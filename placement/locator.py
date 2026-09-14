"""`map-locate` 进程驱动：批量一轮跑与 `--stream` 常驻进程。

批量分片与进度的编排在 `cli/locate_dataset.py`；本模块只管进程与 JSONL 的对接。
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np

from placement.records import Record, parse_record


def run_cli(
    cli_cmd: Sequence[str | Path],
    resource_dir: Path,
    names: Sequence[str],
    out_path: Path,
    progress: Callable[[int, int], None] | None = None,
    progress_every: int = 200,
) -> int:
    """跑一个 map-locate 进程：names 从 stdin 逐行输入，stdout 的 JSONL 落 out_path。

    输入与输出同时进行（喂 stdin 的线程与读 stdout 的主线程分离），避免管道写满死锁。
    返回写出的记录条数；进程非零退出时报错。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(part) for part in cli_cmd] + ["--resource-dir", str(resource_dir)]
    with out_path.with_suffix(".stderr.log").open("w", encoding="utf-8") as errlog:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errlog,
            text=True,
            encoding="utf-8",
        )

        def feed_stdin() -> None:
            try:
                for name in names:
                    proc.stdin.write(name + "\n")
            except BrokenPipeError:
                pass
            finally:
                proc.stdin.close()

        writer = threading.Thread(target=feed_stdin, daemon=True)
        writer.start()

        total = len(names)
        count = 0
        with out_path.open("w", encoding="utf-8") as handle:
            for line in proc.stdout:
                if not line.strip():
                    continue
                parse_record(line)  # 产物损坏立刻暴露，而不是带进下游
                handle.write(line if line.endswith("\n") else line + "\n")
                count += 1
                if progress is not None and (count % progress_every == 0 or count == total):
                    progress(count, total)
        writer.join()
        returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(
            f"map-locate exited with {returncode}: {command} "
            f"(see {out_path.with_suffix('.stderr.log')})"
        )
    return count


class LocalizerStream:
    """map-locate --stream 的常驻进程封装：一帧进、一条定位记录出。

    帧先写入 work_dir 下的图像文件，再把路径喂给 CLI stdin；CLI 流式模式不重置
    追踪状态，连续帧共享同一跟踪上下文。调用方必须串行调用 locate()（一次一帧）；
    进程初始化失败或中途退出都在 locate() 处报 RuntimeError，stderr 尾部附在错误里。
    """

    def __init__(
        self,
        cli_cmd: Sequence[str | Path],
        resource_dir: Path,
        work_dir: Path,
        frame_name: str = "frame.bmp",
    ) -> None:
        self._command = [str(part) for part in cli_cmd] + [
            "--resource-dir",
            str(resource_dir),
            "--stream",
        ]
        self._work_dir = Path(work_dir)
        self._frame_path = self._work_dir / frame_name
        self._log_path = self._work_dir / "map-locate.stderr.log"
        self._process: subprocess.Popen | None = None
        self._stderr = None

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("LocalizerStream already started")
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._stderr = self._log_path.open("w", encoding="utf-8")
        self._process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    def locate(self, frame_bgr: np.ndarray) -> Record:
        """写帧 -> 喂路径 -> 读一条 JSONL；CLI 死亡时抛 RuntimeError。"""
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("LocalizerStream not started")
        if not cv2.imwrite(str(self._frame_path), frame_bgr):
            raise ValueError(f"failed to write frame: {self._frame_path}")
        try:
            process.stdin.write(str(self._frame_path) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise self._died() from exc
        line = process.stdout.readline()
        if not line:
            raise self._died()
        return parse_record(line)

    def close(self) -> None:
        """关 stdin 让 CLI 自然退出，超时再杀；幂等。"""
        process, stderr = self._process, self._stderr
        self._process, self._stderr = None, None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            process.kill()
            process.wait()
        finally:
            if stderr is not None:
                stderr.close()

    def _died(self) -> RuntimeError:
        process = self._process
        code = process.returncode if process is not None else None
        tail = ""
        if self._log_path.exists():
            tail = self._log_path.read_text(encoding="utf-8").strip()[-500:]
        return RuntimeError(f"map-locate exited (returncode={code}): {tail or 'no stderr output'}")
