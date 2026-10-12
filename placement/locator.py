"""`map-locate` 进程驱动：批量一轮跑。

批量分片与进度的编排在 `cli/locate_dataset.py`；本模块只管进程与 JSONL 的对接。
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

from placement.records import parse_record


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

