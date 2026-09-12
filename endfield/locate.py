"""批量定位产物（data/locator/locate.jsonl）的纯逻辑与 CLI 进程封装。

`local/maplocator/bin/map-locate` 每行输出一条定位记录；本模块负责解析、断点续跑、
合并、失败分类与汇总，并提供 `LocalizerStream`（--stream 流式追踪的常驻进程封装，
供 live.py 之类的实时消费方使用）。真正的进程编排（分片、并行、进度）在
`locate_dataset.py`。

记录字段见 local/maplocator/README.local.md：name/status/message/zone/x/y/rot/
locConf/isHeld/latencyMs/attempts/elapsedMs。
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from endfield.data_utils import atomic_path

Record = dict[str, Any]

STATUS_SUCCESS = 0
STATUS_READ_FAILED = -1
STATUS_ROI_FAILED = -2

# 分数门限：kHighConfidenceOverride（旁路冷启动共识）与 MapLocator 默认 loc_threshold。
HIGH_CONF = 0.85
DEFAULT_LOC_THRESHOLD = 0.55

_MESSAGE_CATEGORIES = {
    "Global search failed.": "global_search_failed",
    "Cold-start collecting.": "cold_start_unsettled",
    "Far-jump rejected.": "far_jump_rejected",
}
_STATUS_CATEGORIES = {
    2: "screen_blocked",
    3: "teleported",
    4: "yolo_failed",
    5: "not_initialized",
}

_TIER_RE = re.compile(r"^(.+)_L(\d+)_(\d+)$")
_BASE_RE = re.compile(r"^(.+)_Base$")


def parse_record(line: str) -> Record:
    """解析 CLI 的一行 JSONL；缺 name/status 视为产物损坏，直接报错。"""
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL line: {line[:160]!r}") from exc
    if not isinstance(record, dict) or "name" not in record or "status" not in record:
        raise ValueError(f"record missing name/status: {line[:160]!r}")
    return record


def load_records(path: Path) -> dict[str, Record]:
    """读取现有产物，按 name 索引；文件不存在视为空。"""
    if not path.exists():
        return {}
    records: dict[str, Record] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = parse_record(line)
            records[record["name"]] = record
    return records


def write_jsonl(path: Path, records: Iterable[Record]) -> None:
    """原子写入产物：每行一条，按 name 排序，保证重复运行结果稳定。"""
    ordered = sorted(records, key=lambda r: r["name"])
    with atomic_path(path) as temp:
        with temp.open("w", encoding="utf-8") as handle:
            for record in ordered:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def pending_names(
    names: Iterable[str],
    done: Mapping[str, Record],
    retry_failed: bool = True,
) -> list[str]:
    """待定位清单：已成功的跳过；失败项默认重跑（定位是确定性的，重跑幂等）。"""
    pending = []
    for name in names:
        record = done.get(name)
        if record is None or (retry_failed and record.get("status") != STATUS_SUCCESS):
            pending.append(name)
    return pending


def merge_records(existing: Iterable[Record], new: Iterable[Record]) -> list[Record]:
    """按 name 合并新旧记录，新的覆盖旧的。"""
    by_name: dict[str, Record] = {record["name"]: record for record in existing}
    by_name.update({record["name"]: record for record in new})
    return [by_name[name] for name in sorted(by_name)]


def accept(record: Record) -> tuple[bool, str]:
    """产物入选门：定位失败、held、locConf 低于 loc_threshold 均不入选。

    held 是 MapLocator 对「全局搜索没有过线峰、只能放行裸峰」的显式标记；识别层
    JSON 不带该字段，直连 CLI 才有，因此这道门是精确的（held 分数均 < 0.55）。
    """
    if int(record.get("status", -1)) != STATUS_SUCCESS:
        return False, classify(record)
    if record.get("isHeld"):
        return False, "held"
    if float(record.get("locConf", 0.0)) < DEFAULT_LOC_THRESHOLD:
        return False, "below_loc_threshold"
    return True, "ok"


def classify(record: Record) -> str:
    """把状态与 message 归到稳定的失败类别，供报告与排除决策使用。"""
    """把状态与 message 归到稳定的失败类别，供报告与排除决策使用。"""
    status = int(record.get("status", 0))
    if status == STATUS_SUCCESS:
        return "ok"
    if status == STATUS_READ_FAILED:
        return "read_failed"
    if status == STATUS_ROI_FAILED:
        return "roi_failed"
    if status == 1:
        return _MESSAGE_CATEGORIES.get(str(record.get("message", "")), "tracking_lost")
    return _STATUS_CATEGORIES.get(status, f"unknown_{status}")


def summarize(records: Iterable[Record]) -> dict[str, Any]:
    """全量统计：成败、失败分类、调用次数与 locConf 分布、按命名族成功率。"""
    records = list(records)
    successes = [record for record in records if record.get("status") == STATUS_SUCCESS]
    by_status = Counter(int(record.get("status", 0)) for record in records)
    by_category = Counter(classify(record) for record in records)
    attempts = Counter(int(record.get("attempts", 0)) for record in records)
    accepted = 0
    excluded_by_reason: Counter[str] = Counter()
    for record in records:
        is_accepted, reason = accept(record)
        if is_accepted:
            accepted += 1
        else:
            excluded_by_reason[reason] += 1

    locconf = None
    if successes:
        confidences = sorted(float(record.get("locConf", 0.0)) for record in successes)
        if len(confidences) >= 2:
            p25, median, p75 = statistics.quantiles(confidences, n=4)
        else:
            p25 = median = p75 = confidences[0]
        locconf = {
            "min": confidences[0],
            "p25": p25,
            "median": median,
            "p75": p75,
            "max": confidences[-1],
            "below_high_conf": sum(1 for value in confidences if value < HIGH_CONF),
            "below_loc_threshold": sum(1 for value in confidences if value < DEFAULT_LOC_THRESHOLD),
        }

    families: dict[str, dict[str, int]] = {}
    for record in records:
        family = str(record.get("name", "")).split("_", 1)[0]
        entry = families.setdefault(family, {"total": 0, "ok": 0, "accepted": 0})
        entry["total"] += 1
        if record.get("status") == STATUS_SUCCESS:
            entry["ok"] += 1
        if accept(record)[0]:
            entry["accepted"] += 1

    return {
        "total": len(records),
        "ok": len(successes),
        "failed": len(records) - len(successes),
        "accepted": accepted,
        "excluded_by_reason": dict(sorted(excluded_by_reason.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_category": dict(sorted(by_category.items())),
        "attempts": dict(sorted(attempts.items())),
        "locconf": locconf,
        "held": sum(1 for record in successes if record.get("isHeld")),
        "families": dict(sorted(families.items())),
    }


def split_shards(names: Sequence[str], jobs: int) -> list[list[str]]:
    """把清单切成 jobs 个连续分片（保持顺序，大小尽量均衡）。"""
    if jobs < 1:
        raise ValueError(f"jobs must be >= 1, got {jobs}")
    names = list(names)
    base, extra = divmod(len(names), jobs)
    shards: list[list[str]] = []
    start = 0
    for index in range(jobs):
        size = base + (1 if index < extra else 0)
        shards.append(names[start : start + size])
        start += size
    return shards


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


def zone_asset_path(zone_id: str, assets_root: Path) -> Path | None:
    """把 MapLocator 的 zone_id 映射回底图资产路径（与 zone 命名规则互逆）。

    - `<Parent>_Base`  → `<root>/<Parent>/Base.png`
    - `<Parent>_L<level>_<tier>` → `<root>/<Parent>/Lv<level:03d>Tier<tier>.png`
    - 其它（如 OMVBase01）→ 任意子目录下的 `<zone_id>.png`
    """
    base_match = _BASE_RE.fullmatch(zone_id)
    if base_match:
        candidate = assets_root / base_match.group(1) / "Base.png"
        return candidate if candidate.exists() else None

    tier_match = _TIER_RE.fullmatch(zone_id)
    if tier_match:
        region, level, tier = tier_match.group(1), int(tier_match.group(2)), tier_match.group(3)
        candidate = assets_root / region / f"Lv{level:03d}Tier{tier}.png"
        return candidate if candidate.exists() else None

    hits = sorted(assets_root.glob(f"*/{zone_id}.png"))
    return hits[0] if hits else None


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
        except subprocess.TimeoutExpired, OSError:
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
