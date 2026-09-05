from __future__ import annotations

import json
import math
import os
import random
import re
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from endfield.polar import IMG_H, IMG_W

ANGLE_RE = re.compile(r"_r(\d+(?:\.\d+)?)\.png$")
SEED = 42
POLAR_IMAGE_SIZE = (IMG_W, IMG_H)


def parse_angle(path: Path) -> float:
    """文件名标注的角度可带一位小数。"""
    match = ANGLE_RE.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse angle from filename: {path.name}")
    angle = float(match.group(1))
    if not 0 <= angle < 360:
        raise ValueError(f"angle out of range in filename: {path.name}")
    return angle


def angle_target(angle: float) -> np.ndarray:
    radians = math.radians(angle)
    return np.asarray([math.sin(radians), math.cos(radians)], dtype=np.float32)


def circular_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs((predicted - target + 180.0) % 360.0 - 180.0)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"{path}: expected PNG, got {image.format}")
        if image.size != POLAR_IMAGE_SIZE:
            raise ValueError(f"{path}: expected {IMG_W}x{IMG_H}, got {image.size}")
        if image.mode != "RGB":
            raise ValueError(f"{path}: expected RGB, got {image.mode}")
        return np.asarray(image, dtype=np.uint8).copy()


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """产出临时文件路径，写毕原子替换到 path；中途失败清理临时文件后原样抛出。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        yield Path(temp_name)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_json_dump(path: Path, value: object) -> None:
    with atomic_path(path) as temp:
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=True)
            stream.write("\n")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def png_names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.glob("*.png"))


def validate_manifest_names(names: Iterable[str], available: set[str], label: str) -> list[str]:
    values = list(names)
    if len(values) != len(set(values)):
        raise ValueError(f"{label} manifest contains duplicate filenames")
    if not set(values).issubset(available):
        missing = sorted(set(values) - available)
        raise ValueError(f"{label} manifest references missing files: {missing[:5]}")
    return values


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
