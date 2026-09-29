#!/usr/bin/env python3
"""Shared data and mechanics for deterministic performance programs."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_KERNELS = ("add", "mm", "softmax", "layernorm")
DEFAULT_SHAPES = {
    "add": {"shape": [1024, 1024], "dtype": "float32"},
    "mm": {"m": 256, "n": 256, "k": 256, "dtype": "float32"},
    "softmax": {"shape": [128, 1024], "dim": -1, "dtype": "float32"},
    "layernorm": {
        "shape": [128, 1024],
        "normalized_shape": [1024],
        "dtype": "float32",
        "eps": 1.0e-5,
    },
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def summarize(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "stdev_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def neighboring_compile_benchmark(script_path: Path) -> Path:
    path = script_path.resolve().with_name("compile_benchmark.py")
    if not path.is_file():
        raise FileNotFoundError(f"compile_benchmark.py not found next to {script_path}")
    return path


def write_projected_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {fieldname: row.get(fieldname) for fieldname in fieldnames} for row in rows
        )
