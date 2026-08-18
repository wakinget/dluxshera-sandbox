"""Shared narrow helpers for campaign wrappers."""

from __future__ import annotations

import csv
import json
import math
import os
import shlex
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def write_csv_rows(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    ensure_dir(path.parent)
    rows = list(rows)
    if fieldnames is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(str(key))
        fieldnames = ordered
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json_ready(row.get(k, "")) for k in fieldnames})


def format_shell_command(command: Sequence[str], *, env_prefix: Mapping[str, str] | None = None) -> str:
    parts: list[str] = []
    if env_prefix:
        for key in sorted(env_prefix):
            parts.append(f"{key}={shlex.quote(str(env_prefix[key]))}")
    parts.extend(shlex.quote(str(part)) for part in command)
    return " ".join(parts)


def write_shell_command(path: Path, command: Sequence[str], *, env_prefix: Mapping[str, str] | None = None) -> None:
    ensure_dir(path.parent)
    text = format_shell_command(command, env_prefix=env_prefix)
    path.write_text(text + os.linesep, encoding="utf-8")


def load_existing_campaign_plan(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None
