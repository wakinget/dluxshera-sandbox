from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dluxshera.utils.campaigns import (
    format_shell_command,
    load_existing_campaign_plan,
    write_csv_rows,
    write_json,
    write_shell_command,
)


def test_write_json_normalizes_numpy_and_path(tmp_path: Path) -> None:
    output = tmp_path / "payload.json"
    write_json(
        output,
        {
            "path": tmp_path / "x",
            "scalar": np.float64(1.2),
            "array": np.array([1, 2]),
            "nan_value": float("nan"),
        },
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["path"].endswith("/x")
    assert payload["scalar"] == 1.2
    assert payload["array"] == [1, 2]
    assert payload["nan_value"] == "nan"


def test_write_csv_rows_preserves_field_order(tmp_path: Path) -> None:
    output = tmp_path / "rows.csv"
    write_csv_rows(output, [{"a": 1, "b": 2}, {"b": 3, "a": 4}])
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "a,b"


def test_write_shell_command_quotes_args(tmp_path: Path) -> None:
    output = tmp_path / "cmd.sh"
    write_shell_command(output, ["python", "script.py", "--name", "value with spaces"])
    text = output.read_text(encoding="utf-8")
    assert "'value with spaces'" in text
    assert format_shell_command(["echo", "a b"]).endswith("'a b'")


def test_load_existing_campaign_plan_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "campaign_plan.json"
    path.write_text(json.dumps({"summary_paths": {"case": ["x.json"]}}), encoding="utf-8")
    payload = load_existing_campaign_plan(path)
    assert payload is not None
    assert payload["summary_paths"]["case"] == ["x.json"]
