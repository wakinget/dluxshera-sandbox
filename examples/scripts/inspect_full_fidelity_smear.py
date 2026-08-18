#!/usr/bin/env python3
"""Inspect full-fidelity per-subblock smear templates for a dry-run/campaign."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from dluxshera.utils.smear_audit import (
    DEFAULT_LENGTH_TOL_PIX,
    DEFAULT_PLACEHOLDER_THRESHOLD_PIX,
    DEFAULT_THETA_TOL_DEG,
    build_smear_summary_rows,
    plan_smear_rows,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def inspect_run(
    run_root: Path,
    *,
    strict: bool,
    length_tol_pix: float = DEFAULT_LENGTH_TOL_PIX,
    theta_tol_deg: float = DEFAULT_THETA_TOL_DEG,
    placeholder_threshold_pix: float = DEFAULT_PLACEHOLDER_THRESHOLD_PIX,
    write_summary: bool = False,
) -> list[dict[str, object]]:
    plan_path = run_root / "campaign_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"campaign_plan.json not found under {run_root}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = build_smear_summary_rows(
        plan_smear_rows(plan),
        run_root=run_root,
        strict=strict,
        length_tol_pix=length_tol_pix,
        theta_tol_deg=theta_tol_deg,
        placeholder_threshold_pix=placeholder_threshold_pix,
    )
    if write_summary:
        _write_csv(run_root / "trajectory" / "smear_summary.csv", rows)
    return rows


def _print_table(rows: list[dict[str, object]]) -> None:
    print("subblock  length_pix   theta_deg     render_match  inference_match")
    for row in rows:
        subblock = f"{int(row['subblock_index']):06d}"
        length = float(row["smear_length_pix"])
        theta = float(row["smear_theta_deg"])
        render = "ok" if row.get("render_match") is True else "fail"
        inference = "ok" if row.get("inference_match") is True else "fail"
        print(f"{subblock}  {length:10.6f}  {theta:11.6f}  {render:12s}  {inference}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument("--length-tol-pix", type=float, default=DEFAULT_LENGTH_TOL_PIX)
    parser.add_argument("--theta-tol-deg", type=float, default=DEFAULT_THETA_TOL_DEG)
    parser.add_argument("--placeholder-threshold-pix", type=float, default=DEFAULT_PLACEHOLDER_THRESHOLD_PIX)
    args = parser.parse_args(argv)
    try:
        rows = inspect_run(
            args.run_root,
            strict=bool(args.strict),
            length_tol_pix=float(args.length_tol_pix),
            theta_tol_deg=float(args.theta_tol_deg),
            placeholder_threshold_pix=float(args.placeholder_threshold_pix),
            write_summary=bool(args.write_summary),
        )
    except Exception as exc:
        print(f"smear inspection failed: {exc}", file=sys.stderr)
        return 2
    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
