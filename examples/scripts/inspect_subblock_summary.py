"""Inspect one image-backed observation sub-block summary artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from dluxshera.inference.observation_summary import inspect_subblock_summary_artifact


def _print_report(report: dict[str, Any]) -> None:
    provenance = report["provenance"]
    dimensions = report["dimensions"]
    schur = report["schur"]
    print(f"Summary: {report['summary_json_path']}")
    print(f"Schema: {report['schema_version']}")
    print(f"Subblock: {report['subblock_id']}")
    print(
        "Dimensions: "
        f"n_theta={dimensions['n_theta']} "
        f"n_phi={dimensions['n_phi']} "
        f"combined_dim={dimensions['combined_dim']}"
    )
    print(f"Objective kind: {provenance['objective_kind']}")
    print(f"Variance model: {provenance['variance_model']}")
    print(f"Case root: {provenance['case_root']}")
    print(f"Cube path: {provenance['cube_path']}")
    print(f"Matrix sidecar: {report['matrix_sidecar_path']}")
    print("Theta labels:")
    for label in report["theta_labels"]:
        print(f"  - {label}")
    print("Phi labels:")
    for label in report["phi_labels"]:
        print(f"  - {label}")
    print(
        "H_pp: "
        f"rank={schur['h_pp_rank']} "
        f"cond={schur['h_pp_condition_number']:.6g} "
        f"eig_min={schur['h_pp_min_eigenvalue']:.6g} "
        f"eig_max={schur['h_pp_max_eigenvalue']:.6g}"
    )
    print(
        "Reduced information: "
        f"rank={schur['reduced_information_rank']} "
        f"cond={schur['reduced_information_condition_number']:.6g} "
        f"eig_min={schur['reduced_information_min_eigenvalue']:.6g} "
        f"eig_max={schur['reduced_information_max_eigenvalue']:.6g}"
    )
    print(
        "Schur diagnostics: "
        f"damping={schur['damping']} "
        f"symmetry_residual={schur['symmetry_residual']} "
        f"psd_within_tolerance={schur['psd_within_tolerance']} "
        f"used_pseudoinverse={schur['used_pseudoinverse']}"
    )
    print(f"Reduced score norm: {report['reduced_score_norm']:.6g}")
    print("Top reduced-score entries:")
    for item in report["top_reduced_score_entries"]:
        print(f"  - {item['label']}: {item['value']:+.6g}")
    print("Top reduced-information diagonal entries:")
    for item in report["top_reduced_information_diagonal_entries"]:
        print(f"  - {item['label']}: {item['value']:+.6g}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect one image-backed subblock summary artifact.",
    )
    parser.add_argument(
        "summary_json",
        type=Path,
        help="Path to subblock_summary.json.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path where the compact inspection report will be written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    report = inspect_subblock_summary_artifact(args.summary_json)
    _print_report(report)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return report


if __name__ == "__main__":
    main()
