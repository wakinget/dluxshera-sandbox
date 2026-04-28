"""Run the first worked plate-scale Fisher screening study.

This script is intentionally study-specific. It defines a small fixed matrix
over one shared candidate parameter and one target:

- candidate: ``optics.plate_scale_as_per_pix``
- target: ``ALPHA_CEN``
- frame counts: ``1, 5, 20, 50``
- noise modes: ``noiseless`` and ``shot_noise_only``

Each case is executed through the existing ``fisher_only`` harness path. The
script then writes aggregate CSV/JSON summaries and a small set of review plots
under one study root.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from dluxshera.config.io import load_config_file, load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.params.store import ParameterStore
from dluxshera.systems.base import compose_forward_spec


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "Results" / "plate_scale_fisher_alpha_cen"
DEFAULT_TRACE_TEMPLATE = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "observation_subblock_trace_template"
    / "subblock_trace_prescription.yaml"
)
DEFAULT_RENDER_TEMPLATE = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "observation_subblock_template"
    / "subblock_generation_prescription.yaml"
)
DEFAULT_INFERENCE_TEMPLATE = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "observation_subblock_inference_template"
    / "subblock_inference_prescription.yaml"
)
STUDY_SCHEMA_VERSION = "plate_scale_fisher_screen.v1"
STUDY_MODE = "fisher_only"
CANDIDATE_PARAMETER = "optics.plate_scale_as_per_pix"
TARGET_NAME = "ALPHA_CEN"
DEFAULT_FRAME_COUNTS = (1, 5, 20, 50)
SUPPORTED_NOISE_MODES = ("noiseless", "shot_noise_only")


@dataclass(frozen=True)
class PlateScaleFisherCase:
    """One explicit case in the worked plate-scale Fisher matrix."""

    target_name: str
    frame_count: int
    noise_mode: str
    case_name: str
    case_root: Path


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_study_module():
    return _load_module(
        REPO_ROOT / "examples" / "scripts" / "run_obs_subblock_study.py",
        "obs_subblock_plate_scale_fisher_study_module",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return payload


def _write_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _ensure_mapping(parent: dict[str, Any], key: str, *, path: str) -> dict[str, Any]:
    value = parent.get(key)
    if value is None:
        parent[key] = {}
        return parent[key]
    if not isinstance(value, dict):
        raise ValueError(f"{path}.{key} must be a mapping/dict.")
    return value


def _get_nested_scalar(mapping: dict[str, Any] | None, dotted_key: str) -> float | None:
    if not isinstance(mapping, dict):
        return None
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    return float(current)


def _resolve_target_name(cfg: dict[str, Any] | None) -> str | None:
    if not isinstance(cfg, dict):
        return None
    source_cfg = cfg.get("source")
    if not isinstance(source_cfg, dict):
        return None
    target = source_cfg.get("target")
    if not isinstance(target, str) or not target.strip():
        return None
    return target.strip()


def parse_frame_counts(raw: str | Sequence[int] | None) -> tuple[int, ...]:
    """Parse a comma-separated or sequence-valued frame-count list."""

    if raw is None:
        return DEFAULT_FRAME_COUNTS
    if isinstance(raw, str):
        tokens = [part.strip() for part in raw.split(",")]
    else:
        tokens = [str(value).strip() for value in raw]

    values: list[int] = []
    for token in tokens:
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError("--frame-counts must be a comma-separated list of integers.") from exc
        if value <= 0:
            raise ValueError("--frame-counts must contain only positive integers.")
        values.append(value)
    if not values:
        raise ValueError("--frame-counts must contain at least one value.")
    return tuple(values)


def parse_noise_modes(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse and validate the narrow first-pass noise-mode list."""

    if raw is None:
        return SUPPORTED_NOISE_MODES
    if isinstance(raw, str):
        tokens = [part.strip() for part in raw.split(",")]
    else:
        tokens = [str(value).strip() for value in raw]

    values: list[str] = []
    for token in tokens:
        if not token:
            continue
        if token not in SUPPORTED_NOISE_MODES:
            raise ValueError(
                f"Unsupported noise mode {token!r}. Expected one of: "
                + ", ".join(SUPPORTED_NOISE_MODES)
                + "."
            )
        values.append(token)
    if not values:
        raise ValueError("--noise-modes must contain at least one value.")
    return tuple(values)


def build_plate_scale_fisher_case_specs(
    *,
    study_root: Path,
    frame_counts: Sequence[int] = DEFAULT_FRAME_COUNTS,
    noise_modes: Sequence[str] = SUPPORTED_NOISE_MODES,
    target_name: str = TARGET_NAME,
) -> tuple[PlateScaleFisherCase, ...]:
    """Expand the explicit 8-case study matrix into stable case roots."""

    specs: list[PlateScaleFisherCase] = []
    target_slug = target_name.lower()
    for noise_mode in noise_modes:
        if noise_mode not in SUPPORTED_NOISE_MODES:
            raise ValueError(
                f"Unsupported noise mode {noise_mode!r}. Expected one of: "
                + ", ".join(SUPPORTED_NOISE_MODES)
                + "."
            )
        for frame_count in frame_counts:
            if int(frame_count) <= 0:
                raise ValueError("Frame counts must be positive integers.")
            case_name = f"{target_slug}_n{int(frame_count):03d}_{noise_mode}"
            specs.append(
                PlateScaleFisherCase(
                    target_name=target_name,
                    frame_count=int(frame_count),
                    noise_mode=noise_mode,
                    case_name=case_name,
                    case_root=(study_root / "cases" / case_name).resolve(),
                )
            )
    return tuple(specs)


def resolve_plate_scale_truth_and_target(
    *,
    render_template: Path,
    inference_template: Path,
) -> tuple[float, str]:
    """Resolve the baseline plate scale and target name from the study templates."""

    candidate_value: float | None = None
    resolved_target: str | None = None
    for template_path in (render_template, inference_template):
        user_cfg = load_user_config(
            config_path=template_path.resolve(),
            system_preset=None,
            experiment_preset=None,
        )
        resolved_cfg = resolve_config(user_cfg)
        system_cfg = resolved_cfg.get("system")
        if candidate_value is None:
            candidate_value = _get_nested_scalar(system_cfg, CANDIDATE_PARAMETER)
            if candidate_value is None and isinstance(system_cfg, dict):
                forward_spec = compose_forward_spec(system_cfg)
                store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(
                    forward_spec
                )
                raw_value = store.get(CANDIDATE_PARAMETER)
                if raw_value is not None:
                    candidate_value = float(raw_value)
        if resolved_target is None:
            resolved_target = _resolve_target_name(system_cfg)
    if candidate_value is None:
        raise ValueError(
            "Unable to resolve the baseline plate scale from the configured templates."
        )
    if resolved_target is None:
        resolved_target = TARGET_NAME
    return float(candidate_value), resolved_target


def write_noise_mode_render_template(
    *,
    base_render_template: Path,
    output_path: Path,
    noise_mode: str,
) -> Path:
    """Write one render-template copy with explicit noiseless or shot-noise settings."""

    if noise_mode not in SUPPORTED_NOISE_MODES:
        raise ValueError(
            f"Unsupported noise mode {noise_mode!r}. Expected one of: "
            + ", ".join(SUPPORTED_NOISE_MODES)
            + "."
        )

    cfg = load_config_file(base_render_template.resolve())
    experiment_cfg = _ensure_mapping(cfg, "experiment", path="root")
    noise_cfg = _ensure_mapping(experiment_cfg, "noise", path="experiment")
    noise_cfg["enabled"] = noise_mode == "shot_noise_only"
    noise_cfg["photon_noise"] = True
    noise_cfg["read_noise"] = False
    noise_cfg["dark_current"] = False
    _write_json(output_path, cfg)
    return output_path


def build_case_row(
    *,
    case: PlateScaleFisherCase,
    truth_value: float,
    case_summary: dict[str, Any] | None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Flatten one case summary into the aggregate row contract."""

    base_row: dict[str, Any] = {
        "target": case.target_name,
        "candidate": CANDIDATE_PARAMETER,
        "study_mode": STUDY_MODE,
        "frame_count": int(case.frame_count),
        "noise_mode": case.noise_mode,
        "truth_value": float(truth_value),
        "case_name": case.case_name,
        "case_root": str(case.case_root),
        "case_status": "error" if error_message is not None else "planned",
        "error_message": error_message,
        "case_summary_path": None,
        "fisher_summary_json": None,
        "fisher_blocks_npz": None,
        "reference_value": None,
        "nuisance_keys": None,
        "f_pp": None,
        "i_marg": None,
        "sigma_cond": None,
        "sigma_marg": None,
        "absorption_fraction": None,
        "f_pp_is_finite": None,
        "i_marg_is_finite": None,
        "valid_conditional_sigma": None,
        "valid_marginal_sigma": None,
        "marginalization_status": None,
        "nuisance_block_status": None,
    }
    if case_summary is None:
        return base_row

    base_row["case_summary_path"] = case_summary.get("summary_path")
    fisher_summary = case_summary.get("fisher_summary")
    if not isinstance(fisher_summary, dict):
        if error_message is None:
            base_row["case_status"] = "planned" if case_summary.get("dry_run") else "missing_summary"
        return base_row

    artifacts = fisher_summary.get("artifacts")
    base_row.update(
        {
            "case_status": "ok",
            "fisher_summary_json": (
                None if not isinstance(artifacts, dict) else artifacts.get("fisher_summary_json")
            ),
            "fisher_blocks_npz": (
                None if not isinstance(artifacts, dict) else artifacts.get("fisher_blocks_npz")
            ),
            "reference_value": fisher_summary.get("candidate_reference_value"),
            "nuisance_keys": "|".join(fisher_summary.get("frame_keys", [])),
            "f_pp": fisher_summary.get("f_pp"),
            "i_marg": fisher_summary.get("i_marg"),
            "sigma_cond": fisher_summary.get("sigma_cond"),
            "sigma_marg": fisher_summary.get("sigma_marg"),
            "absorption_fraction": fisher_summary.get("absorption_fraction"),
            "f_pp_is_finite": fisher_summary.get("f_pp_is_finite"),
            "i_marg_is_finite": fisher_summary.get("i_marg_is_finite"),
            "valid_conditional_sigma": fisher_summary.get("valid_conditional_sigma"),
            "valid_marginal_sigma": fisher_summary.get("valid_marginal_sigma"),
            "marginalization_status": fisher_summary.get("marginalization_status"),
            "nuisance_block_status": fisher_summary.get("nuisance_block_status"),
        }
    )
    return base_row


def _augment_case_outputs(
    *,
    case: PlateScaleFisherCase,
    truth_value: float,
    case_summary: dict[str, Any],
) -> dict[str, Any]:
    """Add stable study-matrix metadata to case-local Fisher outputs."""

    summary_path_value = case_summary.get("summary_path")
    if isinstance(summary_path_value, str) and summary_path_value.strip():
        summary_path = Path(summary_path_value).resolve()
        summary_payload = _read_json(summary_path)
        summary_payload["plate_scale_fisher_case"] = {
            "target": case.target_name,
            "candidate": CANDIDATE_PARAMETER,
            "frame_count": int(case.frame_count),
            "noise_mode": case.noise_mode,
            "truth_value": float(truth_value),
            "study_mode": STUDY_MODE,
        }
        _write_json(summary_path, summary_payload)
        case_summary = summary_payload

    fisher_summary = case_summary.get("fisher_summary")
    if not isinstance(fisher_summary, dict):
        return case_summary

    fisher_summary.update(
        {
            "target_name": case.target_name,
            "truth_value": float(truth_value),
            "noise_mode": case.noise_mode,
            "frame_count": int(case.frame_count),
            "study_mode": STUDY_MODE,
        }
    )
    artifacts = fisher_summary.get("artifacts")
    if isinstance(artifacts, dict):
        fisher_summary_path_value = artifacts.get("fisher_summary_json")
        if isinstance(fisher_summary_path_value, str) and fisher_summary_path_value.strip():
            _write_json(Path(fisher_summary_path_value).resolve(), fisher_summary)
    case_summary["fisher_summary"] = fisher_summary
    if isinstance(summary_path_value, str) and summary_path_value.strip():
        _write_json(Path(summary_path_value).resolve(), case_summary)
    return case_summary


def plot_metric_vs_frame_count(
    *,
    rows: Sequence[dict[str, Any]],
    metric_key: str,
    output_path: Path,
    y_label: str,
    title: str,
    log_y: bool,
) -> Path:
    """Plot one scalar summary against frame count for the two study noise modes."""

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for noise_mode in SUPPORTED_NOISE_MODES:
        points: list[tuple[int, float]] = []
        for row in rows:
            if row.get("noise_mode") != noise_mode:
                continue
            value = row.get(metric_key)
            frame_count = row.get("frame_count")
            if value is None or frame_count is None:
                continue
            if isinstance(value, bool):
                continue
            try:
                y = float(value)
                x = int(frame_count)
            except (TypeError, ValueError):
                continue
            if not y_label.startswith("Absorption") and not (y > 0.0):
                continue
            if y != y or y == float("inf") or y == float("-inf"):
                continue
            points.append((x, y))
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        ax.plot(
            [item[0] for item in points],
            [item[1] for item in points],
            marker="o",
            linewidth=1.8,
            label=noise_mode,
        )

    ax.set_xlabel("Frame Count")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(list(DEFAULT_FRAME_COUNTS))
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_plate_scale_fisher_artifacts(
    *,
    study_root: Path,
    rows: Sequence[dict[str, Any]],
    case_summaries: Sequence[dict[str, Any]],
    truth_value: float,
    target_name: str,
    frame_counts: Sequence[int],
    noise_modes: Sequence[str],
    dry_run: bool,
) -> dict[str, Any]:
    """Write the aggregate CSV/JSON/plot outputs for the worked study."""

    csv_path = study_root / "plate_scale_fisher_summary.csv"
    json_path = study_root / "plate_scale_fisher_summary.json"
    sigma_marg_plot = study_root / "sigma_marg_vs_frame_count.png"
    absorption_plot = study_root / "absorption_fraction_vs_frame_count.png"
    sigma_cond_plot = study_root / "sigma_cond_vs_frame_count.png"

    _write_rows_csv(csv_path, rows)
    plot_metric_vs_frame_count(
        rows=rows,
        metric_key="sigma_marg",
        output_path=sigma_marg_plot,
        y_label="Marginalized Sigma",
        title="Plate Scale Fisher Screening: Marginalized Sigma",
        log_y=True,
    )
    plot_metric_vs_frame_count(
        rows=rows,
        metric_key="absorption_fraction",
        output_path=absorption_plot,
        y_label="Absorption Fraction",
        title="Plate Scale Fisher Screening: Absorption Fraction",
        log_y=False,
    )
    plot_metric_vs_frame_count(
        rows=rows,
        metric_key="sigma_cond",
        output_path=sigma_cond_plot,
        y_label="Conditional Sigma",
        title="Plate Scale Fisher Screening: Conditional Sigma",
        log_y=True,
    )

    summary = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_mode": STUDY_MODE,
        "candidate": CANDIDATE_PARAMETER,
        "target": target_name,
        "truth_value": float(truth_value),
        "frame_counts": [int(value) for value in frame_counts],
        "noise_modes": list(noise_modes),
        "dry_run": bool(dry_run),
        "case_count": len(rows),
        "successful_case_count": sum(1 for row in rows if row.get("case_status") == "ok"),
        "failed_case_count": sum(1 for row in rows if row.get("case_status") == "error"),
        "artifacts": {
            "aggregate_csv": str(csv_path.resolve()),
            "aggregate_json": str(json_path.resolve()),
            "sigma_marg_plot": str(sigma_marg_plot.resolve()),
            "absorption_fraction_plot": str(absorption_plot.resolve()),
            "sigma_cond_plot": str(sigma_cond_plot.resolve()),
        },
        "cases": list(rows),
        "case_summaries": list(case_summaries),
    }
    _write_json(json_path, summary)
    return summary


def run_plate_scale_fisher_screen(
    *,
    study_root: Path,
    trace_template: Path = DEFAULT_TRACE_TEMPLATE,
    render_template: Path = DEFAULT_RENDER_TEMPLATE,
    inference_template: Path = DEFAULT_INFERENCE_TEMPLATE,
    frame_counts: Sequence[int] = DEFAULT_FRAME_COUNTS,
    noise_modes: Sequence[str] = SUPPORTED_NOISE_MODES,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the explicit Alpha Cen plate-scale Fisher matrix and aggregate outputs."""

    study_root = study_root.resolve()
    study_root.mkdir(parents=True, exist_ok=True)
    study_module = _load_study_module()
    truth_value, resolved_target = resolve_plate_scale_truth_and_target(
        render_template=render_template,
        inference_template=inference_template,
    )
    target_name = resolved_target or TARGET_NAME
    cases = build_plate_scale_fisher_case_specs(
        study_root=study_root,
        frame_counts=frame_counts,
        noise_modes=noise_modes,
        target_name=target_name,
    )

    template_dir = study_root / "templates"
    noise_templates = {
        noise_mode: write_noise_mode_render_template(
            base_render_template=render_template.resolve(),
            output_path=template_dir / f"render_{noise_mode}.json",
            noise_mode=noise_mode,
        )
        for noise_mode in noise_modes
    }

    rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for case in cases:
        try:
            case_summary = study_module.run_obs_subblock_study(
                mode=STUDY_MODE,
                case_root=case.case_root,
                trace_template=trace_template.resolve(),
                render_template=noise_templates[case.noise_mode],
                inference_template=inference_template.resolve(),
                candidate_key=CANDIDATE_PARAMETER,
                truth_value=truth_value,
                n_frames=int(case.frame_count),
                noise_mode=(
                    "disabled" if case.noise_mode == "noiseless" else "enabled"
                ),
                dry_run=bool(dry_run),
            )
            case_summary = _augment_case_outputs(
                case=case,
                truth_value=truth_value,
                case_summary=case_summary,
            )
            case_summaries.append(case_summary)
            rows.append(
                build_case_row(
                    case=case,
                    truth_value=truth_value,
                    case_summary=case_summary,
                )
            )
        except Exception as exc:
            rows.append(
                build_case_row(
                    case=case,
                    truth_value=truth_value,
                    case_summary=None,
                    error_message=str(exc),
                )
            )

    summary = write_plate_scale_fisher_artifacts(
        study_root=study_root,
        rows=rows,
        case_summaries=case_summaries,
        truth_value=truth_value,
        target_name=target_name,
        frame_counts=frame_counts,
        noise_modes=noise_modes,
        dry_run=dry_run,
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the first worked plate-scale fisher_only screening study."
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Study output root for the 8-case Alpha Cen plate-scale matrix.",
    )
    parser.add_argument(
        "--trace-template",
        type=Path,
        default=DEFAULT_TRACE_TEMPLATE,
        help="Trace template YAML/JSON path.",
    )
    parser.add_argument(
        "--render-template",
        type=Path,
        default=DEFAULT_RENDER_TEMPLATE,
        help="Render template YAML/JSON path.",
    )
    parser.add_argument(
        "--inference-template",
        type=Path,
        default=DEFAULT_INFERENCE_TEMPLATE,
        help="Inference template YAML/JSON path.",
    )
    parser.add_argument(
        "--frame-counts",
        default=None,
        help="Optional comma-separated frame counts. Default: 1,5,20,50.",
    )
    parser.add_argument(
        "--noise-modes",
        default=None,
        help="Optional comma-separated noise modes. Default: noiseless,shot_noise_only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the case matrix and write aggregate placeholders without running Fisher cases.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    frame_counts = parse_frame_counts(args.frame_counts)
    noise_modes = parse_noise_modes(args.noise_modes)
    summary = run_plate_scale_fisher_screen(
        study_root=args.study_root,
        trace_template=args.trace_template,
        render_template=args.render_template,
        inference_template=args.inference_template,
        frame_counts=frame_counts,
        noise_modes=noise_modes,
        dry_run=bool(args.dry_run),
    )
    print(f"Study root: {args.study_root.resolve()}")
    print(f"Cases: {summary['case_count']}")
    print(f"Aggregate JSON: {summary['artifacts']['aggregate_json']}")
    return summary


if __name__ == "__main__":
    main()
