"""Generate scaffold directories for prescribed Monte Carlo multi-YAML sweeps.

This utility creates one experiment directory per sweep point, each containing a
`prescription.yaml` ready for the existing prescribed-MC workflow that discovers
`prescription.*` under `--outdir <experiment-dir>`.

Supported modes
---------------
1) Detector knowledge-error sweep (`--mode detector_ke`, default)
   - Requires `experiment.inference_system.detector.layers`.
   - Patches
     `experiment.inference_system.detector.layers[*].knowledge_error.scale`
     and optional `knowledge_error.realization_policy`.

2) Top-level scalar field sweep (`--mode scalar_field`)
   - Patches an existing dotted field path (for example
     `system.optics.psf_npix`).
   - Does not require `experiment.inference_system`.

Both modes force:
- `experiment.outputs.outdir: .`

Output layout
-------------
By default, output is written under:
`Results/<sweep-name>_<YYYYMMDD-HHMMSS>/`

Example:

    Results/detector_ke_sweep_20260320-113500/
      prescription_base.yaml
      sweep_manifest.json
      ke_0/
        prescription.yaml
      ke_1e-4/
        prescription.yaml

This utility only generates configs. It does not execute Monte Carlo jobs.

Usage examples
--------------
Detector KE sweep:

    python examples/scripts/generate_prescribed_mc_sweep.py \
      --base examples/recipes/prescribed_mc_template/prescription.yaml \
      --mode detector_ke \
      --scales 0 1e-4 3e-4 1e-3 3e-3 1e-2 \
      --layer pixel_offsets \
      --realization-policy per_run \
      --results-orientation row

Scalar field sweep:

    python examples/scripts/generate_prescribed_mc_sweep.py \
      --base Results/detector_crop_sweep_template.yaml \
      --mode scalar_field \
      --field-path system.optics.psf_npix \
      --values 256 224 192 160 128 96 64 \
      --sweep-name detector_crop_sweep \
      --label-prefix psf_npix \
      --results-orientation row
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import shutil
from collections.abc import Mapping, MutableMapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, NamedTuple

from dluxshera.config.io import load_config_file

NOTE_KEYS = ("notes", "note", "comment", "comments")
DEFAULT_SWEEP_NAME = "detector_ke_sweep"
DEFAULT_RESULTS_ROOT = Path("Results")
MODE_DETECTOR_KE = "detector_ke"
MODE_SCALAR_FIELD = "scalar_field"


class SweepPoint(NamedTuple):
    """Single sweep point with parsed scale and directory label."""

    token: str
    scale: float
    scale_label: str
    dirname: str


class ScalarSweepPoint(NamedTuple):
    """Single scalar-field sweep point with typed value and directory label."""

    token: str
    value: Any
    value_label: str
    dirname: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for sweep scaffold generation."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a prescribed-MC multi-YAML sweep scaffold for detector "
            "knowledge-error or top-level scalar-field sweeps."
        )
    )
    parser.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Base prescription file (.yaml/.yml/.json).",
    )
    parser.add_argument(
        "--mode",
        choices=(MODE_DETECTOR_KE, MODE_SCALAR_FIELD),
        default=MODE_DETECTOR_KE,
        help=(
            "Sweep mode: detector_ke (default) or scalar_field "
            "(for dotted config paths such as system.optics.psf_npix)."
        ),
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        required=False,
        help=(
            "Detector-KE mode only: one or more knowledge-error scales "
            "(for example: 0 1e-4 1e-3)."
        ),
    )
    parser.add_argument(
        "--layer",
        default="pixel_offsets",
        help="Detector-KE mode only: detector layer name (default: pixel_offsets).",
    )
    parser.add_argument(
        "--field-path",
        default=None,
        help=(
            "Scalar-field mode only: dotted path to an existing scalar field "
            "(for example: system.optics.psf_npix)."
        ),
    )
    parser.add_argument(
        "--values",
        nargs="+",
        default=None,
        help="Scalar-field mode only: one or more values to assign to --field-path.",
    )
    parser.add_argument(
        "--label-prefix",
        default=None,
        help=(
            "Scalar-field mode only: folder label prefix. Defaults to the "
            "terminal field name."
        ),
    )
    parser.add_argument(
        "--sweep-name",
        default=DEFAULT_SWEEP_NAME,
        help=f"Sweep root prefix (default: {DEFAULT_SWEEP_NAME}).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Default parent directory when --root-dir is omitted (default: {DEFAULT_RESULTS_ROOT}).",
    )
    parser.add_argument(
        "--realization-policy",
        default=None,
        help="Optional knowledge_error.realization_policy override for each sweep point.",
    )
    parser.add_argument(
        "--results-orientation",
        choices=("row", "col"),
        default=None,
        help="Optional override for experiment.monte_carlo.results_orientation.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help="Optional override for experiment.monte_carlo.n_runs.",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help=(
            "Explicit sweep output root. If omitted, uses "
            "<results-root>/<sweep-name>_<timestamp>."
        ),
    )
    parser.add_argument(
        "--notes-suffix",
        default=None,
        help="Optional extra text appended to each generated experiment note.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs without writing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing output root.",
    )
    return parser.parse_args(argv)


def validate_cli_args(args: argparse.Namespace) -> None:
    """Validate mode-specific CLI argument requirements."""

    mode = str(args.mode)
    if mode == MODE_DETECTOR_KE:
        if not args.scales:
            raise ValueError("--scales is required when --mode detector_ke.")
        if args.field_path is not None:
            raise ValueError("--field-path is only valid when --mode scalar_field.")
        if args.values is not None:
            raise ValueError("--values is only valid when --mode scalar_field.")
        if args.label_prefix is not None:
            raise ValueError("--label-prefix is only valid when --mode scalar_field.")
        return

    if mode == MODE_SCALAR_FIELD:
        if args.field_path is None:
            raise ValueError("--field-path is required when --mode scalar_field.")
        if args.values is None:
            raise ValueError("--values is required when --mode scalar_field.")
        if args.scales is not None:
            raise ValueError("--scales is only valid when --mode detector_ke.")
        if args.realization_policy is not None:
            raise ValueError(
                "--realization-policy is only valid when --mode detector_ke."
            )
        return

    raise ValueError(
        f"Unsupported --mode {mode!r}. Supported: "
        f"{MODE_DETECTOR_KE!r}, {MODE_SCALAR_FIELD!r}."
    )


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    """Return a dict from mapping-like values, else raise a clear error."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping/dict.")
    if isinstance(value, dict):
        return value
    return dict(value)


def _load_yaml_module():
    """Return imported `yaml` module, raising a clear error if unavailable."""

    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError(
            "PyYAML is required to write generated prescription YAML files."
        ) from exc
    return yaml


def timestamp_label(now: dt.datetime | None = None) -> str:
    """Return timestamp label used in generated sweep directory names."""

    current = now or dt.datetime.now()
    return current.strftime("%Y%m%d-%H%M%S")


def resolve_root_dir(
    *,
    root_dir: Path | None,
    results_root: Path,
    sweep_name: str,
    now: dt.datetime | None = None,
) -> Path:
    """Resolve output root using explicit root or timestamped default."""

    if root_dir is not None:
        return root_dir.expanduser()
    stamp = timestamp_label(now)
    return results_root.expanduser() / f"{sweep_name}_{stamp}"


def format_scale_label(scale_decimal: Decimal) -> str:
    """Format scale values into stable, human-readable folder labels.

    Examples:
    - 0 -> "0"
    - 1e-4 -> "1e-4"
    - 3e-4 -> "3e-4"
    - 1e-2 -> "1e-2"
    """

    if scale_decimal.is_zero():
        return "0"

    if scale_decimal == scale_decimal.to_integral() and scale_decimal >= 1:
        return str(scale_decimal.to_integral())

    scientific = format(scale_decimal.normalize(), "E").lower()
    mantissa, exponent = scientific.split("e", maxsplit=1)
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    exponent_value = int(exponent)
    if exponent_value == 0:
        return mantissa
    return f"{mantissa}e{exponent_value}"


def parse_sweep_points(scale_tokens: Sequence[str]) -> list[SweepPoint]:
    """Parse scale tokens into sweep points with stable directory names."""

    points: list[SweepPoint] = []
    seen_dirs: set[str] = set()

    for token in scale_tokens:
        try:
            decimal_scale = Decimal(str(token))
        except InvalidOperation as exc:
            raise ValueError(f"Invalid scale value {token!r}.") from exc

        if not decimal_scale.is_finite():
            raise ValueError(f"Scale must be finite, got {token!r}.")
        if decimal_scale < 0:
            raise ValueError(f"Scale must be non-negative, got {token!r}.")

        scale_label = format_scale_label(decimal_scale)
        dirname = f"ke_{scale_label}"
        if dirname in seen_dirs:
            raise ValueError(
                f"Duplicate sweep label {dirname!r} from --scales values. "
                "Ensure each scale maps to a unique label."
            )
        seen_dirs.add(dirname)
        points.append(
            SweepPoint(
                token=str(token),
                scale=float(decimal_scale),
                scale_label=scale_label,
                dirname=dirname,
            )
        )

    return points


def _split_dotted_path(field_path: str) -> list[str]:
    """Split and validate dotted path segments."""

    raw = str(field_path).strip()
    if not raw:
        raise ValueError("--field-path must be a non-empty dotted path.")
    parts = [part.strip() for part in raw.split(".")]
    if any(not part for part in parts):
        raise ValueError(
            f"--field-path must not contain empty segments, got {field_path!r}."
        )
    return parts


def _sanitize_dir_token(value: str, *, context: str) -> str:
    """Return a filesystem-friendly label token."""

    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{context} must not be empty.")
    cleaned = cleaned.replace("/", "_").replace("\\", "_").replace(" ", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    if not cleaned:
        raise ValueError(f"{context} must contain at least one visible character.")
    return cleaned


def _resolve_existing_dotted_field(
    cfg: MutableMapping[str, Any],
    *,
    field_path: str,
) -> tuple[MutableMapping[str, Any], str, Any]:
    """Resolve an existing dotted field path and return (parent, key, value)."""

    parts = _split_dotted_path(field_path)
    current: Any = cfg
    walked: list[str] = []

    for part in parts[:-1]:
        walked.append(part)
        if not isinstance(current, MutableMapping):
            container = ".".join(walked[:-1]) or "<root>"
            raise ValueError(
                f"--field-path {field_path!r} is invalid: {container!r} is not a mapping."
            )
        if part not in current:
            missing = ".".join(walked)
            raise ValueError(
                f"--field-path {field_path!r} does not exist in base config "
                f"(missing key: {missing})."
            )
        current = current[part]

    if not isinstance(current, MutableMapping):
        parent = ".".join(parts[:-1]) or "<root>"
        raise ValueError(
            f"--field-path {field_path!r} is invalid: parent {parent!r} is not a mapping."
        )

    leaf_key = parts[-1]
    if leaf_key not in current:
        missing = ".".join(parts)
        raise ValueError(
            f"--field-path {field_path!r} does not exist in base config "
            f"(missing key: {missing})."
        )
    return current, leaf_key, current[leaf_key]


def _ensure_scalar_target_value(existing_value: Any, *, field_path: str) -> None:
    """Ensure the scalar-field sweep target resolves to a scalar leaf."""

    if isinstance(existing_value, (Mapping, list, tuple)):
        raise ValueError(
            f"--field-path {field_path!r} must point to an existing scalar field, "
            f"got {type(existing_value).__name__}."
        )

    if not isinstance(existing_value, (str, int, float, bool)) and existing_value is not None:
        raise ValueError(
            f"--field-path {field_path!r} points to unsupported scalar type "
            f"{type(existing_value).__name__}."
        )


def _coerce_scalar_token(
    token: str,
    *,
    existing_value: Any,
    field_path: str,
) -> Any:
    """Parse one scalar-field token while preserving the existing value type."""

    raw_token = str(token).strip()

    if isinstance(existing_value, bool):
        lowered = raw_token.lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
        raise ValueError(
            f"Value {token!r} must be boolean for --field-path {field_path!r} "
            "(accepted: true/false)."
        )

    if isinstance(existing_value, int):
        try:
            decimal_value = Decimal(raw_token)
        except InvalidOperation as exc:
            raise ValueError(
                f"Invalid integer value {token!r} for --field-path {field_path!r}."
            ) from exc
        if not decimal_value.is_finite():
            raise ValueError(f"Value must be finite, got {token!r}.")
        if decimal_value != decimal_value.to_integral():
            raise ValueError(
                f"Value {token!r} must be an integer for --field-path {field_path!r}."
            )
        return int(decimal_value)

    if isinstance(existing_value, float):
        try:
            decimal_value = Decimal(raw_token)
        except InvalidOperation as exc:
            raise ValueError(
                f"Invalid numeric value {token!r} for --field-path {field_path!r}."
            ) from exc
        if not decimal_value.is_finite():
            raise ValueError(f"Value must be finite, got {token!r}.")
        return float(decimal_value)

    if isinstance(existing_value, str):
        return raw_token

    if existing_value is None:
        lowered = raw_token.lower()
        if lowered in {"null", "none"}:
            return None
        return raw_token

    raise ValueError(
        f"--field-path {field_path!r} points to unsupported scalar type "
        f"{type(existing_value).__name__}."
    )


def _format_scalar_value_label(value: Any) -> str:
    """Format scalar values into concise, stable label strings."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        decimal_value = Decimal(str(value))
        if decimal_value.is_finite():
            if decimal_value == decimal_value.to_integral():
                return str(decimal_value.to_integral())
            plain = format(decimal_value.normalize(), "f").rstrip("0").rstrip(".")
            if plain:
                return plain
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else "empty"
    return str(value)


def resolve_scalar_label_prefix(*, field_path: str, label_prefix: str | None) -> str:
    """Resolve effective folder-label prefix for scalar-field sweeps."""

    terminal_key = _split_dotted_path(field_path)[-1]
    raw_prefix = label_prefix if label_prefix is not None else terminal_key
    return _sanitize_dir_token(raw_prefix, context="--label-prefix")


def resolve_scalar_note_target(*, field_path: str, label_prefix: str | None) -> str:
    """Resolve note label for scalar-field sweep annotations."""

    if label_prefix is not None and label_prefix.strip():
        return label_prefix.strip()
    return field_path


def parse_scalar_sweep_points(
    value_tokens: Sequence[str],
    *,
    field_path: str,
    existing_value: Any,
    label_prefix: str | None,
) -> list[ScalarSweepPoint]:
    """Parse scalar sweep values into typed points with stable directory labels."""

    prefix = resolve_scalar_label_prefix(field_path=field_path, label_prefix=label_prefix)
    points: list[ScalarSweepPoint] = []
    seen_dirs: set[str] = set()

    for token in value_tokens:
        value = _coerce_scalar_token(
            str(token),
            existing_value=existing_value,
            field_path=field_path,
        )
        value_label = _format_scalar_value_label(value)
        value_label_dir = _sanitize_dir_token(value_label, context="sweep value label")
        dirname = f"{prefix}_{value_label_dir}"
        if dirname in seen_dirs:
            raise ValueError(
                f"Duplicate sweep label {dirname!r} from --values entries. "
                "Ensure each value maps to a unique label."
            )
        seen_dirs.add(dirname)
        points.append(
            ScalarSweepPoint(
                token=str(token),
                value=value,
                value_label=value_label,
                dirname=dirname,
            )
        )
    return points


def validate_base_prescription(base_cfg: dict[str, Any], *, base_path: Path) -> None:
    """Validate required prescribed-MC structure before patching."""

    experiment = _require_mapping(
        base_cfg.get("experiment"),
        context=f"{base_path}: top-level experiment",
    )
    if "monte_carlo" not in experiment:
        raise ValueError(
            f"{base_path}: expected experiment.monte_carlo for prescribed-MC config."
        )
    _require_mapping(
        experiment.get("monte_carlo"),
        context=f"{base_path}: experiment.monte_carlo",
    )

    kind = experiment.get("kind")
    if kind is not None and str(kind) != "prescribed_mc":
        raise ValueError(
            f"{base_path}: experiment.kind must be 'prescribed_mc' when provided, "
            f"got {kind!r}."
        )


def _find_layer_index(
    experiment_cfg: dict[str, Any],
    *,
    layer_name: str,
) -> int:
    """Return detector layer index by name under experiment.inference_system."""

    inference_cfg = experiment_cfg.get("inference_system")
    if not isinstance(inference_cfg, Mapping):
        raise ValueError(
            "experiment.inference_system is required for this sweep generator."
        )

    detector_cfg = inference_cfg.get("detector")
    if not isinstance(detector_cfg, Mapping):
        raise ValueError(
            "experiment.inference_system.detector must be a mapping/dict."
        )

    layers = detector_cfg.get("layers")
    if not isinstance(layers, list):
        raise ValueError(
            "experiment.inference_system.detector.layers must be a list."
        )

    for idx, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            continue
        if str(layer.get("name")) == layer_name:
            return idx

    raise ValueError(
        "Requested detector layer was not found under "
        "experiment.inference_system.detector.layers: "
        f"{layer_name!r}."
    )


def _base_note_text(experiment_cfg: dict[str, Any]) -> str:
    """Return the first non-empty experiment note-like field."""

    for key in NOTE_KEYS:
        value = experiment_cfg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_experiment_note(
    *,
    base_note: str,
    layer_name: str,
    scale_label: str,
    realization_policy: str | None,
    notes_suffix: str | None,
) -> str:
    """Build a per-point note by appending a compact sweep annotation."""

    segments = [seg.strip() for seg in base_note.split("|") if seg.strip()]
    kept_segments = [seg for seg in segments if not seg.lower().startswith("sweep:")]

    sweep_segment = f"sweep: {layer_name} knowledge_error scale={scale_label}"
    if realization_policy:
        sweep_segment += f" realization_policy={realization_policy}"

    output_segments = kept_segments + [sweep_segment]
    if notes_suffix:
        suffix = notes_suffix.strip()
        if suffix:
            output_segments.append(suffix)
    return " | ".join(output_segments)


def build_scalar_field_experiment_note(
    *,
    base_note: str,
    sweep_target: str,
    value_label: str,
    notes_suffix: str | None,
) -> str:
    """Build a per-point note for scalar-field sweeps."""

    segments = [seg.strip() for seg in base_note.split("|") if seg.strip()]
    kept_segments = [seg for seg in segments if not seg.lower().startswith("sweep:")]

    sweep_segment = f"sweep: {sweep_target}={value_label}"
    output_segments = kept_segments + [sweep_segment]
    if notes_suffix:
        suffix = notes_suffix.strip()
        if suffix:
            output_segments.append(suffix)
    return " | ".join(output_segments)


def _apply_common_experiment_overrides(
    *,
    experiment_cfg: dict[str, Any],
    results_orientation: str | None,
    n_runs: int | None,
) -> None:
    """Apply shared experiment-level overrides used by all sweep modes."""

    outputs_cfg = experiment_cfg.get("outputs")
    if outputs_cfg is None:
        outputs: dict[str, Any] = {}
    else:
        outputs = _require_mapping(outputs_cfg, context="experiment.outputs")
    outputs["outdir"] = "."
    experiment_cfg["outputs"] = outputs

    monte_carlo_cfg = _require_mapping(
        experiment_cfg.get("monte_carlo"),
        context="experiment.monte_carlo",
    )
    if results_orientation is not None:
        monte_carlo_cfg["results_orientation"] = results_orientation
    if n_runs is not None:
        if n_runs <= 0:
            raise ValueError("--n-runs must be positive when provided.")
        monte_carlo_cfg["n_runs"] = int(n_runs)
    experiment_cfg["monte_carlo"] = monte_carlo_cfg


def patch_prescription_for_point(
    base_cfg: dict[str, Any],
    *,
    layer_name: str,
    sweep_point: SweepPoint,
    realization_policy: str | None,
    results_orientation: str | None,
    n_runs: int | None,
    notes_suffix: str | None,
) -> dict[str, Any]:
    """Return a patched prescription for one sweep point."""

    cfg = copy.deepcopy(base_cfg)
    experiment_cfg = _require_mapping(
        cfg.get("experiment"),
        context="top-level experiment",
    )
    cfg["experiment"] = experiment_cfg

    layer_idx = _find_layer_index(experiment_cfg, layer_name=layer_name)
    inference_cfg = _require_mapping(
        experiment_cfg.get("inference_system"),
        context="experiment.inference_system",
    )
    detector_cfg = _require_mapping(
        inference_cfg.get("detector"),
        context="experiment.inference_system.detector",
    )
    layers = detector_cfg.get("layers")
    if not isinstance(layers, list):
        raise ValueError("experiment.inference_system.detector.layers must be a list.")

    layer_cfg = _require_mapping(
        layers[layer_idx],
        context=f"experiment.inference_system.detector.layers[{layer_idx}]",
    )
    layers[layer_idx] = layer_cfg

    knowledge_error = layer_cfg.get("knowledge_error")
    if knowledge_error is None:
        knowledge_error_cfg: dict[str, Any] = {}
    else:
        knowledge_error_cfg = _require_mapping(
            knowledge_error,
            context=(
                "experiment.inference_system.detector.layers"
                f"[{layer_idx}].knowledge_error"
            ),
        )
    knowledge_error_cfg["scale"] = sweep_point.scale
    if realization_policy is not None:
        knowledge_error_cfg["realization_policy"] = realization_policy
    layer_cfg["knowledge_error"] = knowledge_error_cfg

    _apply_common_experiment_overrides(
        experiment_cfg=experiment_cfg,
        results_orientation=results_orientation,
        n_runs=n_runs,
    )

    base_note = _base_note_text(experiment_cfg)
    experiment_cfg["notes"] = build_experiment_note(
        base_note=base_note,
        layer_name=layer_name,
        scale_label=sweep_point.scale_label,
        realization_policy=realization_policy,
        notes_suffix=notes_suffix,
    )
    return cfg


def patch_prescription_for_scalar_field_point(
    base_cfg: dict[str, Any],
    *,
    field_path: str,
    sweep_point: ScalarSweepPoint,
    sweep_target: str,
    results_orientation: str | None,
    n_runs: int | None,
    notes_suffix: str | None,
) -> dict[str, Any]:
    """Return a patched prescription for one scalar-field sweep point."""

    cfg = copy.deepcopy(base_cfg)
    experiment_cfg = _require_mapping(
        cfg.get("experiment"),
        context="top-level experiment",
    )
    cfg["experiment"] = experiment_cfg

    field_parent, field_key, _ = _resolve_existing_dotted_field(cfg, field_path=field_path)
    field_parent[field_key] = sweep_point.value

    _apply_common_experiment_overrides(
        experiment_cfg=experiment_cfg,
        results_orientation=results_orientation,
        n_runs=n_runs,
    )

    base_note = _base_note_text(experiment_cfg)
    experiment_cfg["notes"] = build_scalar_field_experiment_note(
        base_note=base_note,
        sweep_target=sweep_target,
        value_label=sweep_point.value_label,
        notes_suffix=notes_suffix,
    )
    return cfg


def build_manifest(
    *,
    created_at: dt.datetime,
    base_path: Path,
    output_root: Path,
    sweep_name: str,
    layer_name: str,
    sweep_points: Sequence[SweepPoint],
    realization_policy: str | None,
    results_orientation: str | None,
    n_runs: int | None,
    notes_suffix: str | None,
) -> dict[str, Any]:
    """Build detector-KE mode sweep manifest for reproducibility."""

    return {
        "timestamp": created_at.isoformat(timespec="seconds"),
        "timestamp_label": timestamp_label(created_at),
        "script_path": str(Path(__file__).as_posix()),
        "base_prescription_path": str(base_path.resolve()),
        "output_root": str(output_root.resolve()),
        "mode": MODE_DETECTOR_KE,
        "sweep_name": sweep_name,
        "layer": layer_name,
        "scales": [point.scale for point in sweep_points],
        "scale_labels": [point.scale_label for point in sweep_points],
        "realization_policy_override": realization_policy,
        "results_orientation_override": results_orientation,
        "n_runs_override": n_runs,
        "notes_suffix": notes_suffix,
        "experiments": [
            {
                "label": point.dirname,
                "scale": point.scale,
                "scale_label": point.scale_label,
                "directory": point.dirname,
                "prescription_path": f"{point.dirname}/prescription.yaml",
            }
            for point in sweep_points
        ],
    }


def build_scalar_field_manifest(
    *,
    created_at: dt.datetime,
    base_path: Path,
    output_root: Path,
    sweep_name: str,
    field_path: str,
    label_prefix: str,
    sweep_points: Sequence[ScalarSweepPoint],
    results_orientation: str | None,
    n_runs: int | None,
    notes_suffix: str | None,
) -> dict[str, Any]:
    """Build scalar-field mode sweep manifest for reproducibility."""

    return {
        "timestamp": created_at.isoformat(timespec="seconds"),
        "timestamp_label": timestamp_label(created_at),
        "script_path": str(Path(__file__).as_posix()),
        "base_prescription_path": str(base_path.resolve()),
        "output_root": str(output_root.resolve()),
        "mode": MODE_SCALAR_FIELD,
        "sweep_name": sweep_name,
        "field_path": field_path,
        "label_prefix": label_prefix,
        "values": [point.value for point in sweep_points],
        "value_labels": [point.value_label for point in sweep_points],
        "results_orientation_override": results_orientation,
        "n_runs_override": n_runs,
        "notes_suffix": notes_suffix,
        "experiments": [
            {
                "label": point.dirname,
                "value": point.value,
                "value_label": point.value_label,
                "directory": point.dirname,
                "prescription_path": f"{point.dirname}/prescription.yaml",
            }
            for point in sweep_points
        ],
    }


def _validate_output_root(output_root: Path, *, force: bool) -> None:
    """Validate output root path with optional force mode."""

    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"Output root exists and is not a directory: {output_root}")
    if output_root.exists() and not force:
        raise ValueError(
            f"Output root already exists: {output_root}. Use --force to reuse it."
        )


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write YAML payload preserving insertion order for readability."""

    yaml = _load_yaml_module()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            payload,
            handle,
            sort_keys=False,
            default_flow_style=False,
        )


def _write_base_copy(
    *,
    base_path: Path,
    base_cfg: dict[str, Any],
    output_root: Path,
) -> Path:
    """Write root-level copy of base prescription for sweep provenance."""

    destination = output_root / "prescription_base.yaml"
    suffix = base_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        shutil.copyfile(base_path, destination)
    else:
        _write_yaml(destination, base_cfg)
    return destination


def generate_sweep_scaffold(
    *,
    base_cfg: dict[str, Any],
    base_path: Path,
    output_root: Path,
    sweep_name: str,
    layer_name: str,
    sweep_points: Sequence[SweepPoint],
    realization_policy: str | None,
    results_orientation: str | None,
    n_runs: int | None,
    notes_suffix: str | None,
    now: dt.datetime,
    dry_run: bool,
    force: bool,
) -> dict[str, Any]:
    """Generate sweep scaffold or return the planned manifest in dry-run mode."""

    _validate_output_root(output_root, force=force)
    manifest = build_manifest(
        created_at=now,
        base_path=base_path,
        output_root=output_root,
        sweep_name=sweep_name,
        layer_name=layer_name,
        sweep_points=sweep_points,
        realization_policy=realization_policy,
        results_orientation=results_orientation,
        n_runs=n_runs,
        notes_suffix=notes_suffix,
    )

    if dry_run:
        return manifest

    output_root.mkdir(parents=True, exist_ok=True)
    _write_base_copy(base_path=base_path, base_cfg=base_cfg, output_root=output_root)

    for point in sweep_points:
        experiment_dir = output_root / point.dirname
        experiment_dir.mkdir(parents=True, exist_ok=True)
        point_cfg = patch_prescription_for_point(
            base_cfg,
            layer_name=layer_name,
            sweep_point=point,
            realization_policy=realization_policy,
            results_orientation=results_orientation,
            n_runs=n_runs,
            notes_suffix=notes_suffix,
        )
        _write_yaml(experiment_dir / "prescription.yaml", point_cfg)

    manifest_path = output_root / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def generate_scalar_field_sweep_scaffold(
    *,
    base_cfg: dict[str, Any],
    base_path: Path,
    output_root: Path,
    sweep_name: str,
    field_path: str,
    label_prefix: str,
    sweep_target: str,
    sweep_points: Sequence[ScalarSweepPoint],
    results_orientation: str | None,
    n_runs: int | None,
    notes_suffix: str | None,
    now: dt.datetime,
    dry_run: bool,
    force: bool,
) -> dict[str, Any]:
    """Generate scalar-field sweep scaffold or planned manifest in dry-run mode."""

    _validate_output_root(output_root, force=force)
    manifest = build_scalar_field_manifest(
        created_at=now,
        base_path=base_path,
        output_root=output_root,
        sweep_name=sweep_name,
        field_path=field_path,
        label_prefix=label_prefix,
        sweep_points=sweep_points,
        results_orientation=results_orientation,
        n_runs=n_runs,
        notes_suffix=notes_suffix,
    )

    if dry_run:
        return manifest

    output_root.mkdir(parents=True, exist_ok=True)
    _write_base_copy(base_path=base_path, base_cfg=base_cfg, output_root=output_root)

    for point in sweep_points:
        experiment_dir = output_root / point.dirname
        experiment_dir.mkdir(parents=True, exist_ok=True)
        point_cfg = patch_prescription_for_scalar_field_point(
            base_cfg,
            field_path=field_path,
            sweep_point=point,
            sweep_target=sweep_target,
            results_orientation=results_orientation,
            n_runs=n_runs,
            notes_suffix=notes_suffix,
        )
        _write_yaml(experiment_dir / "prescription.yaml", point_cfg)

    manifest_path = output_root / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _print_plan(manifest: dict[str, Any], *, dry_run: bool) -> None:
    """Print concise generation summary for users."""

    mode_name = str(manifest.get("mode", MODE_DETECTOR_KE))
    mode = "Dry run (no files written)" if dry_run else "Generated sweep scaffold"
    print(f"{mode}: {manifest['output_root']}")
    print(f"Base prescription: {manifest['base_prescription_path']}")
    if mode_name == MODE_SCALAR_FIELD:
        print(f"Field path: {manifest['field_path']}")
    else:
        print(f"Layer: {manifest['layer']}")
    print("Sweep points:")
    for experiment in manifest["experiments"]:
        if mode_name == MODE_SCALAR_FIELD:
            print(f"  - {experiment['label']}: value={experiment['value_label']}")
        else:
            print(f"  - {experiment['label']}: scale={experiment['scale_label']}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    now = dt.datetime.now()

    try:
        validate_cli_args(args)
        base_path = args.base.expanduser()
        base_cfg = load_config_file(base_path)
        validate_base_prescription(base_cfg, base_path=base_path)
        output_root = resolve_root_dir(
            root_dir=args.root_dir,
            results_root=args.results_root,
            sweep_name=args.sweep_name,
            now=now,
        )
        mode = str(args.mode)

        if mode == MODE_DETECTOR_KE:
            _ = _find_layer_index(
                _require_mapping(base_cfg.get("experiment"), context="top-level experiment"),
                layer_name=args.layer,
            )
            sweep_points = parse_sweep_points(args.scales)
            manifest = generate_sweep_scaffold(
                base_cfg=base_cfg,
                base_path=base_path,
                output_root=output_root,
                sweep_name=args.sweep_name,
                layer_name=args.layer,
                sweep_points=sweep_points,
                realization_policy=args.realization_policy,
                results_orientation=args.results_orientation,
                n_runs=args.n_runs,
                notes_suffix=args.notes_suffix,
                now=now,
                dry_run=bool(args.dry_run),
                force=bool(args.force),
            )
        else:
            field_path = str(args.field_path).strip()
            _, _, existing_value = _resolve_existing_dotted_field(
                base_cfg,
                field_path=field_path,
            )
            _ensure_scalar_target_value(existing_value, field_path=field_path)
            scalar_points = parse_scalar_sweep_points(
                args.values,
                field_path=field_path,
                existing_value=existing_value,
                label_prefix=args.label_prefix,
            )
            label_prefix = resolve_scalar_label_prefix(
                field_path=field_path,
                label_prefix=args.label_prefix,
            )
            sweep_target = resolve_scalar_note_target(
                field_path=field_path,
                label_prefix=args.label_prefix,
            )
            manifest = generate_scalar_field_sweep_scaffold(
                base_cfg=base_cfg,
                base_path=base_path,
                output_root=output_root,
                sweep_name=args.sweep_name,
                field_path=field_path,
                label_prefix=label_prefix,
                sweep_target=sweep_target,
                sweep_points=scalar_points,
                results_orientation=args.results_orientation,
                n_runs=args.n_runs,
                notes_suffix=args.notes_suffix,
                now=now,
                dry_run=bool(args.dry_run),
                force=bool(args.force),
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    _print_plan(manifest, dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
