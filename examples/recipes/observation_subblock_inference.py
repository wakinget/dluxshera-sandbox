"""Run observation sub-block inference from a config-defined active state.

Purpose
-------
This recipe solves for a config-defined active state over a rendered
observation sub-block cube while treating the resolved top-level ``system``
block as the assumed shared state for the run.

Current implemented behavior
----------------------------
- Active inference state is derived from ``experiment.inference.active``:
  ``frame_keys`` vary per frame and ``shared_keys`` vary once per block.
- Initialization is derived from ``experiment.inference.init``.
- The objective is assembled as ``data_term + prior_term + temporal_term``.
- The current data term is Gaussian image NLL with variance from either the
  observed cube or a scalar debug value.
- The current temporal model implementation is ``frame_model.kind: independent``.
- Priors are part of the operational objective structure, but non-empty prior
  configs are not implemented yet.

Tested now vs scaffolded
------------------------
- Tested now: the bundled registration-only prescription with per-frame
  ``source.x_position_as``, ``source.y_position_as``, and
  ``source.position_angle_deg`` and no shared active terms.
- Implemented but lightly exercised: generic frame/shared active-state packing,
  shared initialization, generic recovered-state tables, and mapped JAX block
  prediction/loss helpers.
- Scaffolded only: non-empty priors, non-``independent`` temporal models,
  frame init modes beyond ``shared_guess``/``from_system``, and objective kinds
  beyond Gaussian image NLL.

Examples
--------
Dry-run the bundled template:

``PYTHONPATH=src python examples/recipes/observation_subblock_inference.py --dry-run``

Run the bundled template:

``PYTHONPATH=src python examples/recipes/observation_subblock_inference.py --config examples/recipes/observation_subblock_inference_template/subblock_inference_prescription.yaml``

Override the output root:

``PYTHONPATH=src python examples/recipes/observation_subblock_inference.py --results-dir Results/subblock_inference/demo --run-name block_0001``

Disable optimizer progress for scripted runs:

``PYTHONPATH=src python examples/recipes/observation_subblock_inference.py --config /path/to/prescription.yaml --no-progress``
"""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from astropy.io import fits

from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.inference.losses import gaussian_image_nll
from dluxshera.inference.optimization import fim_theta, run_shera_gd
from dluxshera.params.store import ParameterStore
from dluxshera.systems import SheraBinder
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.obs_subblock_io import (
    find_obs_subblock_sidecar_manifest,
    now_iso_local_ms,
    timestamp_tag,
    to_jsonable_obs_subblock_payload,
    write_obs_subblock_truth_csv,
)
from dluxshera.utils.obs_subblock_keys import (
    ObsSubblockKeyAddress,
    apply_obs_subblock_overrides_preserving_derived,
    canonical_obs_subblock_varying_keys,
    get_obs_subblock_store_value,
    parse_obs_subblock_varying_keys,
    validate_supported_obs_subblock_key_addresses,
)
from dluxshera.utils.obs_subblock_trace import ObsSubblockTrace, load_obs_subblock_trace_csv

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


DEFAULT_PRESCRIPTION_PATH = Path(
    "examples/recipes/observation_subblock_inference_template/subblock_inference_prescription.yaml"
)
DEFAULT_OUTDIR_ROOT = Path("Results/subblock_inference")
MANIFEST_SCHEMA_VERSION = "subblock_inference_manifest.v1"
GENERATOR_ID = "examples/recipes/observation_subblock_inference.py"
TRACE_VALIDATE_DEFAULTS = {
    "require_contiguous_frame_index": True,
    "require_monotonic_time": True,
}


class ActiveState(NamedTuple):
    """Packed active inference state split into frame and shared blocks."""

    frame: jnp.ndarray
    shared: jnp.ndarray


@dataclass(frozen=True)
class ActiveKeySpec:
    """Static metadata for one active key.

    Local to this recipe for now. If another sub-block inference entrypoint needs
    the same frame/shared packing model, this is a candidate for migration.
    """

    canonical: str
    address: ObsSubblockKeyAddress
    kind: str


@dataclass(frozen=True)
class ActiveStateLayout:
    """Describe how frame/shared active variables are packed into optimizer theta."""

    frame_specs: tuple[ActiveKeySpec, ...]
    shared_specs: tuple[ActiveKeySpec, ...]
    n_frame: int

    @property
    def frame_keys(self) -> tuple[str, ...]:
        return tuple(spec.canonical for spec in self.frame_specs)

    @property
    def shared_keys(self) -> tuple[str, ...]:
        return tuple(spec.canonical for spec in self.shared_specs)

    @property
    def all_keys(self) -> tuple[str, ...]:
        return self.frame_keys + self.shared_keys

    @property
    def frame_width(self) -> int:
        return len(self.frame_specs)

    @property
    def shared_width(self) -> int:
        return len(self.shared_specs)

    @property
    def theta_size(self) -> int:
        return self.n_frame * self.frame_width + self.shared_width


@dataclass(frozen=True)
class ObjectiveBundle:
    """Bundle compiled objective helpers for the active-state layout.

    The callables take packed theta so the main recipe flow can stay linear and
    readable without re-encoding state unpacking at each call site.
    """

    total_loss_fn: Callable[[jnp.ndarray], jnp.ndarray]
    objective_terms_fn: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]
    predict_cube_fn: Callable[[jnp.ndarray], jnp.ndarray]
    per_frame_data_terms_fn: Callable[[jnp.ndarray], jnp.ndarray]


@dataclass(frozen=True)
class ThetaPreconditioningBundle:
    """Curvature and learning-rate-vector outputs for theta preconditioning."""

    fim: np.ndarray
    eigvals: np.ndarray
    eigvals_stable: np.ndarray
    preconditioner_diag: np.ndarray
    lr_vec_unclipped: np.ndarray
    lr_vec: np.ndarray
    config: dict[str, Any]


def _required_dict(parent: dict[str, Any], key: str, *, path: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{path}.{key} must be a mapping/dict.")
    return value


def _required_str(parent: dict[str, Any], key: str, *, path: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}.{key} must be a non-empty string.")
    return value.strip()


def _optional_str(parent: dict[str, Any], key: str, *, path: str) -> str | None:
    value = parent.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}.{key} must be a non-empty string when provided.")
    return value.strip()


def _required_list_of_str(parent: dict[str, Any], key: str, *, path: str) -> list[str]:
    value = parent.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{path}.{key} must be a list[str].")
    parsed: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{path}.{key}[{idx}] must be a non-empty string.")
        parsed.append(item.strip())
    return parsed


def _optional_dict(parent: dict[str, Any], key: str, *, path: str) -> dict[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{path}.{key} must be a mapping/dict when provided.")
    return value


def _coerce_numeric_mapping(
    payload: dict[str, Any] | None,
    *,
    path: str,
) -> dict[str, float]:
    """Validate a string-keyed numeric mapping and normalize values to float."""

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a mapping/dict when provided.")

    coerced: dict[str, float] = {}
    for raw_key, raw_value in payload.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError(f"{path} keys must be non-empty strings.")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"{path}.{raw_key} must be numeric.")
        coerced[raw_key.strip()] = float(raw_value)
    return coerced


def _resolve_relative_path(
    value: str,
    *,
    config_path: Path | None,
    field_name: str,
) -> Path:
    """Resolve a config path relative to the prescription file when relative."""

    raw_path = Path(value)
    if raw_path.is_absolute():
        return raw_path
    if config_path is not None:
        return (config_path.parent / raw_path).resolve()
    raise ValueError(
        f"{field_name} is relative but config_path is not available to resolve it."
    )


def _stable_hash_payload(payload: Any) -> str:
    """Return a stable SHA256 hash for a JSON-serializable payload."""

    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _relative_path(path: Path, *, outdir: Path) -> str:
    """Return a POSIX path relative to ``outdir`` when possible."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(outdir.resolve()).as_posix()
    except ValueError:
        return Path(os.path.relpath(resolved, outdir.resolve())).as_posix()


def _load_manifest(path: Path | None) -> dict[str, Any] | None:
    """Load a JSON manifest if provided."""

    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Manifest must decode to a JSON object.")
    return payload


def _resolve_manifest_artifact(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    artifact_key: str,
) -> Path | None:
    """Resolve a manifest artifact path relative to its manifest when needed."""

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    value = artifacts.get(artifact_key)
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()


def _infer_trace_path(
    *,
    trace_path: Path | None,
    manifest: dict[str, Any] | None,
    manifest_path: Path | None,
) -> Path | None:
    """Resolve an optional truth-trace path from config or render manifest."""

    if trace_path is not None:
        return trace_path.resolve()
    if manifest is None or manifest_path is None:
        return None

    candidate = _resolve_manifest_artifact(
        manifest,
        manifest_path=manifest_path,
        artifact_key="frame_truth_csv",
    )
    if candidate is not None and candidate.exists():
        return candidate

    trace_payload = manifest.get("trace")
    if isinstance(trace_payload, dict):
        trace_value = trace_payload.get("path")
        if isinstance(trace_value, str) and trace_value.strip():
            trace_candidate = Path(trace_value)
            if not trace_candidate.is_absolute():
                trace_candidate = (manifest_path.parent / trace_candidate).resolve()
            if trace_candidate.exists():
                return trace_candidate

    return None


def _build_artifact_paths(
    *,
    outdir: Path,
    file_prefix: str,
    timestamp: str,
    include_comparison: bool,
    write_plots: bool,
    include_trace_plots: bool,
    include_parameter_history_heatmap: bool = False,
    include_parameter_residual_history_heatmap: bool = False,
    include_parameter_history_lines: bool = False,
    include_parameter_residual_history_lines: bool = False,
) -> dict[str, Path]:
    """Return artifact paths for one inference run."""

    artifacts: dict[str, Path] = {
        "recovered_trace_csv": outdir / f"{file_prefix}_{timestamp}_recovered_trace.csv",
        "manifest_json": outdir / "manifest.json",
    }
    if include_comparison:
        artifacts["truth_comparison_csv"] = (
            outdir / f"{file_prefix}_{timestamp}_truth_comparison.csv"
        )
    if write_plots:
        artifacts["loss_history_png"] = outdir / f"{file_prefix}_{timestamp}_loss_history.png"
        artifacts["image_fit_png"] = outdir / f"{file_prefix}_{timestamp}_image_fit.png"
        if include_trace_plots:
            if include_comparison:
                artifacts["trace_comparison_png"] = (
                    outdir / f"{file_prefix}_{timestamp}_trace_comparison.png"
                )
                artifacts["trace_residuals_png"] = (
                    outdir / f"{file_prefix}_{timestamp}_trace_residuals.png"
                )
            else:
                artifacts["recovered_traces_png"] = (
                    outdir / f"{file_prefix}_{timestamp}_recovered_traces.png"
                )
        if include_parameter_history_heatmap:
            artifacts["parameter_history_heatmap_png"] = (
                outdir / f"{file_prefix}_{timestamp}_parameter_history_heatmap.png"
            )
        if include_parameter_residual_history_heatmap:
            artifacts["parameter_residual_history_heatmap_png"] = (
                outdir / f"{file_prefix}_{timestamp}_parameter_residual_history_heatmap.png"
            )
        if include_parameter_history_lines:
            artifacts["parameter_history_lines_png"] = (
                outdir / f"{file_prefix}_{timestamp}_parameter_history_lines.png"
            )
        if include_parameter_residual_history_lines:
            artifacts["parameter_residual_history_lines_png"] = (
                outdir / f"{file_prefix}_{timestamp}_parameter_residual_history_lines.png"
            )
    return artifacts


def _theta_labels_for_layout(layout: ActiveStateLayout) -> list[str]:
    """Return readable labels in the exact packed-theta order."""

    labels: list[str] = []
    for frame_index in range(layout.n_frame):
        labels.extend(
            f"frame[{frame_index}].{spec.canonical}" for spec in layout.frame_specs
        )
    labels.extend(f"shared.{spec.canonical}" for spec in layout.shared_specs)
    if len(labels) != layout.theta_size:
        raise ValueError("Theta label count does not match active-state layout.")
    return labels


def _pack_active_state(layout: ActiveStateLayout, state: ActiveState) -> jnp.ndarray:
    """Pack frame and shared state blocks into one optimizer theta vector."""

    frame_flat = jnp.reshape(state.frame, (layout.n_frame * layout.frame_width,))
    if layout.shared_width == 0:
        return frame_flat
    return jnp.concatenate((frame_flat, state.shared), axis=0)


def _unpack_active_state(layout: ActiveStateLayout, theta_flat: jnp.ndarray) -> ActiveState:
    """Unpack one optimizer theta vector into frame and shared state blocks."""

    theta_arr = jnp.asarray(theta_flat)
    frame_size = layout.n_frame * layout.frame_width
    frame_flat = theta_arr[:frame_size]
    shared = theta_arr[frame_size:]
    frame = jnp.reshape(frame_flat, (layout.n_frame, layout.frame_width))
    return ActiveState(frame=frame, shared=shared)


def _build_active_key_specs(
    raw_keys: list[str],
    *,
    forward_spec,
    reference_store: ParameterStore,
    path: str,
) -> tuple[ActiveKeySpec, ...]:
    """Parse, validate, and annotate active keys against the resolved system."""

    addresses = parse_obs_subblock_varying_keys(raw_keys)
    validate_supported_obs_subblock_key_addresses(
        addresses,
        forward_spec=forward_spec,
        reference_store=reference_store,
    )
    canonical_keys = canonical_obs_subblock_varying_keys(addresses)

    specs: list[ActiveKeySpec] = []
    for canonical, address in zip(canonical_keys, addresses):
        field = forward_spec.get(address.base_key)
        kind = getattr(field, "kind", None)
        if kind not in {"primitive", "derived"}:
            raise ValueError(
                f"{path} key {canonical!r} has unsupported resolved kind {kind!r}."
            )
        specs.append(
            ActiveKeySpec(
                canonical=canonical,
                address=address,
                kind=str(kind),
            )
        )
    return tuple(specs)


def _build_active_state_layout(
    *,
    active_cfg: dict[str, Any],
    forward_spec,
    reference_store: ParameterStore,
    n_frame: int,
) -> ActiveStateLayout:
    """Build the active-state layout from config and resolved system metadata."""

    frame_specs = _build_active_key_specs(
        _required_list_of_str(active_cfg, "frame_keys", path="experiment.inference.active"),
        forward_spec=forward_spec,
        reference_store=reference_store,
        path="experiment.inference.active.frame_keys",
    )
    shared_specs = _build_active_key_specs(
        _required_list_of_str(active_cfg, "shared_keys", path="experiment.inference.active"),
        forward_spec=forward_spec,
        reference_store=reference_store,
        path="experiment.inference.active.shared_keys",
    )

    duplicate_keys = set(spec.canonical for spec in frame_specs) & set(
        spec.canonical for spec in shared_specs
    )
    if duplicate_keys:
        raise ValueError(
            "Active keys cannot appear in both frame_keys and shared_keys: "
            + ", ".join(sorted(duplicate_keys))
        )
    if not frame_specs and not shared_specs:
        raise ValueError(
            "experiment.inference.active must define at least one frame or shared key."
        )

    return ActiveStateLayout(
        frame_specs=frame_specs,
        shared_specs=shared_specs,
        n_frame=n_frame,
    )


def _defaults_from_store_for_specs(
    specs: tuple[ActiveKeySpec, ...],
    *,
    base_store: ParameterStore,
) -> dict[str, float]:
    """Return resolved system values for a list of active keys."""

    return {
        spec.canonical: get_obs_subblock_store_value(base_store, address=spec.address)
        for spec in specs
    }


def _extract_frame_init_values(
    frame_init_cfg: dict[str, Any],
    *,
    layout: ActiveStateLayout,
) -> dict[str, float]:
    """Resolve frame init value overrides for ``shared_guess`` mode.

    The preferred config shape is ``init.frame.values`` keyed by canonical active
    key. For compatibility with the current registration template, unique leaf
    aliases such as ``x_position_as`` are also accepted at the top level.
    """

    configured = _coerce_numeric_mapping(
        frame_init_cfg.get("values"),
        path="experiment.inference.init.frame.values",
    )

    valid_keys = set(layout.frame_keys)
    unknown_configured = sorted(set(configured) - valid_keys)
    if unknown_configured:
        raise ValueError(
            "experiment.inference.init.frame.values contains keys not present in "
            "active.frame_keys: " + ", ".join(unknown_configured)
        )

    meta_keys = {"mode", "values", "path"}
    top_level_values = {
        key: value for key, value in frame_init_cfg.items() if key not in meta_keys
    }

    leaf_alias_map: dict[str, str | None] = {}
    for spec in layout.frame_specs:
        if spec.address.index is not None:
            continue
        leaf = spec.address.base_key.split(".")[-1]
        if leaf in leaf_alias_map:
            leaf_alias_map[leaf] = None
        else:
            leaf_alias_map[leaf] = spec.canonical

    for raw_key, raw_value in top_level_values.items():
        if raw_key in configured:
            raise ValueError(
                "Frame init key is specified both in init.frame.values and as a "
                f"top-level alias: {raw_key!r}."
            )

        if raw_key in valid_keys:
            canonical = raw_key
        else:
            canonical = leaf_alias_map.get(raw_key)

        if canonical is None:
            raise ValueError(
                "Unsupported init.frame override key "
                f"{raw_key!r}. Use canonical active keys under init.frame.values."
            )
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"experiment.inference.init.frame.{raw_key} must be numeric.")
        configured[canonical] = float(raw_value)

    return configured


def _resolve_initial_active_state(
    *,
    layout: ActiveStateLayout,
    base_store: ParameterStore,
    init_cfg: dict[str, Any],
) -> ActiveState:
    """Resolve the initial optimizer state from config and the assumed system.

    ``frame`` and ``shared`` init are kept separate so later frame-table or
    propagated-block initializations can be added without reworking the active
    state model itself.
    """

    frame_init_cfg = _required_dict(init_cfg, "frame", path="experiment.inference.init")
    shared_init_cfg = _optional_dict(init_cfg, "shared", path="experiment.inference.init")

    default_frame_values = _defaults_from_store_for_specs(layout.frame_specs, base_store=base_store)
    default_shared_values = _defaults_from_store_for_specs(layout.shared_specs, base_store=base_store)

    frame_mode = _required_str(frame_init_cfg, "mode", path="experiment.inference.init.frame")
    if frame_mode == "shared_guess":
        frame_value_overrides = _extract_frame_init_values(frame_init_cfg, layout=layout)
        frame_seed = np.asarray(
            [
                frame_value_overrides.get(spec.canonical, default_frame_values[spec.canonical])
                for spec in layout.frame_specs
            ],
            dtype=float,
        )
        frame_matrix = np.repeat(frame_seed[None, :], repeats=layout.n_frame, axis=0)
    elif frame_mode == "from_system":
        frame_seed = np.asarray(
            [default_frame_values[spec.canonical] for spec in layout.frame_specs],
            dtype=float,
        )
        frame_matrix = np.repeat(frame_seed[None, :], repeats=layout.n_frame, axis=0)
    elif frame_mode in {"explicit_table", "from_truth_trace", "previous_block"}:
        raise ValueError(
            "experiment.inference.init.frame.mode "
            f"{frame_mode!r} is not implemented yet."
        )
    else:
        raise ValueError(
            "Unsupported experiment.inference.init.frame.mode "
            f"{frame_mode!r}."
        )

    shared_overrides = _coerce_numeric_mapping(
        shared_init_cfg,
        path="experiment.inference.init.shared",
    )
    unknown_shared_overrides = sorted(set(shared_overrides) - set(layout.shared_keys))
    if unknown_shared_overrides:
        raise ValueError(
            "experiment.inference.init.shared contains keys not present in "
            "active.shared_keys: " + ", ".join(unknown_shared_overrides)
        )

    shared_vector = np.asarray(
        [
            shared_overrides.get(spec.canonical, default_shared_values[spec.canonical])
            for spec in layout.shared_specs
        ],
        dtype=float,
    )

    return ActiveState(
        frame=jnp.asarray(frame_matrix, dtype=float),
        shared=jnp.asarray(shared_vector, dtype=float),
    )


def _build_runtime_overrides(
    *,
    reference_store: ParameterStore,
    key_specs: tuple[ActiveKeySpec, ...],
    values: jnp.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build primitive/derived override payloads from active values.

    This helper is intentionally local because it is tuned for JAX-traceable
    runtime application of config-defined active keys rather than one-off config
    validation. It is a good candidate for reuse if another sub-block inference
    recipe needs the same frame/shared store-update semantics.
    """

    primitive_overrides: dict[str, Any] = {}
    derived_overrides: dict[str, Any] = {}

    for idx, key_spec in enumerate(key_specs):
        value = values[idx]
        target = primitive_overrides if key_spec.kind == "primitive" else derived_overrides

        if key_spec.address.index is None:
            target[key_spec.address.base_key] = value
            continue

        if key_spec.address.base_key in target:
            vector_value = jnp.asarray(target[key_spec.address.base_key])
        else:
            vector_value = jnp.asarray(reference_store.get(key_spec.address.base_key))
        target[key_spec.address.base_key] = vector_value.at[key_spec.address.index].set(value)

    return primitive_overrides, derived_overrides


def _apply_runtime_active_values(
    *,
    reference_store: ParameterStore,
    forward_spec,
    key_specs: tuple[ActiveKeySpec, ...],
    values: jnp.ndarray,
) -> ParameterStore:
    """Apply a frame or shared active-value vector to a reference store."""

    if not key_specs:
        return reference_store

    primitive_overrides, derived_overrides = _build_runtime_overrides(
        reference_store=reference_store,
        key_specs=key_specs,
        values=values,
    )
    return apply_obs_subblock_overrides_preserving_derived(
        reference_store,
        forward_spec=forward_spec,
        primitive_overrides=primitive_overrides,
        derived_overrides=derived_overrides,
    )


def _build_variance_cube(
    *,
    data_cube: np.ndarray,
    noise_model_cfg: dict[str, Any],
) -> np.ndarray:
    """Build the Gaussian variance cube requested by ``objective.noise_model``."""

    variance_model = str(noise_model_cfg["variance_model"])
    if variance_model == "data":
        variance_cube = np.asarray(data_cube, dtype=float)
        if not np.all(np.isfinite(variance_cube)):
            raise ValueError(
                "Variance model 'data' produced non-finite values in observation cube."
            )
        floor = 1e-9
        if np.any(variance_cube <= floor):
            clipped = int(np.count_nonzero(variance_cube <= floor))
            total = int(variance_cube.size)
            print(
                "Warning: objective.noise_model.variance_model='data' encountered "
                f"{clipped}/{total} non-positive or near-zero variance values; "
                f"clipping to {floor}."
            )
            variance_cube = np.maximum(variance_cube, floor)
        return variance_cube

    if variance_model == "scalar":
        scalar = noise_model_cfg.get("scalar")
        if scalar is None:
            raise ValueError(
                "objective.noise_model.scalar is required when variance_model='scalar'."
            )
        scalar_value = float(scalar)
        if scalar_value <= 0.0:
            raise ValueError("objective.noise_model.scalar must be > 0.")
        return np.full_like(data_cube, scalar_value, dtype=float)

    raise ValueError(f"Unsupported objective.noise_model.variance_model: {variance_model!r}")


def _build_prior_term_fn(
    priors_cfg: dict[str, Any],
) -> Callable[[ActiveState], jnp.ndarray]:
    """Build the prior contribution to the block objective.

    The current implementation is a deliberate no-op for empty prior blocks, but
    the objective is still assembled as separate data/prior/temporal terms so
    future prior support does not require a recipe rewrite.
    """

    frame_priors = _optional_dict(priors_cfg, "frame", path="experiment.inference.priors")
    shared_priors = _optional_dict(priors_cfg, "shared", path="experiment.inference.priors")
    if frame_priors or shared_priors:
        raise ValueError(
            "Non-empty experiment.inference.priors is not implemented yet."
        )

    def _prior_term(_state: ActiveState) -> jnp.ndarray:
        return jnp.array(0.0, dtype=float)

    return _prior_term


def _build_temporal_term_fn(
    temporal_cfg: dict[str, Any],
) -> Callable[[ActiveState], jnp.ndarray]:
    """Build the temporal contribution to the block objective."""

    frame_model_cfg = _required_dict(
        temporal_cfg,
        "frame_model",
        path="experiment.inference.temporal",
    )
    frame_model_kind = _required_str(
        frame_model_cfg,
        "kind",
        path="experiment.inference.temporal.frame_model",
    )
    if frame_model_kind != "independent":
        raise ValueError(
            "experiment.inference.temporal.frame_model.kind "
            f"{frame_model_kind!r} is not implemented yet."
        )
    if set(frame_model_cfg) != {"kind"}:
        raise ValueError(
            "Independent frame_model currently supports only the 'kind' field."
        )

    def _temporal_term(_state: ActiveState) -> jnp.ndarray:
        return jnp.array(0.0, dtype=float)

    return _temporal_term


def _build_objective_bundle(
    *,
    layout: ActiveStateLayout,
    binder: SheraBinder,
    forward_spec,
    base_store: ParameterStore,
    cube_data: np.ndarray,
    variance_cube: np.ndarray,
    objective_cfg: dict[str, Any],
    priors_cfg: dict[str, Any],
    temporal_cfg: dict[str, Any],
) -> ObjectiveBundle:
    """Build compiled prediction and objective helpers for the active state."""

    objective_kind = str(objective_cfg["kind"])
    if objective_kind != "nll":
        raise ValueError(
            f"Unsupported experiment.inference.objective.kind: {objective_kind!r}."
        )

    noise_model_cfg = _required_dict(
        objective_cfg,
        "noise_model",
        path="experiment.inference.objective",
    )
    noise_model_kind = str(noise_model_cfg["kind"])
    if noise_model_kind != "gaussian":
        raise ValueError(
            "Unsupported experiment.inference.objective.noise_model.kind: "
            f"{noise_model_kind!r}."
        )

    reduce = str(objective_cfg["reduce"])
    if reduce not in {"sum", "mean"}:
        raise ValueError(
            "experiment.inference.objective.reduce must be 'sum' or 'mean'."
        )

    data_cube = jnp.asarray(cube_data)
    var_cube = jnp.asarray(variance_cube)
    prior_term_fn = _build_prior_term_fn(priors_cfg)
    temporal_term_fn = _build_temporal_term_fn(temporal_cfg)

    def _shared_store(shared_values: jnp.ndarray) -> ParameterStore:
        return _apply_runtime_active_values(
            reference_store=base_store,
            forward_spec=forward_spec,
            key_specs=layout.shared_specs,
            values=shared_values,
        )

    def _frame_model(shared_store: ParameterStore, frame_values: jnp.ndarray) -> jnp.ndarray:
        frame_store = _apply_runtime_active_values(
            reference_store=shared_store,
            forward_spec=forward_spec,
            key_specs=layout.frame_specs,
            values=frame_values,
        )
        frame_delta = binder.strip_structural(frame_store)
        return binder.model(frame_delta)

    def _predict_cube_from_state(state: ActiveState) -> jnp.ndarray:
        shared_store = _shared_store(state.shared)
        return jax.vmap(lambda frame_values: _frame_model(shared_store, frame_values))(
            state.frame
        )

    def _per_frame_data_terms_from_state(state: ActiveState) -> jnp.ndarray:
        shared_store = _shared_store(state.shared)

        def _frame_loss(
            frame_values: jnp.ndarray,
            data_frame: jnp.ndarray,
            var_frame: jnp.ndarray,
        ) -> jnp.ndarray:
            model_frame = _frame_model(shared_store, frame_values)
            return gaussian_image_nll(model_frame, data_frame, var_frame, reduce=reduce)

        return jax.vmap(_frame_loss)(state.frame, data_cube, var_cube)

    def _objective_terms(theta_flat: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        state = _unpack_active_state(layout, theta_flat)
        data_term = jnp.sum(_per_frame_data_terms_from_state(state))
        prior_term = prior_term_fn(state)
        temporal_term = temporal_term_fn(state)
        return data_term, prior_term, temporal_term

    def _total_loss(theta_flat: jnp.ndarray) -> jnp.ndarray:
        data_term, prior_term, temporal_term = _objective_terms(theta_flat)
        return data_term + prior_term + temporal_term

    def _predict_cube(theta_flat: jnp.ndarray) -> jnp.ndarray:
        state = _unpack_active_state(layout, theta_flat)
        return _predict_cube_from_state(state)

    def _per_frame_terms(theta_flat: jnp.ndarray) -> jnp.ndarray:
        state = _unpack_active_state(layout, theta_flat)
        return _per_frame_data_terms_from_state(state)

    return ObjectiveBundle(
        total_loss_fn=_total_loss,
        objective_terms_fn=_objective_terms,
        predict_cube_fn=_predict_cube,
        per_frame_data_terms_fn=_per_frame_terms,
    )


def _coerce_lr_clip(
    value: Any,
    *,
    path: str,
) -> tuple[float, float] | None:
    """Validate an optional [min, max] learning-rate clip tuple."""

    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{path} must be [min, max] when provided.")
    lr_min, lr_max = value
    if (
        isinstance(lr_min, bool)
        or isinstance(lr_max, bool)
        or not isinstance(lr_min, (int, float))
        or not isinstance(lr_max, (int, float))
    ):
        raise ValueError(f"{path} values must be numeric.")
    lr_min_f = float(lr_min)
    lr_max_f = float(lr_max)
    if lr_min_f <= 0.0:
        raise ValueError(f"{path}[0] must be > 0.")
    if lr_max_f < lr_min_f:
        raise ValueError(f"{path}[1] must be >= {path}[0].")
    return (lr_min_f, lr_max_f)


def _build_theta_preconditioning_bundle(
    *,
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    theta_ref: jnp.ndarray,
    base_lr: float,
    cfg: dict[str, Any],
) -> ThetaPreconditioningBundle:
    """Build a damped theta-space preconditioner from the full packed FIM."""

    damping = float(cfg.get("damping", 1e-6))
    eig_floor_rel = float(cfg.get("eig_floor_rel", 1e-6))
    eig_floor_abs = float(cfg.get("eig_floor_abs", 1e-8))
    lr_clip = _coerce_lr_clip(
        cfg.get("lr_clip"),
        path="experiment.inference.optimizer.preconditioning.lr_clip",
    )

    if damping < 0.0:
        raise ValueError("experiment.inference.optimizer.preconditioning.damping must be >= 0.")
    if eig_floor_rel < 0.0:
        raise ValueError(
            "experiment.inference.optimizer.preconditioning.eig_floor_rel must be >= 0."
        )
    if eig_floor_abs < 0.0:
        raise ValueError(
            "experiment.inference.optimizer.preconditioning.eig_floor_abs must be >= 0."
        )

    theta_ref_vec = jnp.asarray(theta_ref)
    fim = np.asarray(fim_theta(loss_fn, theta_ref_vec), dtype=float)
    if fim.ndim != 2 or fim.shape[0] != fim.shape[1]:
        raise ValueError("Theta-space curvature must be a square matrix.")
    if fim.shape[0] != int(theta_ref_vec.size):
        raise ValueError("Theta-space curvature dimension must match packed theta dimension.")
    if not np.all(np.isfinite(fim)):
        raise ValueError("Theta-space curvature contains non-finite values.")

    fim_sym = 0.5 * (fim + fim.T)
    eigvals, eigvecs = np.linalg.eigh(fim_sym)
    eig_scale = max(float(np.max(np.abs(eigvals))), 1.0)
    eig_floor = max(eig_floor_abs, eig_floor_rel * eig_scale)
    eigvals_stable = np.maximum(eigvals, eig_floor) + damping

    inv_sqrt_eigs = np.reciprocal(np.sqrt(eigvals_stable))
    precond_full = eigvecs @ (inv_sqrt_eigs[:, None] * eigvecs.T)
    preconditioner_diag = np.diag(precond_full)
    # Convention: lr_vec is a pure preconditioning scale. run_shera_gd applies
    # base_lr via its learning_rate argument, matching canonical theta workflows.
    _ = base_lr
    lr_vec_unclipped = np.array(preconditioner_diag, copy=True)
    lr_vec = np.array(lr_vec_unclipped, copy=True)
    lr_clip_applied_count = 0
    if lr_clip is not None:
        lr_clip_applied_count = int(
            np.count_nonzero((lr_vec < lr_clip[0]) | (lr_vec > lr_clip[1]))
        )
        lr_vec = np.clip(lr_vec, lr_clip[0], lr_clip[1])

    if lr_vec.shape != (int(theta_ref_vec.size),):
        raise ValueError("Preconditioning vector shape does not match packed theta dimension.")
    if not np.all(np.isfinite(lr_vec)):
        raise ValueError("Preconditioning vector contains non-finite values.")
    if np.any(lr_vec <= 0.0):
        raise ValueError("Preconditioning vector must be strictly positive.")

    return ThetaPreconditioningBundle(
        fim=fim_sym,
        eigvals=eigvals,
        eigvals_stable=eigvals_stable,
        preconditioner_diag=preconditioner_diag,
        lr_vec_unclipped=lr_vec_unclipped,
        lr_vec=lr_vec,
        config={
            "enabled": True,
            "damping": damping,
            "eig_floor_rel": eig_floor_rel,
            "eig_floor_abs": eig_floor_abs,
            "eig_floor_effective": eig_floor,
            "lr_clip": None if lr_clip is None else [float(lr_clip[0]), float(lr_clip[1])],
            "lr_clip_applied_count": lr_clip_applied_count,
        },
    )


def _format_scalar(value: float) -> str:
    """Format console diagnostics with compact scientific notation."""

    return f"{float(value):.6g}"


def _format_array_stats(values: np.ndarray) -> str:
    """Return a compact min/median/max summary for finite numeric values."""

    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return "empty"
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "no finite values"
    return (
        f"min={_format_scalar(np.min(finite))} "
        f"median={_format_scalar(np.median(finite))} "
        f"max={_format_scalar(np.max(finite))}"
    )


def _condition_number_from_bounds(min_value: float, max_value: float) -> float:
    """Return max/min when the lower bound is positive, otherwise inf."""

    if min_value <= 0.0:
        return float("inf")
    return float(max_value / min_value)


def _print_active_layout_summary(*, layout: ActiveStateLayout, theta0: jnp.ndarray) -> None:
    """Print active-state packing details before optimization."""

    theta0_np = np.asarray(theta0, dtype=float)
    print("Active inference layout:")
    print(
        f"  frames={layout.n_frame} frame_width={layout.frame_width} "
        f"shared_width={layout.shared_width} theta_size={layout.theta_size}"
    )
    print(f"  frame_keys={list(layout.frame_keys)}")
    print(f"  shared_keys={list(layout.shared_keys)}")
    print(f"  theta0: {_format_array_stats(theta0_np)}")


def _print_preconditioning_summary(
    *,
    bundle: ThetaPreconditioningBundle,
    base_lr: float,
) -> None:
    """Print the numerical details of the theta-space FIM preconditioner."""

    fim_diag = np.diag(bundle.fim)
    fim_abs = np.abs(bundle.fim)
    stable_min = float(np.min(bundle.eigvals_stable))
    stable_max = float(np.max(bundle.eigvals_stable))
    eig_nonpositive = int(np.count_nonzero(bundle.eigvals <= 0.0))
    eig_floor = float(bundle.config["eig_floor_effective"])
    eig_floored = int(np.count_nonzero(bundle.eigvals < eig_floor))
    stable_condition = _condition_number_from_bounds(stable_min, stable_max)
    effective_lr = float(base_lr) * bundle.lr_vec
    effective_lr_unclipped = float(base_lr) * bundle.lr_vec_unclipped

    print("Theta-space Fisher preconditioning summary:")
    print(
        f"  theta_dim={bundle.lr_vec.size} fim_shape={bundle.fim.shape} "
        f"base_lr={_format_scalar(base_lr)}"
    )
    print(
        "  config: "
        f"damping={_format_scalar(bundle.config['damping'])} "
        f"eig_floor_rel={_format_scalar(bundle.config['eig_floor_rel'])} "
        f"eig_floor_abs={_format_scalar(bundle.config['eig_floor_abs'])} "
        f"eig_floor_effective={_format_scalar(eig_floor)} "
        f"lr_clip={bundle.config['lr_clip']}"
    )
    print(
        "  fim: "
        f"diag({_format_array_stats(fim_diag)}) "
        f"trace={_format_scalar(np.trace(bundle.fim))} "
        f"max_abs_entry={_format_scalar(np.max(fim_abs))}"
    )
    print(
        "  raw eigenvalues: "
        f"{_format_array_stats(bundle.eigvals)} "
        f"nonpositive={eig_nonpositive} floor_lifted={eig_floored}"
    )
    print(
        "  stabilized eigenvalues: "
        f"{_format_array_stats(bundle.eigvals_stable)} "
        f"condition_est={_format_scalar(stable_condition)}"
    )
    print(
        "  preconditioner-like diagonal used to build lr_vec: "
        f"{_format_array_stats(bundle.preconditioner_diag)}"
    )
    print(
        "  lr_vec before clipping: "
        f"{_format_array_stats(bundle.lr_vec_unclipped)} "
        "(preconditioning scale only)"
    )
    print(
        "  lr_vec after clipping: "
        f"{_format_array_stats(bundle.lr_vec)} "
        "(preconditioning scale only) "
        f"clipped={bundle.config['lr_clip_applied_count']}"
    )
    print(
        "  implied effective SGD scale: "
        f"before_clip({_format_array_stats(effective_lr_unclipped)}) "
        f"after_clip({_format_array_stats(effective_lr)})"
    )


def _label_for_key(key: str) -> str:
    """Return a readable axis label for one canonical active key."""

    return key


def _to_jsonable_float_list(values: np.ndarray) -> list[float]:
    """Convert a numeric array to a JSON-friendly flat float list."""

    return [float(value) for value in np.asarray(values, dtype=float).ravel()]


def _optimizer_first_step(
    *,
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    theta0: jnp.ndarray,
    learning_rate: float,
    lr_vec: np.ndarray | None,
    optimizer_kind: str,
    optimizer_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate loss/grad at theta0 and simulate the optimizer's first update."""

    theta0_arr = jnp.asarray(theta0)
    loss0, grad0 = jax.value_and_grad(loss_fn)(theta0_arr)
    opt_kwargs = dict(optimizer_kwargs)

    def _scale_by_vector(vec: jnp.ndarray) -> optax.GradientTransformation:
        def init_fn(_params):
            return None

        def update_fn(updates, state, params=None):
            del params
            return jax.tree_util.tree_map(lambda g: g * vec, updates), state

        return optax.GradientTransformation(init_fn, update_fn)

    if optimizer_kind == "sgd":
        if lr_vec is None:
            optimizer = optax.sgd(learning_rate=learning_rate, **opt_kwargs)
        else:
            optimizer = optax.sgd(
                learning_rate=learning_rate * jnp.asarray(lr_vec),
                **opt_kwargs,
            )
    elif optimizer_kind == "adam":
        txs: list[optax.GradientTransformation] = [optax.scale_by_adam(**opt_kwargs)]
        if lr_vec is not None:
            txs.append(_scale_by_vector(jnp.asarray(lr_vec)))
        txs.append(optax.scale(-learning_rate))
        optimizer = optax.chain(*txs)
    else:
        raise ValueError(f"Unsupported optimizer kind for diagnostics: {optimizer_kind!r}.")

    opt_state = optimizer.init(theta0_arr)
    updates, _ = optimizer.update(grad0, opt_state, params=theta0_arr)
    theta1 = optax.apply_updates(theta0_arr, updates)
    loss1 = loss_fn(theta1)
    return {
        "theta0": np.asarray(theta0_arr, dtype=float),
        "theta1": np.asarray(theta1, dtype=float),
        "loss0": float(np.asarray(loss0)),
        "loss1": float(np.asarray(loss1)),
        "grad0": np.asarray(grad0, dtype=float),
        "delta0": np.asarray(theta1 - theta0_arr, dtype=float),
        "lr_vec": None if lr_vec is None else np.asarray(lr_vec, dtype=float),
    }


def _top_labeled_entries(
    values: np.ndarray,
    *,
    labels: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    """Return the largest entries by absolute value with theta labels."""

    arr = np.asarray(values, dtype=float).ravel()
    if arr.size != len(labels):
        raise ValueError("Diagnostic vector length does not match theta label count.")
    k = min(max(int(top_k), 0), arr.size)
    if k == 0:
        return []
    idx = np.argsort(np.abs(arr))[::-1][:k]
    return [
        {
            "index": int(i),
            "label": labels[int(i)],
            "value": float(arr[int(i)]),
            "abs_value": float(abs(arr[int(i)])),
        }
        for i in idx
    ]


def _summarize_first_step(
    *,
    label: str,
    step: dict[str, Any],
    theta_labels: list[str],
    top_k: int,
) -> dict[str, Any]:
    """Build and print a readable first-step diagnostic summary."""

    theta0 = np.asarray(step["theta0"], dtype=float)
    theta1 = np.asarray(step["theta1"], dtype=float)
    grad0 = np.asarray(step["grad0"], dtype=float)
    delta0 = np.asarray(step["delta0"], dtype=float)
    lr_vec = step["lr_vec"]
    payload: dict[str, Any] = {
        "label": label,
        "loss0": float(step["loss0"]),
        "loss1": float(step["loss1"]),
        "loss0_finite": bool(np.isfinite(step["loss0"])),
        "loss1_finite": bool(np.isfinite(step["loss1"])),
        "grad0_finite": bool(np.all(np.isfinite(grad0))),
        "theta0_finite": bool(np.all(np.isfinite(theta0))),
        "theta1_finite": bool(np.all(np.isfinite(theta1))),
        "grad0_min": float(np.min(grad0)),
        "grad0_max": float(np.max(grad0)),
        "delta0_min": float(np.min(delta0)),
        "delta0_max": float(np.max(delta0)),
        "lr_vec_min": None if lr_vec is None else float(np.min(lr_vec)),
        "lr_vec_max": None if lr_vec is None else float(np.max(lr_vec)),
        "top_grad_abs": _top_labeled_entries(grad0, labels=theta_labels, top_k=top_k),
        "top_delta_abs": _top_labeled_entries(delta0, labels=theta_labels, top_k=top_k),
    }

    print(f"First-step diagnostic [{label}]:")
    print(
        f"  loss0={_format_scalar(payload['loss0'])} "
        f"finite={payload['loss0_finite']} | "
        f"loss1={_format_scalar(payload['loss1'])} "
        f"finite={payload['loss1_finite']}"
    )
    print(
        f"  grad_finite={payload['grad0_finite']} "
        f"theta0_finite={payload['theta0_finite']} "
        f"theta1_finite={payload['theta1_finite']}"
    )
    print(
        f"  grad0 min/max={payload['grad0_min']:.3e}/{payload['grad0_max']:.3e} | "
        f"delta0 min/max={payload['delta0_min']:.3e}/{payload['delta0_max']:.3e}"
    )
    if lr_vec is not None:
        print(
            f"  lr_vec scale min/max={payload['lr_vec_min']:.3e}/"
            f"{payload['lr_vec_max']:.3e}"
        )
    top_grad = ", ".join(
        f"{item['label']}={item['value']:.3e}" for item in payload["top_grad_abs"]
    )
    top_delta = ", ".join(
        f"{item['label']}={item['value']:.3e}" for item in payload["top_delta_abs"]
    )
    print(f"  top |grad0|: {top_grad}")
    print(f"  top |delta0|: {top_delta}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write an indented JSON diagnostic artifact."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _save_fim_debug_artifact(
    *,
    bundle: ThetaPreconditioningBundle,
    theta_labels: list[str],
    base_lr: float,
    output_path: Path,
) -> None:
    """Persist full theta-space preconditioning introspection data."""

    _write_json(
        output_path,
        {
            "schema_version": "subblock_theta_preconditioning_debug.v1",
            "description": {
                "fim": "curvature-like full theta-space Fisher information matrix",
                "fim_diagonal": "curvature-like diagonal of fim",
                "eigenvalues": "raw curvature-like eigenvalues of symmetrized fim",
                "stabilized_eigenvalues": "floored and damped curvature-like eigenvalues",
                "preconditioner_diagonal": (
                    "preconditioner-like diagonal used to build lr_vec"
                ),
                "lr_vec_before_clipping": (
                    "preconditioning scale vector before clipping; excludes base_lr"
                ),
                "lr_vec_after_clipping": (
                    "preconditioning scale vector passed to run_shera_gd; excludes base_lr"
                ),
            },
            "theta_labels": list(theta_labels),
            "config": to_jsonable_obs_subblock_payload(bundle.config),
            "fim": np.asarray(bundle.fim, dtype=float).tolist(),
            "fim_diagonal": _to_jsonable_float_list(np.diag(bundle.fim)),
            "eigenvalues": _to_jsonable_float_list(bundle.eigvals),
            "stabilized_eigenvalues": _to_jsonable_float_list(bundle.eigvals_stable),
            "preconditioner_diagonal": _to_jsonable_float_list(
                bundle.preconditioner_diag
            ),
            "base_lr": float(base_lr),
            "lr_scale_before_clipping": _to_jsonable_float_list(bundle.lr_vec_unclipped),
            "lr_scale_after_clipping": _to_jsonable_float_list(bundle.lr_vec),
            "effective_lr_before_clipping": _to_jsonable_float_list(
                float(base_lr) * bundle.lr_vec_unclipped
            ),
            "effective_lr_after_clipping": _to_jsonable_float_list(
                float(base_lr) * bundle.lr_vec
            ),
            "lr_vec_before_clipping": _to_jsonable_float_list(bundle.lr_vec_unclipped),
            "lr_vec_after_clipping": _to_jsonable_float_list(bundle.lr_vec),
        },
    )


def _per_frame_diagnostic_rows(
    *,
    per_frame_terms: np.ndarray,
    label: str,
) -> list[dict[str, Any]]:
    """Return JSON rows for per-frame data-term diagnostics."""

    terms = np.asarray(per_frame_terms, dtype=float).ravel()
    total = float(np.sum(terms))
    return [
        {
            "step": label,
            "frame_index": int(idx),
            "data_term": float(value),
            "fraction_of_step_total": None if total == 0.0 else float(value / total),
        }
        for idx, value in enumerate(terms)
    ]


def _print_per_frame_data_terms(
    *,
    label: str,
    terms: np.ndarray,
) -> None:
    """Print a compact per-frame data-term summary."""

    arr = np.asarray(terms, dtype=float).ravel()
    if arr.size == 0:
        print(f"Per-frame data terms [{label}]: empty")
        return
    dominant_idx = int(np.argmax(arr))
    total = float(np.sum(arr))
    print(
        f"Per-frame data terms [{label}]: total={_format_scalar(total)} "
        f"min={_format_scalar(np.min(arr))} median={_format_scalar(np.median(arr))} "
        f"max={_format_scalar(np.max(arr))} dominant_frame={dominant_idx}"
    )


def _finite_difference_gradient_check(
    *,
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    theta0: jnp.ndarray,
    grad0: np.ndarray,
    theta_labels: list[str],
) -> list[dict[str, Any]]:
    """Compare autodiff gradient signs with central finite differences."""

    theta0_np = np.asarray(theta0, dtype=float).ravel()
    grad0_np = np.asarray(grad0, dtype=float).ravel()
    rows: list[dict[str, Any]] = []
    for idx, value in enumerate(theta0_np):
        eps = max(1e-6, abs(float(value)) * 1e-4)
        direction = np.zeros_like(theta0_np)
        direction[idx] = eps
        loss_plus = float(np.asarray(loss_fn(jnp.asarray(theta0_np + direction))))
        loss_minus = float(np.asarray(loss_fn(jnp.asarray(theta0_np - direction))))
        fd_grad = (loss_plus - loss_minus) / (2.0 * eps)
        autodiff_grad = float(grad0_np[idx])
        sign_consistent = (
            abs(autodiff_grad) < 1e-12
            or abs(fd_grad) < 1e-12
            or np.sign(autodiff_grad) == np.sign(fd_grad)
        )
        rows.append(
            {
                "index": int(idx),
                "label": theta_labels[idx],
                "theta0": float(value),
                "eps": float(eps),
                "autodiff_grad": autodiff_grad,
                "finite_difference_grad": float(fd_grad),
                "sign_consistent": bool(sign_consistent),
                "loss_plus": loss_plus,
                "loss_minus": loss_minus,
            }
        )
    consistent = sum(1 for row in rows if row["sign_consistent"])
    print(
        "Finite-difference gradient sign check: "
        f"{consistent}/{len(rows)} dimensions locally consistent"
    )
    return rows


def _plot_theta_bar(
    *,
    values: np.ndarray,
    labels: list[str],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot a labeled bar chart for one theta diagnostic vector."""

    arr = np.asarray(values, dtype=float).ravel()
    fig_width = max(7.0, 0.45 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ax.bar(np.arange(arr.size), np.abs(arr))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(np.arange(arr.size))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_per_frame_comparison(
    *,
    theta0_terms: np.ndarray,
    theta1_terms: np.ndarray,
    output_path: Path,
) -> None:
    """Plot per-frame data-term contributions before and after step 1."""

    terms0 = np.asarray(theta0_terms, dtype=float).ravel()
    terms1 = np.asarray(theta1_terms, dtype=float).ravel()
    x = np.arange(terms0.size)
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, terms0, width=width, label="theta0")
    ax.bar(x + width / 2, terms1, width=width, label="theta1")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Data-term contribution")
    ax.set_title("Per-frame objective contribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_frame_history_from_theta_trace(
    *,
    layout: ActiveStateLayout,
    theta0: jnp.ndarray,
    theta_trace: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode optimizer theta history into iteration-by-frame-by-key values."""

    theta0_np = np.asarray(theta0, dtype=float).ravel()
    trace_np = np.asarray(theta_trace, dtype=float)
    if trace_np.ndim == 1:
        trace_np = trace_np.reshape((1, trace_np.size))
    if trace_np.ndim != 2:
        raise ValueError("Optimizer theta trace must be a 2D array.")
    if trace_np.shape[1] != theta0_np.size:
        raise ValueError("Optimizer theta trace width does not match theta0 size.")

    if trace_np.shape[0] == 0:
        theta_history = theta0_np[None, :]
    elif np.allclose(trace_np[0], theta0_np, rtol=0.0, atol=1e-12):
        theta_history = trace_np
    else:
        theta_history = np.vstack((theta0_np[None, :], trace_np))

    frame_history = np.asarray(
        [
            np.asarray(
                _unpack_active_state(layout, jnp.asarray(theta_values)).frame,
                dtype=float,
            )
            for theta_values in theta_history
        ],
        dtype=float,
    )
    iterations = np.arange(theta_history.shape[0], dtype=int)
    return iterations, frame_history


def _plot_parameter_history_heatmaps(
    *,
    frame_history: np.ndarray,
    labels: tuple[str, ...],
    output_path: Path,
    title: str,
    colorbar_label: str,
    center_zero: bool = False,
) -> None:
    """Plot frame-varying parameter history as one heatmap per active key."""

    history = np.asarray(frame_history, dtype=float)
    if history.ndim != 3 or history.shape[2] == 0:
        return

    n_step, n_frame, n_key = history.shape
    fig, axes = plt.subplots(
        n_key,
        1,
        figsize=(8, max(3.2, 2.4 * n_key)),
        sharex=True,
        squeeze=False,
    )
    x_extent = (-0.5, max(n_step - 0.5, 0.5))
    extent = (x_extent[0], x_extent[1], -0.5, max(n_frame - 0.5, 0.5))

    for key_index, ax in enumerate(axes[:, 0]):
        values = history[:, :, key_index].T
        cmap = "viridis"
        vmin = vmax = None
        if center_zero:
            finite_abs = np.abs(values[np.isfinite(values)])
            limit = float(np.max(finite_abs)) if finite_abs.size else 0.0
            if not np.isfinite(limit) or limit <= 0.0:
                limit = 1.0
            vmin = -limit
            vmax = limit
            cmap = "RdBu_r"
        image = ax.imshow(
            values,
            aspect="auto",
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylabel("Frame index")
        ax.set_title(_label_for_key(labels[key_index]))
        fig.colorbar(image, ax=ax, label=colorbar_label, fraction=0.025, pad=0.02)

    axes[-1, 0].set_xlabel("Optimizer iteration")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_parameter_history_lines(
    *,
    iterations: np.ndarray,
    frame_history: np.ndarray,
    labels: tuple[str, ...],
    output_path: Path,
    title: str = "Frame-varying active-state optimizer history",
    ylabel_suffix: str = "",
    zero_line: bool = False,
) -> None:
    """Plot frame-wise parameter trajectories as compact line panels."""

    history = np.asarray(frame_history, dtype=float)
    if history.ndim != 3 or history.shape[2] == 0:
        return

    n_step, n_frame, n_key = history.shape
    fig, axes = plt.subplots(
        n_key,
        1,
        figsize=(8, max(3.5, 2.5 * n_key)),
        sharex=True,
        squeeze=False,
    )
    for key_index, ax in enumerate(axes[:, 0]):
        for frame_index in range(n_frame):
            ax.plot(
                iterations,
                history[:, frame_index, key_index],
                linewidth=1.0,
                marker="o" if n_step <= 20 else None,
                markersize=2.5,
                label=f"frame {frame_index}",
            )
        if zero_line:
            ax.axhline(0.0, color="k", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axvline(
            int(iterations[-1]),
            color="k",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )
        ax.set_ylabel(f"{_label_for_key(labels[key_index])}{ylabel_suffix}")
        ax.grid(alpha=0.3)
        if n_frame <= 10:
            ax.legend(loc="best", fontsize="small", ncol=min(n_frame, 4))

    axes[-1, 0].set_xlabel("Optimizer iteration")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_loss_history(*, losses: np.ndarray, output_path: Path) -> None:
    """Plot scalar objective history over optimizer iterations."""

    fig, ax = plt.subplots(figsize=(7, 4))
    iterations = np.arange(losses.shape[0], dtype=int)
    ax.plot(iterations, losses, marker="o", linewidth=1.0, markersize=3.0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Block objective")
    ax.set_title("Observation sub-block inference loss history")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_recovered_traces(
    *,
    times: np.ndarray,
    recovered: np.ndarray,
    labels: tuple[str, ...],
    output_path: Path,
) -> None:
    """Plot recovered frame-varying active terms for any frame-key list."""

    if recovered.shape[1] == 0:
        return

    fig, axes = plt.subplots(
        recovered.shape[1],
        1,
        figsize=(8, max(4, 2.5 * recovered.shape[1])),
        sharex=True,
    )
    axes_arr = np.atleast_1d(axes)
    for idx, ax in enumerate(axes_arr):
        ax.plot(times, recovered[:, idx], marker="o", linewidth=1.0, label="recovered")
        ax.set_ylabel(_label_for_key(labels[idx]))
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    axes_arr[-1].set_xlabel("time_s")
    fig.suptitle("Recovered frame-varying active state")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_trace_comparison(
    *,
    times: np.ndarray,
    recovered: np.ndarray,
    truth: np.ndarray,
    labels: tuple[str, ...],
    output_path: Path,
) -> None:
    """Plot recovered versus truth trace series for any frame-key list."""

    if recovered.shape[1] == 0:
        return

    fig, axes = plt.subplots(
        recovered.shape[1],
        1,
        figsize=(8, max(4, 2.5 * recovered.shape[1])),
        sharex=True,
    )
    axes_arr = np.atleast_1d(axes)
    for idx, ax in enumerate(axes_arr):
        ax.plot(times, recovered[:, idx], marker="o", linewidth=1.0, label="recovered")
        ax.plot(times, truth[:, idx], marker="x", linewidth=1.0, label="truth")
        ax.set_ylabel(_label_for_key(labels[idx]))
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    axes_arr[-1].set_xlabel("time_s")
    fig.suptitle("Recovered vs truth frame-varying active state")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_trace_residuals(
    *,
    times: np.ndarray,
    recovered: np.ndarray,
    truth: np.ndarray,
    labels: tuple[str, ...],
    output_path: Path,
) -> None:
    """Plot recovered minus truth residuals for any frame-key list."""

    if recovered.shape[1] == 0:
        return

    residual = recovered - truth
    fig, axes = plt.subplots(
        residual.shape[1],
        1,
        figsize=(8, max(4, 2.5 * residual.shape[1])),
        sharex=True,
    )
    axes_arr = np.atleast_1d(axes)
    for idx, ax in enumerate(axes_arr):
        ax.plot(times, residual[:, idx], marker="o", linewidth=1.0)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_ylabel(f"{_label_for_key(labels[idx])} residual")
        ax.grid(alpha=0.3)
    axes_arr[-1].set_xlabel("time_s")
    fig.suptitle("Recovered minus truth residual traces")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_image_fit(
    *,
    data_cube: np.ndarray,
    model_cube: np.ndarray,
    output_path: Path,
) -> None:
    """Plot representative data/model/residual image panels."""

    n_frame = int(data_cube.shape[0])
    sample_indices = [0, n_frame // 2, n_frame - 1]
    deduped: list[int] = []
    for idx in sample_indices:
        if idx not in deduped:
            deduped.append(idx)
    sample_indices = deduped

    vmin = float(np.percentile(data_cube, 1.0))
    vmax = float(np.percentile(data_cube, 99.0))
    residual_cube = data_cube - model_cube
    rv = float(np.max(np.abs(residual_cube)))
    if not np.isfinite(rv) or rv <= 0.0:
        rv = 1.0

    fig, axes = plt.subplots(len(sample_indices), 3, figsize=(10, 3.5 * len(sample_indices)))
    if len(sample_indices) == 1:
        axes = np.asarray([axes])

    for row, frame_index in enumerate(sample_indices):
        ax_data, ax_model, ax_resid = axes[row]
        im_data = ax_data.imshow(data_cube[frame_index], cmap="inferno", vmin=vmin, vmax=vmax)
        im_model = ax_model.imshow(
            model_cube[frame_index], cmap="inferno", vmin=vmin, vmax=vmax
        )
        im_resid = ax_resid.imshow(
            residual_cube[frame_index], cmap="RdBu_r", vmin=-rv, vmax=rv
        )
        ax_data.set_title(f"data frame {frame_index}")
        ax_model.set_title(f"model frame {frame_index}")
        ax_resid.set_title(f"residual frame {frame_index}")
        for ax in (ax_data, ax_model, ax_resid):
            ax.set_xlabel("x (pix)")
            ax.set_ylabel("y (pix)")
        fig.colorbar(im_data, ax=ax_data, fraction=0.046, pad=0.04)
        fig.colorbar(im_model, ax=ax_model, fraction=0.046, pad=0.04)
        fig.colorbar(im_resid, ax=ax_resid, fraction=0.046, pad=0.04)

    fig.suptitle("Data/model/residual image diagnostics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_truth_matrix(
    trace: ObsSubblockTrace,
    *,
    frame_keys: tuple[str, ...],
) -> np.ndarray:
    """Return a frame-by-key truth matrix aligned to ``frame_keys``."""

    return np.asarray(
        [
            [float(row[key]) for key in frame_keys]
            for row in trace.rows
        ],
        dtype=float,
    )


def _build_recovered_rows(
    *,
    layout: ActiveStateLayout,
    times: np.ndarray,
    frame_matrix: np.ndarray,
    frame_data_terms: np.ndarray,
) -> list[dict[str, Any]]:
    """Build recovered per-frame rows for CSV output."""

    rows: list[dict[str, Any]] = []
    for frame_index in range(layout.n_frame):
        row: dict[str, Any] = {
            "frame_index": int(frame_index),
            "time_s": float(times[frame_index]),
            "frame_nll": float(frame_data_terms[frame_index]),
        }
        for key_index, key in enumerate(layout.frame_keys):
            row[key] = float(frame_matrix[frame_index, key_index])
        rows.append(row)
    return rows


def _build_truth_comparison_rows(
    *,
    layout: ActiveStateLayout,
    times: np.ndarray,
    recovered_frame_matrix: np.ndarray,
    truth_matrix: np.ndarray,
    frame_data_terms: np.ndarray,
) -> list[dict[str, Any]]:
    """Build truth/recovered/residual rows aligned to the active frame keys."""

    rows: list[dict[str, Any]] = []
    for frame_index in range(layout.n_frame):
        row: dict[str, Any] = {
            "frame_index": int(frame_index),
            "time_s": float(times[frame_index]),
            "frame_nll": float(frame_data_terms[frame_index]),
        }
        for key_index, key in enumerate(layout.frame_keys):
            truth_value = float(truth_matrix[frame_index, key_index])
            recovered_value = float(recovered_frame_matrix[frame_index, key_index])
            row[f"{key}_truth"] = truth_value
            row[f"{key}_recovered"] = recovered_value
            row[f"{key}_residual"] = recovered_value - truth_value
        rows.append(row)
    return rows


def _validate_experiment_cfg(experiment_cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the experiment block for sub-block inference."""

    kind = _required_str(experiment_cfg, "kind", path="experiment")
    if kind != "subblock_inference":
        raise ValueError(
            "experiment.kind must be 'subblock_inference' for this recipe."
        )

    truth_cfg = experiment_cfg.get("truth")
    if truth_cfg is not None:
        if not isinstance(truth_cfg, dict):
            raise ValueError("experiment.truth must be a mapping/dict when provided.")
        if truth_cfg:
            raise ValueError(
                "experiment.truth overrides are not used by this recipe. "
                "Set assumed shared values directly under top-level system."
            )

    inference_cfg = _required_dict(experiment_cfg, "inference", path="experiment")

    data_cfg = _required_dict(inference_cfg, "data", path="experiment.inference")
    cube_path = _required_str(data_cfg, "cube", path="experiment.inference.data")
    truth_trace_path = _optional_str(
        data_cfg, "truth_trace", path="experiment.inference.data"
    )
    manifest_path = _optional_str(
        data_cfg, "manifest", path="experiment.inference.data"
    )

    validate_cfg = _optional_dict(
        inference_cfg,
        "validate",
        path="experiment.inference",
    )
    require_contiguous = validate_cfg.get(
        "require_contiguous_frame_index",
        TRACE_VALIDATE_DEFAULTS["require_contiguous_frame_index"],
    )
    require_monotonic = validate_cfg.get(
        "require_monotonic_time",
        TRACE_VALIDATE_DEFAULTS["require_monotonic_time"],
    )
    if not isinstance(require_contiguous, bool):
        raise ValueError(
            "experiment.inference.validate.require_contiguous_frame_index must be a bool."
        )
    if not isinstance(require_monotonic, bool):
        raise ValueError(
            "experiment.inference.validate.require_monotonic_time must be a bool."
        )

    active_cfg = _required_dict(inference_cfg, "active", path="experiment.inference")
    frame_keys = _required_list_of_str(
        active_cfg,
        "frame_keys",
        path="experiment.inference.active",
    )
    shared_keys = _required_list_of_str(
        active_cfg,
        "shared_keys",
        path="experiment.inference.active",
    )

    init_cfg = _required_dict(inference_cfg, "init", path="experiment.inference")
    init_frame_cfg = _required_dict(init_cfg, "frame", path="experiment.inference.init")
    init_frame_mode = _required_str(
        init_frame_cfg,
        "mode",
        path="experiment.inference.init.frame",
    )
    if init_frame_mode not in {"shared_guess", "from_system", "explicit_table", "from_truth_trace", "previous_block"}:
        raise ValueError(
            "Unsupported experiment.inference.init.frame.mode "
            f"{init_frame_mode!r}."
        )
    init_shared_cfg = _optional_dict(init_cfg, "shared", path="experiment.inference.init")

    priors_cfg = _optional_dict(inference_cfg, "priors", path="experiment.inference")
    priors_frame_cfg = _optional_dict(
        priors_cfg,
        "frame",
        path="experiment.inference.priors",
    )
    priors_shared_cfg = _optional_dict(
        priors_cfg,
        "shared",
        path="experiment.inference.priors",
    )

    temporal_cfg = _required_dict(inference_cfg, "temporal", path="experiment.inference")
    frame_model_cfg = _required_dict(
        temporal_cfg,
        "frame_model",
        path="experiment.inference.temporal",
    )
    frame_model_kind = _required_str(
        frame_model_cfg,
        "kind",
        path="experiment.inference.temporal.frame_model",
    )

    objective_cfg = _required_dict(inference_cfg, "objective", path="experiment.inference")
    objective_kind = _required_str(
        objective_cfg,
        "kind",
        path="experiment.inference.objective",
    )
    reduce = str(objective_cfg.get("reduce", "sum"))
    if reduce not in {"sum", "mean"}:
        raise ValueError(
            "experiment.inference.objective.reduce must be 'sum' or 'mean'."
        )
    noise_model_cfg = _required_dict(
        objective_cfg,
        "noise_model",
        path="experiment.inference.objective",
    )
    noise_model_kind = _required_str(
        noise_model_cfg,
        "kind",
        path="experiment.inference.objective.noise_model",
    )
    variance_model = _required_str(
        noise_model_cfg,
        "variance_model",
        path="experiment.inference.objective.noise_model",
    )
    scalar_variance = noise_model_cfg.get("scalar")
    if scalar_variance is not None:
        if isinstance(scalar_variance, bool) or not isinstance(scalar_variance, (int, float)):
            raise ValueError(
                "experiment.inference.objective.noise_model.scalar must be numeric when provided."
            )
        scalar_variance = float(scalar_variance)
        if scalar_variance <= 0.0:
            raise ValueError(
                "experiment.inference.objective.noise_model.scalar must be > 0."
            )

    optimizer_cfg = _optional_dict(inference_cfg, "optimizer", path="experiment.inference")
    optimizer_kind = str(optimizer_cfg.get("kind", "adam"))
    if optimizer_kind not in {"adam", "sgd"}:
        raise ValueError(
            "experiment.inference.optimizer.kind must be 'adam' or 'sgd'."
        )
    base_lr = optimizer_cfg.get("base_lr", 1e-2)
    if isinstance(base_lr, bool) or not isinstance(base_lr, (int, float)):
        raise ValueError("experiment.inference.optimizer.base_lr must be numeric.")
    base_lr = float(base_lr)
    if base_lr <= 0.0:
        raise ValueError("experiment.inference.optimizer.base_lr must be > 0.")
    n_iter = int(optimizer_cfg.get("n_iter", 100))
    if n_iter <= 0:
        raise ValueError("experiment.inference.optimizer.n_iter must be > 0.")
    optimizer_kwargs = optimizer_cfg.get("kwargs", {})
    if not isinstance(optimizer_kwargs, dict):
        raise ValueError("experiment.inference.optimizer.kwargs must be a mapping/dict.")
    preconditioning_cfg = _optional_dict(
        optimizer_cfg,
        "preconditioning",
        path="experiment.inference.optimizer",
    )
    preconditioning_enabled = bool(preconditioning_cfg.get("enabled", False))
    preconditioning_damping = preconditioning_cfg.get("damping", 1e-6)
    if isinstance(preconditioning_damping, bool) or not isinstance(
        preconditioning_damping, (int, float)
    ):
        raise ValueError(
            "experiment.inference.optimizer.preconditioning.damping must be numeric."
        )
    preconditioning_damping = float(preconditioning_damping)
    if preconditioning_damping < 0.0:
        raise ValueError(
            "experiment.inference.optimizer.preconditioning.damping must be >= 0."
        )
    preconditioning_eig_floor_rel = preconditioning_cfg.get("eig_floor_rel", 1e-6)
    if isinstance(preconditioning_eig_floor_rel, bool) or not isinstance(
        preconditioning_eig_floor_rel, (int, float)
    ):
        raise ValueError(
            "experiment.inference.optimizer.preconditioning.eig_floor_rel must be numeric."
        )
    preconditioning_eig_floor_rel = float(preconditioning_eig_floor_rel)
    if preconditioning_eig_floor_rel < 0.0:
        raise ValueError(
            "experiment.inference.optimizer.preconditioning.eig_floor_rel must be >= 0."
        )
    preconditioning_eig_floor_abs = preconditioning_cfg.get("eig_floor_abs", 1e-8)
    if isinstance(preconditioning_eig_floor_abs, bool) or not isinstance(
        preconditioning_eig_floor_abs, (int, float)
    ):
        raise ValueError(
            "experiment.inference.optimizer.preconditioning.eig_floor_abs must be numeric."
        )
    preconditioning_eig_floor_abs = float(preconditioning_eig_floor_abs)
    if preconditioning_eig_floor_abs < 0.0:
        raise ValueError(
            "experiment.inference.optimizer.preconditioning.eig_floor_abs must be >= 0."
        )
    preconditioning_lr_clip = _coerce_lr_clip(
        preconditioning_cfg.get("lr_clip"),
        path="experiment.inference.optimizer.preconditioning.lr_clip",
    )

    diagnostics_cfg = _optional_dict(
        inference_cfg,
        "diagnostics",
        path="experiment.inference",
    )
    write_plots = diagnostics_cfg.get("plots", True)
    if not isinstance(write_plots, bool):
        raise ValueError("experiment.inference.diagnostics.plots must be a bool.")
    compare_to_truth = diagnostics_cfg.get("compare_to_truth_when_available", True)
    if not isinstance(compare_to_truth, bool):
        raise ValueError(
            "experiment.inference.diagnostics.compare_to_truth_when_available must be a bool."
        )
    first_step_report = diagnostics_cfg.get("first_step_report", False)
    if not isinstance(first_step_report, bool):
        raise ValueError(
            "experiment.inference.diagnostics.first_step_report must be a bool."
        )
    save_first_step_json = diagnostics_cfg.get("save_first_step_json", False)
    if not isinstance(save_first_step_json, bool):
        raise ValueError(
            "experiment.inference.diagnostics.save_first_step_json must be a bool."
        )
    save_fim_debug = diagnostics_cfg.get("save_fim_debug", False)
    if not isinstance(save_fim_debug, bool):
        raise ValueError(
            "experiment.inference.diagnostics.save_fim_debug must be a bool."
        )
    finite_difference_check = diagnostics_cfg.get("finite_difference_check", False)
    if not isinstance(finite_difference_check, bool):
        raise ValueError(
            "experiment.inference.diagnostics.finite_difference_check must be a bool."
        )
    plot_parameter_history_heatmap = diagnostics_cfg.get(
        "plot_parameter_history_heatmap",
        False,
    )
    if not isinstance(plot_parameter_history_heatmap, bool):
        raise ValueError(
            "experiment.inference.diagnostics.plot_parameter_history_heatmap "
            "must be a bool."
        )
    plot_parameter_residual_history_heatmap = diagnostics_cfg.get(
        "plot_parameter_residual_history_heatmap",
        False,
    )
    if not isinstance(plot_parameter_residual_history_heatmap, bool):
        raise ValueError(
            "experiment.inference.diagnostics.plot_parameter_residual_history_heatmap "
            "must be a bool."
        )
    plot_parameter_history_lines = diagnostics_cfg.get(
        "plot_parameter_history_lines",
        False,
    )
    if not isinstance(plot_parameter_history_lines, bool):
        raise ValueError(
            "experiment.inference.diagnostics.plot_parameter_history_lines must be a bool."
        )
    plot_parameter_residual_history_lines = diagnostics_cfg.get(
        "plot_parameter_residual_history_lines",
        False,
    )
    if not isinstance(plot_parameter_residual_history_lines, bool):
        raise ValueError(
            "experiment.inference.diagnostics.plot_parameter_residual_history_lines "
            "must be a bool."
        )
    top_k = diagnostics_cfg.get("top_k", 10)
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("experiment.inference.diagnostics.top_k must be an int.")
    if top_k <= 0:
        raise ValueError("experiment.inference.diagnostics.top_k must be > 0.")

    outputs_cfg = _optional_dict(experiment_cfg, "outputs", path="experiment")
    file_prefix = outputs_cfg.get("file_prefix", "obs_subblock_inference")
    if not isinstance(file_prefix, str) or not file_prefix.strip():
        raise ValueError("experiment.outputs.file_prefix must be a non-empty string.")
    outdir_value = outputs_cfg.get("outdir")
    if outdir_value is not None and (
        not isinstance(outdir_value, str) or not outdir_value.strip()
    ):
        raise ValueError("experiment.outputs.outdir must be a non-empty string when set.")

    notes_value = experiment_cfg.get("notes")
    if notes_value is not None and not isinstance(notes_value, str):
        raise ValueError("experiment.notes must be a string when provided.")

    return {
        "kind": kind,
        "inference": {
            "data": {
                "cube": cube_path,
                "truth_trace": truth_trace_path,
                "manifest": manifest_path,
            },
            "validate": {
                "require_contiguous_frame_index": require_contiguous,
                "require_monotonic_time": require_monotonic,
            },
            "active": {
                "frame_keys": list(frame_keys),
                "shared_keys": list(shared_keys),
            },
            "init": {
                "frame": dict(init_frame_cfg),
                "shared": dict(init_shared_cfg),
            },
            "priors": {
                "frame": dict(priors_frame_cfg),
                "shared": dict(priors_shared_cfg),
            },
            "temporal": {
                "frame_model": dict(frame_model_cfg),
            },
            "objective": {
                "kind": objective_kind,
                "reduce": reduce,
                "noise_model": {
                    "kind": noise_model_kind,
                    "variance_model": variance_model,
                    "scalar": scalar_variance,
                },
            },
            "optimizer": {
                "kind": optimizer_kind,
                "base_lr": base_lr,
                "n_iter": n_iter,
                "kwargs": dict(optimizer_kwargs),
                "preconditioning": {
                    "enabled": preconditioning_enabled,
                    "damping": preconditioning_damping,
                    "eig_floor_rel": preconditioning_eig_floor_rel,
                    "eig_floor_abs": preconditioning_eig_floor_abs,
                    "lr_clip": (
                        None
                        if preconditioning_lr_clip is None
                        else list(preconditioning_lr_clip)
                    ),
                },
            },
            "diagnostics": {
                "plots": write_plots,
                "compare_to_truth_when_available": compare_to_truth,
                "first_step_report": first_step_report,
                "save_first_step_json": save_first_step_json,
                "save_fim_debug": save_fim_debug,
                "finite_difference_check": finite_difference_check,
                "plot_parameter_history_heatmap": plot_parameter_history_heatmap,
                "plot_parameter_residual_history_heatmap": (
                    plot_parameter_residual_history_heatmap
                ),
                "plot_parameter_history_lines": plot_parameter_history_lines,
                "plot_parameter_residual_history_lines": (
                    plot_parameter_residual_history_lines
                ),
                "top_k": top_k,
            },
        },
        "outputs": {
            "outdir": outdir_value,
            "file_prefix": file_prefix.strip(),
        },
        "notes": notes_value,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run observation sub-block inference."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PRESCRIPTION_PATH,
        help="Path to observation sub-block inference prescription YAML/JSON.",
    )
    parser.add_argument(
        "--system-preset",
        type=str,
        default=None,
        help="Optional system preset override merged before config file content.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Optional output root override.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run directory label under the output root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate inputs and print expected outputs without optimization.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable optimizer progress output.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run the observation sub-block inference recipe and return run metadata."""

    args = _build_parser().parse_args(argv)

    cfg_path = Path(args.config) if args.config is not None else DEFAULT_PRESCRIPTION_PATH
    print(f"Loading observation sub-block inference config from: {cfg_path}")
    user_cfg = load_user_config(
        config_path=cfg_path,
        system_preset=args.system_preset,
        experiment_preset=None,
    )
    resolved_cfg = resolve_config(user_cfg)
    system_cfg = resolved_cfg.get("system")
    experiment_cfg = resolved_cfg.get("experiment")
    if system_cfg is None:
        raise ValueError(
            "Observation sub-block inference requires a top-level 'system' block."
        )
    if experiment_cfg is None:
        raise ValueError(
            "Observation sub-block inference requires a top-level 'experiment' block."
        )

    experiment = _validate_experiment_cfg(experiment_cfg)
    inference_cfg = experiment["inference"]
    data_cfg = inference_cfg["data"]

    cube_path = _resolve_relative_path(
        data_cfg["cube"],
        config_path=cfg_path,
        field_name="experiment.inference.data.cube",
    )
    if not cube_path.exists():
        raise FileNotFoundError(f"Observation cube FITS not found: {cube_path}")

    manifest_path_value = data_cfg.get("manifest")
    manifest_path = (
        _resolve_relative_path(
            manifest_path_value,
            config_path=cfg_path,
            field_name="experiment.inference.data.manifest",
        )
        if manifest_path_value is not None
        else find_obs_subblock_sidecar_manifest(cube_path)
    )
    manifest_auto_discovered = manifest_path_value is None and manifest_path is not None
    if manifest_auto_discovered:
        print(f"Using sibling render manifest: {manifest_path}")
    manifest_input = _load_manifest(manifest_path)

    explicit_trace_path = data_cfg.get("truth_trace")
    trace_path = (
        _resolve_relative_path(
            explicit_trace_path,
            config_path=cfg_path,
            field_name="experiment.inference.data.truth_trace",
        )
        if explicit_trace_path is not None
        else None
    )
    trace_path = _infer_trace_path(
        trace_path=trace_path,
        manifest=manifest_input,
        manifest_path=manifest_path,
    )

    with fits.open(cube_path) as hdul:
        cube = np.asarray(hdul[0].data, dtype=float)
    if cube.ndim != 3:
        raise ValueError(
            "Observation sub-block cube must have shape (n_frame, ny, nx), "
            f"got {cube.shape}."
        )
    n_frame = int(cube.shape[0])
    if n_frame <= 0:
        raise ValueError("Observation sub-block cube must contain at least one frame.")
    print(
        "Loaded observation cube: "
        f"path={cube_path} shape={cube.shape} "
        f"data({_format_array_stats(cube)})"
    )

    print("Building fixed shared forward state...")
    forward_spec = compose_forward_spec(system_cfg)
    base_store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(forward_spec)
    active_layout = _build_active_state_layout(
        active_cfg=inference_cfg["active"],
        forward_spec=forward_spec,
        reference_store=base_store,
        n_frame=n_frame,
    )
    binder = SheraBinder(system_cfg, forward_spec, base_store)

    reference_image = np.asarray(binder.model(binder.strip_structural(base_store)))
    frame_shape = tuple(int(v) for v in cube.shape[1:])
    if tuple(reference_image.shape) != frame_shape:
        raise ValueError(
            "Observation sub-block cube frame shape is incompatible with the "
            f"configured fixed shared model. cube_frame_shape={frame_shape}, "
            f"model_frame_shape={tuple(reference_image.shape)}."
        )

    trace: ObsSubblockTrace | None = None
    if trace_path is not None:
        validate_cfg = inference_cfg["validate"]
        trace = load_obs_subblock_trace_csv(
            trace_path,
            required_varying_keys=active_layout.frame_keys,
            require_contiguous_frame_index=validate_cfg["require_contiguous_frame_index"],
            require_monotonic_time=validate_cfg["require_monotonic_time"],
        )

    time_trace: ObsSubblockTrace | None = trace
    comparison_trace: ObsSubblockTrace | None = (
        trace if inference_cfg["diagnostics"]["compare_to_truth_when_available"] else None
    )
    if trace is not None and trace.frame_count != n_frame:
        print(
            "Warning: truth trace frame count does not match cube frame count; "
            "skipping truth-based time/comparison outputs."
        )
        time_trace = None
        comparison_trace = None

    configured_outdir = experiment["outputs"]["outdir"]
    if args.results_dir is not None:
        outdir_root = Path(args.results_dir)
    elif configured_outdir is not None:
        outdir_root = _resolve_relative_path(
            configured_outdir,
            config_path=cfg_path,
            field_name="experiment.outputs.outdir",
        )
    else:
        outdir_root = DEFAULT_OUTDIR_ROOT

    stamp = timestamp_tag()
    run_label = args.run_name or stamp
    outdir = outdir_root / run_label
    artifacts = _build_artifact_paths(
        outdir=outdir,
        file_prefix=experiment["outputs"]["file_prefix"],
        timestamp=stamp,
        include_comparison=comparison_trace is not None and active_layout.frame_width > 0,
        write_plots=bool(inference_cfg["diagnostics"]["plots"]),
        include_trace_plots=active_layout.frame_width > 0,
        include_parameter_history_heatmap=(
            active_layout.frame_width > 0
            and bool(inference_cfg["diagnostics"]["plot_parameter_history_heatmap"])
        ),
        include_parameter_residual_history_heatmap=(
            active_layout.frame_width > 0
            and comparison_trace is not None
            and bool(
                inference_cfg["diagnostics"][
                    "plot_parameter_residual_history_heatmap"
                ]
            )
        ),
        include_parameter_history_lines=(
            active_layout.frame_width > 0
            and bool(inference_cfg["diagnostics"]["plot_parameter_history_lines"])
        ),
        include_parameter_residual_history_lines=(
            active_layout.frame_width > 0
            and comparison_trace is not None
            and bool(
                inference_cfg["diagnostics"]["plot_parameter_residual_history_lines"]
            )
        ),
    )
    diag_cfg = inference_cfg["diagnostics"]
    if bool(diag_cfg["save_first_step_json"]):
        artifacts["first_step_json"] = (
            outdir / f"{experiment['outputs']['file_prefix']}_{stamp}_first_step.json"
        )
        artifacts["per_frame_step_json"] = (
            outdir / f"{experiment['outputs']['file_prefix']}_{stamp}_per_frame_step.json"
        )
    if bool(diag_cfg["finite_difference_check"]):
        artifacts["finite_difference_json"] = (
            outdir / f"{experiment['outputs']['file_prefix']}_{stamp}_finite_difference.json"
        )
    if bool(diag_cfg["save_fim_debug"]):
        artifacts["fim_debug_json"] = (
            outdir / f"{experiment['outputs']['file_prefix']}_{stamp}_fim_debug.json"
        )
    if bool(diag_cfg["plots"]) and (
        bool(diag_cfg["first_step_report"]) or bool(diag_cfg["save_first_step_json"])
        or bool(diag_cfg["finite_difference_check"])
    ):
        artifacts["first_step_grad_png"] = (
            outdir / f"{experiment['outputs']['file_prefix']}_{stamp}_first_step_grad.png"
        )
        artifacts["first_step_delta_png"] = (
            outdir / f"{experiment['outputs']['file_prefix']}_{stamp}_first_step_delta.png"
        )
        artifacts["per_frame_step_png"] = (
            outdir / f"{experiment['outputs']['file_prefix']}_{stamp}_per_frame_step.png"
        )

    initial_state = _resolve_initial_active_state(
        layout=active_layout,
        base_store=base_store,
        init_cfg=inference_cfg["init"],
    )
    theta0 = _pack_active_state(active_layout, initial_state)
    theta_labels = _theta_labels_for_layout(active_layout)
    _print_active_layout_summary(layout=active_layout, theta0=theta0)

    variance_cube = _build_variance_cube(
        data_cube=cube,
        noise_model_cfg=inference_cfg["objective"]["noise_model"],
    )
    print(
        "Objective setup: "
        f"kind={inference_cfg['objective']['kind']} "
        f"reduce={inference_cfg['objective']['reduce']} "
        f"noise_model={inference_cfg['objective']['noise_model']['kind']} "
        f"variance_model={inference_cfg['objective']['noise_model']['variance_model']} "
        f"variance({_format_array_stats(variance_cube)})"
    )
    objective_bundle = _build_objective_bundle(
        layout=active_layout,
        binder=binder,
        forward_spec=forward_spec,
        base_store=base_store,
        cube_data=cube,
        variance_cube=variance_cube,
        objective_cfg=inference_cfg["objective"],
        priors_cfg=inference_cfg["priors"],
        temporal_cfg=inference_cfg["temporal"],
    )

    if args.dry_run:
        print("Dry run: validated configuration and inputs.")
        print(f"  cube_path: {cube_path}")
        print(f"  frame_count: {n_frame}")
        print(f"  frame_keys: {list(active_layout.frame_keys)}")
        print(f"  shared_keys: {list(active_layout.shared_keys)}")
        print(f"  theta_size: {active_layout.theta_size}")
        print(f"  output_dir: {outdir}")
        for key, path in artifacts.items():
            print(f"  expected_{key}: {path}")
        return {
            "dry_run": True,
            "frame_count": n_frame,
            "output_dir": str(outdir),
            "artifacts": {name: str(path) for name, path in artifacts.items()},
            "frame_keys": list(active_layout.frame_keys),
            "shared_keys": list(active_layout.shared_keys),
            "theta_size": int(active_layout.theta_size),
        }

    outdir.mkdir(parents=True, exist_ok=True)

    optimizer_cfg = inference_cfg["optimizer"]
    preconditioning_cfg = optimizer_cfg["preconditioning"]
    preconditioning_bundle: ThetaPreconditioningBundle | None = None
    lr_vec: np.ndarray | None = None
    if bool(preconditioning_cfg["enabled"]):
        print(
            "Computing theta-space Fisher preconditioner at initialization "
            f"(theta_dim={active_layout.theta_size})..."
        )
        preconditioning_bundle = _build_theta_preconditioning_bundle(
            loss_fn=objective_bundle.total_loss_fn,
            theta_ref=jnp.asarray(theta0),
            base_lr=float(optimizer_cfg["base_lr"]),
            cfg=preconditioning_cfg,
        )
        lr_vec = np.asarray(preconditioning_bundle.lr_vec, dtype=float)
        _print_preconditioning_summary(
            bundle=preconditioning_bundle,
            base_lr=float(optimizer_cfg["base_lr"]),
        )
        if bool(inference_cfg["diagnostics"]["save_fim_debug"]):
            _save_fim_debug_artifact(
                bundle=preconditioning_bundle,
                theta_labels=theta_labels,
                base_lr=float(optimizer_cfg["base_lr"]),
                output_path=artifacts["fim_debug_json"],
            )
    else:
        print("Preconditioning disabled; using scalar optimizer learning rate.")
        if bool(inference_cfg["diagnostics"]["save_fim_debug"]):
            _write_json(
                artifacts["fim_debug_json"],
                {
                    "schema_version": "subblock_theta_preconditioning_debug.v1",
                    "enabled": False,
                    "message": "Theta preconditioning was disabled for this run.",
                },
            )

    configured_first_step: dict[str, Any] | None = None
    if (
        bool(inference_cfg["diagnostics"]["first_step_report"])
        or bool(inference_cfg["diagnostics"]["save_first_step_json"])
        or bool(inference_cfg["diagnostics"]["finite_difference_check"])
    ):
        print("Running first-step optimizer diagnostics before main optimization...")
        base_lr = float(optimizer_cfg["base_lr"])
        optimizer_kind = str(optimizer_cfg["kind"])
        optimizer_kwargs = dict(optimizer_cfg["kwargs"])
        top_k = int(inference_cfg["diagnostics"]["top_k"])
        configured_first_step = _optimizer_first_step(
            loss_fn=objective_bundle.total_loss_fn,
            theta0=jnp.asarray(theta0),
            learning_rate=base_lr,
            lr_vec=lr_vec,
            optimizer_kind=optimizer_kind,
            optimizer_kwargs=optimizer_kwargs,
        )
        configured_summary = _summarize_first_step(
            label="configured optimizer settings",
            step=configured_first_step,
            theta_labels=theta_labels,
            top_k=top_k,
        )
        no_lr_vec_step = _optimizer_first_step(
            loss_fn=objective_bundle.total_loss_fn,
            theta0=jnp.asarray(theta0),
            learning_rate=base_lr,
            lr_vec=None,
            optimizer_kind=optimizer_kind,
            optimizer_kwargs=optimizer_kwargs,
        )
        no_lr_vec_summary = _summarize_first_step(
            label="base_lr with lr_vec=None",
            step=no_lr_vec_step,
            theta_labels=theta_labels,
            top_k=top_k,
        )
        tiny_step = _optimizer_first_step(
            loss_fn=objective_bundle.total_loss_fn,
            theta0=jnp.asarray(theta0),
            learning_rate=base_lr * 1.0e-3,
            lr_vec=None,
            optimizer_kind=optimizer_kind,
            optimizer_kwargs=optimizer_kwargs,
        )
        tiny_summary = _summarize_first_step(
            label="tiny scalar lr with lr_vec=None",
            step=tiny_step,
            theta_labels=theta_labels,
            top_k=top_k,
        )

        per_frame_theta0 = np.asarray(
            objective_bundle.per_frame_data_terms_fn(jnp.asarray(theta0)),
            dtype=float,
        )
        per_frame_theta1 = np.asarray(
            objective_bundle.per_frame_data_terms_fn(
                jnp.asarray(configured_first_step["theta1"])
            ),
            dtype=float,
        )
        _print_per_frame_data_terms(label="theta0", terms=per_frame_theta0)
        _print_per_frame_data_terms(
            label="theta1 configured first step",
            terms=per_frame_theta1,
        )

        finite_difference_rows: list[dict[str, Any]] | None = None
        if bool(inference_cfg["diagnostics"]["finite_difference_check"]):
            finite_difference_rows = _finite_difference_gradient_check(
                loss_fn=objective_bundle.total_loss_fn,
                theta0=jnp.asarray(theta0),
                grad0=np.asarray(configured_first_step["grad0"], dtype=float),
                theta_labels=theta_labels,
            )
            _write_json(
                artifacts["finite_difference_json"],
                {
                    "schema_version": "subblock_finite_difference_check.v1",
                    "rows": finite_difference_rows,
                },
            )

        if bool(inference_cfg["diagnostics"]["save_first_step_json"]):
            _write_json(
                artifacts["first_step_json"],
                {
                    "schema_version": "subblock_first_step_diagnostics.v1",
                    "optimizer": {
                        "kind": optimizer_kind,
                        "base_lr": base_lr,
                        "kwargs": optimizer_kwargs,
                        "run_shera_gd_lr_vec_semantics": (
                            "lr_vec is a preconditioning scale only; run_shera_gd "
                            "applies base_lr through learning_rate"
                        ),
                    },
                    "theta_labels": list(theta_labels),
                    "theta0": _to_jsonable_float_list(configured_first_step["theta0"]),
                    "theta1_configured": _to_jsonable_float_list(
                        configured_first_step["theta1"]
                    ),
                    "grad0": _to_jsonable_float_list(configured_first_step["grad0"]),
                    "delta0_configured": _to_jsonable_float_list(
                        configured_first_step["delta0"]
                    ),
                    "loss0": float(configured_first_step["loss0"]),
                    "loss1_configured": float(configured_first_step["loss1"]),
                    "preconditioning": (
                        {"enabled": False, "lr_vec": None}
                        if preconditioning_bundle is None
                        else {
                            **to_jsonable_obs_subblock_payload(
                                preconditioning_bundle.config
                            ),
                            "preconditioner_diagonal": _to_jsonable_float_list(
                                preconditioning_bundle.preconditioner_diag
                            ),
                            "lr_scale_before_clipping": _to_jsonable_float_list(
                                preconditioning_bundle.lr_vec_unclipped
                            ),
                            "lr_scale_after_clipping": _to_jsonable_float_list(
                                preconditioning_bundle.lr_vec
                            ),
                            "effective_lr_before_clipping": _to_jsonable_float_list(
                                base_lr * preconditioning_bundle.lr_vec_unclipped
                            ),
                            "effective_lr_after_clipping": _to_jsonable_float_list(
                                base_lr * preconditioning_bundle.lr_vec
                            ),
                        }
                    ),
                    "summaries": {
                        "configured": configured_summary,
                        "base_lr_lr_vec_none": no_lr_vec_summary,
                        "tiny_scalar_lr_lr_vec_none": tiny_summary,
                    },
                },
            )
            _write_json(
                artifacts["per_frame_step_json"],
                {
                    "schema_version": "subblock_per_frame_first_step.v1",
                    "rows": (
                        _per_frame_diagnostic_rows(
                            per_frame_terms=per_frame_theta0,
                            label="theta0",
                        )
                        + _per_frame_diagnostic_rows(
                            per_frame_terms=per_frame_theta1,
                            label="theta1_configured",
                        )
                    ),
                },
            )

        if bool(inference_cfg["diagnostics"]["plots"]):
            _plot_theta_bar(
                values=np.asarray(configured_first_step["grad0"], dtype=float),
                labels=theta_labels,
                ylabel="|grad0|",
                title="Initial Gradient Magnitudes",
                output_path=artifacts["first_step_grad_png"],
            )
            _plot_theta_bar(
                values=np.asarray(configured_first_step["delta0"], dtype=float),
                labels=theta_labels,
                ylabel="|delta0|",
                title="Configured First-Step Update Magnitudes",
                output_path=artifacts["first_step_delta_png"],
            )
            _plot_per_frame_comparison(
                theta0_terms=per_frame_theta0,
                theta1_terms=per_frame_theta1,
                output_path=artifacts["per_frame_step_png"],
            )

    print("Initializing active inference state...")
    print(
        "Running optimization: "
        f"kind={optimizer_cfg['kind']} n_iter={optimizer_cfg['n_iter']} "
        f"base_lr={optimizer_cfg['base_lr']} "
        f"preconditioned={lr_vec is not None}"
    )
    # lr_vec is scale-only; run_shera_gd applies base_lr through learning_rate.
    theta_final, trace_history = run_shera_gd(
        loss_fn=objective_bundle.total_loss_fn,
        theta0=jnp.asarray(theta0),
        learning_rate=float(optimizer_cfg["base_lr"]),
        lr_vec=None if lr_vec is None else jnp.asarray(lr_vec),
        num_steps=int(optimizer_cfg["n_iter"]),
        optimizer_kind=str(optimizer_cfg["kind"]),
        optimizer_kwargs=dict(optimizer_cfg["kwargs"]),
        return_artifacts=False,
        show_progress=not bool(args.no_progress),
    )

    theta_final_np = np.asarray(theta_final, dtype=float)
    final_state = _unpack_active_state(active_layout, theta_final)
    final_frame_matrix = np.asarray(final_state.frame, dtype=float)
    final_shared_vector = np.asarray(final_state.shared, dtype=float)

    loss_history = np.asarray(trace_history["loss"], dtype=float)
    initial_loss = float(np.asarray(objective_bundle.total_loss_fn(theta0)))
    final_loss = float(np.asarray(objective_bundle.total_loss_fn(theta_final)))

    model_cube = np.asarray(objective_bundle.predict_cube_fn(theta_final), dtype=float)
    frame_data_terms = np.asarray(
        objective_bundle.per_frame_data_terms_fn(theta_final),
        dtype=float,
    )
    final_data_term, final_prior_term, final_temporal_term = objective_bundle.objective_terms_fn(
        theta_final
    )

    if time_trace is not None:
        times = np.asarray([float(row["time_s"]) for row in time_trace.rows], dtype=float)
    else:
        times = np.arange(n_frame, dtype=float)

    recovered_rows = _build_recovered_rows(
        layout=active_layout,
        times=times,
        frame_matrix=final_frame_matrix,
        frame_data_terms=frame_data_terms,
    )
    recovered_fieldnames = ("frame_index", "time_s", *active_layout.frame_keys, "frame_nll")
    write_obs_subblock_truth_csv(
        output_path=artifacts["recovered_trace_csv"],
        rows=recovered_rows,
        fieldnames=recovered_fieldnames,
    )

    truth_matrix: np.ndarray | None = None
    if comparison_trace is not None and active_layout.frame_width > 0:
        truth_matrix = _build_truth_matrix(
            comparison_trace,
            frame_keys=active_layout.frame_keys,
        )
        comparison_rows = _build_truth_comparison_rows(
            layout=active_layout,
            times=times,
            recovered_frame_matrix=final_frame_matrix,
            truth_matrix=truth_matrix,
            frame_data_terms=frame_data_terms,
        )
        comparison_fieldnames = ["frame_index", "time_s"]
        for key in active_layout.frame_keys:
            comparison_fieldnames.extend(
                [f"{key}_truth", f"{key}_recovered", f"{key}_residual"]
            )
        comparison_fieldnames.append("frame_nll")
        write_obs_subblock_truth_csv(
            output_path=artifacts["truth_comparison_csv"],
            rows=comparison_rows,
            fieldnames=tuple(comparison_fieldnames),
        )

    if inference_cfg["diagnostics"]["plots"]:
        _plot_loss_history(losses=loss_history, output_path=artifacts["loss_history_png"])
        if active_layout.frame_width > 0:
            history_plot_requested = (
                bool(inference_cfg["diagnostics"]["plot_parameter_history_heatmap"])
                or bool(
                    inference_cfg["diagnostics"][
                        "plot_parameter_residual_history_heatmap"
                    ]
                )
                or bool(inference_cfg["diagnostics"]["plot_parameter_history_lines"])
                or bool(
                    inference_cfg["diagnostics"][
                        "plot_parameter_residual_history_lines"
                    ]
                )
            )
            if history_plot_requested:
                iterations, frame_history = _build_frame_history_from_theta_trace(
                    layout=active_layout,
                    theta0=jnp.asarray(theta0),
                    theta_trace=np.asarray(trace_history["theta"], dtype=float),
                )
                if "parameter_history_heatmap_png" in artifacts:
                    _plot_parameter_history_heatmaps(
                        frame_history=frame_history,
                        labels=active_layout.frame_keys,
                        output_path=artifacts["parameter_history_heatmap_png"],
                        title="Frame-varying active-state optimizer history",
                        colorbar_label="Parameter value",
                    )
                if (
                    truth_matrix is not None
                    and "parameter_residual_history_heatmap_png" in artifacts
                ):
                    _plot_parameter_history_heatmaps(
                        frame_history=frame_history - truth_matrix[None, :, :],
                        labels=active_layout.frame_keys,
                        output_path=artifacts["parameter_residual_history_heatmap_png"],
                        title="Frame-varying residual optimizer history",
                        colorbar_label="Recovered minus truth",
                        center_zero=True,
                    )
                if "parameter_history_lines_png" in artifacts:
                    _plot_parameter_history_lines(
                        iterations=iterations,
                        frame_history=frame_history,
                        labels=active_layout.frame_keys,
                        output_path=artifacts["parameter_history_lines_png"],
                    )
                if (
                    truth_matrix is not None
                    and "parameter_residual_history_lines_png" in artifacts
                ):
                    _plot_parameter_history_lines(
                        iterations=iterations,
                        frame_history=frame_history - truth_matrix[None, :, :],
                        labels=active_layout.frame_keys,
                        output_path=artifacts["parameter_residual_history_lines_png"],
                        title="Frame-varying residual optimizer history",
                        ylabel_suffix=" residual",
                        zero_line=True,
                    )
            if truth_matrix is not None:
                _plot_trace_comparison(
                    times=times,
                    recovered=final_frame_matrix,
                    truth=truth_matrix,
                    labels=active_layout.frame_keys,
                    output_path=artifacts["trace_comparison_png"],
                )
                _plot_trace_residuals(
                    times=times,
                    recovered=final_frame_matrix,
                    truth=truth_matrix,
                    labels=active_layout.frame_keys,
                    output_path=artifacts["trace_residuals_png"],
                )
            else:
                _plot_recovered_traces(
                    times=times,
                    recovered=final_frame_matrix,
                    labels=active_layout.frame_keys,
                    output_path=artifacts["recovered_traces_png"],
                )
        _plot_image_fit(
            data_cube=cube,
            model_cube=model_cube,
            output_path=artifacts["image_fit_png"],
        )

    initial_state_np = _unpack_active_state(active_layout, theta0)
    initial_frame_values = {
        spec.canonical: float(np.asarray(initial_state_np.frame[0, idx]))
        for idx, spec in enumerate(active_layout.frame_specs)
    }
    initial_shared_values = {
        spec.canonical: float(np.asarray(initial_state_np.shared[idx]))
        for idx, spec in enumerate(active_layout.shared_specs)
    }
    recovered_shared_values = {
        spec.canonical: float(final_shared_vector[idx])
        for idx, spec in enumerate(active_layout.shared_specs)
    }
    optimizer_base_lr = float(optimizer_cfg["base_lr"])

    system_info = {
        "preset": system_cfg.get("preset"),
        "config_hash": _stable_hash_payload(system_cfg),
        "resolved_config": to_jsonable_obs_subblock_payload(system_cfg),
    }
    manifest_payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": now_iso_local_ms(),
        "generator": GENERATOR_ID,
        "frame_count": n_frame,
        "infer_keys": list(active_layout.all_keys),
        "inputs": {
            "config_path": str(cfg_path.resolve()),
            "cube_fits": str(cube_path),
            "truth_trace_csv": None if trace_path is None else str(trace_path),
            "manifest_json": None if manifest_path is None else str(manifest_path),
            "manifest_auto_discovered": bool(manifest_auto_discovered),
        },
        "active": {
            "frame_keys": list(active_layout.frame_keys),
            "shared_keys": list(active_layout.shared_keys),
        },
        "init": {
            "frame": {
                "mode": str(inference_cfg["init"]["frame"]["mode"]),
                "values": initial_frame_values,
            },
            "shared": initial_shared_values,
        },
        "recovered_shared": recovered_shared_values,
        "priors": to_jsonable_obs_subblock_payload(inference_cfg["priors"]),
        "temporal": to_jsonable_obs_subblock_payload(inference_cfg["temporal"]),
        "objective": to_jsonable_obs_subblock_payload(inference_cfg["objective"]),
        "optimizer": {
            "kind": optimizer_cfg["kind"],
            "base_lr": float(optimizer_cfg["base_lr"]),
            "n_iter": int(optimizer_cfg["n_iter"]),
            "kwargs": dict(optimizer_cfg["kwargs"]),
            "preconditioning": (
                {"enabled": False}
                if preconditioning_bundle is None
                else {
                    **preconditioning_bundle.config,
                    "theta_dim": int(preconditioning_bundle.lr_vec.size),
                    "eigval_min": float(np.min(preconditioning_bundle.eigvals)),
                    "eigval_max": float(np.max(preconditioning_bundle.eigvals)),
                    "eigval_stable_min": float(np.min(preconditioning_bundle.eigvals_stable)),
                    "eigval_stable_max": float(np.max(preconditioning_bundle.eigvals_stable)),
                    "lr_scale_unclipped_min": float(
                        np.min(preconditioning_bundle.lr_vec_unclipped)
                    ),
                    "lr_scale_unclipped_max": float(
                        np.max(preconditioning_bundle.lr_vec_unclipped)
                    ),
                    "preconditioner_diag_min": float(
                        np.min(preconditioning_bundle.preconditioner_diag)
                    ),
                    "preconditioner_diag_max": float(
                        np.max(preconditioning_bundle.preconditioner_diag)
                    ),
                    "lr_scale_min": float(np.min(preconditioning_bundle.lr_vec)),
                    "lr_scale_max": float(np.max(preconditioning_bundle.lr_vec)),
                    "effective_lr_unclipped_min": float(
                        np.min(
                            optimizer_base_lr * preconditioning_bundle.lr_vec_unclipped
                        )
                    ),
                    "effective_lr_unclipped_max": float(
                        np.max(
                            optimizer_base_lr * preconditioning_bundle.lr_vec_unclipped
                        )
                    ),
                    "effective_lr_min": float(
                        np.min(optimizer_base_lr * preconditioning_bundle.lr_vec)
                    ),
                    "effective_lr_max": float(
                        np.max(optimizer_base_lr * preconditioning_bundle.lr_vec)
                    ),
                }
            ),
        },
        "diagnostics": to_jsonable_obs_subblock_payload(inference_cfg["diagnostics"]),
        "metrics": {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_delta": final_loss - initial_loss,
            "final_data_term": float(np.asarray(final_data_term)),
            "final_prior_term": float(np.asarray(final_prior_term)),
            "final_temporal_term": float(np.asarray(final_temporal_term)),
            "mean_frame_nll": float(np.mean(frame_data_terms)),
            "preconditioner_trace_fim": (
                None
                if preconditioning_bundle is None
                else float(np.trace(preconditioning_bundle.fim))
            ),
        },
        "truth_comparison_available": truth_matrix is not None,
        "system": system_info,
        "artifacts": {
            name: _relative_path(path, outdir=outdir)
            for name, path in artifacts.items()
        },
    }
    if experiment.get("notes") is not None:
        manifest_payload["notes"] = str(experiment["notes"])
    with artifacts["manifest_json"].open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2, default=str)

    print(f"Finished observation sub-block inference on {n_frame} frames.")
    print(f"initial_loss={initial_loss:.6g} final_loss={final_loss:.6g}")
    print(f"Wrote artifacts under: {outdir}")

    return {
        "dry_run": False,
        "frame_count": n_frame,
        "output_dir": str(outdir),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "frame_keys": list(active_layout.frame_keys),
        "shared_keys": list(active_layout.shared_keys),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "theta_final": theta_final_np,
    }


if __name__ == "__main__":
    main()
