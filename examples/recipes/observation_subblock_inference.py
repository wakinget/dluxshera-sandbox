"""Run registration-only inference on an observation sub-block cube.

This recipe performs the first intentionally narrow block-inference task:
jointly infer per-frame registration parameters while keeping the shared
system/source/optics/detector state fixed.

Inferred per frame
------------------
- ``source.x_position_as``
- ``source.y_position_as``
- ``source.position_angle_deg``

Inputs
------
- required: observation sub-block FITS cube
- optional: frame-truth CSV (for comparison diagnostics)
- optional: manifest JSON (for metadata / truth-path discovery)
- if manifest is omitted, the recipe looks for ``manifest.json`` beside the cube

Outputs
-------
- recovered per-frame registration table
- optional truth-vs-recovered comparison table
- diagnostics plots (loss history, traces, residuals, image-fit panel)
- manifest JSON summarizing inputs, settings, fixed shared state, and artifacts
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib
import numpy as np
from astropy.io import fits

from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.inference.losses import gaussian_image_nll
from dluxshera.inference.optimization import run_shera_gd
from dluxshera.params.store import ParameterStore
from dluxshera.systems import SheraBinder
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.obs_subblock_io import (
    find_obs_subblock_sidecar_manifest,
    now_iso_local_ms,
    timestamp_tag,
    to_jsonable_obs_subblock_payload,
)
from dluxshera.utils.obs_subblock_keys import (
    apply_obs_subblock_overrides_preserving_derived,
    partition_obs_subblock_overrides_by_kind,
)
from dluxshera.utils.obs_subblock_trace import ObsSubblockTrace, load_obs_subblock_trace_csv

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


DEFAULT_PRESCRIPTION_PATH = Path(
    "examples/recipes/observation_subblock_inference_template/prescription.yaml"
)
DEFAULT_OUTDIR_ROOT = Path("Results/observation_subblock_inference")
MANIFEST_SCHEMA_VERSION = "obs_subblock_inference_manifest.v1"
GENERATOR_ID = "examples/recipes/observation_subblock_inference.py"
FRAME_PARAM_KEYS: tuple[str, str, str] = (
    "source.x_position_as",
    "source.y_position_as",
    "source.position_angle_deg",
)
TRACE_VALIDATE_DEFAULTS = {
    "require_contiguous_frame_index": True,
    "require_monotonic_time": True,
}


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


def _optional_number(parent: dict[str, Any], key: str, *, path: str) -> float | None:
    value = parent.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{key} must be numeric when provided.")
    return float(value)


def _resolve_relative_path(
    value: str,
    *,
    config_path: Path | None,
    field_name: str,
) -> Path:
    raw_path = Path(value)
    if raw_path.is_absolute():
        return raw_path
    if config_path is not None:
        return (config_path.parent / raw_path).resolve()
    raise ValueError(
        f"{field_name} is relative but config_path is not available to resolve it."
    )


def _flatten_truth_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                joined = f"{prefix}.{key}" if prefix else str(key)
                _walk(joined, child)
        else:
            flattened[prefix] = value

    _walk("", payload)
    return flattened


def _stable_hash_payload(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _relative_path(path: Path, *, outdir: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(outdir.resolve()).as_posix()
    except ValueError:
        return Path(os.path.relpath(resolved, outdir.resolve())).as_posix()


def _load_manifest(path: Path | None) -> dict[str, Any] | None:
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


def _theta_matrix_to_flat(theta_matrix: np.ndarray) -> np.ndarray:
    return np.asarray(theta_matrix, dtype=float).reshape(-1)


def _theta_flat_to_matrix(theta_flat: np.ndarray, *, n_frame: int) -> np.ndarray:
    return np.asarray(theta_flat, dtype=float).reshape(n_frame, len(FRAME_PARAM_KEYS))


def _write_csv_rows(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _build_artifact_paths(
    *,
    outdir: Path,
    file_prefix: str,
    timestamp: str,
    include_comparison: bool,
    write_plots: bool,
) -> dict[str, Path]:
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
    return artifacts


def _resolve_initial_theta_matrix(
    *,
    base_store: ParameterStore,
    init_cfg: dict[str, Any],
    n_frame: int,
) -> np.ndarray:
    base_values = np.array(
        [
            float(np.asarray(base_store.get("source.x_position_as"))),
            float(np.asarray(base_store.get("source.y_position_as"))),
            float(np.asarray(base_store.get("source.position_angle_deg"))),
        ],
        dtype=float,
    )
    x_init = init_cfg.get("x_position_as", base_values[0])
    y_init = init_cfg.get("y_position_as", base_values[1])
    pa_init = init_cfg.get("position_angle_deg", base_values[2])

    start_vector = np.array([float(x_init), float(y_init), float(pa_init)], dtype=float)
    return np.repeat(start_vector[None, :], repeats=n_frame, axis=0)


def _build_block_loss_fn(
    *,
    binder: SheraBinder,
    forward_spec,
    base_store: ParameterStore,
    cube_data: np.ndarray,
    variance_cube: np.ndarray,
    reduce: str,
):
    n_frame = int(cube_data.shape[0])
    data = jnp.asarray(cube_data)
    var = jnp.asarray(variance_cube)

    def loss_fn(theta_flat: jnp.ndarray) -> jnp.ndarray:
        theta_matrix = jnp.asarray(theta_flat).reshape(n_frame, len(FRAME_PARAM_KEYS))
        total = jnp.array(0.0, dtype=data.dtype)
        for frame_index in range(n_frame):
            frame_theta = theta_matrix[frame_index]
            frame_overrides = {
                FRAME_PARAM_KEYS[0]: frame_theta[0],
                FRAME_PARAM_KEYS[1]: frame_theta[1],
                FRAME_PARAM_KEYS[2]: frame_theta[2],
            }
            frame_store = base_store.replace(frame_overrides).refresh_derived(forward_spec)
            frame_delta = binder.strip_structural(frame_store)
            model_frame = binder.model(frame_delta)
            total = total + gaussian_image_nll(
                model_frame,
                data[frame_index],
                var[frame_index],
                reduce=reduce,
            )
        return total

    return loss_fn


def _predict_cube(
    *,
    binder: SheraBinder,
    forward_spec,
    base_store: ParameterStore,
    theta_matrix: np.ndarray,
) -> np.ndarray:
    frames: list[np.ndarray] = []
    for frame_row in np.asarray(theta_matrix):
        frame_store = base_store.replace(
            {
                FRAME_PARAM_KEYS[0]: float(frame_row[0]),
                FRAME_PARAM_KEYS[1]: float(frame_row[1]),
                FRAME_PARAM_KEYS[2]: float(frame_row[2]),
            }
        ).refresh_derived(forward_spec)
        frame_delta = binder.strip_structural(frame_store)
        frames.append(np.asarray(binder.model(frame_delta)))
    return np.stack(frames, axis=0)


def _compute_per_frame_nll(
    *,
    model_cube: np.ndarray,
    data_cube: np.ndarray,
    variance_cube: np.ndarray,
    reduce: str,
) -> np.ndarray:
    losses = []
    for idx in range(int(data_cube.shape[0])):
        loss_value = gaussian_image_nll(
            jnp.asarray(model_cube[idx]),
            jnp.asarray(data_cube[idx]),
            jnp.asarray(variance_cube[idx]),
            reduce=reduce,
        )
        losses.append(float(np.asarray(loss_value)))
    return np.asarray(losses, dtype=float)


def _plot_loss_history(*, losses: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    iterations = np.arange(losses.shape[0], dtype=int)
    ax.plot(iterations, losses, marker="o", linewidth=1.0, markersize=3.0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Block loss")
    ax.set_title("Observation sub-block inference loss history")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_recovered_traces(
    *,
    times: np.ndarray,
    recovered: np.ndarray,
    output_path: Path,
) -> None:
    labels = ("x_position_as", "y_position_as", "position_angle_deg")
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for idx, ax in enumerate(axes):
        ax.plot(times, recovered[:, idx], marker="o", linewidth=1.0, label="recovered")
        ax.set_ylabel(labels[idx])
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time_s")
    fig.suptitle("Recovered frame-varying registration traces")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_trace_comparison(
    *,
    times: np.ndarray,
    recovered: np.ndarray,
    truth: np.ndarray,
    output_path: Path,
) -> None:
    labels = ("x_position_as", "y_position_as", "position_angle_deg")
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for idx, ax in enumerate(axes):
        ax.plot(times, recovered[:, idx], marker="o", linewidth=1.0, label="recovered")
        ax.plot(times, truth[:, idx], marker="x", linewidth=1.0, label="truth")
        ax.set_ylabel(labels[idx])
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time_s")
    fig.suptitle("Recovered vs truth registration traces")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_trace_residuals(
    *,
    times: np.ndarray,
    recovered: np.ndarray,
    truth: np.ndarray,
    output_path: Path,
) -> None:
    labels = ("x residual (as)", "y residual (as)", "PA residual (deg)")
    residual = recovered - truth
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for idx, ax in enumerate(axes):
        ax.plot(times, residual[:, idx], marker="o", linewidth=1.0)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_ylabel(labels[idx])
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time_s")
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
    n_frame = int(data_cube.shape[0])
    sample_indices = [0, n_frame // 2, n_frame - 1]
    deduped = []
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


def _validate_experiment_cfg(experiment_cfg: dict[str, Any]) -> dict[str, Any]:
    kind = _required_str(experiment_cfg, "kind", path="experiment")
    if kind != "observation_subblock_inference":
        raise ValueError(
            "experiment.kind must be 'observation_subblock_inference' for this recipe."
        )

    truth = experiment_cfg.get("truth", {})
    if not isinstance(truth, dict):
        raise ValueError("experiment.truth must be a mapping/dict.")

    inference_cfg = _required_dict(
        experiment_cfg, "observation_subblock_inference", path="experiment"
    )
    inputs_cfg = _required_dict(inference_cfg, "inputs", path="experiment.observation_subblock_inference")
    cube_path = _required_str(
        inputs_cfg, "cube", path="experiment.observation_subblock_inference.inputs"
    )
    trace_path = _optional_str(
        inputs_cfg, "trace", path="experiment.observation_subblock_inference.inputs"
    )
    manifest_path = _optional_str(
        inputs_cfg, "manifest", path="experiment.observation_subblock_inference.inputs"
    )

    validate_cfg = inference_cfg.get("validate", {})
    if not isinstance(validate_cfg, dict):
        raise ValueError("experiment.observation_subblock_inference.validate must be a mapping/dict.")
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
            "experiment.observation_subblock_inference.validate.require_contiguous_frame_index "
            "must be a bool."
        )
    if not isinstance(require_monotonic, bool):
        raise ValueError(
            "experiment.observation_subblock_inference.validate.require_monotonic_time "
            "must be a bool."
        )

    init_cfg = inference_cfg.get("init", {})
    if not isinstance(init_cfg, dict):
        raise ValueError("experiment.observation_subblock_inference.init must be a mapping/dict.")
    x_init = _optional_number(
        init_cfg, "x_position_as", path="experiment.observation_subblock_inference.init"
    )
    y_init = _optional_number(
        init_cfg, "y_position_as", path="experiment.observation_subblock_inference.init"
    )
    pa_init = _optional_number(
        init_cfg, "position_angle_deg", path="experiment.observation_subblock_inference.init"
    )

    loss_cfg = inference_cfg.get("loss", {})
    if not isinstance(loss_cfg, dict):
        raise ValueError("experiment.observation_subblock_inference.loss must be a mapping/dict.")
    reduce = str(loss_cfg.get("reduce", "sum"))
    if reduce not in {"sum", "mean"}:
        raise ValueError(
            "experiment.observation_subblock_inference.loss.reduce must be 'sum' or 'mean'."
        )
    variance_value = float(loss_cfg.get("variance", 1.0))
    if variance_value <= 0.0:
        raise ValueError(
            "experiment.observation_subblock_inference.loss.variance must be > 0."
        )

    optimizer_cfg = inference_cfg.get("optimizer", {})
    if not isinstance(optimizer_cfg, dict):
        raise ValueError(
            "experiment.observation_subblock_inference.optimizer must be a mapping/dict."
        )
    optimizer_kind = str(optimizer_cfg.get("kind", "adam"))
    if optimizer_kind not in {"adam", "sgd"}:
        raise ValueError(
            "experiment.observation_subblock_inference.optimizer.kind must be 'adam' or 'sgd'."
        )
    base_lr = float(optimizer_cfg.get("base_lr", 1e-2))
    if base_lr <= 0.0:
        raise ValueError(
            "experiment.observation_subblock_inference.optimizer.base_lr must be > 0."
        )
    n_iter = int(optimizer_cfg.get("n_iter", 100))
    if n_iter <= 0:
        raise ValueError(
            "experiment.observation_subblock_inference.optimizer.n_iter must be > 0."
        )
    optimizer_kwargs = optimizer_cfg.get("kwargs", {})
    if not isinstance(optimizer_kwargs, dict):
        raise ValueError(
            "experiment.observation_subblock_inference.optimizer.kwargs must be a mapping/dict."
        )

    diagnostics_cfg = inference_cfg.get("diagnostics", {})
    if not isinstance(diagnostics_cfg, dict):
        raise ValueError(
            "experiment.observation_subblock_inference.diagnostics must be a mapping/dict."
        )
    write_plots = diagnostics_cfg.get("plots", True)
    if not isinstance(write_plots, bool):
        raise ValueError(
            "experiment.observation_subblock_inference.diagnostics.plots must be a bool."
        )

    outputs_cfg = experiment_cfg.get("outputs", {})
    if not isinstance(outputs_cfg, dict):
        raise ValueError("experiment.outputs must be a mapping/dict.")
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
        "truth": truth,
        "observation_subblock_inference": {
            "inputs": {
                "cube": cube_path,
                "trace": trace_path,
                "manifest": manifest_path,
            },
            "validate": {
                "require_contiguous_frame_index": require_contiguous,
                "require_monotonic_time": require_monotonic,
            },
            "init": {
                "x_position_as": x_init,
                "y_position_as": y_init,
                "position_angle_deg": pa_init,
            },
            "loss": {
                "reduce": reduce,
                "variance": variance_value,
            },
            "optimizer": {
                "kind": optimizer_kind,
                "base_lr": base_lr,
                "n_iter": n_iter,
                "kwargs": dict(optimizer_kwargs),
            },
            "diagnostics": {
                "plots": write_plots,
            },
        },
        "outputs": {
            "outdir": outdir_value,
            "file_prefix": file_prefix.strip(),
        },
        "notes": notes_value,
    }


def generate_obs_subblock_inference(
    *,
    config_path: Path | None = None,
    system_preset: str | None = None,
    results_dir: Path | None = None,
    run_name: str | None = None,
    dry_run: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run registration-only inference for an observation sub-block cube."""

    cfg_path = Path(config_path) if config_path is not None else DEFAULT_PRESCRIPTION_PATH
    print(f"Loading observation sub-block inference config from: {cfg_path}")
    user_cfg = load_user_config(
        config_path=cfg_path,
        system_preset=system_preset,
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
    inference_cfg = experiment["observation_subblock_inference"]
    inputs_cfg = inference_cfg["inputs"]

    cube_path = _resolve_relative_path(
        inputs_cfg["cube"],
        config_path=cfg_path,
        field_name="experiment.observation_subblock_inference.inputs.cube",
    )
    if not cube_path.exists():
        raise FileNotFoundError(f"Observation cube FITS not found: {cube_path}")

    manifest_path_value = inputs_cfg.get("manifest")
    manifest_path = (
        _resolve_relative_path(
            manifest_path_value,
            config_path=cfg_path,
            field_name="experiment.observation_subblock_inference.inputs.manifest",
        )
        if manifest_path_value is not None
        else find_obs_subblock_sidecar_manifest(cube_path)
    )
    manifest_auto_discovered = manifest_path_value is None and manifest_path is not None
    if manifest_auto_discovered:
        print(f"Using sibling render manifest: {manifest_path}")
    manifest_input = _load_manifest(manifest_path)

    explicit_trace_path = inputs_cfg.get("trace")
    trace_path = (
        _resolve_relative_path(
            explicit_trace_path,
            config_path=cfg_path,
            field_name="experiment.observation_subblock_inference.inputs.trace",
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

    trace: ObsSubblockTrace | None = None
    if trace_path is not None:
        validate_cfg = inference_cfg["validate"]
        trace = load_obs_subblock_trace_csv(
            trace_path,
            require_contiguous_frame_index=validate_cfg["require_contiguous_frame_index"],
            require_monotonic_time=validate_cfg["require_monotonic_time"],
        )

    comparison_trace: ObsSubblockTrace | None = trace
    if comparison_trace is not None and comparison_trace.frame_count != n_frame:
        print(
            "Warning: truth trace frame count does not match cube frame count; "
            "skipping truth-comparison outputs."
        )
        comparison_trace = None

    configured_outdir = experiment["outputs"]["outdir"]
    if results_dir is not None:
        outdir_root = Path(results_dir)
    elif configured_outdir is not None:
        outdir_root = _resolve_relative_path(
            configured_outdir,
            config_path=cfg_path,
            field_name="experiment.outputs.outdir",
        )
    else:
        outdir_root = DEFAULT_OUTDIR_ROOT

    stamp = timestamp_tag()
    run_label = run_name or stamp
    outdir = outdir_root / run_label
    artifacts = _build_artifact_paths(
        outdir=outdir,
        file_prefix=experiment["outputs"]["file_prefix"],
        timestamp=stamp,
        include_comparison=comparison_trace is not None,
        write_plots=bool(inference_cfg["diagnostics"]["plots"]),
    )

    if dry_run:
        print("Dry run: validated configuration and inputs.")
        print(f"  cube_path: {cube_path}")
        print(f"  frame_count: {n_frame}")
        print(f"  output_dir: {outdir}")
        for key, path in artifacts.items():
            print(f"  expected_{key}: {path}")
        return {
            "dry_run": True,
            "frame_count": n_frame,
            "output_dir": str(outdir),
            "artifacts": {name: str(path) for name, path in artifacts.items()},
        }

    outdir.mkdir(parents=True, exist_ok=True)

    print("Building fixed shared forward state...")
    forward_spec = compose_forward_spec(system_cfg)
    base_store = ParameterStore.from_spec_defaults(forward_spec)
    truth_overrides = _flatten_truth_overrides(experiment["truth"])
    primitive_truth_overrides, derived_truth_overrides, unknown_truth_keys = (
        partition_obs_subblock_overrides_by_kind(
            truth_overrides,
            forward_spec=forward_spec,
        )
    )
    if unknown_truth_keys:
        raise ValueError(
            "experiment.truth contains keys not present or unsupported in "
            "forward_spec: " + ", ".join(sorted(unknown_truth_keys.keys()))
        )
    base_store = apply_obs_subblock_overrides_preserving_derived(
        base_store,
        forward_spec=forward_spec,
        primitive_overrides=primitive_truth_overrides,
        derived_overrides=derived_truth_overrides,
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

    print("Initializing per-frame registration parameters...")
    theta0_matrix = _resolve_initial_theta_matrix(
        base_store=base_store,
        init_cfg=inference_cfg["init"],
        n_frame=n_frame,
    )
    theta0_flat = _theta_matrix_to_flat(theta0_matrix)

    loss_cfg = inference_cfg["loss"]
    variance_cube = np.full_like(cube, float(loss_cfg["variance"]), dtype=float)
    loss_fn = _build_block_loss_fn(
        binder=binder,
        forward_spec=forward_spec,
        base_store=base_store,
        cube_data=cube,
        variance_cube=variance_cube,
        reduce=str(loss_cfg["reduce"]),
    )

    optimizer_cfg = inference_cfg["optimizer"]
    print(
        "Running optimization: "
        f"kind={optimizer_cfg['kind']} n_iter={optimizer_cfg['n_iter']} "
        f"base_lr={optimizer_cfg['base_lr']}"
    )
    theta_final_flat, trace_history = run_shera_gd(
        loss_fn=loss_fn,
        theta0=jnp.asarray(theta0_flat),
        learning_rate=float(optimizer_cfg["base_lr"]),
        num_steps=int(optimizer_cfg["n_iter"]),
        optimizer_kind=str(optimizer_cfg["kind"]),
        optimizer_kwargs=dict(optimizer_cfg["kwargs"]),
        return_artifacts=False,
        show_progress=bool(show_progress),
    )

    theta_final = _theta_flat_to_matrix(np.asarray(theta_final_flat), n_frame=n_frame)
    loss_history = np.asarray(trace_history["loss"], dtype=float)
    initial_loss = float(loss_history[0])
    final_loss = float(loss_history[-1])

    model_cube = _predict_cube(
        binder=binder,
        forward_spec=forward_spec,
        base_store=base_store,
        theta_matrix=theta_final,
    )
    frame_nll = _compute_per_frame_nll(
        model_cube=model_cube,
        data_cube=cube,
        variance_cube=variance_cube,
        reduce=str(loss_cfg["reduce"]),
    )

    if comparison_trace is not None:
        times = np.asarray([float(row["time_s"]) for row in comparison_trace.rows], dtype=float)
    else:
        times = np.arange(n_frame, dtype=float)

    recovered_rows: list[dict[str, Any]] = []
    for frame_index in range(n_frame):
        recovered_rows.append(
            {
                "frame_index": int(frame_index),
                "time_s": float(times[frame_index]),
                FRAME_PARAM_KEYS[0]: float(theta_final[frame_index, 0]),
                FRAME_PARAM_KEYS[1]: float(theta_final[frame_index, 1]),
                FRAME_PARAM_KEYS[2]: float(theta_final[frame_index, 2]),
                "frame_nll": float(frame_nll[frame_index]),
            }
        )

    _write_csv_rows(
        output_path=artifacts["recovered_trace_csv"],
        rows=recovered_rows,
        fieldnames=(
            "frame_index",
            "time_s",
            FRAME_PARAM_KEYS[0],
            FRAME_PARAM_KEYS[1],
            FRAME_PARAM_KEYS[2],
            "frame_nll",
        ),
    )

    truth_matrix: np.ndarray | None = None
    if comparison_trace is not None:
        truth_matrix = np.asarray(
            [
                [
                    float(row[FRAME_PARAM_KEYS[0]]),
                    float(row[FRAME_PARAM_KEYS[1]]),
                    float(row[FRAME_PARAM_KEYS[2]]),
                ]
                for row in comparison_trace.rows
            ],
            dtype=float,
        )
        comparison_rows: list[dict[str, Any]] = []
        for frame_index in range(n_frame):
            comparison_rows.append(
                {
                    "frame_index": int(frame_index),
                    "time_s": float(times[frame_index]),
                    f"{FRAME_PARAM_KEYS[0]}_truth": float(truth_matrix[frame_index, 0]),
                    f"{FRAME_PARAM_KEYS[0]}_recovered": float(theta_final[frame_index, 0]),
                    f"{FRAME_PARAM_KEYS[0]}_residual": float(
                        theta_final[frame_index, 0] - truth_matrix[frame_index, 0]
                    ),
                    f"{FRAME_PARAM_KEYS[1]}_truth": float(truth_matrix[frame_index, 1]),
                    f"{FRAME_PARAM_KEYS[1]}_recovered": float(theta_final[frame_index, 1]),
                    f"{FRAME_PARAM_KEYS[1]}_residual": float(
                        theta_final[frame_index, 1] - truth_matrix[frame_index, 1]
                    ),
                    f"{FRAME_PARAM_KEYS[2]}_truth": float(truth_matrix[frame_index, 2]),
                    f"{FRAME_PARAM_KEYS[2]}_recovered": float(theta_final[frame_index, 2]),
                    f"{FRAME_PARAM_KEYS[2]}_residual": float(
                        theta_final[frame_index, 2] - truth_matrix[frame_index, 2]
                    ),
                    "frame_nll": float(frame_nll[frame_index]),
                }
            )
        _write_csv_rows(
            output_path=artifacts["truth_comparison_csv"],
            rows=comparison_rows,
            fieldnames=(
                "frame_index",
                "time_s",
                f"{FRAME_PARAM_KEYS[0]}_truth",
                f"{FRAME_PARAM_KEYS[0]}_recovered",
                f"{FRAME_PARAM_KEYS[0]}_residual",
                f"{FRAME_PARAM_KEYS[1]}_truth",
                f"{FRAME_PARAM_KEYS[1]}_recovered",
                f"{FRAME_PARAM_KEYS[1]}_residual",
                f"{FRAME_PARAM_KEYS[2]}_truth",
                f"{FRAME_PARAM_KEYS[2]}_recovered",
                f"{FRAME_PARAM_KEYS[2]}_residual",
                "frame_nll",
            ),
        )

    if inference_cfg["diagnostics"]["plots"]:
        _plot_loss_history(losses=loss_history, output_path=artifacts["loss_history_png"])
        if truth_matrix is not None:
            _plot_trace_comparison(
                times=times,
                recovered=theta_final,
                truth=truth_matrix,
                output_path=artifacts["trace_comparison_png"],
            )
            _plot_trace_residuals(
                times=times,
                recovered=theta_final,
                truth=truth_matrix,
                output_path=artifacts["trace_residuals_png"],
            )
        else:
            _plot_recovered_traces(
                times=times,
                recovered=theta_final,
                output_path=artifacts["recovered_traces_png"],
            )
        _plot_image_fit(
            data_cube=cube,
            model_cube=model_cube,
            output_path=artifacts["image_fit_png"],
        )

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
        "infer_keys": list(FRAME_PARAM_KEYS),
        "inputs": {
            "config_path": str(cfg_path.resolve()),
            "cube_fits": str(cube_path),
            "trace_csv": None if trace_path is None else str(trace_path),
            "manifest_json": None if manifest_path is None else str(manifest_path),
            "manifest_auto_discovered": bool(manifest_auto_discovered),
        },
        "init": {
            "x_position_as": float(theta0_matrix[0, 0]),
            "y_position_as": float(theta0_matrix[0, 1]),
            "position_angle_deg": float(theta0_matrix[0, 2]),
        },
        "loss": {
            "noise_model": "gaussian",
            "reduce": loss_cfg["reduce"],
            "variance": float(loss_cfg["variance"]),
        },
        "optimizer": {
            "kind": optimizer_cfg["kind"],
            "base_lr": float(optimizer_cfg["base_lr"]),
            "n_iter": int(optimizer_cfg["n_iter"]),
            "kwargs": dict(optimizer_cfg["kwargs"]),
        },
        "metrics": {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_delta": final_loss - initial_loss,
            "mean_frame_nll": float(np.mean(frame_nll)),
        },
        "truth_comparison_available": comparison_trace is not None,
        "system": system_info,
        "shared_truth": to_jsonable_obs_subblock_payload(experiment["truth"]),
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
        "initial_loss": initial_loss,
        "final_loss": final_loss,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run registration-only observation sub-block inference."
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


def main() -> None:
    args = _build_parser().parse_args()
    generate_obs_subblock_inference(
        config_path=args.config,
        system_preset=args.system_preset,
        results_dir=args.results_dir,
        run_name=args.run_name,
        dry_run=bool(args.dry_run),
        show_progress=not bool(args.no_progress),
    )


if __name__ == "__main__":
    main()
