"""
Generate a forward-model training image sweep for Shera three-plane models.

What this script does
---------------------
This script mirrors the explicit, top-to-bottom workflow in
``examples/recipes/canonical_astrometry.py`` while focusing on dataset
generation for machine-learning workflows. It sweeps one inference-facing
parameter at a time, evaluates the forward model, and writes out a structured
dataset consisting of:

- A run-level ``manifest.json`` capturing settings, baseline values, sweep
  configuration, and reproducibility metadata.
- A line-oriented ``samples.jsonl`` file with one record per sample.
- Paired FITS + JSON sidecars for each sample in ``images/``.

Outputs (run directory layout)
------------------------------
- ``manifest.json``: run configuration and reproducibility metadata.
- ``samples.jsonl``: one record per sample with sweep metadata and file paths.
- ``images/``:
  - ``sample_<id>.fits``: the model image for each sweep sample.
  - ``sample_<id>.json``: a JSON sidecar with the same sweep metadata plus
    nominal/applied parameter values.

Data model summary
------------------
- ``sample_id``: monotonically increasing identifier assigned as samples emit.
- ``sweepkey``: the parameter key being perturbed for a given sample.
- ``delta_sigma``: the delta expressed in sigma units (only for ``fim_sigma``).
- ``delta_value``: the numeric delta added to the nominal value.
- ``applied_value``: nominal + delta_value (the actual value in the model).

Scalar vs. component sweeps
---------------------------
- Scalar parameters (e.g., separation, plate scale) are swept directly.
- Component parameters (Zernike coefficient vectors) are swept one component
  at a time; the component index selects which coefficient is perturbed.

Plate scale policy
------------------
The dataset is intended for direct differencing on a common pixel grid. When
perturbing the system plate scale, it is assumed that the effective focal
length changes but the detector grid does not. Perturbed images should be
treated as living on the same X/Y axes as the nominal image and can therefore
be directly differenced in post-processing.

Key behaviors
-------------
- Sweeps the same ``INFER_KEYS`` as the canonical astrometry recipe.
- Perturbs scalar parameters and individual Zernike components (one at a time).
- Supports two delta selection modes:
    * ``fim_sigma`` (default): uses a Fisher diagonal to scale deltas.
    * ``fixed``: uses explicit per-key step sizes.
- Starts noiseless by default but supports opt-in shot-noise consistent with
  the canonical recipe (Gaussian approximation for bright pixels, Poisson
  otherwise).

Usage (examples)
---------------
Generate a default dataset (FIM-scaled deltas, noiseless). This writes to
``Results/ml_training_dataset_<timestamp>``:

    python work/experiments/generate_training_dataset.py

Specify a custom run name (suffix) with the default base directory. This
writes to ``Results/ml_training_dataset_sweep_v0``:

    python work/experiments/generate_training_dataset.py --run-name sweep_v0

Specify a custom run name and output base directory. When both are provided,
the run name is used directly as the subdirectory (no default prefix):

    python work/experiments/generate_training_dataset.py \\
        --run-name sweep_v0 \\
        --outdir Results/custom_runs

Specify only an output base directory (timestamped subdirectory):

    python work/experiments/generate_training_dataset.py \\
        --outdir Results/custom_runs

Use fixed-step deltas with a JSON override file:

    python work/experiments/generate_training_dataset.py \\
        --delta-mode fixed \\
        --steps-json work/experiments/steps.json

Enable shot noise (using canonical logic) and set seed:

    python work/experiments/generate_training_dataset.py --add-noise --seed 7

File roadmap
------------
- Constants / defaults
- I/O helpers
- Sweep helpers
- Emission helpers
- main()

Notes
-----
- This script is intentionally verbose and explicit to support auditability.
- Helper functions are defined locally; consider migrating shared utilities to
  ``utils/io.py`` or similar if this script becomes productionized.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from astropy.io import fits

from dluxshera.inference.optimization import fim_theta, generate_fim_labels
from dluxshera.inference.optimization import make_binder_nll_fn
from dluxshera.params.packing import build_index_map, pack_params
from dluxshera.params.spec import build_inference_spec_basic, make_inference_subspec
from dluxshera.params.store import ParameterStore, strip_structural
from dluxshera.systems.three_plane import (
    SHERA_TESTBED_CONFIG,
    SheraThreePlaneBinder,
    SheraThreePlaneConfig,
    build_forward_spec_from_config,
)

JAX_ENABLE_X64 = True

# =============================================================================
# Section: Inference sweep defaults
# =============================================================================
INFER_KEYS = (
    "binary.separation_as",
    "binary.position_angle_deg",
    "binary.x_position_as",
    "binary.y_position_as",
    "binary.log_flux_total",
    "binary.contrast",
    "system.plate_scale_as_per_pix",
    "primary.zernike_coeffs_nm",
    "secondary.zernike_coeffs_nm",
)

DEFAULT_FIXED_STEPS = {
    "binary.separation_as": 1e-4,
    "binary.position_angle_deg": 1e-3,
    "binary.x_position_as": 1e-3,
    "binary.y_position_as": 1e-3,
    "binary.log_flux_total": 1e-3,
    "binary.contrast": 1e-3,
    "system.plate_scale_as_per_pix": 1e-5,
    "primary.zernike_coeffs_nm": 5.0,
    "secondary.zernike_coeffs_nm": 5.0,
}
# Example steps.json payloads:
# {
#   "primary.zernike_coeffs_nm": 5.0,  # scalar expands to all components
#   "secondary.zernike_coeffs_nm": [5.0, 5.0, 4.0, 4.0, 3.0, 3.0, 2.0, 2.0]
# }

# =============================================================================
# Section: I/O helpers
# =============================================================================
def _serialize_value(value: Any) -> Any:
    """Return JSON-serializable representations for metadata output.

    TODO: migrate to a shared JSON utility (e.g., utils/io.py).
    """
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return np.asarray(value).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(val) for key, val in value.items()}
    return value


def _parse_csv_levels(levels: str | None, default: Iterable[float]) -> list[float]:
    """Parse comma-separated delta levels, falling back to a default list.

    TODO: migrate to a shared CLI parsing utility.
    """
    # Default levels are supplied by the caller (fim sigma or fixed delta mode).
    if levels is None:
        return list(default)
    parsed = [float(item.strip()) for item in levels.split(",") if item.strip()]
    if not parsed:
        raise ValueError("delta-levels must contain at least one value.")
    return parsed


def _load_steps_json(steps_json: str | None) -> dict[str, float | list[float]]:
    """Load a fixed-step override dictionary from JSON.

    TODO: migrate to utils/io.py if steps overrides are reused elsewhere.
    """
    # Accept either a JSON string or a path to a JSON file to keep the CLI terse.
    if steps_json is None:
        return {}
    path = Path(steps_json)
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        payload = json.loads(steps_json)
    if isinstance(payload, dict) and "steps" in payload:
        payload = payload["steps"]
    if not isinstance(payload, dict):
        raise ValueError("steps-json must decode to a dict of key -> step size.")
    parsed: dict[str, float | list[float]] = {}
    for key, val in payload.items():
        if isinstance(val, list):
            parsed[str(key)] = [float(item) for item in val]
        else:
            parsed[str(key)] = float(val)
    return parsed


def _resolve_run_dir(outdir: str | None, run_name: str | None) -> Path:
    """Resolve the output directory for the sweep run.

    Rules:
      1) outdir is None and run_name is None:
         repo_root/Results/ml_training_dataset_<timestamp>
      2) outdir is None and run_name is provided:
         repo_root/Results/ml_training_dataset_<run_name>
      3) outdir is provided and run_name is None:
         outdir/ml_training_dataset_<timestamp>
      4) outdir is provided and run_name is provided:
         outdir/<run_name>  (note: no default prefix)

    TODO: migrate to utils/io.py if other scripts adopt this pattern.
    """
    # This is intentionally deterministic: same inputs always map to same path.
    repo_root = Path(__file__).resolve().parents[2]
    prefix = "ml_training_dataset_"
    if outdir is None:
        suffix = run_name or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return repo_root / "Results" / f"{prefix}{suffix}"
    base = Path(outdir).expanduser().resolve()
    if run_name is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return base / f"{prefix}{timestamp}"
    return base / run_name


def _git_commit() -> str | None:
    """Return the current git commit hash, or None if unavailable.

    TODO: migrate to utils/versioning.py if reused.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


# =============================================================================
# Section: Sweep helpers
# =============================================================================
def _noise_model(
    rng_key: jax.Array,
    data: jax.Array,
    *,
    add_noise: bool,
) -> tuple[jax.Array, str, int | None]:
    """Apply canonical shot-noise if requested; otherwise return data as-is.

    TODO: migrate to a shared noise helper in inference/sim utils.
    """
    # Keep noise logic mirrored to canonical_astrometry for reproducibility.
    if not add_noise:
        return data, "none", None
    rng_key, split_key = jr.split(rng_key)
    if np.min(np.asarray(data)) > 100:
        noisy = np.sqrt(np.asarray(data)) * jr.normal(split_key, data.shape) + data
        return noisy, "gaussian-approx", int(np.asarray(split_key)[0])
    noisy = jr.poisson(split_key, data)
    return noisy, "poisson", int(np.asarray(split_key)[0])


def _validate_fim_diag(
    fim_diag: np.ndarray,
    *,
    labels: list[str],
) -> None:
    """Validate FIM diagonal entries and raise on invalid values.

    TODO: migrate to inference/optimization helpers if this becomes shared.
    """
    # Guard against zero/negative/NaN sigmas before using 1/sqrt(F).
    for idx, val in enumerate(fim_diag):
        if not np.isfinite(val) or val <= 0:
            label = labels[idx] if idx < len(labels) else f"index {idx}"
            warnings.warn(
                f"Invalid FIM diagonal entry for {label}: {val}.", RuntimeWarning
            )
            raise ValueError(
                f"FIM diagonal entry for {label} is invalid ({val}); "
                "cannot compute sigma scaling."
            )


# =============================================================================
# Section: Emission helpers
# =============================================================================
def _write_fits(
    *,
    output_path: Path,
    image: np.ndarray,
    header_data: dict[str, Any],
) -> None:
    """Write a FITS file with a minimal header.

    TODO: migrate to utils/io.py for shared FITS output.
    """
    # FITS headers ignore None-valued entries to keep headers compact.
    header = fits.Header()
    for key, value in header_data.items():
        if value is None:
            continue

        # Allow (value, comment) tuples.
        if isinstance(value, tuple) and len(value) == 2:
            card_value, comment = value
            header.set(str(key).upper(), card_value, comment=str(comment))
        else:
            header.set(str(key).upper(), value)

    fits.PrimaryHDU(data=image, header=header).writeto(output_path, overwrite=True)


# =============================================================================
# Section: Main entrypoint
# =============================================================================
def main() -> None:
    """Run the forward-model training dataset sweep."""
    # =============================================================================
    # Section: Parse CLI args and set defaults
    # =============================================================================
    parser = argparse.ArgumentParser(
        description="Generate a forward-model training dataset sweep.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help=(
            "Optional base directory for outputs. When set, the run is written "
            "to a subdirectory named by the run name or a timestamp."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional run name. When outdir is unset, it replaces the default "
            "timestamp suffix. When outdir is set, it becomes the subdirectory "
            "name (no default prefix)."
        ),
    )
    parser.add_argument("--exclude-secondary-zernikes", action="store_true")
    parser.add_argument(
        "--delta-mode",
        choices=("fixed", "fim_sigma"),
        default="fim_sigma",
    )
    parser.add_argument(
        "--delta-levels",
        type=str,
        default=None,
        help="Comma-separated list of delta levels (e.g., -1,0,1).",
    )
    parser.add_argument(
        "--sigma-k",
        type=float,
        default=1.0,
        help="Sigma scale k for fim_sigma mode when delta-levels is not set.",
    )
    parser.add_argument(
        "--steps-json",
        type=str,
        default=None,
        help="JSON string or path with fixed-step overrides.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--add-noise", action="store_true")
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", JAX_ENABLE_X64)

    repo_root = Path(__file__).resolve().parents[2]
    # =============================================================================
    # Section: Resolve output directory and run naming policy
    # =============================================================================
    run_dir = _resolve_run_dir(args.outdir, args.run_name)
    run_dir_relative = Path(os.path.relpath(run_dir, repo_root))
    prefix = "ml_training_dataset_"
    if args.run_name is not None:
        resolved_run_name = args.run_name
    else:
        # Derive from run_dir for cases that use a prefixed timestamp subdir.
        resolved_run_name = (
            run_dir.name[len(prefix) :]
            if run_dir.name.startswith(prefix)
            else run_dir.name
        )
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # =============================================================================
    # Section: Initialize config + binder + parameter stores
    # =============================================================================
    cfg: SheraThreePlaneConfig = SHERA_TESTBED_CONFIG
    cfg = cfg.replace(primary_noll_indices=tuple(range(4, 12)))
    cfg = cfg.replace(secondary_noll_indices=tuple(range(4, 12)))
    if args.exclude_secondary_zernikes:
        cfg = cfg.replace(secondary_noll_indices=None)

    forward_spec = build_forward_spec_from_config(cfg)
    inference_spec = build_inference_spec_basic(cfg)

    base_store = ParameterStore.from_spec_defaults(forward_spec)
    base_store = base_store.refresh_derived(forward_spec)

    binder = SheraThreePlaneBinder(cfg, forward_spec, base_store)

    # =============================================================================
    # Section: Define infer keys, sweep keys, and sweep schedule
    # =============================================================================
    infer_keys = list(INFER_KEYS)
    if args.exclude_secondary_zernikes:
        infer_keys = [
            key for key in infer_keys if key != "secondary.zernike_coeffs_nm"
        ]

    inference_subspec = make_inference_subspec(
        base_spec=inference_spec,
        infer_keys=infer_keys,
        cfg=cfg,
    )

    # =============================================================================
    # Section: Noise configuration and seeding model
    # =============================================================================
    rng_key = jr.PRNGKey(args.seed)

    data = binder.model()
    data, noise_mode, noise_seed = _noise_model(rng_key, data, add_noise=args.add_noise)
    data_var = data

    nll_loss_fn, theta0 = make_binder_nll_fn(
        binder=binder,
        infer_keys=infer_keys,
        data=data,
        var=data_var,
        noise_model="gaussian",
        reduce="sum",
        theta0_store=base_store,
    )

    theta_ref = pack_params(inference_subspec, base_store)
    F = fim_theta(nll_loss_fn, theta_ref)
    fim_diag = np.asarray(jnp.diag(F))
    fim_labels = generate_fim_labels(infer_keys, cfg=cfg, store=base_store)
    _validate_fim_diag(fim_diag, labels=fim_labels)

    index_map = build_index_map(inference_subspec, base_store, theta=theta_ref)

    # =============================================================================
    # Section: Define infer keys, sweep keys, and sweep schedule (steps + sizes)
    # =============================================================================
    steps = _parse_csv_levels(
        args.delta_levels,
        default=(
            (-args.sigma_k, 0.0, args.sigma_k)
            if args.delta_mode == "fim_sigma"
            else (-1.0, 0.0, 1.0)
        ),
    )

    step_overrides = _load_steps_json(args.steps_json)
    fixed_steps = {**DEFAULT_FIXED_STEPS, **step_overrides}

    def _sigma_for_key_component(param_key: str, component_index: int | None) -> float:
        """Compute sigma from the FIM diagonal for a key/component.

        Component_index is None for scalar parameters; Zernike sweeps map
        component indices into the flattened inference vector.
        """
        for entry in index_map["entries"]:
            if entry["name"] != param_key:
                continue
            start = int(entry["start"])
            if component_index is None:
                return float(1.0 / math.sqrt(fim_diag[start]))
            return float(1.0 / math.sqrt(fim_diag[start + component_index]))
        raise KeyError(f"Missing FIM mapping for key {param_key}.")

    def _expand_fixed_step(
        param_key: str,
        baseline_value: Any,
        step_value: float | list[float],
    ) -> float | list[float]:
        """Normalize fixed-step sizes to scalars or component-wise lists.

        Scalars remain scalars; vector parameters expand to per-component lists.
        """
        baseline_array = np.asarray(baseline_value)
        is_vector = baseline_array.shape != ()
        if is_vector:
            n_components = int(baseline_array.size)
            if isinstance(step_value, list):
                if len(step_value) != n_components:
                    raise ValueError(
                        f"Fixed-step override for {param_key} must have "
                        f"{n_components} entries (got {len(step_value)})."
                    )
                return [float(item) for item in step_value]
            return [float(step_value)] * n_components
        if isinstance(step_value, list):
            if len(step_value) == 1:
                return float(step_value[0])
            raise ValueError(
                f"Fixed-step override for {param_key} must be a scalar "
                "or a single-element list."
            )
        return float(step_value)

    def _fixed_step_sizes() -> dict[str, float | list[float]]:
        """Build normalized fixed step sizes for all inference keys."""
        sizes: dict[str, float | list[float]] = {}
        for key in infer_keys:
            if key not in fixed_steps:
                raise KeyError(
                    f"Fixed-step mode requires a step size for {key}."
                )
            sizes[key] = _expand_fixed_step(key, base_store.get(key), fixed_steps[key])
        return sizes

    def _infer_values(store: ParameterStore, keys: list[str]) -> dict[str, Any]:
        """Extract infer keys into a JSON-serializable dict."""
        return {key: _serialize_value(store.get(key)) for key in keys}

    # Split scalar keys vs. Zernike-vector keys to simplify sweep loops.
    scalar_keys = [
        key
        for key in infer_keys
        if key not in ("primary.zernike_coeffs_nm", "secondary.zernike_coeffs_nm")
    ]
    zernike_keys = [
        key
        for key in infer_keys
        if key in ("primary.zernike_coeffs_nm", "secondary.zernike_coeffs_nm")
    ]

    # Map each Zernike key to its Noll indices for metadata bookkeeping.
    zernike_map = {
        "primary.zernike_coeffs_nm": tuple(cfg.primary_noll_indices),
        "secondary.zernike_coeffs_nm": tuple(cfg.secondary_noll_indices),
    }

    def _fim_step_sizes() -> dict[str, float | list[float]]:
        """Build FIM-derived step sizes for all inference keys."""
        sizes: dict[str, float | list[float]] = {}
        for key in scalar_keys:
            sizes[key] = _sigma_for_key_component(key, None)
        for key in zernike_keys:
            coeffs = np.asarray(base_store.get(key))
            sizes[key] = [
                _sigma_for_key_component(key, idx) for idx in range(coeffs.size)
            ]
        return sizes

    if args.delta_mode == "fim_sigma":
        step_sizes = _fim_step_sizes()
    else:
        step_sizes = _fixed_step_sizes()

    def _fixed_step_for_component(
        param_key: str,
        component_index: int | None,
    ) -> float:
        """Return a fixed step size for a key/component from normalized sizes."""
        step = step_sizes[param_key]
        if isinstance(step, list):
            if component_index is None:
                raise ValueError(
                    f"Fixed-step size for {param_key} requires a component index."
                )
            return float(step[component_index])
        return float(step)

    # =============================================================================
    # Section: Emitting artifacts (manifest + samples index)
    # =============================================================================
    baseline_infer = _infer_values(base_store, infer_keys)
    manifest = {
        "run_name": resolved_run_name,
        "run_dir": str(run_dir_relative),
        "config_id": cfg.design_name,
        "git_commit": _git_commit(),
        "parameters": infer_keys,
        "nominal_values": baseline_infer,
        "delta_policy": {
            "mode": args.delta_mode,
            "steps": steps,
            "step_sizes": step_sizes,
        },
        "zernike_noll_indices": {
            "primary": zernike_map["primary.zernike_coeffs_nm"],
            "secondary": zernike_map["secondary.zernike_coeffs_nm"],
        },
        "noise": {
            "enabled": bool(args.add_noise),
            "mode": noise_mode,
            "seed": args.seed,
            "realization_seed": noise_seed,
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    samples_path = run_dir / "samples.jsonl"
    samples_path.write_text("")

    sample_id = 0

    # =============================================================================
    # Section: Emission helpers (FITS, sidecar JSON, JSONL index)
    # =============================================================================
    def _emit_sample(
        *,
        param_key: str,
        component_index: int | None,
        noll_index: int | None,
        delta_sigma: float | None,
        delta_value: float,
        nominal_value: float,
        applied_value: float,
        applied_store: ParameterStore,
    ) -> None:
        """Write a single sample to FITS, JSON sidecar, and the JSONL index."""
        nonlocal sample_id
        sample_id += 1
        sample_tag = f"sample_{sample_id:06d}"
        fits_path = images_dir / f"{sample_tag}.fits"
        meta_path = images_dir / f"{sample_tag}.json"
        nominal_plate_scale = float(np.asarray(binder.plate_scale_as_per_pix))

        model = binder.model(
            strip_structural(
                applied_store, structural_keys=binder.structural_store_keys()
            )
        )
        model_np = np.asarray(model)

        # FITS header conventions:
        # - BUNIT: data units (photons)
        # - XUNIT/YUNIT + XSCALE/YSCALE: physical units and pixel scale
        # - COMPIDX/NOLL: component bookkeeping for vector sweeps (None if scalar)
        # - NOISE/SEED: noise toggle + base seed (realization_seed in JSON only)
        # Optional fields: FITS omits None-valued keys; JSON retains nulls for a
        # stable schema across scalar vs component sweeps.
        header_data = {
            "BUNIT": ("photons", "Pixel values are photon counts"),
            "XUNIT": ("arcsec", "X-axis units"),
            "YUNIT": ("arcsec", "Y-axis units"),
            "XSCALE": (nominal_plate_scale, "Arcsec per pixel (nominal)"),
            "YSCALE": (nominal_plate_scale, "Arcsec per pixel (nominal)"),
            "SAMPLEID": (sample_id, "Training dataset sample id"),
            "SWEEPKEY": (param_key, "Swept parameter key"),
            "COMPIDX": (component_index, "Vector component index (if applicable)"),
            "NOLL": (noll_index, "Zernike Noll index (if applicable)"),
            "DELSIG": (delta_sigma, "Delta in sigma units"),
            "DELVAL": (delta_value, "Additive delta applied to nominal"),
            "APPLVAL": (applied_value, "Nominal + delta"),
            "NOISE": (bool(args.add_noise), "Noise added to image"),
            "SEED": (args.seed, "Base RNG seed"),
            "CFGID": (cfg.design_name, "Optics config id"),
        }
        _write_fits(output_path=fits_path, image=model_np, header_data=header_data)

        applied_infer = _infer_values(applied_store, infer_keys)
        sample_meta = {
            "run_name": resolved_run_name,
            "config_id": cfg.design_name,
            "sample_id": sample_id,
            "sweep": {
                "param_key": param_key,
                # Scalars: component_index and noll_index remain null (not 0).
                # Zernike sweeps: component_index is the coefficient index,
                # and noll_index is the corresponding Noll label when known.
                "component_index": component_index,
                "noll_index": noll_index,
                "nominal": nominal_value,
                "delta": delta_value,
                "applied": applied_value,
            },
            "values": applied_infer,
            "noise": {
                "enabled": bool(args.add_noise),
                "mode": noise_mode,
                "seed": args.seed,
                "realization_seed": noise_seed,
            },
        }
        meta_path.write_text(json.dumps(sample_meta, indent=2))

        record = {
            "sample_id": sample_id,
            "sample_tag": sample_tag,
            "fits_path": str(fits_path.relative_to(run_dir)),
            "metadata_path": str(meta_path.relative_to(run_dir)),
            "param_key": param_key,
            "component_index": component_index,
            "noll_index": noll_index,
            "delta_sigma": delta_sigma,
            "delta_value": delta_value,
            "nominal_value": nominal_value,
            "applied_value": applied_value,
            "noise_enabled": bool(args.add_noise),
            "noise_seed": args.seed,
        }
        with samples_path.open("a") as handle:
            handle.write(json.dumps(_serialize_value(record)) + "\n")

    # =============================================================================
    # Section: Dataset generation loop (nominal + perturbed images)
    # =============================================================================
    for key in scalar_keys:
        nominal = float(base_store.get(key))
        for level in steps:
            if args.delta_mode == "fim_sigma":
                sigma = _sigma_for_key_component(key, None)
                delta_value = float(level) * sigma
                delta_sigma = float(level)
            else:
                step = _fixed_step_for_component(key, None)
                delta_value = float(level) * step
                delta_sigma = None
            applied = nominal + delta_value
            store_delta = base_store.replace({key: applied}).refresh_derived(forward_spec)
            _emit_sample(
                param_key=key,
                # Scalars: component_index and noll_index are intentionally null.
                component_index=None,
                noll_index=None,
                delta_sigma=delta_sigma,
                delta_value=delta_value,
                nominal_value=nominal,
                applied_value=applied,
                applied_store=store_delta,
            )

    for key in zernike_keys:
        coeffs = np.asarray(base_store.get(key))
        noll_indices = zernike_map.get(key, ())
        for idx in range(coeffs.size):
            nominal = float(coeffs[idx])
            for level in steps:
                if args.delta_mode == "fim_sigma":
                    sigma = _sigma_for_key_component(key, idx)
                    delta_value = float(level) * sigma
                    delta_sigma = float(level)
                else:
                    step = _fixed_step_for_component(key, idx)
                    delta_value = float(level) * step
                    delta_sigma = None
                updated = coeffs.copy()
                updated[idx] = nominal + delta_value
                store_delta = base_store.replace({key: updated}).refresh_derived(
                    forward_spec
                )
                # Zernikes: component_index refers to coefficient array index,
                # while noll_index is the corresponding Noll label (if available).
                # Use None instead of 0 for scalar sweeps to avoid ambiguity.
                noll_index = int(noll_indices[idx]) if idx < len(noll_indices) else None
                _emit_sample(
                    param_key=key,
                    component_index=idx,
                    noll_index=noll_index,
                    delta_sigma=delta_sigma,
                    delta_value=delta_value,
                    nominal_value=nominal,
                    applied_value=updated[idx],
                    applied_store=store_delta,
                )

    # =============================================================================
    # Section: Final summary printouts / sanity checks
    # =============================================================================
    # (No explicit summary printouts yet; intended for future enhancements.)


if __name__ == "__main__":
    main()
