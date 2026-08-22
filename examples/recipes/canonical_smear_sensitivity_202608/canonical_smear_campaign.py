"""Prepare and review canonical prescribed-MC line-smear sensitivity campaigns.

This script is intentionally campaign-local. It generates deterministic
single-run prescribed-Monte-Carlo prescriptions for controlled line-smear
truth/model mismatches, and it postprocesses finished prescribed-MC runs with
canonical NLL derivative diagnostics at the physical truth parameter vector.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config, resolve_system_config
from dluxshera.inference.optimization import (
    fim_theta,
    generate_fim_labels,
    make_binder_nll_fn,
)
from dluxshera.inference.prior import PriorSpec
from dluxshera.params.packing import pack_params
from dluxshera.params.store import ParameterStore
from dluxshera.systems import SheraBinder
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.chi2_diagnostics import reduced_chi2_between_images
from dluxshera.utils.detector_layer_overrides import (
    apply_detector_layer_overrides,
    detector_layer_stack,
    get_detector_layer,
)
from dluxshera.utils.noise import apply_observation_noise

RECIPE_DIR = Path(__file__).resolve().parent
REPO_ROOT = RECIPE_DIR.parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Results" / "canonical_smear_sensitivity_202608"
PRESCRIBED_RUNNER = REPO_ROOT / "examples" / "recipes" / "prescribed_monte_carlo.py"
SYSTEM_PRESET = "SHERA_FLIGHT_3P_CONV"
GATTACA2_CONDA_ENV_PREFIX = "/scratch-jpl/shera_hpc/dmckeith/conda/envs/dluxshera-py311"

FAMILY_A_LENGTHS = (0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 2.0)
FAMILY_BC_LENGTHS = (0.5, 1.0)
MISMATCH_GRID = (-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20)
ORIENTATIONS = ("parallel", "perpendicular")

SMEAR_LAYER_NAME = "smear"
JITTER_LAYER_NAME = "jitter"
SMEAR_SIGMA_PERP_PIX = 0.1
SMEAR_KERNEL_SIZE = 11
SMEAR_UNITS = "detector_pix"
EXPOSURE_TIME_S = 0.05
SEPARATION_INDEX = 0
NLL_PSEUDOINVERSE_RTOL = 1.0e-10
PARAMETER_COUNT = 23
PRESCRIBED_MC_SEED = 20260821
PRODUCTION_N_ITER = 200
PRODUCTION_OPTIMIZER_KIND = "sgd"
PRODUCTION_BASE_LR = 0.7
PRODUCTION_SCHEDULE = {
    "kind": "linear_warmup",
    "warmup_steps": 10,
    "start_factor": 0.125,
}
PRODUCTION_EARLY_STOPPING = {
    "enabled": True,
    "min_iter": 40,
    "patience": 10,
    "loss_rtol": 1.0e-8,
    "require_finite_loss": True,
    "restore_best": True,
    "monitor": "loss",
}
PRODUCTION_PRIORS = {
    "source.separation_as": {"dist": "Normal", "sigma": 1.0e-4},
    "source.position_angle_deg": {"dist": "Uniform", "sigma": 1.0e-2},
    "source.x_position_as": {"dist": "Normal", "sigma": 1.0e-2},
    "source.y_position_as": {"dist": "Normal", "sigma": 1.0e-2},
    "source.log_flux_total": {"dist": "LogNormal", "sigma": 1.0e-4},
    "source.contrast": {"dist": "LogNormal", "sigma": 1.0e-4},
    "optics.plate_scale_as_per_pix": {"dist": "LogNormal", "sigma": 1.0e-3},
    "optics.primary.zernike_coeffs_nm": {"dist": "Normal", "sigma": 2.0},
    "optics.secondary.zernike_coeffs_nm": {"dist": "Normal", "sigma": 2.0},
}
SLURM_CONDITION_TIME = "00:30:00"
SLURM_AGGREGATE_TIME = "03:00:00"
SLURM_CPUS_PER_TASK = 2
SLURM_MEMORY = "24G"

INFER_KEYS = (
    "source.separation_as",
    "source.position_angle_deg",
    "source.x_position_as",
    "source.y_position_as",
    "source.log_flux_total",
    "source.contrast",
    "optics.plate_scale_as_per_pix",
    "optics.primary.zernike_coeffs_nm",
    "optics.secondary.zernike_coeffs_nm",
)

NUISANCE_KEYS = (
    "source.x_position_as",
    "source.y_position_as",
    "source.position_angle_deg",
)


@dataclasses.dataclass(frozen=True)
class SmearCondition:
    """Single campaign condition resolved into truth and model line kernels."""

    family: str
    run_id: str
    L_truth_pix: float
    L_model_pix: float
    epsilon_L_percent: float | None
    orientation: str
    phi_truth_deg: float
    theta_truth_deg: float
    delta_theta_deg: float
    theta_model_deg: float
    truth_kernel: dict[str, Any] | None
    model_kernel: dict[str, Any] | None
    notes: str = ""


def _length_token(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace("+", "p")
    return text.replace(".", "p")


def _signed_token(value: float, *, suffix: str = "") -> str:
    numeric = float(value)
    prefix = "p" if numeric >= 0.0 else "m"
    mag = f"{abs(numeric):g}".replace(".", "p")
    return f"{prefix}{mag}{suffix}"


def _kernel_from_requested_length(
    *,
    requested_length_pix: float,
    theta_deg: float,
) -> dict[str, Any] | None:
    """Return a native line-kernel mapping, or ``None`` for no physical smear."""

    requested = float(requested_length_pix)
    if requested == 0.0:
        return None
    return {
        "kind": "line",
        "length": requested,
        "requested_length": requested,
        "theta_deg": float(theta_deg),
        "sigma_perp": SMEAR_SIGMA_PERP_PIX,
        "kernel_size": SMEAR_KERNEL_SIZE,
        "units": SMEAR_UNITS,
    }


def line_smear_layer(kernel: Mapping[str, Any]) -> dict[str, Any]:
    """Return a named native detector smear layer for a line kernel."""

    stored_kernel = {
        key: value
        for key, value in dict(kernel).items()
        if key not in {"requested_length"}
    }
    return {
        "name": SMEAR_LAYER_NAME,
        "kind": "ApplyConvolution",
        "kernel": stored_kernel,
    }


def binary_position_angle_deg() -> float:
    """Resolve the current canonical binary position angle from defaults."""

    cfg = resolve_config({"system": {"preset": SYSTEM_PRESET}})
    spec = compose_forward_spec(cfg["system"])
    store = ParameterStore.from_spec_defaults(spec).refresh_derived(spec)
    return float(store.get("source.position_angle_deg"))


def _base_system_config() -> dict[str, Any]:
    """Return the campaign baseline resolved from the authoritative preset."""

    return resolve_system_config(
        {
            "preset": SYSTEM_PRESET,
            "source": {"exposure_time_s": EXPOSURE_TIME_S},
        }
    )


def _smear_layer_override(kernel: Mapping[str, Any] | None) -> dict[str, Any]:
    if kernel is None:
        return {"action": "remove"}
    return {
        "action": "update",
        "kind": "ApplyConvolution",
        "kernel": line_smear_layer(kernel)["kernel"],
    }


def system_config_for_smear(
    smear_kernel: Mapping[str, Any] | None,
    *,
    context: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the preset system and apply only campaign detector deltas."""

    base = _base_system_config()
    return apply_detector_layer_overrides(
        base,
        {
            "layers": {
                JITTER_LAYER_NAME: {"action": "remove"},
                SMEAR_LAYER_NAME: _smear_layer_override(smear_kernel),
            }
        },
        context=context,
    )


def _config_without_smear(system_config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(system_config))
    detector = cfg.get("detector")
    if isinstance(detector, dict):
        detector["layers"] = [
            layer
            for layer in detector.get("layers", [])
            if not (isinstance(layer, Mapping) and layer.get("name") == SMEAR_LAYER_NAME)
        ]
    return cfg


def truth_inference_config_audit(
    condition: SmearCondition,
    *,
    truth_system: Mapping[str, Any] | None = None,
    inference_system: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a focused audit of the deliberate truth/inference smear mismatch."""

    if truth_system is None:
        truth_system, _ = system_config_for_smear(condition.truth_kernel, context="audit.truth")
    if inference_system is None:
        inference_system, _ = system_config_for_smear(condition.model_kernel, context="audit.inference")

    truth_smear = get_detector_layer(truth_system, SMEAR_LAYER_NAME)
    inference_smear = get_detector_layer(inference_system, SMEAR_LAYER_NAME)
    truth_kernel = truth_smear.get("kernel") if isinstance(truth_smear, Mapping) else None
    inference_kernel = (
        inference_smear.get("kernel") if isinstance(inference_smear, Mapping) else None
    )

    mismatch_fields: list[str] = []
    if truth_kernel != inference_kernel:
        if not isinstance(truth_kernel, Mapping) or not isinstance(inference_kernel, Mapping):
            mismatch_fields.append("detector.layers.smear")
        else:
            for key in sorted(set(truth_kernel) | set(inference_kernel)):
                if truth_kernel.get(key) != inference_kernel.get(key):
                    mismatch_fields.append(f"detector.layers.smear.kernel.{key}")

    if condition.family == "B" and condition.epsilon_L_percent != 0.0:
        expected_fields = ["detector.layers.smear.kernel.length"]
    elif condition.family == "C" and condition.delta_theta_deg != 0.0:
        expected_fields = ["detector.layers.smear.kernel.theta_deg"]
    else:
        expected_fields = []

    truth_layers = [row["name"] for row in detector_layer_stack(truth_system)]
    inference_layers = [row["name"] for row in detector_layer_stack(inference_system)]
    base_configs_match = _config_without_smear(truth_system) == _config_without_smear(
        inference_system
    )
    audit = {
        "preset": SYSTEM_PRESET,
        "jitter_removed_truth": JITTER_LAYER_NAME not in truth_layers,
        "jitter_removed_inference": JITTER_LAYER_NAME not in inference_layers,
        "truth_smear_present": truth_smear is not None,
        "inference_smear_present": inference_smear is not None,
        "base_configs_match_after_removing_smear": base_configs_match,
        "mismatch_fields": mismatch_fields,
        "expected_mismatch_fields": expected_fields,
        "matches_expected_mismatch": mismatch_fields == expected_fields,
    }
    if not (
        audit["jitter_removed_truth"]
        and audit["jitter_removed_inference"]
        and audit["base_configs_match_after_removing_smear"]
        and audit["matches_expected_mismatch"]
    ):
        raise ValueError(f"Unexpected truth/inference config mismatch: {audit}")
    return audit


def paired_initialization_seed_provenance(
    *,
    experiment_seed: int = PRESCRIBED_MC_SEED,
    run_index: int = 1,
) -> dict[str, Any]:
    """Return the deterministic one-row prescribed-MC init seed provenance."""

    base_key = jax.random.PRNGKey(int(experiment_seed))
    run_key = jax.random.fold_in(base_key, int(run_index))
    run_seed = int(np.asarray(run_key)[0])
    rng_key = jax.random.PRNGKey(run_seed)
    _, init_key = jax.random.split(rng_key)
    return {
        "experiment_seed": int(experiment_seed),
        "run_index": int(run_index),
        "run_seed": run_seed,
        "init_key": [int(value) for value in np.asarray(init_key)],
        "policy": (
            "Each condition prescription has one enabled row at enabled index 1, so the "
            "common experiment seed produces the same prior-sampled physical init."
        ),
    }


def _refresh_preserving_derived_infer_keys(
    store: ParameterStore,
    *,
    spec: Any,
) -> ParameterStore:
    sampled_derived: dict[str, Any] = {}
    for key in INFER_KEYS:
        if key in spec and spec.get(key).kind == "derived":
            sampled_derived[key] = store.get(key)
    refreshed = store.refresh_derived(spec)
    return refreshed.replace(sampled_derived) if sampled_derived else refreshed


def production_initial_theta_for_condition(condition: SmearCondition) -> np.ndarray:
    """Return the production prior-sampled initial physical theta for audits."""

    inference_system, _ = system_config_for_smear(
        condition.model_kernel,
        context=f"{condition.run_id}.initialization_audit",
    )
    spec = compose_forward_spec(inference_system)
    store = ParameterStore.from_spec_defaults(spec).refresh_derived(spec)
    init_key = jnp.asarray(paired_initialization_seed_provenance()["init_key"], dtype=jnp.uint32)
    prior_spec = PriorSpec.from_info(store, PRODUCTION_PRIORS)
    init_store = prior_spec.sample_near(
        center_store=store,
        rng_key=init_key,
        keys=INFER_KEYS,
    )
    init_store = _refresh_preserving_derived_infer_keys(init_store, spec=spec)
    return np.asarray(pack_params(spec.subset(INFER_KEYS), init_store), dtype=float)


def build_condition(
    *,
    family: str,
    L_truth_pix: float,
    orientation: str,
    epsilon_L_percent: float | None = None,
    delta_theta_deg: float = 0.0,
    binary_pa_deg: float | None = None,
) -> SmearCondition:
    """Resolve one family condition into kernels and deterministic run naming."""

    if orientation not in ORIENTATIONS:
        raise ValueError(f"Unsupported orientation {orientation!r}.")
    if binary_pa_deg is None:
        binary_pa_deg = binary_position_angle_deg()
    phi = 0.0 if orientation == "parallel" else 90.0
    theta_truth = float(binary_pa_deg + phi)
    if family == "A":
        L_model = float(L_truth_pix)
        eps = 0.0 if epsilon_L_percent is None else float(epsilon_L_percent)
        delta = 0.0
        prefix = "matched"
        suffix = ""
    elif family == "B":
        if epsilon_L_percent is None:
            raise ValueError("Family B requires epsilon_L_percent.")
        eps = float(epsilon_L_percent)
        L_model = float(L_truth_pix) * (1.0 + eps / 100.0)
        delta = 0.0
        prefix = "ampke"
        suffix = f"_{_signed_token(eps, suffix='pct')}"
    elif family == "C":
        eps = 0.0 if epsilon_L_percent is None else float(epsilon_L_percent)
        L_model = float(L_truth_pix)
        delta = float(delta_theta_deg)
        prefix = "dirke"
        suffix = f"_{_signed_token(delta, suffix='deg')}"
    else:
        raise ValueError(f"Unsupported family {family!r}.")
    theta_model = float(theta_truth + delta)
    run_id = f"{prefix}_L{_length_token(L_truth_pix)}_{orientation}{suffix}"
    truth_kernel = _kernel_from_requested_length(
        requested_length_pix=float(L_truth_pix),
        theta_deg=theta_truth,
    )
    model_kernel = _kernel_from_requested_length(
        requested_length_pix=float(L_model),
        theta_deg=theta_model,
    )
    return SmearCondition(
        family=family,
        run_id=run_id,
        L_truth_pix=float(L_truth_pix),
        L_model_pix=float(L_model),
        epsilon_L_percent=eps,
        orientation=orientation,
        phi_truth_deg=float(phi),
        theta_truth_deg=theta_truth,
        delta_theta_deg=float(delta),
        theta_model_deg=theta_model,
        truth_kernel=truth_kernel,
        model_kernel=model_kernel,
    )


def campaign_conditions(
    families: Sequence[str] = ("A", "B", "C"),
) -> list[SmearCondition]:
    """Return deterministic campaign A/B/C condition rows."""

    binary_pa = binary_position_angle_deg()
    rows: list[SmearCondition] = []
    if "A" in families:
        for length in FAMILY_A_LENGTHS:
            for orientation in ORIENTATIONS:
                rows.append(
                    build_condition(
                        family="A",
                        L_truth_pix=length,
                        orientation=orientation,
                        binary_pa_deg=binary_pa,
                    )
                )
    if "B" in families:
        for length in FAMILY_BC_LENGTHS:
            for orientation in ORIENTATIONS:
                for eps in MISMATCH_GRID:
                    rows.append(
                        build_condition(
                            family="B",
                            L_truth_pix=length,
                            orientation=orientation,
                            epsilon_L_percent=eps,
                            binary_pa_deg=binary_pa,
                        )
                    )
    if "C" in families:
        for length in FAMILY_BC_LENGTHS:
            for orientation in ORIENTATIONS:
                for delta in MISMATCH_GRID:
                    rows.append(
                        build_condition(
                            family="C",
                            L_truth_pix=length,
                            orientation=orientation,
                            delta_theta_deg=delta,
                            binary_pa_deg=binary_pa,
                        )
                    )
    return rows


def prescription_for_condition(
    condition: SmearCondition,
    *,
    outdir: str = ".",
    n_iter: int = PRODUCTION_N_ITER,
    base_lr: float = PRODUCTION_BASE_LR,
    plots: bool = True,
) -> dict[str, Any]:
    """Return a prescribed-MC prescription for one deterministic condition."""

    system, _ = system_config_for_smear(condition.truth_kernel, context=f"{condition.run_id}.truth")
    inference_system, _ = system_config_for_smear(
        condition.model_kernel,
        context=f"{condition.run_id}.inference",
    )
    truth_inference_config_audit(
        condition,
        truth_system=system,
        inference_system=inference_system,
    )
    return {
        "system": system,
        "experiment": {
            "kind": "prescribed_mc",
            "notes": (
                "Canonical controlled line-smear sensitivity. "
                "Truth/data and inference systems differ only by the recorded "
                "smear kernel for mismatch families."
            ),
            "seed": PRESCRIBED_MC_SEED,
            "inference_system": inference_system,
            "monte_carlo": {
                "n_runs": 1,
                "run_id_prefix": condition.run_id,
                "results_filename": "results.csv",
                "results_orientation": "row",
                "run_plan": None,
                "reuse_fim": False,
            },
            "optimizer": {
                "kind": PRODUCTION_OPTIMIZER_KIND,
                "kwargs": {},
                "n_iter": int(n_iter),
                "base_lr": float(base_lr),
                "loss": "nll",
                "schedule": copy.deepcopy(PRODUCTION_SCHEDULE),
                "early_stopping": copy.deepcopy(PRODUCTION_EARLY_STOPPING),
            },
            "eigenmodes": {
                "enable": True,
                "whiten": True,
                "truncate_k": None,
                "truncate_by_eigval": None,
            },
            "infer_keys": list(INFER_KEYS),
            "noise": {
                "enabled": False,
                "photon_noise": False,
                "read_noise": False,
                "dark_current": False,
                "variance_floor": 1.0,
            },
            "outputs": {
                "outdir": outdir,
                "plots": bool(plots),
                "save_results": True,
            },
            "init": {"sampling": "prior"},
            "priors": copy.deepcopy(PRODUCTION_PRIORS),
            "diagnostics": {
                "objective": "nll",
                "reference_point": "physical_truth_parameter_vector",
                "noise": "disabled; variance=max(model_image, variance_floor)",
                "variance_floor": 1.0,
                "initialization_seed": paired_initialization_seed_provenance(),
            },
        },
    }


def condition_manifest(condition: SmearCondition) -> dict[str, Any]:
    """Return JSON provenance for one condition."""

    payload = dataclasses.asdict(condition)
    truth_system, truth_overrides = system_config_for_smear(
        condition.truth_kernel,
        context=f"{condition.run_id}.truth",
    )
    inference_system, inference_overrides = system_config_for_smear(
        condition.model_kernel,
        context=f"{condition.run_id}.inference",
    )
    payload["smear_definition"] = {
        "native_layer_kind": "ApplyConvolution",
        "kernel_kind": "line",
        "length_units": SMEAR_UNITS,
        "length_definition": "total finite line-segment length",
        "angle_convention": "counter-clockwise from detector +X toward +Y",
        "centered": True,
        "normalization": "kernel divided by sum(kernel)",
        "kernel_size_requirement": "positive odd integer",
        "zero_smear_execution_policy": "requested 0.0 removes the named smear detector layer",
        "nonzero_sigma_perp_detector_pix": SMEAR_SIGMA_PERP_PIX,
    }
    payload["detector_provenance"] = {
        "system_preset": SYSTEM_PRESET,
        "source_exposure_time_s": EXPOSURE_TIME_S,
        "jitter_policy": "named jitter detector layer removed from truth and inference",
        "truth_detector_layer_overrides": truth_overrides,
        "inference_detector_layer_overrides": inference_overrides,
        "truth_detector_layers": detector_layer_stack(truth_system),
        "inference_detector_layers": detector_layer_stack(inference_system),
        "truth_inference_config_audit": truth_inference_config_audit(
            condition,
            truth_system=truth_system,
            inference_system=inference_system,
        ),
    }
    payload["objective_provenance"] = {
        "optimizer_loss": "nll",
        "map_prior_penalty": "disabled",
        "optimizer_kind": PRODUCTION_OPTIMIZER_KIND,
        "optimizer_max_iterations": PRODUCTION_N_ITER,
        "optimizer_base_lr": PRODUCTION_BASE_LR,
        "optimizer_schedule": copy.deepcopy(PRODUCTION_SCHEDULE),
        "optimizer_early_stopping": copy.deepcopy(PRODUCTION_EARLY_STOPPING),
        "eigenmodes": {
            "enable": True,
            "whiten": True,
            "truncate_k": None,
            "truncate_by_eigval": None,
        },
        "init": {
            "sampling": "prior",
            "paired_seed_provenance": paired_initialization_seed_provenance(),
        },
        "priors": copy.deepcopy(PRODUCTION_PRIORS),
        "plots_enabled": True,
        "parameter_count_expected": PARAMETER_COUNT,
        "infer_keys": list(INFER_KEYS),
        "fim_theta_semantics": (
            "current fim_theta is a direct jax.hessian(loss_fn)(theta_ref) wrapper, "
            "so optional exact-Hessian diagnostics are redundant unless fim_theta changes"
        ),
    }
    return payload


def resolve_parameter_labels_for_condition(condition: SmearCondition | None = None) -> list[str]:
    """Resolve expanded canonical FIM labels for the campaign configuration."""

    condition = condition or build_condition(
        family="A",
        L_truth_pix=0.5,
        orientation="parallel",
    )
    prescription = prescription_for_condition(condition)
    inference_system = resolve_system_config(prescription["experiment"]["inference_system"])
    spec = compose_forward_spec(inference_system)
    store = ParameterStore.from_spec_defaults(spec).refresh_derived(spec)
    labels = list(generate_fim_labels(INFER_KEYS, cfg=inference_system, store=store))
    if len(labels) != PARAMETER_COUNT:
        raise RuntimeError(
            f"Expected {PARAMETER_COUNT} expanded labels, resolved {len(labels)}: {labels}"
        )
    return labels


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write prescriptions.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False)


def write_plan_csv(path: Path, conditions: Sequence[SmearCondition]) -> None:
    fieldnames = [
        "family",
        "run_id",
        "L_truth_pix",
        "L_model_pix",
        "epsilon_L_percent",
        "truth_orientation_label",
        "phi_truth_deg",
        "theta_truth_deg",
        "delta_theta_deg",
        "theta_model_deg",
        "truth_kernel_length_effective_pix",
        "model_kernel_length_effective_pix",
        "truth_kernel_json",
        "model_kernel_json",
        "condition_dir",
        "prescription_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition in conditions:
            writer.writerow(plan_row(condition))


def plan_row(condition: SmearCondition) -> dict[str, Any]:
    truth_length = None if condition.truth_kernel is None else condition.truth_kernel["length"]
    model_length = None if condition.model_kernel is None else condition.model_kernel["length"]
    return {
        "family": condition.family,
        "run_id": condition.run_id,
        "L_truth_pix": condition.L_truth_pix,
        "L_model_pix": condition.L_model_pix,
        "epsilon_L_percent": condition.epsilon_L_percent,
        "truth_orientation_label": condition.orientation,
        "phi_truth_deg": condition.phi_truth_deg,
        "theta_truth_deg": condition.theta_truth_deg,
        "delta_theta_deg": condition.delta_theta_deg,
        "theta_model_deg": condition.theta_model_deg,
        "truth_kernel_length_effective_pix": truth_length,
        "model_kernel_length_effective_pix": model_length,
        "truth_kernel_json": json.dumps(condition.truth_kernel, sort_keys=True),
        "model_kernel_json": json.dumps(condition.model_kernel, sort_keys=True),
        "condition_dir": f"{condition.family}/{condition.run_id}",
        "prescription_path": f"{condition.family}/{condition.run_id}/prescription.yaml",
    }


def generate_artifacts(
    *,
    output_root: Path,
    families: Sequence[str],
    dry_run: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    """Generate condition prescriptions, plans, and launch metadata."""

    conditions = campaign_conditions(families)
    if smoke:
        selected = {
            "matched_L0p5_parallel",
            "ampke_L0p5_parallel_m5pct",
            "dirke_L0p5_perpendicular_p2deg",
        }
        conditions = [row for row in conditions if row.run_id in selected]
    counts = {family: sum(1 for row in conditions if row.family == family) for family in families}
    manifest = {
        "campaign": "canonical_smear_sensitivity_202608",
        "output_root": str(output_root),
        "dry_run": bool(dry_run),
        "families": list(families),
        "counts": counts,
        "total_conditions": len(conditions),
        "expected_counts_before_zero_dedup": {"A": 14, "B": 44, "C": 44},
        "system_preset": SYSTEM_PRESET,
        "jitter_policy": "remove named jitter detector layer from truth and inference",
        "zero_smear_policy": "retained duplicate orientation controls with the named smear detector layer absent",
        "production_optimizer": {
            "kind": PRODUCTION_OPTIMIZER_KIND,
            "loss": "nll",
            "n_iter": PRODUCTION_N_ITER,
            "base_lr": PRODUCTION_BASE_LR,
            "schedule": copy.deepcopy(PRODUCTION_SCHEDULE),
            "early_stopping": copy.deepcopy(PRODUCTION_EARLY_STOPPING),
        },
        "production_eigenmodes": {
            "enable": True,
            "whiten": True,
            "truncate_k": None,
            "truncate_by_eigval": None,
        },
        "production_init": {
            "sampling": "prior",
            "paired_seed_provenance": paired_initialization_seed_provenance(),
            "priors": copy.deepcopy(PRODUCTION_PRIORS),
        },
        "plots_enabled": True,
        "conditions": [plan_row(row) for row in conditions],
    }
    if dry_run:
        return manifest

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "slurm").mkdir(parents=True, exist_ok=True)
    write_json(output_root / "campaign_manifest.json", manifest)
    write_json(
        output_root / "parameter_labels.json",
        {
            "parameter_count": PARAMETER_COUNT,
            "infer_keys": list(INFER_KEYS),
            "expanded_labels": resolve_parameter_labels_for_condition(conditions[0]),
            "nuisance_keys_for_schur": list(NUISANCE_KEYS),
        },
    )
    write_plan_csv(output_root / "plan_all.csv", conditions)
    for family in ("A", "B", "C"):
        write_plan_csv(
            output_root / f"plan_family_{family}.csv",
            [row for row in conditions if row.family == family],
        )
    for condition in conditions:
        condition_dir = output_root / condition.family / condition.run_id
        prescription = prescription_for_condition(condition)
        write_yaml(condition_dir / "prescription.yaml", prescription)
        write_json(condition_dir / "condition_manifest.json", condition_manifest(condition))
    write_launch_helpers(output_root, array_max=len(conditions) - 1)
    return manifest


def _array_max_from_manifest(manifest: Mapping[str, Any]) -> int:
    return max(0, int(manifest.get("total_conditions", 0)) - 1)


def write_launch_helpers(output_root: Path, *, array_max: int) -> None:
    """Write Slurm/status helper scripts for generated conditions."""

    slurm_dir = output_root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    sbatch = f"""#!/usr/bin/env bash
#SBATCH --job-name=canonical-smear-202608
#SBATCH --output={output_root}/slurm/%A_%a.out
#SBATCH --error={output_root}/slurm/%A_%a.err
#SBATCH --array=0-{max(0, int(array_max))}
#SBATCH --time={SLURM_CONDITION_TIME}
#SBATCH --cpus-per-task={SLURM_CPUS_PER_TASK}
#SBATCH --mem={SLURM_MEMORY}

set -euo pipefail
mkdir -p {output_root}/slurm
cd {REPO_ROOT}
source /cm/shared/apps/miniforge/etc/profile.d/conda.sh
DLUXSHERA_CONDA_ENV_PREFIX="${{DLUXSHERA_CONDA_ENV_PREFIX:-{GATTACA2_CONDA_ENV_PREFIX}}}"
conda activate "$DLUXSHERA_CONDA_ENV_PREFIX"
export PYTHONPATH="{REPO_ROOT}/src${{PYTHONPATH:+:${{PYTHONPATH}}}}"

echo "Conda env: ${{CONDA_DEFAULT_ENV:-unset}}"
echo "CONDA_PREFIX: ${{CONDA_PREFIX:-unset}}"
echo "Python executable: $(which python)"
echo "PYTHONPATH: ${{PYTHONPATH}}"
python - <<'PYENV'
import sys
print("sys.executable:", sys.executable)
import jax
print("jax:", jax.__version__)
import dluxshera
print("dluxshera import ok")
print("dluxshera path:", dluxshera.__file__)
PYENV

python {RECIPE_DIR / 'canonical_smear_campaign.py'} run-index \\
  --campaign-root {output_root} \\
  --index "${{SLURM_ARRAY_TASK_ID}}"
"""
    helper = f"""#!/usr/bin/env bash
set -euo pipefail
cd {REPO_ROOT}
: "${{PYTHON:=python3}}"
"${{PYTHON}}" {RECIPE_DIR / 'canonical_smear_campaign.py'} status --campaign-root {output_root}
"""
    aggregate = f"""#!/usr/bin/env bash
set -euo pipefail
cd {REPO_ROOT}
: "${{PYTHON:=python3}}"
echo "aggregate_and_plot.sh reconstructs models and computes JAX derivatives; use a compute allocation for cluster campaigns."
"${{PYTHON}}" {RECIPE_DIR / 'canonical_smear_campaign.py'} aggregate --campaign-root {output_root}
"${{PYTHON}}" {RECIPE_DIR / 'canonical_smear_campaign.py'} plot --campaign-root {output_root}
"""
    aggregate_sbatch = f"""#!/usr/bin/env bash
#SBATCH --job-name=canonical-smear-agg-202608
#SBATCH --output={output_root}/slurm/aggregate_%j.out
#SBATCH --error={output_root}/slurm/aggregate_%j.err
#SBATCH --time={SLURM_AGGREGATE_TIME}
#SBATCH --cpus-per-task={SLURM_CPUS_PER_TASK}
#SBATCH --mem={SLURM_MEMORY}

set -euo pipefail
mkdir -p {output_root}/slurm
cd {REPO_ROOT}
source /cm/shared/apps/miniforge/etc/profile.d/conda.sh
DLUXSHERA_CONDA_ENV_PREFIX="${{DLUXSHERA_CONDA_ENV_PREFIX:-{GATTACA2_CONDA_ENV_PREFIX}}}"
conda activate "$DLUXSHERA_CONDA_ENV_PREFIX"
export PYTHONPATH="{REPO_ROOT}/src${{PYTHONPATH:+:${{PYTHONPATH}}}}"

echo "Conda env: ${{CONDA_DEFAULT_ENV:-unset}}"
echo "CONDA_PREFIX: ${{CONDA_PREFIX:-unset}}"
echo "Python executable: $(which python)"
echo "PYTHONPATH: ${{PYTHONPATH}}"
python - <<'PYENV'
import sys
print("sys.executable:", sys.executable)
import jax
print("jax:", jax.__version__)
import dluxshera
print("dluxshera import ok")
print("dluxshera path:", dluxshera.__file__)
PYENV

python {RECIPE_DIR / 'canonical_smear_campaign.py'} aggregate --campaign-root {output_root}
python {RECIPE_DIR / 'canonical_smear_campaign.py'} plot --campaign-root {output_root}
"""
    scripts = {
        "submit_array.sbatch": sbatch,
        "status.sh": helper,
        "aggregate_and_plot.sh": aggregate,
        "aggregate.sbatch": aggregate_sbatch,
    }
    for name, text in scripts.items():
        path = output_root / name
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _condition_dir_from_index(campaign_root: Path, index: int) -> Path:
    rows = _read_csv_rows(campaign_root / "plan_all.csv")
    if index < 0 or index >= len(rows):
        raise IndexError(f"Condition index {index} outside plan length {len(rows)}.")
    return campaign_root / rows[index]["condition_dir"]


def run_condition(condition_dir: Path) -> int:
    """Run prescribed-MC for a single condition directory."""

    prescription = condition_dir / "prescription.yaml"
    if not prescription.exists():
        raise FileNotFoundError(prescription)
    cmd = [
        sys.executable,
        str(PRESCRIBED_RUNNER),
        "--prescription",
        str(prescription),
        "--outdir",
        str(condition_dir),
        "--results-orientation",
        "row",
    ]
    return subprocess.call(cmd, cwd=REPO_ROOT)


def _load_condition_manifest(condition_dir: Path) -> dict[str, Any]:
    return json.loads((condition_dir / "condition_manifest.json").read_text(encoding="utf-8"))


def _deterministic_noiseless_variance(image: Any, *, floor: float = 1.0) -> Any:
    return jnp.maximum(image, float(floor))


def _pinv(matrix: np.ndarray, *, rtol: float = NLL_PSEUDOINVERSE_RTOL) -> np.ndarray:
    return np.linalg.pinv(np.asarray(matrix, dtype=float), rcond=float(rtol))


def _solve_linearized(matrix: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    return -_pinv(matrix) @ np.asarray(gradient, dtype=float)


def _key_indices(labels_by_key: Mapping[str, Sequence[str]]) -> dict[str, list[int]]:
    indices: dict[str, list[int]] = {}
    cursor = 0
    for key in INFER_KEYS:
        labels = list(labels_by_key[key])
        indices[key] = list(range(cursor, cursor + len(labels)))
        cursor += len(labels)
    return indices


def _schur_diagnostics(
    F: np.ndarray,
    *,
    labels_by_key: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    key_indices = _key_indices(labels_by_key)
    nuisance = [idx for key in NUISANCE_KEYS for idx in key_indices[key]]
    all_idx = list(range(F.shape[0]))
    slow = [idx for idx in all_idx if idx not in nuisance]
    F_ss = F[np.ix_(slow, slow)]
    F_sn = F[np.ix_(slow, nuisance)]
    F_ns = F[np.ix_(nuisance, slow)]
    F_nn = F[np.ix_(nuisance, nuisance)]
    F_schur = F_ss - F_sn @ _pinv(F_nn) @ F_ns
    sep_slow_idx = slow.index(SEPARATION_INDEX)
    cov_full = _pinv(F)
    cov_schur = _pinv(F_schur)
    conditional_sigma = math.sqrt(max(0.0, 1.0 / F[SEPARATION_INDEX, SEPARATION_INDEX]))
    marginalized_sigma = math.sqrt(max(0.0, cov_full[SEPARATION_INDEX, SEPARATION_INDEX]))
    registration_marginalized_sigma = math.sqrt(max(0.0, cov_schur[sep_slow_idx, sep_slow_idx]))
    return {
        "slow_indices": slow,
        "nuisance_indices": nuisance,
        "slow_labels": [expanded_labels(labels_by_key)[idx] for idx in slow],
        "nuisance_labels": [expanded_labels(labels_by_key)[idx] for idx in nuisance],
        "F_slow_conditional": F_ss,
        "F_registration_marginalized_slow": F_schur,
        "conditional_separation_sigma_as": conditional_sigma,
        "marginalized_separation_sigma_as": marginalized_sigma,
        "registration_marginalized_separation_sigma_as": registration_marginalized_sigma,
        "nuisance_marginalization_penalty": (
            marginalized_sigma / conditional_sigma if conditional_sigma > 0.0 else np.nan
        ),
    }


def expanded_labels(labels_by_key: Mapping[str, Sequence[str]]) -> list[str]:
    return [label for key in INFER_KEYS for label in labels_by_key[key]]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(entry) for key, entry in value.items()}
    if isinstance(value, list):
        return [_jsonable(entry) for entry in value]
    return value


def compute_derivative_diagnostics(
    condition_dir: Path,
    *,
    include_hessian: bool = False,
) -> dict[str, Any]:
    """Compute canonical NLL derivatives at physical truth for one condition."""

    jax.config.update("jax_enable_x64", True)
    prescription_path = condition_dir / "prescription.yaml"
    cfg = resolve_config(
        load_user_config(
            config_path=prescription_path,
            system_preset=None,
            experiment_preset=None,
        )
    )
    system_cfg = cfg["system"]
    experiment_cfg = cfg["experiment"]
    inference_system_cfg = resolve_system_config(experiment_cfg["inference_system"])

    forward_spec_data = compose_forward_spec(system_cfg)
    forward_spec_infer = compose_forward_spec(inference_system_cfg)
    truth_store_data = ParameterStore.from_spec_defaults(forward_spec_data)
    truth_store_data = truth_store_data.refresh_derived(forward_spec_data)
    truth_store_infer = ParameterStore.from_spec_defaults(forward_spec_infer)
    aligned_truth = {}
    for key in forward_spec_infer.keys():
        try:
            aligned_truth[key] = truth_store_data.get(key)
        except KeyError:
            continue
    truth_store_infer = truth_store_infer.replace(aligned_truth)
    truth_store_infer = truth_store_infer.refresh_derived(forward_spec_infer)

    binder_data = SheraBinder(system_cfg, forward_spec_data, truth_store_data)
    binder_infer = SheraBinder(inference_system_cfg, forward_spec_infer, truth_store_infer)
    data_psf = binder_data.model()
    data, data_var = apply_observation_noise(
        data_psf,
        noise_cfg=experiment_cfg.get("noise", {}),
        rng_key=jax.random.PRNGKey(int(experiment_cfg.get("seed", 0))),
        detector_spec=getattr(binder_data.detector, "spec", None),
        exposure_time_s=truth_store_data.get("source.exposure_time_s", default=None),
    )
    data_var = _deterministic_noiseless_variance(data_psf, floor=1.0)

    inference_subspec = forward_spec_infer.subset(INFER_KEYS)
    nll_loss_fn, _ = make_binder_nll_fn(
        binder=binder_infer,
        infer_keys=INFER_KEYS,
        data=data,
        var=data_var,
        noise_model="gaussian",
        reduce="sum",
        theta0_store=truth_store_infer,
    )
    theta_true = np.asarray(pack_params(inference_subspec, truth_store_infer), dtype=float)
    labels = generate_fim_labels(INFER_KEYS, cfg=inference_system_cfg, store=truth_store_infer)
    labels_by_key = {
        key: list(labels[start:stop])
        for key, (start, stop) in _key_start_stop(labels).items()
    }
    flat_labels = expanded_labels(labels_by_key)
    if theta_true.size != PARAMETER_COUNT:
        raise RuntimeError(
            f"Expected {PARAMETER_COUNT} parameters, resolved {theta_true.size}: {flat_labels}"
        )

    loss_truth = float(nll_loss_fn(theta_true))
    gradient = np.asarray(jax.grad(nll_loss_fn)(theta_true), dtype=float)
    F = np.asarray(fim_theta(nll_loss_fn, theta_true), dtype=float)
    schur = _schur_diagnostics(F, labels_by_key=labels_by_key)
    delta_F = _solve_linearized(F, gradient)

    H = None
    delta_H = None
    hessian_diag: dict[str, Any] = {"computed": False}
    if include_hessian:
        H = np.asarray(jax.hessian(nll_loss_fn)(theta_true), dtype=float)
        delta_H = _solve_linearized(H, gradient)
        hessian_diag = {
            "computed": True,
            "eigenvalues": np.linalg.eigvalsh(H),
            "condition": _condition_number(H),
            "relative_F_difference_norm": _relative_norm(H - F, F),
        }

    final_sep_error_uas = np.nan
    final_gradient_norm = np.nan
    final_nll = np.nan
    initial_nll = np.nan
    status = "not_run"
    summary_path = next((condition_dir / "runs").glob("*/summary.json"), None) if (condition_dir / "runs").exists() else None
    if summary_path is not None:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        status = str(summary.get("status", "unknown"))
        initial_nll = float(summary.get("loss_init", np.nan))
        final_nll = float(summary.get("loss_final", np.nan))
        sep_entry = (summary.get("param_summary") or {}).get("source.separation_as") or {}
        final_delta = sep_entry.get("final_delta")
        if final_delta is not None:
            final_sep_error_uas = float(final_delta) * 1.0e6
        trace_path = summary_path.with_name("trace.npz")
        if trace_path.exists():
            trace = np.load(trace_path, allow_pickle=True)
            if "grad_norm" in trace and trace["grad_norm"].size:
                final_gradient_norm = float(np.asarray(trace["grad_norm"])[-1])

    condition = _load_condition_manifest(condition_dir)
    sidecar = {
        "schema_version": "canonical_smear_derivatives.v1",
        "reference_point": "physical_truth_parameter_vector_evaluated_with_inference_system",
        "objective": "nll",
        "parameter_labels": flat_labels,
        "labels_by_key": labels_by_key,
        "theta_true": theta_true,
        "loss_truth": loss_truth,
        "gradient": gradient,
        "gradient_norm": float(np.linalg.norm(gradient)),
        "gradient_separation_component": float(gradient[SEPARATION_INDEX]),
        "F": F,
        "F_eigenvalues": np.linalg.eigvalsh(F),
        "F_condition": _condition_number(F),
        "pinv_policy": {
            "method": "numpy.linalg.pinv",
            "rcond": NLL_PSEUDOINVERSE_RTOL,
        },
        "delta_theta_F": delta_F,
        "delta_theta_F_separation_uas": float(delta_F[SEPARATION_INDEX] * 1.0e6),
        "H": H,
        "delta_theta_H": delta_H,
        "delta_theta_H_separation_uas": (
            None if delta_H is None else float(delta_H[SEPARATION_INDEX] * 1.0e6)
        ),
        "hessian": hessian_diag,
        "schur": schur,
        "condition": condition,
        "optimizer_result": {
            "status": status,
            "initial_nll": initial_nll,
            "final_nll": final_nll,
            "final_gradient_norm": final_gradient_norm,
            "final_separation_error_uas": final_sep_error_uas,
        },
        "residual_diagnostics": _residual_diagnostics(
            data=data,
            model=binder_infer.model(),
            variance=data_var,
        ),
    }
    np.savez_compressed(
        condition_dir / "derivative_diagnostics.npz",
        F=F,
        gradient=gradient,
        theta_true=theta_true,
        labels=np.asarray(flat_labels),
        delta_theta_F=delta_F,
        H=np.asarray([]) if H is None else H,
        delta_theta_H=np.asarray([]) if delta_H is None else delta_H,
        F_registration_marginalized_slow=schur["F_registration_marginalized_slow"],
        F_slow_conditional=schur["F_slow_conditional"],
    )
    write_json(condition_dir / "derivative_diagnostics.json", _jsonable(sidecar))
    return _summary_row_from_diagnostics(condition, sidecar)


def _key_start_stop(labels: Sequence[str]) -> dict[str, tuple[int, int]]:
    starts: dict[str, tuple[int, int]] = {}
    cursor = 0
    for key in INFER_KEYS:
        if key in {
            "optics.primary.zernike_coeffs_nm",
            "optics.secondary.zernike_coeffs_nm",
        }:
            length = 8
        else:
            length = 1
        starts[key] = (cursor, cursor + length)
        cursor += length
    if cursor != len(labels):
        raise RuntimeError(f"Expected {cursor} labels from INFER_KEYS, got {len(labels)}.")
    return starts


def _condition_number(matrix: np.ndarray) -> float:
    eigvals = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))
    absvals = np.abs(eigvals)
    nonzero = absvals[absvals > 0.0]
    if nonzero.size == 0:
        return float("inf")
    return float(np.max(absvals) / np.min(nonzero))


def _relative_norm(numerator: np.ndarray, denominator: np.ndarray) -> float:
    denom = float(np.linalg.norm(denominator))
    if denom == 0.0:
        return float(np.linalg.norm(numerator))
    return float(np.linalg.norm(numerator) / denom)


def _residual_diagnostics(*, data: Any, model: Any, variance: Any) -> dict[str, Any]:
    residual = np.asarray(model - data, dtype=float)
    return {
        "rms": float(np.sqrt(np.mean(np.square(residual)))),
        "max_abs": float(np.max(np.abs(residual))),
        "reduced_chi2": reduced_chi2_between_images(data, model, variance_image=variance),
    }


def _summary_row_from_diagnostics(condition: Mapping[str, Any], sidecar: Mapping[str, Any]) -> dict[str, Any]:
    opt = sidecar["optimizer_result"]
    schur = sidecar["schur"]
    return {
        "family": condition["family"],
        "run_id": condition["run_id"],
        "L_truth_pix": condition["L_truth_pix"],
        "L_model_pix": condition["L_model_pix"],
        "epsilon_L_percent": condition["epsilon_L_percent"],
        "truth_orientation_label": condition["orientation"],
        "phi_truth_deg": condition["phi_truth_deg"],
        "theta_truth_deg": condition["theta_truth_deg"],
        "delta_theta_deg": condition["delta_theta_deg"],
        "theta_model_deg": condition["theta_model_deg"],
        "final_separation_error_uas": opt["final_separation_error_uas"],
        "linearized_F_predicted_separation_bias_uas": sidecar["delta_theta_F_separation_uas"],
        "linearized_H_predicted_separation_bias_uas": sidecar["delta_theta_H_separation_uas"],
        "marginalized_Fisher_separation_sigma_as": schur["marginalized_separation_sigma_as"],
        "conditional_Fisher_separation_sigma_as": schur["conditional_separation_sigma_as"],
        "nuisance_marginalization_penalty": schur["nuisance_marginalization_penalty"],
        "initial_nll": opt["initial_nll"],
        "final_nll": opt["final_nll"],
        "reference_nll": sidecar["loss_truth"],
        "gradient_norm_at_truth": sidecar["gradient_norm"],
        "final_gradient_norm": opt["final_gradient_norm"],
        "convergence_status": opt["status"],
        "derivative_sidecar": f"{condition['family']}/{condition['run_id']}/derivative_diagnostics.json",
    }


def aggregate_campaign(
    *,
    campaign_root: Path,
    include_hessian: bool = False,
) -> list[dict[str, Any]]:
    """Aggregate condition manifests, prescribed outputs, and derivative sidecars."""

    rows: list[dict[str, Any]] = []
    for plan in _read_csv_rows(campaign_root / "plan_all.csv"):
        condition_dir = campaign_root / plan["condition_dir"]
        rows.append(
            compute_derivative_diagnostics(
                condition_dir,
                include_hessian=include_hessian,
            )
        )
    out_csv = campaign_root / "summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def status_rows(campaign_root: Path) -> list[dict[str, Any]]:
    """Return lightweight status rows for generated condition directories."""

    rows = []
    for plan in _read_csv_rows(campaign_root / "plan_all.csv"):
        condition_dir = campaign_root / plan["condition_dir"]
        run_summaries = list((condition_dir / "runs").glob("*/summary.json")) if (condition_dir / "runs").exists() else []
        rows.append(
            {
                "family": plan["family"],
                "run_id": plan["run_id"],
                "prescription": (condition_dir / "prescription.yaml").exists(),
                "run_summary": bool(run_summaries),
                "derivative_json": (condition_dir / "derivative_diagnostics.json").exists(),
            }
        )
    return rows


def plot_campaign(campaign_root: Path) -> None:
    """Write lightweight review plots from ``summary.csv``."""

    import matplotlib.pyplot as plt

    rows = _read_csv_rows(campaign_root / "summary.csv")
    if not rows:
        return
    plot_dir = campaign_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def f(row: Mapping[str, str], key: str) -> float:
        value = row.get(key, "")
        return float(value) if value not in {"", "None", "nan"} else float("nan")

    family_a = [row for row in rows if row["family"] == "A"]
    if family_a:
        fig, ax = plt.subplots(figsize=(6, 4))
        for orientation in ORIENTATIONS:
            sub = sorted(
                [row for row in family_a if row["truth_orientation_label"] == orientation],
                key=lambda row: f(row, "L_truth_pix"),
            )
            ax.plot(
                [f(row, "L_truth_pix") for row in sub],
                [f(row, "marginalized_Fisher_separation_sigma_as") * 1.0e6 for row in sub],
                marker="o",
                label=orientation,
            )
        ax.set_xlabel("Truth smear length [detector pix]")
        ax.set_ylabel("Marginalized separation Fisher sigma [uas]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "family_A_sigma_vs_smear.png", dpi=160)
        plt.close(fig)

    for family, xkey, xlabel, output in [
        ("B", "epsilon_L_percent", "Smear-length error [%]", "family_B_bias_vs_length_error.png"),
        ("C", "delta_theta_deg", "Smear-direction error [deg]", "family_C_bias_vs_direction_error.png"),
    ]:
        sub_family = [row for row in rows if row["family"] == family]
        if not sub_family:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
        axes_flat = axes.flatten()
        panels = [
            (0.5, "parallel"),
            (0.5, "perpendicular"),
            (1.0, "parallel"),
            (1.0, "perpendicular"),
        ]
        for ax, (length, orientation) in zip(axes_flat, panels):
            panel = sorted(
                [
                    row
                    for row in sub_family
                    if abs(f(row, "L_truth_pix") - length) < 1.0e-12
                    and row["truth_orientation_label"] == orientation
                ],
                key=lambda row: f(row, xkey),
            )
            ax.plot(
                [f(row, xkey) for row in panel],
                [f(row, "final_separation_error_uas") for row in panel],
                marker="o",
                label="optimized",
            )
            ax.plot(
                [f(row, xkey) for row in panel],
                [f(row, "linearized_F_predicted_separation_bias_uas") for row in panel],
                marker="s",
                label="-F^-1 g",
            )
            h_values = [f(row, "linearized_H_predicted_separation_bias_uas") for row in panel]
            if np.isfinite(h_values).any():
                ax.plot([f(row, xkey) for row in panel], h_values, marker="^", label="-H^-1 g")
            ax.set_title(f"L={length:g} pix, {orientation}")
            ax.grid(True, alpha=0.3)
        for ax in axes[-1, :]:
            ax.set_xlabel(xlabel)
        for ax in axes[:, 0]:
            ax.set_ylabel("Separation bias [uas]")
        axes_flat[0].legend()
        fig.tight_layout()
        fig.savefig(plot_dir / output, dpi=160)
        plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--campaign-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p_gen.add_argument("--families", nargs="+", default=["A", "B", "C"])
    p_gen.add_argument("--dry-run", action="store_true")
    p_gen.add_argument("--smoke", action="store_true")

    p_run_index = sub.add_parser("run-index")
    p_run_index.add_argument("--campaign-root", type=Path, required=True)
    p_run_index.add_argument("--index", type=int, required=True)

    p_run = sub.add_parser("run-condition")
    p_run.add_argument("--condition-dir", type=Path, required=True)

    p_agg = sub.add_parser("aggregate")
    p_agg.add_argument("--campaign-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p_agg.add_argument("--hessian", action="store_true")

    p_status = sub.add_parser("status")
    p_status.add_argument("--campaign-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    p_plot = sub.add_parser("plot")
    p_plot.add_argument("--campaign-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "generate":
        manifest = generate_artifacts(
            output_root=args.campaign_root,
            families=tuple(args.families),
            dry_run=bool(args.dry_run),
            smoke=bool(args.smoke),
        )
        print(json.dumps(manifest, indent=2))
        if not args.dry_run:
            array_max = _array_max_from_manifest(manifest)
            print(
                "Generated campaign artifacts. The Slurm helper uses array "
                f"range 0-{array_max}; review smoke outputs before production submission."
            )
        return 0
    if args.command == "run-index":
        return int(run_condition(_condition_dir_from_index(args.campaign_root, args.index)))
    if args.command == "run-condition":
        return int(run_condition(args.condition_dir))
    if args.command == "aggregate":
        rows = aggregate_campaign(
            campaign_root=args.campaign_root,
            include_hessian=bool(args.hessian),
        )
        print(f"Wrote {len(rows)} summary rows to {args.campaign_root / 'summary.csv'}")
        return 0
    if args.command == "status":
        rows = status_rows(args.campaign_root)
        for row in rows:
            print(
                "{family:1s} {run_id:36s} prescription={prescription} "
                "run_summary={run_summary} derivative={derivative_json}".format(**row)
            )
        return 0
    if args.command == "plot":
        plot_campaign(args.campaign_root)
        print(f"Wrote plots under {args.campaign_root / 'plots'}")
        return 0
    raise ValueError(f"Unsupported command {args.command!r}.")


if __name__ == "__main__":
    raise SystemExit(main())
