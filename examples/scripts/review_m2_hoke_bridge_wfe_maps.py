"""Review the imported M2 HO-WFE bridge maps.

This helper treats the HPC import bundle as immutable source data. It verifies
the saved NPY/FITS arrays, recomputes pupil statistics and hashes, and writes a
small review product showing the secondary high-order truth, additive
knowledge-error residual, and inferred/reference map.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import jax
import jax.random as jr
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from dluxshera.config.io import load_config_file
from dluxshera.config.resolver import resolve_config, resolve_system_config
from dluxshera.inference.prior import PriorSpec
from dluxshera.params.store import ParameterStore
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.high_order_wfe import fit_zernike_coefficients_nm


DEFAULT_BUNDLE_ROOT = Path(
    "Results/hpc_imports/m2_hoke_canonical_bridge_0p5nm_center"
)
DEFAULT_OUTDIR = Path("Results/prescribed_mc_m2_hoke_bridge_20260810/wfe_review")
DEFAULT_PRESCRIPTION = Path(
    "examples/recipes/prescribed_mc_m2_hoke_bridge_20260810/prescription.yaml"
)
LOW_ORDER_NOLL = [4, 5, 6, 7, 8, 9, 10, 11]
CASE_NAME = "m2_hoke_0p5nm_xp0p0_yp0p0_w10x30_draw_000"


def _array_hash(array: np.ndarray, *, mask: np.ndarray) -> str:
    arr = np.asarray(array, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if arr.shape != valid.shape:
        raise ValueError(f"Map/hash mask shape mismatch: {arr.shape} vs {valid.shape}.")
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(np.ascontiguousarray(arr[valid]).tobytes())
    return digest.hexdigest()


def _normalised_map_hash(array: np.ndarray, *, mask: np.ndarray) -> str | None:
    arr = np.asarray(array, dtype=float)
    valid = np.asarray(mask, dtype=bool)
    rms = _pupil_rms(arr, valid)
    if rms == 0.0:
        return None
    normalised = np.round(arr / rms, decimals=9)
    return _array_hash(normalised, mask=valid)


def _pupil_rms(array: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        raise ValueError("Pupil mask must contain at least one true pixel.")
    vals = np.asarray(array, dtype=float)[valid]
    return float(np.sqrt(np.mean(np.square(vals))))


def _map_stats(label: str, array: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(mask, dtype=bool)
    vals = np.asarray(array, dtype=float)[valid]
    return {
        "label": label,
        "shape": list(np.asarray(array).shape),
        "pupil_pixels": int(np.count_nonzero(valid)),
        "pupil_mean_nm": float(np.mean(vals)),
        "pupil_rms_nm": _pupil_rms(array, valid),
        "pupil_min_nm": float(np.min(vals)),
        "pupil_max_nm": float(np.max(vals)),
        "map_hash": _array_hash(array, mask=valid),
        "normalised_map_hash": _normalised_map_hash(array, mask=valid),
    }


def _load_fits(path: Path) -> np.ndarray:
    return np.asarray(fits.getdata(path), dtype=float)


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(fits.getdata(path), dtype=bool)


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return os.path.relpath(resolved, repo_root.resolve())


def _read_low_order_rows(path: Path, mirror: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["mirror"] == mirror]


def _low_order_vector(payload: dict[str, Any], mirror: str) -> list[float]:
    coeffs = payload[mirror]["low_order_truth_coefficients_nm"]
    return [float(coeffs[f"Z{noll}"]) for noll in LOW_ORDER_NOLL]


def _load_prescribed_mc_module(repo_root: Path):
    module_path = repo_root / "examples" / "recipes" / "prescribed_monte_carlo.py"
    spec = importlib.util.spec_from_file_location("prescribed_monte_carlo_bridge", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load prescribed_monte_carlo.py from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_float(value: Any) -> float:
    return float(np.asarray(value))


def _store_scalar_entries(store: ParameterStore, infer_keys: list[str]) -> dict[str, float]:
    entries: dict[str, float] = {}
    for key in infer_keys:
        value = store.get(key)
        arr = np.asarray(value)
        if arr.ndim == 0:
            entries[key] = _as_float(arr)
            continue
        flat = arr.reshape(-1)
        for index, item in enumerate(flat):
            entries[f"{key}[{index}]"] = _as_float(item)
    return entries


def _campaign_draw_rows(campaign_plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = campaign_plan["prior_draw_rows_by_case"][CASE_NAME]
    return {str(row["theta_label"]): row for row in rows}


def _low_order_label(mirror: str, index: int) -> str:
    return f"optics.{mirror}.zernike_coeffs_nm[{index}]"


def _campaign_provenance_for_label(
    label: str,
    *,
    draw_rows: dict[str, dict[str, Any]],
    howfe_provenance: dict[str, Any],
) -> dict[str, Any]:
    row = draw_rows.get(label)
    payload: dict[str, Any] = {}
    if row is not None:
        for field in (
            "truth_value",
            "reference_value",
            "prior_mean",
            "theta_reference_offset",
            "prior_sigma",
            "sigma_kind",
            "sigma_config_value",
            "unit",
            "draw_seed",
            "draw_z",
        ):
            payload[f"draw_table_{field}"] = row.get(field)

    for mirror in ("primary", "secondary"):
        prefix = f"optics.{mirror}.zernike_coeffs_nm["
        if label.startswith(prefix) and label.endswith("]"):
            index = int(label[len(prefix):-1])
            noll = LOW_ORDER_NOLL[index]
            coeffs = howfe_provenance[mirror]["low_order_truth_coefficients_nm"]
            payload["physical_truth_source"] = (
                f"campaign_plan.high_order_wfe.provenance.{mirror}."
                "low_order_truth_coefficients_nm"
            )
            payload["campaign_high_order_low_order_truth_value"] = float(
                coeffs[f"Z{noll}"]
            )
            payload["noll_index"] = noll
            if row is not None:
                payload["reference_value_from_physical_truth_plus_offset"] = (
                    float(coeffs[f"Z{noll}"]) + float(row["theta_reference_offset"])
                )
            break
    return payload


def build_state_audit(
    *,
    bundle_root: Path,
    prescription_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Resolve bridge truth/init stores and attach campaign draw provenance."""
    jax.config.update("jax_enable_x64", True)
    pmc = _load_prescribed_mc_module(repo_root)
    campaign_plan = json.loads((bundle_root / "campaign_plan.json").read_text())
    draw_rows = _campaign_draw_rows(campaign_plan)
    howfe_provenance = campaign_plan["high_order_wfe"]["provenance"]

    prescription = load_config_file(prescription_path)
    user_cfg = pmc._strip_private_keys(prescription)
    resolved_cfg = resolve_config(user_cfg)
    system_cfg = resolved_cfg["system"]
    experiment_cfg = resolved_cfg["experiment"]
    mc_cfg_raw = pmc._get_prescribed_mc_cfg(experiment_cfg)
    mc_cfg, mc_defaults = pmc._mc_defaults_from_experiment(experiment_cfg, mc_cfg_raw)
    plan_rows, _ = pmc._apply_experiment_n_runs([], mc_cfg.get("n_runs"))
    run_spec = pmc._resolve_run_spec_with_id(
        mc_cfg,
        plan_rows[0],
        index=1,
        run_id_index=1,
    )

    inference_system_cfg = experiment_cfg.get("inference_system")
    if inference_system_cfg is not None:
        inference_system_cfg = resolve_system_config(inference_system_cfg)
    else:
        inference_system_cfg = system_cfg

    forward_spec_data = compose_forward_spec(system_cfg)
    forward_spec_infer = compose_forward_spec(inference_system_cfg)
    infer_keys = [pmc._migrate_param_key(key) for key in experiment_cfg["infer_keys"]]

    truth_defaults = mc_defaults.get("truth", {}) or {}
    base_store_data = ParameterStore.from_spec_defaults(forward_spec_data)
    if truth_defaults:
        base_store_data = base_store_data.replace(
            pmc._flatten_store_overrides(truth_defaults)
        )
    base_store_data = base_store_data.refresh_derived(forward_spec_data)

    base_store_infer = ParameterStore.from_spec_defaults(forward_spec_infer)
    if truth_defaults:
        base_store_infer = base_store_infer.replace(
            pmc._flatten_store_overrides(truth_defaults)
        )
    base_store_infer = base_store_infer.refresh_derived(forward_spec_infer)

    truth_overrides = pmc._flatten_store_overrides(run_spec.get("truth", {}))
    truth_store_data = base_store_data.replace(truth_overrides)
    truth_store_data = truth_store_data.refresh_derived(forward_spec_data)
    truth_store_infer = base_store_infer.replace(truth_overrides)
    aligned_truth = {}
    for key in forward_spec_infer.keys():
        try:
            aligned_truth[key] = truth_store_data.get(key)
        except KeyError:
            continue
    truth_store_infer = truth_store_infer.replace(aligned_truth)
    truth_store_infer = truth_store_infer.refresh_derived(forward_spec_infer)

    prior_spec = PriorSpec.from_info(
        base_store_infer,
        pmc._migrate_key_mapping(experiment_cfg.get("priors", {})),
    )
    seed = int(run_spec["seed"])
    rng_key = jr.PRNGKey(seed)
    _, init_key = jr.split(rng_key)
    init_cfg = run_spec.get("init", {})
    init_mode = init_cfg.get("mode") or pmc._get_nested(mc_defaults, ["init", "mode"])
    init_overrides = {key: value for key, value in init_cfg.items() if key != "mode"}
    init_overrides_flat = pmc._flatten_store_overrides(init_overrides)
    if init_mode != "prior":
        raise ValueError(f"Bridge state audit expects init.mode='prior', got {init_mode!r}.")
    init_store = prior_spec.sample_near(
        center_store=truth_store_infer,
        rng_key=init_key,
        keys=infer_keys,
    )
    if init_overrides_flat:
        init_store = init_store.replace(init_overrides_flat)
    init_store = pmc._refresh_preserving_derived_infer_keys(
        init_store,
        infer_keys=tuple(infer_keys),
        spec=forward_spec_infer,
    )

    truth_entries = _store_scalar_entries(truth_store_data, infer_keys)
    init_entries = _store_scalar_entries(init_store, infer_keys)
    parameters = []
    for label in truth_entries:
        truth_value = truth_entries[label]
        init_value = init_entries[label]
        parameters.append(
            {
                "label": label,
                "truth": truth_value,
                "init": init_value,
                "init_minus_truth": init_value - truth_value,
                "campaign_provenance": _campaign_provenance_for_label(
                    label,
                    draw_rows=draw_rows,
                    howfe_provenance=howfe_provenance,
                ),
            }
        )

    primary_summary = _low_order_vector(
        json.loads(
            (bundle_root / "model_split" / "high_order_wfe" / "high_order_wfe_summary.json")
            .read_text()
        ),
        "primary",
    )
    secondary_summary = _low_order_vector(
        json.loads(
            (bundle_root / "model_split" / "high_order_wfe" / "high_order_wfe_summary.json")
            .read_text()
        ),
        "secondary",
    )
    primary_plan = _low_order_vector(howfe_provenance, "primary")
    secondary_plan = _low_order_vector(howfe_provenance, "secondary")
    primary_draw_truth = [
        float(draw_rows[_low_order_label("primary", index)]["truth_value"])
        for index in range(8)
    ]
    secondary_draw_truth = [
        float(draw_rows[_low_order_label("secondary", index)]["truth_value"])
        for index in range(8)
    ]

    return {
        "schema_version": "m2_hoke_bridge_state_audit.v1",
        "source_bundle": _repo_relative(bundle_root, repo_root=repo_root),
        "prescription": _repo_relative(prescription_path, repo_root=repo_root),
        "case_name": CASE_NAME,
        "run_id": run_spec.get("run_id"),
        "experiment_seed": int(experiment_cfg["seed"]),
        "run_seed": seed,
        "init_seed_policy": "jax.random.split(jr.PRNGKey(run_seed))[1]",
        "init_mode": init_mode,
        "init_semantics": (
            "sample all infer keys from configured priors around physical truth, "
            "then apply explicit slow-state init overrides"
        ),
        "low_order_truth_checks": {
            "campaign_plan_high_order_provenance_equals_summary": {
                "primary": bool(np.array_equal(primary_plan, primary_summary)),
                "secondary": bool(np.array_equal(secondary_plan, secondary_summary)),
            },
            "campaign_plan_prior_draw_table_truth_equals_summary": {
                "primary": bool(np.array_equal(primary_draw_truth, primary_summary)),
                "secondary": bool(np.array_equal(secondary_draw_truth, secondary_summary)),
            },
            "campaign_plan_prior_draw_table_truth_values": {
                "primary": primary_draw_truth,
                "secondary": secondary_draw_truth,
            },
            "high_order_low_order_truth_coefficients_nm": {
                "primary": primary_plan,
                "secondary": secondary_plan,
            },
        },
        "parameters": parameters,
    }


def write_state_audit_csv(audit: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "label",
        "truth",
        "init",
        "init_minus_truth",
        "draw_table_truth_value",
        "draw_table_reference_value",
        "draw_table_theta_reference_offset",
        "draw_table_prior_sigma",
        "campaign_high_order_low_order_truth_value",
        "reference_value_from_physical_truth_plus_offset",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in audit["parameters"]:
            provenance = row.get("campaign_provenance", {})
            writer.writerow(
                {
                    "label": row["label"],
                    "truth": row["truth"],
                    "init": row["init"],
                    "init_minus_truth": row["init_minus_truth"],
                    "draw_table_truth_value": provenance.get("draw_table_truth_value"),
                    "draw_table_reference_value": provenance.get(
                        "draw_table_reference_value"
                    ),
                    "draw_table_theta_reference_offset": provenance.get(
                        "draw_table_theta_reference_offset"
                    ),
                    "draw_table_prior_sigma": provenance.get("draw_table_prior_sigma"),
                    "campaign_high_order_low_order_truth_value": provenance.get(
                        "campaign_high_order_low_order_truth_value"
                    ),
                    "reference_value_from_physical_truth_plus_offset": provenance.get(
                        "reference_value_from_physical_truth_plus_offset"
                    ),
                }
            )


def build_review_summary(bundle_root: Path, *, repo_root: Path) -> dict[str, Any]:
    how_root = bundle_root / "model_split" / "high_order_wfe"
    config_root = how_root / "config_maps"
    maps_root = how_root / "maps"

    primary_truth = np.load(config_root / "primary_high_order_truth_opd_nm.npy")
    secondary_truth = np.load(config_root / "secondary_high_order_truth_opd_nm.npy")
    secondary_error = np.load(config_root / "secondary_high_order_error_opd_nm.npy")
    primary_mask = _load_mask(maps_root / "primary_mask.fits")
    secondary_mask = _load_mask(maps_root / "secondary_mask.fits")

    primary_truth_fits = _load_fits(maps_root / "primary_high_order_truth_opd_nm.fits")
    primary_error_fits = _load_fits(maps_root / "primary_high_order_error_opd_nm.fits")
    primary_knowledge_fits = _load_fits(
        maps_root / "primary_high_order_knowledge_opd_nm.fits"
    )
    secondary_truth_fits = _load_fits(
        maps_root / "secondary_high_order_truth_opd_nm.fits"
    )
    secondary_error_fits = _load_fits(
        maps_root / "secondary_high_order_error_opd_nm.fits"
    )
    secondary_knowledge_fits = _load_fits(
        maps_root / "secondary_high_order_knowledge_opd_nm.fits"
    )

    low_order_truth = _read_low_order_rows(
        maps_root / "low_order_zernike_truth.csv", "secondary"
    )
    low_order_knowledge = _read_low_order_rows(
        maps_root / "low_order_zernike_knowledge.csv", "secondary"
    )
    low_order_errors = _read_low_order_rows(
        maps_root / "low_order_zernike_errors.csv", "secondary"
    )
    secondary_error_projection = fit_zernike_coefficients_nm(
        secondary_error,
        secondary_mask,
        LOW_ORDER_NOLL,
        input_unit="nm",
    )

    summary_path = how_root / "high_order_wfe_summary.json"
    deck_path = maps_root / "high_order_wfe_deck_manifest.json"
    model_split_path = bundle_root / "model_split" / "model_split_summary.json"
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    return {
        "schema_version": "m2_hoke_bridge_wfe_review.v1",
        "source_bundle": _repo_relative(bundle_root, repo_root=repo_root),
        "source_run": (
            "ff_howfe_production_center_cond_m2_hoke_0p5nm_xp0p0_yp0p0_"
            "w10x30_draw_000"
        ),
        "source_files": {
            "high_order_wfe_summary": _repo_relative(summary_path, repo_root=repo_root),
            "high_order_wfe_deck_manifest": _repo_relative(
                deck_path, repo_root=repo_root
            ),
            "model_split_summary": _repo_relative(model_split_path, repo_root=repo_root),
            "primary_truth_npy": _repo_relative(
                config_root / "primary_high_order_truth_opd_nm.npy",
                repo_root=repo_root,
            ),
            "secondary_truth_npy": _repo_relative(
                config_root / "secondary_high_order_truth_opd_nm.npy",
                repo_root=repo_root,
            ),
            "secondary_error_npy": _repo_relative(
                config_root / "secondary_high_order_error_opd_nm.npy",
                repo_root=repo_root,
            ),
        },
        "provenance": {
            "campaign_base_truth_seed": summary_payload["truth_seed"],
            "primary_truth_seed": summary_payload["primary"]["truth_seed"],
            "secondary_truth_seed": summary_payload["secondary"]["truth_seed"],
            "secondary_knowledge_error_seed": summary_payload["secondary"][
                "knowledge_seed"
            ],
            "secondary_full_truth_rms_nm": summary_payload["secondary"][
                "truth_full_rms_nm"
            ],
            "secondary_high_order_truth_rms_nm": summary_payload["secondary"][
                "truth_high_order_rms_nm"
            ],
            "secondary_requested_knowledge_error_rms_nm": summary_payload[
                "secondary"
            ]["requested_knowledge_error_rms_nm"],
        },
        "array_checks": {
            "primary_npy_truth_equals_fits_truth": bool(
                np.array_equal(primary_truth, primary_truth_fits)
            ),
            "secondary_npy_truth_equals_fits_truth": bool(
                np.array_equal(secondary_truth, secondary_truth_fits)
            ),
            "secondary_npy_error_equals_fits_error": bool(
                np.array_equal(secondary_error, secondary_error_fits)
            ),
            "primary_error_is_zero": bool(np.allclose(primary_error_fits, 0.0)),
            "primary_knowledge_equals_truth": bool(
                np.array_equal(primary_knowledge_fits, primary_truth)
            ),
            "secondary_knowledge_equals_truth_plus_error": bool(
                np.array_equal(secondary_knowledge_fits, secondary_truth + secondary_error)
            ),
            "secondary_knowledge_truth_plus_error_max_abs_nm": float(
                np.max(np.abs(secondary_knowledge_fits - (secondary_truth + secondary_error)))
            ),
        },
        "maps": {
            "primary_high_order_truth": _map_stats(
                "primary_high_order_truth", primary_truth, primary_mask
            ),
            "primary_high_order_error": _map_stats(
                "primary_high_order_error", primary_error_fits, primary_mask
            ),
            "secondary_high_order_truth": _map_stats(
                "secondary_high_order_truth", secondary_truth, secondary_mask
            ),
            "secondary_high_order_error": _map_stats(
                "secondary_high_order_error", secondary_error, secondary_mask
            ),
            "secondary_high_order_knowledge": _map_stats(
                "secondary_high_order_knowledge",
                secondary_knowledge_fits,
                secondary_mask,
            ),
        },
        "low_order_checks": {
            "secondary_truth_coefficients_equal_knowledge": all(
                float(t["truth_coeff_nm"]) == float(k["knowledge_coeff_nm"])
                for t, k in zip(low_order_truth, low_order_knowledge)
            ),
            "secondary_low_order_errors_zero": all(
                float(row["error_nm"]) == 0.0 for row in low_order_errors
            ),
            "secondary_error_z4_z11_projection_nm": secondary_error_projection,
            "secondary_error_z4_z11_max_abs_projection_nm": max(
                abs(float(v)) for v in secondary_error_projection.values()
            ),
        },
    }


def write_review_figure(
    *,
    bundle_root: Path,
    out_path: Path,
) -> None:
    how_root = bundle_root / "model_split" / "high_order_wfe"
    config_root = how_root / "config_maps"
    maps_root = how_root / "maps"
    mask = _load_mask(maps_root / "secondary_mask.fits")
    truth = np.load(config_root / "secondary_high_order_truth_opd_nm.npy")
    error = np.load(config_root / "secondary_high_order_error_opd_nm.npy")
    knowledge = truth + error

    panels = [
        ("M2 high-order truth OPD", truth, "RdBu_r"),
        ("M2 high-order KE residual", error, "RdBu_r"),
        ("M2 inference/reference OPD", knowledge, "RdBu_r"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (title, arr, cmap) in zip(axes, panels):
        masked = np.where(mask, arr, np.nan)
        vmax = float(np.nanpercentile(np.abs(masked), 99.5))
        image = ax.imshow(masked, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("nm")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prescription", type=Path, default=DEFAULT_PRESCRIPTION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    summary = build_review_summary(args.bundle, repo_root=repo_root)
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_path = args.outdir / "wfe_review_summary.json"
    figure_path = args.outdir / "m2_hoke_bridge_wfe_maps.png"
    state_audit_path = args.outdir / "bridge_state_audit.json"
    state_audit_csv_path = args.outdir / "bridge_state_audit.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_review_figure(bundle_root=args.bundle, out_path=figure_path)
    if args.prescription.exists() and (args.bundle / "campaign_plan.json").exists():
        state_audit = build_state_audit(
            bundle_root=args.bundle,
            prescription_path=args.prescription,
            repo_root=repo_root,
        )
        state_audit_path.write_text(
            json.dumps(state_audit, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_state_audit_csv(state_audit, state_audit_csv_path)

    m2_truth = summary["maps"]["secondary_high_order_truth"]
    m2_error = summary["maps"]["secondary_high_order_error"]
    print(f"Wrote {summary_path}")
    print(f"Wrote {figure_path}")
    if state_audit_path.exists():
        print(f"Wrote {state_audit_path}")
        print(f"Wrote {state_audit_csv_path}")
    print(
        "M2 truth RMS/hash: "
        f"{m2_truth['pupil_rms_nm']:.15g} nm / {m2_truth['map_hash']}"
    )
    print(
        "M2 KE RMS/hash: "
        f"{m2_error['pupil_rms_nm']:.15g} nm / {m2_error['map_hash']}"
    )


if __name__ == "__main__":
    main()
