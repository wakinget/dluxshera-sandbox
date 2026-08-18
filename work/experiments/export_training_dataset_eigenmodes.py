"""Export Fisher eigenmodes for the ML dataset v3 SHERA system.

The default configuration source is ``work/experiments/ml_dataset_v3_template.yaml``.
The script copies/resolves the template ``system:`` block, builds the nominal
Binder model, computes the full canonical astrometry FIM, and exports several
unwhitened, untruncated physical eigenbases for downstream ML analysis.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dluxshera.config.io import load_config_file
from dluxshera.config.resolver import resolve_config
from dluxshera.inference.optimization import (
    fim_theta,
    generate_fim_labels,
    make_binder_nll_fn,
)
from dluxshera.params.packing import build_index_map, pack_params
from dluxshera.params.store import ParameterStore
from dluxshera.systems import SheraBinder
from dluxshera.systems.base import compose_forward_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = REPO_ROOT / "work/experiments/ml_dataset_v3_template.yaml"
DEFAULT_VARIANTS = (
    "full_canonical",
    "sweep_fixed_nuisance",
    "sweep_schur_marginalized",
)
RECOMMENDED_VARIANT = "sweep_schur_marginalized"
FULL_CANONICAL_KEYS = (
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
VARIANT_PURPOSES = {
    "full_canonical": (
        "Raw local sensitivity directions of the image model when every "
        "canonical astrometry parameter is treated as an inference parameter. "
        "This can be dominated by X/Y/PA registration modes."
    ),
    "sweep_fixed_nuisance": (
        "Eigenmodes among ML labeled sweep dimensions while registration "
        "nuisance parameters are held fixed. This is directly aligned with ML "
        "labels but optimistic."
    ),
    "sweep_schur_marginalized": (
        "Sweep-parameter sensitivity directions after X/Y/PA registration "
        "nuisance parameters absorb what they can. This is recommended for ML "
        "analysis."
    ),
}
NEAR_ZERO_RELATIVE = 1e-12
NEAR_ZERO_ABSOLUTE = 0.0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    jax.config.update("jax_enable_x64", True)

    template_path = _resolve_path(args.template)
    template_cfg = load_config_file(template_path)
    if "system" not in template_cfg or "experiment" not in template_cfg:
        raise ValueError("Template must contain top-level 'system' and 'experiment' blocks.")

    template_system = deepcopy(template_cfg["system"])
    template_experiment = deepcopy(template_cfg["experiment"])
    effective_cfg_input, overrides = _apply_overrides(
        template_cfg,
        system_preset=args.system_preset,
        exposure_time_s=args.exposure_time_s,
        psf_npix=args.psf_npix,
        noise=args.noise,
    )
    resolved_cfg = resolve_config(effective_cfg_input)
    system_cfg = resolved_cfg["system"]
    experiment_cfg = resolved_cfg["experiment"]

    sweep_keys = _require_string_list(experiment_cfg, "sweep_keys", "experiment.sweep_keys")
    nuisance_keys = _require_string_list(
        experiment_cfg.get("datasets", {}).get("nuisance_replicates", {}),
        "keys",
        "experiment.datasets.nuisance_replicates.keys",
    )
    variants = _parse_variants(args.variants)
    rcond = _parse_schur_inverse(args.schur_inverse)
    outdir = _default_outdir() if args.outdir is None else _resolve_path(args.outdir)

    forward_spec = compose_forward_spec(system_cfg)
    _validate_keys(forward_spec, FULL_CANONICAL_KEYS, "full canonical keys")
    _validate_keys(forward_spec, sweep_keys, "experiment.sweep_keys")
    _validate_keys(forward_spec, nuisance_keys, "nuisance replicate keys")

    truth_store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(forward_spec)
    full_subspec = forward_spec.subset(FULL_CANONICAL_KEYS)
    theta_true = pack_params(full_subspec, truth_store, dtype=jnp.float64)
    full_labels = generate_fim_labels(FULL_CANONICAL_KEYS, cfg=system_cfg, store=truth_store)
    full_index_map = build_index_map(full_subspec, truth_store, theta=theta_true)
    full_param_rows = _parameter_rows(
        keys=FULL_CANONICAL_KEYS,
        labels=full_labels,
        index_map=full_index_map,
        store=truth_store,
        system_cfg=system_cfg,
        included_by="full_canonical",
    )

    print(f"Template: {_rel(template_path)}")
    print(f"Output directory: {_rel(outdir)}")
    print(f"Variants: {', '.join(variants)}")
    print("Effective system:")
    print(f"  preset={system_cfg.get('preset')}")
    print(f"  source.kind={system_cfg.get('source', {}).get('kind')}")
    print(f"  source.target={system_cfg.get('source', {}).get('target')}")
    print(f"  exposure_time_s={system_cfg.get('source', {}).get('exposure_time_s')}")
    print(f"  psf_npix={system_cfg.get('optics', {}).get('psf_npix')}")
    print(f"  primary_noll_indices={system_cfg.get('optics', {}).get('primary_noll_indices')}")
    print(f"  secondary_noll_indices={system_cfg.get('optics', {}).get('secondary_noll_indices')}")
    print(f"  noise.enabled={experiment_cfg.get('noise', {}).get('enabled')}")
    print(f"Full canonical packed parameters: {len(full_labels)}")

    if args.dry_run:
        print("Dry run: validated config, parameter expansion, and output plan; no FIM computed or files written.")
        return 0

    if outdir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {outdir}. Pass --overwrite to replace files.")
    outdir.mkdir(parents=True, exist_ok=True)

    binder = SheraBinder(system_cfg, forward_spec, truth_store)
    data = binder.model()
    var = _variance_for_deterministic_observation(data)
    nll_loss_fn, theta_ref = make_binder_nll_fn(
        binder=binder,
        infer_keys=FULL_CANONICAL_KEYS,
        data=data,
        var=var,
        noise_model="gaussian",
        reduce="sum",
        theta0_store=truth_store,
    )

    print("Computing full canonical FIM...")
    full_fim = _symmetrize(np.asarray(fim_theta(nll_loss_fn, theta_ref), dtype=float))
    variant_summaries: dict[str, dict[str, Any]] = {}

    for variant in variants:
        variant_dir = outdir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        if variant == "full_canonical":
            rows = [dict(row, included_by=variant) for row in full_param_rows]
            fim = full_fim
            manifest_extra = {"fim_construction_source": "full FIM"}
            basis_keys = list(FULL_CANONICAL_KEYS)
        elif variant == "sweep_fixed_nuisance":
            basis_keys = list(sweep_keys)
            rows, indices = _rows_and_indices_for_keys(
                keys=basis_keys,
                full_rows=full_param_rows,
                full_index_map=full_index_map,
                included_by=variant,
            )
            fim = _symmetrize(full_fim[np.ix_(indices, indices)])
            manifest_extra = {"fim_construction_source": "principal block"}
        elif variant == "sweep_schur_marginalized":
            basis_keys = list(sweep_keys)
            rows, theta_indices = _rows_and_indices_for_keys(
                keys=basis_keys,
                full_rows=full_param_rows,
                full_index_map=full_index_map,
                included_by=variant,
            )
            phi_rows, phi_indices = _rows_and_indices_for_keys(
                keys=nuisance_keys,
                full_rows=full_param_rows,
                full_index_map=full_index_map,
                included_by="schur_phi_nuisance",
            )
            fim, schur_diag = _schur_reduce(full_fim, theta_indices, phi_indices, rcond=rcond)
            manifest_extra = {
                "fim_construction_source": "Schur-reduced block",
                "theta_labels": [row["parameter_label"] for row in rows],
                "phi_labels": [row["parameter_label"] for row in phi_rows],
                "F_phiphi_shape": [len(phi_indices), len(phi_indices)],
                "schur": schur_diag,
            }
        else:  # pragma: no cover - guarded by parser
            raise ValueError(f"Unknown variant: {variant}")

        eig = _eigendecompose(fim)
        stats = _write_variant_outputs(
            variant_dir=variant_dir,
            variant=variant,
            purpose=VARIANT_PURPOSES[variant],
            basis_keys=basis_keys,
            rows=rows,
            fim=fim,
            eig=eig,
            manifest_extra=manifest_extra,
        )
        variant_summaries[variant] = stats
        print(f"Exported {variant}: {stats['n_parameters']} parameters, {stats['n_modes']} modes")

    top_manifest = _top_manifest(
        script_path=Path(__file__).resolve(),
        template_path=template_path,
        outdir=outdir,
        template_system=template_system,
        template_experiment=template_experiment,
        system_cfg=system_cfg,
        experiment_cfg=experiment_cfg,
        overrides=overrides,
        variants=variants,
        variant_summaries=variant_summaries,
        full_canonical_keys=FULL_CANONICAL_KEYS,
        sweep_keys=sweep_keys,
        nuisance_keys=nuisance_keys,
        data=data,
        var=var,
        truth_store=truth_store,
    )
    _write_json(outdir / "manifest.json", top_manifest)
    print(f"Wrote top-level manifest: {_rel(outdir / 'manifest.json')}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--noise", choices=("enabled", "disabled"), default=None)
    parser.add_argument("--system-preset", default=None)
    parser.add_argument("--exposure-time-s", type=float, default=None)
    parser.add_argument("--psf-npix", type=int, default=None)
    parser.add_argument("--schur-inverse", default="rcond=1e-12")
    return parser.parse_args(argv)


def _apply_overrides(
    cfg: Mapping[str, Any], *, system_preset: str | None, exposure_time_s: float | None,
    psf_npix: int | None, noise: str | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = deepcopy(dict(cfg))
    overrides: dict[str, Any] = {}
    if system_preset is not None:
        overrides["system.preset"] = {"template": out["system"].get("preset"), "effective": system_preset}
        out["system"]["preset"] = system_preset
    if exposure_time_s is not None:
        src = out["system"].setdefault("source", {})
        overrides["system.source.exposure_time_s"] = {"template": src.get("exposure_time_s"), "effective": exposure_time_s}
        src["exposure_time_s"] = exposure_time_s
    if psf_npix is not None:
        optics = out["system"].setdefault("optics", {})
        overrides["system.optics.psf_npix"] = {"template": optics.get("psf_npix"), "effective": psf_npix}
        optics["psf_npix"] = psf_npix
    if noise is not None:
        enabled = noise == "enabled"
        noise_cfg = out["experiment"].setdefault("noise", {})
        overrides["experiment.noise.enabled"] = {"template": noise_cfg.get("enabled"), "effective": enabled}
        noise_cfg["enabled"] = enabled
    return out, overrides


def _variance_for_deterministic_observation(data: Any) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    # The v2 generator used model counts as Gaussian variance for Fisher scaling.
    return np.maximum(arr, 1.0)


def _parse_variants(text: str) -> list[str]:
    variants = [part.strip() for part in text.split(",") if part.strip()]
    unknown = sorted(set(variants) - set(DEFAULT_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {', '.join(unknown)}")
    return variants


def _parse_schur_inverse(text: str) -> float:
    if text.startswith("rcond="):
        return float(text.split("=", 1)[1])
    return float(text)


def _require_string_list(mapping: Mapping[str, Any], key: str, label: str) -> list[str]:
    value = mapping.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of strings.")
    return list(value)


def _validate_keys(forward_spec: Any, keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in forward_spec]
    if missing:
        raise ValueError(f"Missing {label} from resolved forward spec: {', '.join(missing)}")


def _parameter_rows(
    *, keys: Sequence[str], labels: Sequence[str], index_map: Mapping[str, Any],
    store: ParameterStore, system_cfg: Mapping[str, Any], included_by: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    label_index = 0
    optics_cfg = system_cfg.get("optics", {})
    noll_by_key = {
        "optics.primary.zernike_coeffs_nm": list(optics_cfg.get("primary_noll_indices") or []),
        "optics.secondary.zernike_coeffs_nm": list(optics_cfg.get("secondary_noll_indices") or []),
    }
    entry_by_key = {entry["name"]: entry for entry in index_map["entries"]}
    for key in keys:
        entry = entry_by_key[key]
        size = int(entry["stop"]) - int(entry["start"])
        values = np.asarray(store.get(key)).ravel()
        for component_index in range(size):
            label = labels[label_index]
            noll = None
            if key in noll_by_key and component_index < len(noll_by_key[key]):
                noll = int(noll_by_key[key][component_index])
            rows.append({
                "parameter_index": label_index,
                "parameter_label": label,
                "parameter_group": _parameter_group(key),
                "parameter_unit": _parameter_unit(key),
                "base_key": key,
                "component_index": component_index if size > 1 else "",
                "noll_index": "" if noll is None else noll,
                "truth_value": _json_scalar(values[component_index]),
                "included_by": included_by,
                "description": _parameter_description(key, noll),
            })
            label_index += 1
    return rows


def _rows_and_indices_for_keys(
    *, keys: Sequence[str], full_rows: Sequence[Mapping[str, Any]],
    full_index_map: Mapping[str, Any], included_by: str
) -> tuple[list[dict[str, Any]], list[int]]:
    rows_by_index = {int(row["parameter_index"]): row for row in full_rows}
    indices: list[int] = []
    for entry in full_index_map["entries"]:
        if entry["name"] not in keys:
            continue
        indices.extend(range(int(entry["start"]), int(entry["stop"])))
    rows = []
    for new_index, full_index in enumerate(indices):
        row = dict(rows_by_index[full_index])
        row["parameter_index"] = new_index
        row["included_by"] = included_by
        rows.append(row)
    return rows, indices


def _schur_reduce(full_fim: np.ndarray, theta_idx: Sequence[int], phi_idx: Sequence[int], *, rcond: float) -> tuple[np.ndarray, dict[str, Any]]:
    F_tt = full_fim[np.ix_(theta_idx, theta_idx)]
    F_tphi = full_fim[np.ix_(theta_idx, phi_idx)]
    F_phit = full_fim[np.ix_(phi_idx, theta_idx)]
    F_phiphi = _symmetrize(full_fim[np.ix_(phi_idx, phi_idx)])
    phi_eigs = np.linalg.eigvalsh(F_phiphi)
    max_abs = float(np.max(np.abs(phi_eigs))) if phi_eigs.size else float("nan")
    rank = int(np.linalg.matrix_rank(F_phiphi, tol=rcond * max_abs)) if phi_eigs.size else 0
    pinv = np.linalg.pinv(F_phiphi, rcond=rcond)
    eff = _symmetrize(F_tt - F_tphi @ pinv @ F_phit)
    eff_eigs = np.linalg.eigvalsh(eff)
    cond = float(max_abs / np.min(np.abs(phi_eigs[np.abs(phi_eigs) > 0]))) if phi_eigs.size and np.any(np.abs(phi_eigs) > 0) else float("inf")
    fixed_norm_delta = float(np.linalg.norm(F_tt - eff, ord="fro"))
    warnings = []
    if rank < F_phiphi.shape[0]:
        warnings.append("F_phiphi is rank deficient under the selected rcond.")
    if phi_eigs.size and float(np.min(phi_eigs)) < 0.0:
        warnings.append("F_phiphi has negative eigenvalues before pseudo-inversion.")
    if eff_eigs.size and float(np.min(eff_eigs)) < 0.0:
        warnings.append("Schur-reduced FIM has negative eigenvalues.")
    return eff, {
        "inverse_policy": "numpy.linalg.pinv",
        "rcond": float(rcond),
        "damping": None,
        "estimated_rank": rank,
        "F_phiphi_eigenvalue_min": float(np.min(phi_eigs)) if phi_eigs.size else float("nan"),
        "F_phiphi_eigenvalue_max": float(np.max(phi_eigs)) if phi_eigs.size else float("nan"),
        "F_phiphi_condition_estimate": cond,
        "schur_eigenvalue_min": float(np.min(eff_eigs)) if eff_eigs.size else float("nan"),
        "schur_negative_eigenvalue_count": int(np.count_nonzero(eff_eigs < 0.0)),
        "fixed_minus_schur_frobenius_norm": fixed_norm_delta,
        "warnings": warnings,
    }


def _eigendecompose(fim: np.ndarray) -> dict[str, np.ndarray]:
    fim_sym = _symmetrize(fim)
    eigvals, eigvecs = np.linalg.eigh(fim_sym)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    norms = np.linalg.norm(eigvecs, axis=0)
    norms[norms == 0.0] = 1.0
    eigvecs = eigvecs / norms
    return {"eigenvalues": eigvals, "eigenvectors": eigvecs}


def _write_variant_outputs(
    *, variant_dir: Path, variant: str, purpose: str, basis_keys: Sequence[str],
    rows: Sequence[Mapping[str, Any]], fim: np.ndarray, eig: Mapping[str, np.ndarray],
    manifest_extra: Mapping[str, Any]
) -> dict[str, Any]:
    labels = [str(row["parameter_label"]) for row in rows]
    eigvals = eig["eigenvalues"]
    eigvecs = eig["eigenvectors"]
    threshold = _near_zero_threshold(eigvals)

    _write_csv(variant_dir / "parameter_labels.csv", rows)
    _write_fim_csv(variant_dir / "fim_matrix.csv", labels, fim)
    eigenvalue_rows = _eigenvalue_rows(eigvals, threshold)
    _write_csv(variant_dir / "eigenvalues.csv", eigenvalue_rows)
    _write_csv(variant_dir / "eigenvectors_long.csv", _eigenvectors_long_rows(eigvals, eigvecs, rows, threshold))
    _write_csv(variant_dir / "eigenvectors_wide.csv", _eigenvectors_wide_rows(eigvals, eigvecs, labels, threshold))

    negative_count = int(np.count_nonzero(eigvals < 0.0))
    near_zero_count = int(np.count_nonzero(np.abs(eigvals) <= threshold))
    manifest = {
        "variant": variant,
        "purpose": purpose,
        "recommended_for_ml_analysis": variant == RECOMMENDED_VARIANT,
        "input_effective_basis_keys": list(basis_keys),
        "expanded_parameter_labels": labels,
        "n_parameters": len(labels),
        "fim_shape": list(fim.shape),
        "eigensolver_convention": "numpy.linalg.eigh on symmetrized matrix; eigenvectors are physical packed-basis unit-norm columns",
        "eigenvalue_sort_convention": "descending eigenvalue",
        "normalization_convention": "unit Euclidean norm in physical packed parameter basis",
        "near_zero_eigenvalue_threshold": threshold,
        "sigma_equiv_policy": "1/sqrt(eigenvalue) for eigenvalue > threshold; NaN otherwise",
        "negative_eigenvalue_count": negative_count,
        "near_zero_eigenvalue_count": near_zero_count,
        "whitened": False,
        "truncated": False,
        "all_modes_exported": True,
        **manifest_extra,
    }
    _write_json(variant_dir / "manifest.json", manifest)
    return {
        "path": str(variant_dir),
        "n_parameters": len(labels),
        "n_modes": int(eigvals.size),
        "eigenvalue_min": float(np.min(eigvals)) if eigvals.size else float("nan"),
        "eigenvalue_max": float(np.max(eigvals)) if eigvals.size else float("nan"),
        "negative_eigenvalue_count": negative_count,
        "near_zero_eigenvalue_count": near_zero_count,
        "recommended_for_ml_analysis": variant == RECOMMENDED_VARIANT,
        "manifest": str(variant_dir / "manifest.json"),
    }


def _eigenvalue_rows(eigvals: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    max_eval = float(np.max(eigvals)) if eigvals.size else float("nan")
    rows = []
    for mode_index, val in enumerate(eigvals):
        safe = float(val) > threshold
        rows.append({
            "mode_index": mode_index,
            "mode_rank_descending": mode_index + 1,
            "eigenvalue": float(val),
            "sigma_equiv": 1.0 / math.sqrt(float(val)) if safe else float("nan"),
            "condition_relative_to_max": float(val / max_eval) if max_eval not in (0.0, float("nan")) else float("nan"),
            "is_positive": bool(val > 0.0),
            "is_near_zero": bool(abs(float(val)) <= threshold),
            "normalization": "unit_euclidean_physical_basis",
            "whitened": False,
            "truncated": False,
        })
    return rows


def _eigenvectors_long_rows(eigvals: np.ndarray, eigvecs: np.ndarray, param_rows: Sequence[Mapping[str, Any]], threshold: float) -> list[dict[str, Any]]:
    rows = []
    for mode_index, val in enumerate(eigvals):
        components = eigvecs[:, mode_index]
        order = np.argsort(np.abs(components))[::-1]
        ranks = {int(idx): rank + 1 for rank, idx in enumerate(order)}
        safe = float(val) > threshold
        sigma = 1.0 / math.sqrt(float(val)) if safe else float("nan")
        for pidx, component in enumerate(components):
            prow = param_rows[pidx]
            rows.append({
                "mode_index": mode_index,
                "mode_rank_descending": mode_index + 1,
                "eigenvalue": float(val),
                "sigma_equiv": sigma,
                "parameter_index": pidx,
                "parameter_label": prow["parameter_label"],
                "parameter_group": prow["parameter_group"],
                "parameter_unit": prow["parameter_unit"],
                "component": float(component),
                "abs_component": float(abs(component)),
                "component_rank_within_mode": ranks[pidx],
                "normalization": "unit_euclidean_physical_basis",
                "whitened": False,
                "truncated": False,
            })
    return rows


def _eigenvectors_wide_rows(eigvals: np.ndarray, eigvecs: np.ndarray, labels: Sequence[str], threshold: float) -> list[dict[str, Any]]:
    rows = []
    for mode_index, val in enumerate(eigvals):
        safe = float(val) > threshold
        row: dict[str, Any] = {
            "mode_index": mode_index,
            "mode_rank_descending": mode_index + 1,
            "eigenvalue": float(val),
            "sigma_equiv": 1.0 / math.sqrt(float(val)) if safe else float("nan"),
        }
        for pidx, label in enumerate(labels):
            row[label] = float(eigvecs[pidx, mode_index])
        rows.append(row)
    return rows


def _write_fim_csv(path: Path, labels: Sequence[str], fim: np.ndarray) -> None:
    rows = []
    for i, label in enumerate(labels):
        row = {"parameter_label": label}
        for j, col_label in enumerate(labels):
            row[col_label] = float(fim[i, j])
        rows.append(row)
    _write_csv(path, rows)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _top_manifest(**kwargs: Any) -> dict[str, Any]:
    system_cfg = kwargs["system_cfg"]
    experiment_cfg = kwargs["experiment_cfg"]
    optics = system_cfg.get("optics", {})
    source = system_cfg.get("source", {})
    detector = system_cfg.get("detector", {})
    return {
        "script_name": Path(kwargs["script_path"]).name,
        "script_path": str(kwargs["script_path"]),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "repository_root": str(REPO_ROOT),
        "template_path": str(kwargs["template_path"]),
        "selected_system_block_from_template": kwargs["template_system"],
        "template_experiment_summary": {
            "kind": kwargs["template_experiment"].get("kind"),
            "noise": kwargs["template_experiment"].get("noise"),
            "sweep_keys": kwargs["template_experiment"].get("sweep_keys"),
            "nuisance_replicates": kwargs["template_experiment"].get("datasets", {}).get("nuisance_replicates"),
            "pair_grid": kwargs["template_experiment"].get("datasets", {}).get("pair_grid"),
        },
        "resolved_effective_system_config": system_cfg,
        "cli_overrides_applied": bool(kwargs["overrides"]),
        "cli_overrides": kwargs["overrides"],
        "system_preset": system_cfg.get("preset"),
        "source_kind": source.get("kind"),
        "source_target": source.get("target"),
        "exposure_time_s": source.get("exposure_time_s"),
        "psf_npix": optics.get("psf_npix"),
        "wavelength_settings": {key: optics.get(key) for key in sorted(optics) if "lambda" in key.lower() or "wavelength" in key.lower()},
        "n_lambda": optics.get("n_lambda"),
        "detector_settings": detector,
        "optics_settings": optics,
        "primary_noll_indices": optics.get("primary_noll_indices"),
        "secondary_noll_indices": optics.get("secondary_noll_indices"),
        "primary_zernike_truth_values_nm": _as_list(kwargs["truth_store"].get("optics.primary.zernike_coeffs_nm")),
        "secondary_zernike_truth_values_nm": _as_list(kwargs["truth_store"].get("optics.secondary.zernike_coeffs_nm")),
        "dataset_sweep_keys": kwargs["sweep_keys"],
        "nuisance_replicate_keys": kwargs["nuisance_keys"],
        "full_canonical_keys": list(kwargs["full_canonical_keys"]),
        "exported_variants": kwargs["variants"],
        "recommended_variant": RECOMMENDED_VARIANT,
        "variant_summaries": kwargs["variant_summaries"],
        "image_shape": list(np.asarray(kwargs["data"]).shape),
        "variance_policy": "max(nominal_model_counts, 1.0), matching deterministic Fisher-scaling usage in the dataset generator while avoiding zero variance pixels",
        "variance_summary": _array_summary(kwargs["var"]),
        "whitened": False,
        "truncated": False,
        "all_modes_exported": True,
    }


def _get_store_like(system_cfg: Mapping[str, Any], key: str) -> Any:
    cur: Any = system_cfg
    for part in key.split("."):
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(part)
    return cur


def _parameter_group(key: str) -> str:
    if key.startswith("source."):
        return "source"
    if key.startswith("optics.primary."):
        return "optics.primary"
    if key.startswith("optics.secondary."):
        return "optics.secondary"
    if key.startswith("optics."):
        return "optics"
    return key.split(".", 1)[0]


def _parameter_unit(key: str) -> str:
    if key.endswith("_as") or key.endswith("as_per_pix"):
        return "arcsec" if not key.endswith("as_per_pix") else "arcsec/pixel"
    if key.endswith("_deg"):
        return "deg"
    if key.endswith("_nm") or key.endswith("coeffs_nm"):
        return "nm"
    if key == "source.log_flux_total":
        return "log(counts)"
    if key == "source.contrast":
        return "dimensionless"
    return ""


def _parameter_description(key: str, noll: int | None) -> str:
    if noll is not None:
        mirror = "primary/M1" if "primary" in key else "secondary/M2"
        return f"{mirror} Zernike coefficient for Noll index {noll}"
    return key


def _near_zero_threshold(eigvals: np.ndarray) -> float:
    max_abs = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
    return max(NEAR_ZERO_ABSOLUTE, NEAR_ZERO_RELATIVE * max_abs)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    return 0.5 * (arr + arr.T)


def _array_summary(arr: Any) -> dict[str, Any]:
    data = np.asarray(arr, dtype=float)
    return {"min": float(np.nanmin(data)), "max": float(np.nanmax(data)), "mean": float(np.nanmean(data))}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _json_scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.size != 1:
        return _json_safe(arr)
    return _json_safe(arr.reshape(-1)[0])


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def _as_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    return _json_safe(np.asarray(value).ravel())


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def _default_outdir() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / "Results/ml_eigenmodes" / f"ml_dataset_v3_{stamp}"


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
