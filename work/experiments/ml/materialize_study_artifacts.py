from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _ensure_s10_fim_source_process_start_x64(argv: list[str]) -> None:
    if not argv or argv[0] != "make-s10-fim-source":
        return
    if os.environ.get("JAX_ENABLE_X64") == "1":
        return
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_ensure_s10_fim_source_process_start_x64(sys.argv[1:])

from dluxshera.datasets.schema import json_ready, write_json
from dluxshera.ml import (
    build_artifact_lock,
    artifact_lock_content_sha256,
    NOISY_EVAL_RECIPE_SCHEMA_VERSION,
    expand_study_run_plan,
    fit_intensity_scaler,
    generate_role_preserving_split_registry,
    generate_split_registry,
    build_s10_nominal_physical_fim_source,
    build_science_eigenbasis_from_source,
    PairPolicy,
    generate_frozen_pair_manifest,
    load_intensity_scaler,
    load_pair_manifest,
    noise_config_identity,
    pair_noise_side_seeds,
    load_sample_catalog,
    load_split_registry,
    load_study_contract_artifacts,
    load_study_prescription,
    resolve_study_artifact_contract,
    resolve_study_experiment_config,
    science_fim_sanity_summary,
    science_mode_weight_summary,
    split_registry_content_sha256,
    validate_evaluation_artifact_against_recipe,
    write_artifact_lock,
    write_intensity_scaler,
    write_pair_manifest,
    write_science_eigenbasis,
    write_split_registry,
)


def _materialize_artifact(
    *,
    artifact_key: str,
    recipe: dict,
    study: dict,
    catalog,
    split_registry,
    output_root: Path,
    overwrite: bool,
    config: dict | None = None,
) -> dict:
    policies = dict(study["pair_policies"])
    policy_id = str(recipe["pair_policy_id"])
    if policy_id not in policies:
        raise ValueError(f"Unknown pair policy {policy_id!r} for {artifact_key}.")
    policy_payload = dict(policies[policy_id])
    policy_payload.setdefault("policy_id", policy_id)
    policy = PairPolicy.from_dict(policy_payload)
    manifest = generate_frozen_pair_manifest(
        catalog,
        split_registry,
        policy=policy,
        artifact_id=str(recipe["artifact_id"]),
        split=str(recipe["split"]),
        seed=int(recipe["seed"]),
        pairs_per_slice=int(recipe["pairs_per_slice"]),
        eval_slices=dict(recipe["eval_slices"]),
    )
    validate_evaluation_artifact_against_recipe(
        manifest,
        study=study,
        artifact_key=artifact_key,
        split_registry=split_registry,
        config=config,
    )
    outdir = output_root / str(study["study_id"]) / f"{artifact_key}_pairs" / str(recipe["artifact_id"])
    write_pair_manifest(outdir, manifest, overwrite=overwrite)
    return {"artifact_key": artifact_key, "path": str(outdir), "summary": manifest.summary()}


def _parse_key_path(values: list[str], *, option: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in str(value):
            raise ValueError(f"{option} expects KEY=PATH, got {value!r}.")
        key, path = str(value).split("=", 1)
        if not key.strip() or not path.strip():
            raise ValueError(f"{option} expects non-empty KEY=PATH, got {value!r}.")
        out[key.strip()] = Path(path.strip())
    return out


def _artifact_profile_id(study: dict, recipe: dict) -> str:
    profile = recipe.get("artifact_profile")
    if profile not in (None, ""):
        return str(profile)
    profiles = study.get("artifact_profiles")
    return "standard" if isinstance(profiles, dict) and "standard" in profiles else "default"


def _split_path_for_profile(
    *,
    profile_id: str,
    default_split_registry: Path | None,
    profile_paths: dict[str, Path],
) -> Path:
    if profile_id in profile_paths:
        return profile_paths[profile_id]
    if profile_id == "default" and default_split_registry is not None:
        return default_split_registry
    if profile_id == "standard" and default_split_registry is not None:
        return default_split_registry
    valid = sorted(profile_paths)
    raise ValueError(
        f"No split registry path supplied for artifact profile {profile_id!r}. "
        f"Pass --split-profile {profile_id}=/path/to/split.json. "
        f"Currently supplied profiles: {valid}."
    )


def _valid_artifact_keys(study: dict) -> list[str]:
    artifacts = study.get("evaluation_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Study prescription must define evaluation_artifacts.")
    return sorted(str(key) for key in artifacts)


def _selected_artifacts(study: dict, artifact_arg: str) -> dict[str, dict]:
    artifacts = dict(study["evaluation_artifacts"])
    if artifact_arg == "all":
        return {str(key): dict(value) for key, value in artifacts.items()}
    if artifact_arg not in artifacts:
        valid = ", ".join(_valid_artifact_keys(study))
        raise ValueError(f"Unknown evaluation artifact {artifact_arg!r}. Valid keys: {valid}.")
    return {artifact_arg: dict(artifacts[artifact_arg])}


def _noise_eval_content_sha256(payload: dict) -> str:
    stable = dict(payload)
    stable.pop("generated_at", None)
    stable.pop("content_identity", None)
    raw = json.dumps(json_ready(stable), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return __import__("hashlib").sha256(raw.encode("utf-8")).hexdigest()


def _s10_weight_summaries(eigenvalues) -> dict:
    return {
        "S10-E02_strong_mode_weights": science_mode_weight_summary(
            eigenvalues,
            mode="strong_mode_weighted",
            strength=0.5,
            eigenvalue_floor=1.0e-6,
            weight_cap=10.0,
        ),
        "S10-E03_weak_mode_weights": science_mode_weight_summary(
            eigenvalues,
            mode="weak_mode_weighted",
            strength=0.5,
            eigenvalue_floor=1.0e-4,
            weight_cap=5.0,
        ),
    }


def _validate_s10_weight_summaries(summaries: dict) -> None:
    uniform = [
        key
        for key, summary in summaries.items()
        if bool(summary.get("uniform_allclose_to_one", False))
    ]
    if uniform:
        raise ValueError(
            "S10 weighted eigenmode objectives collapse to ordinary MSE for "
            f"{uniform}; refusing to materialize a no-op weighted basis."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize and audit tracked ML study artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    expand = subparsers.add_parser("expand", help="Expand a tracked study run matrix.")
    expand.add_argument("--study", type=Path, required=True)
    expand.add_argument("--output-dir", type=Path, default=None)
    expand.add_argument("--overwrite", action="store_true", default=False)

    split = subparsers.add_parser("make-split", help="Materialize a split registry.")
    split.add_argument("--prepared-root", type=Path, required=True)
    split.add_argument("--out", type=Path, required=True)
    split.add_argument("--artifact-id", default="SPLIT-ML-v1")
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--role-preserving-v4", action="store_true", default=False)
    split.add_argument("--nuisance-holdout-indices", default=None)

    scaler = subparsers.add_parser("make-scaler", help="Materialize a canonical train-derived image scaler.")
    scaler.add_argument("--prepared-root", type=Path, required=True)
    scaler.add_argument("--split-registry", type=Path, required=True)
    scaler.add_argument("--out", type=Path, required=True)
    scaler.add_argument("--artifact-id", default="SCALER-V4-GLOBAL-MAX-ABS-v1")
    scaler.add_argument("--dataset-family", action="append", default=[])
    scaler.add_argument("--mode", choices=("global_max_abs", "global_p99_abs", "raw"), default="global_max_abs")
    scaler.add_argument("--max-samples", type=int, default=None)
    scaler.add_argument("--overwrite", action="store_true", default=False)

    eigen = subparsers.add_parser("make-eigenbasis", help="Freeze an explicit science FIM/eigenbasis artifact.")
    eigen.add_argument("--prepared-root", type=Path, required=True)
    eigen.add_argument("--matrix-json", type=Path, required=True)
    eigen.add_argument("--out", type=Path, required=True)
    eigen.add_argument("--artifact-id", required=True)
    eigen.add_argument(
        "--coordinate-convention",
        default="delta_z_science = z_B - z_A in prepared-catalog parameter order",
    )
    eigen.add_argument("--source-note", default=None)
    eigen.add_argument("--overwrite", action="store_true", default=False)

    fim = subparsers.add_parser("make-s10-fim-source", help="Export the S10-v1 nominal physical science FIM source.")
    fim.add_argument("--prepared-root", type=Path, required=True)
    fim.add_argument("--out", type=Path, required=True)
    fim.add_argument("--artifact-id", default="s10_science_fim_source")
    fim.add_argument("--overwrite", action="store_true", default=False)

    noisy_eval = subparsers.add_parser("make-noisy-eval", help="Freeze a deterministic noisy-evaluation recipe.")
    noisy_eval.add_argument("--study", type=Path, required=True)
    noisy_eval.add_argument("--experiment-id", required=True)
    noisy_eval.add_argument("--prepared-root", type=Path, required=True)
    noisy_eval.add_argument("--split-registry", type=Path, required=True)
    noisy_eval.add_argument("--pair-manifest", type=Path, required=True)
    noisy_eval.add_argument("--artifact-key", default="validation")
    noisy_eval.add_argument("--out", type=Path, required=True)
    noisy_eval.add_argument("--artifact-id", required=True)
    noisy_eval.add_argument("--overwrite", action="store_true", default=False)

    pairs = subparsers.add_parser("make-pairs", help="Generate frozen pair manifests from a study recipe.")
    pairs.add_argument("--study", type=Path, required=True)
    pairs.add_argument("--prepared-root", type=Path, required=True)
    pairs.add_argument("--split-registry", type=Path, default=None)
    pairs.add_argument(
        "--split-profile",
        action="append",
        default=[],
        help="Profile-specific split path as PROFILE=/path/to/split.json.",
    )
    pairs.add_argument("--output-root", type=Path, required=True)
    pairs.add_argument("--artifact", default="all")
    pairs.add_argument("--overwrite", action="store_true", default=False)

    lock_parser = subparsers.add_parser("make-lock", help="Create a portable artifact lock.")
    lock_parser.add_argument("--study", type=Path, required=True)
    lock_parser.add_argument("--prepared-root", type=Path, required=True)
    lock_parser.add_argument("--split-registry", type=Path, required=True)
    lock_parser.add_argument("--artifact-profile", default=None)
    lock_parser.add_argument("--scaler", type=Path, required=True)
    lock_parser.add_argument("--validation-manifest", type=Path, default=None)
    lock_parser.add_argument("--test-manifest", type=Path, default=None)
    lock_parser.add_argument(
        "--pair-manifest",
        action="append",
        default=[],
        help="Named pair manifest as ARTIFACT_KEY=/path/to/manifest_dir.",
    )
    lock_parser.add_argument("--out", type=Path, required=True)
    lock_parser.add_argument("--overwrite", action="store_true", default=False)

    validate = subparsers.add_parser("validate-lock", help="Validate study artifacts against a lock.")
    validate.add_argument("--study", type=Path, required=True)
    validate.add_argument("--prepared-root", type=Path, required=True)
    validate.add_argument("--split-registry", type=Path, required=True)
    validate.add_argument("--scaler", type=Path, required=True)
    validate.add_argument("--validation-manifest", type=Path, required=True)
    validate.add_argument("--test-manifest", type=Path, default=None)
    validate.add_argument("--audit-manifest", action="append", default=[])
    validate.add_argument("--artifact-lock", type=Path, required=True)
    validate.add_argument("--noisy-eval-artifact", type=Path, default=None)

    audit = subparsers.add_parser("audit-study", help="Validate study expansion without loading large artifacts.")
    audit.add_argument("--study", type=Path, action="append", required=True)
    args = parser.parse_args(argv)

    if args.command == "expand":
        from dluxshera.ml.studies import write_study_run_plan

        study = load_study_prescription(args.study)
        rows = expand_study_run_plan(study)
        payload = {"study_id": study["study_id"], "run_count": len(rows), "rows": [row.to_dict() for row in rows]}
        if args.output_dir is not None:
            payload["written"] = write_study_run_plan(
                study=study,
                output_dir=args.output_dir,
                rows=rows,
                overwrite=args.overwrite,
            )
        print(json.dumps(json_ready(payload), indent=2, sort_keys=True))
        return 0
    if args.command == "make-split":
        catalog = load_sample_catalog(args.prepared_root)
        explicit = None
        if args.nuisance_holdout_indices:
            holdout = {int(v) for v in str(args.nuisance_holdout_indices).split(",") if v.strip()}
            explicit = {
                str(group): ("unseen_nuisance" if int(catalog.nuisance_bank_indices[idx]) in holdout else "train")
                for idx, group in enumerate(catalog.nuisance_group_ids)
            }
        registry = (
            generate_role_preserving_split_registry(catalog, artifact_id=args.artifact_id, explicit_nuisance_assignments=explicit)
            if args.role_preserving_v4
            else generate_split_registry(catalog, artifact_id=args.artifact_id, seed=args.seed)
        )
        write_split_registry(args.out, registry)
        print(json.dumps(json_ready({"path": str(args.out), "counts": registry.counts}), indent=2, sort_keys=True))
        return 0
    if args.command == "make-scaler":
        catalog = load_sample_catalog(args.prepared_root)
        registry = load_split_registry(args.split_registry, catalog=catalog)
        science_groups = registry.science_groups("train")
        nuisance_groups = registry.nuisance_groups("train")
        mask = [
            str(catalog.science_group_ids[idx]) in science_groups
            and str(catalog.nuisance_group_ids[idx]) in nuisance_groups
            and (not args.dataset_family or str(catalog.dataset_families[idx]) in set(args.dataset_family))
            for idx in range(catalog.sample_count)
        ]
        indices = [idx for idx, keep in enumerate(mask) if keep]
        fitted = fit_intensity_scaler(catalog, indices, mode=args.mode, max_samples=args.max_samples)
        payload = dict(fitted.source_population or {})
        payload["science_split"] = "train"
        payload["nuisance_split"] = "train"
        payload["dataset_families"] = list(args.dataset_family)
        fitted = type(fitted)(
            mode=fitted.mode,
            scale=fitted.scale,
            sample_count=fitted.sample_count,
            statistic=fitted.statistic,
            source_population=payload,
        )
        written = write_intensity_scaler(args.out, fitted, artifact_id=args.artifact_id, overwrite=args.overwrite)
        print(json.dumps(json_ready({"path": str(args.out), "content_identity": written["content_identity"], "sample_count": fitted.sample_count}), indent=2, sort_keys=True))
        return 0
    if args.command == "make-eigenbasis":
        catalog = load_sample_catalog(args.prepared_root)
        payload = json.loads(args.matrix_json.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("--matrix-json must contain a JSON object.")
        provenance = {
            "matrix_json": str(args.matrix_json),
            "source_note": args.source_note,
            "source_payload_identity": {
                "sha256": __import__("hashlib").sha256(
                    args.matrix_json.read_bytes()
                ).hexdigest()
            },
            "prepared_dataset": {
                "artifact_id": catalog.artifact_id,
                "prepared_dataset_hash": catalog.prepared_dataset_hash,
            },
            "declared_payload_provenance": payload.get("source_provenance", {}),
        }
        basis = build_science_eigenbasis_from_source(
            artifact_id=args.artifact_id,
            source=payload,
            catalog=catalog,
            coordinate_convention=args.coordinate_convention,
            source_provenance=provenance,
        )
        weight_summaries = _s10_weight_summaries(basis.eigenvalues)
        if str(args.artifact_id).startswith("S10-"):
            _validate_s10_weight_summaries(weight_summaries)
        write_science_eigenbasis(args.out, basis, overwrite=args.overwrite)
        print(
            json.dumps(
                json_ready(
                    {
                        "path": str(args.out),
                        "artifact_id": basis.artifact_id,
                        "content_identity": basis.content_identity,
                        "parameter_count": len(basis.parameter_labels),
                        "transformed_fz_summary": science_fim_sanity_summary(
                            basis.curvature_matrix
                        ),
                        "mode_weight_summaries": weight_summaries,
                    }
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "make-s10-fim-source":
        catalog = load_sample_catalog(args.prepared_root)
        payload = build_s10_nominal_physical_fim_source(
            catalog=catalog,
            artifact_id=args.artifact_id,
        )
        if args.out.exists() and not args.overwrite:
            raise FileExistsError(f"{args.out} exists; pass overwrite=True to replace it.")
        write_json(args.out, payload)
        print(
            json.dumps(
                json_ready(
                    {
                        "path": str(args.out),
                        "artifact_id": payload["artifact_id"],
                        "coordinate_space": payload["coordinate_space"],
                        "content_identity": payload["content_identity"],
                        "fisher_scale_compatibility": payload[
                            "diagonal_fisher_scale_compatibility"
                        ],
                        "physical_fim_summary": payload["diagnostics"]["physical_fim"],
                        "transformed_fz_summary": payload["diagnostics"]["transformed_fz"],
                        "parameter_count": len(payload["parameter_labels"]),
                    }
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "make-pairs":
        study = load_study_prescription(args.study)
        catalog = load_sample_catalog(args.prepared_root)
        profile_paths = _parse_key_path(args.split_profile, option="--split-profile")
        selected = _selected_artifacts(study, str(args.artifact))
        summaries = []
        registry_cache = {}
        for key, recipe in selected.items():
            profile_id = _artifact_profile_id(study, recipe)
            split_path = _split_path_for_profile(
                profile_id=profile_id,
                default_split_registry=args.split_registry,
                profile_paths=profile_paths,
            )
            if str(split_path) not in registry_cache:
                registry_cache[str(split_path)] = load_split_registry(split_path, catalog=catalog)
            split_registry = registry_cache[str(split_path)]
            contract = resolve_study_artifact_contract(study, profile_id)
            summaries.append(
                _materialize_artifact(
                    artifact_key=key,
                    recipe=dict(recipe),
                    study=study,
                    catalog=catalog,
                    split_registry=split_registry,
                    output_root=args.output_root,
                    overwrite=args.overwrite,
                    config={
                        "artifact_profile": profile_id,
                        "artifact_contract": contract,
                    },
                )
            )
        print(json.dumps(json_ready({"artifacts": summaries}), indent=2, sort_keys=True))
        return 0
    if args.command == "make-noisy-eval":
        study = load_study_prescription(args.study)
        config = resolve_study_experiment_config(study, experiment_id=args.experiment_id)
        catalog = load_sample_catalog(args.prepared_root)
        split_registry = load_split_registry(args.split_registry, catalog=catalog)
        pair_manifest = load_pair_manifest(args.pair_manifest, catalog=catalog, split_registry=split_registry)
        validate_evaluation_artifact_against_recipe(
            pair_manifest,
            study=study,
            artifact_key=str(args.artifact_key),
            split_registry=split_registry,
            config=config,
        )
        noise_cfg = dict(config.get("validation_noise", {}) or {})
        records = []
        for index, record in enumerate(pair_manifest.records):
            records.append(
                {
                    "index": int(index),
                    "pair_record_id": record.pair_record_id,
                    "sample_a_id": record.sample_a_id,
                    "sample_b_id": record.sample_b_id,
                    "seeds": pair_noise_side_seeds(
                        noise_cfg,
                        pair_record_id=record.pair_record_id,
                        dynamic_seed_offset=index,
                    ),
                }
            )
        payload = {
            "schema_version": NOISY_EVAL_RECIPE_SCHEMA_VERSION,
            "artifact_id": str(args.artifact_id),
            "study_id": str(study["study_id"]),
            "experiment_id": str(config["experiment_id"]),
            "artifact_key": str(args.artifact_key),
            "underlying_pair_manifest": {
                "artifact_id": pair_manifest.artifact_id,
                "content_sha256": pair_manifest.manifest.get("content_identity", {}).get("sha256"),
            },
            "split_registry": {
                "artifact_id": split_registry.artifact_id,
                "content_sha256": split_registry_content_sha256(split_registry),
            },
            "prepared_dataset": {
                "artifact_id": catalog.artifact_id,
                "prepared_dataset_hash": catalog.prepared_dataset_hash,
            },
            "noise_model": noise_config_identity(noise_cfg),
            "record_count": len(records),
            "records": records,
            "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        }
        payload["content_identity"] = {
            "algorithm": "sha256/json-canonical/noisy-eval-recipe-v1",
            "sha256": _noise_eval_content_sha256(payload),
            "excludes": ["generated_at", "content_identity"],
        }
        if args.out.exists() and not args.overwrite:
            raise FileExistsError(f"{args.out} exists; pass overwrite=True to replace it.")
        write_json(args.out, payload)
        print(
            json.dumps(
                json_ready(
                    {
                        "path": str(args.out),
                        "artifact_id": payload["artifact_id"],
                        "content_identity": payload["content_identity"],
                        "pair_count": len(records),
                    }
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "make-lock":
        study = load_study_prescription(args.study)
        catalog = load_sample_catalog(args.prepared_root)
        registry = load_split_registry(args.split_registry, catalog=catalog)
        scaler_artifact = load_intensity_scaler(args.scaler)
        manifest_paths = _parse_key_path(args.pair_manifest, option="--pair-manifest")
        if args.validation_manifest is not None:
            manifest_paths.setdefault("validation", args.validation_manifest)
        if args.test_manifest is not None:
            manifest_paths.setdefault("test", args.test_manifest)
        if not manifest_paths:
            valid = ", ".join(_valid_artifact_keys(study))
            raise ValueError(
                "make-lock requires at least one --pair-manifest KEY=PATH or "
                f"--validation-manifest. Valid artifact keys: {valid}."
            )
        profile_id = args.artifact_profile
        if profile_id in (None, ""):
            profiles = study.get("artifact_profiles", {})
            profile_id = "standard" if isinstance(profiles, dict) and "standard" in profiles else "default"
        contract = resolve_study_artifact_contract(study, str(profile_id))
        lock_cfg = contract.get("artifact_lock", {})
        split_cfg = contract.get("split_registry", {})
        expected_split_id = split_cfg.get("artifact_id") if isinstance(split_cfg, dict) else None
        if expected_split_id and str(expected_split_id) != str(registry.artifact_id):
            raise ValueError(
                f"Split registry {registry.artifact_id!r} is incompatible with artifact "
                f"profile {profile_id!r}; expected {expected_split_id!r}."
            )
        manifests = {}
        for key, path in manifest_paths.items():
            artifacts = dict(study.get("evaluation_artifacts", {}))
            if key not in artifacts:
                valid = ", ".join(_valid_artifact_keys(study))
                raise ValueError(f"Unknown evaluation artifact {key!r}. Valid keys: {valid}.")
            recipe_profile = _artifact_profile_id(study, dict(artifacts[key]))
            if recipe_profile != str(profile_id):
                raise ValueError(
                    f"Pair manifest {key!r} belongs to artifact profile {recipe_profile!r}, "
                    f"but this lock is for profile {profile_id!r}."
                )
            manifests[str(key)] = load_pair_manifest(path, catalog=catalog, split_registry=registry)
        lock = build_artifact_lock(
            study=study,
            catalog=catalog,
            split_registry=registry,
            scaler=scaler_artifact,
            pair_manifests=manifests,
            lock_id=lock_cfg.get("lock_id") if isinstance(lock_cfg, dict) else None,
        )
        write_artifact_lock(args.out, lock, overwrite=args.overwrite)
        print(json.dumps(json_ready({"path": str(args.out), "lock_id": lock.lock_id, "content_sha256": artifact_lock_content_sha256(lock)}), indent=2, sort_keys=True))
        return 0
    if args.command == "validate-lock":
        study = load_study_prescription(args.study)
        audit_paths = _parse_key_path(args.audit_manifest, option="--audit-manifest")
        load_study_contract_artifacts(
            study=study,
            prepared_root=args.prepared_root,
            split_registry_path=args.split_registry,
            validation_manifest_path=args.validation_manifest,
            test_manifest_path=args.test_manifest,
            artifact_lock_path=args.artifact_lock,
            scaler_path=args.scaler,
            noisy_eval_artifact_path=args.noisy_eval_artifact,
            audit_manifest_paths=audit_paths,
        )
        print(json.dumps(json_ready({"status": "ok", "artifact_lock": str(args.artifact_lock)}), indent=2, sort_keys=True))
        return 0
    if args.command == "audit-study":
        payload = {"studies": [], "total_run_count": 0, "all_evaluate_test_false": True}
        for path in args.study:
            study = load_study_prescription(path)
            rows = expand_study_run_plan(study)
            all_false = True
            for row in rows:
                cfg = resolve_study_experiment_config(study, experiment_id=row.experiment_id, run_id=row.run_id)
                all_false = all_false and (cfg.get("evaluate_test") is False)
            payload["studies"].append({"study_id": study["study_id"], "run_count": len(rows), "all_evaluate_test_false": all_false})
            payload["total_run_count"] += len(rows)
            payload["all_evaluate_test_false"] = payload["all_evaluate_test_false"] and all_false
        print(json.dumps(json_ready(payload), indent=2, sort_keys=True))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
