from __future__ import annotations

import argparse
import json
from pathlib import Path

from dluxshera.datasets.schema import json_ready
from dluxshera.ml import (
    build_artifact_lock,
    artifact_lock_content_sha256,
    expand_study_run_plan,
    fit_intensity_scaler,
    generate_role_preserving_split_registry,
    generate_split_registry,
    PairPolicy,
    generate_frozen_pair_manifest,
    load_intensity_scaler,
    load_pair_manifest,
    load_sample_catalog,
    load_split_registry,
    load_study_contract_artifacts,
    load_study_prescription,
    resolve_study_artifact_contract,
    validate_evaluation_artifact_against_recipe,
    write_artifact_lock,
    write_intensity_scaler,
    write_pair_manifest,
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
                from dluxshera.ml import resolve_study_experiment_config

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
