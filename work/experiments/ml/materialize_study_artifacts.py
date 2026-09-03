from __future__ import annotations

import argparse
import json
from pathlib import Path

from dluxshera.datasets.schema import json_ready
from dluxshera.ml import (
    PairPolicy,
    generate_frozen_pair_manifest,
    load_study_contract_artifacts,
    load_study_prescription,
    validate_evaluation_artifact_against_recipe,
    write_pair_manifest,
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
    )
    outdir = output_root / str(study["study_id"]) / f"{artifact_key}_pairs" / str(recipe["artifact_id"])
    write_pair_manifest(outdir, manifest, overwrite=overwrite)
    return {"artifact_key": artifact_key, "path": str(outdir), "summary": manifest.summary()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize tracked ML study frozen artifacts.")
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--split-registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        choices=("validation", "test", "all"),
        default="all",
        help="Which frozen artifact recipe to materialize.",
    )
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args(argv)

    study = load_study_prescription(args.study)
    loaded = load_study_contract_artifacts(
        study=study,
        prepared_root=args.prepared_root,
        split_registry_path=args.split_registry,
    )
    artifacts = dict(study["evaluation_artifacts"])
    selected = artifacts if args.artifact == "all" else {args.artifact: artifacts[args.artifact]}
    summaries = [
        _materialize_artifact(
            artifact_key=key,
            recipe=dict(recipe),
            study=study,
            catalog=loaded["catalog"],
            split_registry=loaded["split_registry"],
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
        for key, recipe in selected.items()
    ]
    print(json.dumps(json_ready({"artifacts": summaries}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
