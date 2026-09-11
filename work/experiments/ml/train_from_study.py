from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from dluxshera.datasets.schema import json_ready
from dluxshera.ml import (
    load_study_contract_artifacts,
    load_study_prescription,
    resolve_study_experiment_config,
)

COMPACT_OUTPUTS = (
    "run_manifest.json",
    "run_config_resolved.json",
    "history.csv",
    "metrics.json",
    "evaluation_predictions.npz",
    "checkpoint_best.pt",
)


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


def _parse_key_path_json(value: str | None, *, option: str) -> dict[str, Path]:
    if value in (None, ""):
        return {}
    payload = json.loads(str(value))
    if not isinstance(payload, dict):
        raise ValueError(f"{option} expects a JSON object mapping artifact keys to paths.")
    return {str(key): Path(str(path)) for key, path in payload.items()}


def _copy_compact_outputs(run_dir: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in COMPACT_OUTPUTS:
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, destination / name)
            copied.append(name)
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve a tracked ML study experiment and train it.")
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--experiment-id", default="S01-E01")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--split-registry", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--audit-manifest", action="append", default=[])
    parser.add_argument("--audit-manifest-json", default=None)
    parser.add_argument("--scaler", type=Path, default=None)
    parser.add_argument("--artifact-lock", type=Path, default=None)
    parser.add_argument("--eigenbasis-artifact", type=Path, default=None)
    parser.add_argument("--noisy-eval-artifact", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--copy-final-to", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args(argv)

    study = load_study_prescription(args.study)
    config = resolve_study_experiment_config(
        study,
        experiment_id=args.experiment_id,
        run_id=args.run_id,
        device=args.device,
    )
    if config.get("require_frozen_validation_manifest") and args.validation_manifest is None:
        raise ValueError(
            f"{args.experiment_id} requires --validation-manifest; production S01 runs "
            "must not auto-generate validation pairs."
        )
    audit_manifest_paths = _parse_key_path_json(
        args.audit_manifest_json,
        option="--audit-manifest-json",
    )
    audit_manifest_paths.update(_parse_key_path(args.audit_manifest, option="--audit-manifest"))
    if args.resume_checkpoint is not None:
        config["resume_checkpoint"] = str(args.resume_checkpoint)
    load_study_contract_artifacts(
        study=study,
        prepared_root=args.prepared_root,
        split_registry_path=args.split_registry,
        validation_manifest_path=args.validation_manifest,
        test_manifest_path=args.test_manifest,
        artifact_lock_path=args.artifact_lock,
        scaler_path=args.scaler,
        noisy_eval_artifact_path=args.noisy_eval_artifact,
        audit_manifest_paths=audit_manifest_paths,
        experiment_id=args.experiment_id,
        config=config,
    )

    from dluxshera.ml.training import train_pairwise_correction

    summary = train_pairwise_correction(
        config=config,
        prepared_root=args.prepared_root,
        split_registry_path=args.split_registry,
        output_dir=args.output_dir,
        validation_manifest_path=args.validation_manifest,
        test_manifest_path=args.test_manifest,
        scaler_path=args.scaler,
        eigenbasis_path=args.eigenbasis_artifact,
        noisy_eval_artifact_path=args.noisy_eval_artifact,
        overwrite=args.overwrite,
    )
    if args.copy_final_to is not None:
        summary["copied_outputs"] = _copy_compact_outputs(Path(summary["output_dir"]), args.copy_final_to)
        summary["copy_final_to"] = str(args.copy_final_to)
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
