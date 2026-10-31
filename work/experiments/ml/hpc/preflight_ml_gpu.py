#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
from pathlib import Path

from dluxshera.datasets.schema import json_ready
from dluxshera.ml import (
    PairPolicy,
    PairSampler,
    build_science_mode_weights,
    load_artifact_lock,
    load_intensity_scaler,
    load_science_eigenbasis,
    validate_noisy_eval_artifact_for_study,
    load_study_contract_artifacts,
    load_study_prescription,
    persist_study_contract_artifacts,
    resolve_study_experiment_config,
    split_registry_content_sha256,
    validate_science_eigenbasis_expectations,
    validate_science_mode_weights_nonuniform,
)
from dluxshera.ml.scaling import intensity_scaler_content_sha256


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


def _torch_summary(device: str) -> dict:
    import torch
    from dluxshera.ml.training import resolve_device

    resolved = resolve_device(device)
    out = {
        "python_version": platform.python_version(),
        "torch_version": str(getattr(torch, "__version__", "")),
        "torch_cuda_version": None
        if getattr(torch.version, "cuda", None) is None
        else str(getattr(torch.version, "cuda", None)),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "resolved_training_device": str(resolved),
        "cuda_device_name": None,
    }
    if resolved.type != "cuda":
        raise RuntimeError(f"Resolved device is {resolved}; ML GPU preflight requires CUDA.")
    out["cuda_device_name"] = str(torch.cuda.get_device_name(resolved.index or 0))
    return out


def _dataloader_smoke_in_process(
    *,
    config: dict,
    catalog,
    split_registry,
    pair_policy,
    scaler_path: Path | None,
    enabled: bool | None,
    batches: int,
    timeout_seconds: float,
) -> dict:
    """Prove the configured training data pipeline can actually produce batches.

    CUDA availability and valid artifact hashes do not establish pipeline
    readiness.  The six timed-out S12 runs passed every artifact check and then
    stalled before epoch 0 because JAX-backed noise deadlocked in forked
    DataLoader workers.  This check exercises the real dataset and noise path
    under the real worker configuration, bounded by a timeout.

    Returns a diagnostics mapping.  Raises on failure so a batch job stops
    before entering a multi-day training allocation.
    """
    from torch.utils.data import DataLoader

    from dluxshera.ml import NoiseConfig
    from dluxshera.ml.losses import PairConsistencyConfig
    from dluxshera.ml.scaling import load_intensity_scaler
    from dluxshera.ml.torch_data import (
        DynamicPairDataset,
        noise_requires_safe_workers,
        resolve_dataloader_context,
    )
    from dluxshera.ml.training import (
        build_bounded_dataloader_kwargs,
        smoke_test_dataloader,
    )

    noise_config = NoiseConfig.from_dict(config.get("noise"))
    validation_noise_config = NoiseConfig.from_dict(config.get("validation_noise"))
    augmentation_enabled = bool(noise_config.enabled or validation_noise_config.enabled)
    should_run = augmentation_enabled if enabled is None else bool(enabled)
    if not should_run:
        return {
            "status": "skipped",
            "reason": "noise augmentation disabled and smoke test not requested"
            if enabled is None
            else "explicitly disabled with --no-dataloader-smoke",
            "noise_enabled": augmentation_enabled,
        }

    training_cfg = dict(config.get("training", {}))
    batch_size = int(training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 0))
    consistency_config = PairConsistencyConfig.from_dict(config.get("pair_consistency"))
    requires_safe_workers = noise_requires_safe_workers(
        (noise_config, validation_noise_config)
    )
    context, context_name, context_reason = resolve_dataloader_context(
        training_cfg.get("dataloader_multiprocessing_context"),
        requires_safe_workers=requires_safe_workers,
        num_workers=num_workers,
    )
    dataset = DynamicPairDataset(
        catalog=catalog,
        sampler=PairSampler(catalog, split_registry, pair_policy),
        pairs_per_epoch=max(batch_size * max(int(batches), 1), 2),
        seed=int(config.get("seed", 0)),
        science_split="train",
        nuisance_split="train",
        scaler=None if scaler_path is None else load_intensity_scaler(scaler_path),
        noise_config=noise_config,
        second_noise_view=consistency_config.noise_consistency_weight > 0.0,
        shard_cache_size=int(training_cfg.get("shard_cache_size", 4)),
    )
    dataset.set_epoch(0)
    loader_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": False,
        # Bound every fetch, including the first, so a wedged worker raises
        # here instead of stalling the batch job.
        **build_bounded_dataloader_kwargs(
            num_workers=num_workers,
            per_batch_timeout_seconds=float(timeout_seconds),
        ),
    }
    if context is not None:
        loader_kwargs["multiprocessing_context"] = context
    result = smoke_test_dataloader(
        DataLoader(dataset, **loader_kwargs),
        batches=int(batches),
        timeout_seconds=float(timeout_seconds),
        label="preflight_train",
    )
    return {
        "status": "ok",
        "noise_enabled": augmentation_enabled,
        "requires_safe_workers": bool(requires_safe_workers),
        "multiprocessing_method": context_name,
        "multiprocessing_method_reason": context_reason,
        "purpose": (
            "proves the pipeline can produce batches; not a capacity "
            "measurement. Use diagnose_ml_dataloader.py for sustained throughput."
        ),
        **result,
    }


def _dataloader_smoke(**kwargs) -> dict:
    """Bound the entire loader check, including zero-worker fetches and teardown."""
    from functools import partial
    from dluxshera.ml.training import run_supervised

    budget = float(kwargs["timeout_seconds"])
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("timeout_seconds must be finite and > 0")
    outcome = run_supervised(
        partial(_dataloader_smoke_in_process, **kwargs),
        timeout_seconds=budget,
        label="preflight_train",
    )
    if outcome["status"] != "ok":
        raise RuntimeError(f"Data-loader preflight failed: {outcome}")
    return {**outcome["result"], "deadline_kind": "process", "deadline_seconds": budget}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight CUDA and frozen ML study artifacts.")
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--split-registry", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, default=None)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--audit-manifest", action="append", default=[])
    parser.add_argument("--audit-manifest-json", default=None)
    parser.add_argument("--artifact-lock", type=Path, default=None)
    parser.add_argument("--eigenbasis-artifact", type=Path, default=None)
    parser.add_argument("--noisy-eval-artifact", type=Path, default=None)
    parser.add_argument("--persist-artifact-root", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dataloader-smoke",
        dest="dataloader_smoke",
        action="store_true",
        default=None,
        help="Force the bounded data-loader smoke test on. Defaults to on when "
        "noise augmentation is enabled.",
    )
    parser.add_argument(
        "--no-dataloader-smoke",
        dest="dataloader_smoke",
        action="store_false",
        help="Skip the bounded data-loader smoke test.",
    )
    parser.add_argument("--dataloader-smoke-batches", type=int, default=2)
    parser.add_argument("--dataloader-smoke-timeout", type=float, default=300.0)
    args = parser.parse_args(argv)

    study = load_study_prescription(args.study)
    config = resolve_study_experiment_config(
        study,
        experiment_id=args.experiment_id,
        run_id=args.run_id,
        device=args.device,
    )
    device = str(config.get("device", "cuda:0"))
    torch_info = _torch_summary(device)
    audit_manifest_paths = _parse_key_path_json(
        args.audit_manifest_json,
        option="--audit-manifest-json",
    )
    audit_manifest_paths.update(_parse_key_path(args.audit_manifest, option="--audit-manifest"))

    loaded = load_study_contract_artifacts(
        study=study,
        prepared_root=args.prepared_root,
        split_registry_path=args.split_registry,
        scaler_path=args.scaler,
        validation_manifest_path=args.validation_manifest,
        test_manifest_path=args.test_manifest,
        artifact_lock_path=args.artifact_lock,
        noisy_eval_artifact_path=args.noisy_eval_artifact,
        experiment_id=args.experiment_id,
        config=config,
        audit_manifest_paths=audit_manifest_paths,
    )
    catalog = loaded["catalog"]
    split_registry = loaded["split_registry"]
    validation_manifest = loaded["validation_manifest"]
    test_manifest = loaded["test_manifest"]
    audit_manifests = loaded["audit_manifests"]
    pair_policy = PairPolicy.from_dict(config.get("pair_policy", {}))
    noisy_eval_identity = validate_noisy_eval_artifact_for_study(
        study=study,
        config=config,
        catalog=catalog,
        split_registry=split_registry,
        validation_manifest=validation_manifest,
        noisy_eval_artifact_path=args.noisy_eval_artifact,
    )
    eigenbasis_identity = None
    science_loss_cfg = dict(config.get("science_loss", {}) or {})
    if str(science_loss_cfg.get("mode", "ordinary")) != "ordinary":
        if args.eigenbasis_artifact is None:
            raise ValueError("Weighted science_loss preflight requires --eigenbasis-artifact.")
        eigenbasis = load_science_eigenbasis(args.eigenbasis_artifact, catalog=catalog)
        aux = config.get("auxiliary_artifacts", {})
        expected_aux = (
            dict(aux.get("science_eigenbasis", {}) or {})
            if isinstance(aux, dict)
            else {}
        )
        expected = science_loss_cfg.get("eigenbasis", {})
        if isinstance(expected, dict):
            validate_science_eigenbasis_expectations(
                eigenbasis,
                expected={**expected_aux, **dict(expected)},
            )
            expected_id = expected.get("artifact_id")
            expected_hash = expected.get("content_sha256")
            if expected_id and str(expected_id) != eigenbasis.artifact_id:
                raise ValueError(
                    f"Eigenbasis artifact_id {eigenbasis.artifact_id!r} does not match {expected_id!r}."
                )
            actual_hash = eigenbasis.content_identity.get("sha256")
            if expected_hash and str(expected_hash) != str(actual_hash):
                raise ValueError(
                    f"Eigenbasis content_sha256 {actual_hash!r} does not match {expected_hash!r}."
                )
        else:
            validate_science_eigenbasis_expectations(eigenbasis, expected=expected_aux)
        mode_weights = build_science_mode_weights(
            eigenbasis.eigenvalues,
            mode=str(science_loss_cfg["mode"]),
            strength=float(science_loss_cfg.get("strength", 0.5)),
            eigenvalue_floor=float(science_loss_cfg.get("eigenvalue_floor", 1.0e-6)),
            weight_cap=float(science_loss_cfg.get("weight_cap", 10.0)),
        )
        validate_science_mode_weights_nonuniform(mode_weights, mode=str(science_loss_cfg["mode"]))
        eigenbasis_identity = {
            "path": str(args.eigenbasis_artifact),
            "artifact_id": eigenbasis.artifact_id,
            "content_sha256": eigenbasis.content_identity.get("sha256"),
            "coordinate_convention": eigenbasis.coordinate_convention,
            "source_matrix_coordinate_space": eigenbasis.source_matrix_coordinate_space,
            "eigenbasis_coordinate_space": eigenbasis.eigenbasis_coordinate_space,
            "mode_weight_min": float(mode_weights.min()),
            "mode_weight_max": float(mode_weights.max()),
            "mode_weight_std": float(mode_weights.std()),
        }
    sampler = PairSampler(catalog, split_registry, pair_policy)
    train_eligible = sampler.eligible_indices("train", "train", policy=pair_policy)
    if train_eligible.size == 0:
        raise ValueError("Pair policy has no eligible training population.")
    distance_smoke: list[dict] = []
    bins = list(pair_policy.distance_bin_weights) or list(pair_policy.fisher_distance_bins)
    if bins:
        import numpy as np

        for label in bins:
            smoke_policy = PairPolicy.from_dict(
                {**pair_policy.to_dict(), "distance_bin_weights": {str(label): 1.0}, "fisher_distance_bins": []}
            )
            smoke_sampler = PairSampler(catalog, split_registry, smoke_policy)
            record = smoke_sampler.sample_pair(np.random.default_rng(17), epoch=0)
            distance_smoke.append(
                {
                    "distance_bin": str(label),
                    "pair_record_id": record.pair_record_id,
                    "fisher_distance_l2": record.fisher_distance_l2,
                }
            )
    scaler_identity = None
    if args.scaler is not None:
        scaler = load_intensity_scaler(args.scaler)
        scaler_identity = {
            "path": str(args.scaler),
            "content_sha256": intensity_scaler_content_sha256(scaler.to_dict()),
            "mode": scaler.mode,
            "sample_count": scaler.sample_count,
        }
    lock_identity = None
    if args.artifact_lock is not None:
        lock = load_artifact_lock(args.artifact_lock)
        lock_identity = {"path": str(args.artifact_lock), "lock_id": lock.lock_id}
    summary = {
        "identity_table": {
            "study_id": str(study["study_id"]),
            "experiment_id": str(config.get("experiment_id")),
            "run_id": str(config.get("run_id")),
            "repository_sha": os.environ.get("DLUXSHERA_SOURCE_COMMIT") or os.environ.get("ML_SOURCE_COMMIT"),
            "prepared": {"artifact_id": catalog.artifact_id, "hash": catalog.prepared_dataset_hash},
            "split_registry": {"artifact_id": split_registry.artifact_id, "hash": split_registry_content_sha256(split_registry)},
            "scaler": scaler_identity,
            "primary_validation": {"artifact_id": validation_manifest.artifact_id, "hash": validation_manifest.manifest.get("content_identity", {}).get("sha256")},
            "test": {"artifact_id": test_manifest.artifact_id, "hash": test_manifest.manifest.get("content_identity", {}).get("sha256")},
            "audit_artifacts": {
                str(key): {
                    "artifact_id": manifest.artifact_id,
                    "hash": manifest.manifest.get("content_identity", {}).get("sha256"),
                }
                for key, manifest in audit_manifests.items()
            },
            "artifact_lock": lock_identity,
            "eigenbasis": eigenbasis_identity,
            "noisy_eval": noisy_eval_identity,
            "device": torch_info["resolved_training_device"],
            "output_root": os.environ.get("ML_RUN_DIR"),
        },
        "torch": torch_info,
        "slurm": {
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "SLURM_CLUSTER_NAME": os.environ.get("SLURM_CLUSTER_NAME"),
            "SLURM_JOB_PARTITION": os.environ.get("SLURM_JOB_PARTITION"),
            "SLURMD_NODENAME": os.environ.get("SLURMD_NODENAME"),
            "hostname": socket.gethostname(),
        },
        "source": {
            "source_commit": os.environ.get("DLUXSHERA_SOURCE_COMMIT")
            or os.environ.get("ML_SOURCE_COMMIT"),
            "source_archive_id": os.environ.get("DLUXSHERA_SOURCE_ARCHIVE_ID")
            or os.environ.get("ML_SOURCE_ARCHIVE_ID"),
        },
        "prepared_dataset": {
            "path": str(args.prepared_root),
            "artifact_id": catalog.artifact_id,
            "prepared_dataset_hash": catalog.prepared_dataset_hash,
            "sample_count": catalog.sample_count,
            "science_dim": catalog.science_dim,
            "nuisance_dim": catalog.nuisance_dim,
        },
        "split_registry": {
            "path": str(args.split_registry),
            "artifact_id": split_registry.artifact_id,
            "content_sha256": split_registry_content_sha256(split_registry),
            "prepared_dataset_hash": split_registry.prepared_dataset.get("prepared_dataset_hash"),
        },
        "validation_manifest": {
            "path": str(args.validation_manifest),
            "identity": validation_manifest.manifest.get("content_identity"),
            "summary": validation_manifest.summary(),
        },
        "test_manifest": {
            "path": str(args.test_manifest),
            "identity": test_manifest.manifest.get("content_identity"),
            "summary": test_manifest.summary(),
        },
        "audit_manifests": {
            str(key): {
                "path": str(audit_manifest_paths[str(key)]),
                "identity": manifest.manifest.get("content_identity"),
                "summary": manifest.summary(),
            }
            for key, manifest in audit_manifests.items()
        },
        "pair_policy_smoke": {
            "train_eligible_count": int(train_eligible.size),
            "distance_bin_smoke": distance_smoke,
        },
    }
    summary["dataloader_smoke"] = _dataloader_smoke(
        config=config,
        catalog=catalog,
        split_registry=split_registry,
        pair_policy=pair_policy,
        scaler_path=args.scaler,
        enabled=args.dataloader_smoke,
        batches=int(args.dataloader_smoke_batches),
        timeout_seconds=float(args.dataloader_smoke_timeout),
    )
    if args.persist_artifact_root is not None:
        summary["persistent_artifacts"] = persist_study_contract_artifacts(
            artifact_root=args.persist_artifact_root,
            split_registry_path=args.split_registry,
            validation_manifest_path=args.validation_manifest,
            test_manifest_path=args.test_manifest,
        )
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
