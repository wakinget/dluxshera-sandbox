from __future__ import annotations

import argparse
import json
import os
import platform
import socket
from pathlib import Path

from dluxshera.datasets.schema import json_ready
from dluxshera.ml import (
    load_study_contract_artifacts,
    load_study_prescription,
    resolve_study_experiment_config,
    split_registry_content_sha256,
)


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
        raise RuntimeError(f"Resolved device is {resolved}; S01 GPU preflight requires CUDA.")
    out["cuda_device_name"] = str(torch.cuda.get_device_name(resolved.index or 0))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight S01-E01 CUDA and frozen artifacts.")
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--experiment-id", default="S01-E01")
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--split-registry", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    study = load_study_prescription(args.study)
    config = resolve_study_experiment_config(study, experiment_id=args.experiment_id, device=args.device)
    device = str(config.get("device", "cuda:0"))
    torch_info = _torch_summary(device)

    loaded = load_study_contract_artifacts(
        study=study,
        prepared_root=args.prepared_root,
        split_registry_path=args.split_registry,
        validation_manifest_path=args.validation_manifest,
        test_manifest_path=args.test_manifest,
        experiment_id=args.experiment_id,
        config=config,
    )
    catalog = loaded["catalog"]
    split_registry = loaded["split_registry"]
    validation_manifest = loaded["validation_manifest"]
    test_manifest = loaded["test_manifest"]
    summary = {
        "torch": torch_info,
        "slurm": {
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "SLURM_CLUSTER_NAME": os.environ.get("SLURM_CLUSTER_NAME"),
            "SLURM_JOB_PARTITION": os.environ.get("SLURM_JOB_PARTITION"),
            "SLURMD_NODENAME": os.environ.get("SLURMD_NODENAME"),
            "hostname": socket.gethostname(),
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
    }
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
