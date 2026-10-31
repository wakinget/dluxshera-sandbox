#!/usr/bin/env python3
"""Diagnose the ML training data pipeline without running full training.

This exercises the real dataset, noise path, and ``DataLoader`` worker
configuration for a tracked study experiment under process-level supervision,
so a pipeline that cannot produce batches fails in minutes instead of consuming
a multi-day GPU allocation.

Probes run in a supervised child process that is killed, along with its worker
processes, when it exceeds its timeout.  That matters because a deadlocked
DataLoader worker cannot be interrupted from inside the process that owns it.

Use it to distinguish two failure modes that look identical from the outside:

- A worker start-method deadlock.  No batch ever arrives and the worker
  processes sit at ~0% CPU.  This is what JAX-backed noise does in ``fork``
  started workers.
- Slow but healthy loading.  Batches arrive steadily but too slowly for the
  configured epoch budget, and workers are busy.

``--loader-only`` needs no GPU.  The locked test manifest is never read, and no
run directory is created or modified.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path

from dluxshera.datasets.schema import json_ready


def _parse_key_path_json(value: str | None, *, option: str) -> dict[str, Path]:
    if value in (None, ""):
        return {}
    payload = json.loads(str(value))
    if not isinstance(payload, dict):
        raise ValueError(f"{option} expects a JSON object mapping artifact keys to paths.")
    return {str(key): Path(str(path)) for key, path in payload.items()}


@dataclass(frozen=True)
class ProbeRequest:
    """Picklable description of one supervised probe.

    This is a plain dataclass of paths and scalars so it can cross a ``spawn``
    process boundary.  Catalogs, datasets, and loaders are built inside the
    child, which also keeps JAX out of this supervising process.
    """

    study: Path
    experiment_id: str
    run_id: str | None
    prepared_root: Path
    split_registry: Path
    scaler: Path | None
    validation_manifest: Path
    artifact_lock: Path | None
    noisy_eval_artifact: Path | None
    audit_manifest_json: str | None
    loader: str
    batches: int
    warmup_batches: int | None
    num_workers: int | None
    multiprocessing_context: str | None
    batch_size: int | None
    epoch: int
    per_batch_timeout_seconds: float
    allow_unsafe_fork_deadlock: bool
    with_training_step: bool
    device: str | None

    def _build(self) -> dict:
        """Build the catalog, dataset, and loader inside the child process."""
        from dluxshera.ml import (
            NoiseConfig,
            PairPolicy,
            PairSampler,
            load_study_contract_artifacts,
            load_study_prescription,
            resolve_study_experiment_config,
        )
        from dluxshera.ml.losses import PairConsistencyConfig
        from dluxshera.ml.scaling import load_intensity_scaler
        from dluxshera.ml.torch_data import (
            DynamicPairDataset,
            PairManifestDataset,
            noise_requires_safe_workers,
            resolve_dataloader_context,
        )
        from dluxshera.ml.training import build_bounded_dataloader_kwargs
        from torch.utils.data import DataLoader

        study = load_study_prescription(self.study)
        config = resolve_study_experiment_config(
            study,
            experiment_id=self.experiment_id,
            run_id=self.run_id,
            device=self.device,
        )
        loaded = load_study_contract_artifacts(
            study=study,
            prepared_root=self.prepared_root,
            split_registry_path=self.split_registry,
            scaler_path=self.scaler,
            validation_manifest_path=self.validation_manifest,
            # The locked test manifest is intentionally never loaded here.
            test_manifest_path=None,
            artifact_lock_path=self.artifact_lock,
            noisy_eval_artifact_path=self.noisy_eval_artifact,
            experiment_id=self.experiment_id,
            config=config,
            audit_manifest_paths=_parse_key_path_json(
                self.audit_manifest_json,
                option="--audit-manifest-json",
            ),
        )
        catalog = loaded["catalog"]
        split_registry = loaded["split_registry"]
        validation_manifest = loaded["validation_manifest"]

        training_cfg = dict(config.get("training", {}))
        batch_size = int(self.batch_size or training_cfg.get("batch_size", 32))
        num_workers = int(
            training_cfg.get("num_workers", 0)
            if self.num_workers is None
            else self.num_workers
        )
        shard_cache_size = int(training_cfg.get("shard_cache_size", 4))
        noise_config = NoiseConfig.from_dict(config.get("noise"))
        validation_noise_config = NoiseConfig.from_dict(config.get("validation_noise"))
        consistency_config = PairConsistencyConfig.from_dict(config.get("pair_consistency"))
        second_noise_view = consistency_config.noise_consistency_weight > 0.0
        scaler = None if self.scaler is None else load_intensity_scaler(self.scaler)

        if self.loader == "train":
            active_noise = noise_config
            dataset = DynamicPairDataset(
                catalog=catalog,
                sampler=PairSampler(
                    catalog,
                    split_registry,
                    PairPolicy.from_dict(config.get("pair_policy", {})),
                ),
                pairs_per_epoch=max(batch_size * max(int(self.batches), 1), 2),
                seed=int(config.get("seed", 0)),
                science_split="train",
                nuisance_split="train",
                scaler=scaler,
                noise_config=noise_config,
                second_noise_view=second_noise_view,
                shard_cache_size=shard_cache_size,
            )
            dataset.set_epoch(int(self.epoch))
        else:
            active_noise = validation_noise_config
            dataset = PairManifestDataset(
                catalog=catalog,
                pair_manifest=validation_manifest,
                scaler=scaler,
                noise_config=validation_noise_config,
                shard_cache_size=shard_cache_size,
            )

        requires_safe_workers = noise_requires_safe_workers(
            (noise_config, validation_noise_config)
        )
        context, context_name, context_reason = resolve_dataloader_context(
            self.multiprocessing_context,
            requires_safe_workers=requires_safe_workers,
            num_workers=num_workers,
            allow_unsafe_fork_deadlock=self.allow_unsafe_fork_deadlock,
        )
        loader_kwargs: dict = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": False,
            "persistent_workers": False,
            **build_bounded_dataloader_kwargs(
                num_workers=num_workers,
                per_batch_timeout_seconds=self.per_batch_timeout_seconds,
            ),
        }
        if context is not None:
            loader_kwargs["multiprocessing_context"] = context
        return {
            "config": config,
            "study_id": str(study["study_id"]),
            "catalog": catalog,
            "dataset": dataset,
            "loader": DataLoader(dataset, **loader_kwargs),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "active_noise": active_noise,
            "second_noise_view": second_noise_view,
            "consistency_config": consistency_config,
            "requires_safe_workers": requires_safe_workers,
            "context_name": context_name,
            "context_reason": context_reason,
        }

    def describe(self) -> dict:
        """Return static identity/configuration facts without fetching batches."""
        built = self._build()
        catalog = built["catalog"]
        return {
            "study_id": built["study_id"],
            "experiment_id": str(built["config"].get("experiment_id")),
            "run_id": str(built["config"].get("run_id")),
            "loader": str(self.loader),
            "prepared_dataset": {
                "artifact_id": catalog.artifact_id,
                "prepared_dataset_hash": catalog.prepared_dataset_hash,
                "sample_shape": list(catalog.sample_shape),
            },
            "noise": {
                "active": built["active_noise"].to_dict(),
                "second_noise_view": bool(built["second_noise_view"]),
                "pair_consistency": built["consistency_config"].to_dict(),
                "requires_safe_workers": bool(built["requires_safe_workers"]),
            },
            "dataloader": {
                "num_workers": built["num_workers"],
                "batch_size": built["batch_size"],
                "multiprocessing_method": built["context_name"],
                "multiprocessing_method_reason": built["context_reason"],
                "persistent_workers": False,
                "per_batch_timeout_seconds": built["loader"].timeout or None,
                "requested_context": self.multiprocessing_context,
                "unsafe_fork_reproduction": bool(self.allow_unsafe_fork_deadlock),
            },
        }

    def run_loader_probe(self) -> dict:
        """Drain batches and report sustained throughput. Runs in the child."""
        from dluxshera.ml.training import smoke_test_dataloader

        built = self._build()
        return smoke_test_dataloader(
            built["loader"],
            batches=int(self.batches),
            timeout_seconds=float("inf"),  # the supervisor owns the wall clock
            label=str(self.loader),
            warmup_batches=self.warmup_batches,
        )

    def run_training_step_probe(self) -> dict:
        """Run bounded train steps on real noisy batches. Runs in the child.

        For E02 (``noise_consistency_weight > 0``) this includes the actual
        second-view consistency term, so the probe exercises the objective the
        experiment will really optimise rather than a cheaper stand-in.
        """
        import time

        import torch

        from dluxshera.ml.losses import pair_consistency_losses
        from dluxshera.ml.models import build_pairwise_correction_model
        from dluxshera.ml.training import resolve_device

        built = self._build()
        config = built["config"]
        consistency_config = built["consistency_config"]
        device = resolve_device(
            str(config.get("device", "auto")) if self.device is None else self.device
        )
        model = build_pairwise_correction_model(
            built["catalog"].science_dim,
            config.get("model"),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.get("training", {}).get("learning_rate", 1.0e-4)),
        )
        criterion = torch.nn.MSELoss()
        model.train()

        step_seconds: list[float] = []
        consistency_values: list[float] = []
        exercised_second_view = False
        for index, batch in enumerate(built["loader"]):
            started = time.perf_counter()
            image_a = batch["image_a"].to(device)
            image_b = batch["image_b"].to(device)
            target = batch["target_delta_z"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(image_a, image_b)
            loss = criterion(pred, target)
            if consistency_config.noise_consistency_weight > 0.0:
                view2_a = batch.get("image_a_noise_view2")
                view2_b = batch.get("image_b_noise_view2")
                if view2_a is None or view2_b is None:
                    raise RuntimeError(
                        "noise_consistency_weight > 0 but the batch has no second "
                        "noise view; the dataset was not configured for E02."
                    )
                aux = pair_consistency_losses(
                    model,
                    image_a,
                    image_b,
                    image_a_noise_view2=view2_a.to(device),
                    image_b_noise_view2=view2_b.to(device),
                    pred_ab=pred,
                    config=consistency_config,
                )
                loss = loss + (
                    consistency_config.noise_consistency_weight * aux["noise_consistency"]
                )
                consistency_values.append(float(aux["noise_consistency"].detach().cpu()))
                exercised_second_view = True
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds.append(time.perf_counter() - started)
            if index + 1 >= int(self.batches):
                break

        warmup = 1 if len(step_seconds) > 1 else 0
        measured = step_seconds[warmup:]
        return {
            "device": str(device),
            "batches": len(step_seconds),
            "first_step_seconds": step_seconds[0] if step_seconds else None,
            "warmup_steps_discarded": warmup,
            "mean_step_seconds": (sum(measured) / len(measured)) if measured else None,
            "noise_consistency_weight": consistency_config.noise_consistency_weight,
            "exercised_second_view_objective": exercised_second_view,
            "mean_noise_consistency_loss": (
                sum(consistency_values) / len(consistency_values)
                if consistency_values
                else None
            ),
            "objective_completeness": (
                "includes second-view noise-consistency term"
                if exercised_second_view
                else "science loss only; this experiment has no consistency term"
            ),
        }


def _epoch_projection(loader_smoke: dict, *, pairs_per_epoch: int, epochs: int) -> dict:
    """Project epoch and campaign wall time from sustained throughput."""
    pairs_per_second = loader_smoke.get("sustained_pairs_per_second")
    if not pairs_per_second or pairs_per_second <= 0.0:
        return {"status": "unavailable", "reason": "no sustained throughput measured"}
    epoch_seconds = float(pairs_per_epoch) / float(pairs_per_second)
    return {
        "status": "ok"
        if loader_smoke.get("sustained_measurement_valid")
        else "not_defensible",
        "caveat": None
        if loader_smoke.get("sustained_measurement_valid")
        else loader_smoke.get("sustained_measurement_note"),
        "pairs_per_epoch": int(pairs_per_epoch),
        "sustained_pairs_per_second": float(pairs_per_second),
        "projected_epoch_minutes": epoch_seconds / 60.0,
        "projected_campaign_hours": epoch_seconds * int(epochs) / 3600.0,
        "epochs": int(epochs),
        "note": "data loading only; excludes GPU step time",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exercise the ML dataset and noise path under supervision.",
    )
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--split-registry", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, default=None)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--artifact-lock", type=Path, default=None)
    parser.add_argument("--noisy-eval-artifact", type=Path, default=None)
    parser.add_argument("--audit-manifest-json", default=None)
    parser.add_argument(
        "--loader",
        choices=("train", "validation"),
        default="train",
        help="Which dataset to exercise. 'train' uses dynamic sampling and dynamic noise.",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=24,
        help="Total batches to fetch, including warmup. Use >= 24 for capacity planning.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=None,
        help="Batches discarded before measuring. Defaults to the prefetch depth.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override the study's num_workers. Use 0 for the in-process fallback.",
    )
    parser.add_argument(
        "--multiprocessing-context",
        default=None,
        choices=("fork", "forkserver", "spawn"),
        help="Override the automatically selected worker start method.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=900.0,
        help="Wall-clock budget per probe. The child process is killed on expiry.",
    )
    parser.add_argument(
        "--per-batch-timeout-seconds",
        type=float,
        default=300.0,
        help="DataLoader per-batch timeout. Applies to every fetch, including the first.",
    )
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument(
        "--with-training-step",
        action="store_true",
        default=False,
        help="Also run bounded forward/backward passes (requires a device). For "
        "experiments with noise_consistency_weight > 0 this includes the "
        "second-view consistency term.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for --with-training-step. Defaults to the study device.",
    )
    parser.add_argument(
        "--i-understand-this-will-deadlock-reproduce-fork-hang",
        dest="allow_unsafe_fork_deadlock",
        action="store_true",
        default=False,
        help="Permit the known-deadlocking fork configuration in order to "
        "demonstrate the failure. The probe is expected to time out and be "
        "killed. Never use this for a real run.",
    )
    args = parser.parse_args(argv)

    if args.allow_unsafe_fork_deadlock and args.multiprocessing_context != "fork":
        parser.error(
            "--i-understand-this-will-deadlock-reproduce-fork-hang only applies "
            "with --multiprocessing-context fork."
        )

    from dluxshera.ml.training import run_supervised

    request = ProbeRequest(
        study=args.study,
        experiment_id=args.experiment_id,
        run_id=args.run_id,
        prepared_root=args.prepared_root,
        split_registry=args.split_registry,
        scaler=args.scaler,
        validation_manifest=args.validation_manifest,
        artifact_lock=args.artifact_lock,
        noisy_eval_artifact=args.noisy_eval_artifact,
        audit_manifest_json=args.audit_manifest_json,
        loader=args.loader,
        batches=args.batches,
        warmup_batches=args.warmup_batches,
        num_workers=args.num_workers,
        multiprocessing_context=args.multiprocessing_context,
        batch_size=args.batch_size,
        epoch=args.epoch,
        per_batch_timeout_seconds=args.per_batch_timeout_seconds,
        allow_unsafe_fork_deadlock=args.allow_unsafe_fork_deadlock,
        with_training_step=args.with_training_step,
        device=args.device,
    )

    summary: dict = {
        "host": {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurmd_nodename": os.environ.get("SLURMD_NODENAME"),
            "cpu_count": os.cpu_count(),
        },
        "supervision": {
            "timeout_seconds": float(args.timeout_seconds),
            "per_batch_timeout_seconds": float(args.per_batch_timeout_seconds),
            "mode": "probes run in a killable child process",
        },
    }

    # Identity/configuration facts are themselves gathered under supervision,
    # because artifact loading imports JAX and can be slow on a cold filesystem.
    describe = run_supervised(
        request.describe,
        timeout_seconds=float(args.timeout_seconds),
        label="describe",
    )
    if describe.get("status") != "ok":
        summary["identity"] = describe
        print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
        return 1
    summary["identity"] = describe["result"]

    loader_probe = run_supervised(
        request.run_loader_probe,
        timeout_seconds=float(args.timeout_seconds),
        label=f"loader:{args.loader}",
    )
    summary["loader_smoke"] = loader_probe
    exit_code = 0 if loader_probe.get("status") == "ok" else 1

    if loader_probe.get("status") == "ok":
        config_training = {}
        try:
            from dluxshera.ml import (
                load_study_prescription,
                resolve_study_experiment_config,
            )

            config_training = dict(
                resolve_study_experiment_config(
                    load_study_prescription(args.study),
                    experiment_id=args.experiment_id,
                    run_id=args.run_id,
                ).get("training", {})
            )
        except Exception:  # pragma: no cover - projection is advisory
            config_training = {}
        summary["capacity_projection"] = _epoch_projection(
            loader_probe["result"],
            pairs_per_epoch=int(config_training.get("pairs_per_epoch", 16384)),
            epochs=int(config_training.get("epochs", 600)),
        )

    if args.with_training_step:
        step_probe = run_supervised(
            request.run_training_step_probe,
            timeout_seconds=float(args.timeout_seconds),
            label="training_step",
        )
        summary["training_step_smoke"] = step_probe
        if step_probe.get("status") != "ok":
            exit_code = 1

    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
