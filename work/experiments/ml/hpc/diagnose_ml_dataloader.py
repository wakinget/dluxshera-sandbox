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


class EquivalenceMismatch(AssertionError):
    """Raised when the batched path does not exactly reproduce the reference."""


def compare_noise_implementations(
    dataset,
    *,
    indices: list[int],
    batch_size: int,
) -> dict:
    """Compare the batched path against the reference on exactly ``indices``.

    The comparison is deliberately count-driven rather than ``zip``-driven.  A
    ``zip`` of a truncated optimized batch against the full reference list
    silently compares only the overlapping prefix and reports success, which
    would let a dropped or reordered record pass as "exact".  Here every
    requested index must be returned exactly once, and the per-index pairing is
    established by position within the requested batch before any field is
    compared.

    Parameters
    ----------
    dataset:
        Dataset exposing ``__getitem__`` (reference) and ``__getitems__``
        (optimized).  Its ``noise_implementation`` is set to ``"batched"`` for
        the optimized fetch and restored afterwards.
    indices:
        Ordered dataset indices to compare.  Must be unique.
    batch_size:
        Partition width.  The final batch may be partial; that is the point of
        varying this value.

    Returns
    -------
    dict
        Report with expected/compared counts, batch partitioning, and explicit
        missing / duplicated / mismatched record lists.

    Raises
    ------
    EquivalenceMismatch
        On any count, coverage, field-set, or value discrepancy.
    """
    import torch

    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1.")
    requested = [int(index) for index in indices]
    if len(set(requested)) != len(requested):
        raise ValueError("indices must be unique; duplicate requests cannot be verified.")

    batches = [
        requested[start : start + int(batch_size)]
        for start in range(0, len(requested), int(batch_size))
    ]
    compared = 0
    seen: dict[int, int] = {}
    mismatched: list[dict] = []
    original = getattr(dataset, "noise_implementation", "reference")
    try:
        for batch_indices in batches:
            expected = [dataset[index] for index in batch_indices]
            dataset.noise_implementation = "batched"
            actual = dataset.__getitems__(list(batch_indices))
            dataset.noise_implementation = original
            # Count first: a short or over-long optimized batch is a failure in
            # its own right, independent of any field value.
            if not isinstance(actual, list):
                raise EquivalenceMismatch(
                    f"__getitems__ returned {type(actual).__name__}, expected list, "
                    f"for indices {batch_indices[0]}..{batch_indices[-1]}."
                )
            if len(actual) != len(batch_indices):
                raise EquivalenceMismatch(
                    f"__getitems__ returned {len(actual)} record(s) for "
                    f"{len(batch_indices)} requested indices "
                    f"({batch_indices[0]}..{batch_indices[-1]}); a truncated or "
                    "padded optimized batch is never equivalent."
                )
            for index, actual_item, expected_item in zip(batch_indices, actual, expected):
                seen[index] = seen.get(index, 0) + 1
                record_id = expected_item.get("pair_record_id")
                if set(actual_item.keys()) != set(expected_item.keys()):
                    raise EquivalenceMismatch(
                        f"record {record_id!r} (index {index}) field sets differ: "
                        f"optimized-only={sorted(set(actual_item) - set(expected_item))}, "
                        f"reference-only={sorted(set(expected_item) - set(actual_item))}."
                    )
                if actual_item.get("pair_record_id") != record_id:
                    raise EquivalenceMismatch(
                        f"index {index} identity differs: optimized returned "
                        f"{actual_item.get('pair_record_id')!r}, reference {record_id!r}."
                    )
                for key in expected_item:
                    lhs, rhs = actual_item[key], expected_item[key]
                    if isinstance(rhs, torch.Tensor):
                        equal = (
                            isinstance(lhs, torch.Tensor)
                            and lhs.shape == rhs.shape
                            and lhs.dtype == rhs.dtype
                            and torch.equal(lhs, rhs)
                        )
                    else:
                        equal = lhs == rhs
                    if not equal:
                        mismatched.append(
                            {"index": index, "pair_record_id": record_id, "field": key}
                        )
                compared += 1
    finally:
        dataset.noise_implementation = original

    missing = sorted(set(requested) - set(seen))
    duplicated = sorted(index for index, count in seen.items() if count > 1)
    report = {
        "expected_records": len(requested),
        "compared_records": compared,
        "batch_size": int(batch_size),
        "batches": len(batches),
        "final_batch_records": len(batches[-1]) if batches else 0,
        "final_batch_is_partial": bool(batches) and len(batches[-1]) != int(batch_size),
        "comparison_implementation": "batched vs reference (in-process, single worker)",
        "missing_indices": missing,
        "duplicated_indices": duplicated,
        "mismatched_fields": mismatched[:32],
        "mismatched_field_count": len(mismatched),
        "exact": not (missing or duplicated or mismatched) and compared == len(requested),
    }
    if not report["exact"]:
        raise EquivalenceMismatch(
            f"Batched/reference comparison is not exact: {json.dumps(report)}"
        )
    return report


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
    noise_implementation: str = "reference"
    clean_baseline: bool = False
    profile: bool = False
    equivalence_full_manifest: bool = False
    equivalence_batch_sizes: tuple[int, ...] = ()
    numpy_shared_rng_ceiling: bool = False

    def resolved_noise_implementation(self) -> str:
        """Return the strategy the selected loader will actually use.

        The validation loader exercises the frozen noisy-validation recipe,
        which is JAX-backed, so it mirrors ``training.py``'s separation rather
        than benchmarking a configuration production will never run.
        """
        from dluxshera.ml.torch_data import resolve_validation_noise_implementation

        if self.loader == "train":
            return str(self.noise_implementation)
        return resolve_validation_noise_implementation(self.noise_implementation)

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

        dataset.noise_implementation = self.resolved_noise_implementation()
        if self.clean_baseline:
            dataset.noise_config = NoiseConfig()
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
        from dluxshera.ml.noise import effective_noise_implementation, numpy_noise_provenance

        built = self._build()
        catalog = built["catalog"]
        return {
            "study_id": built["study_id"],
            "experiment_id": str(built["config"].get("experiment_id")),
            "run_id": str(built["config"].get("run_id")),
            "loader": str(self.loader),
            "noise_implementation": built["dataset"].noise_implementation,
            "requested_noise_implementation": self.noise_implementation,
            "effective_noise_implementation": effective_noise_implementation(
                built["active_noise"], built["dataset"].noise_implementation
            ),
            "numpy_noise_rng": numpy_noise_provenance(),
            "clean_baseline": self.clean_baseline,
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
        result = smoke_test_dataloader(
            built["loader"],
            batches=int(self.batches),
            timeout_seconds=float("inf"),  # the supervisor owns the wall clock
            label=str(self.loader),
            warmup_batches=self.warmup_batches,
        )

        if self.profile:
            import cProfile
            import pstats
            import io
            import time
            import resource
            import jax
            from torch.utils.data import default_collate
            dataset = built["dataset"]
            indices = list(range(min(len(dataset), built["batch_size"])))
            started = time.perf_counter()
            default_collate(dataset.__getitems__(indices))
            result["profile_first_inprocess_batch_seconds"] = time.perf_counter() - started
            profiler = cProfile.Profile()
            profiler.enable()
            for _ in range(4):
                default_collate(dataset.__getitems__(indices))
            profiler.disable()
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).sort_stats("cumtime").print_stats(35)
            result["profile_warm_four_batches"] = stream.getvalue()
            result["peak_rss_platform_units_self"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result["jax_backend"] = jax.default_backend()
            from dluxshera.ml.noise import (
                apply_pair_noise,
                apply_pair_noise_batch,
                apply_pair_noise_numpy,
                numpy_noise_provenance,
            )
            records = [dataset._record_and_offset(i) for i in indices]
            reader = dataset._get_reader()
            pairs = [(reader.get(int(r.sample_a_index)), reader.get(int(r.sample_b_index)))
                     for r, _ in records]
            seeds = [(r.pair_record_id, offset) for r, offset in records]
            draws = {
                "reference": lambda: [
                    apply_pair_noise(a, b, dataset.noise_config, pair_record_id=r,
                                     dynamic_seed_offset=o)
                    for (a, b), (r, o) in zip(pairs, seeds)
                ],
                "batched": lambda: apply_pair_noise_batch(pairs, dataset.noise_config, seeds),
                "numpy": lambda: apply_pair_noise_numpy(pairs, dataset.noise_config, seeds),
            }
            if self.numpy_shared_rng_ceiling:
                # NON-PRODUCTION reference point: one RNG over the stacked
                # batch. Realization depends on batch composition and ordering,
                # so this bounds achievable speed; it is not a candidate.
                import numpy as np
                stack = np.stack([np.maximum(np.asarray(b, dtype=np.float32), 0.0)
                                  for _, b in pairs])
                draws["numpy_shared_rng_ceiling_NONPRODUCTION"] = lambda: (
                    np.random.Generator(np.random.PCG64(0)).poisson(stack).astype(np.float32)
                )
            resident = {}
            for strategy, draw in draws.items():
                started = time.perf_counter()
                draw()
                first = time.perf_counter() - started
                started = time.perf_counter()
                for _ in range(8):
                    draw()
                warm = (time.perf_counter() - started) / 8
                resident[strategy] = {
                    "first_call_seconds": first,
                    "warm_batch_seconds": warm,
                    "images_per_second": (len(indices) / warm) if warm > 0 else None,
                    "production_candidate": not strategy.endswith("NONPRODUCTION"),
                }
            result["resident_noise_one_view"] = resident
            result["numpy_noise_rng"] = numpy_noise_provenance()
        return result

    def verify_equivalence(self):
        """Compare every output field; never regenerate a manifest or recipe.

        This is an **in-process** equivalence check: the reference and batched
        paths run in the same process on identical records.  It deliberately
        says nothing about multiprocessing behaviour; forkserver/spawn
        equivalence is covered by ``tests/ml/test_batched_noise.py``.

        ``--equivalence-full-manifest`` compares every record in the dataset
        rather than ``--batches`` worth, which is what certifies the complete
        frozen 1,024-record S12 validation manifest.  Each configured batch size
        is compared independently so a partition-dependent result cannot hide.

        This is a *bitwise* comparison, and only ``batched`` is bitwise
        identical to the reference.  ``--noise-implementation numpy`` is
        rejected here rather than reported as a mismatch: NumPy is a different
        RNG stream by design, so "not exact" would be the expected result and
        would say nothing about correctness.  Use the statistical tests in
        ``tests/ml/test_numpy_noise.py`` for the NumPy pathway instead.
        """
        if self.noise_implementation == "numpy":
            raise ValueError(
                "--verify-equivalence certifies bitwise equality against the "
                "reference JAX sampler, which only 'batched' provides. The "
                "NumPy strategy samples the same photon-noise model from a "
                "different RNG stream and is validated statistically, not "
                "bitwise. Re-run with --noise-implementation batched."
            )
        built = self._build()
        dataset = built["dataset"]
        available = len(dataset)
        requested_span = (
            available
            if self.equivalence_full_manifest
            else min(available, int(self.batches) * built["batch_size"])
        )
        indices = list(range(requested_span))
        sizes = [built["batch_size"]]
        for extra in self.equivalence_batch_sizes:
            if int(extra) not in sizes:
                sizes.append(int(extra))
        partitions = [
            compare_noise_implementations(dataset, indices=indices, batch_size=size)
            for size in sizes
        ]
        return {
            "exact": all(part["exact"] for part in partitions),
            "records": requested_span,
            "dataset_records_available": available,
            "coverage": (
                "complete manifest"
                if requested_span == available
                else f"partial sample of {available} available records"
            ),
            "complete_manifest_compared": requested_span == available,
            "loader": self.loader,
            "scope": "in-process; not a multiprocessing equivalence test",
            "partitions": partitions,
        }

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
    parser.add_argument(
        "--noise-implementation",
        choices=("reference", "batched", "numpy"),
        default="reference",
        help="Noise execution strategy to exercise. reference and batched are "
        "bitwise identical; numpy samples the same photon-noise model from "
        "deterministic per-record PCG64 generators.",
    )
    parser.add_argument(
        "--numpy-shared-rng-ceiling",
        action="store_true",
        help="Additionally microbenchmark one shared NumPy RNG over the whole "
        "image stack. NON-PRODUCTION: its realization depends on batch "
        "composition and ordering, so it is a performance ceiling only.",
    )
    parser.add_argument("--clean-baseline", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--verify-equivalence", action="store_true")
    parser.add_argument(
        "--equivalence-full-manifest",
        action="store_true",
        help="Compare every record in the dataset instead of --batches worth. "
        "Use this to certify the complete frozen validation manifest.",
    )
    parser.add_argument(
        "--equivalence-batch-size",
        type=int,
        action="append",
        default=[],
        dest="equivalence_batch_sizes",
        help="Additional partition width to repeat the exact comparison with, "
        "for example 17 to exercise a partial final batch. Repeatable.",
    )
    args = parser.parse_args(argv)

    if args.allow_unsafe_fork_deadlock and args.multiprocessing_context != "fork":
        parser.error(
            "--i-understand-this-will-deadlock-reproduce-fork-hang only applies "
            "with --multiprocessing-context fork."
        )

    from dluxshera.ml.training import run_supervised

    request = ProbeRequest(
        noise_implementation=args.noise_implementation,
        clean_baseline=args.clean_baseline,
        profile=args.profile,
        numpy_shared_rng_ceiling=args.numpy_shared_rng_ceiling,
        equivalence_full_manifest=args.equivalence_full_manifest,
        equivalence_batch_sizes=tuple(int(size) for size in args.equivalence_batch_sizes),
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
            # Recorded so a qualification gate can reject a stale report left
            # in a reused directory by an earlier source revision. Taken from
            # the environment the operator exports, not from `git rev-parse`,
            # because the runbook deploys a detached worktree.
            "source_commit": os.environ.get("DLUXSHERA_SOURCE_COMMIT")
            or os.environ.get("ML_SOURCE_COMMIT"),
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
    # Carry the supervised status into the identity block. The failure branch
    # above reports a "status", so omitting it on success would make
    # `identity.status` present only on failure -- and any gate checking
    # `identity.status == "ok"` would reject every healthy report.
    summary["identity"] = {"status": "ok", **describe["result"]}

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

    if args.verify_equivalence:
        equivalence = run_supervised(
            request.verify_equivalence,
            timeout_seconds=float(args.timeout_seconds),
            label="equivalence",
        )
        summary["equivalence"] = equivalence
        # A supervised child that reported "ok" without an exact result is
        # still a failure; never let a non-exact comparison exit zero.
        if equivalence.get("status") != "ok" or not equivalence.get("result", {}).get("exact"):
            exit_code = 1

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
