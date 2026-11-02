#!/usr/bin/env python3
"""Local engineering benchmark, NOT a V4 capacity measurement.

Writes only temporary synthetic shards. Use diagnose_ml_dataloader.py for V4.
"""
import argparse
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Probe:
    implementation: str
    second: bool
    workers: int
    batches: int
    epochs: int = 1

    def run(self):
        import jax
        import numpy
        import resource
        from tests.ml.test_s12_dataloader_runtime import _dynamic_dataset
        from dluxshera.ml.noise import NoiseConfig
        from dluxshera.ml.torch_data import resolve_dataloader_context
        from dluxshera.ml.training import smoke_test_dataloader
        from torch.utils.data import DataLoader
        with tempfile.TemporaryDirectory() as root:
            ds = _dynamic_dataset(Path(root), image_size=160,
                                  second_noise_view=self.second, pairs_per_epoch=self.batches*32)
            ds.noise_implementation = self.implementation if self.implementation != "clean" else "reference"
            if self.implementation == "clean":
                ds.noise_config = NoiseConfig()
            context, _, _ = resolve_dataloader_context("forkserver", requires_safe_workers=True,
                                                       num_workers=self.workers)
            kwargs = {"multiprocessing_context": context, "timeout": 120} if self.workers else {}
            loader = DataLoader(ds, batch_size=32, num_workers=self.workers, **kwargs)
            epoch_results = []
            for epoch in range(self.epochs):
                ds.set_epoch(epoch)
                epoch_results.append(smoke_test_dataloader(
                    loader, batches=self.batches, timeout_seconds=600, warmup_batches=8))
            result = dict(epoch_results[-1])
            result["epoch_lifecycle"] = epoch_results
            result.update(implementation=self.implementation, second_view=self.second,
                          workers=self.workers, image_shape=[160,160], batch_size=32,
                          jax_version=jax.__version__, jax_backend=jax.default_backend(),
                          numpy_version=numpy.__version__,
                          peak_rss_self_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return result


if __name__ == "__main__":
    from dluxshera.ml.training import run_supervised
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=240)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--implementations", nargs="+",
                        choices=("clean", "reference", "batched", "numpy"),
                        default=["clean", "reference", "batched", "numpy"])
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    for workers in args.workers:
        for second in (False, True):
            for implementation in args.implementations:
                result = run_supervised(Probe(implementation, second, workers, args.batches, args.epochs).run,
                                        timeout_seconds=600, label="fixture")
                print(json.dumps(result), flush=True)
