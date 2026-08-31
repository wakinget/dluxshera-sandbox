from __future__ import annotations

import argparse
import json
from pathlib import Path

from dluxshera.datasets.schema import json_ready


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the ML-S01 pairwise SHERA correction baseline."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--preset", choices=("s01_e00", "s01_e01"), default="s01_e01")
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--split-registry", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--study-id", default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--comparator", choices=("concat_diff", "difference"), default=None)
    parser.add_argument("--pairs-per-epoch", type=int, default=None)
    parser.add_argument("--min-fisher-distance", type=float, default=None)
    parser.add_argument("--max-fisher-distance", type=float, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--evaluate-test", action="store_true", default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args(argv)

    from dluxshera.ml.training import load_run_config, train_pairwise_correction

    config = load_run_config(args.config, preset=args.preset)
    for key in ("study_id", "experiment_id", "run_id", "seed", "device", "resume_checkpoint"):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.pairs_per_epoch is not None:
        config.setdefault("training", {})["pairs_per_epoch"] = args.pairs_per_epoch
    if args.comparator is not None:
        config.setdefault("model", {})["comparator"] = args.comparator
    if args.min_fisher_distance is not None:
        config.setdefault("pair_policy", {})["min_fisher_distance"] = args.min_fisher_distance
    if args.max_fisher_distance is not None:
        config.setdefault("pair_policy", {})["max_fisher_distance"] = args.max_fisher_distance
    if args.evaluate_test is not None:
        config["evaluate_test"] = bool(args.evaluate_test)

    summary = train_pairwise_correction(
        config=config,
        prepared_root=args.prepared_root,
        split_registry_path=args.split_registry,
        output_dir=args.output_dir,
        validation_manifest_path=args.validation_manifest,
        test_manifest_path=args.test_manifest,
        overwrite=args.overwrite,
    )
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
