from __future__ import annotations

import argparse
import json
from pathlib import Path

from dluxshera.datasets.master_v4 import PRODUCTION_IDENTITIES
from dluxshera.datasets.prepared_v4 import prepare_shera_v4_dataset
from dluxshera.datasets.schema import json_ready
from dluxshera.datasets.shera import prepare_shera_v3_dataset


def main(argv: list[str] | None = None) -> int:
    """Prepare SHERA ML raw image datasets into reusable array shards."""
    parser = argparse.ArgumentParser(
        description="Prepare SHERA ML FITS datasets into sample-centric array shards."
    )
    parser.add_argument(
        "--dataset-kind",
        choices=("v3", "v4"),
        default="v3",
        help="Use v3 for legacy manifest/samples.jsonl roots or v4 for master V4 render roots.",
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--target-shard-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-samples-per-shard", type=int, default=None)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--v4-source-audit", choices=("off", "sample", "full"), default="sample")
    parser.add_argument("--v4-source-audit-samples", type=int, default=256)
    parser.add_argument("--allow-incomplete-source", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--v4-plan-root",
        type=Path,
        default=Path("work/experiments/ml/datasets/materialized/master_v4"),
        help="Frozen V4 state-plan root used when --dataset-kind=v4.",
    )
    parser.add_argument(
        "--no-v4-production-identity-check",
        action="store_true",
        default=False,
        help="Allow tiny/development V4-like plan roots instead of the frozen production identities.",
    )
    args = parser.parse_args(argv)

    if args.dataset_kind == "v4":
        summary = prepare_shera_v4_dataset(
            source_root=args.source_root,
            plan_root=args.v4_plan_root,
            outdir=args.outdir,
            dtype=args.dtype,
            target_shard_bytes=args.target_shard_bytes,
            max_samples_per_shard=args.max_samples_per_shard or 1024,
            validation_samples=args.validation_samples,
            seed=args.seed,
            max_samples=args.max_samples,
            source_audit=args.v4_source_audit,
            source_audit_samples=args.v4_source_audit_samples,
            overwrite=args.overwrite,
            resume=args.resume,
            dry_run=args.dry_run,
            expected=None if args.no_v4_production_identity_check else PRODUCTION_IDENTITIES,
        )
    else:
        summary = prepare_shera_v3_dataset(
            source_root=args.source_root,
            outdir=args.outdir,
            dtype=args.dtype,
            target_shard_bytes=args.target_shard_bytes,
            max_samples_per_shard=args.max_samples_per_shard,
            validation_samples=args.validation_samples,
            seed=args.seed,
            max_samples=args.max_samples,
            allow_incomplete_source=args.allow_incomplete_source,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    print(json.dumps(json_ready(summary.to_dict()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
