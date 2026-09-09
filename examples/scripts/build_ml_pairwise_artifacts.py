from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from dluxshera.datasets.schema import json_ready
from dluxshera.ml import (
    PairPolicy,
    generate_frozen_pair_manifest,
    generate_role_preserving_split_registry,
    generate_split_registry,
    load_sample_catalog,
    load_split_registry,
    write_pair_manifest,
    write_split_registry,
)


def _fractions(value: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for chunk in value.split(","):
        if not chunk.strip():
            continue
        key, raw = chunk.split(":", 1)
        out[key.strip()] = float(raw)
    if not out:
        raise argparse.ArgumentTypeError("fractions must look like train:0.8,validation:0.1,test:0.1")
    return out


def _weights(value: str | None) -> dict[str, float]:
    if value in (None, ""):
        return {}
    return _fractions(str(value))


def _json_mapping(path_or_text: str | None) -> Mapping[str, object] | None:
    if path_or_text in (None, ""):
        return None
    path = Path(path_or_text)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(path_or_text)


def _add_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy-id", default="s01_clean_same_pair_grid_v1")
    parser.add_argument(
        "--family-weights",
        default="same_nuisance_different_science:1.0",
        help="Pair-family weights, e.g. same_nuisance_different_science:0.5,different_nuisance_same_science:0.25,different_science_different_nuisance:0.25.",
    )
    parser.add_argument("--same-pair-id", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-fisher-distance", type=float, default=0.0)
    parser.add_argument("--max-fisher-distance", type=float, default=5000.0)
    parser.add_argument("--include-reverse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-changed-science-dimensions", type=int, default=None)
    parser.add_argument("--dataset-family", action="append", default=[])
    parser.add_argument("--dataset-family-weights", default=None)
    parser.add_argument("--require-same-dataset-family", action="store_true", default=False)
    parser.add_argument("--joint-train-prefix-size", type=int, default=None)
    parser.add_argument(
        "--fisher-distance-bin-edges",
        default=None,
        help="Comma-separated Fisher-distance edges; defaults to the standard 0,100,...,5000 edges.",
    )
    parser.add_argument(
        "--fisher-distance-bins",
        default=None,
        help="Comma-separated allowed bin labels such as 0-100,100-250.",
    )
    parser.add_argument(
        "--distance-bin-weights",
        default=None,
        help="Optional bin-label weights for approximately balanced distance sampling.",
    )


def _edges(value: str | None) -> tuple[float, ...] | None:
    if value in (None, ""):
        return None
    return tuple(float(chunk) for chunk in str(value).split(",") if chunk.strip())


def _labels(value: str | None) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    return tuple(chunk.strip() for chunk in str(value).split(",") if chunk.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build SHERA ML split and pair-evaluation artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-catalog", help="Summarize a prepared dataset catalog.")
    inspect_parser.add_argument("--prepared-root", type=Path, required=True)

    split_parser = subparsers.add_parser("make-split", help="Generate SPLIT-ML-v1.")
    split_parser.add_argument("--prepared-root", type=Path, required=True)
    split_parser.add_argument("--out", type=Path, required=True)
    split_parser.add_argument("--seed", type=int, default=0)
    split_parser.add_argument(
        "--science-fractions",
        type=_fractions,
        default={"train": 0.8, "validation": 0.1, "test": 0.1},
    )
    split_parser.add_argument(
        "--nuisance-fractions",
        type=_fractions,
        default={"train": 0.8, "validation": 0.1, "test": 0.1},
    )
    split_parser.add_argument(
        "--explicit-nuisance-assignments",
        default=None,
        help="JSON object or path mapping nuisance id to train/validation/test.",
    )
    split_parser.add_argument(
        "--preserve-catalog-science-splits",
        action="store_true",
        default=False,
        help="Use catalog sample_role/split_role as the science split registry instead of drawing random grouped science splits.",
    )
    split_parser.add_argument(
        "--default-nuisance-split",
        default="train",
        help="Nuisance split assigned to every nuisance state when explicit assignments are not provided.",
    )

    pair_parser = subparsers.add_parser("make-eval", help="Generate frozen PAIR-EVAL-v1 pairs.")
    pair_parser.add_argument("--prepared-root", type=Path, required=True)
    pair_parser.add_argument("--split-registry", type=Path, required=True)
    pair_parser.add_argument("--outdir", type=Path, required=True)
    pair_parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    pair_parser.add_argument("--seed", type=int, default=0)
    pair_parser.add_argument("--pairs-per-slice", type=int, default=256)
    pair_parser.add_argument(
        "--eval-slices",
        default=None,
        help="Optional JSON object or path mapping slice names to science/nuisance split selections.",
    )
    _add_policy_args(pair_parser)

    args = parser.parse_args(argv)
    if args.command == "inspect-catalog":
        catalog = load_sample_catalog(args.prepared_root)
        print(json.dumps(json_ready(catalog.summary()), indent=2, sort_keys=True))
        return 0
    if args.command == "make-split":
        catalog = load_sample_catalog(args.prepared_root)
        if args.preserve_catalog_science_splits:
            registry = generate_role_preserving_split_registry(
                catalog,
                seed=args.seed,
                explicit_nuisance_assignments=_json_mapping(args.explicit_nuisance_assignments),
                default_nuisance_split=args.default_nuisance_split,
            )
        else:
            registry = generate_split_registry(
                catalog,
                seed=args.seed,
                science_fractions=args.science_fractions,
                nuisance_fractions=args.nuisance_fractions,
                explicit_nuisance_assignments=_json_mapping(args.explicit_nuisance_assignments),
            )
        write_split_registry(args.out, registry)
        print(json.dumps(json_ready(registry.counts), indent=2, sort_keys=True))
        return 0
    if args.command == "make-eval":
        catalog = load_sample_catalog(args.prepared_root)
        registry = load_split_registry(args.split_registry, catalog=catalog)
        edges = _edges(args.fisher_distance_bin_edges)
        policy = PairPolicy(
            policy_id=args.policy_id,
            family_weights=_weights(args.family_weights),
            same_pair_id=args.same_pair_id,
            min_fisher_distance=args.min_fisher_distance,
            max_fisher_distance=args.max_fisher_distance,
            max_changed_science_dimensions=args.max_changed_science_dimensions,
            include_reverse=args.include_reverse,
            dataset_families=tuple(args.dataset_family),
            dataset_family_weights=_weights(args.dataset_family_weights),
            require_same_dataset_family=args.require_same_dataset_family,
            joint_train_prefix_size=args.joint_train_prefix_size,
            fisher_distance_bin_edges=PairPolicy().fisher_distance_bin_edges
            if edges is None
            else edges,
            fisher_distance_bins=_labels(args.fisher_distance_bins),
            distance_bin_weights=_weights(args.distance_bin_weights),
        )
        manifest = generate_frozen_pair_manifest(
            catalog,
            registry,
            policy=policy,
            split=args.split,
            seed=args.seed,
            pairs_per_slice=args.pairs_per_slice,
            eval_slices=_json_mapping(args.eval_slices),
        )
        write_pair_manifest(args.outdir, manifest)
        print(json.dumps(json_ready(manifest.summary()), indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
