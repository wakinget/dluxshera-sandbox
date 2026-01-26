"""
Scaffold a testbed vs flight Fisher Information Matrix (FIM) comparison.

This script mirrors the setup flow in ``examples/recipes/canonical_astrometry.py``
while skipping any optimization. It builds Shera three-plane binders for the
Testbed and Flight configs, computes their FIMs for a shared inference key set,
then writes a 1×3 comparison plot (testbed, flight, ratio) plus an ``.npz``
artifact for downstream interactive exploration.

How to run
----------
PYTHONPATH=src python work/experiments/fim_comparison.py \
    --outdir work/experiments/outputs/fim_compare_run1 \
    --infer-keys binary.separation_as \
    --infer-keys binary.x_position_as \
    --infer-keys binary.y_position_as
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from dluxshera.inference.optimization import fim_theta_shera
from dluxshera.params.store import ParameterStore
from dluxshera.plot.plotting import apply_plot_defaults, get_default_cmaps, plot_fim
from dluxshera.systems.three_plane import (
    SHERA_FLIGHT_CONFIG,
    SHERA_TESTBED_CONFIG,
    SheraThreePlaneBinder,
    SheraThreePlaneConfig,
    build_forward_spec_from_config,
)

JAX_ENABLE_X64 = True
DEFAULT_INFER_KEYS = (
    "binary.separation_as",
    "binary.position_angle_deg",
    "binary.x_position_as",
    "binary.y_position_as",
    "binary.log_flux_total",
    "binary.contrast",
    "system.plate_scale_as_per_pix",
    "primary.zernike_coeffs_nm",
    "secondary.zernike_coeffs_nm",
)
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DEFAULT_RESULTS_DIR = Path(f"work/experiments/Results/fim_comparison_{TIMESTAMP}")


def parse_infer_keys(raw_keys: list[str] | None) -> tuple[str, ...]:
    """Normalize inference keys from CLI inputs."""
    if not raw_keys:
        return DEFAULT_INFER_KEYS

    parsed: list[str] = []
    for entry in raw_keys:
        parsed.extend([key.strip() for key in entry.split(",") if key.strip()])

    return tuple(parsed) or DEFAULT_INFER_KEYS


def configure_shera_config(config: SheraThreePlaneConfig) -> SheraThreePlaneConfig:
    """Apply canonical astrometry-style overrides to a Shera config."""
    return config.replace(
        primary_noll_indices=tuple(range(4, 12)),
        secondary_noll_indices=tuple(range(4, 12)),
    )


def compute_fim(
    cfg: SheraThreePlaneConfig,
    *,
    infer_keys: tuple[str, ...],
    noise_model: str,
    reduce: str,
) -> tuple[np.ndarray, list[str]]:
    """Build a binder and compute the FIM for the requested inference keys."""
    forward_spec = build_forward_spec_from_config(cfg)
    base_store = ParameterStore.from_spec_defaults(forward_spec)
    base_store = base_store.replace(
        {
            "binary.separation_as": 10.0,
            "binary.position_angle_deg": 90.0,
            "binary.x_position_as": 0.0,
            "binary.y_position_as": 0.0,
            "imaging.exposure_time_s": 1800.0,
        }
    )
    base_store = base_store.refresh_derived(forward_spec)

    binder = SheraThreePlaneBinder(cfg, forward_spec, base_store)
    data = binder.model()
    var = data

    fim, _, labels = fim_theta_shera(
        cfg,
        forward_spec,
        base_store,
        infer_keys,
        data,
        var,
        noise_model=noise_model,
        reduce=reduce,
        return_labels=True,
    )

    return np.asarray(fim), labels


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compare Fisher Information Matrices for Shera testbed/flight configs.",
    )
    parser.add_argument(
        "--infer-keys",
        action="append",
        help=(
            "Inference keys to include. May be repeated or comma-separated. "
            "Defaults mirror canonical_astrometry.py."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to write outputs.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-20,
        help="Small value added to the flight FIM when forming ratios.",
    )
    parser.add_argument(
        "--log-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot FIMs with log scaling (default: true).",
    )
    parser.add_argument(
        "--noise-model",
        choices=("gaussian", "poisson"),
        default="gaussian",
        help="Noise model for the FIM computation.",
    )
    parser.add_argument(
        "--reduce",
        choices=("sum", "mean"),
        default="sum",
        help="Reduction used in the NLL when computing the FIM.",
    )
    return parser


def main() -> None:
    """Run the FIM comparison script."""
    args = build_parser().parse_args()
    jax.config.update("jax_enable_x64", JAX_ENABLE_X64)

    infer_keys = parse_infer_keys(args.infer_keys)
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    _ = get_default_cmaps()
    apply_plot_defaults()
    plt.rcParams["image.cmap"] = "inferno_nan"

    testbed_cfg = configure_shera_config(SHERA_TESTBED_CONFIG)
    flight_cfg = configure_shera_config(SHERA_FLIGHT_CONFIG)

    fim_testbed, labels_testbed = compute_fim(
        testbed_cfg,
        infer_keys=infer_keys,
        noise_model=args.noise_model,
        reduce=args.reduce,
    )
    fim_flight, labels_flight = compute_fim(
        flight_cfg,
        infer_keys=infer_keys,
        noise_model=args.noise_model,
        reduce=args.reduce,
    )

    if labels_testbed != labels_flight:
        raise ValueError(
            "Mismatch between FIM labels for testbed and flight configs. "
            "Ensure infer keys are consistent across configs."
        )

    fim_ratio = fim_testbed / (fim_flight + args.eps)

    scale_label = "log" if args.log_scale else "linear"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_fim(
        fim_testbed,
        labels_testbed,
        log_scale=args.log_scale,
        ax=axes[0],
        close=False,
    )
    axes[0].set_title(f"Testbed FIM ({scale_label})")

    plot_fim(
        fim_flight,
        labels_flight,
        log_scale=args.log_scale,
        ax=axes[1],
        close=False,
    )
    axes[1].set_title(f"Flight FIM ({scale_label})")

    plot_fim(
        fim_ratio,
        labels_flight,
        log_scale=args.log_scale,
        ax=axes[2],
        close=False,
    )
    axes[2].set_title(f"Ratio: Testbed / Flight ({scale_label})")

    fig_path = outdir / "fim_comparison.png"
    fig.savefig(fig_path)
    plt.close(fig)

    np.savez(
        outdir / "fim_comparison.npz",
        fim_testbed=fim_testbed,
        fim_flight=fim_flight,
        fim_ratio=fim_ratio,
        labels=np.array(labels_testbed, dtype=str),
    )

    print(f"Saved figure: {fig_path}")
    print(f"Saved data: {outdir / 'fim_comparison.npz'}")


if __name__ == "__main__":
    main()
