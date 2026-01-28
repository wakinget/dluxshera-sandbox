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
import time

import jax
import numpy as np

from dluxshera.inference.optimization import fim_theta_shera
from dluxshera.params.store import ParameterStore
from dluxshera.plot.plotting import apply_plot_defaults, get_default_cmaps, plot_fim, plot_fim_plotly, merge_cbar
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

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
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "work/experiments/Results" / f"fim_comparison_{TIMESTAMP}"


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


def plot_fim_comparison_plotly(
    fim_testbed, fim_flight, fim_ratio, labels,
    log_scale=True, eps=1e-20, vmin=None, vmax=None,
    save_path=None, show=False
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def _prep(z):
        z = np.asarray(z)
        return np.log10(np.abs(z) + eps) if log_scale else z

    z1, z2, z3 = _prep(fim_testbed), _prep(fim_flight), _prep(fim_ratio)

    if vmin is None:
        vmin = np.nanmin([np.nanmin(z1), np.nanmin(z2), np.nanmin(z3)])
    if vmax is None:
        vmax = np.nanmax([np.nanmax(z1), np.nanmax(z2), np.nanmax(z3)])

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Testbed FIM", "Flight FIM", "Ratio (Flight/Testbed)"),
        horizontal_spacing=0.08,
    )

    # Share one colorscale range across all three for comparability
    common = dict(colorscale="Viridis", zmin=vmin, zmax=vmax, x=labels, y=labels)

    fig.add_trace(go.Heatmap(z=z1, colorbar=dict(title="log10(|FIM|+eps)"), **common), row=1, col=1)
    fig.add_trace(go.Heatmap(z=z2, showscale=False, **common), row=1, col=2)
    fig.add_trace(go.Heatmap(z=z3, showscale=False, **common), row=1, col=3)

    fig.update_layout(
        xaxis=dict(tickangle=-45),
        xaxis2=dict(tickangle=-45),
        xaxis3=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
        yaxis2=dict(autorange="reversed"),
        yaxis3=dict(autorange="reversed"),
        margin=dict(l=80, r=40, t=60, b=140),
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path if str(save_path).endswith(".html") else save_path.with_suffix(".html")))

    if show:
        fig.show()

    return fig



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
    t0 = time.time()
    print("Running fim_comparison.py...")
    args = build_parser().parse_args()
    jax.config.update("jax_enable_x64", JAX_ENABLE_X64)

    infer_keys = parse_infer_keys(args.infer_keys)
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    save_plots = True
    show_plots = True
    plots_block = True
    _ = get_default_cmaps()
    apply_plot_defaults()
    plt.rcParams["image.cmap"] = "inferno_nan"

    testbed_cfg = configure_shera_config(SHERA_TESTBED_CONFIG)
    flight_cfg = configure_shera_config(SHERA_FLIGHT_CONFIG)

    # Update config for speed
    testbed_cfg = testbed_cfg.replace(n_lambda=1,
                      primary_noll_indices=tuple(range(4, 7)),
                      secondary_noll_indices=tuple(range(4, 7)))
    flight_cfg = flight_cfg.replace(n_lambda=1,
                      primary_noll_indices=tuple(range(4, 7)),
                      secondary_noll_indices=tuple(range(4, 7)))

    print("Computing FIM for Testbed point design...")
    fim_testbed, labels_testbed = compute_fim(
        testbed_cfg,
        infer_keys=infer_keys,
        noise_model=args.noise_model,
        reduce=args.reduce,
    )
    print("Computing FIM for Flight point design...")
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

    fim_ratio = fim_flight / (fim_testbed + args.eps)
    abs_ratio = np.abs(fim_ratio)
    log_ratio = np.log10(abs_ratio)
    # log_label = r"$\log_{10}(|\mathrm{Ratio}|)$"

    print("Plotting the two FIMs...")
    scale_label = "log" if args.log_scale else "linear"
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_fim(
        fim_testbed,
        labels_testbed,
        log_scale=args.log_scale,
        vmin=4,
        vmax=14,
        ax=axes[0],
        save_path=None,
        show=False,
        close=False,
    )
    axes[0].set_title(f"Testbed FIM ({scale_label})")

    plot_fim(
        fim_flight,
        labels_flight,
        log_scale=args.log_scale,
        vmin=4,
        vmax=14,
        ax=axes[1],
        save_path=None,
        show=False,
        close=False,
    )
    axes[1].set_title(f"Flight FIM ({scale_label})")

    plot_fim(
        log_ratio,
        labels_flight,
        log_scale=False,
        vmin=-3.0,
        vmax=3.0,
        # cmap="vididis",
        cbar_label=r"$\log_{10}(|\mathrm{Ratio}|)$",
        ax=axes[2],
        save_path=None,
        show=False,
        close=False,
    )
    axes[2].set_title(f"Log Ratio: Flight / Testbed")

    # plot_fim input signature for reference
    # def plot_fim(
    #         fim: np.ndarray,
    #         labels: Sequence[str],
    #         log_scale: bool = True,
    #         vmin=None,
    #         vmax=None,
    #         cmap: str = "viridis",
    #         cbar_label: Optional[str] = None,
    #         figsize=(8, 6),
    #         eps: float = 1e-20,
    #         ax=None,
    #         save_path: Optional[Union[str, Path]] = None,
    #         show: bool = False,
    #         close: bool = True,
    # ):

    print("Saving the Results...")
    if save_plots:
        fig_path = outdir / "fim_comparison.png"
        fig.savefig(fig_path)
        print(f"Saved figure: {fig_path}")

    np.savez(
        outdir / "fim_comparison.npz",
        fim_testbed=fim_testbed,
        fim_flight=fim_flight,
        fim_ratio=fim_ratio,
        labels=np.array(labels_testbed, dtype=str),
    )
    print(f"Saved data: {outdir / 'fim_comparison.npz'}")

    if show_plots:
        plt.show(block=plots_block)
    else:
        plt.close(fig)

    # Try using plotly
    plot_fim_comparison_plotly(
        fim_testbed, fim_flight, fim_ratio, labels_testbed,
        save_path=outdir / "fim_comparison_plotly.html",
        show=True,
    )

    t1 = time.time()
    print("Finished in %.3f sec" % (t1 - t0))

if __name__ == "__main__":
    out = main()
