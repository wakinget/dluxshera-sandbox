"""Minimal two-plane astrometry demo mirroring the canonical three-plane example."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import jax.numpy as jnp
import numpy as np

from dluxshera.core.binder import SheraTwoPlaneBinder
from dluxshera.inference.optimization import make_binder_image_nll_fn, run_simple_gd
from dluxshera.optics.config import SheraTwoPlaneConfig
from dluxshera.params.packing import unpack_params as store_unpack_params
from dluxshera.params.spec import ParamKey, ParamSpec, build_inference_spec_basic
from dluxshera.params.spec import build_shera_twoplane_forward_spec_from_config
from dluxshera.params.store import ParameterStore, refresh_derived
from dluxshera.params.transforms import DEFAULT_SYSTEM_ID, TRANSFORMS
import dluxshera.params.shera_threeplane_transforms  # Registers default transforms
from dluxshera.plot.plotting import (
    plot_parameter_history_grid,
    plot_psf_comparison,
    plot_psf_single,
)


DEFAULT_RESULTS_DIR = Path("Results/TwoplaneAstrometryDemo")


@dataclass
class DemoData:
    """Container for key demo artifacts returned to callers and tests."""

    cfg: SheraTwoPlaneConfig
    forward_spec: ParamSpec
    inference_spec: ParamSpec
    truth_store: ParameterStore
    binder: SheraTwoPlaneBinder
    truth_psf: np.ndarray
    noisy_psf: np.ndarray
    init_psf: Optional[np.ndarray]
    gd_psf: Optional[np.ndarray]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _evaluate_psf(binder: SheraTwoPlaneBinder, store: Optional[ParameterStore] = None) -> np.ndarray:
    return np.array(binder.model(store))


def _format_parameter_summary(
    keys: Iterable[ParamKey],
    truth_store: ParameterStore,
    init_store: ParameterStore,
    final_store: ParameterStore,
) -> str:
    lines = ["Parameter summary (truth → init → recovered):"]
    for key in keys:
        truth_val = truth_store.get(key)
        init_val = init_store.get(key)
        final_val = final_store.get(key)
        lines.append(f"  - {key}: {truth_val} → {init_val} → {final_val}")
    return "\n".join(lines)


def _build_param_history(
    theta_sequence: Sequence[np.ndarray],
    base_store: ParameterStore,
    sub_spec: ParamSpec,
    keys: Sequence[str],
) -> Mapping[str, np.ndarray]:
    histories = {key: [] for key in keys}
    for theta_vec in theta_sequence:
        store = store_unpack_params(sub_spec, theta_vec, base_store)
        for key in keys:
            histories[key].append(np.array(store.get(key)))
    return {k: np.stack(v) for k, v in histories.items()}


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------


def main(
    fast: bool = False,
    save_plots: bool = False,
    add_noise: bool = False,
    save_plots_dir: Optional[Path] = None,
) -> DemoData:
    rng = np.random.default_rng(42)
    make_plots = save_plots or save_plots_dir is not None
    output_dir = Path(save_plots_dir) if save_plots_dir is not None else DEFAULT_RESULTS_DIR

    # 1) Build config + specs
    pupil_npix = 64 if fast else 128
    psf_npix = 64 if fast else 128
    n_lambda = 1 if fast else 3

    cfg = SheraTwoPlaneConfig(
        design_name="shera_twoplane_demo",
        pupil_npix=pupil_npix,
        psf_npix=psf_npix,
        oversample=2,
        n_lambda=n_lambda,
        primary_noll_indices=(4, 5),
    )
    forward_spec = build_shera_twoplane_forward_spec_from_config(cfg)
    inference_spec = build_inference_spec_basic()

    base_store = ParameterStore.from_spec_defaults(forward_spec)
    base_store = refresh_derived(base_store, forward_spec, TRANSFORMS, system_id=DEFAULT_SYSTEM_ID)

    # 2) Build binder
    binder = SheraTwoPlaneBinder(cfg, forward_spec, base_store, use_system_graph=True)

    # 3) Truth parameters and synthetic data
    truth_updates = {
        "binary.separation_as": 0.6,
        "binary.position_angle_deg": 25.0,
        "binary.contrast": 0.05,
    }
    truth_store = base_store.replace(truth_updates)
    truth_psf = _evaluate_psf(binder, truth_store)

    noisy_psf = truth_psf.copy()
    if add_noise:
        noisy_psf = noisy_psf + rng.normal(scale=0.01 * noisy_psf.max(), size=noisy_psf.shape)

    # 4) Inference setup
    infer_keys = (
        "binary.separation_as",
        "binary.position_angle_deg",
        "binary.contrast",
    )
    sub_spec = forward_spec.subset(infer_keys)

    init_updates = {
        "binary.separation_as": truth_updates["binary.separation_as"] * 0.8,
        "binary.position_angle_deg": truth_updates["binary.position_angle_deg"] + 10.0,
        "binary.contrast": truth_updates["binary.contrast"] * 1.5,
    }
    init_store = base_store.replace(init_updates)
    init_psf = _evaluate_psf(binder, init_store)

    var = jnp.ones_like(noisy_psf)
    loss_fn, theta0 = make_binder_image_nll_fn(
        cfg,
        forward_spec,
        base_store,
        infer_keys,
        noisy_psf,
        var,
        use_system_graph=True,
    )

    num_steps = 5 if fast else 15
    learning_rate = 5e-2 if fast else 1e-2

    theta_final, history = run_simple_gd(
        loss_fn,
        theta0,
        learning_rate=learning_rate,
        num_steps=num_steps,
    )

    theta_history = history["theta"]
    final_store = store_unpack_params(sub_spec, theta_final, base_store)
    gd_psf = _evaluate_psf(binder, final_store)

    # 5) Plotting
    output_dir.mkdir(parents=True, exist_ok=True)
    if make_plots:
        plot_psf_single(truth_psf, title="Truth PSF", save_path=output_dir / "psf_truth.png")
        plot_psf_single(noisy_psf, title="Noisy observation", save_path=output_dir / "psf_noisy.png")
        plot_psf_single(gd_psf, title="Recovered PSF", save_path=output_dir / "psf_recovered.png")

        plot_psf_comparison(
            noisy_psf,
            gd_psf,
            var=jnp.ones_like(noisy_psf),
            model_label="Recovered PSF",
            save_path=output_dir / "psf_comparison.png",
        )

        param_history = _build_param_history(theta_history, base_store, sub_spec, infer_keys)
        plot_parameter_history_grid(
            param_history,
            true_vals={key: truth_store.get(key) for key in infer_keys},
            save_path=output_dir / "parameter_history.png",
        )

    print(
        _format_parameter_summary(
            infer_keys,
            truth_store,
            init_store,
            final_store,
        )
    )

    return DemoData(
        cfg=cfg,
        forward_spec=forward_spec,
        inference_spec=inference_spec,
        truth_store=truth_store,
        binder=binder,
        truth_psf=truth_psf,
        noisy_psf=noisy_psf,
        init_psf=init_psf,
        gd_psf=gd_psf,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Shera two-plane astrometry demo.")
    parser.add_argument("--fast", action="store_true", help="Use reduced grid sizes and fewer GD steps for quick runs")
    parser.add_argument("--save-plots", action="store_true", help="Save PSF and history plots")
    parser.add_argument("--add-noise", action="store_true", help="Inject light Gaussian noise into the synthetic PSF")
    parser.add_argument("--save-plots-dir", type=Path, help="Optional output directory for plots (implies --save-plots)")
    args = parser.parse_args()

    main(
        fast=args.fast,
        save_plots=args.save_plots or args.save_plots_dir is not None,
        add_noise=args.add_noise,
        save_plots_dir=args.save_plots_dir,
    )
