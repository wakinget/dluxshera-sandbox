from pathlib import Path

import numpy as np

from dluxshera.inference.optimization import (
    make_binder_image_nll_fn,
    run_image_gd,
    run_simple_gd,
)
from dluxshera.inference.run_artifacts import load_meta, load_summary, load_trace
from dluxshera.optics.config import SheraThreePlaneConfig
from dluxshera.params.spec import build_forward_model_spec_from_config
from dluxshera.params.store import ParameterStore


def test_run_simple_gd_writes_artifacts(tmp_path: Path):
    theta0 = np.array([1.0, -2.0], dtype=float)

    def loss_fn(theta: np.ndarray) -> np.ndarray:
        return np.sum(theta**2)

    run_dir = tmp_path / "run_artifacts"
    theta_final, history = run_simple_gd(
        loss_fn,
        theta0,
        learning_rate=0.1,
        num_steps=6,
        run_dir=run_dir,
        save_checkpoints=True,
    )

    assert theta_final.shape == theta0.shape
    assert "loss" in history
    assert "theta" in history

    assert (run_dir / "trace.npz").exists()
    assert (run_dir / "meta.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "checkpoint_best.npz").exists()
    assert (run_dir / "checkpoint_final.npz").exists()

    trace = load_trace(run_dir)
    assert trace["loss"].shape == (6,)
    assert trace["theta"].shape == (6, 2)
    assert trace["grad_norm"].shape == (6,)
    assert trace["step_norm"].shape == (6,)
    assert trace["base_lr"].shape == (6,)

    meta = load_meta(run_dir)
    assert meta["artifact_schema"] == "dluxshera-run-v0"
    assert meta["theta"]["dim"] == 2
    assert meta["theta"]["theta_space"] == "primitive"
    assert meta["optimizer"]["num_steps"] == 6

    summary = load_summary(run_dir)
    assert summary["status"] == "ok"
    assert summary["run_id"] == run_dir.name
    assert summary["num_steps_completed"] == 6
    assert summary["loss_init"] is not None
    assert summary["loss_final"] is not None
    assert summary["loss_best"] is not None
    assert summary["best_step"] is not None
    assert summary["has_checkpoint_best"] is True
    assert summary["has_checkpoint_final"] is True

    checkpoint_best = np.load(run_dir / "checkpoint_best.npz")
    assert "theta_best" in checkpoint_best
    assert "best_step" in checkpoint_best
    assert "best_loss" in checkpoint_best

    checkpoint_final = np.load(run_dir / "checkpoint_final.npz")
    assert "theta_final" in checkpoint_final
    assert "final_step" in checkpoint_final
    assert "final_loss" in checkpoint_final


def test_run_image_gd_writes_index_map_metadata(tmp_path: Path):
    cfg = SheraThreePlaneConfig(
        design_name="shera_threeplane_smoke",
        pupil_npix=8,
        psf_npix=8,
        oversample=1,
        primary_noll_indices=(4,),
    )

    forward_spec = build_forward_model_spec_from_config(cfg)
    store_init = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(forward_spec)
    infer_keys = ("binary.separation_as",)

    # Build synthetic data directly from the binder-backed predict_fn to keep the
    # smoke test cheap and deterministic.
    _loss_fn, theta0, predict_fn = make_binder_image_nll_fn(
        cfg,
        forward_spec,
        store_init,
        infer_keys,
        data=np.zeros((cfg.psf_npix, cfg.psf_npix)),
        var=np.ones((cfg.psf_npix, cfg.psf_npix)),
        reduce="mean",
        return_predict_fn=True,
    )
    data = np.asarray(predict_fn(theta0))
    var = np.ones_like(data)

    run_dir = tmp_path / "binder_run"
    theta_final, store_final, history = run_image_gd(
        cfg,
        forward_spec,
        store_init,
        infer_keys,
        data,
        var,
        learning_rate=1e-3,
        num_steps=3,
        run_dir=run_dir,
    )

    assert theta_final.shape == theta0.shape
    assert store_final.get(infer_keys[0]) is not None
    assert history["loss"].shape[0] == 3

    assert (run_dir / "trace.npz").exists()
    assert (run_dir / "meta.json").exists()
    assert (run_dir / "summary.json").exists()

    trace = load_trace(run_dir)
    assert trace["theta"].shape == (3, theta0.size)
    assert trace["loss"].shape == (3,)

    meta = load_meta(run_dir)
    assert meta["artifact_schema"] == "dluxshera-run-v0"
    assert meta["theta"]["dim"] == int(theta0.size)
    index_map = meta["theta"]["index_map"]
    assert index_map["entries"]
    assert index_map["entries"][-1]["stop"] == theta0.size

    summary = load_summary(run_dir)
    assert summary["status"] == "ok"
    assert summary["num_steps_completed"] == 3
