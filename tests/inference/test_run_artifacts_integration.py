from pathlib import Path

import numpy as np

from dluxshera.inference.optimization import run_simple_gd
from dluxshera.inference.run_artifacts import load_meta, load_summary, load_trace


def test_run_simple_gd_writes_artifacts(tmp_path: Path):
    theta0 = np.array([1.0, -2.0], dtype=float)

    def loss_fn(theta: np.ndarray) -> np.ndarray:
        return np.sum(theta**2)

    run_dir = tmp_path / "run_artifacts"
    theta_final, history = run_simple_gd(
        loss_fn,
        theta0,
        learning_rate=0.1,
        num_steps=5,
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
    assert trace["loss"].shape == (5,)
    assert trace["theta"].shape == (5, 2)
    assert trace["grad_norm"].shape == (5,)
    assert trace["step_norm"].shape == (5,)
    assert "base_lr" in trace

    meta = load_meta(run_dir)
    assert meta["artifact_schema"] == "dluxshera-run-v0"
    assert meta["theta"]["dim"] == 2
    assert meta["theta"]["theta_space"] == "primitive"
    assert meta["optimizer"]["num_steps"] == 5

    summary = load_summary(run_dir)
    assert summary["status"] == "ok"
    assert summary["run_id"] == run_dir.name
    assert summary["num_steps_completed"] == 5
    assert summary["loss_init"] is not None
    assert summary["loss_final"] is not None
    assert summary["loss_best"] is not None
    assert summary["best_step"] is not None
