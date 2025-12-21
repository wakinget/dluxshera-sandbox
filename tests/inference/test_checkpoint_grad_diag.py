from pathlib import Path
import json

import jax.numpy as jnp
import numpy as np

from dluxshera.inference.diagnostics import compute_checkpoint_gradients


def quadratic_builder():
    def loss(theta: np.ndarray) -> np.ndarray:
        return 0.5 * jnp.sum(theta**2)

    return loss


def test_compute_checkpoint_gradients(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    meta = {
        "run_id": "run",
        "theta": {
            "dim": 2,
            "index_map": {
                "entries": [
                    {"name": "a", "start": 0, "stop": 1, "block": "a"},
                    {"name": "b", "start": 1, "stop": 2, "block": "b"},
                ]
            },
        },
        "optimizer": {"base_lr": 1e-3},
    }
    summary = {"run_id": "run", "status": "ok"}

    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    theta_best = np.array([1.0, -2.0], dtype=float)
    np.savez_compressed(run_dir / "checkpoint_best.npz", theta_best=theta_best)

    result = compute_checkpoint_gradients(run_dir, builder=quadratic_builder, compute_curvature=True)

    diag_path = run_dir / "diag" / "grad_at_best.npz"
    summary_path = run_dir / "diag" / "grad_summary.json"

    assert diag_path.exists()
    assert summary_path.exists()
    diag = np.load(diag_path)

    assert np.allclose(diag["grad"], theta_best)
    assert np.isclose(diag["grad_norm"], np.linalg.norm(theta_best))
    assert "curv_diag" in diag and "lr_vec" in diag
    assert "block_grad_norms" in result
