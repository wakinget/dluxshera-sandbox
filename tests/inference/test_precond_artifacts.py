from pathlib import Path

import numpy as np
import pytest

from dluxshera.inference.preconditioning import (
    PreconditioningConfig,
    compute_precond_vectors,
)
from dluxshera.inference.run_artifacts import save_run


def test_quadratic_precond_vectors_shapes_and_values():
    coeffs = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    def loss(theta):
        return 0.5 * np.sum(coeffs * theta**2)

    theta0 = np.array([1.0, -0.5, 0.25], dtype=np.float32)
    cfg = PreconditioningConfig(eps=1e-6, lr_clip=(0.0, 10.0), base_lr=0.5)

    outputs = compute_precond_vectors(
        loss_fn=loss,
        theta0=theta0,
        method_cfg=cfg,
    )

    for key in ("lr_vec", "precond", "curv_diag"):
        assert key in outputs
        assert outputs[key].shape == theta0.shape
        assert np.all(np.isfinite(outputs[key]))

    expected_curv = (coeffs * theta0) ** 2
    np.testing.assert_allclose(outputs["curv_diag"], expected_curv, rtol=1e-5)

    expected_precond = 1.0 / np.sqrt(expected_curv + cfg.eps)
    np.testing.assert_allclose(outputs["precond"], expected_precond, rtol=1e-5)

    expected_lr = expected_precond * cfg.base_lr
    assert outputs["lr_vec"].min() >= cfg.lr_clip[0]
    assert outputs["lr_vec"].max() <= cfg.lr_clip[1]
    np.testing.assert_allclose(outputs["lr_vec"], expected_lr, rtol=1e-5)


def test_index_map_mismatch_raises():
    theta0 = np.ones(2, dtype=np.float32)
    cfg = PreconditioningConfig()
    bad_index_map = {"entries": [{"name": "a", "start": 0, "stop": 1, "shape": [1]}]}

    with pytest.raises(ValueError):
        compute_precond_vectors(
            loss_fn=lambda th: np.sum(th**2),
            theta0=theta0,
            method_cfg=cfg,
            index_map=bad_index_map,
        )


def test_artifact_writing(tmp_path: Path):
    run_dir = tmp_path / "run"
    trace = {"loss": np.array([1.0]), "theta": np.zeros((1, 3))}
    meta = {"theta": {"dim": 3}}
    summary = {}

    precond = {"lr_vec": np.ones(3), "precond": np.full(3, 0.5)}
    curvature = {"curv_diag": np.arange(3)}

    save_run(
        run_dir,
        trace=trace,
        meta=meta,
        summary=summary,
        precond=precond,
        curvature=curvature,
    )

    with np.load(run_dir / "precond.npz", allow_pickle=False) as npz:
        np.testing.assert_allclose(npz["lr_vec"], precond["lr_vec"])
        np.testing.assert_allclose(npz["precond"], precond["precond"])

    with np.load(run_dir / "curvature.npz", allow_pickle=False) as npz:
        np.testing.assert_allclose(npz["curv_diag"], curvature["curv_diag"])
