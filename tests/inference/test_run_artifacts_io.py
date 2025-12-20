from pathlib import Path

import numpy as np
import pytest

from dluxshera.inference.run_artifacts import (
    build_index_map,
    load_checkpoint,
    load_meta,
    load_summary,
    load_trace,
    save_run,
)
from dluxshera.params.packing import pack_params
from dluxshera.params.spec import build_inference_spec_basic
from dluxshera.params.store import ParameterStore


def test_save_and_load_required_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run"
    trace = {"loss": [1.0, 0.5], "theta": [[0.1, 0.2], [0.15, 0.25]]}
    meta = {"run_id": "abc123", "theta": {"dim": 2}}
    summary = {"final_loss": 0.5}

    save_run(run_dir, trace=trace, meta=meta, summary=summary)

    loaded_trace = load_trace(run_dir)
    np.testing.assert_allclose(loaded_trace["loss"], np.asarray(trace["loss"]))
    np.testing.assert_allclose(loaded_trace["theta"], np.asarray(trace["theta"]))

    assert load_meta(run_dir) == meta
    assert load_summary(run_dir) == summary

    # Optional artifacts should not be created when omitted.
    assert not (run_dir / "signals.npz").exists()
    assert not (run_dir / "grads.npz").exists()
    assert not (run_dir / "curvature.npz").exists()
    assert not (run_dir / "precond.npz").exists()
    assert not (run_dir / "diag_steps.jsonl").exists()
    assert not (run_dir / "checkpoint_best.npz").exists()
    assert not (run_dir / "checkpoint_final.npz").exists()


def test_optional_artifacts_and_checkpoints(tmp_path: Path):
    run_dir = tmp_path / "run_opt"
    trace = {"loss": np.array([1.0]), "theta": np.array([[0.3]])}
    meta = {}
    summary = {}

    signals = {"s1": np.array([0.1, 0.2, 0.3])}
    grads = {"g1": np.array([1.0])}
    checkpoints = {
        "best": {
            "theta_best": np.array([1.0, 2.0]),
            "best_step": 1,
            "best_loss": 0.5,
        },
        "final": {
            "theta_final": np.array([3.0, 4.0]),
            "final_step": 2,
            "final_loss": 0.25,
        },
    }

    save_run(
        run_dir,
        trace=trace,
        meta=meta,
        summary=summary,
        signals=signals,
        grads=grads,
        checkpoints=checkpoints,
    )

    with np.load(run_dir / "signals.npz", allow_pickle=False) as npz:
        np.testing.assert_allclose(npz["s1"], signals["s1"])

    best = load_checkpoint(run_dir, "best")
    np.testing.assert_allclose(best["theta_best"], checkpoints["best"]["theta_best"])
    assert best["best_step"] == checkpoints["best"]["best_step"]
    assert best["best_loss"] == checkpoints["best"]["best_loss"]

    final = load_checkpoint(run_dir, "final")
    np.testing.assert_allclose(final["theta_final"], checkpoints["final"]["theta_final"])
    assert final["final_step"] == checkpoints["final"]["final_step"]
    assert final["final_loss"] == checkpoints["final"]["final_loss"]


def test_build_index_map_matches_pack_params(tmp_path: Path):
    spec = build_inference_spec_basic()
    store = ParameterStore.from_spec_defaults(spec).replace(
        {
            "primary.zernike_coeffs_nm": np.zeros(3),
            "secondary.zernike_coeffs_nm": np.zeros(2),
        }
    )

    theta = pack_params(spec, store)
    index_map = build_index_map(spec, store, theta=theta)

    entries = index_map["entries"]
    assert entries, "IndexMap should have entries"
    assert entries[-1]["stop"] == theta.size

    first_non_empty = next(e for e in entries if e["stop"] > e["start"])
    name = first_non_empty["name"]
    start, stop = first_non_empty["start"], first_non_empty["stop"]

    np.testing.assert_allclose(
        np.asarray(store.get(name)).ravel(),
        np.asarray(theta[start:stop]).ravel(),
    )

    layout_hash = index_map.get("layout_hash")
    assert isinstance(layout_hash, str) and len(layout_hash) == 64
