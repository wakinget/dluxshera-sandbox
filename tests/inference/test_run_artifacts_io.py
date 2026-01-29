from pathlib import Path

import numpy as np
import pytest

from dluxshera.inference.run_artifacts import (
    load_checkpoint,
    load_meta,
    load_npz_artifact,
    load_summary,
    load_trace,
    save_run,
)
from dluxshera.params.packing import build_index_map, pack_params
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

    loaded_meta = load_meta(run_dir)
    loaded_summary = load_summary(run_dir)

    assert loaded_meta["run_id"] == meta["run_id"]
    assert loaded_meta["theta"] == meta["theta"]
    assert loaded_summary["final_loss"] == summary["final_loss"]

    for name in ("trace", "meta", "summary"):
        assert name in loaded_meta["manifest"]
        assert name in loaded_summary["manifest"]

    # Optional artifacts should not be created when omitted.
    assert not (run_dir / "signals.npz").exists()
    assert not (run_dir / "grads.npz").exists()
    assert not (run_dir / "metric.npz").exists()
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
        "checkpoint_best": {
            "theta_best": np.array([1.0, 2.0]),
            "best_step": 1,
            "best_loss": 0.5,
        },
        "checkpoint_final": {
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
        artifacts={
            "signals": {"kind": "npz", "content": signals},
            "grads": {"kind": "npz", "content": grads},
            "checkpoint_best": {
                "kind": "npz",
                "content": checkpoints["checkpoint_best"],
                "filename": "best_model.npz",
            },
            "checkpoint_final": {
                "kind": "npz",
                "content": checkpoints["checkpoint_final"],
                "filename": "final_model.npz",
            },
        },
    )

    loaded_signals = load_npz_artifact(run_dir, "signals")
    assert loaded_signals is not None
    np.testing.assert_allclose(loaded_signals["s1"], signals["s1"])

    loaded_summary = load_summary(run_dir)
    for name in ("signals", "grads", "checkpoint_best", "checkpoint_final"):
        assert name in loaded_summary["manifest"]

    assert (run_dir / "best_model.npz").exists()
    assert (run_dir / "final_model.npz").exists()
    assert not (run_dir / "checkpoint_best.npz").exists()
    assert not (run_dir / "checkpoint_final.npz").exists()

    best = load_checkpoint(run_dir, "best")
    np.testing.assert_allclose(best["theta_best"], checkpoints["checkpoint_best"]["theta_best"])
    assert best["best_step"] == checkpoints["checkpoint_best"]["best_step"]
    assert best["best_loss"] == checkpoints["checkpoint_best"]["best_loss"]

    final = load_checkpoint(run_dir, "final")
    np.testing.assert_allclose(final["theta_final"], checkpoints["checkpoint_final"]["theta_final"])
    assert final["final_step"] == checkpoints["checkpoint_final"]["final_step"]
    assert final["final_loss"] == checkpoints["checkpoint_final"]["final_loss"]


def test_build_index_map_matches_pack_params(tmp_path: Path):
    spec = build_inference_spec_basic()
    store = ParameterStore.from_spec_defaults(spec).replace(
        {
            "primary.zernike_coeffs_nm": np.zeros(3),
            "secondary.zernike_coeffs_nm": np.zeros(2),
        }
    )
    store = store.refresh_derived(spec)

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
