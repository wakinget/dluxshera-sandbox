"""Exact equality including fallback branches and real worker serialization."""
import json
from dataclasses import replace
import numpy as np
import pytest
import torch
from dluxshera.ml.noise import apply_pair_noise, apply_pair_noise_batch
from tests.ml.test_s12_dataloader_runtime import (
    S12_TRAINING_NOISE, _dynamic_dataset, _safe_loader,
)


@pytest.mark.parametrize("apply_to", ["observation", "model", "both"])
@pytest.mark.parametrize("seed", [11, 23, 47])
def test_noise_exact(apply_to, seed):
    cfg = replace(S12_TRAINING_NOISE, apply_to=apply_to, seed=seed)
    # Mixed threshold branches, equality at threshold, zeros and clipped negatives.
    arrays = [np.full((32, 32), x, np.float32) for x in (-1, 0, 12, 100, 101, 1000)]
    pairs = [(x, x) for x in arrays]
    records = [(str(i), 1_000_003 + i) for i in range(len(pairs))]
    expected = [apply_pair_noise(a, b, cfg, pair_record_id=r, dynamic_seed_offset=o)
                for (a, b), (r, o) in zip(pairs, records)]
    for width in (1, 2, 4, 6):
        actual = sum([apply_pair_noise_batch(pairs[i:i+width], cfg, records[i:i+width])
                      for i in range(0, len(pairs), width)], [])
        for a, b in zip(actual, expected):
            assert all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("second", [False, True])
@pytest.mark.parametrize("workers,method", [(0, None), (4, "forkserver"), (2, "spawn")])
def test_dataset_exact(tmp_path, second, workers, method):
    from dluxshera.ml.torch_data import resolve_dataloader_context
    ds = _dynamic_dataset(tmp_path, second_noise_view=second, pairs_per_epoch=10, image_size=32)
    context, _, _ = resolve_dataloader_context(method, requires_safe_workers=True, num_workers=workers)
    for epoch in (0, 2):
        ds.set_epoch(epoch)
        ds.noise_implementation = "reference"
        expected = [ds[i] for i in range(len(ds))]
        ds.noise_implementation = "batched"
        actual = []
        loader = _safe_loader(ds, num_workers=workers, batch_size=4, context=context)
        if workers:
            loader.timeout = 90
        for batch in loader:
            for i in range(len(batch["pair_record_id"])):
                actual.append({key: value[i] for key, value in batch.items()})
        for a, b in zip(actual, expected):
            assert a.keys() == b.keys()
            for key in a:
                if isinstance(a[key], torch.Tensor):
                    assert torch.equal(a[key], b[key]), key
                else:
                    assert a[key] == b[key], key


@pytest.mark.parametrize("options", [dict(enabled=False), dict(photon_noise=False),
    dict(read_noise=True, detector_model="GSENSE2020BSI"),
    dict(dark_current=True, detector_model="GSENSE2020BSI", exposure_time_s=1.0)])
def test_fallback_exact(options):
    cfg = replace(S12_TRAINING_NOISE, **options)
    image = np.full((32, 32), 5, np.float32)
    expected = apply_pair_noise(image, image, cfg, pair_record_id="f", dynamic_seed_offset=2)
    actual = apply_pair_noise_batch([(image, image)], cfg, [("f", 2)])[0]
    assert all(np.array_equal(a, b) for a, b in zip(expected, actual))


def test_photon_statistics():
    cfg = S12_TRAINING_NOISE
    for mean in (25, 1000):
        image = np.full((160, 160), mean, np.float32)
        output = apply_pair_noise_batch([(image, image)], cfg, [("statistics", 0)])[0][1]
        assert abs(output.mean() - mean) < 6 * np.sqrt(mean / image.size)
        assert abs(output.var() / mean - 1) < 0.06


def test_manifest_exact(tmp_path):
    from types import SimpleNamespace
    from dluxshera.ml.torch_data import PairManifestDataset
    ds = _dynamic_dataset(tmp_path, pairs_per_epoch=10)
    records = [ds._record_and_offset(i)[0] for i in range(10)]
    frozen = PairManifestDataset(catalog=ds.catalog, pair_manifest=SimpleNamespace(records=records),
        scaler=ds.scaler, noise_config=replace(S12_TRAINING_NOISE, training_dynamic=False, seed=12031),
        noise_implementation="batched")
    for indices in ([0, 1, 2, 3], [4, 5, 6], [7, 8, 9]):
        for index, actual in zip(indices, frozen.__getitems__(indices)):
            expected = frozen[index]
            for key in actual:
                if isinstance(actual[key], torch.Tensor):
                    assert torch.equal(actual[key], expected[key]), key
                else:
                    assert actual[key] == expected[key]


def test_runtime_selection_and_submission(monkeypatch):
    from argparse import Namespace
    from work.experiments.ml.hpc.submit_study_run import _dataloader_smoke_env
    from dluxshera.ml.hpc import SUBMITTED_ENV_KEYS
    from dluxshera.ml.training import _runtime_info
    env = _dataloader_smoke_env(Namespace(noise_implementation="batched",
        dataloader_smoke=True, dataloader_smoke_batches=4, dataloader_smoke_timeout=900))
    assert env["DLUXSHERA_NOISE_IMPLEMENTATION"] == "batched"
    assert "DLUXSHERA_NOISE_IMPLEMENTATION" in SUBMITTED_ENV_KEYS
    monkeypatch.setenv("DLUXSHERA_NOISE_IMPLEMENTATION", "batched")
    assert _runtime_info(torch.device("cpu"))["noise_implementation"] == "batched"


def _equivalence_request(dataset, monkeypatch, *, batch_size=2, **overrides):
    """Build a ProbeRequest bound to an in-memory dataset, skipping artifacts."""
    from work.experiments.ml.hpc.diagnose_ml_dataloader import ProbeRequest
    request = object.__new__(ProbeRequest)
    fields = dict(batches=4, warmup_batches=1, loader="train", profile=False)
    fields.update(overrides)
    for name, value in fields.items():
        object.__setattr__(request, name, value)
    monkeypatch.setattr(ProbeRequest, "_build", lambda self: dict(
        dataset=dataset, loader=_safe_loader(dataset, num_workers=0), batch_size=batch_size))
    return request


def test_diagnostic_profile_and_equivalence(tmp_path, monkeypatch):
    ds = _dynamic_dataset(tmp_path, pairs_per_epoch=8)
    ds.noise_implementation = "batched"
    request = _equivalence_request(ds, monkeypatch, profile=True)
    report = request.verify_equivalence()
    assert report["exact"] is True
    assert report["records"] == 8
    assert report["loader"] == "train"
    assert report["complete_manifest_compared"] is True
    assert report["partitions"][0]["compared_records"] == 8
    assert report["partitions"][0]["expected_records"] == 8
    assert report["partitions"][0]["missing_indices"] == []
    assert report["partitions"][0]["duplicated_indices"] == []
    result = request.run_loader_probe()
    assert result["batches_observed"] == 4
    assert "resident_noise_one_view" in result
    assert "profile_warm_four_batches" in result


def test_equivalence_rejects_truncated_optimized_batch(tmp_path, monkeypatch):
    """A short __getitems__ result must fail, not silently compare a prefix.

    This is the regression that motivated replacing the original zip()-based
    comparison: zip(short_actual, full_expected) stops at the shorter sequence
    and reports success, so a dropped record looked exactly like equivalence.
    """
    from work.experiments.ml.hpc.diagnose_ml_dataloader import (
        EquivalenceMismatch,
        compare_noise_implementations,
    )
    ds = _dynamic_dataset(tmp_path, pairs_per_epoch=8)
    ds.noise_implementation = "batched"
    indices = list(range(8))
    # Control: the unmodified dataset is exact under this partition.
    assert compare_noise_implementations(ds, indices=indices, batch_size=4)["exact"] is True

    full = type(ds).__getitems__
    monkeypatch.setattr(type(ds), "__getitems__",
                        lambda self, idx: full(self, idx)[:-1])
    with pytest.raises(EquivalenceMismatch, match="truncated or"):
        compare_noise_implementations(ds, indices=indices, batch_size=4)


def test_equivalence_rejects_duplicated_and_mismatched_records(tmp_path, monkeypatch):
    """Duplicate records and corrupted field values must both fail."""
    from work.experiments.ml.hpc.diagnose_ml_dataloader import (
        EquivalenceMismatch,
        compare_noise_implementations,
    )
    ds = _dynamic_dataset(tmp_path, pairs_per_epoch=8)
    ds.noise_implementation = "batched"
    full = type(ds).__getitems__

    # Right count, wrong coverage: the first record is returned twice.
    monkeypatch.setattr(type(ds), "__getitems__",
                        lambda self, idx: [full(self, idx)[0], *full(self, idx)[:-1]])
    with pytest.raises(EquivalenceMismatch, match="identity differs"):
        compare_noise_implementations(ds, indices=list(range(8)), batch_size=4)

    def corrupt(self, idx):
        items = full(self, idx)
        items[-1]["image_b"] = items[-1]["image_b"] + 1.0
        return items

    monkeypatch.setattr(type(ds), "__getitems__", corrupt)
    with pytest.raises(EquivalenceMismatch, match="not exact"):
        compare_noise_implementations(ds, indices=list(range(8)), batch_size=4)


def test_equivalence_reports_partial_final_batch_for_each_partition(tmp_path, monkeypatch):
    """A second batch size must be compared independently, partial tail included.

    S12's frozen manifest holds 1,024 records: 32 batches of 32 divide evenly,
    so only an indivisible width such as 17 exercises the final partial batch.
    """
    ds = _dynamic_dataset(tmp_path, pairs_per_epoch=10)
    ds.noise_implementation = "batched"
    request = _equivalence_request(
        ds, monkeypatch, batch_size=4,
        equivalence_full_manifest=True, equivalence_batch_sizes=(3,))
    report = request.verify_equivalence()
    assert report["exact"] is True
    assert report["complete_manifest_compared"] is True
    assert [part["batch_size"] for part in report["partitions"]] == [4, 3]
    for part in report["partitions"]:
        assert part["compared_records"] == 10
        assert part["expected_records"] == 10
        assert part["final_batch_is_partial"] is True
        assert part["mismatched_field_count"] == 0
    assert [part["batches"] for part in report["partitions"]] == [3, 4]
    assert "not a multiprocessing equivalence test" in report["scope"]


def test_equivalence_partial_sample_is_labelled_as_partial(tmp_path, monkeypatch):
    """Sampling fewer records than the dataset holds must not read as complete."""
    ds = _dynamic_dataset(tmp_path, pairs_per_epoch=10)
    ds.noise_implementation = "batched"
    request = _equivalence_request(ds, monkeypatch, batch_size=2, batches=2)
    report = request.verify_equivalence()
    assert report["records"] == 4
    assert report["dataset_records_available"] == 10
    assert report["complete_manifest_compared"] is False
    assert "partial sample" in report["coverage"]


def test_equivalence_restores_the_original_implementation(tmp_path):
    """The comparison must not leave the dataset switched to batched."""
    from work.experiments.ml.hpc.diagnose_ml_dataloader import compare_noise_implementations
    ds = _dynamic_dataset(tmp_path, pairs_per_epoch=4)
    ds.noise_implementation = "reference"
    compare_noise_implementations(ds, indices=[0, 1, 2, 3], batch_size=2)
    assert ds.noise_implementation == "reference"
    with pytest.raises(ValueError, match="unique"):
        compare_noise_implementations(ds, indices=[0, 0], batch_size=2)


def test_backend_guard(monkeypatch):
    import jax
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    image = np.ones((32, 32), np.float32)
    with pytest.raises(ValueError, match="requires JAX CPU"):
        apply_pair_noise_batch([(image, image)], S12_TRAINING_NOISE, [("p", 0)])


@pytest.mark.parametrize("previous,current,allowed", [
    ("batched", "batched", True), ("reference", "reference", True),
    ("reference", "batched", False), ("batched", "reference", False),
    (None, "reference", True), (None, "batched", False)])
def test_resume_implementation(previous, current, allowed):
    from dluxshera.ml.training import _validate_resume_noise_implementation
    checkpoint = {} if previous is None else {"runtime_provenance": {"noise_implementation": previous}}
    if allowed:
        _validate_resume_noise_implementation(checkpoint, current)
    else:
        with pytest.raises(ValueError, match="Resume noise implementation mismatch"):
            _validate_resume_noise_implementation(checkpoint, current)


def _train_batched_probe():
    import tempfile
    import os
    import json
    from pathlib import Path
    from argparse import Namespace
    from work.experiments.ml.hpc.submit_study_run import _dataloader_smoke_env
    from tests.ml.test_models_training import _tiny_training_inputs
    from tests.ml.test_catalog_splits_pairs import _write_prepared_fixture
    from dluxshera.ml.training import train_pairwise_correction
    os.environ.update({k: v for k, v in _dataloader_smoke_env(Namespace(
        noise_implementation="batched", dataloader_smoke=True,
        dataloader_smoke_batches=2, dataloader_smoke_timeout=90)).items() if v is not None})
    with tempfile.TemporaryDirectory() as root:
        path = Path(root)
        prepared, split, config = _tiny_training_inputs(path)
        # Replace the toy images while preserving fixture metadata/split identity.
        import shutil
        shutil.rmtree(prepared)  # disposable fixture created immediately above
        _write_prepared_fixture(prepared, image_size=32)
        from dluxshera.ml import load_sample_catalog, generate_split_registry, write_split_registry
        registry = generate_split_registry(load_sample_catalog(prepared), seed=7,
            science_fractions={"train": 1.0}, nuisance_fractions={"train": 1.0})
        write_split_registry(split, registry)
        config["training"].update(epochs=1, num_workers=2, pairs_per_epoch=8)
        config["noise"] = S12_TRAINING_NOISE.to_dict()
        config["validation_noise"] = S12_TRAINING_NOISE.to_dict()
        config["pair_consistency"] = {"noise_consistency_weight": 0.05}
        from work.experiments.ml.hpc.preflight_ml_gpu import _dataloader_smoke_in_process
        from dluxshera.ml import PairPolicy
        preflight = _dataloader_smoke_in_process(config=config, catalog=load_sample_catalog(prepared),
            split_registry=registry, pair_policy=PairPolicy.from_dict(config.get("pair_policy", {})),
            scaler_path=None, enabled=True, batches=2, timeout_seconds=90)
        assert preflight["noise_implementation"] == "batched"
        assert preflight["batches_observed"] == 2
        train_pairwise_correction(config=config, prepared_root=prepared,
            split_registry_path=split, output_dir=path / "run")
        checkpoint = torch.load(path / "run/checkpoint_last.pt", weights_only=True)
        assert checkpoint["runtime_provenance"]["noise_implementation"] == "batched"
        assert "noise_implementation" not in checkpoint["training_identity"]
        assert (path / "run/history.csv").exists()
        return {"trained": True}


def test_batched_cpu_training_integration():
    from dluxshera.ml.training import run_supervised
    result = run_supervised(_train_batched_probe, timeout_seconds=180, label="batched_training")
    assert result["status"] == "ok", result


# ---------------------------------------------------------------------------
# Skipped-preflight regression
#
# The skipped early-return path reported dataset.noise_implementation before
# any dataset existed, so a noiseless study or --no-dataloader-smoke raised
# UnboundLocalError instead of returning a skip report.  S10/S11 are noiseless,
# so this would have failed their preflight outright.
# ---------------------------------------------------------------------------
def _preflight_module():
    import sys
    from pathlib import Path
    from unittest.mock import patch
    hpc = Path(__file__).resolve().parents[2] / "work/experiments/ml/hpc"
    with patch.object(sys, "path", [str(hpc), *sys.path]):
        import preflight_ml_gpu
    return preflight_ml_gpu


def _smoke(tmp_path, monkeypatch, *, noise, enabled, implementation=None, env=None):
    from dluxshera.ml import PairPolicy
    from tests.ml.test_s12_dataloader_runtime import _catalog_registry_sampler
    monkeypatch.delenv("DLUXSHERA_NOISE_IMPLEMENTATION", raising=False)
    if env is not None:
        monkeypatch.setenv("DLUXSHERA_NOISE_IMPLEMENTATION", env)
    # The prepared fixture refuses to overwrite itself, so each call to this
    # helper within one test gets its own root.
    root = tmp_path / f"case{next(_smoke.counter)}"
    catalog, registry, sampler = _catalog_registry_sampler(root)
    config = {"training": {"num_workers": 0, "batch_size": 2}}
    if noise is not None:
        config["noise"] = noise.to_dict()
    return _preflight_module()._dataloader_smoke_in_process(
        config=config, catalog=catalog, split_registry=registry,
        pair_policy=PairPolicy.from_dict(config.get("pair_policy", {})),
        scaler_path=None, enabled=enabled, batches=2, timeout_seconds=90,
        noise_implementation_override=implementation)


_smoke.counter = __import__("itertools").count()


def test_preflight_skips_cleanly_for_a_noiseless_study(tmp_path, monkeypatch):
    """S10/S11-style noiseless configuration with default preflight behaviour."""
    outcome = _smoke(tmp_path, monkeypatch, noise=None, enabled=None)
    assert outcome["status"] == "skipped"
    assert outcome["noise_enabled"] is False
    assert outcome["noise_implementation"] == "reference"
    assert "noise augmentation disabled" in outcome["reason"]


def test_preflight_skips_cleanly_when_explicitly_disabled(tmp_path, monkeypatch):
    outcome = _smoke(tmp_path, monkeypatch, noise=S12_TRAINING_NOISE, enabled=False)
    assert outcome["status"] == "skipped"
    # Noise is configured; only the smoke test was turned off.
    assert outcome["noise_enabled"] is True
    assert outcome["noise_implementation"] == "reference"
    assert "--no-dataloader-smoke" in outcome["reason"]


@pytest.mark.parametrize("implementation", ["reference", "batched"])
def test_preflight_skip_reports_selected_implementation(tmp_path, monkeypatch, implementation):
    """Provenance must be correct on the skipped path too, for both strategies."""
    explicit = _smoke(tmp_path, monkeypatch, noise=None, enabled=None,
                      implementation=implementation)
    assert explicit["status"] == "skipped"
    assert explicit["noise_implementation"] == implementation
    from_env = _smoke(tmp_path, monkeypatch, noise=None, enabled=None, env=implementation)
    assert from_env["noise_implementation"] == implementation


@pytest.mark.parametrize("implementation", ["reference", "batched"])
def test_preflight_noisy_run_succeeds_and_reports_implementation(
    tmp_path, monkeypatch, implementation
):
    outcome = _smoke(tmp_path, monkeypatch, noise=S12_TRAINING_NOISE, enabled=True,
                     implementation=implementation)
    assert outcome["status"] == "ok"
    assert outcome["noise_enabled"] is True
    assert outcome["batches_observed"] == 2
    assert outcome["noise_implementation"] == implementation
    assert outcome["noise_implementation_source"] == "explicit override"


def test_preflight_noisy_run_honours_the_environment_default(tmp_path, monkeypatch):
    outcome = _smoke(tmp_path, monkeypatch, noise=S12_TRAINING_NOISE, enabled=True,
                     env="batched")
    assert outcome["status"] == "ok"
    assert outcome["noise_implementation"] == "batched"
    assert "DLUXSHERA_NOISE_IMPLEMENTATION" in outcome["noise_implementation_source"]


def test_preflight_rejects_an_unknown_implementation(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="noise_implementation"):
        _smoke(tmp_path, monkeypatch, noise=None, enabled=None, implementation="typo")


def test_resolve_noise_implementation_precedence(monkeypatch):
    from dluxshera.ml.torch_data import resolve_noise_implementation
    monkeypatch.delenv("DLUXSHERA_NOISE_IMPLEMENTATION", raising=False)
    assert resolve_noise_implementation() == "reference"
    assert resolve_noise_implementation("batched") == "batched"
    monkeypatch.setenv("DLUXSHERA_NOISE_IMPLEMENTATION", "batched")
    assert resolve_noise_implementation() == "batched"
    # An explicit argument wins over the environment.
    assert resolve_noise_implementation("reference") == "reference"
    monkeypatch.setenv("DLUXSHERA_NOISE_IMPLEMENTATION", "")
    assert resolve_noise_implementation() == "reference"
    with pytest.raises(ValueError, match="noise_implementation"):
        resolve_noise_implementation("typo")


def test_runtime_default_and_invalid(tmp_path, monkeypatch):
    from dluxshera.ml.torch_data import PairManifestDataset
    from types import SimpleNamespace
    monkeypatch.delenv("DLUXSHERA_NOISE_IMPLEMENTATION", raising=False)
    assert PairManifestDataset(catalog=None, pair_manifest=SimpleNamespace(records=[])).noise_implementation == "reference"
    with pytest.raises(ValueError, match="noise_implementation"):
        PairManifestDataset(catalog=None, pair_manifest=SimpleNamespace(records=[]), noise_implementation="typo")


# ---------------------------------------------------------------------------
# Noise-regime characterization
# ---------------------------------------------------------------------------
def test_image_statistics_reports_branch_selection(tmp_path, capsys):
    """The branch predicate is min(image) > threshold, not max and not per-pixel.

    S12's 50 ms exposures leave faint PSF outskirts well below 100, so the exact
    Poisson branch is expected to be selected for effectively every image. This
    test pins the predicate the diagnostic reports, so a future reading of
    "fraction above threshold" cannot be mistaken for branch selection.
    """
    import sys
    from unittest.mock import patch
    from pathlib import Path
    from tests.ml.test_catalog_splits_pairs import _write_prepared_fixture
    hpc = Path(__file__).resolve().parents[2] / "work/experiments/ml/hpc"
    with patch.object(sys, "path", [str(hpc), *sys.path]):
        import analyze_image_statistics as stats

    # A faint floor with a bright core: most pixels exceed 100, yet the image
    # minimum does not, so Poisson is selected. This is exactly the case that
    # a max-based or per-pixel reading would get backwards.
    mixed = np.full((64, 64), 500.0, dtype=np.float32)
    mixed[0, 0] = 2.0
    report = stats._summarize([mixed], bright_threshold=100.0, label="mixed")
    assert report["images_selecting_poisson_branch"] == 1
    assert report["images_selecting_gaussian_branch"] == 0
    assert report["poisson_branch_exclusive"] is True
    assert report["pooled_fraction_pixels_above_threshold"] > 0.99
    assert report["fraction_images_every_pixel_above_threshold"] == 0.0
    assert report["count_units"].startswith("1 DN = 1 electron")

    # A uniformly bright image is the only way to reach the Gaussian branch.
    bright = stats._summarize([np.full((8, 8), 500.0, dtype=np.float32)],
                              bright_threshold=100.0, label="bright")
    assert bright["images_selecting_gaussian_branch"] == 1
    assert bright["poisson_branch_exclusive"] is False
    # Strictly greater: equality at the threshold stays on the Poisson branch.
    at_threshold = stats._summarize([np.full((8, 8), 100.0, dtype=np.float32)],
                                    bright_threshold=100.0, label="equal")
    assert at_threshold["images_selecting_poisson_branch"] == 1

    assert stats._summarize([], bright_threshold=100.0, label="none")["images"] == 0
    # Strided sampling must span the catalog rather than take a leading block.
    assert stats._sample_indices(100, 5) == [0, 25, 50, 74, 99]
    assert stats._sample_indices(3, 10) == [0, 1, 2]
    assert stats._sample_indices(10, 0) == []

    prepared = _write_prepared_fixture(tmp_path / "prepared", image_size=32)
    assert stats.main(["--prepared-root", str(prepared), "--samples", "4"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["catalog_samples"]["images"] == 4
    assert payload["sampling"]["strategy"].startswith("evenly strided")


def test_branch_predicate_matches_the_shared_noise_utility():
    """Pin the diagnostic's predicate to the shipped implementation."""
    from pathlib import Path
    source = Path("src/dluxshera/utils/noise.py").read_text(encoding="utf-8")
    assert "if jnp.min(image) > bright_threshold:" in source
    # S12's configured threshold, unchanged.
    from dluxshera.ml import load_study_prescription
    study = load_study_prescription(Path("work/experiments/ml/s12/study.yaml"))
    for experiment in ("S12-E01", "S12-E02"):
        config = study["experiments"][experiment]
        assert float(config["noise"]["bright_threshold"]) == 100.0
        assert float(config["validation_noise"]["bright_threshold"]) == 100.0
        assert config["noise"]["read_noise"] is False
        assert config["noise"]["dark_current"] is False
        assert config["noise"]["detector_model"] is None
