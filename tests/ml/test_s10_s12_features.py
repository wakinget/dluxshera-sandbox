from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dluxshera.ml import (
    IntensityScaler,
    PairPolicy,
    PairSampler,
    build_science_mode_weights,
    build_science_eigenbasis,
    build_science_eigenbasis_from_source,
    build_s10_nominal_physical_fim_source,
    expand_study_run_plan,
    generate_frozen_pair_manifest,
    generate_split_registry,
    load_sample_catalog,
    load_study_prescription,
    pair_noise_side_seeds,
    resolve_study_experiment_config,
    science_eigenbasis_content_sha256,
)
import dluxshera.ml.eigenbasis as eigenbasis_module
from dluxshera.ml.noise import NoiseConfig, apply_pair_noise
from tests.ml.test_catalog_splits_pairs import _write_prepared_fixture


ROOT = Path("work/experiments/ml")


def _catalog_registry_sampler(tmp_path: Path) -> tuple:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    registry = generate_split_registry(
        catalog,
        seed=7,
        science_fractions={"train": 1.0, "validation": 0.0, "test": 0.0},
        nuisance_fractions={"train": 1.0, "validation": 0.0, "test": 0.0},
    )
    policy = PairPolicy(
        family_weights={"A": 1.0},
        same_pair_id=True,
        min_fisher_distance=0.5,
        max_fisher_distance=3.0,
        include_reverse=True,
        max_sampling_attempts=4000,
    )
    return catalog, registry, PairSampler(catalog, registry, policy)


def test_s10_eigenbasis_identity_and_coordinate_ordering(tmp_path: Path) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    basis = build_science_eigenbasis(
        artifact_id="BASIS-v1",
        parameter_labels=catalog.parameter_labels,
        curvature_matrix=[[4.0, 1.0], [1.0, 2.0]],
        fisher_scales=catalog.fisher_sigmas,
        source_provenance={"source": "unit-test"},
    )
    assert basis.coordinate_convention == "delta_z_science = z_B - z_A in prepared-catalog parameter order"
    assert basis.parameter_labels == catalog.parameter_labels
    assert basis.eigenvalues[0] > basis.eigenvalues[1]
    assert basis.content_identity["sha256"] == science_eigenbasis_content_sha256(basis)
    basis.validate_catalog(catalog)

    wrong = build_science_eigenbasis(
        artifact_id="WRONG-v1",
        parameter_labels=tuple(reversed(catalog.parameter_labels)),
        curvature_matrix=[[4.0, 0.0], [0.0, 2.0]],
    )
    with pytest.raises(ValueError, match="parameter_labels"):
        wrong.validate_catalog(catalog)


def test_s10_weighted_loss_and_floor_cap_behavior(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from dluxshera.ml.losses import ScienceLossHelper

    basis = build_science_eigenbasis(
        artifact_id="BASIS-v1",
        parameter_labels=["strong", "weak"],
        curvature_matrix=[[100.0, 0.0], [0.0, 1.0e-12]],
        fisher_scales=[1.0, 1.0],
    )
    strong = build_science_mode_weights(
        basis.eigenvalues,
        mode="strong_mode_weighted",
        strength=0.5,
        eigenvalue_floor=1.0e-6,
    )
    weak = build_science_mode_weights(
        basis.eigenvalues,
        mode="weak_mode_weighted",
        strength=1.0,
        eigenvalue_floor=1.0e-6,
        weight_cap=3.0,
    )
    assert np.mean(strong) == pytest.approx(1.0)
    assert np.mean(weak) == pytest.approx(1.0)
    assert strong[0] > strong[1]
    assert weak[1] > weak[0]
    assert np.all(np.isfinite(weak))

    helper = ScienceLossHelper(
        {"mode": "weak_mode_weighted", "strength": 1.0, "eigenvalue_floor": 1.0e-6, "weight_cap": 3.0},
        eigenbasis=basis,
    )
    pred = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    target = torch.zeros_like(pred)
    loss, diag = helper(pred, target)
    assert loss.item() != pytest.approx(diag["ordinary_mse"].item())
    assert "weak_mode_mse" in diag


def test_s10_physical_theta_fim_is_transformed_to_fisher_scaled_z(tmp_path: Path) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    source = {
        "coordinate_space": "physical_theta",
        "parameter_labels": list(catalog.parameter_labels),
        "curvature_matrix": [[4.0, 0.25], [0.25, 0.25]],
        "nominal_provenance": {"source": "unit-test"},
        "weighting_variance_convention": {"fim_diagonal_definition": "1/sigma^2"},
    }
    basis = build_science_eigenbasis_from_source(
        artifact_id="BASIS-v1",
        source=source,
        catalog=catalog,
    )
    np.testing.assert_allclose(basis.curvature_matrix, [[1.0, 0.25], [0.25, 1.0]])
    assert basis.source_matrix_coordinate_space == "physical_theta"
    assert basis.eigenbasis_coordinate_space == "prepared_v4_fisher_scaled_science_delta"
    assert basis.physical_fim_identity["sha256"]
    assert basis.transformed_fz_identity["sha256"]


def test_s10_physical_fim_rejects_incompatible_prepared_fisher_scales(tmp_path: Path) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    source = {
        "coordinate_space": "physical_theta",
        "parameter_labels": list(catalog.parameter_labels),
        "curvature_matrix": [[1.0, 0.0], [0.0, 0.25]],
    }
    with pytest.raises(ValueError, match="diagonal-derived Fisher sigmas"):
        build_science_eigenbasis_from_source(
            artifact_id="BASIS-v1",
            source=source,
            catalog=catalog,
        )


def test_s10_source_fim_rejects_missing_or_unknown_coordinate_space(tmp_path: Path) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    base = {
        "parameter_labels": list(catalog.parameter_labels),
        "curvature_matrix": [[4.0, 0.0], [0.0, 0.25]],
    }
    with pytest.raises(ValueError, match="coordinate_space"):
        build_science_eigenbasis_from_source(
            artifact_id="BASIS-v1",
            source=base,
            catalog=catalog,
        )
    with pytest.raises(ValueError, match="Unsupported science FIM coordinate_space"):
        build_science_eigenbasis_from_source(
            artifact_id="BASIS-v1",
            source={**base, "coordinate_space": "ambiguous"},
            catalog=catalog,
        )


def test_s10_source_fim_rejects_materially_indefinite_matrix(tmp_path: Path) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    source = {
        "coordinate_space": "physical_theta",
        "parameter_labels": list(catalog.parameter_labels),
        "curvature_matrix": [[4.0, 5.0], [5.0, 0.25]],
    }
    with pytest.raises(ValueError, match="materially indefinite"):
        build_science_eigenbasis_from_source(
            artifact_id="BASIS-v1",
            source=source,
            catalog=catalog,
        )


def test_s10_nominal_source_records_physical_and_z_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    f_theta = np.asarray([[4.0, 0.25], [0.25, 0.25]], dtype=np.float64)

    def fake_compute(catalog):
        return {
            "fim_theta": f_theta,
            "parameter_labels": list(catalog.parameter_labels),
            "theta_ref": [0.0, 0.0],
            "index_map": {"entries": []},
            "image_shape": [8, 8],
            "system_cfg": {"preset": "unit-test"},
            "loss_convention": {
                "maker": "dluxshera.inference.optimization.make_binder_nll_fn",
                "fim": "dluxshera.inference.optimization.fim_theta",
                "var": "data",
                "noise_model": "gaussian",
                "reduce": "sum",
            },
            "source_implementation": "unit-test",
            "script_version": "unit-test",
            "git_info": {"commit": "unit-test"},
        }

    monkeypatch.setattr(
        eigenbasis_module,
        "_compute_s10_nominal_full_physical_fim",
        fake_compute,
    )
    source = build_s10_nominal_physical_fim_source(catalog=catalog)
    assert source["coordinate_space"] == "physical_theta"
    assert source["diagonal_fisher_scale_compatibility"]["status"] == "PASS"
    assert source["curvature_matrix"][0][1] == pytest.approx(0.25)
    assert source["diagnostics"]["transformed_fz"]["max_off_diagonal_abs_correlation"] > 0.0
    basis = build_science_eigenbasis_from_source(
        artifact_id="BASIS-v1",
        source=source,
        catalog=catalog,
    )
    np.testing.assert_allclose(basis.curvature_matrix, [[1.0, 0.25], [0.25, 1.0]])
    assert basis.physical_fim_identity["coordinate_space"] == "physical_theta"
    assert basis.transformed_fz_identity["coordinate_space"] == "prepared_v4_fisher_scaled_science_delta"
    assert basis.fisher_scales.tolist() == pytest.approx(catalog.fisher_sigmas.tolist())
    strong = build_science_mode_weights(
        basis.eigenvalues,
        mode="strong_mode_weighted",
        strength=0.5,
    )
    weak = build_science_mode_weights(
        basis.eigenvalues,
        mode="weak_mode_weighted",
        strength=0.5,
    )
    assert not np.allclose(strong, np.ones_like(strong))
    assert not np.allclose(weak, np.ones_like(weak))


def test_s11_antisymmetry_and_identity_losses() -> None:
    torch = pytest.importorskip("torch")
    from dluxshera.ml.losses import PairConsistencyConfig, pair_consistency_losses

    class _LinearPairModel(torch.nn.Module):
        def __init__(self, bias: float = 0.0) -> None:
            super().__init__()
            self.bias = float(bias)

        def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
            delta = (image_b - image_a).mean(dim=(1, 2, 3), keepdim=False)
            return torch.stack([delta + self.bias, 2.0 * delta + self.bias], dim=1)

    a = torch.zeros((3, 1, 4, 4), dtype=torch.float32)
    b = torch.ones((3, 1, 4, 4), dtype=torch.float32)
    antisymmetric = pair_consistency_losses(
        _LinearPairModel(),
        a,
        b,
        config=PairConsistencyConfig(antisymmetry_weight=1.0, identity_weight=1.0),
    )
    assert antisymmetric["antisymmetry"].item() == pytest.approx(0.0)
    assert antisymmetric["identity"].item() == pytest.approx(0.0)

    biased = pair_consistency_losses(
        _LinearPairModel(bias=0.5),
        a,
        b,
        config=PairConsistencyConfig(antisymmetry_weight=1.0, identity_weight=1.0),
    )
    assert biased["antisymmetry"].item() > 0.0
    assert biased["identity"].item() > 0.0


def test_s11_identity_record_preserves_metadata_semantics(tmp_path: Path) -> None:
    catalog, registry, _ = _catalog_registry_sampler(tmp_path)
    identity_policy = PairPolicy(
        family_weights={"I": 1.0},
        same_pair_id=True,
        allow_identity_pairs=True,
    )
    sampler = PairSampler(catalog, registry, identity_policy)
    record = sampler.sample_pair(np.random.default_rng(4))
    assert record.sample_a_id == record.sample_b_id
    assert record.science_a_id == record.science_b_id
    assert record.nuisance_a_id == record.nuisance_b_id
    assert record.dataset_family_a == record.dataset_family_b
    assert record.pair_id_a == record.pair_id_b
    np.testing.assert_allclose(record.target_delta_z, [0.0, 0.0])
    np.testing.assert_allclose(record.nuisance_delta, [0.0, 0.0])


def test_s12_dynamic_noise_determinism_and_independent_views(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from dluxshera.ml.torch_data import DynamicPairDataset

    image_a = np.full((5, 5), 100.0, dtype=np.float32)
    image_b = np.full((5, 5), 120.0, dtype=np.float32)
    cfg = NoiseConfig(
        enabled=True,
        apply_to="observation",
        noise_model="shera_observation",
        photon_noise=True,
        seed=123,
        training_dynamic=True,
        negative_policy="clip",
    )
    _, first = apply_pair_noise(image_a, image_b, cfg, pair_record_id="pair", dynamic_seed_offset=1)
    _, again = apply_pair_noise(image_a, image_b, cfg, pair_record_id="pair", dynamic_seed_offset=1)
    _, different = apply_pair_noise(image_a, image_b, cfg, pair_record_id="pair", dynamic_seed_offset=2)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, different)

    catalog, _, sampler = _catalog_registry_sampler(tmp_path)
    dataset = DynamicPairDataset(
        catalog=catalog,
        sampler=sampler,
        pairs_per_epoch=2,
        seed=9,
        scaler=IntensityScaler(mode="global_max_abs", scale=100.0),
        noise_config=cfg,
        second_noise_view=True,
    )
    item = dataset[0]
    assert "image_b_noise_view2" in item
    assert not torch.equal(item["image_b"], item["image_b_noise_view2"])


def test_s12_noise_is_injected_before_scaling(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from dluxshera.ml.torch_data import DynamicPairDataset

    catalog, _, sampler = _catalog_registry_sampler(tmp_path)
    cfg = NoiseConfig(
        enabled=True,
        apply_to="observation",
        noise_model="legacy_numpy",
        photon_noise=False,
        read_noise=True,
        read_noise_sigma=1.0,
        seed=55,
        training_dynamic=False,
    )
    dataset = DynamicPairDataset(
        catalog=catalog,
        sampler=sampler,
        pairs_per_epoch=2,
        seed=9,
        scaler=IntensityScaler(mode="global_max_abs", scale=10.0),
        noise_config=cfg,
    )
    item = dataset[0]
    record_id = str(item["pair_record_id"])
    clean_b = catalog.image_reader().get(int(item["sample_b_id"].split("_")[-1]))
    _, expected_b = apply_pair_noise(
        clean_b,
        clean_b,
        cfg,
        pair_record_id=record_id,
        dynamic_seed_offset=0,
    )
    np.testing.assert_allclose(item["image_b"].numpy()[0], expected_b / 10.0)


def test_s12_frozen_noisy_validation_is_reproducible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    from dluxshera.ml.torch_data import PairManifestDataset

    catalog, registry, sampler = _catalog_registry_sampler(tmp_path)
    manifest = generate_frozen_pair_manifest(
        catalog,
        registry,
        policy=sampler.policy,
        artifact_id="VAL-v1",
        seed=3,
        pairs_per_slice=2,
    )
    cfg = NoiseConfig(
        enabled=True,
        apply_to="observation",
        noise_model="shera_observation",
        photon_noise=True,
        seed=12031,
        training_dynamic=False,
        negative_policy="clip",
    )
    first = PairManifestDataset(
        catalog=catalog,
        pair_manifest=manifest,
        scaler=IntensityScaler(mode="global_max_abs", scale=100.0),
        noise_config=cfg,
    )
    second = PairManifestDataset(
        catalog=catalog,
        pair_manifest=manifest,
        scaler=IntensityScaler(mode="global_max_abs", scale=100.0),
        noise_config=cfg,
    )
    torch.testing.assert_close(first[0]["image_b"], second[0]["image_b"])
    assert manifest.manifest["content_identity"]["sha256"]

    import dluxshera.ml.torch_data as torch_data

    observed: list[tuple[str | None, int]] = []

    def fake_apply_pair_noise(
        image_a,
        image_b,
        noise_config=None,
        *,
        pair_record_id=None,
        dynamic_seed_offset=0,
    ):
        observed.append((pair_record_id, dynamic_seed_offset))
        return image_a, image_b

    monkeypatch.setattr(torch_data, "apply_pair_noise", fake_apply_pair_noise)
    seed_probe = PairManifestDataset(
        catalog=catalog,
        pair_manifest=manifest,
        scaler=IntensityScaler(mode="global_max_abs", scale=100.0),
        noise_config=cfg,
    )
    _ = seed_probe[0]
    _ = seed_probe[1]
    for index, (pair_record_id, offset) in enumerate(observed):
        assert offset == index
        assert pair_noise_side_seeds(
            cfg,
            pair_record_id=pair_record_id,
            dynamic_seed_offset=index,
        ) == pair_noise_side_seeds(
            cfg,
            pair_record_id=manifest.records[index].pair_record_id,
            dynamic_seed_offset=index,
        )


def test_s10_s12_study_expansion_and_shared_reference_accounting() -> None:
    s10 = load_study_prescription(ROOT / "s10" / "study.yaml")
    s11 = load_study_prescription(ROOT / "s11" / "study.yaml")
    s12 = load_study_prescription(ROOT / "s12" / "study.yaml")
    rows = []
    for study in (s10, s11, s12):
        rows.extend(expand_study_run_plan(study))
    assert len(rows) == 21
    assert sum(1 for row in rows if row.experiment_id == "S10-E01") == 3
    assert all(resolve_study_experiment_config(s10, experiment_id="S10-E01", run_id=f"S10-E01-R00{i}")["evaluate_test"] is False for i in (1, 2, 3))
    for study in (s11, s12):
        assert study["shared_reference"]["experiment_id"] == "S10-E01"
        assert study["shared_reference"]["retrain_in_this_study"] is False
    assert {row.study_id: row.artifact_lock_id for row in rows if row.run_id.endswith("R001")}["S10"] == "S10-ARTIFACT-LOCK-v1"
    s12_e01 = resolve_study_experiment_config(s12, experiment_id="S12-E01")
    assert s12_e01["validation_noise"]["training_dynamic"] is False
    assert s12_e01["noise"]["noise_model"] == "shera_observation"
    assert s12["auxiliary_artifacts"]["noisy_validation_recipe"]["underlying_validation_artifact"] == "S12-VALIDATION-PAIRS-v1"
