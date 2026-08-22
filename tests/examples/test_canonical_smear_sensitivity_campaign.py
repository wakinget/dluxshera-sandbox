from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "recipes"
    / "canonical_smear_sensitivity_202608"
    / "canonical_smear_campaign.py"
)
PRESCRIBED_MC_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "recipes"
    / "prescribed_monte_carlo.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("canonical_smear_campaign", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_prescribed_module():
    spec = importlib.util.spec_from_file_location("prescribed_monte_carlo", PRESCRIBED_MC_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sweep_generation_counts_are_nominal() -> None:
    module = _load_module()
    rows = module.campaign_conditions(("A", "B", "C"))

    assert sum(row.family == "A" for row in rows) == 14
    assert sum(row.family == "B" for row in rows) == 44
    assert sum(row.family == "C" for row in rows) == 44
    assert len(rows) == 102


def test_smoke_selection_is_exactly_three_nominal_conditions() -> None:
    module = _load_module()
    rows = module.generate_artifacts(
        output_root=Path("unused"),
        families=("A", "B", "C"),
        dry_run=True,
        smoke=True,
    )["conditions"]

    assert [row["run_id"] for row in rows] == [
        "matched_L0p5_parallel",
        "ampke_L0p5_parallel_m5pct",
        "dirke_L0p5_perpendicular_p2deg",
    ]


def test_zero_smear_is_represented_by_absent_kernel() -> None:
    module = _load_module()
    condition = module.build_condition(
        family="A",
        L_truth_pix=0.0,
        orientation="parallel",
        binary_pa_deg=17.0,
    )

    assert condition.L_truth_pix == 0.0
    assert condition.truth_kernel is None
    assert condition.model_kernel is None


def test_matched_cases_use_identical_truth_and_model_kernels() -> None:
    module = _load_module()
    condition = module.build_condition(
        family="A",
        L_truth_pix=0.5,
        orientation="perpendicular",
        binary_pa_deg=12.0,
    )

    assert condition.truth_kernel == condition.model_kernel


def test_amplitude_knowledge_error_changes_length_only() -> None:
    module = _load_module()
    condition = module.build_condition(
        family="B",
        L_truth_pix=1.0,
        orientation="parallel",
        epsilon_L_percent=-20.0,
        binary_pa_deg=30.0,
    )

    assert condition.L_model_pix == pytest.approx(0.8)
    assert condition.model_kernel["length"] == pytest.approx(0.8)
    assert condition.model_kernel["theta_deg"] == pytest.approx(condition.truth_kernel["theta_deg"])
    assert {
        key: value
        for key, value in condition.model_kernel.items()
        if key not in {"length", "requested_length"}
    } == {
        key: value
        for key, value in condition.truth_kernel.items()
        if key not in {"length", "requested_length"}
    }


def test_direction_knowledge_error_changes_angle_only() -> None:
    module = _load_module()
    condition = module.build_condition(
        family="C",
        L_truth_pix=0.5,
        orientation="parallel",
        delta_theta_deg=2.0,
        binary_pa_deg=30.0,
    )

    assert condition.L_model_pix == pytest.approx(condition.L_truth_pix)
    assert condition.model_kernel["length"] == pytest.approx(condition.truth_kernel["length"])
    assert condition.model_kernel["theta_deg"] == pytest.approx(32.0)
    assert {
        key: value
        for key, value in condition.model_kernel.items()
        if key != "theta_deg"
    } == {
        key: value
        for key, value in condition.truth_kernel.items()
        if key != "theta_deg"
    }


def test_family_b_and_c_grids_are_preserved() -> None:
    module = _load_module()
    rows = module.campaign_conditions(("B", "C"))

    for length in module.FAMILY_BC_LENGTHS:
        for orientation in module.ORIENTATIONS:
            b = [
                row.epsilon_L_percent
                for row in rows
                if row.family == "B"
                and row.L_truth_pix == length
                and row.orientation == orientation
            ]
            c = [
                row.delta_theta_deg
                for row in rows
                if row.family == "C"
                and row.L_truth_pix == length
                and row.orientation == orientation
            ]
            assert b == list(module.MISMATCH_GRID)
            assert c == [float(value) for value in module.MISMATCH_GRID]


@pytest.mark.parametrize(
    ("epsilon", "expected"),
    [(-1.0, 0.99), (1.0, 1.01), (-2.0, 0.98), (2.0, 1.02)],
)
def test_amplitude_mismatch_signs(epsilon: float, expected: float) -> None:
    module = _load_module()
    condition = module.build_condition(
        family="B",
        L_truth_pix=1.0,
        orientation="parallel",
        epsilon_L_percent=epsilon,
        binary_pa_deg=0.0,
    )

    assert condition.L_model_pix == pytest.approx(expected)


def test_parallel_perpendicular_are_relative_to_binary_pa() -> None:
    module = _load_module()
    parallel = module.build_condition(
        family="A",
        L_truth_pix=0.5,
        orientation="parallel",
        binary_pa_deg=23.0,
    )
    perpendicular = module.build_condition(
        family="A",
        L_truth_pix=0.5,
        orientation="perpendicular",
        binary_pa_deg=23.0,
    )

    assert parallel.phi_truth_deg == pytest.approx(0.0)
    assert parallel.theta_truth_deg == pytest.approx(23.0)
    assert perpendicular.phi_truth_deg == pytest.approx(90.0)
    assert perpendicular.theta_truth_deg == pytest.approx(113.0)


def test_run_plan_generation_is_deterministic() -> None:
    module = _load_module()
    first = [module.plan_row(row) for row in module.campaign_conditions(("A", "B", "C"))]
    second = [module.plan_row(row) for row in module.campaign_conditions(("A", "B", "C"))]

    assert first == second


def test_generate_dry_run_does_not_write_files(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "campaign"

    manifest = module.generate_artifacts(
        output_root=root,
        families=("A", "B", "C"),
        dry_run=True,
    )

    assert manifest["total_conditions"] == 102
    assert not root.exists()


def test_generated_smoke_artifacts_include_prescriptions_and_helpers(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "smoke"

    manifest = module.generate_artifacts(
        output_root=root,
        families=("A", "B", "C"),
        smoke=True,
    )

    assert manifest["total_conditions"] == 3
    assert (root / "plan_all.csv").exists()
    assert (root / "parameter_labels.json").exists()
    assert (root / "submit_array.sbatch").exists()
    assert (root / "aggregate.sbatch").exists()
    assert (root / "slurm").is_dir()
    assert "#SBATCH --array=0-2" in (root / "submit_array.sbatch").read_text(encoding="utf-8")
    assert (root / "A" / "matched_L0p5_parallel" / "prescription.yaml").exists()
    assert (root / "B" / "ampke_L0p5_parallel_m5pct" / "condition_manifest.json").exists()
    labels = json.loads((root / "parameter_labels.json").read_text(encoding="utf-8"))
    assert labels["parameter_count"] == 23
    assert labels["expanded_labels"][0] == "Binary Separation"


def test_generated_sbatch_initializes_gattaca2_environment(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "smoke"
    module.generate_artifacts(
        output_root=root,
        families=("A", "B", "C"),
        smoke=True,
    )

    text = (root / "submit_array.sbatch").read_text(encoding="utf-8")
    assert "PYTHONPAT:" not in text
    assert "DLUXSHERA_CONDA_EN_PREFIX" not in text
    for expected in (
        "source /cm/shared/apps/miniforge/etc/profile.d/conda.sh",
        'DLUXSHERA_CONDA_ENV_PREFIX="${DLUXSHERA_CONDA_ENV_PREFIX:-/scratch-jpl/shera_hpc/dmckeith/conda/envs/dluxshera-py311}"',
        'conda activate "$DLUXSHERA_CONDA_ENV_PREFIX"',
        f'export PYTHONPATH="{module.REPO_ROOT}/src${{PYTHONPATH:+:${{PYTHONPATH}}}}"',
        'echo "Conda env: ${CONDA_DEFAULT_ENV:-unset}"',
        'echo "CONDA_PREFIX: ${CONDA_PREFIX:-unset}"',
        'echo "Python executable: $(which python)"',
        'echo "PYTHONPATH: ${PYTHONPATH}"',
        "python - <<'PYENV'\nimport sys",
        'print("sys.executable:", sys.executable)',
        "import jax",
        'print("jax:", jax.__version__)',
        "import dluxshera",
        'print("dluxshera import ok")',
        'print("dluxshera path:", dluxshera.__file__)',
        "#SBATCH --time=00:30:00",
        "#SBATCH --cpus-per-task=2",
        "#SBATCH --mem=24G",
        "\nPYENV\n",
    ):
        assert expected in text
    assert "sbatch -M edge" not in text

    aggregate = (root / "aggregate.sbatch").read_text(encoding="utf-8")
    assert "PYTHONPAT:" not in aggregate
    assert "DLUXSHERA_CONDA_EN_PREFIX" not in aggregate
    assert 'DLUXSHERA_CONDA_ENV_PREFIX="${DLUXSHERA_CONDA_ENV_PREFIX:-/scratch-jpl/shera_hpc/dmckeith/conda/envs/dluxshera-py311}"' in aggregate
    assert 'conda activate "$DLUXSHERA_CONDA_ENV_PREFIX"' in aggregate
    assert "#SBATCH --time=03:00:00" in aggregate
    assert "#SBATCH --mem=24G" in aggregate
    assert f'export PYTHONPATH="{module.REPO_ROOT}/src${{PYTHONPATH:+:${{PYTHONPATH}}}}"' in aggregate


def test_prescription_for_condition_uses_production_optimizer_and_init_settings() -> None:
    module = _load_module()
    condition = module.build_condition(
        family="A",
        L_truth_pix=0.5,
        orientation="parallel",
        binary_pa_deg=0.0,
    )
    prescription = module.prescription_for_condition(condition)
    experiment = prescription["experiment"]

    assert experiment["optimizer"]["kind"] == "sgd"
    assert experiment["optimizer"]["loss"] == "nll"
    assert experiment["optimizer"]["n_iter"] == 200
    assert experiment["optimizer"]["base_lr"] == pytest.approx(0.7)
    assert experiment["optimizer"]["kwargs"] == {}
    schedule = experiment["optimizer"]["schedule"]
    assert schedule["kind"] == "linear_warmup"
    assert schedule["warmup_steps"] == 10
    assert schedule["start_factor"] == pytest.approx(0.125)
    early = experiment["optimizer"]["early_stopping"]
    assert early["enabled"] is True
    assert early["min_iter"] == 40
    assert early["patience"] == 10
    assert early["loss_rtol"] == pytest.approx(1.0e-8)
    assert early["require_finite_loss"] is True
    assert early["restore_best"] is True
    assert early["monitor"] == "loss"
    assert "loss_atol" not in early
    assert "step_atol" not in early
    assert "grad_norm_atol" not in early
    assert experiment["eigenmodes"]["enable"] is True
    assert experiment["eigenmodes"]["whiten"] is True
    assert experiment["eigenmodes"]["truncate_k"] is None
    assert experiment["eigenmodes"]["truncate_by_eigval"] is None
    assert experiment["noise"]["enabled"] is False
    assert experiment["init"]["sampling"] == "prior"
    assert experiment["outputs"]["plots"] is True
    assert len(experiment["infer_keys"]) == 9


def test_production_prior_scales_and_distribution_families_are_exact() -> None:
    module = _load_module()
    condition = module.build_condition(
        family="A",
        L_truth_pix=0.5,
        orientation="parallel",
        binary_pa_deg=0.0,
    )
    priors = module.prescription_for_condition(condition)["experiment"]["priors"]

    assert priors == {
        "source.separation_as": {"dist": "Normal", "sigma": 1.0e-4},
        "source.position_angle_deg": {"dist": "Uniform", "sigma": 1.0e-2},
        "source.x_position_as": {"dist": "Normal", "sigma": 1.0e-2},
        "source.y_position_as": {"dist": "Normal", "sigma": 1.0e-2},
        "source.log_flux_total": {"dist": "LogNormal", "sigma": 1.0e-4},
        "source.contrast": {"dist": "LogNormal", "sigma": 1.0e-4},
        "optics.plate_scale_as_per_pix": {"dist": "LogNormal", "sigma": 1.0e-3},
        "optics.primary.zernike_coeffs_nm": {"dist": "Normal", "sigma": 2.0},
        "optics.secondary.zernike_coeffs_nm": {"dist": "Normal", "sigma": 2.0},
    }


def test_prescribed_mc_preview_resolves_production_settings() -> None:
    module = _load_module()
    prescribed = _load_prescribed_module()
    condition = module.build_condition(
        family="A",
        L_truth_pix=0.5,
        orientation="parallel",
        binary_pa_deg=0.0,
    )
    experiment = module.prescription_for_condition(condition)["experiment"]

    mc_cfg, _ = prescribed._mc_defaults_from_experiment(
        experiment,
        experiment["monte_carlo"],
    )
    run_spec = prescribed._resolve_run_spec_with_id(
        mc_cfg,
        {},
        index=0,
        run_id_index=0,
    )

    assert run_spec["init"]["mode"] == "prior"
    assert run_spec["eigen"]["use_eigen"] is True
    assert run_spec["eigen"]["whiten_basis"] is True
    assert run_spec["eigen"]["truncate_k"] is None
    assert run_spec["eigen"]["truncate_by_eigval"] is None
    assert run_spec["optimizer"]["kind"] == "sgd"
    assert run_spec["optimizer"]["loss"] == "nll"
    assert run_spec["optimizer"]["n_iter"] == 200
    assert run_spec["optimizer"]["base_lr"] == pytest.approx(0.7)
    assert run_spec["optimizer"]["schedule"]["kind"] == "linear_warmup"
    assert run_spec["optimizer"]["schedule"]["warmup_steps"] == 10
    assert run_spec["optimizer"]["schedule"]["start_factor"] == pytest.approx(0.125)
    assert run_spec["optimizer"]["early_stopping"]["enabled"] is True
    assert run_spec["optimizer"]["early_stopping"]["min_iter"] == 40
    assert run_spec["outputs"]["plots"] is True


def test_representative_conditions_have_paired_prior_initial_theta() -> None:
    module = _load_module()
    conditions = [
        module.build_condition(
            family="A",
            L_truth_pix=0.5,
            orientation="parallel",
            binary_pa_deg=0.0,
        ),
        module.build_condition(
            family="B",
            L_truth_pix=0.5,
            orientation="parallel",
            epsilon_L_percent=-5.0,
            binary_pa_deg=0.0,
        ),
        module.build_condition(
            family="C",
            L_truth_pix=0.5,
            orientation="perpendicular",
            delta_theta_deg=2.0,
            binary_pa_deg=0.0,
        ),
    ]

    theta = [module.production_initial_theta_for_condition(condition) for condition in conditions]
    assert theta[0].shape == (23,)
    assert np.allclose(theta[0], theta[1])
    assert np.allclose(theta[0], theta[2])
    provenance = module.paired_initialization_seed_provenance()
    assert provenance["experiment_seed"] == module.PRESCRIBED_MC_SEED
    assert provenance["run_index"] == 1


def test_prescriptions_start_from_preset_and_remove_jitter() -> None:
    module = _load_module()
    preset = module._base_system_config()
    preset_baseline_layers = [
        row
        for row in module.detector_layer_stack(preset)
        if row["name"] not in {module.JITTER_LAYER_NAME, module.SMEAR_LAYER_NAME}
    ]
    condition = module.build_condition(
        family="A",
        L_truth_pix=0.5,
        orientation="parallel",
        binary_pa_deg=0.0,
    )
    prescription = module.prescription_for_condition(condition)

    for system in (prescription["system"], prescription["experiment"]["inference_system"]):
        names = [row["name"] for row in module.detector_layer_stack(system)]
        assert module.JITTER_LAYER_NAME not in names
        assert names[:4] == ["pixel_mtf", "diffusion", "pixel_offsets", "pixel_response"]
        assert [
            row
            for row in module.detector_layer_stack(system)
            if row["name"] not in {module.JITTER_LAYER_NAME, module.SMEAR_LAYER_NAME}
        ] == preset_baseline_layers
        assert system["preset"] == module.SYSTEM_PRESET
        assert system["source"]["exposure_time_s"] == pytest.approx(module.EXPOSURE_TIME_S)


def test_zero_smear_family_a_prescriptions_have_no_smear_layer() -> None:
    module = _load_module()
    for orientation in module.ORIENTATIONS:
        condition = module.build_condition(
            family="A",
            L_truth_pix=0.0,
            orientation=orientation,
            binary_pa_deg=0.0,
        )
        prescription = module.prescription_for_condition(condition)
        assert module.get_detector_layer(prescription["system"], module.SMEAR_LAYER_NAME) is None
        assert (
            module.get_detector_layer(
                prescription["experiment"]["inference_system"],
                module.SMEAR_LAYER_NAME,
            )
            is None
        )


def test_nonzero_matched_family_a_truth_and_model_smear_layers_agree() -> None:
    module = _load_module()
    condition = module.build_condition(
        family="A",
        L_truth_pix=1.0,
        orientation="perpendicular",
        binary_pa_deg=10.0,
    )
    prescription = module.prescription_for_condition(condition)
    truth_smear = module.get_detector_layer(prescription["system"], module.SMEAR_LAYER_NAME)
    inference_smear = module.get_detector_layer(
        prescription["experiment"]["inference_system"],
        module.SMEAR_LAYER_NAME,
    )

    assert truth_smear is not None
    assert inference_smear is not None
    assert truth_smear == inference_smear


def test_all_generated_conditions_remove_jitter_and_have_expected_config_mismatch() -> None:
    module = _load_module()

    for condition in module.campaign_conditions(("A", "B", "C")):
        audit = module.truth_inference_config_audit(condition)
        assert audit["jitter_removed_truth"] is True
        assert audit["jitter_removed_inference"] is True
        assert audit["base_configs_match_after_removing_smear"] is True
        assert audit["matches_expected_mismatch"] is True


def test_truth_inference_mismatch_fields_are_family_specific() -> None:
    module = _load_module()
    family_a = module.build_condition(
        family="A",
        L_truth_pix=0.5,
        orientation="parallel",
        binary_pa_deg=0.0,
    )
    family_b = module.build_condition(
        family="B",
        L_truth_pix=0.5,
        orientation="parallel",
        epsilon_L_percent=5.0,
        binary_pa_deg=0.0,
    )
    family_c = module.build_condition(
        family="C",
        L_truth_pix=0.5,
        orientation="parallel",
        delta_theta_deg=-2.0,
        binary_pa_deg=0.0,
    )

    assert module.truth_inference_config_audit(family_a)["mismatch_fields"] == []
    assert module.truth_inference_config_audit(family_b)["mismatch_fields"] == [
        "detector.layers.smear.kernel.length"
    ]
    assert module.truth_inference_config_audit(family_c)["mismatch_fields"] == [
        "detector.layers.smear.kernel.theta_deg"
    ]


def test_derivative_summary_row_dimensions_and_labels_are_preserved() -> None:
    module = _load_module()
    labels_by_key = {
        "source.separation_as": ["source.separation_as"],
        "source.position_angle_deg": ["source.position_angle_deg"],
        "source.x_position_as": ["source.x_position_as"],
        "source.y_position_as": ["source.y_position_as"],
        "source.log_flux_total": ["source.log_flux_total"],
        "source.contrast": ["source.contrast"],
        "optics.plate_scale_as_per_pix": ["optics.plate_scale_as_per_pix"],
        "optics.primary.zernike_coeffs_nm": [f"m1[{idx}]" for idx in range(8)],
        "optics.secondary.zernike_coeffs_nm": [f"m2[{idx}]" for idx in range(8)],
    }
    labels = module.expanded_labels(labels_by_key)
    F = np.eye(len(labels))

    schur = module._schur_diagnostics(F, labels_by_key=labels_by_key)

    assert len(labels) == 23
    assert schur["F_registration_marginalized_slow"].shape == (20, 20)
    assert schur["F_slow_conditional"].shape == (20, 20)


def test_condition_manifest_records_kernel_and_objective_provenance() -> None:
    module = _load_module()
    condition = module.build_condition(
        family="C",
        L_truth_pix=1.0,
        orientation="perpendicular",
        delta_theta_deg=-2.0,
        binary_pa_deg=10.0,
    )

    manifest = module.condition_manifest(condition)

    assert manifest["smear_definition"]["centered"] is True
    assert manifest["smear_definition"]["normalization"] == "kernel divided by sum(kernel)"
    assert "removes the named smear" in manifest["smear_definition"]["zero_smear_execution_policy"]
    assert manifest["detector_provenance"]["system_preset"] == module.SYSTEM_PRESET
    assert "jitter" in manifest["detector_provenance"]["jitter_policy"]
    assert manifest["detector_provenance"]["truth_inference_config_audit"]["matches_expected_mismatch"] is True
    assert manifest["objective_provenance"]["optimizer_loss"] == "nll"
    assert manifest["objective_provenance"]["parameter_count_expected"] == 23
    assert "jax.hessian" in manifest["objective_provenance"]["fim_theta_semantics"]
    assert manifest["objective_provenance"]["optimizer_max_iterations"] == 200
    assert manifest["objective_provenance"]["eigenmodes"]["enable"] is True
    assert manifest["objective_provenance"]["init"]["sampling"] == "prior"
    assert manifest["objective_provenance"]["plots_enabled"] is True


def test_run_condition_uses_current_interpreter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_module()
    condition_dir = tmp_path / "condition"
    condition_dir.mkdir()
    (condition_dir / "prescription.yaml").write_text("system: {}\n", encoding="utf-8")
    captured = {}

    def fake_call(cmd, cwd):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(module.subprocess, "call", fake_call)

    assert module.run_condition(condition_dir) == 0
    assert captured["cmd"][0] == sys.executable


def test_source_no_longer_contains_hard_coded_python_child_command() -> None:
    module = _load_module()
    source = inspect.getsource(module.run_condition)

    assert '"python"' not in source
    assert "sys.executable" in source
