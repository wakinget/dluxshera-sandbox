from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "generate_prescribed_mc_sweep.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "generate_prescribed_mc_sweep_script",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_prescription_payload() -> dict:
    return {
        "system": {"preset": "SHERA_TESTBED_3P"},
        "experiment": {
            "kind": "prescribed_mc",
            "notes": "base note",
            "seed": 42,
            "monte_carlo": {
                "n_runs": 5,
                "results_orientation": "col",
                "run_plan": None,
            },
            "outputs": {"outdir": "Results/original"},
            "inference_system": {
                "preset": "SHERA_TESTBED_3P",
                "detector": {
                    "layers": [
                        {"name": "downsample", "kernel_size": 3},
                        {
                            "name": "pixel_offsets",
                            "dx_path": "dx.fits",
                            "dy_path": "dy.fits",
                            "knowledge_error": {
                                "model": "gaussian",
                                "scale": 0.0,
                                "realization_policy": "fixed_per_experiment",
                            },
                        },
                    ]
                },
            },
        },
    }


def _write_yaml(path: Path, payload: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _read_layer_knowledge_error(cfg: dict, layer_name: str) -> dict:
    layers = cfg["experiment"]["inference_system"]["detector"]["layers"]
    for layer in layers:
        if layer.get("name") == layer_name:
            return layer["knowledge_error"]
    raise AssertionError(f"Layer not found in test payload: {layer_name}")


def test_timestamped_root_generation(tmp_path):
    module = _load_script_module()
    root = module.resolve_root_dir(
        root_dir=None,
        results_root=tmp_path / "Results",
        sweep_name="detector_ke_sweep",
        now=dt.datetime(2026, 3, 20, 11, 35, 0),
    )
    assert root == tmp_path / "Results" / "detector_ke_sweep_20260320-113500"


def test_folder_labels_for_representative_scale_values():
    module = _load_script_module()
    points = module.parse_sweep_points(["0", "1e-4", "3e-4", "1e-3", "1e-2"])
    assert [point.dirname for point in points] == [
        "ke_0",
        "ke_1e-4",
        "ke_3e-4",
        "ke_1e-3",
        "ke_1e-2",
    ]


def test_patch_updates_layer_scale():
    module = _load_script_module()
    base_cfg = _base_prescription_payload()
    point = module.parse_sweep_points(["1e-3"])[0]

    patched = module.patch_prescription_for_point(
        base_cfg,
        layer_name="pixel_offsets",
        sweep_point=point,
        realization_policy=None,
        results_orientation=None,
        n_runs=None,
        notes_suffix=None,
    )

    knowledge_error = _read_layer_knowledge_error(patched, "pixel_offsets")
    assert knowledge_error["scale"] == 1e-3


def test_patch_updates_realization_policy_when_requested():
    module = _load_script_module()
    base_cfg = _base_prescription_payload()
    point = module.parse_sweep_points(["1e-4"])[0]

    patched = module.patch_prescription_for_point(
        base_cfg,
        layer_name="pixel_offsets",
        sweep_point=point,
        realization_policy="per_run",
        results_orientation=None,
        n_runs=None,
        notes_suffix=None,
    )

    knowledge_error = _read_layer_knowledge_error(patched, "pixel_offsets")
    assert knowledge_error["realization_policy"] == "per_run"


def test_patch_forces_outputs_outdir_dot():
    module = _load_script_module()
    base_cfg = _base_prescription_payload()
    point = module.parse_sweep_points(["0"])[0]

    patched = module.patch_prescription_for_point(
        base_cfg,
        layer_name="pixel_offsets",
        sweep_point=point,
        realization_policy=None,
        results_orientation=None,
        n_runs=None,
        notes_suffix=None,
    )

    assert patched["experiment"]["outputs"]["outdir"] == "."


def test_fails_when_inference_system_missing():
    module = _load_script_module()
    bad_cfg = _base_prescription_payload()
    bad_cfg["experiment"].pop("inference_system")
    point = module.parse_sweep_points(["1e-3"])[0]

    try:
        module.patch_prescription_for_point(
            bad_cfg,
            layer_name="pixel_offsets",
            sweep_point=point,
            realization_policy=None,
            results_orientation=None,
            n_runs=None,
            notes_suffix=None,
        )
    except ValueError as exc:
        assert "experiment.inference_system is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing experiment.inference_system")


def test_fails_when_requested_layer_missing():
    module = _load_script_module()
    bad_cfg = _base_prescription_payload()
    bad_cfg["experiment"]["inference_system"]["detector"]["layers"] = [
        {"name": "downsample", "kernel_size": 3}
    ]
    point = module.parse_sweep_points(["1e-3"])[0]

    try:
        module.patch_prescription_for_point(
            bad_cfg,
            layer_name="pixel_offsets",
            sweep_point=point,
            realization_policy=None,
            results_orientation=None,
            n_runs=None,
            notes_suffix=None,
        )
    except ValueError as exc:
        assert "Requested detector layer was not found" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing detector layer")


def test_dry_run_does_not_write_files(tmp_path):
    module = _load_script_module()
    base_cfg = _base_prescription_payload()
    base_path = tmp_path / "base_prescription.yaml"
    _write_yaml(base_path, base_cfg)

    output_root = tmp_path / "Results" / "detector_ke_sweep_20260320-113500"
    points = module.parse_sweep_points(["0", "1e-3"])
    manifest = module.generate_sweep_scaffold(
        base_cfg=base_cfg,
        base_path=base_path,
        output_root=output_root,
        sweep_name="detector_ke_sweep",
        layer_name="pixel_offsets",
        sweep_points=points,
        realization_policy="per_run",
        results_orientation="row",
        n_runs=7,
        notes_suffix="dry run note",
        now=dt.datetime(2026, 3, 20, 11, 35, 0),
        dry_run=True,
        force=False,
    )

    assert manifest["layer"] == "pixel_offsets"
    assert output_root.exists() is False


def test_manifest_and_generated_files(tmp_path):
    module = _load_script_module()
    base_cfg = _base_prescription_payload()
    base_path = tmp_path / "base_prescription.yaml"
    _write_yaml(base_path, base_cfg)

    output_root = tmp_path / "Results" / "detector_ke_sweep_20260320-113500"
    points = module.parse_sweep_points(["0", "1e-4"])
    module.generate_sweep_scaffold(
        base_cfg=base_cfg,
        base_path=base_path,
        output_root=output_root,
        sweep_name="detector_ke_sweep",
        layer_name="pixel_offsets",
        sweep_points=points,
        realization_policy="per_run",
        results_orientation="row",
        n_runs=9,
        notes_suffix="manifest-check",
        now=dt.datetime(2026, 3, 20, 11, 35, 0),
        dry_run=False,
        force=False,
    )

    assert (output_root / "prescription_base.yaml").exists()
    assert (output_root / "ke_0" / "prescription.yaml").exists()
    assert (output_root / "ke_1e-4" / "prescription.yaml").exists()

    manifest_path = output_root / "sweep_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["timestamp_label"] == "20260320-113500"
    assert manifest["layer"] == "pixel_offsets"
    assert manifest["realization_policy_override"] == "per_run"
    assert manifest["results_orientation_override"] == "row"
    assert manifest["n_runs_override"] == 9
    assert [item["label"] for item in manifest["experiments"]] == ["ke_0", "ke_1e-4"]

    generated = module.load_config_file(output_root / "ke_1e-4" / "prescription.yaml")
    ke_cfg = _read_layer_knowledge_error(generated, "pixel_offsets")
    assert ke_cfg["scale"] == 1e-4
    assert ke_cfg["realization_policy"] == "per_run"
    assert generated["experiment"]["outputs"]["outdir"] == "."
