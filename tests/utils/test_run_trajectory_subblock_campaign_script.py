from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "run_trajectory_subblock_campaign.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "run_trajectory_subblock_campaign_script_tests",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_dry_run_writes_plan_and_subblock_artifacts(tmp_path):
    module = _load_script_module()
    airbus = tmp_path / "airbus.csv"
    airbus.write_text(
        "\n".join(
            [
                "0.0,1.0,2.0,3600.0",
                "0.1,1.1,2.2,7200.0",
                "0.2,1.2,2.4,10800.0",
                "0.3,1.3,2.6,14400.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    plan = module.main(
        [
            "--run-name",
            "tiny",
            "--results-root",
            str(tmp_path / "results"),
            "--trajectory-csv",
            str(airbus),
            "--duration-s",
            "0.1",
            "--subblock-duration-s",
            "0.1",
            "--frame-dt-s",
            "0.05",
            "--n-frames-per-subblock",
            "2",
            "--dry-run",
        ]
    )

    run_root = Path(plan["run_root"])
    assert (run_root / "campaign_plan.json").exists()
    assert (run_root / "trajectory_ingest_summary.json").exists()
    assert (run_root / "subblock_plan.csv").exists()
    subblock = run_root / "subblocks" / "subblock_000000"
    assert (subblock / "frame_truth.csv").exists()
    assert (subblock / "starting_guess_prediction.csv").exists()
    assert (subblock / "render_config.json").exists()
    assert (subblock / "inference_config.json").exists()
    assert (subblock / "command.sh").exists()

    campaign_plan = json.loads((run_root / "campaign_plan.json").read_text(encoding="utf-8"))
    assert campaign_plan["n_subblocks"] == 1
    assert campaign_plan["child_results"] == []
    assert "source.position_angle_deg" in campaign_plan["active_frame_keys"]


def test_script_smear_metadata_and_constant_layer_modes(tmp_path):
    module = _load_script_module()
    airbus = tmp_path / "airbus.csv"
    airbus.write_text(
        "\n".join(
            [
                "0.0,1.0,2.0,3600.0",
                "0.1,1.1,2.2,7200.0",
                "0.2,1.2,2.4,10800.0",
                "0.3,1.3,2.6,14400.0",
                "0.4,1.4,2.8,18000.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    smear_cfg = tmp_path / "smear.yaml"
    smear_cfg.write_text(
        """
trajectory_processing:
  smear:
    enabled: true
    exposure:
      time_s: from_subblocks
      interval: centered
      edge_policy: error
    inference:
      mode: scaled
      length_scale: 0.8
    render:
      mode: subblock_constant_layer
      target_layer: trajectory_smear
      require_existing_layer: true
      allow_layer_injection: false
""",
        encoding="utf-8",
    )
    render_template = tmp_path / "render_template.yaml"
    render_template.write_text(
        """
system:
  preset: SHERA_FLIGHT_3P
  source:
    target: ALPHA_CEN
    exposure_time_s: 0.05
  detector:
    layers:
      - name: trajectory_smear
        kind: ApplyConvolution
        kernel:
          kind: line
          length: 1.0e-12
          theta_deg: 0.0
          sigma_perp: 0.1
          kernel_size: 11
          units: detector_pix

experiment:
  kind: subblock_generation
  seed: 42
  subblock:
    varying_keys:
      - source.x_position_as
      - source.y_position_as
      - source.position_angle_deg
    trace:
      format: csv
      path: frame_truth.csv
    validate:
      require_contiguous_frame_index: true
      require_monotonic_time: true
  noise:
    enabled: false
    photon_noise: true
    read_noise: false
    dark_current: false
  outputs:
    outdir: unused
    file_prefix: obs_subblock
    frame_truth_format: csv
""",
        encoding="utf-8",
    )

    plan = module.main(
        [
            "--run-name",
            "smear",
            "--results-root",
            str(tmp_path / "results"),
            "--trajectory-csv",
            str(airbus),
            "--start-s",
            "0.05",
            "--duration-s",
            "0.1",
            "--subblock-duration-s",
            "0.1",
            "--frame-dt-s",
            "0.05",
            "--n-frames-per-subblock",
            "2",
            "--trajectory-processing-config",
            str(smear_cfg),
            "--render-template",
            str(render_template),
            "--dry-run",
        ]
    )

    run_root = Path(plan["run_root"])
    subblock = run_root / "subblocks" / "subblock_000000"
    assert (subblock / "frame_smear_truth.csv").exists()
    assert (subblock / "frame_smear_model.csv").exists()
    assert (subblock / "smear_provenance.json").exists()
    assert (run_root / "smear_summary.json").exists()
    render_cfg = json.loads((subblock / "render_config.json").read_text(encoding="utf-8"))
    provenance = json.loads((subblock / "smear_provenance.json").read_text(encoding="utf-8"))
    expected_kernel = provenance["representative_kernel"]
    layers = render_cfg["system"]["detector"]["layers"]
    smear_layer = next(layer for layer in layers if layer.get("name") == "trajectory_smear")
    assert smear_layer["kernel"]["kind"] == "line"
    assert smear_layer["kernel"]["length"] == expected_kernel["length"]
    assert smear_layer["kernel"]["theta_deg"] == expected_kernel["theta_deg"]
    assert smear_layer["kernel"]["sigma_perp"] == expected_kernel["sigma_perp"]
    assert smear_layer["kernel"]["kernel_size"] == expected_kernel["kernel_size"]
    assert plan["subblocks"][0]["smear_model_policy"] == "scaled"
    assert plan["subblocks"][0]["smear_layer_name"] == "trajectory_smear"
