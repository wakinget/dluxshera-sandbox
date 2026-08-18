from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "scripts" / "run_observation_bias_campaign.py"


def _load_observation_bias():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "run_observation_bias_early_stopping_flags_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reference_early_stopping_flag_forwarded_once(tmp_path: Path) -> None:
    module = _load_observation_bias()
    subblock_cfg = {
        "n_frames": 1,
        "reference_early_stopping_enabled": True,
        "reference_early_stopping_patience": 3,
    }

    options = module.resolve_subblock_command_options(subblock_cfg)
    command = module.build_subblock_command(
        case_root_parent=tmp_path,
        case_subblock_name="case/subblock_000000",
        theta_labels=("source.x_position_as",),
        offsets={},
        subblock_cfg=subblock_cfg,
    )

    assert options["forwarded_flags"].count("--reference-early-stopping") == 1
    assert command.count("--reference-early-stopping") == 1
    assert "--reference-early-stopping-patience" in options["forwarded_flags"]
    assert "--reference-early-stopping-patience" in command


def test_reference_early_stopping_disabled_omits_switch(tmp_path: Path) -> None:
    module = _load_observation_bias()
    subblock_cfg = {
        "n_frames": 1,
        "reference_early_stopping_enabled": False,
    }

    options = module.resolve_subblock_command_options(subblock_cfg)
    command = module.build_subblock_command(
        case_root_parent=tmp_path,
        case_subblock_name="case/subblock_000000",
        theta_labels=("source.x_position_as",),
        offsets={},
        subblock_cfg=subblock_cfg,
    )

    assert "--reference-early-stopping" not in options["forwarded_flags"]
    assert "--reference-early-stopping" not in command
