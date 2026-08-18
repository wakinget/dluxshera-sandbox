from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from dluxshera.config.io import load_config_file
from dluxshera.utils.detector_layer_overrides import get_detector_layer
from dluxshera.utils.full_fidelity_review import build_model_split_from_smoke


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "scripts" / "run_full_fidelity_binary_iterative_campaign.py"
CONFIG = (
    ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_binary_iterative_smoke.yaml"
)


def _module():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("run_full_fidelity_binary_iterative_detector_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_detector_overrides_are_forwarded_and_applied(tmp_path: Path) -> None:
    module = _module()
    raw = load_config_file(CONFIG)
    translated = module._full_fidelity_to_observation_bias(raw, run_name="unit")

    assert translated["experiment"]["detector_overrides"]["layers"]["jitter"]["action"] == "update"

    split = build_model_split_from_smoke(raw, tmp_path, write_artifacts=False)
    jitter = get_detector_layer(split["base_system_cfg"], "jitter")

    assert jitter is not None
    assert jitter["kernel"]["sigma_x"] == 0.001
    assert jitter["kernel"]["sigma_y"] == 0.001
