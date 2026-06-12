from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from dluxshera.utils import full_fidelity_review as review


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "examples/scripts/run_full_fidelity_binary_iterative_campaign.py"
CONFIG_PATH = REPO_ROOT / "examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_binary_iterative_review.yaml"


def _load_wrapper():
    scripts_dir = str(SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("_test_full_fidelity_wrapper_noise", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_render_template_receives_structured_noise_provenance() -> None:
    module = _load_wrapper()
    translated = module._full_fidelity_to_observation_bias(review.load_smoke_config(CONFIG_PATH), run_name="noise_test")
    sub = translated["experiment"]["subblocks"]
    assert sub["noise"] == "enabled"
    model = sub["noise_model"]
    assert model["normalized"]["shot_noise"] is True
    assert model["normalized"]["read_noise"] is True
    assert model["normalized"]["dark_current"] is False
    assert model["render_template_terms"]["read_noise"] is True
    assert model["separate_term_control"] is False
    assert model["warnings"]


def test_inference_template_use_render_variance_auto_is_reported(tmp_path: Path) -> None:
    cfg = review.load_smoke_config(CONFIG_PATH)
    ctx = review.build_model_split_from_smoke(cfg, tmp_path / "review", run_label="noise_test", write_artifacts=False)
    summary = review.summarize_noise_config(ctx["translated_config"], ctx["truth_system_cfg"])
    assert summary["inference_noise_model"]["use_render_variance"] == "auto"
    assert summary["inference_noise_model"]["variance_floor"] == 1.0
    assert summary["render_noise"]["read_noise"] is True
    assert summary["render_noise"]["read_noise_source"].startswith("detector_spec")
