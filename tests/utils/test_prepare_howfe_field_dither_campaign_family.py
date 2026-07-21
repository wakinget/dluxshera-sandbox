from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "examples" / "scripts"
SCRIPT_PATH = SCRIPT_DIR / "prepare_howfe_field_dither_campaign_family.py"
SHARD_SCRIPT_PATH = SCRIPT_DIR / "prepare_full_fidelity_campaign_shards.py"


def _load_module(path: Path, name: str) -> Any:
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _base_config() -> dict[str, Any]:
    return {
        "experiment": {
            "kind": "full_fidelity_binary_iterative",
            "schema_version": "full_fidelity_binary_iterative.v1",
            "run_name": "unit",
            "seed": 42,
            "high_order_wfe": {
                "enabled": True,
                "truth": {
                    "enabled": True,
                    "mirrors": ["primary", "secondary"],
                    "npix": 16,
                    "amplitude_nm_rms": 1.0,
                    "power_law_alpha": 2.5,
                    "seed": 101,
                    "pairing": "independent",
                    "remove_low_order_zernikes": True,
                    "remove_zernike_modes": [4, 5, 6, 7, 8, 9, 10, 11],
                },
                "inference": {
                    "enabled": True,
                    "mode": "knowledge_error",
                    "knowledge_error": {"enabled": True, "amplitude_nm_rms": 0.1},
                },
                "artifacts": {"write_maps": False, "write_summary_json": False},
                "validation": {"require_nonzero_difference_when_enabled": False},
            },
            "subblocks": {
                "n_frames": 2,
                "trace_source": {"mode": "iid_jitter"},
            },
            "iterative": {
                "enabled": True,
                "windows_per_draw": 2,
                "subblocks_per_window": 3,
            },
            "iterative_forecast": {
                "enabled": True,
                "actual_windows": 2,
                "projected_windows": 2,
                "subblocks_per_window": 3,
            },
            "observation_theta": {
                "source": {
                    "separation_as": True,
                    "log_flux_total": True,
                    "contrast": True,
                },
                "optics": {
                    "plate_scale_as_per_pix": True,
                    "primary_zernikes": {"enabled": True, "indices": [0, 1]},
                    "secondary_zernikes": {"enabled": True, "indices": [0, 1]},
                },
            },
            "prior_draws": {
                "enabled": True,
                "n_cases": 2,
                "draws_per_condition": 2,
                "draw_seed": 12345,
                "conditions": [
                    {
                        "condition_name": "base",
                        "sigmas": {
                            "optics.primary.zernike_coeffs_nm[*]": {
                                "kind": "absolute",
                                "sigma": 0.01,
                                "unit": "nm",
                            },
                            "optics.secondary.zernike_coeffs_nm[*]": {
                                "kind": "absolute",
                                "sigma": 0.01,
                                "unit": "nm",
                            },
                        },
                    }
                ],
            },
        }
    }


def test_condition_config_sets_mirror_field_metadata_and_fixed_hashes() -> None:
    module = _load_module(SCRIPT_PATH, "prepare_howfe_field_dither_campaign_family")
    config = module._condition_config(
        _base_config(),
        mirror="secondary",
        amplitude_nm=0.1,
        field_token="xp1p0_yp0p0",
        x_as=1.0,
        y_as=0.0,
        pa_deg=0.0,
        cadence_token="w1x30",
        windows=1,
        subblocks_per_window=30,
        draws=1,
    )
    experiment = config["experiment"]
    metadata = experiment["campaign_family_condition"]

    assert "m2_hoke_0p1nm_xp1p0_yp0p0_w1x30" in experiment["run_name"]
    assert experiment["subblocks"]["trace_source"]["processing"]["offsets"]["source.x_position_as"] == 1.0
    assert metadata["ho_ke_primary_enabled"] is False
    assert metadata["ho_ke_secondary_enabled"] is True
    assert metadata["ho_ke_secondary_amplitude_nm_rms"] == 0.1
    assert metadata["secondary_ke_map_hash"]
    assert metadata["map_group"]


def test_combined_shard_manifest_contains_new_fields_and_preserves_draw_positions(tmp_path: Path) -> None:
    family = _load_module(SCRIPT_PATH, "prepare_howfe_field_dither_campaign_family_manifest")
    sharder = _load_module(SHARD_SCRIPT_PATH, "prepare_full_fidelity_campaign_shards_manifest")
    config = family._condition_config(
        _base_config(),
        mirror="primary",
        amplitude_nm=0.05,
        field_token="xp0p0_yp0p0",
        x_as=0.0,
        y_as=0.0,
        pa_deg=0.0,
        cadence_token="w2x3",
        windows=2,
        subblocks_per_window=3,
        draws=2,
    )
    config_path = tmp_path / "source.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    rows = sharder.prepare_shards_for_configs(
        config_paths=[config_path],
        outdir=tmp_path / "shards",
        run_name_prefix="unit",
        mode="draw",
        results_root=tmp_path / "results",
        resources=sharder.Resources(
            time="00:10:00",
            cpus_per_task=1,
            mem="1G",
            max_workers=1,
        ),
        dry_run=False,
        overwrite=False,
    )

    assert len(rows) == 2
    assert rows[0]["draw_index"] == 0
    assert rows[1]["draw_index"] == 1
    assert rows[0]["expected_subblocks"] == 6
    assert rows[0]["field_offset_x_as"] == 0.0
    assert rows[0]["ho_ke_active_mirror"] == "primary"
    assert rows[0]["primary_ke_map_hash"]
    shard_config = yaml.safe_load(
        (tmp_path / "shards" / "configs" / f"{rows[1]['shard_name']}.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert shard_config["experiment"]["prior_draws"]["rng_skip_draws"] == 1

    with (tmp_path / "shards" / "shard_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    assert manifest_rows[0]["map_group"]
