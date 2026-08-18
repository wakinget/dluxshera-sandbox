from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "examples" / "scripts" / "export_full_fidelity_poster_figures.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("export_full_fidelity_poster_figures", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_defaults_include_png_and_expected_paths():
    module = _load_script_module()
    parser = module.build_arg_parser()
    args = parser.parse_args([])

    assert args.config == str(module.DEFAULT_CONFIG)
    assert args.outdir == str(module.DEFAULT_OUTDIR)
    assert args.wavelength_min_nm == 450.0
    assert args.wavelength_max_nm == 650.0
    assert args.dp_opd_cmap == "inferno"
    assert args.line_figsize == (6.0, 3.5)
    assert args.opd_figsize == (5.3, 4.8)
    assert module._formats(args.formats) == ["png"]


def test_formats_normalize_common_aliases_and_keep_overrides():
    module = _load_script_module()

    assert module._formats("tif,png,jpg") == ["png", "tiff", "jpeg"]
    assert module._formats("png,tiff,pdf") == ["png", "tiff", "pdf"]


def test_formats_reject_empty_or_unsupported_values():
    module = _load_script_module()

    with pytest.raises(module.argparse.ArgumentTypeError):
        module._formats("")
    with pytest.raises(module.argparse.ArgumentTypeError):
        module._formats("png,svg")


def test_export_sections_inventory_includes_notebook_plot_categories():
    module = _load_script_module()

    assert set(module.EXPORT_SECTIONS) >= {
        "spectral",
        "sed",
        "dp_opd",
        "high_order_wfe",
        "trajectory",
        "detector_calibration",
    }


def test_record_exports_groups_flat_and_category_manifest_entries():
    module = _load_script_module()
    manifest = {
        "figure_exports": {},
        "figure_exports_by_category": {section: {} for section in module.EXPORT_SECTIONS},
    }

    module._record_exports(manifest, "trajectory", "trajectory_x", ["trajectory_x.png"])

    assert manifest["figure_exports"]["trajectory_x"] == ["trajectory_x.png"]
    assert manifest["figure_exports_by_category"]["trajectory"]["trajectory_x"] == ["trajectory_x.png"]


def test_poster_light_theme_sets_dark_text_on_white_background():
    module = _load_script_module()

    with module.poster_light_theme(dpi=321):
        assert plt.rcParams["figure.facecolor"] == "white"
        assert plt.rcParams["axes.facecolor"] == "white"
        assert plt.rcParams["savefig.transparent"] is False
        assert plt.rcParams["text.color"] == "black"
        assert plt.rcParams["xtick.color"] == "black"
        assert plt.rcParams["ytick.color"] == "black"
        assert plt.rcParams["figure.dpi"] == 321
        assert plt.rcParams["image.cmap"] == "inferno_nan"


def test_repo_relative_paths_resolve_to_existing_data_files():
    module = _load_script_module()

    assert module._resolve_repo_path(module.DEFAULT_CONFIG).is_file()
    for rel_path in module.DEFAULT_SED_FILES.values():
        assert module._resolve_repo_path(rel_path).is_file()


def test_configured_response_paths_resolve_to_existing_package_data():
    module = _load_script_module()
    config = module.load_config_file(module._resolve_repo_path(module.DEFAULT_CONFIG))

    m2_spec = module._component_spec(config, "m2_filter_response")
    qe_spec = module._component_spec(config, "detector_qe")

    assert module.resolve_response_curve_path(m2_spec["path"]).is_file()
    assert module.resolve_response_curve_path(qe_spec["path"]).is_file()


class _FakeStore:
    def __init__(self, values):
        self.values = values

    def get(self, key, *args, **kwargs):
        if key not in self.values:
            raise KeyError(key)
        return self.values[key]


class _FakeBinder:
    def __init__(self, store):
        self.base_forward_store = store


def test_resolved_pupil_extent_prefers_physical_m1_diameter_without_warning():
    module = _load_script_module()
    store = _FakeStore({"optics.m1_diameter_m": 0.22})
    binder = _FakeBinder(store)

    extent, label, warnings, metadata = module.resolved_pupil_extent(binder, store, {"optics": {}})

    assert np.allclose(extent, np.array([-0.11, 0.11, -0.11, 0.11]))
    assert label == "M1 pupil coordinate (m)"
    assert warnings == []
    assert metadata["units"] == "m"


def test_collect_source_artifacts_keeps_fits_separate_from_figure_exports(tmp_path):
    module = _load_script_module()
    maps = tmp_path / "_model_split_for_poster" / "model_split" / "high_order_wfe" / "maps"
    maps.mkdir(parents=True)
    fits_path = maps / "primary_high_order_truth_opd_nm.fits"
    manifest_path = maps / "high_order_wfe_deck_manifest.json"
    fits_path.write_bytes(b"source fits")
    manifest_path.write_text("{}", encoding="utf-8")

    artifacts = module._collect_source_artifacts(tmp_path)

    assert artifacts["high_order_wfe_fits"] == [str(fits_path)]
    assert artifacts["high_order_wfe_manifests"] == [str(manifest_path)]
    assert str(fits_path) in artifacts["all"]
