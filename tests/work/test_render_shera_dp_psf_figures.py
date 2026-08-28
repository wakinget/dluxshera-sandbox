from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import PowerNorm
import numpy as np


def _load_script_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "work"
        / "experiments"
        / "render_shera_dp_psf_figures.py"
    )
    spec = importlib.util.spec_from_file_location(
        "render_shera_dp_psf_figures",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _configure_small_system(module, tmp_path: Path) -> None:
    module.PUPIL_NPIX = 64
    module.PSF_NPIX = 16
    module.N_LAMBDA = 1
    module.OUTPUT_DIR = tmp_path


def _small_figure_system(module, tmp_path: Path):
    _configure_small_system(module, tmp_path)
    return module.build_figure_system()


def test_nominal_mode_retains_preset_m1_and_uses_it_for_plot_support(tmp_path):
    module = _load_script_module()
    system_cfg, _store, binder, _payload = _small_figure_system(module, tmp_path)
    nominal_m1 = module._extract_m1_transmission(binder)
    nominal_m2 = module._extract_aperture_transmission(binder, "m2_aperture")

    modeled_binder, support, info = module._resolve_and_apply_m1_transmission(
        binder,
        system_cfg=system_cfg,
        secondary_obscuration_enabled=True,
        custom_obscuration=False,
    )

    final_m1 = module._extract_m1_transmission(modeled_binder)
    final_m2 = module._extract_aperture_transmission(modeled_binder, "m2_aperture")
    center = support[support.shape[0] // 2, support.shape[1] // 2]

    assert info["pupil_mode"] == "nominal"
    np.testing.assert_array_equal(final_m1, nominal_m1)
    np.testing.assert_array_equal(final_m2, nominal_m2)
    np.testing.assert_array_equal(support, final_m1 > 0.0)
    assert center == np.bool_(False)


def test_custom_mode_installs_custom_m1_without_changing_m2(tmp_path):
    module = _load_script_module()
    module.CUSTOM_N_STRUTS = 4
    module.CUSTOM_STRUT_ROTATION_DEG = 90.0
    module.CUSTOM_STRUT_WIDTH_M = 2.0e-3
    module.CUSTOM_CENTRAL_OBSCURATION_DIAMETER_M = 0.025
    system_cfg, _store, binder, _payload = _small_figure_system(module, tmp_path)
    nominal_m1 = module._extract_m1_transmission(binder)
    nominal_m2 = module._extract_aperture_transmission(binder, "m2_aperture")

    modeled_binder, support, info = module._resolve_and_apply_m1_transmission(
        binder,
        system_cfg=system_cfg,
        secondary_obscuration_enabled=True,
        custom_obscuration=True,
    )

    final_m1 = module._extract_m1_transmission(modeled_binder)
    final_m2 = module._extract_aperture_transmission(modeled_binder, "m2_aperture")
    central_only = module._build_m1_transmission(
        npix=final_m1.shape[-1],
        m1_diameter_m=system_cfg["optics"]["m1_diameter_m"],
        central_obscuration_diameter_m=0.025,
        n_struts=0,
        strut_width_m=0.0,
        strut_rotation_deg=90.0,
    )

    assert info["pupil_mode"] == "custom"
    assert info["selected_n_struts"] == 4
    assert not np.array_equal(final_m1, nominal_m1)
    np.testing.assert_array_equal(final_m2, nominal_m2)
    np.testing.assert_array_equal(support, final_m1 > 0.0)
    assert np.sum(final_m1) < np.sum(central_only)


def test_custom_geometry_parameters_change_selected_m1_transmission():
    module = _load_script_module()
    base = dict(
        npix=96,
        m1_diameter_m=0.22,
        central_obscuration_diameter_m=0.025,
        n_struts=4,
        strut_width_m=2.0e-3,
        strut_rotation_deg=0.0,
    )

    nominal_width = module._build_m1_transmission(**base)
    wider_struts = module._build_m1_transmission(
        **{**base, "strut_width_m": 4.0e-3}
    )
    rotated = module._build_m1_transmission(
        **{**base, "strut_rotation_deg": 45.0}
    )
    larger_obscuration = module._build_m1_transmission(
        **{**base, "central_obscuration_diameter_m": 0.05}
    )

    assert not np.array_equal(nominal_width, rotated)
    assert np.mean(wider_struts > 0.0) < np.mean(nominal_width > 0.0)
    assert np.mean(larger_obscuration > 0.0) < np.mean(nominal_width > 0.0)
    assert larger_obscuration[48, 48] == 0.0


def test_clear_mode_ignores_custom_flag_and_installs_clear_m1(tmp_path):
    module = _load_script_module()
    system_cfg, _store, binder, _payload = _small_figure_system(module, tmp_path)
    nominal_m2 = module._extract_aperture_transmission(binder, "m2_aperture")

    modeled_binder, support, info = module._resolve_and_apply_m1_transmission(
        binder,
        system_cfg=system_cfg,
        secondary_obscuration_enabled=False,
        custom_obscuration=True,
    )

    final_m1 = module._extract_m1_transmission(modeled_binder)
    final_m2 = module._extract_aperture_transmission(modeled_binder, "m2_aperture")
    expected_clear = module._build_m1_transmission(
        npix=final_m1.shape[-1],
        m1_diameter_m=system_cfg["optics"]["m1_diameter_m"],
        central_obscuration_diameter_m=0.0,
        n_struts=0,
        strut_width_m=0.0,
        strut_rotation_deg=0.0,
    )

    assert info["pupil_mode"] == "clear"
    np.testing.assert_array_equal(final_m1, expected_clear)
    np.testing.assert_array_equal(final_m2, nominal_m2)
    np.testing.assert_array_equal(support, final_m1 > 0.0)
    assert support[support.shape[0] // 2, support.shape[1] // 2] == np.bool_(True)
    assert support[0, 0] == np.bool_(False)


def test_modeled_psf_changes_when_m1_transmission_changes(tmp_path):
    module = _load_script_module()
    system_cfg, store, binder, _payload = _small_figure_system(module, tmp_path)
    nominal_binder, _support, _info = module._resolve_and_apply_m1_transmission(
        binder,
        system_cfg=system_cfg,
        secondary_obscuration_enabled=True,
        custom_obscuration=False,
    )
    clear_binder, _support, _info = module._resolve_and_apply_m1_transmission(
        binder,
        system_cfg=system_cfg,
        secondary_obscuration_enabled=False,
        custom_obscuration=False,
    )

    nominal_psf = module.render_noiseless_psf(nominal_binder, store)
    clear_psf = module.render_noiseless_psf(clear_binder, store)

    assert nominal_psf.shape == clear_psf.shape
    assert not np.allclose(nominal_psf, clear_psf)


def test_sqrt_psf_display_uses_unit_range_and_fixed_ticks():
    module = _load_script_module()
    psf = np.array(
        [
            [0.0, 0.1, 0.3],
            [0.2, 1.0, 0.4],
            [0.05, 0.2, 0.6],
        ]
    )

    fig, ax = module.build_single_star_psf_plot(
        psf=psf,
        plate_scale_as_per_pix=0.1,
        oversample=1,
        show_diameter_circle=False,
        stretch="sqrt",
    )

    image = ax.images[0]
    cbar_ax = fig.axes[-1]
    assert isinstance(image.norm, PowerNorm)
    assert image.norm.gamma == 0.5
    assert image.norm.vmin == 0.0
    assert image.norm.vmax == 1.0
    np.testing.assert_allclose(cbar_ax.get_yticks(), module.PSF_COLORBAR_TICKS)
    assert cbar_ax.get_ylabel() == "Normalized Intensity [sqrt]"
    plt.close(fig)


def test_configure_font_preferences_sets_sans_serif_order():
    module = _load_script_module()

    resolved_fonts = module._configure_font_preferences()
    normal_path = font_manager.findfont(
        font_manager.FontProperties(family=[module.FONT_FAMILY], weight="normal"),
        fallback_to_default=True,
    )
    bold_path = font_manager.findfont(
        font_manager.FontProperties(family=[module.FONT_FAMILY], weight="bold"),
        fallback_to_default=True,
    )
    normal_name = font_manager.FontProperties(fname=normal_path).get_name()
    bold_name = font_manager.FontProperties(fname=bold_path).get_name()
    bold_capable = [
        family
        for family in module.SANS_SERIF_FONT_PREFERENCE
        if module._font_family_has_bold_face(family)
    ]

    assert plt.rcParams["font.family"] == [module.FONT_FAMILY]
    if bold_capable:
        assert plt.rcParams["font.sans-serif"][0] == bold_capable[0]
    else:
        assert plt.rcParams["font.sans-serif"] == list(
            module.SANS_SERIF_FONT_PREFERENCE
        )
    assert resolved_fonts == f"normal={normal_name}, bold={bold_name}"


def test_title_bold_toggle_affects_titles_and_axis_labels_not_ticks():
    module = _load_script_module()
    psf = np.array(
        [
            [0.0, 0.1, 0.3],
            [0.2, 1.0, 0.4],
            [0.05, 0.2, 0.6],
        ]
    )

    module.TITLES_BOLD = True
    module._configure_font_preferences()
    fig, ax = module.build_single_star_psf_plot(
        psf=psf,
        plate_scale_as_per_pix=0.1,
        oversample=1,
        show_diameter_circle=False,
        stretch="sqrt",
    )
    cbar_ax = fig.axes[-1]

    assert ax.title.get_fontweight() == "bold"
    assert cbar_ax.yaxis.label.get_fontweight() == "bold"
    assert ax.xaxis.label.get_fontweight() == "bold"
    assert ax.yaxis.label.get_fontweight() == "bold"
    assert ax.get_xticklabels()[0].get_fontweight() == "normal"
    assert ax.get_yticklabels()[0].get_fontweight() == "normal"
    expected_bold_path = font_manager.findfont(
        font_manager.FontProperties(family=[module.FONT_FAMILY], weight="bold"),
        fallback_to_default=True,
    )
    assert font_manager.findfont(ax.title.get_fontproperties()) == expected_bold_path
    assert (
        font_manager.findfont(cbar_ax.yaxis.label.get_fontproperties())
        == expected_bold_path
    )
    plt.close(fig)

    module.TITLES_BOLD = False
    module._configure_font_preferences()
    fig, ax = module.build_single_star_psf_plot(
        psf=psf,
        plate_scale_as_per_pix=0.1,
        oversample=1,
        show_diameter_circle=False,
        stretch="sqrt",
    )
    cbar_ax = fig.axes[-1]

    assert ax.title.get_fontweight() == "normal"
    assert cbar_ax.yaxis.label.get_fontweight() == "normal"
    assert ax.xaxis.label.get_fontweight() == "normal"
    assert ax.yaxis.label.get_fontweight() == "normal"
    plt.close(fig)


def test_dp_plot_centimetre_axes_use_zero_decimal_tick_format():
    module = _load_script_module()
    shape = (8, 8)
    dp_payload = {
        "dp_mask_opd_m": np.zeros(shape),
        "grating_opd_m": np.ones(shape) * 1e-9,
        "combined_opd_m": np.ones(shape) * 2e-9,
    }
    support = np.ones(shape, dtype=bool)

    fig, ax = module.build_dp_opd_plot(
        dp_payload=dp_payload,
        support=support,
        pupil_diameter_m=0.22,
    )

    assert ax.xaxis.get_major_formatter()(1.4) == "1"
    assert ax.xaxis.get_major_formatter()(1.6) == "2"
    assert ax.yaxis.get_major_formatter()(1.4) == "1"
    assert ax.yaxis.get_major_formatter()(1.6) == "2"
    plt.close(fig)
