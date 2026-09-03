from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from dluxshera.ml.models import build_pairwise_correction_model
from dluxshera.ml.visualization import (
    ArchitectureVisualizationError,
    _compressed_visual_size,
    describe_pairwise_correction_architecture,
    render_pairwise_correction_architecture_set,
    render_pairwise_correction_architecture,
    render_pairwise_correction_model_overview,
    render_shared_cnn_encoder_detail,
)


BASELINE_CONFIG = {
    "comparator": "concat_diff",
    "embedding_dim": 128,
    "adaptive_pool_shape": [4, 4],
}


def test_pairwise_architecture_description_extracts_baseline_model() -> None:
    model = build_pairwise_correction_model(13, BASELINE_CONFIG)

    desc = describe_pairwise_correction_architecture(model, input_shape=(1, 160, 160))

    assert desc.input_shape == (1, 160, 160)
    assert [(s.input_channels, s.output_channels) for s in desc.encoder_stages] == [
        (1, 16),
        (16, 32),
        (32, 64),
        (64, 128),
    ]
    assert [s.kernel_size for s in desc.encoder_stages] == [
        (5, 5),
        (5, 5),
        (3, 3),
        (3, 3),
    ]
    assert [s.output_shape for s in desc.encoder_stages] == [
        (16, 80, 80),
        (32, 40, 40),
        (64, 20, 20),
        (128, 10, 10),
    ]
    assert all(stage.has_batch_norm for stage in desc.encoder_stages)
    assert all(stage.activation == "ReLU" for stage in desc.encoder_stages)
    assert desc.adaptive_pool_shape == (4, 4)
    assert desc.flattened_features == 2048
    assert [(l.input_features, l.output_features) for l in desc.projection_layers] == [
        (2048, 256),
        (256, 128),
    ]
    assert desc.embedding_dim == 128
    assert desc.comparator == "concat_diff"
    assert desc.comparator_expression == "[h_A, h_B, h_B - h_A]"
    assert desc.comparator_dim == 384
    assert [(l.input_features, l.output_features) for l in desc.regression_head_layers] == [
        (384, 256),
        (256, 13),
    ]
    assert desc.output_dim == 13
    assert desc.trainable_parameter_count == 765421


def test_pairwise_architecture_description_tracks_difference_comparator() -> None:
    model = build_pairwise_correction_model(
        5,
        {
            **BASELINE_CONFIG,
            "comparator": "difference",
        },
    )

    desc = describe_pairwise_correction_architecture(model, input_shape=(1, 160, 160))

    assert desc.comparator == "difference"
    assert desc.comparator_expression == "h_B - h_A"
    assert desc.comparator_dim == 128
    assert [(l.input_features, l.output_features) for l in desc.regression_head_layers] == [
        (128, 256),
        (256, 5),
    ]
    assert desc.trainable_parameter_count == 697829


def test_render_pairwise_architecture_set_writes_two_views_without_latex(tmp_path: Path) -> None:
    model = build_pairwise_correction_model(13, BASELINE_CONFIG)

    result = render_pairwise_correction_architecture_set(
        model,
        input_shape=(1, 160, 160),
        model_config=BASELINE_CONFIG,
        output_dir=tmp_path,
        compile_pdf=False,
    )

    assert result.overview.tex_path.name == "pairwise_correction_model_overview.tex"
    assert result.encoder.tex_path.name == "shared_cnn_encoder_detail.tex"
    for rendered in (result.overview, result.encoder):
        assert rendered.tex_path.exists()
        assert rendered.metadata_path.exists()
        assert rendered.pdf_path is None
        assert "PlotNeuralNet" in rendered.backend

    assert result.overview.metadata["view"] == "model_overview"
    assert result.encoder.metadata["view"] == "shared_encoder_detail"


def test_render_pairwise_architecture_compatibility_wrapper_returns_overview(tmp_path: Path) -> None:
    model = build_pairwise_correction_model(13, BASELINE_CONFIG)

    result = render_pairwise_correction_architecture(
        model,
        input_shape=(1, 160, 160),
        model_config=BASELINE_CONFIG,
        output_dir=tmp_path,
        compile_pdf=False,
    )

    assert result.tex_path.name == "pairwise_correction_architecture.tex"
    assert result.metadata["view"] == "model_overview"
    assert "Shared CNN Encoder" in result.tex_path.read_text(encoding="utf-8")


def test_model_overview_tex_has_pairwise_structure_without_conv_towers(tmp_path: Path) -> None:
    model = build_pairwise_correction_model(13, BASELINE_CONFIG)

    result = render_pairwise_correction_model_overview(
        model,
        input_shape=(1, 160, 160),
        model_config=BASELINE_CONFIG,
        output_dir=tmp_path,
        compile_pdf=False,
    )

    tex = result.tex_path.read_text(encoding="utf-8")
    for expected in (
        "Image A",
        "Image B",
        "Shared CNN Encoder",
        "same weights",
        "h_A",
        "h_B",
        "Comparator",
        "[h_A, h_B, h_B - h_A]",
        "384-D",
        "Regression Head",
        "13",
        "Delta theta",
        r"\Delta\theta",
    ):
        assert expected in tex
    for omitted in ("Conv 1", "Conv 2", "Conv 3", "Conv 4"):
        assert omitted not in tex


def test_encoder_detail_tex_has_one_encoder_sequence_and_tensor_labels(tmp_path: Path) -> None:
    model = build_pairwise_correction_model(20, BASELINE_CONFIG)

    result = render_shared_cnn_encoder_detail(
        model,
        input_shape=(1, 160, 160),
        model_config=BASELINE_CONFIG,
        output_dir=tmp_path,
        compile_pdf=False,
    )

    tex = result.tex_path.read_text(encoding="utf-8")
    for expected in (
        "Shared CNN Encoder",
        "Conv 1",
        "Conv 2",
        "Conv 3",
        "Conv 4",
        "Adaptive\\\\AvgPool",
        "Flatten",
        "FC + ReLU",
        "Embedding",
        "1 x 160 x 160",
        "16 x 80 x 80",
        "32 x 40 x 40",
        "64 x 20 x 20",
        "128 x 10 x 10",
        "128 x 4 x 4",
        "2048",
        "256",
        "128",
    ):
        assert expected in tex
    assert tex.count("Conv 1") == 1
    assert tex.count("Conv 2") == 1
    assert tex.count("Conv 3") == 1
    assert tex.count("Conv 4") == 1


def test_compressed_visual_size_is_deterministic_monotonic_and_bounded() -> None:
    kwargs = {
        "lower_value": 4,
        "upper_value": 160,
        "min_size": 16.0,
        "max_size": 42.0,
    }

    sizes = [_compressed_visual_size(value, **kwargs) for value in (4, 10, 20, 40, 80, 160)]

    assert sizes == [_compressed_visual_size(value, **kwargs) for value in (4, 10, 20, 40, 80, 160)]
    assert sizes == sorted(sizes)
    assert min(sizes) >= 16.0
    assert max(sizes) <= 42.0
    assert _compressed_visual_size(80, **kwargs) == _compressed_visual_size(80, **kwargs)
    assert sizes[-1] / sizes[0] < 4.0


def test_model_overview_tracks_comparator_variants(tmp_path: Path) -> None:
    concat_model = build_pairwise_correction_model(20, BASELINE_CONFIG)
    concat_result = render_pairwise_correction_model_overview(
        concat_model,
        input_shape=(1, 160, 160),
        model_config=BASELINE_CONFIG,
        output_dir=tmp_path,
        basename="concat",
        compile_pdf=False,
    )

    difference_config = {**BASELINE_CONFIG, "comparator": "difference"}
    difference_model = build_pairwise_correction_model(20, difference_config)
    difference_result = render_pairwise_correction_model_overview(
        difference_model,
        input_shape=(1, 160, 160),
        model_config=difference_config,
        output_dir=tmp_path,
        basename="difference",
        compile_pdf=False,
    )

    concat_tex = concat_result.tex_path.read_text(encoding="utf-8")
    difference_tex = difference_result.tex_path.read_text(encoding="utf-8")
    assert "concat\\_diff" in concat_tex
    assert "[h_A, h_B, h_B - h_A]" in concat_tex
    assert "384-D" in concat_tex
    assert "Difference" in difference_tex
    assert "h_B - h_A" in difference_tex
    assert "128-D" in difference_tex


def test_render_pairwise_architecture_missing_latex_error_is_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_pairwise_correction_model(3, BASELINE_CONFIG)
    monkeypatch.setattr("dluxshera.ml.visualization.resolve_pdflatex", lambda: None)

    with pytest.raises(ArchitectureVisualizationError, match="pdflatex.*not found"):
        render_pairwise_correction_architecture(
            model,
            input_shape=(1, 160, 160),
            output_dir=tmp_path,
            compile_pdf=True,
        )


def test_render_pairwise_architecture_optional_pdf_compile_smoke(tmp_path: Path) -> None:
    from dluxshera.ml.visualization import resolve_pdflatex

    if resolve_pdflatex() is None:
        pytest.skip("pdflatex is not installed")

    model = build_pairwise_correction_model(3, BASELINE_CONFIG)
    result = render_pairwise_correction_architecture_set(
        model,
        input_shape=(1, 160, 160),
        output_dir=tmp_path,
        compile_pdf=True,
        cleanup_aux=True,
    )

    for rendered in (result.overview, result.encoder):
        assert rendered.pdf_path is not None
        assert rendered.pdf_path.exists()
        assert rendered.pdf_path.stat().st_size > 0
