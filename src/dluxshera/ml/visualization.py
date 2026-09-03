from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._vendor.plotneuralnet import layers_path

__all__ = [
    "ArchitectureRenderResult",
    "ArchitectureVisualizationError",
    "ConvStageDescription",
    "LinearLayerDescription",
    "PairwiseArchitectureRenderSet",
    "PairwiseCorrectionArchitecture",
    "describe_pairwise_correction_architecture",
    "render_pairwise_correction_architecture_set",
    "render_pairwise_correction_architecture",
    "render_pairwise_correction_model_overview",
    "render_shared_cnn_encoder_detail",
    "resolve_pdflatex",
]

PLOTNEURALNET_BACKEND = "PlotNeuralNet/TikZ"
PLOTNEURALNET_SOURCE = "HarisIqbal88/PlotNeuralNet"
PLOTNEURALNET_COMMIT = "e96bc852189c2089dd500527a0a01a5a36e8977e"


class ArchitectureVisualizationError(RuntimeError):
    """Report architecture visualization failures with actionable context."""


@dataclass(frozen=True)
class ConvStageDescription:
    """Describe one convolutional encoder stage extracted from the model."""

    name: str
    input_channels: int
    output_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    has_batch_norm: bool
    activation: str | None
    output_shape: tuple[int, int, int]


@dataclass(frozen=True)
class LinearLayerDescription:
    """Describe one linear layer extracted from the model."""

    name: str
    input_features: int
    output_features: int


@dataclass(frozen=True)
class PairwiseCorrectionArchitecture:
    """Capture the pairwise correction CNN structure used for rendering.

    This intentionally small description is specific to the dLuxShera
    shared-encoder pairwise correction baseline.  It records the architectural
    fields needed by notebook visualizations without attempting to represent an
    arbitrary PyTorch computation graph.
    """

    input_shape: tuple[int, int, int]
    model_class: str
    encoder_class: str
    shared_encoder: bool
    encoder_stages: tuple[ConvStageDescription, ...]
    adaptive_pool_shape: tuple[int, int]
    pooled_shape: tuple[int, int, int]
    flattened_features: int
    projection_layers: tuple[LinearLayerDescription, ...]
    embedding_dim: int
    comparator: str
    comparator_expression: str
    comparator_dim: int
    regression_head_layers: tuple[LinearLayerDescription, ...]
    output_dim: int
    trainable_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation of this architecture."""
        return asdict(self)


@dataclass(frozen=True)
class ArchitectureRenderResult:
    """Return generated architecture artifacts and provenance."""

    tex_path: Path
    pdf_path: Path | None
    metadata_path: Path
    backend: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PairwiseArchitectureRenderSet:
    """Return the two complementary views for one pairwise architecture."""

    overview: ArchitectureRenderResult
    encoder: ArchitectureRenderResult


def _pair2(value: Any, *, field: str) -> tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ArchitectureVisualizationError(f"{field} must have length 2, got {value!r}.")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _conv2d_output_size(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return int(math.floor((size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1))


def _iter_linear_layers(module: Any) -> tuple[LinearLayerDescription, ...]:
    try:
        from torch import nn
    except ModuleNotFoundError as exc:  # pragma: no cover - no-torch environment
        raise ModuleNotFoundError(
            "ML architecture visualization requires PyTorch because it inspects "
            "the model instance. Install the ML environment with "
            "`python -m pip install -e .[ml]`."
        ) from exc

    layers: list[LinearLayerDescription] = []
    for name, child in module.named_modules():
        if name and isinstance(child, nn.Linear):
            layers.append(
                LinearLayerDescription(
                    name=str(name),
                    input_features=int(child.in_features),
                    output_features=int(child.out_features),
                )
            )
    return tuple(layers)


def _count_parameters(model: Any, *, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        return int(sum(param.numel() for param in params if param.requires_grad))
    return int(sum(param.numel() for param in params))


def describe_pairwise_correction_architecture(
    model: Any,
    *,
    input_shape: Sequence[int] = (1, 160, 160),
    model_config: Mapping[str, Any] | None = None,
) -> PairwiseCorrectionArchitecture:
    """Extract a compact architecture description from a pairwise CNN model.

    Parameters
    ----------
    model
        Actual ``PairwiseCorrectionCNN``-like PyTorch module to inspect.
    input_shape
        Single-image ``(channels, height, width)`` shape.  The spatial input
        size is not encoded in the model, so callers provide it explicitly.
    model_config
        Optional model configuration mapping used only as a fallback for
        semantic fields that are not exposed by the module.

    Returns
    -------
    PairwiseCorrectionArchitecture
        Immutable model-specific description suitable for rendering and JSON
        provenance.

    Raises
    ------
    ArchitectureVisualizationError
        If the supplied model does not expose the shared encoder and regression
        head structure expected by the dLuxShera pairwise correction baseline.
    """
    try:
        from torch import nn
    except ModuleNotFoundError as exc:  # pragma: no cover - no-torch environment
        raise ModuleNotFoundError(
            "ML architecture visualization requires PyTorch because it inspects "
            "the model instance. Install the ML environment with "
            "`python -m pip install -e .[ml]`."
        ) from exc

    if len(tuple(input_shape)) != 3:
        raise ArchitectureVisualizationError(
            f"input_shape must be (channels, height, width), got {tuple(input_shape)!r}."
        )
    in_channels, height, width = (int(v) for v in input_shape)
    if min(in_channels, height, width) < 1:
        raise ArchitectureVisualizationError(f"input_shape values must be positive, got {tuple(input_shape)!r}.")

    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise ArchitectureVisualizationError("Expected model.encoder on the pairwise correction model.")
    features = getattr(encoder, "features", None)
    if features is None:
        raise ArchitectureVisualizationError("Expected model.encoder.features to contain convolutional stages.")

    stages: list[ConvStageDescription] = []
    modules = list(features.children())
    spatial_h = height
    spatial_w = width
    for index, layer in enumerate(modules):
        if not isinstance(layer, nn.Conv2d):
            continue
        kernel = _pair2(layer.kernel_size, field="kernel_size")
        stride = _pair2(layer.stride, field="stride")
        padding = _pair2(layer.padding, field="padding")
        dilation = _pair2(layer.dilation, field="dilation")
        spatial_h = _conv2d_output_size(spatial_h, kernel[0], stride[0], padding[0], dilation[0])
        spatial_w = _conv2d_output_size(spatial_w, kernel[1], stride[1], padding[1], dilation[1])

        following = modules[index + 1 :]
        next_conv_index = next(
            (offset for offset, child in enumerate(following) if isinstance(child, nn.Conv2d)),
            len(following),
        )
        block_layers = following[:next_conv_index]
        batch_norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        activation = next(
            (child.__class__.__name__ for child in block_layers if isinstance(child, (nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU))),
            None,
        )
        stages.append(
            ConvStageDescription(
                name=f"conv{len(stages) + 1}",
                input_channels=int(layer.in_channels),
                output_channels=int(layer.out_channels),
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                has_batch_norm=any(isinstance(child, batch_norm_types) for child in block_layers),
                activation=activation,
                output_shape=(int(layer.out_channels), int(spatial_h), int(spatial_w)),
            )
        )

    if not stages:
        raise ArchitectureVisualizationError("No Conv2d stages were found in model.encoder.features.")

    pool = getattr(encoder, "pool", None)
    if not isinstance(pool, nn.AdaptiveAvgPool2d):
        raise ArchitectureVisualizationError("Expected model.encoder.pool to be nn.AdaptiveAvgPool2d.")
    adaptive_pool_shape = _pair2(pool.output_size, field="adaptive pool output_size")
    pooled_shape = (stages[-1].output_channels, adaptive_pool_shape[0], adaptive_pool_shape[1])
    flattened_features = int(pooled_shape[0] * pooled_shape[1] * pooled_shape[2])

    projection = getattr(encoder, "projection", None)
    if projection is None:
        raise ArchitectureVisualizationError("Expected model.encoder.projection to contain Linear layers.")
    projection_layers = _iter_linear_layers(projection)
    if not projection_layers:
        raise ArchitectureVisualizationError("No Linear layers were found in model.encoder.projection.")
    embedding_dim = int(getattr(encoder, "embedding_dim", projection_layers[-1].output_features))
    if projection_layers[-1].output_features != embedding_dim:
        raise ArchitectureVisualizationError(
            "The encoder embedding_dim does not match the final projection layer output."
        )

    fallback_comparator = None if model_config is None else model_config.get("comparator")
    comparator = str(getattr(model, "comparator", fallback_comparator or "concat_diff"))
    if comparator == "difference":
        comparator_dim = embedding_dim
        comparator_expression = "h_B - h_A"
    elif comparator == "concat_diff":
        comparator_dim = 3 * embedding_dim
        comparator_expression = "[h_A, h_B, h_B - h_A]"
    else:
        raise ArchitectureVisualizationError(
            f"Unsupported pairwise comparator {comparator!r}; expected 'concat_diff' or 'difference'."
        )

    regression_head = getattr(model, "regression_head", None)
    if regression_head is None:
        raise ArchitectureVisualizationError("Expected model.regression_head to contain Linear layers.")
    head_layers = _iter_linear_layers(regression_head)
    if not head_layers:
        raise ArchitectureVisualizationError("No Linear layers were found in model.regression_head.")
    if head_layers[0].input_features != comparator_dim:
        raise ArchitectureVisualizationError(
            "Comparator dimension does not match regression head input: "
            f"{comparator_dim} vs {head_layers[0].input_features}."
        )

    output_dim = int(getattr(model, "output_dim", head_layers[-1].output_features))
    return PairwiseCorrectionArchitecture(
        input_shape=(in_channels, height, width),
        model_class=model.__class__.__name__,
        encoder_class=encoder.__class__.__name__,
        shared_encoder=True,
        encoder_stages=tuple(stages),
        adaptive_pool_shape=adaptive_pool_shape,
        pooled_shape=pooled_shape,
        flattened_features=flattened_features,
        projection_layers=projection_layers,
        embedding_dim=embedding_dim,
        comparator=comparator,
        comparator_expression=comparator_expression,
        comparator_dim=comparator_dim,
        regression_head_layers=head_layers,
        output_dim=output_dim,
        trainable_parameter_count=_count_parameters(model),
    )


def _shape_label(shape: Sequence[int]) -> str:
    return " x ".join(str(int(v)) for v in shape)


def _compressed_visual_size(
    value: int | float,
    *,
    lower_value: int | float,
    upper_value: int | float,
    min_size: float,
    max_size: float,
    exponent: float = 0.35,
) -> float:
    """Map tensor magnitudes to bounded presentation sizes.

    Tensor dimensions remain scientific/model metadata and are rendered exactly
    in labels.  Visual dimensions are a deterministic nonlinear presentation
    mapping so large inputs do not dominate the figure and small feature maps
    remain visible.
    """
    numeric_value = float(value)
    lower = float(lower_value)
    upper = float(upper_value)
    if min(numeric_value, lower, upper, min_size, max_size, exponent) <= 0.0:
        raise ArchitectureVisualizationError("visual-size inputs must be positive.")
    if lower > upper:
        lower, upper = upper, lower
    clamped = min(max(numeric_value, lower), upper)
    if math.isclose(lower, upper):
        return float((min_size + max_size) / 2.0)

    low_t = lower**exponent
    high_t = upper**exponent
    value_t = clamped**exponent
    ratio = (value_t - low_t) / (high_t - low_t)
    return float(min_size + ratio * (max_size - min_size))


def _architecture_spatial_values(desc: PairwiseCorrectionArchitecture) -> tuple[int, ...]:
    values = [max(desc.input_shape[1], desc.input_shape[2])]
    values.extend(max(stage.output_shape[1], stage.output_shape[2]) for stage in desc.encoder_stages)
    values.append(max(desc.pooled_shape[1], desc.pooled_shape[2]))
    return tuple(int(value) for value in values)


def _architecture_channel_values(desc: PairwiseCorrectionArchitecture) -> tuple[int, ...]:
    values = [desc.input_shape[0]]
    values.extend(stage.output_channels for stage in desc.encoder_stages)
    values.append(desc.pooled_shape[0])
    return tuple(int(value) for value in values)


def _encoder_box_size(
    *,
    channels: int,
    spatial_h: int,
    spatial_w: int,
    spatial_values: Sequence[int],
    channel_values: Sequence[int],
) -> tuple[float, float, float]:
    spatial = max(int(spatial_h), int(spatial_w))
    height = _compressed_visual_size(
        spatial,
        lower_value=min(spatial_values),
        upper_value=max(spatial_values),
        min_size=16.0,
        max_size=42.0,
    )
    width = _compressed_visual_size(
        channels,
        lower_value=min(channel_values),
        upper_value=max(channel_values),
        min_size=0.85,
        max_size=4.0,
        exponent=0.42,
    )
    return width, height, height


def _vector_box_size(
    value: int,
    *,
    vector_values: Sequence[int],
    min_size: float = 7.5,
    max_size: float = 14.0,
) -> float:
    return _compressed_visual_size(
        value,
        lower_value=min(vector_values),
        upper_value=max(vector_values),
        min_size=min_size,
        max_size=max_size,
        exponent=0.28,
    )


def _emit_right_banded_box(
    *,
    name: str,
    offset: str,
    to: str,
    width: float,
    height: float,
    depth: float,
    xlabel: str,
    zlabel: str,
    caption: str,
    fill: str = r"\ConvColor",
    bandfill: str = r"\ConvReluColor",
) -> str:
    return rf"""
\pic[shift={{{offset}}}] at {to}
    {{RightBandedBox={{
        name={name},
        caption={{{caption}}},
        xlabel={{{{{{{xlabel}}}, }} }},
        zlabel={{{zlabel}}},
        fill={fill},
        bandfill={bandfill},
        height={height:.3g},
        width={{{width:.3g}}},
        depth={depth:.3g}
        }}
    }};
"""


def _emit_box(
    *,
    name: str,
    offset: str,
    to: str,
    width: float,
    height: float,
    depth: float,
    xlabel: str = "",
    zlabel: str = "",
    caption: str,
    fill: str,
    opacity: float = 0.75,
) -> str:
    return rf"""
\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={{{caption}}},
        xlabel={{{{{{{xlabel}}}, }} }},
        zlabel={{{zlabel}}},
        fill={fill},
        opacity={opacity:.3g},
        height={height:.3g},
        width={{{width:.3g}}},
        depth={depth:.3g}
        }}
    }};
"""


def _connection(source: str, target: str) -> str:
    return rf"\draw [connection] ({source}-east) -- node {{\midarrow}} ({target}-west);" + "\n"


def _metadata_comments(desc: PairwiseCorrectionArchitecture, *, view: str) -> str:
    return "\n".join(
        [
            f"% view: {view}",
            f"% input_shape: {_shape_label(desc.input_shape)}",
            f"% comparator: {desc.comparator}",
            f"% comparator_expression: {desc.comparator_expression}",
            f"% comparator_dim: {desc.comparator_dim}",
            f"% embedding_dim: {desc.embedding_dim}",
            f"% output_dim: {desc.output_dim}",
            "% output_label: Delta theta",
            f"% trainable_parameter_count: {desc.trainable_parameter_count}",
        ]
    )


def _tex_preamble(desc: PairwiseCorrectionArchitecture, *, view: str) -> list[str]:
    layer_dir = layers_path().replace("\\", "/")
    return [
        r"\documentclass[border=8pt, multi, tikz]{standalone}",
        r"\usepackage{xcolor}",
        r"\usepackage{import}",
        rf"\subimport{{{layer_dir}/}}{{init}}",
        r"\usetikzlibrary{positioning,3d,fit,calc}",
        r"\def\ConvColor{rgb:yellow,5;red,2.5;white,5}",
        r"\def\ConvReluColor{rgb:yellow,5;red,5;white,5}",
        r"\def\PoolColor{rgb:red,1;black,0.3}",
        r"\def\FcColor{rgb:blue,5;red,2.5;white,5}",
        r"\def\SoftmaxColor{rgb:magenta,5;black,7}",
        r"\def\InputColor{rgb:green,2;blue,1;white,8}",
        r"\def\EmbeddingColor{rgb:cyan,4;blue,2;white,7}",
        r"\def\CompareColor{rgb:green,5;blue,2;white,6}",
        r"\def\OutputColor{rgb:magenta,4;red,1;white,6}",
        r"\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}",
        _metadata_comments(desc, view=view),
        r"\pagecolor{white}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]",
        r"\tikzstyle{shapeLabel}=[align=center,font=\scriptsize,text width=2.6cm]",
        r"\tikzstyle{opLabel}=[align=center,font=\scriptsize,text width=2.7cm]",
    ]


def _finish_tex(lines: list[str]) -> str:
    lines.extend([r"\end{tikzpicture}", r"\end{document}"])
    return "\n".join(lines)


def _comparator_visual_title(desc: PairwiseCorrectionArchitecture) -> str:
    if desc.comparator == "difference":
        return "Difference"
    return "Comparator"


def _comparator_visual_subtitle(desc: PairwiseCorrectionArchitecture) -> str:
    if desc.comparator == "concat_diff":
        return r"concat\_diff"
    return "pairwise difference"


def _head_dimension_chain(desc: PairwiseCorrectionArchitecture) -> str:
    dims = [desc.regression_head_layers[0].input_features]
    dims.extend(layer.output_features for layer in desc.regression_head_layers)
    return " $\\rightarrow$ ".join(str(int(dim)) for dim in dims)


def _plotneuralnet_overview_tex(desc: PairwiseCorrectionArchitecture) -> str:
    lines = _tex_preamble(desc, view="model_overview")
    c, h, w = desc.input_shape
    conv_count = len(desc.encoder_stages)
    comparator_title = _comparator_visual_title(desc)
    comparator_subtitle = _comparator_visual_subtitle(desc)
    lines.extend(
        [
            r"\tikzstyle{overviewBlock}=[draw,very thick,fill=white,align=center,font=\small,minimum height=1.1cm]",
            r"\tikzstyle{inputBlock}=[overviewBlock,fill=\InputColor,fill opacity=0.45,text opacity=1,minimum width=2.2cm]",
            r"\tikzstyle{latentBlock}=[overviewBlock,fill=\EmbeddingColor,fill opacity=0.58,text opacity=1,minimum width=1.45cm]",
            r"\tikzstyle{conceptBlock}=[overviewBlock,fill=\CompareColor,fill opacity=0.55,text opacity=1,minimum width=3.5cm]",
            r"\tikzstyle{headBlock}=[overviewBlock,fill=\FcColor,fill opacity=0.55,text opacity=1,minimum width=3.3cm]",
            r"\tikzstyle{outBlock}=[overviewBlock,fill=\OutputColor,fill opacity=0.62,text opacity=1,minimum width=2.7cm]",
            r"\node[inputBlock] (image_a) at (0,1.65) {Image A\\{\scriptsize " + _shape_label(desc.input_shape) + r"}};",
            r"\node[inputBlock] (image_b) at (0,-1.65) {Image B\\{\scriptsize " + _shape_label(desc.input_shape) + r"}};",
            r"\node[draw,very thick,fill=\ConvColor,fill opacity=0.35,text opacity=1,align=center,font=\small,minimum width=5.2cm,minimum height=4.4cm] (encoder) at (5.0,0) "
            + rf"{{\textbf{{Shared CNN Encoder $E$}}\\same weights\\[3pt]{{\scriptsize {_shape_label((c, h, w))} $\rightarrow$ {desc.embedding_dim}-D embedding}}\\{{\scriptsize {conv_count} convolution stages + projection}}}};",
            r"\node[latentBlock] (ha) at (10.0,1.65) {$h_A$\\{\scriptsize " + str(desc.embedding_dim) + r"-D}};",
            r"\node[latentBlock] (hb) at (10.0,-1.65) {$h_B$\\{\scriptsize " + str(desc.embedding_dim) + r"-D}};",
            r"\node[conceptBlock] (comparator) at (14.1,0) "
            + rf"{{\textbf{{{comparator_title}}}\\{{\scriptsize {comparator_subtitle}}}\\[2pt]{{\scriptsize ${{{desc.comparator_expression}}}$}}\\{{\scriptsize {desc.comparator_dim}-D}}}};",
            r"\node[headBlock] (head) at (14.1,-3.65) {\textbf{Regression Head}\\{\scriptsize "
            + _head_dimension_chain(desc)
            + r"}};",
            r"\node[outBlock] (output) at (14.1,-6.3) {\textbf{$\Delta\theta$}\\{\scriptsize science correction}\\{\scriptsize "
            + str(desc.output_dim)
            + r" parameters}};",
            r"\coordinate (encoder_a_west) at ($(encoder.west)+(0,1.35)$);",
            r"\coordinate (encoder_b_west) at ($(encoder.west)+(0,-1.35)$);",
            r"\coordinate (encoder_a_east) at ($(encoder.east)+(0,1.35)$);",
            r"\coordinate (encoder_b_east) at ($(encoder.east)+(0,-1.35)$);",
            r"\draw[connection] (image_a.east) -- node {\midarrow} (encoder_a_west);",
            r"\draw[connection] (image_b.east) -- node {\midarrow} (encoder_b_west);",
            r"\draw[connection] (encoder_a_east) -- node {\midarrow} (ha.west);",
            r"\draw[connection] (encoder_b_east) -- node {\midarrow} (hb.west);",
            r"\draw[connection] (ha.east) -- node {\midarrow} (comparator.west);",
            r"\draw[connection] (hb.east) -- node {\midarrow} (comparator.west);",
            r"\draw[connection] (comparator.south) -- node {\midarrow} (head.north);",
            r"\draw[connection] (head.south) -- node {\midarrow} (output.north);",
        ]
    )
    return _finish_tex(lines)


def _conv_operation_label(stage: ConvStageDescription, *, index: int) -> str:
    operations = [rf"Conv {index}: {stage.kernel_size[0]}x{stage.kernel_size[1]}"]
    suffix: list[str] = []
    if stage.has_batch_norm:
        suffix.append("BN")
    if stage.activation:
        suffix.append(stage.activation)
    if suffix:
        operations.append(" + ".join(suffix))
    stride = stage.stride[0] if stage.stride[0] == stage.stride[1] else f"{stage.stride[0]}x{stage.stride[1]}"
    operations.append(rf"stride {stride}")
    return r"\\".join(operations)


def _tensor_node(name: str, label: str) -> str:
    return rf"\node[shapeLabel] at ($({name}-north)+(0,1.15,0)$) {{{label}}};"


def _operation_node(name: str, label: str) -> str:
    return rf"\node[opLabel] at ($({name}-south)+(0,-1.35,0)$) {{{label}}};"


def _plotneuralnet_encoder_detail_tex(desc: PairwiseCorrectionArchitecture) -> str:
    lines = _tex_preamble(desc, view="shared_encoder_detail")
    spatial_values = _architecture_spatial_values(desc)
    channel_values = _architecture_channel_values(desc)
    vector_values = [desc.flattened_features]
    vector_values.extend(layer.output_features for layer in desc.projection_layers)

    c, h, w = desc.input_shape
    input_width, input_height, input_depth = _encoder_box_size(
        channels=c,
        spatial_h=h,
        spatial_w=w,
        spatial_values=spatial_values,
        channel_values=channel_values,
    )
    x_positions = [0.0, 4.0, 7.9, 11.8, 15.8, 20.5, 24.0, 27.0, 29.8]
    lines.append(r"\node[align=center,font=\small] at (14.7,6.45) {\textbf{Shared CNN Encoder $E$} \quad same weights for Image A and Image B};")
    lines.append(
        _emit_box(
            name="input",
            offset="(0,0,0)",
            to=f"({x_positions[0]:.3g},0,0)",
            width=input_width,
            height=input_height,
            depth=input_depth,
            xlabel='""',
            zlabel="",
            caption="",
            fill=r"\InputColor",
            opacity=0.5,
        )
    )
    lines.append(_tensor_node("input", _shape_label(desc.input_shape)))
    lines.append(_operation_node("input", "Input image"))

    previous = "input"
    for index, stage in enumerate(desc.encoder_stages, start=1):
        channels, spatial_h, spatial_w = stage.output_shape
        width_box, height_box, depth_box = _encoder_box_size(
            channels=channels,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
            spatial_values=spatial_values,
            channel_values=channel_values,
        )
        name = f"conv{index}"
        lines.append(
            _emit_right_banded_box(
                name=name,
                offset="(0,0,0)",
                to=f"({x_positions[index]:.3g},0,0)",
                width=width_box,
                height=height_box,
                depth=depth_box,
                xlabel='""',
                zlabel="",
                caption="",
            )
        )
        lines.append(_connection(previous, name))
        lines.append(_tensor_node(name, _shape_label(stage.output_shape)))
        lines.append(_operation_node(name, _conv_operation_label(stage, index=index)))
        previous = name

    pool_channels, pool_h, pool_w = desc.pooled_shape
    pool_width, pool_height, pool_depth = _encoder_box_size(
        channels=pool_channels,
        spatial_h=pool_h,
        spatial_w=pool_w,
        spatial_values=spatial_values,
        channel_values=channel_values,
    )
    lines.append(
        _emit_box(
            name="pool",
            offset="(0,0,0)",
            to=f"({x_positions[5]:.3g},0,0)",
            width=pool_width,
            height=pool_height,
            depth=pool_depth,
            xlabel='""',
            zlabel="",
            caption="",
            fill=r"\PoolColor",
            opacity=0.45,
        )
    )
    lines.append(_connection(previous, "pool"))
    lines.append(_tensor_node("pool", _shape_label(desc.pooled_shape)))
    lines.append(_operation_node("pool", r"Adaptive\\AvgPool"))

    flatten_size = _vector_box_size(desc.flattened_features, vector_values=vector_values)
    lines.append(
        _emit_box(
            name="flatten",
            offset="(0,0,0)",
            to=f"({x_positions[6]:.3g},0,0)",
            width=0.9,
            height=flatten_size,
            depth=flatten_size,
            xlabel='""',
            zlabel="",
            caption="",
            fill=r"\FcColor",
            opacity=0.5,
        )
    )
    lines.append(_connection("pool", "flatten"))
    lines.append(_tensor_node("flatten", str(desc.flattened_features)))
    lines.append(_operation_node("flatten", "Flatten"))

    first_projection = desc.projection_layers[0]
    fc_size = _vector_box_size(first_projection.output_features, vector_values=vector_values)
    lines.append(
        _emit_box(
            name="fc",
            offset="(0,0,0)",
            to=f"({x_positions[7]:.3g},0,0)",
            width=1.05,
            height=fc_size,
            depth=fc_size,
            xlabel='""',
            zlabel="",
            caption="",
            fill=r"\FcColor",
            opacity=0.68,
        )
    )
    lines.append(_connection("flatten", "fc"))
    lines.append(_tensor_node("fc", str(first_projection.output_features)))
    lines.append(_operation_node("fc", r"FC + ReLU"))

    embedding_size = _vector_box_size(desc.embedding_dim, vector_values=vector_values)
    lines.append(
        _emit_box(
            name="embedding",
            offset="(0,0,0)",
            to=f"({x_positions[8]:.3g},0,0)",
            width=1.0,
            height=embedding_size,
            depth=embedding_size,
            xlabel='""',
            zlabel="",
            caption="",
            fill=r"\EmbeddingColor",
            opacity=0.78,
        )
    )
    lines.append(_connection("fc", "embedding"))
    lines.append(_tensor_node("embedding", str(desc.embedding_dim)))
    lines.append(_operation_node("embedding", "Embedding"))
    return _finish_tex(lines)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_pdflatex() -> str | None:
    """Return a usable ``pdflatex`` executable path, if one is discoverable.

    GUI-launched Jupyter processes on macOS often do not inherit the terminal
    ``PATH`` that includes MacTeX.  This helper first respects the caller's
    environment and then checks the standard MacTeX executable path without
    mutating ``PATH`` globally.
    """
    executable = shutil.which("pdflatex")
    if executable is not None:
        return executable
    if sys.platform == "darwin":
        candidate = Path("/Library/TeX/texbin/pdflatex")
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _compile_tex(tex_path: Path, *, cleanup_aux: bool = False) -> Path:
    pdflatex = resolve_pdflatex()
    if pdflatex is None:
        raise ArchitectureVisualizationError(
            "PDF compilation requested, but `pdflatex` was not found. Install a "
            "LaTeX distribution such as MacTeX/BasicTeX on macOS, TeX Live on "
            "Linux, or MiKTeX/TeX Live on Windows; or call with compile_pdf=False "
            "to generate only the .tex source."
        )

    cmd = [
        pdflatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    result = subprocess.run(
        cmd,
        cwd=tex_path.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        tail = output[-3000:] if output else "<no pdflatex output captured>"
        raise ArchitectureVisualizationError(
            f"pdflatex failed while compiling {tex_path}. Command: {' '.join(cmd)}\n"
            f"Last compiler output:\n{tail}"
        )
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise ArchitectureVisualizationError(f"pdflatex completed but did not create a non-empty PDF at {pdf_path}.")

    if cleanup_aux:
        for suffix in (".aux", ".log"):
            tex_path.with_suffix(suffix).unlink(missing_ok=True)
    return pdf_path


def _coerce_pairwise_architecture(
    description_or_model: Any,
    *,
    input_shape: Sequence[int] = (1, 160, 160),
    model_config: Mapping[str, Any] | None = None,
) -> PairwiseCorrectionArchitecture:
    if isinstance(description_or_model, PairwiseCorrectionArchitecture):
        return description_or_model
    return describe_pairwise_correction_architecture(
        description_or_model,
        input_shape=input_shape,
        model_config=model_config,
    )


def _architecture_metadata(
    desc: PairwiseCorrectionArchitecture,
    *,
    view: str,
) -> dict[str, Any]:
    return {
        "backend": PLOTNEURALNET_BACKEND,
        "view": view,
        "plotneuralnet": {
            "source": PLOTNEURALNET_SOURCE,
            "commit": PLOTNEURALNET_COMMIT,
            "integration": "vendored minimal LaTeX/TikZ layer resources",
        },
        **desc.to_dict(),
    }


def _write_render_result(
    *,
    desc: PairwiseCorrectionArchitecture,
    output_dir: Path | str,
    basename: str,
    view: str,
    tex_source: str,
    compile_pdf: bool,
    cleanup_aux: bool,
) -> ArchitectureRenderResult:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    tex_path = outdir / f"{basename}.tex"
    metadata_path = outdir / f"{basename}.json"
    metadata = _architecture_metadata(desc, view=view)
    tex_path.write_text(tex_source, encoding="utf-8")
    _write_json(metadata_path, metadata)
    pdf_path = _compile_tex(tex_path, cleanup_aux=cleanup_aux) if compile_pdf else None
    return ArchitectureRenderResult(
        tex_path=tex_path,
        pdf_path=pdf_path,
        metadata_path=metadata_path,
        backend=PLOTNEURALNET_BACKEND,
        metadata=metadata,
    )


def render_pairwise_correction_model_overview(
    description_or_model: Any,
    *,
    input_shape: Sequence[int] = (1, 160, 160),
    model_config: Mapping[str, Any] | None = None,
    output_dir: Path | str,
    basename: str = "pairwise_correction_model_overview",
    compile_pdf: bool = True,
    cleanup_aux: bool = False,
) -> ArchitectureRenderResult:
    """Render the high-level shared-encoder pairwise model overview.

    This view intentionally omits convolution internals and emphasizes the
    Siamese/pairwise topology: two images, one shared encoder, embeddings,
    comparator, regression head, and physical correction output.
    """
    desc = _coerce_pairwise_architecture(
        description_or_model,
        input_shape=input_shape,
        model_config=model_config,
    )
    return _write_render_result(
        desc=desc,
        output_dir=output_dir,
        basename=basename,
        view="model_overview",
        tex_source=_plotneuralnet_overview_tex(desc),
        compile_pdf=compile_pdf,
        cleanup_aux=cleanup_aux,
    )


def render_shared_cnn_encoder_detail(
    description_or_model: Any,
    *,
    input_shape: Sequence[int] = (1, 160, 160),
    model_config: Mapping[str, Any] | None = None,
    output_dir: Path | str,
    basename: str = "shared_cnn_encoder_detail",
    compile_pdf: bool = True,
    cleanup_aux: bool = False,
) -> ArchitectureRenderResult:
    """Render the shared CNN encoder architecture once in detail.

    This view uses PlotNeuralNet's CNN box language for the encoder progression
    and keeps exact tensor dimensions in labels while using compressed visual
    box dimensions for readability.
    """
    desc = _coerce_pairwise_architecture(
        description_or_model,
        input_shape=input_shape,
        model_config=model_config,
    )
    return _write_render_result(
        desc=desc,
        output_dir=output_dir,
        basename=basename,
        view="shared_encoder_detail",
        tex_source=_plotneuralnet_encoder_detail_tex(desc),
        compile_pdf=compile_pdf,
        cleanup_aux=cleanup_aux,
    )


def render_pairwise_correction_architecture_set(
    description_or_model: Any,
    *,
    input_shape: Sequence[int] = (1, 160, 160),
    model_config: Mapping[str, Any] | None = None,
    output_dir: Path | str,
    overview_basename: str = "pairwise_correction_model_overview",
    encoder_basename: str = "shared_cnn_encoder_detail",
    compile_pdf: bool = True,
    cleanup_aux: bool = False,
) -> PairwiseArchitectureRenderSet:
    """Render both pairwise architecture views from one description/model."""
    desc = _coerce_pairwise_architecture(
        description_or_model,
        input_shape=input_shape,
        model_config=model_config,
    )
    overview = render_pairwise_correction_model_overview(
        desc,
        output_dir=output_dir,
        basename=overview_basename,
        compile_pdf=compile_pdf,
        cleanup_aux=cleanup_aux,
    )
    encoder = render_shared_cnn_encoder_detail(
        desc,
        output_dir=output_dir,
        basename=encoder_basename,
        compile_pdf=compile_pdf,
        cleanup_aux=cleanup_aux,
    )
    return PairwiseArchitectureRenderSet(overview=overview, encoder=encoder)


def render_pairwise_correction_architecture(
    model: Any,
    *,
    input_shape: Sequence[int] = (1, 160, 160),
    model_config: Mapping[str, Any] | None = None,
    output_dir: Path | str,
    basename: str = "pairwise_correction_architecture",
    compile_pdf: bool = True,
    cleanup_aux: bool = False,
) -> ArchitectureRenderResult:
    """Render the pairwise correction CNN overview to PlotNeuralNet/TikZ.

    Parameters
    ----------
    model
        Actual ``PairwiseCorrectionCNN``-like model instance being trained.
    input_shape
        Single-image ``(channels, height, width)`` shape used to compute
        displayed convolution and pooling dimensions.
    model_config
        Optional model configuration mapping for semantic fallback metadata.
    output_dir
        Directory where ``.tex``, ``.json``, and optionally ``.pdf`` artifacts
        are written.  It is created only when this function is called.
    basename
        Basename for generated architecture artifact files.
    compile_pdf
        If ``True``, run ``pdflatex`` in the output directory.  Missing LaTeX is
        reported as ``ArchitectureVisualizationError`` and never affects imports
        or training unless this rendering path is explicitly called.
    cleanup_aux
        Remove LaTeX ``.aux`` and ``.log`` files after a successful compile.

    Returns
    -------
    ArchitectureRenderResult
        Paths and metadata for the overview artifact.  This function preserves
        the original return type and basename default for callers that used the
        previous single-figure API.  New code should call
        :func:`render_pairwise_correction_architecture_set` when it needs both
        the overview and shared-encoder detail figures.
    """
    return render_pairwise_correction_model_overview(
        model,
        input_shape=input_shape,
        model_config=model_config,
        output_dir=output_dir,
        basename=basename,
        compile_pdf=compile_pdf,
        cleanup_aux=cleanup_aux,
    )
