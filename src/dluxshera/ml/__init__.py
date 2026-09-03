from __future__ import annotations

from .catalog import SampleCatalog, load_sample_catalog
from .metrics import compute_regression_metrics, metrics_by_group, transform_z_to_physical
from .noise import NoiseConfig, apply_pair_noise
from .pairs import (
    PairManifest,
    PairPolicy,
    PairRecord,
    PairSampler,
    generate_frozen_pair_manifest,
    load_pair_manifest,
    make_reverse_pair_record,
    write_pair_manifest,
)
from .scaling import IntensityScaler, fit_intensity_scaler
from .splits import SplitRegistry, generate_split_registry, load_split_registry, write_split_registry
from .visualization import (
    ArchitectureRenderResult,
    ArchitectureVisualizationError,
    PairwiseArchitectureRenderSet,
    PairwiseCorrectionArchitecture,
    describe_pairwise_correction_architecture,
    render_pairwise_correction_architecture_set,
    render_pairwise_correction_architecture,
    render_pairwise_correction_model_overview,
    render_shared_cnn_encoder_detail,
    resolve_pdflatex,
)

__all__ = [
    "ArchitectureRenderResult",
    "ArchitectureVisualizationError",
    "IntensityScaler",
    "NoiseConfig",
    "PairwiseArchitectureRenderSet",
    "PairManifest",
    "PairPolicy",
    "PairRecord",
    "PairSampler",
    "PairwiseCorrectionArchitecture",
    "SampleCatalog",
    "SplitRegistry",
    "apply_pair_noise",
    "compute_regression_metrics",
    "describe_pairwise_correction_architecture",
    "fit_intensity_scaler",
    "generate_frozen_pair_manifest",
    "generate_split_registry",
    "load_pair_manifest",
    "load_sample_catalog",
    "load_split_registry",
    "make_reverse_pair_record",
    "render_pairwise_correction_architecture_set",
    "metrics_by_group",
    "render_pairwise_correction_architecture",
    "render_pairwise_correction_model_overview",
    "render_shared_cnn_encoder_detail",
    "resolve_pdflatex",
    "transform_z_to_physical",
    "write_pair_manifest",
    "write_split_registry",
]
