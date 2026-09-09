from __future__ import annotations

from .arrays import ArrayShardReader, ArrayShardStore, ShardRecord
from .master_v4 import (
    FrozenMasterV4,
    FrozenV4Identities,
    ResolvedRenderState,
    SlurmArrayPlan,
    array_task_range,
    build_slurm_array_plan,
    locate_science_global_index,
    render_index_location,
    render_index_to_science_nuisance,
    render_output_paths,
    render_state_id,
    science_nuisance_to_render_index,
    shard_name_for_render_index,
)
from .prepared_v4 import PREPARED_V4_ARTIFACT_ID, PreparedV4Summary, prepare_shera_v4_dataset
from .schema import VectorComponentSpec, VectorSpaceSpec
from .splitting import GroupedSplitResult, assign_grouped_split
from .transforms import (
    CompositeTransform,
    CoordinateTransform,
    DiagonalScaleTransform,
    LinearTransform,
)
from .validation import ArrayComparisonResult, compare_arrays
from .rendering import (
    RenderResult,
    apply_absolute_vector_to_store,
    apply_sample_deltas_to_store,
    refresh_preserving_derived_keys,
    render_image,
    set_scalar_label,
    write_fits,
)

__all__ = [
    "ArrayComparisonResult",
    "ArrayShardReader",
    "ArrayShardStore",
    "CompositeTransform",
    "CoordinateTransform",
    "DiagonalScaleTransform",
    "FrozenMasterV4",
    "FrozenV4Identities",
    "GroupedSplitResult",
    "LinearTransform",
    "PREPARED_V4_ARTIFACT_ID",
    "PreparedV4Summary",
    "RenderResult",
    "ResolvedRenderState",
    "ShardRecord",
    "SlurmArrayPlan",
    "VectorComponentSpec",
    "VectorSpaceSpec",
    "assign_grouped_split",
    "apply_absolute_vector_to_store",
    "apply_sample_deltas_to_store",
    "array_task_range",
    "build_slurm_array_plan",
    "compare_arrays",
    "locate_science_global_index",
    "refresh_preserving_derived_keys",
    "render_image",
    "render_index_location",
    "render_index_to_science_nuisance",
    "render_output_paths",
    "render_state_id",
    "prepare_shera_v4_dataset",
    "science_nuisance_to_render_index",
    "set_scalar_label",
    "shard_name_for_render_index",
    "write_fits",
]
