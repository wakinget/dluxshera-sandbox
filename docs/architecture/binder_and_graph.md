# Binder and SystemGraph

## Binder as the model interface
- The Binder wraps an optical system and exposes a concise API: supply a parameter delta (or updated store) and receive PSFs/images.
- It owns the base `ParameterStore` and configuration, applying deltas before evaluation so callers do not manipulate optics internals directly.
- Designed to be the stable facade, replacing earlier monolithic model classes.

## SystemGraph as execution DAG
- Internally, the Binder uses a `SystemGraph` DAG to organise computation nodes (e.g., optics builder, detector stages).
- The graph allows clearer structuring of intermediate values and paves the way for caching and multi-node systems beyond the current single-node scaffold.
- Binder construction can enable `use_system_graph=True` so forward passes run through the DAG while keeping the external API unchanged.

## Migration from legacy models
- Older model classes coupled parameter handling, optics construction, and execution; the Binder/SystemGraph split separates these concerns.
- New examples and demos (e.g., the canonical astrometry script) use the Binder as the canonical entry point.
