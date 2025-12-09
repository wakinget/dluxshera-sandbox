import jax.numpy as jnp
import pytest

from dluxshera.params.store import ParameterStore
from dluxshera.params.transforms import (
    DEFAULT_SYSTEM_ID,
    DERIVED_RESOLVER,
    DerivedResolver,
    Transform,
    TransformCycleError,
    TransformDepthError,
    TransformMissingDependencyError,
    TransformRegistry,
)


def test_simple_transform_separation_pix():
    """
    separation_pix = separation_as / plate_scale_as_per_pix
    """
    registry = TransformRegistry()

    transform = Transform(
        key="binary.separation_pix",
        depends_on=("binary.separation_as", "imaging.plate_scale_as_per_pix"),
        doc="Convert binary separation from arcsec to pixels.",
        fn=lambda ctx: ctx["binary.separation_as"]
        / ctx["imaging.plate_scale_as_per_pix"],
    )
    registry.register(transform)

    store = ParameterStore.from_dict(
        {
            "binary.separation_as": 10.0,  # arcsec
            "imaging.plate_scale_as_per_pix": 0.5,  # arcsec/pixel
        }
    )

    value = registry.compute("binary.separation_pix", store)
    assert value == pytest.approx(20.0)


def test_recursive_transforms_chain():
    """
    Test a chain of transforms:
      c = primitive
      b = c + 1
      a = 2 * b
    """
    registry = TransformRegistry()

    t_b = Transform(
        key="b",
        depends_on=("c",),
        fn=lambda ctx: ctx["c"] + 1.0,
        doc="b = c + 1",
    )
    registry.register(t_b)

    t_a = Transform(
        key="a",
        depends_on=("b",),
        fn=lambda ctx: 2.0 * ctx["b"],
        doc="a = 2 * b",
    )
    registry.register(t_a)

    store = ParameterStore.from_dict({"c": 3.0})

    # a = 2 * (c + 1) = 2 * 4 = 8
    value_a = registry.compute("a", store)
    assert value_a == pytest.approx(8.0)


def test_missing_dependency_raises():
    """
    If a dependency cannot be resolved from either the store or the registry,
    TransformMissingDependencyError is raised.
    """
    registry = TransformRegistry()

    t_a = Transform(
        key="a",
        depends_on=("b",),
        fn=lambda ctx: ctx["b"] + 1.0,
        doc="a = b + 1",
    )
    registry.register(t_a)

    # Store has no 'b', and there is no transform for 'b' either.
    store = ParameterStore.from_dict({})

    with pytest.raises(TransformMissingDependencyError):
        registry.compute("a", store)


def test_cycle_detection():
    """
    A simple cycle:
      a depends on b
      b depends on a
    should raise TransformCycleError.
    """
    registry = TransformRegistry()

    t_a = Transform(
        key="a",
        depends_on=("b",),
        fn=lambda ctx: ctx["b"] + 1.0,
        doc="a = b + 1 (cycle test)",
    )
    t_b = Transform(
        key="b",
        depends_on=("a",),
        fn=lambda ctx: ctx["a"] + 1.0,
        doc="b = a + 1 (cycle test)",
    )
    registry.register(t_a)
    registry.register(t_b)

    store = ParameterStore.from_dict({})

    with pytest.raises(TransformCycleError):
        registry.compute("a", store)


def test_max_depth_guard():
    """
    Ensure that very deep chains trigger TransformDepthError when they exceed
    the configured max_depth.
    """
    # Note: the transform functions use a default argument dep_key=dep to capture
    # the current dep value in the loop (classic Python closure pattern), so each
    # lambda reads from the correct dependency.

    registry = TransformRegistry()

    # Build a chain of transforms: t0 depends on t1, t1 on t2, ..., tN on "leaf"
    chain_length = 10
    for i in range(chain_length):
        key = f"t{i}"
        dep = "leaf" if i == chain_length - 1 else f"t{i+1}"
        registry.register(
            Transform(
                key=key,
                depends_on=(dep,),
                fn=lambda ctx, dep_key=dep: ctx[dep_key] + 1.0,
                doc=f"{key} = {dep} + 1",
            )
        )

    store = ParameterStore.from_dict({"leaf": 0.0})

    # With a low max_depth, this should fail.
    with pytest.raises(TransformDepthError):
        registry.compute("t0", store, max_depth=3)

    # With a sufficiently large max_depth, it should succeed.
    value = registry.compute("t0", store, max_depth=32)
    # The value should be chain_length steps above leaf.
    assert value == pytest.approx(float(chain_length))


def test_scoped_resolver_isolates_systems():
    resolver = DerivedResolver(default_system_id="sys_a")

    @resolver.register_transform("value", depends_on=("p",), system_id="sys_a")
    def value_a(ctx):
        return ctx["p"] + 1

    store = ParameterStore.from_dict({"p": 1})

    assert resolver.compute("value", store, system_id="sys_a") == pytest.approx(2)

    with pytest.raises(TransformMissingDependencyError) as excinfo:
        resolver.compute("value", store, system_id="sys_b")
    assert "sys_b" in str(excinfo.value)

    @resolver.register_transform("value", depends_on=("p",), system_id="sys_b")
    def value_b(ctx):
        return ctx["p"] * 10

    assert resolver.compute("value", store, system_id="sys_b") == pytest.approx(10)


def test_global_resolver_default_system_matches_constant():
    assert DERIVED_RESOLVER.default_system_id == DEFAULT_SYSTEM_ID
