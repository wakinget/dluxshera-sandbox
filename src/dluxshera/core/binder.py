# src/dluxshera/core/binder.py
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Optional

import jax.numpy as jnp
import dLux as dl

from ..optics.config import SheraThreePlaneConfig, SheraTwoPlaneConfig
from ..optics.builder import (
    apply_runtime_bindings,
    build_shera_threeplane_optics,
    build_shera_twoplane_optics,
    structural_hash_for_twoplane,
    structural_hash_from_config,
    THREEPLANE_RUNTIME_BINDINGS,
    TWOPLANE_RUNTIME_BINDINGS,
)
from ..params.spec import ParamSpec
from ..params.store import ParameterStore, strip_structural
from ..params.store_namespace import StoreNamespace


BINDER_RESERVED_NAMES = {
    "cfg",
    "forward_spec",
    "base_forward_store",
    "get",
    "ns",
    "model",
    "with_store",
}
from .universe import build_alpha_cen_source


class BaseSheraBinder:
    """Shared backbone for Shera binder implementations.

    Encapsulates the common binder behaviour: storing config/spec/base-store,
    eager detector construction, functional store merge, and the public
    ``.model`` / ``.with_store`` helpers. Concrete subclasses remain the public
    entry points and supply system-specific optics/source builders via
    protected hooks. Methods that update the base store, such as
    :meth:`update_store`, always return a new binder instance. The binder owns
    a persistent telescope built from the baseline store; fast-path
    evaluations reuse these cached objects instead of rebuilding optics each
    call.
    """

    def __init__(
        self,
        cfg,
        forward_spec: ParamSpec,
        base_forward_store: ParameterStore,
    ) -> None:
        self.cfg = cfg
        self.forward_spec = forward_spec
        self.structural_hash = self._compute_structural_hash()

        # Validate and freeze the base forward store; derived values are allowed
        # because forward_spec includes them explicitly.
        self.base_forward_store = base_forward_store.validate_against(
            forward_spec, allow_derived=True
        )

        # Shared detector construction; subclasses can override if needed.
        self._detector = self._build_detector()
        self.telescope = self._build_telescope(self.base_forward_store)


    def __dir__(self):
        entries = set(super().__dir__())
        reserved = BINDER_RESERVED_NAMES

        def _maybe_add(name: str):
            if (
                isinstance(name, str)
                and name not in reserved
                and name.isidentifier()
                and not name.startswith("__")
            ):
                entries.add(name)

        if is_dataclass(self.cfg):
            for field in fields(self.cfg):
                _maybe_add(field.name)

        prefixes = set()
        for key in self.base_forward_store.keys():
            if "." not in key:
                continue
            prefix, _ = key.split(".", 1)
            prefixes.add(prefix)

        for prefix in prefixes:
            _maybe_add(prefix)

        for leaf, candidates in self._leaf_index().items():
            if len(candidates) == 1:
                _maybe_add(leaf)

        return sorted(entries)

    def __getattr__(self, name):
        if name in BINDER_RESERVED_NAMES:
            raise AttributeError(name)

        if hasattr(self.cfg, name):
            return getattr(self.cfg, name)

        has_prefix = name.isidentifier() and any(
            key.startswith(f"{name}.") for key in self.base_forward_store.keys()
        )
        if has_prefix:
            return self.ns(name)

        leaf_index = self._leaf_index()
        if name in leaf_index:
            candidates = leaf_index[name]
            if len(candidates) == 1:
                return self.base_forward_store.get(candidates[0])

            candidate_list = ", ".join(sorted(candidates))
            raise AttributeError(
                "Ambiguous leaf name {leaf!r} found in store keys: {candidates}. "
                "Use binder.<prefix>.{leaf} or binder.get(\"<full.key>\")".format(
                    leaf=name, candidates=candidate_list
                )
            )

        raise AttributeError(name)

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _build_detector(self) -> dl.LayeredDetector:
        return dl.LayeredDetector(layers=[("downsample", dl.Downsample(self.cfg.oversample))])

    def _build_optics(self, store: ParameterStore):  # pragma: no cover - abstract hook
        raise NotImplementedError

    def _build_source(self, store: ParameterStore):  # pragma: no cover - abstract hook
        raise NotImplementedError

    def _build_telescope(self, store: ParameterStore) -> dl.Telescope:
        return dl.Telescope(
            source=self._build_source(store),
            optics=self._build_optics(store),
            detector=self._detector,
        )

    def _direct_model(self, eff_store: ParameterStore) -> jnp.ndarray:  # pragma: no cover - abstract hook
        raise NotImplementedError

    def _runtime_bindings(self) -> tuple[tuple[str, str], ...]:
        return ()

    def _compute_structural_hash(self) -> Optional[str]:
        return None

    def _update_telescope_runtime(self, store: ParameterStore) -> dl.Telescope:
        optics = apply_runtime_bindings(
            self.telescope.optics,
            store,
            self._runtime_bindings(),
        )
        source = self._build_source(store)
        return dl.Telescope(source=source, optics=optics, detector=self._detector)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _merge_store(self, store_delta: Optional[ParameterStore]) -> ParameterStore:
        """Merge a (possibly partial) store into the base forward store."""

        if store_delta is None:
            return self.base_forward_store

        store_delta = store_delta.validate_against(
            self.forward_spec,
            allow_missing=True,
            allow_extra=False,
            allow_derived=True,
        )
        return self.base_forward_store.replace(store_delta.as_dict())

    def _leaf_index(self) -> dict[str, list[str]]:
        """Build an index mapping leaf names to full store paths."""

        leaf_index: dict[str, list[str]] = {}

        for key in self.base_forward_store.keys():
            if "." not in key:
                continue

            leaf = key.split(".")[-1]
            leaf_index.setdefault(leaf, []).append(key)

        return leaf_index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, paths, default=None):
        """Retrieve values from the binder configuration or base store."""

        if isinstance(paths, (list, tuple)):
            return [self.get(path, default=default) for path in paths]

        path = paths
        if isinstance(path, str) and "." in path:
            if default is None:
                return self.base_forward_store.get(path)
            return self.base_forward_store.get(path, default)

        if hasattr(self.cfg, path):
            return getattr(self.cfg, path)

        if default is None:
            return self.base_forward_store.get(path)
        return self.base_forward_store.get(path, default)

    def ns(self, prefix: str) -> StoreNamespace:
        """Return a StoreNamespace proxy for a prefix in the base forward store."""

        if not isinstance(prefix, str) or not prefix.isidentifier():
            raise ValueError(f"Invalid namespace prefix: {prefix!r}")

        if prefix in BINDER_RESERVED_NAMES:
            raise ValueError(f"Namespace prefix {prefix!r} is reserved")

        has_prefix = any(
            key.startswith(f"{prefix}.") for key in self.base_forward_store.keys()
        )
        if not has_prefix:
            raise ValueError(f"No store keys found under prefix {prefix!r}")

        return StoreNamespace(self.base_forward_store, prefix)

    def _structural_store_keys(self) -> set[str]:
        """Return store keys treated as structural for this binder."""

        structural_keys = {
            key
            for key in self.forward_spec.keys()
            if key.startswith(("system.", "band."))
        }
        runtime_keys = {store_key for store_key, _ in self._runtime_bindings()}
        return structural_keys - runtime_keys

    def model(
        self,
        store_delta: Optional[ParameterStore] = None,
        *,
        allow_rebuild: bool = False,
    ) -> jnp.ndarray:
        """Evaluate the Shera PSF for an optional store overlay.

        With ``store_delta=None`` this uses the stored telescope directly
        for a fast-path evaluation that reuses the persistent binder state.
        When providing a ``store_delta`` the binder only accepts non-structural
        keys by default; this is the slower path that merges an overlay and
        evaluates with updated values. Pass ``allow_rebuild=True`` to validate
        a full store and delegate to :meth:`update_store` when structural keys
        need to change.
        """

        if store_delta is None:
            return self.telescope.model()

        if allow_rebuild:
            return self.update_store(store_delta).model()

        structural_keys = self._structural_store_keys()
        non_structural = strip_structural(store_delta, structural_keys=structural_keys)
        provided_structural = sorted(
            key for key in store_delta.keys() if key not in non_structural
        )
        if provided_structural:
            joined = ", ".join(provided_structural)
            raise ValueError(
                "model() only accepts non-structural store keys; "
                f"found structural keys: {joined}. "
                "Use allow_rebuild=True with a full store to rebuild."
            )

        eff_store = self._merge_store(non_structural)

        return self._direct_model(eff_store)

    @property
    def optics(self):
        return self.telescope.optics

    @property
    def source(self):
        return self.telescope.source

    @property
    def detector(self):
        return self.telescope.detector

    # ------------------------------------------------------------------
    # Mostly immutable helpers
    # ------------------------------------------------------------------

    def with_store(self, new_base_store: ParameterStore):
        """Return a new Binder sharing cfg/spec but with a different base store."""

        return self.__class__(
            cfg=self.cfg,
            forward_spec=self.forward_spec,
            base_forward_store=new_base_store,
        )

    def update_store(self, store: ParameterStore):
        """Return a new Binder with an updated base store.

        The incoming store is validated against ``forward_spec``. If the
        configuration structure has changed (based on the stored structural
        hash), the telescope is rebuilt and a warning is emitted. Otherwise the
        telescope optics are updated in-place via runtime bindings for a
        lightweight refresh. Use ``update_store`` when you want a new baseline
        (e.g., to make a structural change or to persist a new truth store)
        rather than supplying a per-call overlay.
        """

        validated_store = store.validate_against(
            self.forward_spec,
            allow_derived=True,
        )

        new_structural_hash = self._compute_structural_hash()
        structural_hash_changed = new_structural_hash != self.structural_hash
        structural_store_changed = False
        for key in self._structural_store_keys():
            current_value = self.base_forward_store.get(key)
            incoming_value = validated_store.get(key)
            try:
                values_equal = bool(
                    jnp.array_equal(jnp.asarray(current_value), jnp.asarray(incoming_value))
                )
            except Exception:
                values_equal = current_value == incoming_value
            if not values_equal:
                structural_store_changed = True
                break

        structural_changed = structural_hash_changed or structural_store_changed

        if structural_changed:
            import warnings

            reasons = []
            if structural_hash_changed:
                reasons.append("structural config hash changed")
            if structural_store_changed:
                reasons.append("structural store values changed")
            reason_text = " and ".join(reasons)
            warnings.warn(
                f"{reason_text.capitalize()}; rebuilding telescope and binder state.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.with_store(validated_store)

        updated = self.__class__.__new__(self.__class__)
        updated.cfg = self.cfg
        updated.forward_spec = self.forward_spec
        updated.base_forward_store = validated_store
        updated.structural_hash = new_structural_hash
        updated._detector = self._detector
        updated.telescope = self._update_telescope_runtime(validated_store)
        return updated


@dataclass
class SheraThreePlaneBinder(BaseSheraBinder):
    """
    Canonical generative model for the Shera three-plane system.

    Binder is the successor to the legacy ``SheraThreePlane_Model`` facade and
    is intentionally treated as **mostly immutable**: instantiate it once for a
    given configuration + base forward store (with deriveds populated), then use
    ``.model(store_delta)`` to evaluate PSFs without mutating internal state.

    Key properties
    --------------
    - Holds the Shera config, forward ParamSpec, and a *forward-style* base
      ParameterStore (derived values already populated).
    - ``.model()`` is the primary API and is intentionally lightweight: with
      ``store_delta=None`` it fast-paths through the cached telescope. For
      non-structural overlays it merges ``store_delta`` onto the base store,
      then evaluates the direct builder path. Structural overrides require
      ``allow_rebuild=True`` and delegate to ``update_store()``.
    - ``.update_store()`` returns a new binder instance with the refreshed base
      store; the original binder remains unchanged.
    """

    cfg: SheraThreePlaneConfig
    forward_spec: ParamSpec
    base_forward_store: ParameterStore
    # Internal detector references (prepared eagerly)
    _detector: Optional[dl.LayeredDetector] = None

    def __init__(
        self,
        cfg: SheraThreePlaneConfig,
        forward_spec: ParamSpec,
        base_forward_store: ParameterStore,
    ) -> None:
        super().__init__(
            cfg=cfg,
            forward_spec=forward_spec,
            base_forward_store=base_forward_store,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _direct_model(self, eff_store: ParameterStore) -> jnp.ndarray:
        return self._build_telescope(eff_store).model()

    def _build_optics(self, store: ParameterStore):
        return build_shera_threeplane_optics(
            self.cfg, store=store, spec=self.forward_spec
        )

    def _build_source(self, store: ParameterStore):
        return build_alpha_cen_source(store, cfg=self.cfg)

    def _runtime_bindings(self) -> tuple[tuple[str, str], ...]:
        return THREEPLANE_RUNTIME_BINDINGS

    def _compute_structural_hash(self) -> Optional[str]:
        return structural_hash_from_config(self.cfg)

    with_store = BaseSheraBinder.with_store


@dataclass
class SheraTwoPlaneBinder(BaseSheraBinder):
    """Generative model for the Shera two-plane system.

    Mirrors :class:`SheraThreePlaneBinder` semantics: mostly immutable, owns a
    forward-spec-validated base store, and exposes ``.model(store_delta)`` as the
    canonical evaluation path. ``.model`` fast-paths through the cached
    telescope when ``store_delta`` is omitted, and accepts non-structural
    overlays by default when an explicit delta is provided. Structural updates
    require ``allow_rebuild=True`` to rebuild the binder state. When
    ``.update_store()`` returns a new binder instance with the refreshed base
    store so the original binder remains unchanged.
    """

    cfg: SheraTwoPlaneConfig
    forward_spec: ParamSpec
    base_forward_store: ParameterStore
    _detector: Optional[dl.LayeredDetector] = None

    def __init__(
        self,
        cfg: SheraTwoPlaneConfig,
        forward_spec: ParamSpec,
        base_forward_store: ParameterStore,
    ) -> None:
        super().__init__(
            cfg=cfg,
            forward_spec=forward_spec,
            base_forward_store=base_forward_store,
        )

    def _direct_model(self, eff_store: ParameterStore) -> jnp.ndarray:
        return self._build_telescope(eff_store).model()

    def _build_optics(self, store: ParameterStore):
        return build_shera_twoplane_optics(self.cfg, store=store, spec=self.forward_spec)

    def _build_source(self, store: ParameterStore):
        return build_alpha_cen_source(store, cfg=self.cfg)

    def _runtime_bindings(self) -> tuple[tuple[str, str], ...]:
        return TWOPLANE_RUNTIME_BINDINGS

    def _compute_structural_hash(self) -> Optional[str]:
        return structural_hash_for_twoplane(self.cfg)

    with_store = BaseSheraBinder.with_store
