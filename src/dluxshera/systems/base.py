"""Base system interfaces and shared helpers for Shera system binders."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace as dataclass_replace
from typing import Optional, Sequence, Self

import jax.numpy as jnp
import dLux as dl

from ..params.spec import ParamSpec
from ..params.store import ParameterStore, StoreNamespace


class BaseConfig:
    """Shared helpers for immutable Shera configuration dataclasses."""

    def replace(self, **kwargs) -> Self:
        """Return a new config with the provided fields updated.

        This is a thin wrapper around :func:`dataclasses.replace` that keeps the
        configs frozen/immutable while providing an ergonomic, discoverable
        update path mirroring :meth:`dluxshera.params.store.ParameterStore.replace`.
        """

        return dataclass_replace(self, **kwargs)


BINDER_RESERVED_NAMES = {
    "cfg",
    "forward_spec",
    "base_forward_store",
    "get",
    "ns",
    "model",
    "with_store",
}


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

    cfg: "SheraThreePlaneConfig | SheraTwoPlaneConfig"
    forward_spec: ParamSpec
    base_forward_store: ParameterStore
    structural_hash: Optional[str]
    telescope: dl.Telescope

    def __init__(
        self,
        cfg,
        forward_spec: ParamSpec,
        base_forward_store: ParameterStore,
    ) -> None:
        """Initialize the binder with a config, parameter spec, and base store.

        This sets up the shared baseline state used by all binder evaluations:
        the configuration, parameter spec, a validated base forward store, and
        cached telescope instance for fast-path evaluations. Use this
        when constructing a new binder from a known-good configuration and
        fully populated forward store.

        Parameters
        ----------
        cfg : Any
            Configuration object for the optics/source builders. The object is
            stored as-is and may be a dataclass with attributes accessed via
            ``__getattr__``/``get``.
        forward_spec : ParamSpec
            Parameter specification describing the allowed store keys and
            their metadata (including structural keys).
        base_forward_store : ParameterStore
            Forward-style store containing the baseline parameter values.
            Derived values are allowed and validated against ``forward_spec``.

        Returns
        -------
        None
            Initializes the binder instance in-place.

        Raises
        ------
        ValueError
            If ``base_forward_store`` fails validation against ``forward_spec``.
        """
        self.cfg = cfg
        self.forward_spec = forward_spec
        self.structural_hash = self._compute_structural_hash()

        # Validate and freeze the base forward store; derived values are allowed
        # because forward_spec includes them explicitly.
        self.base_forward_store = base_forward_store.validate_against(
            forward_spec, allow_derived=True
        )

        detector = self._build_detector()
        self.telescope = self._build_telescope(self.base_forward_store, detector=detector)

    def __dir__(self) -> list[str]:
        """List attribute names available on the binder.

        This augments the default ``dir()`` output with configuration fields,
        store namespace prefixes, and unique leaf keys (when unambiguous).
        Use this for discovery in interactive sessions; it does not mutate the
        binder and only reflects the current base store/config.

        Returns
        -------
        list[str]
            Sorted attribute names that can be accessed on the binder.
        """
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

    def __getattr__(self, name: str) -> object:
        """Resolve dynamic attributes from the config or base store.

        Resolution order:
        1) Configuration attributes (e.g., ``binder.oversample``).
        2) Namespace proxies for store prefixes (``binder.ns("prefix")``).
        3) Unique leaf names in the store (unambiguous suffixes).

        Use this for ergonomic access to configuration fields and store values.
        For ambiguous leaf names, prefer ``binder.<prefix>.<leaf>`` or
        ``binder.get("full.key")``.

        Parameters
        ----------
        name : str
            Attribute name being requested.

        Returns
        -------
        Any
            Configuration attribute value, store namespace proxy, or store
            value.

        Raises
        ------
        AttributeError
            If ``name`` is reserved, missing, or refers to an ambiguous leaf
            name in the store.
        """
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
        """Construct the detector instance for the binder.

        Called during binder initialization to create a detector that can be
        cached across model evaluations. Subclasses may override this to
        provide a different detector topology or to derive detector parameters
        from the configuration.

        Returns
        -------
        dl.LayeredDetector
            Detector instance used by the cached telescope and runtime updates.

        Notes
        -----
        The returned detector is intended to be immutable at runtime; changing
        detector structure should trigger a full binder rebuild instead of
        runtime bindings.
        """
        from ..builders.detector import build_detector

        return build_detector(self.cfg)

    def _build_optics(self, store: ParameterStore):  # pragma: no cover - abstract hook
        """Build the optics model for the given store.

        Called when constructing a telescope (either at initialization or via
        direct model evaluation). Implementations must read any required
        structural and non-structural keys from ``store``.

        Parameters
        ----------
        store : ParameterStore
            Fully validated store providing both structural and runtime values.

        Returns
        -------
        dl.OpticalLayer | dl.Optics
            Optics object compatible with ``dl.Telescope``.

        Notes
        -----
        Subclasses should treat structural changes in ``store`` as requiring a
        rebuild. Non-structural keys that can be updated at runtime should be
        surfaced via :meth:`_optics_runtime_bindings`.
        """
        raise NotImplementedError

    def _build_source(self, store: ParameterStore):  # pragma: no cover - abstract hook
        """Build the source model for the given store.

        Called when constructing a telescope or applying runtime updates that
        change source parameters. Subclasses should read store values needed
        to build the source and return a ``dl.Source`` compatible object.

        Parameters
        ----------
        store : ParameterStore
            Store providing source parameters. Typically contains non-
            structural keys, but may include structural keys if the source
            configuration is structural for the system.

        Returns
        -------
        dl.Source
            Source object to inject into the telescope model.

        Notes
        -----
        Source construction is typically lightweight; it is rebuilt for
        runtime updates even when optics are updated via bindings.
        """
        raise NotImplementedError

    def _build_telescope(
        self,
        store: ParameterStore,
        *,
        detector: Optional[dl.LayeredDetector] = None,
    ) -> dl.Telescope:
        """Build a full telescope model from the given store.

        Called during initialization (to create the cached telescope) and by
        direct model evaluation paths. Subclasses may override this if they
        need custom telescope assembly logic.

        Parameters
        ----------
        store : ParameterStore
            Validated store containing all parameters required to build the
            source, optics, and detector.
        detector : dl.LayeredDetector, optional
            Detector instance to reuse. When omitted, a new detector is built
            via :meth:`_build_detector`.

        Returns
        -------
        dl.Telescope
            Fully constructed telescope.
        """
        if detector is None:
            detector = self._build_detector()

        return dl.Telescope(
            source=self._build_source(store),
            optics=self._build_optics(store),
            detector=detector,
        )

    def _direct_model(self, eff_store: ParameterStore) -> jnp.ndarray:  # pragma: no cover - abstract hook
        """Evaluate the model using a fully merged effective store.

        Called by :meth:`model` after merging a non-structural store delta with
        the base store. Subclasses should implement this as a direct modeling
        path (usually by building a telescope and calling ``model()``).

        Parameters
        ----------
        eff_store : ParameterStore
            Fully validated store that includes all values needed for a model
            evaluation.

        Returns
        -------
        jax.numpy.ndarray
            The evaluated PSF model output.

        Notes
        -----
        This method should not mutate the binder; use runtime bindings or
        rebuild logic for structural changes instead of modifying state here.
        """
        raise NotImplementedError

    def _optics_runtime_bindings(self) -> tuple[tuple[str, str], ...]:
        """Return runtime binding pairs for non-structural optics keys.

        Runtime bindings map store keys to optics attributes (or paths) that
        can be updated without rebuilding the full optics model. Subclasses
        should override this to list the non-structural optics keys eligible
        for fast-path updates.
        """
        return ()

    def _source_runtime_bindings(self) -> tuple[tuple[str, str], ...]:
        """Return runtime binding pairs for non-structural source keys."""
        from ..builders.source import SOURCE_RUNTIME_BINDINGS

        return SOURCE_RUNTIME_BINDINGS

    def _detector_runtime_bindings(self) -> tuple[tuple[str, str], ...]:
        """Return runtime binding pairs for non-structural detector keys."""
        from ..builders.detector import DETECTOR_RUNTIME_BINDINGS

        return DETECTOR_RUNTIME_BINDINGS

    def _compute_structural_hash(self) -> Optional[str]:
        """Compute a structural hash for the current configuration.

        Called during initialization and store updates to detect structural
        configuration changes that require a full rebuild. Subclasses should
        return a stable hash string or ``None`` if structural hashing is not
        applicable.

        Returns
        -------
        str | None
            Hash representing structural config state, or ``None``.

        Notes
        -----
        The structural hash is compared against the stored value to determine
        if runtime bindings are safe or if a rebuild is required.
        """
        return None

    def _optics_structural_keys(self) -> set[str]:
        """Return store keys treated as structural for the optics component."""

        structural_keys = {
            key
            for key in self.forward_spec.keys()
            if key.startswith(("system.", "band."))
        }
        runtime_keys = {store_key for store_key, _ in self._optics_runtime_bindings()}
        return structural_keys - runtime_keys

    def _source_structural_keys(self) -> set[str]:
        """Return store keys treated as structural for the source component."""

        return set()

    def _detector_structural_keys(self) -> set[str]:
        """Return store keys treated as structural for the detector component."""

        return set()

    def _apply_runtime_updates(self, store: ParameterStore) -> dl.Telescope:
        """Apply runtime bindings to update cached telescope components."""

        from ..builders import detector as detector_builder
        from ..builders import optics as optics_builder
        from ..builders import source as source_builder

        optics = optics_builder.apply_runtime_bindings(
            self.telescope.optics,
            store,
            self._optics_runtime_bindings(),
        )
        source = source_builder.apply_runtime_bindings(
            self.telescope.source,
            store,
            cfg=self.cfg,
            bindings=self._source_runtime_bindings(),
        )
        detector = detector_builder.apply_runtime_bindings(
            self.telescope.detector,
            store,
            self._detector_runtime_bindings(),
        )
        return dl.Telescope(source=source, optics=optics, detector=detector)

    def _rebuild_telescope(
        self,
        store: ParameterStore,
        *,
        structural_components: set[str],
    ) -> dl.Telescope:
        """Rebuild structural components while applying runtime updates."""

        from ..builders import detector as detector_builder
        from ..builders import optics as optics_builder
        from ..builders import source as source_builder

        if "optics" in structural_components:
            optics = self._build_optics(store)
        else:
            optics = optics_builder.apply_runtime_bindings(
                self.telescope.optics,
                store,
                self._optics_runtime_bindings(),
            )

        if "source" in structural_components:
            source = self._build_source(store)
        else:
            source = source_builder.apply_runtime_bindings(
                self.telescope.source,
                store,
                cfg=self.cfg,
                bindings=self._source_runtime_bindings(),
            )

        if "detector" in structural_components:
            detector = self._build_detector()
        else:
            detector = self.telescope.detector

        detector = detector_builder.apply_runtime_bindings(
            detector,
            store,
            self._detector_runtime_bindings(),
        )

        return dl.Telescope(source=source, optics=optics, detector=detector)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _merge_store(self, store_delta: Optional[ParameterStore]) -> ParameterStore:
        """Merge a (possibly partial) store into the base forward store.

        Called by :meth:`model` to overlay a delta of non-structural values
        onto the baseline store. The delta is validated against the forward
        spec and may be partial; the resulting store is a full, validated
        forward store.

        Parameters
        ----------
        store_delta : ParameterStore | None
            Partial overlay of values. When ``None``, the base store is
            returned unchanged.

        Returns
        -------
        ParameterStore
            The merged forward store used for evaluation.

        Notes
        -----
        This helper does not accept structural changes; those are handled by
        :meth:`update_store` and require a rebuild.
        """

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
        """Build an index mapping leaf names to full store paths.

        Called by ``__dir__`` and ``__getattr__`` to allow ergonomic access to
        store values by leaf name (suffix). This is a read-only helper that
        scans the base store keys.

        Returns
        -------
        dict[str, list[str]]
            Mapping of leaf name to all matching fully-qualified store keys.

        Notes
        -----
        Leaf-name access is only provided when the leaf is unambiguous; this
        helper surfaces all candidates so callers can enforce uniqueness.
        """

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

    def get(
        self,
        paths: str | Sequence[str],
        default: object | None = None,
    ) -> object | list[object]:
        """Retrieve values from the configuration or base store.

        This method is a convenience accessor that reads configuration fields
        (by attribute name) or store values (by key). Use it when you need a
        uniform accessor that works for both config fields and store entries.
        When ``paths`` is a sequence, a list of resolved values is returned.

        Parameters
        ----------
        paths : str | Sequence[str]
            A single config attribute name or store key, or a sequence of them.
            Store keys containing ``"."`` are treated as fully-qualified keys.
        default : Any, optional
            Default to return if the store key is missing. When ``None``, a
            missing key raises the underlying store error.

        Returns
        -------
        Any | list[Any]
            The resolved value(s) from config or store.

        Raises
        ------
        KeyError
            If a store key is missing and ``default`` is ``None``.
        AttributeError
            If a configuration attribute is missing.
        """

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
        """Return a namespace proxy for a store prefix.

        This is a typed convenience wrapper around store paths: it returns a
        ``StoreNamespace`` that exposes ``<prefix>.<key>`` entries as attributes.
        Use this to access grouped store values (e.g., ``binder.ns("system")``).

        Parameters
        ----------
        prefix : str
            Namespace prefix (must be a valid Python identifier and not a
            reserved binder name).

        Returns
        -------
        StoreNamespace
            Proxy object for accessing store values under the given prefix.

        Raises
        ------
        ValueError
            If the prefix is invalid, reserved, or no keys exist under it.
        """

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

    def _structural_keys_by_component(self) -> dict[str, set[str]]:
        """Return structural keys grouped by component."""

        return {
            "optics": self._optics_structural_keys(),
            "source": self._source_structural_keys(),
            "detector": self._detector_structural_keys(),
        }

    def _structural_keys(self) -> set[str]:
        """Return the union of structural keys across all components."""

        structural_keys: set[str] = set()
        for keys in self._structural_keys_by_component().values():
            structural_keys |= set(keys)
        return structural_keys

    def _structural_keys_in_store(self, store: ParameterStore) -> list[str]:
        """Return the structural keys present in ``store``."""

        structural_keys = self._structural_keys()
        return sorted(key for key in store.keys() if key in structural_keys)

    @staticmethod
    def _values_equal(current_value: object, incoming_value: object) -> bool:
        """Return whether two values are equivalent for structural checks."""

        try:
            return bool(
                jnp.array_equal(jnp.asarray(current_value), jnp.asarray(incoming_value))
            )
        except Exception:
            return current_value == incoming_value

    def _detect_structural_changes(self, store: ParameterStore) -> dict[str, set[str]]:
        """Return structural keys that changed, grouped by component."""

        changes: dict[str, set[str]] = {}
        for component, keys in self._structural_keys_by_component().items():
            changed: set[str] = set()
            for key in keys:
                current_value = self.base_forward_store.get(key)
                incoming_value = store.get(key)
                if not self._values_equal(current_value, incoming_value):
                    changed.add(key)
            if changed:
                changes[component] = changed
        return changes

    def structural_store_keys(self) -> set[str]:
        """Return the structural store keys for this binder.

        Public wrapper around the component-aware structural key helper. Use
        this to inspect which keys are treated as structural (rebuild-required)
        versus non-structural (runtime-bound) for the current binder.

        Returns
        -------
        set[str]
            Structural store keys after accounting for runtime bindings.
        """

        return self._structural_keys()

    def model(
        self,
        store_delta: Optional[ParameterStore] = None,
        *,
        allow_rebuild: bool = False,
    ) -> jnp.ndarray:
        """Evaluate the Shera PSF for an optional store overlay.

        Use this as the primary evaluation API. With ``store_delta=None``, the
        cached telescope is reused for a fast-path model evaluation. When a
        ``store_delta`` is provided, runtime updates are applied per component
        to the cached telescope without rebuilding. Structural keys are not
        accepted by default; pass ``allow_rebuild=True`` to rebuild the binder
        state via :meth:`update_store` when structural changes are required.

        Parameters
        ----------
        store_delta : ParameterStore, optional
            Overlay of parameter values to merge onto the base store. May
            contain only non-structural keys unless ``allow_rebuild=True``.
        allow_rebuild : bool, optional
            When ``True``, structural keys are accepted and the binder is
            rebuilt via :meth:`update_store` before evaluation.

        Returns
        -------
        jax.numpy.ndarray
            PSF model evaluated with the effective store.

        Raises
        ------
        ValueError
            If structural keys are present in ``store_delta`` and
            ``allow_rebuild`` is ``False``.
        """

        if store_delta is None:
            return self.telescope.model()

        store_delta = store_delta.validate_against(
            self.forward_spec,
            allow_missing=True,
            allow_extra=False,
            allow_derived=True,
        )

        structural_keys = self._structural_keys_in_store(store_delta)
        if structural_keys:
            if allow_rebuild:
                eff_store = self._merge_store(store_delta)
                rebuilt = self.update_store(eff_store, allow_rebuild=True)
                return rebuilt.telescope.model()

            joined = ", ".join(structural_keys)
            raise ValueError(
                "model() only accepts non-structural store keys; "
                f"found structural keys: {joined}. "
                "Use allow_rebuild=True with a full store to rebuild."
            )

        eff_store = self._merge_store(store_delta)
        return self._apply_runtime_updates(eff_store).model()

    @property
    def optics(self) -> dl.OpticalLayer | dl.Optics:
        return self.telescope.optics

    @property
    def source(self) -> dl.Source:
        return self.telescope.source

    @property
    def detector(self) -> dl.LayeredDetector:
        return self.telescope.detector

    # ------------------------------------------------------------------
    # Mostly immutable helpers
    # ------------------------------------------------------------------

    def with_store(self, new_base_store: ParameterStore) -> "BaseSheraBinder":
        """Return a new binder that uses a different base store.

        This is an immutable-style helper: it constructs a new binder instance
        with the same configuration and parameter specification but a new base
        store. Use this when you want to swap the baseline store while keeping
        the current config/spec.

        Parameters
        ----------
        new_base_store : ParameterStore
            Fully populated base store to use for the new binder.

        Returns
        -------
        BaseSheraBinder
            New binder instance with the updated base store.

        Raises
        ------
        ValueError
            If ``new_base_store`` fails validation against ``forward_spec``.
        """

        return self.__class__(
            cfg=self.cfg,
            forward_spec=self.forward_spec,
            base_forward_store=new_base_store,
        )

    def update_store(
        self,
        store: ParameterStore,
        *,
        allow_rebuild: bool = False,
    ) -> "BaseSheraBinder":
        """Return a new binder with an updated base store.

        This immutable-style helper validates the incoming store and applies
        runtime bindings when only non-structural keys change. Structural
        changes are allowed only when ``allow_rebuild=True``; otherwise a
        clear error is raised.

        Parameters
        ----------
        store : ParameterStore
            Full base store with derived values populated.
        allow_rebuild : bool, optional
            When ``True``, structural changes trigger a rebuild of the
            affected components. When ``False``, structural changes raise.

        Raises
        ------
        ValueError
            If structural keys change while ``allow_rebuild`` is ``False``.
        """

        validated_store = store.validate_against(
            self.forward_spec,
            allow_derived=True,
        )

        new_structural_hash = self._compute_structural_hash()
        structural_hash_changed = new_structural_hash != self.structural_hash
        structural_changes = self._detect_structural_changes(validated_store)
        structural_changed = structural_hash_changed or bool(structural_changes)

        if structural_changed and not allow_rebuild:
            structural_keys = sorted(
                {key for keys in structural_changes.values() for key in keys}
            )
            if structural_hash_changed:
                structural_keys.append("structural config hash")
            joined = ", ".join(structural_keys) if structural_keys else "structural config hash"
            raise ValueError(
                "update_store() requires allow_rebuild=True when structural "
                f"keys change. Detected structural keys: {joined}."
            )

        if structural_changed:
            structural_components = set(structural_changes)
            if structural_hash_changed:
                structural_components.add("optics")
            telescope = self._rebuild_telescope(
                validated_store,
                structural_components=structural_components,
            )
        else:
            telescope = self._apply_runtime_updates(validated_store)

        updated = self.__class__.__new__(self.__class__)
        updated.cfg = self.cfg
        updated.forward_spec = self.forward_spec
        updated.base_forward_store = validated_store
        updated.structural_hash = new_structural_hash
        updated.telescope = telescope
        return updated


__all__ = [
    "BaseConfig",
    "BaseSheraBinder",
    "BINDER_RESERVED_NAMES",
]
