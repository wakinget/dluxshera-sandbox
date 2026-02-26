"""Base system interfaces and shared helpers for Shera system binders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace as dataclass_replace
from typing import Callable, Optional, Sequence, Self

import jax.numpy as jnp
import dLux as dl

from ..params.spec import ParamSpec
from ..params.store import ParameterStore, StoreNamespace


def compose_forward_spec(system_cfg, detector_contract: ParamSpec | None = None) -> ParamSpec:
    """Compose a forward spec from source, optics, and detector contracts.

    The composition order is deterministic:
      1) source contract
      2) optics contract (dispatched by ``system.optics.kind``)
      3) detector contract

    Key collisions across contracts raise ``ValueError`` with clear component
    names to avoid silently shadowing fields.
    """

    from ..components.optics import build_threeplane_optics_contract, build_twoplane_optics_contract
    from ..components.sources import build_alpha_cen_contract

    source_contract = build_alpha_cen_contract(system_cfg)

    optics_kind = _detect_optics_kind_from_cfg(system_cfg)
    optics_contract_builders: dict[str, Callable[..., ParamSpec]] = {
        "two_plane": build_twoplane_optics_contract,
        "three_plane": build_threeplane_optics_contract,
    }
    try:
        optics_contract = optics_contract_builders[optics_kind](system_cfg)
    except KeyError as exc:
        supported = ", ".join(sorted(optics_contract_builders))
        raise ValueError(
            f"Unknown optics kind {optics_kind!r} when composing forward spec. "
            f"Supported optics kinds: {supported}."
        ) from exc

    detector_contract = detector_contract or ParamSpec()

    contracts = (
        ("source", source_contract),
        ("optics", optics_contract),
        ("detector", detector_contract),
    )

    merged_fields = []
    seen_keys: dict[str, str] = {}
    for contract_name, contract in contracts:
        for field in contract.values():
            owner = seen_keys.get(field.key)
            if owner is not None:
                raise ValueError(
                    "Forward spec contract key collision on "
                    f"{field.key!r}: present in both {owner!r} and {contract_name!r}."
                )
            seen_keys[field.key] = contract_name
            merged_fields.append(field)

    system_id = (
        "shera_threeplane"
        if optics_kind == "three_plane"
        else "shera_twoplane"
    )
    return ParamSpec(merged_fields, system_id=system_id)


def _cfg_get(root, path: str, default=None):
    """Read a dotted path from mapping- or attribute-based configs."""

    cur = root
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, Mapping):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def _detect_optics_kind_from_cfg(cfg) -> str:
    """Return optics kind from config with compatibility fallbacks."""

    kind = _cfg_get(cfg, "system.optics.kind", default=None)
    if kind is not None:
        return str(kind)

    cfg_name = type(cfg).__name__.lower()
    if "threeplane" in cfg_name:
        return "three_plane"
    if "twoplane" in cfg_name:
        return "two_plane"

    kind = _cfg_get(cfg, "optics_kind", default=None)
    if kind is not None:
        return str(kind)

    raise ValueError(
        "Unable to resolve optics kind from config. Expected "
        "`system.optics.kind` (e.g. 'two_plane' or 'three_plane')."
    )


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


class SheraBinder:
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

    cfg: object
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

    def _cfg_get(self, path: str, default=None):
        """Read a dotted path from mapping- or attribute-based configs."""
        return _cfg_get(self.cfg, path, default=default)

    def _detect_optics_kind(self) -> str:
        """Return the configured optics kind with backward-compatible fallback."""
        return _detect_optics_kind_from_cfg(self.cfg)

    def _detect_source_kind(self) -> str:
        """Return the configured source kind with backward-compatible fallback."""

        kind = self._cfg_get("system.source.kind", default=None)
        if kind is not None:
            return str(kind)

        kind = self._cfg_get("source_kind", default=None)
        if kind is not None:
            return str(kind)

        return "binary"

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

        detector, _detector_contract = build_detector(self.cfg)
        return detector

    def _build_optics(self, store: ParameterStore):
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
        from ..builders.optics import build_shera_threeplane_optics, build_shera_twoplane_optics

        optics_builders: dict[str, Callable[..., object]] = {
            "two_plane": build_shera_twoplane_optics,
            "three_plane": build_shera_threeplane_optics,
        }

        optics_kind = self._detect_optics_kind()
        try:
            builder = optics_builders[optics_kind]
        except KeyError as exc:
            supported = ", ".join(sorted(optics_builders))
            raise ValueError(
                f"Unknown optics kind {optics_kind!r}. Supported optics kinds: {supported}."
            ) from exc

        return builder(self.cfg, store=store, spec=self.forward_spec)

    def _build_source(self, store: ParameterStore):
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
        from ..builders.source import build_alpha_cen_source

        source_builders: dict[str, Callable[..., object]] = {
            "binary": build_alpha_cen_source,
            "alpha_cen": build_alpha_cen_source,
        }

        source_kind = self._detect_source_kind()
        try:
            builder = source_builders[source_kind]
        except KeyError as exc:
            supported = ", ".join(sorted(source_builders))
            raise ValueError(
                f"Unknown source kind {source_kind!r}. Supported source kinds: {supported}."
            ) from exc

        return builder(store, cfg=self.cfg)

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

    def _group_names_for_component(self, component: str) -> tuple[str, ...]:
        """Return ParamField groups associated with a binder component."""

        group_aliases = {
            "optics": ("optics", "system", "band", "primary", "secondary"),
            "source": ("source",),
            "detector": ("detector", "imaging"),
        }
        try:
            return group_aliases[component]
        except KeyError as exc:
            raise ValueError(f"Unknown binder component: {component!r}") from exc

    def _runtime_bindings_for_group(self, group: str) -> tuple[tuple[str, str], ...]:
        """Return runtime bindings declared by ParamField metadata for a component."""

        groups = set(self._group_names_for_component(group))
        return tuple(
            (field.key, field.binding)
            for field in self.forward_spec.values()
            if field.group in groups and field.binding is not None
        )

    def _optics_runtime_bindings(self) -> tuple[tuple[str, str], ...]:
        """Return runtime binding pairs for non-structural optics keys."""

        return self._runtime_bindings_for_group("optics")

    def _source_runtime_bindings(self) -> tuple[tuple[str, str], ...]:
        """Return runtime binding pairs for non-structural source keys."""
        return self._runtime_bindings_for_group("source")

    def _detector_runtime_bindings(self) -> tuple[tuple[str, str], ...]:
        """Return runtime binding pairs for non-structural detector keys."""
        return self._runtime_bindings_for_group("detector")

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
        from ..builders.optics import structural_hash_for_twoplane, structural_hash_from_config

        optics_kind = self._detect_optics_kind()
        hash_fns: dict[str, Callable[..., str]] = {
            "three_plane": structural_hash_from_config,
            "two_plane": structural_hash_for_twoplane,
        }
        try:
            struct_hash = hash_fns[optics_kind](self.cfg)
        except KeyError as exc:
            supported = ", ".join(sorted(hash_fns))
            raise ValueError(
                f"Unknown optics kind {optics_kind!r}. Supported optics kinds: {supported}."
            ) from exc

        return f"optics_kind={optics_kind}:{struct_hash}"

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
        """Return structural keys grouped by binder component."""

        structural_by_group = self.forward_spec.structural_keys_by_group()
        return {
            component: {
                key
                for group in self._group_names_for_component(component)
                for key in structural_by_group.get(group, set())
            }
            for component in ("optics", "source", "detector")
        }

    def _structural_keys(self) -> set[str]:
        """Return the union of structural keys across all components."""

        return self.forward_spec.structural_keys()

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

    def strip_structural(self, store: ParameterStore) -> ParameterStore:
        """
        Return a new store with structural keys removed according to this binder's
        forward spec (contract-driven).

        Parameters
        ----------
        store:
            Store to remove structural keys from.

        Returns
        -------
        ParameterStore
            Store with any keys in `self.structural_store_keys()` removed.
        """

        structural = self.structural_store_keys()
        filtered = {key: value for key, value in store.items() if key not in structural}
        return ParameterStore.from_dict(filtered)

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

    def with_store(self, new_base_store: ParameterStore) -> "SheraBinder":
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
        SheraBinder
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
    ) -> "SheraBinder":
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
    "SheraBinder",
    "BINDER_RESERVED_NAMES",
]
