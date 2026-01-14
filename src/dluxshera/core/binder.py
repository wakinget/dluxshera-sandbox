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
        """Initialize the binder with a config, parameter spec, and base store.

        This sets up the shared baseline state used by all binder evaluations:
        the configuration, parameter spec, a validated base forward store, and
        cached telescope/detector instances for fast-path evaluations. Use this
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

        # Shared detector construction; subclasses can override if needed.
        self._detector = self._build_detector()
        self.telescope = self._build_telescope(self.base_forward_store)


    def __dir__(self):
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

    def __getattr__(self, name):
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
        return dl.LayeredDetector(layers=[("downsample", dl.Downsample(self.cfg.oversample))])

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
        surfaced via :meth:`_runtime_bindings`.
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

    def _build_telescope(self, store: ParameterStore) -> dl.Telescope:
        """Build a full telescope model from the given store.

        Called during initialization (to create the cached telescope) and by
        direct model evaluation paths. Subclasses may override this if they
        need custom telescope assembly logic.

        Parameters
        ----------
        store : ParameterStore
            Validated store containing all parameters required to build the
            source, optics, and detector.

        Returns
        -------
        dl.Telescope
            Fully constructed telescope.
        """
        return dl.Telescope(
            source=self._build_source(store),
            optics=self._build_optics(store),
            detector=self._detector,
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

    def _runtime_bindings(self) -> tuple[tuple[str, str], ...]:
        """Return runtime binding pairs for non-structural keys.

        Runtime bindings map store keys to optics attributes (or paths) that
        can be updated without rebuilding the full optics model. Subclasses
        should override this to list the non-structural keys eligible for
        fast-path updates.

        Returns
        -------
        tuple[tuple[str, str], ...]
            Sequence of ``(store_key, optics_path)`` bindings. Defaults to
            empty, meaning no runtime updates are supported.

        Notes
        -----
        Keys returned here are considered **non-structural** for the purposes
        of :meth:`_structural_store_keys`, and updates to them will use runtime
        bindings instead of triggering a rebuild.
        """
        return ()

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

    def _update_telescope_runtime(self, store: ParameterStore) -> dl.Telescope:
        """Apply runtime bindings to update the cached telescope.

        Called from :meth:`update_store` when structural changes are absent.
        Uses the runtime binding map to update optics parameters in-place (via
        ``apply_runtime_bindings``) and rebuilds the source as needed.

        Parameters
        ----------
        store : ParameterStore
            Validated store containing updated non-structural values.

        Returns
        -------
        dl.Telescope
            Telescope with updated optics and source, reusing the cached
            detector.

        Notes
        -----
        This path is performance-oriented: it avoids rebuilding the full
        optics when only non-structural parameters change. Structural keys
        must be excluded from ``store`` for this path to remain valid.
        """
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

    def get(self, paths, default=None):
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

    def _structural_store_keys(self) -> set[str]:
        """Return store keys treated as structural for this binder.

        Called by :meth:`model` and :meth:`update_store` to separate
        structural keys from non-structural runtime bindings. Structural keys
        are expected to require a full rebuild when they change.

        Returns
        -------
        set[str]
            Keys considered structural for this binder instance.

        Notes
        -----
        The base structural set is derived from the forward spec (keys under
        ``system.`` and ``band.``). Keys listed in :meth:`_runtime_bindings`
        are explicitly treated as **non-structural** and removed from the
        set so they can be updated via runtime bindings.
        """

        structural_keys = {
            key
            for key in self.forward_spec.keys()
            if key.startswith(("system.", "band."))
        }
        runtime_keys = {store_key for store_key, _ in self._runtime_bindings()}
        return structural_keys - runtime_keys

    def structural_store_keys(self) -> set[str]:
        """Return the structural store keys for this binder.

        Public wrapper around :meth:`_structural_store_keys`. Use this to
        inspect which keys are treated as structural (rebuild-required)
        versus non-structural (runtime-bound) for the current binder.

        Returns
        -------
        set[str]
            Structural store keys after accounting for runtime bindings.
        """

        return self._structural_store_keys()

    def model(
        self,
        store_delta: Optional[ParameterStore] = None,
        *,
        allow_rebuild: bool = False,
    ) -> jnp.ndarray:
        """Evaluate the Shera PSF for an optional store overlay.

        Use this as the primary evaluation API. With ``store_delta=None``, the
        cached telescope is reused for a fast-path model evaluation. When a
        ``store_delta`` is provided, the delta is merged with the base store
        and evaluated through the direct model path. Structural keys are not
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

    def update_store(self, store: ParameterStore):
        """Return a new binder with an updated base store.

        This is the immutable-style path for changing the baseline store. The
        incoming store is validated against ``forward_spec`` and then compared
        against the structural hash and structural store keys. If structural
        changes are detected, a new binder is created via ``with_store`` and a
        warning is emitted; otherwise runtime bindings are applied to refresh
        the cached telescope without a full rebuild. Use this to persist a new
        baseline store or when structural changes are intended.

        Parameters
        ----------
        store : ParameterStore
            Full base store with derived values populated.

        Returns
        -------
        BaseSheraBinder
            New binder instance with refreshed base store and telescope state.

        Raises
        ------
        ValueError
            If ``store`` fails validation against ``forward_spec``.

        Notes
        -----
        This method never mutates the existing binder instance; it always
        returns a new binder (either a full rebuild or a runtime-updated copy).
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

    The ``with_store`` attribute is an alias of
    :meth:`BaseSheraBinder.with_store`, provided for parity with legacy APIs.
    It preserves the binder's immutable-style semantics by always returning a
    fresh binder instance rather than mutating in-place.
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
        """Construct a binder for the three-plane Shera configuration.

        Parameters
        ----------
        cfg : SheraThreePlaneConfig
            Fully prepared Shera three-plane configuration. Any derived
            configuration values needed by the optics/source builders should
            already be present on this object.
        forward_spec : ParamSpec
            Parameter specification describing the full forward store,
            including structural keys and derived entries.
        base_forward_store : ParameterStore
            Forward-style base store with derived values populated. The store
            is validated against ``forward_spec`` and treated as immutable
            baseline state for subsequent evaluations.
        """
        super().__init__(
            cfg=cfg,
            forward_spec=forward_spec,
            base_forward_store=base_forward_store,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _direct_model(self, eff_store: ParameterStore) -> jnp.ndarray:
        """Evaluate the Shera three-plane model directly.

        Uses the three-plane optics and alpha Cen source builders to assemble
        a fresh telescope from ``eff_store`` and returns the PSF model output.
        This path is used when a non-structural store overlay is supplied.
        """
        return self._build_telescope(eff_store).model()

    def _build_optics(self, store: ParameterStore):
        """Build the Shera three-plane optics stack.

        Delegates to ``build_shera_threeplane_optics`` with the configured
        three-plane configuration, validated store, and forward specification.
        """
        return build_shera_threeplane_optics(
            self.cfg, store=store, spec=self.forward_spec
        )

    def _build_source(self, store: ParameterStore):
        """Build the Shera alpha Cen source for the three-plane system."""
        return build_alpha_cen_source(store, cfg=self.cfg)

    def _runtime_bindings(self) -> tuple[tuple[str, str], ...]:
        """Return the three-plane runtime bindings for non-structural keys."""
        return THREEPLANE_RUNTIME_BINDINGS

    def _compute_structural_hash(self) -> Optional[str]:
        """Return the structural hash derived from the three-plane config."""
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

    The ``with_store`` attribute is an alias of
    :meth:`BaseSheraBinder.with_store` to keep a stable public API. It preserves
    immutable-style semantics by always returning a new binder instance.
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
        """Construct a binder for the two-plane Shera configuration.

        Parameters
        ----------
        cfg : SheraTwoPlaneConfig
            Fully prepared Shera two-plane configuration. Derived config
            values expected by the optics/source builders should already be
            present.
        forward_spec : ParamSpec
            Parameter specification describing the forward store, including
            structural keys and derived entries.
        base_forward_store : ParameterStore
            Forward-style base store with derived values populated. The store
            is validated against ``forward_spec`` and used as the immutable
            baseline for evaluations.
        """
        super().__init__(
            cfg=cfg,
            forward_spec=forward_spec,
            base_forward_store=base_forward_store,
        )

    def _direct_model(self, eff_store: ParameterStore) -> jnp.ndarray:
        """Evaluate the Shera two-plane model directly.

        Builds a fresh telescope using the two-plane optics and alpha Cen
        source with ``eff_store`` and returns the modeled PSF output.
        """
        return self._build_telescope(eff_store).model()

    def _build_optics(self, store: ParameterStore):
        """Build the Shera two-plane optics stack."""
        return build_shera_twoplane_optics(self.cfg, store=store, spec=self.forward_spec)

    def _build_source(self, store: ParameterStore):
        """Build the Shera alpha Cen source for the two-plane system."""
        return build_alpha_cen_source(store, cfg=self.cfg)

    def _runtime_bindings(self) -> tuple[tuple[str, str], ...]:
        """Return the two-plane runtime bindings for non-structural keys."""
        return TWOPLANE_RUNTIME_BINDINGS

    def _compute_structural_hash(self) -> Optional[str]:
        """Return the structural hash derived from the two-plane config."""
        return structural_hash_for_twoplane(self.cfg)

    with_store = BaseSheraBinder.with_store
