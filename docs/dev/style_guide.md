# Style Guide (dLux-inspired)

This document defines formatting and API conventions used across **dLuxShera**.
The goal is consistency, readability, and JAX-friendly design (immutability,
functional updates, predictable IO).

## 1) File and module structure

- Prefer a consistent top-of-file layout:
  - `from __future__ import annotations` (when helpful for typing)
  - Imports grouped as: **stdlib → third-party → local**
  - `__all__` at top to define the public surface area of the module
- Use clear file sectioning for large modules (optional but encouraged):
  - “Private helpers” vs “Public API”
  - Visually distinct comment banners are acceptable when they improve
    scanability

## 2) Naming conventions

- Classes: `CamelCase`
- Methods and functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Booleans: prefer verb-ish or predicate names:
  - `return_wf`, `return_psf`, `use_eigen`, `whiten_basis`
- Domain parameters should be explicit and unit-aware:
  - e.g., `pupil_npix`, `wavelength_m`, `binary.x_position_as`,
    `system.plate_scale_as_per_pix`, etc.

## 3) Type hints and explicit attributes

- Public classes should declare key instance fields as **class attributes with
  type annotations**.
  - This advertises the “shape” of the object before reading `__init__`.
  - Example:

```python
psf_npixels: int
oversample: int
psf_pixel_scale: float
```

- Method signatures should include argument and return type hints where
  practical.
  - Prefer concrete JAX types (`jax.Array`) for arrays and `float/int/bool` for
    scalars.
- Be consistent about typing style:
  - If annotating `self`, do so consistently within a module or subsystem.
  - Use `list[...]`, `dict[...]` generics (Python 3.9+) rather than
    `typing.List`.

## 4) Docstrings (detailed NumPy style, binder-level verbosity)

- Use **NumPy-style docstrings** for public classes and public methods:
  - One-line *action-oriented* summary (imperative mood is fine: “Return…”,
    “Build…”, “Evaluate…”).
  - Follow immediately with **1–3 short paragraphs** that explain:
    - what the function/class *does* in the system,
    - when to use it (and when not to),
    - key semantics (immutability, caching, fast-path vs rebuild, etc.),
    - any important constraints or resolution order.

- Prefer **binder-style “narrative docstrings”** over minimal one-liners:
  - Include usage cues like “Use this for …” / “Called by …” /
    “This path is performance-oriented …”.
  - When behavior depends on options or internal routing, document it
    explicitly (e.g., “Resolution order:” lists).
  - Use `Notes` to capture invariants and design intent (immutability, runtime
    bindings, structural rebuild triggers).

- Include the standard sections when applicable:
  - `Parameters` (include **units** where relevant: metres, radians, arcseconds,
    microns, etc.)
  - `Returns`
  - `Raises` (user-facing; list common error cases)
  - `Notes` (design intent, performance/caching implications, immutability
    expectations)

- Flag-dependent return types:
  - If return type varies by flags (e.g., `return_wf` / `return_psf`), keep type
    hints simple in the signature if needed, but document all return variants
    clearly in `Returns`.

- Private helpers:
  - Private methods may use shorter docstrings, but should still be explicit
    when they encode important invariants (e.g., “This must not mutate the
    binder”, “store must be validated”, “structural keys excluded”).

- Formatting details:
  - Wrap lines at the repo’s typical docstring line length (generally ~88
    characters).
  - Use double-backticks for inline code and refer to related methods with
    `:meth:` and classes with `:class:` where helpful.

### Recommended templates

- Public method template (binder-style):

```python
"""<One-line summary>.

<1–3 paragraphs describing purpose, when to use, and key semantics.>
<Optional: Resolution order / behavior notes.>

Parameters
----------
...

Returns
-------
...

Raises
------
...

Notes
-----
<Design intent / immutability / performance implications.>
"""
```

- Public class template (binder-style overview + properties):

```python
"""<One-line summary>.

<Overview paragraph(s) describing the object’s role in the system.>

Key properties
--------------
- <bullet list of key semantics and intended usage>
- ...

Notes
-----
<Important invariants: immutability, caching, structural rebuild rules, etc.>
"""
```

## 5) Abstract base classes: `@abstractmethod` vs `NotImplementedError`

Use `@abstractmethod` for **required** subclass hooks (true interface methods).
Use `NotImplementedError` for **optional / conditionally-used** hooks where
instantiating the base class is still valid.

### Prefer `@abstractmethod` when:
- The base class is not meaningful to instantiate on its own.
- The method must be implemented for correctness (it is part of the contract).
- You want failures to happen at **construction time** (fast feedback) instead
  of at runtime.
- You want clearer intent for reviewers and better support from type checkers
  and IDEs.

Behavior: classes with unimplemented abstract methods cannot be instantiated;
Python raises a `TypeError` at instantiation time.

### Prefer `NotImplementedError` when:
- The base class *can* be instantiated, but a method is only valid in some
  configurations or subclasses.
- The method is a “soft hook” where a default implementation is intentionally
  absent.
- You want the error to happen only if/when the method is actually called.

Behavior: instantiation succeeds, but calling the method triggers a runtime
exception.

### Repository convention
- If a method is a documented “hook for subclasses” and must exist (e.g., optics
  or source builders), express it as an abstract method on an `ABC` base class.
- If a hook is truly optional, either provide a real default implementation or
  raise `NotImplementedError` with a clear message explaining when/why the hook
  is unavailable.

## 6) Input normalization and validation

- Normalize inputs early, then validate shapes/types before compute.
  - Examples of normalization patterns:
    - `np.atleast_1d(...)` for vector inputs
    - Coerce numeric types in constructors (`int(...)`, `float(...)`)
- Validate mutually exclusive flags with explicit guards:
  - If flags are mutually exclusive and are both enabled, raise a `ValueError`
    immediately.
- Error messages should be user-facing:
  - Include received shapes/values when relevant
  - Provide a clear resolution (“expected X, got Y”)

## 7) Constructors: cast at the boundary

- In `__init__`, cast and store parameters immediately:
  - `self.psf_npixels = int(psf_npixels)`
  - `self.psf_pixel_scale = float(psf_pixel_scale)`
- Prefer establishing invariants at construction time rather than deep in model
  execution.

## 8) Immutability and functional updates (JAX-first)

- Objects intended to be used as JAX pytrees should be treated as immutable.
- Prefer functional update patterns (`.set(...)`, `.replace(...)`, etc.) over
  in-place mutation.
- Methods that “modify” an object should generally return a new updated object.

## 9) Keep shared logic centralized; override only the delta

- Put common control flow in a base class or shared helper.
- Subclasses should override the smallest possible portion (e.g., the
  “propagate to image plane” step).
- Avoid copy/paste across classes when differences are parameterization-only.

## 10) Small local helpers for vectorization / transforms

- When mapping across wavelength or batch dimensions, keep per-item logic in a
  small local function, then apply transforms (`vmap`, `filter_vmap`, `scan`,
  etc.).
- This improves readability and makes tracing behavior clearer.

## 11) Plotting utilities IO policy (required)

Plotting helpers must:
- Accept optional Matplotlib `fig` / `ax` inputs; create new ones only when
  omitted
- Return `(fig, ax)` or `(fig, axes)` for downstream customization
- Never call `plt.show()` implicitly; caller controls display
- Save figures only when explicitly requested (e.g., `save_path`) to keep tests
  headless

## 12) Testing-related expectations (docs-level)

- Tests should not rely on interactive GUI backends.
- Keep deterministic behavior where feasible (seed RNGs where appropriate).
- Prefer small, fast smoke tests for end-to-end demo flows, with heavier
  validation tests isolated.

## 13) When it’s okay to deviate

- Performance-critical sections may trade some readability for efficiency, but
  should include comments describing assumptions and constraints.
- If a deviation is repeated, update this guide so the exception becomes
  documented convention.
