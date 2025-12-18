# dLuxShera Roadmap

## Purpose

This document captures **longer-term goals, themes, and priorities** for the dLuxShera project.

Unlike the Working Plan, this roadmap is:
- intentionally **non-binding**
- not tied to fixed timelines or deadlines
- expected to evolve as the project matures

Themes may advance in parallel, stall, or resurface over time.  
Priorities reflect *current emphasis*, not commitments.

---

## How to Read This Document

- **Themes** describe broad areas of development or capability.
- **Priorities** are lightweight indicators of where focus is currently desired.
- Items listed under a theme are *illustrative*, not exhaustive task lists.
- Concrete implementation work should live in the **Working Plan**, not here.

Priority levels are intentionally coarse:
- **High** — active focus or near-term importance
- **Medium** — important but not currently dominant
- **Low** — exploratory, deferred, or opportunistic

---

## Theme Overview (Current Priorities)

| Theme | Priority |
|------|----------|
| Core Model & Architecture Maturity | High |
| Inference & Optimization Robustness | High |
| Canonical Demos & Reproducibility | High |
| System Performance & Sensitivity Studies | Medium |
| Performance, Scaling & Caching | Medium |
| Diagnostics, Introspection & Debuggability | Medium |
| Testbed & Hardware Integration | Medium |
| Extensibility & Advanced Optical Modeling | Low |
| Documentation & Knowledge Capture | Medium |

(These priorities are expected to change.)

---

## Themes

### 1. Core Model & Architecture Maturity  
**Priority: High**

Establish dLuxShera as a stable, principled modeling framework with a clear internal contract.

Focus areas:
- Binder / SystemGraph / ParamSpec architecture
- Clear separation of structural vs parametric concerns
- Canonical θ-space definitions and transformations
- Robust handling of derived parameters
- Reduction (and eventual retirement) of legacy paths

Success here underpins *everything else*.

---

### 2. Inference & Optimization Robustness  
**Priority: High**

Ensure inference workflows are:
- numerically stable
- interpretable
- comparable to legacy results

Focus areas:
- Consistent loss construction (Binder-based NLLs)
- Eigenmode inference workflows
- Fisher Information Matrix validation
- Agreement between eager vs JIT execution paths
- Clear diagnostics when inference fails or stalls

This theme bridges modeling and scientific credibility.

---

### 3. Canonical Demos & Reproducibility  
**Priority: High**

Provide well-defined, reproducible examples that demonstrate:
- how the system is intended to be used
- what “correct” behavior looks like

Focus areas:
- Canonical two-plane and three-plane demos
- Minimal examples suitable for testing and onboarding
- Stable outputs that can be regression-tested
- Clear separation between demo logic and infrastructure

These demos act as *living specifications* for the framework.

---

### 4. System Performance & Sensitivity Studies  
**Priority: Medium**

Use the dLuxShera model to **quantify sensitivity to non-ideal system effects**, with the explicit goal of informing system requirements (e.g. for SMEX proposal development).

This theme focuses on *model-driven requirement justification*, not implementation fidelity for its own sake.

Key target questions include (but are not limited to):

#### Pixel Position Errors
- Sensitivity of astrometric recovery to sub-pixel detector geometry errors
- Modeling effective pixel location perturbations (e.g. X/Y offsets at the ~0.1% pixel level)
- Use of synthetic data with intentionally distorted pixel grids
- Quantifying induced separation and position-angle biases

#### Pixel Response Non-Uniformity
- Impact of pixel-to-pixel response variations (e.g. 0.1–1%)
- Use of detector-layer response models (e.g. ApplyPixelResponse)
- Comparison between assumed and actual response models during inference
- Translation of response errors into astrometric error budgets

#### Broader Detector & Imaging Effects (Exploratory)
- Field distortion
- Higher-order detector effects
- Correlated pixel behavior or systematic patterns

The intent is not to exhaustively model all effects, but to:
- identify dominant sensitivities
- validate or challenge existing assumptions
- guide requirement-setting with quantitative evidence

---

### 5. Performance, Scaling & Caching  
**Priority: Medium**

Improve runtime performance without compromising correctness or clarity.

Focus areas:
- Structural hashing and optics caching
- JAX compilation boundaries and cost
- Avoiding accidental re-tracing
- Scaling to larger parameter sets or higher resolutions
- Identifying true performance bottlenecks vs perceived ones

This theme is opportunistic and often reactive.

---

### 6. Diagnostics, Introspection & Debuggability  
**Priority: Medium**

Make it easier to understand *what the system is doing* at runtime.

Focus areas:
- Introspection tools (tree views, parameter summaries)
- Clear printing and inspection of binders and stores
- Snapshotting and context capture
- Better error messages around transforms and validation
- Tools that support human reasoning, not just tests

This theme supports both development velocity and long-term maintainability.

---

### 7. Testbed & Hardware Integration  
**Priority: Medium**

Align the software framework with physical testbed usage.

Focus areas:
- Mapping between physical configurations and model configs
- Consistent parameter naming between hardware and simulation
- Support for calibration states and experiment scripts
- Validation of model assumptions against real data

Progress here may be bursty and experiment-driven.

---

### 8. Extensibility & Advanced Optical Modeling  
**Priority: Low**

Enable future modeling capabilities without prematurely committing to specific designs.

Potential directions include:

- Generalized multi-plane propagation beyond the current fixed layouts
- Support for additional free-space propagation stages
- Modeling of downstream optics (e.g. filters placed after the secondary)
- Non-standard entrance pupils, including diffractive masks upstream of the primary
- Accommodation of alternative point designs from external partners

This theme exists to ensure architectural choices made today do not block tomorrow’s science cases.

---

### 9. Documentation & Knowledge Capture  
**Priority: Medium**

Ensure that decisions, rationale, and usage patterns are preserved.

Focus areas:
- Architecture documentation
- Migration and refactor history
- Clear distinction between legacy and refactor-era APIs
- Capturing “why we did this” alongside “what we did”

This theme reduces future cognitive load.

---

## Relationship to the Working Plan

- The **Roadmap** answers: *Where is this project headed?*
- The **Working Plan** answers: *What are we doing right now?*

Items may move:
- from Roadmap → Working Plan as they become concrete
- from Working Plan → Roadmap when deferred or re-scoped

Neither document is expected to be perfectly up to date at all times.

---

## Notes

- This document is expected to change.
- Reordering priorities is encouraged.
- Removing themes is acceptable.
- Adding speculative ideas is allowed.

The roadmap exists to support clarity, not constrain progress.
