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

### 10. Time-Domain Simulation & Inference  
**Priority: Low → Medium (Long-Term)**

Enable the dLuxShera framework to **simulate and process time-series data**, spanning a wide range of temporal scales and physical assumptions.

This theme recognizes that many scientifically relevant effects are inherently time-dependent, and that future analyses may require reasoning over sequences of observations rather than single static images.

The emphasis is on *capability and structure*, not on committing to a specific temporal model.

#### Time-Series Simulation

Potential regimes of interest include (non-exhaustive):

- **Short-timescale frames**
  - Simulation of individual short exposures (e.g. ~50 ms frames)
  - Aggregation into longer effective observations
  - Inclusion of effects such as telescope jitter or pointing noise
  - Exploration of frame-to-frame correlations or independence assumptions

- **Observation-level sequences**
  - Simulation of full observation blocks (e.g. 30-minute integrations)
  - Repeated observations over mission-relevant timescales (e.g. years)
  - Use of simplified statistical models (e.g. Gaussian jitter approximations)
  - Investigation of how long-term trends impact astrometric recovery

The intent is to support both:
- high-fidelity simulations when needed, and
- lightweight approximations when only aggregate behavior matters.

#### Time-Series Processing & Inference

In addition to data generation, the framework should eventually support:

- Inference over time-indexed data
- Aggregation of likelihoods across frames or epochs
- Clear handling of shared vs time-varying parameters
- Distinction between nuisance temporal effects and science parameters
- Diagnostics for identifying time-correlated biases or drift

This may require:
- explicit representation of time or epoch indices
- structured parameter tying across observations
- new abstractions for batching or grouping data

#### Relationship to Other Themes

This theme naturally interacts with:
- **System Performance & Sensitivity Studies** (e.g. time-dependent error sources)
- **Inference & Optimization Robustness** (e.g. scaling inference to many frames)
- **Performance & Scaling** (e.g. managing large simulated datasets)
- **Extensibility & Advanced Optical Modeling** (e.g. time-varying optics)

The goal is not immediate implementation, but to ensure that architectural choices made today do not preclude time-domain modeling tomorrow.

---

### 11. External Model Comparison & Advanced Source Modeling  
**Priority: Low (Strategic / Exploratory)**

Evaluate and learn from other established physical optics and propagation codebases, and assess how their capabilities might inform or augment dLuxShera.

This theme is mostly exploratory. The goal is not wholesale adoption, but:
- identifying best-in-class ideas,
- validating modeling assumptions,
- and selectively incorporating capabilities where they align with dLuxShera’s goals (e.g. differentiability, composability, inference).

#### External Optical Propagation Libraries

Candidate libraries of interest include (non-exhaustive):

- **PROPER Optical Propagation Library**
  - Well-established physical optics modeling
  - Rich library of propagation primitives and examples
  - Widely used in mission studies and optical design

- **prysm**
  - Modern Python-based physical optics framework
  - Strong support for wavefront analysis and PSF modeling
  - Clear conceptual separation of optics and analysis

- **Lentil**
  - Lightweight, composable optics modeling
  - Emphasis on clarity and physical interpretability

Potential evaluation questions:
- What physical effects are modeled that dLuxShera currently omits?
- What abstractions are particularly clean or reusable?
- Which components might be adaptable to JAX / auto-diff?
- Where does dLuxShera’s architecture offer advantages or limitations by comparison?

Outcomes may include:
- direct use of external libraries for validation or benchmarking
- inspiration for new abstractions or APIs
- selective re-implementation of ideas in a JAX-compatible form

---

#### Advanced Source & Spectral Modeling

Current source modeling is intentionally minimal and specialized (e.g. AlphaCen models from dLuxToliman). Future science or proposal needs may require more sophisticated treatments.

Potential directions include:
- More realistic stellar spectra
- Broadband or multi-band source models
- Spectral mismatch effects between source, optics, and detector
- Parameterized or data-driven spectral representations

Key questions:
- When does simplified source modeling break down for astrometric inference?
- How sensitive are recovered parameters to spectral assumptions?
- What level of spectral realism is required for proposal-level studies vs detailed analysis?

The intent is to ensure that dLuxShera can:
- support increased realism when needed,
- without imposing unnecessary complexity on all use cases.

---

#### Relationship to Core Principles

Any integration or adaptation should respect core dLuxShera principles:
- compatibility with JAX and automatic differentiation
- clear separation between model structure and parameters
- composability with existing inference workflows

This theme exists to broaden perspective and reduce blind spots, not to dilute architectural focus.

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
