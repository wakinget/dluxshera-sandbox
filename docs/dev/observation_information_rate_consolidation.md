# Observation Information-Rate Consolidation

Status: developer consolidation note, 2026-08-10. This records the current
scientific and algorithmic understanding from the reviewed observation-level
information-rate work on `observation-level-update`. It is not a controller
design freeze and does not describe a newly implemented acquisition controller.

## Provenance

Authoritative reviewed result family:

- `analysis_reviews/m2_center_information_rate_family_20260805`
- per-root reviewed products: `per_root_psd_421fd09`
- family aggregation: `family_aggregate_psd_421fd09_agg_12cf205`

Relevant analysis commits:

- `421fd09550b9083cbe071051d0b574620e2a31aa`
- `12cf2051a8836345b7823d5170bc75e872d7e151`

The implementation paths to inspect first are:

- `examples/scripts/analyze_full_fidelity_binary_iterative_campaign.py`
- `examples/scripts/aggregate_full_fidelity_information_rate_family.py`
- `src/dluxshera/inference/observation_information_rate.py`
- `src/dluxshera/inference/observation_belief.py`
- `src/dluxshera/inference/observation_forecast.py`
- `tests/inference/test_observation_information_rate.py`
- `tests/utils/test_analyze_full_fidelity_binary_iterative_campaign_script.py`
- `tests/utils/test_aggregate_full_fidelity_information_rate_family_script.py`

`analysis_reviews/` is not present in this checkout, so the numerical family
summaries below are recorded as reviewed-family results by provenance, while
the mathematical and product semantics were verified against the code above.
Do not replace these relative provenance names with cluster-local absolute
paths in long-lived docs.

## Current Iterative Workflow

The established full-fidelity iterative campaign workflow is:

```text
short-exposure frames
  -> 1-s subblocks
  -> subblock inference under a shared slow state
  -> Schur-reduced slow-state information
  -> information-weighted block/window update
  -> new reference state
  -> next window
```

The historical iterative result is a local/window update workflow. Its reported
posterior sigma is the local window posterior associated with the update that
set the next reference. It should not be described as a fully accumulated
observation-level science posterior unless the execution path has explicitly
combined all observation likelihood factors with the initial observation prior.

The retrospective cumulative and information-rate analyzers answer a different
question: how much reduced information is present across the saved summaries
when those summaries are treated as additive likelihood factors. That formal
information can be combined into an observation-level covariance diagnostic,
and the `ObservationLikelihoodState` machinery can rebase local quadratics
through their `theta_ref` and `reduced_score`, but the reviewed cadence replay
itself is covariance/information only.

Keep these concepts separate:

- local/window iterative posterior: the posterior used to update a moving
  reference inside the current campaign workflow;
- full-observation information: the additive Schur information present in all
  accepted one-second summaries;
- future observation-level science posterior: a proposed final product that
  must consistently accumulate information and score terms under a controlled
  linearization/rebasing policy.

## Validated Information Products

The retrospective analysis showed that the saved one-second Schur-reduced
information products are coherent and additive under the analyzer's acceptance
rules. In the reviewed 10-window / 300-s campaigns:

| Quantity | Representative result |
| --- | --- |
| Final iterative/window-local separation sigma | about 17.4--17.6 uas |
| Formal accumulated 300-s information sigma | about 5.6 uas |
| Contraction from 30 s to 300 s | close to the expected `sqrt(10)` |
| Formal stationary 1800-s projection | about 2.3 uas |

The 1800-s value is a formal stationary information projection from late-tail
rates. It is not a demonstrated 30-minute astrometric accuracy.

## Prior Assumptions

The production family used a diagonal physical prior covariance. No physical
cross-covariances were assumed in that prior. The reviewed baseline priors were:

| Parameter | Prior sigma |
| --- | --- |
| `source.separation_as` | 100 uas |
| `source.log_flux_total` | `1e-4` |
| `source.contrast` | about `3.366e-4` for this target |
| `optics.plate_scale_as_per_pix` | about 12.321 uas/pixel |
| each M1 Z4--Z11 coefficient | 1 nm |
| each M2 Z4--Z11 coefficient | 1 nm |

These values follow the production prior-draw configuration: absolute
`1e-4 arcsec` separation, absolute `1e-4` log flux, fractional `1e-4` contrast
and plate-scale priors, and absolute `1 nm` Zernike priors for the enabled
low-order M1/M2 coefficients.

## Canonical Information Modes

For physical-basis information matrix `S` and diagonal prior covariance `C0`,
define the prior-whitened gain matrix:

```text
G = C0^(1/2) S C0^(1/2)
```

Equivalently, with prior sigmas `sigma_i`,
`G_ij = sigma_i S_ij sigma_j`. The canonical information modes are the
eigenvectors of a prior-whitened information-rate matrix, normally pooled over
the configured late-tail windows. Eigenvalues are prior-relative information
rates in `1/s`.

For a whitened eigenvector `v_k`, the corresponding one-prior-sigma physical
direction is:

```text
d_k = C0^(1/2) v_k
d_ik = sigma_i * v_ik
```

Squared whitened coefficients are the dimensionless physical composition
fractions used for interpretation. This avoids comparing raw arcsec, nm, flux,
and contrast coefficients directly.

Canonical mode IDs are root-local. Treat integer IDs as local labels attached
to one canonical spectrum, and associate modes with physical interpretations
through their loading tables and assignment diagnostics.

## Information Spectrum

The baseline `0.01 nm` M2 high-order WFE knowledge-error family showed a strong
hierarchy:

| Mode region | Dominant interpretation |
| --- | --- |
| 0--7 | very high-information mixed WFE directions |
| 8 | plate scale |
| 9 | total/log flux |
| 10 | source separation |
| 11--17 | weaker mixed WFE directions |
| 18 | contrast |
| 19 | weakest mixed M2 WFE direction |

Representative prior-relative rates from the reviewed baseline family:

| Mode | Interpretation | Rate |
| --- | --- | --- |
| 8 | plate scale | about 497.7 1/s |
| 9 | total/log flux | about 9.56 1/s |
| 10 | separation | about 1.126 1/s |
| 18 | contrast | about 0.245 1/s |
| 19 | weak M2 WFE | about 0.071 1/s |

Modes 0--7 have much larger WFE-dominated rates, roughly `1e4--4e4 1/s`
relative to the adopted 1-nm WFE priors.

Do not interpret sub-second or microsecond crossing-time extrapolations as
demonstrated operational sampling rates. The reviewed inputs are one-second
Schur summaries. Such values are normalized observability rankings, not proof
of usable subsecond control cadence.

## Fixed-Prior Gain

For a canonical direction with approximately stationary prior-relative rate
`r`, fixed-prior gain is:

```text
gamma(t) = r t
t(gamma) = gamma / r
sigma(t) / sigma_prior = 1 / sqrt(1 + gamma)
```

Useful anchors:

| Gain | Sigma ratio |
| --- | --- |
| `gamma = 1` | `1 / sqrt(2)` |
| `gamma = 2` | `1 / sqrt(3)` |
| `gamma = 3` | `1 / 2` |

This normalization is independent of any operational minimum or maximum block
duration.

Because canonical eigenvalues are ordered from highest to lowest information
rate, a selected prefix `modes 0:k` is normally limited by mode `k` under the
stationary fixed-basis approximation. The reviewed baseline analysis found:

| Selected prefix | Limiter | Ideal `gamma=3` crossing |
| --- | --- | --- |
| modes 0:3 | WFE mode 3 | about `1e-4 s` |
| modes 0:8 | plate scale | about `0.006 s` |
| modes 0:9 | log flux | about `0.314 s` |
| modes 0:10 | separation | about `2.67 s` |
| modes 0:18 | contrast | about `12.24 s` |
| modes 0:19 | weakest M2 WFE | about `42.2 s` |

This supports the interpretation that including all high-information modes
through mode 10 remains separation-limited in the baseline fixed-prior view.

## Named Mode Sets

The cadence experiments used named analysis constructs. They should not be
treated as immutable public API concepts unless later code formalizes them.

`astrometric_core`:

- source separation;
- plate scale.

These mapped approximately to modes 10 and 8 in representative roots.

`source_core`:

- separation;
- total/log flux;
- contrast;
- plate scale.

Contrast becomes the cadence bottleneck for this set.

`high_information_calibration`:

- `astrometric_core`;
- plus the four highest-information WFE-dominated canonical modes.

In the analyzed family, adding these WFE modes did not lengthen the
information-only schedule relative to `astrometric_core`.

The analyzer also contains `all_trackable`, whose membership depends on the
requested gain threshold and maximum duration. Treat it as exploratory
trackability bookkeeping.

## Sequential Relative Gain

Sequential cadence replay differs from fixed-prior gain. For the first block,
gain is measured relative to the initial prior precision. After a block is
incorporated, later gain is measured relative to the stronger accumulated
precision:

```text
gamma_k = (d_k.T S_buffer d_k) / (d_k.T P_current d_k)
P_after = P_before + S_buffer
```

The replay in `simulate_sequential_information_gate` records no score, mean,
innovation, requested update, relinearization, or reference trajectory.

For stationary scalar information rate `r` and relative-gain threshold `g`, if
each block closes exactly at threshold:

```text
dt_1 = g / r
dt_2 = g (1 + g) / r = (1 + g) dt_1
dt_n = dt_1 (1 + g)^(n - 1)
```

For gain 3, the second exact-threshold block is four times longer than the
first. This explains why a relative-information cadence naturally updates
frequently during acquisition and increasingly slowly as precision accumulates.

## The Historical 5-s / 18-s / 23-s Result

The historical replay imposed:

- 5-s minimum block length;
- 30-s maximum block length;
- gain threshold 3.

For the separation-limited `astrometric_core`, the unconstrained first
`gamma=3` crossing is about 2.67 s. The imposed 5-s minimum forced the first
update to wait until 5 s, so the separation mode overshot the requested gain to
about 5.6. The stronger post-update precision then required about 17.7
additional seconds to reach another gain of 3, producing the observed pattern:

```text
first block    = 5 s
second block   = about 18 s
second closure = about 23 s cumulative
```

The 5-s value is therefore not a physically derived optimal cadence. It came
from the imposed minimum duration.

## Information Versus Accuracy

Across the M2 high-order WFE knowledge-error sweep, formal accumulated 300-s
separation sigma remained near 5.6 uas while actual separation bias grew
dramatically with M2 knowledge error. Representative mean signed final errors
from the reviewed family were:

| M2 knowledge error | Mean signed final separation error |
| --- | --- |
| 0.01 nm | about -9 uas |
| 0.05 nm | about -29 uas |
| 0.10 nm | about -70 uas |
| 0.50 nm | about -604 uas |
| 1.00 nm | about -1294 uas |

The conclusion is:

```text
statistical information != model fidelity != astrometric accuracy
```

More exposure can shrink formal uncertainty around a systematically biased
solution. Do not imply that the about-2.3-uas 30-minute formal projection
represents total expected astrometric error.

## PSD and Degeneracy Diagnostics

At larger M2 knowledge errors, more reduced information matrices required
clipping of tiny negative eigenvalues. The implementation treats tiny negative
eigenvalues below
`PSD_ATOL + PSD_RTOL * max(max(abs(raw_eigenvalues)), 1.0)` as numerical
roundoff, projects them to zero, and rejects materially indefinite matrices.
The reviewed family found relative corrections around machine precision and no
materially indefinite accepted matrices. Treat this as numerical
Schur/eigendecomposition roundoff, not physically negative information.

Some WFE directions become quasi-degenerate. In that case individual
eigenvector labels can rotate substantially even when the corresponding
subspace remains stable. Interpret the eigenspace through subspace singular
values, principal angles, and grouped physical composition instead of assigning
too much meaning to an individual eigenvector ID.

## Candidate Future Architecture

The current findings motivate a possible two-phase architecture. This is
proposed future work, not current campaign behavior:

```text
acquisition / reference-establishment phase
  -> accumulate short subblocks
  -> test information support
  -> test innovation/reference movement
  -> update/relinearize slow state
  -> repeat until stable

handoff

fixed-reference precision-accumulation phase
  -> accumulate observation-level information
  -> accumulate/rebase score information consistently
  -> avoid contaminating the final science posterior with incompatible early
     local linearizations
  -> form final observation-level posterior
```

The eventual controller probably needs at least two gates:

1. information support: is there enough likelihood curvature to justify an
   update?
2. innovation/reference stability: does the data actually request a meaningful
   state change?

The current information-rate replay addresses the first gate only.

Separation and plate scale are useful acquisition-gate directions because they
are directly relevant to astrometric reference establishment and, in the
reviewed family, are well supported without forcing the cadence to wait on
contrast or weak WFE directions. High-information WFE modes do not lengthen the
baseline `astrometric_core` schedule because their rates are much larger than
the separation-limited rate. Contrast and weak WFE modes should continue to be
estimated and monitored, but should not control the main astrometric cadence
without additional evidence from controlling-mode, innovation, and systematic
error diagnostics.

## Unresolved Work

Record these as future investigations rather than current implementation:

- cumulative-posterior start-window sensitivity;
- whether early acquisition likelihood factors should be excluded or rebased;
- non-oracle acquisition/handoff criteria;
- innovation, score, and reference-stability diagnostics;
- prospective acquire-then-accumulate validation;
- score rebasing and relinearization consistency;
- treatment of weak and quasi-degenerate modes;
- whether prefix `0:10` remains exactly schedule-equivalent under full
  historical replay;
- validation across additional targets, pointing conditions, and optical-error
  families;
- final choice of operational cadence rules;
- broader model-fidelity and systematic-error requirements.

## Reproduction Pointers

Per-root review products to inspect:

- `information_rate/information_rate_summary.json`
- `information_rate/information_rate_input_inventory.csv`
- `information_rate/information_rate_by_mode.csv`
- `information_rate/information_rate_by_window_mode.csv`
- `information_rate/information_mode_loadings.csv`
- `information_rate/information_by_physical_label.csv`
- `information_rate/adaptive_mode_set_resolution.csv`
- `information_rate/adaptive_cadence_sequential_updates.csv`
- `information_rate/adaptive_cadence_sequential_mode_gains.csv`
- `information_rate/adaptive_cadence_sequential_summary.csv`
- `information_rate/adaptive_cadence_candidates.csv`
- `information_rate/adaptive_cadence_prefix_diagnostics.csv`
- `information_rate/mode_overlap.csv`
- `information_rate/quasi_degenerate_subspace_summary.csv`

Family aggregation products to inspect:

- `family_information_rate_summary.json`
- `family_information_rate_summary.md`
- `family_input_inventory.csv`
- `family_physical_mode_assignments_by_root.csv`
- `family_physical_information_rates_by_root.csv`
- `family_fixed_prior_candidates.csv`
- `family_sequential_policy_by_root.csv`
- `family_gain3_acquisition_by_root.csv`
- `family_policy_schedule_equivalence.csv`
- `family_controlling_modes.csv`
- `family_quasi_degenerate_subspaces_by_root.csv`
- `family_formal_uncertainty_by_root.csv`
- `family_accuracy_and_information_by_root.csv`
- `family_psd_projection_by_root.csv`
