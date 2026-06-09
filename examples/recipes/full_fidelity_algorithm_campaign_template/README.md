# Full-Fidelity Algorithm Campaign Template

This directory contains an architecture/config skeleton for a future
full-fidelity SHERA end-to-end algorithm demonstration campaign. It is not yet a
fully executable campaign prescription.

The skeleton follows the repository's experiment-centered prescription pattern:
the root YAML payload contains only `experiment`, and all campaign blocks live
inside that mapping. A future runner should consume the full `experiment`
mapping, matching existing loaders that use `payload["experiment"]` or
`payload.get("experiment", payload)`.

The goal is to make high-fidelity realism terms first-class from the beginning:
spectral mismatch, high-order WFE maps, detector calibration maps, trajectory
knowledge error, noise, and future smear all have explicit schema blocks even
when implementation is deferred.

Simpler runs should be represented as knockdowns or zero-amplitude settings of
this same schema. They should not use a separate minimal campaign architecture.

The active inference state is intentionally limited in the first version:

- local sub-block solves handle X/Y registration and binary PA where needed;
- observation-level updates handle separation, total flux, contrast, plate
  scale, and low-order M1/M2 Z4-Z11 WFE terms;
- spectral shape, high-order WFE maps, detector maps, QE/filter response, and
  smear are nuisance realism terms for now.

Follow-on tasks should implement the schema blocks incrementally:

1. spectral throughput deck and diagnostics;
2. high-order WFE truth/knowledge deck;
3. detector pixel-offset and flat-field deck;
4. trajectory-derived smear model;
5. full binary iterative campaign wrapper and proposal-facing diagnostics.
