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

## Tiny Binary Iterative Smoke

`full_fidelity_binary_iterative_smoke.yaml` is the first executable, intentionally tiny wrapper recipe. It is not the production full-fidelity campaign. The wrapper at `examples/scripts/run_full_fidelity_binary_iterative_campaign.py` translates this narrow schema into the existing observation-bias campaign runner.

It composes:

- binary/Alpha Cen target source config;
- target-aware spectral truth/reference grids with component-specific SEDs;
- high-order WFE truth maps and reference knowledge-error maps;
- trajectory frame-center truth plus smear sidecars in `metadata_only` mode;
- conservative physical iterative windows.

The Data/Inference split contract is written under the run root as:

- `model_split/model_split.json`
- `model_split/model_split_summary.json`
- `template_hashes.csv`
- `campaign_plan.json` top-level `model_split` and `template_hashes`
- per-row hash/path columns in `subblock_plan.csv`, `expected_outputs.csv`, and `iterative_plan.csv`

Trace and render templates use the truth system config. Inference templates use the reference/inference system config. Spectral shape and high-order map pixels are not optimizer-visible parameters.

Dry-run:

```bash
PYTHONPATH=src python examples/scripts/run_full_fidelity_binary_iterative_campaign.py \
  --config examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_binary_iterative_smoke.yaml \
  --run-name full_fidelity_binary_iterative_smoke_dryrun \
  --dry-run \
  --no-resource-time
```

Optional tiny execution should remain opt-in and local-scale:

```bash
PYTHONPATH=src python examples/scripts/run_full_fidelity_binary_iterative_campaign.py \
  --config examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_binary_iterative_smoke.yaml \
  --run-name full_fidelity_binary_iterative_smoke_exec \
  --max-workers 1 \
  --no-resource-time
```

Deferred: production-scale campaigns, dynamic crop/ROI-origin realism, per-frame dynamic smear kernels, active high-order map inference, spectral-shape inference, and full Bayesian recursive filtering.
