# ML research prototypes

Exploratory code supporting optimization decisions. **Nothing here is production
code.** These modules are not imported by `src/dluxshera`, any training path, or
any DataLoader, and they must not become so without a separate design and
scientific review. `tests/ml/test_poisson_lookup_research.py` asserts that
separation directly.

## `poisson_lookup_prototype.py`

Evaluates whether precomputed Poisson sample tables, indexed cheaply, could
replace a live Poisson sampler for S12-scale photon-noise generation.

**Outcome: rejected for S12.** The full analysis, measurements, and decision
rationale are in
[the S12 performance review](../../../../docs/dev/s12_pipeline_performance.md#research-precomputed-poisson-lookup-tables).
In short:

- Measured against vectorized NumPy Poisson — not against the slower JAX
  kernel — the lookup advantage is under 2×. NumPy has since been integrated as
  an opt-in production strategy, so that is now the live comparison.
- Nearest-integer level lookup maps every `λ < 0.5` to a deterministic zero,
  destroying the noise in exactly the faint pixels S12 studies. The mixture
  alternative fixes the mean and is overdispersed by construction.
- A table satisfying both adequate pool size and full count coverage is
  ~20.7 GiB, or ~82.8 GiB if duplicated across four workers.
- Spatial and cross-realization correlations are at the finite-sample floor, so
  correlation is *not* the failure mode; parameter quantization and
  campaign-scale table reuse are.
- With `offset_per_level=True`, translations are drawn once per count level and
  reused for every image in the call, so two identical expected images in one
  batch get the **same** draw rather than two independent ones.
  `correlation_report`'s duplicate-input probe measures this; correlation across
  differing images cannot detect it.
- The one distributionally exact construction (dyadic superposition of
  independent Poisson pools) is 3.5–4.5× *slower* than `rng.poisson`, and its
  exactness is **conditional**: it holds at lattice `λ` under the idealization
  that summed pool entries are independent. A finite table cannot supply
  unlimited independent draws, so "exact" here never means "equivalent to
  unlimited independent Poisson draws".

It writes no files and allocates no table above a 512 MiB default budget;
larger configurations require raising `--memory-budget-mib` deliberately.

```bash
PYTHONPATH=src:. python work/experiments/ml/research/poisson_lookup_prototype.py --help
PYTHONPATH=src:. python work/experiments/ml/research/poisson_lookup_prototype.py \
  --mode all --table-side 256 --table-levels 1024
```

Modes: `performance`, `fidelity`, `correlation`, `reuse`, `memory`, `exact`,
`all`. Pass `--prepared-root` to repeat the performance measurement on resident
real images.
