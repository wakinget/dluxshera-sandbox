# Examples

These examples are runnable artifacts (scripts + notebooks). Install the
project in editable mode before executing them:

```bash
python -m pip install -e .
```

## Recipes + runners (start here)

Read-first recipes live under `examples/recipes/`. Execute-first runners live
under `examples/runners/`.

- **Canonical 3-plane runner + recipe** — Shera three-plane end-to-end workflow
  (config → resolve → binder → simulate → loss/inference → plotting). Uses the
  native `system`/`experiment` config structure, detector layer composition, and
  binder-first evaluation.
  - Recipe: `examples/recipes/canonical_astrometry.py`
  - Runner: `examples/runners/run_canonical_astrometry.py`
  - How to run:

    ```bash
    python examples/runners/run_canonical_astrometry.py
    python examples/runners/run_canonical_astrometry.py --fast
    ```
- **2-plane runner + recipe** — Same workflow on the two-plane optical system
  for a faster, simpler baseline with the same config-resolution flow.
  - Recipe: `examples/recipes/twoplane_astrometry.py`
  - Runner: `examples/runners/run_twoplane_astrometry.py`
  - How to run:

    ```bash
    python examples/runners/run_twoplane_astrometry.py
    python examples/runners/run_twoplane_astrometry.py --fast
    ```

## Notebooks

Launch Jupyter after installation, select the environment kernel, and open any
notebook under `examples/notebooks/`. The notebooks are written to use the
installed `dluxshera` package—no `sys.path` tweaks should be required.

- **minimal_example.ipynb** — Minimal forward-model + optimization walkthrough
  for quick orientation.
- **Shera_Eigen_Inference_Example.ipynb** — Eigen-θ inference in the Shera
  workflow, with plots that mirror the scripts.
- **notebook_setup.py** — Shared notebook utilities (imports, plotting defaults).

To launch Jupyter:

```bash
jupyter lab
```

## Utility scripts

Scripts under `examples/scripts/` operate on run directories produced by the
recipes/runners.

- **summarize_runs.py** — Crawl run directories and emit a CSV summary from per-run artifacts (independent of aggregate `results.csv` orientation).
  - How to run:

    ```bash
    python examples/scripts/summarize_runs.py --help
    python examples/scripts/summarize_runs.py Results/
    ```
- **analyze_checkpoint_gradients.py** — Inspect saved checkpoints and write
  gradient diagnostics under a run’s `diag/` directory.
  - How to run:

    ```bash
    python examples/scripts/analyze_checkpoint_gradients.py --help
    python examples/scripts/analyze_checkpoint_gradients.py Results/<run_dir>
    ```
- **aggregate_detector_ke_sweep.py** — Aggregate multiple prescribed-MC detector
  knowledge-error experiment directories (for example `ke_0`, `ke_1e-3`, ...)
  into cross-experiment `sweep_runs.csv` and `sweep_summary.csv`.
  - Use this when each sweep point is its own experiment directory and you want
    one outer table for all runs plus grouped per-KE statistics.
  - The aggregator reads configured detector KE settings from each
    `prescription.*`, run outcomes from row-oriented `results.csv`, and
    per-run realized detector KE metadata from `runs*/<run_id>/meta.json` when
    available (older runs without meta KE fields are still supported).
  - How to run:

    ```bash
    python examples/scripts/aggregate_detector_ke_sweep.py --help
    python examples/scripts/aggregate_detector_ke_sweep.py \
      --root Results/detector_ke_sweep
    ```
- **generate_prescribed_mc_sweep.py** — Scaffold multi-YAML prescribed-MC
  sweeps from one base prescription by creating a timestamped root,
  per-point subdirectories, per-point `prescription.yaml` files, and a
  root-level `sweep_manifest.json`.
  - Use `--mode detector_ke` for inference-side detector knowledge-error sweeps
    (for example `pixel_offsets`/`pixel_response` `knowledge_error.scale`).
  - Use `--mode scalar_field` for structural top-level scalar sweeps (for
    example `system.optics.psf_npix`) when data and inference should share the
    same top-level `system`.
  - Detector-KE example:

    ```bash
    python examples/scripts/generate_prescribed_mc_sweep.py \
      --base examples/recipes/prescribed_mc_template/prescription.yaml \
      --mode detector_ke \
      --scales 0 1e-4 3e-4 1e-3 3e-3 1e-2 \
      --layer pixel_offsets \
      --realization-policy per_run \
      --results-orientation row
    ```

  - Scalar-field `psf_npix` crop example:

    ```bash
    python examples/scripts/generate_prescribed_mc_sweep.py \
      --base Results/detector_crop_sweep_template.yaml \
      --mode scalar_field \
      --field-path system.optics.psf_npix \
      --values 256 224 192 160 128 96 64 \
      --sweep-name detector_crop_sweep \
      --label-prefix psf_npix \
      --results-orientation row
    ```

  - Then run generated sweep points with the same shell loop pattern:

    ```bash
    for d in Results/detector_crop_sweep_*/psf_npix_*; do
      PYTHONPATH=src python examples/recipes/prescribed_monte_carlo.py \
        --outdir "$d" \
        --results-orientation row
    done
    ```

## Artifact outputs (what to look for)

Recipes/runners create a run directory (defaults to `Results/<run_id>/`) that
holds traces, summaries, plots, and optional checkpoints. For a quick tour of
the file layout and optional artifacts, see
`docs/architecture/optimization_artifacts_and_plotting.md`.

## Policy

Examples under this directory are not a Python package. Keep shared library
logic under `src/dluxshera/`; the files here stay as runnable or read-first
references.
