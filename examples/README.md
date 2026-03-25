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
- **Observation sub-block trace + renderer recipes** — Two-step time-domain
  workflow for short frame stacks:
  1) generate explicit per-frame trace CSV (`observation_subblock_trace.py`),
  2) render a central-field sub-block cube (`observation_subblock.py`).
  - Trace recipe: `examples/recipes/observation_subblock_trace.py`
  - Renderer recipe: `examples/recipes/observation_subblock.py`
  - Template directories:
    - `examples/recipes/observation_subblock_trace_template/`
    - `examples/recipes/observation_subblock_template/`

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
- **visualize_obs_subblock.py** — Generate quick-look diagnostics for
  observation sub-block renderer outputs (`*_cube.fits` plus optional trace CSV
  and manifest).
  - Writes `preview.gif`, `summary.png`, and `trace_summary.png` (when trace is
    available).
  - How to run:

    ```bash
    PYTHONPATH=src python examples/scripts/visualize_obs_subblock.py \
      --cube Results/observation_subblock/<run>/obs_subblock_*_cube.fits \
      --manifest Results/observation_subblock/<run>/manifest.json
    ```
- **generate_binary_rois.py** — Generate a static two-circle binary-star ROI
  mask as a detector `pixel_response` FITS map, plus a PNG quick-look preview.
  - Uses geometry from a resolved `--system-preset` and supports optional
    source overrides (`--separation-as`, `--position-angle-deg`,
    `--x-position-as`, `--y-position-as`) for quick studies.
  - Core controls include `--npix`, `--oversample`, and `--roi-diameter-as`;
    output paths are set by `--outfile` and `--preview`.
  - Binary clipping defaults to enabled (`--clip-to-binary`): output mask values
    are forced to `{0,1}` at threshold `0.5`. Use `--no-clip-to-binary` to keep
    anti-aliased soft edges.
  - Valid ranges: `--npix > 0`, `--oversample > 0`, `--roi-diameter-as > 0`,
    and `--separation-as >= 0` (if provided).
  - Ensure the image field of view is large enough for your geometry and ROI
    diameter; otherwise the generated mask can be all zeros.
  - How to run:

    ```bash
    python examples/scripts/generate_binary_rois.py --help
    python examples/scripts/generate_binary_rois.py
    python examples/scripts/generate_binary_rois.py \
      --system-preset SHERA_TESTBED_3P \
      --npix 192 --oversample 8 --roi-diameter-as 6.0 \
      --outfile Results/roi_mask_testbed.fits \
      --preview Results/roi_mask_testbed.png
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
