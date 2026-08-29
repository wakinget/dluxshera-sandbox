# Canonical Smear Sensitivity 202608

This recipe prepares controlled, single-image prescribed-Monte-Carlo campaigns
for line-smear sensitivity in the canonical 23-scalar binary astrometry
estimator.

The campaigns are a canonical sensitivity study only. They are not a validated
replacement for the full-fidelity iterative ADORA estimator.

## Audit Summary

Native line smear uses the detector `ApplyConvolution` layer with
`kernel.kind: line`.

- Schema: `kind: line`, `length`, `sigma_perp`, `theta_deg`, `kernel_size`,
  `units`.
- Length: total finite line-segment length.
- Units: `detector_pix` or `psf_pix`; this campaign uses `detector_pix`.
- Angle: degrees counter-clockwise from detector +X toward detector +Y.
- Centering: the generated kernel grid is centered at
  `(kernel_size - 1) / 2`, and the finite segment is symmetric about zero
  along the rotated line coordinate.
- Normalization: kernels are divided by `sum(kernel)`.
- Support: `kernel_size` must be a positive odd integer. Even supports are
  rejected.
- Zero smear: requested `L_truth_pix=0.0` is genuine no-smear execution. The
  named `smear` detector layer is removed from both truth and inference systems.
- Trajectory conversion: `trajectory_smear.py` computes displacement from
  exposure endpoints, divides by plate scale for detector-pixel length, and
  uses `atan2(dy, dx)` for `theta_deg`.

The prescribed-MC runner already supports separate truth/data and inference
systems via `experiment.inference_system`. Generic plan rows currently forbid
`system.*` and `experiment.*` overrides, so this campaign writes one compact
prescription per condition. Each prescription starts from the resolved
`SHERA_FLIGHT_3P_CONV` system preset, preserves the preset detector stack, sets
the source exposure to `0.05 s`, removes the named `jitter` layer, and then
updates or removes only the named `smear` layer for the condition.

Detector behavior owned by the preset and detector builder is not duplicated in
this recipe. In particular, `pixel_mtf`, diffusion, pixel offsets, pixel
response, and native pixel-MTF / `optics.oversample` behavior remain inherited
from the preset path.

Derivative support uses the inference-layer `fim_theta` machinery.
`fim_theta()` computes the fixed-variance Gaussian Fisher/Gauss-Newton matrix
`J.T @ W @ J` from the Binder prediction Jacobian. Optional `--hessian`
diagnostics use `hessian_theta()` to compute the observed scalar NLL Hessian;
under model mismatch this Hessian can differ from the PSD Fisher matrix and may
be indefinite.

## Campaign Families

Family A is matched smear/information loss:

- `L_truth_pix = [0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 2.0]`
- orientations: parallel and perpendicular
- truth and model kernels match exactly
- count: 14 rows; the zero-smear duplicate orientations are retained
  as duplicated no-smear optical systems with distinct condition metadata

Family B is smear-length knowledge error:

- `L_truth_pix = [0.5, 1.0]`
- orientations: parallel and perpendicular
- `epsilon_L_percent = [-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20]`
- `L_model = L_truth * (1 + epsilon_L_percent / 100)`
- `theta_model = theta_truth`
- count: 44 rows

Family C is smear-direction knowledge error:

- `L_truth_pix = [0.5, 1.0]`
- orientations: parallel and perpendicular
- `delta_theta_deg = [-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20]`
- `L_model = L_truth`
- `theta_model = theta_truth + delta_theta_deg`
- count: 44 rows

Parallel and perpendicular are relative to the resolved canonical binary PA.
The generator records `phi_truth_deg`, `theta_truth_deg`, and
`theta_model_deg` for every condition.

## Objective and Parameters

The generated prescriptions force:

- one deterministic image per condition;
- exposure time `0.05 s`;
- observation noise disabled;
- deterministic variance `max(model_image, 1.0)`, matching the canonical
  disabled-noise convention;
- optimizer loss `nll`;
- no MAP prior penalty;
- full physical 23-scalar fit, not Schur-reduced optimization;
- SGD with `base_lr=0.7`, a 10-step linear warmup from factor `0.125`,
  and a maximum of 200 optimizer updates;
- existing early stopping enabled after at least 40 updates, with patience 10
  and `loss_rtol=1.0e-8`;
- full whitened FIM eigenbasis optimization with no `truncate_k` or eigenvalue
  truncation;
- deterministic paired prior initialization;
- per-run diagnostic plots enabled.

The initialization prior is used only to draw the starting point. The objective
remains NLL; no MAP prior penalty is included.

Initialization prior scales:

- `source.separation_as`: `Normal`, `sigma=1.0e-4` arcsec
- `source.position_angle_deg`: `Uniform`, `sigma=1.0e-2` deg
- `source.x_position_as`: `Normal`, `sigma=1.0e-2` arcsec
- `source.y_position_as`: `Normal`, `sigma=1.0e-2` arcsec
- `source.log_flux_total`: `LogNormal`, `sigma=1.0e-4`
- `source.contrast`: `LogNormal`, `sigma=1.0e-4`
- `optics.plate_scale_as_per_pix`: `LogNormal`, `sigma=1.0e-3`
- `optics.primary.zernike_coeffs_nm`: `Normal`, scalar `sigma=2.0`
- `optics.secondary.zernike_coeffs_nm`: `Normal`, scalar `sigma=2.0`

Each campaign condition is written as a one-row prescription with the common
experiment seed `20260821`. The prescribed-MC runner folds that seed with run
index 0 and then splits the result for initialization, so A/B/C conditions share
the same deterministic physical prior draw. This keeps condition comparisons
paired with respect to the optimizer starting point.

Truth/inference mismatch isolation is audited in each condition manifest:

- Family A: truth and inference systems match exactly after the common
  preset-derived no-jitter construction.
- Family B: truth and inference systems differ only in
  `detector.layers.smear.kernel.length`, except the zero-error rows which match.
- Family C: truth and inference systems differ only in
  `detector.layers.smear.kernel.theta_deg`, except the zero-error rows which
  match.

The prescribed-MC runtime must preserve these inference-system defaults when it
constructs the effective inference `ParameterStore`. Shared physical
truth/run overrides may be applied to both data and inference stores where
supported, but common `ParamSpec` values must not be blindly copied from the
data store into the inference store. For model-mismatch campaigns, audit the
effective runtime store/binder state, not only the prescription dictionaries.

The parameter layout is resolved from the current config and written into the
derivative sidecars. Expected scalar count is 23:

- `source.separation_as`
- `source.position_angle_deg`
- `source.x_position_as`
- `source.y_position_as`
- `source.log_flux_total`
- `source.contrast`
- `optics.plate_scale_as_per_pix`
- `optics.primary.zernike_coeffs_nm[0:8]`
- `optics.secondary.zernike_coeffs_nm[0:8]`

## Commands

Generate the full A-C campaign scaffold:

```bash
python3 examples/recipes/canonical_smear_sensitivity_202608/canonical_smear_campaign.py generate \
  --campaign-root Results/canonical_smear_sensitivity_202608 \
  --families A B C
```

Generate a small smoke scaffold:

```bash
python3 examples/recipes/canonical_smear_sensitivity_202608/canonical_smear_campaign.py generate \
  --campaign-root Results/canonical_smear_sensitivity_202608_smoke \
  --families A B C \
  --smoke
```

Dry-run without writing files:

```bash
python3 examples/recipes/canonical_smear_sensitivity_202608/canonical_smear_campaign.py generate \
  --dry-run \
  --families A B C
```

Run one indexed condition locally:

```bash
python3 examples/recipes/canonical_smear_sensitivity_202608/canonical_smear_campaign.py run-index \
  --campaign-root Results/canonical_smear_sensitivity_202608_smoke \
  --index 0
```

Aggregate and plot:

```bash
python3 examples/recipes/canonical_smear_sensitivity_202608/canonical_smear_campaign.py aggregate \
  --campaign-root Results/canonical_smear_sensitivity_202608_smoke

python3 examples/recipes/canonical_smear_sensitivity_202608/canonical_smear_campaign.py plot \
  --campaign-root Results/canonical_smear_sensitivity_202608_smoke
```

Add `--hessian` to `aggregate` only after a benchmark confirms Hessian cost is
acceptable.

## Gattaca2 Sequence

The generated `submit_array.sbatch` follows the repository Gattaca2 convention:
it initializes shared Miniforge, activates
`/scratch-jpl/shera_hpc/dmckeith/conda/envs/dluxshera-py311` by explicit
prefix, prepends the exact generated worktree `src` directory to `PYTHONPATH`,
and prints Python/JAX/dLuxShera import diagnostics including
`dluxshera.__file__` before running science. It does not hard-code
`sbatch -M edge`; submit with `sbatch -M edge <file>` externally if Edge
execution is desired.

Generated Slurm resources are:

- condition array: 2 CPUs, 24 GB, 30 minutes per condition
- aggregate job: 2 CPUs, 24 GB, 3 hours

Generation creates `<campaign_root>/slurm/` before submission so Slurm can open
the declared output/error paths.

1. Generate the smoke scaffold.
2. Run `status.sh` to confirm prescriptions exist. `status.sh` is lightweight
   and login-node safe.
3. Submit the generated `submit_array.sbatch` for the smoke root.
4. Run aggregation and plotting on a compute allocation, preferably with the
   generated `aggregate.sbatch` on Gattaca2. `aggregate` reconstructs models and
   computes JAX derivatives; production aggregation over 102 conditions should
   not run directly on a login node. The 3-condition smoke aggregation should
   also preferably run on a compute node.
5. Generate the full A-C scaffold.
6. Submit the full generated `submit_array.sbatch`.
7. Re-run `status.sh` and compute-node aggregation/plotting until all rows have
   run summaries and derivative sidecars.

Do not submit production jobs before the smoke aggregation verifies:

- matched cases have near-zero NLL gradient at truth;
- matched optimization bias is near zero;
- `optimizer.loss` is `nll`;
- parameter label count is 23;
- matrix shapes are 23 x 23.

Post-patch production preflight:

1. Generate a fresh 3-condition smoke scaffold using the patched code.
2. Confirm the resolved prescribed-MC preview reports `init.mode=prior`,
   `eigen.use_eigen=True`, `eigen.whiten_basis=True`, no truncation, SGD/NLL,
   `base_lr=0.7`, a 10-step linear warmup from factor `0.125`, `n_iter=200`,
   early stopping enabled with `min_iter=40`, and plots enabled.
3. Run the three representative smoke conditions on a compute node.
4. Inspect initial/final PSF comparison plots, loss history, signal history,
   FIM, and eigenvalue spectrum.
5. Confirm optimizer convergence and early-stopping behavior.
6. Aggregate the smoke and verify 23 labels, a 23 x 23 local-curvature matrix,
   matched near-zero final separation bias and truth-point gradient, and
   sensible nonzero mismatch response in B/C.
7. Only then generate and submit the complete 102-condition A/B/C campaign to
   Edge.

## Outputs

The generator writes:

- `campaign_manifest.json`
- `parameter_labels.json`
- `plan_all.csv`
- `plan_family_A.csv`
- `plan_family_B.csv`
- `plan_family_C.csv`
- one `prescription.yaml` and `condition_manifest.json` per condition
- `submit_array.sbatch`
- `aggregate.sbatch`
- `status.sh`
- `aggregate_and_plot.sh`

Aggregation writes:

- `summary.csv`
- per-condition `derivative_diagnostics.json`
- per-condition `derivative_diagnostics.npz`

Review plots are written under `plots/`.

For each completed prescribed-MC run, the existing per-run plots are under
`<condition>/runs/<run_id>/plots/` and include, where applicable:

- `fim.png`
- `eigenvalue_spectrum.png`
- `initial_psf_comparison.png`
- `final_psf_comparison.png`
- `loss_history.png`
- signal and parameter-history grid products
- detector-map diagnostics for detector layers with map products

Convergence metadata is written by the optimizer core to each run's
`summary.json`, `meta.json`, and `trace.npz`. Review `loss_true`/`loss_init`/
`loss_final`, `chi2_init`/`chi2_final`, `num_steps_completed`,
`optimizer.actual_num_steps`, and `optimizer.early_stopping` in those files or
the aggregate `results.csv`.
