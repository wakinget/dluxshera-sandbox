from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = (
    ROOT
    / "examples/recipes/full_fidelity_algorithm_campaign_template/"
    / "full_fidelity_info_damped_detector_ke_projected_30min_v1.yaml"
)
CONFIG_DIR = ROOT / "examples/recipes/full_fidelity_next_campaigns_20260703"
BENCH_DIR = ROOT / "work/full_fidelity_runtime_benchmarks_20260703"
RESULTS_ROOT = "/projects/shera_hpc/dmckeith/dLuxShera-Results"


PRIMARY_KEY = "optics.primary.zernike_coeffs_nm[*]"
SECONDARY_KEY = "optics.secondary.zernike_coeffs_nm[*]"


SCIENCE_FAMILIES = [
    {
        "name": "full_fidelity_info_damped_hoke_0p1nm_loz0p01nm_n10_w10x30_projected_30min_v1",
        "condition": "m1_0p01nm_m2_0p01nm",
        "low_order_sigma": 0.01,
        "windows": 10,
        "subblocks": 30,
        "draws": 10,
        "high_order_enabled": True,
        "high_order_ke_nm": 0.1,
        "detector_ke": None,
    },
    {
        "name": "full_fidelity_info_damped_hoke_1p0nm_loz0p01nm_n10_w10x30_projected_30min_v1",
        "condition": "m1_0p01nm_m2_0p01nm",
        "low_order_sigma": 0.01,
        "windows": 10,
        "subblocks": 30,
        "draws": 10,
        "high_order_enabled": True,
        "high_order_ke_nm": 1.0,
        "detector_ke": None,
    },
    {
        "name": "full_fidelity_info_damped_pixelposke_1em4pix_n10_w10x30_projected_30min_v1",
        "condition": "m1_0p3nm_m2_0p3nm",
        "low_order_sigma": 0.3,
        "windows": 10,
        "subblocks": 30,
        "draws": 10,
        "high_order_enabled": False,
        "detector_ke": 1.0e-4,
    },
    {
        "name": "full_fidelity_info_damped_pixelposke_5em4pix_n10_w10x30_projected_30min_v1",
        "condition": "m1_0p3nm_m2_0p3nm",
        "low_order_sigma": 0.3,
        "windows": 10,
        "subblocks": 30,
        "draws": 10,
        "high_order_enabled": False,
        "detector_ke": 5.0e-4,
    },
    {
        "name": "full_fidelity_info_damped_pixelposke_1em3pix_n10_w10x30_projected_30min_v1",
        "condition": "m1_0p3nm_m2_0p3nm",
        "low_order_sigma": 0.3,
        "windows": 10,
        "subblocks": 30,
        "draws": 10,
        "high_order_enabled": False,
        "detector_ke": 1.0e-3,
    },
    {
        "name": "full_fidelity_info_damped_no_ke_single_w30x30_actual15min_v1",
        "condition": "m1_0p3nm_m2_0p3nm",
        "low_order_sigma": 0.3,
        "windows": 30,
        "subblocks": 30,
        "draws": 1,
        "high_order_enabled": False,
        "detector_ke": None,
    },
    {
        "name": "full_fidelity_info_damped_no_ke_single_w60x30_actual30min_v1",
        "condition": "m1_0p3nm_m2_0p3nm",
        "low_order_sigma": 0.3,
        "windows": 60,
        "subblocks": 30,
        "draws": 1,
        "high_order_enabled": False,
        "detector_ke": None,
    },
]


BENCHMARK_FAMILIES = [
    {
        "name": "full_fidelity_runtime_benchmark_truth_2x20f_v1",
        "phi_ref": "truth_when_available",
    },
    {
        "name": "full_fidelity_runtime_benchmark_recovered_2x20f_v1",
        "phi_ref": "recovered",
    },
]


def _load_template() -> dict:
    payload = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Template is not a YAML mapping: {TEMPLATE}")
    return payload


def _set_detector_ke(experiment: dict, sigma_pix: float | None) -> None:
    if sigma_pix is None:
        experiment["detector_calibration_knowledge_error"] = {"enabled": False}
        return
    experiment["detector_calibration_knowledge_error"] = {
        "enabled": True,
        "apply_to": "inference",
        "realization_policy": "fixed_per_experiment",
        "pixel_offsets": {
            "enabled": True,
            "sigma_pix": float(sigma_pix),
            "distribution": "normal",
        },
        "pixel_response": {
            "enabled": False,
            "sigma_fractional": 0.0,
            "distribution": "normal",
        },
    }


def _set_high_order_wfe(experiment: dict, enabled: bool, ke_nm: float | None) -> None:
    hoke = copy.deepcopy(experiment.get("high_order_wfe", {}))
    if not enabled:
        experiment["high_order_wfe"] = {"enabled": False}
        return
    hoke["enabled"] = True
    hoke.setdefault("truth", {})["enabled"] = True
    hoke.setdefault("inference", {})["enabled"] = True
    hoke["inference"]["mode"] = "knowledge_error"
    hoke["inference"]["use_truth_common_map"] = True
    hoke["inference"].setdefault("knowledge_error", {})["enabled"] = True
    hoke["inference"]["knowledge_error"]["amplitude_nm_rms"] = float(ke_nm)
    experiment["high_order_wfe"] = hoke


def _set_prior_draws(
    experiment: dict,
    *,
    draws: int,
    condition: str,
    low_order_sigma: float,
) -> None:
    prior = copy.deepcopy(experiment["prior_draws"])
    prior["n_cases"] = int(draws)
    prior["draws_per_condition"] = int(draws)
    prior["conditions"] = [
        {
            "condition_name": condition,
            "sigmas": {
                PRIMARY_KEY: {
                    "kind": "absolute",
                    "sigma": float(low_order_sigma),
                    "unit": "nm",
                },
                SECONDARY_KEY: {
                    "kind": "absolute",
                    "sigma": float(low_order_sigma),
                    "unit": "nm",
                },
            },
        }
    ]
    experiment["prior_draws"] = prior


def _base_campaign(family: dict) -> dict:
    cfg = copy.deepcopy(_load_template())
    experiment = cfg["experiment"]
    experiment["run_name"] = family["name"]
    experiment["iterative"]["windows_per_draw"] = int(family["windows"])
    experiment["iterative"]["subblocks_per_window"] = int(family["subblocks"])
    experiment["subblocks"]["n_frames"] = 20
    experiment["iterative_forecast"]["projected_windows"] = 60
    experiment["iterative_forecast"]["observation_duration_s"] = 1800.0
    _set_detector_ke(experiment, family.get("detector_ke"))
    _set_high_order_wfe(
        experiment,
        bool(family["high_order_enabled"]),
        family.get("high_order_ke_nm"),
    )
    _set_prior_draws(
        experiment,
        draws=int(family["draws"]),
        condition=str(family["condition"]),
        low_order_sigma=float(family["low_order_sigma"]),
    )
    return cfg


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _benchmark_sbatch(name: str, config_path: Path) -> str:
    rel_config = config_path.relative_to(ROOT).as_posix()
    return f"""#!/bin/bash
#SBATCH --job-name={name[:80]}
#SBATCH --output={RESULTS_ROOT}/slurm_logs/%x-%j.out
#SBATCH --error={RESULTS_ROOT}/slurm_logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -euo pipefail

REPO="${{REPO:-$HOME/dluxshera-sandbox}}"
RESULTS_ROOT="${{RESULTS_ROOT:-{RESULTS_ROOT}}}"
RUN_RESULTS_ROOT="${{RESULTS_ROOT%/}}/observation_bias_campaign"
CONFIG="${{CONFIG:-{rel_config}}}"
RUN_NAME="${{RUN_NAME:-{name}}}"
MAX_WORKERS="${{MAX_WORKERS:-1}}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export JAX_COMPILATION_CACHE_DIR="/scratch/shera_hpc/$USER/jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1

mkdir -p "$RUN_RESULTS_ROOT" "$JAX_COMPILATION_CACHE_DIR" "$RESULTS_ROOT/slurm_logs"
cd "$REPO"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dluxshera-py311

PYTHONPATH=src python examples/scripts/run_full_fidelity_binary_iterative_campaign.py \\
  --config "$CONFIG" \\
  --run-name "$RUN_NAME" \\
  --results-root "$RUN_RESULTS_ROOT" \\
  --max-workers "$MAX_WORKERS" \\
  --fail-fast \\
  --resource-time auto \\
  --profile-runtime \\
  --profile-runtime-detail basic \\
  --memory-diagnostics
"""


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    for family in SCIENCE_FAMILIES:
        path = CONFIG_DIR / f"{family['name']}.yaml"
        _write_yaml(path, _base_campaign(family))
    for family in BENCHMARK_FAMILIES:
        payload = _base_campaign(
            {
                "name": family["name"],
                "condition": "m1_0p3nm_m2_0p3nm",
                "low_order_sigma": 0.3,
                "windows": 1,
                "subblocks": 2,
                "draws": 1,
                "high_order_enabled": False,
                "detector_ke": None,
            }
        )
        payload["experiment"]["subblocks"]["phi_ref"] = family["phi_ref"]
        payload["experiment"]["subblocks"]["profile_runtime"] = True
        payload["experiment"]["subblocks"]["profile_runtime_detail"] = "basic"
        payload["experiment"]["subblocks"]["memory_diagnostics"] = True
        path = CONFIG_DIR / f"{family['name']}.yaml"
        _write_yaml(path, payload)
        sbatch = BENCH_DIR / f"{family['name']}.sbatch"
        sbatch.write_text(_benchmark_sbatch(family["name"], path), encoding="utf-8")
        sbatch.chmod(0o755)


if __name__ == "__main__":
    main()
