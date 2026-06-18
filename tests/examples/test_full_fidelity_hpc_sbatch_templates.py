from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = ROOT / "examples" / "recipes" / "full_fidelity_algorithm_campaign_template"


def test_hpc_sbatch_templates_are_parameterized() -> None:
    for name, default_workers, resource_time in (
        (
            "full_fidelity_registration_solve_smoke_hpc.sbatch",
            'MAX_WORKERS="${MAX_WORKERS:-1}"',
            'USE_RESOURCE_TIME="${USE_RESOURCE_TIME:-0}"',
        ),
        (
            "full_fidelity_zernike_2x2_self_correction_hpc.sbatch",
            'MAX_WORKERS="${MAX_WORKERS:-4}"',
            'USE_RESOURCE_TIME="${USE_RESOURCE_TIME:-0}"',
        ),
        (
            "full_fidelity_iterative_campaign_hpc.sbatch",
            'MAX_WORKERS="${MAX_WORKERS:-5}"',
            'USE_RESOURCE_TIME="${USE_RESOURCE_TIME:-1}"',
        ),
    ):
        text = (TEMPLATE_DIR / name).read_text(encoding="utf-8")
        for expected in (
            'CONFIG="${CONFIG:',
            'RUN_NAME="${RUN_NAME:',
            'RESULTS_ROOT="${RESULTS_ROOT:-/scratch/shera_hpc/$USER/dluxshera}"',
            default_workers,
            'FAIL_FAST="${FAIL_FAST:-1}"',
            'ANALYZE_AFTER_RUN="${ANALYZE_AFTER_RUN:-1}"',
            resource_time,
            "conda activate dluxshera-py311",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export NUMEXPR_NUM_THREADS=1",
            'export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"',
            'export JAX_COMPILATION_CACHE_DIR="/scratch/shera_hpc/$USER/jax_cache"',
            "export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0",
            "export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1",
            "PYTHONPATH=src python examples/scripts/run_full_fidelity_binary_iterative_campaign.py",
            "--config",
            "--run-name",
            "--results-root",
            "--max-workers",
            "--fail-fast",
            "--no-resource-time",
            "PYTHONPATH=src python examples/scripts/analyze_full_fidelity_binary_iterative_campaign.py",
            "--strict",
            "--max-image-examples 4",
        ):
            assert expected in text
