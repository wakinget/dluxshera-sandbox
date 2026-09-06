from __future__ import annotations

from dataclasses import dataclass
import errno
import os
from pathlib import Path
import re
import shutil
from typing import Callable, Mapping, Sequence
import uuid

from .pairs import load_pair_manifest
from .splits import load_split_registry, split_registry_content_sha256

__all__ = [
    "PreparedSbatchSubmission",
    "SlurmProfile",
    "build_sbatch_command",
    "parse_sbatch_job_id",
    "persist_study_contract_artifacts",
    "prepare_sbatch_submission",
    "slurm_profile",
]

SBATCH_JOB_ID_PATTERN = re.compile(r"(?P<job_id>\d+)(?:;[A-Za-z0-9_.-]+)?")

SUBMITTED_ENV_KEYS = (
    "ML_SITE",
    "ML_REPO_ROOT",
    "ML_CONDA_SH",
    "ML_CONDA_ENV",
    "ML_CONDA_PREFIX",
    "ML_STUDY_PATH",
    "ML_EXPERIMENT_ID",
    "ML_RUN_ID",
    "ML_PREPARED_ROOT",
    "ML_SPLIT_REGISTRY",
    "ML_VALIDATION_MANIFEST",
    "ML_TEST_MANIFEST",
    "ML_RUN_DIR",
    "ML_PERSIST_DIR",
    "ML_PERSIST_ARTIFACT_ROOT",
    "ML_DEVICE",
    "ML_RESUME_CHECKPOINT",
    "ML_OVERWRITE",
    "ML_SOURCE_COMMIT",
    "ML_SOURCE_ARCHIVE_ID",
    "DLUXSHERA_SOURCE_COMMIT",
    "DLUXSHERA_SOURCE_ARCHIVE_ID",
)

REQUIRED_RUN_ENV_KEYS = (
    "ML_STUDY_PATH",
    "ML_EXPERIMENT_ID",
    "ML_RUN_ID",
    "ML_PREPARED_ROOT",
    "ML_SPLIT_REGISTRY",
    "ML_VALIDATION_MANIFEST",
    "ML_TEST_MANIFEST",
    "ML_RUN_DIR",
)


@dataclass(frozen=True)
class SlurmProfile:
    """Describe the scheduler options that differ across ML execution sites."""

    name: str
    account: str
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int = 8
    time: str = "08:00:00"
    partition: str | None = None
    mem: str | None = None
    gres: str | None = None


@dataclass(frozen=True)
class PreparedSbatchSubmission:
    """Hold the command, environment, and log paths for one Slurm submission."""

    command: list[str]
    environment: dict[str, str]
    output: Path
    error: Path
    log_directories: tuple[Path, ...]
    missing_required_environment: tuple[str, ...]


def slurm_profile(name: str) -> SlurmProfile:
    """Return a tracked Slurm profile for known ML execution sites."""
    key = str(name).lower().replace("-", "_")
    if key == "gattaca2":
        return SlurmProfile(
            name="gattaca2",
            account="shera_hpc",
            cpus_per_task=8,
            time="24:00:00",
            mem="64G",
        )
    if key in {"tacc_ls6", "ls6", "lonestar6"}:
        return SlurmProfile(
            name="tacc_ls6",
            account="JPL-PUB",
            partition="gpu-a100-small",
            cpus_per_task=8,
            time="08:00:00",
        )
    raise ValueError(f"Unknown ML Slurm site profile {name!r}.")


def build_sbatch_command(
    profile: SlurmProfile,
    *,
    script: Path,
    job_name: str,
    output: Path | None = None,
    error: Path | None = None,
    export: str | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build an ``sbatch`` command without invoking Slurm."""
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--account={profile.account}",
        f"--nodes={int(profile.nodes)}",
        f"--ntasks={int(profile.ntasks)}",
        f"--cpus-per-task={int(profile.cpus_per_task)}",
        f"--time={profile.time}",
        "--parsable",
    ]
    if export is not None:
        cmd.append(f"--export={export}")
    if profile.partition:
        cmd.append(f"--partition={profile.partition}")
    if profile.mem:
        cmd.append(f"--mem={profile.mem}")
    if profile.gres:
        cmd.append(f"--gres={profile.gres}")
    if output is not None:
        cmd.append(f"--output={output}")
    if error is not None:
        cmd.append(f"--error={error}")
    cmd.extend(str(arg) for arg in extra_args)
    cmd.append(str(script))
    return cmd


def _resolve_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _submitted_environment(
    *,
    base_env: Mapping[str, str],
    explicit_env: Mapping[str, Path | str | int | bool | None],
) -> dict[str, str]:
    """Return a submit environment with stale ML provenance stripped first."""
    cleaned = {
        str(key): str(value)
        for key, value in base_env.items()
        if not str(key).startswith("ML_")
        and not str(key).startswith("DLUXSHERA_SOURCE_")
    }
    for key, value in explicit_env.items():
        if value is not None:
            cleaned[str(key)] = str(value)
    return cleaned


def prepare_sbatch_submission(
    profile: SlurmProfile,
    *,
    script: Path,
    job_name: str,
    submitted_env: Mapping[str, Path | str | int | bool | None],
    log_root: Path | None = None,
    output: Path | None = None,
    error: Path | None = None,
    extra_args: Sequence[str] = (),
    base_env: Mapping[str, str] | None = None,
) -> PreparedSbatchSubmission:
    """Prepare a Slurm submission without invoking Slurm.

    The returned environment strips parent-shell ``ML_*`` and source-provenance
    variables before applying the submitted values.  This keeps an explicit CLI
    run identity from being overridden by stale shell state.
    """
    resolved_log_root = _resolve_path(
        log_root if log_root is not None else Path("work/experiments/ml/hpc/logs")
    )
    resolved_output = (
        _resolve_path(output)
        if output is not None
        else resolved_log_root / "%x-%j.out"
    )
    resolved_error = (
        _resolve_path(error)
        if error is not None
        else resolved_log_root / "%x-%j.err"
    )
    log_directories = tuple(dict.fromkeys((resolved_output.parent, resolved_error.parent)))
    filtered_env = {
        key: submitted_env.get(key)
        for key in SUBMITTED_ENV_KEYS
        if submitted_env.get(key) not in (None, "")
    }
    environment = _submitted_environment(
        base_env=os.environ if base_env is None else base_env,
        explicit_env=filtered_env,
    )
    missing = tuple(key for key in REQUIRED_RUN_ENV_KEYS if key not in environment)
    command = build_sbatch_command(
        profile,
        script=_resolve_path(script),
        job_name=job_name,
        output=resolved_output,
        error=resolved_error,
        export="ALL",
        extra_args=extra_args,
    )
    return PreparedSbatchSubmission(
        command=command,
        environment=environment,
        output=resolved_output,
        error=resolved_error,
        log_directories=log_directories,
        missing_required_environment=missing,
    )


def parse_sbatch_job_id(stdout: str) -> str:
    """Return the final Slurm job ID from ``sbatch --parsable`` stdout.

    TACC's ``sbatch --parsable`` wrapper can print a banner before the numeric
    job ID, so callers must scan from the end instead of assuming one-line
    output.  Gattaca2 Edge can return Slurm's cluster-qualified parsable form,
    for example ``576430;edge``.  The canonical stored ID is the numeric prefix.
    """
    for line in reversed(str(stdout).splitlines()):
        candidate = line.strip()
        match = SBATCH_JOB_ID_PATTERN.fullmatch(candidate)
        if match:
            return match.group("job_id")
    raise ValueError(
        f"Could not find a valid Slurm job ID in sbatch stdout: {stdout!r}."
    )


def _split_identity(path: Path) -> dict[str, str]:
    registry = load_split_registry(path)
    return {
        "artifact_id": registry.artifact_id,
        "content_sha256": split_registry_content_sha256(registry),
    }


def _pair_manifest_identity(path: Path) -> dict[str, str | None]:
    manifest = load_pair_manifest(path)
    return {
        "artifact_id": manifest.artifact_id,
        "content_sha256": manifest.manifest.get("content_identity", {}).get("sha256"),
    }


def _copy_file_artifact(
    *,
    source: Path,
    destination: Path,
    identity_fn: Callable[[Path], dict[str, str]],
) -> dict[str, str]:
    source_identity = identity_fn(source)
    status = "copied"
    if destination.exists():
        destination_identity = identity_fn(destination)
        if destination_identity != source_identity:
            raise FileExistsError(
                "Persistent artifact destination already contains a different artifact "
                f"({destination}: expected={source_identity}, actual={destination_identity})."
            )
        status = "exists"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_destination = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            shutil.copy2(source, temp_destination)
            try:
                os.link(temp_destination, destination)
            except FileExistsError:
                destination_identity = identity_fn(destination)
                if destination_identity != source_identity:
                    raise FileExistsError(
                        "Persistent artifact destination already contains a different artifact "
                        f"({destination}: expected={source_identity}, actual={destination_identity})."
                    )
                status = "exists"
        finally:
            temp_destination.unlink(missing_ok=True)
    return {
        "source": str(source),
        "destination": str(destination),
        "status": status,
        **source_identity,
    }


def _is_publication_race(exc: OSError) -> bool:
    return isinstance(exc, FileExistsError) or exc.errno in {errno.EEXIST, errno.ENOTEMPTY}


def _copy_directory_artifact(
    *,
    source: Path,
    destination: Path,
    identity_fn: Callable[[Path], dict[str, str | None]],
) -> dict[str, str | None]:
    source_identity = identity_fn(source)
    status = "copied"
    if destination.exists():
        destination_identity = identity_fn(destination)
        if destination_identity != source_identity:
            raise FileExistsError(
                "Persistent artifact destination already contains a different artifact "
                f"({destination}: expected={source_identity}, actual={destination_identity})."
            )
        status = "exists"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_destination = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            shutil.copytree(source, temp_destination)
            try:
                temp_destination.rename(destination)
            except OSError as exc:
                if not _is_publication_race(exc) or not destination.exists():
                    raise
                destination_identity = identity_fn(destination)
                if destination_identity != source_identity:
                    raise FileExistsError(
                        "Persistent artifact destination already contains a different artifact "
                        f"({destination}: expected={source_identity}, actual={destination_identity})."
                    )
                status = "exists"
        finally:
            if temp_destination.exists():
                shutil.rmtree(temp_destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "status": status,
        **source_identity,
    }


def persist_study_contract_artifacts(
    *,
    artifact_root: Path,
    split_registry_path: Path,
    validation_manifest_path: Path,
    test_manifest_path: Path,
) -> dict[str, dict[str, str | None]]:
    """Persist compact split and frozen-evaluation artifacts idempotently."""
    root = _resolve_path(artifact_root)
    validation_identity = _pair_manifest_identity(Path(validation_manifest_path))
    test_identity = _pair_manifest_identity(Path(test_manifest_path))
    return {
        "split": _copy_file_artifact(
            source=Path(split_registry_path),
            destination=root / "split" / Path(split_registry_path).name,
            identity_fn=_split_identity,
        ),
        "validation_pairs": _copy_directory_artifact(
            source=Path(validation_manifest_path),
            destination=root / "validation_pairs" / str(validation_identity["artifact_id"]),
            identity_fn=_pair_manifest_identity,
        ),
        "test_pairs": _copy_directory_artifact(
            source=Path(test_manifest_path),
            destination=root / "test_pairs" / str(test_identity["artifact_id"]),
            identity_fn=_pair_manifest_identity,
        ),
    }
