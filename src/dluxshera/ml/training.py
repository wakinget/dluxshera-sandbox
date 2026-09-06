from __future__ import annotations

import csv
import datetime as dt
import json
import os
import shutil
import socket
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in no-torch envs
    raise ModuleNotFoundError(
        "dluxshera.ml.training requires PyTorch. Install the optional ML "
        "environment, for example `python -m pip install -e .[ml]`."
    ) from exc

from dluxshera.datasets.schema import read_json, write_json

from .catalog import SampleCatalog, load_sample_catalog
from .metrics import compute_regression_metrics, metrics_by_group
from .models import build_pairwise_correction_model, count_parameters
from .noise import NoiseConfig
from .pairs import (
    PairManifest,
    PairPolicy,
    PairSampler,
    generate_frozen_pair_manifest,
    load_pair_manifest,
    write_pair_manifest,
)
from .scaling import IntensityScaler, fit_intensity_scaler
from .splits import SplitRegistry, load_split_registry, split_registry_content_sha256
from .torch_data import DynamicPairDataset, PairManifestDataset

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "DEFAULT_S01_FISHER_DISTANCE_BIN_EDGES",
    "EarlyStoppingConfig",
    "EarlyStoppingState",
    "LRSchedulerConfig",
    "default_s01_e00_config",
    "default_s01_e01_config",
    "load_run_config",
    "resolve_device",
    "train_pairwise_correction",
    "validate_fisher_distance_bin_edges",
]

CHECKPOINT_SCHEMA_VERSION = "dluxshera_ml_checkpoint/1"
DEFAULT_S01_FISHER_DISTANCE_BIN_EDGES = (0.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_info() -> dict[str, Any]:
    root = _repo_root()
    env = os.environ
    info: dict[str, Any] = {
        "source_commit": env.get("DLUXSHERA_SOURCE_COMMIT")
        or env.get("ML_SOURCE_COMMIT"),
        "source_archive_id": env.get("DLUXSHERA_SOURCE_ARCHIVE_ID")
        or env.get("ML_SOURCE_ARCHIVE_ID"),
    }
    for key, cmd in {
        "commit": ["git", "-C", str(root), "rev-parse", "HEAD"],
        "branch": ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
        "dirty": ["git", "-C", str(root), "status", "--short"],
    }.items():
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            info[key] = None
        else:
            info[key] = bool(result.stdout.strip()) if key == "dirty" else result.stdout.strip()
    info["has_git_metadata"] = info.get("commit") is not None
    return info


def _runtime_info(device: torch.device) -> dict[str, Any]:
    """Return optional execution provenance for local and SLURM runs."""
    cuda_device_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            cuda_device_name = torch.cuda.get_device_name(device.index or 0)
        except Exception:  # pragma: no cover - defensive provenance only
            cuda_device_name = None
    env = os.environ
    return {
        "python_version": ".".join(str(v) for v in os.sys.version_info[:3]),
        "torch_version": str(getattr(torch, "__version__", "")),
        "torch_cuda_version": None
        if getattr(torch.version, "cuda", None) is None
        else str(getattr(torch.version, "cuda", None)),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "torch_cuda_device_name": None if cuda_device_name is None else str(cuda_device_name),
        "resolved_device": str(device),
        "slurm_job_id": env.get("SLURM_JOB_ID"),
        "slurm_cluster_name": env.get("SLURM_CLUSTER_NAME"),
        "slurm_job_partition": env.get("SLURM_JOB_PARTITION"),
        "slurmd_nodename": env.get("SLURMD_NODENAME"),
        "hostname": socket.gethostname(),
    }


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Configuration for validation-loss early stopping."""

    enabled: bool = False
    monitor: str = "validation_loss"
    min_epochs: int = 0
    patience: int = 0
    min_delta_relative: float = 0.0

    def __post_init__(self) -> None:
        if self.monitor != "validation_loss":
            raise ValueError("Only monitor='validation_loss' is currently supported.")
        if int(self.min_epochs) < 0:
            raise ValueError("early_stopping.min_epochs must be >= 0.")
        if int(self.patience) < 0:
            raise ValueError("early_stopping.patience must be >= 0.")
        if bool(self.enabled) and int(self.patience) < 1:
            raise ValueError("early_stopping.patience must be >= 1 when enabled.")
        if float(self.min_delta_relative) < 0.0 or not np.isfinite(float(self.min_delta_relative)):
            raise ValueError("early_stopping.min_delta_relative must be finite and >= 0.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "EarlyStoppingConfig":
        if payload is None:
            return cls()
        return cls(
            enabled=bool(payload.get("enabled", False)),
            monitor=str(payload.get("monitor", "validation_loss")),
            min_epochs=int(payload.get("min_epochs", 0)),
            patience=int(payload.get("patience", 0)),
            min_delta_relative=float(payload.get("min_delta_relative", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "monitor": self.monitor,
            "min_epochs": int(self.min_epochs),
            "patience": int(self.patience),
            "min_delta_relative": float(self.min_delta_relative),
        }


@dataclass
class EarlyStoppingState:
    """Mutable early-stop progress and absolute-best checkpoint state."""

    absolute_best_loss: float = float("inf")
    best_epoch: int = -1
    reference_loss: float = float("inf")
    bad_epochs: int = 0
    epochs_completed: int = 0
    early_stopped: bool = False
    stop_epoch: int | None = None

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> "EarlyStoppingState":
        state = checkpoint.get("early_stopping_state")
        if isinstance(state, Mapping):
            return cls(
                absolute_best_loss=float(state.get("absolute_best_loss", float("inf"))),
                best_epoch=int(state.get("best_epoch", -1)),
                reference_loss=float(state.get("reference_loss", float("inf"))),
                bad_epochs=int(state.get("bad_epochs", 0)),
                epochs_completed=int(state.get("epochs_completed", 0)),
                early_stopped=bool(state.get("early_stopped", False)),
                stop_epoch=state.get("stop_epoch"),
            )

        epoch = int(checkpoint.get("epoch", -1))
        validation_metrics = checkpoint.get("validation_metrics", {})
        if isinstance(validation_metrics, Mapping) and "fisher_overall_rmse" in validation_metrics:
            loss = float(validation_metrics["fisher_overall_rmse"]) ** 2
        else:
            loss = float(checkpoint.get("best_validation_loss", float("inf")))
        return cls(
            absolute_best_loss=loss,
            best_epoch=int(checkpoint.get("best_epoch", epoch)),
            reference_loss=loss,
            bad_epochs=0,
            epochs_completed=max(epoch + 1, 0),
        )

    def update(self, *, epoch: int, metric: float, config: EarlyStoppingConfig) -> tuple[bool, bool]:
        """Update state and return ``(is_absolute_best, should_stop)``."""
        value = float(metric)
        is_best = value < float(self.absolute_best_loss)
        if is_best:
            self.absolute_best_loss = value
            self.best_epoch = int(epoch)

        if not np.isfinite(self.reference_loss):
            meaningful = True
        else:
            threshold = float(self.reference_loss) * (1.0 - float(config.min_delta_relative))
            meaningful = value <= threshold

        self.epochs_completed = int(epoch) + 1
        if meaningful:
            self.reference_loss = value
            self.bad_epochs = 0
        elif bool(config.enabled) and self.epochs_completed >= int(config.min_epochs):
            self.bad_epochs += 1

        should_stop = (
            bool(config.enabled)
            and self.epochs_completed >= int(config.min_epochs)
            and self.bad_epochs >= int(config.patience)
        )
        if should_stop:
            self.early_stopped = True
            self.stop_epoch = int(epoch)
        return is_best, should_stop

    def to_dict(self) -> dict[str, Any]:
        return {
            "absolute_best_loss": float(self.absolute_best_loss),
            "best_epoch": int(self.best_epoch),
            "reference_loss": float(self.reference_loss),
            "bad_epochs": int(self.bad_epochs),
            "epochs_completed": int(self.epochs_completed),
            "early_stopped": bool(self.early_stopped),
            "stop_epoch": self.stop_epoch,
        }


@dataclass(frozen=True)
class LRSchedulerConfig:
    """Configuration for the epoch-level learning-rate scheduler."""

    name: str = "none"
    monitor: str = "validation_loss"
    factor: float = 0.1
    patience: int = 10
    threshold_relative: float = 0.0001
    min_lr: float = 0.0
    t_max: int | None = None

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        initial_learning_rate: float,
        max_epochs: int | None = None,
    ) -> "LRSchedulerConfig":
        if payload is None:
            config = cls()
        else:
            name = str(payload.get("name", "none")).lower()
            config = cls(
                name=name,
                monitor=str(payload.get("monitor", "validation_loss")),
                factor=float(payload.get("factor", 0.1)),
                patience=_scheduler_int(payload.get("patience", 10), "lr_scheduler.patience"),
                threshold_relative=float(payload.get("threshold_relative", 0.0001)),
                min_lr=float(payload.get("min_lr", 0.0)),
                t_max=None
                if "t_max" not in payload
                else _scheduler_int(payload.get("t_max"), "lr_scheduler.t_max"),
            )
        config.validate(initial_learning_rate=initial_learning_rate, max_epochs=max_epochs)
        return config

    @property
    def is_active(self) -> bool:
        return self.name != "none"

    def validate(
        self,
        *,
        initial_learning_rate: float,
        max_epochs: int | None = None,
    ) -> None:
        lr = float(initial_learning_rate)
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("training.learning_rate must be finite and > 0.")
        if self.name not in {"none", "reduce_on_plateau", "cosine_annealing"}:
            raise ValueError(f"Unsupported lr_scheduler.name {self.name!r}.")
        if self.name == "none":
            return
        if not np.isfinite(float(self.min_lr)) or float(self.min_lr) < 0.0:
            raise ValueError("lr_scheduler.min_lr must be finite and >= 0.")
        if float(self.min_lr) > lr:
            raise ValueError("lr_scheduler.min_lr must be <= training.learning_rate.")
        if self.name == "reduce_on_plateau":
            if self.monitor != "validation_loss":
                raise ValueError("Only lr_scheduler.monitor='validation_loss' is supported.")
            if not 0.0 < float(self.factor) < 1.0:
                raise ValueError("lr_scheduler.factor must satisfy 0 < factor < 1.")
            if int(self.patience) < 0:
                raise ValueError("lr_scheduler.patience must be a nonnegative integer.")
            if (
                not np.isfinite(float(self.threshold_relative))
                or float(self.threshold_relative) < 0.0
            ):
                raise ValueError("lr_scheduler.threshold_relative must be finite and >= 0.")
            return
        if self.name == "cosine_annealing":
            if self.t_max is None or int(self.t_max) <= 0:
                raise ValueError("lr_scheduler.t_max must be a finite positive integer.")
            if max_epochs is not None and int(max_epochs) > int(self.t_max):
                raise ValueError(
                    "training.epochs must be <= lr_scheduler.t_max for "
                    "cosine_annealing."
                )
            return

    def to_dict(self) -> dict[str, Any]:
        if self.name == "none":
            return {"name": "none"}
        if self.name == "reduce_on_plateau":
            return {
                "name": self.name,
                "monitor": self.monitor,
                "factor": float(self.factor),
                "patience": int(self.patience),
                "threshold_relative": float(self.threshold_relative),
                "min_lr": float(self.min_lr),
            }
        if self.name == "cosine_annealing":
            return {
                "name": self.name,
                "t_max": int(self.t_max or 0),
                "min_lr": float(self.min_lr),
            }
        raise ValueError(f"Unsupported lr_scheduler.name {self.name!r}.")


def _scheduler_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite integer.") from exc
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"{field} must be a finite integer.")
    return int(number)


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_update(dict(out[key]), value)
        else:
            out[key] = value
    return out


def default_s01_e00_config() -> dict[str, Any]:
    """Return a quick CPU-friendly S01-E00 tiny-overfit preset."""
    return {
        "study_id": "S01",
        "experiment_id": "S01-E00",
        "run_id": "S01-E00-R001",
        "seed": 1,
        "device": "auto",
        "model": {
            "channels": [8, 16, 32],
            "embedding_dim": 32,
            "encoder_hidden_dim": 64,
            "head_hidden_dim": 64,
            "comparator": "concat_diff",
            "normalization": "batch",
            "adaptive_pool_shape": [2, 2],
        },
        "pair_policy": {
            "policy_id": "s01_clean_same_pair_grid_v1",
            "family_weights": {"same_nuisance_different_science": 1.0},
            "same_pair_id": True,
            "min_fisher_distance": 0.0,
            "max_fisher_distance": 5000.0,
            "include_reverse": True,
            "max_sampling_attempts": 4000,
        },
        "image_scaling": {"mode": "global_max_abs", "max_samples": 64},
        "noise": {"enabled": False, "apply_to": "observation"},
        "training": {
            "epochs": 6,
            "batch_size": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "optimizer": "adamw",
            "pairs_per_epoch": 64,
            "num_workers": 0,
            "shard_cache_size": 2,
            "early_stopping": {"enabled": False},
        },
        "validation": {
            "pairs_per_slice": 16,
            "split": "validation",
            "eval_slices": {
                "tiny_train_seen": {
                    "science_split": "train",
                    "nuisance_split": "train",
                }
            },
        },
        "evaluate_test": False,
    }


def default_s01_e01_config() -> dict[str, Any]:
    """Return the first clean same-nuisance held-out science baseline preset."""
    return {
        "study_id": "S01",
        "experiment_id": "S01-E01",
        "run_id": "S01-E01-R001",
        "seed": 11,
        "device": "auto",
        "model": {
            "channels": [16, 32, 64, 128],
            "embedding_dim": 128,
            "encoder_hidden_dim": 256,
            "head_hidden_dim": 256,
            "comparator": "concat_diff",
            "normalization": "batch",
            "adaptive_pool_shape": [4, 4],
        },
        "pair_policy": {
            "policy_id": "s01_clean_same_pair_grid_v1",
            "family_weights": {"same_nuisance_different_science": 1.0},
            "same_pair_id": True,
            "min_fisher_distance": 0.0,
            "max_fisher_distance": 5000.0,
            "include_reverse": True,
            "max_sampling_attempts": 4000,
        },
        "image_scaling": {"mode": "global_max_abs", "max_samples": 512},
        "noise": {"enabled": False, "apply_to": "observation"},
        "training": {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.0005,
            "weight_decay": 0.0001,
            "optimizer": "adamw",
            "pairs_per_epoch": 8192,
            "num_workers": 4,
            "shard_cache_size": 4,
            "early_stopping": {
                "enabled": True,
                "monitor": "validation_loss",
                "min_epochs": 20,
                "patience": 12,
                "min_delta_relative": 0.001,
            },
        },
        "validation": {"pairs_per_slice": 512, "split": "validation"},
        "evaluation": {"fisher_distance_bin_edges": list(DEFAULT_S01_FISHER_DISTANCE_BIN_EDGES)},
        "evaluate_test": False,
    }


def load_run_config(path: Path | None, *, preset: str | None = None) -> dict[str, Any]:
    """Load a JSON/YAML run config and merge it over an optional preset."""
    if preset in {None, "", "s01_e01"}:
        config = default_s01_e01_config()
    elif preset == "s01_e00":
        config = default_s01_e00_config()
    else:
        raise ValueError(f"Unknown ML run preset {preset!r}.")
    if path is None:
        return config
    raw_text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith((".yaml", ".yml")):
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("YAML config files require PyYAML.") from exc
        payload = yaml.safe_load(raw_text)
    else:
        payload = json.loads(raw_text)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON/YAML object.")
    return _deep_update(config, payload)


def _mps_is_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    is_available = getattr(backend, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def resolve_device(value: str) -> torch.device:
    """Resolve an ML training device string with CUDA, MPS, then CPU auto-priority."""
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} requested but CUDA is not available.")
    if device.type == "mps" and not _mps_is_available():
        raise RuntimeError(f"MPS device {value!r} requested but Apple MPS is not available.")
    return device


_resolve_device = resolve_device


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _train_indices(catalog: SampleCatalog, split_registry: SplitRegistry) -> np.ndarray:
    return catalog.indices_for_groups(
        science_groups=split_registry.science_groups("train"),
        nuisance_groups=split_registry.nuisance_groups("train"),
    )


def _build_optimizer(model: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    name = str(config.get("optimizer", "adamw")).lower()
    lr = float(config.get("learning_rate", 5.0e-4))
    weight_decay = float(config.get("weight_decay", 1.0e-4))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer {name!r}.")


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: LRSchedulerConfig,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if config.name == "none":
        return None
    if config.name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config.factor),
            patience=int(config.patience),
            threshold=float(config.threshold_relative),
            threshold_mode="rel",
            min_lr=float(config.min_lr),
        )
    if config.name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.t_max or 0),
            eta_min=float(config.min_lr),
        )
    raise ValueError(f"Unsupported lr_scheduler.name {config.name!r}.")


def _optimizer_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    groups = optimizer.param_groups
    if len(groups) != 1:
        raise ValueError("Expected optimizer to contain exactly one parameter group.")
    return float(groups[0]["lr"])


def _step_lr_scheduler(
    scheduler: (
        torch.optim.lr_scheduler.LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
        | None
    ),
    *,
    config: LRSchedulerConfig,
    validation_loss: float,
) -> None:
    if scheduler is None:
        return
    if config.name == "reduce_on_plateau":
        scheduler.step(float(validation_loss))
    elif config.name == "cosine_annealing":
        scheduler.step()
    else:
        raise ValueError(f"Unsupported lr_scheduler.name {config.name!r}.")


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_a = batch["image_a"].to(device)
    image_b = batch["image_b"].to(device)
    target = batch["target_delta_z"].to(device)
    return image_a, image_b, target


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    catalog: SampleCatalog,
    fisher_distance_bin_edges: Sequence[float] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    pair_ids: list[str] = []
    eval_slices: list[str] = []
    pair_families: list[str] = []
    distances: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            image_a, image_b, target = _move_batch(batch, device)
            pred = model(image_a, image_b)
            preds.append(pred.detach().cpu().numpy())
            truths.append(target.detach().cpu().numpy())
            pair_ids.extend(str(v) for v in batch["pair_record_id"])
            eval_slices.extend(str(v) for v in batch["eval_slice"])
            pair_families.extend(str(v) for v in batch["pair_family"])
            distances.extend(float(v) for v in batch["fisher_distance_l2"].cpu().numpy())
    y_pred = np.concatenate(preds, axis=0) if preds else np.zeros((0, catalog.science_dim))
    y_true = np.concatenate(truths, axis=0) if truths else np.zeros((0, catalog.science_dim))
    metrics = compute_regression_metrics(y_pred, y_true, catalog=catalog)
    metrics["by_eval_slice"] = metrics_by_group(y_pred, y_true, eval_slices, catalog=catalog)
    metrics["by_pair_family"] = metrics_by_group(y_pred, y_true, pair_families, catalog=catalog)
    metrics["by_distance_bin"] = _distance_binned_metrics(
        y_pred,
        y_true,
        np.asarray(distances, dtype=np.float64),
        catalog=catalog,
        bin_edges=fisher_distance_bin_edges,
    )
    predictions = {
        "pair_record_id": np.asarray(pair_ids, dtype=str),
        "eval_slice": np.asarray(eval_slices, dtype=str),
        "pair_family": np.asarray(pair_families, dtype=str),
        "fisher_distance_l2": np.asarray(distances, dtype=np.float32),
        "y_pred_z": y_pred.astype(np.float32),
        "y_true_z": y_true.astype(np.float32),
    }
    return metrics, predictions


def _distance_binned_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    distances: np.ndarray,
    *,
    catalog: SampleCatalog,
    bin_edges: Sequence[float] | None = None,
) -> dict[str, Any]:
    edges = validate_fisher_distance_bin_edges(
        DEFAULT_S01_FISHER_DISTANCE_BIN_EDGES if bin_edges is None else bin_edges
    )
    labels = [_distance_bin_label(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
    below = int(np.count_nonzero(distances < edges[0]))
    above = int(np.count_nonzero(distances > edges[-1]))
    bins: dict[str, Any] = {}
    for index, (lo, hi, label) in enumerate(zip(edges[:-1], edges[1:], labels)):
        if index == len(labels) - 1:
            mask = (distances >= lo) & (distances <= hi)
        else:
            mask = (distances >= lo) & (distances < hi)
        if not np.any(mask):
            bins[label] = {"sample_count": 0}
            continue
        bins[label] = compute_regression_metrics(y_pred[mask], y_true[mask], catalog=catalog)
    return {
        "bin_edges": [float(v) for v in edges],
        "below_range_count": below,
        "above_range_count": above,
        "outside_range_count": below + above,
        "bins": bins,
    }


def validate_fisher_distance_bin_edges(edges: Sequence[float]) -> tuple[float, ...]:
    """Validate finite, strictly increasing Fisher-distance bin edges."""
    values = tuple(float(v) for v in edges)
    if len(values) < 2:
        raise ValueError("evaluation.fisher_distance_bin_edges must contain at least two edges.")
    if not all(np.isfinite(v) for v in values):
        raise ValueError("evaluation.fisher_distance_bin_edges must be finite.")
    if any(hi <= lo for lo, hi in zip(values[:-1], values[1:])):
        raise ValueError("evaluation.fisher_distance_bin_edges must be strictly increasing.")
    return values


def _distance_bin_label(lo: float, hi: float) -> str:
    def fmt(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else f"{value:g}"

    return f"{fmt(lo)}-{fmt(hi)}"


def _write_history(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "validation_loss",
        "validation_overall_rmse",
        "epoch_seconds",
        "is_best",
        "early_stopping_bad_epochs",
        "learning_rate",
        "learning_rate_next",
        "lr_reduced",
    ]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    tmp_path.replace(path)


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _latest_history_epoch(rows: Sequence[Mapping[str, Any]]) -> int | None:
    if not rows:
        return None
    epochs = [int(row["epoch"]) for row in rows if row.get("epoch") not in (None, "")]
    return None if not epochs else max(epochs)


def _optimization_summary(
    *,
    history: Sequence[Mapping[str, Any]],
    initial_learning_rate: float,
    final_learning_rate: float,
    lr_scheduler_config: LRSchedulerConfig,
    early_stopping_state: EarlyStoppingState,
    max_epochs: int,
) -> dict[str, Any]:
    reduction_epochs = [
        int(row["epoch"])
        for row in history
        if str(row.get("lr_reduced", "")).lower() == "true" or row.get("lr_reduced") is True
    ]
    reached_max_epochs = (
        not bool(early_stopping_state.early_stopped)
        and int(early_stopping_state.epochs_completed) >= int(max_epochs)
    )
    return {
        "initial_learning_rate": float(initial_learning_rate),
        "final_learning_rate": float(final_learning_rate),
        "lr_scheduler": lr_scheduler_config.to_dict(),
        "lr_reduction_count": len(reduction_epochs),
        "lr_reduction_epochs": reduction_epochs,
        "epochs_completed": int(early_stopping_state.epochs_completed),
        "best_epoch": int(early_stopping_state.best_epoch),
        "early_stopped": bool(early_stopping_state.early_stopped),
        "stop_epoch": early_stopping_state.stop_epoch,
        "reached_max_epochs": reached_max_epochs,
    }


def _training_identity(
    *,
    catalog: SampleCatalog,
    split_registry: SplitRegistry,
    pair_policy: PairPolicy,
    validation_manifest: PairManifest,
) -> dict[str, Any]:
    return {
        "prepared_dataset": {
            "artifact_id": catalog.artifact_id,
            "prepared_dataset_hash": catalog.prepared_dataset_hash,
        },
        "split_registry": {
            "artifact_id": split_registry.artifact_id,
            "content_sha256": split_registry_content_sha256(split_registry),
        },
        "pair_policy": pair_policy.to_dict(),
        "validation_manifest_identity": validation_manifest.manifest.get("content_identity"),
    }


def _checkpoint_metadata_value(value: Any) -> Any:
    """Normalize checkpoint metadata to PyTorch weights-only-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _checkpoint_metadata_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_checkpoint_metadata_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_checkpoint_metadata_value(item) for item in value)
    raise TypeError(
        "Checkpoint metadata contains a non weights-only-safe value "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def _checkpoint_metadata_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _checkpoint_metadata_value(value) for key, value in payload.items()}


def _load_training_checkpoint(path: Path, *, map_location: torch.device | str) -> Mapping[str, Any]:
    return torch.load(str(path), map_location=map_location, weights_only=True)


def _validate_resume_identity(
    *,
    checkpoint: Mapping[str, Any],
    current_identity: Mapping[str, Any],
) -> None:
    previous = checkpoint.get("training_identity")
    if not isinstance(previous, Mapping):
        warnings.warn(
            "Resume checkpoint does not contain training_identity; continuing without "
            "strict scientific identity validation for this older checkpoint.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if dict(previous) != dict(current_identity):
        raise ValueError(
            "Resume checkpoint scientific identity does not match the current run inputs: "
            f"expected={previous}, actual={current_identity}."
        )


def _validate_resume_lr_scheduler(
    *,
    checkpoint: Mapping[str, Any],
    current_config: LRSchedulerConfig,
    current_initial_learning_rate: float,
) -> None:
    previous_raw = checkpoint.get("lr_scheduler")
    previous = previous_raw if isinstance(previous_raw, Mapping) else {"name": "none"}
    previous_name = str(previous.get("name", "none"))
    current = current_config.to_dict()
    if previous_name != current_config.name:
        raise ValueError(
            "Resume checkpoint lr_scheduler does not match the current run "
            f"({previous_name!r} != {current_config.name!r})."
        )
    if current_config.is_active:
        checkpoint_config = checkpoint.get("config")
        checkpoint_training = (
            checkpoint_config.get("training")
            if isinstance(checkpoint_config, Mapping)
            else None
        )
        if not isinstance(checkpoint_training, Mapping) or "learning_rate" not in checkpoint_training:
            raise ValueError(
                "Resume checkpoint is missing config.training.learning_rate for "
                "an active scheduler."
            )
        previous_learning_rate = float(checkpoint_training["learning_rate"])
        if previous_learning_rate != float(current_initial_learning_rate):
            raise ValueError(
                "Resume checkpoint training.learning_rate does not match the "
                "current active scheduled run "
                f"({previous_learning_rate} != {float(current_initial_learning_rate)})."
            )
        if dict(previous) != current:
            raise ValueError(
                "Resume checkpoint lr_scheduler configuration does not match "
                f"the current run: expected={previous}, actual={current}."
            )
        if "lr_scheduler_state_dict" not in checkpoint:
            raise ValueError(
                "Resume checkpoint is missing lr_scheduler_state_dict for an "
                "active scheduler."
            )
        if not isinstance(checkpoint["lr_scheduler_state_dict"], Mapping):
            raise ValueError(
                "Resume checkpoint lr_scheduler_state_dict is not compatible "
                "with an active scheduler."
            )
    elif "lr_scheduler_state_dict" in checkpoint and isinstance(
        checkpoint.get("lr_scheduler_state_dict"), Mapping
    ):
        raise ValueError(
            "Resume checkpoint contains active scheduler state but the current "
            "run config requests lr_scheduler.name='none'."
        )


def _prepare_validation_manifest(
    *,
    catalog: SampleCatalog,
    split_registry: SplitRegistry,
    policy: PairPolicy,
    config: Mapping[str, Any],
    output_dir: Path,
    explicit_path: Path | None,
) -> PairManifest:
    if explicit_path is not None:
        return load_pair_manifest(explicit_path, catalog=catalog, split_registry=split_registry)
    validation_cfg = dict(config.get("validation", {}))
    outdir = output_dir / "generated_validation_pairs"
    if config.get("resume_checkpoint") and outdir.exists():
        return load_pair_manifest(outdir, catalog=catalog, split_registry=split_registry)
    manifest = generate_frozen_pair_manifest(
        catalog,
        split_registry,
        policy=policy,
        artifact_id="PAIR-EVAL-v1",
        split=str(validation_cfg.get("split", "validation")),
        seed=int(config.get("seed", 0)) + 911,
        pairs_per_slice=int(validation_cfg.get("pairs_per_slice", 256)),
        eval_slices=validation_cfg.get("eval_slices"),
    )
    write_pair_manifest(outdir, manifest)
    return manifest


def train_pairwise_correction(
    *,
    config: Mapping[str, Any],
    prepared_root: Path,
    split_registry_path: Path,
    output_dir: Path,
    validation_manifest_path: Path | None = None,
    test_manifest_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run a noninteractive pairwise-correction training job."""
    output_dir = Path(output_dir).resolve()
    resume_checkpoint = config.get("resume_checkpoint")
    resume_checkpoint_path = (
        None if not resume_checkpoint else Path(str(resume_checkpoint)).resolve()
    )
    if resume_checkpoint_path is not None and overwrite:
        raise ValueError(
            "Resume cannot be combined with overwrite; resume continues the existing run directory."
        )
    if resume_checkpoint_path is not None and resume_checkpoint_path.parent != output_dir:
        raise ValueError(
            "Resume continues the same logical run directory; --resume-checkpoint must be "
            f"inside output_dir ({resume_checkpoint_path.parent} != {output_dir})."
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite and not resume_checkpoint_path:
            raise FileExistsError(f"{output_dir} exists and is non-empty; pass overwrite=True.")
        if overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    seed = int(config.get("seed", 0))
    _set_seed(seed)
    device = _resolve_device(str(config.get("device", "auto")))
    runtime = _runtime_info(device)

    catalog = load_sample_catalog(prepared_root)
    split_registry = load_split_registry(split_registry_path, catalog=catalog)
    pair_policy = PairPolicy.from_dict(config.get("pair_policy", {}))
    noise_config = NoiseConfig.from_dict(config.get("noise"))
    train_indices = _train_indices(catalog, split_registry)
    scaling_cfg = dict(config.get("image_scaling", {}))
    scaler = fit_intensity_scaler(
        catalog,
        train_indices,
        mode=str(scaling_cfg.get("mode", "global_max_abs")),
        max_samples=scaling_cfg.get("max_samples", 512),
        cache_size=int(config.get("training", {}).get("shard_cache_size", 4)),
    )
    validation_manifest = _prepare_validation_manifest(
        catalog=catalog,
        split_registry=split_registry,
        policy=pair_policy,
        config=config,
        output_dir=output_dir,
        explicit_path=validation_manifest_path,
    )

    training_cfg = dict(config.get("training", {}))
    epochs = int(training_cfg.get("epochs", 25))
    initial_learning_rate = float(training_cfg.get("learning_rate", 5.0e-4))
    lr_scheduler_config = LRSchedulerConfig.from_dict(
        training_cfg.get("lr_scheduler"),
        initial_learning_rate=initial_learning_rate,
        max_epochs=epochs,
    )
    training_cfg["lr_scheduler"] = lr_scheduler_config.to_dict()
    config["training"] = training_cfg

    model = build_pairwise_correction_model(catalog.science_dim, config.get("model")).to(device)
    criterion = nn.MSELoss()
    optimizer = _build_optimizer(model, training_cfg)
    lr_scheduler = _build_lr_scheduler(optimizer, lr_scheduler_config)
    start_epoch = 0
    checkpoint = None
    current_identity = _training_identity(
        catalog=catalog,
        split_registry=split_registry,
        pair_policy=pair_policy,
        validation_manifest=validation_manifest,
    )
    if resume_checkpoint_path is not None:
        checkpoint = _load_training_checkpoint(resume_checkpoint_path, map_location=device)
        _validate_resume_identity(checkpoint=checkpoint, current_identity=current_identity)
        _validate_resume_lr_scheduler(
            checkpoint=checkpoint,
            current_config=lr_scheduler_config,
            current_initial_learning_rate=initial_learning_rate,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1

    batch_size = int(training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 0))
    shard_cache_size = int(training_cfg.get("shard_cache_size", 4))
    early_stopping_config = EarlyStoppingConfig.from_dict(training_cfg.get("early_stopping"))
    sampler = PairSampler(catalog, split_registry, pair_policy)
    train_dataset = DynamicPairDataset(
        catalog=catalog,
        sampler=sampler,
        pairs_per_epoch=int(training_cfg.get("pairs_per_epoch", 2048)),
        seed=seed,
        science_split="train",
        nuisance_split="train",
        scaler=scaler,
        noise_config=noise_config,
        shard_cache_size=shard_cache_size,
    )
    val_dataset = PairManifestDataset(
        catalog=catalog,
        pair_manifest=validation_manifest,
        scaler=scaler,
        noise_config=NoiseConfig(enabled=False),
        shard_cache_size=shard_cache_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    training_pair_stream = {
        "pairs_per_epoch_ordered": len(train_dataset),
        "include_reverse": bool(pair_policy.include_reverse),
        "reverse_pair_augmentation": bool(pair_policy.include_reverse),
        "base_pairs_per_epoch": len(train_dataset) // 2
        if pair_policy.include_reverse
        else len(train_dataset),
    }
    resolved_config = _deep_update(
        dict(config),
        {
            "image_scaling_resolved": scaler.to_dict(),
            "training_pair_stream": training_pair_stream,
            "runtime": runtime,
        },
    )
    write_json(output_dir / "run_config_resolved.json", resolved_config)
    run_manifest = {
        "schema_version": "dluxshera_ml_run_manifest/1",
        "study_id": config.get("study_id"),
        "experiment_id": config.get("experiment_id"),
        "run_id": config.get("run_id"),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "prepared_dataset": catalog.summary(),
        "split_registry": {
            "artifact_id": split_registry.artifact_id,
            "content_sha256": split_registry_content_sha256(split_registry),
            "prepared_dataset_hash": split_registry.prepared_dataset.get(
                "prepared_dataset_hash"
            ),
            "counts": split_registry.counts,
            "science_group_policy": split_registry.science_group_policy,
            "nuisance_group_policy": split_registry.nuisance_group_policy,
        },
        "pair_eval_manifest": validation_manifest.manifest,
        "validation_manifest_identity": validation_manifest.manifest.get("content_identity"),
        "training_identity": current_identity,
        "pair_policy": pair_policy.to_dict(),
        "noise": noise_config.to_dict(),
        "image_scaling": scaler.to_dict(),
        "model": {
            **dict(config.get("model", {})),
            "parameter_count": count_parameters(model),
        },
        "training": training_cfg,
        "early_stopping": early_stopping_config.to_dict(),
        "training_pair_stream": training_pair_stream,
        "git": _git_info(),
        "runtime": runtime,
        "test_evaluated": False,
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    early_stopping_state = EarlyStoppingState()
    if checkpoint is not None:
        early_stopping_state = EarlyStoppingState.from_checkpoint(checkpoint)
    history: list[dict[str, Any]] = []
    if resume_checkpoint_path is not None:
        history = _read_history(output_dir / "history.csv")
        latest_epoch = _latest_history_epoch(history)
        expected_latest = start_epoch - 1
        if latest_epoch is None:
            warnings.warn(
                "Resuming without existing history.csv; new history will contain only "
                "epochs completed after resume.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif latest_epoch != expected_latest:
            raise ValueError(
                "Existing history.csv is inconsistent with resume checkpoint "
                f"(latest epoch {latest_epoch} != checkpoint epoch {expected_latest})."
            )
    evaluation_cfg = dict(config.get("evaluation", {}))
    fisher_distance_bin_edges = validate_fisher_distance_bin_edges(
        evaluation_cfg.get("fisher_distance_bin_edges", DEFAULT_S01_FISHER_DISTANCE_BIN_EDGES)
    )
    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        learning_rate = _optimizer_learning_rate(optimizer)
        model.train()
        train_dataset.set_epoch(epoch)
        running = 0.0
        seen = 0
        for batch in train_loader:
            image_a, image_b, target = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(image_a, image_b)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            batch_n = int(target.shape[0])
            running += float(loss.detach().cpu()) * batch_n
            seen += batch_n
        train_loss = running / max(seen, 1)
        val_metrics, val_predictions = _evaluate(
            model,
            val_loader,
            device=device,
            catalog=catalog,
            fisher_distance_bin_edges=fisher_distance_bin_edges,
        )
        val_loss = float(val_metrics["fisher_overall_rmse"]) ** 2
        is_best, should_stop = early_stopping_state.update(
            epoch=epoch,
            metric=val_loss,
            config=early_stopping_config,
        )
        _step_lr_scheduler(
            lr_scheduler,
            config=lr_scheduler_config,
            validation_loss=val_loss,
        )
        learning_rate_next = _optimizer_learning_rate(optimizer)
        lr_reduced = bool(learning_rate_next < learning_rate)
        epoch_seconds = time.perf_counter() - epoch_started
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_overall_rmse": val_metrics["fisher_overall_rmse"],
                "epoch_seconds": epoch_seconds,
                "is_best": is_best,
                "early_stopping_bad_epochs": early_stopping_state.bad_epochs,
                "learning_rate": learning_rate,
                "learning_rate_next": learning_rate_next,
                "lr_reduced": lr_reduced,
            }
        )
        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": _checkpoint_metadata_mapping(resolved_config),
            "image_scaling": _checkpoint_metadata_mapping(scaler.to_dict()),
            "validation_metrics": _checkpoint_metadata_mapping(val_metrics),
            "best_validation_loss": early_stopping_state.absolute_best_loss,
            "best_epoch": early_stopping_state.best_epoch,
            "early_stopping": _checkpoint_metadata_mapping(early_stopping_config.to_dict()),
            "early_stopping_state": _checkpoint_metadata_mapping(early_stopping_state.to_dict()),
            "lr_scheduler": _checkpoint_metadata_mapping(lr_scheduler_config.to_dict()),
            "training_identity": _checkpoint_metadata_mapping(current_identity),
            "runtime_provenance": _checkpoint_metadata_mapping(runtime),
        }
        if lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = _checkpoint_metadata_mapping(
                lr_scheduler.state_dict()
            )
        torch.save(checkpoint, output_dir / "checkpoint_last.pt")
        if is_best:
            torch.save(checkpoint, output_dir / "checkpoint_best.pt")
            np.savez(output_dir / "evaluation_predictions.npz", **val_predictions)
            write_json(
                output_dir / "metrics.json",
                {"schema_version": "dluxshera_ml_metrics/2", "validation": val_metrics},
            )
        _write_history(output_dir / "history.csv", history)
        print(
            "epoch "
            f"{epoch + 1:03d}/{epochs:03d} "
            f"lr={learning_rate:.2e}"
            f"{f'->{learning_rate_next:.2e}' if lr_reduced else ''} "
            f"train={train_loss:.6g} "
            f"val={val_loss:.6g} "
            f"best={early_stopping_state.absolute_best_loss:.6g} "
            f"patience={early_stopping_state.bad_epochs}/{early_stopping_config.patience} "
            f"time={epoch_seconds:.1f}s",
            flush=True,
        )
        if should_stop:
            break

    final_metrics = read_json(output_dir / "metrics.json")
    final_learning_rate = _optimizer_learning_rate(optimizer)
    optimization = _optimization_summary(
        history=history,
        initial_learning_rate=initial_learning_rate,
        final_learning_rate=final_learning_rate,
        lr_scheduler_config=lr_scheduler_config,
        early_stopping_state=early_stopping_state,
        max_epochs=epochs,
    )
    final_metrics["schema_version"] = "dluxshera_ml_metrics/2"
    final_metrics["best_epoch"] = early_stopping_state.best_epoch
    final_metrics["best_validation_loss"] = early_stopping_state.absolute_best_loss
    final_metrics["early_stopping"] = {
        **early_stopping_config.to_dict(),
        "early_stopped": bool(early_stopping_state.early_stopped),
        "stop_epoch": early_stopping_state.stop_epoch,
        "epochs_completed": int(early_stopping_state.epochs_completed),
        "final_bad_epoch_count": int(early_stopping_state.bad_epochs),
        "reached_max_epochs": not early_stopping_state.early_stopped
        and early_stopping_state.epochs_completed >= epochs,
    }
    final_metrics["optimization"] = optimization

    if bool(config.get("evaluate_test", False)):
        if test_manifest_path is None:
            raise ValueError("evaluate_test=true requires a test_manifest_path.")
        test_manifest = load_pair_manifest(
            test_manifest_path,
            catalog=catalog,
            split_registry=split_registry,
        )
        test_dataset = PairManifestDataset(
            catalog=catalog,
            pair_manifest=test_manifest,
            scaler=scaler,
            noise_config=NoiseConfig(enabled=False),
            shard_cache_size=shard_cache_size,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=False,
        )
        checkpoint = _load_training_checkpoint(output_dir / "checkpoint_best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics, test_predictions = _evaluate(model, test_loader, device=device, catalog=catalog)
        final_metrics["test"] = test_metrics
        np.savez(output_dir / "test_predictions.npz", **test_predictions)
        run_manifest["test_evaluated"] = True
        run_manifest["test_pair_manifest"] = test_manifest.manifest
        run_manifest["test_manifest_identity"] = test_manifest.manifest.get("content_identity")
    write_json(output_dir / "metrics.json", final_metrics)
    run_manifest["best_epoch"] = early_stopping_state.best_epoch
    run_manifest["best_validation_loss"] = early_stopping_state.absolute_best_loss
    run_manifest["early_stopping"].update(final_metrics["early_stopping"])
    run_manifest["optimization"] = optimization
    write_json(output_dir / "run_manifest.json", run_manifest)
    return {
        "output_dir": str(output_dir),
        "run_id": config.get("run_id"),
        "best_epoch": early_stopping_state.best_epoch,
        "best_validation_loss": early_stopping_state.absolute_best_loss,
        "early_stopped": bool(early_stopping_state.early_stopped),
        "epochs_completed": int(early_stopping_state.epochs_completed),
        "metrics_path": str(output_dir / "metrics.json"),
        "checkpoint_best": str(output_dir / "checkpoint_best.pt"),
    }
