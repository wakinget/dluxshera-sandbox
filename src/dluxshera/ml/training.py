from __future__ import annotations

import csv
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

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
from .pairs import PairManifest, PairPolicy, PairSampler, generate_frozen_pair_manifest, load_pair_manifest, write_pair_manifest
from .scaling import IntensityScaler, fit_intensity_scaler
from .splits import SplitRegistry, load_split_registry
from .torch_data import DynamicPairDataset, PairManifestDataset

__all__ = [
    "default_s01_e00_config",
    "default_s01_e01_config",
    "load_run_config",
    "resolve_device",
    "train_pairwise_correction",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_info() -> dict[str, Any]:
    root = _repo_root()
    info: dict[str, Any] = {}
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
    return info


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
            "policy_id": "s01_e00_tiny_same_nuisance_v1",
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
            "policy_id": "s01_e01_clean_same_pair_grid_v1",
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
            "epochs": 25,
            "batch_size": 32,
            "learning_rate": 0.0005,
            "weight_decay": 0.0001,
            "optimizer": "adamw",
            "pairs_per_epoch": 2048,
            "num_workers": 0,
            "shard_cache_size": 4,
        },
        "validation": {"pairs_per_slice": 512, "split": "validation"},
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
) -> dict[str, Any]:
    if distances.size == 0:
        return {}
    edges = [0.0, 1.0, 2.0, 4.0, np.inf]
    labels = ["0-1", "1-2", "2-4", "4-inf"]
    out: dict[str, Any] = {}
    for lo, hi, label in zip(edges[:-1], edges[1:], labels):
        mask = (distances >= lo) & (distances < hi)
        if not np.any(mask):
            continue
        out[label] = compute_regression_metrics(y_pred[mask], y_true[mask], catalog=catalog)
    return out


def _write_history(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "validation_loss", "validation_overall_rmse"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


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
    outdir = output_dir / "generated_validation_pairs"
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
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists and is non-empty; pass overwrite=True.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    seed = int(config.get("seed", 0))
    _set_seed(seed)
    device = _resolve_device(str(config.get("device", "auto")))

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

    model = build_pairwise_correction_model(catalog.science_dim, config.get("model")).to(device)
    criterion = nn.MSELoss()
    optimizer = _build_optimizer(model, config.get("training", {}))
    resume_checkpoint = config.get("resume_checkpoint")
    start_epoch = 0
    if resume_checkpoint:
        checkpoint = torch.load(str(resume_checkpoint), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1

    training_cfg = dict(config.get("training", {}))
    batch_size = int(training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 0))
    shard_cache_size = int(training_cfg.get("shard_cache_size", 4))
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
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
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
            "prepared_dataset_hash": split_registry.prepared_dataset.get(
                "prepared_dataset_hash"
            ),
            "counts": split_registry.counts,
            "science_group_policy": split_registry.science_group_policy,
            "nuisance_group_policy": split_registry.nuisance_group_policy,
        },
        "pair_eval_manifest": validation_manifest.manifest,
        "pair_policy": pair_policy.to_dict(),
        "noise": noise_config.to_dict(),
        "image_scaling": scaler.to_dict(),
        "model": {
            **dict(config.get("model", {})),
            "parameter_count": count_parameters(model),
        },
        "training": training_cfg,
        "training_pair_stream": training_pair_stream,
        "git": _git_info(),
        "test_evaluated": False,
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    best_val = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    epochs = int(training_cfg.get("epochs", 25))
    for epoch in range(start_epoch, epochs):
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
        val_metrics, val_predictions = _evaluate(model, val_loader, device=device, catalog=catalog)
        val_loss = float(val_metrics["fisher_overall_rmse"]) ** 2
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_overall_rmse": val_metrics["fisher_overall_rmse"],
            }
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": resolved_config,
            "image_scaling": scaler.to_dict(),
            "validation_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "checkpoint_last.pt")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(checkpoint, output_dir / "checkpoint_best.pt")
            np.savez(output_dir / "evaluation_predictions.npz", **val_predictions)
            write_json(output_dir / "metrics.json", {"validation": val_metrics})
        _write_history(output_dir / "history.csv", history)

    final_metrics = read_json(output_dir / "metrics.json")
    final_metrics["best_epoch"] = best_epoch
    final_metrics["best_validation_loss"] = best_val

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
        )
        checkpoint = torch.load(output_dir / "checkpoint_best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics, test_predictions = _evaluate(model, test_loader, device=device, catalog=catalog)
        final_metrics["test"] = test_metrics
        np.savez(output_dir / "test_predictions.npz", **test_predictions)
        run_manifest["test_evaluated"] = True
        run_manifest["test_pair_manifest"] = test_manifest.manifest
    write_json(output_dir / "metrics.json", final_metrics)
    run_manifest["best_epoch"] = best_epoch
    run_manifest["best_validation_loss"] = best_val
    write_json(output_dir / "run_manifest.json", run_manifest)
    return {
        "output_dir": str(output_dir),
        "run_id": config.get("run_id"),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "metrics_path": str(output_dir / "metrics.json"),
        "checkpoint_best": str(output_dir / "checkpoint_best.pt"),
    }
