from __future__ import annotations

import copy
import csv
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

import dluxshera.ml.training as training_module
from dluxshera.datasets.schema import read_json
from dluxshera.ml.training import LRSchedulerConfig, train_pairwise_correction
from tests.ml.test_models_training import _fake_evaluate_losses, _tiny_training_inputs


def _optimizer(lr: float = 1.0e-3) -> torch.optim.Optimizer:
    model = torch.nn.Linear(1, 1)
    return torch.optim.AdamW(model.parameters(), lr=lr)


def _history(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_lr_scheduler_omitted_and_none_preserve_fixed_lr() -> None:
    omitted = LRSchedulerConfig.from_dict(None, initial_learning_rate=1.0e-3)
    explicit = LRSchedulerConfig.from_dict(
        {"name": "none"},
        initial_learning_rate=1.0e-3,
    )
    assert omitted.to_dict() == {"name": "none"}
    assert explicit.to_dict() == {"name": "none"}
    assert training_module._build_lr_scheduler(_optimizer(), omitted) is None
    assert training_module._build_lr_scheduler(_optimizer(), explicit) is None


def test_reduce_on_plateau_config_constructs_scheduler() -> None:
    config = LRSchedulerConfig.from_dict(
        {
            "name": "reduce_on_plateau",
            "monitor": "validation_loss",
            "factor": 0.3,
            "patience": 8,
            "threshold_relative": 0.001,
            "min_lr": 1.0e-6,
        },
        initial_learning_rate=5.0e-4,
    )
    scheduler = training_module._build_lr_scheduler(_optimizer(lr=5.0e-4), config)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert config.to_dict()["factor"] == pytest.approx(0.3)


def test_cosine_config_constructs_scheduler() -> None:
    config = LRSchedulerConfig.from_dict(
        {"name": "cosine_annealing", "t_max": 300, "min_lr": 1.0e-6},
        initial_learning_rate=5.0e-4,
    )
    scheduler = training_module._build_lr_scheduler(_optimizer(lr=5.0e-4), config)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert config.to_dict()["t_max"] == 300


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"name": "one_cycle"}, "Unsupported lr_scheduler.name"),
        (
            {
                "name": "reduce_on_plateau",
                "monitor": "training_loss",
                "factor": 0.3,
                "patience": 1,
                "threshold_relative": 0.0,
                "min_lr": 0.0,
            },
            "monitor='validation_loss'",
        ),
        (
            {
                "name": "reduce_on_plateau",
                "factor": 1.0,
                "patience": 1,
                "threshold_relative": 0.0,
                "min_lr": 0.0,
            },
            "factor",
        ),
        (
            {
                "name": "reduce_on_plateau",
                "factor": 0.5,
                "patience": -1,
                "threshold_relative": 0.0,
                "min_lr": 0.0,
            },
            "patience",
        ),
        (
            {
                "name": "reduce_on_plateau",
                "factor": 0.5,
                "patience": 1,
                "threshold_relative": -0.1,
                "min_lr": 0.0,
            },
            "threshold_relative",
        ),
        (
            {
                "name": "reduce_on_plateau",
                "factor": 0.5,
                "patience": 1,
                "threshold_relative": 0.0,
                "min_lr": -1.0e-6,
            },
            "min_lr",
        ),
        ({"name": "cosine_annealing", "t_max": 0, "min_lr": 0.0}, "t_max"),
        ({"name": "cosine_annealing", "t_max": 10, "min_lr": 2.0e-3}, "min_lr"),
        (
            {"name": "cosine_annealing", "t_max": 3, "min_lr": 0.0, "max_epochs": 4},
            "training.epochs",
        ),
    ],
)
def test_invalid_lr_scheduler_configs_fail_clearly(
    payload: dict[str, object],
    message: str,
) -> None:
    max_epochs = payload.pop("max_epochs", None)
    with pytest.raises(ValueError, match=message):
        LRSchedulerConfig.from_dict(
            payload,
            initial_learning_rate=1.0e-3,
            max_epochs=None if max_epochs is None else int(max_epochs),
        )


def test_step_lr_scheduler_passes_validation_loss_to_plateau() -> None:
    class Recorder:
        values: list[float]

        def __init__(self) -> None:
            self.values = []

        def step(self, value: float) -> None:
            self.values.append(float(value))

    recorder = Recorder()
    config = LRSchedulerConfig.from_dict(
        {
            "name": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 0,
            "threshold_relative": 0.0,
            "min_lr": 0.0,
        },
        initial_learning_rate=1.0e-3,
    )
    training_module._step_lr_scheduler(recorder, config=config, validation_loss=12.5)
    assert recorder.values == [12.5]


def test_scheduler_epoch_steps_update_lr_as_expected() -> None:
    plateau_optimizer = _optimizer()
    plateau_config = LRSchedulerConfig.from_dict(
        {
            "name": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 0,
            "threshold_relative": 0.0,
            "min_lr": 1.0e-6,
        },
        initial_learning_rate=1.0e-3,
    )
    plateau = training_module._build_lr_scheduler(plateau_optimizer, plateau_config)
    training_module._step_lr_scheduler(plateau, config=plateau_config, validation_loss=1.0)
    assert training_module._optimizer_learning_rate(plateau_optimizer) == pytest.approx(1.0e-3)
    training_module._step_lr_scheduler(plateau, config=plateau_config, validation_loss=1.1)
    assert training_module._optimizer_learning_rate(plateau_optimizer) == pytest.approx(5.0e-4)

    cosine_optimizer = _optimizer()
    cosine_config = LRSchedulerConfig.from_dict(
        {"name": "cosine_annealing", "t_max": 4, "min_lr": 0.0},
        initial_learning_rate=1.0e-3,
    )
    cosine = training_module._build_lr_scheduler(cosine_optimizer, cosine_config)
    before = training_module._optimizer_learning_rate(cosine_optimizer)
    cosine_optimizer.zero_grad(set_to_none=True)
    next(iter(cosine_optimizer.param_groups[0]["params"])).sum().backward()
    cosine_optimizer.step()
    training_module._step_lr_scheduler(cosine, config=cosine_config, validation_loss=1.0)
    after = training_module._optimizer_learning_rate(cosine_optimizer)
    assert after < before

    fixed_optimizer = _optimizer()
    fixed_config = LRSchedulerConfig.from_dict(None, initial_learning_rate=1.0e-3)
    training_module._step_lr_scheduler(None, config=fixed_config, validation_loss=1.0)
    assert training_module._optimizer_learning_rate(fixed_optimizer) == pytest.approx(1.0e-3)


def test_history_contains_lr_semantics_for_plateau_scheduler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, split_path, config = _tiny_training_inputs(tmp_path)
    config["training"]["epochs"] = 3
    config["training"]["learning_rate"] = 1.0e-3
    config["training"]["lr_scheduler"] = {
        "name": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 0,
        "threshold_relative": 0.0,
        "min_lr": 1.0e-6,
    }
    _fake_evaluate_losses(monkeypatch, [1.0, 1.1, 1.2])

    train_pairwise_correction(
        config=config,
        prepared_root=prepared,
        split_registry_path=split_path,
        output_dir=tmp_path / "run",
    )

    rows = _history(tmp_path / "run" / "history.csv")
    assert float(rows[0]["learning_rate"]) == pytest.approx(1.0e-3)
    assert float(rows[0]["learning_rate_next"]) == pytest.approx(1.0e-3)
    assert rows[0]["lr_reduced"] == "False"
    assert float(rows[1]["learning_rate"]) == pytest.approx(1.0e-3)
    assert float(rows[1]["learning_rate_next"]) == pytest.approx(5.0e-4)
    assert rows[1]["lr_reduced"] == "True"
    assert float(rows[2]["learning_rate"]) == pytest.approx(5.0e-4)
    assert float(rows[2]["learning_rate_next"]) == pytest.approx(2.5e-4)
    assert rows[2]["lr_reduced"] == "True"


def test_scheduled_checkpoint_and_resume_preserve_scheduler_progression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, split_path, base_config = _tiny_training_inputs(tmp_path)
    base_config["training"]["learning_rate"] = 1.0e-3
    base_config["training"]["lr_scheduler"] = {
        "name": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 0,
        "threshold_relative": 0.0,
        "min_lr": 1.0e-6,
    }

    uninterrupted = copy.deepcopy(base_config)
    uninterrupted["training"]["epochs"] = 3
    _fake_evaluate_losses(monkeypatch, [1.0, 1.1, 1.2])
    train_pairwise_correction(
        config=uninterrupted,
        prepared_root=prepared,
        split_registry_path=split_path,
        output_dir=tmp_path / "uninterrupted",
    )

    segmented = copy.deepcopy(base_config)
    segmented["training"]["epochs"] = 2
    _fake_evaluate_losses(monkeypatch, [1.0, 1.1])
    train_pairwise_correction(
        config=segmented,
        prepared_root=prepared,
        split_registry_path=split_path,
        output_dir=tmp_path / "segmented",
    )
    checkpoint = torch.load(
        tmp_path / "segmented" / "checkpoint_last.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["lr_scheduler"] == {
        "name": "reduce_on_plateau",
        "monitor": "validation_loss",
        "factor": 0.5,
        "patience": 0,
        "threshold_relative": 0.0,
        "min_lr": 1.0e-6,
    }
    assert "lr_scheduler_state_dict" in checkpoint

    resumed = copy.deepcopy(base_config)
    resumed["training"]["epochs"] = 3
    resumed["resume_checkpoint"] = str(tmp_path / "segmented" / "checkpoint_last.pt")
    _fake_evaluate_losses(monkeypatch, [1.2])
    summary = train_pairwise_correction(
        config=resumed,
        prepared_root=prepared,
        split_registry_path=split_path,
        output_dir=tmp_path / "segmented",
    )
    assert summary["epochs_completed"] == 3

    segmented_rows = _history(tmp_path / "segmented" / "history.csv")
    uninterrupted_rows = _history(tmp_path / "uninterrupted" / "history.csv")
    assert float(segmented_rows[-1]["learning_rate_next"]) == pytest.approx(
        float(uninterrupted_rows[-1]["learning_rate_next"])
    )


def test_scheduled_resume_with_changed_initial_lr_fails_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, split_path, config = _tiny_training_inputs(tmp_path)
    config["training"]["epochs"] = 1
    config["training"]["learning_rate"] = 1.0e-3
    config["training"]["lr_scheduler"] = {
        "name": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 0,
        "threshold_relative": 0.0,
        "min_lr": 1.0e-6,
    }
    _fake_evaluate_losses(monkeypatch, [1.0])
    train_pairwise_correction(
        config=config,
        prepared_root=prepared,
        split_registry_path=split_path,
        output_dir=tmp_path / "run",
    )

    resume_config = copy.deepcopy(config)
    resume_config["training"]["epochs"] = 2
    resume_config["training"]["learning_rate"] = 5.0e-4
    resume_config["resume_checkpoint"] = str(tmp_path / "run" / "checkpoint_last.pt")
    with pytest.raises(ValueError, match="training.learning_rate"):
        train_pairwise_correction(
            config=resume_config,
            prepared_root=prepared,
            split_registry_path=split_path,
            output_dir=tmp_path / "run",
        )


def test_cosine_run_and_resume_reject_epochs_beyond_t_max(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, split_path, config = _tiny_training_inputs(tmp_path)
    config["training"]["lr_scheduler"] = {
        "name": "cosine_annealing",
        "t_max": 3,
        "min_lr": 1.0e-6,
    }

    invalid_config = copy.deepcopy(config)
    invalid_config["training"]["epochs"] = 4
    with pytest.raises(ValueError, match="training.epochs.*t_max"):
        train_pairwise_correction(
            config=invalid_config,
            prepared_root=prepared,
            split_registry_path=split_path,
            output_dir=tmp_path / "invalid_run",
        )

    valid_config = copy.deepcopy(config)
    valid_config["training"]["epochs"] = 2
    _fake_evaluate_losses(monkeypatch, [1.0, 0.9])
    train_pairwise_correction(
        config=valid_config,
        prepared_root=prepared,
        split_registry_path=split_path,
        output_dir=tmp_path / "run",
    )

    resume_config = copy.deepcopy(config)
    resume_config["training"]["epochs"] = 4
    resume_config["resume_checkpoint"] = str(tmp_path / "run" / "checkpoint_last.pt")
    with pytest.raises(ValueError, match="training.epochs.*t_max"):
        train_pairwise_correction(
            config=resume_config,
            prepared_root=prepared,
            split_registry_path=split_path,
            output_dir=tmp_path / "run",
        )


def test_legacy_fixed_lr_checkpoint_without_scheduler_state_remains_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, split_path, config = _tiny_training_inputs(tmp_path)
    config["training"]["epochs"] = 1
    _fake_evaluate_losses(monkeypatch, [1.0])
    train_pairwise_correction(
        config=config,
        prepared_root=prepared,
        split_registry_path=split_path,
        output_dir=tmp_path / "run",
    )
    checkpoint_path = tmp_path / "run" / "checkpoint_last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint.pop("lr_scheduler", None)
    checkpoint.pop("lr_scheduler_state_dict", None)
    torch.save(checkpoint, checkpoint_path)

    resume_config = copy.deepcopy(config)
    resume_config["training"]["epochs"] = 2
    resume_config["resume_checkpoint"] = str(checkpoint_path)
    _fake_evaluate_losses(monkeypatch, [0.9])
    summary = train_pairwise_correction(
        config=resume_config,
        prepared_root=prepared,
        split_registry_path=split_path,
        output_dir=tmp_path / "run",
    )
    assert summary["epochs_completed"] == 2


def test_active_scheduler_resume_without_state_fails_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, split_path, config = _tiny_training_inputs(tmp_path)
    config["training"]["epochs"] = 1
    config["training"]["lr_scheduler"] = {
        "name": "cosine_annealing",
        "t_max": 3,
        "min_lr": 1.0e-6,
    }
    _fake_evaluate_losses(monkeypatch, [1.0])
    train_pairwise_correction(
        config=config,
        prepared_root=prepared,
        split_registry_path=split_path,
        output_dir=tmp_path / "run",
    )
    checkpoint_path = tmp_path / "run" / "checkpoint_last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint.pop("lr_scheduler_state_dict", None)
    torch.save(checkpoint, checkpoint_path)

    resume_config = copy.deepcopy(config)
    resume_config["training"]["epochs"] = 2
    resume_config["resume_checkpoint"] = str(checkpoint_path)
    with pytest.raises(ValueError, match="missing lr_scheduler_state_dict"):
        train_pairwise_correction(
            config=resume_config,
            prepared_root=prepared,
            split_registry_path=split_path,
            output_dir=tmp_path / "run",
        )


def test_optimizer_summary_records_scheduler_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, split_path, config = _tiny_training_inputs(tmp_path)
    config["training"]["epochs"] = 2
    config["training"]["lr_scheduler"] = {
        "name": "cosine_annealing",
        "t_max": 2,
        "min_lr": 1.0e-6,
    }
    _fake_evaluate_losses(monkeypatch, [1.0, 0.9])
    train_pairwise_correction(
        config=config,
        prepared_root=prepared,
        split_registry_path=split_path,
        output_dir=tmp_path / "run",
    )

    metrics = read_json(tmp_path / "run" / "metrics.json")
    manifest = read_json(tmp_path / "run" / "run_manifest.json")
    assert metrics["schema_version"] == "dluxshera_ml_metrics/2"
    assert metrics["optimization"]["initial_learning_rate"] == pytest.approx(1.0e-3)
    assert metrics["optimization"]["final_learning_rate"] == pytest.approx(1.0e-6)
    assert metrics["optimization"]["lr_scheduler"] == config["training"]["lr_scheduler"]
    assert metrics["optimization"]["lr_reduction_count"] == 2
    assert metrics["optimization"]["lr_reduction_epochs"] == [0, 1]
    assert manifest["optimization"] == metrics["optimization"]
