from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dluxshera.ml.training import EarlyStoppingConfig, EarlyStoppingState


def _run(values: list[float], config: EarlyStoppingConfig) -> EarlyStoppingState:
    state = EarlyStoppingState()
    for epoch, value in enumerate(values):
        _, should_stop = state.update(epoch=epoch, metric=value, config=config)
        if should_stop:
            break
    return state


def test_disabled_early_stopping_never_requests_stop() -> None:
    config = EarlyStoppingConfig(enabled=False, patience=1)
    state = _run([1.0, 1.1, 1.2, 1.3], config)
    assert state.early_stopped is False
    assert state.epochs_completed == 4


def test_early_stopping_cannot_stop_before_min_epochs() -> None:
    config = EarlyStoppingConfig(enabled=True, min_epochs=4, patience=1)
    state = _run([1.0, 1.1, 1.2, 1.3], config)
    assert state.early_stopped is True
    assert state.stop_epoch == 3
    assert state.epochs_completed == 4


def test_patience_counts_consecutive_bad_epochs() -> None:
    config = EarlyStoppingConfig(enabled=True, min_epochs=1, patience=2)
    state = _run([1.0, 1.1, 1.2, 0.8], config)
    assert state.early_stopped is True
    assert state.stop_epoch == 2
    assert state.bad_epochs == 2


def test_meaningful_improvement_resets_patience() -> None:
    config = EarlyStoppingConfig(
        enabled=True,
        min_epochs=1,
        patience=2,
        min_delta_relative=0.01,
    )
    state = _run([1.0, 1.1, 0.98, 0.99, 1.0], config)
    assert state.early_stopped is True
    assert state.stop_epoch == 4
    assert state.reference_loss == pytest.approx(0.98)


def test_tiny_improvement_updates_absolute_best_without_resetting_patience() -> None:
    config = EarlyStoppingConfig(
        enabled=True,
        min_epochs=1,
        patience=2,
        min_delta_relative=0.01,
    )
    state = EarlyStoppingState()
    is_best, should_stop = state.update(epoch=0, metric=1.0, config=config)
    assert is_best is True
    assert should_stop is False
    is_best, should_stop = state.update(epoch=1, metric=0.9995, config=config)
    assert is_best is True
    assert should_stop is False
    assert state.absolute_best_loss == pytest.approx(0.9995)
    assert state.reference_loss == pytest.approx(1.0)
    assert state.bad_epochs == 1
    _, should_stop = state.update(epoch=2, metric=1.1, config=config)
    assert should_stop is True


def test_early_stopping_state_round_trips_from_checkpoint_metadata() -> None:
    state = EarlyStoppingState(
        absolute_best_loss=0.7,
        best_epoch=5,
        reference_loss=0.72,
        bad_epochs=3,
        epochs_completed=9,
        early_stopped=False,
    )
    loaded = EarlyStoppingState.from_checkpoint({"early_stopping_state": state.to_dict()})
    assert loaded.to_dict() == state.to_dict()


def test_older_checkpoint_without_early_stopping_state_is_loadable() -> None:
    loaded = EarlyStoppingState.from_checkpoint(
        {
            "epoch": 4,
            "validation_metrics": {"fisher_overall_rmse": 2.0},
        }
    )
    assert loaded.absolute_best_loss == pytest.approx(4.0)
    assert loaded.best_epoch == 4
    assert loaded.reference_loss == pytest.approx(4.0)
    assert loaded.bad_epochs == 0
    assert loaded.epochs_completed == 5


def test_enabled_early_stopping_requires_positive_patience() -> None:
    with pytest.raises(ValueError, match="patience.*>= 1"):
        EarlyStoppingConfig(enabled=True, patience=0)
