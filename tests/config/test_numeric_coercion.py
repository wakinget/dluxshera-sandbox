from __future__ import annotations

import math
import json

import pytest

from dluxshera.config.io import load_config_file
from dluxshera.config.numeric import coerce_numeric_value, normalize_optimizer_kwargs


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1e-3", 1e-3),
        ("1.0e-8", 1.0e-8),
        ("5e-4", 5e-4),
        (2, 2.0),
        (0.25, 0.25),
    ],
)
def test_coerce_numeric_value_accepts_expected_numeric_fields(raw, expected):
    assert coerce_numeric_value(raw, path="field") == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "   ", "abc", True, False])
def test_coerce_numeric_value_rejects_bad_numeric_fields(raw):
    with pytest.raises(ValueError, match="expected a numeric value"):
        coerce_numeric_value(raw, path="field")


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", math.inf])
def test_coerce_numeric_value_rejects_nonfinite_by_default(raw):
    with pytest.raises(ValueError, match="finite numeric"):
        coerce_numeric_value(raw, path="field")


def test_normalize_optimizer_kwargs_coerces_adam_scientific_notation():
    kwargs = normalize_optimizer_kwargs(
        "adam",
        {"b1": "0.9", "b2": 0.999, "eps": "1e-8", "eps_root": "1.0e-8"},
        path="experiment.optimizer.kwargs",
    )

    assert kwargs == {
        "b1": pytest.approx(0.9),
        "b2": pytest.approx(0.999),
        "eps": pytest.approx(1e-8),
        "eps_root": pytest.approx(1.0e-8),
    }


def test_normalize_optimizer_kwargs_coerces_sgd_momentum():
    kwargs = normalize_optimizer_kwargs(
        "sgd",
        {"momentum": "0.05", "nesterov": True},
        path="experiment.optimizer.kwargs",
    )

    assert kwargs == {"momentum": pytest.approx(0.05), "nesterov": True}


@pytest.mark.parametrize("kwargs", [{"eps": "abc"}, {"eps": ""}, {"eps": True}])
def test_normalize_optimizer_kwargs_rejects_bad_adam_numeric_values(kwargs):
    with pytest.raises(ValueError, match="experiment.optimizer.kwargs.eps"):
        normalize_optimizer_kwargs("adam", kwargs, path="experiment.optimizer.kwargs")


def test_normalize_optimizer_kwargs_rejects_unknown_kwargs():
    with pytest.raises(ValueError, match="not supported"):
        normalize_optimizer_kwargs("adam", {"not_a_kwarg": "1e-3"})


def test_config_loading_does_not_globally_coerce_numeric_strings(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "experiment": {
                    "optimizer": {"base_lr": "1e-3"},
                    "notes": "1e-3 is text here",
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = load_config_file(path)

    assert loaded["experiment"]["optimizer"]["base_lr"] == "1e-3"
    assert loaded["experiment"]["notes"] == "1e-3 is text here"
    assert (
        coerce_numeric_value(
            loaded["experiment"]["optimizer"]["base_lr"],
            path="experiment.optimizer.base_lr",
        )
        == pytest.approx(1e-3)
    )
