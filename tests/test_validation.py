"""Tests for input validation rules enforced by ``BinaryABTest.from_counts``."""

from __future__ import annotations

import pytest

from abtestwise import BinaryABTest
from abtestwise.validation import validate_margin


def make(**overrides):
    """Build a valid set of kwargs, with optional overrides for one bad field."""
    kwargs = dict(
        control_successes=120,
        control_total=1000,
        treatment_successes=145,
        treatment_total=1000,
        prior_alpha=1,
        prior_beta=1,
        n_simulations=1000,
        credible_interval=0.95,
        seed=42,
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_inputs_construct():
    test = BinaryABTest.from_counts(**make())
    assert test.control_total == 1000


@pytest.mark.parametrize(
    "overrides",
    [
        {"control_total": 0},
        {"treatment_total": -5},
        {"control_successes": -1},
        {"control_successes": 1001},  # successes > total
        {"treatment_successes": 1001},
        {"prior_alpha": 0},
        {"prior_beta": -1},
        {"credible_interval": 0.0},
        {"credible_interval": 1.0},
        {"credible_interval": 1.5},
        {"n_simulations": 0},
        {"n_simulations": -10},
        {"seed": -1},
    ],
)
def test_invalid_inputs_raise(overrides):
    with pytest.raises(ValueError):
        BinaryABTest.from_counts(**make(**overrides))


def test_keyword_only_arguments_enforced():
    # Priors and later args must be passed by keyword.
    with pytest.raises(TypeError):
        BinaryABTest.from_counts(120, 1000, 145, 1000, 1, 1)  # type: ignore[misc]


@pytest.mark.parametrize("value", [0, 0.005])
def test_validate_margin_accepts_valid(value):
    validate_margin(value)  # should not raise


@pytest.mark.parametrize(
    "value",
    [-0.1, True, "x", float("nan"), float("inf"), float("-inf")],
)
def test_validate_margin_rejects_invalid(value):
    with pytest.raises(ValueError):
        validate_margin(value)
