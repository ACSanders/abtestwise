"""Tests for Bayesian A/B testing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from abtestwise import bayesian
from abtestwise.bayesian import (
    expected_loss_control,
    expected_loss_treatment,
    simulate_lift_samples,
    simulate_mean_difference_samples,
)


def lift(seed, n=50_000):
    rng = np.random.default_rng(seed)
    return simulate_lift_samples(
        120,
        1000,
        145,
        1000,
        1.0,
        1.0,
        n,
        rng,
    )


def mean_difference(seed, n=50_000):
    control = np.array([10.0, 11.5, 9.5, 12.0, 10.5])
    treatment = np.array([12.0, 13.5, 11.0, 14.0, 12.5, 13.0])
    rng = np.random.default_rng(seed)

    return simulate_mean_difference_samples(
        control,
        treatment,
        n,
        rng,
    )


def test_reproducible_with_same_seed():
    a = lift(42)
    b = lift(42)
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    a = lift(1)
    b = lift(2)
    assert not np.array_equal(a, b)


def test_sample_shape():
    samples = lift(0, n=12_345)
    assert samples.shape == (12_345,)


def test_strong_treatment_signal():
    rng = np.random.default_rng(0)
    samples = simulate_lift_samples(
        100,
        1000,
        300,
        1000,
        1.0,
        1.0,
        50_000,
        rng,
    )

    assert np.mean(samples) > 0
    assert np.mean(samples > 0) > 0.99
    assert expected_loss_treatment(samples) < 1e-3


def test_identical_arms_are_symmetric():
    rng = np.random.default_rng(7)
    samples = simulate_lift_samples(
        100,
        1000,
        100,
        1000,
        1.0,
        1.0,
        100_000,
        rng,
    )

    assert abs(np.mean(samples)) < 0.01
    assert abs(np.mean(samples > 0) - 0.5) < 0.02


def test_expected_loss_formulas_on_known_array():
    samples = np.array([-2.0, -1.0, 1.0, 3.0])

    assert expected_loss_treatment(samples) == 0.75
    assert expected_loss_control(samples) == 1.0


def test_credible_interval_brackets_median():
    samples = lift(3)
    lower, upper = bayesian.credible_interval_bounds(samples, 0.95)
    median = float(np.median(samples))

    assert lower < median < upper


def test_continuous_mean_difference_reproducible_with_same_seed():
    a = mean_difference(42)
    b = mean_difference(42)

    assert np.array_equal(a, b)


def test_continuous_mean_difference_changes_with_seed():
    a = mean_difference(1)
    b = mean_difference(2)

    assert not np.array_equal(a, b)


def test_continuous_mean_difference_sample_shape():
    samples = mean_difference(0, n=12_345)

    assert samples.shape == (12_345,)


def test_continuous_mean_difference_preserves_treatment_minus_control():
    control = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    treatment = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

    rng = np.random.default_rng(42)
    samples = simulate_mean_difference_samples(
        control,
        treatment,
        50_000,
        rng,
    )

    assert np.mean(samples) > 0
    assert np.mean(samples > 0) > 0.99


def test_continuous_swapping_arms_reverses_posterior_direction():
    control = np.array([10.0, 11.5, 9.5, 12.0, 10.5])
    treatment = np.array([12.0, 13.5, 11.0, 14.0, 12.5, 13.0])

    forward = simulate_mean_difference_samples(
        control,
        treatment,
        50_000,
        np.random.default_rng(42),
    )
    reversed_ = simulate_mean_difference_samples(
        treatment,
        control,
        50_000,
        np.random.default_rng(42),
    )

    assert np.mean(forward) > 0
    assert np.mean(reversed_) < 0
    assert np.mean(forward) == pytest.approx(-np.mean(reversed_), abs=0.05)


def test_continuous_credible_interval_brackets_median():
    samples = mean_difference(3)

    lower, upper = bayesian.credible_interval_bounds(
        samples,
        0.95,
    )
    median = float(np.median(samples))

    assert lower < median < upper


def test_continuous_treatment_probability_in_unit_range():
    samples = mean_difference(42)
    probability = float(np.mean(samples > 0))

    assert 0.0 <= probability <= 1.0


def test_continuous_expected_loss_uses_native_units():
    samples = np.array([-2.0, -1.0, 1.0, 3.0])

    assert expected_loss_treatment(samples) == 0.75
    assert expected_loss_control(samples) == 1.0