"""Tests for the Bayesian beta-binomial simulation helpers."""

from __future__ import annotations

import numpy as np

from xpkit import bayesian
from xpkit.bayesian import (
    expected_loss_control,
    expected_loss_treatment,
    simulate_lift_samples,
)


def lift(seed, n=50_000):
    rng = np.random.default_rng(seed)
    return simulate_lift_samples(120, 1000, 145, 1000, 1.0, 1.0, n, rng)


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
    samples = simulate_lift_samples(100, 1000, 300, 1000, 1.0, 1.0, 50_000, rng)
    assert np.mean(samples) > 0
    assert np.mean(samples > 0) > 0.99
    assert expected_loss_treatment(samples) < 1e-3


def test_identical_arms_are_symmetric():
    rng = np.random.default_rng(7)
    samples = simulate_lift_samples(100, 1000, 100, 1000, 1.0, 1.0, 100_000, rng)
    assert abs(np.mean(samples)) < 0.01
    assert abs(np.mean(samples > 0) - 0.5) < 0.02


def test_expected_loss_formulas_on_known_array():
    samples = np.array([-2.0, -1.0, 1.0, 3.0])
    # treatment loss: mean(max(-lift, 0)) = mean(2, 1, 0, 0) = 0.75
    # control loss:   mean(max(lift, 0))  = mean(0, 0, 1, 3) = 1.0
    assert expected_loss_treatment(samples) == 0.75
    assert expected_loss_control(samples) == 1.0


def test_credible_interval_brackets_median():
    samples = lift(3)
    lower, upper = bayesian.credible_interval_bounds(samples, 0.95)
    median = float(np.median(samples))
    assert lower < median < upper
