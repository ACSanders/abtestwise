"""Tests for the two-sided pooled two-proportion z-test."""

from __future__ import annotations

import math

from scipy.stats import norm

from xpkit.frequentist import two_proportion_z_test


def test_known_case_matches_manual_calculation():
    # control 120/1000, treatment 145/1000
    z, p = two_proportion_z_test(120, 1000, 145, 1000)

    pooled = (120 + 145) / 2000
    se = math.sqrt(pooled * (1 - pooled) * (1 / 1000 + 1 / 1000))
    expected_z = (0.145 - 0.120) / se
    expected_p = 2 * (1 - norm.cdf(abs(expected_z)))

    assert abs(z - expected_z) < 1e-9
    assert abs(p - expected_p) < 1e-12


def test_treatment_higher_gives_positive_z():
    z, _ = two_proportion_z_test(100, 1000, 150, 1000)
    assert z > 0


def test_swapping_arms_flips_sign_keeps_pvalue():
    z1, p1 = two_proportion_z_test(120, 1000, 145, 1000)
    z2, p2 = two_proportion_z_test(145, 1000, 120, 1000)
    assert abs(z1 + z2) < 1e-12
    assert abs(p1 - p2) < 1e-12


def test_identical_rates_give_zero_z_and_p_one():
    z, p = two_proportion_z_test(100, 1000, 100, 1000)
    assert abs(z) < 1e-12
    assert abs(p - 1.0) < 1e-12


def test_degenerate_zero_standard_error():
    # Both arms all failures -> pooled rate 0 -> se 0 -> defined fallback.
    z, p = two_proportion_z_test(0, 1000, 0, 1000)
    assert z == 0.0
    assert p == 1.0
