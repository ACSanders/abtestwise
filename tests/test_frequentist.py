"""Tests for frequentist A/B testing methods."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats
from scipy.stats import norm

from abtestwise.frequentist import (
    newcombe_difference_interval,
    two_proportion_z_test,
    welch_t_test,
    wilson_interval,
)


def test_known_case_matches_manual_calculation():
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
    z, p = two_proportion_z_test(0, 1000, 0, 1000)

    assert z == 0.0
    assert p == 1.0


def test_newcombe_interval_matches_reference_equal_sizes():
    interval = newcombe_difference_interval(
        control_successes=50,
        control_total=100,
        treatment_successes=60,
        treatment_total=100,
    )

    assert interval[0] == pytest.approx(-0.03730210321546354)
    assert interval[1] == pytest.approx(0.23212305420266283)


def test_newcombe_interval_matches_reference_unequal_sizes():
    interval = newcombe_difference_interval(
        control_successes=100,
        control_total=1000,
        treatment_successes=220,
        treatment_total=2000,
    )

    assert interval[0] == pytest.approx(-0.01396629006581825)
    assert interval[1] == pytest.approx(0.03239195055889762)


def test_newcombe_interval_matches_reference_small_samples():
    interval = newcombe_difference_interval(
        control_successes=3,
        control_total=20,
        treatment_successes=12,
        treatment_total=35,
    )

    assert interval[0] == pytest.approx(-0.05689706064637151)
    assert interval[1] == pytest.approx(0.38511467799951216)


def test_newcombe_interval_handles_boundary_proportion():
    interval = newcombe_difference_interval(
        control_successes=0,
        control_total=50,
        treatment_successes=8,
        treatment_total=60,
    )

    assert interval[0] == pytest.approx(0.037358735708434496)
    assert interval[1] == pytest.approx(0.2416515967649962)


def test_newcombe_interval_preserves_treatment_minus_control():
    interval = newcombe_difference_interval(
        control_successes=49,
        control_total=50,
        treatment_successes=30,
        treatment_total=50,
    )

    assert interval[0] == pytest.approx(-0.5191625749176154)
    assert interval[1] == pytest.approx(-0.22975867184362034)


def test_newcombe_interval_custom_confidence_level_is_narrower():
    interval_95 = newcombe_difference_interval(
        50,
        100,
        60,
        100,
        confidence_level=0.95,
    )
    interval_90 = newcombe_difference_interval(
        50,
        100,
        60,
        100,
        confidence_level=0.90,
    )

    width_95 = interval_95[1] - interval_95[0]
    width_90 = interval_90[1] - interval_90[0]

    assert width_90 < width_95


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.1, 1.1])
def test_wilson_interval_rejects_invalid_confidence_level(confidence_level):
    with pytest.raises(ValueError, match="confidence_level"):
        wilson_interval(
            successes=50,
            total=100,
            confidence_level=confidence_level,
        )


def test_welch_t_test_matches_scipy():
    control = np.array([10.0, 11.5, 9.5, 12.0])
    treatment = np.array([12.0, 13.5, 11.0, 14.0, 12.5])

    statistic, p_value, df, interval = welch_t_test(
        control,
        treatment,
        confidence_level=0.95,
    )

    expected = stats.ttest_ind(
        treatment,
        control,
        equal_var=False,
        alternative="two-sided",
    )
    expected_interval = expected.confidence_interval(
        confidence_level=0.95,
    )

    assert statistic == pytest.approx(expected.statistic)
    assert p_value == pytest.approx(expected.pvalue)
    assert df == pytest.approx(expected.df)
    assert interval[0] == pytest.approx(expected_interval.low)
    assert interval[1] == pytest.approx(expected_interval.high)


def test_welch_t_test_preserves_treatment_minus_control():
    control = np.array([1.0, 2.0, 3.0, 4.0])
    treatment = np.array([5.0, 6.0, 7.0, 8.0])

    statistic, _, _, interval = welch_t_test(
        control,
        treatment,
    )

    assert statistic > 0
    assert interval[0] > 0


def test_welch_t_test_swapping_arms_reverses_direction():
    control = np.array([10.0, 11.5, 9.5, 12.0])
    treatment = np.array([12.0, 13.5, 11.0, 14.0, 12.5])

    forward = welch_t_test(control, treatment)
    reversed_ = welch_t_test(treatment, control)

    assert reversed_[0] == pytest.approx(-forward[0])
    assert reversed_[1] == pytest.approx(forward[1])
    assert reversed_[2] == pytest.approx(forward[2])
    assert reversed_[3][0] == pytest.approx(-forward[3][1])
    assert reversed_[3][1] == pytest.approx(-forward[3][0])


def test_welch_t_test_supports_unequal_sample_sizes():
    control = np.array([10.0, 11.0, 9.5, 12.0])
    treatment = np.array([11.0, 12.0, 13.0, 12.5, 14.0, 10.5])

    statistic, p_value, df, interval = welch_t_test(
        control,
        treatment,
    )

    assert np.isfinite(statistic)
    assert 0.0 <= p_value <= 1.0
    assert df > 0.0
    assert interval[0] < interval[1]


def test_welch_t_test_custom_confidence_level_is_narrower():
    control = np.array([10.0, 11.5, 9.5, 12.0])
    treatment = np.array([12.0, 13.5, 11.0, 14.0, 12.5])

    _, _, _, interval_95 = welch_t_test(
        control,
        treatment,
        confidence_level=0.95,
    )
    _, _, _, interval_90 = welch_t_test(
        control,
        treatment,
        confidence_level=0.90,
    )

    width_95 = interval_95[1] - interval_95[0]
    width_90 = interval_90[1] - interval_90[0]

    assert width_90 < width_95