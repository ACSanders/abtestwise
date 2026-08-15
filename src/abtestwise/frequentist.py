"""Frequentist methods for A/B testing."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.stats import norm


def two_proportion_z_test(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
) -> tuple[float, float]:
    """Run a two-sided pooled two-proportion z-test."""
    control_rate = control_successes / control_total
    treatment_rate = treatment_successes / treatment_total

    pooled_rate = (control_successes + treatment_successes) / (
        control_total + treatment_total
    )
    standard_error = math.sqrt(
        pooled_rate
        * (1.0 - pooled_rate)
        * (1.0 / control_total + 1.0 / treatment_total)
    )

    if standard_error == 0.0:
        return 0.0, 1.0

    z_statistic = (treatment_rate - control_rate) / standard_error
    p_value = 2.0 * (1.0 - norm.cdf(abs(z_statistic)))

    return float(z_statistic), float(p_value)


def wilson_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Calculate a Wilson score confidence interval for one proportion."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    if total <= 0:
        raise ValueError("total must be positive.")

    rate = successes / total
    alpha = 1.0 - confidence_level
    z = norm.ppf(1.0 - alpha / 2.0)
    z_squared = z**2

    denominator = 1.0 + z_squared / total
    center = (rate + z_squared / (2.0 * total)) / denominator

    half_width = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / total
            + z_squared / (4.0 * total**2)
        )
        / denominator
    )

    return (
        float(center - half_width),
        float(center + half_width),
    )


def newcombe_difference_interval(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Calculate a Newcombe confidence interval for Treatment - Control."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    control_rate = control_successes / control_total
    treatment_rate = treatment_successes / treatment_total
    difference = treatment_rate - control_rate

    control_low, control_high = wilson_interval(
        control_successes,
        control_total,
        confidence_level,
    )
    treatment_low, treatment_high = wilson_interval(
        treatment_successes,
        treatment_total,
        confidence_level,
    )

    lower_distance = math.sqrt(
        (treatment_rate - treatment_low) ** 2
        + (control_high - control_rate) ** 2
    )
    upper_distance = math.sqrt(
        (treatment_high - treatment_rate) ** 2
        + (control_rate - control_low) ** 2
    )

    return (
        float(difference - lower_distance),
        float(difference + upper_distance),
    )


def welch_t_test(
    control: np.ndarray,
    treatment: np.ndarray,
    confidence_level: float = 0.95,
) -> tuple[float, float, float, tuple[float, float]]:
    """Run a two-sided Welch t-test for Treatment - Control."""
    result = stats.ttest_ind(
        treatment,
        control,
        equal_var=False,
        alternative="two-sided",
    )

    interval = result.confidence_interval(
        confidence_level=confidence_level,
    )

    return (
        float(result.statistic),
        float(result.pvalue),
        float(result.df),
        (float(interval.low), float(interval.high)),
    )