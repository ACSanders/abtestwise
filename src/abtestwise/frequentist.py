"""Frequentist two-sided pooled two-proportion z-test."""

from __future__ import annotations

import math

from scipy.stats import norm


def two_proportion_z_test(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
) -> tuple[float, float]:
    """Run a two-sided pooled two-proportion z-test.

    Returns ``(z_statistic, p_value)``.

    The pooled estimate combines both arms under the null hypothesis that the
    two proportions are equal:

        p_pool = (x_c + x_t) / (n_c + n_t)
        se     = sqrt(p_pool * (1 - p_pool) * (1/n_c + 1/n_t))
        z      = (rate_t - rate_c) / se
        p      = 2 * (1 - Phi(|z|))

    If the standard error is zero, then return ``(0.0, 1.0)`` to avoid dividing by zero.
    """
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
