"""Bayesian analysis: beta-binomial posterior simulation.

For a binary metric with a Beta prior, the posterior for each group's true
success rate is conjugate and also Beta:

posterior = Beta(prior_alpha + successes, prior_beta + failures)

We draw samples from each group's posterior, form the lift distribution
(Treatment B - Control A), and summarize it using NumPy operations.
"""

from __future__ import annotations

import numpy as np


def posterior_samples(
    successes: int,
    total: int,
    prior_alpha: float,
    prior_beta: float,
    n_simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """"Draw posterior samples of a group's true success rate from its Beta posterior."""
    failures = total - successes
    alpha = prior_alpha + successes
    beta = prior_beta + failures
    return rng.beta(alpha, beta, size=n_simulations)


def simulate_lift_samples(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    prior_alpha: float,
    prior_beta: float,
    n_simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return posterior samples of the lift (treatment rate - control rate)."""
    control = posterior_samples(
        control_successes, control_total, prior_alpha, prior_beta, n_simulations, rng
    )
    treatment = posterior_samples(
        treatment_successes,
        treatment_total,
        prior_alpha,
        prior_beta,
        n_simulations,
        rng,
    )
    return treatment - control


def credible_interval_bounds(
    lift_samples: np.ndarray, credible_interval: float
) -> tuple[float, float]:
    """Equal-tailed credible interval for the lift at the given level.

    For a 0.95 interval this returns the 2.5th and 97.5th percentiles.
    """
    tail = (1.0 - credible_interval) / 2.0
    lower = float(np.quantile(lift_samples, tail))
    upper = float(np.quantile(lift_samples, 1.0 - tail))
    return lower, upper


def expected_loss_treatment(lift_samples: np.ndarray) -> float:
    """Expected loss from choosing treatment: mean(max(-lift, 0))."""
    return float(np.mean(np.maximum(-lift_samples, 0.0)))


def expected_loss_control(lift_samples: np.ndarray) -> float:
    """Expected loss from choosing control: mean(max(lift, 0))."""
    return float(np.mean(np.maximum(lift_samples, 0.0)))
