"""Bayesian analysis helpers for A/B testing."""

from __future__ import annotations

import numpy as np
from scipy import stats


def posterior_samples(
    successes: int,
    total: int,
    prior_alpha: float,
    prior_beta: float,
    n_simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw posterior samples of a binary success rate."""
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
    """Return posterior samples of binary lift: Treatment - Control."""
    control = posterior_samples(
        control_successes,
        control_total,
        prior_alpha,
        prior_beta,
        n_simulations,
        rng,
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


def simulate_mean_difference_samples(
    control: np.ndarray,
    treatment: np.ndarray,
    n_simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return posterior samples of the mean difference: Treatment - Control."""
    control_mean_distribution, _, _ = stats.mvsdist(control)
    treatment_mean_distribution, _, _ = stats.mvsdist(treatment)

    control_mean_samples = control_mean_distribution.rvs(
        size=n_simulations,
        random_state=rng,
    )
    treatment_mean_samples = treatment_mean_distribution.rvs(
        size=n_simulations,
        random_state=rng,
    )

    return np.asarray(treatment_mean_samples) - np.asarray(control_mean_samples)


def credible_interval_bounds(
    samples: np.ndarray,
    credible_interval: float,
) -> tuple[float, float]:
    """Return an equal-tailed credible interval."""
    tail = (1.0 - credible_interval) / 2.0
    lower = float(np.quantile(samples, tail))
    upper = float(np.quantile(samples, 1.0 - tail))
    return lower, upper


def expected_loss_treatment(difference_samples: np.ndarray) -> float:
    """Expected loss from choosing treatment."""
    return float(np.mean(np.maximum(-difference_samples, 0.0)))


def expected_loss_control(difference_samples: np.ndarray) -> float:
    """Expected loss from choosing control."""
    return float(np.mean(np.maximum(difference_samples, 0.0)))