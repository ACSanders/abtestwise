"""Run binary A/B tests from aggregate count data.

This module validates inputs, sets up the random number generator so results are
reproducible from a seed, and combines the frequentist and Bayesian helpers to
build a :class:BinaryABResult.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import bayesian, frequentist, validation
from .result import BinaryABResult


@dataclass(frozen=True)
class BinaryABTest:
    """A binary A/B test using aggregate success and total counts."""

    control_successes: int
    control_total: int
    treatment_successes: int
    treatment_total: int
    prior_alpha: float
    prior_beta: float
    n_simulations: int
    credible_interval: float
    seed: int | None

    @classmethod
    def from_counts(
        cls,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
        *,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_simulations: int = 100_000,
        credible_interval: float = 0.95,
        seed: int | None = None,
    ) -> "BinaryABTest":
        """Create a binary A/B test from aggregate counts.

        The four count inputs can be positional. All other settings have to be names.
        The default prior is Beta(1, 1).
        """
        validation.validate_count("control_successes", control_successes)
        validation.validate_total("control_total", control_total)
        validation.validate_count("treatment_successes", treatment_successes)
        validation.validate_total("treatment_total", treatment_total)
        validation.validate_successes_le_total(
            "control_successes", control_successes, "control_total", control_total
        )
        validation.validate_successes_le_total(
            "treatment_successes",
            treatment_successes,
            "treatment_total",
            treatment_total,
        )
        validation.validate_prior("prior_alpha", prior_alpha)
        validation.validate_prior("prior_beta", prior_beta)
        validation.validate_n_simulations(n_simulations)
        validation.validate_credible_interval(credible_interval)
        validation.validate_seed(seed)

        return cls(
            control_successes=control_successes,
            control_total=control_total,
            treatment_successes=treatment_successes,
            treatment_total=treatment_total,
            prior_alpha=float(prior_alpha),
            prior_beta=float(prior_beta),
            n_simulations=n_simulations,
            credible_interval=credible_interval,
            seed=seed,
        )

    def run(self) -> BinaryABResult:
        """Run the frequentist and Bayesian analyses and return the result."""
        control_rate = self.control_successes / self.control_total
        treatment_rate = self.treatment_successes / self.treatment_total
        absolute_lift = treatment_rate - control_rate

        # Relative lift is undefined when the control rate is zero.
        relative_lift = (
            absolute_lift / control_rate if control_rate != 0 else math.nan
        )

        # --- Frequentist ---
        z_statistic, p_value = frequentist.two_proportion_z_test(
            self.control_successes,
            self.control_total,
            self.treatment_successes,
            self.treatment_total,
        )

        # --- Bayesian ---
        rng = np.random.default_rng(self.seed)
        lift_samples = bayesian.simulate_lift_samples(
            self.control_successes,
            self.control_total,
            self.treatment_successes,
            self.treatment_total,
            self.prior_alpha,
            self.prior_beta,
            self.n_simulations,
            rng,
        )

        lower, upper = bayesian.credible_interval_bounds(
            lift_samples, self.credible_interval
        )

        return BinaryABResult(
            control_successes=self.control_successes,
            control_total=self.control_total,
            treatment_successes=self.treatment_successes,
            treatment_total=self.treatment_total,
            prior_alpha=self.prior_alpha,
            prior_beta=self.prior_beta,
            n_simulations=self.n_simulations,
            credible_interval=self.credible_interval,
            seed=self.seed,
            control_rate=control_rate,
            treatment_rate=treatment_rate,
            absolute_lift=absolute_lift,
            relative_lift=relative_lift,
            z_statistic=z_statistic,
            p_value=p_value,
            posterior_mean_lift=float(np.mean(lift_samples)),
            posterior_median_lift=float(np.median(lift_samples)),
            prob_treatment_better=float(np.mean(lift_samples > 0)),
            prob_control_better=float(np.mean(lift_samples < 0)),
            credible_interval_bounds=(lower, upper),
            expected_loss_treatment=bayesian.expected_loss_treatment(lift_samples),
            expected_loss_control=bayesian.expected_loss_control(lift_samples),
            lift_samples=lift_samples,
        )
