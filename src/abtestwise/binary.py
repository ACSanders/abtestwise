"""Run binary A/B tests from counts, raw samples, or DataFrame-like data."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import bayesian, frequentist, validation
from .result import BinaryABResult


@dataclass(frozen=True)
class BinaryABTest:
    """A binary A/B test with frequentist and Bayesian analysis."""

    control_successes: int
    control_total: int
    treatment_successes: int
    treatment_total: int
    prior_alpha: float
    prior_beta: float
    n_simulations: int
    confidence_level: float
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
        confidence_level: float = 0.95,
        credible_interval: float = 0.95,
        seed: int | None = None,
    ) -> "BinaryABTest":
        """Create a binary A/B test from aggregate counts."""
        validation.validate_count("control_successes", control_successes)
        validation.validate_total("control_total", control_total)
        validation.validate_count("treatment_successes", treatment_successes)
        validation.validate_total("treatment_total", treatment_total)
        validation.validate_successes_le_total(
            "control_successes",
            control_successes,
            "control_total",
            control_total,
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
        validation.validate_confidence_level(confidence_level)
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
            confidence_level=float(confidence_level),
            credible_interval=float(credible_interval),
            seed=seed,
        )

    @classmethod
    def from_samples(
        cls,
        control: object,
        treatment: object,
        *,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_simulations: int = 100_000,
        confidence_level: float = 0.95,
        credible_interval: float = 0.95,
        seed: int | None = None,
    ) -> "BinaryABTest":
        """Create a binary A/B test from raw control and treatment samples."""
        control_values = validation.validate_binary_sample("control", control)
        treatment_values = validation.validate_binary_sample("treatment", treatment)

        return cls.from_counts(
            control_successes=int(control_values.sum()),
            control_total=int(control_values.size),
            treatment_successes=int(treatment_values.sum()),
            treatment_total=int(treatment_values.size),
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            n_simulations=n_simulations,
            confidence_level=confidence_level,
            credible_interval=credible_interval,
            seed=seed,
        )

    @classmethod
    def from_dataframe(
        cls,
        data: object,
        *,
        group_col: str,
        outcome_col: str,
        control: object,
        treatment: object,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_simulations: int = 100_000,
        confidence_level: float = 0.95,
        credible_interval: float = 0.95,
        seed: int | None = None,
    ) -> "BinaryABTest":
        """Create a binary A/B test from a DataFrame-like object."""
        if control == treatment:
            raise ValueError("control and treatment must identify different groups.")

        try:
            groups = np.asarray(data[group_col])
            outcomes = np.asarray(data[outcome_col])
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError(
                "data must provide the requested group_col and outcome_col."
            ) from exc

        if groups.ndim != 1 or outcomes.ndim != 1:
            raise ValueError("group_col and outcome_col must each be one-dimensional.")

        if groups.size != outcomes.size:
            raise ValueError("group_col and outcome_col must have the same length.")

        control_mask = groups == control
        treatment_mask = groups == treatment

        if not np.any(control_mask):
            raise ValueError(f"control group {control!r} was not found in group_col.")

        if not np.any(treatment_mask):
            raise ValueError(
                f"treatment group {treatment!r} was not found in group_col."
            )

        return cls.from_samples(
            control=outcomes[control_mask],
            treatment=outcomes[treatment_mask],
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            n_simulations=n_simulations,
            confidence_level=confidence_level,
            credible_interval=credible_interval,
            seed=seed,
        )

    def run(self) -> BinaryABResult:
        """Run the frequentist and Bayesian analyses and return the result."""
        control_rate = self.control_successes / self.control_total
        treatment_rate = self.treatment_successes / self.treatment_total
        absolute_lift = treatment_rate - control_rate

        relative_lift = (
            absolute_lift / control_rate if control_rate != 0 else math.nan
        )

        z_statistic, p_value = frequentist.two_proportion_z_test(
            self.control_successes,
            self.control_total,
            self.treatment_successes,
            self.treatment_total,
        )

        confidence_interval_bounds = frequentist.newcombe_difference_interval(
            self.control_successes,
            self.control_total,
            self.treatment_successes,
            self.treatment_total,
            confidence_level=self.confidence_level,
        )

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

        credible_interval_bounds = bayesian.credible_interval_bounds(
            lift_samples,
            self.credible_interval,
        )

        return BinaryABResult(
            control_successes=self.control_successes,
            control_total=self.control_total,
            treatment_successes=self.treatment_successes,
            treatment_total=self.treatment_total,
            prior_alpha=self.prior_alpha,
            prior_beta=self.prior_beta,
            n_simulations=self.n_simulations,
            confidence_level=self.confidence_level,
            credible_interval=self.credible_interval,
            seed=self.seed,
            control_rate=control_rate,
            treatment_rate=treatment_rate,
            absolute_lift=absolute_lift,
            relative_lift=relative_lift,
            z_statistic=z_statistic,
            p_value=p_value,
            confidence_interval_bounds=confidence_interval_bounds,
            posterior_mean_lift=float(np.mean(lift_samples)),
            posterior_median_lift=float(np.median(lift_samples)),
            prob_treatment_better=float(np.mean(lift_samples > 0)),
            prob_control_better=float(np.mean(lift_samples < 0)),
            credible_interval_bounds=credible_interval_bounds,
            expected_loss_treatment=bayesian.expected_loss_treatment(lift_samples),
            expected_loss_control=bayesian.expected_loss_control(lift_samples),
            lift_samples=lift_samples,
        )