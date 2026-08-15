"""Run continuous A/B tests from raw samples or DataFrame-like data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import bayesian, frequentist, validation
from .result import ContinuousABResult


@dataclass(frozen=True)
class ContinuousABTest:
    """A continuous A/B test with frequentist and Bayesian analysis."""

    control: np.ndarray
    treatment: np.ndarray
    n_simulations: int
    confidence_level: float
    credible_interval: float
    seed: int | None

    @classmethod
    def from_samples(
        cls,
        control: object,
        treatment: object,
        *,
        n_simulations: int = 100_000,
        confidence_level: float = 0.95,
        credible_interval: float = 0.95,
        seed: int | None = None,
    ) -> "ContinuousABTest":
        """Create a continuous A/B test from raw samples."""
        control_values = validation.validate_continuous_sample(
            "control",
            control,
        )
        treatment_values = validation.validate_continuous_sample(
            "treatment",
            treatment,
        )

        validation.validate_n_simulations(n_simulations)
        validation.validate_confidence_level(confidence_level)
        validation.validate_credible_interval(credible_interval)
        validation.validate_seed(seed)

        return cls(
            control=control_values,
            treatment=treatment_values,
            n_simulations=n_simulations,
            confidence_level=float(confidence_level),
            credible_interval=float(credible_interval),
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
        n_simulations: int = 100_000,
        confidence_level: float = 0.95,
        credible_interval: float = 0.95,
        seed: int | None = None,
    ) -> "ContinuousABTest":
        """Create a continuous A/B test from a DataFrame-like object."""
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
            n_simulations=n_simulations,
            confidence_level=confidence_level,
            credible_interval=credible_interval,
            seed=seed,
        )

    def run(self) -> ContinuousABResult:
        """Run the frequentist and Bayesian analyses and return the result."""
        control_n = int(self.control.size)
        treatment_n = int(self.treatment.size)

        control_mean = float(np.mean(self.control))
        treatment_mean = float(np.mean(self.treatment))
        mean_difference = treatment_mean - control_mean

        (
            t_statistic,
            p_value,
            degrees_of_freedom,
            confidence_interval_bounds,
        ) = frequentist.welch_t_test(
            self.control,
            self.treatment,
            confidence_level=self.confidence_level,
        )

        rng = np.random.default_rng(self.seed)
        mean_difference_samples = bayesian.simulate_mean_difference_samples(
            self.control,
            self.treatment,
            self.n_simulations,
            rng,
        )

        credible_interval_bounds = bayesian.credible_interval_bounds(
            mean_difference_samples,
            self.credible_interval,
        )

        return ContinuousABResult(
            n_simulations=self.n_simulations,
            confidence_level=self.confidence_level,
            credible_interval=self.credible_interval,
            seed=self.seed,
            control_n=control_n,
            treatment_n=treatment_n,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            mean_difference=mean_difference,
            t_statistic=t_statistic,
            p_value=p_value,
            degrees_of_freedom=degrees_of_freedom,
            confidence_interval_bounds=confidence_interval_bounds,
            posterior_mean_difference=float(np.mean(mean_difference_samples)),
            posterior_median_difference=float(np.median(mean_difference_samples)),
            prob_treatment_better=float(np.mean(mean_difference_samples > 0)),
            prob_control_better=float(np.mean(mean_difference_samples < 0)),
            credible_interval_bounds=credible_interval_bounds,
            expected_loss_treatment=bayesian.expected_loss_treatment(
                mean_difference_samples
            ),
            expected_loss_control=bayesian.expected_loss_control(
                mean_difference_samples
            ),
            mean_difference_samples=mean_difference_samples,
        )