"""Result objects for A/B tests."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import validation


def _format_percent(x: float, digits: int = 2) -> str:
    """Format a proportion as a percent string."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "undefined"
    return f"{x * 100:.{digits}f}%"


def _format_pp(x: float, digits: int = 2) -> str:
    """Format a proportion difference as signed percentage points."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "undefined"
    return f"{x * 100:+.{digits}f}"


def _format_probability(x: float, digits: int = 1) -> str:
    """Format a probability as a percent string."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "undefined"
    return f"{x * 100:.{digits}f}%"


def _format_number(x: float, digits: int = 4) -> str:
    """Format a continuous quantity."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "undefined"
    return f"{x:.{digits}f}"


def _format_signed_number(x: float, digits: int = 4) -> str:
    """Format a signed continuous quantity."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "undefined"
    return f"{x:+.{digits}f}"


@dataclass(frozen=True)
class BinaryABResult:
    """Results from a binary A/B test.

    Lift is always Treatment B - Control A.
    """

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

    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift: float
    z_statistic: float
    p_value: float
    confidence_interval_bounds: tuple[float, float]

    posterior_mean_lift: float
    posterior_median_lift: float
    prob_treatment_better: float
    prob_control_better: float
    credible_interval_bounds: tuple[float, float]
    expected_loss_treatment: float
    expected_loss_control: float

    lift_samples: np.ndarray = field(repr=False)

    def prob_lift_above(self, threshold: float) -> float:
        """Return the posterior probability that lift is above a threshold."""
        return float(np.mean(self.lift_samples > threshold))

    def prob_no_harm(self, margin: float = 0.0) -> float:
        """Return P(lift >= -margin | data)."""
        validation.validate_margin(margin)
        return float(np.mean(self.lift_samples >= -margin))

    def prob_harm_above(self, margin: float = 0.0) -> float:
        """Return P(lift < -margin | data)."""
        validation.validate_margin(margin)
        return float(np.mean(self.lift_samples < -margin))

    def plot_lift_distribution(
        self,
        ax: Any = None,
        *,
        bins: int = 50,
        density: bool = True,
        title: str | None = "Posterior Distribution of Lift",
    ) -> Any:
        """Plot the posterior lift distribution."""
        from . import plotting

        return plotting.plot_lift_distribution(
            self.lift_samples,
            self.posterior_median_lift,
            self.credible_interval_bounds,
            self.credible_interval,
            ax=ax,
            bins=bins,
            density=density,
            title=title,
        )

    def plot_probability_bar(
        self,
        ax: Any = None,
        *,
        title: str | None = "Posterior Probability of Being Better",
    ) -> Any:
        """Plot the probability that each group is better."""
        from . import plotting

        return plotting.plot_probability_bar(
            self.prob_treatment_better,
            self.prob_control_better,
            ax=ax,
            title=title,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return result fields as a dictionary."""
        return {
            "control_successes": self.control_successes,
            "control_total": self.control_total,
            "treatment_successes": self.treatment_successes,
            "treatment_total": self.treatment_total,
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "n_simulations": self.n_simulations,
            "confidence_level": self.confidence_level,
            "credible_interval": self.credible_interval,
            "seed": self.seed,
            "control_rate": self.control_rate,
            "treatment_rate": self.treatment_rate,
            "absolute_lift": self.absolute_lift,
            "relative_lift": self.relative_lift,
            "z_statistic": self.z_statistic,
            "p_value": self.p_value,
            "confidence_interval_bounds": self.confidence_interval_bounds,
            "posterior_mean_lift": self.posterior_mean_lift,
            "posterior_median_lift": self.posterior_median_lift,
            "prob_treatment_better": self.prob_treatment_better,
            "prob_control_better": self.prob_control_better,
            "credible_interval_bounds": self.credible_interval_bounds,
            "expected_loss_treatment": self.expected_loss_treatment,
            "expected_loss_control": self.expected_loss_control,
        }

    def summary(self) -> str:
        """Return a formatted summary."""
        confidence_pct = self.confidence_level * 100
        credible_pct = self.credible_interval * 100

        confidence_lower, confidence_upper = self.confidence_interval_bounds
        credible_lower, credible_upper = self.credible_interval_bounds

        if math.isnan(self.relative_lift):
            relative_lift_str = "undefined"
        else:
            relative_lift_str = f"{self.relative_lift * 100:+.2f}%"

        lines = [
            "Binary A/B test result",
            "=" * 40,
            "Observed (lift is always Treatment B - Control A)",
            f"  Control (A):   {self.control_successes} / {self.control_total} "
            f"= {_format_percent(self.control_rate)}",
            f"  Treatment (B): {self.treatment_successes} / {self.treatment_total} "
            f"= {_format_percent(self.treatment_rate)}",
            f"  Observed lift (B - A): {_format_pp(self.absolute_lift)} "
            "percentage points",
            f"  Relative lift: {relative_lift_str}",
            "",
            "Frequentist (two-sided pooled z-test)",
            f"  z statistic: {self.z_statistic:+.4f}",
            f"  p-value: {self.p_value:.4f}",
            f"  {confidence_pct:g}% Newcombe confidence interval for lift: "
            f"[{_format_pp(confidence_lower)}, {_format_pp(confidence_upper)}] "
            "percentage points",
            "",
            f"Bayesian (Beta({self.prior_alpha:g}, {self.prior_beta:g}) prior, "
            f"{self.n_simulations:,} sims)",
            f"  Posterior mean lift: {_format_pp(self.posterior_mean_lift)} "
            "percentage points",
            f"  Posterior median lift: {_format_pp(self.posterior_median_lift)} "
            "percentage points",
            f"  P(Treatment B > Control A): "
            f"{_format_probability(self.prob_treatment_better)}",
            f"  P(Control A > Treatment B): "
            f"{_format_probability(self.prob_control_better)}",
            f"  {credible_pct:g}% credible interval for lift: "
            f"[{_format_pp(credible_lower)}, {_format_pp(credible_upper)}] "
            "percentage points",
            "  Expected loss",
            f"    Choosing treatment B: "
            f"{self.expected_loss_treatment * 100:.2f} percentage points",
            f"    Choosing control A:   "
            f"{self.expected_loss_control * 100:.2f} percentage points",
        ]

        return "\n".join(lines)


@dataclass(frozen=True)
class ContinuousABResult:
    """Results from a continuous A/B test.

    Differences are always Treatment B - Control A.
    """

    n_simulations: int
    confidence_level: float
    credible_interval: float
    seed: int | None

    control_n: int
    treatment_n: int
    control_mean: float
    treatment_mean: float
    mean_difference: float

    t_statistic: float
    p_value: float
    degrees_of_freedom: float
    confidence_interval_bounds: tuple[float, float]

    posterior_mean_difference: float
    posterior_median_difference: float
    prob_treatment_better: float
    prob_control_better: float
    credible_interval_bounds: tuple[float, float]
    expected_loss_treatment: float
    expected_loss_control: float

    mean_difference_samples: np.ndarray = field(repr=False)

    def prob_difference_above(self, threshold: float) -> float:
        """Return the posterior probability that the difference exceeds a threshold."""
        return float(np.mean(self.mean_difference_samples > threshold))

    def prob_no_harm(self, margin: float = 0.0) -> float:
        """Return P(mean difference >= -margin | data)."""
        validation.validate_margin(margin)
        return float(np.mean(self.mean_difference_samples >= -margin))

    def prob_harm_above(self, margin: float = 0.0) -> float:
        """Return P(mean difference < -margin | data)."""
        validation.validate_margin(margin)
        return float(np.mean(self.mean_difference_samples < -margin))

    def plot_difference_distribution(
        self,
        ax: Any = None,
        *,
        bins: int = 50,
        density: bool = True,
        title: str | None = "Posterior Distribution of Mean Difference",
    ) -> Any:
        """Plot the posterior mean-difference distribution."""
        from . import plotting

        return plotting.plot_mean_difference_distribution(
            self.mean_difference_samples,
            self.posterior_median_difference,
            self.credible_interval_bounds,
            self.credible_interval,
            ax=ax,
            bins=bins,
            density=density,
            title=title,
        )

    def plot_probability_bar(
        self,
        ax: Any = None,
        *,
        title: str | None = "Posterior Probability of Being Better",
    ) -> Any:
        """Plot the probability that each group is better."""
        from . import plotting

        return plotting.plot_probability_bar(
            self.prob_treatment_better,
            self.prob_control_better,
            ax=ax,
            title=title,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return result fields as a dictionary."""
        return {
            "n_simulations": self.n_simulations,
            "confidence_level": self.confidence_level,
            "credible_interval": self.credible_interval,
            "seed": self.seed,
            "control_n": self.control_n,
            "treatment_n": self.treatment_n,
            "control_mean": self.control_mean,
            "treatment_mean": self.treatment_mean,
            "mean_difference": self.mean_difference,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "degrees_of_freedom": self.degrees_of_freedom,
            "confidence_interval_bounds": self.confidence_interval_bounds,
            "posterior_mean_difference": self.posterior_mean_difference,
            "posterior_median_difference": self.posterior_median_difference,
            "prob_treatment_better": self.prob_treatment_better,
            "prob_control_better": self.prob_control_better,
            "credible_interval_bounds": self.credible_interval_bounds,
            "expected_loss_treatment": self.expected_loss_treatment,
            "expected_loss_control": self.expected_loss_control,
        }

    def summary(self) -> str:
        """Return a formatted summary."""
        confidence_pct = self.confidence_level * 100
        credible_pct = self.credible_interval * 100

        confidence_lower, confidence_upper = self.confidence_interval_bounds
        credible_lower, credible_upper = self.credible_interval_bounds

        lines = [
            "Continuous A/B test result",
            "=" * 40,
            "Observed (difference is always Treatment B - Control A)",
            f"  Control (A):   n={self.control_n:,}, "
            f"mean={_format_number(self.control_mean)}",
            f"  Treatment (B): n={self.treatment_n:,}, "
            f"mean={_format_number(self.treatment_mean)}",
            f"  Observed mean difference (B - A): "
            f"{_format_signed_number(self.mean_difference)}",
            "",
            "Frequentist (two-sided Welch t-test)",
            f"  t statistic: {self.t_statistic:+.4f}",
            f"  degrees of freedom: {self.degrees_of_freedom:.4f}",
            f"  p-value: {self.p_value:.4f}",
            f"  {confidence_pct:g}% confidence interval for mean difference: "
            f"[{_format_signed_number(confidence_lower)}, "
            f"{_format_signed_number(confidence_upper)}]",
            "",
            f"Bayesian (Normal mean/variance reference-prior model, "
            f"{self.n_simulations:,} sims)",
            f"  Posterior mean difference: "
            f"{_format_signed_number(self.posterior_mean_difference)}",
            f"  Posterior median difference: "
            f"{_format_signed_number(self.posterior_median_difference)}",
            f"  P(Treatment B > Control A): "
            f"{_format_probability(self.prob_treatment_better)}",
            f"  P(Control A > Treatment B): "
            f"{_format_probability(self.prob_control_better)}",
            f"  {credible_pct:g}% credible interval for mean difference: "
            f"[{_format_signed_number(credible_lower)}, "
            f"{_format_signed_number(credible_upper)}]",
            "  Expected loss",
            f"    Choosing treatment B: "
            f"{_format_number(self.expected_loss_treatment)}",
            f"    Choosing control A:   "
            f"{_format_number(self.expected_loss_control)}",
        ]

        return "\n".join(lines)