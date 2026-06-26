"""Result object for binary A/B tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import math

import numpy as np


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


@dataclass(frozen=True)
class BinaryABResult:
    """Results from a binary A/B test.

    Lift is always Treatment B - Control A.
    """

    # Inputs
    control_successes: int
    control_total: int
    treatment_successes: int
    treatment_total: int
    prior_alpha: float
    prior_beta: float
    n_simulations: int
    credible_interval: float
    seed: int | None

    # Observed and frequentist results
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift: float
    z_statistic: float
    p_value: float

    # Bayesian results
    posterior_mean_lift: float
    posterior_median_lift: float
    prob_treatment_better: float
    prob_control_better: float
    credible_interval_bounds: tuple[float, float]
    expected_loss_treatment: float
    expected_loss_control: float

    # Posterior samples
    lift_samples: np.ndarray = field(repr=False)

    def prob_lift_above(self, threshold: float) -> float:
        """Return the posterior probability that lift is above a threshold."""
        return float(np.mean(self.lift_samples > threshold))

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
            "credible_interval": self.credible_interval,
            "seed": self.seed,
            "control_rate": self.control_rate,
            "treatment_rate": self.treatment_rate,
            "absolute_lift": self.absolute_lift,
            "relative_lift": self.relative_lift,
            "z_statistic": self.z_statistic,
            "p_value": self.p_value,
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
        ci_pct = self.credible_interval * 100
        lower, upper = self.credible_interval_bounds

        # Relative lift is a ratio, not percentage points.
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
            f"  {ci_pct:g}% credible interval for lift: "
            f"[{_format_pp(lower)}, {_format_pp(upper)}] percentage points",
            "  Expected loss",
            f"    Choosing treatment B: "
            f"{self.expected_loss_treatment * 100:.2f} percentage points",
            f"    Choosing control A:   "
            f"{self.expected_loss_control * 100:.2f} percentage points",
        ]
        return "\n".join(lines)