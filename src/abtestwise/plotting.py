"""Plotting for binary A/B test results.

Lift is always Treatment B - Control A. Plots show the lift in percentage points,
so a raw lift of 0.025 will be displayed as +2.5 percentage points.

These functions return matplotlib Axes objects. Note that they do not call plt.show()
or save files automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

# Raw lift is a proportion difference; multiply by 100 to get percentage points.
_PCT_POINTS = 100.0


def plot_lift_distribution(
    lift_samples: np.ndarray,
    median_lift: float,
    credible_interval_bounds: tuple[float, float],
    credible_interval: float,
    *,
    ax: "Axes | None" = None,
    bins: int = 50,
    density: bool = True,
    title: str | None = "Posterior Distribution of Lift",
) -> "Axes":
    """Histogram of the posterior lift distribution (Treatment B - Control A).

    Marks zero, the posterior median, and the credible-interval bounds. All
    lift values are shown in percentage points.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    samples_pp = np.asarray(lift_samples) * _PCT_POINTS
    median_pp = median_lift * _PCT_POINTS
    lower_pp = credible_interval_bounds[0] * _PCT_POINTS
    upper_pp = credible_interval_bounds[1] * _PCT_POINTS
    ci_pct = credible_interval * 100

    ax.hist(
        samples_pp,
        bins=bins,
        density=density,
        color="#4C72B0",
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
    )

    # Zero reference: where Treatment B and Control A are equal.
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1.5, label="No difference")
    # Posterior median lift.
    ax.axvline(
        median_pp,
        color="#C44E52",
        linestyle="-",
        linewidth=2.0,
        label=f"Median {median_pp:+.2f} pp",
    )
    # Credible interval bounds.
    ax.axvline(
        lower_pp,
        color="#55A868",
        linestyle=":",
        linewidth=1.8,
        label=f"{ci_pct:g}% CI [{lower_pp:+.2f}, {upper_pp:+.2f}] pp",
    )
    ax.axvline(upper_pp, color="#55A868", linestyle=":", linewidth=1.8)

    ax.set_xlabel("Lift: Treatment B - Control A (percentage points)")
    ax.set_ylabel("Density" if density else "Frequency")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.margins(x=0.02)

    return ax


def plot_probability_bar(
    prob_treatment_better: float,
    prob_control_better: float,
    *,
    ax: "Axes | None" = None,
    title: str | None = "Posterior Probability of Being Better",
) -> "Axes":
    """Two-bar chart comparing P(Treatment B better) vs P(Control A better)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    labels = ["Treatment B better", "Control A better"]
    values = [prob_treatment_better, prob_control_better]
    colors = ["#4C72B0", "#C44E52"]

    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="white")

    # Percentage labels. For tall bars (near the top boundary) we place the
    # label *inside* the bar in white so it does not overlap the chart top; for
    # shorter bars we place it just above the bar in dark text.
    high_bar_threshold = 0.9
    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        if value >= high_bar_threshold:
            ax.text(
                x,
                value - 0.03,
                f"{value:.1%}",
                ha="center",
                va="top",
                fontsize=11,
                fontweight="bold",
                color="white",
            )
        else:
            ax.text(
                x,
                value + 0.02,
                f"{value:.1%}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#222222",
            )

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Posterior probability")
    if title is not None:
        ax.set_title(title)

    return ax
