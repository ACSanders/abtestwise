"""Smoke tests for the plotting methods.

These confirm each method runs and returns a matplotlib ``Axes``. We use the
non-interactive "Agg" backend so the tests work in headless CI with no display
and never open a window.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from abtestwise import BinaryABTest


def make_result():
    return BinaryABTest.from_counts(
        control_successes=120,
        control_total=1000,
        treatment_successes=145,
        treatment_total=1000,
        n_simulations=5_000,
        seed=42,
    ).run()


def test_plot_lift_distribution_returns_axes():
    result = make_result()
    ax = result.plot_lift_distribution()
    assert isinstance(ax, Axes)
    plt.close("all")


def test_plot_probability_bar_returns_axes():
    result = make_result()
    ax = result.plot_probability_bar()
    assert isinstance(ax, Axes)
    plt.close("all")


def test_plot_lift_distribution_accepts_existing_ax():
    result = make_result()
    fig, ax = plt.subplots()
    returned = result.plot_lift_distribution(ax=ax)
    assert returned is ax
    plt.close(fig)


def test_plot_probability_bar_accepts_existing_ax():
    result = make_result()
    fig, ax = plt.subplots()
    returned = result.plot_probability_bar(ax=ax)
    assert returned is ax
    plt.close(fig)


def test_plot_lift_distribution_accepts_optional_params():
    result = make_result()
    ax = result.plot_lift_distribution(bins=30, density=False, title=None)
    assert isinstance(ax, Axes)
    plt.close("all")


def test_probability_bar_y_axis_is_zero_to_one():
    result = make_result()
    ax = result.plot_probability_bar()
    assert ax.get_ylim() == (0.0, 1.0)
    plt.close("all")
