"""Smoke tests for plotting methods."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from abtestwise import BinaryABTest, ContinuousABTest


def make_binary_result():
    return BinaryABTest.from_counts(
        control_successes=120,
        control_total=1000,
        treatment_successes=145,
        treatment_total=1000,
        n_simulations=5_000,
        seed=42,
    ).run()


def make_continuous_result():
    control = np.array([10.0, 11.5, 9.5, 12.0, 10.5])
    treatment = np.array([12.0, 13.5, 11.0, 14.0, 12.5, 13.0])

    return ContinuousABTest.from_samples(
        control,
        treatment,
        n_simulations=5_000,
        seed=42,
    ).run()


def test_binary_plot_lift_distribution_returns_axes():
    result = make_binary_result()
    ax = result.plot_lift_distribution()

    assert isinstance(ax, Axes)

    plt.close("all")


def test_binary_plot_probability_bar_returns_axes():
    result = make_binary_result()
    ax = result.plot_probability_bar()

    assert isinstance(ax, Axes)

    plt.close("all")


def test_binary_plot_lift_distribution_accepts_existing_ax():
    result = make_binary_result()
    fig, ax = plt.subplots()

    returned = result.plot_lift_distribution(ax=ax)

    assert returned is ax

    plt.close(fig)


def test_binary_plot_probability_bar_accepts_existing_ax():
    result = make_binary_result()
    fig, ax = plt.subplots()

    returned = result.plot_probability_bar(ax=ax)

    assert returned is ax

    plt.close(fig)


def test_binary_plot_lift_distribution_accepts_optional_params():
    result = make_binary_result()
    ax = result.plot_lift_distribution(
        bins=30,
        density=False,
        title=None,
    )

    assert isinstance(ax, Axes)

    plt.close("all")


def test_binary_probability_bar_y_axis_is_zero_to_one():
    result = make_binary_result()
    ax = result.plot_probability_bar()

    assert ax.get_ylim() == (0.0, 1.0)

    plt.close("all")


def test_continuous_plot_difference_distribution_returns_axes():
    result = make_continuous_result()
    ax = result.plot_difference_distribution()

    assert isinstance(ax, Axes)

    plt.close("all")


def test_continuous_plot_probability_bar_returns_axes():
    result = make_continuous_result()
    ax = result.plot_probability_bar()

    assert isinstance(ax, Axes)

    plt.close("all")


def test_continuous_plot_difference_distribution_accepts_existing_ax():
    result = make_continuous_result()
    fig, ax = plt.subplots()

    returned = result.plot_difference_distribution(ax=ax)

    assert returned is ax

    plt.close(fig)


def test_continuous_plot_probability_bar_accepts_existing_ax():
    result = make_continuous_result()
    fig, ax = plt.subplots()

    returned = result.plot_probability_bar(ax=ax)

    assert returned is ax

    plt.close(fig)


def test_continuous_plot_difference_distribution_accepts_optional_params():
    result = make_continuous_result()
    ax = result.plot_difference_distribution(
        bins=30,
        density=False,
        title=None,
    )

    assert isinstance(ax, Axes)

    plt.close("all")


def test_continuous_probability_bar_y_axis_is_zero_to_one():
    result = make_continuous_result()
    ax = result.plot_probability_bar()

    assert ax.get_ylim() == (0.0, 1.0)

    plt.close("all")


def test_continuous_difference_plot_uses_native_units():
    result = make_continuous_result()
    ax = result.plot_difference_distribution()

    assert ax.get_xlabel() == "Mean difference: Treatment B - Control A"

    plt.close("all")


def test_plot_methods_do_not_call_show(monkeypatch):
    calls = []

    def fake_show(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(plt, "show", fake_show)

    binary_result = make_binary_result()
    continuous_result = make_continuous_result()

    binary_result.plot_lift_distribution()
    binary_result.plot_probability_bar()
    continuous_result.plot_difference_distribution()
    continuous_result.plot_probability_bar()

    assert calls == []

    plt.close("all")