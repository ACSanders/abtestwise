"""End-to-end tests for ``BinaryABTest.run()`` and ``BinaryABResult``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from abtestwise import BinaryABResult, BinaryABTest


def build():
    return BinaryABTest.from_counts(
        control_successes=120,
        control_total=1000,
        treatment_successes=145,
        treatment_total=1000,
        prior_alpha=1,
        prior_beta=1,
        n_simulations=50_000,
        confidence_level=0.95,
        credible_interval=0.95,
        seed=42,
    )


def test_run_returns_result():
    result = build().run()
    assert isinstance(result, BinaryABResult)


def test_observed_quantities():
    result = build().run()
    assert result.control_rate == 120 / 1000
    assert result.treatment_rate == 145 / 1000
    assert math.isclose(result.absolute_lift, 0.025)
    assert math.isclose(result.relative_lift, 0.025 / 0.120)


def test_default_confidence_level():
    test = BinaryABTest.from_counts(120, 1000, 145, 1000)
    assert test.confidence_level == 0.95


def test_custom_confidence_level():
    test = BinaryABTest.from_counts(
        120,
        1000,
        145,
        1000,
        confidence_level=0.90,
    )
    assert test.confidence_level == 0.90


def test_invalid_confidence_level_rejected():
    with pytest.raises(ValueError, match="confidence_level"):
        BinaryABTest.from_counts(
            120,
            1000,
            145,
            1000,
            confidence_level=1.0,
        )


def test_confidence_interval_contains_observed_lift():
    result = build().run()
    lower, upper = result.confidence_interval_bounds

    assert lower < result.absolute_lift < upper


def test_custom_confidence_level_changes_interval_width():
    result_95 = BinaryABTest.from_counts(
        120,
        1000,
        145,
        1000,
        confidence_level=0.95,
        n_simulations=2000,
        seed=1,
    ).run()

    result_90 = BinaryABTest.from_counts(
        120,
        1000,
        145,
        1000,
        confidence_level=0.90,
        n_simulations=2000,
        seed=1,
    ).run()

    width_95 = (
        result_95.confidence_interval_bounds[1]
        - result_95.confidence_interval_bounds[0]
    )
    width_90 = (
        result_90.confidence_interval_bounds[1]
        - result_90.confidence_interval_bounds[0]
    )

    assert width_90 < width_95


def test_lift_samples_length_matches_n_simulations():
    result = build().run()
    assert result.lift_samples.shape == (50_000,)


def test_reproducible_end_to_end():
    a = build().run()
    b = build().run()
    assert np.array_equal(a.lift_samples, b.lift_samples)
    assert a.posterior_mean_lift == b.posterior_mean_lift


def test_probabilities_in_range_and_consistent():
    result = build().run()
    assert 0.0 <= result.prob_treatment_better <= 1.0
    assert 0.0 <= result.prob_control_better <= 1.0
    assert abs(
        result.prob_treatment_better + result.prob_control_better - 1.0
    ) < 1e-6


def test_prob_lift_above_zero_equals_prob_treatment_better():
    result = build().run()
    assert result.prob_lift_above(0.0) == result.prob_treatment_better


def test_prob_lift_above_is_monotonic():
    result = build().run()
    assert result.prob_lift_above(-1.0) >= result.prob_lift_above(0.0)
    assert result.prob_lift_above(0.0) >= result.prob_lift_above(1.0)


def test_summary_is_nonempty_string():
    text = build().run().summary()
    assert isinstance(text, str)
    assert "Binary A/B test result" in text


def test_summary_uses_percent_and_percentage_points():
    text = build().run().summary()

    assert "%" in text
    assert "12.00%" in text
    assert "14.50%" in text
    assert "percentage points" in text
    assert "+2.50 percentage points" in text
    assert "Relative lift: +20.83%" in text


def test_summary_includes_newcombe_confidence_interval():
    text = build().run().summary()

    assert "95% Newcombe confidence interval for lift" in text


def test_summary_includes_credible_interval():
    text = build().run().summary()

    assert "95% credible interval for lift" in text


def test_summary_distinguishes_custom_interval_levels():
    text = BinaryABTest.from_counts(
        120,
        1000,
        145,
        1000,
        confidence_level=0.90,
        credible_interval=0.95,
        n_simulations=2000,
        seed=1,
    ).run().summary()

    assert "90% Newcombe confidence interval for lift" in text
    assert "95% credible interval for lift" in text


def test_summary_renders_nan_relative_lift_gracefully():
    text = BinaryABTest.from_counts(
        control_successes=0,
        control_total=1000,
        treatment_successes=50,
        treatment_total=1000,
        n_simulations=2000,
        seed=1,
    ).run().summary()

    assert "Relative lift: undefined" in text
    assert "nan" not in text.lower()


def test_to_dict_keeps_raw_decimal_values():
    d = build().run().to_dict()

    assert d["control_rate"] == 120 / 1000
    assert isinstance(d["absolute_lift"], float)
    assert math.isclose(d["absolute_lift"], 0.025)


def test_to_dict_includes_confidence_interval():
    d = build().run().to_dict()

    assert d["confidence_level"] == 0.95
    assert "confidence_interval_bounds" in d
    assert isinstance(d["confidence_interval_bounds"], tuple)
    assert len(d["confidence_interval_bounds"]) == 2


def test_to_dict_has_scalars_and_excludes_samples():
    d = build().run().to_dict()

    assert "lift_samples" not in d
    assert d["control_successes"] == 120
    assert "expected_loss_treatment" in d


def test_relative_lift_nan_when_control_rate_zero():
    result = BinaryABTest.from_counts(
        control_successes=0,
        control_total=1000,
        treatment_successes=50,
        treatment_total=1000,
        n_simulations=2000,
        seed=1,
    ).run()

    assert math.isnan(result.relative_lift)


@pytest.mark.parametrize("margin", [0.0, 0.005, 0.02])
def test_no_harm_and_harm_above_are_complementary(margin):
    result = build().run()
    total = result.prob_no_harm(margin) + result.prob_harm_above(margin)
    assert total == pytest.approx(1.0)


@pytest.mark.parametrize("margin", [0.0, 0.005, 0.02])
def test_no_harm_and_harm_above_in_unit_range(margin):
    result = build().run()
    assert 0.0 <= result.prob_no_harm(margin) <= 1.0
    assert 0.0 <= result.prob_harm_above(margin) <= 1.0


def test_prob_no_harm_non_decreasing_in_margin():
    result = build().run()
    assert result.prob_no_harm(0.0) <= result.prob_no_harm(0.005)
    assert result.prob_no_harm(0.005) <= result.prob_no_harm(0.02)


def test_prob_harm_above_non_increasing_in_margin():
    result = build().run()
    assert result.prob_harm_above(0.0) >= result.prob_harm_above(0.005)
    assert result.prob_harm_above(0.005) >= result.prob_harm_above(0.02)


def test_no_harm_methods_reject_invalid_margin():
    result = build().run()

    with pytest.raises(ValueError):
        result.prob_no_harm(-0.1)

    with pytest.raises(ValueError):
        result.prob_harm_above(-0.1)