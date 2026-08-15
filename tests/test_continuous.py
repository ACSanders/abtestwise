"""End-to-end tests for ContinuousABTest and ContinuousABResult."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from abtestwise import ContinuousABResult, ContinuousABTest


CONTROL = np.array([10.0, 11.5, 9.5, 12.0, 10.5])
TREATMENT = np.array([12.0, 13.5, 11.0, 14.0, 12.5, 13.0])


def build():
    return ContinuousABTest.from_samples(
        CONTROL,
        TREATMENT,
        n_simulations=50_000,
        confidence_level=0.95,
        credible_interval=0.95,
        seed=42,
    )


def test_run_returns_result():
    result = build().run()

    assert isinstance(result, ContinuousABResult)


def test_observed_quantities():
    result = build().run()

    assert result.control_n == CONTROL.size
    assert result.treatment_n == TREATMENT.size
    assert result.control_mean == pytest.approx(np.mean(CONTROL))
    assert result.treatment_mean == pytest.approx(np.mean(TREATMENT))
    assert result.mean_difference == pytest.approx(
        np.mean(TREATMENT) - np.mean(CONTROL)
    )


def test_welch_results_match_scipy():
    result = build().run()

    expected = stats.ttest_ind(
        TREATMENT,
        CONTROL,
        equal_var=False,
        alternative="two-sided",
    )
    expected_interval = expected.confidence_interval(
        confidence_level=0.95,
    )

    assert result.t_statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)
    assert result.degrees_of_freedom == pytest.approx(expected.df)
    assert result.confidence_interval_bounds[0] == pytest.approx(
        expected_interval.low
    )
    assert result.confidence_interval_bounds[1] == pytest.approx(
        expected_interval.high
    )


def test_treatment_minus_control_orientation():
    result = build().run()

    assert result.mean_difference > 0
    assert result.t_statistic > 0
    assert result.posterior_mean_difference > 0
    assert result.posterior_median_difference > 0


def test_reproducible_end_to_end():
    a = build().run()
    b = build().run()

    np.testing.assert_array_equal(
        a.mean_difference_samples,
        b.mean_difference_samples,
    )

    assert a.posterior_mean_difference == b.posterior_mean_difference
    assert a.posterior_median_difference == b.posterior_median_difference
    assert a.prob_treatment_better == b.prob_treatment_better


def test_posterior_sample_shape():
    result = build().run()

    assert result.mean_difference_samples.shape == (50_000,)


def test_probabilities_in_range_and_consistent():
    result = build().run()

    assert 0.0 <= result.prob_treatment_better <= 1.0
    assert 0.0 <= result.prob_control_better <= 1.0
    assert (
        result.prob_treatment_better + result.prob_control_better
        == pytest.approx(1.0)
    )


def test_prob_difference_above_zero_equals_prob_treatment_better():
    result = build().run()

    assert result.prob_difference_above(0.0) == result.prob_treatment_better


def test_prob_difference_above_uses_native_units():
    result = build().run()

    assert result.prob_difference_above(1.0) <= result.prob_difference_above(0.5)
    assert result.prob_difference_above(0.5) <= result.prob_difference_above(0.0)


@pytest.mark.parametrize("margin", [0.0, 0.5, 1.0])
def test_no_harm_and_harm_above_are_complementary(margin):
    result = build().run()

    total = result.prob_no_harm(margin) + result.prob_harm_above(margin)

    assert total == pytest.approx(1.0)


def test_prob_no_harm_non_decreasing_in_margin():
    result = build().run()

    assert result.prob_no_harm(0.0) <= result.prob_no_harm(0.5)
    assert result.prob_no_harm(0.5) <= result.prob_no_harm(1.0)


def test_prob_harm_above_non_increasing_in_margin():
    result = build().run()

    assert result.prob_harm_above(0.0) >= result.prob_harm_above(0.5)
    assert result.prob_harm_above(0.5) >= result.prob_harm_above(1.0)


def test_no_harm_methods_reject_invalid_margin():
    result = build().run()

    with pytest.raises(ValueError):
        result.prob_no_harm(-0.1)

    with pytest.raises(ValueError):
        result.prob_harm_above(-0.1)


def test_expected_loss_is_in_native_units():
    result = build().run()

    assert result.expected_loss_treatment >= 0.0
    assert result.expected_loss_control >= 0.0


def test_credible_interval_brackets_posterior_median():
    result = build().run()
    lower, upper = result.credible_interval_bounds

    assert lower < result.posterior_median_difference < upper


def test_custom_interval_levels_are_preserved():
    result = ContinuousABTest.from_samples(
        CONTROL,
        TREATMENT,
        confidence_level=0.90,
        credible_interval=0.80,
        n_simulations=20_000,
        seed=42,
    ).run()

    assert result.confidence_level == 0.90
    assert result.credible_interval == 0.80


def test_to_dict_contains_continuous_results_and_excludes_samples():
    result = build().run()
    values = result.to_dict()

    assert values["control_n"] == CONTROL.size
    assert values["treatment_n"] == TREATMENT.size
    assert "mean_difference" in values
    assert "t_statistic" in values
    assert "degrees_of_freedom" in values
    assert "confidence_interval_bounds" in values
    assert "posterior_mean_difference" in values
    assert "credible_interval_bounds" in values
    assert "expected_loss_treatment" in values
    assert "expected_loss_control" in values
    assert "mean_difference_samples" not in values


def test_summary_contains_frequentist_and_bayesian_sections():
    text = build().run().summary()

    assert "Continuous A/B test result" in text
    assert "Treatment B - Control A" in text
    assert "Welch t-test" in text
    assert "95% confidence interval for mean difference" in text
    assert "95% credible interval for mean difference" in text
    assert "P(Treatment B > Control A)" in text
    assert "Expected loss" in text