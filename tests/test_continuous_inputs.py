"""Tests for ContinuousABTest input constructors."""

import numpy as np
import pandas as pd
import pytest

from abtestwise.continuous import ContinuousABTest


CONTROL = np.array([10.0, 11.5, 9.5, 12.0])
TREATMENT = np.array([12.0, 13.5, 11.0, 14.0, 12.5])


def test_from_samples_constructs():
    test = ContinuousABTest.from_samples(
        control=CONTROL,
        treatment=TREATMENT,
        seed=123,
    )

    np.testing.assert_array_equal(test.control, CONTROL)
    np.testing.assert_array_equal(test.treatment, TREATMENT)
    assert test.confidence_level == 0.95
    assert test.credible_interval == 0.95
    assert test.seed == 123


def test_from_samples_supports_unequal_sizes():
    test = ContinuousABTest.from_samples(
        control=CONTROL,
        treatment=TREATMENT,
    )

    assert test.control.size == 4
    assert test.treatment.size == 5


def test_from_dataframe_matches_from_samples():
    df = pd.DataFrame(
        {
            "variant": ["control"] * 4 + ["treatment"] * 5,
            "metric": np.concatenate([CONTROL, TREATMENT]),
        }
    )

    from_dataframe = ContinuousABTest.from_dataframe(
        df,
        group_col="variant",
        outcome_col="metric",
        control="control",
        treatment="treatment",
        confidence_level=0.90,
        credible_interval=0.95,
        seed=123,
    )

    from_samples = ContinuousABTest.from_samples(
        CONTROL,
        TREATMENT,
        confidence_level=0.90,
        credible_interval=0.95,
        seed=123,
    )

    np.testing.assert_array_equal(
        from_dataframe.control,
        from_samples.control,
    )
    np.testing.assert_array_equal(
        from_dataframe.treatment,
        from_samples.treatment,
    )
    assert from_dataframe.n_simulations == from_samples.n_simulations
    assert from_dataframe.confidence_level == from_samples.confidence_level
    assert from_dataframe.credible_interval == from_samples.credible_interval
    assert from_dataframe.seed == from_samples.seed


def test_from_dataframe_ignores_unselected_groups():
    df = pd.DataFrame(
        {
            "variant": [
                "control",
                "control",
                "treatment",
                "treatment",
                "other",
                "other",
            ],
            "metric": [1.0, 2.0, 3.0, 5.0, 100.0, 200.0],
        }
    )

    test = ContinuousABTest.from_dataframe(
        df,
        group_col="variant",
        outcome_col="metric",
        control="control",
        treatment="treatment",
    )

    np.testing.assert_array_equal(test.control, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(test.treatment, np.array([3.0, 5.0]))


def test_from_dataframe_rejects_same_group_labels():
    df = pd.DataFrame(
        {
            "variant": ["control", "control", "control"],
            "metric": [1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="different groups"):
        ContinuousABTest.from_dataframe(
            df,
            group_col="variant",
            outcome_col="metric",
            control="control",
            treatment="control",
        )


def test_from_dataframe_rejects_missing_group():
    df = pd.DataFrame(
        {
            "variant": ["control", "control", "control"],
            "metric": [1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="treatment group"):
        ContinuousABTest.from_dataframe(
            df,
            group_col="variant",
            outcome_col="metric",
            control="control",
            treatment="treatment",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence_level": 0.0},
        {"confidence_level": 1.0},
        {"credible_interval": 0.0},
        {"credible_interval": 1.0},
        {"n_simulations": 0},
        {"seed": -1},
    ],
)
def test_from_samples_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        ContinuousABTest.from_samples(
            CONTROL,
            TREATMENT,
            **kwargs,
        )