"""Tests for BinaryABTest input constructors."""

import numpy as np
import pandas as pd
import pytest

from abtestwise import BinaryABTest


CONTROL = np.array([0, 1, 0, 1, 1, 0, 0, 1])
TREATMENT = np.array([1, 1, 0, 1, 1, 0, 1, 1])


def test_from_samples_matches_from_counts():
    from_counts = BinaryABTest.from_counts(
        control_successes=4,
        control_total=8,
        treatment_successes=6,
        treatment_total=8,
        confidence_level=0.90,
        seed=123,
    )

    from_samples = BinaryABTest.from_samples(
        control=CONTROL,
        treatment=TREATMENT,
        confidence_level=0.90,
        seed=123,
    )

    assert from_samples == from_counts


def test_from_dataframe_matches_from_samples():
    df = pd.DataFrame(
        {
            "variant": ["control"] * 8 + ["treatment"] * 8,
            "converted": np.concatenate([CONTROL, TREATMENT]),
        }
    )

    from_dataframe = BinaryABTest.from_dataframe(
        df,
        group_col="variant",
        outcome_col="converted",
        control="control",
        treatment="treatment",
        confidence_level=0.90,
        seed=123,
    )

    from_samples = BinaryABTest.from_samples(
        control=CONTROL,
        treatment=TREATMENT,
        confidence_level=0.90,
        seed=123,
    )

    assert from_dataframe == from_samples


def test_all_constructors_produce_identical_results():
    df = pd.DataFrame(
        {
            "variant": ["control"] * 8 + ["treatment"] * 8,
            "converted": np.concatenate([CONTROL, TREATMENT]),
        }
    )

    tests = [
        BinaryABTest.from_counts(
            4,
            8,
            6,
            8,
            confidence_level=0.90,
            seed=123,
        ),
        BinaryABTest.from_samples(
            CONTROL,
            TREATMENT,
            confidence_level=0.90,
            seed=123,
        ),
        BinaryABTest.from_dataframe(
            df,
            group_col="variant",
            outcome_col="converted",
            control="control",
            treatment="treatment",
            confidence_level=0.90,
            seed=123,
        ),
    ]

    results = [test.run() for test in tests]

    assert results[0].z_statistic == results[1].z_statistic == results[2].z_statistic
    assert results[0].p_value == results[1].p_value == results[2].p_value
    assert (
        results[0].confidence_interval_bounds
        == results[1].confidence_interval_bounds
        == results[2].confidence_interval_bounds
    )
    assert (
        results[0].prob_treatment_better
        == results[1].prob_treatment_better
        == results[2].prob_treatment_better
    )

    np.testing.assert_array_equal(
        results[0].lift_samples,
        results[1].lift_samples,
    )
    np.testing.assert_array_equal(
        results[0].lift_samples,
        results[2].lift_samples,
    )


def test_from_samples_propagates_confidence_level():
    test = BinaryABTest.from_samples(
        control=CONTROL,
        treatment=TREATMENT,
        confidence_level=0.90,
    )

    assert test.confidence_level == 0.90


def test_from_dataframe_propagates_confidence_level():
    df = pd.DataFrame(
        {
            "variant": ["control"] * 8 + ["treatment"] * 8,
            "converted": np.concatenate([CONTROL, TREATMENT]),
        }
    )

    test = BinaryABTest.from_dataframe(
        df,
        group_col="variant",
        outcome_col="converted",
        control="control",
        treatment="treatment",
        confidence_level=0.90,
    )

    assert test.confidence_level == 0.90


def test_from_samples_accepts_boolean_values():
    test = BinaryABTest.from_samples(
        control=[False, True, False, True],
        treatment=[True, True, False, True],
    )

    assert test.control_successes == 2
    assert test.control_total == 4
    assert test.treatment_successes == 3
    assert test.treatment_total == 4


@pytest.mark.parametrize(
    "bad_sample",
    [
        [],
        [0, 1, 2],
        [0, 1, np.nan],
        [[0, 1], [1, 0]],
    ],
)
def test_from_samples_rejects_invalid_binary_data(bad_sample):
    with pytest.raises(ValueError):
        BinaryABTest.from_samples(
            control=bad_sample,
            treatment=[0, 1],
        )


def test_from_dataframe_rejects_same_group_labels():
    df = pd.DataFrame(
        {
            "variant": ["control", "control"],
            "converted": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="different groups"):
        BinaryABTest.from_dataframe(
            df,
            group_col="variant",
            outcome_col="converted",
            control="control",
            treatment="control",
        )


def test_from_dataframe_rejects_missing_group():
    df = pd.DataFrame(
        {
            "variant": ["control", "control"],
            "converted": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="treatment group"):
        BinaryABTest.from_dataframe(
            df,
            group_col="variant",
            outcome_col="converted",
            control="control",
            treatment="treatment",
        )


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
            "converted": [0, 1, 1, 1, 0, 0],
        }
    )

    test = BinaryABTest.from_dataframe(
        df,
        group_col="variant",
        outcome_col="converted",
        control="control",
        treatment="treatment",
    )

    assert test.control_successes == 1
    assert test.control_total == 2
    assert test.treatment_successes == 2
    assert test.treatment_total == 2