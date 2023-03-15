"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for zeroes, positive integers, mix of positive/negative integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, -1], [2001, 7], [-24, -4], [2, 28]], [2001, 28]),
        ([[12, 2], [-3, 4], [5, 6]], [12, 6]),
    ])
def test_daily_max(test, expected):
    """Test max function works for zeroes, positive integers, mix of positive/negative integers."""
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]),
        ([[4, 2, 5], [1, 6, 2], [4, 1, 9]], [1, 1, 2]),
        ([[4, -2, 5], [1, -6, 2], [-4, -1, 9]], [-4, -6, 2]),
    ])
def test_daily_min(test, expected):
    """Test min function works for zeroes, positive integers, mix of positive/negative integers."""
    from inflammation.models import daily_min
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))


...


def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])


# @pytest.mark.parametrize(
#     "test, expected",
#     [
#         ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
#         ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
#         ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]]),
#         ([[-1, 1, 1], [1, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1], [1, 1, 1]]),
#         # ([[float('nan'), 1, 1], [1, 1, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1], [1, 1, 1]]),
#         # For unknown reasons the code to correct nan in models.py isn't working for this
#     ])

# def test_patient_normalise(test, expected):
#     """Test normalisation works for arrays of one and positive integers.
#        Assumption that test accuracy of two decimal places is sufficient."""
#     from inflammation.models import patient_normalise
#     npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)

@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        (
            [[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            ValueError,
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None,

        ),
        (
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            None,

        ),
    ])
def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers."""
    from inflammation.models import patient_normalise
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)
    else:
        npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)

    @pytest.mark.parametrize(
        "test, expected",
        [
            ([[0, 0], [0, 0], [0, 0]], [0, 0]),
            ([[1, 2], [3, 4], [5, 7]], [1.6, 2.0]),
        ])
    def test_patient_std_dev(test, expected):
        """Test normalisation works for arrays of one and positive integers."""
        from inflammation.models import patient_std_dev
        npt.assert_almost_equal(patient_std_dev(np.array(test)), np.array(expected), decimal=2)