"""Tests for statistics functions within the Model layer."""

import numpy as np
import math
import numpy.testing as npt
import pytest
from unittest.mock import Mock

from inflammation.models import daily_mean, daily_min, daily_max, patient_normalise

def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0],[0, 0],[0, 0]], [0, 0]),
        ([[1, 2],[3, 4],[5, 6]], [3, 4]),
    ]
)
def test_daily_mean(test, expected):

    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


def test_daily_min_string():
    """Test that the min function fails if passed a string"""

    with pytest.raises(TypeError):
        error_expected = daily_min(np.array([["1", "2"],
                                             ["3", "4"],
                                             ["5", "6"]]))


def test_daily_min():
    """Test that the min of array is correct"""

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = [1,2]

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)


def test_daily_max():
    """Test that the max of array is correct"""

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = [5, 6]

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), test_result)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]])
    ])
def test_patient_normalise(test, expected):
    """Test normalisation works for arrays of one and positive integers.
       Test with a relative and absolute tolerance of 0.01."""

    result = patient_normalise(np.array(test))
    npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)
def test_compute_data_mock_source():
    from inflammation.compute_data import analyse_data
    data_source = Mock()
    data_source.load_inflammation_data.return_value = [[[0, 2, 0]],[[0, 1, 0]]]

    result = analyse_data(data_source)
    print(result)
    print(type(result))
    npt.assert_array_almost_equal(result, [0, math.sqrt(0.25), 0])