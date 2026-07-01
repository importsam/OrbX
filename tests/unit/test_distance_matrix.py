import numpy as np
import pytest

from orbx.tools.distance_matrix import _validate_matrix


def test_validate_matrix_rejects_non_square():
    with pytest.raises(ValueError, match="not square"):
        _validate_matrix(np.ones((2, 3)))


def test_validate_matrix_zeroes_negative_values():
    matrix = np.array([[0.0, -2.0], [-2.0, 0.0]])

    assert _validate_matrix(matrix) is True
    assert np.all(matrix >= 0)


def test_validate_matrix_symmetrizes_asymmetric_input():
    matrix = np.array([[0.0, 1.0], [2.0, 0.0]])

    _validate_matrix(matrix)

    assert np.allclose(matrix, matrix.T)
