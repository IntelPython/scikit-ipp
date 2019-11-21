import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_allclose)

from skipp.skipp.filters import laplace as skipp_laplace
from skimage.filters import laplace as skimage_laplace


def test_laplace_zeros():
    """Laplace on a square image."""
    # Create a synthetic 2D image
    image = np.zeros((9, 9), dtype=np.float32)
    image[3:-3, 3:-3] = 1
    result = skipp_laplace(image)
    check_result = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., -1., -1., -1., 0., 0., 0.],
                             [0., 0., -1., 2., 1., 2., -1., 0., 0.],
                             [0., 0., -1., 1., 0., 1., -1., 0., 0.],
                             [0., 0., -1., 2., 1., 2., -1., 0., 0.],
                             [0., 0., 0., -1., -1., -1., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    assert_allclose(result, check_result)