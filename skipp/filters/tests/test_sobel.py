import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_allclose)

from skipp.skipp.filters import sobel_h as skipp_sobel_h
from skipp.skipp.filters import sobel_v as skipp_sobel_v
from skimage.filters import sobel_h as skimage_sobel_h
from skimage.filters import sobel_v as skimage_sobel_h


def test_sobel_h_zeros():
    """Horizontal sobel on an array of all zeros."""
    result = skipp_sobel_h(np.zeros((10, 10), dtype=np.float32))
    assert (np.all(result == 0))


def test_sobel_v_zeros():
    """Vertical sobel on an array of all zeros."""
    result = skipp_sobel_v(np.zeros((10, 10), dtype=np.float32))
    assert_allclose(result, 0)


@pytest.mark.skip(reason="deveoloping _mask_filter_result in progress")
def test_sobel_v_mask():
    """Vertical Sobel on a masked array should be zero."""
    result = skipp_sobel_v(np.random.uniform(size=(10, 10)),
                           np.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)


@pytest.mark.skip(reason="deveoloping _mask_filter_result in progress")
def test_sobel_h_mask():
    """Horizontal Sobel on a masked array should be zero."""
    result = skipp_sobel_h(np.random.uniform(size=(10, 10)),
                           np.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)

# TODO
# add tests
