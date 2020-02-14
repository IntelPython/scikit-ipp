import pytest
import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_array_less, assert_array_almost_equal_nulp,
                           assert_equal, TestCase, assert_allclose,
                           assert_almost_equal, assert_, assert_warns,
                           assert_no_warnings)
from skimage.filters import median as skimage_median
from skipp.filters import median as skipp_median

@pytest.fixture
def image():
    return np.array([[1, 2, 3, 2, 1],
                     [1, 1, 2, 2, 3],
                     [3, 2, 1, 2, 1],
                     [3, 2, 1, 1, 1],
                     [1, 2, 1, 2, 3]],
                    dtype=np.uint8)

# TODO
# for all dtypes
@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint16, np.int16, np.float32]
)
def test_median_preserve_dtype(image, dtype):
    skipp_median_image = skipp_median(image.astype(dtype), selem = np.ones((3,3), dtype=np.bool_), behavior='ipp')
    assert skipp_median_image.dtype == dtype

# TODO
# for all dtypes
@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint16, np.int16, np.float32]
)
def test_median_skimage_similarity(image, dtype):
    """
    # Testing scikit-image's and scikit-ipp's median filtering results
    """
    skimage_median_result = skimage_median(image.astype(dtype), selem = np.ones((3,3), dtype=np.bool_), behavior='ndimage')
    skipp_median_result = skipp_median(image.astype(dtype), selem = np.ones((3,3), dtype=np.bool_), behavior='ipp')
    assert_array_almost_equal(skimage_median_result, skipp_median_result, decimal=3)
