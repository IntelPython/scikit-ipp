import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_allclose)

from skipp.filters import laplace as skipp_laplace
from skimage.filters import laplace as skimage_laplace


@pytest.fixture
def image():
    return np.array([[1, 2, 3, 2, 1],
                     [1, 1, 2, 2, 3],
                     [3, 2, 1, 2, 1],
                     [3, 2, 1, 1, 1],
                     [1, 2, 1, 2, 3]],
                    dtype=np.uint8)


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

# TODO
@pytest.mark.skip(reason="deveoloping _mask_filter_result in progress")
def test_laplace_mask():
    """Laplace on a masked array should be zero."""
    # Create a synthetic 2D image
    image = np.zeros((9, 9), dtype=np.float32)
    image[3:-3, 3:-3] = 1
    # Define the mask
    result = skipp_laplace(image, ksize=3, mask=np.zeros((9, 9), dtype=bool))
    assert (np.all(result == 0))


def test_laplace_skimage_similarity(image):
    """
    # Testing scikit-image's and scikit-ipp's laplace filtering results
    # for float32 input/output dtypes
    """
    image = image.astype(np.float32)
    skimage_laplace_result = skimage_laplace(image)
    skipp_laplace_result = skipp_laplace(image)
    assert_array_almost_equal(skimage_laplace_result, skipp_laplace_result)

# TODO
# similarity test for 3 channel image and other dtypes
