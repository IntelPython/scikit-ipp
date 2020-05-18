import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

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