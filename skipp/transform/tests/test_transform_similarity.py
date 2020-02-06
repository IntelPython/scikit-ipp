import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_allclose,
                           assert_array_equal, assert_allclose, assert_almost_equal)
from skimage._shared.testing import (assert_equal,TestCase, parametrize)

import skipp.skipp.transform
import skimage.transform

# TODO
# add checks
# "image_dtype", [np.float32])
# @pytest.mark.parametrize("preserve_range", [True, False])
@pytest.mark.skip(reason="in progress")
@pytest.mark.parametrize("image_dtype", [np.float32])
@pytest.mark.parametrize("preserve_range", [True])
@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("angle", [i for i in range(0, 91, 1)])
def test_rotate_skimage_similarity(image_dtype, angle, order, preserve_range):
    """
    # Testing scikit-image's and scikit-ipp's rotate transform results
    """
    #image = np.arange(50*40, dtype=image_dtype).reshape((50, 40))
    image = np.random.RandomState(0).randn(40, 50).astype(np.float32)
    skimage_rotate = skimage.transform.rotate(image, angle=angle, order=order, preserve_range=preserve_range)
    skipp_rotate = skipp.skipp.transform.rotate(image, angle=angle, order=order, preserve_range=preserve_range)
    assert_allclose(skipp_rotate, skimage_rotate, rtol=1e-05)
