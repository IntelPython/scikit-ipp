import pytest
import numpy as np
# from skipp.skipp.filters import (median, median_1)

# TODO
@pytest.mark.skip(reason="in progress")
def test_median_unsupported_behavior():
    image = np.arange(5*5, dtype=np.uint8).reshape((5, 5))
    with pytest.raises(ValueError):
        filtered_img = median(image, behavior='ndimage')

# TODO
@pytest.mark.skip(reason="in progress")
def test_median_1_unsupported_behavior():
    image = np.arange(5*5, dtype=np.uint8).reshape((5, 5))
    with pytest.raises(ValueError):
        filtered_img = median_1(image, behavior='ndimage')

# TODO
@pytest.mark.skip(reason="in progress")
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_median_preserve_dtype(image, dtype):
    mediani_mage = median(image.astype(dtype), behavior='ndimage')
    assert median_image.dtype == dtype

# TODO
@pytest.mark.skip(reason="in progress")
def test_median_error_ndim():
    img = np.random.randint(0, 10, size=(5, 5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        median(img, behavior='rank')
# TODO
# tests
