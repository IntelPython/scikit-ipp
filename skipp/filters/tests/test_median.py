import pytest
import numpy as np
from skipp.filters import median

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
    median_image = median(image.astype(dtype), selem = np.ones((3, 3),
                          dtype=np.bool_), behavior='ipp')
    assert median_image.dtype == dtype
