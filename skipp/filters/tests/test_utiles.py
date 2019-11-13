import pytest
import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal)
from scipy.ndimage.filters import gaussian_filter as scipy_gaussian
from skimage import img_as_float as skimage_img_as_float
from skimage import img_as_float32 as skimage_img_as_float32
from skimage import img_as_float64 as skimage_img_as_float64
# TODO
# Change modul
# from skipp.skipp.filters import img_as_float

unsigned_dtype_range = {np.uint8: (0, 255),
                        np.uint16: (0, 65535),
                        np.uint32: (0, 4294967295)}

signed_dtype_range = {np.int8: (-128, 127),
                      np.int16: (-32768, 32767),
                      np.int32: (-2147483648, 2147483647)}


dtype_range_floats_for_signed = {np.float32: (-1.0, 1.0),
                                   np.float64: (-1.0, 1.0)}

dtype_range_floats_for_unsigned = {np.float32: (0.0, 1.0),
                                   np.float64: (0.0, 1.0)}


# TODO
@pytest.mark.skip(reason="dev in progress")
@pytest.mark.parametrize("input_dtype", [np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32])
def test_img_as_float(input_dtype):
    if input_dtype in unsigned_dtype_range:
        input_image = np.asanyarray(unsigned_dtype_range[input_dtype], dtype=input_dtype)
        output_image = img_as_float(input_image)
        correct_output = np.asanyarray(dtype_range_floats_for_unsigned[np.float64], dtype=np.float64)
    if input_dtype in signed_dtype_range:
        input_image = np.asanyarray(signed_dtype_range[input_dtype], dtype=input_dtype)
        output_image = img_as_float(input_image)
        correct_output = np.asanyarray(dtype_range_floats_for_signed[np.float64], dtype=np.float64)
    assert_array_equal(output_image, correct_output)
