import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_allclose)
from numpy import allclose
from scipy.ndimage.filters import gaussian_filter as scipy_gaussian
from skipp.skipp.filters import gaussian
from skimage.filters import gaussian as skimage_gaussian


@pytest.fixture
def image():
    return np.array([[1, 2, 3, 2, 1],
                     [1, 1, 2, 2, 3],
                     [3, 2, 1, 2, 1],
                     [3, 2, 1, 1, 1],
                     [1, 2, 1, 2, 3]],
                    dtype=np.uint8)


# TODO
# add checks for np.uint32, np.int32, np.uint64, np.int64, np.float64, np.double
# TODO
# add also checks for different output and input dtypes
@pytest.mark.parametrize("input_dtype", [np.uint8, np.int8, np.uint16, np.int16, np.float32])
def test_gaussian_preserve_dtype(image, input_dtype):
    gaussian_image = gaussian(image.astype(input_dtype))
    assert gaussian_image.dtype == input_dtype


def test_gaussian_default_sigma():
    a = np.zeros((3, 3), dtype=np.uint8)
    a[1, 1] = 1.
    assert np.all(gaussian(a) == gaussian(a, sigma=1))

# TODO
# add checks for np.uint32, np.int32, np.uint64, np.int64, np.float64, np.double
# currently supported dtypes: np.uint8, np.uint16, np.int16, np.float32
@pytest.mark.parametrize("output_dtype", [np.uint8, np.uint16, np.int16, np.float32])
def test_gaussian_preserve_output(output_dtype):
    input_image = np.zeros((3, 3), dtype=output_dtype)
    output_image = np.zeros((3, 3), dtype=output_dtype)
    returned_image = gaussian(input_image, output=output_image)
    assert id(output_image) == id(returned_image)


def test_gaussian_null_sigma():
    a = np.zeros((3, 3))
    a[1, 1] = 1.
    assert np.all(gaussian(a, 0) == a)


# TODO: investigate why skimage in test_energy_decrease_gaussian uses `reflect` mode
# Intel IPP's GaussianFilterBorder doesn't support `reflect` mode
def test_gaussian_energy_decrease():
    a = np.zeros((3, 3), dtype=np.float32)
    a[1, 1] = 1.
    gaussian_a = gaussian(a, sigma=1)
    assert gaussian_a.std() < a.std()


def test_gaussian_unsupported_mode():
    """
    Intel IPP's GaussianFilterBorder doesn't support `reflect` mode
    """
    a = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(RuntimeError):
        gaussian_a = gaussian(a, mode='reflect')


def test_gaussian_preserve_range():
    img = np.array([[10.0, -10.0], [-4, 3]], dtype=np.float32)
    gaussian(img, 1, preserve_range=True)


# TODO
# update
def test_gaussian_dimension_error():
    image_4d = np.arange(5*5*5*4, dtype=np.uint8).reshape((5, 5, 5, 4))
    with pytest.raises(ValueError):
        filtered_img = gaussian(image_4d, sigma=1, multichannel=True)

# TODO
# add checks
# "input_dtype", [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float32, np.float64])
# "output_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("input_dtype", [np.float32])
@pytest.mark.parametrize("output_dtype", [np.float32])
@pytest.mark.parametrize("preserve_range", [True, False])
def test_gaussian_skimage_similarity(input_dtype, output_dtype, preserve_range):
    """
    # Testing scikit-image's and scikit-ipp's gaussian filtering results
    """
    image = np.arange(5*4, dtype=input_dtype).reshape((5, 4))
    skimage_gaussian_result = skimage_gaussian(image, output=output_dtype, sigma=3, preserve_range=preserve_range)
    skipp_gaussian_result = gaussian(image, output=output_dtype, sigma=3, preserve_range=preserve_range)
    assert_array_almost_equal(skimage_gaussian_result, skipp_gaussian_result, decimal=3)


# TODO
@pytest.mark.skip(reason="in progress")
@pytest.mark.parametrize("data_type", [np.uint8, np.uint16, np.int16])
def test_gaussian_scipy_similarity(data_type):
    """
    # Note: there is a bug in gaussian scikit-image version 0.17.dev0
    # skimage.filters.gaussian doesn't use the value of output parameter
    # skimage.filters.gaussian is a wrapper around scipy.ndi.gaussian_filter. But 
    # the scikit-image's gaussian doesn't pass the output to scipy.ndi.gaussian_filter.
    # This leads that skimage.filters.gaussian returns only the image of float64/float32 dtype,
    # even if explicitly specify the integer output dtype.
    # Thats why this test compares scikit-ipp's gaussian filter with
    # scipy.ndimage.filters.gaussian_filter
    """
    rtol = 1e-05
    atol = 1e-08
    image = np.arange(5*4, dtype=data_type).reshape((5, 4))
    scipy_gaussian_result = scipy_gaussian(image, sigma=3)
    skipp_gaussian_result = gaussian(image, sigma=3)
    assert_allclose(scipy_gaussian_result, skipp_gaussian_result, rtol=rtol, atol=atol)


def test_gaussian_wrong_output_shape():
    image = np.zeros(3*3, dtype=np.uint8).reshape((3, 3))
    output_image = np.zeros(3*2, dtype=np.uint8).reshape((3, 2))
    with pytest.raises(RuntimeError):
        gaussian(image, output=output_image)
