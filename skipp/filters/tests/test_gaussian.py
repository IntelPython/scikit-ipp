import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
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


@pytest.mark.parametrize("input_dtype", [np.uint8, np.int8, np.uint16, np.int16, np.float32])
def test_gaussian_output_is_none(image, input_dtype):
    gaussian_image = gaussian(image.astype(input_dtype))
    assert gaussian_image.dtype == input_dtype


# TODO
@pytest.mark.skip(reason="dev in progress")
@pytest.mark.parametrize("input_dtype", [np.uint32, np.int32, np.uint64, np.int64, np.float64,
                                         np.double])
def test_gaussian_output_is_none_dev(image, input_dtype):
    gaussian_image = gaussian(image.astype(input_dtype))
    assert gaussian_image.dtype == input_dtype


def test_gaussian_default_sigma():
    a = np.zeros((3, 3), dtype=np.uint8)
    a[1, 1] = 1.
    assert np.all(gaussian(a) == gaussian(a, sigma=1))


@pytest.mark.parametrize("ipp_supported", [np.uint8, np.uint16, np.int16, np.float32])
def test_gaussian_given_and_returned_output(ipp_supported):
    input_image = np.zeros((3, 3), dtype=ipp_supported)
    output_image = np.zeros((3, 3), dtype=ipp_supported)
    returned_image = gaussian(input_image, output=output_image)
    assert id(output_image) == id(returned_image)


def test_gaussian_null_sigma():
    a = np.zeros((3, 3))
    a[1, 1] = 1.
    assert np.all(gaussian(a, 0) == a)


# Intel IPP's GaussianFilterBorder doesn't support `reflect` mode
# TODO: investigate why skimage in test_energy_decrease_gaussian uses `reflect` mode
def test_gaussian_energy_decrease():
    a = np.zeros((3, 3), dtype=np.float32)
    a[1, 1] = 1.
    gaussian_a = gaussian(a, sigma=1)
    assert gaussian_a.std() < a.std()


def test_gaussian_unsupported_mode_in_IPP():
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

def test_gaussian_skimage_similarity_float32():
    """
    # Testing scikit-image's and scikit-ipp's gaussian filtering results
    # for float32 input/output dtypes
    """
    image = np.arange(5*4, dtype=np.float32).reshape((5, 4))
    skimage_gaussian_result = skimage_gaussian(image, sigma=3)
    skipp_gaussian_result = gaussian(image, sigma=3)
    assert_array_almost_equal(skimage_gaussian_result, skipp_gaussian_result, decimal=3)


# TODO
# add np.float64 input image
# TODO
@pytest.mark.skip(reason="dev in progress")
@pytest.mark.parametrize("input_dtype", [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float32])
@pytest.mark.parametrize("output_dtype", [np.float32, np.float64])
def test_gaussian_skimage_similarity_preserve_range_false(input_dtype, output_dtype):
    """
    # Testing scikit-image's and scikit-ipp's gaussian filtering results when
    # preserve_range parameter is False and output dtype float
    """
    image = np.arange(3*4,dtype=input_dtype).reshape((4,3))
    skimage_gaussian_filtered = skimage_gaussian(image, output=output_dtype, sigma=1, preserve_range=False)
    skipp_gaussian_filtered = gaussian(image, output=output_dtype, sigma=1, preserve_range=False)
    assert_array_almost_equal(skimage_gaussian_filtered, skipp_gaussian_filtered, decimal=3)

# TODO
@pytest.mark.skip(reason="dev in progress")
def test_gaussian_skimage_similarity_uint8():
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
    image = np.arange(5*4, dtype=np.uint8).reshape((5, 4))
    scipy_gaussian_result = scipy_gaussian(image, sigma=3)
    skipp_gaussian_result = gaussian(image, sigma=3)
    assert_array_almost_equal(scipy_gaussian_result, skipp_gaussian_result, decimal=3)


def test_gaussian_wrong_output_shape():
    image = np.zeros(3*3, dtype=np.uint8).reshape((3, 3))
    output_image = np.zeros(3*2, dtype=np.uint8).reshape((3, 2))
    with pytest.raises(RuntimeError):
        gaussian(image, output=output_image)
