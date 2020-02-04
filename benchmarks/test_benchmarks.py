import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.ndimage.filters import gaussian_filter as scipy_gaussian
from skipp.skipp.filters import gaussian
from skimage.filters import gaussian as skimage_gaussian


def get_image_data(image_dtype):
    image = np.arange(3000*40000, dtype=image_dtype).reshape((3000, 40000))
    return image


def run_scipy_gaussian(image, output, sigma):
    result = scipy_gaussian(image, output=output, sigma=sigma)


def run_skimage_gaussian(image, output, sigma, preserve_range):
    result = skimage_gaussian(image, output=output, sigma=sigma, preserve_range=preserve_range)


def run_skipp_gaussian(image, output, sigma, preserve_range):
    result = gaussian(image, output=output, sigma=sigma, preserve_range=preserve_range)

@pytest.mark.skip(reason="skipp")
@pytest.mark.parametrize("input_dtype", [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float32])
def test_bench_skimage_gaussian_preserve_range(benchmark, input_dtype):
    """
    Speed test, that measures the speed of scikit-image's gaussian filter with given values of parameters:
    sigma - big sigma (100)
    output dtype = np.float32 (all input images will be converted into float32)
    preserve_range is False: signed/unsigned images will be converted into float32 in range [-1..1] or [0..1]
    """
    image = get_image_data(input_dtype)
    # gaussian(image, output=np.float32, sigma=100, preserve_range=False
    result = benchmark(run_skimage_gaussian, image, np.float32, 100, False)

@pytest.mark.skip(reason="skipp")
@pytest.mark.parametrize("input_dtype", [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float32])
def test_bench_skipp_gaussian_preserve_range(benchmark, input_dtype):
    """
    Speed test, that measures the speed of scikit-ipp's gaussian filter with given values of parameters:
    sigma - big sigma (100)
    output dtype = np.float32 (all input images will be converted into float32)
    preserve_range is False: signed/unsigned images will be converted into float32 in range [-1..1] or [0..1]
    """
    image = get_image_data(input_dtype)
    result = benchmark(run_skipp_gaussian, image, np.float32, 100, False)

@pytest.mark.skip(reason="skipp")
@pytest.mark.parametrize("input_dtype", [np.uint8, np.uint16, np.int16, np.float32])
def test_bench_skipp_gaussian_with_ipp_supported(benchmark, input_dtype):
    """
    Speed test, that measures  the speed of scikit-ipp's gaussian filter.
    scikit-ipp uses Intel IPP ippiFilterGaussianBorder function for the gaussian blur.
    ippiFilterGaussianBorder supportes only np.uint8, np.uint16, np.int16, np.float32,
    those no need in converting if the given input & ouput are the same dtype and they are
    uint8, uint16, int16 or float32.
    """
    image = get_image_data(input_dtype)
    result = benchmark(run_skipp_gaussian, image, None, 100, False)

@pytest.mark.skip(reason="skipp")
@pytest.mark.parametrize("input_dtype", [np.uint8, np.uint16, np.int16, np.float32])
def test_bench_scipy_gaussian_with_ipp_supported(benchmark, input_dtype):
    """
    # Speed test, that measures  the speed of scipy's gaussian filter.
    #
    # Note: there is a bug in gaussian scikit-image version 0.17.dev0
    # skimage.filters.gaussian doesn't use the value of output parameter
    # skimage.filters.gaussian is a wrapper around scipy.ndi.gaussian_filter. But 
    # the scikit-image's gaussian doesn't pass the output to scipy.ndi.gaussian_filter.
    # This leads that skimage.filters.gaussian returns only the image of float64/float32 dtype,
    # even if explicitly specify the integer output dtype.
    """
    image = get_image_data(input_dtype)
    result = benchmark(run_scipy_gaussian, image, None, 100)

