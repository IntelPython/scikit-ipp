import pytest
import numpy as np

import skimage.filters
import skipp.skipp.filters
import scipy.ndimage.filters

def get_image_data(image_dtype, shape=(300, 400)):
    image = np.arange(shape[0] * shape[1], dtype=image_dtype).reshape(shape)
    return image


@pytest.mark.parametrize("input_dtype", [np.uint8, np.uint16, np.int16, np.float32])
@pytest.mark.parametrize("sigma", [pytest.param(1, id="1.0"),
                                   pytest.param(100, id="100.0")
                                   ])
@pytest.mark.parametrize("shape", [pytest.param((300, 400), id="300x400")
                                   ])
@pytest.mark.parametrize("function",[pytest.param(skipp.skipp.filters.gaussian, id="skipp_gaussina"),
                                     pytest.param(skimage.filters.gaussian, id="skimage_gaussian"),
                                     pytest.param(scipy.ndimage.filters.gaussian_filter, id="scipy_gaussian")
                                     ])
def test_gaussian(benchmark, function, input_dtype, shape, sigma):
    """
    # Speed test, that measures the speed of scikit-ipp's and scikit-image's gaussian filter
    """
    image = get_image_data(input_dtype, shape)
    result = benchmark(function, image, sigma)


@pytest.mark.parametrize("input_dtype", [np.float32])
@pytest.mark.parametrize("shape", [pytest.param((300, 400), id="300x400")
                                   ])
@pytest.mark.parametrize("function",[pytest.param(skipp.skipp.filters.sobel, id="skipp_sobel"),
                                     pytest.param(skimage.filters.sobel, id="skimage_sobel"),
                                     pytest.param(skipp.skipp.filters.sobel_v, id="skipp_sobel_v"),
                                     pytest.param(skimage.filters.sobel_v, id="skimage_sobel_v"),
                                     pytest.param(skipp.skipp.filters.sobel_h, id="skipp_sobel_h"),
                                     pytest.param(skimage.filters.sobel_h, id="skimage_sobel_h"),
                                     pytest.param(skipp.skipp.filters.prewitt, id="skipp_prewitt"),
                                     pytest.param(skimage.filters.prewitt, id="skimage_prewitt"),
                                     pytest.param(skipp.skipp.filters.prewitt_v, id="skipp_prewitt_v"),
                                     pytest.param(skimage.filters.prewitt_v, id="skimage_prewitt_v"),
                                     pytest.param(skipp.skipp.filters.prewitt_h, id="skipp_prewitt_h"),
                                     pytest.param(skimage.filters.prewitt_h, id="skimage_prewitt_h"),
                                     pytest.param(skipp.skipp.filters.laplace, id="skipp_laplace"),
                                     pytest.param(skimage.filters.laplace, id="skimage_laplace")
                                     ])
def test_edges(benchmark, function, input_dtype, shape):
    """
    # Speed test, that measures the speed of scikit-ipp's and scikit-image's filters:
    # * sobel
    # * sobel_h
    # * sobel_v
    # * prewitt
    # * prewitt_h
    # * prewitt_v
    # * laplace
    """
    image = get_image_data(input_dtype, shape)
    result = benchmark(function, image)


@pytest.mark.parametrize("input_dtype", [np.uint8])
@pytest.mark.parametrize("shape", [pytest.param((300, 400), id="300x400")])
@pytest.mark.parametrize("mask_shape", [pytest.param((3, 5), id="3x5")])
@pytest.mark.parametrize("function",[pytest.param(skipp.skipp.filters.median, id="skipp_median"),
                                     pytest.param(skimage.filters.median, id="skimage_median")])
def test_median(benchmark, function, input_dtype, shape, mask_shape):
    """
    # Speed test, that measures the speed of scikit-ipp's and scikit-image's median filter
    """
    image = get_image_data(input_dtype, shape)
    mask = np.ones(mask_shape, input_dtype)
    # TODO
    # set behaviour param to avoid warning
    result = benchmark(function, image, mask)
