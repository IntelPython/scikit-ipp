# ******************************************************************************
# Copyright (c) 2020, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************

import pytest
import numpy as np

import skimage.filters
import skipp.filters
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
@pytest.mark.parametrize("function",[pytest.param(skipp.filters.gaussian, id="skipp_gaussina"),
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
@pytest.mark.parametrize("function",[pytest.param(skipp.filters.sobel, id="skipp_sobel"),
                                     pytest.param(skimage.filters.sobel, id="skimage_sobel"),
                                     pytest.param(skipp.filters.sobel_v, id="skipp_sobel_v"),
                                     pytest.param(skimage.filters.sobel_v, id="skimage_sobel_v"),
                                     pytest.param(skipp.filters.sobel_h, id="skipp_sobel_h"),
                                     pytest.param(skimage.filters.sobel_h, id="skimage_sobel_h"),
                                     pytest.param(skipp.filters.prewitt, id="skipp_prewitt"),
                                     pytest.param(skimage.filters.prewitt, id="skimage_prewitt"),
                                     pytest.param(skipp.filters.prewitt_v, id="skipp_prewitt_v"),
                                     pytest.param(skimage.filters.prewitt_v, id="skimage_prewitt_v"),
                                     pytest.param(skipp.filters.prewitt_h, id="skipp_prewitt_h"),
                                     pytest.param(skimage.filters.prewitt_h, id="skimage_prewitt_h"),
                                     pytest.param(skipp.filters.laplace, id="skipp_laplace"),
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
@pytest.mark.parametrize("function",[pytest.param(skipp.filters.median, id="skipp_median"),
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
