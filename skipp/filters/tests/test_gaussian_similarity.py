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
from numpy.testing import (assert_array_almost_equal, assert_allclose)
from scipy.ndimage.filters import gaussian_filter as scipy_gaussian
from skipp.filters import gaussian
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
    skimage_gaussian_result = skimage_gaussian(image, output=output_dtype, sigma=3,
                                               preserve_range=preserve_range)
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