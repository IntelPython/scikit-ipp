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

from skimage.morphology import erosion as skimage_erosion
from skimage.morphology import dilation as skimage_dilation
from skipp.morphology import erosion as skipp_erosion
from skipp.morphology import dilation as skipp_dilation
# import scipy.ndimage.filters

import skipp
import skimage

def get_image_data(image_dtype, shape=(300, 400)):
    image = np.arange(shape[0] * shape[1], dtype=image_dtype).reshape(shape)
    return image

@pytest.mark.parametrize("input_dtype", [np.uint16, np.float32])
@pytest.mark.parametrize("shape", [pytest.param((8000, 8000), id="8000x8000")
                                   ])
@pytest.mark.parametrize("function",[skipp_erosion, skimage_erosion, skipp_dilation, skimage_dilation],
                                    ids=["skipp_erosion", "skimage_erosion", "skipp_dilation", "skimage_dilation"])

@pytest.mark.parametrize("selem", [pytest.param(None, id="default_selem")])
def test_morphology(benchmark, function, input_dtype, shape, selem):
    """
    # Speed test, that measures the speed of scikit-ipp's and scikit-image's morphology funcs
    """
    image = get_image_data(input_dtype, shape)
    result = benchmark.pedantic(target=function, args=(image, selem),
                                rounds=10, iterations=10)
