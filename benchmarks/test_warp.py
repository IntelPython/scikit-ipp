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

from skimage.transform import warp as skimage_warp
from skipp.transform import warp as skipp_warp

from skipp.transform import AffineTransform as skp_AffineTransform
from skimage.transform import AffineTransform as skm_AffineTransform



def get_image_data(image_dtype, shape=(300, 400)):
    image = np.arange(shape[0] * shape[1], dtype=image_dtype).reshape(shape)
    return image


@pytest.mark.parametrize("input_dtype", [np.float32])
@pytest.mark.parametrize("shape", [pytest.param((8000, 8000), id="800x800")
                                   ])
@pytest.mark.parametrize("function",[skipp_warp], ids=["skipp_warp"])
@pytest.mark.parametrize("preserve_range", [True])
@pytest.mark.parametrize("order", [1])
def test_warp_skipp(benchmark, function, shape, input_dtype, preserve_range, order):
    """
    # Speed test, that measures the speed of scikit-ipp's and scikit-image's
      transform.rotate funcs
    """
    image = get_image_data(input_dtype, shape)
    mat = np.array([[ 0.70710678,  -0.70710678,  0.        ],
                    [ 0.70710678,   0.70710678,  0.        ],
                    [ 0.        ,   0.        ,  1.        ]], dtype=np.double)

    transf_skp = skp_AffineTransform(matrix=mat)
    result = benchmark.pedantic(target=function, args=(image, transf_skp.params), kwargs={'preserve_range': preserve_range, 'order':order},rounds=10, iterations=25)
    #result = skipp.transform.warp(image, transf_skp.params, order=order, preserve_range=preserve_range)


@pytest.mark.parametrize("input_dtype", [np.float32])
@pytest.mark.parametrize("shape", [pytest.param((8000, 8000), id="800x800")
                                   ])
@pytest.mark.parametrize("function",[skimage_warp], ids=["skimage_warp"])
@pytest.mark.parametrize("preserve_range", [True])
@pytest.mark.parametrize("order", [1])
def test_warp_skimage(benchmark, function, shape, input_dtype, preserve_range, order):
    """
    # Speed test, that measures the speed of scikit-ipp's and scikit-image's
      transform.rotate funcs
    """
    image = get_image_data(input_dtype, shape)
    mat = np.array([[ 0.70710678,  0.70710678,  0.        ],
                    [-0.70710678,  0.70710678,  0.        ],
                    [ 0.        ,  0.        ,  1.        ]], dtype=np.double)
    
    transf_skm = skm_AffineTransform(matrix=mat)
    result = benchmark.pedantic(target=function, args=(image, transf_skm.params), kwargs={'preserve_range': preserve_range, 'order':order},rounds=10, iterations=25)