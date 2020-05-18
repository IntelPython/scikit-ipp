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

from skimage.transform import rotate as skimage_rotate
from skipp.transform import rotate as skipp_rotate


def get_image_data(image_dtype, shape=(300, 400)):
    image = np.arange(shape[0] * shape[1], dtype=image_dtype).reshape(shape)
    return image

# preserve_range = True
@pytest.mark.parametrize("input_dtype", [np.float32])
@pytest.mark.parametrize("shape", [pytest.param((8000, 8000), id="8000x8000")
                                   ])
@pytest.mark.parametrize("function",[skipp_rotate, skimage_rotate],
                                    ids=["skipp_rotate", "skimage_rotate"])
@pytest.mark.parametrize("preserve_range", [True])
@pytest.mark.parametrize("angle", [30])
@pytest.mark.parametrize("order", [0, 1])
def test_rotate(benchmark, function, input_dtype, shape, angle, preserve_range, order):
    """
    # Speed test, that measures the speed of scikit-ipp's and scikit-image's
      transform.rotate funcs
    """
    image = get_image_data(input_dtype, shape)
    result = benchmark.pedantic(target=function, args=(image, angle), rounds=10, iterations=25)
    #result = benchmark.pedantic(target=function, args=(image, angle), kwargs={'preserve_range': preserve_range, 'order':order}, rounds=10, iterations=10)
