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

from skimage.transform import resize as skimage_resize
from skipp.transform import resize as skipp_resize


def get_image_data(image_dtype, shape=(300, 400)):
    image = np.arange(shape[0] * shape[1], dtype=image_dtype).reshape(shape)
    return image

# preserve_range = True
#@pytest.mark.parametrize("function",[skipp_resize, skimage_resize],
#                                    ids=["skipp_resize", "skimage_resize"])
# @pytest.mark.parametrize("function",[skipp_resize, skimage_resize], ids=["skipp_resize", "skimage_resize"])
@pytest.mark.parametrize("input_dtype", [np.float32])
@pytest.mark.parametrize("shape", [pytest.param((4000, 4000), id="4000x4000")
                                   ])
@pytest.mark.parametrize("output_shape", [(8000, 8000)], ids=["8000x8000"])
@pytest.mark.parametrize("function",[skimage_resize], ids=["skimage_resize"])
@pytest.mark.parametrize("preserve_range", [True])
@pytest.mark.parametrize("clip_mode", [False])
@pytest.mark.parametrize("anti_aliasing", [False])
@pytest.mark.parametrize("order", [0])
def test_resize_skimage(benchmark, function, input_dtype, shape, output_shape, preserve_range, clip_mode, anti_aliasing, order):
    """
    # Speed test, that measures the speed of scikit-ipp's and scikit-image's
      transform.resize funcs
    """
    image = get_image_data(input_dtype, shape)
    #result = benchmark.pedantic(target=function, args=(image, angle), rounds=10, iterations=25)
    #result = benchmark.pedantic(target=function, args=(image), kwargs={'preserve_range': preserve_range,'output_shape': output_shape,'clip_mode': clip_mode,'anti_aliasing': anti_aliasing, 'order':order}, rounds=10, iterations=10)
    result = benchmark.pedantic(target=function, args=(image, output_shape), kwargs={'mode':'edge','preserve_range': preserve_range,'clip': clip_mode,'anti_aliasing': anti_aliasing, 'order':order}, rounds=25, iterations=15)
    #result = benchmark(function, image, output_shape)


@pytest.mark.parametrize("input_dtype", [np.float32])
@pytest.mark.parametrize("shape", [pytest.param((4000, 4000), id="4000x4000")
                                   ])
@pytest.mark.parametrize("output_shape", [(8000, 8000)], ids=["8000x8000"])
@pytest.mark.parametrize("function",[skipp_resize], ids=["skipp_resize"])
@pytest.mark.parametrize("preserve_range", [True])
@pytest.mark.parametrize("clip_mode", [False])
@pytest.mark.parametrize("anti_aliasing", [False])
@pytest.mark.parametrize("order", [0])
def test_resize_skipp(benchmark, function, input_dtype, shape, output_shape, preserve_range, clip_mode, anti_aliasing, order):
    """
    # Speed test, that measures the speed of scikit-ipp's and scikit-image's
      transform.resize funcs
    """
    image = get_image_data(input_dtype, shape)
    #result = benchmark.pedantic(target=function, args=(image, angle), rounds=10, iterations=25)
    #result = benchmark.pedantic(target=function, args=(image), kwargs={'preserve_range': preserve_range,'output_shape': output_shape,'clip_mode': clip_mode,'anti_aliasing': anti_aliasing, 'order':order}, rounds=10, iterations=10)
    result = benchmark.pedantic(target=function, args=(image, output_shape), kwargs={'mode':'nearest','preserve_range': preserve_range,'clip': clip_mode,'anti_aliasing': anti_aliasing, 'order':order}, rounds=25, iterations=15)
