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

from skipp.filters import (gaussian, laplace, median)
from skipp.filters import (sobel, sobel_v, sobel_h)
from skipp.filters import (prewitt, prewitt_v, prewitt_h)
from skipp.morphology import (erosion, dilation)
from skipp.transform import (resize, rotate, warp)
from skipp.transform import AffineTransform


ROUNDS=30
ITERATIONS=1

IMAGE_DTYPE=np.float32
IMAGE_NUM_CHANNELS=1

SRC_IMAGE_WIDTH=4000
SRC_IMAGE_HEIGHT=4000

DST_IMAGE_WIDTH=20000
DST_IMAGE_HEIGHT=20000


def get_image_data(image_dtype=IMAGE_DTYPE,
                   shape=(SRC_IMAGE_WIDTH, SRC_IMAGE_HEIGHT),
                   number_of_channels=IMAGE_NUM_CHANNELS):
    if number_of_channels==3 or number_of_channels==4:
        shape=(SRC_IMAGE_WIDTH, SRC_IMAGE_HEIGHT, number_of_channels)
    elif number_of_channels != 1:
        raise ValueError("No test suits for provided "\
                          "number_of_channels: {}".format(number_of_channels))

    if image_dtype==np.float32:
        image = np.random.random(shape).astype(image_dtype)
    elif image_dtype==np.uint8 or image_dtype==np.uint16 or image_dtype==np.int16:
        image = np.random.random(shape).astype(image_dtype)
        np.random.randint(255, size=None, dtype=image_dtype)
    else:
        raise ValueError("No test suits for provided dtype: {}".format(image_dtype))
    return image


@pytest.mark.parametrize("sigma", [pytest.param(1, id="1.0"),
                                   pytest.param(10, id="10.0")])
def test_gaussian(benchmark, sigma):
    image = get_image_data()
    result = benchmark.pedantic(target=gaussian, args=(image, sigma),
                                rounds=ROUNDS, iterations=ITERATIONS)


@pytest.mark.parametrize("function",[pytest.param(sobel, id="sobel"),
                                     pytest.param(sobel_v, id="sobel_v"),
                                     pytest.param(sobel_h, id="sobel_h"),
                                     pytest.param(prewitt, id="prewitt"),
                                     pytest.param(prewitt_v, id="prewitt_v"),
                                     pytest.param(prewitt_h, id="prewitt_h"),
                                     pytest.param(laplace, id="laplace"),])
def test_edges(benchmark, function):
    image = get_image_data(image_dtype=IMAGE_DTYPE,
                           shape=(13000, 13000),
                           number_of_channels=IMAGE_NUM_CHANNELS)
    result = benchmark.pedantic(target=function, args=(image, None),
                                rounds=ROUNDS, iterations=ITERATIONS)


def test_median(benchmark):
    image = get_image_data()
    result = benchmark.pedantic(target=median, args=(image, None),
                                kwargs={'behavior':"ipp"},
                                rounds=ROUNDS, iterations=ITERATIONS)


@pytest.mark.parametrize("function",[erosion, dilation],
                                    ids=["erosion", "dilation"])
def test_morphology(benchmark, function):
    image = get_image_data()
    result = benchmark.pedantic(target=function, args=(image, None),
                                rounds=ROUNDS, iterations=ITERATIONS)


@pytest.mark.parametrize("anti_aliasing", [False])
@pytest.mark.parametrize("order", [0, 1, 3])
def test_resize(benchmark, anti_aliasing, order):
    output_shape = (DST_IMAGE_WIDTH, DST_IMAGE_HEIGHT)
    image = get_image_data()
    result = benchmark.pedantic(target=resize, args=(image, output_shape),
                                kwargs={'mode':'edge','preserve_range': True,
                                        'clip': False,
                                        'anti_aliasing': anti_aliasing,
                                        'order':order}, 
                                        rounds=ROUNDS, iterations=ITERATIONS)


@pytest.mark.parametrize("angle", [45])
@pytest.mark.parametrize("order", [0, 1, 3])  # supported by scikit-ipp
def test_rotate(benchmark, angle, order):
    image = get_image_data(image_dtype=np.float32, shape=(13000, 13000))
    result = benchmark.pedantic(target=rotate, args=(image, angle),
                                kwargs={'preserve_range': True, 'order':order,
                                        'resize':True},
                                rounds=ROUNDS, iterations=ITERATIONS)


@pytest.mark.parametrize("order", [0, 1, 3])  # supported by scikit-ipp
def test_warp(benchmark, order):
    image = get_image_data(image_dtype=np.float32, shape=(13000, 13000))
    mat = np.array([[ 0.70710678,  -0.70710678,  0.        ],
                    [ 0.70710678,   0.70710678,  0.        ],
                    [ 0.        ,   0.        ,  1.        ]],
                    dtype=np.double)
    transf = AffineTransform(matrix=mat)
    result = benchmark.pedantic(target=warp, args=(image, transf.params),
                                kwargs={'preserve_range': True, 'order':order},
                                rounds=ROUNDS, iterations=ITERATIONS)
