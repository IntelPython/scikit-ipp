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

import os

import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_array_equal)
from skimage.morphology import grey, selem
from skimage._shared.testing import (TestCase, parametrize)
from skimage import img_as_uint

from skipp.morphology import (erosion, dilation)


class TestEccentricStructuringElements(TestCase):
    def setUp(self):
        self.black_pixel = 255 * np.ones((4, 4), dtype=np.uint8)
        self.black_pixel[1, 1] = 0
        self.white_pixel = 255 - self.black_pixel
        self.selems = [selem.square(2), selem.rectangle(2, 2),
                       selem.rectangle(2, 1), selem.rectangle(1, 2)]

    def test_dilate_erode_symmetry(self):
        for s in self.selems:
            c = erosion(self.black_pixel, s)
            d = dilation(self.white_pixel, s)
            assert np.all(c == (255 - d))


@pytest.mark.parametrize("function", [pytest.param(dilation, id="dilation"),
                                      pytest.param(erosion, id="erosion")])
def test_default_selem(function):
    strel = selem.diamond(radius=1)
    image = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], np.uint8)
    im_expected = function(image, strel)
    im_test = function(image)
    assert_allclose(im_expected, im_test)


# float test images
im = np.array([[0.55, 0.72, 0.60, 0.54, 0.42],
               [0.65, 0.44, 0.89, 0.96, 0.38],
               [0.79, 0.53, 0.57, 0.93, 0.07],
               [0.09, 0.02, 0.83, 0.78, 0.87],
               [0.98, 0.80, 0.46, 0.78, 0.12]], dtype=np.float32)

eroded = np.array([[0.55, 0.44, 0.54, 0.42, 0.38],
                   [0.44, 0.44, 0.44, 0.38, 0.07],
                   [0.09, 0.02, 0.53, 0.07, 0.07],
                   [0.02, 0.02, 0.02, 0.78, 0.07],
                   [0.09, 0.02, 0.46, 0.12, 0.12]], dtype=np.float32)

dilated = np.array([[0.72, 0.72, 0.89, 0.96, 0.54],
                    [0.79, 0.89, 0.96, 0.96, 0.96],
                    [0.79, 0.79, 0.93, 0.96, 0.93],
                    [0.98, 0.83, 0.83, 0.93, 0.87],
                    [0.98, 0.98, 0.83, 0.78, 0.87]], dtype=np.float32)


def test_float():
    assert_allclose(erosion(im), eroded)
    assert_allclose(dilation(im), dilated)


def test_uint16():
    im16, eroded16, dilated16 = (
        map(img_as_uint, [im, eroded, dilated]))
    assert_allclose(erosion(im16), eroded16)
    assert_allclose(dilation(im16), dilated16)


@pytest.mark.skip(reason="needs __get_output implementation")
def test_discontiguous_out_array():
    image = np.array([[5, 6, 2],
                      [7, 2, 2],
                      [3, 5, 1]], np.uint8)
    out_array_big = np.zeros((5, 5), np.uint8)
    out_array = out_array_big[::2, ::2]
    expected_dilation = np.array([[7, 0, 6, 0, 6],
                                  [0, 0, 0, 0, 0],
                                  [7, 0, 7, 0, 2],
                                  [0, 0, 0, 0, 0],
                                  [7, 0, 5, 0, 5]], np.uint8)
    expected_erosion = np.array([[5, 0, 2, 0, 2],
                                 [0, 0, 0, 0, 0],
                                 [2, 0, 2, 0, 1],
                                 [0, 0, 0, 0, 0],
                                 [3, 0, 1, 0, 1]], np.uint8)
    dilation(image, out=out_array)
    assert_array_equal(out_array_big, expected_dilation)
    erosion(image, out=out_array)
    assert_array_equal(out_array_big, expected_erosion)


def test_1d_erosion():
    image = np.array([1, 2, 3, 2, 1], dtype=np.uint8)
    expected = np.array([1, 1, 2, 1, 1], dtype=np.uint8)
    eroded = erosion(image)
    assert_array_equal(eroded, expected)


def test_1d_dilation():
    image = np.array([1, 2, 3, 2, 1], dtype=np.uint8)
    expected = np.array([2, 3, 3, 3, 2], dtype=np.uint8)
    dilated = dilation(image)
    assert_array_equal(dilated, expected)
