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

from skipp.morphology import (erosion, dilation)


def test_dilate_erode_symmetry():
    # Test eccentric structuring elements
    # test dilate and erode symmetry
    black_pixel = 255 * np.ones((4, 4), dtype=np.uint8)
    black_pixel[1, 1] = 0
    white_pixel = 255 - black_pixel
    selem_list = []
    # rectacle 2 x 2
    # analog of skimage.morphology.selem.square(2)
    # or .skimage.morphology.selem.rectangle(2, 2)
    selem_list.append(np.array([[1, 1],
                                [1, 1]], dtype=np.uint8))
    # rectacle 1 x 2
    # analog of skimage.morphology.selem.rectangle(1, 2)
    selem_list.append(np.array([[1, 1]], dtype=np.uint8))
    # rectacle 2 x 1
    # analog of skimage.morphology.selem.rectangle(2, 1)
    selem_list.append(np.array([[1],
                                [1]], dtype=np.uint8))
    for s in selem_list:
        c = erosion(black_pixel, s)
        d = dilation(white_pixel, s)
        assert np.all(c == (255 - d))


@pytest.mark.parametrize("function", [pytest.param(dilation, id="dilation"),
                                      pytest.param(erosion, id="erosion")])
def test_default_selem(function):
    # selem_diamond
    # the same as is scikit-images's
    # skimage.morphology.selem.diamond(radius=1)
    selem_diamond = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=np.uint8)

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
    im_expected = function(image, selem_diamond)
    im_test = function(image)
    assert_allclose(im_expected, im_test)


def test_float():
    # float test images
    im = np.array([[0.55, 0.72, 0.60, 0.54, 0.42],
                   [0.65, 0.44, 0.89, 0.96, 0.38],
                   [0.79, 0.53, 0.57, 0.93, 0.07],
                   [0.09, 0.02, 0.83, 0.78, 0.87],
                   [0.98, 0.80, 0.46, 0.78, 0.12]],
                   dtype=np.float32)

    eroded = np.array([[0.55, 0.44, 0.54, 0.42, 0.38],
                       [0.44, 0.44, 0.44, 0.38, 0.07],
                       [0.09, 0.02, 0.53, 0.07, 0.07],
                       [0.02, 0.02, 0.02, 0.78, 0.07],
                       [0.09, 0.02, 0.46, 0.12, 0.12]],
                       dtype=np.float32)

    dilated = np.array([[0.72, 0.72, 0.89, 0.96, 0.54],
                        [0.79, 0.89, 0.96, 0.96, 0.96],
                        [0.79, 0.79, 0.93, 0.96, 0.93],
                        [0.98, 0.83, 0.83, 0.93, 0.87],
                        [0.98, 0.98, 0.83, 0.78, 0.87]],
                        dtype=np.float32)

    assert_allclose(erosion(im), eroded)
    assert_allclose(dilation(im), dilated)


def test_uint16():
    # given images match with given data on test suit
    # `test_float`
    # see: skimage.img_as_uint
    # e.g. im16 = skimage.img_as_uint(im)
    im16 = np.array([[36044, 47185, 39321, 35389, 27525],
                     [42598, 28835, 58326, 62914, 24903],
                     [51773, 34734, 37355, 60948,  4587],
                     [ 5898,  1311, 54394, 51117, 57015],
                     [64224, 52428, 30146, 51117,  7864]],
                     dtype=np.uint16)
    
    eroded16 = np.array([[36044, 28835, 35389, 27525, 24903],
                         [28835, 28835, 28835, 24903,  4587],
                         [ 5898,  1311, 34734,  4587,  4587],
                         [ 1311,  1311,  1311, 51117,  4587],
                         [ 5898,  1311, 30146,  7864,  7864]],
                         dtype=np.uint16)
    
    dilated16 = np.array([[47185, 47185, 58326, 62914, 35389],
                          [51773, 58326, 62914, 62914, 62914],
                          [51773, 51773, 60948, 62914, 60948],
                          [64224, 54394, 54394, 60948, 57015],
                          [64224, 64224, 54394, 51117, 57015]],
                          dtype=np.uint16)
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
