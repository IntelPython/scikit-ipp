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
from numpy.testing import (assert_allclose, assert_equal, assert_almost_equal)

from skipp.transform import AffineTransform
from skipp.transform import resize
from skipp.transform import rotate
from skipp.transform import warp

# Used acronyms
# Intel(R) Integrated Performance Primitives (Intel(R) IPP)

def test_invalid_input():
    with pytest.raises(ValueError):
        AffineTransform(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        AffineTransform(matrix=np.zeros((2, 3)), scale=1)


@pytest.mark.skip(reason="__repr__ in progress")
def test_affine_init():
    # init with implicit parameters
    scale = (0.1, 0.13)
    rotation = 1
    shear = 0.1
    translation = (1, 1)
    tform = AffineTransform(scale=scale, rotation=rotation, shear=shear,
                            translation=translation)
    assert_almost_equal(tform.scale, scale)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.shear, shear)
    assert_almost_equal(tform.translation, translation)

    # init with transformation matrix
    tform2 = AffineTransform(tform.params)
    assert_almost_equal(tform2.scale, scale)
    assert_almost_equal(tform2.rotation, rotation)
    assert_almost_equal(tform2.shear, shear)
    assert_almost_equal(tform2.translation, translation)


def test_AffineTransform_inverse():
    # TODO
    # add check orthogonal matrix
    expected = np.array([[ 0.        ,  0.        ],
                         [ 4.46998332, -2.24036808],
                         [ 2.67768885, -5.81635474],
                         [-1.79229446, -3.57598665]])

    transf = AffineTransform(rotation=90)
    rows = 6; cols = 5
    corners = np.array([[0, 0],
                        [0, rows - 1],
                        [cols - 1, rows - 1],
                        [cols - 1, 0]])
    results = transf.inverse(corners)
    assert_allclose(results, expected)


def test_warp_tform():
    x = np.zeros((5, 5), dtype=np.uint8)
    x[2, 2] = 1
    theta = - np.pi / 2
    tform = AffineTransform(scale=(1, 1), rotation=theta,
                            translation=(0, 4))

    x90 = warp(x, tform, order=1)
    assert_almost_equal(x90, np.rot90(x))


def test_warp_matrix():
    x = np.zeros((5, 5), dtype=np.uint8)
    x[2, 2] = 1
    refx = np.zeros((5, 5), dtype=np.uint8)
    refx[1, 1] = 1

    matrix = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])

    outx = warp(x, matrix, order=1)
    assert_almost_equal(outx, refx)


def test_rotate():
    x = np.zeros((5, 5), dtype=np.uint8)
    x[1, 1] = 1
    x90 = rotate(x, angle=90)
    assert_allclose(x90, np.rot90(x), rtol=1e-06)


def test_rotate_resize():
    x = np.zeros((10, 10), dtype=np.uint8)

    x45 = rotate(x, 45, resize=False)
    assert x45.shape == (10, 10)

    x45 = rotate(x, 45, resize=True)
    # new dimension should be d = sqrt(2 * (10/2)^2)
    assert x45.shape == (14, 14)


def test_rotate_center():
    x = np.zeros((10, 10), dtype=np.double)
    x[4, 4] = 1
    refx = np.zeros((10, 10), dtype=np.double)
    refx[2, 5] = 1
    x20 = rotate(x, 20, order=0, center=(0, 0))
    assert_allclose(x20, refx, rtol=1e-06)
    x0 = rotate(x20, -20, order=0, center=(0, 0))
    assert_allclose(x0, x, rtol=1e-06)


def test_rotate_resize_center():
    x = np.zeros((10, 10), dtype=np.double)
    x[0, 0] = 1

    ref_x45 = np.zeros((14, 14), dtype=np.double)
    ref_x45[6, 0] = 1
    ref_x45[7, 0] = 1

    x45 = rotate(x, 45, resize=True, center=(3, 3), order=0)
    # new dimension should be d = sqrt(2 * (10/2)^2)
    assert x45.shape == (14, 14)
    assert_equal(x45, ref_x45)


def test_rotate_resize_90():
    x90 = rotate(np.zeros((470, 230), dtype=np.double), 90,
                 resize=True)
    assert x90.shape == (230, 470)


@pytest.mark.parametrize("image_dtype", [np.uint8,  np.uint16,
                                         np.int16, np.float32])
@pytest.mark.parametrize("order", [1, 3])
def test_resize_with_antialiasing(image_dtype, order):
    image = np.zeros((5, 5), dtype=image_dtype)
    expected_shape = np.zeros((10, 10), dtype=image_dtype).shape
    resized = resize(image, (10, 10), order=order,
                     anti_aliasing=True)
    assert resized.dtype == image.dtype
    assert resized.shape == expected_shape


@pytest.mark.parametrize("image_dtype", [np.uint8,  np.uint16,
                                         np.int16, np.float32])
@pytest.mark.parametrize("order", [0, 1, 3, 6])
def test_resize_without_antialiasing(image_dtype, order):
    image = np.zeros((10, 10), dtype=image_dtype)
    expected_shape = np.zeros((20, 20), dtype=image_dtype).shape
    resized = resize(image, (20, 20), order=order,
                     anti_aliasing=False)
    assert resized.dtype == image.dtype
    assert resized.shape == expected_shape


@pytest.mark.parametrize("image_dtype", [np.uint8,  np.uint16,
                                         np.int16, np.float32])
def test_resize_super(image_dtype):
    image = np.zeros((20, 20), dtype=image_dtype)
    expected_shape = np.zeros((10, 10), dtype=image_dtype).shape
    resized = resize(image, (10, 10), order=7,
                     anti_aliasing=False)
    assert resized.dtype == image.dtype
    assert resized.shape == expected_shape


@pytest.mark.parametrize("image_dtype", [np.uint8,  np.uint16,
                                         np.int16, np.float32])
def test_resize2d(image_dtype):
     x = np.zeros((5, 5), dtype=image_dtype)
     x[1, 1] = 1
     resized = resize(x, (10, 10), order=0, anti_aliasing=False)
     ref = np.zeros((10, 10), dtype=image_dtype)
     ref[2:4, 2:4] = 1
     assert_allclose(resized, ref, rtol=1e-06)
