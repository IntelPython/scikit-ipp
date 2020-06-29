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
from numpy.testing import assert_allclose
from skipp.filters import (sobel, sobel_h, sobel_v)


def test_sobel_zeros():
    """Sobel on an array of all zeros."""
    result = sobel(np.zeros((10, 10), dtype=np.float32))
    assert(np.all(result == 0))


@pytest.mark.skip(reason='non-default values for '\
                         'mask param are not supported')
def test_sobel_mask():
    """Sobel on a masked array should be zero."""
    image = np.random.uniform(size=(10, 10)).astype(np.float32)
    mask = np.zeros((10, 10), dtype=bool)
    result = sobel(image=image, mask=mask)
    assert(np.all(result == 0))


def test_sobel_horizontal():
    """Sobel on a horizontal edge should be a horizontal line."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(np.float32)
    result = sobel(image) * np.sqrt(2)
    # Check if result match transform direction
    i[np.abs(j) == 5] = 10000
    assert_allclose(result[i == 0], 1)
    assert (np.all(result[np.abs(i) > 1] == 0))


def test_sobel_vertical():
    """Sobel on a vertical edge should be a vertical line."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(np.float32)
    result = sobel(image) * np.sqrt(2)
    j[np.abs(i) == 5] = 10000
    assert_allclose(result[j == 0], 1)
    assert (np.all(result[np.abs(j) > 1] == 0))


def test_sobel_h_zeros():
    """Horizontal sobel on an array of all zeros."""
    result = sobel_h(np.zeros((10, 10), dtype=np.float32))
    assert(np.all(result == 0))


@pytest.mark.skip(reason='non-default values for '\
                         'mask param are not supported')
def test_sobel_h_mask():
    """Horizontal Sobel on a masked array should be zero."""
    image = np.random.uniform(size=(10, 10)).astype(np.float32)
    mask = np.zeros((10, 10), dtype=bool)
    result = sobel_h(image=image, mask=mask)
    assert_allclose(result, 0)


def test_sobel_h_horizontal():
    """Horizontal Sobel on an edge should be a horizontal line."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(np.float32)
    result = sobel_h(image)
    # Check if result match transform direction
    i[np.abs(j) == 5] = 10000
    assert (np.all(result[i == 0] == 1))
    assert (np.all(result[np.abs(i) > 1] == 0))


def test_sobel_h_vertical():
    """Horizontal Sobel on a vertical edge should be zero."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(np.float32) * np.sqrt(2)
    result = sobel_h(image)
    assert_allclose(result, 0, atol=1e-10)


def test_sobel_v_zeros():
    """Vertical sobel on an array of all zeros."""
    result = sobel_v(np.zeros((10, 10), dtype=np.float32))
    assert_allclose(result, 0)


@pytest.mark.skip(reason='non-default values for '\
                         'mask param are not supported')
def test_sobel_v_mask():
    """Vertical Sobel on a masked array should be zero."""
    image = np.random.uniform(size=(10, 10)).astype(np.float32)
    mask = np.zeros((10, 10), dtype=bool)
    result = sobel_v(image=image, mask=mask)
    assert_allclose(result, 0)


def test_sobel_v_vertical():
    """Vertical Sobel on an edge should be a vertical line."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(np.float32)
    result = sobel_v(image)
    # Check if result match transform direction
    j[np.abs(i) == 5] = 10000
    assert (np.all(result[j == 0] == 1))
    assert (np.all(result[np.abs(j) > 1] == 0))


def test_sobel_v_horizontal():
    """vertical Sobel on a horizontal edge should be zero."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(np.float32)
    result = sobel_v(image)
    assert_allclose(result, 0)
