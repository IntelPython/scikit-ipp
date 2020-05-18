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
from skipp.filters import laplace as laplace


@pytest.fixture
def image():
    return np.array([[1, 2, 3, 2, 1],
                     [1, 1, 2, 2, 3],
                     [3, 2, 1, 2, 1],
                     [3, 2, 1, 1, 1],
                     [1, 2, 1, 2, 3]],
                    dtype=np.uint8)


def test_laplace_zeros():
    """Laplace on a square image."""
    # Create a synthetic 2D image
    image = np.zeros((9, 9), dtype=np.float32)
    image[3:-3, 3:-3] = 1
    result = laplace(image)
    check_result = np.array([[0., 0., 0.,  0., 0.,  0.,  0., 0., 0.],
                             [0., 0., 0.,  0., 0.,  0.,  0., 0., 0.],
                             [0., 0., 0., -1., -1., -1., 0., 0., 0.],
                             [0., 0., -1., 2., 1.,  2., -1., 0., 0.],
                             [0., 0., -1., 1., 0.,  1., -1., 0., 0.],
                             [0., 0., -1., 2., 1.,  2., -1., 0., 0.],
                             [0., 0., 0., -1., -1., -1., 0., 0., 0.],
                             [0., 0., 0.,  0., 0.,  0.,  0., 0., 0.],
                             [0., 0., 0.,  0., 0.,  0.,  0., 0., 0.]])
    assert_allclose(result, check_result)

# TODO
@pytest.mark.skip(reason="deveoloping _mask_filter_result in progress")
def test_laplace_mask():
    """Laplace on a masked array should be zero."""
    # Create a synthetic 2D image
    image = np.zeros((9, 9), dtype=np.float32)
    image[3:-3, 3:-3] = 1
    # Define the mask
    result = laplace(image, ksize=3, mask=np.zeros((9, 9), dtype=bool))
    assert (np.all(result == 0))
