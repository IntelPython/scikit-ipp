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
from numpy.testing import (assert_array_almost_equal, assert_allclose,
                           assert_array_equal, assert_allclose, assert_almost_equal)
from skimage._shared.testing import (assert_equal,TestCase, parametrize)

import skipp.transform
import skimage.transform

# TODO
# add checks
# @pytest.mark.parametrize("preserve_range", [True, False])
@pytest.mark.skip(reason="in progress")
@pytest.mark.parametrize("image_dtype", [np.float32])
@pytest.mark.parametrize("preserve_range", [True])
@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("angle", [i for i in range(0, 91, 1)])
def test_rotate_skimage_similarity(image_dtype, angle, order, preserve_range):
    """
    # Testing scikit-image's and scikit-ipp's rotate transform results
    """
    image = np.random.RandomState(0).randn(40, 50).astype(np.float32)
    skimage_rotate = skimage.transform.rotate(image, angle=angle, order=order,
                                              preserve_range=preserve_range)
    skipp_rotate = skipp.transform.rotate(image, angle=angle, order=order,
                                          preserve_range=preserve_range)
    assert_allclose(skipp_rotate, skimage_rotate, rtol=1e-05)
