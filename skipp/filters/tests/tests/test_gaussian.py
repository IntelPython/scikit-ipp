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

# Used acronyms
# Intel(R) Integrated Performance Primitives (Intel(R) IPP)

import pytest
import numpy as np
from skipp.filters import gaussian


@pytest.mark.parametrize("supported_dtype", [np.uint8, np.uint16,
                                             np.int16, np.float32])
def test_gaussian_preserve_dtype(supported_dtype):
    image = np.ones((5, 5), dtype=supported_dtype)
    filtered_image = gaussian(image)
    assert filtered_image.dtype == image.dtype


def test_gaussian_default_sigma():
    a = np.zeros((3, 3), dtype=np.uint8)
    a[1, 1] = 1.
    assert np.all(gaussian(a) == gaussian(a, sigma=1))


def test_gaussian_null_sigma():
    a = np.zeros((3, 3))
    a[1, 1] = 1.
    assert np.all(gaussian(a, 0) == a)


# TODO: investigate why skimage in test_energy_decrease_gaussian uses `reflect` mode
# Intel(R) IPP GaussianFilterBorder doesn't support `reflect` mode
def test_gaussian_energy_decrease():
    a = np.zeros((3, 3), dtype=np.float32)
    a[1, 1] = 1.
    gaussian_a = gaussian(a, sigma=1)
    assert gaussian_a.std() < a.std()


def test_gaussian_unsupported_mode():
    """
    Intel(R) IPP GaussianFilterBorder doesn't support `reflect` mode
    """
    a = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(RuntimeError):
        gaussian_a = gaussian(a, mode='reflect')

# TODO
@pytest.mark.skip(reason="`preserve_range` param is not enabled")
def test_gaussian_preserve_range():
    img = np.array([[10.0, -10.0], [-4, 3]], dtype=np.float32)
    gaussian(img, 1, preserve_range=True)


# TODO
# update
def test_gaussian_dimension_error():
    image_4d = np.arange(5*5*5*4, dtype=np.uint8).reshape((5, 5, 5, 4))
    with pytest.raises(ValueError):
        filtered_img = gaussian(image_4d, sigma=1, multichannel=True)
