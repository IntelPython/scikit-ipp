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

from __future__ import absolute_import
include "../_ipp_utils/_ippi.pxd"


cdef extern from "own_filters.h":
    ctypedef enum own_EdgeFilterKernel:
        own_filterSobelVert = 0
        own_filterSobelHoriz = 1
        own_filterSobel = 2
        own_filterPrewittVert = 3
        own_filterPrewittHoriz = 4
        own_filterPrewitt = 5 

    IppStatus own_FilterGaussian(IppDataType ippDataType,
                                 void * pSrc,
                                 void * pDst,
                                 int img_width,
                                 int img_height,
                                 int numChannels,
                                 float sigma_,
                                 int kernelSize,
                                 IppiBorderType ippBorderType,
                                 float ippBorderValue)

    IppStatus own_FilterMedian(IppDataType ippDataType,
                               void * pSrc,
                               void * pDst,
                               int img_width,
                               int img_height,
                               int numChannels,
                               int mask_width,
                               int mask_height,
                               IppiBorderType ippBorderType,
                               float ippBorderValue)

    IppStatus own_FilterLaplace(IppDataType ippDataType,
                                void * pSrc,
                                void * pDst,
                                int img_width,
                                int img_height,
                                int numChannels,
                                IppiBorderType ippBorderType,
                                float ippBorderValue)

    IppStatus own_FilterEdge(own_EdgeFilterKernel edgeKernel,
                             IppDataType ippSrcDataType,
                             IppDataType ippDstDataType,
                             void * pSrc,
                             void * pDst,
                             int img_width,
                             int img_height,
                             int numChannels)

    IppStatus own_FilterPrewitt(own_EdgeFilterKernel edgeKernel,
                                IppDataType ippSrcDataType,
                                IppDataType ippDstDataType,
                                void * pSrc,
                                void * pDst,
                                int img_width,
                                int img_height,
                                int numChannels)

    IppStatus own_mask_filter_result(IppDataType ippSrcDataType,
                                     void * pDst,
                                     int img_width,
                                     int img_height,
                                     int numChannels)
