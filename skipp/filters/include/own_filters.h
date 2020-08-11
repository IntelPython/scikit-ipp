/*******************************************************************************
* Copyright (c) 2020, Intel Corporation
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*     * Redistributions of source code must retain the above copyright notice,
*       this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of Intel Corporation nor the names of its contributors
*       may be used to endorse or promote products derived from this software
*       without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

////////////////////////////////////////////////////////////////////////////////////////
//
//    scikit-ipp's own functions for filterring images, that uses
//    Intel(R) Integrated Performance Primitives (Intel(R) IPP)
//
////////////////////////////////////////////////////////////////////////////////////////

#ifndef OWN_FILTERS_H
#define OWN_FILTERS_H
#include <stddef.h>
#include "ipp.h"
#include <omp.h>
#include "_ipp_wr.h"
#include "utils.h"

typedef enum {
    own_filterSobelVert,
    own_filterSobelHoriz,
    own_filterSobel,
    own_filterPrewittVert,
    own_filterPrewittHoriz,
    own_filterPrewitt
}own_EdgeFilterKernel;

#ifndef own_Laplace_KERNEL_3x3
#define own_Laplace_KERNEL_3x3 { \
   0, -1,  0, \
  -1,  4, -1, \
   0, -1,  0  \
}
#endif

////////////////////////////////////////////////////////////////////////////////////////
//
//    own_FilterBorder
//
//          General border filter
//
//    Note: own_FilterBorder uses Intel IPP functions on the backend
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
own_FilterBorder(
    IppDataType ippImageDataType,
    IppDataType ippKernelDataType,
    void * pSrc,
    void * pDst,
    void * pKernel,
    int img_width,
    int img_height,
    int kernel_width,
    int kernel_height,
    int numChannels,
    IppRoundMode roundMode,
    IppiBorderType ippBorderType,
    float ippBorderValue);

////////////////////////////////////////////////////////////////////////////////////////
//
//    own_FilterGaussian
//
//          Gaussian filter
//
//    Note: own_FilterGaussian uses Intel IPP functions on the backend for
//          implementing gaussian filtering of image as is in scikit-image.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
own_FilterGaussian(
    IppDataType ippDataType,
    void * pSrc,
    void * pDst,
    int img_width,
    int img_height,
    int numChannels,
    float sigma_,
    int kernelSize,
    IppiBorderType ippBorderType,
    float ippBorderValue);

////////////////////////////////////////////////////////////////////////////////////////
//
//    own_FilterGaussian
//
//          Median filter
//
//          own_FilterGaussian uses Intel IPP functions on the backend for
//          implementing median filtering of image as is in scikit-image.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
own_FilterMedian(
    IppDataType ippDataType,
    void * pSrc,
    void * pDst,
    int img_width,
    int img_height,
    int numChannels,
    int mask_width,
    int mask_height,
    IppiBorderType ippBorderType,
    float ippBorderValue);

////////////////////////////////////////////////////////////////////////////////////////
//
//    own_FilterLaplace
//
//                                0 -1  0
//              Laplace (3x3)    -1  4 -1
//                                0 -1  0
//    Note: own_FilterLaplace uses own_FilterBorder on the backend for implementing
//          laplace filtering as is in scikit-image.
//          This func doesn't use ippiFilterLaplaceBorder, because of different laplace
//          kernel coeffs.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
own_FilterLaplace(
    IppDataType ippDataType,
    void * pSrc,
    void * pDst,
    int img_width,
    int img_height,
    int numChannels,
    IppiBorderType ippBorderType,
    float ippBorderValue);

///////////////////////////////////////////////////////////////////////////////////////////
//
//    own_FilterEdge
//                                1  0 -1
//              Prewitt_h (3x3)   1  0 -1
//                                1  0 -1
//
//                                1  1  1
//              Prewitt_v (3x3)   0  0  0
//                               -1 -1 -1
//
//              Sobel (3x3)       TODO: description
//
//                                1  0 -1
//              Sobel_h (3x3)     2  0 -2
//                                1  0 -1
//
//                                1  2  1
//              Sobel_v (3x3)     0  0  0
//                               -1 -2 -1
//
//    Note: own_FilterEdge       TODO: description
//
///////////////////////////////////////////////////////////////////////////////////////////
IppStatus
own_FilterEdge(
    own_EdgeFilterKernel edgeKernel,
    IppDataType ippSrcDataType,
    IppDataType ippDstDataType,
    void * pSrc,
    void * pDst,
    int img_width,
    int img_height,
    int numChannels);

///////////////////////////////////////////////////////////////////////////////////////////
//
//    own_FilterPrewitt
//         Prewitt (3x3)
//         computes the square root of the sum of squares of the horizontal
//         and vertical Prewitt transforms.
//
//              sqrt(A**2 + B**2)/sqrt(2)
//
//    Note: currently supporeted only Ipp32f input/output.
//
///////////////////////////////////////////////////////////////////////////////////////////
IppStatus
own_FilterPrewitt(
    own_EdgeFilterKernel edgeKernel,
    IppDataType ippSrcDataType,
    IppDataType ippDstDataType,
    void * pSrc,
    void * pDst,
    int img_width,
    int img_height,
    int numChannels);

IppStatus
own_mask_filter_result(
    IppDataType ippSrcDataType,
    void * pDst,
    int img_width,
    int img_height,
    int numChannels);
#endif // OWN_FILTERS_H
