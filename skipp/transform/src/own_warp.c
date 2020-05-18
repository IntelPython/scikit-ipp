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
//    scikit-ipp's own functions for image transformations, that uses Intel(R) IPP.
//
////////////////////////////////////////////////////////////////////////////////////////
#include "own_warp.h"

#define EXIT_FUNC exitLine:             /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine

////////////////////////////////////////////////////////////////////////////////////////
//
//    own_RotateCoeffs
//
//    own_RotateCoeffs uses Intel(R) IPP's ippiGetRotateShift and
//    ippiGetRotateTransform functions for getting affine coefficients for the rotation
//    transform. ippiGetRotateShift, computes shift values for rotation of an image
//    around the specified center. ippiGetRotateTransform computes the affine
//    coefficients for the rotation transform.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
own_RotateCoeffs(
    double angle,
    double xCenter,
    double yCenter,
    double *coeffs)
{
    IppStatus status = ippStsNoErr;
    double xShift;
    double yShift;
    status = ippiGetRotateShift(xCenter, yCenter, angle, &xShift, &yShift);
    check_sts(status);
    status = ippiGetRotateTransform(angle, xShift, yShift, (double(*)[3])coeffs);
    check_sts(status);
EXIT_FUNC
    return status;
}

////////////////////////////////////////////////////////////////////////////////////////
//
//    own_GetAffineDstSize
//
//    own_GetAffineDstSize uses Intel(R) IPP's ippiGetAffineBound for computing size
//    destination image for the provided coeffs for the affine transformations.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
own_GetAffineDstSize(
    int img_width,
    int img_height,
    int * dst_width,
    int * dst_height,
    double * coeffs)
{
    IppStatus status = ippStsNoErr;
    double bound[2][2] = { 0 };
    IppiRect srcRoi;
    srcRoi.x = 0;
    srcRoi.y = 0;
    srcRoi.width = img_width;
    srcRoi.height = img_height;

    status = ippiGetAffineBound(srcRoi, bound, (double(*)[3])coeffs);
    check_sts(status);
    // TODO
    // more correct formula
    *dst_width = (int)(bound[1][0] - bound[0][0] + 2);
    *dst_height = (int)(bound[1][1] - bound[0][1] + 2);

EXIT_FUNC
    return status;
}

////////////////////////////////////////////////////////////////////////////////////////
//
//    own_Warp
//
//    own_Warp uses Intel(R) IPP's funcstions for implementing image warp
//    transformations
//
//    TODO: complete the description.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
own_Warp(
    IppDataType ippDataType,
    void * pSrc,
    void * pDst,
    int img_width,
    int img_height,
    int dst_width,
    int dst_height,
    int numChannels,
    double * coeffs,
    IppiInterpolationType interpolation,
    IppiWarpDirection direction,
    IppiBorderType ippBorderType,
    double ippBorderValue)
{
    IppStatus status = ippStsNoErr;
    IppiWarpSpec* pSpec = NULL;                          // Pointer to the specification
                                                         // structure
    Ipp8u* pInitBuf = NULL;

    // ``scikit-image`` uses Catmull-Rom spline (0.0, 0.5)
    // Catmull-Rom spline (0.0, 0.5)
    // Don P. Mitchell, Arun N. Netravali. Reconstruction Filters in Computer Graphics.
    // Computer Graphics, Volume 22, Number 4, AT&T Bell Laboratories, Murray Hill, 
    // New Jersey, August 1988.
    Ipp64f valueB = 0.0;
    Ipp64f valueC = 0.5;

    Ipp8u * pBuffer = NULL;

    Ipp64f pBorderValue[4];
    IppiSize srcSize = { img_width, img_height };        // Size of source image
    IppiSize dstSize = { dst_width, dst_height };        // size of destination images
    int srcStep, dstStep;                                // Steps, in bytes, through the
                                                         // source/destination images

    IppiPoint dstOffset = { 0, 0 };                      // Offset of the destination
                                                         // image ROI with respect to
                                                         // the destination image origin
    int specSize = 0, initSize = 0, bufSize = 0;         // Work buffer size

    int sizeof_src;

    status = get_sizeof(ippDataType, &sizeof_src);
    check_sts(status);

    srcStep = numChannels * img_width * sizeof_src;
    dstStep = numChannels * dst_width * sizeof_src;;

    if (numChannels == 1) {
        pBorderValue[0] = (Ipp64f)ippBorderValue;
    }
    else if (numChannels == 3)
    {
        pBorderValue[0] = (Ipp64f)ippBorderValue;
        pBorderValue[1] = (Ipp64f)ippBorderValue;
        pBorderValue[2] = (Ipp64f)ippBorderValue;
    }
    else if (numChannels == 4)
    {
        pBorderValue[0] = (Ipp64f)ippBorderValue;
        pBorderValue[1] = (Ipp64f)ippBorderValue;
        pBorderValue[2] = (Ipp64f)ippBorderValue;
        pBorderValue[3] = (Ipp64f)ippBorderValue;
    }
    else
    {
        status = ippStsErr;
        check_sts(status);
    }
    // Spec and init buffer sizes
    status = ippiWarpAffineGetSize(srcSize, dstSize, ippDataType,
                                  (double(*)[3])coeffs,  interpolation, direction,
                                  ippBorderType, &specSize, &initSize);
    check_sts(status);

    pInitBuf = ippsMalloc_8u(initSize);
    if (pInitBuf == NULL)
    {
        status = ippStsNoMemErr;
        check_sts(status);
    }
    pSpec = (IppiWarpSpec*)ippsMalloc_8u(specSize);
    if (pSpec == NULL)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    }
    // Filter initialization
    switch (interpolation)
    {
    case ippCubic:
    {
        status = ippiWarpAffineCubicInit(srcSize, dstSize, ippDataType,
                                        (double(*)[3])coeffs, direction, numChannels,
                                        valueB, valueC, ippBorderType, pBorderValue,
                                        0, pSpec, pInitBuf);
        break;
    }
    case ippNearest:
    {
        status = ippiWarpAffineNearestInit(srcSize, dstSize, ippDataType,
                                          (double(*)[3])coeffs, direction,
                                          numChannels, ippBorderType,
                                          pBorderValue, 0, pSpec);
        break;
    }
    case ippLinear:
    {
        status = ippiWarpAffineLinearInit(srcSize, dstSize, ippDataType,
                                         (double(*)[3])coeffs, direction,
                                         numChannels, ippBorderType, pBorderValue,
                                         0, pSpec);
        break;
    }
    default:
    {
        status = ippStsErr;
    }
    }
    check_sts(status);
    // Get work buffer size
    status = ippiWarpGetBufferSize(pSpec, dstSize, &bufSize);
    check_sts(status);

    pBuffer = ippsMalloc_8u(bufSize);
    if (pBuffer == NULL)
    {
        check_sts(status = ippStsMemAllocErr);
    };
    status = _ippiWarpAffine_interpolation(ippDataType, interpolation, numChannels,
        pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
    check_sts(status);
EXIT_FUNC
    ippsFree(pInitBuf);
    ippsFree(pBuffer);
    ippsFree(pSpec);
    return status;
}
