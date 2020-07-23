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
//    scikit-ipp's own functions for image warp transformations, that uses
//    Intel(R) Integrated Performance Primitives (Intel(R) IPP).
//
////////////////////////////////////////////////////////////////////////////////////////
#include "own_warp.h"
#include "omp.h"

#define EXIT_FUNC exitLine:             /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine

////////////////////////////////////////////////////////////////////////////////////////
//
//    own_RotateCoeffs
//
//    own_RotateCoeffs uses Intel(R) IPP ippiGetRotateShift and
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
//    own_GetAffineDstSize uses Intel(R) IPP ippiGetAffineBound for computing size
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
//    own_Warp uses Intel(R) IPP funcstions for implementing image warp
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

    int specSize = 0, initSize = 0, bufSize = 0;         // Work buffer size

    int numThreads, slice, tail;
    int bufSize1, bufSize2;
    IppiSize dstTileSize, dstLastTileSize;

    int number_of_threads;

#ifdef MAX_NUM_THREADS
    number_of_threads = MAX_NUM_THREADS;
#else
    number_of_threads = omp_get_max_threads();
#endif

    IppStatus * pStatus = NULL;

    // checking supported dtypes
    if (!(ippDataType==ipp8u ||
          ippDataType==ipp16u ||
          ippDataType==ipp16s ||
          ippDataType==ipp32f))
    {
        status = ippStsDataTypeErr;
        check_sts(status);
    }

    int sizeof_src;

    status = get_sizeof(ippDataType, &sizeof_src);
    check_sts(status);

    srcStep = numChannels * img_width * sizeof_src;
    dstStep = numChannels * dst_width * sizeof_src;;

    pBorderValue[0] = (Ipp64f)ippBorderValue;
    pBorderValue[1] = (Ipp64f)ippBorderValue;
    pBorderValue[2] = (Ipp64f)ippBorderValue;
    pBorderValue[3] = (Ipp64f)ippBorderValue;

    pStatus = (IppStatus*)ippsMalloc_8u(sizeof(IppStatus) * number_of_threads);
    if (pStatus == NULL)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    }
    for (int i = 0; i < number_of_threads; ++i) pStatus[i] = ippStsNoErr;

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

    // General transform function
    // Parallelized only by Y-direction here
omp_set_num_threads(number_of_threads);
#pragma omp parallel
    {
#pragma omp master
        {
            numThreads = omp_get_num_threads();
            // ippSetNumThreads(numThreads);
            slice = dstSize.height / numThreads;
            tail = dstSize.height % numThreads;

            dstTileSize.width = dstSize.width;
            dstTileSize.height = slice;
            dstLastTileSize.width = dstSize.width;
            dstLastTileSize.height = slice + tail;

            ippiWarpGetBufferSize(pSpec, dstTileSize, &bufSize1);
            ippiWarpGetBufferSize(pSpec, dstLastTileSize, &bufSize2);
            pBuffer = ippsMalloc_8u(bufSize1 * (numThreads - 1) + bufSize2);
            if (pBuffer == NULL)
            {
                status = ippStsMemAllocErr;
            }
        }
#pragma omp barrier
        {
            if (pBuffer)
            {
                // ippSetNumThreads(1);
                Ipp32u  i;
                void * pDstT = NULL;
                Ipp8u * pOneBuf = NULL;
                i = omp_get_thread_num();
                IppiPoint srcOffset = { 0, 0 };
                IppiPoint dstOffset = { 0, 0 };
                IppiSize  srcSizeT = srcSize;
                IppiSize  dstSizeT = dstTileSize;
                
                dstSizeT.height = slice;
                dstOffset.y += i * slice;

                if (i == numThreads - 1) dstSizeT = dstLastTileSize;

                pDstT = (void*)((Ipp8u*)pDst + dstOffset.y * (Ipp32s)dstStep);

                if(status == ippStsNoErr)
                {
                    pOneBuf = pBuffer + i * bufSize1;
                    pStatus[i] = _ippiWarpAffine_interpolation(ippDataType, interpolation,
                        numChannels, pSrc, srcStep, pDstT, dstStep, dstOffset, dstSizeT,
                        pSpec, pOneBuf);
                }
            }
        }
    }
    // checking status for pBuffer allocation
    // and ippDataType checking in switch case
    // for getting pDstT
    check_sts(status);

    // Checking status for tiles
    for (Ipp32u i = 0; i < numThreads; ++i)
    {
        check_sts(pStatus[i]);
    }
EXIT_FUNC
    ippsFree(pInitBuf);
    ippsFree(pSpec);
    ippsFree(pBuffer);
    ippFree(pStatus);
    return status;
}
