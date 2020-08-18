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

#define EXIT_FUNC exitLine:             /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine


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
    IppiPoint dstOffset = { 0, 0 };

#ifdef USE_OPENMP
    IppStatus * pStatus = NULL;
    int numThreads, slice, tail;
    int bufSize1, bufSize2;
    IppiSize dstTileSize, dstLastTileSize;

    int max_num_threads;

#ifdef MAX_NUM_THREADS
    max_num_threads = MAX_NUM_THREADS;
#else
    max_num_threads = omp_get_max_threads();
    if(dstSize.height / max_num_threads < 2)
    {
        max_num_threads = 1;
    }
#endif

#endif

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

#ifdef USE_OPENMP
    if (max_num_threads != 1)
    {
        // General transform function
        // Parallelized only by Y-direction here
    #pragma omp parallel num_threads(max_num_threads)
        {
    #pragma omp master
            {
                numThreads = omp_get_num_threads();
                pStatus = (IppStatus*)ippsMalloc_8u(sizeof(IppStatus) * numThreads);
                if (pStatus == NULL)
                {
                    status = ippStsMemAllocErr;
                }
                if(status == ippStsNoErr)
                {
                    for (int i = 0; i < max_num_threads; ++i) pStatus[i] = ippStsNoErr;
                    // ippSetNumThreads(numThreads);
                    slice = dstSize.height / numThreads;
                    tail = dstSize.height % numThreads;

                    dstTileSize.width = dstSize.width;
                    dstTileSize.height = slice;
                    dstLastTileSize.width = dstSize.width;
                    dstLastTileSize.height = slice + tail;

                    status = ippiWarpGetBufferSize(pSpec, dstTileSize, &bufSize1);
                    if (status == ippStsNoErr) status = ippiWarpGetBufferSize(pSpec, dstLastTileSize, &bufSize2);
                    if (status == ippStsNoErr)
                    {
                        pBuffer = ippsMalloc_8u(bufSize1 * (numThreads - 1) + bufSize2);
                        if (pBuffer == NULL) status = ippStsMemAllocErr;
                    }
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
            status = pStatus[i];
            check_sts(status);
        }
    }
    else
    {
#endif
        status = ippiWarpGetBufferSize(pSpec, dstSize, &bufSize);
        check_sts(status);
        pBuffer = ippsMalloc_8u(bufSize);
        if (pBuffer == NULL)
        {
            status = ippStsMemAllocErr;
            check_sts(status);
        }
        status = _ippiWarpAffine_interpolation(ippDataType, interpolation, numChannels,
            pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
#ifdef USE_OPENMP
    }
#endif

EXIT_FUNC
    ippsFree(pInitBuf);
    ippsFree(pSpec);
    ippsFree(pBuffer);
#ifdef USE_OPENMP
    ippFree(pStatus);
#endif
    return status;
}
