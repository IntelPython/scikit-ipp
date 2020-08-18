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
//    scikit-ipp's own functions for resizing images, that uses
//    Intel(R) Integrated Performance Primitives (Intel(R) IPP)
//
////////////////////////////////////////////////////////////////////////////////////////
#include "own_resize.h"

#define EXIT_FUNC exitLine:             /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine


////////////////////////////////////////////////////////////////////////////////////////
//
//    own_Resize
//
//          Changes an image size using the different interpolation methods.
//          own_Resize uses Intel(R) IPP functions on the backend for implementing
//          resizing of image as is in scikit-image.
//
//    Note: currently ippBorderValue is disabled. TODO: needs implementing.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
own_Resize(
    IppDataType ippDataType,
    void * pSrc,
    void * pDst,
    int img_width,
    int img_height,
    int dst_width,
    int dst_height,
    int numChannels,
    Ipp32u antialiasing,
    Ipp32u numLobes,
    IppiInterpolationType interpolation,
    IppiBorderType ippBorderType,
    double ippBorderValue)
{
    // TODO:
    // currently ippBorderValue is not used
    IppStatus status = ippStsNoErr;
    IppiResizeSpec_32f *pSpec = NULL;                    // Pointer to the specification
                                                         // structure
    Ipp8u * pBuffer = NULL;                              // Pointer to the work buffer
    Ipp8u * pInitBuf = NULL;

    Ipp32f valueB = 0.0;                                 // for ippCubic interpolation
    Ipp32f valueC = 0.5;                                 // for ippCubic interpolation

    // Ipp32u numLobes                                   // for ippLanczos interpolation

    Ipp64f pBorderValue[4];

    IppiSize srcSize = { img_width, img_height };        // Size of source image
    IppiSize dstSize = { dst_width, dst_height };        // size of destination images
    int srcStep, dstStep;                                // Steps, in bytes, through the
                                                         // source/destination images

    IppiPoint srcOffset = {0, 0};                        // Offset of the source and
    IppiPoint dstOffset = {0, 0};                      // destination image ROI with
                                                         // respect to the destination
                                                         // image origin
    int specSize = 0, initSize = 0, bufSize = 0;         // Work buffer size

#ifdef USE_OPENMP
    IppiBorderSize borderSize = {0, 0, 0, 0};
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

    IppStatus * pStatus = NULL;
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

    // Calculation of work buffer size
    status = ippiResizeGetSize(ippDataType, srcSize, dstSize, interpolation,
        antialiasing, &specSize, &initSize);
    check_sts(status);

    // Memory allocation
    pSpec = (IppiResizeSpec_32f*)ippsMalloc_8u(specSize + initSize);
    if (pSpec == NULL)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    }
    pInitBuf = (Ipp8u*)pSpec + initSize;

    // Filter initialization
    if (antialiasing == 0)
    {
        // ippiResizeNearestInit
        // ippiResizeLinearInit
        // ippiResizeCubicInit
        // ippiResizeLanczosInit
        // ippiResizeSuperInit
        switch (interpolation)
        {
        case ippNearest:
        {
            status = ippiResizeNearestInit(ippDataType, srcSize, dstSize, pSpec);
            break;
        }
        case ippLinear:
        {
            status = ippiResizeLinearInit(ippDataType, srcSize, dstSize, pSpec);
            break;
        }
        case ippCubic:
        {
            status = ippiResizeCubicInit(ippDataType, srcSize, dstSize, valueB,
                valueC, pSpec, pInitBuf);
            break;
        }
        case ippLanczos:
        {
            status = ippiResizeLanczosInit(ippDataType, srcSize, dstSize, numLobes,
                pSpec, pInitBuf);
            break;
        }
        case ippSuper:
        {
            status = ippiResizeSuperInit(ippDataType, srcSize, dstSize, pSpec);
            break;
        }
        default:
        {
            status = ippStsInterpolationErr;
        }
        }
    }
    else if (antialiasing == 1)
    {
        // ippiResizeAntialiasingLinearInit
        // ippiResizeAntialiasingCubicInit
        switch (interpolation)
        {
        case ippLinear:
        {
            status = ippiResizeAntialiasingLinearInit(srcSize, dstSize,
                pSpec, pInitBuf);
            break;
        }
        case ippCubic:
        {
            status = ippiResizeAntialiasingCubicInit(srcSize, dstSize,
                valueB, valueC, pSpec, pInitBuf);
            break;
        }
        default:
        {
            status = ippStsInterpolationErr;
        }
        }
    }
    else
    {
        status = ippStsErr;
    }
    check_sts(status);
#ifdef USE_OPENMP
    if (max_num_threads != 1)
    {
        status = ippiResizeGetBorderSize(ippDataType, pSpec, &borderSize);
        check_sts(status);
        /* General transform function */
        /* Parallelized only by Y-direction here */
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
                    slice = dstSize.height / numThreads;
                    tail  = dstSize.height % numThreads;

                    dstTileSize.width = dstLastTileSize.width = dstSize.width;
                    dstTileSize.height = slice;
                    dstLastTileSize.height = slice + tail;

                    status = ippiResizeGetBufferSize(ippDataType, pSpec, dstTileSize,
                        numChannels, &bufSize1);
                    if (status == ippStsNoErr) ippiResizeGetBufferSize(ippDataType, pSpec,
                        dstTileSize, numChannels, &bufSize2);
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
                    Ipp32u  i;
                    void  *pSrcT;
                    void *pDstT;
                    Ipp8u  *pOneBuf;
                    IppiPoint srcOffset = {0, 0};
                    IppiPoint dstOffset = {0, 0};
                    IppiSize  srcSizeT = srcSize;
                    IppiSize  dstSizeT = dstTileSize;

                    i = omp_get_thread_num();
                    dstSizeT.height = slice;
                    dstOffset.y += i * slice;

                    if (i == numThreads - 1) dstSizeT = dstLastTileSize;

                    pStatus[i] = ippiResizeGetSrcRoi(ippDataType, pSpec, dstOffset,
                                                     dstSizeT, &srcOffset, &srcSizeT);
                    if(pStatus[i] == ippStsNoErr)
                    {
                        pSrcT = (void*)((Ipp8u*)pSrc + srcOffset.y * srcStep);
                        pDstT = (void*)((Ipp8u*)pDst + dstOffset.y * dstStep);

                        pOneBuf = pBuffer + i * bufSize1;

                        if (antialiasing == 0)
                        {
                            // TODO pBorderValue
                            pStatus[i] = ippiResize(ippDataType, pSrcT, srcStep, pDstT, dstStep, dstOffset,
                                dstSizeT, numChannels, ippBorderType, 0, interpolation, pSpec, pOneBuf);
                        }
                        else if (antialiasing == 1)
                        {
                            // TODO pBorderValue
                            pStatus[i] = ippiResizeAntialiasing(ippDataType, pSrcT, srcStep, pDstT, dstStep,
                                dstOffset, dstSizeT, numChannels, ippBorderType, 0, pSpec, pOneBuf);
                        }
                        else
                        {
                            pStatus[i] = ippStsErr;
                        }
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
        status = ippiResizeGetBufferSize(ippDataType, pSpec, dstSize,
            numChannels, &bufSize);
        check_sts(status);

        pBuffer = ippsMalloc_8u(bufSize);

        if (pBuffer == NULL)
        {
            status = ippStsNoMemErr;
            check_sts(status);
        }
        if (antialiasing == 0)
        {
            // TODO pBorderValue
            status = ippiResize(ippDataType, pSrc, srcStep, pDst, dstStep, dstOffset,
                dstSize, numChannels, ippBorderType, 0, interpolation, pSpec, pBuffer);
        }
        else if (antialiasing == 1)
        {
            // TODO pBorderValue
            status = ippiResizeAntialiasing(ippDataType, pSrc, srcStep, pDst, dstStep,
                dstOffset, dstSize, numChannels, ippBorderType, 0, pSpec, pBuffer);
        }
        else
        {
            status = ippStsErr;
        }
        check_sts(status);
#ifdef USE_OPENMP
    }
#endif
EXIT_FUNC
    ippsFree(pSpec);
    ippsFree(pBuffer);
#ifdef USE_OPENMP
    ippFree(pStatus);
#endif
    return status;
}
