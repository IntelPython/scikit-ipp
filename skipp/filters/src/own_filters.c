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
#include "own_filters.h"

#define EXIT_FUNC exitLine:                                  /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine 


////////////////////////////////////////////////////////////////////////////////////////
//
//    own_FilterBorder
//
//          General border filter
//
//    Note: own_FilterBorder functions on the backend for implementing
//          gaussian filtering of image as is in scikit-image.
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
    float ippBorderValue)
{
    IppStatus status = ippStsNoErr;
    int sizeof_src;
    IppiSize roiSize = { img_width, img_height };         // Size of source and
                                                          // destination ROI in pixels
    IppiSize kernelSize = { kernel_width, kernel_height };

    Ipp8u * pBuffer = NULL;                               // Pointer to the work buffer
    IppiFilterBorderSpec * pSpec = NULL;                  // context structure
    int iTmpBufSize = 0;                                  // Common work buffer size
    int iSpecSize = 0;                                    // Common work buffer size
    int srcStep;                                          // Steps, in bytes, through the
    int dstStep;

    status = get_sizeof(ippImageDataType, &sizeof_src);
    check_sts(status);
    srcStep = numChannels * img_width * sizeof_src;
    dstStep = srcStep;

    status = ippiFilterBorderGetSize(kernelSize, roiSize, ippImageDataType, ippKernelDataType,
        numChannels, &iSpecSize, &iTmpBufSize);
    check_sts(status);
    pSpec = (IppiFilterBorderSpec *)ippsMalloc_8u(iSpecSize);
    if (pSpec == NULL)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    };
    pBuffer = ippsMalloc_8u(iTmpBufSize);
    if (pBuffer == NULL)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    };
    status = ippiFilterBorderInit(ippKernelDataType, pKernel, kernelSize, 4,
        ippImageDataType, numChannels, roundMode, pSpec);
    check_sts(status);
    status = ippiFilterBorder(ippImageDataType, pSrc, srcStep, pDst, dstStep, roiSize,
        numChannels, ippBorderType, ippBorderValue, pSpec, pBuffer);
    check_sts(status);
    EXIT_FUNC
        ippsFree(pBuffer);
    ippsFree(pSpec);
    return status;
}

////////////////////////////////////////////////////////////////////////////////////////
//
//    own_FilterGaussian
//
//          Gaussian filter
//
//    Note: own_FilterGaussian functions on the backend for implementing
//          gaussian filtering of image as is in scikit-image.
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
    float ippBorderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp8u *pBuffer = NULL;
    IppFilterGaussianSpec* pSpec = NULL;                      // context structure 
    int iTmpBufSize = 0, iSpecSize = 0;                       // Common work buffer size
    int sizeof_src;
    status = get_sizeof(ippDataType, &sizeof_src);
    check_sts(status);
    int srcStep = numChannels * img_width * sizeof_src;       // Steps, in bytes, through
    int dstStep = srcStep;                                    // the source/destination
                                                              //images

    IppiSize roiSize = { img_width, img_height };             // Size of source/destination
                                                              //ROI in pixels
    Ipp32f sigma = (Ipp32f)sigma_;
    // Pointer to the work buffer 
    status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ippDataType,
        numChannels, &iSpecSize, &iTmpBufSize);
    check_sts(status);
    pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
    if (pSpec == NULL)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    }
    pBuffer = (Ipp8u *)ippsMalloc_8u(iTmpBufSize);
    if (pBuffer == NULL)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    }
    status = ippiFilterGaussianInit(roiSize, kernelSize, sigma,
        ippBorderType, ippDataType, numChannels, pSpec, pBuffer);
    check_sts(status);
    status = ippiFilterGaussianBorder(ippDataType, pSrc, srcStep, pDst, dstStep,
        roiSize, numChannels, ippBorderValue, pSpec, pBuffer);
    EXIT_FUNC
        ippsFree(pBuffer);
    ippsFree(pSpec);
    return status;
}

////////////////////////////////////////////////////////////////////////////////////////
//
//    own_FilterGaussian
//
//          Median filter
//
//          own_FilterGaussian uses functions on the backend for
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
    float ippBorderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp8u * pBuffer = NULL;                                 // Pointer to the work buffer
    int bufferSize;
    int sizeof_src;
    status = get_sizeof(ippDataType, &sizeof_src);
    check_sts(status);

    int srcStep = numChannels * img_width * sizeof_src;     // Steps, in bytes, through
    int dstStep = srcStep;                                  // thesource/destination
                                                            //images

    IppiSize dstRoiSize = { img_width, img_height };        // Size of source and
                                                            // destination ROI in pixels

    IppiSize maskSize = { mask_width, mask_height };        // Size of source and
                                                            // destination ROI in pixels
    status = ippiFilterMedianBorderGetBufferSize(dstRoiSize,
        maskSize,
        ippDataType,
        numChannels,
        &bufferSize);
    check_sts(status);
    pBuffer = ippsMalloc_8u(bufferSize);
    if (pBuffer == NULL)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    }
    status = ippiFilterMedianBorder(ippDataType, pSrc, srcStep, pDst, dstStep, dstRoiSize,
        numChannels, maskSize, ippBorderType, ippBorderValue, pBuffer);
    check_sts(status);
    EXIT_FUNC
        ippsFree(pBuffer);
    return status;
}

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
    float ippBorderValue)
{
    IppStatus status = ippStsNoErr;
    int kernel_width = 3;
    int kernel_height = 3;

    Ipp32f own_kernel_laplace[] = own_Laplace_KERNEL_3x3;
    IppDataType ippKernelDataType = ipp32f;

    IppRoundMode roundMode = ippRndNear;

    status = own_FilterBorder(ippDataType,
        ippKernelDataType,
        pSrc,
        pDst,
        own_kernel_laplace,
        img_width,
        img_height,
        kernel_width,
        kernel_height,
        numChannels,
        roundMode,
        ippBorderType,
        ippBorderValue);
    return status;
}

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
    int numChannels)
{
    IppStatus status = ippStsNoErr;
    IppStatus * pStatus = NULL;

    Ipp8u * pBuffer = NULL;                         // Pointer to the work buffer
    int bufferSize;
    IppiBorderType ippBorderType = ippBorderRepl;
    float ippBorderValue = 0.0;
    int srcStep;                                    // Steps, in bytes, through the
    int dstStep;
    IppiSize roiSize = { img_width, img_height };   // Size of source and
                                                    // destination ROI in pixels
    Ipp32f value;
    IppiMaskSize maskSize = ippMskSize3x3;
    IppNormType normType = ippNormL2;               // As is in skimage.filters.prewitt

    if (!( numChannels == 1 &&
          ippSrcDataType == ipp32f &&
          ippDstDataType == ipp32f ))
    {
        status = ippStsErr;
        check_sts(status);
    }
    int sizeof_src;
    status = get_sizeof(ippSrcDataType, &sizeof_src);
    check_sts(status);

    int sizeof_dst;
    status = get_sizeof(ippDstDataType, &sizeof_dst);
    check_sts(status);
    // currently Intel(R) IPP supports only 1C images for filtering by prewitt kernels
    srcStep = numChannels * img_width * sizeof_src;
    dstStep = numChannels * img_width * sizeof_dst;

    int max_num_threads;
    int numThreads, slice, tail;
    IppiSize dstTileSize, dstLastTileSize;

#ifdef MAX_NUM_THREADS
    max_num_threads = MAX_NUM_THREADS;
#else
    max_num_threads = omp_get_max_threads();
    if(roiSize.height / max_num_threads < 2)
    {
        max_num_threads = 1;
    }
#endif

    switch (edgeKernel)
    {
    case own_filterSobelVert:
    {
        status = ippiFilterSobelVertBorderGetBufferSize(roiSize, maskSize, ippSrcDataType,
            ippDstDataType, numChannels, &bufferSize);
        break;
    }
    case own_filterSobelHoriz:
    {
        status = ippiFilterSobelHorizBorderGetBufferSize(roiSize, maskSize, ippSrcDataType,
            ippDstDataType, numChannels, &bufferSize);
        break;
    }
    case own_filterSobel:
    {
        status = ippiFilterSobelGetBufferSize(roiSize, maskSize, normType, ippSrcDataType,
            ippDstDataType, numChannels, &bufferSize);
        break;
    }
    case own_filterPrewittVert:
    {
        status = ippiFilterPrewittVertBorderGetBufferSize(roiSize, maskSize, ippSrcDataType,
                       ippDstDataType, numChannels, &bufferSize);
        break;
    }
    case own_filterPrewittHoriz:
    {
        status = ippiFilterPrewittHorizBorderGetBufferSize(roiSize, maskSize, ippSrcDataType,
                       ippDstDataType, numChannels, &bufferSize);
        break;
    }
    default:
    {
        status = ippStsErr;
    }
    }
    check_sts(status);
    pBuffer = ippsMalloc_8u(bufferSize);
    if (pBuffer == NULL)
    {
        check_sts(status = ippStsMemAllocErr);
    };

    switch (edgeKernel)
    {
    case own_filterSobelVert:
    {
        status = ippiFilterSobelVertBorder(ippSrcDataType, ippDstDataType, pSrc, srcStep, pDst, dstStep,
            roiSize, numChannels, maskSize, ippBorderType, ippBorderValue, pBuffer);
        break;
    }
    case own_filterSobelHoriz:
    {
        status = ippiFilterSobelHorizBorder(ippSrcDataType, ippDstDataType, pSrc, srcStep, pDst, dstStep,
            roiSize, numChannels, maskSize, ippBorderType, ippBorderValue, pBuffer);
        break;
    }
    case own_filterSobel:
    {
        status = ippiFilterSobel(ippSrcDataType, ippDstDataType, pSrc, srcStep, pDst, dstStep, roiSize,
            numChannels, maskSize, normType, ippBorderType, ippBorderValue, pBuffer);
        break;
    }
    case own_filterPrewittVert:
    {
        status = ippiFilterPrewittVertBorder(ippSrcDataType, ippDstDataType, pSrc, srcStep, pDst, dstStep,
            roiSize, numChannels, maskSize, ippBorderType, ippBorderValue, pBuffer);
        break;
    }
    case own_filterPrewittHoriz:
    {
        status = ippiFilterPrewittHorizBorder(ippSrcDataType, ippDstDataType, pSrc, srcStep, pDst, dstStep,
            roiSize, numChannels, maskSize, ippBorderType, ippBorderValue, pBuffer);
        break;
    }
    default:
    {
        status = ippStsErr;
    }
    }
    check_sts(status);
    // suppose that processing only ipp32f
    switch (edgeKernel)
    {
    case own_filterSobelVert:
    {
        // NOTE:
        // scikit-image's Sobel filter is a wrapper on scipy.ndimage's
        // `convolve` func. `convolve` uses `reflect` border mode.
        // `reflect` border mode is equivalent of Intel IPP ippBorderMirrorR border type
        // ippiFilterSobelVertBorder_<mode> doesn't supports this border type
        value = (Ipp32f)(-4.0);
        break;
    }
    case own_filterSobelHoriz:
    {
        // NOTE:
        // scikit-image's Sobel filter is a wrapper on scipy.ndimage's
        // `convolve` func. `convolve` uses `reflect` border mode.
        // `reflect` border mode is equivalent of Intel IPP ippBorderMirrorR border type
        // ippiFilterSobelHorizBorder_<mode> doesn't supports this border type
        value = (Ipp32f)4.0;
        break;
    }
    case own_filterSobel:
    {
        // NOTE:
        // scikit-image's Sobel filter is a wrapper on scipy.ndimage's
        // `convolve` func. `convolve` uses `reflect` border mode.
        // `reflect` border mode is equivalent of Intel IPP ippBorderMirrorR border type
        // ippiFilterSobelBorder_<mode> doesn't supports this border type
        value = (Ipp32f)(4.0 * (Ipp32f)IPP_SQRT2);
        break;
    }
    case own_filterPrewittVert:
    {
        // NOTE:
        // scikit-image's prewitt filter is a wrapper on scipy.ndimage's
        // `convolve` func. `convolve` uses `reflect` border mode.
        // `reflect` border mode is equivalent of Intel IPP ippBorderMirrorR border type  
        // ippiFilterPrewittVertBorder_<mode> doesn't supports this border type
        value = (Ipp32f)-3.0;
        break;
    }
    case own_filterPrewittHoriz:
    {
        // NOTE:
        // scikit-image's prewitt filter is a wrapper on scipy.ndimage's
        // `convolve` func. `convolve` uses `reflect` border mode.
        // `reflect` border mode is equivalent of Intel IPP ippBorderMirrorR border type
        // ippiFilterPrewittHorizBorder_<mode> doesn't supports this border type
        value = (Ipp32f)3.0;
        break;
    }
    default:
    {
        status = ippStsErr;
    }
    }
    if (max_num_threads != 1)
    {
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
                    for (int i = 0; i < numThreads; ++i) pStatus[i] = ippStsNoErr;

                    slice = roiSize.height / numThreads;
                    tail = roiSize.height % numThreads;

                    dstTileSize.width = roiSize.width;
                    dstTileSize.height = slice;
                    dstLastTileSize.width = roiSize.width;
                    dstLastTileSize.height = slice + tail;
                }
            }
    #pragma omp barrier
            {
                if (status == ippStsNoErr)
                {
                    Ipp32u  i;
                    void * pDstT = NULL;
                    i = omp_get_thread_num();
                    IppiPoint dstOffset = { 0, 0 };
                    IppiSize  dstSizeT = dstTileSize;

                    dstSizeT.height = slice;
                    dstOffset.y += i * slice;

                    if (i == numThreads - 1) dstSizeT = dstLastTileSize;

                    pDstT = (void*)((Ipp8u*)pDst + dstOffset.y * (Ipp32s)dstStep);
                    pStatus[i] = ippiDivC_32f_C1IR(value, pDstT, dstStep, dstSizeT);
                }
            }
        }
        check_sts(status);
        // Checking status for slices and tile
        for (Ipp32u i = 0; i < numThreads; ++i)
        {
            status = pStatus[i];
            check_sts(status);
        }
    }
    else
    {
        status = ippiDivC_32f_C1IR(value, pDst, dstStep, roiSize);
        check_sts(status);
    }
EXIT_FUNC
    ippsFree(pStatus);
    ippsFree(pBuffer);
    return status;
}

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
    int numChannels)
{
    // computes the square root of the sum of squares of the horizontal
    // and vertical Prewitt transforms.
    // sqrt(A**2 + B**2)/sqrt(2) 
    IppStatus status = ippStsNoErr;
    Ipp32f * pAsrcDst = NULL;    // Pointer to sobel_h result
    Ipp32f * pBsrcDst = NULL;    // Pointer to sobel_v result

    IppiSize roiSize = { img_width, img_height };  // Size of source-destination
                                                   // ROI in pixels

    // currently supporeted only Ipp32f input/output
    if (!(ippSrcDataType == ipp32f &&
        ippSrcDataType == ipp32f &&
        numChannels == 1 &&
        edgeKernel == own_filterPrewitt))
    {
        status = ippStsErr;
        check_sts(status);
    }
    int sizeofIppDataType = sizeof(Ipp32f);
    pAsrcDst = (void *)ippsMalloc_8u((img_width * sizeofIppDataType * numChannels) * img_height);
    if (pAsrcDst == NULL)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    };
    status = own_FilterEdge(own_filterSobelHoriz, ippSrcDataType, ippDstDataType, pSrc, pAsrcDst,
        img_width, img_height, numChannels);
    check_sts(status);

    pBsrcDst = (void *)ippsMalloc_8u((img_width * sizeofIppDataType * numChannels) * img_height);
    if (pBsrcDst == NULL)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    };
    status = own_FilterEdge(own_filterSobelVert, ippSrcDataType, ippDstDataType, pSrc, pBsrcDst,
        img_width, img_height, numChannels);
    check_sts(status);

    int srcDstStep = sizeofIppDataType * img_width * numChannels;

    status = ippiSqr_32f_C1IR(pAsrcDst, srcDstStep, roiSize);
    check_sts(status);

    status = ippiSqr_32f_C1IR(pBsrcDst, srcDstStep, roiSize);
    check_sts(status);

    status = ippiAdd_32f_C1R(pAsrcDst, srcDstStep, pBsrcDst, srcDstStep, pDst, srcDstStep, roiSize);
    check_sts(status);

    status = ippiSqrt_32f_C1IR(pDst, srcDstStep, roiSize);
    check_sts(status);

    Ipp32f sqrt2 = (Ipp32f)IPP_SQRT2;

    status = ippiDivC_32f_C1IR(sqrt2, pDst, srcDstStep, roiSize);
    check_sts(status);

EXIT_FUNC
    ippsFree(pAsrcDst);
    ippsFree(pBsrcDst);
    return status;
}
