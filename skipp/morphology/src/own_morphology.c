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
//    scikit-ipp's own functions for morphology transfomations of image, that uses
//    Intel(R) Integrated Performance Primitives (Intel(R) IPP).
//
////////////////////////////////////////////////////////////////////////////////////////
#include "own_morphology.h"

#define EXIT_FUNC exitLine:             /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine

////////////////////////////////////////////////////////////////////////////////////////
//
//    own_Morphology
//
//        scikit-ipp's own image processing functions that perform morphological
//        operations on images.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
own_Morphology(
    IppDataType datatype,
    ippiMorphologyFunction ippiFunc,
    void * pSrc,
    void * pDst,
    int img_width,
    int img_height,
    int numChannels,
    void * pMask,
    int mask_width,
    int mask_height,
    IppiBorderType ippBorderType,
    float ippBorderValue)
{
    IppStatus status = ippStsNoErr;

    IppiMorphState* pSpec = NULL;
    Ipp8u* pBuffer = NULL;
    IppiSize roiSize = { img_width, img_height };
    IppiSize maskSize = { mask_width, mask_height };

    int sizeof_src;
    status = get_sizeof(datatype, &sizeof_src);
    check_sts(status);

    int srcStep = numChannels * img_width * sizeof_src;
    int dstStep = srcStep;

    int specSize = 0;
    int bufferSize = 0;

    status = ippiMorphologyBorderGetSize(datatype, roiSize, maskSize, numChannels,
        &specSize, &bufferSize);
    check_sts(status);
    pSpec = (IppiMorphState*)ippsMalloc_8u(specSize);
    if (pSpec == NULL)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    }
    pBuffer = (Ipp8u*)ippsMalloc_8u(bufferSize);
    if (pBuffer == NULL)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    }

    status = ippiMorphologyBorderInit(datatype, numChannels, roiSize, pMask, maskSize,
        pSpec, pBuffer);
    check_sts(status);

    if (ippiFunc == IppiErodeBorder) {
        status = ippiErodeBorder(datatype, numChannels, pSrc, srcStep, pDst, dstStep,
            roiSize, ippBorderType,
            ippBorderValue, pSpec, pBuffer);
    }
    else if (ippiFunc == IppiDilateBorder) {
        status = ippiDilateBorder(datatype, numChannels, pSrc, srcStep, pDst, dstStep,
            roiSize, ippBorderType,
            ippBorderValue, pSpec, pBuffer);
    }
    else
    {
        status = ippStsErr;
    }
    check_sts(status);
EXIT_FUNC
    ippsFree(pBuffer);
    ippsFree(pSpec);
    return status;
}
