
////////////////////////////////////////////////////////////////////////////////////////
//
//    scikit-ipp's own functions for morphology transfomations of image, that uses
//    Intel(R) IPP.
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
