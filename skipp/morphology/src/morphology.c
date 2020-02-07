#include "morphology.h"

#define EXIT_FUNC exitLine:             /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine

IppStatus
ippiMorphologyBorderGetSize(
    IppDataType datatype,
    IppiSize roiSize,
    IppiSize maskSize,
    int numChannels,
    int * pSpecSize,
    int * pBufferSize)
{
    IppStatus status = ippStsNoErr;
    if (numChannels == 1)
    {
        switch (datatype)
        {
        case ipp1u:
        {
            status = ippiMorphologyBorderGetSize_1u_C1R(roiSize, maskSize, pSpecSize, pBufferSize);
            break;
        }
        case ipp8u:
        {
            status = ippiMorphologyBorderGetSize_8u_C1R(roiSize, maskSize, pSpecSize, pBufferSize);
            break;
        }
        case ipp16u:
        {
            status = ippiMorphologyBorderGetSize_16u_C1R(roiSize, maskSize, pSpecSize, pBufferSize);
            break;
        }
        case ipp16s:
        {
            status = ippiMorphologyBorderGetSize_16s_C1R(roiSize, maskSize, pSpecSize, pBufferSize);
            break;
        }
        case ipp32f:
        {
            status = ippiMorphologyBorderGetSize_32f_C1R(roiSize, maskSize, pSpecSize, pBufferSize);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else if (numChannels == 3)
    {
        switch (datatype)
        {

        case ipp8u:
        {
            status = ippiMorphologyBorderGetSize_8u_C3R(roiSize, maskSize, pSpecSize, pBufferSize);
            break;
        }
        case ipp32f:
        {
            status = ippiMorphologyBorderGetSize_32f_C3R(roiSize, maskSize, pSpecSize, pBufferSize);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else if (numChannels == 4)
    {
        switch (datatype)
        {

        case ipp8u:
        {
            status = ippiMorphologyBorderGetSize_8u_C4R(roiSize, maskSize, pSpecSize, pBufferSize);
            break;
        }
        case ipp32f:
        {
            status = ippiMorphologyBorderGetSize_32f_C4R(roiSize, maskSize, pSpecSize, pBufferSize);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else
    {
        status = ippStsErr;
    }
    check_sts(status);

    EXIT_FUNC
        return status;
}


IppStatus
ippiMorphologyBorderInit(
    IppDataType datatype,
    int numChannels,
    IppiSize roiSize,
    const Ipp8u * pMask,
    IppiSize maskSize,
    IppiMorphState* pSpec,
    Ipp8u* pBuffer)
{
    IppStatus status = ippStsNoErr;
    if (numChannels == 1)
    {
        switch (datatype)
        {
        case ipp1u:
        {
            status = ippiMorphologyBorderInit_1u_C1R(roiSize, pMask, maskSize, pSpec, pBuffer);
            break;
        }
        case ipp8u:
        {
            status = ippiMorphologyBorderInit_8u_C1R(roiSize, pMask, maskSize, pSpec, pBuffer);
            break;
        }
        case ipp16u:
        {
            status = ippiMorphologyBorderInit_16u_C1R(roiSize, pMask, maskSize, pSpec, pBuffer);
            break;
        }
        case ipp16s:
        {
            status = ippiMorphologyBorderInit_16s_C1R(roiSize, pMask, maskSize, pSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            status = ippiMorphologyBorderInit_32f_C1R(roiSize, pMask, maskSize, pSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else if (numChannels == 3)
    {
        switch (datatype)
        {

        case ipp8u:
        {
            status = ippiMorphologyBorderInit_8u_C3R(roiSize, pMask, maskSize, pSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            status = ippiMorphologyBorderInit_32f_C3R(roiSize, pMask, maskSize, pSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else if (numChannels == 4)
    {
        switch (datatype)
        {

        case ipp8u:
        {
            status = ippiMorphologyBorderInit_8u_C4R(roiSize, pMask, maskSize, pSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            status = ippiMorphologyBorderInit_32f_C4R(roiSize, pMask, maskSize, pSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else
    {
        status = ippStsErr;
    }
    check_sts(status);
    EXIT_FUNC
        return status;
}


IppStatus
ippiErodeBorder(
    IppDataType datatype,
    int numChannels,
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    IppiSize roiSize,
    IppiBorderType ippBorderType,
    float ippBorderValue,
    const IppiMorphState * pSpec,
    Ipp8u* pBuffer
    )
{
    IppStatus status = ippStsNoErr;

    if (numChannels == 1)
    {
        switch (datatype)
        {
        case ipp1u:
        {
            //status = ippiErodeBorder_1u_C1R(pSrc, srcStep, pDst, dstStep, roiSize, borderType,
            //  borderValue, pSpec, pBuffer);
            status = ippStsErr;
            break;
        }
        case ipp8u:
        {
            Ipp8u ippbordervalue = (Ipp8u)ippBorderValue;
            status = ippiErodeBorder_8u_C1R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
                ippbordervalue, pSpec, pBuffer);
            break;
        }
        case ipp16u:
        {
            Ipp16u ippbordervalue = (Ipp16u)ippBorderValue;
            status = ippiErodeBorder_16u_C1R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
                ippbordervalue, pSpec, pBuffer);
            break;
        }
        case ipp16s:
        {
            Ipp16s ippbordervalue = (Ipp16s)ippBorderValue;
            status = ippiErodeBorder_16s_C1R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
                ippbordervalue, pSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            Ipp32f ippbordervalue = (Ipp32f)ippBorderValue;
            status = ippiErodeBorder_32f_C1R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
                ippbordervalue, pSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else if (numChannels == 3)
    {
        switch (datatype)
        {
        case ipp8u:
        {
            Ipp8u ippbordervalue[3] = { (Ipp8u)ippBorderValue, (Ipp8u)ippBorderValue , (Ipp8u)ippBorderValue };
            status = ippiErodeBorder_8u_C3R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
                ippbordervalue, pSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            Ipp32f ippbordervalue[3] = { (Ipp32f)ippBorderValue, (Ipp32f)ippBorderValue , (Ipp32f)ippBorderValue };
            status = ippiErodeBorder_32f_C3R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
                ippbordervalue, pSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else if (numChannels == 4)
    {
        switch (datatype)
        {

        case ipp8u:
        {
            Ipp8u ippbordervalue[4] = { (Ipp8u)ippBorderValue, (Ipp8u)ippBorderValue , (Ipp8u)ippBorderValue, (Ipp8u)ippBorderValue };
            status = ippiErodeBorder_8u_C4R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
                ippbordervalue, pSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            Ipp32f ippbordervalue[4] = { (Ipp32f)ippBorderValue, (Ipp32f)ippBorderValue , (Ipp32f)ippBorderValue, (Ipp32f)ippBorderValue };
            status = ippiErodeBorder_32f_C4R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
                ippbordervalue, pSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else
    {
        status = ippStsErr;
    }
    return status;
}

IppStatus
ippiDilateBorder(
    IppDataType datatype,
    int numChannels,
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    IppiSize roiSize,
    IppiBorderType ippBorderType,
    float ippBorderValue,
    const IppiMorphState * pSpec,
    Ipp8u* pBuffer
)
{
    IppStatus status = ippStsNoErr;

    if (numChannels == 1)
    {
        switch (datatype)
        {
        case ipp1u:
        {
            //status = ippiDilateBorder_1u_C1R(pSrc, srcStep, pDst, dstStep, roiSize, borderType,
            //  borderValue, pSpec, pBuffer);
            status = ippStsErr;
            break;
        }
        case ipp8u:
        {
            Ipp8u ippbordervalue = (Ipp8u)ippBorderValue;
            status = ippiDilateBorder_8u_C1R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
                ippbordervalue, pSpec, pBuffer);
            break;
        }
        case ipp16u:
        {
            Ipp16u ippbordervalue = (Ipp16u)ippBorderValue;
            status = ippiDilateBorder_16u_C1R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
                ippbordervalue, pSpec, pBuffer);
            break;
        }
        case ipp16s:
        {
            Ipp16s ippbordervalue = (Ipp16s)ippBorderValue;
            status = ippiDilateBorder_16s_C1R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
                ippbordervalue, pSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            Ipp32f ippbordervalue = (Ipp32f)ippBorderValue;
            status = ippiDilateBorder_32f_C1R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
                ippbordervalue, pSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else if (numChannels == 3)
    {
        switch (datatype)
        {
        case ipp8u:
        {
            Ipp8u ippbordervalue[3] = { (Ipp8u)ippBorderValue, (Ipp8u)ippBorderValue , (Ipp8u)ippBorderValue };
            status = ippiDilateBorder_8u_C3R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
                ippbordervalue, pSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            Ipp32f ippbordervalue[3] = { (Ipp32f)ippBorderValue, (Ipp32f)ippBorderValue , (Ipp32f)ippBorderValue };
            status = ippiDilateBorder_32f_C3R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
                ippbordervalue, pSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else if (numChannels == 4)
    {
        switch (datatype)
        {

        case ipp8u:
        {
            Ipp8u ippbordervalue[4] = { (Ipp8u)ippBorderValue, (Ipp8u)ippBorderValue , (Ipp8u)ippBorderValue, (Ipp8u)ippBorderValue };
            status = ippiDilateBorder_8u_C4R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
                ippbordervalue, pSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            Ipp32f ippbordervalue[4] = { (Ipp32f)ippBorderValue, (Ipp32f)ippBorderValue , (Ipp32f)ippBorderValue, (Ipp32f)ippBorderValue };
            status = ippiDilateBorder_32f_C4R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
                ippbordervalue, pSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else
    {
        status = ippStsErr;
    }
    return status;
}

IppStatus
ippiMorphology(
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

    status = ippiMorphologyBorderGetSize(datatype, roiSize, maskSize, numChannels, &specSize, &bufferSize);
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

    status = ippiMorphologyBorderInit(datatype, numChannels, roiSize, pMask, maskSize, pSpec, pBuffer);
    check_sts(status);

    if (ippiFunc == IppiErodeBorder) {
        status = ippiErodeBorder(datatype, numChannels, pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
            ippBorderValue, pSpec, pBuffer);
    }
    else if (ippiFunc == IppiDilateBorder) {
        status = ippiDilateBorder(datatype, numChannels, pSrc, srcStep, pDst, dstStep, roiSize, ippBorderType,
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

/*
int 
ippiDilate_L(
    IppDataType datatype,
    void * pSrc,
    void * pDst,
    IppSizeL img_width,
    IppSizeL img_height,
    int numChannels,
    void * pMask,
    IppSizeL mask_width,
    IppSizeL mask_height,
    IppiBorderType ippBorderType,
    float ippBorderValue
)
{
    IppStatus status = ippStsNoErr;

    Ipp8u* pBuffer = NULL;
    IppiMorphStateL* pMorphSpec = NULL;

    IppiSizeL roiSize = { img_width, img_height };
    IppiSizeL maskSize = { mask_width, mask_height };

    IppSizeL pSpecSize = 0;

    int sizeof_src;
    status = get_sizeof(datatype, &sizeof_src);
    check_sts(status);

    IppSizeL srcStep = numChannels * img_width * (IppSizeL)sizeof_src;
    IppSizeL dstStep = srcStep;

    IppSizeL bufferSize = 0;

    status = ippiDilateGetBufferSize_L(roiSize, maskSize, datatype, numChannels, &bufferSize);
    check_sts(status);

    status = ippiDilateGetSpecSize_L(roiSize, maskSize, &pSpecSize);
    check_sts(status);

    pMorphSpec = (IppiMorphStateL*)ippsMalloc_8u_L(pSpecSize);
    if (pMorphSpec == NULL)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    }
    pBuffer = (Ipp8u*)ippsMalloc_8u_L(bufferSize);
    if (pBuffer == NULL)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    }
    status = ippiDilateInit_L(roiSize, pMask, maskSize, pMorphSpec);
    check_sts(status);
    if (numChannels == 1)
    {
        switch(datatype)
        {
        case ipp1u:
        {
            // TODO
            //Ipp8u ippbordervalue = (Ipp8u)ippBorderValue;
            //status = ippiDilate_1u_C1R_L(pSrc, srcStep, pDst, dstStep, roiSize,
            //  borderType, borderValue, pMorphSpec, pBuffer);
            status = ippStsErr;
            break;
        }
        case ipp8u:
        {
            Ipp8u ippbordervalue = (Ipp8u)ippBorderValue;
            status = ippiDilate_8u_C1R_L(pSrc, srcStep, pDst, dstStep, roiSize,
                ippBorderType, ippbordervalue, pMorphSpec, pBuffer);
            break;
        }
        case ipp16u:
        {
            Ipp16u ippbordervalue = (Ipp16u)ippBorderValue;
            status = ippiDilate_16u_C1R_L(pSrc, srcStep, pDst, dstStep, roiSize,
                ippBorderType, ippbordervalue, pMorphSpec, pBuffer);
            break;
        }
        case ipp16s:
        {
            Ipp16s ippbordervalue = (Ipp16s)ippBorderValue;
            status = ippiDilate_16s_C1R_L(pSrc, srcStep, pDst, dstStep, roiSize,
                ippBorderType, ippbordervalue, pMorphSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            Ipp32f ippbordervalue[] = { (Ipp32f)ippBorderValue }; // ~~ TODO check
            status = ippiDilate_32f_C1R_L(pSrc, srcStep, pDst, dstStep, roiSize,
                ippBorderType, ippbordervalue, pMorphSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsErr;
        }
        }
    }
    else if (numChannels == 3)
    {
        switch (datatype)
        {
        case ipp8u:
        {
            Ipp8u ippbordervalue[3] = { (Ipp8u)ippBorderValue, (Ipp8u)ippBorderValue , (Ipp8u)ippBorderValue };
            status = ippiDilate_8u_C3R_L(pSrc, srcStep, pDst, dstStep, roiSize,
                ippBorderType, ippbordervalue, pMorphSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            Ipp32f ippbordervalue[3] = { (Ipp32f)ippBorderValue, (Ipp32f)ippBorderValue , (Ipp32f)ippBorderValue };
            status = ippiDilate_32f_C3R_L(pSrc, srcStep, pDst, dstStep, roiSize,
                ippBorderType, ippbordervalue, pMorphSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsErr;
        }
        }

    }
    else // TODO also for 4 channels
    {
        status = ippStsErr;
    }
    check_sts(status);
EXIT_FUNC
    ippsFree(pBuffer);
    ippsFree(pMorphSpec);
    return status;
}

int
ippiErode_L(
    IppDataType datatype,
    void * pSrc,
    void * pDst,
    IppSizeL img_width,
    IppSizeL img_height,
    int numChannels,
    void * pMask,
    IppSizeL mask_width,
    IppSizeL mask_height,
    IppiBorderType ippBorderType,
    float ippBorderValue
)
{
    IppStatus status = ippStsNoErr;

    Ipp8u* pBuffer = NULL;
    IppiMorphStateL* pMorphSpec = NULL;

    IppiSizeL roiSize = { img_width, img_height };
    IppiSizeL maskSize = { mask_width, mask_height };

    IppSizeL pSpecSize = 0;

    int sizeof_src;
    status = get_sizeof(datatype, &sizeof_src);
    check_sts(status);

    IppSizeL srcStep = numChannels * img_width * (IppSizeL)sizeof_src;
    IppSizeL dstStep = srcStep;

    IppSizeL bufferSize = 0;

    status = ippiErodeGetBufferSize_L(roiSize, maskSize, datatype, numChannels, &bufferSize);
    check_sts(status);

    status = ippiErodeGetSpecSize_L(roiSize, maskSize, &pSpecSize);
    check_sts(status);

    pMorphSpec = (IppiMorphStateL*)ippsMalloc_8u_L(pSpecSize);
    if (pMorphSpec == NULL)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    }
    pBuffer = (Ipp8u*)ippsMalloc_8u_L(bufferSize);
    if (pBuffer == NULL)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    }
    status = ippiErodeInit_L(roiSize, pMask, maskSize, pMorphSpec);
    check_sts(status);
    if (numChannels == 1)
    {
        switch (datatype)
        {
        case ipp1u:
        {
            // TODO
            //Ipp8u ippbordervalue = (Ipp8u)ippBorderValue;
            //status = ippiErode_1u_C1R_L(pSrc, srcStep, pDst, dstStep, roiSize,
            //  borderType, borderValue, pMorphSpec, pBuffer);
            status = ippStsErr;
            break;
        }
        case ipp8u:
        {
            Ipp8u ippbordervalue = (Ipp8u)ippBorderValue;
            status = ippiErode_8u_C1R_L(pSrc, srcStep, pDst, dstStep, roiSize,
                ippBorderType, ippbordervalue, pMorphSpec, pBuffer);
            break;
        }
        case ipp16u:
        {
            Ipp16u ippbordervalue = (Ipp16u)ippBorderValue;
            status = ippiErode_16u_C1R_L(pSrc, srcStep, pDst, dstStep, roiSize,
                ippBorderType, ippbordervalue, pMorphSpec, pBuffer);
            break;
        }
        case ipp16s:
        {
            Ipp16s ippbordervalue = (Ipp16s)ippBorderValue;
            status = ippiErode_16s_C1R_L(pSrc, srcStep, pDst, dstStep, roiSize,
                ippBorderType, ippbordervalue, pMorphSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            Ipp32f ippbordervalue[] = { (Ipp32f)ippBorderValue }; // ~~ TODO check
            status = ippiErode_32f_C1R_L(pSrc, srcStep, pDst, dstStep, roiSize,
                ippBorderType, ippbordervalue, pMorphSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsErr;
        }
        }
    }
    else if (numChannels == 3)
    {
        switch (datatype)
        {
        case ipp8u:
        {
            Ipp8u ippbordervalue[3] = { (Ipp8u)ippBorderValue, (Ipp8u)ippBorderValue , (Ipp8u)ippBorderValue };
            status = ippiErode_8u_C3R_L(pSrc, srcStep, pDst, dstStep, roiSize,
                ippBorderType, ippbordervalue, pMorphSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            Ipp32f ippbordervalue[3] = { (Ipp32f)ippBorderValue, (Ipp32f)ippBorderValue , (Ipp32f)ippBorderValue };
            status = ippiErode_32f_C3R_L(pSrc, srcStep, pDst, dstStep, roiSize,
                ippBorderType, ippbordervalue, pMorphSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsErr;
        }
        }

    }
    else // TODO also for 4 channels
    {
        status = ippStsErr;
    }
    check_sts(status);
EXIT_FUNC
    ippsFree(pBuffer);
    ippsFree(pMorphSpec);
    return status;
}
*/