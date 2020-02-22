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
