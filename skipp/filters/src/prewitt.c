#include "prewitt.h"

#define EXIT_FUNC exitLine:             /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine

int
ippiFilterPrewittHorizBorder(
    IppDataTypeIndex input_index,
    IppDataTypeIndex output_index,
    void * pInput,
    void * pOutput,
    int img_width,
    int img_height,
    int numChannels,
    IppiMaskSize mask,
    IppiBorderType ippBorderType,
    float ippBorderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp8u *pBuffer = NULL;                         // Pointer to the work buffer
    int bufferSize;
    int srcStep;                                   // Steps, in bytes, through the
    int dstStep;
    IppiSize roiSize = { img_width, img_height };  // Size of source and
                                                   // destination ROI in pixels
    IppDataType ippSrcDataType;
    IppDataType ippDstDataType;
    int sizeof_src;
    status = sizeof_ipp_dtype_by_index(&sizeof_src, input_index);
    check_sts(status);

    int sizeof_dst;
    status = sizeof_ipp_dtype_by_index(&sizeof_dst, output_index);
    check_sts(status);

    srcStep = numChannels * img_width * sizeof_src;
    dstStep = numChannels * img_width * sizeof_dst;

    status = ipp_type_index_as_IppDataType(&ippSrcDataType, input_index);
    check_sts(status);

    status = ipp_type_index_as_IppDataType(&ippDstDataType, output_index);
    check_sts(status);

    status = ippiFilterPrewittHorizBorderGetBufferSize(roiSize, mask, ippSrcDataType, ippDstDataType,
        numChannels, &bufferSize);
    check_sts(status);
    pBuffer = ippsMalloc_8u(bufferSize);
    if (NULL == pBuffer)
    {
        check_sts(status = ippStsMemAllocErr);
    };
    if (numChannels == 1)
    {
        if (input_index == ipp8u_index && output_index == ipp16s_index) {
            Ipp8u ippbordervalue = (Ipp8u)ippBorderValue;
            status = ippiFilterPrewittHorizBorder_8u16s_C1R(pInput, srcStep, pOutput, dstStep, roiSize,
                mask, ippBorderType, ippbordervalue, pBuffer);
        }
        else if (input_index == ipp16s_index && output_index == ipp16s_index) {
            // 16s_C1R
            Ipp16s ippbordervalue = (Ipp16s)ippBorderValue;
            status = ippiFilterPrewittHorizBorder_16s_C1R(pInput, srcStep, pOutput, dstStep, roiSize,
                mask, ippBorderType, ippbordervalue, pBuffer);
        }
        else if (input_index == ipp32f_index && output_index == ipp32f_index) {
            // 32f_C1R
            Ipp32f ippbordervalue = (Ipp32f)ippBorderValue;
            status = ippiFilterPrewittHorizBorder_32f_C1R(pInput, srcStep, pOutput, dstStep, roiSize,
                mask, ippBorderType, ippbordervalue, pBuffer);
        }
        else
        {
            status = ippStsErr;
        }
    }
    else
    {
        status = ippStsErr;
    }
    check_sts(status);
EXIT_FUNC
    ippsFree(pBuffer);
    return (int)status;
}

int
ippiFilterPrewittVertBorder(
    IppDataTypeIndex input_index,
    IppDataTypeIndex output_index,
    void * pInput,
    void * pOutput,
    int img_width,
    int img_height,
    int numChannels,
    IppiMaskSize mask,
    IppiBorderType ippBorderType,
    float ippBorderValue
)
{
    IppStatus status = ippStsNoErr;
    Ipp8u *pBuffer = NULL;                         // Pointer to the work buffer
    int bufferSize;
    int srcStep;                                   // Steps, in bytes, through the
    int dstStep;
    IppiSize roiSize = { img_width, img_height };  // Size of source and
                                                   // destination ROI in pixels
    IppDataType ippSrcDataType;
    IppDataType ippDstDataType;
    int sizeof_src;
    status = sizeof_ipp_dtype_by_index(&sizeof_src, input_index);
    check_sts(status);

    int sizeof_dst;
    status = sizeof_ipp_dtype_by_index(&sizeof_dst, output_index);
    check_sts(status);

    srcStep = numChannels * img_width * sizeof_src;
    dstStep = numChannels * img_width * sizeof_dst;

    status = ipp_type_index_as_IppDataType(&ippSrcDataType, input_index);
    check_sts(status);

    status = ipp_type_index_as_IppDataType(&ippDstDataType, output_index);
    check_sts(status);
    
    status = ippiFilterPrewittVertBorderGetBufferSize(roiSize, mask, ippSrcDataType, ippDstDataType,
        numChannels, &bufferSize);
    check_sts(status);
    pBuffer = ippsMalloc_8u(bufferSize);
    if (NULL == pBuffer)
    {
        check_sts(status = ippStsMemAllocErr);
    };
    if (numChannels == 1)
    {
        if (input_index == ipp8u_index && output_index == ipp16s_index) {
            Ipp8u ippbordervalue = (Ipp8u)ippBorderValue;
            status = ippiFilterPrewittVertBorder_8u16s_C1R(pInput, srcStep, pOutput, dstStep, roiSize,
                                                           mask, ippBorderType, ippbordervalue, pBuffer);
        }
        else if (input_index == ipp16s_index && output_index == ipp16s_index) {
            // 16s_C1R
            Ipp16s ippbordervalue = (Ipp16s)ippBorderValue;
            status = ippiFilterPrewittVertBorder_16s_C1R(pInput, srcStep, pOutput, dstStep, roiSize,
                                                         mask, ippBorderType, ippbordervalue, pBuffer);
        }
        else if (input_index == ipp32f_index && output_index == ipp32f_index) {
            // 32f_C1R
            Ipp32f ippbordervalue = (Ipp32f)ippBorderValue;
            status = ippiFilterPrewittVertBorder_32f_C1R(pInput, srcStep, pOutput, dstStep, roiSize,
                                                         mask, ippBorderType, ippbordervalue, pBuffer);
        }
        else
        {
            status = ippStsErr;
        }
    }
    else
    {
        status = ippStsErr;
    }
    check_sts(status);
EXIT_FUNC
    ippsFree(pBuffer);
    return (int)status;
}

int
FilterPrewitt(
    IppDataTypeIndex input_index,
    IppDataTypeIndex output_index,
    void * pInput,
    void * pOutput,
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
    if (!(input_index == ipp32f_index &&
          output_index == ipp32f_index &&
          numChannels == 1))
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

    status = FilterPrewittHoriz(input_index, output_index, pInput, pAsrcDst, img_width, img_height, numChannels);
    check_sts(status);

    pBsrcDst = (void *)ippsMalloc_8u((img_width * sizeofIppDataType * numChannels) * img_height);
    if (pBsrcDst == NULL)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    };

    status = FilterPrewittVert(input_index, output_index, pInput, pBsrcDst, img_width, img_height, numChannels);
    check_sts(status);

    int srcDstStep = sizeofIppDataType * img_width * numChannels;

    status = ippiSqr_32f_C1IR(pAsrcDst, srcDstStep, roiSize);
    check_sts(status);

    status = ippiSqr_32f_C1IR(pBsrcDst, srcDstStep, roiSize);
    check_sts(status);

    status = ippiAdd_32f_C1R(pAsrcDst, srcDstStep, pBsrcDst, srcDstStep, pOutput, srcDstStep, roiSize);
    check_sts(status);

    status = ippiSqrt_32f_C1IR(pOutput, srcDstStep, roiSize);
    check_sts(status);

    Ipp32f sqrt2 = (Ipp32f)IPP_SQRT2;

    status = ippiDivC_32f_C1IR(sqrt2, pOutput, srcDstStep, roiSize);
    check_sts(status);

EXIT_FUNC
if (pAsrcDst !=  NULL)
    ippsFree(pAsrcDst);
if (pBsrcDst != NULL)
    ippsFree(pBsrcDst);
    return (int)status;
}

int
FilterPrewittHoriz(
    IppDataTypeIndex input_index,
    IppDataTypeIndex output_index,
    void * pInput,
    void * pOutput,
    int img_width,
    int img_height,
    int numChannels)
{
    IppStatus status = ippStsNoErr;
    IppiMaskSize mask = ippMskSize3x3;
    // TODO
    // scikit-image's prewitt filter is a wrapper on scipy.ndimage's
    // `convolve` func. `convolve` uses `reflect` border mode.
    // In `reflect` border mode is equalen of IPP's ippBorderMirrorR border type  
    // ippiFilterPrewittHorizBorder_<mode> doesn't supports this border type
    IppiBorderType ippBorderType = ippBorderRepl;
    float ippBorderValue = 0.0;
    IppiSize roiSize = { img_width, img_height };  // Size of source and
                                                   // destination ROI in pixels

    status = ippiFilterPrewittHorizBorder(input_index, output_index, pInput, pOutput,
        img_width, img_height, numChannels, mask, ippBorderType, ippBorderValue);
    if (output_index == ipp32f_index)
    {
        Ipp32f value = (Ipp32f)3.0;
        int srcStep = numChannels * img_width * sizeof(Ipp32f);
        status = ippiDivC_32f_C1IR(value, pOutput, srcStep, roiSize);
    }
    else
    {
        status = ippStsErr;
    }
    check_sts(status);
EXIT_FUNC
    return (int)status;
}

int
FilterPrewittVert(
    IppDataTypeIndex input_index,
    IppDataTypeIndex output_index,
    void * pInput,
    void * pOutput,
    int img_width,
    int img_height,
    int numChannels)
{
    IppStatus status = ippStsNoErr;
    IppiMaskSize mask = ippMskSize3x3;

    // TODO
    // scikit-image's prewitt filter is a wrapper on scipy.ndimage's
    // `convolve` func. `convolve` uses `reflect` border mode.
    // In `reflect` border mode is equalen of IPP's ippBorderMirrorR border type  
    // ippiFilterPrewittVertBorder_<mode> doesn't supports this border type
    IppiBorderType ippBorderType = ippBorderRepl; 
    float ippBorderValue = 0.0;

    IppiSize roiSize = { img_width, img_height };  // Size of source and
                                                   // destination ROI in pixels

    status = ippiFilterPrewittVertBorder(input_index, output_index, pInput, pOutput,
        img_width, img_height, numChannels, mask, ippBorderType, ippBorderValue);
    if (output_index == ipp32f_index)
    {
        Ipp32f value = (Ipp32f)(-3.0);
        int srcStep = numChannels * img_width * sizeof(Ipp32f);
        status = ippiDivC_32f_C1IR(value, pOutput, srcStep, roiSize);
    }
    else
    {
        status = ippStsErr;
    }
    check_sts(status);
EXIT_FUNC
    return (int)status;
}
