#include "gaussian.h"

#define EXIT_FUNC exitLine:                                  /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine 

int
ippiFilterGaussianBorder(
    IppDataTypeIndex ipp_src_dst_index,
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
    int sizeof_src;
    status = sizeof_ipp_dtype_by_index(&sizeof_src, ipp_src_dst_index);

    int srcStep = numChannels * img_width * sizeof_src;      // Steps, in bytes, through the
    int dstStep = srcStep;                                   //source/destination images

    IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels
    Ipp32f sigma = (Ipp32f)sigma_;
    Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer 
    IppFilterGaussianSpec* pSpec = NULL;   // context structure 
    int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size

    IppDataType ippDataType;
    status = ipp_type_index_as_IppDataType(&ippDataType, ipp_src_dst_index);
    check_sts(status);

    status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ippDataType,
        numChannels, &iSpecSize, &iTmpBufSize);
    check_sts(status);
    pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
    if (NULL == pSpec)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    }

    pBuffer = ippsMalloc_8u(iTmpBufSize);
    if (NULL == pBuffer)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    }

    status = ippiFilterGaussianInit(roiSize, kernelSize, sigma,
        ippBorderType, ippDataType, numChannels, pSpec, pBuffer);
    
    check_sts(status);

    if (numChannels == 1)
    {
        switch (ipp_src_dst_index)
        {
        case ipp8u_index:
        {
            Ipp8u ippbordervalue = (Ipp8u)ippBorderValue;
            status = ippiFilterGaussianBorder_8u_C1R(pSrc, srcStep, pDst, dstStep, roiSize, ippbordervalue, pSpec, pBuffer);
            break;
        }
        case ipp16u_index:
        {
            Ipp16u ippbordervalue = (Ipp16u)ippBorderValue;
            status = ippiFilterGaussianBorder_16u_C1R(pSrc, srcStep, pDst, dstStep, roiSize, ippbordervalue, pSpec, pBuffer);
            break;
        }
        case ipp16s_index:
        {
            Ipp16s ippbordervalue = (Ipp16s)ippBorderValue;
            status = ippiFilterGaussianBorder_16s_C1R(pSrc, srcStep, pDst, dstStep, roiSize, ippbordervalue, pSpec, pBuffer);
            break;
        }
        case ipp32f_index:
        {
            Ipp32f ippbordervalue = (Ipp32f)ippBorderValue;
            status = ippiFilterGaussianBorder_32f_C1R(pSrc, srcStep, pDst, dstStep, roiSize, ippbordervalue, pSpec, pBuffer);
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
        switch (ipp_src_dst_index)
        {
        case ipp8u_index:
        {
            Ipp8u ippbordervalue[3] = { (Ipp8u)ippBorderValue, (Ipp8u)ippBorderValue , (Ipp8u)ippBorderValue };
            status = ippiFilterGaussianBorder_8u_C3R(pSrc, srcStep, pDst, dstStep, roiSize, ippbordervalue, pSpec, pBuffer);
            break;
        }
        case ipp16u_index:
        {
            Ipp16u ippbordervalue[3] = { (Ipp16u)ippBorderValue, (Ipp16u)ippBorderValue , (Ipp16u)ippBorderValue };
            status = ippiFilterGaussianBorder_16u_C3R(pSrc, srcStep, pDst, dstStep, roiSize, ippbordervalue, pSpec, pBuffer);
            break;
        }
        case ipp16s_index:
        {
            Ipp16s ippbordervalue[3] = { (Ipp16s)ippBorderValue, (Ipp16s)ippBorderValue , (Ipp16s)ippBorderValue };
            status = ippiFilterGaussianBorder_16s_C3R(pSrc, srcStep, pDst, dstStep, roiSize, ippbordervalue, pSpec, pBuffer);
            break;
        }
        case ipp32f_index:
        {
            Ipp32f ippbordervalue[3] = { (Ipp32f)ippBorderValue, (Ipp32f)ippBorderValue , (Ipp32f)ippBorderValue };
            status = ippiFilterGaussianBorder_32f_C3R(pSrc, srcStep, pDst, dstStep, roiSize, ippbordervalue, pSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsErr;
        }
        }
    }
    else
    {
        status = ippStsErr;
    }
    check_sts(status);
EXIT_FUNC
    ippsFree(pBuffer);
    ippsFree(pSpec);
return (int)status;
}

int
GaussianFilter(
    IppDataTypeIndex input_index,
    IppDataTypeIndex output_index,
    void * pInput,
    void * pOutput,
    int img_width,
    int img_height,
    int numChannels,
    float sigma,
    int kernelSize,
    IppiBorderType ippBorderType,
    float ippBorderValue,
    preserve_range_flag preserve_range)
{
    IppStatus status = ippStsNoErr;

    void * ipp_src = NULL;  // pointer to src array that is passes to ipp func
    void * ipp_dst = NULL;  // pointers to dst array that is passes to ipp func
    IppDataTypeIndex ipp_src_dst_index;

    if (!(preserve_range == preserve_range_false && ((output_index == ipp32f_index) || (output_index == ipp64f_index)))) {
        preserve_range = preserve_range_true;
    }
    // IPP_GAUSSIAN_SUPPORTED_DTYPES
    // int -> ippDtypeMask and int -> ippDtypeIndex
    int ipp_src_dst_mask_for_gaussian_filter = get_ipp_src_dst_index(output_index, IPP_GAUSSIAN_SUPPORTED_DTYPES);
    if (ipp_src_dst_mask_for_gaussian_filter == -1)
    {
        status = ippStsErr;
        check_sts(status);
    }
    ipp_src_dst_index = ippDtypeMask_as_ippDtypeIndex(ipp_src_dst_mask_for_gaussian_filter);
    if (ipp_src_dst_index == ippUndef_index)
    {
        status = ippStsErr;
        check_sts(status);
    }
    // if there is no need in convertation ipp_src --> pOutput 
    if ((ipp_src_dst_index == output_index) && (preserve_range != preserve_range_false)) {
        preserve_range = preserve_range_true_for_small_bitsize_src;
    }
    //  input  --> ipp_src
    //  output --> ipp_dst
    if (input_index != ipp_src_dst_index)
    {
        ipp_src = malloc_by_dtype_index(ipp_src_dst_index, numChannels, img_width, img_height);
        if (ipp_src == NULL) {
            // specify corret error status
            status = ippStsErr;
            check_sts(status);
        }
        status = image_ScaleC(input_index, ipp_src_dst_index, pInput, ipp_src,
            numChannels, img_width, img_height, preserve_range);

        check_sts(status);
    }
    else
    {
        ipp_src = pInput;
    }

    if (output_index != ipp_src_dst_index)
    {
        ipp_dst = malloc_by_dtype_index(ipp_src_dst_index, numChannels, img_width, img_height);
        if (ipp_dst == NULL) {
            // specify correct error status
            status = ippStsErr;
            check_sts(status);
        }
    }
    else
    {
        ipp_dst = pOutput;
    }
    // pass to ipp func
    status = ippiFilterGaussianBorder(ipp_src_dst_index, ipp_src, ipp_dst, img_width, img_height, numChannels,
        sigma, kernelSize, ippBorderType, ippBorderValue);
    check_sts(status);
    // convert ipp_dst into output array dtype, if they are different
    if (output_index != ipp_src_dst_index)
    {
        status = image_ScaleC(ipp_src_dst_index, output_index, ipp_dst, pOutput, numChannels, img_width, img_height, preserve_range);
        check_sts(status);
    }
EXIT_FUNC
    if (ipp_src != pInput && ipp_src != NULL)
        ippsFree(ipp_src);
    if (ipp_dst != pOutput && ipp_dst != NULL)
        ippsFree(ipp_dst);
    return (int)status;
};
