#include "gaussian.h"

#define EXIT_FUNC exitLine:                                  /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine 


static ippiFilterGaussianBorder_C1R
ippiFilterGaussianBorder_C1R_arr[IPP_GAUSSIAN_SUPPORTED_TYPES_NUMBER] = {
    ippiFilterGaussianBorder_8u_C1R,
    ippiFilterGaussianBorder_16u_C1R,
    ippiFilterGaussianBorder_16s_C1R,
    ippiFilterGaussianBorder_32f_C1R
};

static ippiFilterGaussianBorder_C3R
ippiFilterGaussianBorder_C3R_arr[IPP_GAUSSIAN_SUPPORTED_TYPES_NUMBER] = {
    ippiFilterGaussianBorder_8u_C3R,
    ippiFilterGaussianBorder_16u_C3R,
    ippiFilterGaussianBorder_16s_C3R,
    ippiFilterGaussianBorder_32f_C3R
};

func_jumpt_table_index 
ippiFilterGaussianBorder_table_array[IPP_TYPES_NUMBER] = {
    ippiFilterGaussianBorder_8u,
    ippiFilterGaussianBorder_16u,
    ippiFilterGaussianBorder_16s,
    undef,
    undef,
    undef,
    undef,
    undef,
    ippiFilterGaussianBorder_32f,
    undef,
};

int
dtype_index_for_ippiFilterGaussianBorder_table(func_jumpt_table_index * jumpt_table_index, IppDataTypeIndex type_index)
{
    IppStatus status = ippStsNoErr;

    if (type_index > ipp64f_index || type_index < ipp8u_index)
    {
        jumpt_table_index = NULL;
        status = ippStsErr;
        check_sts(status);
    }
    *jumpt_table_index = ippiFilterGaussianBorder_table_array[type_index];
    if (*jumpt_table_index == undef)
    {
        jumpt_table_index = NULL;
        status = ippStsErr;
        check_sts(status);
    }

EXIT_FUNC
    return (int)status;
}

IppStatus
get_borderValue_C3(
    IppDataTypeIndex ipp_src_dst_index,
    void * borderValue_C3,
    float ippBorderValue)
{
    IppStatus status = ippStsNoErr;
    // borderValue_C3 = NULL;
    if (ipp_src_dst_index == ipp8u_index) {
        Ipp8u * borderValue_C3_ipp8u = (Ipp8u*)borderValue_C3;
        borderValue_C3_ipp8u[0] = (Ipp8u)(ippBorderValue);
        borderValue_C3_ipp8u[1] = (Ipp8u)(ippBorderValue);
        borderValue_C3_ipp8u[2] = (Ipp8u)(ippBorderValue);
    }
    else if (ipp_src_dst_index == ipp16u_index)
    {
        Ipp16u * borderValue_C3_ipp16u = (Ipp16u*)borderValue_C3;
        borderValue_C3_ipp16u[0] = (Ipp16u)(ippBorderValue);
        borderValue_C3_ipp16u[1] = (Ipp16u)(ippBorderValue);
        borderValue_C3_ipp16u[2] = (Ipp16u)(ippBorderValue);
    }
    else if (ipp_src_dst_index == ipp16s_index)
    {
        Ipp16s * borderValue_C3_ipp16s = (Ipp16s*)borderValue_C3;
        borderValue_C3_ipp16s[0] = (Ipp16s)(ippBorderValue);
        borderValue_C3_ipp16s[1] = (Ipp16s)(ippBorderValue);
        borderValue_C3_ipp16s[2] = (Ipp16s)(ippBorderValue);
    }

    else if (ipp_src_dst_index = ipp32f_index)
    {
        Ipp32f * borderValue_C3_ipp32f = (Ipp32f*)borderValue_C3;
        borderValue_C3_ipp32f[0] = (Ipp32f)(ippBorderValue);
        borderValue_C3_ipp32f[1] = (Ipp32f)(ippBorderValue);
        borderValue_C3_ipp32f[2] = (Ipp32f)(ippBorderValue);
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
    void * borderValue_C3 = NULL;
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

    func_jumpt_table_index ippiFilterGaussianBorder_table;

    status = dtype_index_for_ippiFilterGaussianBorder_table(&ippiFilterGaussianBorder_table, ipp_src_dst_index);
    check_sts(status);

    if (numChannels == 1)
    {
        status = ippiFilterGaussianBorder_C1R_arr[ippiFilterGaussianBorder_table](pSrc, srcStep, pDst, dstStep,
            roiSize, ippBorderValue, pSpec, pBuffer);
    }
    else if (numChannels == 3)
    {
        borderValue_C3 = malloc_by_dtype_index(ipp_src_dst_index, numChannels, 1, 1);
        if (borderValue_C3 == NULL)
        {
            status = ippStsMemAllocErr;
            check_sts(status);
        }
        status = get_borderValue_C3(ipp_src_dst_index, borderValue_C3, ippBorderValue);
        check_sts(status);

        status = ippiFilterGaussianBorder_C3R_arr[ippiFilterGaussianBorder_table](pSrc, srcStep, pDst, dstStep,
            roiSize, borderValue_C3, pSpec, pBuffer);
    }
    else
    {
        status = ippStsErr;
    }
    check_sts(status);
EXIT_FUNC
    ippsFree(pBuffer);
    ippsFree(pSpec);
if(borderValue_C3 != NULL)
{
    ippsFree(borderValue_C3);
}
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

    if ((output_index == ipp32f_index) && (preserve_range == preserve_range_false))
    {
        ipp_src_dst_index = output_index;  //  ipp32f_index
        ipp_dst = pOutput;
        if (input_index == ipp_src_dst_index)
        {
            ipp_src = pInput;
        }
        else
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
        
        // pass to ipp func
        status = ippiFilterGaussianBorder(ipp_src_dst_index, ipp_src, ipp_dst, img_width, img_height, numChannels, 
                    sigma, kernelSize, ippBorderType, ippBorderValue);

        check_sts(status);
    }

    else if ((output_index == ipp64f_index) && (preserve_range == preserve_range_false))
    {
        ipp_src_dst_index = ipp32f_index;

        // malloc ipp_dst
        ipp_dst = malloc_by_dtype_index(ipp_src_dst_index, numChannels, img_width, img_height);
        if (ipp_dst == NULL) {
            // specify corret error status
            status = ippStsErr;
            check_sts(status);
        }

        if (input_index == ipp_src_dst_index)
        {
            ipp_src = pInput;
        }
        // convert to float32 input image
        // if input_index is not ipp32f_index, then malloc and convert
        else
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

        // pass to ipp func
        status = ippiFilterGaussianBorder(ipp_src_dst_index, ipp_src, ipp_dst, img_width, img_height, numChannels,
            sigma, kernelSize, ippBorderType, ippBorderValue);
        check_sts(status);

        // convert_from_32f_to_64f output image
        status = image_ScaleC(ipp_src_dst_index, output_index, ipp_dst, pOutput, numChannels, img_width, img_height, preserve_range);
        check_sts(status);
    }
    else
    {
        if (preserve_range == preserve_range_false) {
            preserve_range = preserve_range_true;
        }

        // IPP_GAUSSIAN_SUPPORTED_DTYPES
        int ipp_src_dst_mask_for_gaussian_filter = get_ipp_src_dst_index(output_index, IPP_GAUSSIAN_SUPPORTED_DTYPES);
        if (ipp_src_dst_mask_for_gaussian_filter == -1)
        {
            status = ippStsErr;
            check_sts(status);
        }
        int ipp_src_dst_index = ippDtypeMask_as_ippDtypeIndex(ipp_src_dst_mask_for_gaussian_filter);
        if (ipp_src_dst_index == -1)
        {
            status = ippStsErr;
            check_sts(status);
        }

        if (ipp_src_dst_index == output_index) {
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
    }
EXIT_FUNC
    if (ipp_src != pInput && ipp_src != NULL)
        ippsFree(ipp_src);
    if (ipp_dst != pOutput && ipp_dst != NULL)
        ippsFree(ipp_dst);
    return (int)status;
};
