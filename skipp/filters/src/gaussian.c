#include "gaussian.h"

#define EXIT_FUNC exitLine:                                  /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine 

int
GaussianFilter_Ipp8u(
    void * pSRC,
    void * pDST,
    int img_width,
    int img_height,
    int numChannels,
    float sigma_,
    int kernelSize,
    int ippBorderType,
    float ippBorderValue)
{

    IppStatus status = ippStsNoErr;
    Ipp8u * pSrc = NULL;     // Pointers to source
    Ipp8u * pDst = NULL;     // and destination images       
    int srcStep = numChannels * img_width * sizeof(Ipp8u);   // Steps, in bytes, through the
    int dstStep = srcStep;                                   //source/destination images
    
    IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels
    Ipp32f sigma = (Ipp32f)sigma_;
    Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer 
    IppFilterGaussianSpec* pSpec = NULL;   // context structure 
    int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size
    IppiBorderType borderType = ippBorderType;
    Ipp8u borderValue_C1;
    Ipp8u borderValue_C3[3];
    if (numChannels == 1) {
        borderValue_C1 = (Ipp8u)ippBorderValue;
    }
    else if (numChannels == 3)
    {
        borderValue_C3[0] = (Ipp8u)ippBorderValue;
        borderValue_C3[1] = (Ipp8u)ippBorderValue;
        borderValue_C3[2] = (Ipp8u)ippBorderValue;
    }
    else
    {
        status = ippStsErr;
        check_sts(status);
    }
    
    pSrc = (Ipp8u *)pSRC;
    pDst = (Ipp8u *)pDST;

    check_sts(status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp8u,
                                               numChannels, &iSpecSize, &iTmpBufSize));
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
                           borderType, ipp8u, numChannels, pSpec, pBuffer);
    check_sts(status);
    if (numChannels == 1)
    {
        status = ippiFilterGaussianBorder_8u_C1R(pSrc, srcStep, pDst, dstStep,
                                                 roiSize, borderValue_C1, pSpec, pBuffer);
    }
    else
    {
        status = ippiFilterGaussianBorder_8u_C3R(pSrc, srcStep, pDst, dstStep,
            roiSize, borderValue_C3, pSpec, pBuffer);
    }
    check_sts(status);
EXIT_FUNC
    ippsFree(pBuffer);
    ippsFree(pSpec);
    return (int)status;
}

int
GaussianFilter_Ipp16u(
    void * pSRC,
    void * pDST,
    int img_width,
    int img_height,
    int numChannels,
    float sigma_,
    int kernelSize,
    int ippBorderType,
    float ippBorderValue)
{

    IppStatus status = ippStsNoErr;
    Ipp16u * pSrc = NULL;     // Pointers to source
    Ipp16u * pDst = NULL;     // and destination images       
    int srcStep = numChannels * img_width * sizeof(Ipp8u);   // Steps, in bytes, through the
    int dstStep = srcStep;                                   //source/destination images

    IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels
    Ipp32f sigma = (Ipp32f)sigma_;
    Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer 
    IppFilterGaussianSpec* pSpec = NULL;   // context structure 
    int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size
    IppiBorderType borderType = ippBorderType;
    Ipp16u borderValue_C1;
    Ipp16u borderValue_C3[3];
    if (numChannels == 1) {
        borderValue_C1 = (Ipp16u)ippBorderValue;
    }
    else if (numChannels == 3)
    {
        borderValue_C3[0] = (Ipp16u)ippBorderValue;
        borderValue_C3[1] = (Ipp16u)ippBorderValue;
        borderValue_C3[2] = (Ipp16u)ippBorderValue;
    }
    else
    {
        status = ippStsErr;
        check_sts(status);
    }

    pSrc = (Ipp16u *)pSRC;
    pDst = (Ipp16u *)pDST;

    check_sts(status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp16u,
        numChannels, &iSpecSize, &iTmpBufSize));
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
        borderType, ipp16u, numChannels, pSpec, pBuffer);
    check_sts(status);
    if (numChannels == 1)
    {
        status = ippiFilterGaussianBorder_16u_C1R(pSrc, srcStep, pDst, dstStep,
            roiSize, borderValue_C1, pSpec, pBuffer);
    }
    else
    {
        status = ippiFilterGaussianBorder_16u_C3R(pSrc, srcStep, pDst, dstStep,
            roiSize, borderValue_C3, pSpec, pBuffer);
    }
    check_sts(status);
EXIT_FUNC
    ippsFree(pBuffer);
    ippsFree(pSpec);
    return (int)status;
}

int
GaussianFilter_Ipp16s(
    void * pSRC,
    void * pDST,
    int img_width,
    int img_height,
    int numChannels,
    float sigma_,
    int kernelSize,
    int ippBorderType,
    float ippBorderValue)
{

    IppStatus status = ippStsNoErr;
    Ipp16s * pSrc = NULL;     // Pointers to source
    Ipp16s * pDst = NULL;     // and destination images       
    int srcStep = numChannels * img_width * sizeof(Ipp8u);   // Steps, in bytes, through the
    int dstStep = srcStep;                                   //source/destination images

    IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels
    Ipp32f sigma = (Ipp32f)sigma_;
    Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer 
    IppFilterGaussianSpec* pSpec = NULL;   // context structure 
    int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size
    IppiBorderType borderType = ippBorderType;
    Ipp16s borderValue_C1;
    Ipp16s borderValue_C3[3];
    if (numChannels == 1) {
        borderValue_C1 = (Ipp16s)ippBorderValue;
    }
    else if (numChannels == 3)
    {
        borderValue_C3[0] = (Ipp16s)ippBorderValue;
        borderValue_C3[1] = (Ipp16s)ippBorderValue;
        borderValue_C3[2] = (Ipp16s)ippBorderValue;
    }
    else
    {
        status = ippStsErr;
        check_sts(status);
    }

    pSrc = (Ipp16s *)pSRC;
    pDst = (Ipp16s *)pDST;

    check_sts(status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp16s,
        numChannels, &iSpecSize, &iTmpBufSize));
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
        borderType, ipp16s, numChannels, pSpec, pBuffer);
    check_sts(status);
    if (numChannels == 1)
    {
        status = ippiFilterGaussianBorder_16s_C1R(pSrc, srcStep, pDst, dstStep,
            roiSize, borderValue_C1, pSpec, pBuffer);
    }
    else
    {
        status = ippiFilterGaussianBorder_16s_C3R(pSrc, srcStep, pDst, dstStep,
            roiSize, borderValue_C3, pSpec, pBuffer);
    }
    check_sts(status);
EXIT_FUNC
    ippsFree(pBuffer);
    ippsFree(pSpec);
    return (int)status;
}

int
GaussianFilter_Ipp32f(
    void * pSRC,
    void * pDST,
    int img_width,
    int img_height,
    int numChannels,
    float sigma_,
    int kernelSize,
    int ippBorderType,
    float ippBorderValue)
{

    IppStatus status = ippStsNoErr;
    Ipp32f * pSrc = NULL;     // Pointers to source
    Ipp32f * pDst = NULL;     // and destination images       
    int srcStep = numChannels * img_width * sizeof(Ipp8u);   // Steps, in bytes, through the
    int dstStep = srcStep;                                   //source/destination images

    IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels
    Ipp32f sigma = (Ipp32f)sigma_;
    Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer 
    IppFilterGaussianSpec* pSpec = NULL;   // context structure 
    int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size
    IppiBorderType borderType = ippBorderType;
    Ipp32f borderValue_C1;
    Ipp32f borderValue_C3[3];
    if (numChannels == 1) {
        borderValue_C1 = (Ipp32f)ippBorderValue;
    }
    else if (numChannels == 3)
    {
        borderValue_C3[0] = (Ipp32f)ippBorderValue;
        borderValue_C3[1] = (Ipp32f)ippBorderValue;
        borderValue_C3[2] = (Ipp32f)ippBorderValue;
    }
    else
    {
        status = ippStsErr;
        check_sts(status);
    }

    pSrc = (Ipp32f *)pSRC;
    pDst = (Ipp32f *)pDST;

    check_sts(status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp32f,
        numChannels, &iSpecSize, &iTmpBufSize));
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
        borderType, ipp32f, numChannels, pSpec, pBuffer);
    check_sts(status);
    if (numChannels == 1)
    {
        status = ippiFilterGaussianBorder_32f_C1R(pSrc, srcStep, pDst, dstStep,
            roiSize, borderValue_C1, pSpec, pBuffer);
    }
    else
    {
        status = ippiFilterGaussianBorder_32f_C3R(pSrc, srcStep, pDst, dstStep,
            roiSize, borderValue_C3, pSpec, pBuffer);
    }
    check_sts(status);
EXIT_FUNC
    ippsFree(pBuffer);
    ippsFree(pSpec);
    return (int)status;
}

int
get_IppGaussianFilterIndex(IppDataTypeIndex ippDataTypeIndex)
{
    IppDataTypeIndex index;
    if (ippDataTypeIndex == ipp8u_index)
        index = gaussianFilter_Ipp8u;
    else if (ippDataTypeIndex == ipp16u_index)
        index = gaussianFilter_Ipp16u;
    else if (ippDataTypeIndex == ipp16s_index)
        index = gaussianFilter_Ipp16s;
    else if (ippDataTypeIndex == ipp32f_index)
        index = gaussianFilter_Ipp32f;
    else
        index = -1;
    return index;
}

static gaussianFuncHandler
gaussianFilter[] = {
    GaussianFilter_Ipp8u,
    GaussianFilter_Ipp16u,
    GaussianFilter_Ipp16s,
    GaussianFilter_Ipp32f
};

int
GaussianFilter(
    int input_index,
    int output_index,
    void * pInput,
    void * pOutput,
    int img_width,
    int img_height,
    int numChannels,
    float sigma_,
    int kernelSize,
    int ippBorderType,
    float ippBorderValue)
{
    IppStatus status = ippStsNoErr;

    void * ipp_src = NULL;  // pointer to src array that is passes to ipp func
    void * ipp_dst = NULL;  // pointers to dst array that is passes to ipp func

    // IPP_GAUSSIAN_SUPPORTED_DTYPES
    // int -> ippDtypeMask and int -> ippDtypeIndex
    int ipp_src_dst_mask_for_gaussian_filter = get_ipp_src_dst_index(output_index, IPP_GAUSSIAN_SUPPORTED_DTYPES);
    if (ipp_src_dst_mask_for_gaussian_filter == -1)
    {
        status = ippStsErr;
        check_sts(status);
    }

    int ipp_src_dst_index_for_gaussian_filter = ippDtypeMask_as_ippDtypeIndex(ipp_src_dst_mask_for_gaussian_filter);
    if (ipp_src_dst_index_for_gaussian_filter == -1)
    {
        status = ippStsErr;
        check_sts(status);
    }

    //  input  --> ipp_src
    //  output --> ipp_dst
    if (input_index != ipp_src_dst_index_for_gaussian_filter)
    {
        ipp_src = malloc_by_dtype_index(ipp_src_dst_index_for_gaussian_filter, numChannels, img_width, img_height);
        if (ipp_src == NULL) {
            // specify corret error status
            status = ippStsErr;
            check_sts(status);
        }
        status = convert(input_index, ipp_src_dst_index_for_gaussian_filter,
            pInput, ipp_src, numChannels, img_width, img_height);

        check_sts(status);
    }
    else
    {
        ipp_src = pInput;
    }

    if (output_index != ipp_src_dst_index_for_gaussian_filter)
    {

        ipp_dst = malloc_by_dtype_index(ipp_src_dst_index_for_gaussian_filter, numChannels, img_width, img_height);
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

    // get ipp_gaussian filter index
    int ippGaussianFilter_index = get_IppGaussianFilterIndex(ipp_src_dst_index_for_gaussian_filter);

    // pass to ipp func
    status = gaussianFilter[ippGaussianFilter_index](ipp_src, ipp_dst, img_width, img_height, numChannels, sigma_,
        kernelSize, ippBorderType, ippBorderValue);
    check_sts(status);

    // convert ipp_dst into output array dtype, if they are different
    if (output_index != ipp_src_dst_index_for_gaussian_filter)
    {
        status = convert(ipp_src_dst_index_for_gaussian_filter, output_index,
            ipp_dst, pOutput, numChannels, img_width, img_height);
        check_sts(status);
    }
EXIT_FUNC
    if (ipp_src != pInput && ipp_src != NULL)
        ippsFree(ipp_src);
    if (ipp_dst != pOutput && ipp_dst != NULL)
        ippsFree(ipp_dst);
    return (int)status;
}
