#include "laplace.h"

#define EXIT_LINE exitLine:                // Label for Exit
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine;


//
//                                0 -1  0
//              Laplace (3x3)    -1  4 -1
//                                0 -1  0

const 
Ipp32f kernelLaplace[3 * 3] = { 0, -1, 0,
                               -1, 4, -1,
                                0, -1, 0 };


//
//                               -1 -1 -1
//              Laplace (3x3)    -1  8 -1
//                               -1 -1 -1

int
ippiFilterLaplaceBorder(
    IppDataTypeIndex ipp_src_dst_index,
    void * pSrc,
    void * pDst,
    int img_width,
    int img_height,
    int numChannels,
    IppiMaskSize mask,
    IppiBorderType ippBorderType,
    float ippBorderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp8u *pBuffer = NULL;                  // Pointer to the work buffer
    int bufferSize;
    int srcStep;                            // Steps, in bytes, through the
    int dstStep;
    IppDataType ippDataType;

    IppiSize dstRoiSize = { img_width, img_height };  // Size of source and
                                                      // destination ROI in pixels
    int sizeof_src;
    status = sizeof_ipp_dtype_by_index(&sizeof_src, ipp_src_dst_index);
    check_sts(status);

    srcStep = numChannels * img_width * sizeof_src;
    dstStep = srcStep;

    status = ipp_type_index_as_IppDataType(&ippDataType, ipp_src_dst_index);
    check_sts(status);

    status = ippiFilterLaplaceBorderGetBufferSize(dstRoiSize, mask, ippDataType, ippDataType,
                                                  numChannels, &bufferSize);
    check_sts(status);
    pBuffer = ippsMalloc_8u(bufferSize);
    if (NULL == pBuffer)
    {
        check_sts(status = ippStsMemAllocErr);
    };
    if (numChannels == 1) 
    {
        switch (ipp_src_dst_index)
        {
        case ipp8u_index:
        {
            Ipp8u ippbordervalue = (Ipp8u)ippBorderValue;
            status = ippiFilterLaplaceBorder_8u_C1R(pSrc, srcStep, pDst, dstStep, dstRoiSize,
                                                     mask, ippBorderType, ippbordervalue, pBuffer);
            break;
        }
        case ipp16s_index:
        {
            Ipp16s ippbordervalue = (Ipp16s)ippBorderValue;
            status = ippiFilterLaplaceBorder_16s_C1R(pSrc, srcStep, pDst, dstStep, dstRoiSize,
                                                     mask, ippBorderType, ippbordervalue, pBuffer);
            break;
        }
        case ipp32f_index:
        {
            Ipp32f ippbordervalue = (Ipp32f)ippBorderValue;
            status = ippiFilterLaplaceBorder_32f_C1R(pSrc, srcStep, pDst, dstStep, dstRoiSize,
                                                     mask, ippBorderType, ippbordervalue, pBuffer);
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
            status = ippiFilterLaplaceBorder_8u_C3R(pSrc, srcStep, pDst, dstStep, dstRoiSize,
                                                    mask, ippBorderType, ippbordervalue, pBuffer);
            break;
        }
        case ipp16s_index:
        {
            Ipp16s ippbordervalue[3] = { (Ipp16s)ippBorderValue, (Ipp16s)ippBorderValue , (Ipp16s)ippBorderValue };
            status = ippiFilterLaplaceBorder_16s_C3R(pSrc, srcStep, pDst, dstStep, dstRoiSize,
                                                     mask, ippBorderType, ippbordervalue, pBuffer);
            break;
        }
        case ipp32f_index:
        {
            Ipp32f ippbordervalue[3] = { (Ipp32f)ippBorderValue, (Ipp32f)ippBorderValue , (Ipp32f)ippBorderValue };
            status = ippiFilterLaplaceBorder_32f_C3R(pSrc, srcStep, pDst, dstStep, dstRoiSize,
                                                     mask, ippBorderType, ippbordervalue, pBuffer);
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
EXIT_LINE
    ippsFree(pBuffer);
    return (int)status;
}

int
LaplaceFilter(
    IppDataTypeIndex input_index,
    IppDataTypeIndex output_index,
    void * pInput,
    void * pOutput,
    int img_width,
    int img_height,
    int numChannels,
    IppiBorderType ippBorderType,
    float ippBorderValue)
{
    // Kernel Values
    //
    //                                0 -1  0
    //              Laplace (3x3)    -1  4 -1
    //                                0 -1  0

    IppStatus status = ippStsNoErr;
    void * ipp_src = NULL;              // pointer to src array that is passes to ipp func
    void * ipp_dst = NULL;              // pointers to dst array that is passes to ipp func
    IppDataTypeIndex ipp_src_dst_index;
    IppDataTypeIndex border_dtype_index;

    preserve_range_flag preserve_range = preserve_range_false;

    int kernel_width = 3;
    int kernel_height = 3;
    IppRoundMode roundMode = ippRndNear;

    if (output_index = ipp32f_index)
    {
        ipp_src_dst_index = ipp32f_index;
        border_dtype_index = ipp32f_index;
        ipp_dst = pOutput;
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
        Ipp32f ippbordervalue = (Ipp32f)ippBorderValue;
    }
    else if (output_index == ipp64f_index)
    {
        // TODO
        // currently not supported
        status = ippStsErr;
    }
    else
    {
        status = ippStsErr;
    }
    check_sts(status);
    status = ippiFilterBorder(ipp_src_dst_index, border_dtype_index, ipp_src, ipp_dst, kernelLaplace, img_width, img_height, kernel_width, kernel_height,
        numChannels, roundMode, ippBorderType, ippBorderValue);
    check_sts(status);

EXIT_LINE
if (ipp_src != pInput && ipp_src != NULL)
ippsFree(ipp_src);
if (ipp_dst != pOutput && ipp_dst != NULL)
ippsFree(ipp_dst);
    return (int)status;
}
