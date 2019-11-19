#include "laplace.h"

#define EXIT_LINE exitLine:                // Label for Exit
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine;


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
