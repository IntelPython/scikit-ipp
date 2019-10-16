// #include "numpy/npy_common.h" /* npy_intp */
#include "dtypes.h"

#define EXIT_FUNC exitLine:             /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine;

static
int IppDataTypeMaskArray[IPP_TYPES_NUMBER] = {ipp8u_c,
                                              ipp8s_c,
                                              ipp16u_c,
                                              ipp16s_c,
                                              ipp32u_c,
                                              ipp32s_c,
                                              ipp64u_c,
                                              ipp64s_c,
                                              ipp32f_c,
                                              ipp64f_c
};

static
int IppDataTypeConversionRecomendationMaskArray[IPP_TYPES_NUMBER] = {ipp8u_r,
                                                                     ipp8s_r,
                                                                     ipp16u_r,
                                                                     ipp16s_r,
                                                                     ipp32u_r,
                                                                     ipp32s_r,
                                                                     ipp64u_r,
                                                                     ipp64s_r,
                                                                     ipp32f_r,
                                                                     ipp64f_r
};

int
get_ipp_src_dst_index(int output_index, int ipp_func_support_dtypes) {
    if (output_index > 9 || output_index < 0)
        return -1;

    IppDataTypeMask output_mask = IppDataTypeMaskArray[output_index];

    int result = output_mask & ipp_func_support_dtypes; // if result is not 0 then ipp func supports output dtype

    if (result == 0) // if result is 0 then ipp func doesn't support output dtype
    {
        IppDataTypeConversionRecomendationMask output_conv_recom_dtypes
            = IppDataTypeConversionRecomendationMaskArray[output_index];
        int output_conv_recom_dtypes_for_ipp_support = ipp_func_support_dtypes & output_conv_recom_dtypes;
        if ((output_conv_recom_dtypes_for_ipp_support > 0)
            && (output_conv_recom_dtypes_for_ipp_support < 0x400))  // 0x400 --> 10000000000
        { // case when converting into recomended dtypes, that ipp func supports
            int mask_checker = 0x200;  // 1000000000
            while (mask_checker > 0) {
                result = mask_checker & output_conv_recom_dtypes_for_ipp_support;
                if (result == 0)
                    mask_checker >>= 1;
                else
                    return result;
            }
        }
        else if (output_conv_recom_dtypes_for_ipp_support == 0)
        {  // case when converting into only ipp func supported dtypes
            int mask_checker = 0x1;   // 0000000001
            while (mask_checker < 513) {
                result = mask_checker & ipp_func_support_dtypes;
                if (result == 0)
                    mask_checker <<= 1;
                else
                    return result;
            }
        }
        else // some error
            return -1;
    }
    return result;
};

int
image_no_convert(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsErr;
    // currently not supported
    return (int)status;
};

int
image_8u_as_8s_XorC(void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp8u * pSRC = NULL;     // Pointers to source and
    Ipp8s * pDST = NULL;     // destination images

    if (numChannels == 3) {
        if (img_width < MAX_C3_IMG_WIDTH_BY_INT32_ROI_DTYPE) {
            img_width = img_height * 3;
        }
        else
        {
            status = ippStsSizeErr;
            check_sts(status)
        }
    }

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp8u *)pSrc;
    pDST = (Ipp8s *)pDst;

    status = ippiXorC_8u_C1R(pSRC, sizeof(Ipp8u) * img_width, 0x80,
        (Ipp8u *)pDST, sizeof(Ipp8s) * img_width, roiSize);
    check_sts(status)
EXIT_FUNC
    return (int)status;
}

int
image_8u_as_8s_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp8u * pSRC = NULL;     // Pointers to source and
    Ipp8s * pDST = NULL;     // destination images

    if (numChannels == 3) {
        if (img_width < MAX_C3_IMG_WIDTH_BY_INT32_ROI_DTYPE) {
            img_width = img_height * 3;
        }
        else
        {
            status = ippStsSizeErr;
            check_sts(status)
        }
    }

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp8u *)pSrc;
    pDST = (Ipp8s *)pDst;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_8U);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_8U);
    Ipp64f minDst = (Ipp64f)(IPP_MIN_8S);
    Ipp64f maxDst = (Ipp64f)(IPP_MAX_8S);

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    status = ippiScaleC_8u8s_C1R(pSRC, sizeof(Ipp8u) * img_width, mVal,aVal, pDST, sizeof(Ipp8s) * img_width, roiSize, ippAlgHintAccurate);
    check_sts(status);

EXIT_FUNC
    return (int)status;
}

int
image_8u_as_16u_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp8u * pSRC = NULL;     // Pointers to source and
    Ipp16u * pDST = NULL;    // destination images

    if (numChannels == 3) {
        if (img_width < MAX_C3_IMG_WIDTH_BY_INT32_ROI_DTYPE) {
            img_width = img_height * 3;
        }
        else
        {
            status = ippStsSizeErr;
            check_sts(status)
        }
    }

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp8u *)pSrc;
    pDST = (Ipp16u *)pDst;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_8U);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_8U);
    Ipp64f minDst = (Ipp64f)(IPP_MIN_16U);
    Ipp64f maxDst = (Ipp64f)(IPP_MAX_16U);

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    status = ippiScaleC_8u16u_C1R(pSRC, sizeof(Ipp8u) * img_width, mVal, aVal, pDST, sizeof(Ipp16u) * img_width, roiSize, ippAlgHintAccurate);
    check_sts(status);

EXIT_FUNC
    return (int)status;
}

int
image_8u_as_16s_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp8u * pSRC = NULL;     // Pointers to source and
    Ipp16s * pDST = NULL;    // destination images

    if (numChannels == 3) {
        if (img_width < MAX_C3_IMG_WIDTH_BY_INT32_ROI_DTYPE) {
            img_width = img_height * 3;
        }
        else
        {
            status = ippStsSizeErr;
            check_sts(status)
        }
    }

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels

    pSRC = (Ipp8u *)pSrc;
    pDST = (Ipp16s *)pDst;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_8U);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_8U);
    Ipp64f minDst = (Ipp64f)(IPP_MIN_16S);
    Ipp64f maxDst = (Ipp64f)(IPP_MAX_16S);

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    status = ippiScaleC_8u16s_C1R(pSRC, sizeof(Ipp8u) * img_width, mVal, aVal, pDST, sizeof(Ipp16s) * img_width, roiSize, ippAlgHintAccurate);
    check_sts(status);

EXIT_FUNC
    return (int)status;
}

int
image_8u_as_32u_Convert(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    // ** Doesn't take the range of values **
    IppStatus status = ippStsNoErr;
    status = image_8u_as_32s_Convert(pSrc, pDst, numChannels, img_width, img_height);
    return (int)status;
}

int
image_8u_as_32s_Convert(void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    // ** Doesn't take the range of values **
    IppStatus status = ippStsNoErr;
    Ipp8u * pSRC = NULL;     // Pointers to source and
    Ipp32s * pDST = NULL;    // destination images

    if (numChannels == 3) {
        if (img_width < MAX_C3_IMG_WIDTH_BY_INT32_ROI_DTYPE) {
            img_width = img_height * 3;
        }
        else
        {
            status = ippStsSizeErr;
            check_sts(status)
        }
    }

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp8u *)pSrc;
    pDST = (Ipp32s *)pDst;

    status = ippiConvert_8u32s_C1R(pSRC, img_width * sizeof(Ipp8u),
        pDST, img_width * sizeof(Ipp32s), roiSize);

    check_sts(status)

EXIT_FUNC
    return (int)status;
}

int
image_8u_as_32s_Scale(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    // ** Takes into account the range of values **
    IppStatus status = ippStsNoErr;
    Ipp8u * pSRC = NULL;     // Pointers to source and
    Ipp32s * pDST = NULL;    // destination images

    if (numChannels == 3) {
        if (img_width < MAX_C3_IMG_WIDTH_BY_INT32_ROI_DTYPE) {
            img_width = img_height * 3;
        }
        else
        {
            status = ippStsSizeErr;
            check_sts(status)
        }
    }

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp8u *)pSrc;
    pDST = (Ipp32s *)pDst;

    status = ippiScale_8u32s_C1R(pSRC, img_width * sizeof(Ipp8u),
        pDST, img_width * sizeof(Ipp32s), roiSize);
    check_sts(status);

EXIT_FUNC
    return (int)status;
}

int
image_8u_as_32s_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    // ** Takes into account the range of values **
    IppStatus status = ippStsNoErr;
    Ipp8u * pSRC = NULL;     // Pointers to source and
    Ipp32s * pDST = NULL;    // destination images

    if (numChannels == 3) {
        if (img_width < MAX_C3_IMG_WIDTH_BY_INT32_ROI_DTYPE) {
            img_width = img_height * 3;
        }
        else
        {
            status = ippStsSizeErr;
            check_sts(status)
        }
    }

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp8u *)pSrc;
    pDST = (Ipp32s *)pDst;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_8U);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_8U);
    Ipp64f minDst = (Ipp64f)(IPP_MIN_32S);
    Ipp64f maxDst = (Ipp64f)(IPP_MAX_32S);

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    status = ippiScaleC_8u32s_C1R(pSRC, img_width * sizeof(Ipp8u), mVal, aVal, pDST,
        img_width * sizeof(Ipp32s), roiSize, ippAlgHintAccurate);
    check_sts(status);

EXIT_FUNC
    return (int)status;
}

int
image_8s_as_8u_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp8s * pSRC = NULL;     // Pointers to source and
    Ipp8u * pDST = NULL;     // destination images

    if (numChannels == 3) {
        if (img_width < MAX_C3_IMG_WIDTH_BY_INT32_ROI_DTYPE) {
            img_width = img_height * 3;
        }
        else
        {
            status = ippStsSizeErr;
            check_sts(status)
        }
    }

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp8s *)pSrc;
    pDST = (Ipp8u *)pDst;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_8S);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_8S);
    Ipp64f minDst = (Ipp64f)(IPP_MIN_8U);
    Ipp64f maxDst = (Ipp64f)(IPP_MAX_8U);

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    status = ippiScaleC_8s8u_C1R(pSRC, sizeof(Ipp8s) * img_width, mVal, aVal, pDST, sizeof(Ipp8u) * img_width, roiSize, ippAlgHintAccurate);
    check_sts(status);

EXIT_FUNC
    return (int)status;
}

int
image_8s_as_8u_XorC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp8s * pSRC = NULL;     // Pointers to source and
    Ipp8u * pDST = NULL;     // destination images

    if (numChannels == 3) {
        if (img_width < MAX_C3_IMG_WIDTH_BY_INT32_ROI_DTYPE) {
            img_width = img_height * 3;
        }
        else
        {
            status = ippStsSizeErr;
            check_sts(status)
        }
    }

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp8s *)pSrc;
    pDST = (Ipp8u *)pDst;

    status = ippiXorC_8u_C1R((Ipp8u *)pSRC, sizeof(Ipp8s) * img_width,
        0x80, pDST, sizeof(Ipp8u) * img_width, roiSize);
    check_sts(status)
EXIT_FUNC
    return (int)status;
}

int
image_16u_as_16s_XorC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp16u * pSRC = NULL;     // Pointers to source and
    Ipp16s * pDST = NULL;     // destination images

    if (numChannels == 3) {
        if (img_width < MAX_C3_IMG_WIDTH_BY_INT32_ROI_DTYPE) {
            img_width = img_height * 3;
        }
        else
        {
            status = ippStsSizeErr;
            check_sts(status)
        }
    }

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp16u *)pSrc;
    pDST = (Ipp16s *)pDst;

    status = ippiXorC_16u_C1R(pSRC, sizeof(Ipp16u) * img_width, 0x8000,
        (Ipp16u *)pDST, sizeof(Ipp16s) * img_width, roiSize);
    check_sts(status)
EXIT_FUNC
    return (int)status;
}

int
image_16s_as_16u_XorC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp16s * pSRC = NULL;     // Pointers to source and
    Ipp16u * pDST = NULL;     // destination images

    if (numChannels == 3) {
        if (img_width < MAX_C3_IMG_WIDTH_BY_INT32_ROI_DTYPE) {
            img_width = img_height * 3;
        }
        else
        {
            status = ippStsSizeErr;
            check_sts(status)
        }
    }

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp16s *)pSrc;
    pDST = (Ipp16u *)pDst;

    status = ippiXorC_16u_C1R((Ipp16u *)pSRC, sizeof(Ipp16s) * img_width,
        0x8000, pDST, sizeof(Ipp16u) * img_width, roiSize);
    check_sts(status)
EXIT_FUNC
    return (int)status;
}

int
image_32u_as_32s_XorC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp32u * pSRC = NULL;     // Pointers to source and
    Ipp32s * pDST = NULL;     // destination images

    if (numChannels == 3) {
        if (img_width < MAX_C3_IMG_WIDTH_BY_INT32_ROI_DTYPE) {
            img_width = img_height * 3;
        }
        else
        {
            status = ippStsSizeErr;
            check_sts(status)
        }
    }

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp32u *)pSrc;
    pDST = (Ipp32s *)pDst;

    status = ippiXorC_32s_C1R((Ipp32s *)pSRC, sizeof(Ipp32u) * img_width,
        0x80000000, pDST, sizeof(Ipp32s) * img_width, roiSize);
    check_sts(status)

EXIT_FUNC
    return (int)status;
}

int
image_32s_as_32u_XorC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp32s * pSRC = NULL;     // Pointers to source and
    Ipp32u * pDST = NULL;     // destination images

    if (numChannels == 3) {
        if (img_width < MAX_C3_IMG_WIDTH_BY_INT32_ROI_DTYPE) {
            img_width = img_height * 3;
        }
        else
        {
            status = ippStsSizeErr;
            check_sts(status)
        }
    }

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp32s *)pSrc;
    pDST = (Ipp32u *)pDst;

    status = ippiXorC_32s_C1R(pSRC, sizeof(Ipp32s) * img_width, 0x80000000,
        (Ipp32s *)pDST, sizeof(Ipp32s) * img_width, roiSize);
    check_sts(status)

EXIT_FUNC
    return (int)status;
}

// image_as_float32
int
image_UINT8_as_float32(
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp8u * pSRC = NULL;     // Pointers to source and
    Ipp32f * pDST = NULL;    // destination images

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp8u *)pSrc;
    pDST = (Ipp32f *)pDst;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_8U);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_8U);
    Ipp64f minDst = 0;
    Ipp64f maxDst = 1;

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    check_sts(ippiScaleC_8u32f_C1R(pSRC,
        srcStep,
        mVal,
        aVal,
        pDST,
        dstStep,
        roiSize,
        ippAlgHintAccurate));

EXIT_FUNC
    return (int)status;
};

int
image_INT8_as_float32(
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp8s * pSRC = NULL;     // Pointers to source and
    Ipp32f * pDST = NULL;    // destination images

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp8s *)pSrc;
    pDST = (Ipp32f *)pDst;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_8S);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_8S);
    Ipp64f minDst = 0;
    Ipp64f maxDst = 1;

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    check_sts(ippiScaleC_8s32f_C1R(pSRC,
        srcStep,
        mVal,
        aVal,
        pDST,
        dstStep,
        roiSize,
        ippAlgHintAccurate));

EXIT_FUNC
    return (int)status;
}

int
image_UINT16_as_float32(
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp16u * pSRC = NULL;     // Pointers to source and
    Ipp32f * pDST = NULL;    // destination images

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp16u *)pSrc;
    pDST = (Ipp32f *)pDst;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_16U);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_16U);
    Ipp64f minDst = 0;
    Ipp64f maxDst = 1;

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    check_sts(ippiScaleC_16u32f_C1R(pSRC,
        srcStep,
        mVal,
        aVal,
        pDST,
        dstStep,
        roiSize,
        ippAlgHintAccurate));

EXIT_FUNC
    return (int)status;
};

int
image_INT16_as_float32(
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp16s * pSRC = NULL;     // Pointers to source and
    Ipp32f * pDST = NULL;    // destination images

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp16s *)pSrc;
    pDST = (Ipp32f *)pDst;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_16S);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_16S);
    Ipp64f minDst = 0;
    Ipp64f maxDst = 1;

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    check_sts(ippiScaleC_16s32f_C1R(pSRC,
        srcStep,
        mVal,
        aVal,
        pDST,
        dstStep,
        roiSize,
        ippAlgHintAccurate));

EXIT_FUNC
        return (int)status;
}

int
convert(int index1,
        int index2,
        void * pSrc,
        void * pDst,
        int numChannels,
        int img_width,
        int img_height)
{
    IppStatus status = ippStsNoErr;
    status = covertTable[index1][index2](pSrc, pDst, numChannels, img_width, img_height);
    return (int)status;
}

static covertHandler
covertTable[IPP_TYPES_NUMBER][IPP_TYPES_NUMBER] = { 
                        {image_no_convert,  // from Ipp8u
                         image_8u_as_8s_XorC,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert
                       },
                        {image_8s_as_8u_XorC,  // from Ipp8s
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert
                       },
                        {image_no_convert,   // from Ipp16u
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert
                       },
                        {image_no_convert,   // from Ipp16s
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert
                       },
                        {image_no_convert,   // from Ipp32u
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert
                       },
                        {image_no_convert,   // from Ipp32s
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert
                       },
                        {image_no_convert,   // from Ipp64u
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert
                       },
                        {image_no_convert,   // from Ipp64s
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert
                       },
                        {image_no_convert,   // from Ipp32f
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert
                       },
                        {image_no_convert,   // from Ipp64f
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert
                       }
};
