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
ippDtypeMask_as_ippDtypeIndex(int ippDtypeMask)
{
    int ippDtypeIndex = -1;
    /*
    ipp8u_index = 0,
    ipp8s_index = 1,
    ipp16u_index = 2,
    ipp16s_index = 3,
    ipp32u_index = 4,
    ipp32s_index = 5,
    ipp64u_index = 6,
    ipp64s_index = 7,
    ipp32f_index = 8,
    ipp64f_index = 9,
    */
    if (ippDtypeMask == ipp8u_c)
        ippDtypeIndex = ipp8u_index;

    else if (ippDtypeMask == ipp8s_c)
        ippDtypeIndex = ipp8s_index;

    else if (ippDtypeMask == ipp16u_c)
        ippDtypeIndex = ipp16u_index;

    else if (ippDtypeMask == ipp16s_c)
        ippDtypeIndex = ipp16s_index;

    else if (ippDtypeMask == ipp32u_c)
        ippDtypeIndex = ipp32u_index;

    else if (ippDtypeMask == ipp32s_c)
        ippDtypeIndex = ipp32s_index;

    else if (ippDtypeMask == ipp64u_c)
        ippDtypeIndex = ipp64u_index;

    else if (ippDtypeMask == ipp64s_c)
        ippDtypeIndex = ipp64s_index;

    else if (ippDtypeMask == ipp32f_c)
        ippDtypeIndex = ipp32f_index;

    else if (ippDtypeMask == ipp64f_c)
        ippDtypeIndex = ipp64f_index;

    /*
    ipp8u_c = 512,     // 1000000000
    ipp8s_c = 256,     // 0100000000
    ipp16u_c = 128,    // 0010000000
    ipp16s_c = 64,     // 0001000000
    ipp32u_c = 32,     // 0000100000
    ipp32s_c = 16,     // 0000010000
    ipp64u_c = 8,      // 0000001000
    ipp64s_c = 4,      // 0000000100
    ipp32f_c = 2,      // 0000000010
    ipp64f_c = 1,      // 0000000001
    */
    return ippDtypeIndex;
}

void *
malloc_by_dtype_index(
    int index,
    int numChannels,
    int img_width,
    int img_height
)
{
    void * ipp_arr_p = NULL;
    int sizeofIppDataType = 0;

    // better use array with sizeof(IppDtype)
    if (index == ipp8u_index)
    {
        sizeofIppDataType = sizeof(Ipp8u);
    }
    else if (index == ipp8s_index)
    {
        sizeofIppDataType = sizeof(Ipp8s);
    }
    else if (index == ipp16u_index)
    {
        sizeofIppDataType = sizeof(Ipp16u);
    }
    else if (index == ipp16s_index)
    {
        sizeofIppDataType = sizeof(Ipp16s);
    }
    else if (index == ipp32u_index)
    {
        sizeofIppDataType = sizeof(Ipp32u);
    }
    else if (index == ipp32s_index)
    {
        sizeofIppDataType = sizeof(Ipp32s);
    }
    else if (index == ipp64u_index)
    {
        sizeofIppDataType = sizeof(Ipp64u);
    }
    else if (index == ipp64s_index)
    {
        sizeofIppDataType = sizeof(Ipp64s);
    }
    else if (index == ipp32f_index)
    {
        sizeofIppDataType = sizeof(Ipp32f);
    }
    else if (index == ipp64f_index)
    {
        sizeofIppDataType = sizeof(Ipp64f);
    }

    // ~~~~ check mul
    // ~~~~ is it correct allocate by ippsMalloc_8u ?
    ipp_arr_p = (void *)ippsMalloc_8u((img_width * sizeofIppDataType * numChannels) * img_height);

    return ipp_arr_p;
}

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

// Image scaling and converting funcs
// functions naming rule
// image_<from dtype>_<to dtype>_<Functionality>_<backend function(s) from IPP>
//
// E.g. image_8u_as_8s_Converting_XorC: functions that does convertation
// from Ipp8u to Ipp8s by using IPP's XorC library func

int
image_8u_as_8s_Converting_XorC(
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

    status = ippiScaleC_8u8s_C1R(pSRC, sizeof(Ipp8u) * img_width, mVal, aVal,
                                 pDST, sizeof(Ipp8s) * img_width, roiSize, ippAlgHintAccurate);
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

    status = ippiScaleC_8u16u_C1R(pSRC, sizeof(Ipp8u) * img_width, mVal, aVal,
                                  pDST, sizeof(Ipp16u) * img_width, roiSize, ippAlgHintAccurate);
    check_sts(status);

EXIT_FUNC
    return (int)status;
}

int
image_8u_as_16u_Converting_ScaleC(
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
    Ipp64f minDst = (Ipp64f)(IPP_MIN_8U);

    Ipp64f mVal = 1;
    Ipp64f aVal = minDst - minSrc * mVal;

    status = ippiScaleC_8u16u_C1R(pSRC, sizeof(Ipp8u) * img_width, mVal, aVal,
                                  pDST, sizeof(Ipp16u) * img_width, roiSize, ippAlgHintAccurate);
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

    status = ippiScaleC_8u16s_C1R(pSRC, sizeof(Ipp8u) * img_width, mVal, aVal,
                                  pDST, sizeof(Ipp16s) * img_width, roiSize, ippAlgHintAccurate);
    check_sts(status);

EXIT_FUNC
    return (int)status;
}

int
image_8u_as_16s_Converting_ScaleC(
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
    Ipp64f minDst = (Ipp64f)(IPP_MIN_8U);


    Ipp64f mVal = 1;
    Ipp64f aVal = minDst - minSrc * mVal;

    status = ippiScaleC_8u16s_C1R(pSRC, sizeof(Ipp8u) * img_width, mVal, aVal,
                                  pDST, sizeof(Ipp16s) * img_width, roiSize, ippAlgHintAccurate);
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
image_8u_as_32f_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
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

    int srcStep = sizeof(Ipp8u) * img_width * numChannels;
    int dstStep = sizeof(Ipp32f) * img_width * numChannels;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_8U);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_8U);
    Ipp64f minDst = (Ipp64f)(IPP_MINABS_32F);  // IPP_MINABS_32F
    Ipp64f maxDst = (Ipp64f)(IPP_MAXABS_32F);

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
image_8u_as_32f_Converting_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
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

    int srcStep = sizeof(Ipp8u) * img_width * numChannels;
    int dstStep = sizeof(Ipp32f) * img_width * numChannels;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_8U);
    Ipp64f minDst = (Ipp64f)(IPP_MIN_8U); // IPP_MINABS_32F


    Ipp64f mVal = 1;
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
image_8u_as_64f_Converting_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp8u * pSRC = NULL;     // Pointers to source and
    Ipp64f * pDST = NULL;    // destination images

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp8u *)pSrc;
    pDST = (Ipp64f *)pDst;

    int srcStep = sizeof(Ipp8u) * img_width * numChannels;
    int dstStep = sizeof(Ipp64f) * img_width * numChannels;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_8U);
    Ipp64f minDst = (Ipp64f)(IPP_MIN_8U); // IPP_MINABS_32F


    Ipp64f mVal = 1;
    Ipp64f aVal = minDst - minSrc * mVal;

    check_sts(ippiScaleC_8u64f_C1R(pSRC,
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

    status = ippiScaleC_8s8u_C1R(pSRC, sizeof(Ipp8s) * img_width, mVal, aVal,
                                 pDST, sizeof(Ipp8u) * img_width, roiSize, ippAlgHintAccurate);
    check_sts(status);

EXIT_FUNC
    return (int)status;
}


int
image_8s_as_8u_Converting_XorC(
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
image_8s_as_16u_Converting_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp8s * pSRC = NULL;     // Pointers to source and
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
    pSRC = (Ipp8s *)pSrc;
    pDST = (Ipp16u *)pDst;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_8S);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_8S);
    Ipp64f minDst = (Ipp64f)(IPP_MIN_8U);
    Ipp64f maxDst = (Ipp64f)(IPP_MAX_8U);

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    status = ippiScaleC_8s16u_C1R(pSRC, sizeof(Ipp8s) * img_width, mVal, aVal, pDST,
                                  sizeof(Ipp16u) * img_width, roiSize, ippAlgHintAccurate);
    check_sts(status);

EXIT_FUNC
    return (int)status;
}

int
image_8s_as_16s_Converting_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp8s * pSRC = NULL;     // Pointers to source and
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
    pSRC = (Ipp8s *)pSrc;
    pDST = (Ipp16s *)pDst;

    int srcStep = sizeof(Ipp8s) * img_width * numChannels;
    int dstStep = sizeof(Ipp16s) * img_width * numChannels;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_8S);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_8S);
    Ipp64f minDst = (Ipp64f)(IPP_MIN_8S);
    Ipp64f maxDst = (Ipp64f)(IPP_MAX_8S);

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    status = ippiScaleC_8s16s_C1R(pSRC, srcStep, mVal, aVal, pDST, dstStep, roiSize, ippAlgHintAccurate);
    check_sts(status);

EXIT_FUNC
    return (int)status;
}

//~~~ doesn't work correct unsafe convert
int
image_8s_as_32u_Convert(   // 8s32u_C1Rs
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    // ** Doesn't take the range of values **
    IppStatus status = ippStsNoErr;
    Ipp8s * pSRC = NULL;     // Pointers to source and
    Ipp32u * pDST = NULL;    // destination images

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
    pDST = (Ipp32u *)pDst;

    int srcStep = sizeof(Ipp8s) * img_width * numChannels;
    int dstStep = sizeof(Ipp32u) * img_width * numChannels;

    status = ippiConvert_8s32u_C1Rs(pSRC, srcStep, pDST, dstStep, roiSize);

    check_sts(status)

EXIT_FUNC
    return (int)status;
}

int
image_8s_as_32s_Converting_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp8s * pSRC = NULL;     // Pointers to source and
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
    pSRC = (Ipp8s *)pSrc;
    pDST = (Ipp32s *)pDst;

    int srcStep = sizeof(Ipp8s) * img_width * numChannels;
    int dstStep = sizeof(Ipp32s) * img_width * numChannels;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_8S);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_8S);
    Ipp64f minDst = (Ipp64f)(IPP_MIN_8S);
    Ipp64f maxDst = (Ipp64f)(IPP_MAX_8S);

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    status = ippiScaleC_8s32s_C1R(pSRC, srcStep, mVal, aVal, pDST, dstStep, roiSize, ippAlgHintAccurate);
    check_sts(status);

EXIT_FUNC
    return (int)status;
}

int
image_16u_as_8s_Converting_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp16u * pSRC = NULL;     // Pointers to source and
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
    pSRC = (Ipp16u *)pSrc;
    pDST = (Ipp8s *)pDst;

    int srcStep = sizeof(Ipp16u) * img_width * numChannels;
    int dstStep = sizeof(Ipp8s) * img_width * numChannels;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_16U);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_16U);
    Ipp64f minDst = (Ipp64f)(IPP_MIN_8S);
    Ipp64f maxDst = (Ipp64f)(IPP_MAX_8S);

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    status = ippiScaleC_16u8s_C1R(pSRC, srcStep, mVal, aVal, pDST, dstStep, roiSize, ippAlgHintAccurate);
    check_sts(status);

EXIT_FUNC
    return (int)status;
}

int
image_16u_as_16s_Converting_XorC(
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
image_16s_as_8s_Converting_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp16s * pSRC = NULL;     // Pointers to source and
    Ipp8s * pDST = NULL;      // destination images

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
    pDST = (Ipp8s *)pDst;

    int srcStep = sizeof(Ipp16s) * img_width * numChannels;
    int dstStep = sizeof(Ipp8s) * img_width * numChannels;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_16S);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_16S);
    Ipp64f minDst = (Ipp64f)(IPP_MIN_8S);
    Ipp64f maxDst = (Ipp64f)(IPP_MAX_8S);

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    status = ippiScaleC_16s8s_C1R(pSRC, srcStep, mVal, aVal, pDST, dstStep, roiSize, ippAlgHintAccurate);
    check_sts(status);

EXIT_FUNC
    return (int)status;
}
int
image_16s_as_16u_Converting_XorC(
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
image_32u_as_32s_Converting_XorC(
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
image_32s_as_32u_Converting_XorC(
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

static covertHandler
covertTable[IPP_TYPES_NUMBER][IPP_TYPES_NUMBER] = {
                        {image_no_convert,                // from Ipp8u
                         image_8u_as_8s_Converting_XorC,
                         image_8u_as_16u_ScaleC,
                         image_8u_as_16s_Converting_ScaleC,
                         image_8u_as_32u_Convert,
                         image_8u_as_32s_Convert,
                         image_no_convert,
                         image_no_convert,
                         image_8u_as_32f_Converting_ScaleC,
                         image_8u_as_64f_Converting_ScaleC
                       },
                        {image_8s_as_8u_Converting_XorC,  // from Ipp8s
                         image_no_convert,
                         image_8s_as_16u_Converting_ScaleC,
                         image_8s_as_16s_Converting_ScaleC,
                         image_no_convert,
                         image_8s_as_32s_Converting_ScaleC,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert
                       },
                        {image_no_convert,               // from Ipp16u
                         image_16u_as_8s_Converting_ScaleC,
                         image_no_convert,
                         image_16u_as_16s_Converting_XorC,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert
                       },
                        {image_no_convert,               // from Ipp16s
                         image_16s_as_8s_Converting_ScaleC,
                         image_16s_as_16u_Converting_XorC,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert
                       },
                        {image_no_convert,               // from Ipp32u
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_32u_as_32s_Converting_XorC,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert
                       },
                        {image_no_convert,               // from Ipp32s
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_32s_as_32u_Converting_XorC,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert,
                         image_no_convert
                       },
                        {image_no_convert,               // from Ipp64u
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
                        {image_no_convert,               // from Ipp64s
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
                        {image_no_convert,              // from Ipp32f
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
                        {image_no_convert,              // from Ipp64f
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

int
convert(
    int index1,
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


// Image as float 
// The range of a floating point image is [0.0, 1.0] or [-1.0, 1.0] when
// converting from unsigned or signed datatypes, respectively.
//
// functions naming rule
// image_<from dtype>_<to dtype>_<Functionality>_<backend function(s) from IPP>
//
// E.g. image_8u_as_32f_Converting_range_01_ScaleC: function that does scaling 
// conv range [0.0, 1.0] from Ipp8u to Ipp32f by using IPP's ScaleC library func

int
image_8u_as_32f_Converting_range_01_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
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

    int srcStep = sizeof(Ipp8u) * img_width * numChannels;
    int dstStep = sizeof(Ipp32f) * img_width * numChannels;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_8U);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_8U);
    Ipp64f minDst = 0;
    Ipp64f maxDst = 1;

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    check_sts(ippiScaleC_8u32f_C1R(pSRC, srcStep, mVal, aVal, pDST,
                                   dstStep, roiSize, ippAlgHintAccurate));

EXIT_FUNC
    return (int)status;
}

int
image_8u_as_64f_Converting_range_01_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp8u * pSRC = NULL;     // Pointers to source and
    Ipp64f * pDST = NULL;    // destination images

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp8u *)pSrc;
    pDST = (Ipp64f *)pDst;

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

    int srcStep = sizeof(Ipp8u) * img_width * numChannels;
    int dstStep = sizeof(Ipp64f) * img_width * numChannels;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_8U);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_8U);
    Ipp64f minDst = 0;
    Ipp64f maxDst = 1;

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    check_sts(ippiScaleC_8u64f_C1R(pSRC, srcStep, mVal, aVal, pDST,
                                   dstStep, roiSize, ippAlgHintAccurate));

EXIT_FUNC
    return (int)status;
}

int
image_8s_as_32f_Converting_range_11_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
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

    int srcStep = sizeof(Ipp8s) * img_width * numChannels;
    int dstStep = sizeof(Ipp32f) * img_width * numChannels;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_8S);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_8S);
    Ipp64f minDst = -1;
    Ipp64f maxDst = 1;

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    check_sts(ippiScaleC_8s32f_C1R(pSRC, srcStep, mVal, aVal,
                                   pDST, dstStep, roiSize, ippAlgHintAccurate));

EXIT_FUNC
    return (int)status;
}

int
image_8s_as_64f_Converting_range_11_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp8s * pSRC = NULL;     // Pointers to source and
    Ipp64f * pDST = NULL;    // destination images

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp8s *)pSrc;
    pDST = (Ipp64f *)pDst;

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

    int srcStep = sizeof(Ipp8s) * img_width * numChannels;
    int dstStep = sizeof(Ipp64f) * img_width * numChannels;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_8S);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_8S);
    Ipp64f minDst = -1;
    Ipp64f maxDst = 1;

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    check_sts(ippiScaleC_8s64f_C1R(pSRC, srcStep, mVal, aVal, pDST,
                                   dstStep, roiSize, ippAlgHintAccurate));

EXIT_FUNC
    return (int)status;
}

int
image_16u_as_32f_Converting_range_01_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp16u * pSRC = NULL;     // Pointers to source and
    Ipp32f * pDST = NULL;     // destination images

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp16u *)pSrc;
    pDST = (Ipp32f *)pDst;

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

    int srcStep = sizeof(Ipp16u) * img_width * numChannels;
    int dstStep = sizeof(Ipp32f) * img_width * numChannels;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_16U);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_16U);
    Ipp64f minDst = 0;
    Ipp64f maxDst = 1;

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    check_sts(ippiScaleC_16u32f_C1R(pSRC, srcStep, mVal, aVal,
                                    pDST, dstStep, roiSize, ippAlgHintAccurate));

EXIT_FUNC
    return (int)status;
}

int
image_16u_as_64f_Converting_range_01_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp16u * pSRC = NULL;     // Pointers to source and
    Ipp64f * pDST = NULL;     // destination images

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp16u *)pSrc;
    pDST = (Ipp64f *)pDst;

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

    int srcStep = sizeof(Ipp16u) * img_width * numChannels;
    int dstStep = sizeof(Ipp64f) * img_width * numChannels;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_16U);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_16U);
    Ipp64f minDst = 0;
    Ipp64f maxDst = 1;

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    check_sts(ippiScaleC_16u64f_C1R(pSRC, srcStep, mVal, aVal,
                                    pDST, dstStep, roiSize, ippAlgHintAccurate));
EXIT_FUNC
    return (int)status;
}

int
image_16s_as_32f_Converting_range_11_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp16s * pSRC = NULL;     // Pointers to source and
    Ipp32f * pDST = NULL;     // destination images

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp16s *)pSrc;
    pDST = (Ipp32f *)pDst;

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

    int srcStep = sizeof(Ipp16s) * img_width * numChannels;
    int dstStep = sizeof(Ipp32f) * img_width * numChannels;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_16S);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_16S);
    Ipp64f minDst = -1;
    Ipp64f maxDst = 1;

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    check_sts(ippiScaleC_16s32f_C1R(pSRC, srcStep, mVal, aVal, pDST,
                                    dstStep, roiSize, ippAlgHintAccurate));

EXIT_FUNC
    return (int)status;
}

int
image_16s_as_64f_Converting_range_11_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp16s * pSRC = NULL;     // Pointers to source and
    Ipp64f * pDST = NULL;     // destination images

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp16s *)pSrc;
    pDST = (Ipp64f *)pDst;

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

    int srcStep = sizeof(Ipp16s) * img_width * numChannels;
    int dstStep = sizeof(Ipp64f) * img_width * numChannels;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_16S);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_16S);
    Ipp64f minDst = -1;
    Ipp64f maxDst = 1;

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    check_sts(ippiScaleC_16s64f_C1R(pSRC, srcStep, mVal, aVal, pDST,
                                    dstStep, roiSize, ippAlgHintAccurate));

EXIT_FUNC
    return (int)status;
}

int
image_32s_as_32f_Converting_range_11_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp32s * pSRC = NULL;     // Pointers to source and
    Ipp32f * pDST = NULL;     // destination images

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp32s *)pSrc;
    pDST = (Ipp32f *)pDst;

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

    int srcStep = sizeof(Ipp32s) * img_width * numChannels;
    int dstStep = sizeof(Ipp32f) * img_width * numChannels;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_32S);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_32S);
    Ipp64f minDst = -1;
    Ipp64f maxDst = 1;

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;

    check_sts(ippiScaleC_32s32f_C1R(pSRC, srcStep, mVal, aVal, pDST,
        dstStep, roiSize, ippAlgHintAccurate));

EXIT_FUNC
    return (int)status;
}
