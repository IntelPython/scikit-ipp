///////////////////////////////////////////////////////////////////////////////////////////
//                                  
//    Data types conversion module.
//    Most of the Intel IPP function support only small scope of data types. This module
//    was created in order to support all the functionality of `scikit-image`, which
//    processes on all types of numpy array.          
//                                                                                       
//    Note: this modules was implemented for data types conversion. this implementation is
//          deprecated. Currently it is unused. It will be re-implemented. 
//                                                                                         
///////////////////////////////////////////////////////////////////////////////////////////

// #include "numpy/npy_common.h" /* npy_intp */
//#include "dtypes.h"

//#define EXIT_FUNC exitLine:             /* Label for Exit */
//#define check_sts(st) if((st) != ippStsNoErr) goto exitLine;


/*
static
int IppDataTypeMaskArray[IPP_TYPES_NUMBER] = { ipp8u_c,
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
int IppDataTypeConversionRecomendationMaskArray[IPP_TYPES_NUMBER] = { ipp8u_r,
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

// contains IppDataType enums that equalent of ipp<type>_index
// ex. IppDataTypeArray[ipp8u_index] is ipp8u
static
int IppDataTypeArray[IPP_TYPES_NUMBER] = {
    ipp8u,                  //  ipp8u_index
    ipp8s,                  //  ipp8s_index
    ipp16u,                 //  ipp16u_index
    ipp16s,                 //  ipp16s_index
    ipp32u,                 //  ipp32u_index
    ipp32s,                 //  ipp32s_index
    ipp64u,                 //  ipp64u_index
    ipp64s,                 //  ipp64s_index
    ipp32f,                 //  ipp32f_index
    ipp64f                  //  ipp64f_index
};

// returns IppDataType enum by ipp<type>_index
int
ipp_type_index_as_IppDataType(
    IppDataType * ippDataType,
    IppDataTypeIndex ipp_type_index
)
{
    IppStatus status = ippStsNoErr;
    if (ipp_type_index < ipp8u_index || ipp_type_index > ipp64f_index)
    {
        status = ippStsSizeErr;
        ippDataType = NULL;
        check_sts(status);
    }
    *ippDataType = IppDataTypeArray[ipp_type_index];
EXIT_FUNC
    return (int)status;
}

int
get_ipp_src_dst_index(int output_index, int ipp_func_support_dtypes) {
    if (output_index > ipp64f_index || output_index < ipp8u_index)
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

IppDataTypeIndex
ippDtypeMask_as_ippDtypeIndex(int ippDtypeMask)
{
    IppDataTypeIndex ippDtypeIndex;

    // ipp8u_index = 0
    // ipp8s_index = 1
    // ipp16u_index = 2
    // ipp16s_index = 3
    // ipp32u_index = 4
    // ipp32s_index = 5
    // ipp64u_index = 6
    // ipp64s_index = 7
    // ipp32f_index = 8
    // ipp64f_index = 9
    // ippUndef_index = -1

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

    // ipp8u_c = 512  ----> 1000000000
    // ipp8s_c = 256  ----> 0100000000
    // ipp16u_c = 128 ----> 0010000000
    // ipp16s_c = 64  ----> 0001000000
    // ipp32u_c = 32  ----> 0000100000
    // ipp32s_c = 16  ----> 0000010000
    // ipp64u_c = 8   ----> 0000001000
    // ipp64s_c = 4   ----> 0000000100
    // ipp32f_c = 2   ----> 0000000010
    // ipp64f_c = 1   ----> 0000000001
    else 
        ippDtypeIndex = ippUndef_index;

    return ippDtypeIndex;
}

int
sizeof_ipp_dtype[IPP_TYPES_NUMBER] =
{
    sizeof(Ipp8u),
    sizeof(Ipp8s),
    sizeof(Ipp16u),
    sizeof(Ipp16s),
    sizeof(Ipp32u),
    sizeof(Ipp32s),
    sizeof(Ipp64u),
    sizeof(Ipp64s),
    sizeof(Ipp32f),
    sizeof(Ipp64f)
};

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

    if (index >= ipp8u_index && index <= ipp64f_index)
    {
        sizeofIppDataType = sizeof_ipp_dtype[index];
        ipp_arr_p = (void *)ippsMalloc_8u((img_width * sizeofIppDataType * numChannels) * img_height);
    }
    return ipp_arr_p;
}

Ipp64f
ipp_type_max[IPP_TYPES_NUMBER] =
{
    (Ipp64f)IPP_MAX_8U,
    (Ipp64f)IPP_MAX_8S,
    (Ipp64f)IPP_MAX_16U,
    (Ipp64f)IPP_MAX_16S,
    (Ipp64f)IPP_MAX_32U,
    (Ipp64f)IPP_MAX_32S,
    (Ipp64f)IPP_MAX_64U,
    (Ipp64f)IPP_MAX_64S,
    (Ipp64f)IPP_MAXABS_32F,
    (Ipp64f)IPP_MAXABS_64F    //  TODO: problems, when IPP_MAXABS_64F
};

Ipp64f
ipp_type_min[IPP_TYPES_NUMBER] =
{
    (Ipp64f)IPP_MIN_8U,
    (Ipp64f)IPP_MIN_8S,
    (Ipp64f)IPP_MIN_16U,
    (Ipp64f)IPP_MIN_16S,
    (Ipp64f)IPP_MIN_32U,
    (Ipp64f)IPP_MIN_32S,
    (Ipp64f)IPP_MIN_64U,
    (Ipp64f)IPP_MIN_64S,
    (Ipp64f)IPP_MINABS_32F,
    (Ipp64f)IPP_MINABS_64F    //  TODO: problems, when IPP_MINABS_64F
};

int
sizeof_ipp_dtype_by_index(
    int * sizeof_type,
    int type_index
)
{
    IppStatus status = ippStsNoErr;
    if (type_index < ipp8u_index || type_index > ipp64f_index)
    {
        status = ippStsSizeErr;
        sizeof_type = NULL;
        check_sts(status);
    }
    *sizeof_type = sizeof_ipp_dtype[type_index];
EXIT_FUNC
    return (int)status;
}


typedef enum {
    ipp8u_ipp_ScaleC = 0,
    ipp8s_ipp_ScaleC = 1,
    ipp16u_ipp_ScaleC = 2,
    ipp16s_ipp_ScaleC = 3,
    ipp32s_ipp_ScaleC = 4,
    ipp32f_ipp_ScaleC = 5,
    ipp64f_ipp_ScaleC = 6,
    undef_ipp_ScaleC = -1
} ScaleC_C1R_dtype_index;

func_jumpt_table_index ScaleC_C1R_table_array[IPP_TYPES_NUMBER] = {
    ipp8u_ipp_ScaleC,
    ipp8s_ipp_ScaleC,
    ipp16u_ipp_ScaleC,
    ipp16s_ipp_ScaleC,
    undef_ipp_ScaleC,
    ipp32s_ipp_ScaleC,
    undef_ipp_ScaleC,
    undef_ipp_ScaleC,
    ipp32f_ipp_ScaleC,
    ipp64f_ipp_ScaleC,
};

int
dtype_index_for_ScaleC_C1R_table(func_jumpt_table_index * jumpt_table_index, IppDataTypeIndex type_index)
{
    IppStatus status = ippStsNoErr;

    if (type_index > ipp64f_index || type_index < ipp8u_index)
    {
        jumpt_table_index = NULL;
        status = ippStsErr;
        check_sts(status);
    }
    *jumpt_table_index = ScaleC_C1R_table_array[type_index];
    if (*jumpt_table_index == undef_ipp_ScaleC)
    {
        jumpt_table_index = NULL;
        status = ippStsErr;
        check_sts(status);
    }

EXIT_FUNC
    return (int)status;
}

static ScaleC_Handler
ipp_scaleC_table[IPPi_ScaleC_SUPPORTED_TYPES_NUMBER][IPPi_ScaleC_SUPPORTED_TYPES_NUMBER] = {
    {
        ippiScaleC_8u_C1R,
        ippiScaleC_8u8s_C1R,
        ippiScaleC_8u16u_C1R,
        ippiScaleC_8u16s_C1R,
        ippiScaleC_8u32s_C1R,
        ippiScaleC_8u32f_C1R,
        ippiScaleC_8u64f_C1R,
    },
    {
        ippiScaleC_8s8u_C1R,
        ippiScaleC_8s_C1R,
        ippiScaleC_8s16u_C1R,
        ippiScaleC_8s16s_C1R,
        ippiScaleC_8s32s_C1R,
        ippiScaleC_8s32f_C1R,
        ippiScaleC_8s64f_C1R,
    },
    {
        ippiScaleC_16u8u_C1R,
        ippiScaleC_16u8s_C1R,
        ippiScaleC_16u_C1R,
        ippiScaleC_16u16s_C1R,
        ippiScaleC_16u32s_C1R,
        ippiScaleC_16u32f_C1R,
        ippiScaleC_16u64f_C1R,
    },
    {
        ippiScaleC_16s8u_C1R,
        ippiScaleC_16s8s_C1R,
        ippiScaleC_16s16u_C1R,
        ippiScaleC_16s_C1R,
        ippiScaleC_16s32s_C1R,
        ippiScaleC_16s32f_C1R,
        ippiScaleC_16s64f_C1R,
    },
    {
        ippiScaleC_32s8u_C1R,
        ippiScaleC_32s8s_C1R,
        ippiScaleC_32s16u_C1R,
        ippiScaleC_32s16s_C1R,
        ippiScaleC_32s_C1R,
        ippiScaleC_32s32f_C1R,
        ippiScaleC_32s64f_C1R,
    },
    {
        ippiScaleC_32f8u_C1R,
        ippiScaleC_32f8s_C1R,
        ippiScaleC_32f16u_C1R,
        ippiScaleC_32f16s_C1R,
        ippiScaleC_32f32s_C1R,
        ippiScaleC_32f_C1R,
        ippiScaleC_32f64f_C1R,
    },
    {
        ippiScaleC_64f8u_C1R,
        ippiScaleC_64f8s_C1R,
        ippiScaleC_64f16u_C1R,
        ippiScaleC_64f16s_C1R,
        ippiScaleC_64f32s_C1R,
        ippiScaleC_64f32f_C1R,
        ippiScaleC_64f_C1R,
    }
};

int
image_ScaleC(
    IppDataTypeIndex src_index,
    IppDataTypeIndex dst_index,
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height,
    preserve_range_flag preserve_range)
{
    IppStatus status = ippStsNoErr;
    // if (src_index == dst_index)
    //    goto exitLine;

    void * intermediateSrc = NULL;
    void * intermediateDst = NULL;

    if (numChannels == 3) {
        if (img_width < MAX_C3_IMG_WIDTH_BY_INT32_ROI_DTYPE) {
            img_width = img_height * 3;
            numChannels = 1;
        }
        else
        {
            status = ippStsSizeErr;
            check_sts(status);
        }
    }
    if (src_index == ipp32u_index) 
    {
        if (dst_index == ipp32s_index) {
            status = image_32u_as_32s_Converting_XorC(pSrc, pDst, numChannels, img_width, img_height);
            check_sts(status)
        }
        else
        {
            intermediateSrc = malloc_by_dtype_index(ipp32s_index, numChannels, img_width, img_height);
            if (intermediateSrc == NULL) {
                // specify corret error status
                status = ippStsErr;
                check_sts(status);
            }
            status = image_32u_as_32s_Converting_XorC(pSrc, intermediateSrc, numChannels, img_width, img_height);
            check_sts(status);
            src_index = ipp32s_index;
            status = image_ScaleC(src_index, dst_index, intermediateSrc, pDst, numChannels, img_width, img_height, preserve_range);
        }
        goto exitLine;
    }
    if (dst_index == ipp32u_index)
    {
        if (src_index == ipp32s_index)
        {
            status = image_32s_as_32u_Converting_XorC(pSrc, pDst, numChannels, img_width, img_height);
            check_sts(status)
        }
        else
        {
            intermediateDst = malloc_by_dtype_index(ipp32s_index, numChannels, img_width, img_height);
            if (intermediateDst == NULL) {
                // specify corret error status
                status = ippStsErr;
                check_sts(status);
            }
            dst_index = ipp32s_index;
            status = image_ScaleC(src_index, dst_index, pSrc, intermediateDst, numChannels, img_width, img_height, preserve_range);
            check_sts(status);

            status = image_32u_as_32s_Converting_XorC(intermediateDst, pDst, numChannels, img_width, img_height);
        }
        goto exitLine;
    }

    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels

    Ipp64f minSrc;
    Ipp64f maxSrc;
    Ipp64f minDst;
    Ipp64f maxDst;

    Ipp64f mVal;
    Ipp64f aVal;

    int sizeof_src;
    status = sizeof_ipp_dtype_by_index(&sizeof_src, src_index);
    check_sts(status);

    int sizeof_dst;
    status = sizeof_ipp_dtype_by_index(&sizeof_dst, dst_index);
    check_sts(status);

    minSrc = ipp_type_min[src_index];
    maxSrc = ipp_type_max[src_index];

    if (preserve_range == preserve_range_true) {
        minDst = ipp_type_min[dst_index];
        maxDst = ipp_type_max[dst_index];
    }
    else if (preserve_range == preserve_range_true_for_small_bitsize_src) {
        if ((sizeof_src < sizeof_dst) && 
            (((src_index <= ipp64s_index) && (dst_index <= ipp64s_index) && ((src_index % 2) == (dst_index % 2))) ||
            (dst_index == ipp32f_index || dst_index == ipp64f_index))
            ) 
        {
            minDst = ipp_type_min[src_index];
            maxDst = ipp_type_max[src_index];
        }
        else
        {
            minDst = ipp_type_min[dst_index];
            maxDst = ipp_type_max[dst_index];
        }
    }
    else if ((preserve_range == preserve_range_false) && (dst_index == ipp32f_index || dst_index == ipp64f_index)) {
        if ((src_index <= ipp64s_index) && ((src_index % 2) == 1))
        {
            minDst = 0;
            maxDst = 1;
        }
        else
        {
            minDst = -1;
            maxDst = 1;
        }
    }
    else
    {
        status = ippStsSizeErr;    // preserve_range_false and dst_index is not floating
        check_sts(status);
    }

    mVal = (maxDst - minDst) / (maxSrc - minSrc);
    aVal = minDst - minSrc * mVal;

    func_jumpt_table_index src_jumpt_table_index;
    func_jumpt_table_index dst_jumpt_table_index;

    status = dtype_index_for_ScaleC_C1R_table(&src_jumpt_table_index, src_index);
    check_sts(status);
    status = dtype_index_for_ScaleC_C1R_table(&dst_jumpt_table_index, dst_index);
    check_sts(status);
    status = ipp_scaleC_table[src_jumpt_table_index][dst_jumpt_table_index](pSrc, sizeof_src * img_width, mVal, aVal,
        pDst, sizeof_dst * img_width, roiSize, ippAlgHintAccurate);

    check_sts(status);

EXIT_FUNC
if(intermediateSrc != NULL){
    ippsFree(intermediateSrc);
}
if(intermediateDst != NULL){
    ippsFree(intermediateDst);
}
    return (int)status;
};


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
*/