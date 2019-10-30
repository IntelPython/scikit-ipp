#include "ipp.h"
// #include "ippcore_tl.h"
#ifndef DTYPES_H
#define DTYPES_H

#define MAX_C3_IMG_WIDTH_BY_INT32_ROI_DTYPE 0x2AAAAAAA

#define IPP_TYPES_NUMBER 10

typedef enum {
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
} IppDataTypeIndex;

typedef enum {
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
} IppDataTypeMask;

typedef enum {
    ipp8u_r = 0x1FF, // 0111111111
    ipp8s_r = 0x2FF, // 1011111111
    ipp16u_r = 0x7F, // 0001111111
    ipp16s_r = 0xBF, // 0010111111
    ipp32u_r = 0x1F, // 0000011111
    ipp32s_r = 0x2F, // 0000101111
    ipp64u_r = 0x7,  // 0000000111
    ipp64s_r = 0xB,  // 0000001011
    ipp32f_r = 0x1,  // 0000000001
    ipp64f_r = 0x2,  // 0000000010
} IppDataTypeConversionRecomendationMask;

int
get_ipp_src_dst_index(int output_index, int ipp_func_support_dtypes);

int
ippDtypeMask_as_ippDtypeIndex(int ippDtypeMask);

void *
malloc_by_dtype_index(
    int index,
    int numChannels,
    int img_width,
    int img_height
);

int
image_no_convert(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

// image scaling and converting funcs
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
    int img_height);

int
image_8u_as_8s_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_8u_as_16u_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_8u_as_16u_Converting_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_8u_as_16s_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_8u_as_16s_Converting_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_8u_as_32u_Convert(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_8u_as_32s_Convert(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_8u_as_32s_Scale(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_8u_as_32s_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_8u_as_32f_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_8u_as_32f_Converting_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_8u_as_64f_Converting_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_8s_as_8u_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_8s_as_8u_Converting_XorC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_8s_as_16u_Converting_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_8s_as_16s_Converting_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

//~~~ doesn't work correct unsafe convert
int
image_8s_as_32u_Convert(   // 8s32u_C1Rs
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_8s_as_32s_Converting_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_16u_as_8s_Converting_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_16u_as_16s_Converting_XorC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_16s_as_8s_Converting_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_16s_as_16u_Converting_XorC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_32u_as_32s_Converting_XorC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_32s_as_32u_Converting_XorC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

typedef
int(*covertHandler)(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

static covertHandler covertTable[IPP_TYPES_NUMBER][IPP_TYPES_NUMBER];

int
convert(int index1,
    int index2,
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);


// Image as float
//
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
    int img_height);

int
image_8u_as_64f_Converting_range_01_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_8s_as_32f_Converting_range_11_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);
int
image_8s_as_64f_Converting_range_11_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_16u_as_32f_Converting_range_01_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);
int
image_16u_as_64f_Converting_range_01_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);
int
image_16s_as_32f_Converting_range_11_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_16s_as_64f_Converting_range_11_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_32s_as_32f_Converting_range_11_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_32s_as_64f_Converting_range_11_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);
#endif
