#include "ipp.h"
// #include "ippcore_tl.h"
#ifndef DTYPES_H
#define DTYPES_H

#define IPP_GAUSSIAN_SUPPORTED_DTYPES  0x2C2 // 1011000010

#define MAX_C3_IMG_WIDTH_BY_INT32_ROI_DTYPE 0x2AAAAAAA

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
image_no_convert(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_8u_as_8s_XorC(
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
image_8u_as_16s_ScaleC(
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
image_8s_as_8u_ScaleC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_8s_as_8u_XorC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_16u_as_16s_XorC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_16s_as_16u_XorC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_32u_as_32s_XorC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

int
image_32s_as_32u_XorC(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

// image_as_float32
int
image_UINT8_as_float32(
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    int img_width,
    int img_height);

int
image_INT8_as_float32(void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    int img_width,
    int img_height);

int
image_UINT16_as_float32(void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    int img_width,
    int img_height);

int
image_INT16_as_float32(void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    int img_width,
    int img_height);

typedef
int(*covertHandler)(
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);

static covertHandler covertTable[10][10];

int
convert(int index1,
    int index2,
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height);
#endif
