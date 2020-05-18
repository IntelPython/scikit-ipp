/*******************************************************************************
* Copyright (c) 2020, Intel Corporation
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*     * Redistributions of source code must retain the above copyright notice,
*       this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of Intel Corporation nor the names of its contributors
*       may be used to endorse or promote products derived from this software
*       without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

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

// #include "ipp.h"
// #include "ippcore_tl.h"
// #ifndef DTYPES_H
// #define DTYPES_H
// #include <stddef.h>

// #define MAX_C3_IMG_WIDTH_BY_INT32_ROI_DTYPE 0x2AAAAAAA

// #define IPP_TYPES_NUMBER 10
/*
#define IPPi_ScaleC_SUPPORTED_TYPES_NUMBER 7

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
    ippUndef_index = -1,
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
    ipp8u_r = 0x1FF,   // 0111111111
    ipp8s_r = 0x2FF,   // 1011111111
    ipp16u_r = 0x7F,   // 0001111111
    ipp16s_r = 0xBF,   // 0010111111
    ipp32u_r = 0x1F,   // 0000011111
    ipp32s_r = 0x2F,   // 0000101111
    ipp64u_r = 0x7,    // 0000000111
    ipp64s_r = 0xB,    // 0000001011
    ipp32f_r = 0x1,    // 0000000001
    ipp64f_r = 0x2,    // 0000000010
} IppDataTypeConversionRecomendationMask;

// returns IppDataType enum by ipp<type>_index
int
ipp_type_index_as_IppDataType(
    IppDataType * ippDataType,
    IppDataTypeIndex ipp_type_index
);

int
get_ipp_src_dst_index(int output_index, int ipp_func_support_dtypes);

IppDataTypeIndex
ippDtypeMask_as_ippDtypeIndex(int ippDtypeMask);

void *
malloc_by_dtype_index(
    int index,
    int numChannels,
    int img_width,
    int img_height
);

int
sizeof_ipp_dtype_by_index(
    int * sizeof_type,
    int type_index
);

typedef int func_jumpt_table_index;

typedef enum {
    preserve_range_false = 0,                         //  [0...1] or [-1...1] for unsigned and signed
    preserve_range_true = 1,                          //  [dst_dtype_min...dst_dtype_max]
    preserve_range_true_for_small_bitsize_src = 2     //  [dst_dtype_min...dst_dtype_max], where 
                                                      //  dst_dtype_min is src_dtype_min and
                                                      //  dst_dtype_max is src_dtype_max
} preserve_range_flag;

int
image_ScaleC(
    IppDataTypeIndex src_index,
    IppDataTypeIndex dst_index,
    void * pSrc,
    void * pDst,
    int numChannels,
    int img_width,
    int img_height,
    preserve_range_flag preserve_range);

typedef
IppStatus(*ScaleC_Handler)(
    void * pSrc,
    int srcStep,
    Ipp64f mVal,
    Ipp64f aVal,
    void * pDst,
    int dstStep,
    IppiSize roiSize,
    IppHintAlgorithm hint);
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
#endif
*/
