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

////////////////////////////////////////////////////////////////////////////////////////
//
//    scikit-ipp's own functions for image warp transformations, that uses
//    Intel(R) Integrated Performance Primitives (Intel(R) IPP).
//
////////////////////////////////////////////////////////////////////////////////////////
#ifndef WARP_H
#define WARP_H
#include <stddef.h>
#include <ipp.h>
#ifdef USE_OPENMP
   #include <omp.h>
#endif
#include "_ipp_wr.h"
#include "utils.h"


////////////////////////////////////////////////////////////////////////////////////////
//
//    own_Warp
//
//    own_Warp uses Intel(R) IPP funcstions for implementing image warp
//    transformations
//
//    TODO: complete the description.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
own_Warp(
    IppDataType ippDataType,
    void * pSrc,
    void * pDst,
    int img_width,
    int img_height,
    int dst_width,
    int dst_height,
    int numChannels,
    double * coeffs,
    IppiInterpolationType interpolation,
    IppiWarpDirection direction,
    IppiBorderType ippBorderType,
    double ippBorderValue);
#endif // WARP_H
