# ******************************************************************************
# Copyright (c) 2020, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************

cdef extern from "ippdefs.h":
    cdef int COMPILER_SUPPORT_SHORT_FLOAT = 0

cdef extern from "ippbase.h":
    ctypedef enum IppDataType:
        ippUndef = -1
        ipp1u = 0
        ipp8u = 1
        ipp8uc = 2
        ipp8s = 3
        ipp8sc = 4
        ipp16u = 5
        ipp16uc = 6
        ipp16s = 7
        ipp16sc = 8
        ipp32u = 9
        ipp32uc = 10
        ipp32s = 11
        ipp32sc = 12
        ipp32f = 13
        ipp32fc = 14
        ipp64u = 15
        ipp64uc = 16
        ipp64s = 17
        ipp64sc = 18
        ipp64f = 19
        ipp64fc = 20
        ipp16fc = 21


cdef extern from "ipptypes.h":
    ctypedef enum IppiBorderType:
        ippBorderRepl = 1
        ippBorderWrap = 2
        ippBorderMirror = 3    # left border: 012... -> 21012...
        ippBorderMirrorR = 4   # left border: 012... -> 210012...
        ippBorderDefault = 5
        ippBorderConst = 6
        ippBorderTransp = 7
        ippBorderUndef = -1    # added for scikit-ipp


cdef extern from "ipptypes.h":
    ctypedef enum IppiWarpDirection:
        ippWarpForward = 0
        ippWarpBackward = 1

#cdef extern from "ippbase.h":
#    ctypedef IPP_MIN_32S = (-2147483647 - 1)

cdef extern from "ipptypes.h":
    ctypedef enum:
        IPPI_INTER_NN     = 1
        IPPI_INTER_LINEAR = 2
        IPPI_INTER_CUBIC  = 4
        IPPI_INTER_CUBIC2P_BSPLINE = 5     # two-parameter cubic filter (B=1, C=0)
        IPPI_INTER_CUBIC2P_CATMULLROM = 6  # two-parameter cubic filter (B=0, C=1/2)
        IPPI_INTER_CUBIC2P_B05C03 = 7      # two-parameter cubic filter (B=1/2, C=3/10)
        IPPI_INTER_SUPER  = 8
        IPPI_INTER_LANCZOS = 16
        IPPI_ANTIALIASING  = (1 << 29)
        IPPI_SUBPIXEL_EDGE = (1 << 30)
        #IPPI_SMOOTH_EDGE   = IPP_MIN_32S    # GCC gives warning for (1 << 31) definition


cdef extern from "ipptypes.h":
    ctypedef enum IppiInterpolationType:
        ippNearest = IPPI_INTER_NN
        ippLinear = IPPI_INTER_LINEAR
        ippCubic = IPPI_INTER_CUBIC2P_CATMULLROM
        ippLanczos = IPPI_INTER_LANCZOS
        ippHahn = 0
        ippSuper = IPPI_INTER_SUPER

# TODO:
# for different os versions
# update defining of IPP_UINT64 and IPP_INT64
# and usage of __int64

# cdef extern from "ippbase.h":
#IF _WIN32:
    #cdef __int64 IPP_INT64
    #cdef unsigned __int64 IPP_UINT64
#ELIF _WIN64:
    #cdef __int64 IPP_INT64
    #cdef unsigned __int64 IPP_UINT64
#ELSE:
    #cdef long long IPP_INT64
    #cdef unsigned long long IPP_UINT64


cdef extern from "ippbase.h":
    ctypedef unsigned char  Ipp8u
    ctypedef unsigned short Ipp16u
    ctypedef unsigned int   Ipp32u
    ctypedef signed char    Ipp8s
    ctypedef signed short   Ipp16s
    ctypedef signed int     Ipp32s
    ctypedef float          Ipp32f
    #ctypedef IPP_INT64      Ipp64s
    ctypedef long long      Ipp64s
    #ctypedef IPP_UINT64     Ipp64u
    ctypedef double         Ipp64f


cdef extern from "ipptypes.h":
    ctypedef int IppStatus

    ctypedef enum IppiMaskSize:
        ippMskSize1x3 = 13,
        ippMskSize1x5 = 15,
        ippMskSize3x1 = 31,
        ippMskSize3x3 = 33,
        ippMskSize5x1 = 51,
        ippMskSize5x5 = 55

    ctypedef enum  IppRoundMode:
        ippRndZero = 0
        ippRndNear = 1
        ippRndFinancial = 2 
        ippRndHintAccurate = 0x10

    ctypedef enum IppNormType:
        ippNormInf  =   0x00000001
        ippNormL1   =   0x00000002
        ippNormL2   =   0x00000004


cdef extern from "ippcore.h":
    const char * ippGetStatusString(IppStatus stsCode)


#cdef extern from "ipptypes_l.h":
#IF _M_AMD64 :
#    ctypedef Ipp64s IppSizeL
#ELIF __x86_64__:
#    ctypedef Ipp64s IppSizeL
#ELSE:
#    ctypedef int IppSizeL


#cdef extern from "ipptypes_l.h":
#    cdef struct IppiSizeL:
#        IppSizeL width
#        IppSizeL height
