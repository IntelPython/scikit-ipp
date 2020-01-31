

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


cdef extern from "ippbase.h":
    ctypedef unsigned char  Ipp8u
    ctypedef unsigned short Ipp16u
    ctypedef unsigned int   Ipp32u
    ctypedef signed char    Ipp8s
    ctypedef signed short   Ipp16s
    ctypedef signed int     Ipp32s
    ctypedef float          Ipp32f
    # ctypedef IPP_INT64    Ipp64s
    # ctypedef IPP_UINT64   Ipp64u
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
