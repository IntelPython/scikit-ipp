#include "warp.h"

#define EXIT_FUNC exitLine:             /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine

IppStatus
ippiWarpAffineCubic(
    IppDataType ippDataType,
    int numChannels,
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    IppiPoint dstOffset,
    IppiSize dstSize,
    IppiWarpSpec* pSpec,
    Ipp8u * pBuffer
)
{
    IppStatus status = ippStsNoErr;
    if (numChannels == 1)
    {
        switch (ippDataType)
        {
        case ipp8u:
        {
            status = ippiWarpAffineCubic_8u_C1R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp16u:
        {
            status = ippiWarpAffineCubic_16u_C1R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp16s:
        {
            status = ippiWarpAffineCubic_16s_C1R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            status = ippiWarpAffineCubic_32f_C1R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp64f:
        {
            status = ippiWarpAffineCubic_64f_C1R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else if (numChannels == 3)
    {
        switch (ippDataType)
        {
        case ipp8u:
        {
            status = ippiWarpAffineCubic_8u_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp16u:
        {
            status = ippiWarpAffineCubic_16u_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp16s:
        {
            status = ippiWarpAffineCubic_16s_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            status = ippiWarpAffineCubic_32f_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp64f:
        {
            status = ippiWarpAffineCubic_64f_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else if (numChannels == 4)
    {
        switch (ippDataType)
        {
        case ipp8u:
        {
            status = ippiWarpAffineCubic_8u_C4R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp16u:
        {
            status = ippiWarpAffineCubic_16u_C4R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp16s:
        {
            status = ippiWarpAffineCubic_16s_C4R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            status = ippiWarpAffineCubic_32f_C4R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp64f:
        {
            status = ippiWarpAffineCubic_64f_C4R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else
    {
        status = ippStsErr;
    }
    return status;
}

IppStatus
ippiWarpAffineNearest(
    IppDataType ippDataType,
    int numChannels,
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    IppiPoint dstOffset,
    IppiSize dstSize,
    IppiWarpSpec* pSpec,
    Ipp8u * pBuffer
)
{
    IppStatus status = ippStsNoErr;
    if (numChannels == 1)
    {
        switch (ippDataType)
        {
        case ipp8u:
        {
            status = ippiWarpAffineNearest_8u_C1R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp16u:
        {
            status = ippiWarpAffineNearest_16u_C1R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp16s:
        {
            status = ippiWarpAffineNearest_16s_C1R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            status = ippiWarpAffineNearest_32f_C1R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp64f:
        {
            status = ippiWarpAffineNearest_64f_C1R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else if (numChannels == 3)
    {
        switch (ippDataType)
        {
        case ipp8u:
        {
            status = ippiWarpAffineNearest_8u_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp16u:
        {
            status = ippiWarpAffineNearest_16u_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp16s:
        {
            status = ippiWarpAffineNearest_16s_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            status = ippiWarpAffineNearest_32f_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp64f:
        {
            status = ippiWarpAffineNearest_64f_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else if (numChannels == 4)
    {
        switch (ippDataType)
        {
        case ipp8u:
        {
            status = ippiWarpAffineNearest_8u_C4R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp16u:
        {
            status = ippiWarpAffineNearest_16u_C4R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp16s:
        {
            status = ippiWarpAffineNearest_16s_C4R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            status = ippiWarpAffineNearest_32f_C4R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp64f:
        {
            status = ippiWarpAffineNearest_64f_C4R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else
    {
        status = ippStsErr;
    }
    return status;
}

IppStatus
ippiWarpAffineLinear(
    IppDataType ippDataType,
    int numChannels,
    void * pSrc,
    int srcStep,
    void * pDst,
    int dstStep,
    IppiPoint dstOffset,
    IppiSize dstSize,
    IppiWarpSpec* pSpec,
    Ipp8u * pBuffer
)
{
    IppStatus status = ippStsNoErr;
    if (numChannels == 1)
    {
        switch (ippDataType)
        {
        case ipp8u:
        {
            status = ippiWarpAffineLinear_8u_C1R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp16u:
        {
            status = ippiWarpAffineLinear_16u_C1R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp16s:
        {
            status = ippiWarpAffineLinear_16s_C1R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            status = ippiWarpAffineLinear_32f_C1R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp64f:
        {
            status = ippiWarpAffineLinear_64f_C1R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else if (numChannels == 3)
    {
        switch (ippDataType)
        {
        case ipp8u:
        {
            status = ippiWarpAffineLinear_8u_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp16u:
        {
            status = ippiWarpAffineLinear_16u_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp16s:
        {
            status = ippiWarpAffineLinear_16s_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            status = ippiWarpAffineLinear_32f_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp64f:
        {
            status = ippiWarpAffineLinear_64f_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else if (numChannels == 4)
    {
        switch (ippDataType)
        {
        case ipp8u:
        {
            status = ippiWarpAffineLinear_8u_C4R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp16u:
        {
            status = ippiWarpAffineLinear_16u_C4R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp16s:
        {
            status = ippiWarpAffineLinear_16s_C4R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp32f:
        {
            status = ippiWarpAffineLinear_32f_C4R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        case ipp64f:
        {
            status = ippiWarpAffineLinear_64f_C4R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
            break;
        }
        default:
        {
            status = ippStsDataTypeErr;
        }
        }
    }
    else
    {
        status = ippStsErr;
    }
    return status;
}

IppStatus
ippi_RotateCoeffs(
    double angle,
    double xCenter,
    double yCenter,
    double *coeffs)
{
    IppStatus status = ippStsNoErr;
    double xShift;
    double yShift;
    status = ippiGetRotateShift(xCenter, yCenter, angle, &xShift, &yShift);
    check_sts(status);
    status = ippiGetRotateTransform(angle, xShift, yShift, (double(*)[3])coeffs);
    check_sts(status);
EXIT_FUNC
    return status;
}

IppStatus
ippi_GetAffineDstSize(
    int img_width,
    int img_height,
    int * dst_width,
    int * dst_height,
    double * coeffs)
{
    IppStatus status = ippStsNoErr;
    double bound[2][2] = { 0 };
    IppiRect srcRoi;
    srcRoi.x = 0;
    srcRoi.y = 0;
    srcRoi.width = img_width;
    srcRoi.height = img_height;

    status = ippiGetAffineBound(srcRoi, bound, (double(*)[3])coeffs);
    check_sts(status);
    // TODO
    // more correct formula
    *dst_width = (int)(bound[1][0] - bound[0][0] + 2);
    *dst_height = (int)(bound[1][1] - bound[0][1] + 2);

    // unused
    //IppiSize dstSize_new = { *dst_width, *dst_height };  // size of destination images
EXIT_FUNC
    return status;
}

IppStatus
ippi_Warp(
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
    double ippBorderValue)
{
    IppStatus status = ippStsNoErr;
    IppiWarpSpec* pSpec = NULL;                    // Pointer to the specification structure

    Ipp8u* pInitBuf = NULL;

    // ``scikit-image`` uses Catmull-Rom spline (0.0, 0.5)
    // Catmull-Rom spline (0.0, 0.5)
    // Don P. Mitchell, Arun N. Netravali. Reconstruction Filters in Computer Graphics.
    // Computer Graphics, Volume 22, Number 4, AT&T Bell Laboratories, Murray Hill, 
    // New Jersey, August 1988.
    Ipp64f valueB = 0.0;
    Ipp64f valueC = 0.5;

    Ipp8u * pBuffer = NULL;

    Ipp64f pBorderValue[4];
    IppiSize srcSize = { img_width, img_height };  // Size of source image
    IppiSize dstSize = { dst_width, dst_height };  // size of destination images
    int srcStep, dstStep;                // Steps, in bytes, through the source/destination images

    IppiPoint dstOffset = { 0, 0 };      // Offset of the destination image ROI with respect to
                                         // the destination image origin
    int specSize = 0, initSize = 0, bufSize = 0; // Work buffer size

    int sizeof_src;

    status = get_sizeof(ippDataType, &sizeof_src);
    check_sts(status);

    srcStep = numChannels * img_width * sizeof_src;
    dstStep = numChannels * dst_width * sizeof_src;;

    if (numChannels == 1) {
        pBorderValue[0] = (Ipp64f)ippBorderValue;
    }
    else if (numChannels == 3)
    {
        pBorderValue[0] = (Ipp64f)ippBorderValue;
        pBorderValue[1] = (Ipp64f)ippBorderValue;
        pBorderValue[2] = (Ipp64f)ippBorderValue;
    }
    else if (numChannels == 4)
    {
        pBorderValue[0] = (Ipp64f)ippBorderValue;
        pBorderValue[1] = (Ipp64f)ippBorderValue;
        pBorderValue[2] = (Ipp64f)ippBorderValue;
        pBorderValue[3] = (Ipp64f)ippBorderValue;
    }
    else
    {
        status = ippStsErr;
        check_sts(status);
    }
    // Spec and init buffer sizes
    status = ippiWarpAffineGetSize(srcSize, dstSize, ippDataType, (double(*)[3])coeffs, interpolation, direction,
        ippBorderType, &specSize, &initSize);
    check_sts(status);

    pInitBuf = ippsMalloc_8u(initSize);
    if (pInitBuf == NULL)
    {
        status = ippStsNoMemErr;
        check_sts(status);
    }

    pSpec = (IppiWarpSpec*)ippsMalloc_8u(specSize);
    if (pSpec == NULL)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    }
    // Filter initialization
    switch (interpolation)
    {
    case ippCubic:
    {
        status = ippiWarpAffineCubicInit(srcSize, dstSize, ippDataType, (double(*)[3])coeffs, direction,
            numChannels, valueB, valueC, ippBorderType, pBorderValue, 0, pSpec, pInitBuf);
        break;
    }
    case ippNearest:
    {
        status = ippiWarpAffineNearestInit(srcSize, dstSize, ippDataType, (double(*)[3])coeffs, direction,
            numChannels, ippBorderType, pBorderValue, 0, pSpec);
        break;
    }
    case ippLinear:
    {
        status = ippiWarpAffineLinearInit(srcSize, dstSize, ippDataType, (double(*)[3])coeffs, direction,
            numChannels, ippBorderType, pBorderValue, 0, pSpec);
        break;
    }
    default:
    {
        status = ippStsErr;
    }
    }
    check_sts(status);

    // Get work buffer size
    status = ippiWarpGetBufferSize(pSpec, dstSize, &bufSize);
    check_sts(status);

    pBuffer = ippsMalloc_8u(bufSize);
    if (pBuffer == NULL)
    {
        check_sts(status = ippStsMemAllocErr);
    };

    switch (interpolation)
    {
    case ippCubic:
    {
        status = ippiWarpAffineCubic(ippDataType, numChannels, pSrc, srcStep,
            pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
        break;
    }
    case ippNearest:
    {
        status = ippiWarpAffineNearest(ippDataType, numChannels, pSrc, srcStep,
            pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
        break;
    }
    case ippLinear:
    {
        status = ippiWarpAffineLinear(ippDataType, numChannels, pSrc, srcStep,
            pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);
        break;
    }
    default:
    {
        status = ippStsErr;
    }
    }
    check_sts(status);

EXIT_FUNC
    ippsFree(pInitBuf);
    ippsFree(pBuffer);
    ippsFree(pSpec);
    return status;
}
