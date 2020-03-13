
////////////////////////////////////////////////////////////////////////////////////////
//
//    scikit-ipp's own functions for image transformations, that uses Intel(R) IPP.
//
////////////////////////////////////////////////////////////////////////////////////////
#include "own_resize.h"

#define EXIT_FUNC exitLine:             /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine

////////////////////////////////////////////////////////////////////////////////////////
//
//    own_Resize
//
//          Changes an image size using the different interpolation methods.
//          own_Resize uses Intel(R) IPP functions on the backend for implementing
//          resizing of image as is in scikit-image.
//
//    Note: currently ippBorderValue is disabled. TODO: needs implementing.
//
////////////////////////////////////////////////////////////////////////////////////////
IppStatus
own_Resize(
    IppDataType ippDataType,
    void * pSrc,
    void * pDst,
    int img_width,
    int img_height,
    int dst_width,
    int dst_height,
    int numChannels,
    Ipp32u antialiasing,
    IppiInterpolationType interpolation,
    IppiBorderType ippBorderType,
    double ippBorderValue)
{
    // TODO:
    // currently ippBorderValue is not used
    IppStatus status = ippStsNoErr;
    IppiResizeSpec_32f *pSpec = NULL;                    // Pointer to the specification
                                                         // structure
    Ipp8u * pBuffer = NULL;                              // Pointer to the work buffer
    Ipp8u * pInit = NULL;

    Ipp32f valueB = 0.0;                                 // for ippCubic interpolation
    Ipp32f valueC = 0.5;                                 // for ippCubic interpolation

    Ipp64f pBorderValue[4];

    IppiSize srcSize = { img_width, img_height };        // Size of source image
    IppiSize dstSize = { dst_width, dst_height };        // size of destination images
    int srcStep, dstStep;                                // Steps, in bytes, through the
                                                         // source/destination images

    IppiPoint dstOffset = { 0, 0 };                      // Offset of the destination
                                                         // image ROI with respect to
                                                         // the destination image origin
    int specSize = 0, initSize = 0, bufSize = 0;         // Work buffer size

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
    // Calculation of work buffer size
    status = ippiResizeGetSize(ippDataType, srcSize, dstSize, interpolation,
        antialiasing, &specSize, &initSize);
    check_sts(status);

    // Memory allocation
    pSpec = (IppiResizeSpec_32f*)ippsMalloc_8u(specSize + initSize);
    if (pSpec == NULL)
    {
        status = ippStsMemAllocErr;
        check_sts(status);
    }
    pInit = (Ipp8u*)pSpec + initSize;

    // Filter initialization
    if (antialiasing == 0)
    {
        // ippiResizeNearestInit
        // ippiResizeLinearInit
        // ippiResizeCubicInit

        switch (interpolation)
        {
        case ippNearest:
        {
            status = ippiResizeNearestInit(ippDataType, srcSize, dstSize, pSpec);
            break;
        }
        case ippLinear:
        {
            status = ippiResizeLinearInit(ippDataType, srcSize, dstSize, pSpec);
            break;
        }
        case ippCubic:
        {
            status = ippiResizeCubicInit(ippDataType, srcSize, dstSize, valueB,
                valueC, pSpec, pInit);
            break;
        }
        default:
        {
            status = ippStsInterpolationErr;
        }
        }
    }
    else if (antialiasing == 1)
    {
        // ippiResizeAntialiasingLinearInit
        // ippiResizeAntialiasingCubicInit
        switch (interpolation)
        {
        case ippLinear:
        {
            status = ippiResizeAntialiasingLinearInit(srcSize, dstSize,
                pSpec, pInit);
            break;
        }
        case ippCubic:
        {
            status = ippiResizeAntialiasingCubicInit(srcSize, dstSize,
                valueB, valueC, pSpec, pInit);
            break;
        }
        default:
        {
            status = ippStsInterpolationErr;
        }
        }
    }
    else
    {
        status = ippStsErr;
    }
    check_sts(status);

    status = ippiResizeGetBufferSize(ippDataType, pSpec, dstSize,
        numChannels, &bufSize);
    check_sts(status);

    pBuffer = ippsMalloc_8u(bufSize);
    if (pBuffer == NULL)
    {
        status = ippStsNoMemErr;
        check_sts(status);
    }

    if (antialiasing == 0)
    {
        // TODO pBorderValue
        status = ippiResize(ippDataType, pSrc, srcStep, pDst, dstStep, dstOffset,
            dstSize, numChannels, ippBorderType, 0, interpolation, pSpec, pBuffer);
    }
    else if (antialiasing == 1)
    {

        status = ippiResizeAntialiasing(ippDataType, pSrc, srcStep, pDst, dstStep,
            dstOffset, dstSize, numChannels, ippBorderType, 0, pSpec, pBuffer);
    }
    else
    {
        status = ippStsErr;
    }
    check_sts(status);

EXIT_FUNC
    ippsFree(pSpec);
    ippsFree(pBuffer);
    return status;
}
