#include "dtypes.h"

#define EXIT_FUNC exitLine:             /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine;

int
image_UINT8_as_float32(void * pSrc,
                       int srcStep,
                       void * pDst,
                       int dstStep,
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

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_8U);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_8U);
    Ipp64f minDst = 0;
    Ipp64f maxDst = 1;

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;
    
    check_sts( ippiScaleC_8u32f_C1R(pSRC,
                                    srcStep,
                                    mVal,
                                    aVal,
                                    pDST,
                                    dstStep,
                                    roiSize,
                                    ippAlgHintAccurate) );

EXIT_FUNC
    return (int)status;
};

int
image_INT8_as_float32(void * pSrc,
                      int srcStep,
                      void * pDst,
                      int dstStep,
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

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_8S);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_8S);
    Ipp64f minDst = 0;
    Ipp64f maxDst = 1;

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;
    
    check_sts( ippiScaleC_8s32f_C1R(pSRC,
                                    srcStep,
                                    mVal,
                                    aVal,
                                    pDST,
                                    dstStep,
                                    roiSize,
                                    ippAlgHintAccurate) );

EXIT_FUNC
    return (int)status;
}

int
image_UINT16_as_float32(void * pSrc,
                        int srcStep,
                        void * pDst,
                        int dstStep,
                        int img_width,
                        int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp16u * pSRC = NULL;     // Pointers to source and
    Ipp32f * pDST = NULL;    // destination images
                                             
    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp16u *)pSrc;
    pDST = (Ipp32f *)pDst;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_16U);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_16U);
    Ipp64f minDst = 0;
    Ipp64f maxDst = 1;

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;
    
    check_sts( ippiScaleC_16u32f_C1R(pSRC,
                                     srcStep,
                                     mVal,
                                     aVal,
                                     pDST,
                                     dstStep,
                                     roiSize,
                                     ippAlgHintAccurate) );

EXIT_FUNC
    return (int)status;
};

int
image_INT16_as_float32(void * pSrc,
                       int srcStep,
                       void * pDst,
                       int dstStep,
                       int img_width,
                       int img_height)
{
    IppStatus status = ippStsNoErr;
    Ipp16s * pSRC = NULL;     // Pointers to source and
    Ipp32f * pDST = NULL;    // destination images
                                             
    IppiSize roiSize = { img_width, img_height }; // Size of source and
                                                  // destination ROI in pixels
    pSRC = (Ipp16s *)pSrc;
    pDST = (Ipp32f *)pDst;

    Ipp64f minSrc = (Ipp64f)(IPP_MIN_16S);
    Ipp64f maxSrc = (Ipp64f)(IPP_MAX_16S);
    Ipp64f minDst = 0;
    Ipp64f maxDst = 1;

    Ipp64f mVal = (maxDst - minDst) / (maxSrc - minSrc);
    Ipp64f aVal = minDst - minSrc * mVal;
    
    check_sts( ippiScaleC_16s32f_C1R(pSRC,
                                     srcStep,
                                     mVal,
                                     aVal,
                                     pDST,
                                     dstStep,
                                     roiSize,
                                     ippAlgHintAccurate) );

EXIT_FUNC
    return (int)status;
}
