#include "edges.h"

#define EXIT_FUNC exitLine:             /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine;

int
PrewittFilterFLOAT32(void * pA_srcDst,
                     void * pB_srcDst,
                     int stepsize,
                     int img_width,
                     int img_height)
{
    // computes the square root of the sum of squares of the horizontal
    // sqrt(A**2 + B**2)/sqrt(2) and vertical Prewitt transforms.
    IppStatus status = ippStsNoErr;
    Ipp32f* pAsrcDst = NULL, *pBsrcDst = NULL;     // Pointers to source-destination
                                                   // images for IPP in-place funcs
    IppiSize roiSize = { img_width, img_height };  // Size of source-destination
                                                   // ROI in pixels
    pAsrcDst = (Ipp32f *) pA_srcDst;  // A
    pBsrcDst = (Ipp32f *) pB_srcDst;  // B pointer to the image array, that contains
                                      // results

    int aSrcDstStep = stepsize;
    int bSrcDstStep = stepsize;

    check_sts( ippiSqr_32f_C1IR(pAsrcDst, aSrcDstStep, roiSize) );
    check_sts( ippiSqr_32f_C1IR(pBsrcDst, bSrcDstStep, roiSize) );
    check_sts( ippiAdd_32f_C1IR(pAsrcDst, aSrcDstStep, pBsrcDst, bSrcDstStep, roiSize) );
    check_sts( ippiSqrt_32f_C1IR(pBsrcDst, bSrcDstStep, roiSize) );

    Ipp32f sqrt2 = (Ipp32f)IPP_SQRT2;
   
    check_sts( ippiDivC_32f_C1IR(sqrt2, pBsrcDst, bSrcDstStep, roiSize) );
EXIT_FUNC
    return (int)status; 
};

int
SobelFilterCrossFLOAT32(void * pSRC,
                        int srcStep,
                        void * pDST,
                        int dstStep,
                        int img_width,
                        int img_height,
                        int maskSize,
                        int borderType,
                        Ipp32f borderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp32f* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source and
                                             // destination images
    IppiSize roiSize = { img_width, img_height };  // Size of source and
                                                   // destination ROI in pixels
    pSrc1 = (Ipp32f *) pSRC;
    pDst1 = (Ipp32f *) pDST;
    int bufferSize;
    Ipp8u *pBuffer = NULL;
    check_sts( ippiFilterSobelCrossGetBufferSize_32f_C1R(roiSize,
                                                         maskSize,
                                                         &bufferSize) );
    pBuffer = ippsMalloc_8u(bufferSize);
    if(NULL == pBuffer)
    {
        check_sts( status = ippStsMemAllocErr);
    }
    check_sts( ippiFilterSobelCrossBorder_32f_C1R(pSrc1,
                                                  srcStep,
                                                  pDst1,
                                                  dstStep,
                                                  roiSize,
                                                  maskSize,
                                                  borderType,
                                                  borderValue,
                                                  pBuffer) );
EXIT_FUNC
    ippsFree(pBuffer);
    return (int)status;
}
