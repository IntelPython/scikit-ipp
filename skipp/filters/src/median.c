#include "median.h"


#define EXIT_LINE exitLine:  // Label for Exit
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine;

// new
int
MedianFilter_32f_C1_3D(
    void * pSRC,
    void * pDST,
    int img_width,
    int img_height,
    int img_depth,
    int mask_width,
    int mask_height,
    int mask_depth,
    int borderType)
{
    IppStatus status = ippStsNoErr;
    Ipp32f * pSrc = NULL;  // Pointers to source and 
    Ipp32f * pDst = NULL;  // destination images
                                           
    pSrc = (Ipp32f *)pSRC;
    pDst = (Ipp32f *)pDST;

    int numChannels = 1;

    Ipp8u * pBuffer = NULL;  // Pointer to the work buffer

    IpprVolume srcRoiVolume = { img_width, img_height, img_depth };
    IpprVolume dstRoiVolume = { img_width, img_height, img_depth };
    IpprVolume maskVolume = { mask_width, mask_height, mask_depth };

    // srcPlaneStep & dstPlaneStep
    int planeStep = img_width * img_height * sizeof(Ipp32f);

    // srcStep & dstStep
    int srcDstStep = img_width * sizeof(Ipp32f);

    // ~~~~ correct
    Ipp32f pBorderValue = 0; // cval

    IpprFilterMedianSpec * pSpec = NULL;
    int pBufferSize = 0;     // Common work buffer size
    int pSpecSize = 0; 


    status = ipprFilterMedianGetSize(
        maskVolume,
        dstRoiVolume,
        ipp32f,
        numChannels,
        &pSpecSize,
        &pBufferSize);
    

    pSpec = (IpprFilterMedianSpec *)ippsMalloc_8u(pSpecSize);
    if (pSpec == NULL)
    {
        check_sts(status = ippStsMemAllocErr);
    };

    pBuffer = ippsMalloc_8u(pBufferSize);
    if (NULL == pBuffer)
    {
        check_sts(status = ippStsMemAllocErr);
    };

    status = ipprFilterMedianInit(
        maskVolume,
        ipp32f,
        numChannels,
        pSpec);

    check_sts(status);

    status = ipprFilterMedian_32f_C1V(
        pSrc,
        planeStep,
        srcDstStep,
        pDst,
        planeStep,
        srcDstStep,
        dstRoiVolume,
        borderType,
        &pBorderValue,
        pSpec,
        pBuffer);

    check_sts(status);

EXIT_LINE
    ippsFree(pBuffer);
    ippsFree(pSpec);
    return(int)status;
}

int
MedianFilterFLOAT32_3D(void * pSRC,
                       int srcStep,
                       void * pDST,
                       int dstStep,
                       int img_width,
                       int img_height,
                       int img_depth,
                       int kSize,
                       IpprBorderType borderType,
                       const Ipp32f * pBorderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp32f* pSrc = NULL, *pDst = NULL;     // Pointers to source and 
                                           // destination images
    
    pSrc = (Ipp32f *)pSRC;
    pDst = (Ipp32f *)pDST;
    int numChannels = 1;

    Ipp8u * pBuffer = NULL;                // Pointer to the work buffer

    IpprVolume srcRoiVolume = { img_width, img_height, img_depth };
    IpprVolume dstRoiVolume = { img_width, img_height, img_depth };
    IpprVolume maskVolume = { kSize, kSize, 1 };

    // check is needed
    int dstPlaneStep = img_width * img_height * sizeof(Ipp32f);
    int srcPlaneStep = dstPlaneStep;


    IpprFilterMedianSpec* pSpec = NULL;
    int pBufferSize = 0, pSpecSize = 0;    // Common work buffer size


    check_sts( status = ipprFilterMedianGetSize(maskVolume,
                                                dstRoiVolume,
                                                ipp32f,
                                                numChannels,
                                                &pSpecSize,
                                                &pBufferSize) );

    pSpec = (IpprFilterMedianSpec *)ippsMalloc_8u(pSpecSize);
    if (NULL == pSpec)
    {
        check_sts(status = ippStsMemAllocErr);
    };

    pBuffer = ippsMalloc_8u(pBufferSize);
    if (NULL == pBuffer)
    {
        check_sts(status = ippStsMemAllocErr);
    };

    check_sts( status = ipprFilterMedianInit(maskVolume,
                                             ipp32f,
                                             numChannels,
                                             pSpec) );

    check_sts( status = ipprFilterMedian_32f_C1V(pSrc,
                                                 srcPlaneStep,
                                                 srcStep,
                                                 pDst,
                                                 dstPlaneStep,
                                                 dstStep,
                                                 dstRoiVolume,
                                                 borderType,
                                                 pBorderValue,
                                                 pSpec,
                                                 pBuffer) );
    
EXIT_LINE
    ippsFree(pBuffer);
    ippsFree(pSpec);
    // printf("Exit status %d (%s)\n", (int)status, ippGetStatusString(status));
    return(int)status;
}

int
MedianFilterFLOAT32(void * pSRC,
                    int stepSize,
                    void * pDST,
                    int img_width,
                    int img_height,
                    int mask_width,
                    int mask_height,
                    int borderType) // const float * pBorderValue) <-----~~
{
    IppStatus status = ippStsNoErr;
    Ipp32f* pSrc = NULL, *pDst = NULL;     // Pointers to source and 
                                          // destination images
    int srcStep = stepSize, dstStep = stepSize;    // Steps, in bytes, through
                                          //  the source/destination images
    IppiSize dstRoiSize = { img_width, img_height };  // Size of source and
                                                 // destination ROI in pixels

    IppiSize maskSize = { mask_width, mask_height };  // Size of source and
                                                 // destination ROI in pixels
    Ipp8u *pBuffer = NULL;                   // Pointer to the work buffer
    int bufferSize;

    pSrc = (Ipp32f *)pSRC;
    pDst = (Ipp32f *)pDST;

    // IppiBorderType borderType = ippBorderRepl;
    Ipp32f borderValue = 0;

    int numChannels = 1;

    // get buffer size
    status = ippiFilterMedianBorderGetBufferSize(dstRoiSize,
                                                 maskSize,
                                                 ipp32f,
                                                 numChannels,
                                                 &bufferSize);
    pBuffer = ippsMalloc_8u(bufferSize);
    if(NULL == pBuffer)
    {
        check_sts( status = ippStsMemAllocErr);
    };

    status =  ippiFilterMedianBorder_32f_C1R(pSrc, 
                                             srcStep,
                                             pDst,
                                             dstStep,
                                             dstRoiSize,
                                             maskSize,
                                             borderType,
                                             borderValue,
                                             pBuffer);

EXIT_LINE
    ippsFree(pBuffer);
    return (int)status;
}
