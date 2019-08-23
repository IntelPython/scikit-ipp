#include "median.h"


#define EXIT_LINE exitLine:                         // Label for Exit
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine;

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
MedianFilterUINT8_3D(void * pSRC,
                     int srcStep,
                     void * pDST,
                     int dstStep,
                     int img_width,
                     int img_height,
                     int img_depth,
                     int kSize,
                     IpprBorderType borderType,
                     const Ipp8u * pBorderValue)
{
    IppStatus status = ippStsNoErr;
    Ipp8u* pSrc = NULL, *pDst = NULL;     // Pointers to source and 
                                           // destination images
    
    pSrc = (Ipp8u *)pSRC;
	pDst = (Ipp8u *)pDST;
	int numChannels = 1;

	Ipp8u * pBuffer = NULL;                // Pointer to the work buffer

	IpprVolume srcRoiVolume = { img_width, img_height, img_depth };
	IpprVolume dstRoiVolume = { img_width, img_height, img_depth };
	IpprVolume maskVolume = { kSize, kSize, 1 };

	// check is needed
	int dstPlaneStep = img_width * img_height * sizeof(Ipp8u);
	int srcPlaneStep = dstPlaneStep;


	IpprFilterMedianSpec* pSpec = NULL;
	int pBufferSize = 0, pSpecSize = 0;    // Common work buffer size


	check_sts( status = ipprFilterMedianGetSize(maskVolume,
                                                dstRoiVolume,
		                                        ipp8u,
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
                                             ipp8u,
                                             numChannels,
                                             pSpec) );

	check_sts( status = ipprFilterMedian_8u_C1V(pSrc,
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
