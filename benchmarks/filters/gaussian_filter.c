//#include "gaussian.h"
#include <stdio.h>
#include <stdlib.h>
#include "ipp.h"

#define EXIT_FUNC exitLine:                                  /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine; 


int GaussianFilterUINT8(void *pSRC, void * pDST, int img_width, int img_height,
	int numChannels, float sigma_, int kernelSize, int stepSize, int ippBorderType, float ippBorderValue) {

	IppStatus status = ippStsNoErr;
	Ipp8u* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source/destination images
	int srcStep = stepSize, dstStep = stepSize;          // Steps, in bytes, through the source/destination images
	IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels
	Ipp32f sigma = (Ipp32f)sigma_;
	Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer 
	IppFilterGaussianSpec* pSpec = NULL;   // context structure 
	int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size
	IppiBorderType borderType = ippBorderType; // ~~~~ added here
	Ipp8u borderValue = (Ipp8u)ippBorderValue;
	//// border values i
	pSrc1 = (Ipp8u *)pSRC;
	pDst1 = (Ipp8u *)pDST;
	check_sts(status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp8u,
		numChannels, &iSpecSize, &iTmpBufSize));
	//check is needed 
	pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
	//check is needed 
	pBuffer = ippsMalloc_8u(iTmpBufSize);
	check_sts(status = ippiFilterGaussianInit(roiSize, kernelSize, sigma,
		borderType, ipp8u, numChannels, pSpec, pBuffer));
	check_sts(status = ippiFilterGaussianBorder_8u_C1R(pSrc1, srcStep, pDst1, dstStep,
		roiSize, borderValue, pSpec, pBuffer));

	EXIT_FUNC
		ippsFree(pBuffer);
	ippsFree(pSpec);
	return (int)status;
}

int GaussianFilterUINT8RGB(void *pSRC, void * pDST, int img_width, int img_height,
	int numChannels, float sigma_, int kernelSize, int stepSize, int ippBorderType, float ippBorderValue) {

	IppStatus status = ippStsNoErr;
	Ipp8u* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source/destination images 
	int srcStep = stepSize, dstStep = stepSize;          //Steps, in bytes, through the source/destination images 
	IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels 
	Ipp32f sigma = (Ipp32f)sigma_;
	Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer
	IppFilterGaussianSpec* pSpec = NULL;   // context structure
	int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size
	IppiBorderType borderType = ippBorderType;
	Ipp8u borderValue[] = { 0, 0, 0 };
	pSrc1 = (Ipp8u *)pSRC;
	pDst1 = (Ipp8u *)pDST;
	check_sts(status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp8u,
		numChannels, &iSpecSize, &iTmpBufSize));
	//check is needed 
	pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
	//check is needed 
	pBuffer = ippsMalloc_8u(iTmpBufSize);
	check_sts(status = ippiFilterGaussianInit(roiSize, kernelSize, sigma,
		borderType, ipp8u, numChannels, pSpec, pBuffer));
	check_sts(status = ippiFilterGaussianBorder_8u_C3R(pSrc1, srcStep, pDst1, dstStep,
		roiSize, borderValue, pSpec, pBuffer));

	EXIT_FUNC
		ippsFree(pBuffer);
	ippsFree(pSpec);
	return (int)status;
}

int GaussianFilterUINT16(void *pSRC, void * pDST, int img_width, int img_height,
	int numChannels, float sigma_, int kernelSize, int stepSize, int ippBorderType, float ippBorderValue) {

	IppStatus status = ippStsNoErr;
	Ipp16u* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source/destination images
	int srcStep = stepSize, dstStep = stepSize;          // Steps, in bytes, through the source/destination images
	IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels
	Ipp32f sigma = (Ipp32f)sigma_;
	Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer
	IppFilterGaussianSpec* pSpec = NULL;   // context structure
	int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size
	IppiBorderType borderType = ippBorderType;
	Ipp16u borderValue = (Ipp16u)ippBorderValue;
	pSrc1 = (Ipp16u *)pSRC;
	pDst1 = (Ipp16u *)pDST;
	check_sts(status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp16u,
		numChannels, &iSpecSize, &iTmpBufSize));
	//check is needed 
	pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
	//check is needed 
	pBuffer = ippsMalloc_8u(iTmpBufSize);
	check_sts(status = ippiFilterGaussianInit(roiSize, kernelSize, sigma,
		borderType, ipp16u, numChannels, pSpec, pBuffer));
	check_sts(status = ippiFilterGaussianBorder_16u_C1R(pSrc1, srcStep, pDst1, dstStep,
		roiSize, borderValue, pSpec, pBuffer));

	EXIT_FUNC
		ippsFree(pBuffer);
	ippsFree(pSpec);
	return (int)status;
}

int GaussianFilterINT16(void *pSRC, void * pDST, int img_width, int img_height,
	int numChannels, float sigma_, int kernelSize, int stepSize, int ippBorderType, float ippBorderValue) {

	IppStatus status = ippStsNoErr;
	Ipp16s* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source/destination images
	int srcStep = stepSize, dstStep = stepSize;          // Steps, in bytes, through the source/destination images
	IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels
	Ipp32f sigma = (Ipp32f)sigma_;
	Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer 
	IppFilterGaussianSpec* pSpec = NULL;   // context structure 
	int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size 
	IppiBorderType borderType = ippBorderType;
	Ipp16s borderValue = (Ipp16s)ippBorderValue;
	pSrc1 = (Ipp16s *)pSRC;
	pDst1 = (Ipp16s *)pDST;
	check_sts(status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp16s,
		numChannels, &iSpecSize, &iTmpBufSize));
	//check is needed 
	pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
	//check is needed 
	pBuffer = ippsMalloc_8u(iTmpBufSize);
	check_sts(status = ippiFilterGaussianInit(roiSize, kernelSize, sigma,
		borderType, ipp16s, numChannels, pSpec, pBuffer));
	check_sts(status = ippiFilterGaussianBorder_16s_C1R(pSrc1, srcStep, pDst1, dstStep,
		roiSize, borderValue, pSpec, pBuffer));

	EXIT_FUNC
		ippsFree(pBuffer);
	ippsFree(pSpec);
	return (int)status;
}

int GaussianFilterUINT16RGB(void *pSRC, void * pDST, int img_width, int img_height,
	int numChannels, float sigma_, int kernelSize, int stepSize, int ippBorderType, float ippBorderValue) {

	IppStatus status = ippStsNoErr;
	Ipp16u* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source/destination images
	int srcStep = stepSize, dstStep = stepSize;          // Steps, in bytes, through the source/destination images
	IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels
	Ipp32f sigma = (Ipp32f)sigma_;
	Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer
	IppFilterGaussianSpec* pSpec = NULL;   // context structure 
	int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size
	IppiBorderType borderType = ippBorderType;
	Ipp16u borderValue[] = { 0, 0, 0 };
	pSrc1 = (Ipp16u *)pSRC;
	pDst1 = (Ipp16u *)pDST;
	check_sts(status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp16u,
		numChannels, &iSpecSize, &iTmpBufSize));
	//check is needed 
	pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
	//check is needed 
	pBuffer = ippsMalloc_8u(iTmpBufSize);
	check_sts(status = ippiFilterGaussianInit(roiSize, kernelSize, sigma,
		borderType, ipp16u, numChannels, pSpec, pBuffer));
	check_sts(status = ippiFilterGaussianBorder_16u_C3R(pSrc1, srcStep, pDst1, dstStep,
		roiSize, borderValue, pSpec, pBuffer));

	EXIT_FUNC
		ippsFree(pBuffer);
	ippsFree(pSpec);
	return (int)status;
}

int GaussianFilterINT16RGB(void *pSRC, void * pDST, int img_width, int img_height,
	int numChannels, float sigma_, int kernelSize, int stepSize, int ippBorderType, float ippBorderValue) {

	IppStatus status = ippStsNoErr;
	Ipp16s* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source/destination images
	int srcStep = stepSize, dstStep = stepSize;          // Steps, in bytes, through the source/destination images
	IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels
	Ipp32f sigma = (Ipp32f)sigma_;
	Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer 
	IppFilterGaussianSpec* pSpec = NULL;   // context structure 
	int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size 
	IppiBorderType borderType = ippBorderType;
	Ipp16s borderValue[] = { 0, 0, 0 };
	pSrc1 = (Ipp16s *)pSRC;
	pDst1 = (Ipp16s *)pDST;
	check_sts(status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp16s,
		numChannels, &iSpecSize, &iTmpBufSize));
	//check is needed 
	pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
	//check is needed 
	pBuffer = ippsMalloc_8u(iTmpBufSize);
	check_sts(status = ippiFilterGaussianInit(roiSize, kernelSize, sigma,
		borderType, ipp16s, numChannels, pSpec, pBuffer));
	check_sts(status = ippiFilterGaussianBorder_16s_C3R(pSrc1, srcStep, pDst1, dstStep,
		roiSize, borderValue, pSpec, pBuffer));

	EXIT_FUNC
		ippsFree(pBuffer);
	ippsFree(pSpec);
	return (int)status;
}

int GaussianFilterFLOAT32(void *pSRC, void * pDST, int img_width, int img_height,
	int numChannels, float sigma_, int kernelSize, int stepSize, int ippBorderType, float ippBorderValue) {

	IppStatus status = ippStsNoErr;
	Ipp32f* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source/destination images
	int srcStep = stepSize, dstStep = stepSize;          // Steps, in bytes, through the source/destination images
	IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels
	Ipp32f sigma = (Ipp32f)sigma_;
	Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer 
	IppFilterGaussianSpec* pSpec = NULL;   // context structure 
	int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size 
	IppiBorderType borderType = ippBorderType;
	Ipp32f borderValue = (Ipp32f)ippBorderValue;
	pSrc1 = (Ipp32f *)pSRC;
	pDst1 = (Ipp32f *)pDST;
	check_sts(status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp32f,
		numChannels, &iSpecSize, &iTmpBufSize));
	//check is needed 
	pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
	//check is needed
	pBuffer = ippsMalloc_8u(iTmpBufSize);
	check_sts(status = ippiFilterGaussianInit(roiSize, kernelSize, sigma,
		borderType, ipp32f, numChannels, pSpec, pBuffer));
	check_sts(status = ippiFilterGaussianBorder_32f_C1R(pSrc1, srcStep, pDst1, dstStep,
		roiSize, borderValue, pSpec, pBuffer));

	EXIT_FUNC
		ippsFree(pBuffer);
	ippsFree(pSpec);
	return (int)status;
}

int GaussianFilterFLOAT32RGB(void *pSRC, void * pDST, int img_width, int img_height,
	int numChannels, float sigma_, int kernelSize, int stepSize, int ippBorderType, float ippBorderValue) {

	IppStatus status = ippStsNoErr;
	Ipp32f* pSrc1 = NULL, *pDst1 = NULL;     // Pointers to source/destination images
	int srcStep = stepSize, dstStep = stepSize;          // Steps, in bytes, through the source/destination images
	IppiSize roiSize = { img_width, img_height };  // Size of source/destination ROI in pixels
	Ipp32f sigma = (Ipp32f)sigma_;
	Ipp8u *pBuffer = NULL;                 // Pointer to the work buffer 
	IppFilterGaussianSpec* pSpec = NULL;   // context structure 
	int iTmpBufSize = 0, iSpecSize = 0;    // Common work buffer size 
	IppiBorderType borderType = ippBorderType;
	Ipp32f borderValue[] = { 0, 0, 0 };
	pSrc1 = (Ipp32f *)pSRC;
	pDst1 = (Ipp32f *)pDST;
	check_sts(status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp32f,
		numChannels, &iSpecSize, &iTmpBufSize));
	//check is needed 
	pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
	//check is needed 
	pBuffer = ippsMalloc_8u(iTmpBufSize);
	check_sts(status = ippiFilterGaussianInit(roiSize, kernelSize, sigma,
		borderType, ipp32f, numChannels, pSpec, pBuffer));
	check_sts(status = ippiFilterGaussianBorder_32f_C3R(pSrc1, srcStep, pDst1, dstStep,
		roiSize, borderValue, pSpec, pBuffer));

	EXIT_FUNC
		ippsFree(pBuffer);
	ippsFree(pSpec);
	return (int)status;
}

void
showImageArray_Ipp8u(Ipp8u * ImageROI, int img_width, int img_height)
{
	for (int i = 0; i != img_height; i++) {
		for (int j = 0; j != img_width; j++) {
			printf(" %d \t", ImageROI[i*img_width + j]);
		};
		printf("\n");
	};
	printf("\n");
};

typedef int(*funcHandler)(void *pSRC,
	void * pDST,
	int img_width,
	int img_height,
	int numChannels,
	float sigma_,
	int kernelSize,
	int stepSize,
	int ippBorderType,
	float ippBorderValue);

funcHandler gaussianFilterTable[] = { GaussianFilterUINT8, GaussianFilterUINT16 };


int main() {
	//return (int)filter_sobel_32f_c1();

	char arr[] = { 123, 123, 123, 123, 123, 123, 123,
				  123, 123, 123, 123, 123, 123, 123,
				  123, 123, 123, 123, 123, 123, 123,
				  123, 123, 0, 0, 0, 123, 123,
				  123, 123, 0, 123, 123, 123, 123,
				  123, 123, 123, 123, 0, 123, 123,
				  123, 123, 0, 123, 0, 123, 123,
				  123, 0, 123, 123, 123, 123, 0,
				  123, 123, 123, 123, 123, 0, 0 };
	char arr_dst[7 * 9];

	char source[5 * 5] = { 0, 1, 2, 3, 4,
					  5, 6, 7, 8, 9,
					  10, 11, 12, 13, 14,
					  15, 16, 17, 18, 19,
					  20, 21, 22, 23, 24
	};
	char destination[5 * 5];
	//void * ptr = (void)arr;
	//int result = 
	int result = gaussianFilterTable[0](source,
		destination,
		5,
		5,
		1,
		1.25,
		11,
		5,
		1,
		5);

	showImageArray_Ipp8u(source, 5, 5);
	printf("result is %d \n", result);
	showImageArray_Ipp8u(destination, 5, 5);
	return 0;
};
