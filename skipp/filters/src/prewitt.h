#ifndef PREWITT_H
#define PREWITT_H
#include "ipp.h"
#include "dtypes.h"

int
ippiFilterPrewittHorizBorder(
	IppDataTypeIndex input_index,
	IppDataTypeIndex output_index,
	void * pInput,
	void * pOutput,
	int img_width,
	int img_height,
	int numChannels,
	IppiMaskSize mask,
	IppiBorderType ippBorderType,
	float ippBorderValue);

int
ippiFilterPrewittVertBorder(
	IppDataTypeIndex input_index,
	IppDataTypeIndex output_index,
	void * pInput,
	void * pOutput,
	int img_width,
	int img_height,
	int numChannels,
	IppiMaskSize mask,
	IppiBorderType ippBorderType,
	float ippBorderValue);

int
FilterPrewittHoriz(
	IppDataTypeIndex input_index,
	IppDataTypeIndex output_index,
	void * pInput,
	void * pOutput,
	int img_width,
	int img_height,
	int numChannels);

int
FilterPrewittVert(
	IppDataTypeIndex input_index,
	IppDataTypeIndex output_index,
	void * pInput,
	void * pOutput,
	int img_width,
	int img_height,
	int numChannels);
#endif