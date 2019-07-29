cimport numpy as cnp
import numpy as np
cimport cython

from skimage.color import guess_spatial_dimensions

cnp.import_array()

cdef extern from "ippfuncswrappers.c":
	int ippConvertUINT8toFLOAT32(void *pSRC, void * pDST, int img_width, int img_height);

	int GaussianFilterUINT8(void *pSRC, void * pDST, int img_width, int img_height, int numChannels, 
							float sigma_, int kernelSize, int stepSize, int ippBorderType, float ippBorderValue);
	int GaussianFilterUINT8RGB(void *pSRC, void * pDST, int img_width, int img_height, 
							int numChannels, float sigma_, int kernelSize, int stepSize,int ippBorderType, float ippBorderValue);
	int GaussianFilterUINT16(void *pSRC, void * pDST, int img_width, int img_height, 
							int numChannels, float sigma_, int kernelSize, int stepSize,int ippBorderType, float ippBorderValue);
	int GaussianFilterUINT16RGB(void *pSRC, void * pDST, int img_width, int img_height, 
							int numChannels, float sigma_, int kernelSize, int stepSize,int ippBorderType, float ippBorderValue);
	int GaussianFilterINT16(void *pSRC, void * pDST, int img_width, int img_height, 
							int numChannels, float sigma_, int kernelSize, int stepSize,int ippBorderType, float ippBorderValue);
	int GaussianFilterINT16RGB(void *pSRC, void * pDST, int img_width, int img_height, 
							int numChannels, float sigma_, int kernelSize, int stepSize,int ippBorderType, float ippBorderValue);
	int GaussianFilterFLOAT32(void *pSRC, void * pDST, int img_width, int img_height, 
							int numChannels, float sigma_, int kernelSize, int stepSize,int ippBorderType, float ippBorderValue);
	int GaussianFilterFLOAT32RGB(void *pSRC, void * pDST, int img_width, int img_height, 
							int numChannels, float sigma_, int kernelSize, int stepSize,int ippBorderType, float ippBorderValue);

__all__ = ['gaussian']

cdef extern from "ippbase.h":
	ctypedef unsigned char  Ipp8u;
	ctypedef unsigned short Ipp16u;
	ctypedef unsigned int   Ipp32u;
	ctypedef signed char    Ipp8s;
	ctypedef signed short   Ipp16s;
	ctypedef signed int     Ipp32s;
	ctypedef float          Ipp32f;
	#ctypedef IPP_INT64        Ipp64s;
	#ctypedef IPP_UINT64       Ipp64u;
	ctypedef double			Ipp64f;


#from _ni_support.py scipy/ndimage/_ni_support.py :_extend_mode_to_code
cdef int _getIppBorderType(mode):
	"""Convert an extension mode to the corresponding integer code.
	"""
	if mode == 'nearest':
		return 1
	elif mode == 'wrap':
		return 2
	elif mode == 'mirror':
		return 3
	elif mode == 'reflect':
		return 4
	elif mode == 'constant':
		return 6
	else:
		raise RuntimeError('boundary mode not supported')

#from _ni_support.py scipy/ndimage/_ni_support.py
def _get_output(output, input, shape=None):
	if shape is None:
		shape = input.shape

	if output is None:
		output = np.zeros(shape, dtype=input.dtype.name)
	elif type(output) in [type(type), type(np.zeros((4,)).dtype)]:
		output = np.zeros(shape, dtype=output)
	elif output.shape != shape:
		raise RuntimeError("output shape not correct")
	return output

# maybe cdef needed
# needed more correct version (guest_spatial_dim skimage)
cdef int _get_number_of_channels(image):
	if image.ndim == 2:
		channels = 1 #single (grayscale)
	elif image.ndim == 3:
		channels = image.shape[-1] #RGB 
	else:
		raise ValueError('invalid axis')
	return channels

def gaussian(cnp.ndarray image, float sigma=1.0, output=None, mode='nearest', 
				cval=0, multichannel=None, preserve_range=False, float truncate=4.0):
	image = np.asarray(image)
	if not image.flags.f_contiguous:
		image = np.ascontiguousarray(image)
	destination = _get_output(output, image)

	cdef float sd = float(sigma)
	cdef float tr = float(truncate)
	cdef float ippBorderValue = float(cval)
	# make the radius of the filter equal to truncate standard deviations
	# as is in SciPy
	kernelSize = int(tr * sd + 0.5) * 2 - 1
	cdef void* cyimage
	cdef void* cydestination

	# needed more correct way. Warning: conversion from 'npy_intp' to 'int', possible loss of data
	cdef int img_width = image.shape[0]
	cdef int img_height = image.shape[1]
	cdef int stepsize = image.strides[0]

	cdef int numChannels = _get_number_of_channels(image)
	cdef int ippBorderType = _getIppBorderType(mode)


	"""
	## numpy convertations has some bug
	if(numChannels == 1):
		if(destination.dtype == np.uint8):
			if not image.dtype == destination.dtype:
				image = image.astype(dtype = destination.dtype.name)
			print(image.dtype)
			print(destination.dtype)
			cyimage = <void*> cnp.PyArray_DATA(image)
			cydestination = <void*> cnp.PyArray_DATA(destination)
			GaussianFilterUINT8(cyimage, cydestination, img_width, img_height, 
									numChannels, sigma, kernelSize, stepsize, ippBorderType, ippBorderValue)
		elif(destination.dtype == np.uint16):
			if not image.dtype == destination.dtype:
				image = np.asarray(image, dtype = destination.dtype)
			cyimage = <void*> cnp.PyArray_DATA(image)
			cydestination = <void*> cnp.PyArray_DATA(destination)
			GaussianFilterUINT16(cyimage, cydestination, img_width, img_height, 
									numChannels, sigma, kernelSize, stepsize, ippBorderType, ippBorderValue)
		elif(destination.dtype == np.int16):
			if not image.dtype == destination.dtype:
				image = np.asarray(image, dtype = destination.dtype)
			cyimage = <void*> cnp.PyArray_DATA(image)
			cydestination = <void*> cnp.PyArray_DATA(destination)
			GaussianFilterINT16(cyimage, cydestination, img_width, img_height, 
									numChannels, sigma, kernelSize, stepsize, ippBorderType, ippBorderValue)
		elif(destination.dtype == np.float32):
			if not image.dtype == destination.dtype:
				image = np.asarray(image, dtype = destination.dtype)
			cyimage = <void*> cnp.PyArray_DATA(image)
			cydestination = <void*> cnp.PyArray_DATA(destination)
			GaussianFilterFLOAT32(cyimage, cydestination, img_width, img_height, 
									numChannels, sigma, kernelSize, stepsize, ippBorderType, ippBorderValue)
		# other dtypes for one channel
		elif(destination.dtype == np.int8): 
			print("case is this")
			if not image.dtype == np.float32:
				image = np.asarray(image, dtype = np.float32)
			print(image)
			print("output dtype is ", destination.dtype)
			destination = np.asarray(destination, dtype = np.float32)
			cyimage = <void*> cnp.PyArray_DATA(image)
			cydestination = <void*> cnp.PyArray_DATA(destination)
			GaussianFilterFLOAT32(cyimage, cydestination, img_width, img_height, 
									numChannels, sigma, kernelSize, stepsize, ippBorderType, ippBorderValue)
			destination = np.asarray(destination, dtype=np.int8)
		else:
			raise ValueError("Currently not supported")
	elif(numChannels == 3):
		if(destination.dtype == np.uint8):
			if not image.dtype == destination.dtype:
				image = np.asarray(image, dtype = destination.dtype)
			cyimage = <void*> cnp.PyArray_DATA(image)
			cydestination = <void*> cnp.PyArray_DATA(destination)
			GaussianFilterUINT8RGB(cyimage, cydestination, img_width, img_height, 
									numChannels, sigma, kernelSize, stepsize, ippBorderType, ippBorderValue)
		else:
			raise ValueError("Currently not supported")
	else:
		raise ValueError("Currently not supported")
	return destination
	"""


	
	cyimage = <void*> cnp.PyArray_DATA(image)
	cydestination = <void*> cnp.PyArray_DATA(destination)

	
	if(numChannels == 1):
		if(image.dtype == np.uint8):
			GaussianFilterUINT8(cyimage, cydestination, img_width, img_height, 
									numChannels, sigma, kernelSize, stepsize, ippBorderType, ippBorderValue)
		elif(image.dtype == np.float32):
			GaussianFilterFLOAT32(cyimage, cydestination, img_width, img_height, 
									numChannels, sigma, kernelSize, stepsize, ippBorderType, ippBorderValue)
		elif(image.dtype == np.uint16):
			GaussianFilterUINT16(cyimage, cydestination, img_width, img_height,
									numChannels, sigma, kernelSize, stepsize, ippBorderType, ippBorderValue)
		elif(image.dtype == np.int16):
			GaussianFilterINT16(cyimage, cydestination, img_width, img_height, 
									numChannels, sigma, kernelSize, stepsize, ippBorderType, ippBorderValue)
		else:
			raise ValueError("Currently not supported")
	elif(numChannels == 3):
		if(image.dtype == np.uint8):
			GaussianFilterUINT8RGB(cyimage, cydestination, img_width, img_height,
									numChannels, sigma, kernelSize, stepsize, ippBorderType, ippBorderValue)
		elif(image.dtype == np.float32):
			GaussianFilterFLOAT32RGB(cyimage, cydestination, img_width, img_height, 
									numChannels, sigma, kernelSize, stepsize, ippBorderType, ippBorderValue)
		elif(image.dtype == np.uint16):
			GaussianFilterUINT16RGB(cyimage, cydestination, img_width, img_height, 
									numChannels, sigma, kernelSize, stepsize, ippBorderType, ippBorderValue)
		elif(image.dtype == np.int16):
			GaussianFilterINT16(cyimage, cydestination, img_width, img_height,
									numChannels, sigma, kernelSize, stepsize, ippBorderType, ippBorderValue)
		else:
			raise ValueError("Currently not supported")
	else:
		raise ValueError("Currently not supported")
	return destination
	
	
