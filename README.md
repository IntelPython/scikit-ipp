# Scikit-IPP (skipp)
__scikit-ipp__ is optimization of open-source image processing library [scikit-image](https://scikit-image.org/) by using Intel® Integrated Performance Primitives (Intel® IPP) library.

### Modules:
* __Filters:__
  + Gaussian
  + Sobel (sobel, sobel_h, sobel_v, sobel_c)
  + Prewitt (prewitt, prewitt_h, prewitt_v)
  + Laplace
  + Median
* __Morpholgy:__
  + Erosion
  + Dilation
* __Transform:__
  + Rotate

### Building `scikit-ipp`
````
cd <checkout-dir>
conda build -c intel conda-recipe
````

This will build the conda package and tell you where to find it (```.../scikit-ipp*.tar.bz2```).

## Installing the built scikit-ipp conda package
```
conda install <path-to-conda-package-as-built-above>
```
## Examples
Introductory examples for scikit-ipp [link](https://github.intel.com/SAT/scikit-ipp/blob/master/examples/scikit-ipp_examples.ipynb)
