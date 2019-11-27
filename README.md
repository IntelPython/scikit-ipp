# Scikit-IPP (skipp)
__scikit-ipp__ is optimization of open-source image processing library [scikit-image](https://scikit-image.org/) by using Intel® Integrated Performance Primitives (Intel® IPP) library.

* Filters:
  + Gaussian
  + Sobel (sobel, sobel_h, sobel_v, sobel_c)
  + Prewitt (prewitt, prewitt_h, prewitt_v)
  + Laplace
  + Median

### Building scikit-ipp
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
Introductory examples for scikit-ipp [link](https://github.intel.com/SAT/scikit-ipp/blob/features/gaussian_filter/examples/scikit-ipp_examples.ipynb)