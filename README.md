# scikit-IPP (skipp)
`scikit-ipp` is optimization of open-source image processing library [scikit-image](https://scikit-image.org/) by using Intel® Integrated Performance Primitives (Intel® IPP) library.

`scikit-ipp` is a standalone package, provided scikit-image-like API to some of Intel® IPP functions.

- [Documentation](https://intelpython.github.io/scikit-ipp/)
- [Source Code](https://github.com/IntelPython/scikit-ipp)
- [About Intel® IPP](https://software.intel.com/en-us/intel-ipp)


# Getting started
`scikit-ipp` is easily built from source with the majority of the necessary prerequisites available on conda.  The instructions below detail how to gather the prerequisites, setting one's build environment, and finally building and installing the completed package.  `scikit-ipp` can be built for all three major platforms (Windows, Linux, macOS).

The build-process (using setup.py) happens in 2 stages:
1. Running cython on C and Cython sources
2. Compiling and linking


# Building scikit-ipp using conda-build
The easiest way to build `scikit-ipp` is using the conda-build with the provided recipe.

## Prerequisites
* Python version >= 3.6
* conda-build version >= 3
* C compiler

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
To actually use your `scikit-ipp`, dependent packages need to be installed. To ensure, do

Linux or Windows:
```
conda install -c intel numpy ipp
```
# Building documentation for scikit-ipp
## Prerequisites for creating documentation
* sphinx >= 3.0
* sphinx_rtd_theme >= 0.4
* sphinx-gallery >= 0.3.1
* matplotlib > = 3.0.1

## Building documentation
1. Install scikit-ipp into your python environment
2. ```cd doc && make html```
3. The documentation will be in ```doc/_build/html```

# Examples
Introductory examples for `scikit-ipp` [link](examples/scikit-ipp_examples.ipynb)
