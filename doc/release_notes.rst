Release notes
=============


Version 1.2.0
-------------


New features
^^^^^^^^^^^^
added option to disable OpenMP (#4)

update becnhmarks (#142)

`transform.resize` threaded version (#143)

example for affine warp transform (#151)

`transform.warp` threaded version (#139)

Multi-threading with OpenMP for some statements in `own_FilterEdge`  (#2)

re-implement `_mask_filter_results` (#3)


Commits notes
^^^^^^^^^^^^^

Multi-threading with OpenMP for some statements in `own_FilterEdge`  (#2)

* Multi-threading with OpenMP for some statements in own_FilterEdge

* added USE_OPENMP directive

* Multi-threading with OpenMP for some statements in own_FilterEdge function, that is a backend for `scikit-ipp` edge filters

* added external multi-threading with OpenMP for some part of code


re-implement `_mask_filter_results` (#3)

* implemented native `own_mask_filter_results`

* template `ippiSet_<mode>_C1R`: added `ippiSet_C1R adpater`


added option to disable OpenMP (#4)

* added option to disable OpenMP

* added using USE_OPENMP env variable

* disabled using OpenMP for osx


Create CODE_OF_CONDUCT.md


update becnhmarks (#142)

* added test_benchmarks.py

* corrected test suits inputs and settings

* updated benchmarks/readme.md:


update own_resize.h (#159)

added missed include


refactor: own_resize.c (#156)

* removed extra statement


update `own_warp.c` and `own_warp.h` (#149)

* update own_warp.h/.c

* removed unused own_RotateCoeffs, own_RotateCoeffs

* renamed number_of_threads to max_num_threads

* added setting max_num_threads =1 for small images

* added case when num_threads == 1, that doesn't uses openmp for initializing single thread

* removed unused variable srcOffset


example for affine warp transform (#151)

* added example for affine warp transform: plot_geometric.py


update transform.pyx (#150)

* implemented  `__rer__` and `__str__` for AffineTransform


`transform.resize` threaded version (#143)

* added parallel verision of own_Resize

* update resize func

* added `ippiResizeGetSrcRoi_<mod>`

* added `ippiResizeGetBorderSize_<mod>`

* update `own_Resize`

* refactor own_resize.h /own_resize.c

* update docstrings

* Update meta.yaml

* using `num_threads` instdead of `omp_set_num_threads`

* added case for when number of threads is 1


fix `own_Warp` threads statuses init (#145)

* fix warp thread status init


update setup.py (#147)

fixed ignoring flags for openmp [win]


bump up numpy version (#146)


Update meta.yaml (#144)

* fix buildnumber setting


feat: `transform.warp` threaded version (#139)

* added `openmp` dependecy

* update `warp` and `rotate` backend funcs

* implemented multithreaded `transfrom.warp`

* update `own_warp.c`

* update test suits

* update _transform.pxd

* update docsting for


update version (#140)

incremented to dev version 1.2.0dev


removed not used dependecy (#138)

* removed `scipy` and `scikit-image` since they are not used in tests


Version 1.1.0
-------------

API changes
^^^^^^^^^^^

* extend `skipp.filters.gaussian` filter supported dtypes(uint8, uint16, int16), update docstring (#119)

* extend `skipp.filters.laplace` filter supported dtypes, more supported dtypes(uint8, uint16, int16) (#118)

* extend `skipp.transform.resize` interpolation methods support [Lanczos and Super] (#135)

* extend `skipp.transform.resize` function params, added `num_lobes` param, that is used in conjunction with `order=6` (Lanczos) (#135)



Bug fixes
^^^^^^^^^

* `skipp.filters.median` filter bug with default mode of `selem=None` (#117)

* `skipp.transform.resize` function bug with cubic interpolation and update test suits (#128)

* `skipp.transform.resize` problems with resized image shape (#130)

* `skipp.transform.rotate` problems related with resized images center (#130)

* `skipp.transfrom.warp` bug for all provided non-default value of `mode` param (#130), (#123)



New features
^^^^^^^^^^^^

* added/update docstrings for all methods and structures (#108) and pep8 align docstrings - numpy style (#110). Also (#122), (#126), (#130)

* update README.md (#111), (#109): new modules were added.

* created `scikit-ipp` documentation by using sphinx (#89)

* extend laplace filter supported dtypes (#118)

* package was checked for MacOS and was added MacOS support in documentation (#121)

* extended setup metadata (#120)

* update gaussian filter test suits (#131)

* re-implement edge filters tests (#132)

* added `inverse`, `__add__` methods to `AffineTransform` class (#130)

* re-implemented `transform.rotate` function (#130)

* extend `skipp.transform.resize` interpolation methods support [Lanczos and Super] (#135)

* update processing mode for `transform` funcs - make them as is in scikit-image (`numpy.pad`). Created `__get_numpy_pad_IppBorderType` for given boundary mode processing (#123)



Refactoring
^^^^^^^^^^^
* fix legal names [dtypes.c/.h] (#112)

* separate similarity tests from scikit-ipp own functional/unit tests (#113)

* removed skimage dependence from skipp own tests and re-implemented test suits without skimage use (#113)

* removed outdated, unused src file `own_morphology_tl.c` (#133)



Commits notes
^^^^^^^^^^^^^

update README.md (#109)

* Getting started

* Prerequisites

* Building scikit-ipp using conda-build

* Building documentation for scikit-ipp



update docstrings (#108)

docstrings for:

* filters: gaussian, median, laplace, prewitt, prewitt_h, prewitt_v, sobel, sobel_v, sobel_h

* morphology: dilation, erosion

* transfrom: warp, rotate, resize, AffineTransform

Docstrings for all methods were added/updated.   



update README.md (#111)

* added documentations web link



fix legal names [dtypes.c/.h] (#112)



refactor scikit-ipp own tests (#113)

* separate similarity tests from scikit-ipp own functional/unit tests

* removed skimage dependence from skipp own tests and re-implemented test suits without skimage use



pep8 align docstrings - numpy style (#110)

* pep8 align docstrings morphology funcs - numpy style

* pep8 align docstrings filters funcs - numpy style

* pep8 align docstrings tranform funcs - numpy style



scikit-ipp docs initial (#89)

creating `scikit-ipp` documentation by using sphinx

* updated main README.md

* added configuration conf.py

* Added: Makefile, make.bat, release_notes.rst, index.rst, installing.rst, license.rst, contents.rst, contribute.rst, api.rst, examples.rst

* some misc. updates



fix median filter (#117)

* fix `median` filter bug with default mode of selem=None



extend laplace filter supported dtypes (#118)

* extended `laplace` filter supported dtypes: uint8, uint16, int16 and float32 [was only float32]

* added `test_laplace_preserve_dtype` test suit

* update `laplace` filter docstring



update gaussian filter docstring (#119)

* update gaussian filter docstring - added supported dtypes, removed outdated notes



added MacOS support in documentation (#121)

* added MacOS support in documentation



update setup metadata (#120)

* update setup.py

* extended metadata



update Gaussian filter docstrings (#122)

* correct supported modes list



update processing boundary mode (#123)

* update processing mode for `transform` funcs - make them as is in scikit-image (`numpy.pad`)

* created `__get_numpy_pad_IppBorderType` for given boundary mode processing

* update docstrings for `__get_IppBorderType`

* update docstrings for transform functions: update info about supported modes

* test suits `test_transform` were updated - all checks passed



update resize func docstrings (#126)



fixed `skipp.transform.resize` function with cubic interpolation and update test suits (#128)

* fix `transfrom.resize` function when interpolation method is cubic

* rewrite test suits for `transform.resize`

  - added `test_resize2d` test suit

  - added parameterized `test_resize_without_antialiasing` and `test_resize_with_antialiasing` test suits



update gaussian filter test suits (#131)

* update and enabled `test_gaussian_preserve_dtype` test suit

* removed outdated test suit `test_gaussian_preserve_output`



re-implement edge filters tests (#132)

* re-implement `test_sobel.py` and `test_prewitt.py`



update transform module (#130)

* update `AffineTransform` class

  + added `inverse` method

  + added test suit `test_AffineTransform_inverse`

  + implemented `__add__` method for AffineTransform

  + added `test_invalid_input` test suit

  + added `test_affine_init` test suit

* update `transform.warp` transform func

* added `test_warp_matrix` and `test_warp_tform` test suits

* enabled `test_rotate`, `test_rotate_resize`, `test_rotate_center`, `test_rotate_resize_center`, `test_rotate_resize_90` test suits

* re-implemented `transform.rotate` function

* update `transform.rotate` function docstrings

* removed unused `own_RotateCoeffs` and `own_GetAffineDstSize` from `tranform.pxd`



refactor: removed unused own_morphology_tl.c (#133)

* removed outdated, unused src file `own_morphology_tl.c`



extend `skipp.transform.resize` interpolation methods support [Lanczos and Super] (#135)

* added `Lanczos`, `Super` interpolation method to `__get_IppiInterpolationType` function

* update `transform.resize` function

* added `ippiResizeLanczos`, that is adapter for `ippiResizeLanczos_<mode>` funcs

* added `ippiResizeLanczosInit`, that is adapter for `ippiResizeLanczosInit_<mode>` funcs

* added `ippiResizeSuper`, that is adapter for `ippiResizeSuper_<mode>` funcs

* added `ippiResizeSuperInit`, that is adapter for `ippiResizeSuperInit_<mode>` funcs

* updated `ippiResize` function

* updated `own_Resize` function

* update `transform.resize` function:

  + update docstrings for `transform.resize` function

  + extended `transform.resize function params, added `num_lobes` param, that is used in conjunction with `order=6` (Lanczos)

* added tests suits for check:

  + added `test_resize_super` for checking `transform.resize` with `super` interpolation method

  + update parameterized `test_resize_without_antialiasing` test suit for checking `transform.resize` with `Lanczos` interpolation method


Version 1.0.0
-------------


New Features
^^^^^^^^^^^^

Gaussian filter

* `skipp.filters.gaussian`

* This function uses Intel(R) Integrated Performance Primitives (Intel(R) IPP) funcs: ippiFilterGaussianBorder_<mod> on the backend, that performs Gaussian filtering of an image with user-defined borders, see: `FilterGaussianBorder` on https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/


Median filter

* `skipp.filters.median`

* This function uses Intel(R) Integrated Performance Primitives (Intel(R) IPP) funcs: ippiFilterMedianBorder_<mod> on the backend, that performs median filtering of an image with user-defined borders, see: `FilterMedianBorder` on https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/


Laplace filter

* `skipp.filters.laplace`. Find the edges of an image using the Laplace operator.

* This function uses Intel(R) Integrated Performance Primitives (Intel(R) IPP) funcs: ippiFilterBorder_<mod> on the backend, that filters an image using a rectangular filter with coeffs (Laplace (3x3)) [[0 -1 0], [-1 4 -1], [0 -1 0]] for implementing laplace filtering as is in `scikit-image`, see: `FilterBorder` on https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/


Sobel filter

* `skipp.filters.sobel`. Find edges in an image using the Sobel filter. 

* This function uses Intel(R) Integrated Performance Primitives (Intel(R) IPP) funcs: ippiFilterSobel_<mod> on the backend, see: `FilterSobel` on https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/


Horizontal Sobel filter

* `skipp.filters.sobel_h`. Find the horizontal edges of an image using the Sobel transform.

* This function uses Intel(R) Integrated Performance Primitives (Intel(R) IPP) funcs: ippiFilterSobelHorizBorder_<mod> on the backend, see: `FilterSobelHorizBorder` on https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/


Vertical Sobel filter

* `skipp.filters.sobel_v`. Find the vertical edges of an image using the Sobel transform. 

* This function uses Intel(R) Integrated Performance Primitives (Intel(R) IPP) funcs: ippiFilterSobelVertBorder_<mod> on the backend, see: `FilterSobelVertBorder` on https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/


Prewitt filter

* `skipp.filters.prewitt`. Find the edge magnitude using the Prewitt transform.

* This function uses Intel(R) Integrated Performance Primitives (Intel(R) IPP) funcs: ippiFilterPrewittVertBorder_<mod> and ippiFilterPrewittHorizBorder_<mod> on the backend see: `FilterPrewittHorizBorder`, `FilterPrewittVertBorder` https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/


Horizontal Prewitt filter

* `skipp.filters.prewitt_h`. Find the horizontal edges of an image using the Prewitt transform.

* This function uses Intel(R) Integrated Performance Primitives (Intel(R) IPP) funcs: ippiFilterPrewittHorizBorder_<mod> on the backend see: `FilterPrewittHorizBorder` https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/


Vertical Prewitt filter

* `skipp.filters.prewitt_v`. Find the vertical edges of an image using the Prewitt transform.

* This function uses Intel(R) Integrated Performance Primitives (Intel(R) IPP) funcs: ippiFilterPrewittVertBorder_<mod> on the backend see: `FilterPrewittVertBorder` https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/


Morphological dilation

* `skipp.morphology.dilation`. Morphological dilation sets a pixel at (i,j) to the maximum over all pixels in the neighborhood centered at (i,j). Dilation enlarges bright regions and shrinks dark regions.

* This function uses Intel(R) Integrated Performance Primitives (Intel(R) IPP) funcs: ippiDilateBorder_<mod> on the backend, that performs dilation of an image, see: `DilateBorder` on https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/


Morphological erosion

* `skipp.morphology.erosion`. Return greyscale morphological erosion of an image. Morphological erosion sets a pixel at (i,j) to the minimum over all pixels in the neighborhood centered at (i,j). Erosion shrinks bright regions and enlarges dark regions.

* This function uses Intel(R) Integrated Performance Primitives (Intel(R) IPP) funcs: ippiErodeBorder_<mod> on the backend, that performs dilation of an image, see: `ErodeBorder` on https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/


2D affine transformation object

* `skipp.transform.AffineTransform` class. Contains homogeneous transformation matrix.


Image warping

* `skipp.transform.warp`. Warp an image according to a given coordinate transformation.

* This function uses Intel(R) Integrated Performance Primitives (Intel(R) IPP) funcs: ippiWarpAffineLinear_<mod>,  ippiWarpAffineNearest_<mod> and ippiWarpAffineCubic_<mod> on the backend, that performs warp affine transformation of an image using the linear, nearest neighbor or cubic interpolation method, see: `WarpAffineLinear`, `WarpAffineCubic`, `WarpAffineNearest` on https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/


Image rotation

* `skipp.transform.rotate`. Rotate image by a certain angle around its center.

* This function uses `skipp.transform.warp` on the backend, and `skipp.transform.warp` in turn uses Intel(R) Integrated Performance Primitives (Intel(R) IPP) funcs: ippiWarpAffineLinear_<mod>,  ippiWarpAffineNearest_<mod> and ippiWarpAffineCubic_<mod> on the backend, that performs warp affine transformation of an image using the linear, nearest neighbor or cubic interpolation method, see: `WarpAffineLinear`, `WarpAffineCubic`, `WarpAffineNearest` on https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/


Image resizing

* `skipp.transform.resize`. Resize image to match a certain size.

* This function uses Intel(R) Integrated Performance Primitives (Intel(R) IPP) funcs on the backend: ippiResizeNearest_<mod>, ippiResizeLinear_<mod>, ippiResizeCubic_<mod>, ippiResizeLanczos_<mod>, ippiResizeSuper_<mod> that changes an image size using nearest neighbor, linear, cubic, Lanczos or super interpolation method, and ippiResizeAntialiasing_<mod>, that changes an image size using using the linear and cubic interpolation method with antialiasing, see: `ResizeNearest`, `ResizeLinear`, `ResizeCubic`, `ResizeLanczos`, `ResizeSuper`,`ResizeAntialiasing` on https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/
