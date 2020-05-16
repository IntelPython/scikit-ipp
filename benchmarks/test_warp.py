import pytest
import numpy as np

from skimage.transform import warp as skimage_warp
from skipp.transform import warp as skipp_warp

from skipp.transform import AffineTransform as skp_AffineTransform
from skimage.transform import AffineTransform as skm_AffineTransform



def get_image_data(image_dtype, shape=(300, 400)):
    image = np.arange(shape[0] * shape[1], dtype=image_dtype).reshape(shape)
    return image


@pytest.mark.parametrize("input_dtype", [np.float32])
@pytest.mark.parametrize("shape", [pytest.param((8000, 8000), id="800x800")
                                   ])
@pytest.mark.parametrize("function",[skipp_warp], ids=["skipp_warp"])
@pytest.mark.parametrize("preserve_range", [True])
@pytest.mark.parametrize("order", [1])
def test_warp_skipp(benchmark, function, shape, input_dtype, preserve_range, order):
    """
    # Speed test, that measures the speed of scikit-ipp's and scikit-image's
      transform.rotate funcs
    """
    image = get_image_data(input_dtype, shape)
    mat = np.array([[ 0.70710678,  -0.70710678,  0.        ],
                    [ 0.70710678,   0.70710678,  0.        ],
                    [ 0.        ,   0.        ,  1.        ]], dtype=np.double)

    transf_skp = skp_AffineTransform(matrix=mat)
    result = benchmark.pedantic(target=function, args=(image, transf_skp.params), kwargs={'preserve_range': preserve_range, 'order':order},rounds=10, iterations=25)
    #result = skipp.transform.warp(image, transf_skp.params, order=order, preserve_range=preserve_range)


@pytest.mark.parametrize("input_dtype", [np.float32])
@pytest.mark.parametrize("shape", [pytest.param((8000, 8000), id="800x800")
                                   ])
@pytest.mark.parametrize("function",[skimage_warp], ids=["skimage_warp"])
@pytest.mark.parametrize("preserve_range", [True])
@pytest.mark.parametrize("order", [1])
def test_warp_skimage(benchmark, function, shape, input_dtype, preserve_range, order):
    """
    # Speed test, that measures the speed of scikit-ipp's and scikit-image's
      transform.rotate funcs
    """
    image = get_image_data(input_dtype, shape)
    mat = np.array([[ 0.70710678,  0.70710678,  0.        ],
                    [-0.70710678,  0.70710678,  0.        ],
                    [ 0.        ,  0.        ,  1.        ]], dtype=np.double)
    
    transf_skm = skm_AffineTransform(matrix=mat)
    result = benchmark.pedantic(target=function, args=(image, transf_skm.params), kwargs={'preserve_range': preserve_range, 'order':order},rounds=10, iterations=25)