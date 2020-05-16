import pytest
import numpy as np

from skimage.morphology import erosion as skimage_erosion
from skimage.morphology import dilation as skimage_dilation
from skipp.morphology import erosion as skipp_erosion
from skipp.morphology import dilation as skipp_dilation
# import scipy.ndimage.filters

import skipp
import skimage

def get_image_data(image_dtype, shape=(300, 400)):
    image = np.arange(shape[0] * shape[1], dtype=image_dtype).reshape(shape)
    return image

@pytest.mark.parametrize("input_dtype", [np.uint16, np.float32])
@pytest.mark.parametrize("shape", [pytest.param((8000, 8000), id="8000x8000")
                                   ])
@pytest.mark.parametrize("function",[skipp_erosion, skimage_erosion, skipp_dilation, skimage_dilation],
                                    ids=["skipp_erosion", "skimage_erosion", "skipp_dilation", "skimage_dilation"])

@pytest.mark.parametrize("selem", [pytest.param(None, id="default_selem")])
def test_morphology(benchmark, function, input_dtype, shape, selem):
    """
    # Speed test, that measures the speed of scikit-ipp's and scikit-image's morphology funcs
    """
    image = get_image_data(input_dtype, shape)
    result = benchmark.pedantic(target=function, args=(image, selem),
                                rounds=10, iterations=10)
