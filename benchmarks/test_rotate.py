import pytest
import numpy as np

from skimage.transform import rotate as skimage_rotate
from skipp.transform import rotate as skipp_rotate


def get_image_data(image_dtype, shape=(300, 400)):
    image = np.arange(shape[0] * shape[1], dtype=image_dtype).reshape(shape)
    return image

# preserve_range = True
@pytest.mark.parametrize("input_dtype", [np.float32])
@pytest.mark.parametrize("shape", [pytest.param((8000, 8000), id="8000x8000")
                                   ])
@pytest.mark.parametrize("function",[skipp_rotate, skimage_rotate],
                                    ids=["skipp_rotate", "skimage_rotate"])
@pytest.mark.parametrize("preserve_range", [True])
@pytest.mark.parametrize("angle", [30])
@pytest.mark.parametrize("order", [0, 1])
def test_rotate(benchmark, function, input_dtype, shape, angle, preserve_range, order):
    """
    # Speed test, that measures the speed of scikit-ipp's and scikit-image's
      transform.rotate funcs
    """
    image = get_image_data(input_dtype, shape)
    result = benchmark.pedantic(target=function, args=(image, angle), rounds=10, iterations=25)
    #result = benchmark.pedantic(target=function, args=(image, angle), kwargs={'preserve_range': preserve_range, 'order':order}, rounds=10, iterations=10)
