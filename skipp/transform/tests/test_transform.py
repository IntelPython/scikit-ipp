import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_allclose,
                           assert_array_equal, assert_allclose, assert_almost_equal)
from skimage._shared.testing import (assert_equal,TestCase, parametrize, raises)

from skipp.transform import AffineTransform
from skipp.transform import resize
from skipp.transform import rotate
from skipp.transform import warp

 
@pytest.mark.skip(reason="skip: adding invers method")
def test_warp_tform():
    x = np.zeros((5, 5), dtype=np.double)
    x[2, 2] = 1
    theta = - np.pi / 2
    tform = AffineTransform(scale=(1, 1), rotation=theta,
                            translation=(0, 4))

    x90 = warp(x, tform, order=1)
    assert_almost_equal(x90, np.rot90(x))

    #x90 = skipp.transform.warp(x, tform.inverse, order=1)
    #assert_almost_equal(x90, np.rot90(x))


@pytest.mark.skip(reason="needs resize implementation")
def test_warp_matrix():
    x = np.zeros((5, 5), dtype=np.double)
    x[2, 2] = 1
    refx = np.zeros((5, 5), dtype=np.double)
    refx[1, 1] = 1

    matrix = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])

    # _warp_fast
    outx = warp(x, matrix, order=1)
    assert_allclose(outx, refx)
    # check for ndimage.map_coordinates
    #outx = warp(x, matrix, order=5)


def test_rotate():
    x = np.zeros((5, 5), dtype=np.double)
    x[1, 1] = 1
    x90 = rotate(x, angle=90)
    assert_allclose(x90, np.rot90(x), rtol=1e-06)


def test_rotate_resize():
    x = np.zeros((10, 10), dtype=np.double)

    x45 = rotate(x, 45, resize=False)
    assert x45.shape == (10, 10)

    x45 = rotate(x, 45, resize=True)
    # new dimension should be d = sqrt(2 * (10/2)^2)
    assert x45.shape == (14, 14)


def test_rotate_center():
    x = np.zeros((10, 10), dtype=np.double)
    x[4, 4] = 1
    refx = np.zeros((10, 10), dtype=np.double)
    refx[2, 5] = 1
    x20 = rotate(x, 20, order=0, center=(0, 0))
    assert_allclose(x20, refx, rtol=1e-06)
    x0 = rotate(x20, -20, order=0, center=(0, 0))
    assert_allclose(x0, x, rtol=1e-06)


@pytest.mark.skip(reason="needs resize implementation")
def test_rotate_resize_center():
    x = np.zeros((10, 10), dtype=np.double)
    x[0, 0] = 1

    ref_x45 = np.zeros((14, 14), dtype=np.double)
    ref_x45[6, 0] = 1
    ref_x45[7, 0] = 1

    x45 = rotate(x, 45, resize=True, center=(3, 3), order=0)
    # new dimension should be d = sqrt(2 * (10/2)^2)
    assert x45.shape == (14, 14)
    assert_equal(x45, ref_x45)


@pytest.mark.skip(reason="needs resize implementation")
def test_rotate_resize_90():
    x90 = rotate(np.zeros((470, 230), dtype=np.double), 90,
                 resize=True)
    assert x90.shape == (230, 470)

@pytest.mark.parametrize("order", [2, 4, 5, -1, ''])
def test_rotate_order_error(order):
    with pytest.raises(ValueError):
        rotate(np.zeros((5, 5), dtype=np.float32), angle=20,
               order=order)

# TODO
# add other ``order`` options
# Intel IPP antialiasing resize function doesnot support Nearest interpolation method
# TODO
# add order 3 to tests and float32 type
@pytest.mark.parametrize("image_dtype", [np.uint8,  np.uint16, np.int16])
#@pytest.mark.parametrize("anti_aliasing", [False])
@pytest.mark.parametrize("order", [0, 1])
def test_resize2d_without_anti_aliasing(image_dtype, order):
    x = np.zeros((5, 5), dtype=image_dtype)
    x[1, 1] = 1
    resized = resize(x, (10, 10), order=order, anti_aliasing=False,
                     mode='nearest')
    ref = np.zeros((10, 10), dtype=image_dtype)
    ref[2:4, 2:4] = 1
    assert_allclose(resized, ref, rtol=1e-06)

# TODO
# add other ``order`` options
# Intel IPP antialiasing resize function doesnot support Nearest interpolation method
@pytest.mark.parametrize("image_dtype", [np.uint8,  np.uint16, np.int16])
@pytest.mark.parametrize("order", [1])
def test_resize2d_anti_aliasing(image_dtype, order):
    x = np.zeros((5, 5), dtype=image_dtype)
    x[1, 1] = 1
    resized = resize(x, (10, 10), order=order, anti_aliasing=True,
                                           mode='nearest')
    ref = np.zeros((10, 10), dtype=image_dtype)
    ref[2:4, 2:4] = 1
    assert_allclose(resized, ref, rtol=1e-06)


@pytest.mark.parametrize("image_dtype", [np.uint32,  np.int32, np.uint64,
                         np.int64, np.float64])
def test_resize_not_supported_dtype(image_dtype):
    x = np.zeros((5, 5), dtype=image_dtype)
    with raises(RuntimeError):
        # output_shape too short
        resize(x, (10, 10), order=0, anti_aliasing=False,
               mode='nearest')


@pytest.mark.parametrize("image_shape", [(10,10,10), (10,)])
def test_resize_not_supported_ndim(image_shape):
    x = np.zeros(image_shape, dtype=np.float32)
    with raises(ValueError):
        # output_shape too short
        resize(x, (10, 10), order=0, anti_aliasing=False,
               mode='nearest')
