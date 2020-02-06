import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_allclose,
                           assert_array_equal, assert_allclose, assert_almost_equal)
from skimage._shared.testing import (assert_equal,TestCase, parametrize)

import skipp.skipp.transform


def test_rotate():
    x = np.zeros((5, 5), dtype=np.double)
    x[1, 1] = 1
    x90 = skipp.skipp.transform.rotate(x, angle=90)
    assert_allclose(x90, np.rot90(x), rtol=1e-06)


@pytest.mark.skip(reason="needs resize implementation")
def test_rotate_resize():
    x = np.zeros((10, 10), dtype=np.double)

    x45 = skipp.skipp.transform.rotate(x, 45, resize=False)
    assert x45.shape == (10, 10)

    x45 = skipp.skipp.transform.rotate(x, 45, resize=True)
    # new dimension should be d = sqrt(2 * (10/2)^2)
    assert x45.shape == (14, 14)


def test_rotate_center():
    x = np.zeros((10, 10), dtype=np.double)
    x[4, 4] = 1
    refx = np.zeros((10, 10), dtype=np.double)
    refx[2, 5] = 1
    x20 = skipp.skipp.transform.rotate(x, 20, order=0, center=(0, 0))
    assert_allclose(x20, refx, rtol=1e-06)
    x0 = skipp.skipp.transform.rotate(x20, -20, order=0, center=(0, 0))
    assert_allclose(x0, x, rtol=1e-06)


@pytest.mark.skip(reason="needs resize implementation")
def test_rotate_resize_center():
    x = np.zeros((10, 10), dtype=np.double)
    x[0, 0] = 1

    ref_x45 = np.zeros((14, 14), dtype=np.double)
    ref_x45[6, 0] = 1
    ref_x45[7, 0] = 1

    x45 = skipp.skipp.transform.rotate(x, 45, resize=True, center=(3, 3), order=0)
    # new dimension should be d = sqrt(2 * (10/2)^2)
    assert x45.shape == (14, 14)
    assert_equal(x45, ref_x45)


@pytest.mark.skip(reason="needs resize implementation")
def test_rotate_resize_90():
    x90 = skipp.skipp.transform.rotate(np.zeros((470, 230), dtype=np.double), 90,
                                       resize=True)
    assert x90.shape == (230, 470)

@pytest.mark.parametrize("order", [2, 4, 5, -1, ''])
def test_rotate_order_error(order):
    with pytest.raises(ValueError):
        skipp.skipp.transform.rotate(np.zeros((5, 5), dtype=np.float32), angle=20,
                                     order=order)
