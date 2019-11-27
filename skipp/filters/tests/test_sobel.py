import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_allclose)

from skipp.skipp.filters import sobel as skipp_sobel
from skipp.skipp.filters import sobel_h as skipp_sobel_h
from skipp.skipp.filters import sobel_v as skipp_sobel_v

from skimage.filters import sobel as skimage_sobel
from skimage.filters import sobel_h as skimage_sobel_h
from skimage.filters import sobel_v as skimage_sobel_h


def test_sobel_zeros():
    """Sobel on an array of all zeros."""
    result = skipp_sobel(np.zeros((10, 10), dtype=np.float32))
    assert(np.all(result == 0))


@pytest.mark.skip(reason="deveoloping _mask_filter_result in progress")
def test_sobel_mask():
    """Sobel on a masked array should be zero."""
    result = filters.sobel(np.random.uniform(size=(10, 10)),
                           np.zeros((10, 10), dtype=bool))
    assert(np.all(result == 0))


def test_sobel_horizontal():
    """Sobel on a horizontal edge should be a horizontal line."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(np.float32)
    result = skipp_sobel(image) * np.sqrt(2)
    # Check if result match transform direction
    i[np.abs(j) == 5] = 10000
    assert_allclose(result[i == 0], 1)
    assert (np.all(result[np.abs(i) > 1] == 0))


def test_sobel_vertical():
    """Sobel on a vertical edge should be a vertical line."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(np.float32)
    result = skipp_sobel(image) * np.sqrt(2)
    j[np.abs(i) == 5] = 10000
    assert_allclose(result[j == 0], 1)
    assert (np.all(result[np.abs(j) > 1] == 0))


def test_sobel_h_zeros():
    """Horizontal sobel on an array of all zeros."""
    result = skipp_sobel_h(np.zeros((10, 10), dtype=np.float32))
    assert(np.all(result == 0))


@pytest.mark.skip(reason="deveoloping _mask_filter_result in progress")
def test_sobel_h_mask():
    """Horizontal Sobel on a masked array should be zero."""
    result = skipp_sobel_h(np.random.uniform(size=(10, 10)),
                           np.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)


def test_sobel_h_horizontal():
    """Horizontal Sobel on an edge should be a horizontal line."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(np.float32)
    result = skipp_sobel_h(image)
    # Check if result match transform direction
    i[np.abs(j) == 5] = 10000
    assert (np.all(result[i == 0] == 1))
    assert (np.all(result[np.abs(i) > 1] == 0))


def test_sobel_h_vertical():
    """Horizontal Sobel on a vertical edge should be zero."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(np.float32) * np.sqrt(2)
    result = skipp_sobel_h(image)
    assert_allclose(result, 0, atol=1e-10)


def test_sobel_v_zeros():
    """Vertical sobel on an array of all zeros."""
    result = skipp_sobel_v(np.zeros((10, 10), dtype=np.float32))
    assert_allclose(result, 0)


@pytest.mark.skip(reason="deveoloping _mask_filter_result in progress")
def test_sobel_v_mask():
    """Vertical Sobel on a masked array should be zero."""
    result = skipp_sobel_v(np.random.uniform(size=(10, 10)),
                           np.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)


def test_sobel_v_vertical():
    """Vertical Sobel on an edge should be a vertical line."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(np.float32)
    result = skipp_sobel_v(image)
    # Check if result match transform direction
    j[np.abs(i) == 5] = 10000
    assert (np.all(result[j == 0] == 1))
    assert (np.all(result[np.abs(j) > 1] == 0))


def test_sobel_v_horizontal():
    """vertical Sobel on a horizontal edge should be zero."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(np.float32)
    result = skipp_sobel_v(image)
    assert_allclose(result, 0)

# TODO
# add tests
