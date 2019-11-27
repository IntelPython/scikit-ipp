import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_allclose)

from skipp.skipp.filters import prewitt as skipp_prewitt
from skipp.skipp.filters import prewitt_h as skipp_prewitt_h
from skipp.skipp.filters import prewitt_v as skipp_prewitt_v

from skimage.filters import prewitt as skimage_prewitt
from skimage.filters import prewitt_h as skimage_prewitt_h
from skimage.filters import prewitt_v as skimage_prewitt_v


def test_prewitt_zeros():
    """Horizontal prewitt on an array of all zeros."""
    result = skipp_prewitt(np.zeros((10, 10), dtype=np.float32))
    assert_allclose(result, 0)

# TODO
@pytest.mark.skip(reason="deveoloping _mask_filter_result in progress")
def test_prewitt_mask():
    """Prewitt on a masked array should be zero."""
    result = filters.prewitt(np.random.uniform(size=(10, 10)),
                             np.zeros((10, 10), dtype=bool))
    assert_allclose(np.abs(result), 0)


def test_prewitt_horizontal():
    """Prewitt on an edge should be a horizontal line."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(np.float32)
    result = skipp_prewitt(image) * np.sqrt(2)
    # Check if result match transform direction
    i[np.abs(j) == 5] = 10000
    assert_allclose(result[i == 0], 1)
    assert_allclose(result[np.abs(i) > 1], 0, atol=1e-10)


def test_prewitt_vertical():
    """Prewitt on a vertical edge should be a vertical line."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(np.float32)
    result = skipp_prewitt(image) * np.sqrt(2)
    j[np.abs(i) == 5] = 10000
    assert_allclose(result[j == 0], 1)
    assert_allclose(result[np.abs(j) > 1], 0, atol=1e-10)


def test_prewitt_h_zeros():
    """Horizontal prewitt on an array of all zeros."""
    result = skipp_prewitt_h(np.zeros((10, 10), dtype=np.float32))
    assert_allclose(result, 0)

# TODO
@pytest.mark.skip(reason="deveoloping _mask_filter_result in progress")
def test_prewitt_h_mask():
    """Horizontal prewitt on a masked array should be zero."""
    result = filters.prewitt_h(np.random.uniform(size=(10, 10)),
                               np.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)


def test_prewitt_h_horizontal():
    """Horizontal prewitt on an edge should be a horizontal line."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(np.float32)
    result = skipp_prewitt_h(image)
    # Check if result match transform direction
    i[np.abs(j) == 5] = 10000
    assert (np.all(result[i == 0] == 1))
    assert_allclose(result[np.abs(i) > 1], 0, atol=1e-10)


def test_prewitt_h_vertical():
    """Horizontal prewitt on a vertical edge should be zero."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(np.float32)
    result = skipp_prewitt_h(image)
    assert_allclose(result, 0, atol=1e-10)


def test_prewitt_v_zeros():
    """Horizontal prewitt on an array of all zeros."""
    result = skipp_prewitt_v(np.zeros((10, 10), dtype=np.float32))
    assert_allclose(result, 0)

# TODO
@pytest.mark.skip(reason="deveoloping _mask_filter_result in progress")
def test_prewitt_v_mask():
    """Vertical prewitt on a masked array should be zero."""
    result = filters.prewitt_v(np.random.uniform(size=(10, 10)),
                               np.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)


def test_prewitt_v_vertical():
    """Vertical prewitt on an edge should be a vertical line."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(np.float32)
    result = skipp_prewitt_v(image)
    # Check if result match transform direction
    j[np.abs(i) == 5] = 10000
    assert (np.all(result[j == 0] == 1))
    assert_allclose(result[np.abs(j) > 1], 0, atol=1e-10)


def test_prewitt_v_horizontal():
    """Vertical prewitt on a horizontal edge should be zero."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(np.float32)
    result = skipp_prewitt_v(image)
    assert_allclose(result, 0)

# TODO
# add tests
