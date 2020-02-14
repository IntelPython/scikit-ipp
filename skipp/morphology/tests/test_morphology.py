import os

import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_allclose,
                           assert_array_equal)
from skimage.morphology import grey, selem
from skimage._shared.testing import (assert_equal, TestCase, parametrize)
from skimage import img_as_uint

import skipp.morphology


class TestEccentricStructuringElements(TestCase):
    def setUp(self):
        self.black_pixel = 255 * np.ones((4, 4), dtype=np.uint8)
        self.black_pixel[1, 1] = 0
        self.white_pixel = 255 - self.black_pixel
        self.selems = [selem.square(2), selem.rectangle(2, 2),
                       selem.rectangle(2, 1), selem.rectangle(1, 2)]

    def test_dilate_erode_symmetry(self):
        for s in self.selems:
            c = skipp.morphology.erosion(self.black_pixel, s)
            d = skipp.morphology.dilation(self.white_pixel, s)
            assert np.all(c == (255 - d))


@pytest.mark.parametrize("function", [pytest.param(skipp.morphology.dilation, id="dilation"),
                                      pytest.param(skipp.morphology.erosion, id="erosion")])
def test_default_selem(function):
    strel = selem.diamond(radius=1)
    image = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], np.uint8)
    im_expected = function(image, strel)
    im_test = function(image)
    np.testing.assert_allclose(im_expected, im_test)


# float test images
im = np.array([[0.55, 0.72, 0.60, 0.54, 0.42],
               [0.65, 0.44, 0.89, 0.96, 0.38],
               [0.79, 0.53, 0.57, 0.93, 0.07],
               [0.09, 0.02, 0.83, 0.78, 0.87],
               [0.98, 0.80, 0.46, 0.78, 0.12]], dtype=np.float32)

eroded = np.array([[0.55, 0.44, 0.54, 0.42, 0.38],
                   [0.44, 0.44, 0.44, 0.38, 0.07],
                   [0.09, 0.02, 0.53, 0.07, 0.07],
                   [0.02, 0.02, 0.02, 0.78, 0.07],
                   [0.09, 0.02, 0.46, 0.12, 0.12]], dtype=np.float32)

dilated = np.array([[0.72, 0.72, 0.89, 0.96, 0.54],
                    [0.79, 0.89, 0.96, 0.96, 0.96],
                    [0.79, 0.79, 0.93, 0.96, 0.93],
                    [0.98, 0.83, 0.83, 0.93, 0.87],
                    [0.98, 0.98, 0.83, 0.78, 0.87]], dtype=np.float32)


def test_float():
    np.testing.assert_allclose(skipp.morphology.erosion(im), eroded)
    np.testing.assert_allclose(skipp.morphology.dilation(im), dilated)


def test_uint16():
    im16, eroded16, dilated16 = (
        map(img_as_uint, [im, eroded, dilated]))
    np.testing.assert_allclose(skipp.morphology.erosion(im16), eroded16)
    np.testing.assert_allclose(skipp.morphology.dilation(im16), dilated16)


@pytest.mark.skip(reason="needs __get_output implementation")
def test_discontiguous_out_array():
    image = np.array([[5, 6, 2],
                      [7, 2, 2],
                      [3, 5, 1]], np.uint8)
    out_array_big = np.zeros((5, 5), np.uint8)
    out_array = out_array_big[::2, ::2]
    expected_dilation = np.array([[7, 0, 6, 0, 6],
                                  [0, 0, 0, 0, 0],
                                  [7, 0, 7, 0, 2],
                                  [0, 0, 0, 0, 0],
                                  [7, 0, 5, 0, 5]], np.uint8)
    expected_erosion = np.array([[5, 0, 2, 0, 2],
                                 [0, 0, 0, 0, 0],
                                 [2, 0, 2, 0, 1],
                                 [0, 0, 0, 0, 0],
                                 [3, 0, 1, 0, 1]], np.uint8)
    skipp.morphology.dilation(image, out=out_array)
    assert_array_equal(out_array_big, expected_dilation)
    skipp.morphology.erosion(image, out=out_array)
    assert_array_equal(out_array_big, expected_erosion)


def test_1d_erosion():
    image = np.array([1, 2, 3, 2, 1], dtype=np.uint8)
    expected = np.array([1, 1, 2, 1, 1], dtype=np.uint8)
    eroded = skipp.morphology.erosion(image)
    assert_array_equal(eroded, expected)


def test_1d_dilation():
    image = np.array([1, 2, 3, 2, 1], dtype=np.uint8)
    expected = np.array([2, 3, 3, 3, 2], dtype=np.uint8)
    dilated = skipp.morphology.dilation(image)
    assert_array_equal(dilated, expected)
