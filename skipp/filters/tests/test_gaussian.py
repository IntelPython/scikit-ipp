import pytest
import numpy as np
from skipp.skipp.filters import gaussian


@pytest.fixture
def image():
    return np.array([[1, 2, 3, 2, 1],
                     [1, 1, 2, 2, 3],
                     [3, 2, 1, 2, 1],
                     [3, 2, 1, 1, 1],
                     [1, 2, 1, 2, 3]],
                    dtype=np.uint8)


@pytest.mark.parametrize(
    "output_dtype", [np.uint8, np.uint16, np.int16, np.float32]
)
@pytest.mark.parametrize(
    "input_dtype", [np.uint8, np.uint16, np.int16, np.float32]
)
def test_gaussian_output_dtype(image, input_dtype, output_dtype):
    gaussian_image = gaussian(image.astype(input_dtype), output=output_dtype)
    assert gaussian_image.dtype == output_dtype


def test_null_sigma():
    a = np.zeros((3, 3), dtype=np.uint8)
    a[1, 1] = 1
    assert np.all(gaussian(a, output=np.uint8, sigma=0) == a)


def test_default_sigma():
    a = np.zeros((3, 3))
    a[1, 1] = 1.
    assert np.all(gaussian(a) == gaussian(a, sigma=1))


def test_energy_decrease():
    a = np.zeros((3, 3))
    a[1, 1] = 1.
    gaussian_a = gaussian(a, sigma=1, mode='reflect')
    assert gaussian_a.std() < a.std()


def test_multichannel():
    pass


def test_preserve_range():
    img = np.array([[10.0, -10.0], [-4, 3]], dtype=np.float32)
    gaussian(img, 1, preserve_range=True)


def test_gaussian_error_ndim():
    img = np.zeros((5,) * 4)
    with pytest.raises(ValueError):
        gaussian(img, sigma=1)


def test_output_type():
    img = np.arange(16, dtype=np.uint8).reshape((4, 4))
    output_type = np.uint8
    gaussian_img = gaussian(img, 1, output=output_type)
    assert gaussian_img.dtype == output_type
    output_image = np.zeros_like(img, dtype=output_type)
    gaussian_img = gaussian(img, 1, output=output_image)
    assert gaussian_img.dtype == output_type
