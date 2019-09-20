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


# def test_null_sigma():
#     pass

def test_default_sigma():
    a = np.zeros((3, 3))
    a[1, 1] = 1.
    assert np.all(gaussian(a) == gaussian(a, sigma=1))


def test_energy_decrease():
    a = np.zeros((3, 3))
    a[1, 1] = 1.
    gaussian_a = gaussian(a, sigma=1, mode='reflect')
    assert gaussian_a.std() < a.std()


# def test_multichannel():
#    pass


def test_preserve_range():
    img = np.array([[10.0, -10.0], [-4, 3]], dtype=np.float32)
    gaussian(img, 1, preserve_range=True)
