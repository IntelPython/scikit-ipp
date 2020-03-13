import pytest
import numpy as np
from skipp.filters import gaussian


@pytest.fixture
def image():
    return np.array([[1, 2, 3, 2, 1],
                     [1, 1, 2, 2, 3],
                     [3, 2, 1, 2, 1],
                     [3, 2, 1, 1, 1],
                     [1, 2, 1, 2, 3]],
                    dtype=np.uint8)


# TODO
# add checks for np.uint32, np.int32, np.uint64, np.int64, np.float64, np.double
# TODO
# add also checks for different output and input dtypes
# TODO
@pytest.mark.skip(reason="update test")
@pytest.mark.parametrize("input_dtype", [np.uint8, np.int8, np.uint16,
                                        np.int16, np.float32])
def test_gaussian_preserve_dtype(image, input_dtype):
    gaussian_image = gaussian(image.astype(input_dtype))
    assert gaussian_image.dtype == input_dtype


def test_gaussian_default_sigma():
    a = np.zeros((3, 3), dtype=np.uint8)
    a[1, 1] = 1.
    assert np.all(gaussian(a) == gaussian(a, sigma=1))

# TODO
# add checks for np.uint32, np.int32, np.uint64, np.int64, np.float64, np.double
# currently supported dtypes: np.uint8, np.uint16, np.int16, np.float32
# TODO
@pytest.mark.skip(reason="update test")
@pytest.mark.parametrize("output_dtype", [np.uint8, np.uint16, np.int16, np.float32])
def test_gaussian_preserve_output(output_dtype):
    input_image = np.zeros((3, 3), dtype=output_dtype)
    output_image = np.zeros((3, 3), dtype=output_dtype)
    returned_image = gaussian(input_image, output=output_image)
    assert id(output_image) == id(returned_image)


def test_gaussian_null_sigma():
    a = np.zeros((3, 3))
    a[1, 1] = 1.
    assert np.all(gaussian(a, 0) == a)


# TODO: investigate why skimage in test_energy_decrease_gaussian uses `reflect` mode
# Intel(R) IPP GaussianFilterBorder doesn't support `reflect` mode
def test_gaussian_energy_decrease():
    a = np.zeros((3, 3), dtype=np.float32)
    a[1, 1] = 1.
    gaussian_a = gaussian(a, sigma=1)
    assert gaussian_a.std() < a.std()


def test_gaussian_unsupported_mode():
    """
    Intel(R) IPP GaussianFilterBorder doesn't support `reflect` mode
    """
    a = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(RuntimeError):
        gaussian_a = gaussian(a, mode='reflect')

# TODO
@pytest.mark.skip(reason="`preserve_range` param is not enabled")
def test_gaussian_preserve_range():
    img = np.array([[10.0, -10.0], [-4, 3]], dtype=np.float32)
    gaussian(img, 1, preserve_range=True)


# TODO
# update
def test_gaussian_dimension_error():
    image_4d = np.arange(5*5*5*4, dtype=np.uint8).reshape((5, 5, 5, 4))
    with pytest.raises(ValueError):
        filtered_img = gaussian(image_4d, sigma=1, multichannel=True)
