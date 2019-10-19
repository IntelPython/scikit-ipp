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


@pytest.mark.parametrize("input_dtype", [np.uint8, np.int8, np.uint16, np.int16, np.float32])
def test_gaussian_output_is_none(image, input_dtype):
    gaussian_image = gaussian(image.astype(input_dtype))
    assert gaussian_image.dtype == input_dtype


# skipped: in progress
# TODO
@pytest.mark.skip(reason="dev in progress")
@pytest.mark.parametrize("input_dtype", [np.uint32, np.int32, np.uint64, np.int64, np.float64,
                                         np.double])
def test_gaussian_output_is_none_dev(image, input_dtype):
    gaussian_image = gaussian(image.astype(input_dtype))
    assert gaussian_image.dtype == input_dtype


def test_default_sigma_gaussian():
    a = np.zeros((3, 3), dtype=np.uint8)
    a[1, 1] = 1.
    assert np.all(gaussian(a) == gaussian(a, sigma=1))


@pytest.mark.parametrize("ipp_supported", [np.uint8, np.uint16, np.int16, np.float32])
def test_given_and_returned_output_gaussian(ipp_supported):
    input_image = np.zeros((3, 3), dtype=ipp_supported)
    output_image = np.zeros((3, 3), dtype=ipp_supported)
    returned_image = gaussian(input_image, output=output_image)
    assert id(output_image) == id(returned_image)


def test_null_sigma():
    a = np.zeros((3, 3))
    a[1, 1] = 1.
    assert np.all(gaussian(a, 0) == a)


# Intel IPP's GaussianFilterBorder doesn't support `reflect` mode
# TODO: investigate why skimage in test_energy_decrease_gaussian uses `reflect` mode
def test_energy_decrease_gaussian():
    a = np.zeros((3, 3), dtype=np.float32)
    a[1, 1] = 1.
    gaussian_a = gaussian(a, sigma=1)
    assert gaussian_a.std() < a.std()


def test_gaussian_unsupported_mode_in_IPP():
    a = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(RuntimeError):
        gaussian_a = gaussian(a, mode='reflect')


# TODO
@pytest.mark.skip(reason="dev in progress")
def test_preserve_range_gaussian():
    img = np.array([[10.0, -10.0], [-4, 3]], dtype=np.float32)
    gaussian(img, 1, preserve_range=True)

# skipped: fails
# TODO
@pytest.mark.skip(reason="dev in progress")
def test_dimensiona_error_gaussian():
    image_4d = np.arange(5*5*5*4, dtype=np.uint8).reshape((5, 5, 5, 4))
    with pytest.raises(ValueError):
        filtered_img = gaussian(image_4d, sigma=1, multichannel=True)
