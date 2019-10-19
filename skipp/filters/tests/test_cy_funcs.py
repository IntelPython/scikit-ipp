import pytest
import numpy as np
# this funcs currently exist in _filter.pyx
from skipp.skipp.filters import (_get_cy__ipp_equalent_number_for_numpy,
	                             _get_cy__get_IppBorderType,
	                             _get_cy__get_number_of_channels)

# TODO
# correct test names
@pytest.mark.parametrize("input_numpy_dtype, expected_int_output_value",
	                     [(np.uint8, 0), (np.int8, 1), (np.uint16, 2), (np.int16, 3), (np.uint32, 4),
	                      (np.int32, 5), (np.uint64, 6), (np.int64, 7), (np.float32, 8), (np.float64, 9),
	                      (np.double, 9)])
def test__ipp_equalent_number_for_numpy__supported_dtypes(input_numpy_dtype, expected_int_output_value):
	input_image = np.zeros((3, 3), dtype=input_numpy_dtype)
	returned_int_output_value = _get_cy__ipp_equalent_number_for_numpy(input_image)
	assert returned_int_output_value == expected_int_output_value


@pytest.mark.parametrize("input_numpy_dtype", np.sctypes['complex'] + np.sctypes['others'])
@pytest.mark.parametrize("expected_int_output_value", [-1])
def test__ipp_equalent_number_for_numpy__unsupported_dtypes(input_numpy_dtype, expected_int_output_value):
	input_image = np.zeros((3, 3), dtype=input_numpy_dtype)
	returned_int_output_value = _get_cy__ipp_equalent_number_for_numpy(input_image)
	assert returned_int_output_value == expected_int_output_value


@pytest.mark.parametrize("input_mode_str, expected_output_IppBoredType_int",
	                     [(str('nearest'), 1), (str('wrap'), 2), (str('mirror'), 3), (str('reflect'), 4),
	                      (str('default'), 5), (str('constant'), 6), (str('transp'), 7), (str('*'), -1)])
def test__get_IppBorderType(input_mode_str, expected_output_IppBoredType_int):
	returned_output_IppBoredType_int = _get_cy__get_IppBorderType(input_mode_str)
	assert returned_output_IppBoredType_int == expected_output_IppBoredType_int


# TODO
# _get_cy__get_number_of_channels
@pytest.mark.skip(reason="in progress")
def test__get_number_of_channels():
	pass
