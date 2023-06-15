import unittest

import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt

from gaiaxpy import convert
from gaiaxpy.core.generic_functions import str_to_array
from gaiaxpy.input_reader.input_reader import InputReader
from tests.files.paths import *
from tests.utils.utils import pos_file_to_array, missing_bp_source_id

# Load solution
solution_path = join(files_path, 'converter_solution')
converters = dict([(column, lambda x: str_to_array(x)) for column in ['flux', 'flux_error']])
with_missing_solution_df = pd.read_csv(join(solution_path, 'with_missing_converter_solution.csv'),
                                       converters=converters)
solution_sampling = pos_file_to_array(join(solution_path, 'with_missing_converter_solution_sampling.csv'))

missing_solution_df = with_missing_solution_df[with_missing_solution_df['source_id'] ==
                                               missing_bp_source_id].reset_index(drop=True)

_rtol, _atol = 1e-7, 1e-7


class TestConverterMissingBPFileInput(unittest.TestCase):

    def test_missing_bp_csv_file(self):
        output_df, sampling = convert(with_missing_bp_csv_file, save_file=False)
        pdt.assert_frame_equal(output_df, with_missing_solution_df, rtol=_rtol, atol=_atol)
        npt.assert_array_equal(sampling, solution_sampling)

    def test_missing_bp_ecsv_file(self):
        output_df, sampling = convert(with_missing_bp_ecsv_file, save_file=False)
        pdt.assert_frame_equal(output_df, with_missing_solution_df, rtol=_rtol, atol=_atol)
        npt.assert_array_equal(sampling, solution_sampling)

    def test_missing_bp_fits_file(self):
        output_df, sampling = convert(with_missing_bp_fits_file, save_file=False)
        pdt.assert_frame_equal(output_df, with_missing_solution_df, rtol=_rtol, atol=_atol)
        npt.assert_array_equal(sampling, solution_sampling)

    def test_missing_bp_xml_file(self):
        output_df, sampling = convert(with_missing_bp_xml_file, save_file=False)
        pdt.assert_frame_equal(output_df, with_missing_solution_df, rtol=_rtol, atol=_atol)
        npt.assert_array_equal(sampling, solution_sampling)

    def test_missing_bp_xml_plain_file(self):
        output_df, sampling = convert(with_missing_bp_xml_plain_file, save_file=False)
        pdt.assert_frame_equal(output_df, with_missing_solution_df, rtol=_rtol, atol=_atol)
        npt.assert_array_equal(sampling, solution_sampling)

    def test_missing_bp_csv_file_isolated(self):
        output_df, sampling = convert(missing_bp_csv_file, save_file=False)
        pdt.assert_frame_equal(output_df, missing_solution_df, atol=_atol, rtol=_rtol)
        npt.assert_array_equal(sampling, solution_sampling)

    def test_missing_bp_ecsv_file_isolated(self):
        output_df, sampling = convert(missing_bp_ecsv_file, save_file=False)
        pdt.assert_frame_equal(output_df, missing_solution_df, atol=_atol, rtol=_rtol)
        npt.assert_array_equal(sampling, solution_sampling)

    def test_missing_bp_fits_file_isolated(self):
        output_df, sampling = convert(missing_bp_fits_file, save_file=False)
        pdt.assert_frame_equal(output_df, missing_solution_df, atol=_atol, rtol=_rtol)
        npt.assert_array_equal(sampling, solution_sampling)

    def test_missing_bp_xml_file_isolated(self):
        output_df, sampling = convert(missing_bp_xml_file, save_file=False)
        pdt.assert_frame_equal(output_df, missing_solution_df, atol=_atol, rtol=_rtol)
        npt.assert_array_equal(sampling, solution_sampling)

    def test_missing_bp_xml_plain_file_isolated(self):
        output_df, sampling = convert(missing_bp_xml_plain_file, save_file=False)
        pdt.assert_frame_equal(output_df, missing_solution_df, atol=_atol, rtol=_rtol)
        npt.assert_array_equal(sampling, solution_sampling)


class TestConverterMissingBPDataFrameInput(unittest.TestCase):

    def test_missing_bp_simple_dataframe(self):
        # Test dataframe containing strings instead of arrays
        df = pd.read_csv(with_missing_bp_csv_file)
        output_df, sampling = convert(df, save_file=False)
        pdt.assert_frame_equal(output_df, with_missing_solution_df, rtol=_rtol, atol=_atol)
        npt.assert_array_equal(sampling, solution_sampling)

    def test_missing_bp_simple_dataframe_isolated(self):
        # Test dataframe containing strings instead of arrays
        df = pd.read_csv(missing_bp_csv_file)
        output_df, sampling = convert(df, save_file=False)
        pdt.assert_frame_equal(output_df, missing_solution_df, atol=_atol, rtol=_rtol)
        npt.assert_array_equal(sampling, solution_sampling)

    def test_missing_bp_array_dataframe(self):
        # Convert columns in the dataframe to arrays (but not to matrices)
        array_columns = ['bp_coefficients', 'bp_coefficient_errors', 'bp_coefficient_correlations', 'rp_coefficients',
                         'rp_coefficient_errors', 'rp_coefficient_correlations']
        _converters = dict([(column, lambda x: str_to_array(x)) for column in array_columns])
        df = pd.read_csv(with_missing_bp_csv_file, converters=_converters)
        output_df, sampling = convert(df, save_file=False)
        pdt.assert_frame_equal(output_df, with_missing_solution_df, rtol=_rtol, atol=_atol)
        npt.assert_array_equal(sampling, solution_sampling)

    def test_missing_bp_array_dataframe_isolated(self):
        array_columns = ['bp_coefficients', 'bp_coefficient_errors', 'bp_coefficient_correlations', 'rp_coefficients',
                         'rp_coefficient_errors', 'rp_coefficient_correlations']
        converters = dict([(column, lambda x: str_to_array(x)) for column in array_columns])
        df = pd.read_csv(missing_bp_csv_file, converters=converters)
        output_df, sampling = convert(df, save_file=False)
        pdt.assert_frame_equal(output_df, missing_solution_df, atol=_atol, rtol=_rtol)
        npt.assert_array_equal(sampling, solution_sampling)

    def test_missing_bp_parsed_dataframe(self):
        # Test completely parsed (arrays + matrices) dataframe
        df, _ = InputReader(with_missing_bp_csv_file, convert).read()
        output_df, sampling = convert(df, save_file=False)
        pdt.assert_frame_equal(output_df, with_missing_solution_df, rtol=_rtol, atol=_atol)
        npt.assert_array_equal(sampling, solution_sampling)

    def test_missing_bp_parsed_dataframe_isolated(self):
        # Test completely parsed (arrays + matrices) dataframe
        df, _ = InputReader(missing_bp_csv_file, convert).read()
        output_df, sampling = convert(df, save_file=False)
        pdt.assert_frame_equal(output_df, missing_solution_df, atol=_atol, rtol=_rtol)
        npt.assert_array_equal(sampling, solution_sampling)
