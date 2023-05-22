"""
cast.py
====================================
Module to cast the data after parsing.
"""
import numpy as np
import pandas as pd
from numpy.ma import getdata

# Fields of all formats including AVRO.
__type_map = {'source_id': 'int64', 'solution_id': 'int64', 'rp_n_parameters': 'int64', 'bp_n_parameters': 'int64',
              'rp_n_rejected_measurements': 'int64', 'bp_n_rejected_measurements': 'int64',
              'rp_n_measurements': 'int64', 'bp_n_measurements': 'int64',
              'rp_standard_deviation': 'float64', 'bp_standard_deviation': 'float64',
              'rp_num_of_transits': 'int64', 'bp_num_of_transits': 'int64',
              'rp_num_of_blended_transits': 'int64', 'bp_num_of_blended_transits': 'int64',
              'rp_num_of_contaminated_transits': 'int64', 'bp_num_of_contaminated_transits': 'int64',
              'rp_coefficients': 'O', 'bp_coefficients': 'O',
              'rp_coefficient_covariances': 'O', 'bp_coefficient_covariances': 'O',
              'rp_degrees_of_freedom': 'int64', 'bp_degrees_of_freedom': 'int64',
              'rp_n_relevant_bases': 'int16', 'bp_n_relevant_bases': 'int16',
              'rp_basis_function_id': 'int64', 'bp_basis_function_id': 'int64',
              'rp_chi_squared': 'float64', 'bp_chi_squared': 'float64',
              'rp_coefficient_errors': 'O', 'bp_coefficient_errors': 'O',
              'rp_coefficient_correlations': 'O', 'bp_coefficient_correlations': 'O',
              'rp_relative_shrinking': 'float64', 'bp_relative_shrinking': 'float64'}


def __replace_masked_constant(value):
    return float('NaN') if isinstance(value, np.ma.core.MaskedConstant) else value


def __replace_masked_array(value):
    if (isinstance(value, np.ma.core.MaskedArray) and getdata(value).size == 0) or\
            (isinstance(value, float) and value == 0.0) or (isinstance(value, float) and pd.isna(value)) or\
            isinstance(value, pd._libs.missing.NAType) or np.isnan(np.sum(value)):
        return np.array([])
    elif isinstance(value, np.ma.core.MaskedArray):
        return getdata(value)
    else:
        return value

def _cast(df):
    """
    Cast types to the defined ones to standardise the different input formats.

    Args:
        df (DataFrame): a DataFrame with parsed data from input files.
    """
    for column in df.columns:
        if column not in __type_map.keys():
            pass  # Parsing is not required
        else:
            if __type_map[column] == 'O':
                df[column] = df[column].apply(lambda x: __replace_masked_array(x))
            elif __type_map[column] in ['int16', 'int64', 'float64']:
                if any(isinstance(x, np.ma.core.MaskedConstant) for x in df[column].values):
                    df[column] = df[column].apply(lambda x: __replace_masked_constant(x))
                    _req_dtype = 'float64'
                elif any((pd.isna(x) for x in df[column].values)):
                    _req_dtype = 'float64'
                else:
                    _req_dtype = __type_map[column]
                df[column].astype(_req_dtype)
    return df
