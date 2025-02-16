from gaiaxpy.colour_equation.xp_filter_system_colour_equation import apply_colour_equation
from gaiaxpy.core.generic_functions import cast_output, _validate_arguments
from gaiaxpy.error_correction.error_correction import apply_error_correction
from gaiaxpy.input_reader.input_reader import InputReader
from gaiaxpy.output.photometry_data import PhotometryData
from .multi_synthetic_photometry_generator import MultiSyntheticPhotometryGenerator
from .photometric_system import PhotometricSystem


def generate(input_object, photometric_system, output_path='.', output_file='output_synthetic_photometry',
             output_format=None, save_file=True, error_correction=False, username=None, password=None):
    """
    Synthetic photometry utility: generates synthetic photometry in a set of available systems from the input
        internally-calibrated continuously-represented mean spectra.

        Some standardised photometric systems include a colour-correction to the U bands which will be applied
        automatically when generating the corresponding synthetic photometry.

    Args:
        input_object (object): Path to the file containing the mean spectra as downloaded from the archive in their
            continuous representation, a list of sources ids (string or long), or a pandas DataFrame.
        photometric_system (obj): Desired photometric system or list of photometric systems.
        output_path (str): Path where to save the output data.
        output_file (str): Name of the output file.
        output_format (str): Format to be used for the output file. If no format is given, then the output file will
            have the same format as the input file.
        save_file (bool): Whether to save the output in a file or not. If false, output_format and output_file_name will
            be ignored.
        error_correction (bool): Whether to apply to the photometric errors the tabulated factors to mitigate
            underestimated errors (see Montegriffo et al., 2022, for more details).
        username (str): Cosmos username, only suggested when input_object is a list or ADQL query.
        password (str): Cosmos password, only suggested when input_object is a list or ADQL query.
        
    Returns:
        DataFrame: A DataFrame of all synthetic photometry results.
    """

    def is_gaia_initially_in_systems(_internal_photometric_system, _gaia_system):
        system_names = [s.name for s in _internal_photometric_system]
        return _gaia_system.name in system_names

    # colour_equation should be always true as it is part of the definition of standardised systems.
    colour_equation = True
    # TODO: merge this statement with _validate_arguments. "if not photometric_system" didn't seem to work, check again.
    if photometric_system in (None, [], ''):
        raise ValueError('At least one photometric system is required as input.')
    _validate_arguments(generate.__defaults__[1], output_file, save_file)
    # Load data
    parsed_input_data, extension = InputReader(input_object, generate, username, password)._read()
    # Prepare systems, keep track of original systems
    internal_photometric_system = photometric_system.copy() if isinstance(photometric_system, list) else \
        [photometric_system].copy()
    gaia_system = PhotometricSystem.Gaia_DR3_Vega
    # Create multi generator
    gaia_initially_in_systems = is_gaia_initially_in_systems(internal_photometric_system, gaia_system)
    if error_correction and not gaia_initially_in_systems:
        internal_photometric_system.append(gaia_system)
    if isinstance(internal_photometric_system, list):
        generator = MultiSyntheticPhotometryGenerator(internal_photometric_system, bp_model='v375wi', rp_model='v142r')
    else:
        raise ValueError('Photometry generation not implemented for the input type.')
    photometry_df = generator._generate(parsed_input_data, extension, output_file=None, output_format=None,
                                        save_file=False)
    if colour_equation:
        photometry_df = apply_colour_equation(photometry_df, photometric_system=internal_photometric_system,
                                              save_file=False)
    if error_correction:
        photometry_df = apply_error_correction(photometry_df, photometric_system=photometric_system, save_file=False)
    if not gaia_initially_in_systems:
        # Remove Gaia_DR3_Vega system from the final result
        gaia_label = gaia_system.get_system_label()
        gaia_columns = [column for column in photometry_df if column.startswith(gaia_label)]
        photometry_df = photometry_df.drop(columns=gaia_columns)
    output_data = PhotometryData(photometry_df)
    output_data.data = cast_output(output_data)
    output_data.save(save_file, output_path, output_file, output_format, extension)
    return photometry_df
