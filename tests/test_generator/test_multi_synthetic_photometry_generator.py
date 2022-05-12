import unittest
import pandas as pd
import pandas.testing as pdt
from configparser import ConfigParser
from os import path
from gaiaxpy.generator import generate, _generate_synthetic_photometry, PhotometricSystem
from gaiaxpy.core import _load_xpmerge_from_csv, _load_xpsampling_from_csv, \
                         satellite
from gaiaxpy.config import config_path
from gaiaxpy.spectrum import SampledBasisFunctions, SingleSyntheticPhotometry
from tests.files import files_path

# Files to test parse
continuous_path = path.join(files_path, 'xp_continuous')
covariance_avro_file = path.join(continuous_path, 'MeanSpectrumSolutionWithCov.avro')
correlation_csv_file = path.join(continuous_path, 'XP_CONTINUOUS_RAW_dr3int6.csv')
correlation_fits_file = path.join(continuous_path, 'XP_CONTINUOUS_RAW_dr3int6.fits')
correlation_xml_plain_file = path.join(continuous_path, 'XP_CONTINUOUS_RAW_votable_plain_dr3int6.xml')
correlation_xml_file = path.join(continuous_path, 'XP_CONTINUOUS_RAW_votable_dr3int6.xml')


class TestMultiSyntheticPhotometryGenerator(unittest.TestCase):

    def test_generate_empty_list(self):
        with self.assertRaises(ValueError):
            synthetic_photometry = generate(
                correlation_csv_file,
                photometric_system=[],
                save_file=False)

    def test_generate_one_element_list(self):
        phot_system = PhotometricSystem.JKC_Std
        one_element_synthetic_photometry = generate(
                correlation_csv_file,
                photometric_system=[phot_system],
                save_file=False)
        single_synthetic_photometry = generate(
                correlation_csv_file,
                photometric_system=phot_system,
                save_file=False)
        # Rename columns
        pdt.assert_frame_equal(one_element_synthetic_photometry, single_synthetic_photometry)

    def test_generate_csv_mix(self):
        phot_list = [PhotometricSystem.Euclid_VIS, PhotometricSystem.Gaia_2]
        # Current multi-result
        multi_synthetic_photometry = generate(
                correlation_csv_file,
                photometric_system=phot_list,
                save_file=False)
        # Generate right multi-result from single photometries
        single_synthetic_photometry_euclid_vis = generate(
                correlation_fits_file,
                photometric_system=phot_list[0],
                save_file=False)
        single_synthetic_photometry_gaia_2 = generate(
                correlation_xml_file,
                photometric_system=phot_list[1],
                save_file=False)
        # Concatenate but avoid duplicated source_id column
        concatenated_photometry = pd.concat([single_synthetic_photometry_euclid_vis,
                                             single_synthetic_photometry_gaia_2.drop(columns=['source_id'])],
                                             axis=1)
        # Rename multi-photometry columns
        pdt.assert_frame_equal(multi_synthetic_photometry, concatenated_photometry)

    def test_no_system_given_is_none(self):
        f = path.join(continuous_path, 'XP_CONTINUOUS_RAW_dr3int6.csv')
        with self.assertRaises(ValueError):
            generate(f, photometric_system=None)
        with self.assertRaises(ValueError):
            generate(f, photometric_system=[])
        with self.assertRaises(ValueError):
            generate(f, photometric_system='')
