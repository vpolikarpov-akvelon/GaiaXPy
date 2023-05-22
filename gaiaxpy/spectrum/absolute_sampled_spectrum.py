"""
absolute_sampled_spectrum.py
====================================
Module to represent an absolute sampled spectrum.
"""

from numbers import Number

import numpy as np
import pandas as pd

from gaiaxpy.core.satellite import BANDS, BP_WL, RP_WL
from .sampled_spectrum import SampledSpectrum
from .utils import _list_to_array
from ..core.generic_functions import correlation_from_covariance


class AbsoluteSampledSpectrum(SampledSpectrum):
    """
    A spectrum calibrated onto the absolute system of wavelength and flux. The spectrum is represented by a set of
    discrete measurements or samples.
    """

    def __init__(self, source_id, xp_spectra, sampled_bases, merge, truncation=None, with_correlation=False):
        """
        Initialise an absolute sampled spectrum.

        Args:
            source_id (str): Source identifier.
            xp_spectra (dict): A dictionary containing the BP and RP continuous spectra.
            sampled_bases (dict): The set of basis functions sampled onto the grid defining the resolution of the final
                sampled spectrum.
            merge (dict): The weighting factors for BP and RP sampled onto the grid defining the resolution of the final
                sampled spectrum.
            truncation (dict): Number of bases to be used for this spectrum. The set of bases functions used for the
                continuous representation of the spectra has been optimised to ensure that the first bases are the ones
                that contribute most. In many cases, the last bases contribution will be below the noise. Truncation of
                the basis function set to preserve only the significant bases is optional. By default, no truncation
                will be applied, i.e. all bases will be used.
            with_correlation (bool): Whether correlation information should be computed.
        """
        truncation = dict() if truncation is None else truncation
        # Bands available, float type indicates the coefficients are nan, as they should be an array
        bands = [band for band in xp_spectra.keys() if not len(xp_spectra[band].get_coefficients()) == 0]
        if not bands:
            raise ValueError('At least one band must be present.')
        pos = sampled_bases[bands[0]].get_sampling_grid()
        SampledSpectrum.__init__(self, source_id, pos)
        split_spectrum = self.__generate_spectra(xp_spectra, sampled_bases, bands, truncation,
                                                 with_correlation=with_correlation)
        self.pos = pos
        self.__merge_output(split_spectrum, bands, merge, with_correlation=with_correlation)

    def __generate_spectra(self, xp_spectra, sampled_bases, bands, truncation, with_correlation):
        split_spectrum = {band: {} for band in BANDS}
        for band in bands:
            band_truncation = truncation.get(band)
            split_spectrum[band]['xp_spectra'] = xp_spectra[band]
            stdev = split_spectrum[band]['xp_spectra'].get_standard_deviation()
            design_matrix = sampled_bases[band].get_design_matrix()
            spectra_covariance = split_spectrum[band]['xp_spectra'].get_covariance()
            coefficients = split_spectrum[band]['xp_spectra'].get_coefficients()
            if isinstance(band_truncation, Number) and band_truncation > 0:
                design_matrix = design_matrix[:band_truncation][:]
                spectra_covariance = spectra_covariance[:band_truncation, :band_truncation]
                coefficients = coefficients[:band_truncation]
            split_spectrum[band]['flux'] = self._sample_flux(coefficients, design_matrix)
            split_spectrum[band]['error'] = self._sample_error(spectra_covariance, design_matrix, stdev)
            if with_correlation:
                split_spectrum[band]['cov'] = self._sample_covariance(spectra_covariance, design_matrix)
                split_spectrum[band]['stdev'] = stdev
        return split_spectrum

    def __merge_output(self, split_spectrum, bands, merge, with_correlation):
        # If both bands are present
        if len(bands) == 2:
            self.flux = np.add(np.multiply(split_spectrum[BANDS.bp]['flux'], merge[BANDS.bp]),
                               np.multiply(split_spectrum[BANDS.rp]['flux'], merge[BANDS.rp]))
            self.error = np.sqrt(np.add(np.multiply(split_spectrum[BANDS.bp]['error'] ** 2, merge[BANDS.bp] ** 2),
                                        np.multiply(split_spectrum[BANDS.rp]['error'] ** 2, merge[BANDS.rp] ** 2)))
            if with_correlation:
                self.covariance = np.add(np.multiply(split_spectrum[BANDS.bp]['cov'], merge[BANDS.bp]),
                                         np.multiply(split_spectrum[BANDS.rp]['cov'], merge[BANDS.rp]))
        # If only one is
        elif len(bands) == 1:
            existing_band = bands[0]
            self.flux = split_spectrum[existing_band]['flux']
            self.error = split_spectrum[existing_band]['error']
            if with_correlation:
                self.covariance = split_spectrum[existing_band]['cov']
            # Patch values if there's a band missing
            masked_pos = self.pos.copy()
            masked_pos = masked_pos.astype(float)
            if existing_band == BANDS.rp:
                masked_pos[masked_pos <= RP_WL.low] = np.nan
            elif existing_band == BANDS.bp:
                masked_pos[masked_pos >= BP_WL.high] = np.nan
            # Get the indices of all the values in pos that are less than the lowest RP range value
            self.flux[np.argwhere(np.isnan(masked_pos))] = np.nan
            self.error[np.argwhere(np.isnan(masked_pos))] = np.nan
            if with_correlation:
                self.covariance[:, np.argwhere(np.isnan(masked_pos))] = np.nan
                self.covariance[np.argwhere(np.isnan(masked_pos)), :] = np.nan

    def _get_fluxes(self):
        return self.flux

    def _get_flux_errors(self):
        return self.error

    def get_positions(self):
        return self.pos

    @classmethod
    def get_units(cls):
        return {'flux': 'W.nm**-1.m**-2', 'flux_error': 'W.nm**-1.m**-2', 'pos': 'nm'}

    @classmethod
    def _get_flux_label(cls):
        return 'Flux [W nm^-1 m^-2]'

    @classmethod
    def get_position_label(cls):
        return 'Wavelength [nm]'

    def spectrum_to_dict(self):
        """
        Represent the spectrum as a dictionary.

        Returns:
            dict: A dictionary populated with the minimum set of parameters that need to be stored for this object.
                This is optimised for writing large number of sampled spectra and for this reason the array of positions
                is NOT included as it is expected to be the same for a batch of spectra. The array of positions can be
                retrieved calling the sampling_to_dict method.
        """
        spectrum_dict = {'source_id': self.source_id, 'flux': _list_to_array(self.flux),
                         'flux_error': _list_to_array(self.error)}
        if self.covariance is not None:
            full_correlation = correlation_from_covariance(self.covariance)
            spectrum_dict['correlation'] = full_correlation[np.tril_indices(full_correlation.shape[0], k=-1)]
        return spectrum_dict

    def _sampling_to_dict(self):
        """
        Represent the sampling grid as a dictionary.

        Returns:
            dict: A dictionary populated with the sampling grid used for this spectrum.
        """
        return {'pos': _list_to_array(self.pos)}
