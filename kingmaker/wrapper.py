from typing import Any, Dict, List, Optional, Tuple, Union
import numpy.typing as npt

from os.path import exists
import logging
import numpy as np
import healpy as hp
from scipy.interpolate import interpn

from .pdf import InterpolatedKingPDF, TemplateSmearedKingPDF
from .fitting import KingPSFFitter


class KingSpatialLikelihood:
    """Wrapper class to encapsulate King distribution functionality, including PDF evaluation and parameter fitting.
    This class provides a unified interface for working with King distributions, allowing users to easily fit simulation
    and evaluate the PDF over events for likelihood calculations.

    Users create an instance of KingSpatialLikelihood by passing in parameters, binning, and simulated events. The
    class then fits King distribution parameters using the requested parameter binning. From this point onward, users
    only need to call either the "PDF" or "template" evaluation methods with their events to obtain likelihood values
    for their analyses. These methods interpolate the fitted King distribution parameters per-event using the event's
    observable parameters and the provided binning, and then evaluate the PDF or template-smoothed PDF at the event's
    reconstructed equatorial position.
    """
    multiple_source_warning_logged: bool = False

    def __init__(
        self,
        signal_events: npt.NDArray[Any],
        parametrization_bins: Dict[str, Union[int, List, Tuple, npt.NDArray]],
        dpsi_nbins: int = 101,
        minimum_counts: int = 100,
        spectral_indices: Optional[Union[List[float], npt.NDArray[np.floating]]] = [
            1.0,
            2.0,
            3.0,
            4.0,
        ],
        angular_cutoff: float = np.pi,
        source_ras: Optional[Union[float, npt.NDArray[np.floating]]] = None,
        source_decs: Optional[Union[float, npt.NDArray[np.floating]]] = None,
        skymap: Union[npt.NDArray[np.floating], None] = None,
        interpolated_pvalues: bool = False,
        cache_parameters: bool = True,
        cache_name: Optional[str] = "king_parameters_cache.npz",
    ):
        self.spectral_indices = spectral_indices
        self.angular_cutoff = angular_cutoff
        self.source_ras = source_ras
        self.source_decs = source_decs
        self.skymap = skymap

        # Have some place to cache the per-event information so we don't need to
        # recalculate it every time we evaluate the PDF.
        self.event_distances: Union[npt.NDArray[np.floating], List[float]] = []
        self.event_alpha: Dict[float, npt.NDArray[np.floating]] = {}
        self.event_beta: Dict[float, npt.NDArray[np.floating]] = {}

        # Also have a place to cache the per-event PDF values for each spectral
        # index so we can simply call or interpolate pvalues instead of calculating
        # them each time. This will be filled most important in the case of the
        # template-smoothed PDF, which is more expensive to evaluate, but it should
        # help with the computational efficiency of the standard PDF as well.
        self.interpolated_pvalues = interpolated_pvalues
        self.event_pvalue: Dict[float, Union[List, npt.NDArray[np.floating]]] = {}

        # Fit the King distribution parameters for all bins. If we're caching parameters
        # and a cache file exists, load from the cache instead of fitting.
        fitted_parameters : Dict[str, Dict[str, Union[List, npt.NDArray[np.floating]]]] = {}
        if cache_parameters and (cache_name is not None) and exists(cache_name):
            fitted_parameters_npz = np.load(cache_name)
            for key in fitted_parameters_npz.files:
                fitted_parameters[key] = fitted_parameters_npz[key]
        else:
            fitter = KingPSFFitter(
                signal_events,
                parametrization_bins,
                dpsi_nbins,
                minimum_counts,
                spectral_indices,
                angular_cutoff,
            )
            fitted_parameters = fitter.fit_all_bins(verbose=True)
            if cache_parameters and (cache_name is not None):
                np.savez(cache_name, **fitted_parameters)

        self.parametrization_bins: Dict[str, npt.NDArray[np.floating]]
        self.parametrization_bins = fitted_parameters["parametrization_bins"]
        self.alpha_values: npt.NDArray[np.floating] = fitted_parameters["alpha"]
        self.beta_values: npt.NDArray[np.floating] = fitted_parameters["beta"]

        # Instantiate the PDF object. If we have a template, use the template-smoothed
        # PDF; otherwise, use the standard interpolated PDF.
        self.pdf: Union[InterpolatedKingPDF, TemplateSmearedKingPDF]
        if self.skymap is not None:
            self.nside = hp.npix2nside(len(self.skymap))
            self.pdf = TemplateSmearedKingPDF(skymap = self.skymap,
                                              angular_cutoff=self.angular_cutoff)
        else:
            self.pdf = InterpolatedKingPDF(angular_cutoff=self.angular_cutoff)

        return

    def calculate_parameters(self, events: npt.NDArray[Any]) -> None:
        """Calculate the King distribution parameters (alpha and beta) for a given set of events by interpolating
        the fitted parameters based on the event parameters and the provided binning.
        """
        # Extract the bin edges and keys for interpolation
        keys, bins = [], []
        for item in self.parametrization_bins.items():
            keys.append(item[0])
            bins.append(item[1])

        # Get the event parameter values for each parameter
        event_param_values = np.array([events[key] for key in keys]).T
        
        # If we're using a skymap, then we can calculate the bin
        # index for each event now.
        if self.skymap is not None:
            self.map_index = hp.ang2pix(self.nside, events["dec"], events["ra"], lonlat=True)
        # Otherwise, we can calculate the angular separation for each event now.
        else:
            if len(self.source_ras) > 1:
                # TODO: I'm not sure if this will work with multiple sources. It might
                # need to be reshaped first? Need to check it!
                logging.log_warn("Multiple source positions provided. This has not been tested and"
                                 " may not work as expected. Please check the results carefully!")
                self.multiple_source_warning_logged = True
            self.event_distances = self.pdf.angular_separation(
                events["ra"], events["dec"], self.source_ras, self.source_decs
            )

        # Interpolate to get the alpha and beta values for each spectral index for
        # each event in the given sample.
        for i, gamma in enumerate(self.spectral_indices):
            self.event_alpha[gamma] = interpn(
                bins, self.alpha_values[i], event_param_values, bounds_error=False, fill_value=np.pi
            )
            self.event_beta[gamma] = interpn(
                bins, self.beta_values[i], event_param_values, bounds_error=False, fill_value=100
            )

        # If we're going to do interpolated pvalues as a function of spectral index instead of 
        # direct PDF evaluation at runtime, we can calculate and cache those values here as well.
        if self.interpolated_pvalues:
            if self.skymap is not None:
                self.event_pvalue[gamma] = self.pdf.pdf(self.event_distances,
                                                        self.event_alpha[gamma],
                                                        self.event_beta[gamma])
            else:
                # TODO: How can I calculate the pvalues for the PDF-convolved skymap without calculating
                # each event individually? This needs to be done efficiently, but I'm not yet sure how
                # to do that. I suppose I could calculate the alm values for a grid of alpha/beta values?
                # That should at least be less expensive to store in memory and still be reasonable efficient
                # if I need to do evaluations, but it's still going to be slower than I'd like.
                raise NotImplementedError("Interpolated pvalues for the template-smoothed PDF are not yet implemented.")


        return

    def evaluate_pdf(self, events: npt.NDArray[Any], gamma: float = 2) -> npt.NDArray[np.floating]:
        # If we haven't already calculated the per-event alpha and beta parameters, do so now.
        if len(self.event_alpha) == 0 or len(self.event_beta) == 0:
            self.calculate_parameters(events)

        if gamma in self.spectral_indices:
            # If the requested gamma is one of the fitted spectral indices, we can directly use those parameters.
            alpha = self.event_alpha[gamma]
            beta = self.event_beta[gamma]
        else:
            # Otherwise we have to interpolate the gamma values.
            alpha = np.interp(
                gamma, self.spectral_indices, [self.event_alpha[g] for g in self.spectral_indices]
            )
            beta = np.interp(
                gamma, self.spectral_indices, [self.event_beta[g] for g in self.spectral_indices]
            )

        return self.pdf.pdf()

        # Evaluate the PDF for each event using the interpolated alpha and beta values
