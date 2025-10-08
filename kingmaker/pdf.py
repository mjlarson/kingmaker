import numpy as np
import healpy as hp
from scipy.integrate import trapezoid
from scipy.special import legendre_p_all

from .distribution import _log10pi
from .distribution import _norm, _unnormalized_pdf, _unnormalized_cdf
from .utils import _interp2d, adaptive_bins, angular_distance, get_Ylm, map2nside, meshgrid2d 

class KingPDF():
    def __init__(self, *, angular_cutoff=np.pi):
        self.angular_cutoff = angular_cutoff

    def norm(self, alpha, beta):
        return _norm(alpha, beta, self.angular_cutoff)
    
    def pdf(self, x, alpha, beta):        
        # Scalar-like: check if we can shortcut using the angular cutoff
        if np.isscalar(x):
            if x > self.angular_cutoff:
                return 0
        elif len(x) == 1:
            if x[0] > self.angular_cutoff:
                return 0

        # Broadcast
        x, alpha, beta = np.broadcast_arrays(x, alpha, beta)

        # And mask
        normalized_pdf = np.zeros_like(x)
        mask = (x <= self.angular_cutoff)
        x, alpha, beta = x[mask], alpha[mask], beta[mask]

        # Nope. Do the calculations.
        norm = self.norm(alpha, beta)
        unnormalized_pdf = _unnormalized_pdf(x, alpha, beta)
        normalized_pdf[mask] = norm * unnormalized_pdf
        
        return normalized_pdf

    def cdf(self, x, alpha, beta):
        # Scalar-like: check if we can shortcut using the angular cutoff
        if np.isscalar(x):
            if x > self.angular_cutoff:
                return 0
        elif len(x) == 1:
            if x[0] > self.angular_cutoff:
                return 0

        # Broadcast
        x, alpha, beta = np.broadcast_arrays(x, alpha, beta)
        
        # And mask
        normalized_cdf = np.ones_like(x)
        mask = (x <= self.angular_cutoff)
        x, alpha, beta = x[mask], alpha[mask], beta[mask]

        # Nope. Do the calculations.
        norm = self.norm(alpha, beta)
        unnormalized_cdf = _unnormalized_cdf(x, alpha, beta)
        normalized_cdf[mask] = norm * unnormalized_cdf
        
        return normalized_cdf

    def marginalize(self, dec, alpha, beta, threshold=1e-6, nbins=None):
        # Generate the sindec binning        
        sindec = np.sin(dec)
        if nbins is None:
            delta = min(0.1, (np.sin(dec+alpha)-np.sin(dec-alpha))/2)
            sindec_bins = np.concatenate([np.arange(-1, sindec, delta),
                                          np.arange(sindec, 1, delta)])
        else:
            sindec_bins = np.concatenate([[sindec, 1],
                                          np.linspace(-1, 1, nbins)])
            sindec_bins = np.unique(sindec_bins)
    
        # Mask unnecessary declination bins
        pdf_check = self.pdf(np.abs(dec-np.arcsin(sindec_bins)), alpha, beta)
        decmask = (pdf_check/pdf_check.max() > threshold)
    
        # Generate a binning in RA from 0 to 2pi.
        nbins = int(2*np.pi/alpha*5)
        ra_bins = np.linspace(0, 2*np.pi, nbins)
    
        # Mask unnecessary RA bins
        pdf_check = self.pdf(ra_bins, alpha, beta)
        ramask =  (pdf_check/pdf_check.max() > threshold)
        ra_bins = ra_bins[ramask]
    
        # Calculate the PDF at each grid point
        ra_points, dec_points = meshgrid2d(ra_bins, np.arcsin(sindec_bins[decmask]))
        distance = angular_distance(0, dec, ra_points, dec_points)
        pdf = self.pdf(distance, alpha, beta)
    
        # Integrate over RA using a simple trapezoid rule integration.
        marginalized = np.zeros_like(sindec_bins)
        marginalized[decmask] = ((pdf[:,:-1]+pdf[:,1:])/2 * np.diff(ra_bins)[None,:]).sum(axis=1)
        return sindec_bins, marginalized

class InterpolatedKingPDF(KingPDF):
    def __init__(self, 
                 *,
                 angular_cutoff=np.pi,
                 points_alpha=np.logspace(-5, _log10pi+1e-2, 200),
                 points_beta=np.logspace(0, 1, 200)):
        super().__init__(angular_cutoff=angular_cutoff)
        self.points_alpha, self.points_beta = points_alpha, points_beta
        grid_alpha, grid_beta = np.meshgrid(points_alpha, points_beta)
        self.log10_grid_norms = np.log10(_norm(grid_alpha, grid_beta, self.angular_cutoff).T)
    
    def norm(self, alpha, beta):
        return 10**_interp2d(alpha,
                             beta,
                             self.points_alpha, 
                             self.points_beta, 
                             self.log10_grid_norms)


class TemplateSmearedKingPDF(InterpolatedKingPDF):
    def __init__(self,
                 skymap,
                 eval_decs,
                 eval_ras,
                 *,
                 angular_cutoff=np.pi,
                 points_alpha=np.logspace(-5, _log10pi, 200),
                 points_beta=np.logspace(0, 1, 200),
                 lmax=None,
                 scheme="RING",
                 ):
        super().__init__(angular_cutoff=angular_cutoff, 
                         points_alpha=np.logspace(-5, _log10pi, 200),
                         points_beta=np.logspace(0, 1, 200))
        self.nside = hp.npix2nside(len(skymap))
        self.lmax = (3*self.nside-1) if lmax is None else lmax
        self.mmax = self.lmax

        # Set up the evaluation coordinates
        self.set_coordinates(eval_decs, eval_ras)

        # While we're here, normalize the skymap so it integrates to 1.
        self.skymap = skymap
        self.skymap /= (self.skymap * hp.nside2resol(self.nside)).sum()
        self.skymap_alm = self.skymap_to_alm()

    def set_coordinates(self, eval_decs, eval_ras):
        self._eval_decs = np.atleast_1d(eval_decs)
        self._eval_ras = np.atleast_1d(eval_ras)
        assert self._eval_decs.shape == self._eval_ras.shape
        
        # Evaluate the Y_lm harmonic functions now so we can use them later without cost.
        # The theta, phi here are co-latitude and azimuth/RA, both in radians.
        self._Y_lm = get_Ylm(self.lmax, self.mmax, 
                             self._eval_decs-np.pi/2, self._eval_ras)
        return

    def skymap_to_alm(self):
        return hp.map2alm(self.skymap, lmax=self.lmax, mmax=self.mmax)

    def get_king_b_l(self, alpha, beta, npoints=100, scale=np.pi/4):
        theta = adaptive_bins(alpha, npoints, scale=scale)
        P_all = legendre_p_all(self.lmax, np.cos(theta))[0]
        pdf_vals = self.pdf(theta, alpha, beta)
        return 2*np.pi*trapezoid(P_all * pdf_vals * np.sin(theta), dx=np.diff(theta))
    
    def convolve_map(self, alpha, beta):
        b_l = self.get_king_b_l(alpha, beta)
        harmonic_convolution = hp.almxfl(self.skymap_alm.copy(),
                                         b_l,
                                         self.lmax,
                                         self.mmax)
        return hp.alm2map(harmonic_convolution,
                          nside=self.nside,
                          lmax=self.lmax,
                          mmax=self.mmax)
        
    def convolve_at_grid_point(self, alpha, beta, npoints=100, scale=np.pi/4):
        b_l = self.get_king_b_l(alpha, beta, npoints=npoints, scale=scale)
        harmonic_convolution = hp.almxfl(self.skymap_alm.copy(),
                                         b_l,
                                         self.lmax,
                                         self.mmax)
        if len(self._Y_lm.shape) > 1:
            return (np.real(harmonic_convolution[:, None] * self._Y_lm).sum(axis=0))
        else:
            return (np.real(harmonic_convolution * self._Y_lm).sum(axis=0))

    def marginalize(self, dec, alpha, beta, threshold=1e-6):
        raise NotImplementedError("Signal subtraction for templates is not implemented.")


        
# class KingSpatialPDF:
#     def __init__(self,
#                  grid_alpha=np.logspace(-3, np.log10(np.pi), 200),
#                  grid_beta=np.linspace(1, 10, 100),
#                  grid_sindec_true=np.linspace(-1, 1, 20),
#                  grid_sindec_rec=np.linspace(-1, 1, 20)):
#         """
#         Initialize a KingSpatialPDF instance with a grid of alpha and beta values
#         used for interpolating King function normalization constants.

#         Parameters
#         ----------
#         grid_alpha : ndarray
#             Grid of alpha values (in radians) used for normalization interpolation.
#         grid_beta : ndarray
#             Grid of beta values used for normalization interpolation.
#         grid_sindec_true, grid_sindec_rec : ndarray
#             Grid of true/reconstructed declination points to use when pre-calculating the
#             signal-subtracted likelihood. These are only used when evaluating a likelihood
#         """
#         self._grid_alpha = grid_alpha
#         self._grid_beta = grid_beta
#         self._grid_sindec_true = grid_sindec_true
#         self._grid_sindec_rec = grid_sindec_rec
#         self._precompute_norms()
#         self._precompute_cdf_isf()
#         self._precompute_marginalization_over_ra()



#     def _precompute_norms(self):
#         """
#         Precompute normalization constants over a grid of (alpha, beta)
#         values using the _raw_norm method. Results are stored for interpolation
#         in the `norm` method.
#         """
#         self._grid_norms = self._raw_norm(self._grid_alpha, self._grid_beta)

#     def _precompute_cdf_isf(self, n_psi=200, n_prob=200):
#         """
#         Precompute the King CDF over a grid of (psi, alpha, beta).
#         Normalization is done by dividing by the cumulative value at psi = pi.
#         """
#         self._grid_psi = np.logspace(-3, np.log10(np.pi), n_psi) # Note: this is in radians
#         psi_vals = self._grid_psi  # shape (n_psi,)

#         alpha_grid = self._grid_alpha[:, None, None]  # (n_alpha, 1, 1)
#         beta_grid = self._grid_beta[None, :, None]    # (1, n_beta, 1)
#         psi_grid = psi_vals[None, None, :]            # (1, 1, n_psi)

#         # Evaluate unnormalized PDF and include solid angle element
#         integrand = self._unnormalized_pdf(psi_grid, alpha_grid, beta_grid) * np.sin(psi_grid)

#         # Integrate using trapezoidal rule along psi axis
#         cdf_vals = cumulative_trapezoid(integrand, psi_vals, axis=-1, initial=0.0)
#         cdf_vals = np.maximum.accumulate(cdf_vals, axis=-1)

#         # Normalize each (alpha, beta) slice by the value at psi = pi
#         norm = cdf_vals[..., -1]
#         with np.errstate(divide='ignore', invalid='ignore'):
#             cdf_vals /= norm[..., None]

#         # Build interpolator: axes are (alpha, beta, psi)
#         self._cdf_interp = RegularGridInterpolator(
#             (self._grid_alpha, self._grid_beta, self._grid_psi),
#             cdf_vals,
#             bounds_error=False,
#             fill_value=1.0  # CDF should approach 1 outside the domain
#         )

#         # Now construct inverse survival function
#         # We define a fixed grid in [0,1] for survival (1 - cdf)
#         grid_prob = np.logspace(-10, 0, n_prob)
#         inv_surv = np.zeros((len(self._grid_alpha), len(self._grid_beta), n_prob))

#         # Now broadcast grid_prob for each (alpha, beta)
#         # Output: shape (n_alpha * n_beta, n_prob)
#         n_alpha, n_beta = len(self._grid_alpha), len(self._grid_beta)
#         cdf_flat = cdf_vals.reshape(n_alpha * n_beta, n_psi)
#         isf_flat = np.empty((n_alpha * n_beta, n_prob))

#         # psi is 1D, same for all (alpha, beta). Interpolate to
#         # the common axis in probability. Note that we need to flip
#         # the order of some axes to satisfy np.interp's monotonicity
#         # requirement.
#         for k in range(n_alpha * n_beta):
#             sf = 1 - cdf_flat[k]
#             isf_flat[k] = np.interp(grid_prob, sf[::-1], self._grid_psi[::-1],
#                                     left=self._grid_psi[0], right=self._grid_psi[-1])

#         # Reshape back to (n_alpha, n_beta, n_p)
#         isf_vals = isf_flat.reshape(n_alpha, n_beta, n_prob)

#         # Save inverse survival interpolator
#         self._isf_interp = RegularGridInterpolator(
#             (self._grid_alpha, self._grid_beta, grid_prob),
#             isf_vals,
#             bounds_error=True,  # Enforce 0 < p < 1
#         )

#         self._grid_cdf_p = grid_prob

#     def _precompute_marginalization_over_ra(self, n_ra=360):
#         """
#         Precompute King PDF marginalized over RA for grids of (dec_true, dec_rec, alpha, beta).
#         Stores results for 4D interpolation in `marginalization_over_ra_interp` for use in
#         signal-subtracted likelihood evaluations
#         """
#         grid_sindec_true = self._grid_sindec_true
#         grid_sindec_rec = self._grid_sindec_rec

#         alpha_grid, beta_grid = np.meshgrid(self._grid_alpha, self._grid_beta, indexing='ij')
#         alpha_flat = alpha_grid.ravel()
#         beta_flat = beta_grid.ravel()
#         n_params = alpha_flat.shape[0]

#         ra_vals = np.linspace(0, 2*np.pi, n_ra, endpoint=False)
#         cos_ra = np.cos(ra_vals)

#         # Pre-allocate result: shape (N_dec_true, N_dec_rec, N_alpha, N_beta)
#         shape = (len(grid_sindec_true), len(grid_sindec_rec),
#                  len(self._grid_alpha), len(self._grid_beta))
#         pdf_vals = np.zeros(shape)

#         for i, sin_dec_true in enumerate(grid_sindec_true):
#             # Precompute trig terms for current dec_true
#             cos_dec_true = np.sqrt(1 - sin_dec_true**2)

#             for i_rec, sin_dec_rec in enumerate(grid_sindec_rec):
#                 cos_dec_rec = np.sqrt(1 - sin_dec_rec**2)

#                 # Compute angular distances psi for each RA
#                 cos_psi = sin_dec_true * sin_dec_rec + cos_dec_true * cos_dec_rec * cos_ra
#                 cos_psi = np.clip(cos_psi, -1, 1)
#                 psi = np.arccos(cos_psi)  # shape (N_ra,)

#                 # Evaluate PDF for all (alpha, beta) pairs at this psi[None, :]
#                 psi_exp = psi[None, :]           # shape (1, N_ra)
#                 alpha_exp = alpha_flat[:, None]  # (N_params, 1)
#                 beta_exp = beta_flat[:, None]    # (N_params, 1)

#                 # Evaluate King PDF → shape (N_params, N_ra)
#                 pdf_ra = self.pdf(psi_exp, alpha_exp, beta_exp)

#                 # Average over RA: → shape (N_params,)
#                 mean_pdf = np.mean(pdf_ra, axis=1)

#                 # Normalize over sin(dec_rec) later — after filling all i_rec
#                 pdf_vals[i, i_rec, :, :] = mean_pdf.reshape(len(self._grid_alpha), len(self._grid_beta))

#             # Normalize over sin(dec_rec)
#             pdf_slice = pdf_vals[i, :, :, :]  # shape (N_rec, N_alpha, N_beta)
#             norm = np.trapz(pdf_slice, grid_sindec_rec, axis=0)  # shape (N_alpha, N_beta)

#             with np.errstate(divide='ignore', invalid='ignore'):
#                 normed_pdf = np.where(norm > 0, pdf_slice / norm[None, :, :], 0.0)

#             pdf_vals[i, :, :, :] = normed_pdf

#         self._grid_pdf_marginalized = pdf_vals
#         self._marginalized_pdf_interp = RegularGridInterpolator(
#                     (self._grid_sindec_true,
#                      self._grid_sindec_rec,
#                      self._grid_alpha,
#                      self._grid_beta),
#                     self._grid_pdf_marginalized,
#                     bounds_error=False,
#                     fill_value=0.0)


#     def norm(self, alpha, beta):
#         """
#         Calculate the King function normalization constant for given (alpha, beta)
#         using interpolation over precomputed values on a grid.

#         Parameters
#         ----------
#         alpha : float or ndarray
#             King distribution alpha parameter (scale).
#         beta : float or ndarray
#             King distribution beta parameter (tail weight).

#         Returns
#         -------
#         ndarray
#             Normalization constant(s) for the King PDF in units of probability/sterradian.
#         """
#         alpha, beta = np.asarray(alpha), np.asarray(beta)
#         points = np.array([alpha.ravel(), beta.ravel()]).T
#         out = interpn((self._grid_alpha, self._grid_beta),
#                         self._grid_norms,
#                         points,
#                         method='linear',
#                         bounds_error=False,
#                         fill_value=None)
#         return out.reshape(alpha.shape)

#     def pdf(self, x, alpha, beta, norm=None, normalize_over_distance=False):
#         """
#         Evaluate the King probability density function over angular distance.

#         This returns a PDF normalized over solid angle,
#         suitable for use in unbinned likelihoods on the sphere.

#         Parameters
#         ----------
#         x : float or ndarray
#             Angular separation from the source, in radians.
#         alpha : float or ndarray
#             King distribution alpha parameter (scale).
#         beta : float or ndarray
#             King distribution beta parameter (tail weight).

#         Returns
#         -------
#         ndarray
#             Normalized King PDF values at the specified angles.
#         """
#         p = self._unnormalized_pdf(x, alpha, beta)
#         if norm is not None:
#             p *= norm
#         else:
#             p *= self.norm(alpha, beta)
#         if normalize_over_distance:
#             p *= 2*np.pi*np.sin(x)
#         return p

#     def cdf(self, psi, alpha, beta):
#         """
#         Evaluate the King CDF by interpolation.

#         Parameters
#         ----------
#         psi : float or array_like
#             Angular distance(s) in radians.
#         alpha : float or array_like
#             Alpha parameter(s).
#         beta : float or array_like
#             Beta parameter(s).

#         Returns
#         -------
#         ndarray
#             CDF values.
#         """
#         psi = np.asarray(psi)
#         alpha = np.asarray(alpha)
#         beta = np.asarray(beta)

#         # Broadcast inputs and reshape for interpolation
#         psi, alpha, beta = np.broadcast_arrays(psi, alpha, beta)
#         pts = np.stack([alpha.ravel(), beta.ravel(), psi.ravel()], axis=-1)

#         cdf_vals = self._cdf_interp(pts).reshape(psi.shape)
#         return cdf_vals

#     def isf(self, prob, alpha, beta):
#         """
#         Evaluate the King ISF by interpolation.

#         Parameters
#         ----------
#         prob : float or array_like
#             Probability remaining in the tail
#         alpha : float or array_like
#             Alpha parameter(s).
#         beta : float or array_like
#             Beta parameter(s).

#         Returns
#         -------
#         ndarray
#             CDF values.
#         """
#         prob = np.asarray(prob)
#         alpha = np.asarray(alpha)
#         beta = np.asarray(beta)

#         # Broadcast inputs and reshape for interpolation
#         prob, alpha, beta = np.broadcast_arrays(prob, alpha, beta)
#         pts = np.stack([alpha.ravel(), beta.ravel(), prob.ravel()], axis=-1)

#         psi_vals = self._isf_interp(pts).reshape(prob.shape)
#         return psi_vals

#     def marginalized_over_ra(self, ev_sindec, true_sindec, alpha, beta):
#         """
#         Interpolated version of King PDF marginalized over RA for use in signal-subtracted
#         likelihood calculations.

#         Parameters
#         ----------
#         ev_sindec : ndarray
#             Array of reconstructed declination values.
#         true_sindec : float
#             True sin(declination).
#         alpha : float
#             Alpha parameter.
#         beta : float
#             Beta parameter.

#         Returns
#         -------
#         pdf : ndarray
#             Interpolated marginalized King PDF values.
#         """
#         # Interpolation points: shape (N, 4)
#         pts = np.column_stack([
#             np.full_like(ev_sindec, true_sindec),
#             ev_sindec,
#             np.full_like(ev_sindec, alpha),
#             np.full_like(ev_sindec, beta)
#         ])

#         return self._marginalized_pdf_interp(pts)


    








# class KingPointSourceSpacePDFRatioModel(AccWeightedPDFRatioModel):
#     r"""Space PDF ratio model for one or more point or point-like sources using the King distribution

#     This class implements the space PDF ratio using the King distribution for the signal
#     spatial PDFs. This PSF model is used in the gamma-ray community and provides a better
#     description of our point spread function than the historically used 2D Gaussian PDF.
#     The PDF for this function takes the form

#         f(x; \alpha, \beta) = \frac{1}{2\pi\beta^2} \left(1-\frac{1}{\alpha}\right)\left(1+\frac{x^2}{2\alpha \beta^2}\right)^{-\alpha}

#     For a description of this distribution, see

#     https://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/psf/psf_king/index.html

#     The source list should give the coordinates :math:`(\alpha,\delta)` for one or more
#     sources; multiple sources will result in a stacking analysis.
#     """
#     def __init__(self, ana, src, bg_param=None, acc_param=None,
#                  fit_cut_n_sigma=5, cut_pdf_threshold=1,
#                  sigsub=False, angular_floor=0.2,
#                  mp_cpus=1,
#                  sindec_bandwidth=np.radians(1),
#                  king_gammas = np.r_[1.:5:1],
#                  parametrization_bins={"log10energy":10, "dec":10, "sigma":10},
#                  dpsi_nbins=101, minimum_counts=100, n_sigma_max=3,
#                  diagnostic_plots=False):
#         """Construct a KingPointSourceSpacePDFRatioModel.

#         Args:
#             ana (:class:`analysis.SubAnalysis`): the sub analysis
#             src (:class:`utils.Sources`): the source list.  required keys: ``ra`` and ``dec``.
#                  optional keys: ``weight``, ``extension``
#             bg_param: the background space PDF parameterization
#             acc_param: the signal acceptance parameterization
#             fit_cut_n_sigma (float): The maximum distance (in units of sigma) for events to be included in fits.
#                 Note that setting this to include all events could cause problems.
#             cut_pdf_threshold (float): the minimum pvalue below which we ignore events during likelihood
#                 evaluation
#             sigsub (bool): whether signal subtraction will be enabled
#             angular_floor (float): The minimum distance to use in spatial likelihood evaluation in degrees.
#                 Note that this does not affect the construction of the King PDF parameters: only evaluation.
#             mp_cpus (int): The number of CPU cores to use when generating the King parametrizations.
#             sindec_bandwidth (float): The true distance away from each source to select signal events for the signal subtraction calculations
#             king_gammas (array-like): Spectral indices to use in King parametrization fitting
#             parametrization_bins (dict{observable:bins}): A dictionary of bins to use when fitting the King models.
#             dpsi_nbins (int): The number of bins number of bins to use in true angular distance
#                 when fitting the King models.
#             minimum_counts (int): The minimum number of events required in a bin for parametrization.
#             n_sigma_max (float): A limit on the range of weight values to include. Increasing this too high
#                 will introduce events with large weights, increasing the uncertainty on the King fit parameters.
#             diagnostic_plots (bool): Draw diagnostic plots directly using matplotlib. The histograms
#                 are stored internally in the model.histograms, model.uncertainties, and model.dpsi_bins.
#         """
#         self.ana, self.src = ana, src
#         self.bg_param = bg_param = bg_param or ana.bg_space_param
#         self.acc_param = acc_param = acc_param or ana.acc_param
#         self._acc_model = self.AccModel(src, acc_param, ana.livetime)
#         self.fit_cut_n_sigma = fit_cut_n_sigma
#         self.cut_pdf_threshold = cut_pdf_threshold
#         self.sigsub = sigsub
#         self.angular_floor = angular_floor
#         self.mp_cpus = mp_cpus
#         self.king_gammas = king_gammas
#         self.minimum_counts = minimum_counts
#         self.dpsi_nbins = dpsi_nbins
#         self.n_sigma_max = n_sigma_max

#         # Build the signal subtraction parametrization
#         self.sigsub_param = SigSubSinDecParameterization(src, ana.sig, log=False)

#         #if len(self.king_gammas) > 1:
#         #    raise NotImplementedError("Interpolating across gamma isn't included yet.")

#         #-----------------------------
#         # Build the list of values to keep. Explicitly include anything in the dimension_bins
#         #-----------------------------
#         self.keep = list(set(['ra','dec','king_alpha','king_beta','king_norm'] + list(parametrization_bins.keys())))
#         for k in bg_param.keep:
#             if k not in self.keep:
#                 self.keep.append(k)

#         #-----------------------------
#         # Get the mc and angular distances we need for paramatrizations
#         # Use the standard csky method of calculating angular distances,
#         # even if I know that it's not accurate at larger angles.
#         #-----------------------------
#         sig = self.ana.sig
#         dpsi = cext.delta_angle(sig.dec, sig.ra,
#                                 sig.true_dec, sig.true_ra)

#         if np.any(dpsi >= np.deg2rad(75)):
#             print("Warning: At least some of your MC events have angular errors"
#                   " of 75 degrees or more. Csky uses an approximation for angular"
#                   " distances which is off by ~10% at 75 degrees and the calculation"
#                   " error increases to nearly 80% around 100 degrees. Consider"
#                   " removing these events. If you keep them, be aware that their"
#                   " spatial likelihood values will potentially be incorrect.")

#         #-----------------------------
#         # Instantiate the binning and assign events to each bin
#         # And ensure that all keys we want are valid observables
#         #-----------------------------
#         available = set(sig.keys()) & set(self.ana.data.keys())
#         for key in parametrization_bins.keys():
#             if not hasattr(sig, key):
#                 msg = ("User requested a King parametrization built with variable"
#                        f" {key} which doesn't appear to be in the MC. Please provide"
#                         " only variables that are available in both the data and MC:"
#                        f"{available}")
#                 raise RuntimeError(msg)
#             if not hasattr(self.ana.data, key):
#                 msg = ("User requested a King parametrization built with variable"
#                        f" {key} which doesn't appear to be in the data. Please provide"
#                         " only variables that are available in both the data and MC:"
#                        f"{available}")
#                 raise RuntimeError(msg)

#         # Are there any spatially extended sources?
#         self.fit_extensions = False
#         self.extensions = np.unique(src.extension)
#         if np.any(self.extensions>0):
#             print(f"* Found source spatial extensions {self.extensions}")
#             self.fit_extensions = True

#         self.parametrization_bins = {}
#         for key, val in parametrization_bins.items():
#             if isinstance(val, int):
#                 self.parametrization_bins[key] = self._get_bins(int(val), sig[key])
#             elif any(isinstance(val, _) for _ in [tuple, list, np.array]):
#                 self.parametrization_bins[key] = val
#             else:
#                 msg = (f"Unknown binning argument for parameter {key}: Unable"
#                        f" to interpret binning {val} of type {type(val)}. Use"
#                        " an integer for the number of bins or specify the binning"
#                        " directly using a tuple, list, or np.array")
#                 raise RuntimeError(msg)
#         self.parametrization_shape = [len(b) for b in self.parametrization_bins.values()]

#         # Bin the events
#         event_indices = {key:np.digitize(sig[key], b)
#                          for key, b in self.parametrization_bins.items()}

#         #-----------------------------
#         # Begin the fitting process
#         #-----------------------------
#         print("Building the KingSpatialPDF objects", end="")
#         self.king = KingSpatialPDF(grid_sindec_rec=self.bg_param.hkw['bins'],
#                                    grid_sindec_true=np.linspace(-1, 1, 51),
#                                   )
#         print(" -- Done!")

#         print('Getting King parametrizations for {}...'.format(ana.key))
#         self.alpha_interp = [[None for _ in range(len(self.king_gammas))]
#                                    for __ in range(len(self.extensions))]
#         self.beta_interp = [[None for _ in range(len(self.king_gammas))]
#                                   for __ in range(len(self.extensions))]
#         self.fit_alpha =  np.ones([len(self.king_gammas), len(self.extensions)]
#                                     + self.parametrization_shape) * np.median(dpsi)
#         self.fit_beta  = np.ones_like(self.fit_alpha) * 2.25
#         self.histograms = np.zeros(list(self.fit_alpha.shape) + [self.dpsi_nbins,], dtype=float)
#         self.uncertainties = np.zeros(list(self.fit_alpha.shape) + [self.dpsi_nbins,], dtype=float)
#         self.dpsi_bins = np.zeros(list(self.fit_alpha.shape) + [self.dpsi_nbins+1,], dtype=float)

#         print("Starting fitting King functions")
#         for e_index, extension in enumerate(self.extensions):
#             if self.fit_extensions:
#                 print("\r  * Extension = {:.4f}".format(np.rad2deg(extension)), end="")
#             if extension > 0:
#                 # We're doing extended sources. Add some variation to the true positions
#                 # using a 2d Gaussian.
#                 delta_dec, delta_ra = np.random.multivariate_normal([0,0],
#                                                                    np.eye(2)*extension**2,
#                                                                    size=len(sig)).T
#                 # And recalculate the dpsi to the new "true" locations
#                 dpsi = cext.delta_angle(sig.dec, sig.ra,
#                                         sig.true_dec + delta_dec,
#                                         sig.true_ra + delta_ra)
#             for g_index, gamma in enumerate(self.king_gammas):
#                 print("  * gamma = {:.4f} ...".format(gamma), end="")
#                 sys.stdout.flush()

#                 w = sig.oneweight * sig.true_energy**-gamma
#                 for bin_indices in np.ndindex(*self.parametrization_shape):
#                     mask = np.ones(len(w), dtype=bool)

#                     for i, key in enumerate(self.parametrization_bins.keys()):
#                         mask &= (event_indices[key] == bin_indices[i])

#                     # Trim outliers. These aren't inherently problematic as long we
#                     # we include uncertainties, but removing them should make the
#                     # calculation a little more stable.
#                     if mask.sum() < self.minimum_counts: continue

#                     # We have our events. Histogram them and get the correct uncertainties
#                     masked_sig = sig[mask]
#                     masked_dpsi = dpsi[mask]
#                     masked_w = w[mask] / w[mask].sum()
#                     dpsi_bins = self._get_bins(self.dpsi_nbins, masked_dpsi, masked_w)
#                     dpsi_bins = np.unique(dpsi_bins)
#                     dpsi_bin_centers = (dpsi_bins[:-1]+dpsi_bins[1:])/2

#                     # Fit these by normalizing over dpsi in 1d. This turns out to be useful
#                     # since the peak location in this version of the PDF corresponds to the
#                     # shape parameter alpha, so we get an excellent seed value.
#                     h, bins = np.histogram(masked_dpsi,
#                                         bins=dpsi_bins,
#                                         weights=masked_w)
#                     h2, bins = np.histogram(masked_dpsi,
#                                         bins=dpsi_bins,
#                                         weights=(masked_w)**2)
#                     delta = np.diff(bins)
#                     try:
#                         h /= delta
#                         h2 /= delta**2
#                     except FloatingPointError:
#                         print("WTF?", dpsi_bins, delta)
#                     alpha = dpsi_bin_centers[np.argmax(h)]

#                     # The fitting process seems to be somewhat sensitive to the seed. Since
#                     # the fit itself is pretty quick to run, do a scan over a small grid of
#                     # seed values and choose the one with the best final likelihood. We can
#                     # speed up the process by stopping the inner loop as soon as the likelihood
#                     # begins to rise again since then we're climbing out of the minimum.
#                     test_alphas = np.arange(0.5, 1.5, 0.25)*alpha
#                     test_betas = np.arange(1.5, 5, 0.5)
#                     popts, llhs = [], []
#                     try:
#                         for a in test_alphas:
#                             previous = np.inf
#                             for b in test_betas:
#                                 p, l = self.fit_histogram(h, bins,
#                                                            err2 = h2,
#                                                            alpha_guess=a,
#                                                            beta_guess=b)
#                                 if l > previous:
#                                     break
#                                 popts.append(p)
#                                 llhs.append(l)
#                                 previous = l
#                     except:
#                         continue
#                     popt = popts[np.argmin(llhs)]

#                     # Store the results. In python3.11+, we can just unpack the
#                     # indices directly without wrapping in a tuple, but we want
#                     # to support some earlier python versions. See PEP 646.
#                     param_idx = tuple([g_index, e_index, *bin_indices])
#                     self.fit_alpha[param_idx] = popt[0]
#                     self.fit_beta[param_idx] = popt[1]
#                     self.histograms[param_idx] = h
#                     self.uncertainties[param_idx] = np.sqrt(h2)
#                     self.dpsi_bins[param_idx] = dpsi_bins

#                     ###################################
#                     # Plot it
#                     ###################################
#                     if diagnostic_plots:
#                         fig, ax = plt.subplots()
#                         ax.plot(np.rad2deg(dpsi_bin_centers), h, label='mc', color='k', linewidth=3, zorder=800)
#                         ax.fill_between(np.rad2deg(dpsi_bin_centers),
#                                        h-np.sqrt(h2),
#                                        h+np.sqrt(h2),
#                                        color='k',
#                                        alpha=0.25, )
#                         xmax = min(8*self.fit_alpha[param_idx], np.pi)
#                         dpsi_bin_centers = np.linspace(0, xmax, 1000)
#                         fit = self.king.pdf(dpsi_bin_centers,
#                                            alpha = self.fit_alpha[param_idx],
#                                            beta = self.fit_beta[param_idx],
#                                            normalize_over_distance=True)
#                         fit *= h.max()/fit.max()
#                         ax.plot(np.rad2deg(dpsi_bin_centers), fit,
#                                label='King, best fit', linewidth=3,
#                                 color='b', linestyle='dashed', zorder=900)

#                         # The King becomes a rayleigh at very large beta. Use
#                         # that instead of pulling in extra code.
#                         from scipy.stats import rayleigh
#                         ray = rayleigh.pdf(dpsi_bin_centers,
#                                            scale=self.fit_alpha[param_idx])
#                         ray *= h.max()/ray.max()
#                         ax.plot(np.rad2deg(dpsi_bin_centers), ray,
#                                label='Rayleigh', linewidth=2,)

#                         ax.plot([], [], label=r'$\gamma$={:4.2}'.format(float(gamma)),
#                                 color='w', linewidth=0)

#                         for i, key in enumerate(self.parametrization_bins.keys()):
#                             idx = bin_indices[i]
#                             left = float(self.parametrization_bins[key][idx-1])
#                             right = float(self.parametrization_bins[key][idx])
#                             ax.plot([], [], label=f'{left:4.2}<={key}<{right:4.2}',
#                                     color='w', linewidth=0)
#                         ax.legend()
#                         ax.set_xlabel("Angular distance (degrees)")
#                         ax.set_ylabel("Density, (1/rad, weighted events)")
#                         ax.set_yscale('log')
#                         ax.grid(alpha=0.25)
#                         ax.set_ylim(fit.max()/1000, fit.max()*2)
#                         ax.set_xlim(0, np.rad2deg(xmax))
#                         #ax.set_ylim(ymin=1e-1)
#                         #if self.cut_pdf_threshold:
#                         #    ax.set_ylim(ymin = self.cut_pdf_threshold,
#                         #                ymax = min(h.max(), 1000)*1.25)

#             self.alpha_interp[e_index][g_index] = RegularGridInterpolator(
#                                                     self.parametrization_bins.values(),
#                                                     self.fit_alpha[g_index, e_index])
#             self.beta_interp[e_index][g_index] = RegularGridInterpolator(
#                                                     self.parametrization_bins.values(),
#                                                     self.fit_beta[g_index, e_index])

#     # Now interpolate all of these and store the alpha, beta, and norm for each
#     # event in the data and simulatino.
#     def set_shape_params(self, ev):
#         """
#         Interpolate King shape parameters (alpha, beta) from event observables.
#         """
#         params = self.parametrization_bins
#         values = np.array([ev[key] for key in params.keys()]).T
#         ev['king_alpha'] = self.alpha_interp(values)
#         ev['king_beta'] = self.beta_interp(values)
#         ev['king_norm'] = self.king.norm(ev['king_alpha'], ev['king_beta'])
#         return ev

#     def _get_bins(self, nbins, x, weights=None):
#         if weights is None:
#             weights = np.ones(len(x))
#         i = np.argsort(x)
#         cumulative = np.cumsum(weights[i])/weights.sum()
#         positions = np.searchsorted(cumulative, np.linspace(0, 1, nbins+1))
#         positions = np.clip(positions, 0, len(x)-1)
#         # If there are duplicates, try shifting slightly
#         p, count = np.unique(positions, return_counts=True)
#         if np.any(count>1):
#             # Fall back on unweighted
#             return self._get_bins(nbins, x)
#         xpos = x[i][positions]
#         return xpos

#     def _residuals(self, hist_vals, bin_edges, err2, alpha, beta):
#         centers = (bin_edges[:-1]+bin_edges[1:])/2.0
#         expected = self.king.pdf(centers, alpha, beta, normalize_over_distance=True)
#         expected *= hist_vals.sum()/expected.sum()
#         res = (hist_vals - expected)**2/(err2 + 1e-6)
#         return res

#     def fit_histogram(self, hist_vals, bin_edges, err2, alpha_guess=np.deg2rad(0.2), beta_guess=2.0):
#         def residuals(params):
#             alpha, beta = params
#             res = self._residuals(hist_vals, bin_edges, err2, alpha, beta)
#             return res.sum()

#         initial_guess = [alpha_guess, beta_guess]
#         result = least_squares(residuals,
#                                initial_guess,
#                                bounds=([0, 1],
#                                        [2*alpha_guess, 100])
#                               )
#         return result.x, np.sum(result.fun)

#     def set_ra(self, ra):
#         self.src["ra"] = ra

#     @property
#     def acc_model(self):
#         return self._acc_model

#     def get_updated(self, evs):
#         self = copy(self)
#         for ev in evs[1:]:
#             self.bg_param = self.bg_param + ev
#         return self

#     class AccModel(AccModel):
#         """Acceptance model for PointSourceSpacePDFRatioModel.

#         This acceptance model evaluates the time-independent acceptance for each source,
#         weights these by the per-source intrinsic weights, and finally multiplies by the
#         livetime.

#         """

#         def __init__(self, src, acc_param, livetime):
#             self.src, self.acc_param, self.livetime = src, acc_param, livetime

#         def get_acc_per_source(self, **params):
#             self.acc_param = np.atleast_1d(self.acc_param)
#             if len(self.acc_param) == 1:
#                 return self.livetime * self.src.weight * self.acc_param[0](self.src, **params)
#             acc_params = []
#             # for sources with custom or mixed types of fluxes, better to handle separately
#             for weight, acc_param, src in zip(self.src.weight, self.acc_param, self.src):
#                 acc_params.append(self.livetime * weight * acc_param(src, **params))
#             return acc_params

#     def __call__(self, ev, i=(None, None)):
#         i_ev, i_src = i
#         self.ana.sig = self.set_shape_params(self.ana.sig)
#         self.ana.data = self.set_shape_params(self.ana.data)
#         return KingPointSourceSpacePDFRatioEvaluator(ev, self, (i_ev, i_src))
