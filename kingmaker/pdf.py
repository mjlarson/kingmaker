import numpy as np
import healpy as hp
from scipy.integrate import trapezoid
from scipy.special import legendre_p_all

from .distribution import _log10pi
from .distribution import _norm, _unnormalized_pdf, _unnormalized_cdf
from .utils import _interp2d, adaptive_bins, angular_distance, get_Ylm, meshgrid2d


class KingPDF:
    """
    Calculate the probability density function (PDF) and cumulative distribution
    function (CDF) for the King spatial distribution.

    This class manages PDF and CDF evaluations with support for angular cutoffs
    and proper normalization over the sphere.

    Parameters
    ----------
    angular_cutoff : float, optional
        Maximum angular separation in radians. Default is pi (full sphere).
    """

    def __init__(self, *, angular_cutoff=np.pi):
        self.angular_cutoff = angular_cutoff

    def norm(self, alpha, beta):
        """
        Compute the normalization constant for given King parameters.

        Parameters
        ----------
        alpha : float or ndarray
            King distribution alpha parameter (scale) in radians.
        beta : float or ndarray
            King distribution beta parameter (tail weight).

        Returns
        -------
        float or ndarray
            Normalization constant(s) such that PDF integrates to 1.
        """
        return _norm(alpha, beta, self.angular_cutoff)

    def pdf(self, x, alpha, beta):
        """
        Evaluate the normalized King PDF at given angular separation(s).

        Returns zero for points beyond the angular cutoff. Handles broadcasting
        of input arrays and masks invalid regions.

        Parameters
        ----------
        x : float or ndarray
            Angular separation(s) from the source in radians.
        alpha : float or ndarray
            King distribution alpha parameter (scale) in radians.
        beta : float or ndarray
            King distribution beta parameter (tail weight).

        Returns
        -------
        ndarray
            Normalized PDF value(s) with units of probability/steradian.
        """
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
        mask = x <= self.angular_cutoff
        x, alpha, beta = x[mask], alpha[mask], beta[mask]

        # Nope. Do the calculations.
        norm = self.norm(alpha, beta)
        unnormalized_pdf = _unnormalized_pdf(x, alpha, beta)
        normalized_pdf[mask] = norm * unnormalized_pdf

        return normalized_pdf

    def cdf(self, x, alpha, beta):
        """
        Evaluate the normalized King CDF at given angular separation(s).

        Returns 1 for points beyond the angular cutoff. Handles broadcasting
        of input arrays and masks invalid regions.

        Parameters
        ----------
        x : float or ndarray
            Angular separation(s) from the source in radians.
        alpha : float or ndarray
            King distribution alpha parameter (scale) in radians.
        beta : float or ndarray
            King distribution beta parameter (tail weight).

        Returns
        -------
        ndarray
            Normalized CDF value(s) (cumulative probability).
        """
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
        mask = x <= self.angular_cutoff
        x, alpha, beta = x[mask], alpha[mask], beta[mask]

        # Nope. Do the calculations.
        norm = self.norm(alpha, beta)
        unnormalized_cdf = _unnormalized_cdf(x, alpha, beta)
        normalized_cdf[mask] = norm * unnormalized_cdf

        return normalized_cdf

    def marginalize(self, dec, alpha, beta, threshold=1e-6, nbins=None):
        """
        Marginalize the King PDF over right ascension to obtain declination profile.

        Integrates the 2D King PDF over RA using adaptive binning to produce a
        1D profile in sin(declination). Only bins with PDF above threshold are computed.

        Parameters
        ----------
        dec : float
            Source declination in radians.
        alpha : float
            King distribution alpha parameter (scale) in radians.
        beta : float
            King distribution beta parameter (tail weight).
        threshold : float, optional
            Minimum relative PDF value for bin inclusion. Default is 1e-6.
        nbins : int, optional
            Number of bins in sin(declination). If None, uses adaptive binning.

        Returns
        -------
        sindec_bins : ndarray
            Bin edges in sin(declination).
        marginalized : ndarray
            Marginalized PDF values at each bin.
        """
        # Generate the sindec binning
        sindec = np.sin(dec)
        if nbins is None:
            delta = min(0.025, (np.sin(dec + alpha) - np.sin(dec - alpha)) / 2)
            sindec_bins = np.concatenate(
                [np.arange(-1, sindec, delta), np.arange(sindec, 1, delta)]
            )
        else:
            sindec_bins = np.concatenate([[sindec, 1], np.linspace(-1, 1, nbins)])
            sindec_bins = np.unique(sindec_bins)

        # Mask unnecessary declination bins
        pdf_check = self.pdf(np.abs(dec - np.arcsin(sindec_bins)), alpha, beta)
        decmask = pdf_check / pdf_check.max() > threshold

        # Generate a binning in RA from 0 to 2pi.
        nbins = int(2 * np.pi / alpha * 5)
        ra_bins = np.linspace(0, 2 * np.pi, nbins)

        # Mask unnecessary RA bins
        pdf_check = self.pdf(ra_bins, alpha, beta)
        ramask = pdf_check / pdf_check.max() > threshold
        ra_bins = ra_bins[ramask]

        # Calculate the PDF at each grid point
        ra_points, dec_points = meshgrid2d(ra_bins, np.arcsin(sindec_bins[decmask]))
        distance = angular_distance(0, dec, ra_points, dec_points)
        pdf = self.pdf(distance, alpha, beta)

        # Integrate over RA using a simple trapezoid rule integration.
        # This should probably be checked for correct units. For example,
        # should this include a np.diff(sindec_bins) in the denominator?
        # I don't think it should, but someone should help check this.
        marginalized = np.zeros_like(sindec_bins)
        marginalized[decmask] = ((pdf[:, :-1] + pdf[:, 1:]) / 2 * np.diff(ra_bins)[None, :]).sum(
            axis=1
        )
        return sindec_bins, marginalized


class InterpolatedKingPDF(KingPDF):
    """
    King PDF with pre-computed normalization interpolation for efficiency.

    Pre-calculates normalization constants on a grid of (alpha, beta) values
    and uses 2D interpolation for fast norm() evaluations.

    Parameters
    ----------
    angular_cutoff : float, optional
        Maximum angular separation in radians. Default is pi.
    points_alpha : ndarray, optional
        Grid of alpha values for normalization interpolation.
    points_beta : ndarray, optional
        Grid of beta values for normalization interpolation.
    """

    def __init__(
        self,
        *,
        angular_cutoff=np.pi,
        points_alpha=np.logspace(-5, _log10pi + 1e-2, 200),
        points_beta=np.logspace(0, 1, 200),
    ):
        super().__init__(angular_cutoff=angular_cutoff)
        self.points_alpha, self.points_beta = points_alpha, points_beta
        grid_alpha, grid_beta = np.meshgrid(points_alpha, points_beta)
        self.log10_grid_norms = np.log10(_norm(grid_alpha, grid_beta, self.angular_cutoff).T)

    def norm(self, alpha, beta):
        return 10 ** _interp2d(
            alpha, beta, self.points_alpha, self.points_beta, self.log10_grid_norms
        )


class TemplateSmearedKingPDF(InterpolatedKingPDF):
    """
    King PDF convolved with a HEALPix template map using spherical harmonics.

    Uses spherical harmonic decomposition to efficiently convolve a King PSF
    with a template skymap. Pre-computes template harmonics for fast evaluation.

    Parameters
    ----------
    skymap : ndarray
        HEALPix map to convolve with King PDF. Will be normalized to integrate to 1.
    eval_decs : float or ndarray
        Declination(s) in radians where PDF will be evaluated.
    eval_ras : float or ndarray
        Right ascension(s) in radians where PDF will be evaluated.
    angular_cutoff : float, optional
        Maximum angular separation in radians. Default is pi.
    points_alpha : ndarray, optional
        Grid of alpha values for normalization interpolation.
    points_beta : ndarray, optional
        Grid of beta values for normalization interpolation.
    lmax : int, optional
        Maximum spherical harmonic degree. Default is 3*nside-1.
    scheme : str, optional
        HEALPix ordering scheme ('RING' or 'NESTED'). Default is 'RING'.
    """

    def __init__(
        self,
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
        super().__init__(
            angular_cutoff=angular_cutoff,
            points_alpha=np.logspace(-5, _log10pi, 200),
            points_beta=np.logspace(0, 1, 200),
        )
        self.nside = hp.npix2nside(len(skymap))
        self.lmax = (3 * self.nside - 1) if lmax is None else lmax
        self.mmax = self.lmax

        # Set up the evaluation coordinates
        self.set_coordinates(eval_decs, eval_ras)

        # While we're here, normalize the skymap so it integrates to 1.
        self.skymap = skymap
        self.skymap /= (self.skymap * hp.nside2resol(self.nside)).sum()
        self.skymap_alm = self.skymap_to_alm()

    def set_coordinates(self, eval_decs, eval_ras):
        """
        Set evaluation coordinates and pre-compute spherical harmonics.

        Parameters
        ----------
        eval_decs : float or ndarray
            Declination(s) in radians.
        eval_ras : float or ndarray
            Right ascension(s) in radians.
        """
        self._eval_decs = np.atleast_1d(eval_decs)
        self._eval_ras = np.atleast_1d(eval_ras)
        assert self._eval_decs.shape == self._eval_ras.shape

        # Evaluate the Y_lm harmonic functions now so we can use them later without cost.
        # The theta, phi here are co-latitude and azimuth/RA, both in radians.
        self._Y_lm = get_Ylm(self.lmax, self.mmax, self._eval_decs - np.pi / 2, self._eval_ras)
        return

    def skymap_to_alm(self):
        """
        Convert the HEALPix skymap to spherical harmonic coefficients.

        Returns
        -------
        ndarray
            Complex spherical harmonic coefficients a_lm.
        """
        return hp.map2alm(self.skymap, lmax=self.lmax, mmax=self.mmax)

    def get_king_b_l(self, alpha, beta, npoints=100, scale=np.pi / 4):
        """
        Compute spherical harmonic expansion coefficients b_l for the King PDF.

        Integrates King PDF against Legendre polynomials to obtain coefficients
        for harmonic convolution.

        Parameters
        ----------
        alpha : float
            King distribution alpha parameter (scale) in radians.
        beta : float
            King distribution beta parameter (tail weight).
        npoints : int, optional
            Number of integration points. Default is 100.
        scale : float, optional
            Scale factor for adaptive binning. Default is pi/4.

        Returns
        -------
        ndarray
            Spherical harmonic coefficients b_l for degrees 0 to lmax.
        """
        theta = adaptive_bins(alpha, npoints, scale=scale)
        P_all = legendre_p_all(self.lmax, np.cos(theta))[0]
        pdf_vals = self.pdf(theta, alpha, beta)
        return 2 * np.pi * trapezoid(P_all * pdf_vals * np.sin(theta), dx=np.diff(theta))

    def convolve_map(self, alpha, beta):
        """
        Convolve the template skymap with a King PSF and return full HEALPix map.

        Parameters
        ----------
        alpha : float
            King distribution alpha parameter (scale) in radians.
        beta : float
            King distribution beta parameter (tail weight).

        Returns
        -------
        ndarray
            Convolved HEALPix skymap at the same resolution as input.
        """
        b_l = self.get_king_b_l(alpha, beta)
        harmonic_convolution = hp.almxfl(self.skymap_alm.copy(), b_l, self.lmax, self.mmax)
        return hp.alm2map(harmonic_convolution, nside=self.nside, lmax=self.lmax, mmax=self.mmax)

    def convolve_at_grid_point(self, alpha, beta, npoints=100, scale=np.pi / 4):
        """
        Evaluate convolved PDF only at pre-set grid points (eval_decs, eval_ras).

        More efficient than convolve_map() when only specific points are needed.
        Uses pre-computed spherical harmonics from set_coordinates().

        Parameters
        ----------
        alpha : float
            King distribution alpha parameter (scale) in radians.
        beta : float
            King distribution beta parameter (tail weight).
        npoints : int, optional
            Number of integration points for b_l calculation. Default is 100.
        scale : float, optional
            Scale factor for adaptive binning. Default is pi/4.

        Returns
        -------
        float or ndarray
            Convolved PDF value(s) at evaluation coordinates.
        """
        b_l = self.get_king_b_l(alpha, beta, npoints=npoints, scale=scale)
        harmonic_convolution = hp.almxfl(self.skymap_alm.copy(), b_l, self.lmax, self.mmax)
        if len(self._Y_lm.shape) > 1:
            return np.real(harmonic_convolution[:, None] * self._Y_lm).sum(axis=0)
        else:
            return np.real(harmonic_convolution * self._Y_lm).sum(axis=0)

    def marginalize(self, dec, alpha, beta, threshold=1e-6):
        raise NotImplementedError("Signal subtraction for templates is not implemented.")
