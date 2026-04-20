from typing import Optional, Tuple, Union
import os
import numpy as np
import numpy.typing as npt
import healpy as hp
from scipy.integrate import trapezoid
from scipy.special import legendre_p_all, sph_harm_y_all

from .distribution import _log10pi
from .distribution import _norm, _unnormalized_pdf, _unnormalized_cdf
from .utils import _interp2d, angular_distance, meshgrid2d


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

    def __init__(self, *, angular_cutoff: float = np.pi) -> None:
        self.angular_cutoff = angular_cutoff

    def norm(
        self,
        alpha: Union[float, npt.NDArray[np.floating]],
        beta: Union[float, npt.NDArray[np.floating]],
    ) -> Union[float, npt.NDArray[np.floating]]:
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

    def pdf(
        self,
        x: Union[float, npt.NDArray[np.floating]],
        alpha: Union[float, npt.NDArray[np.floating]],
        beta: Union[float, npt.NDArray[np.floating]],
    ) -> Union[float, npt.NDArray[np.floating]]:
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

    def cdf(
        self,
        x: Union[float, npt.NDArray[np.floating]],
        alpha: Union[float, npt.NDArray[np.floating]],
        beta: Union[float, npt.NDArray[np.floating]],
    ) -> Union[float, npt.NDArray[np.floating]]:
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

    def marginalize(
        self,
        dec: float,
        alpha: float,
        beta: float,
        threshold: float = 1e-6,
        nbins: Optional[int] = None,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
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
        Grid of alpha values for normalization interpolation. Unit: radians
    points_beta : ndarray, optional
        Grid of beta values for normalization interpolation. 
    """

    def __init__(
        self,
        *,
        angular_cutoff: float = np.pi,
        points_alpha: npt.NDArray[np.floating] = np.logspace(-4, _log10pi + 1e-2, 200),
        points_beta: npt.NDArray[np.floating] = np.logspace(0, 1, 200),
    ) -> None:
        super().__init__(angular_cutoff=angular_cutoff)
        self.points_alpha, self.points_beta = points_alpha, points_beta
        grid_alpha, grid_beta = np.meshgrid(points_alpha, points_beta)
        self.log10_grid_norms = np.log10(_norm(grid_alpha, grid_beta, self.angular_cutoff).T)

    def norm(
        self,
        alpha: Union[float, npt.NDArray[np.floating]],
        beta: Union[float, npt.NDArray[np.floating]],
    ) -> Union[float, npt.NDArray[np.floating]]:
        if np.any(alpha < self.points_alpha[0]) or np.any(alpha > self.points_alpha[-1]):
            raise ValueError(f"Alpha value {alpha} must be within the interpolation grid: alpha in [{self.points_alpha[0]}, {self.points_alpha[-1]}]")
        if np.any(beta < self.points_beta[0]) or np.any(beta > self.points_beta[-1]):
            raise ValueError(f"Beta value {beta} must be within the interpolation grid: beta in [{self.points_beta[0]}, {self.points_beta[-1]}]")
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
    """
    skymap : npt.NDArray[np.floating]
    eval_decs : npt.NDArray[np.floating] = np.array([], dtype=np.float32)
    eval_ras: npt.NDArray[np.floating] = np.array([], dtype=np.float32)

    def __init__(
        self,
        skymap: npt.NDArray[np.floating],
        *,
        eval_decs: Union[float, npt.NDArray[np.floating], None] = None,
        eval_ras: Union[float, npt.NDArray[np.floating], None] = None,
        angular_cutoff: float = np.pi,
        points_alpha: npt.NDArray[np.floating] = np.logspace(-4, _log10pi, 100),
        points_beta: npt.NDArray[np.floating] = np.logspace(0, 1, 200),
        lmax: Optional[int] = None,
    ) -> None:
        super().__init__(
            angular_cutoff=angular_cutoff,
            points_alpha=points_alpha,
            points_beta=points_beta,
        )
        self.nside = hp.npix2nside(len(skymap))
        self.lmax = (3 * self.nside - 1) if lmax is None else lmax
        self.mmax = self.lmax

        # While we're here, normalize the skymap so it integrates to 1.
        # Then calculate the alm coefficients needed for convolution.
        self.skymap = skymap
        normalized_skymap = skymap / skymap.sum() / hp.nside2pixarea(self.nside)
        self.skymap_alm = hp.map2alm(normalized_skymap, lmax=self.lmax, mmax=self.mmax)

        # Precompute the indices for the spherical harmonic coefficients
        # we'll need for convolution.
        self.ls = np.concatenate([np.full(min(l, self.mmax) + 1, l) 
                             for l in range(self.lmax + 1)]) # noqa: E741
        self.ms = np.concatenate([np.arange(min(l, self.mmax) + 1) 
                             for l in range(self.lmax + 1)])  # noqa: E741
        self._alm_indices = hp.Alm.getidx(self.lmax, self.ls, self.ms)
        self._m_vals = self.ms 

        # Set up the evaluation coordinates
        if eval_decs is not None and eval_ras is not None:
            self.set_coordinates(eval_decs, eval_ras)

        # Choose a reasonable enough grid for the b_l calculation
        # and pre-generate the Legendre polynomials we'll need.
        # The actual values here are little awkward and arbitrary,
        # but they should be fine for most people. Note that these
        # are in radians, so a value of 1e-4 corresponds to about
        # 0.0057 degrees: this is much smaller than IceCube's 
        # typical angular resolution. Since these are expensive,
        # we can cache them to disk once and just load them for
        # all future coefficients.
        if not os.path.exists("precomputed_legendre.npz"):
            #print("Generating and storing Legendre polynomials for convolution...")
            self.theta_grid = np.concatenate([[0.0], np.logspace(-4, np.log10(self.angular_cutoff), 1000)])
            P_all = legendre_p_all(1024, np.cos(self.theta_grid))[0]
            np.savez("precomputed_legendre.npz", theta_grid=self.theta_grid, P_all=P_all)
            self.P_all = P_all[:self.lmax + 1]
        else:
            #print("Loading precomputed Legendre polynomials for convolution...")
            precomputed = np.load("precomputed_legendre.npz")
            self.theta_grid = precomputed["theta_grid"]
            self.P_all = precomputed["P_all"][:self.lmax + 1]

    def set_coordinates(
        self,
        eval_decs: Union[float, npt.NDArray[np.floating]],
        eval_ras: Union[float, npt.NDArray[np.floating]],
    ) -> None:
        """
        Set evaluation coordinates and pre-compute spherical harmonics.

        Parameters
        ----------
        eval_decs : float or ndarray
            Declination(s) in radians.
        eval_ras : float or ndarray
            Right ascension(s) in radians.
        """
        # If there's a different number of declination and RA points, 
        # there's something wrong. 
        if np.atleast_1d(eval_decs).shape != np.atleast_1d(eval_ras).shape:
            raise RuntimeError(
                "TemplateSmearedKingPDF::set_coordinates received different numbers"
                f" of declination values ({np.atleast_1d(eval_decs).shape}) and"
                f" right ascension values ({np.atleast_1d(eval_ras).shape})."
                " These need to match since each is assumed to be one source."
            )

        # If the coordinates match what we already have, do nothing.
        if (np.equal(self.eval_decs, np.atleast_1d(eval_decs))
            and np.equal(self.eval_ras, np.atleast_1d(eval_ras))):
            return

        self.eval_decs = np.atleast_1d(eval_decs)
        self.eval_ras = np.atleast_1d(eval_ras)
        assert self.eval_decs.shape == self.eval_ras.shape

        # Compute all Y_lm at once: shape (lmax+1, 2*mmax+1, npts)
        # m axis ordering: 0, +1, +2, ..., +mmax, -mmax, ..., -1
        raw = sph_harm_y_all(self.lmax, self.mmax, np.pi/2 - self.eval_decs, self.eval_ras)
        
        # Extract all needed Y_lm at once: shape (nalm, npts)
        Y_lm_all = raw[self.ls, self.ms, :]  # advanced indexing, no loop
        a_lm_all = self.skymap_alm[self._alm_indices]  # shape (nalm,)
        contribs = np.real(a_lm_all[:, None] * Y_lm_all)  # (nalm, npts)
        factors = np.where(self._m_vals == 0, 1.0, 2.0)   # (nalm,)

        # Accumulate per l
        npts = len(self.eval_decs)
        self._c_l = np.zeros((self.lmax + 1, npts), dtype=np.float64)
        np.add.at(self._c_l, self.ls, factors[:, None] * contribs)
        return

    def skymap_to_alm(self) -> npt.NDArray[np.complexfloating]:
        """
        Convert the HEALPix skymap to spherical harmonic coefficients.

        Returns
        -------
        ndarray
            Complex spherical harmonic coefficients a_lm.
        """
        return hp.map2alm(self.skymap, lmax=self.lmax, mmax=self.mmax)

    def get_king_b_l(
        self, alpha: float, beta: float) -> npt.NDArray[np.floating]:
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

        Returns
        -------
        ndarray
            Spherical harmonic coefficients b_l for degrees 0 to lmax.
        """
        pdf_vals = self.pdf(self.theta_grid, alpha, beta)
        return 2 * np.pi * trapezoid(self.P_all * pdf_vals * np.sin(self.theta_grid), x=self.theta_grid)

    #--------------------------------------------------------
    # Vectorized over the whole grid at once
    #def _precompute_bl_grid(self):
    #    # pdf evaluates fine over arrays
    #    # self._theta_grid: (npoints,)
    #    # pdf_vals: (n_alpha, n_beta, npoints) via broadcasting
    #    alpha_grid, beta_grid = np.meshgrid(self.points_alpha, self.points_beta, indexing="ij")
    #    # shape: (n_alpha, n_beta, npoints)
    #    pdf_vals = self.pdf(
    #        self._theta_grid[None, None, :], alpha_grid[:, :, None], beta_grid[:, :, None]
    #    )
    #    # _P_all: (lmax+1, npoints) -> integrate against each pdf
    #    # trapezoid over last axis
    #    integrand = pdf_vals * self._sin_theta[None, None, :]  # (n_alpha, n_beta, npoints)
    #    # (lmax+1, npoints) @ (npoints, n_alpha*n_beta) -> (lmax+1, n_alpha*n_beta)
    #    bl_grid = (
    #        2
    #        * np.pi
    #        * np.trapz(
    #            self._P_all[:, None, None, :] * integrand[None, :, :, :], dx=self._dtheta, axis=-1
    #        )
    #    )  # (lmax+1, n_alpha, n_beta)
    #    self._bl_grid = bl_grid.transpose(1, 2, 0)  # (n_alpha, n_beta, lmax+1)
    # --------------------------------------------------------

    def convolve_map(self, alpha: float, beta: float) -> npt.NDArray[np.floating]:
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
        harmonic_convolution = hp.almxfl(alm=self.skymap_alm, fl=b_l, mmax=self.mmax, inplace=False)
        return hp.alm2map(harmonic_convolution, nside=self.nside, lmax=self.lmax, mmax=self.mmax)

    def convolve_at_grid_point(
        self, alpha: float, beta: float
    ) -> Union[float, npt.NDArray[np.floating]]:
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

        Returns
        -------
        float or ndarray
            Convolved PDF value(s) at evaluation coordinates.
        """
        b_l = self.get_king_b_l(alpha, beta)
        return b_l @ self._c_l

    def marginalize(
        self,
        dec: float,
        alpha: float,
        beta: float,
        threshold: float = 1e-6,
        nbins: Optional[int] = None,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        raise NotImplementedError("Signal subtraction for templates is not implemented.")
