import numpy as np
from scipy.integrate import cumulative_trapezoid

class KingSpatialPDF():
    def __init__():
        pass

    def pdf(self, x, alpha, beta):
        return self._norm(alpha, beta) * self._unnormalized_pdf(x, alpha, beta)

    def cdf(self, x, alpha, beta):
        return self._norm(alpha, beta) * self._unnormalized_cdf(x, alpha, beta)

    def _unnormalized_pdf(self, x, alpha, beta):
        """
        Evaluate the unnormalized radial King function (without solid angle Jacobian).

        This function returns the radial shape:
            f(x) = [1 + (x/alpha)^2 / (2*beta)]^(-beta)

        Parameters
        ----------
        x : float or ndarray
            Angular separation from the source, in radians.
        alpha : float or ndarray
            King distribution alpha parameter (scale).
        beta : float or ndarray
            King distribution beta parameter (tail weight).

        Returns
        -------
        ndarray
            Unnormalized King function values with units of probability/sterradian
        """
        x = np.atleast_1d(x)
        return (1 + (x / alpha)**2 / (2 * beta))**-beta

    def _unnormalized_cdf(self, x, alpha, beta):
        """
        Evaluate the CDF of the radial King function (without solid angle Jacobian).

        The integral includes the sin(theta) for spherical coordinates and
        normalizes over solid angle (i.e., integral(PDF * sin(theta) dtheta dphi = 1)).

        Parameters
        ----------
        x : scalar or ndarray
            The locations at which to calculate the CDF. If this is a scalar, use log-binning
            with 1000 bins from 1e-5 to this value, assuming radians.
        alpha : float or ndarray
            King distribution alpha parameter (scale).
        beta : float or ndarray
            King distribution beta parameter (tail weight).

        Returns
        -------
        float or ndarray
            King function CDF values with units of probability. If x is a single value,
            only return the last value.
        """
        # Check if x is a float or ndarray
        isscalar = np.isscalar(x)
        if isscalar:
            x = np.logspace(np.log10(pi)-5, np.log10(x), 1000)
        
        # Broadcast
        alpha = np.atleast_1d(alpha)[:, None, None]
        beta = np.atleast_1d(beta)[None, :, None]
        x = x[None, None, :]

        # Use the unnormalized PDF (which doesn't include sin(x))
        unnormalized = self._unnormalized_pdf(x, alpha, beta)
        integrand = unnormalized * np.sin(x)
        cdf = cumulative_trapezoid(integrand, x, axis=-1, initial=0)

        # If we only want a scalar value out, grab only what we need
        if isscalar:
            return cdf[...,-1]
            
    def _norm(self, alpha, beta):
        """
        Compute the normalization constant for the King PDF over the sphere.

        The integral includes the sin(theta) for spherical coordinates and
        normalizes over solid angle (i.e., integral(PDF * sin(theta) dtheta dphi = 1)).

        Parameters
        ----------
        alpha : float or ndarray
            Alpha parameters for the King distribution.
        beta : float or ndarray
            Beta parameters for the King distribution.

        Returns
        -------
        ndarray
            Normalization constants such that PDF integrates to 1 over the sphere.
        """
        cdf = self._unnormalized_cdf(np.pi, alpha, beta)
        return cdf
