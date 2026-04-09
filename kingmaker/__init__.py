# Import main classes for convenience
from .pdf import KingPDF, InterpolatedKingPDF, TemplateSmearedKingPDF
from .fitting import KingPSFFitter

'''class KingSpatialPDF:
    def __init__(self):
        pass

    def pdf(self, x, alpha, beta):
        return self._norm(alpha, beta) * self._unnormalized_pdf(x, alpha, beta)

    def cdf(self, x, alpha, beta):
        return self._norm(alpha, beta) * self._unnormalized_cdf(x, alpha, beta)

    def _to_1d(self, *args):
        return (np.atleast_1d(_) for _ in args)

    def _check_args(self, **kwargs):
        # Ensure all args are the same shape
        if len(kwargs.keys()) > 1:
            shape = list(kwargs.values())[0].shape
            for key, val in kwargs.items():
                if len(val.shape) > 1:
                    message = (
                        f"The variable {key} has more than one dimension (shape = {val.shape})."
                        " This is not supported at the moment. Try calling again with 1d values."
                    )
                    raise NotImplementedError(message)
                if val.shape != shape:
                    message = (
                        f"The variables {list(kwargs.keys())} have different shapes "
                        f"{list(_.shape for _ in kwargs.values())}."
                    )
                    raise ValueError(message)
        return

    def _unnormalized_pdf(self, x, alpha, beta):
        """
        Evaluate the unnormalized radial King function (without solid angle Jacobian).

        This function returns the radial shape:
            f(x) = [1 + (x/alpha)^2 / (2*beta)]^(-beta)

        Parameters
        ----------
        x : float or ndarray
            Angular separation from the source, in radians with shape M.
        alpha : float or ndarray
            King distribution alpha parameter (scale) with shape N.
        beta : float or ndarray
            King distribution beta parameter (tail weight) with shape N.

        Returns
        -------
        ndarray
            Unnormalized King function values with units of probability/sterradian.
            The output shape will be either N (if `x` is a scalar) or (N, M) (if `x`
            is an ndarray).
        """
        print("_unnormalized_pdf")

        # Check if x is a float or ndarray
        isscalar = np.isscalar(x)

        # Ensure everything is 1d
        x, alpha, beta = self._to_1d(x, alpha, beta)
        self._check_args(alpha=alpha, beta=beta)

        # Broadcast
        alpha = alpha[:, None]
        beta = beta[:, None]
        x = x[None, :]

        # Calculate the King function value and return
        return (1 + (x / alpha) ** 2 / (2 * beta)) ** -beta

    def _unnormalized_cdf(self, x, alpha, beta):
        """
        Evaluate the CDF of the radial King function (without solid angle Jacobian).

        The integral includes the sin(theta) for spherical coordinates and
        normalizes over solid angle (i.e., integral(PDF * sin(theta) dtheta dphi = 1)).

        Parameters
        ----------
        x : scalar or ndarray
            The locations at which to calculate the CDF. If this is a scalar, use log-binning
            with 1000 bins from 1e-5 to this value, assuming radians. If this is an ndarray,
            the shape can be M, indepedent of the shape of alpha and beta.
        alpha : float or ndarray
            King distribution alpha parameter (scale) with shape N.
        beta : float or ndarray
            King distribution beta parameter (tail weight) with shape N.

        Returns
        -------
        float or ndarray
            King function CDF values with units of probability. If x is a single value,
            only return the last value. Otherwise, the shape will be (N, M).
        """
        print("_unnormalized_cdf")
        # Check if x is a float or ndarray
        isscalar = np.isscalar(x)
        if isscalar:
            x = np.logspace(np.log10(np.pi) - 5, np.log10(x), 1000)

        # Broadcast
        x, alpha, beta = self._to_1d(x, alpha, beta)
        alpha = alpha[:, None]
        beta = beta[:, None]
        x = x[None, :]

        # Use the unnormalized PDF (which doesn't include sin(x))
        unnormalized = np.array([self._unnormalized_pdf(x, a, b) for a, b in zip(alpha, beta)])
        integrand = unnormalized * np.sin(x)

        print(x.shape, integrand.shape)
        cdf = cumulative_trapezoid(integrand, x, axis=-1, initial=0)

        # If we only want a scalar value out, grab only what we need
        if isscalar:
            return cdf[..., -1]

    def _norm(self, alpha, beta):
        """
        Compute the normalization constant for the King PDF over the sphere.

        The integral includes the sin(theta) for spherical coordinates and
        normalizes over solid angle (i.e., integral(PDF * sin(theta) dtheta dphi = 1)).

        Parameters
        ----------
        alpha : float or ndarray
            King distribution alpha parameter (scale) with shape N.
        beta : float or ndarray
            King distribution beta parameter (tail weight) with shape N.

        Returns
        -------
        ndarray
            Normalization constants such that PDF integrates to 1 over the sphere
            with shape N.
        """
        print("norm")
        cdf = self._unnormalized_cdf(np.pi, alpha, beta)
        return cdf
'''

__all__ = ["KingPDF", "InterpolatedKingPDF", "TemplateSmearedKingPDF", "KingPSFFitter"]
