from typing import Union
import numpy as np
import numpy.typing as npt
from numba import vectorize, float32, float64

_log10pi: float = np.log10(np.pi)


@vectorize([float32(float32, float32, float32), float64(float64, float64, float64)], target="cpu")
def _unnormalized_pdf(
    x: Union[float, npt.NDArray[np.floating]],
    alpha: Union[float, npt.NDArray[np.floating]],
    beta: Union[float, npt.NDArray[np.floating]],
) -> Union[float, npt.NDArray[np.floating]]:
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
        Unnormalized King function values with units of probability/sterradian.
    """
    # Calculate the King function value and return
    return (1 + (x / alpha) ** 2 / (2 * beta)) ** -beta


@vectorize  # ([float32(float32,float32,float32), float64(float64,float64,float64)],
# target='cpu')
def _unnormalized_cdf(
    x: Union[float, npt.NDArray[np.floating]], alpha: float, beta: float
) -> Union[float, npt.NDArray[np.floating]]:
    """
    Evaluate the CDF of the radial King function (without solid angle Jacobian).

    The integral includes the sin(theta) for spherical coordinates and
    normalizes over solid angle (i.e., integral(PDF * sin(theta) dtheta dphi = 1)).
    Use log-binning with 1000 bins from 1e-5 radians to this value.

    Parameters
    ----------
    x : scalar
        The location at which to calculate the CDF in radians.
    alpha : float
        King distribution alpha parameter (scale) in radians.
    beta : float
        King distribution beta parameter (tail weight).

    Returns
    -------
    float
        King function CDF value with units of probability.
    """
    # Define the grid points for the evaluation. Add some padding to the maximum.
    points = np.append(
        [
            0.0,
        ],
        np.logspace(_log10pi - 5, _log10pi, 1000),
    )

    # Use the unnormalized PDF (given in 1/sr) and scale by the annulus area (~cos(dx))
    unnormalized = _unnormalized_pdf(points, alpha, beta)
    solid_angle = 2 * np.pi * np.abs(np.cos(points[1:]) - np.cos(points[:-1]))
    integrand = np.append([0.0], np.cumsum(unnormalized[1:] * np.abs(solid_angle)))
    return np.interp(x, points, integrand)


def _norm(
    alpha: Union[float, npt.NDArray[np.floating]],
    beta: Union[float, npt.NDArray[np.floating]],
    maximum: float,
) -> Union[float, npt.NDArray[np.floating]]:
    """
    Compute the normalization constant for the King PDF over the sphere.

    The integral includes the sin(theta) for spherical coordinates and
    normalizes over solid angle (i.e., integral(PDF * sin(theta) dtheta dphi = 1)).

    Parameters
    ----------
    alpha : float or ndarray
        King distribution alpha parameter (scale).
    beta : float or ndarray
        King distribution beta parameter (tail weight).
    maximum : float
        The maximum angular value for the distribution. This would normally
        just be pi, but can be smaller if the user is truncating the distribution.

    Returns
    -------
    ndarray
        Normalization constants such that PDF integrates to 1 over the sphere.
    """
    return 1.0 / _unnormalized_cdf(maximum, alpha, beta)
