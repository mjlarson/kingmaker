from typing import Tuple, Union
import numpy as np
import numpy.typing as npt
from numba import guvectorize, njit
from numba import float32, float64

_log10pi: float = np.log10(np.pi)


@njit
def _interp1d(x: float, xlow: float, xhigh: float, ylow: float, yhigh: float) -> float:
    """
    Perform 1D linear interpolation.

    Parameters
    ----------
    x : float
        Point at which to interpolate.
    xlow : float
        Lower x-coordinate of the interval.
    xhigh : float
        Upper x-coordinate of the interval.
    ylow : float
        Function value at xlow.
    yhigh : float
        Function value at xhigh.

    Returns
    -------
    float
        Linearly interpolated value at x.
    """
    return ylow + (yhigh - ylow) / (xhigh - xlow) * (x - xlow)


@guvectorize(
    [
        (float32, float32, float32[:], float32[:], float32[:, :], float32[:]),
        (float64, float64, float64[:], float64[:], float64[:, :], float64[:]),
    ],
    "(),(),(n),(m),(n,m)->()",
    cache=True,
    target="parallel",
)
def _interp2d(x, y, xp, yp, z, result):
    """
    Perform 2D bilinear interpolation on a regular grid.

    Vectorized interpolation supporting both float32 and float64. Uses
    bilinear interpolation within grid cells and handles edge cases by
    clamping to boundary values.

    Parameters
    ----------
    x : float or ndarray
        X-coordinate(s) at which to interpolate.
    y : float or ndarray
        Y-coordinate(s) at which to interpolate.
    xp : ndarray
        Sorted 1D array of x-coordinates defining the grid.
    yp : ndarray
        Sorted 1D array of y-coordinates defining the grid.
    z : ndarray
        2D array of function values at grid points (shape: len(xp) x len(yp)).
    result : ndarray
        Output array for interpolated values.
    """
    xmax, ymax = int(len(xp) - 1), int(len(yp) - 1)
    xidx = np.searchsorted(xp, x, "right")
    yidx = np.searchsorted(yp, y, "right")

    # Check for edge cases
    # Handle underflow by pinning it to the first value
    if xidx == 0:
        xidx = 1
    if yidx == 0:
        yidx = 1

    # Overflow is at or above the max edge. Set it to the last value.
    if xidx > xmax:
        xidx = xmax
    if yidx > ymax:
        yidx = ymax

    # Get the coordinates for the surrounding box
    left, right = xp[xidx - 1], xp[xidx]
    bottom, top = yp[yidx - 1], yp[yidx]

    # Get the values at the 4 surrounding points
    z_left_bottom, z_right_bottom = z[xidx - 1, yidx - 1], z[xidx, yidx - 1]
    z_left_top, z_right_top = z[xidx - 1, yidx], z[xidx, yidx]

    z_bottom = _interp1d(x, left, right, z_left_bottom, z_right_bottom)
    z_top = _interp1d(x, left, right, z_left_top, z_right_top)

    # And return the interpolation between the low and high
    result[0] = _interp1d(y, bottom, top, z_bottom, z_top)


def map2nside(skymap: npt.NDArray[np.floating]) -> int:
    """
    Compute HEALPix nside parameter from skymap size.

    Parameters
    ----------
    skymap : ndarray
        HEALPix map array.

    Returns
    -------
    int
        HEALPix nside parameter (npix = 12 * nside^2).
    """
    return int(np.sqrt(len(skymap) / 12))


@njit
def angular_distance(
    src_ra: Union[float, npt.NDArray[np.floating]],
    src_dec: Union[float, npt.NDArray[np.floating]],
    ra: Union[float, npt.NDArray[np.floating]],
    dec: Union[float, npt.NDArray[np.floating]],
) -> Union[float, npt.NDArray[np.floating]]:
    """
    Calculate angular distance on the sphere using the haversine formula.

    Computes the great-circle distance between celestial coordinates using
    spherical trigonometry.

    Parameters
    ----------
    src_ra : float or ndarray
        Source right ascension in radians.
    src_dec : float or ndarray
        Source declination in radians.
    ra : float or ndarray
        Target right ascension(s) in radians.
    dec : float or ndarray
        Target declination(s) in radians.

    Returns
    -------
    float or ndarray
        Angular separation(s) in radians.
    """
    cosDist = np.cos(src_ra - ra) * np.cos(src_dec) * np.cos(dec) + np.sin(src_dec) * np.sin(dec)
    return np.arccos(cosDist)


@njit
def meshgrid2d(
    a: npt.NDArray[np.floating], b: npt.NDArray[np.floating]
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Create 2D meshgrid from 1D coordinate arrays (numba-compatible).

    Similar to numpy.meshgrid but optimized for use with numba JIT compilation.
    Returns transposed grids in matrix indexing ('ij') convention.

    Parameters
    ----------
    a : ndarray
        1D array of coordinates for first dimension.
    b : ndarray
        1D array of coordinates for second dimension.

    Returns
    -------
    grid_a : ndarray
        2D grid of 'a' values with shape (len(b), len(a)).
    grid_b : ndarray
        2D grid of 'b' values with shape (len(b), len(a)).
    """
    output_a = np.empty((len(a), len(b)), dtype=a.dtype)
    output_b = np.empty((len(a), len(b)), dtype=b.dtype)

    for i in range(len(a)):
        output_a[i, :] = a[i]
    for j in range(len(b)):
        output_b[:, j] = b[j]
    return output_a.T, output_b.T
