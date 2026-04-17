from typing import Tuple, Union
import numpy as np
import numpy.typing as npt
from numba import guvectorize, njit
from numba import float32, float64
from scipy.special import sph_harm_y_all

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

    # If it's in the corners, return a value immediately.
    if (xidx in [1, xmax]) and (yidx in [1, ymax]):
        result[0] = z[xidx, yidx]

    # Otherwise get the coordinates for the surrounding box
    else:
        left, right = xp[xidx - 1], xp[xidx]
        bottom, top = yp[yidx - 1], yp[yidx]

        # Get the values at the 4 surrounding points
        z_left_bottom, z_right_bottom = z[xidx - 1, yidx - 1], z[xidx, yidx - 1]
        z_left_top, z_right_top = z[xidx - 1, yidx], z[xidx, yidx]

        z_bottom = _interp1d(x, left, right, z_left_bottom, z_right_bottom)
        z_top = _interp1d(x, left, right, z_left_top, z_right_top)

        # And return the interpolation between the low and high
        result[0] = _interp1d(y, bottom, top, z_bottom, z_top)


@njit
def adaptive_bins(alpha: float, npoints: int, scale: float = np.pi / 4) -> npt.NDArray[np.floating]:
    """
    Generate adaptive binning transitioning from logarithmic to linear spacing.

    Creates bins that are logarithmically spaced for small alpha and linearly
    spaced for large alpha, with smooth transition controlled by the scale parameter.

    Parameters
    ----------
    alpha : float
        Characteristic scale in radians controlling the transition.
    npoints : int
        Number of bin edges to generate.
    scale : float, optional
        Transition scale in radians. Default is pi/4.

    Returns
    -------
    ndarray
        Array of bin edges from 0 to pi.
    """
    linear = np.linspace(0, np.pi, npoints)
    logarithmic = np.logspace(-4, _log10pi, npoints)

    weight = alpha / scale
    if weight > 1:
        weight = 1
    return linear * weight + logarithmic * (1 - weight)


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
def alm_index(l: int, m: int, lmax: int) -> int:  # noqa: E741
    """
    Compute flat array index for spherical harmonic coefficient a_lm.

    Parameters
    ----------
    l : int
        Degree of the spherical harmonic.
    m : int
        Order of the spherical harmonic (0 <= m <= l).
    lmax : int
        Maximum degree in the coefficient array.

    Returns
    -------
    int
        Index in the flattened a_lm array.
    """
    return int(m * (2 * lmax + 1 - m) / 2 + l)


def _reshape_alm(
    Ylm_all: npt.NDArray[np.complexfloating], lmax: int, mmax: int
) -> Tuple[npt.NDArray[np.complexfloating], Union[int, Tuple[int, int]]]:
    """
    Reshape spherical harmonic array and compute packed dimensions.

    Parameters
    ----------
    Ylm_all : ndarray
        Array of spherical harmonic values.
    lmax : int
        Maximum degree.
    mmax : int
        Maximum order.

    Returns
    -------
    reshaped : ndarray
        Reshaped spherical harmonic array.
    packed_shape : int or tuple
        Shape for packed a_lm coefficients.
    """
    nalm = (mmax + 1) * (2 * lmax - mmax + 2) // 2
    if Ylm_all.shape[-1] == 1:
        return Ylm_all.reshape(Ylm_all.shape[:-1]), nalm
    else:
        return Ylm_all, (nalm, Ylm_all.shape[-1])


def _repack_alm(
    Ylm: npt.NDArray[np.complexfloating],
    packed_shape: Union[int, Tuple[int, int]],
    lmax: int,
    mmax: int,
) -> npt.NDArray[np.complexfloating]:
    """
    Repack Y_lm values into a_lm coefficient ordering.

    Converts from (l, m) indexing to the flattened a_lm convention used by HEALPix.

    Parameters
    ----------
    Ylm : ndarray
        Spherical harmonic values indexed by (l, m).
    packed_shape : int or tuple
        Target shape for packed coefficients.
    lmax : int
        Maximum degree.
    mmax : int
        Maximum order.

    Returns
    -------
    ndarray
        Packed a_lm coefficients in HEALPix ordering.
    """
    # pack into alm order
    Ylm_packed = np.empty(packed_shape, dtype=np.complex128)
    idx = 0
    for l in range(lmax + 1):  # noqa: E741
        idxmax = min(l, mmax) + 1
        Ylm_packed[idx : idx + idxmax] = Ylm[l, :idxmax]
        idx += idxmax
    return Ylm_packed


def get_Ylm(
    lmax: int,
    mmax: int,
    theta: Union[float, npt.NDArray[np.floating]],
    phi: Union[float, npt.NDArray[np.floating]],
) -> npt.NDArray[np.complexfloating]:
    """
    Compute spherical harmonics Y_lm at given angles.

    Evaluates spherical harmonics up to degree lmax and order mmax,
    returning coefficients in HEALPix a_lm ordering.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree.
    mmax : int
        Maximum spherical harmonic order.
    theta : float or ndarray
        Co-latitude angle(s) in radians (0 at north pole).
    phi : float or ndarray
        Azimuthal angle(s) in radians.

    Returns
    -------
    ndarray
        Complex spherical harmonic values in a_lm ordering.
    """
    Ylm = sph_harm_y_all(lmax, mmax, theta, phi)
    reduced_Ylm, shape = _reshape_alm(Ylm, lmax=lmax, mmax=mmax)
    return _repack_alm(reduced_Ylm, shape, lmax=lmax, mmax=mmax)


@njit
def almxfl(
    alm: npt.NDArray[np.complexfloating], bl: npt.NDArray[np.floating], lmax: int, mmax: int
) -> npt.NDArray[np.complexfloating]:
    """
    Multiply spherical harmonic coefficients a_lm by mode-coupling factors b_l.

    Each coefficient a_lm is multiplied by the corresponding b_l factor for its
    degree l. This operation is used for convolving maps with azimuthally-symmetric
    kernels in harmonic space.

    Parameters
    ----------
    alm : ndarray
        Complex spherical harmonic coefficients.
    bl : ndarray
        Real-valued multiplicative factors for each degree l.
    lmax : int
        Maximum degree of the input a_lm.
    mmax : int
        Maximum order of the input a_lm.

    Returns
    -------
    ndarray
        Modified spherical harmonic coefficients a_lm * b_l.
    """

    # Make a copy so that we don't modify the original
    result = alm.copy()

    for l in range(lmax + 1):  # noqa: E741
        current = bl[l]
        if l < len(bl):
            current = bl[l]
        else:
            current = 0.0
        current_mmax = min([l, mmax])
        for m in range(current_mmax + 1):
            i = alm_index(l, m, lmax)
            result[i] = alm[i] * current

    return result


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
