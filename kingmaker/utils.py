import numpy as np
from numba import guvectorize, njit
from numba import float32, float64
from scipy.special import sph_harm_y_all

_log10pi = np.log10(np.pi)

@njit
def _interp1d(x, xlow, xhigh, ylow, yhigh):
    return ylow + (yhigh-ylow)/(xhigh-xlow) * (x-xlow)

@guvectorize([(float32, float32, float32[:], float32[:], float32[:,:], float32[:]),
              (float64, float64, float64[:], float64[:], float64[:,:], float64[:])],
             '(),(),(n),(m),(n,m)->()',
             cache=True,
             target='parallel')
def _interp2d(x, y, xp, yp, z, result):
    xmax, ymax = int(len(xp)-1), int(len(yp)-1)
    xidx = np.searchsorted(xp, x, "right")
    yidx = np.searchsorted(yp, y, "right")

    # Check for edge cases
    # Handle underflow by pinning it to the first value
    if xidx == 0: xidx = 1
    if yidx == 0: yidx = 1

    # Overflow is at or above the max edge. Set it to the last value.
    if xidx > xmax: xidx = xmax
    if yidx > ymax: yidx = ymax

    # If it's in the corners, return a value immediately.
    if (xidx in [1, xmax]) and (yidx in [1, ymax]):
        result[0] = z[xidx, yidx]

    # Otherwise get the coordinates for the surrounding box
    else:
        left, right = xp[xidx-1], xp[xidx]
        bottom, top = yp[yidx-1], yp[yidx]
    
        # Get the values at the 4 surrounding points
        z_left_bottom, z_right_bottom  = z[xidx-1, yidx-1], z[xidx, yidx-1]
        z_left_top,    z_right_top     = z[xidx-1, yidx],   z[xidx, yidx]
    
        z_bottom = _interp1d(x, left, right, z_left_bottom, z_right_bottom)
        z_top    = _interp1d(x, left, right, z_left_top, z_right_top)
    
        # And return the interpolation between the low and high
        result[0] = _interp1d(y, bottom, top, z_bottom, z_top)

@njit
def adaptive_bins(alpha, npoints, scale=np.pi/4):
    linear = np.linspace(0, np.pi, npoints)
    logarithmic = np.logspace(-4, _log10pi, npoints)

    weight = alpha/scale
    if weight > 1:
        weight = 1
    return linear*weight + logarithmic*(1-weight)
    
def map2nside(skymap):
    return int(np.sqrt(len(skymap)/12))

@njit
def alm_index(l, m, lmax):
    return int(m*(2*lmax+1-m)/2+l)
    
def _reshape_alm(Ylm_all, lmax, mmax):
    nalm = (mmax+1)*(2*lmax - mmax + 2)//2
    if Ylm_all.shape[-1] == 1:
        return Ylm_all.reshape(Ylm_all.shape[:-1]), nalm
    else:
        return Ylm_all, (nalm, Ylm_all.shape[-1])

def _repack_alm(Ylm, packed_shape, lmax, mmax):
    # pack into alm order
    Ylm_packed = np.empty(packed_shape, dtype=np.complex128)
    idx = 0
    for l in range(lmax+1):
        idxmax = min(l, mmax)+1
        Ylm_packed[idx:idx+idxmax] = Ylm[l,:idxmax]
        idx += idxmax
    return Ylm_packed

def get_Ylm(lmax, mmax, theta, phi):
    Ylm = sph_harm_y_all(lmax, mmax, theta, phi)
    reduced_Ylm, shape = _reshape_alm(Ylm, lmax=lmax, mmax=mmax)
    return _repack_alm(reduced_Ylm, shape, lmax=lmax, mmax=mmax)

@njit
def almxfl(alm, bl, lmax, mmax):
    """Multiply an a_lm by a vector b_l.

    Parameters
    ----------
    alm : array, double
      The array representing the spherical harmonics coefficients
    bl : array, double
      The array giving the factor b_l by which to multiply a_lm
    lmax : Int, optional
      The maximum l of the input alm
    mmax : Int, optional
      The maximum m of the input alm
      
    Returns
    -------
    alm : array, double
      The result of a_lm * b_l.
    """

    # Make a copy so that we don't modify the original
    result = alm.copy()

    for l in range(lmax + 1):
        current = bl[l]
        if l < len(bl):
            current = bl[l]
        else:
            current = 0.
        current_mmax = min([l, mmax])
        for m in range(current_mmax + 1):
            i = alm_index(l, m, lmax)
            result[i] = alm[i] * current

    return result

@njit
def angular_distance(src_ra, src_dec, ra, dec):
   cosDist = (
       np.cos(src_ra - ra) * np.cos(src_dec) * np.cos(dec) +
       np.sin(src_dec) * np.sin(dec)
   )
   return np.arccos(cosDist)
    
@njit
def meshgrid2d(a, b):
    output_a = np.empty((len(a), len(b)), dtype=a.dtype)
    output_b = np.empty((len(a), len(b)), dtype=b.dtype)

    for i in range(len(a)):
        output_a[i,:] = a[i]
    for j in range(len(b)):
        output_b[:,j] = b[j]
    return output_a.T, output_b.T