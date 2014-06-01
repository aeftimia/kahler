__all__ = ['det']

from cython cimport boundscheck, wraparound, profile
from scipy.linalg._flinalg import zdet_r

@profile(True)
@boundscheck(False)
@wraparound(False)
cdef complex det(ndarray[complex, ndim=2] m):
    return zdet_r(m, True)[0]