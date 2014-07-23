__all__ = ['det']

from scipy.linalg._flinalg import zdet_r

from cython cimport boundscheck, wraparound

@boundscheck(False)
@wraparound(False)
cdef complex det(ndarray[complex, ndim=2] m):
    return zdet_r(m, True)[0]