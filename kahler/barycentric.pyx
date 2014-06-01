from numpy import vstack, atleast_2d
from scipy.linalg import solve

from cython cimport boundscheck, wraparound, profile

@profile(True)
@boundscheck(False)
@wraparound(False)
cdef ndarray[complex, ndim=2] compute_barycentric_gradients(ndarray[complex, ndim=2] points, ndarray[complex, ndim=2] metric):
        cdef ndarray[complex, ndim=2] V = points[1:] - points[0], V1 = metric.dot(V.conj().T).T
        solve(V.dot(metric).dot(V1.T), V1, overwrite_a=True, overwrite_b=True)
        return vstack((atleast_2d(-V1.sum(0)), V1))