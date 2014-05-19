__all__ = ['compute_circumcenter']

from numpy import ones, empty
from numpy.linalg import solve
from cython cimport boundscheck, wraparound, profile

@profile(True)
@boundscheck(False)
@wraparound(False)
cdef ndarray[complex, ndim=1] compute_circumcenter(ndarray[complex, ndim=2] points, ndarray[complex, ndim=2] metric):
    cdef list A = [], b = []
    cdef unsigned char i, k0 = points.shape[0] 
    cdef ndarray[complex, ndim=1] circumcenter = empty((points.shape[1],), dtype="complex"), ref = points[0]
    cdef ndarray[complex, ndim=2] points_conj = points.conj()
    cdef complex ref2 = ref.conj().dot(metric).dot(ref)
    
    for i in range(1, k0):
        A.append(2 * points_conj.dot(metric).dot(points[i] - ref).real)
        b.append(points_conj[i].dot(metric).dot(points[i]) - ref2)
    A.append(ones(k0))
    b.append(1)

    return solve(A, b).dot(points)
