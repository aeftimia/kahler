from numpy cimport complex, ndarray

cdef ndarray[complex, ndim=1] compute_circumcenter(ndarray[complex, ndim=2] points, ndarray[complex, ndim=2] metric)
