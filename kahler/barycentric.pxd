from numpy cimport complex, ndarray

cdef ndarray[complex, ndim=2] compute_barycentric_gradients(ndarray[complex, ndim=2] points, ndarray[complex, ndim=2] metric)