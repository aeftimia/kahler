from numpy cimport complex, ndarray

cdef complex det(ndarray[complex, ndim=2] m)