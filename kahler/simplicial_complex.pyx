__all__ = ['SimplicialComplex']
from numpy import asarray, arange, empty, identity, vstack, atleast_2d
from itertools import product
from scipy.sparse import csr_matrix
from scipy.linalg import solve

from .skeleton import Skeleton
from .parallel import parmap
from .grid_utils import stitch

from cython cimport boundscheck, wraparound, profile
from numpy cimport complex, ndarray   

class SimplicialComplex(_SimplicialComplex):
    """simplicial complex"""
        
    def __init__(self, simplices, vertices, metric=None, stitches={}, subdivisions=1):

        simplices.sort()

        self.complex_dimension = simplices.shape[1] - 1
        self.vertices = vertices.astype("complex")
        self.embedding_dimension = vertices.shape[1]
        self.stitches = stitches
        self.subdivisions = subdivisions
        
        if metric is None:
            default_metric = identity(self.embedding_dimension, dtype="complex")
            self.metric = lambda _: default_metric
        else:
            self.metric = metric
        
        for dim in range(self.complex_dimension):
            skeleton = Skeleton()
            skeleton.complex = self
            skeleton.dim = dim
            self.append(skeleton)
            
        skeleton = Skeleton()
        skeleton.complex = self
        skeleton.dim = self.complex_dimension
        skeleton.simplices = asarray([stitch(s, stitches) for s in simplices])
        skeleton.simplex_to_index = dict([(frozenset(stitch(simplex, stitches)), index) for index, simplex in enumerate(skeleton.simplices)])
        skeleton.num_simplices = simplices.shape[0]
        skeleton.exterior_derivative = csr_matrix((1, skeleton.num_simplices), dtype="int8")
        skeleton.boundary = asarray([], dtype="uint")
        skeleton.interior = arange(skeleton.num_simplices, dtype="uint")
        skeleton.unstitched = [tuple(s) for s in simplices]
        skeleton.points = vertices[simplices]
        
        self.append(skeleton)

    def __repr__(self):
        output = "SimplicialComplex:\n"
        for i in reversed(range(len(self))):
            output += "   %10d: %2d-simplices\n" % (self[i].num_simplices, i)
        return output

    def __getattr__(self, attr):
        if attr == "barycentric_gradients":
            barycentric_gradients = parmap(lambda stuff: (self[-1].simplex_to_index[frozenset(stitch(stuff[0], self.stitches))], \
                                            self.compute_barycentric_gradients(stuff[1], stuff[2])), \
                                            zip(self[-1].unstitched, self[-1].points, self[-1].metrics))
            self.barycentric_gradients = empty((self[-1].num_simplices, self.complex_dimension + 1, self.embedding_dimension), dtype="complex")
            for index, bc in barycentric_gradients:
                self.barycentric_gradients[index] = bc
            return self.barycentric_gradients
        else:
            raise AttributeError(attr + " not found")

cdef class _SimplicialComplex(list):
    @profile(True)
    @boundscheck(False)
    @wraparound(False)
    cpdef ndarray[complex, ndim=2] compute_barycentric_gradients(self, ndarray[complex, ndim=2] points, ndarray[complex, ndim=2] metric):
        cdef ndarray[complex, ndim=2] V = points[1:] - points[0], V1 = metric.dot(V.conj().T).T
        solve(V.dot(metric).dot(V1.T), V1, overwrite_a=True, overwrite_b=True, check_finite=False)
        return vstack((atleast_2d(-V1.sum(0)), V1))

    @profile(True)
    @boundscheck(False)
    @wraparound(False)
    cpdef list compute_dual_cells(self, tuple simplex, unsigned char p):
        cdef unsigned char i, k1 = len(simplex), k = k1 - 1
        cdef long unsigned int index
        cdef list new_circumcenters = [], old_circumcenters, old_points, new_points, circumcenter = [self[k].circumcenters[simplex]]

        if p == k:
            return [(circumcenter, circumcenter, self[k].simplex_to_index[frozenset(stitch(simplex, self.stitches))])]

        for i in range(k1):
            new_points = [self.vertices[simplex[i]]]
            for old_circumcenters, old_points, index in self.compute_dual_cells(simplex[:i] + simplex[i + 1:], p):
                new_circumcenters.append((old_circumcenters + circumcenter, old_points + new_points, index))
            
        return new_circumcenters