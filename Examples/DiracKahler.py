import kahler
from numpy import linspace, asarray, identity, zeros, pi, sqrt
from numpy.random import rand
from scipy.linalg import eigvals
from scipy.sparse import bmat
from itertools import product

N = 10
dim = 2

shape = [N]*dim

grid_indices = kahler.grid_indices(shape)
simplices = kahler.grid(grid_indices)
coordinates = [linspace(0,pi,s) for s in shape]
vertices = kahler.embed(grid_indices, coordinates)
dmetric = 0.0005 * (rand(dim, dim) - 0.5)
dmetric += dmetric.T
g=lambda _: identity(dim, dtype="complex") + dmetric
sc = kahler.SimplicialComplex(simplices, vertices, metric=g)


K=[]
for i, i_skel in enumerate(sc):
    k = []
    i_interior = i_skel.interior
    for j, j_skel in enumerate(sc):
        j_interior = j_skel.interior
        if j == i - 1:
            k.append(j_skel.exterior_derivative[i_interior, :][:, j_interior])
        elif j == i + 1:
            k.append((-1) ** dim * j_skel.codifferential[i_interior, :][:, j_interior])
        else:
            k.append(None)
    K.append(k)

K = bmat(K)

val0 = eigvals(K.todense())
eigenvalues = [val.real for val in val0 if val.real > .9 and abs(val.imag) < 0.1]
eigenvalues.sort()

print eigenvalues[:10]

theory_eigenvalues = sorted([asarray(m).dot(m) for m in product(*[range(0, 40)]*dim)])
theory_eigenvalues = asarray([sqrt(t) for t in theory_eigenvalues if t >= .2])

print theory_eigenvalues[:10]

errors=(abs(eigenvalues[:len(theory_eigenvalues)] - theory_eigenvalues[:len(eigenvalues)]) /  theory_eigenvalues[:len(eigenvalues)])
print errors[:10] * 100

