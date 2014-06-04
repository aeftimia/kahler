import kahler
from numpy import identity, linspace, diag, asarray, pi
from numpy.random import rand
from scipy.sparse.linalg import lsqr
from scipy.linalg import eig
from itertools import product

N = 20
dim = 2

shape = [N]*dim

grid_indices = kahler.grid_indices(shape)
simplices = kahler.grid(grid_indices)
coordinates = [linspace(-pi,pi,s) for s in shape]
vertices = kahler.embed(grid_indices, coordinates)
dmetric = 0.0005 * (rand(dim, dim) - 0.5)
dmetric += dmetric.T
g=lambda _: identity(dim, dtype="complex") + dmetric
sc = kahler.SimplicialComplex(simplices, vertices, metric=g)

V_flat = kahler.derham_map(lambda pt: pt.dot(pt), sc[-1].points)
V = lsqr(sc[0].sharp, V_flat)[0]

interior = sc[0].interior
K = -sc[0].laplace_beltrami[interior, :][:, interior] + diag(V[interior])
K /= 2

val0, vec0 = eig(K)
vec0 = vec0.T

values = [(val.real, vec.real) for val, vec in zip(val0, vec0) if val.real >=.7 * dim / 2]
values.sort(key=lambda x:x[0])
eigenvalues = []
eigenvectors = []
for val, vec in values:
    eigenvalues.append(val)
    eigenvectors.append(vec)
eigenvalues = asarray(eigenvalues)

print eigenvalues[:10]

import itertools
theory_eigenvalues = asarray(sorted([sum(0.5 + asarray(m))\
    for m in product(*[range(0, dim * N)] * dim)]))
    
print theory_eigenvalues[:10]

errors=(abs(eigenvalues - theory_eigenvalues[:len(eigenvalues)]) /  theory_eigenvalues[:len(eigenvalues)])

print errors[:10] * 100
