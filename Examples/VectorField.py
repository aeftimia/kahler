import kahler
from numpy import linspace, asarray, identity, zeros, pi
from numpy.random import rand
from scipy.linalg import eig
from matplotlib.pylab import show

N = 20
dim = 2

shape = [N]*dim

grid_indices = kahler.grid_indices(shape)
simplices = kahler.grid(grid_indices)
coordinates = [linspace(0,pi,s) for s in shape]
vertices = kahler.embed(grid_indices, coordinates)
dmetric = 0.005 * (rand(dim, dim) - 0.5)
dmetric += dmetric.T
g=lambda _: identity(dim, dtype="complex") + dmetric
sc = kahler.SimplicialComplex(simplices, vertices, metric=g)

interior = sc[1].interior
#interior = range(sc[1].num_simplices) #uncomment for Nuemann conditions
K = sc[1].laplace_beltrami[interior, :][:, interior]

val0, vec0 = eig(K.todense())
vec0 = vec0.T

values = [(val.real, vec) for val, vec in zip(val0, vec0) if val.real > .9 and abs(val.imag) < 0.1]
values.sort(key=lambda x: x[0])
eigenvalues = []
eigenvectors = []
for i,value in enumerate(values):
    eigenvalues.append(value[0])
    eigenvectors.append(value[1])
eigenvalues = asarray(eigenvalues)
eigenvectors = asarray(eigenvectors)


n = 2

barycenters = sc[-1].points.mean(1).real
z = zeros(sc[1].num_simplices)
z[interior] = eigenvectors[n].real

kahler.vector_field2d(barycenters, sc[1].sharpen(z).real, 20)

show()