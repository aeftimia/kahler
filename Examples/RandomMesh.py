import kahler
from numpy import linspace, asarray, identity, zeros, pi
from scipy.linalg import eig
from matplotlib.pylab import show

simplices, vertices, stitches = kahler.random_mesh(150, 2)
vertices *= pi
sc = kahler.SimplicialComplex(simplices, vertices)

p = 0
#p = 1 #uncomment for vector field
interior = sc[p].interior
#interior = range(sc[p].num_simplices) #uncomment for Nuemann conditions
K = (-1) ** (p + 1) * sc[p].laplace_beltrami[interior, :][:, interior]

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

from matplotlib import tri
from matplotlib import pylab
fig=pylab.figure(figsize=(10,10))
pylab.triplot(tri.Triangulation(sc.vertices[:,0].real, sc.vertices[:,1].real, sc[-1].simplices), figure=fig)

if p == 0:
    z = zeros(sc[0].num_simplices)
    z[interior] = eigenvectors[n].real

    kahler.scalar_field2d(barycenters, sc[0].sharpen(z).real, 1000, fig)
    
if p == 1:
    z = zeros(sc[1].num_simplices)
    z[interior] = eigenvectors[n].real

    kahler.vector_field2d(barycenters, sc[1].sharpen(z).real, 20, fig)

show()