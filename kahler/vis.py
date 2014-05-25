__all__ = ['scalar_field2d', 'vector_field2d']

from numpy import atleast_2d, hstack, asarray, zeros, mgrid, complex, nanmin, nanmax
from scipy.interpolate import griddata

def scalar_field2d(barycenters, scalar_field, N, figure=None):
    N = complex(0, N)
    xmin, xmax = barycenters[:, 0].min(), barycenters[:, 0].max()
    ymin, ymax = barycenters[:, 1].min(), barycenters[:, 1].max()
    
    grid = mgrid[xmin:xmax:N, ymin:ymax:N]
    z1 = griddata(barycenters, scalar_field, grid.transpose((1,2,0)), method='linear')
    vmin = nanmin(z1)
    vmax = nanmax(z1)
    from matplotlib import pylab
    if figure == None:
        fig = pylab.figure(figsize=(10,10))
    return pylab.imshow(z1, vmin=vmin,vmax=vmax, extent=[xmin, xmax, ymin, ymax], figure=fig)
    
def vector_field2d(barycenters, vector_field, N, figure=None):
    dn = 1. / N
    N = complex(0, N)
    xmin, xmax = barycenters[:, 0].min(), barycenters[:, 0].max()
    ymin, ymax = barycenters[:, 1].min(), barycenters[:, 1].max()
    
    vector_field *= dn / sqrt((vector_field ** 2).sum(1).max())
    grid = mgrid[xmin + dn*(xmax-xmin):xmax:N, ymin + dn*(ymax-ymin):ymax:N]
    z1 = [griddata(barycenters, z0, grid.transpose((1,2,0)), method='linear') for z0 in vector_field.T]
        
    from matplotlib import pylab
    X, Y = grid
    VX, VY = z1
    if figure == None:
        fig = pylab.figure(figsize=(10,10))
    pylab.quiver(X, Y, VX, VY, sqrt(VX ** 2 + VY ** 2), scale=1, figure=fig)
    pylab.axis([xmin, xmax, ymin, ymax])