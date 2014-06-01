from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy

name = 'kahler'
modules = ['grid_utils', 'form_utils', 'skeleton', 'simplicial_complex', 'circumcenter', 'linalg', 'barycentric']
np_include = numpy.get_include()
setup(name=name,
      packages=[name],
      version='0.1',
      ext_modules=[Extension("%s.%s" % (name, x), include_dirs = [np_include], sources = ["%s/%s.pyx" % (name, x)]) for x in modules],
      cmdclass = {'build_ext': build_ext}
      )
