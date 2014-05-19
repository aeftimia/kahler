from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy

modules = ['grid_utils', 'form_utils', 'skeleton', 'simplicial_complex', 'circumcenter']
np_include = numpy.get_include()
setup(name='kahler',
      packages=['kahler'],
      version='0.1',
      ext_modules=[Extension("kahler.%s" % x, include_dirs = [np_include], sources = ["kahler/%s.pyx" % x]) for x in modules],
      cmdclass = {'build_ext': build_ext}
      )
