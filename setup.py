try:
    from setuptools import setup, Extension
except:
    from distutils.core import setup
    from distutils.extension import Extension
import numpy as np
from findblas.distutils import build_ext_with_blas

## If you do not wish to use findblas:
## -uncomment the SciPy imports in 'wrapper_untyped.pxi'
##  (or manually add linkage to another blas/lapack implementation
##   in this setup file)
## -comment out the line that imports 'build_ext_with_blas'
## -uncomment the next line
# from Cython.Distutils import build_ext as build_ext_with_blas

class build_ext_subclass( build_ext_with_blas ):
    def build_extensions(self):
        compiler = self.compiler.compiler_type
        if compiler == 'msvc': # visual studio
            for e in self.extensions:
                e.extra_compile_args += ['/O2', '/openmp']
        else: # everything else that cares about following standards
            for e in self.extensions:
                e.extra_compile_args += ['-O3', '-fopenmp', '-march=native', '-std=c99', '-ggdb']
                e.extra_link_args += ['-fopenmp']
                # e.extra_compile_args += ['-O3', '-march=native', '-std=c99']
                # e.extra_compile_args += ['-fsanitize=address', '-static-libasan']
                # e.extra_link_args += ['-fsanitize=address', '-static-libasan']
                # e.extra_link_args += ["-lopenblas"]
        build_ext_with_blas.build_extensions(self)

setup(
    name  = "cmfrec",
    packages = ["cmfrec"],
    version = '1.0.0',
    description = 'Collective matrix factorization',
    author = 'David Cortes',
    author_email = 'david.cortes.rivera@gmail.com',
    url = 'https://github.com/david-cortes/cmfrec',
    keywords = ['collaborative filtering', 'collective matrix factorization',
                'relational learning'],
    install_requires=[
        'cython',
        'numpy',
        'pandas>=0.25.0',
        'findblas'
    ],
    cmdclass = {'build_ext': build_ext_subclass},
    ext_modules = [
        Extension("cmfrec.wrapper_double",
            sources=["cmfrec/cfuns_double.pyx", "src/collective.c", "src/common.c",
                     "src/offsets.c", "src/helpers.c", "src/lbfgs.c",
                     "src/cblas_wrappers.c"],
            include_dirs=[np.get_include(), "src"],
            define_macros = [("_FOR_PYTHON", None), ("USE_DOUBLE", None)]
            ),
        Extension("cmfrec.wrapper_float",
          sources=["cmfrec/cfuns_float.pyx", "src/collective.c", "src/common.c",
                   "src/offsets.c", "src/helpers.c", "src/lbfgs.c",
                   "src/cblas_wrappers.c"],
          include_dirs=[np.get_include(), "src"],
          define_macros = [("_FOR_PYTHON", None)]
          ),
        ]
    )
