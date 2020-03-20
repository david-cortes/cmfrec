try:
    from setuptools import setup, Extension
except:
    from distutils.core import setup
    from distutils.extension import Extension
from findblas.distutils import build_ext_with_blas
import numpy as np

class build_ext_subclass( build_ext_with_blas ):
    def build_extensions(self):
        compiler = self.compiler.compiler_type
        if compiler == 'msvc':
            for e in self.extensions:
                e.extra_compile_args += ['/O2', '/openmp']
        else:
            for e in self.extensions:
                e.extra_compile_args += ['-O2', '-fopenmp', '-march=native', '-std=c99', "-ggdb"]
                e.extra_link_args += ['-fopenmp']

                # e.extra_compile_args += ["-fsanitize=address", "-static-libasan"]
                # e.extra_link_args += ["-fsanitize=address", "-static-libasan"]
        build_ext_with_blas.build_extensions(self)


setup(
    name  = "test_math",
    packages = ["test_math"],
    cmdclass = {'build_ext': build_ext_subclass},
    ext_modules = [
        Extension("test_math.test_math",
          sources=["c_math_funs.pyx", "../src/collective.c", "../src/common.c",
                   "../src/offsets.c", "../src/helpers.c", "../src/lbfgs.c"],
          include_dirs=[np.get_include(), "../src"],
          define_macros = [("_FOR_PYTHON", None), ("USE_DOUBLE", None)]
          ),
        ]
    )
