try:
    import setuptools
    from setuptools import setup, Extension
except:
    from distutils.core import setup
    from distutils.extension import Extension
import numpy as np
from Cython.Distutils import build_ext
import sys, os, subprocess, warnings, re


found_omp = True
def set_omp_false():
    global found_omp
    found_omp = False

## Modify this to pass your own BLAS/LAPACK linkage parameters:
custom_blas_link_args = []
custom_blas_compile_args = []
# example:
# custom_blas_link_args = ["-lopenblas"]
if len(custom_blas_link_args) or len(custom_blas_compile_args):
    build_ext_with_blas = build_ext

if not (len(custom_blas_link_args) or len(custom_blas_compile_args)):
    use_findblas = (("findblas" in sys.argv)
                     or ("-findblas" in sys.argv)
                     or ("--findblas" in sys.argv))
    if os.environ.get('USE_FINDBLAS') is not None:
        use_findblas = True
    if use_findblas:
        sys.argv = [a for a in sys.argv if a not in ("findblas", "-findblas", "--findblas")]
        from findblas.distutils import build_ext_with_blas
    else:
        build_ext_with_blas = build_ext


class build_ext_subclass( build_ext_with_blas ):
    def build_extensions(self):
        
        if self.compiler.compiler_type == 'msvc':
            for e in self.extensions:
                e.extra_compile_args += ['/O2', '/openmp']
        else:
            self.add_march_native()
            self.add_openmp_linkage()
            if sys.platform[:3].lower() != "win":
                self.add_link_time_optimization()

            ### Now add arguments as appropriate for good performance
            for e in self.extensions:
                e.extra_compile_args += ['-O3', '-std=c99']
                
                # e.extra_compile_args += ['-O2', '-fopenmp', '-march=native', '-std=c99', '-ggdb']
                # e.extra_link_args += ['-fopenmp']
                # e.extra_link_args += ['-fopenmp=libiomp5']

                # e.extra_compile_args += ['-O2', '-march=native', '-std=c99', '-ggdb']
                

                # e.extra_compile_args += ['-fsanitize=address', '-static-libasan', '-ggdb']
                # e.extra_link_args += ['-fsanitize=address', '-static-libasan']                

        ## If a custom BLAS/LAPACK is provided:
        if len(custom_blas_link_args) or len(custom_blas_compile_args):
            for e in self.extensions:
                e.extra_compile_args += custom_blas_compile_args
                e.extra_link_args += custom_blas_link_args
                e.define_macros = [m for m in e.define_macros if m[0] != "USE_FINDBLAS"]

        ## SYR in OpenBLAS is currently 10-15x slower than MKL, avoid it:
        ## https://github.com/xianyi/OpenBLAS/issues/3237
        for e in self.extensions:
            has_openblas_or_atlas = False
            if not has_openblas_or_atlas:
                for arg in e.extra_compile_args:
                    if (bool(re.search("openblas", str(arg).lower()))
                        or bool(re.search("atlas", str(arg).lower()))
                    ):
                        has_openblas_or_atlas = True
                        break
            if not has_openblas_or_atlas:
                for arg in e.extra_link_args:
                    if (bool(re.search("openblas", str(arg).lower()))
                        or bool(re.search("atlas", str(arg).lower()))
                    ):
                        has_openblas_or_atlas = True
                        break
            if not has_openblas_or_atlas:
                for arg in e.define_macros:
                    if (bool(re.search("openblas", str(arg[0]).lower()))
                        or bool(re.search("atlas", str(arg[0]).lower()))
                    ):
                        has_openblas_or_atlas = True
                        break
            if has_openblas_or_atlas:
                if "AVOID_BLAS_SYR" not in [m[0] for m in e.define_macros]:
                    e.define_macros += [("AVOID_BLAS_SYR", None)]
                e.define_macros = [macro for macro in e.define_macros if macro[0] != "USE_BLAS_SYR"]

        build_ext_with_blas.build_extensions(self)

    def add_march_native(self):
        arg_march_native = "-march=native"
        arg_mcpu_native = "-mcpu=native"
        if self.test_supports_compile_arg(arg_march_native):
            for e in self.extensions:
                e.extra_compile_args.append(arg_march_native)
        elif self.test_supports_compile_arg(arg_mcpu_native):
            for e in self.extensions:
                e.extra_compile_args.append(arg_mcpu_native)

    def add_link_time_optimization(self):
        arg_lto = "-flto"
        if self.test_supports_compile_arg(arg_lto):
            for e in self.extensions:
                e.extra_compile_args.append(arg_lto)
                e.extra_link_args.append(arg_lto)

    def add_openmp_linkage(self):
        arg_omp1 = "-fopenmp"
        arg_omp2 = "-qopenmp"
        arg_omp3 = "-xopenmp"
        args_apple_omp = ["-Xclang", "-fopenmp", "-lomp"]
        if self.test_supports_compile_arg(arg_omp1):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp1)
                e.extra_link_args.append(arg_omp1)
        elif (sys.platform[:3].lower() == "dar") and self.test_supports_compile_arg(args_apple_omp):
            for e in self.extensions:
                e.extra_compile_args += ["-Xclang", "-fopenmp"]
                e.extra_link_args += ["-lomp"]
        elif self.test_supports_compile_arg(arg_omp2):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp2)
                e.extra_link_args.append(arg_omp2)
        elif self.test_supports_compile_arg(arg_omp3):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp3)
                e.extra_link_args.append(arg_omp3)
        else:
            set_omp_false()

    def test_supports_compile_arg(self, comm):
        is_supported = False
        try:
            if not hasattr(self.compiler, "compiler"):
                return False
            if not isinstance(comm, list):
                comm = [comm]
            print("--- Checking compiler support for option '%s'" % " ".join(comm))
            fname = "cmfrec_compiler_testing.c"
            with open(fname, "w") as ftest:
                ftest.write(u"int main(int argc, char**argv) {return 0;}\n")
            try:
                cmd = [self.compiler.compiler[0]]
            except:
                cmd = list(self.compiler.compiler)
            val_good = subprocess.call(cmd + [fname])
            try:
                val = subprocess.call(cmd + comm + [fname])
                is_supported = (val == val_good)
            except:
                is_supported = False
        except:
            pass
        try:
            os.remove(fname)
        except:
            pass
        return is_supported


force_openblas = (("openblas" in sys.argv)
                  or ("-openblas" in sys.argv)
                  or ("--openblas" in sys.argv))
if force_openblas:
    sys.argv = [a for a in sys.argv if a not in ("openblas", "-openblas", "--openblas")]
if os.environ.get('USE_OPENBLAS') is not None:
    force_openblas = True
if (force_openblas):
    custom_blas_link_args = ["-lopenblas"]
    from Cython.Distutils import build_ext
    build_ext_with_blas = build_ext

setup(
    name  = "cmfrec",
    packages = ["cmfrec"],
    version = '3.2.2-3',
    description = 'Collective matrix factorization',
    author = 'David Cortes',
    author_email = 'david.cortes.rivera@gmail.com',
    url = 'https://github.com/david-cortes/cmfrec',
    keywords = ['collaborative filtering', 'collective matrix factorization',
                'relational learning'],
    install_requires=[
        'cython',
        'numpy>=1.17',
        'scipy',
        'pandas>=0.25.0',
        'findblas'
    ],
    cmdclass = {'build_ext': build_ext_subclass},
    ext_modules = [
        Extension("cmfrec.wrapper_double",
            sources=["cmfrec/cfuns_double.pyx" if use_findblas else "cmfrec/cfuns_double_plusblas.pyx",
                     "src/collective.c", "src/common.c",
                     "src/offsets.c", "src/helpers.c", "src/lbfgs.c",
                     "src/cblas_wrappers.c"],
            include_dirs=[np.get_include(), "src"],
            define_macros = [("_FOR_PYTHON", None),
                             ("USE_DOUBLE", None),
                             ("USE_FINDBLAS" if use_findblas else "NO_FINDBLAS", None),
                             ("USE_BLAS_SYR" if use_findblas else "AVOID_BLAS_SYR", None)]
            ),
        Extension("cmfrec.wrapper_float",
            sources=["cmfrec/cfuns_float.pyx" if use_findblas else "cmfrec/cfuns_float_plusblas.pyx",
                     "src/collective.c", "src/common.c",
                     "src/offsets.c", "src/helpers.c", "src/lbfgs.c",
                     "src/cblas_wrappers.c"],
            include_dirs=[np.get_include(), "src"],
            define_macros = [("_FOR_PYTHON", None),
                             ("USE_FLOAT", None),
                             ("USE_FINDBLAS" if use_findblas else "NO_FINDBLAS", None),
                             ("USE_BLAS_SYR" if use_findblas else "AVOID_BLAS_SYR", None)]
            ),
        ]
)

if not found_omp:
    omp_msg  = "\n\n\nCould not detect OpenMP. Package will be built without multi-threading capabilities. "
    omp_msg += " To enable multi-threading, first install OpenMP"
    if (sys.platform[:3] == "dar"):
        omp_msg += " - for macOS: 'brew install libomp'\n"
    else:
        omp_msg += " modules for your compiler. "
    
    omp_msg += "Then reinstall this package from scratch: 'pip install --force-reinstall cmfrec'.\n"
    warnings.warn(omp_msg)
