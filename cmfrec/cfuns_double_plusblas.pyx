#cython: freethreading_compatible=True, language_level=3
import ctypes

ctypedef double real_t
c_real_t = ctypes.c_double
include "wrapper_untyped.pxi"
include "cython_cblas.pxi"
