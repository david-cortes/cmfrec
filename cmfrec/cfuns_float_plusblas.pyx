#cython: language_level=3
import ctypes

ctypedef float real_t
c_real_t = ctypes.c_float
include "wrapper_untyped.pxi"
include "cython_cblas.pxi"
