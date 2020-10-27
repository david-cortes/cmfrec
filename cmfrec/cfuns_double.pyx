import ctypes

ctypedef double real_t
c_real_t = ctypes.c_double
include "wrapper_untyped.pxi"
