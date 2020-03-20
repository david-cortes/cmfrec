import ctypes

ctypedef double FPnum
c_FPnum = ctypes.c_double
include "wrapper_untyped.pxi"
