import ctypes

ctypedef float FPnum
c_FPnum = ctypes.c_float
include "wrapper_untyped.pxi"
