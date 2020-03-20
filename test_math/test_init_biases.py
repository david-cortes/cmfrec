import numpy as np
import ctypes
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import test_math

m = int(1e1)
n = int(9e0)
nz = int(.25*m*n)
nthreads = 4

np.random.seed(123)
X = np.random.gamma(1,1, size=(m,n))
X[np.random.randint(m, size=nz), np.random.randint(n, size=nz)] = 0
all_NA_row = (X == 0).sum(axis = 1) == n
X[all_NA_row, 0] = 1.234
all_NA_col = (X == 0).sum(axis = 0) == m
X[0, all_NA_col] = 5.678
Xcoo = coo_matrix(X)
Xcsr = csr_matrix(Xcoo)
Xcsc = csc_matrix(Xcoo)
X[X == 0] = np.nan

glob_mean = np.nanmean(X)
bias_B = np.nanmean(X - glob_mean, axis=0)
X_minusB = X - glob_mean - bias_B.reshape((1,-1))
bias_A_AB = np.nanmean(X_minusB, axis=1)
bias_A_A = np.nanmean(X - glob_mean, axis=1)

empty_1d = np.empty(0, dtype=ctypes.c_double)
empty_2d = np.empty((0,0), dtype=ctypes.c_double)
empty_int = np.empty(0, dtype=ctypes.c_int)
empty_long = np.empty(0, dtype=ctypes.c_long)
def get_biases():
    biasA = np.empty(m, dtype=ctypes.c_double)
    biasB = np.empty(n, dtype=ctypes.c_double)
    glob_mean, resA, resB = test_math.py_initialize_biases(
        biasA if user_bias else empty_1d,
        biasB if item_bias else empty_1d,
        X.copy() if xtype=="dense" else empty_2d,
        Xcoo.row.astype(ctypes.c_int) if xtype!="dense" else empty_int,
        Xcoo.col.astype(ctypes.c_int) if xtype!="dense" else empty_int,
        Xcoo.data.astype(ctypes.c_double).copy() if xtype!="dense" else empty_1d,
        Xcsr.indptr.astype(ctypes.c_long) if xtype=="csr" else empty_long,
        Xcsr.indices.astype(ctypes.c_int) if xtype=="csr" else empty_int,
        Xcsr.data.astype(ctypes.c_double).copy() if xtype=="csr" else empty_1d,
        Xcsc.indptr.astype(ctypes.c_long) if xtype=="csr" else empty_long,
        Xcsc.indices.astype(ctypes.c_int) if xtype=="csr" else empty_int,
        Xcsc.data.astype(ctypes.c_double).copy() if xtype=="csr" else empty_1d,
        m, n, user_bias, item_bias,
        has_trans,
        nthreads
    )
    return glob_mean, resA, resB

xtry = ["dense", "sparse", "csr"]
btry = [False,True]
ttry = [False,True]
for xtype in xtry:
    for user_bias in btry:
        for item_bias in btry:
            for has_trans in ttry:
                
                if (has_trans) and (xtype!="dense"):
                    continue
                
                res_mean, resA, resB = get_biases()
                    
                diff0 = (glob_mean - res_mean)**2
                if user_bias:
                    if item_bias:
                        diff1 = np.linalg.norm(resA - bias_A_AB)
                    else:
                        diff1 = np.linalg.norm(resA - bias_A_A)
                else:
                    diff1 = 0.
                if item_bias:
                    diff2 = np.linalg.norm(resB - bias_B)
                else:
                    diff2 = 0.
                    
                is_wrong = (diff0>1e-1) or (diff1>1e0) or (diff2>1e0) or np.isnan(diff0) or np.isnan(diff1) or np.isnan(diff2)
                if is_wrong:
                    print("\n\n\n****ERROR BELOW****", flush=True)
                    
                print("[X %s] [b:%d,%d] [t:%d] - err:%.2f, %.2f, %.2f"
                      % (xtype[0], user_bias, item_bias, has_trans, diff0, diff1, diff2),
                      flush=True)
                
                if is_wrong:
                    print("****ERROR ABOVE****\n", flush=True)
