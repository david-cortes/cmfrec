import numpy as np
import ctypes
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import test_math

m = int(1e2)
n = int(9e1)
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

colmeans = np.nanmean(X, axis=0)
X_centered = X - colmeans.reshape((1,-1))

empty_1d = np.empty(0, dtype=ctypes.c_double)
empty_2d = np.empty((0,0), dtype=ctypes.c_double)
empty_int = np.empty(0, dtype=ctypes.c_int)
empty_size_t = np.empty(0, dtype=ctypes.c_size_t)
def get_colmeans():
    res = np.empty(n, dtype=ctypes.c_double)
    Xfull_pass = X.copy() if xtype=="dense" else empty_2d
    ixA_pass = Xcoo.row.astype(ctypes.c_int) if xtype!="dense" else empty_int
    ixB_pass = Xcoo.col.astype(ctypes.c_int) if xtype!="dense" else empty_int
    X_pass = Xcoo.data.astype(ctypes.c_double).copy() if xtype!="dense" else empty_1d
    Xcsr_p_pass = Xcsr.indptr.astype(ctypes.c_size_t) if xtype=="csr" else empty_size_t
    Xcsr_i_pass = Xcsr.indices.astype(ctypes.c_int) if xtype=="csr" else empty_int
    Xcsr_pass = Xcsr.data.astype(ctypes.c_double).copy() if xtype=="csr" else empty_1d
    Xcsc_p_pass = Xcsc.indptr.astype(ctypes.c_size_t) if xtype=="csr" else empty_size_t
    Xcsc_i_pass = Xcsc.indices.astype(ctypes.c_int) if xtype=="csr" else empty_int
    Xcsc_pass = Xcsc.data.astype(ctypes.c_double).copy() if xtype=="csr" else empty_1d
    
    res = test_math.py_center_by_cols(
        res,
        Xfull_pass,
        ixA_pass,
        ixB_pass,
        X_pass,
        Xcsr_p_pass,
        Xcsr_i_pass,
        Xcsr_pass,
        Xcsc_p_pass,
        Xcsc_i_pass,
        Xcsc_pass,
        m, n,
        nthreads
    )
    
    if xtype=="dense":
        return res, Xfull_pass
    elif xtype=="sparse":
        X_pass = np.ascontiguousarray(coo_matrix((X_pass, (Xcoo.row, Xcoo.col)), shape=(m,n)).todense())
        return res, X_pass
    else:
        Xcsr_pass = np.ascontiguousarray(csr_matrix((Xcsr_pass, Xcsr.indices, Xcsr.indptr), shape=(m,n)).todense())
        Xcsc_pass = np.ascontiguousarray(csc_matrix((Xcsc_pass, Xcsc.indices, Xcsc.indptr), shape=(m,n)).todense())
        return res, Xcsr_pass, Xcsc_pass

xtry = ["dense", "sparse", "csr"]
for xtype in xtry:
    if xtype=="csr":
        res_means, res_csr, res_csc = get_colmeans()
    else:
        res_means, res_x = get_colmeans()
    
    err1 = np.linalg.norm(res_means - colmeans)
    if xtype=="csr":
        err2 = (np.linalg.norm(res_csr[~np.isnan(X)] - X_centered[~np.isnan(X)]) +\
                np.linalg.norm(res_csc[~np.isnan(X)] - X_centered[~np.isnan(X)])) / 2
    else:
        err2 = np.linalg.norm(res_x[~np.isnan(X)] - X_centered[~np.isnan(X)])
        
    is_wrong = (err1>1e0) or (err2>1e0) or np.isnan(err1) or np.isnan(err2)
    if is_wrong:
        print("\n\n\n****ERROR BELOW****", flush=True)
    
    print("[X %s] - err:%.2f, %.2f"
          % (xtype[0], err1, err2),
          flush=True)
        
    if is_wrong:
        print("****ERROR ABOVE****\n\n\n", flush=True)
