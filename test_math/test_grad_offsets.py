import numpy as np
import ctypes
from scipy.optimize import minimize, check_grad, approx_fprime
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import test_math

m = int(1e1)
n = int(2e1)
p = int(3e1)
q = int(1e2)
nzX = int(.25*m*n)
nzU = int(.25*m*p)
nzI = int(.25*n*q)

lam = 6e-1
w_user = 0.1234
w_item = 8.432
nthreads = 16
user_bias = True
item_bias = True

np.random.seed(123)
X = np.random.gamma(1,1, size = (m,n))
U = np.random.gamma(1,1, size = (m,p))
I = np.random.gamma(1,1, size = (n,q))
W = np.random.gamma(1,1, size = (m,n))

def assign_zeros(X, m, n, nzX):
    np.random.seed(345)
    X[np.random.randint(m, size=nzX), np.random.randint(n, size=nzX)] = 0
    all_NA_row = (X == 0).sum(axis = 1) == n
    X[all_NA_row, 0] = 1.234
    all_NA_col = (X == 0).sum(axis = 0) == m
    X[0, all_NA_col] = 4.567
    return X
X = assign_zeros(X, m, n, nzX)
U = assign_zeros(U, m, p, nzU)
I = assign_zeros(I, n, q, nzI)

Xcoo = coo_matrix(X)
Ucsr = csr_matrix(U)
Ucsc = csc_matrix(U)
Icsr = csr_matrix(I)
Icsc = csc_matrix(I)
X[X == 0] = np.nan
Wcoo = W.copy()
Wcoo[np.isnan(X)] = 0
Wcoo = coo_matrix(Wcoo)


def py_eval(values):
    pass

empty_1d = np.empty(0, dtype=ctypes.c_double)
empty_2d = np.empty((0,0), dtype=ctypes.c_double)
empty_int = np.empty(0, dtype=ctypes.c_int)
empty_long = np.empty(0, dtype=ctypes.c_long)
buffer_double = np.empty(int(1e6), dtype=ctypes.c_double)
def offsets_fun_grad(values):
    grad = np.empty(values.shape[0], dtype=ctypes.c_double)
    return test_math.py_fun_grad_offsets(
        values,
        grad,
        X.copy() if xtype=="dense" else empty_2d,
        Xcoo.row.astype(ctypes.c_int).copy() if xtype=="sparse" else empty_int,
        Xcoo.col.astype(ctypes.c_int).copy() if xtype=="sparse" else empty_int,
        Xcoo.data.astype(ctypes.c_double).copy() if xtype=="sparse" else empty_1d,
        U.copy() if utype=="dense" else empty_2d,
        Ucsr.indptr.astype(ctypes.c_long).copy() if utype=="sparse" else empty_long,
        Ucsr.indices.astype(ctypes.c_int).copy() if utype=="sparse" else empty_int,
        Ucsr.data.astype(ctypes.c_double).copy() if utype=="sparse" else empty_1d,
        Ucsc.indptr.astype(ctypes.c_long).copy() if utype=="sparse" else empty_long,
        Ucsc.indices.astype(ctypes.c_int).copy() if utype=="sparse" else empty_int,
        Ucsc.data.astype(ctypes.c_double).copy() if utype=="sparse" else empty_1d,
        I.copy() if itype=="dense" else empty_2d,
        Icsr.indptr.astype(ctypes.c_long).copy() if itype=="sparse" else empty_long,
        Icsr.indices.astype(ctypes.c_int).copy() if itype=="sparse" else empty_int,
        Icsr.data.astype(ctypes.c_double).copy() if itype=="sparse" else empty_1d,
        Icsc.indptr.astype(ctypes.c_long).copy() if itype=="sparse" else empty_long,
        Icsc.indices.astype(ctypes.c_int).copy() if itype=="sparse" else empty_int,
        Icsc.data.astype(ctypes.c_double).copy() if itype=="sparse" else empty_1d,
        empty_1d if not wtype else (W.reshape(-1).copy() if xtype=="dense" else Wcoo.data.astype(ctypes.c_double).copy()),
        m, n, p, q,
        k, k_main, k_sec,
        w_user, w_item,
        lam,
        user_bias, item_bias,
        btype,
        nthreads,
        buffer_double
    )
ff = lambda xval: offsets_fun_grad(xval)[0]
gg = lambda xval: offsets_fun_grad(xval)[1].copy().astype(ctypes.c_double)

ktry = [0,3]
xtry = ["dense", "sparse"]
wtry = [True, False]
utry = ["dense", "sparse", "missing"]
itry = ["dense", "sparse", "missing"]
btry = [True, False]


eps_fev = 1e-8
for k in ktry:
    for k_sec in ktry:
        for k_main in ktry:
            for xtype in xtry:
                for utype in utry:
                    for itype in itry:
                        for wtype in wtry:
                            for btype in btry:
                            
                                if (k==0) and (k_sec==0):
                                    continue
                                if (k==0) and (k_main==0) and (utype=="missing" and itype=="missing"):
                                    continue
                                
                                nvars = m + n + m*(k+k_main) + n*(k+k_main)
                                if utype != "missing":
                                    nvars += p*(k_sec+k)
                                    if btype:
                                        nvars += k_sec+k
                                else:
                                    nvars += m*k_sec
                                if itype != "missing":
                                    nvars += q*(k_sec+k)
                                    if btype:
                                        nvars += k_sec+k
                                else:
                                    nvars += n*k_sec
                                
                                np.random.seed(456)
                                x0 = np.random.normal(size = nvars)
                                err = check_grad(ff, gg, x0, epsilon=eps_fev)
                                
                                is_wrong = (err>1e1) or np.isnan(err)
                                if is_wrong:
                                    print("\n\n\n****ERROR BELOW****")
                                
                                print("[X %s] [U %s] [I %s] [b:%d] [k:%d,%d,%d] [w:%d] - err:%.2f"
                                      % (xtype[0], utype[0], itype[0], btype, k, k_sec, k_main, wtype, err),
                                      flush=True)
                                
                                if is_wrong:
                                    print("****ERROR ABOVE****\n\n\n")
