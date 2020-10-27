import numpy as np
import ctypes
from scipy.optimize import minimize, check_grad, approx_fprime
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import test_math

m = int(1e1)
n = int(7e0)
k = int(3e0)
nz = int(.25*m*n)


np.random.seed(123)
X = np.random.gamma(1,1, size = (m,n))
W = np.random.gamma(1,1, size = (m,n))
X[np.random.randint(m, size=nz), np.random.randint(n, size=nz)] = np.nan

empty_1d = np.empty(0, dtype=ctypes.c_double)
empty_2d = np.empty((0,0), dtype=ctypes.c_double)
empty_int = np.empty(0, dtype=ctypes.c_int)
empty_size_t = np.empty(0, dtype=ctypes.c_size_t)
buffer_double = np.empty(int(1e6), dtype=ctypes.c_double)
buffer_mt = np.empty(int(1e6), dtype=ctypes.c_double)

def dense_to_sp(X, m, n):
    X_sp = X[~np.isnan(X)].reshape(-1)
    X_sp_row = np.repeat(np.arange(m), n).reshape(-1)[~np.isnan(X).reshape(-1)].astype(ctypes.c_int)
    X_sp_col = np.tile(np.arange(n), m).reshape(-1)[~np.isnan(X).reshape(-1)].astype(ctypes.c_int)
    W_sp = W[~np.isnan(X)].reshape(-1)
    return X_sp_row, X_sp_col, X_sp, W_sp
X_sp_row, X_sp_col, X_sp, W_sp = dense_to_sp(X, m, n)
Xcoo = coo_matrix((X_sp, (X_sp_row, X_sp_col)))
Wcoo = coo_matrix((W_sp, (X_sp_row, X_sp_col)))
Xcsr = csr_matrix(Xcoo)
Xcsc = csc_matrix(Xcoo)
Wcsr = csr_matrix(Wcoo)
Wcsc = csc_matrix(Wcoo)

def classic_fun_grad(values):
    res = test_math.py_fun_grad_classic(
        values.copy(),
        np.zeros(values.shape[0], dtype=ctypes.c_double),
        X_sp_row if xtype=="sparse" else empty_int,
        X_sp_col if xtype=="sparse" else empty_int,
        X_sp.copy() if xtype=="sparse" else empty_1d,
        X.copy() if xtype=="dense" else empty_2d,
        Xcsr.indptr.astype(ctypes.c_size_t) if (xtype=="sparse" and not mt1p) else empty_size_t,
        Xcsr.indices.astype(ctypes.c_int) if (xtype=="sparse" and not mt1p) else empty_int,
        Xcsr.data.astype(ctypes.c_double).copy() if (xtype=="sparse" and not mt1p) else empty_1d,
        Xcsc.indptr.astype(ctypes.c_size_t) if (xtype=="sparse" and not mt1p) else empty_size_t,
        Xcsc.indices.astype(ctypes.c_int) if (xtype=="sparse" and not mt1p) else empty_int,
        Xcsc.data.astype(ctypes.c_double).copy() if (xtype=="sparse" and not mt1p) else empty_1d,
        empty_1d if not wtype else (W.reshape(-1).copy() if xtype=="dense" else W_sp.copy()),
        empty_1d if not wtype else Wcsr.data.astype(ctypes.c_double).copy(),
        empty_1d if not wtype else Wcsc.data.astype(ctypes.c_double).copy(),
        m, n, k,
        bias_u, bias_i,
        nthreads,
        buffer_double,
        buffer_mt if mt1p else empty_1d
    )
    return float(res[0]), res[1].copy().astype(ctypes.c_double)
ff = lambda xval: classic_fun_grad(xval)[0]
gg = lambda xval: classic_fun_grad(xval)[1]

xtry = ["sparse", "dense"]
wtry = [False, True]
btry = [False, True]
nttry = [1, 4]
mt1ptry = [False,True]



eps_fev = 1e-8
for xtype in xtry:
    for wtype in wtry:
        for bias_u in btry:
            for bias_i in btry:
                for nthreads in nttry:
                    for mt1p in mt1ptry:
                        
                        if (mt1p) and (nthreads==1):
                            continue
                        
                        np.random.seed(123)
                        x0 = np.random.normal(size = m*bias_u + n*bias_i + k*(m+n))
                        err1 = check_grad(ff, gg, x0, epsilon=eps_fev)
                        options = {"iprint":-1}
                        r = minimize(classic_fun_grad,x0,jac=True,method="L-BFGS-B",options=options)
                        err2 = check_grad(ff, gg, r["x"], epsilon=eps_fev)
                        
                        is_wrong = (err1>1e1) or (err2>5e-1) or np.isnan(err1) or np.isnan(err2)
                        if (is_wrong):
                            print("\n\n\n\n****ERROR BELOW****", flush=True)
                        
                        print("[X %s] [b:%d,%d] [w:%d] [nt:%d] [1p:%d] - err: %.2f,  %.2f"
                              % (xtype[0], bias_u, bias_i, wtype, nthreads, mt1p, err1, err2),
                              flush=True)
                        
                        if (is_wrong):
                            print("****ERROR ABOVE****\n", flush=True)
