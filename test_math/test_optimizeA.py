import numpy as np
import ctypes
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import test_math

m = int(7e0)
n = int(15e0)
k = int(4e0)

lam = 2.5
w = 3.2
nthreads = 8

np.random.seed(123)
X = np.random.gamma(1,1, size=(m,n))
W = np.random.gamma(1,1, size=(m,n))

def get_sol():
    if not is_B:
        return test_math.py_optimizeA(
            A0.copy(),
            B0.copy(),
            int(m), int(n), int(k),
            int(lda), int(ldb),
            Xcsr_p.astype(ctypes.c_long) if xtype=="sparse" else empty_long,
            Xcsr_i.astype(ctypes.c_int) if xtype=="sparse" else empty_int,
            Xcsr.copy() if xtype=="sparse" else empty_1d,
            Xpass.copy() if xtype=="dense" else empty_2d,
            empty_1d if not has_weight else (Wpass.reshape(-1) if xtype=="dense" else Wcsr),
            0,
            lam, w,
            NA_as_zero,
            near_dense,
            int(nthreads),
            buffer1,
            buffer2
            )
    else:
        return test_math.py_optimizeA(
            B0.copy(),
            A0.copy(),
            int(n), int(m), int(k),
            int(ldb), int(lda),
            Xcsc_p.astype(ctypes.c_long) if xtype=="sparse" else empty_long,
            Xcsc_i.astype(ctypes.c_int) if xtype=="sparse" else empty_int,
            Xcsc.copy() if xtype=="sparse" else empty_1d,
            Xpass.copy() if xtype=="dense" else empty_2d,
            empty_1d if not has_weight else (Wpass.reshape(-1) if xtype=="dense" else Wcsc),
            1,
            lam, w,
            NA_as_zero,
            near_dense,
            int(nthreads),
            buffer1,
            buffer2
            )

def py_evalA(A, B, X, W):
    Ax = A.reshape((m,lda))[:,:k]
    Bx = B[:,:k]
    Xuse = X.copy()
    if NA_as_zero:
        Xuse[np.isnan(X)] = 0
    E = Xuse - Ax.dot(Bx.T)
    E[np.isnan(Xuse)] = 0
    if not has_weight:
        res = w * (E**2).sum()
    else:
        Wcopy = W.copy()
        if not NA_as_zero:
            Wcopy[np.isnan(X)] = 0
        else:
            Wcopy[np.isnan(X)] = 1
        res = w * (Wcopy * (E**2)).sum()
    res += lam * (Ax**2).sum()
    return res/2

def py_evalB(B, A, X, W):
    Ax = A[:,:k]
    Bx = B.reshape((n,ldb))[:,:k]
    Xuse = X.copy()
    if NA_as_zero:
        Xuse[np.isnan(X)] = 0
    E = Xuse - Ax.dot(Bx.T)
    E[np.isnan(Xuse)] = 0
    if not has_weight:
        res = w * (E**2).sum()
    else:
        Wcopy = W.copy()
        if not NA_as_zero:
            Wcopy[np.isnan(X)] = 0
        else:
            Wcopy[np.isnan(X)] = 1
        res = w * (Wcopy * (E**2)).sum()
    res += lam * (Bx**2).sum()
    return res/2

empty_1d = np.empty(0, dtype=ctypes.c_double)
empty_2d = np.empty((0,0), dtype=ctypes.c_double)
empty_int = np.empty(0, dtype=ctypes.c_int)
empty_long = np.empty(0, dtype=ctypes.c_long)
buffer1 = np.empty(int(1e6), dtype=ctypes.c_double)
buffer2 = np.empty(int(1e6), dtype=ctypes.c_double)

def dense_to_sp(X, W, m, n):
    X_sp = X[~np.isnan(X)].reshape(-1)
    W_sp = W[~np.isnan(X)].reshape(-1)
    X_sp_row = np.repeat(np.arange(m), n).reshape(-1)[~np.isnan(X).reshape(-1)].astype(ctypes.c_int)
    X_sp_col = np.tile(np.arange(n), m).reshape(-1)[~np.isnan(X).reshape(-1)].astype(ctypes.c_int)
    Xcoo = coo_matrix((X_sp, (X_sp_row, X_sp_col)))
    Wcoo = coo_matrix((W_sp, (X_sp_row, X_sp_col)))
    Xcsr = csr_matrix(Xcoo)
    Xcsc = csc_matrix(Xcoo)
    Wcsr = csr_matrix(Wcoo)
    Wcsc = csc_matrix(Wcoo)
    return (
        Xcsr.indptr.astype(ctypes.c_long),
        Xcsr.indices.astype(ctypes.c_int),
        Xcsr.data.astype(ctypes.c_double),
        Wcsr.data.astype(ctypes.c_double),
        Xcsc.indptr.astype(ctypes.c_long),
        Xcsc.indices.astype(ctypes.c_int),
        Xcsc.data.astype(ctypes.c_double),
        Wcsc.data.astype(ctypes.c_double)
    )

ld_pad = [0, 2]
nz_try = [0, 3, int(m*n*0.5)]
xtry = ["dense", "sparse"]
ndtry = [False, True]
wtry = [False, True]
natry = [False, True]


for xtype in xtry:
    for nz in nz_try:
        for ldA in ld_pad:
            for ldB in ld_pad:
                for near_dense in ndtry:
                    for has_weight in wtry:
                        for NA_as_zero in natry:
                    
                            if (near_dense) and (nz_try==0):
                                continue
                            if (NA_as_zero) and (xtype!="sparse"):
                                continue
                            lda = k + ldA
                            ldb = k + ldB
                            np.random.seed(123)
                            A0 = np.random.gamma(1,1, size=(m,lda))
                            B0 = np.random.normal(size = (n,ldb))
                            Xpass = X.copy()
                            Xpass[np.random.randint(m, size=nz), np.random.randint(n, size=nz)] = np.nan
                            Wpass = W.copy()
                                
                            Xcsr_p, Xcsr_i, Xcsr, Wcsr, Xcsc_p, Xcsc_i, Xcsc, Wcsc = dense_to_sp(Xpass, W, m, n)

                            is_B = False
                            res_scipyA = minimize(py_evalA, A0.copy().reshape(-1), (B0, Xpass, Wpass))["x"]
                            res_moduleA = get_sol().reshape(-1)

                            is_B = True
                            res_scipyB = minimize(py_evalB, B0.copy().reshape(-1), (A0, Xpass, Wpass))["x"]
                            res_moduleB = get_sol().reshape(-1)
                            
                            diffA = np.linalg.norm(res_scipyA - res_moduleA)
                            diffB = np.linalg.norm(res_scipyB - res_moduleB)
                            dfA = py_evalA(res_moduleA, B0, Xpass, Wpass) - py_evalA(res_scipyA, B0, Xpass, Wpass)
                            dfB = py_evalB(res_moduleB, A0, Xpass, Wpass) - py_evalB(res_scipyB, A0, Xpass, Wpass)
                            
                            is_wrong = (diffA>1e1) or (dfA>5e0) or (diffB>1e1) or (dfB>5e0) \
                                        or np.any(np.isnan(res_moduleA)) or np.any(np.isnan(res_moduleB))
                            
                            if is_wrong:
                                print("*****ERROR BELOW*****\n\n\n")
                                
                            print("[X %s] [w:%d] [nz:%d] [na:%d] [nd:%d] [pa:%d] [pb:%d] - err:%.2f,%.2f - df:%.2f,%.2f"
                                  % (xtype[0], has_weight, nz, NA_as_zero, near_dense, ldA, ldB, diffA, diffB, dfA, dfB),
                                  flush=True)
                                
                            if is_wrong:
                                print("\n\n\n*****ERROR ABOVE*****")
