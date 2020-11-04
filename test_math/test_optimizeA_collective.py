import numpy as np
import ctypes
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import test_math

m0 = int(11e0)
m1 = int(11e0)
m2 = int(13e0)
n0 = int(12e0)
n1 = int(14e0)
n2 = int(16e0)
p = int(3e0)
q = int(3e0)
k = int(4e0)

lam = 2.5
w_main = 3.2
w_user = 11.123
w_item = 0.234
w_implicit = 0.456
nthreads = 16

def gen_data():
    np.random.seed(123)
    X = np.random.gamma(1,1, size = (m,n))
    W = np.random.gamma(1,1, size = (m,n))
    U = np.random.gamma(1,1, size = (m_u,p))
    I = np.random.gamma(1,1, size = (n_i,q))
    A = np.random.normal(size = (m,k_user+k+k_main))
    B = np.random.normal(size = (n,k_item+k+k_main))
    C = np.random.normal(size = (p,k_user+k))
    D = np.random.normal(size = (q,k_item+k))
    Ai = np.empty((0,0), dtype="float64")
    Bi = np.empty((0,0), dtype="float64")
    Xones = np.empty((0,0), dtype="float64")
    if nzX > 0:
        X[np.random.randint(m,size=nzX),np.random.randint(n,size=nzX)] = np.nan
        all_NA_row = (np.isnan(X).sum(axis=1) == X.shape[1]).astype(bool)
        X[all_NA_row, 0] = 1.
        all_NA_col = (np.isnan(X).sum(axis=0) == X.shape[0]).astype(bool)
        X[0,all_NA_col] = 1.
    if nzU > 0:
        U[np.random.randint(m_u,size=nzU),np.random.randint(p,size=nzU)] = np.nan
        all_NA_row = (np.isnan(U).sum(axis=1) == U.shape[1]).astype(bool)
        U[all_NA_row, 0] = 1.
        all_NA_col = (np.isnan(U).sum(axis=0) == U.shape[0]).astype(bool)
        U[0,all_NA_col] = 1.
        
        I[np.random.randint(n_i,size=nzU),np.random.randint(q,size=nzU)] = np.nan
        all_NA_row = (np.isnan(I).sum(axis=1) == I.shape[1]).astype(bool)
        I[all_NA_row, 0] = 1.
        all_NA_col = (np.isnan(I).sum(axis=0) == I.shape[0]).astype(bool)
        I[0,all_NA_col] = 1.
    if i_f:
        Ai = np.random.normal(size = (m,k+k_main))
        Bi = np.random.normal(size = (n,k+k_main))
        Xones = (~np.isnan(X)).astype("float64")
        
    return X, W, U, I, A, B, C, D, Ai, Bi, Xones

def dense_to_sp(X, W):
    m = X.shape[0]
    n = X.shape[1]
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
        Xcsr.indptr.astype(ctypes.c_size_t),
        Xcsr.indices.astype(ctypes.c_int),
        Xcsr.data.astype(ctypes.c_double),
        Wcsr.data.astype(ctypes.c_double),
        Xcsc.indptr.astype(ctypes.c_size_t),
        Xcsc.indices.astype(ctypes.c_int),
        Xcsc.data.astype(ctypes.c_double),
        Wcsc.data.astype(ctypes.c_double)
    )
def dense_to_sp_simple(X):
    m = X.shape[0]
    n = X.shape[1]
    X_sp = X[~np.isnan(X)].reshape(-1)
    X_sp_row = np.repeat(np.arange(m), n).reshape(-1)[~np.isnan(X).reshape(-1)].astype(ctypes.c_int)
    X_sp_col = np.tile(np.arange(n), m).reshape(-1)[~np.isnan(X).reshape(-1)].astype(ctypes.c_int)
    Xcoo = coo_matrix((X_sp, (X_sp_row, X_sp_col)))
    Xcsr = csr_matrix(Xcoo)
    return (
        Xcsr.indptr.astype(ctypes.c_size_t),
        Xcsr.indices.astype(ctypes.c_int),
        Xcsr.data.astype(ctypes.c_double)
    )

empty_1d = np.empty(0, dtype=ctypes.c_double)
empty_2d = np.empty((0,0), dtype=ctypes.c_double)
empty_int = np.empty(0, dtype=ctypes.c_int)
empty_size_t = np.empty(0, dtype=ctypes.c_size_t)
buffer1 = np.empty(int(1e6), dtype=ctypes.c_double)
def get_solA():
    A = np.empty((max(m,m_u),k_user+k+k_main), dtype=ctypes.c_double)
    return test_math.py_optimizeA_collective(
        A,
        B.copy(),
        C.copy(),
        Bi.copy() if i_f else empty_2d,
        m, n,
        k, k_user, k_item, k_main,
        m_u, p,
        Xcsr_p.copy() if xtype=="sparse" else empty_size_t,
        Xcsr_i.copy() if xtype=="sparse" else empty_int,
        Xcsr.copy() if xtype=="sparse" else empty_1d,
        X.copy() if xtype=="dense" else empty_2d,
        Wpass.copy() if wtype else empty_1d,
        U_csr_p.copy() if utype=="sparse" else empty_size_t,
        U_csr_i.copy() if utype=="sparse" else empty_int,
        U_csr.copy() if utype=="sparse" else empty_1d,
        U.copy() if utype=="dense" else empty_2d,
        False,
        lam, w_main, w_user, w_implicit,
        NA_as_zero_X, NA_as_zero_U,
        as_near_dense_x, as_near_dense_u,
        nthreads,
        buffer1
        )
def get_solB():
    B = np.empty((max(n,n_i),k_item+k+k_main), dtype=ctypes.c_double)
    if n <= n_i:
        pass_isB = True
        pass_X = X
        pass_W = Wpass
    else:
        pass_isB = False
        pass_X = np.ascontiguousarray(X.T)
        if xtype=="dense":
            pass_W = np.ascontiguousarray(Wpass.reshape((m,n)).T).reshape(-1)
        else:
            pass_W = Wpass
    
    return test_math.py_optimizeA_collective(
        B,
        A.copy(),
        D.copy(),
        Ai.copy() if i_f else empty_2d,
        n, m,
        k, k_item, k_user, k_main,
        n_i, q,
        Xcsc_p.copy() if xtype=="sparse" else empty_size_t,
        Xcsc_i.copy() if xtype=="sparse" else empty_int,
        Xcsc.copy() if xtype=="sparse" else empty_1d,
        pass_X.copy() if xtype=="dense" else empty_2d,
        pass_W.copy() if wtype else empty_1d,
        I_csr_p.copy() if utype=="sparse" else empty_size_t,
        I_csr_i.copy() if utype=="sparse" else empty_int,
        I_csr.copy() if utype=="sparse" else empty_1d,
        I.copy() if utype=="dense" else empty_2d,
        pass_isB,
        lam, w_main, w_item, w_implicit,
        NA_as_zero_X, NA_as_zero_U,
        as_near_dense_x, as_near_dense_u,
        nthreads,
        buffer1
        )

def py_evalA(x):
    A = x.reshape((max(m,m_u),k_user+k+k_main))
    res = lam * A.reshape(-1).dot(A.reshape(-1))
    if wtype:
        Wuse = W.copy()
    if NA_as_zero_X:
        X_use = X.copy()
        X_use[np.isnan(X)] = 0
        if wtype:
            Wuse[np.isnan(X)] = 1
    else:
        X_use = X
    E = X_use - A[:m,k_user:].dot(B[:n,k_item:].T)
    E[np.isnan(X_use)] = 0
    if wtype:
        res += w_main * (Wuse * (E ** 2)).sum()
    else:
        res += w_main * E.reshape(-1).dot(E.reshape(-1))
    
    if NA_as_zero_U:
        U_use = U.copy()
        U_use[np.isnan(U)] = 0
    else:
        U_use = U
    E2 = U_use - A[:m_u,:k+k_user].dot(C.T)
    E2[np.isnan(U_use)] = 0
    res += w_user * E2.reshape(-1).dot(E2.reshape(-1))
    if i_f:
        Eones = A[:m,k_user:].dot(Bi.T) - Xones
        res += w_implicit * Eones.reshape(-1).dot(Eones.reshape(-1))
    return res / 2

def py_evalB(x):
    B = x.reshape((max(n,n_i),k_item+k+k_main))
    res = lam * B.reshape(-1).dot(B.reshape(-1))
    if wtype:
        Wuse = W.copy()
    if NA_as_zero_X:
        X_use = X.copy()
        X_use[np.isnan(X)] = 0
        if wtype:
            Wuse[np.isnan(X)] = 1
    else:
        X_use = X
    E = X_use - A[:m,k_user:].dot(B[:n,k_item:].T)
    E[np.isnan(X_use)] = 0
    if wtype:
        res += w_main * (Wuse * (E ** 2)).sum()
    else:
        res += w_main * E.reshape(-1).dot(E.reshape(-1))
    
    if NA_as_zero_U:
        I_use = I.copy()
        I_use[np.isnan(I)] = 0
    else:
        I_use = I
    E2 = I_use - B[:n_i,:k+k_item].dot(D.T)
    E2[np.isnan(I_use)] = 0
    res += w_item * E2.reshape(-1).dot(E2.reshape(-1))
    if i_f:
        Eones = Ai.dot(B[:n,k_item:].T) - Xones
        res += w_implicit * Eones.reshape(-1).dot(Eones.reshape(-1))
    return res / 2

xtry = ["dense", "sparse"]
utry = ["dense", "sparse"]
wtry = [False,True]
nztry = [0,1,25]
natry = [False,True]
ktry = [0,2]
ndtry = [False, True]
xlength = ["smaller", "longer", "even"]
imp_f = [False, True]


for xtype in xtry:
    for utype in utry:
        for nzX in nztry:
            for nzU in nztry:
                for NA_as_zero_X in natry:
                    for NA_as_zero_U in natry:
                        for as_near_dense_x in ndtry:
                            for as_near_dense_u in ndtry:
                                for k_user in ktry:
                                    for k_item in ktry:
                                        for k_main in ktry:
                                            for xlen in xlength:
                                                for wtype in wtry:
                                                    for i_f in imp_f:
                                
                                                        if (nzX == 0) and (as_near_dense_x or NA_as_zero_X):
                                                            continue
                                                        if (nzU == 0) and (as_near_dense_u or NA_as_zero_U):
                                                            continue
                                                        if (NA_as_zero_X) and (xtype!="sparse"):
                                                            continue
                                                        if (NA_as_zero_U) and (utype!="sparse"):
                                                            continue
                                                        if (as_near_dense_x) and (xtype!="dense"):
                                                            continue
                                                        if (as_near_dense_u) and (utype!="dense"):
                                                            continue

                                                        if xlen == "even":
                                                            m = m1
                                                            m_u = m1
                                                            n_i = n1
                                                        elif xlen == "smaller":
                                                            m = m2
                                                            m_u = m1
                                                            n_i = n1
                                                        else:
                                                            m = m1
                                                            m_u = m2
                                                            n_i = n2
                                                        n = n0
                                                            
                                                        X, W, U, I, A, B, C, D, Ai, Bi, Xones = gen_data()
                                                        Xcsr_p, Xcsr_i, Xcsr, Wcsr, \
                                                        Xcsc_p, Xcsc_i, Xcsc, Wcsc = dense_to_sp(X, W)
                                                        U_csr_p, U_csr_i, U_csr = dense_to_sp_simple(U)
                                                        
                                                        
                                                        if xtype=="sparse":
                                                            Wpass = Wcsr
                                                        else:
                                                            Wpass = W.reshape(-1).copy()
                                                        
                                                        np.random.seed(456)
                                                        x0 = np.random.normal(size = max(m,m_u)*(k_user+k+k_main))
                                                        res_scipy = minimize(py_evalA, x0)["x"].reshape((max(m,m_u),k_user+k+k_main))
                                                        res_module = get_solA()
                                                        
                                                        err1 = np.linalg.norm(res_module - res_scipy)
                                                        df1 = py_evalA(res_module.reshape(-1)) - py_evalA(res_scipy.reshape(-1))

                                                        
                                                        np.random.seed(456)
                                                        if xlen == "even":
                                                            n = n1
                                                        elif xlen == "smaller":
                                                            n = n2
                                                        else:
                                                            n = n1
                                                        m = m0
                                                        X, W, U, I, A, B, C, D, Ai, Bi, Xones = gen_data()
                                                        Xcsr_p, Xcsr_i, Xcsr, Wcsr, \
                                                        Xcsc_p, Xcsc_i, Xcsc, Wcsc = dense_to_sp(X, W)
                                                        I_csr_p, I_csr_i, I_csr = dense_to_sp_simple(I)
                                                        if xtype=="sparse":
                                                            Wpass = Wcsc
                                                        else:
                                                            Wpass = W.reshape(-1).copy()
                                                        
                                                        np.random.seed(456)
                                                        x0 = np.random.normal(size = max(n,n_i)*(k_item+k+k_main))
                                                        res_scipy = minimize(py_evalB, x0)["x"].reshape((max(n,n_i),k_item+k+k_main))
                                                        res_module = get_solB()
                                                        
                                                        err2 = np.linalg.norm(res_module - res_scipy)
                                                        df2 = py_evalB(res_module.reshape(-1)) - py_evalB(res_scipy.reshape(-1))
                                                        
                                                        
                                                        is_wrong = (err1 > 5e0) or (err2 > 5e0) or (df1 > 5e0) or (df2 > 5e0) or np.any(np.isnan(res_module))
                                                        if is_wrong:
                                                            print("\n\n\n\n****ERROR BELOW****", flush=True)
                                                        
                                                        print("[X %s] [U %s] [l:%s] [w:%d] [nz:%d,%d] [nd:%d,%d] [if:%d] [u:%d] [m:%d] [i:%d] [na:%d,%d] -> err:%.2f,%.2f df:%.2f,%.2f"
                                                              % (xtype[0], utype[0], xlen[0], int(wtype), nzX, nzU, int(as_near_dense_x), int(as_near_dense_u),
                                                                 int(i_f), k_user, k_main, k_item, int(NA_as_zero_X), int(NA_as_zero_U), err1, err2, df1, df2), flush=True)
                                                        
                                                        if is_wrong:
                                                            print("****ERROR ABOVE****\n\n\n\n", flush=True)




