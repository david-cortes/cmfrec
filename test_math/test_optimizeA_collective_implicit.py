import numpy as np
import ctypes
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import test_math

m1 = int(7e0)
m2 = int(9e0)
n = int(12e0)
p = int(4e0)
k = int(3e0)
nzX = int(m1*n*0.25)

lam = 2.5
alpha = 3.456
w_main = 0.234
w_user = 11.123
nthreads = 16

def gen_data():
    np.random.seed(456)
    X = np.random.gamma(1,1, size=(m,n)) + 1
    X[np.random.randint(m, size=nzX), np.random.randint(n, size=nzX)] = 0
    all_NA_row = (X==0).sum(axis=1) == n
    X[all_NA_row,0] = 3.546
    all_NA_col = (X==0).sum(axis=0) == m
    X[0,all_NA_col] = 1.123
    W = alpha*X + 1
    Xcsr = csr_matrix(X)
    X[X != 0] = 1
    
    U = np.random.normal(size = (m_u,p))
    U[np.random.randint(m_u,size=nzU), np.random.randint(p,size=nzU)] = 0
    all_NA_row = (U==0).sum(axis=1) == p
    U[all_NA_row,0] = 3.546
    all_NA_col = (U==0).sum(axis=0) == m_u
    U[0,all_NA_col] = 1.123
    Ucsr = csr_matrix(U)
    U[U == 0] = np.nan
    
    B = np.random.gamma(1,1, size=(n,k_item+k+k_main))
    C = np.random.normal(size=(p,k_user+k))
    return X, W, U, Xcsr, Ucsr, B, C

empty_1d = np.empty(0, dtype=ctypes.c_double)
empty_2d = np.empty((0,0), dtype=ctypes.c_double)
empty_int = np.empty(0, dtype=ctypes.c_int)
empty_long = np.empty(0, dtype=ctypes.c_long)
buffer1 = np.empty(int(1e6), dtype=ctypes.c_double)
buffer2 = np.empty(int(1e6), dtype=ctypes.c_double)
def get_sol():
#     A = np.empty((max(m,m_u), k_user+k+k_main))
    A = np.zeros((max(m,m_u), k_user+k+k_main), dtype=ctypes.c_double)
    return test_math.py_optimizeA_collective_implicit(
        A,
        B,
        C,
        m, n,
        k, k_user, k_item, k_main,
        m_u, p,
        Xcsr.indptr.astype(ctypes.c_long).copy(),
        Xcsr.indices.astype(ctypes.c_int).copy(),
        Xcsr.data.astype(ctypes.c_double).copy(),
        Ucsr.indptr.astype(ctypes.c_long).copy() if utype=="sparse" else empty_long,
        Ucsr.indices.astype(ctypes.c_int).copy() if utype=="sparse" else empty_int,
        Ucsr.data.astype(ctypes.c_double).copy() if utype=="sparse" else empty_1d,
        U.copy() if utype=="dense" else empty_2d,
        lam, alpha, w_main, w_user,
        na_as_zero_u, as_near_dense_u,
        nthreads,
        buffer1,
        buffer2
    )

def py_eval(values):
    A = values.reshape((max(m,m_u),k_user+k+k_main))
    errX = X - A[:m,k_user:].dot(B[:,k_item:].T)
    res = w_main * (W*(errX**2)).sum() + lam*(A**2).sum()
    predU = A[:m_u,:k_user+k].dot(C.T)
    if na_as_zero_u:
        U_use = U.copy()
        U_use[np.isnan(U)] = 0
        errU = U_use - predU
    else:
        errU = U - predU
        errU[np.isnan(U)] = 0
    res += w_user * (errU**2).sum()
    return res

utry = ["dense", "sparse"]
ktry = [0,2]
nztry = [0,1,15]
ndtry = [False,True]
natry = [False,True]
ltry = ["even", "longer", "smaller"]



for utype in utry:
    for nzU in nztry:
        for na_as_zero_u in natry:
            for as_near_dense_u in ndtry:
                for k_user in ktry:
                    for k_item in ktry:
                        for k_main in ktry:
                            for xlen in ltry:
                            
                                if (utype == "sparse") and (as_near_dense_u):
                                    continue
                                if (utype == "dense") and (na_as_zero_u):
                                    continue
                                    
                                if xlen == "even":
                                    m = m1
                                    m_u = m1
                                elif xlen == "longer":
                                    m = m2
                                    m_u = m1
                                else:
                                    m = m1
                                    m_u = m2
                                    
                                X, W, U, Xcsr, Ucsr, B, C = gen_data()
                                
                                U *= 100
                                Ucsr.data *= 100

                                x0 = np.zeros(max(m,m_u)*(k_user+k+k_main), dtype=ctypes.c_double)
                                res_scipy = minimize(py_eval, x0)["x"]
                                f1 = py_eval(res_scipy)
                                
                                res_module = get_sol()
                                f2 = py_eval(res_module.reshape(-1))

                                err = np.linalg.norm(res_scipy - res_module.reshape(-1))
                                df = f2 / f1
                                
                                is_wrong = (f2/f1 > 1.05) or np.any(np.isnan(res_module))
                                if is_wrong:
                                    print("\n\n\n****ERROR BELOW****")

                                print("[U %s] [l:%s] [nz:%d] [na:%d] [nd:%d] [u:%d] [i:%d] [m:%d] - err:%.2f, %.2f, %.2g"
                                      % (utype[0], xlen[0], nzU, na_as_zero_u, as_near_dense_u, k_user, k_item, k_main, err, df, f1), flush=True)
                                
                                if is_wrong:
                                    print("****ERROR ABOVE****\n\n\n")
