import numpy as np
import ctypes
import test_math
from scipy.optimize import minimize

k = int(7e0)
p = int(8e0)
pbin = int(9e0)
nzU = int(p*.25)
nzUb = int(pbin*.25)

np.random.seed(123)
U_full = np.random.gamma(1,1, size = p)
Ub_full = (np.random.random(size = pbin) >= .3).astype(ctypes.c_double)

def gen_matrices(nz):
    U_sp = U_full.copy()
    Ub_sp = Ub_full.copy()
    U_sp[np.random.randint(p, size=nzU)] = np.nan
    Ub_sp[np.random.randint(pbin, size=nzUb)] = np.nan
    U_sp_as_zero = U_sp.copy()
    U_sp_as_zero[np.isnan(U_sp_as_zero)] = 0
    return U_sp, Ub_sp, U_sp_as_zero

def vector_dense_to_sp(X):
    X_sp = X[~np.isnan(X)]
    X_ix = np.arange(X.shape[0])[~np.isnan(X)].astype(ctypes.c_int)
    return X_ix, X_sp

def py_eval(values):
    if na_as_zero:
        errU = U_sp_as_zero - C.dot(values[:k_user+k])
    else:
        errU = U_full - C.dot(values[:k_user+k])
    errUb = Ub_full - 1/(1+np.exp(-Cb.dot(values[:k_user+k])))
    res = lam*values.dot(values)
    if utype != "full" and not na_as_zero:
        errU[np.isnan(U_sp)] = 0
    if ubin_type != "full":
        errUb[np.isnan(Ub_sp)] = 0
    if utype != "missing":
        res += w_user * (errU**2).sum()
    if ubin_type != "missing":
        res += w_user * (errUb**2).sum()
    return res / 2

empty_1d = np.empty(0, dtype=ctypes.c_double)
empty_2d = np.empty((0,0), dtype=ctypes.c_double)
empty_int = np.empty(0, dtype=ctypes.c_int)

def py_cold_start():
    return test_math.py_collective_cold_start(
        U_sp_ix if utype=="sparse" else empty_int,
        U_sp_val.copy() if utype=="sparse" else empty_1d,
        empty_1d if (utype=="missing" or utype=="sparse") else (U_full.copy() if utype=="full" else U_sp.copy()),
        empty_1d if ubin_type=="missing" else (Ub_full.copy() if ubin_type=="full" else Ub_sp.copy()),
        C if utype!="missing" else empty_2d,
        Cb if ubin_type!="missing" else empty_2d,
        k, k_main, k_user,
        lam, w_user,
        na_as_zero,
        precompute
    )
    

utry = ["dense", "full", "sparse", "missing"]
ubin_try = ["dense", "full", "missing"]
ktry = [0,2]
wtry = [1., 4.567]
pc_try = [False,True]
na_zero_try = [False,True]
nz_try = [0, 1, int(pbin*.5)]
lam = 2.5

for utype in utry:
    for ubin_type in ubin_try:
        for k_main in ktry:
            for k_user in ktry:
                for w_user in wtry:
                    for na_as_zero in na_zero_try:
                        for nz in nz_try:
                            for precompute in pc_try:

                                if (utype=="missing") and (ubin_type=="missing"):
                                    continue
                                if (na_as_zero) and (ubin_type!="missing" or utype!="sparse"):
                                    continue
                                    
                                U_sp, Ub_sp, U_sp_as_zero = gen_matrices(nz)
                                U_sp_ix, U_sp_val = vector_dense_to_sp(U_sp)

                                np.random.seed(123)
                                C = np.random.gamma(1,1, size=(p,k_user+k))
                                Cb = np.random.gamma(1,1, size=(pbin, k_user+k))

                                nvars = k_user + k + k_main
                                x0 = np.random.normal(size = nvars)
                                res_scipy = minimize(py_eval, x0)["x"]
                                res_module = py_cold_start()
                                diff = np.linalg.norm(res_scipy - res_module)
                                fdif = py_eval(res_module) - py_eval(res_scipy)

                                is_wrong = diff>1e1 or np.isnan(diff) or np.any(np.isnan(res_module)) or (fdif>1e0)
                                if is_wrong:
                                    print("\n\n\n*****ERROR BELOW****")
                                print("[U: %s] [Ub: %s] [u:%d] [m:%d] [w:%.1f] [pc:%d] [na:%d] [nz:%d] -> diff: %.2f, df: %.2f"
                                      % (utype[0], ubin_type[0], k_user, k_main, w_user, precompute, na_as_zero, nz, diff, fdif),
                                      flush=True)
                                if is_wrong:
                                    print("****ERROR ABOVE****\n\n\n")
