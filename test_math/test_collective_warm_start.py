import numpy as np
import ctypes
import test_math
from scipy.optimize import minimize

n = int(12)
k = int(7e0)
p = int(11)
pbin = int(5e0)
nzUb = int(pbin*.25)

X_full = np.random.gamma(1,1, size = n)
U_full = np.random.gamma(1,1, size = p)
Ub_full = (np.random.random(size = pbin) >= .3).astype(ctypes.c_double)
W_full = np.random.gamma(1,1, size=n)

def gen_data():
    np.random.seed(123)
    X_sp = X_full.copy()
    U_sp = U_full.copy()
    Ub_sp = Ub_full.copy()
    X_sp[np.random.randint(n, size=nzX)] = np.nan
    U_sp[np.random.randint(p, size=nzU)] = np.nan
    Ub_sp[np.random.randint(pbin, size=nzUb)] = np.nan
    X_sp_ix, X_sp_val = vector_dense_to_sp(X_sp)
    U_sp_ix, U_sp_val = vector_dense_to_sp(U_sp)
    W_sp = W_full[~np.isnan(X_sp)]
    return X_sp, U_sp, Ub_sp, X_sp_ix, X_sp_val, U_sp_ix, U_sp_val, W_sp

def vector_dense_to_sp(X):
    X_sp = X[~np.isnan(X)]
    X_ix = np.arange(X.shape[0])[~np.isnan(X)].astype(ctypes.c_int)
    return X_ix, X_sp

def py_eval(values):
    errX = X_full - B[:,k_item:].dot(values[k_user:])
    errU = U_full - C.dot(values[:k_user+k])
    errUb = Ub_full - 1/(1+np.exp(-Cb.dot(values[:k_user+k])))
    res = lam*values.dot(values)
    if xtype != "full" and not na_as_zero_x:
        errX[np.isnan(X_sp)] = 0
    if utype != "full" and not na_as_zero_u:
        errU[np.isnan(U_sp)] = 0
    if ubin_type != "full":
        errUb[np.isnan(Ub_sp)] = 0
    if weighted:
        res += w_main * ((errX**2)*W_full).sum()
    else:
        res += w_main * (errX**2).sum()
    if utype != "missing":
        res += w_user * (errU**2).sum()
    if ubin_type != "missing":
        res += w_user * (errUb**2).sum()
    return res / 2

empty_1d = np.empty(0, dtype=ctypes.c_double)
empty_2d = np.empty((0,0), dtype=ctypes.c_double)
empty_int = np.empty(0, dtype=ctypes.c_int)
def py_warm_factors():
    outp = np.zeros(k_user+k+k_main, dtype=ctypes.c_double)
    return test_math.py_collective_factors(
        outp,
        X_sp_ix if xtype=="sparse" else empty_int,
        X_sp_val.copy() if xtype=="sparse" else empty_1d,
        X_full.copy() if xtype=="full" else (X_sp.copy() if xtype=="dense" else empty_1d),
        U_sp_ix if utype=="sparse" else empty_int,
        U_sp_val.copy() if utype=="sparse" else empty_1d,
        U_full.copy() if utype=="full" else (U_sp.copy() if utype=="dense" else empty_1d),
        Ub_full.copy() if ubin_type=="full" else (Ub_sp.copy() if ubin_type=="dense" else empty_1d),
        B,
        C if utype!="missing" else empty_2d,
        Cb if ubin_type!="missing" else empty_2d,
        empty_1d if not weighted else (W_full if xtype in ["dense","full"] else W_sp),
        k, k_main, k_user, k_item,
        lam, w_user, w_main,
        na_as_zero_x, na_as_zero_u,
        precompute
    )
    

xtry = ["dense", "full", "sparse"]
utry = ["dense", "full", "sparse", "missing"]
ubin_try = ["dense", "full", "missing"]
ktry = [0,2]
weights = [False,True]
wtry = [1., 0.24]
pctry = [False,True]
na_try = [False, True]
nz_try = [0,1,4]
lam = 2.5


for xtype in xtry:
    for utype in utry:
        for ubin_type in ubin_try:
            for k_main in ktry:
                for k_user in ktry:
                    for k_item in ktry:
                        for weighted in weights:
                            for w_main in wtry:
                                for w_user in wtry:
                                    for precompute in pctry:
                                        for na_as_zero_u in na_try:
                                            for na_as_zero_x in na_try:
                                                for nzU in nz_try:
                                                    for nzX in nz_try:

                                                        if (na_as_zero_x) and (xtype!="sparse"):
                                                            continue
                                                        if (na_as_zero_u) and (utype!="sparse"):
                                                            continue
                                                        if (na_as_zero_x or na_as_zero_u) and (ubin_type!="missing"):
                                                            continue
                                                        if (w_user != 1) and (utype=="missing" and ubin_type=="missing"):
                                                            continue
                                                        if (nzX) and (xtype=="full"):
                                                            continue
                                                        if (nzU) and (utype=="missing" or utype=="full"):
                                                            continue
                                                        if (nzU < 2) and (utype=="sparse"):
                                                            continue
                                                        if (nzX < 2) and (xtype=="sparse"):
                                                            continue
                                                        if (ubin_type=="missing") and (nzX>1 or nzU>1):
                                                            continue

                                                        np.random.seed(123)
                                                        B = np.random.gamma(1,1, size=(n,k_item+k+k_main))
                                                        C = np.random.gamma(1,1, size=(p,k_user+k))
                                                        Cb = np.random.gamma(1,1, size=(pbin,k_user+k))
                                                        
                                                        X_sp, U_sp, Ub_sp, X_sp_ix, X_sp_val, U_sp_ix, U_sp_val, W_sp =\
                                                            gen_data()

                                                        nvars = k_user + k + k_main
                                                        x0 = np.random.normal(size = nvars)
                                                        r_scipy = minimize(py_eval, x0)["x"]
                                                        r_module = py_warm_factors()
                                                        diff = np.linalg.norm(r_scipy - r_module)
                                                        df = py_eval(r_module) - py_eval(r_scipy)
                                                        #is_wrong = (diff >= .25*np.linalg.norm(r_scipy - x0)) or np.any(np.isnan(r_module))
                                                        is_wrong = (diff > 1e1) or (np.isnan(diff)) or (np.any(np.isnan(r_module))) or (df > 1e1) or (np.isnan(df))
                                                        if is_wrong:
                                                            print("\n\n\n\n****ERROR BELOW****", flush=True)

                                                        print(
                                                        "[X %s] [w:%d] [U %s] [Ub %s] [m:%d] [u:%d] [i:%d] [wu:%.2f] [wm:%.2f] [pc:%d] [nax:%d] [nau:%d] [nz:%d,%d] -> diff: %.2f, %.2f"
                                                        % (xtype[0], weighted, utype[0], ubin_type[0], k_main, k_user, k_item,
                                                        w_user, w_main, precompute, na_as_zero_x, na_as_zero_u, nzX, nzU, diff, df)
                                                        , flush=True
                                                        )

                                                        if is_wrong:
                                                            print("****ERROR ABOVE****\n\n\n\n", flush=True)
