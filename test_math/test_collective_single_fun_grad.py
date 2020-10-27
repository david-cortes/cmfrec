import numpy as np
import ctypes
import test_math
from scipy.optimize import minimize,check_grad

n = int(1e1)
k = int(7e0)
p = int(5e0)
pbin = int(9e0)
nzX = int(n*.25)
nzU = int(p*.25)
nzUb = int(pbin*.25)

np.random.seed(123)
X_full = np.random.gamma(1,1, size = n)
U_full = np.random.gamma(1,1, size = p)
Ub_full = (np.random.random(size = pbin) >= .3).astype(ctypes.c_double)
W_full = np.random.gamma(1,1, size=n)

X_sp = X_full.copy()
U_sp = U_full.copy()
Ub_sp = Ub_full.copy()
X_sp[np.random.randint(n, size=nzX)] = np.nan
U_sp[np.random.randint(p, size=nzU)] = np.nan
Ub_sp[np.random.randint(pbin, size=nzUb)] = np.nan

def vector_dense_to_sp(X):
    X_sp = X[~np.isnan(X)]
    X_ix = np.arange(X.shape[0])[~np.isnan(X)].astype(ctypes.c_int)
    return X_ix, X_sp
X_sp_ix, X_sp_val = vector_dense_to_sp(X_sp)
U_sp_ix, U_sp_val = vector_dense_to_sp(U_sp)
W_sp = W_full[~np.isnan(X_full)]

def py_eval(values):
    errX = X_full - B[:,k_item:].dot(values[k_user:])
    errU = U_full - C.dot(values[:k_user+k])
    errUb = Ub_full - 1/(1+np.exp(-Cb.dot(values[:k_user+k])))
    res = lam*values.dot(values)
    if xtype != "full":
        errX[np.isnan(X_sp)] = 0
    if utype != "full":
        errU[np.isnan(U_sp)] = 0
    if ubin_type != "full":
        errUb[np.isnan(Ub_sp)] = 0
    if xtype != "missing":
        if weighted:
            res += w_main * ((errX**2)*W_full).sum()
        else:
            res += w_main * (errX**2).sum()
    if utype != "missing":
        res += w_user * (errU**2).sum()
    if ubin_type != "missing":
        res += w_user * (errUb**2).sum()
    return res / 2

buffer_double = np.empty(int(1e5), dtype=ctypes.c_double)
empty_1d = np.empty(0, dtype=ctypes.c_double)
empty_2d = np.empty((0,0), dtype=ctypes.c_double)
empty_int = np.empty(0, dtype=ctypes.c_int)
def py_fun_grad_single(values):
    grad = np.empty(values.shape[0], dtype=ctypes.c_double)
    return test_math.py_fun_grad_collective_single(
        values,
        grad,
        X_sp_ix if xtype=="sparse" else empty_int,
        X_sp_val if xtype=="sparse" else empty_1d,
        X_full if xtype=="full" else (X_sp if xtype=="dense" else empty_1d),
        U_sp_ix if utype=="sparse" else empty_int,
        U_sp_val if utype=="sparse" else empty_1d,
        U_full if utype=="full" else (U_sp if utype=="dense" else empty_1d),
        Ub_full if ubin_type=="full" else (Ub_sp if ubin_type=="dense" else empty_1d),
        B,
        C if utype!="missing" else empty_2d,
        Cb if ubin_type!="missing" else empty_2d,
        empty_1d if not weighted else (W_full if xtype in ["dense","full"] else W_sp),
        k, k_main, k_user, k_item,
        lam, w_user, w_main,
        buffer_double
    )
ff = lambda xval: py_fun_grad_single(xval)[0]
gg = lambda xval: py_fun_grad_single(xval)[1]
    

xtry = ["dense", "full", "sparse", "missing"]
utry = ["dense", "full", "sparse", "missing"]
ubin_try = ["dense", "full", "missing"]
ktry = [0,2]
weights = [False,True]
wtry = [1., 0.24]
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
                                    
                                    if (xtype=="missing") and (utype=="missing") and (ubin_type=="missing"):
                                        continue

                                    np.random.seed(123)
                                    B = np.random.gamma(1,1, size=(n,k_item+k+k_main))
                                    C = np.random.gamma(1,1, size=(p,k_user+k))
                                    Cb = np.random.gamma(1,1, size=(pbin,k_user+k))

                                    nvars = k_user + k + k_main
                                    x0 = np.random.normal(size = nvars)
                                    res = check_grad(ff, gg, x0)
                                    
                                    py_f = py_eval(x0)
                                    f = ff(x0)
                                    df = (f - py_f)
                                    
                                    is_wrong = res > 1e1 or np.isnan(res) or (np.abs(df) > 1e1)
                                    
                                    if is_wrong:
                                        print("\n\n****ERROR BELOW****", flush=True)
                                        
                                    print(
                                    "[X %s] [w:%d] [U %s] [Ub %s] [m:%d] [u:%d] [i:%d] [wu:%.2f] [wm:%.2f] -> err: %.2f, df: %.2f"
                                    % (xtype[0], weighted, utype[0], ubin_type[0], k_main, k_user, k_item,
                                       w_user, w_main, res, df)
                                    , flush=True
                                    )
                                        
                                    if is_wrong:
                                        print("****ERROR ABOVE****\n\n", flush=True)
