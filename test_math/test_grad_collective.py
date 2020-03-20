import numpy as np
import ctypes
from scipy.optimize import minimize, check_grad
import test_math

m = int(1e1)
n = int(2e1)
p = int(3e1)
q = int(1e2)
pbin = int(3e0)
qbin = int(5e0)
k = int(7e0)
nz = int(.25*m*n)

lam = 6e-1
w_main = 2.345
w_user = 0.1234
w_item = 8.432
nthreads = 1

np.random.seed(123)
X = np.random.gamma(1,1, size = (m,n))
U = np.random.gamma(1,1, size = (m,p))
I = np.random.gamma(1,1, size = (n,q))
Ubin = (np.random.random(size = (m,pbin)) >= .7).astype(ctypes.c_double)
Ibin = (np.random.random(size = (n,qbin)) >= .3).astype(ctypes.c_double)
W = np.random.gamma(1,1, size = (m,n))

X[np.random.randint(m, size=nz), np.random.randint(n, size=nz)] = np.nan
U[np.random.randint(m, size=nz), np.random.randint(p, size=nz)] = np.nan
I[np.random.randint(n, size=nz), np.random.randint(q, size=nz)] = np.nan
w = W[~np.isnan(X)].reshape(-1)
W = W.reshape(-1)

empty_1d = np.empty(0, dtype=ctypes.c_double)
empty_2d = np.empty((0,0), dtype=ctypes.c_double)
empty_int = np.empty(0, dtype=ctypes.c_int)
buffer_float = np.empty(int(1e6), dtype=ctypes.c_double)

def dense_to_sp(X, m, n):
    X_sp = X[~np.isnan(X)].reshape(-1)
    X_sp_row = np.repeat(np.arange(m), n).reshape(-1)[~np.isnan(X).reshape(-1)].astype(ctypes.c_int)
    X_sp_col = np.tile(np.arange(n), m).reshape(-1)[~np.isnan(X).reshape(-1)].astype(ctypes.c_int)
    return X_sp_row, X_sp_col, X_sp
X_sp_row, X_sp_col, X_sp = dense_to_sp(X, m, n)
U_sp_row, U_sp_col, U_sp = dense_to_sp(U, m, p)
I_sp_row, I_sp_col, I_sp = dense_to_sp(I, n, q)

def sq_sum(x):
    x1 = x.reshape(-1)
    x1 = x1[~np.isnan(x1)]
    return x1.dot(x1)

def calc_nvars(xtype, utype, itype, ubin_type, ibin_type, bias_u, bias_i):
    nvars = (m+n)*k
    if bias_u:
        nvars += m
    if bias_i:
        nvars += n
    if xtype != "missing":
        nvars += (m+n)*k_main
    if utype != "missing":
        nvars += p*(k_user+k)
    if ubin_type != "missing":
        nvars += pbin*(k_user+k)
    if (utype != "missing") or (ubin_type != "missing"):
        nvars += m*k_user
    if itype != "missing":
        nvars += q*(k_item+k)
    if ibin_type != "missing":
        nvars += qbin*(k_item+k)
    if (itype != "missing") or (ibin_type != "missing"):
        nvars += n*k_item
    return nvars

def py_eval_fun(values):
    edge = 0
    predX = np.zeros((m,n))
    predU = np.zeros((m,p))
    predI = np.zeros((n,q))
    predUbin = np.zeros((m,pbin))
    predIbin = np.zeros((n,qbin))
    reg = 0
    has_u = (utype != "missing") or (ubin_type != "missing")
    has_i = (itype != "missing") or (ibin_type != "missing")
    
    if bias_u and xtype!="missing":
        biasA = values[edge:edge+m].reshape((-1,1))
        edge += m
        predX += biasA
        reg += sq_sum(biasA)
    if bias_i and xtype!="missing":
        biasB = values[edge:edge+n].reshape((1,-1))
        predX += biasB
        edge += n
        reg += sq_sum(biasB)
    k_a = k
    k_b = k
    if xtype != "missing":
        k_a += k_main
        k_b += k_main
    if has_u:
        k_a += k_user
    if has_i:
        k_b += k_item
    if has_u or xtype!="missing":
        A = values[edge:edge+m*k_a].reshape((m,k_a))
        edge += m*k_a
        reg += sq_sum(A)
    if has_i or xtype!="missing":
        B = values[edge:edge+n*k_b].reshape((n,k_b))
        edge += n*k_b
        reg += sq_sum(B)
    if has_u:
        Ax = A[:,k_user:]
    elif xtype!="missing":
        Ax = A
    if has_i:
        Bx = B[:,k_item:]
    elif xtype!="missing":
        Bx = B
    if xtype != "missing":
        predX += Ax.dot(Bx.T)
    
    if utype!="missing":
        Au = A[:,:k_user+k]
        C = values[edge:edge+p*(k_user+k)].reshape((p,k_user+k))
        predU += Au.dot(C.T)
        reg += sq_sum(C)
        edge += p*(k_user+k)
    if (ubin_type != "missing"):
        Au = A[:,:k_user+k]
        Cb = values[edge:edge+pbin*(k_user+k)].reshape((pbin,k_user+k))
        predUbin += Au.dot(Cb.T)
        reg += sq_sum(Cb)
        edge += pbin*(k_user+k)
    if itype!="missing":
        Bi = B[:,:k_item+k]
        D = values[edge:edge+q*(k_item+k)].reshape((q,k_item+k))
        predI += Bi.dot(D.T)
        reg += sq_sum(D)
        edge += q*(k_item+k)
    if (ibin_type != "missing"):
        Bi = B[:,:k_item+k]
        Db = values[edge:edge+qbin*(k_item+k)].reshape((qbin,k_item+k))
        predIbin += Bi.dot(Db.T)
        reg += sq_sum(Db)
        edge += qbin*(k_item+k)
        
    fval = 0
    if xtype != "missing":
        if not weighted:
            fval += w_main * sq_sum(X - predX)
        else:
            Esq = (X - predX) ** 2
            Esq *= W.reshape((m,n))
            Esq = Esq[~np.isnan(X)]
            fval += w_main * np.sum(Esq)
    if utype != "missing":
        fval += w_user * sq_sum(U - predU)
    if itype != "missing":
        fval += w_item * sq_sum(I - predI)
    if ubin_type != "missing":
        fval += w_user * sq_sum(Ubin - 1/(1+np.exp(-predUbin)))
    if ibin_type != "missing":
        fval += w_item * sq_sum(Ibin - 1/(1+np.exp(-predIbin)))
    return (fval + lam*reg) / 2

def collective_fun_grad(values):
    return test_math.py_fun_grad_collective(
        values,
        np.zeros(values.shape[0], dtype=ctypes.c_double),
        X if xtype=="dense" else empty_2d,
        X_sp_row if xtype=="sparse" else empty_int,
        X_sp_col if xtype=="sparse" else empty_int,
        X_sp if xtype=="sparse" else empty_1d,
        U if utype=="dense" else empty_2d,
        U_sp_row if utype=="sparse" else empty_int,
        U_sp_col if utype=="sparse" else empty_int,
        U_sp if utype=="sparse" else empty_1d,
        I if itype=="dense" else empty_2d,
        I_sp_row if itype=="sparse" else empty_int,
        I_sp_col if itype=="sparse" else empty_int,
        I_sp if itype=="sparse" else empty_1d,
        Ubin if ubin_type=="dense" else empty_2d,
        Ibin if ibin_type=="dense" else empty_2d,
        W if (xtype=="dense" and weighted) else (w if xtype=="sparse" and weighted else empty_1d),
        m, n,
        p if utype!="missing" else 0, q if itype!="missing" else 0,
        pbin if ubin_type!="missing" else 0, qbin if ibin_type!="missing" else 0,
        k, k_main if xtype!="missing" else 0,
        k_user if (utype!="missing" or ubin_type!="missing") else 0,
        k_item if (itype!="missing" or ibin_type!="missing") else 0,
        w_main, w_user, w_item,
        lam, empty_1d,
        bias_u, bias_i,
        nthreads,
        buffer_float
    )

ff = lambda xval: collective_fun_grad(xval)[0]
gg = lambda xval: collective_fun_grad(xval)[1].copy().astype(ctypes.c_double)

xtry = ["dense", "sparse"]
utry = ["dense", "sparse", "missing"]
itry = ["dense", "sparse", "missing"]
ubin_try = ["dense", "missing"]
ibin_try = ["dense", "missing"]
ktry = [0, 2]
bias = [True, False]
wtry = [True, False]

eps_fev = 1e-8
for xtype in xtry:
    for k_main in ktry:
        for k_user in ktry:
            for k_item in ktry:
                for utype in utry:
                    for itype in itry:
                        for ubin_type in ubin_try:
                            for ibin_type in ibin_try:
                                for bias_u in bias:
                                    for bias_i in bias:
                                        for weighted in wtry:
                                            if (xtype == "missing") and (utry == "missing")\
                                                and (itry == "missing") and (ubin_try == "missing")\
                                                and (ibin_try == "missing"):
                                                continue
                                            if (xtype == "missing") and (bias_u or bias_i or k_main or weighted):
                                                continue
                                            if (utype=="missing") and (ubin_type=="missing") and (k_user):
                                                continue
                                            if (itype=="missing") and (ibin_type=="missing") and (k_item):
                                                continue

                                            nvars = calc_nvars(xtype, utype, itype, ubin_type, ibin_type, bias_u, bias_i)
                                            np.random.seed(321)
                                            x0 = np.random.normal(size = nvars)
                                            err0 = check_grad(ff, gg, x0, epsilon=eps_fev)
                                            f_py = py_eval_fun(x0)
                                            f_cy = ff(x0)
                                            df = np.abs(f_cy - f_py)
                                            if (err0 > 1e1) or (df > 1e1):
                                                print("\n\n\n\n****ERROR BELOW****", flush=True)
                                            print("[X %s] [b:%d,%d] [W:%d] [U %s] [I %s] [Ub %s] [Ib %s] [m:%d] [u:%d] [i:%d] -> err: %.2f, df:%.1f" %
                                                  (xtype[0], bias_u, bias_i, weighted, utype[0], itype[0], ubin_type[0], ibin_type[0], k_main, k_user, k_item, err0, df), flush=True)
                                            if (err0 > 1e1) or (df > 1e1):
                                                print("****ERROR ABOVE****\n\n\n\n", flush=True)

