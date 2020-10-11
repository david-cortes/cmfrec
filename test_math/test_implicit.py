import numpy as np
import ctypes
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csr_matrix
import test_math

m = int(1e1)
n = int(9e0)
k = int(3e0)
nz = int(m*n*0.25)


lam = 2.5
alpha = 4.234
w = 3.4456

nthreads = 4
buffer_double = np.empty(int(1e6), dtype=ctypes.c_double)

np.random.seed(123)
X = np.random.gamma(1,1, size = (m,n)) + 1
X[np.random.randint(m, size=nz), np.random.randint(n, size=nz)] = 0
all_NA_row = ((X==0).sum(axis = 1) == n)
X[all_NA_row, 0] = 4.567
all_NA_col = ((X==0).sum(axis = 0) == m)
X[0, all_NA_col] = 8.9123
Xcsr = csr_matrix(X)

C = alpha*X + 1
X[X != 0] = 1
B = np.random.normal(size = (n,k))

empty_A = np.zeros((m,k), dtype=ctypes.c_double)

def py_eval(values):
    A = values.reshape((m,k))
    E = (X - A.dot(B.T)) ** 2
    return (C*E).sum() + lam * (A**2).sum()

x0 = np.random.normal(size = m*k)
res_scipy = minimize(py_eval, x0)["x"].reshape((m,k))
res_module = test_math.py_optimizeA_implicit(
    empty_A,
    B.copy(),
    m, n, k,
    Xcsr.indptr.astype(ctypes.c_long).copy(),
    Xcsr.indices.astype(ctypes.c_int).copy(),
    Xcsr.data.astype(ctypes.c_double).copy(),
    lam, alpha, w,
    nthreads,
    buffer_double
    )

err = np.linalg.norm(res_scipy - res_module)
f1 = py_eval(res_module.reshape(-1))
f2 = py_eval(res_scipy.reshape(-1))
df_pct = 100 * (f1/f2 - 1)
is_wrong = (df_pct > 5.) or (err > 1e1) or np.any(np.isnan(res_module))
if is_wrong:
    print("****ERROR****\n\n", flush=True)
print("err:%.2f - df(pct):%.2f%%" % (err, df_pct), flush=True)
if is_wrong:
    print("\n\n****ERROR****", flush=True)
