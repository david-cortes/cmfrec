## Test that the closed-form function with no side info gives
## the same result as gradient-based approaches and SciPy's
## optimizers with finite-differencing.

import numpy as np
import ctypes
from scipy.optimize import minimize
import test_math

n = int(1e1)
k = int(5e0)
nz = int(.5*n)
lam = 3e0


np.random.seed(123)
B = np.random.gamma(1,1, size = (n,k)).astype(ctypes.c_double)
x = np.random.gamma(1,1, size = n).astype(ctypes.c_double)
x[np.random.randint(n, size = nz)] = np.nan
x_sp_ix = np.arange(n)[~np.isnan(x)].astype(ctypes.c_int)
x_sp_val = x[~np.isnan(x)]
w = np.random.gamma(10,12, size = n).astype(ctypes.c_double)
W = np.diag(w)
w_sp = w[~np.isnan(x)]

a0 = np.random.random(size = k).astype(ctypes.c_double)
catcher = np.empty(k, dtype=ctypes.c_double)
buffer = np.empty(int(1e6), dtype=ctypes.c_double)

empty_1d = np.empty(0, dtype=ctypes.c_double)
empty_2d = np.empty((0,0), dtype=ctypes.c_double)
empty_int = np.empty(0, dtype=ctypes.c_int)

def feval(a):
    pred = B.dot(a.reshape((-1,1))).reshape(-1)
    err = pred - x
    err = np.sum((err[~np.isnan(err)] ** 2) * w[~np.isnan(err)])
    return err + lam*a.dot(a)
def py_closed_form():
    Bsel = B[~np.isnan(x)]
    xsel = x[~np.isnan(x)]
    Wsel = np.diag(w[~np.isnan(x)])
    return np.linalg.inv(Bsel.T.dot(Wsel.dot(Bsel)) + lam*np.eye(k)).dot(Wsel.dot(Bsel).T.dot(xsel))

### First with weights
res_scipy = minimize(feval, a0)["x"]
res_scipy_nn = minimize(feval, a0, bounds=[(0., None)]*k, method="TNC")["x"]
res_py = py_closed_form()
res_closedform_dense = test_math.py_factors_closed_form(
    catcher,
    x.copy(),
    empty_int,
    empty_1d,
    B.copy(),
    w.copy(),
    buffer,
    k, lam
).copy()
res_closedform_sparse = test_math.py_factors_closed_form(
    catcher,
    empty_1d,
    x_sp_ix,
    x_sp_val.copy(),
    B.copy(),
    w_sp.copy(),
    buffer,
    k, lam
).copy()
res_closedform_dense_nn = test_math.py_factors_closed_form(
    catcher,
    x.copy(),
    empty_int,
    empty_1d,
    B.copy(),
    w.copy(),
    buffer,
    k, lam,
    nonneg=True
).copy()
res_closedform_sparse_nn = test_math.py_factors_closed_form(
    catcher,
    empty_1d,
    x_sp_ix,
    x_sp_val.copy(),
    B.copy(),
    w_sp.copy(),
    buffer,
    k, lam,
    nonneg=True
).copy()
res_lbfgs_dense = test_math.py_factors_lbfgs(
    catcher,
    x.copy(),
    empty_int,
    empty_1d,
    B.copy(),
    w.copy(),
    buffer,
    k, lam
).copy()
res_lbfgs_sparse = test_math.py_factors_lbfgs(
    catcher,
    empty_1d,
    x_sp_ix,
    x_sp_val.copy(),
    B.copy(),
    w_sp.copy(),
    buffer,
    k, lam
).copy()
res_closedform_dense_pc = test_math.py_factors_closed_form(
    catcher,
    x.copy(),
    empty_int,
    empty_1d,
    B.copy(),
    w.copy(),
    buffer,
    k, lam, precompute=True
).copy()
res_closedform_sparse_pc = test_math.py_factors_closed_form(
    catcher,
    empty_1d,
    x_sp_ix,
    x_sp_val.copy(),
    B.copy(),
    w_sp.copy(),
    buffer,
    k, lam, precompute=True
).copy()
res_closedform_dense_nn_pc = test_math.py_factors_closed_form(
    catcher,
    x.copy(),
    empty_int,
    empty_1d,
    B.copy(),
    w.copy(),
    buffer,
    k, lam, precompute=True, nonneg=True
).copy()
res_closedform_sparse_nn_pc = test_math.py_factors_closed_form(
    catcher,
    empty_1d,
    x_sp_ix,
    x_sp_val.copy(),
    B.copy(),
    w_sp.copy(),
    buffer,
    k, lam, precompute=True, nonneg=True
).copy()
print("Single-row prediction with weights")
with np.printoptions(precision=4, suppress=True):
    print("scipy           :", res_scipy)
    print("numpy           :", res_py)
    print("formula (d)     :", res_closedform_dense)
    print("formula (s)     :", res_closedform_sparse)
    print("formula[pc] (d) :", res_closedform_dense_pc)
    print("formula[pc] (s) :", res_closedform_sparse_pc)
    print("lbfgs (d)       :", res_lbfgs_dense)
    print("lbfgs (s)       :", res_lbfgs_sparse)
    print("----")
    print("scipy  (nn)       :", res_scipy_nn)
    print("formula (d)(nn)   :", res_closedform_dense_nn)
    print("formula (s)(nn)   :", res_closedform_sparse_nn)
    print("formula[pc](d)(nn):", res_closedform_dense_nn_pc)
    print("formula[pc](s)(nn):", res_closedform_sparse_nn_pc)


### Now without weights
w = np.ones(n).astype(ctypes.c_double)
W = np.diag(w)
res_scipy = minimize(feval, a0)["x"]
res_scipy = minimize(feval, a0, bounds=[(0.,None) for i in range(k)], method="TNC")["x"]
res_py = py_closed_form()
res_closedform_dense = test_math.py_factors_closed_form(
    catcher,
    x.copy(),
    empty_int,
    empty_1d,
    B.copy(),
    empty_1d,
    buffer,
    k, lam
).copy()
res_closedform_sparse = test_math.py_factors_closed_form(
    catcher,
    empty_1d,
    x_sp_ix,
    x_sp_val.copy(),
    B.copy(),
    empty_1d,
    buffer,
    k, lam
).copy()
res_closedform_dense_nn = test_math.py_factors_closed_form(
    catcher,
    x.copy(),
    empty_int,
    empty_1d,
    B.copy(),
    empty_1d,
    buffer,
    k, lam, nonneg=True
).copy()
res_closedform_sparse_nn = test_math.py_factors_closed_form(
    catcher,
    empty_1d,
    x_sp_ix,
    x_sp_val.copy(),
    B.copy(),
    empty_1d,
    buffer,
    k, lam, nonneg=True
).copy()
res_lbfgs_dense = test_math.py_factors_lbfgs(
    catcher,
    x.copy(),
    empty_int,
    empty_1d,
    B.copy(),
    empty_1d,
    buffer,
    k, lam
).copy()
res_lbfgs_sparse = test_math.py_factors_lbfgs(
    catcher,
    empty_1d,
    x_sp_ix,
    x_sp_val.copy(),
    B.copy(),
    empty_1d,
    buffer,
    k, lam
).copy()

### Now using pre-computed matrix
res_closedform_dense_pc = test_math.py_factors_closed_form(
    catcher,
    x.copy(),
    empty_int,
    empty_1d,
    B.copy(),
    empty_1d,
    buffer,
    k, lam, precompute=True
).copy()
res_closedform_sparse_pc = test_math.py_factors_closed_form(
    catcher,
    empty_1d,
    x_sp_ix,
    x_sp_val.copy(),
    B.copy(),
    empty_1d,
    buffer,
    k, lam, precompute=True
).copy()
res_closedform_dense_nn_pc = test_math.py_factors_closed_form(
    catcher,
    x.copy(),
    empty_int,
    empty_1d,
    B.copy(),
    empty_1d,
    buffer,
    k, lam, precompute=True, nonneg=True
).copy()
res_closedform_sparse_nn_pc = test_math.py_factors_closed_form(
    catcher,
    empty_1d,
    x_sp_ix,
    x_sp_val.copy(),
    B.copy(),
    empty_1d,
    buffer,
    k, lam, precompute=True, nonneg=True
).copy()

print("\nSingle-row prediction without weights")
with np.printoptions(precision=4, suppress=True):
    print("scipy           :", res_scipy)
    print("numpy           :", res_py)
    print("formula (d)     :", res_closedform_dense)
    print("formula (s)     :", res_closedform_sparse)
    print("formula[pc] (d) :", res_closedform_dense_pc)
    print("formula[pc] (s) :", res_closedform_sparse_pc)
    print("lbfgs (d)       :", res_lbfgs_dense)
    print("lbfgs (s)       :", res_lbfgs_sparse)
    print("-------")
    print("scipy      (nn)     :", res_scipy_nn)
    print("formula (d)(nn)     :", res_closedform_dense_nn)
    print("formula (s)(nn)     :", res_closedform_sparse_nn)
    print("formula[pc] (d)(nn) :", res_closedform_dense_nn_pc)
    print("formula[pc] (s)(nn) :", res_closedform_sparse_nn_pc)
