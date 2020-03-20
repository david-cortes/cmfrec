import numpy as np
import ctypes
import test_math

n = int(5e2)
k = int(5e0)

lam = 2.345
glob_mean = -4.567
np.random.seed(456)
biasA = np.random.normal()
biasB = np.random.normal(size = n)
rng = np.arange(n).astype(ctypes.c_int)

empty_1d = np.empty(0, dtype=ctypes.c_double)
empty_int = np.empty(0, dtype=ctypes.c_int)
def get_topN():
    return test_math.py_topN(
        a_vec,
        B,
        biasB if item_bias else empty_1d,
        glob_mean, biasA,
        incl_vec,
        excl_vec,
        n_top,
        k, k_user = k_user, k_item = k_item, k_main = k_main,
        output_score = True
    )
    
def py_topN():
    scores = B[:,k_item:].dot(a_vec[k_user:]) + glob_mean + biasA
    if item_bias:
        scores += biasB
    if incl>0:
        scores = scores[incl_vec]
        take = np.argsort(-scores)[:n_top]
        return incl_vec[take], scores[take]
    elif excl>0:
        ix = np.setdiff1d(rng, excl_vec)
        scores = scores[ix]
        take = np.argsort(-scores)[:n_top]
        return ix[take], scores[take]
    else:
        take = np.argsort(-scores)[:n_top]
        return rng[take], scores[take]

ktry = [0,2]
ttry = [1, 5, 60, int(n/2), int(n-1), int(n)]
btry = [True,False]
etry = [0, 1, 2, 100, int(n-2)]
itry = [0, 1, 100, int(n-2)]



for k_user in ktry:
    for k_item in ktry:
        for k_main in ktry:
            for item_bias in btry:
                for n_top in ttry:
                    for excl in etry:
                        for incl in itry:

                            if excl and incl:
                                continue
                            if (incl>0) and (n_top>incl):
                                continue
                            if (n-excl)<n_top:
                                continue

                            np.random.seed(123)
                            B = np.random.normal(size = (n, k_item+k+k_main))
                            a_vec = np.random.normal(size = k_user+k+k_main)
                            incl_vec = np.random.choice(rng, size=incl, replace=False)
                            excl_vec = np.random.choice(rng, size=excl, replace=False)

                            res_py_ix, res_py_s = py_topN()
                            res_module_ix, res_module_s = get_topN()

                            res1 = np.mean(res_py_ix == res_module_ix)
                            res2 = np.linalg.norm(res_py_s - res_module_s)
                            is_wrong = res1 < 1. or res2 > 1e-5
                            if is_wrong:
                                print("\n\n\n****ERROR BELOW****")

                            print("[n:%d] [i:%d] [e:%d] [k:%d,%d,%d] [b:%d] -> agr:%.2f, diff:%.2f"
                                  % (n_top, incl, excl, k_user,k_item,k_main, item_bias, res1, res2),
                                  flush = True)
                            if is_wrong:
                                print("****ERROR ABOVE****\n\n\n")
