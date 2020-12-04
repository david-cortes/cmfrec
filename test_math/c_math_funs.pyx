#cython: language_level=3
import numpy as np
cimport numpy as np
from scipy.linalg.cython_lapack cimport dpotrf
from libcpp cimport bool as c_bool
from libc.stdio cimport printf

## TODO: test also the non-negative version

cdef extern from "cmfrec.h":
    void factors_closed_form(
        double *a_vec, int k,
        double *B, int n, int ldb,
        double *Xa_dense, bint full_dense,
        double *Xa, int ixB[], size_t nnz,
        double *weight,
        double *buffer_double,
        double lam, double lam_last,
        double *precomputedTransBtBinvBt,
        double *precomputedBtBw, int cnt_NA, int ld_BtB,
        bint BtB_has_diag, bint BtB_is_scaled, double scale_BtB, int n_BtB,
        double *precomputedBtBchol, bint NA_as_zero,
        bint use_cg, int max_cg_steps,
        bint nonneg, int max_cd_steps, double *a_prev,
        double *bias_BtX, double *bias_X,
        bint force_add_diag
    )

    int collective_factors_lbfgs(
        double *a_vec,
        int k, int k_user, int k_item, int k_main,
        double *u_vec, int p,
        int u_vec_ixB[], double *u_vec_sp, size_t nnz_u_vec,
        double *u_bin_vec, int pbin,
        bint u_vec_has_NA, bint u_bin_vec_has_NA,
        double *B, int n,
        double *C, double *Cb,
        double *Xa, int ixB[], double *weight, size_t nnz,
        double *Xa_dense,
        double *buffer_double,
        double lam, double w_main, double w_user, double lam_last
    )

    double collective_fun_grad(
        double *values, double *grad,
        int m, int n, int k,
        int ixA[], int ixB[], double *X, size_t nnz,
        double *Xfull,
        size_t Xcsr_p[], int Xcsr_i[], double *Xcsr,
        size_t Xcsc_p[], int Xcsc_i[], double *Xcsc,
        double *weight, double *weightR, double *weightC,
        bint user_bias, bint item_bias,
        double lam, double *lam_unique,
        double *U, int m_u, int p, bint U_has_NA,
        double *I, int n_i, int q, bint I_has_NA,
        double *Ub, int m_ubin, int pbin, bint Ub_has_NA,
        double *Ib, int n_ibin, int qbin, bint Ib_has_NA,
        int U_row[], int U_col[], double *U_sp, size_t nnz_U,
        int I_row[], int I_col[], double *I_sp, size_t nnz_I,
        size_t U_csr_p[], int U_csr_i[], double *U_csr,
        size_t U_csc_p[], int U_csc_i[], double *U_csc,
        size_t I_csr_p[], int I_csr_i[], double *I_csr,
        size_t I_csc_p[], int I_csc_i[], double *I_csc,
        double *buffer_double, double *buffer_mt,
        int k_main, int k_user, int k_item,
        double w_main, double w_user, double w_item,
        int nthreads
    )

    double collective_fun_grad_single(
        double *a_vec, double *g_A,
        int k, int k_user, int k_item, int k_main,
        double *u_vec, int p,
        int u_vec_ixB[], double *u_vec_sp, size_t nnz_u_vec,
        double *u_bin_vec, int pbin,
        bint u_vec_has_NA, bint u_bin_vec_has_NA,
        double *B, int n,
        double *C, double *Cb,
        double *Xa, int ixB[], size_t nnz,
        double *Xa_dense,
        double *weight,
        double *buffer_double,
        double lam, double w_main, double w_user, double lam_last
    )

    int collective_factors_cold(
        double *a_vec,
        double *u_vec, int p,
        double *u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
        double *u_bin_vec, int pbin,
        double *C, double *Cb,
        double *TransCtCinvCt,
        double *CtCw,
        double *col_means,
        int k, int k_user, int k_main,
        double lam, double l1_lam, double w_main, double w_user,
        bint scale_lam,
        bint NA_as_zero_U,
        bint nonneg
    )

    int collective_factors_warm(
        double *a_vec, double *a_bias,
        double *u_vec, int p,
        double *u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
        double *u_bin_vec, int pbin,
        double *C, double *Cb,
        double glob_mean, double *biasB,
        double *col_means,
        double *Xa, int ixB[], size_t nnz,
        double *Xa_dense, int n,
        double *weight,
        double *B,
        double *Bi, bint add_implicit_features,
        int k, int k_user, int k_item, int k_main,
        double lam, double w_main, double w_user, double w_implicit, double lam_bias,
        double l1_lam, double l1_lam_bias,
        bint scale_lam, bint scale_lam_sideinfo,
        int n_max, bint include_all_X,
        double *TransBtBinvBt,
        double *BtXbias,
        double *BtB,
        double *BeTBeChol,
        double *BiTBi,
        double *CtCw,
        bint NA_as_zero_U, bint NA_as_zero_X,
        bint nonneg,
        double *B_plus_bias
    )

    double fun_grad_cannonical_form(
        double *A, int lda, double *B, int ldb,
        double *g_A, double *g_B,
        int m, int n, int k,
        int ixA[], int ixB[], double *X, size_t nnz,
        double *Xfull, bint full_dense,
        size_t Xcsr_p[], int Xcsr_i[], double *Xcsr,
        size_t Xcsc_p[], int Xcsc_i[], double *Xcsc,
        bint user_bias, bint item_bias,
        double *biasA, double *biasB,
        double *g_biasA, double *g_biasB,
        double *weight, double *weightR, double *weightC,
        double scaling,
        double *buffer_double,
        double *buffer_mt,
        int nthreads
    )

    void optimizeA(
        double *A, int lda,
        double *B, int ldb,
        int m, int n, int k,
        size_t Xcsr_p[], int Xcsr_i[], double *Xcsr,
        double *Xfull, int ldX, bint full_dense, bint near_dense,
        int cnt_NA[], double *weight, bint NA_as_zero,
        double lam, double lam_last,
        bint do_B, bint is_first_iter,
        int nthreads,
        bint use_cg, int max_cg_steps,
        bint nonneg, int max_cd_steps, double *A_prev,
        double *bias_BtX, double *bias_X,
        bint keep_precomputedBtB,
        double *precomputedBtB, c_bool *filled_BtB,
        double *buffer_double
    )

    void optimizeA_implicit(
        double *A, size_t lda,
        double *B, size_t ldb,
        int m, int n, int k,
        size_t Xcsr_p[], int Xcsr_i[], double *Xcsr,
        double lam,
        int nthreads,
        bint use_cg, int max_cg_steps, bint force_set_to_zero,
        bint nonneg, int max_cd_steps, double *A_prev,
        double *precomputedBtB,
        double *buffer_double
    )

    void optimizeA_collective(
        double* A, int lda, double* B, int ldb, double* C, double *Bi,
        int m, int m_u, int n, int p,
        int k, int k_main, int k_user, int k_item,
        size_t Xcsr_p[], int Xcsr_i[], double* Xcsr,
        double* Xfull, bint full_dense, bint near_dense, int ldX,
        int cnt_NA_x[], double* weight, bint NA_as_zero_X,
        double *Xones, int k_main_i, int ldXones,
        bint add_implicit_features,
        size_t U_csr_p[], int U_csr_i[], double* U_csr,
        double* U, int cnt_NA_u[],
        bint full_dense_u, bint near_dense_u, bint NA_as_zero_U,
        double lam, double w_user, double w_implicit, double lam_last,
        bint do_B,
        int nthreads,
        bint use_cg, int max_cg_steps, bint is_first_iter,
        bint nonneg, int max_cd_steps, double *A_prev,
        double *bias_BtX, double *bias_X,
        bint keep_precomputed,
        double *precomputedBtB,
        double *precomputedCtCw,
        double *precomputedBeTBeChol,
        double *precomputedBiTBi,
        c_bool *filled_BtB, c_bool *filled_CtCw, c_bool *filled_BeTBeChol,
        c_bool *CtC_is_scaled,
        double* buffer_double
    )

    void optimizeA_collective_implicit(
        double *A, double *B, double *C,
        int m, int m_u, int n, int p,
        int k, int k_main, int k_user, int k_item,
        size_t Xcsr_p[], int Xcsr_i[], double *Xcsr,
        size_t U_csr_p[], int U_csr_i[], double *U_csr,
        double *U, int cnt_NA_u[],
        bint full_dense_u, bint near_dense_u, bint NA_as_zero_U,
        double lam, double w_user,
        int nthreads,
        bint use_cg, int max_cg_steps, bint is_first_iter,
        bint nonneg, int max_cd_steps, double *A_prev,
        double *precomputedBtB,
        double *precomputedBeTBe,
        double *precomputedBeTBeChol,
        double *precomputedCtC,
        bint *filled_BeTBe,
        bint *filled_BeTBeChol,
        bint *filled_CtC,
        double *buffer_double
    )

    double offsets_fun_grad(
        double *values, double *grad,
        int ixA[], int ixB[], double *X,
        size_t nnz, int m, int n, int k,
        double *Xfull, bint full_dense,
        size_t Xcsr_p[], int Xcsr_i[], double *Xcsr,
        size_t Xcsc_p[], int Xcsc_i[], double *Xcsc,
        double *weight, double *weightR, double *weightC,
        bint user_bias, bint item_bias,
        bint add_intercepts,
        double lam, double *lam_unique,
        double *U, int p,
        double *II, int q,
        size_t U_csr_p[], int U_csr_i[], double *U_csr,
        size_t U_csc_p[], int U_csc_i[], double *U_csc,
        size_t I_csr_p[], int I_csr_i[], double *I_csr,
        size_t I_csc_p[], int I_csc_i[], double *I_csc,
        int k_main, int k_sec,
        double w_user, double w_item,
        int nthreads,
        double *buffer_double,
        double *buffer_mt
    )

    int offsets_factors_warm(
        double *a_vec, double *a_bias,
        double *u_vec,
        int u_vec_ixB[], double *u_vec_sp, size_t nnz_u_vec,
        int ixB[], double *Xa, size_t nnz,
        double *Xa_dense, int n,
        double *weight,
        double *Bm, double *C,
        double *C_bias,
        double glob_mean, double *biasB,
        int k, int k_sec, int k_main,
        int p, double w_user,
        double lam, bint exact, double lam_bias,
        bint implicit, double alpha,
        double *precomputedTransBtBinvBt,
        double *precomputedBtBw,
        double *output_a,
        double *Bm_plus_bias
    )

    int precompute_collective_explicit(
        double *B, int n, int n_max, bint include_all_X,
        double *C, int p,
        double *Bi, bint add_implicit_features,
        double *biasB, double glob_mean, bint NA_as_zero_X,
        int k, int k_user, int k_item, int k_main,
        bint user_bias,
        bint nonneg,
        double lam, double *lam_unique,
        bint scale_lam, bint scale_lam_sideinfo,
        double w_main, double w_user, double w_implicit,
        double *B_plus_bias,
        double *BtB,
        double *TransBtBinvBt,
        double *BtXbias,
        double *BeTBeChol,
        double *BiTBi,
        double *TransCtCinvCt,
        double *CtCw
    )

    int initialize_biases(
        double *glob_mean, double *biasA, double *biasB,
        bint user_bias, bint item_bias,
        double lam_user, double lam_item,
        bint scale_lam,
        int m, int n,
        int m_bias, int n_bias,
        int ixA[], int ixB[], double *X, size_t nnz,
        double *Xfull, double *Xtrans,
        size_t Xcsr_p[], int Xcsr_i[], double *Xcsr,
        size_t Xcsc_p[], int Xcsc_i[], double *Xcsc,
        bint nonneg,
        int nthreads
    )

    int center_by_cols(
        double *col_means,
        double *Xfull, int m, int n,
        int ixA[], int ixB[], double *X, size_t nnz,
        size_t Xcsr_p[], int Xcsr_i[], double *Xcsr,
        size_t Xcsc_p[], int Xcsc_i[], double *Xcsc,
        int nthreads
    )

    int topN(
        double *a_vec, int k_user,
        double *B, int k_item,
        double *biasB,
        double glob_mean, double biasA,
        int k, int k_main,
        int *include_ix, int n_include,
        int *exclude_ix, int n_exclude,
        int *outp_ix, double *outp_score,
        int n_top, int n, int nthreads
    )

    void fill_lower_triangle(double A[], size_t n, size_t lda)

### TODO: add tests for the precomputed matrices too
### TODO: add tests for mean centering with NA_as_zero_X
import ctypes

def py_factors_closed_form(
    np.ndarray[double, ndim=1] outp,
    np.ndarray[double, ndim=1] Xa_dense,
    np.ndarray[int, ndim=1] ixB,
    np.ndarray[double, ndim=1] Xa,
    np.ndarray[double, ndim=2] B,
    np.ndarray[double, ndim=1] weight,
    np.ndarray[double, ndim=1] buffer_double,
    int k, double lam,
    bint nonneg=0,
    bint precompute=0
    ):
    
    cdef double *ptr_Xa_dense = NULL
    cdef double *ptr_Xa = NULL
    cdef double *ptr_weight = NULL
    cdef int   *ptr_ixB = NULL
    cdef size_t nnz = 0

    if Xa_dense.shape[0]:
        ptr_Xa_dense = &Xa_dense[0]
    else:
        ptr_Xa = &Xa[0]
        ptr_ixB = &ixB[0]
        nnz = Xa.shape[0]
    if weight.shape[0]:
        ptr_weight = &weight[0]

    cdef double *ptr_TransBtBinvBt = NULL
    cdef double *ptr_BtBw = NULL
    cdef double *ptr_BtBchol = NULL
    cdef np.ndarray[double, ndim=2] TransBtBinvBt
    cdef np.ndarray[double, ndim=2] BtBw
    cdef np.ndarray[double, ndim=2] BtBchol
    cdef char lo = 76#'L'
    cdef int ignore = 0
    cdef int n = B.shape[1]
    if precompute:
        BtBw = np.empty((k,k))
        TransBtBinvBt = np.empty((n, k))
        BtBchol = np.empty((k,k))
        precompute_collective_explicit(
            &B[0,0], n, n, 0,
            <double*>NULL, 0,
            <double*>NULL, 0,
            <double*>NULL, 0., 0,
            k, 0, 0, 0,
            0, nonneg,
            lam, <double*>NULL,
            0, 0,
            1., 1., 1.,
            <double*>NULL,
            &BtBw[0,0],
            &TransBtBinvBt[0,0],
            <double*>NULL,
            <double*>NULL,
            <double*>NULL,
            <double*>NULL,
            <double*>NULL
        )
        dpotrf(&lo, &k, &BtBchol[0,0], &k, &ignore)

    cdef np.ndarray[double, ndim=1] a_prev = np.zeros(k, dtype=np.float64)

    factors_closed_form(
        &outp[0], k,
        &B[0,0], B.shape[0], B.shape[1],
        ptr_Xa_dense, np.sum(np.isnan(Xa_dense))==0,
        ptr_Xa, ptr_ixB, nnz,
        ptr_weight,
        &buffer_double[0],
        lam, lam,
        ptr_TransBtBinvBt,
        ptr_BtBw, np.sum(np.isnan(Xa_dense)), k,
        1, 0, 1., 0,
        ptr_BtBchol, 0,
        0, 0,
        nonneg, 10000, &a_prev[0],
        <double*>NULL, <double*>NULL,
        0
    )
    return outp

def py_factors_lbfgs(
    np.ndarray[double, ndim=1] outp,
    np.ndarray[double, ndim=1] Xa_dense,
    np.ndarray[int, ndim=1] ixB,
    np.ndarray[double, ndim=1] Xa,
    np.ndarray[double, ndim=2] B,
    np.ndarray[double, ndim=1] weight,
    np.ndarray[double, ndim=1] buffer_double,
    int k, double lam
    ):

    cdef double *ptr_Xa_dense = NULL
    cdef double *ptr_Xa = NULL
    cdef double *ptr_weight = NULL
    cdef int   *ptr_ixB = NULL
    cdef size_t nnz = 0

    if Xa_dense.shape[0]:
        ptr_Xa_dense = &Xa_dense[0]
    else:
        ptr_Xa = &Xa[0]
        ptr_ixB = &ixB[0]
        nnz = Xa.shape[0]
    if weight.shape[0]:
        ptr_weight = &weight[0]

    collective_factors_lbfgs(
        &outp[0],
        k, 0, 0, 0,
        NULL, 0,
        NULL, NULL, 0,
        NULL, 0,
        0, 0,
        &B[0,0], B.shape[0],
        NULL, NULL,
        ptr_Xa, ptr_ixB, ptr_weight, nnz,
        ptr_Xa_dense,
        &buffer_double[k],
        lam, 1, 1, lam
    )
    return outp

def py_collective_factors(
        np.ndarray[double, ndim=1] outp,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[double, ndim=1] Xa,
        np.ndarray[double, ndim=1] Xfull,
        np.ndarray[int, ndim=1] ixB_U,
        np.ndarray[double, ndim=1] U_sp,
        np.ndarray[double, ndim=1] U,
        np.ndarray[double, ndim=1] Ubin,
        np.ndarray[double, ndim=2] B,
        np.ndarray[double, ndim=2] C,
        np.ndarray[double, ndim=2] Cb,
        np.ndarray[double, ndim=2] Bi,
        np.ndarray[double, ndim=1] weight,
        int k, int k_main, int k_user, int k_item,
        double lam, double w_user, double w_main, double w_implicit,
        bint NA_as_zero_X,
        bint NA_as_zero_U,
        bint precompute
    ):

    cdef int *ptr_ixB = NULL
    cdef double *ptr_Xa = NULL
    cdef size_t nnz = 0
    cdef double *ptr_Xfull = NULL
    if Xfull.shape[0]:
        ptr_Xfull = &Xfull[0]
    elif Xa.shape[0]:
        ptr_ixB = &ixB[0]
        ptr_Xa = &Xa[0]
        nnz = Xa.shape[0]

    cdef int *ptr_ixB_U = NULL
    cdef double *ptr_U_sp = NULL
    cdef size_t nnz_u_vec = 0
    cdef double *ptr_U = NULL
    if U.shape[0]:
        ptr_U = &U[0]
    elif U_sp.shape[0]:
        ptr_ixB_U = &ixB_U[0]
        ptr_U_sp = &U_sp[0]
        nnz_u_vec = U_sp.shape[0]

    cdef double *ptr_Ubin = NULL
    if Ubin.shape[0]:
        ptr_Ubin = &Ubin[0]

    cdef double *ptr_C = NULL
    cdef double *ptr_Cb = NULL
    cdef int p = 0
    cdef int pbin = 0
    if C.shape[0]:
        ptr_C = &C[0,0]
        p = C.shape[0]
    if Cb.shape[0]:
        ptr_Cb = &Cb[0,0]
        pbin = Cb.shape[0]

    cdef c_bool add_implicit_features = 0
    cdef double *ptr_Bi = NULL
    if Bi.shape[0]:
        ptr_Bi = &Bi[0,0]
        add_implicit_features = 1

    cdef double *ptr_weight = NULL
    if weight.shape[0]:
        ptr_weight = &weight[0]

    cdef double *ptr_TransBtBinvBt = NULL
    cdef double *ptr_BtB = NULL
    cdef double *ptr_BeTBeChol = NULL
    cdef double *ptr_BiTBi = NULL
    cdef double *ptr_CtCw = NULL
    cdef np.ndarray[double, ndim=2] TransBtBinvBt
    cdef np.ndarray[double, ndim=2] BtB
    cdef np.ndarray[double, ndim=2] BeTBeChol
    cdef np.ndarray[double, ndim=2] BiTBi
    cdef np.ndarray[double, ndim=2] CtCw
    lam /= w_main
    w_user /= w_main
    w_implicit /= w_main
    w_main = 1.

    cdef int one = 1
    cdef char lo = 76#'L'
    cdef int ignore = 0
    cdef int k_totA = k_user + k + k_main
    if precompute:
        TransBtBinvBt = np.empty((B.shape[0], k+k_main))
        BtB = np.empty((k+k_main, k+k_main))
        ptr_TransBtBinvBt = &TransBtBinvBt[0,0]
        ptr_BtB = &BtB[0,0]
        if C.shape[0]:
            CtCw = np.empty((k_user+k,k_user+k))
            ptr_CtCw = &CtCw[0,0]
        if add_implicit_features:
            BiTBi = np.empty((k+k_main, k+k_main))
            ptr_BiTBi = &BiTBi[0,0]
        if p or add_implicit_features:
            BeTBeChol = np.empty((k_user+k+k_main,k_user+k+k_main))
            ptr_BeTBeChol = &BeTBeChol[0,0]
        precompute_collective_explicit(
            &B[0,0], B.shape[0], B.shape[0], 0,
            ptr_C, p,
            ptr_Bi, add_implicit_features,
            <double*>NULL, 0., NA_as_zero_X,
            k, k_user, k_item, k_main,
            0, 0,
            lam, <double*>NULL,
            0, 0,
            1., w_user, w_implicit,
            <double*>NULL,
            ptr_BtB,
            ptr_TransBtBinvBt,
            <double*>NULL,
            ptr_BeTBeChol,
            ptr_BiTBi,
            <double*>NULL,
            ptr_CtCw
        )

        # if U.shape[0] or U_sp.shape[0]:
        #     CtCw = w_user * C.T.dot(C)
        #     ptr_CtCw = &CtCw[0,0]

        # BtB = B[:,k_item:].T.dot(B[:,k_item:])
        # TransBtBinvBt = np.linalg.inv(BtB + lam*np.eye(BtB.shape[0])).dot(B[:,k_item:].T)
        # TransBtBinvBt = np.ascontiguousarray(TransBtBinvBt.T)
        
        # if C.shape[0]:
        #     BeTBeChol = np.zeros((k_user+k+k_main, k_user+k+k_main))
        #     BeTBeChol[k_user:,k_user:] = BtB
        #     if C.shape[0]:
        #         BeTBeChol[:k_user+k, :k_user+k] += w_user * (C.T.dot(C))
        #     BeTBeChol[:,:] += lam * np.eye(BeTBeChol.shape[0])
        #     dpotrf(&lo, &k_totA, &BeTBeChol[0,0], &k_totA, &ignore)
        #     ptr_BeTBeChol = &BeTBeChol[0,0]


    cdef np.ndarray[double, ndim=1] col_means = np.zeros(p, dtype=ctypes.c_double)
    cdef double *ptr_col_means = NULL
    if col_means.shape[0]:
        ptr_col_means = &col_means[0]

    collective_factors_warm(
        &outp[0], <double*>NULL,
        ptr_U, p,
        ptr_U_sp, ptr_ixB_U, nnz_u_vec,
        ptr_Ubin, pbin,
        ptr_C, ptr_Cb,
        0., <double*>NULL, ptr_col_means,
        ptr_Xa, ptr_ixB, nnz,
        ptr_Xfull, B.shape[0],
        ptr_weight,
        &B[0,0],
        ptr_Bi, add_implicit_features,
        k, k_user, k_item, k_main,
        lam, w_main, w_user, w_implicit, lam,
        0., 0.,
        0, 0,
        B.shape[0], 0,
        ptr_TransBtBinvBt, <double*>NULL, ptr_BtB, ptr_BeTBeChol,
        ptr_BiTBi, ptr_CtCw,
        NA_as_zero_U, NA_as_zero_X,
        0,
        <double*>NULL
    )
    return outp

def py_fun_grad_classic(
        np.ndarray[double, ndim=1] values,
        np.ndarray[double, ndim=1] grad,
        np.ndarray[int, ndim=1] ixA,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[double, ndim=1] X,
        np.ndarray[double, ndim=2] Xfull,
        np.ndarray[size_t, ndim=1] Xcsr_p,
        np.ndarray[int, ndim=1] Xcsr_i,
        np.ndarray[double, ndim=1] Xcsr,
        np.ndarray[size_t, ndim=1] Xcsc_p,
        np.ndarray[int, ndim=1] Xcsc_i,
        np.ndarray[double, ndim=1] Xcsc,
        np.ndarray[double, ndim=1] weight,
        np.ndarray[double, ndim=1] weightR,
        np.ndarray[double, ndim=1] weightC,
        int m, int n, int k,
        bint bias_u, bint bias_i,
        int nthreads,
        np.ndarray[double, ndim=1] buffer_double,
        np.ndarray[double, ndim=1] buffer_mt
    ):

    cdef double scaling = 1
    cdef bint full_dense = 0

    cdef double* biasA = &values[0]
    cdef double* biasB = biasA + (m if bias_u else 0)
    cdef double* A = biasB + (n if bias_i else 0)
    cdef double* B = A + m*k

    cdef double* g_biasA = &grad[0]
    cdef double* g_biasB = g_biasA + (m if bias_u else 0)
    cdef double* g_A = g_biasB + (n if bias_i else 0)
    cdef double* g_B = g_A + m*k

    cdef int *ptr_ixA = NULL
    cdef int *ptr_ixB = NULL
    cdef double *ptr_X = NULL
    cdef size_t nnz = 0
    cdef double *ptr_Xfull = NULL
    if Xfull.shape[0]:
        ptr_Xfull = &Xfull[0,0]
    else:
        ptr_ixA = &ixA[0]
        ptr_ixB = &ixB[0]
        ptr_X = &X[0]
        nnz = X.shape[0]

    cdef size_t *ptr_Xcsr_p = NULL
    cdef int *ptr_Xcsr_i = NULL
    cdef double *ptr_Xcsr = NULL
    cdef size_t *ptr_Xcsc_p = NULL
    cdef int *ptr_Xcsc_i = NULL
    cdef double *ptr_Xcsc = NULL
    if Xcsr.shape[0]:
        ptr_Xcsr_p = &Xcsr_p[0]
        ptr_Xcsr_i = &Xcsr_i[0]
        ptr_Xcsr = &Xcsr[0]
        ptr_Xcsc_p = &Xcsc_p[0]
        ptr_Xcsc_i = &Xcsc_i[0]
        ptr_Xcsc = &Xcsc[0]

    cdef double *ptr_weight = NULL
    cdef double *ptr_weightR = NULL
    cdef double *ptr_weightC = NULL
    if weight.shape[0]:
        ptr_weight = &weight[0]
    if weightR.shape[0]:
        ptr_weightR = &weightR[0]
    if weightC.shape[0]:
        ptr_weightC = &weightC[0]

    cdef double *ptr_buffer_mt = NULL
    if buffer_mt.shape[0]:
        ptr_buffer_mt = &buffer_mt[0]

    cdef double f = fun_grad_cannonical_form(
        A, k, B, k,
        g_A, g_B,
        m, n, k,
        ptr_ixA, ptr_ixB, ptr_X, nnz,
        ptr_Xfull, full_dense,
        ptr_Xcsr_p, ptr_Xcsr_i, ptr_Xcsr,
        ptr_Xcsc_p, ptr_Xcsc_i, ptr_Xcsc,
        bias_u, bias_i,
        biasA, biasB,
        g_biasA, g_biasB,
        ptr_weight, ptr_weightR, ptr_weightC,
        scaling,
        &buffer_double[0],
        ptr_buffer_mt,
        nthreads
    )
    return f, grad

def py_fun_grad_collective(
    np.ndarray[double, ndim=1] values,
    np.ndarray[double, ndim=1] grad,
    np.ndarray[double, ndim=2] Xfull,
    np.ndarray[int, ndim=1] ixA,
    np.ndarray[int, ndim=1] ixB,
    np.ndarray[double, ndim=1] X,
    np.ndarray[double, ndim=2] U,
    np.ndarray[int, ndim=1] Urow,
    np.ndarray[int, ndim=1] Ucol,
    np.ndarray[double, ndim=1] Usp,
    np.ndarray[double, ndim=2] II,
    np.ndarray[int, ndim=1] Irow,
    np.ndarray[int, ndim=1] Icol,
    np.ndarray[double, ndim=1] Isp,
    np.ndarray[double, ndim=2] Ub,
    np.ndarray[double, ndim=2] Ib,
    np.ndarray[double, ndim=1] weight,
    int m, int n, int p, int q, int pbin, int qbin,
    int k, int k_main, int k_user, int k_item,
    double w_main, double w_user, double w_item,
    double lam, np.ndarray[double, ndim=1] lam_unique,
    bint user_bias, bint item_bias,
    int nthreads,
    np.ndarray[double, ndim=1] buffer_double
    ):

    cdef double *ptr_lam_unique = NULL
    if lam_unique.shape[0]:
        ptr_lam_unique = &lam_unique[0]

    cdef double *ptr_weight = NULL
    if weight.shape[0]:
        ptr_weight = &weight[0]

    cdef double *ptr_X = NULL
    cdef int *ptr_ixA = NULL
    cdef int *ptr_ixB = NULL
    cdef size_t nnz = 0
    cdef double *ptr_Xfull = NULL
    if Xfull.shape[0]:
        ptr_Xfull = &Xfull[0,0]
    elif X.shape[0]:
        ptr_X = &X[0]
        ptr_ixA = &ixA[0]
        ptr_ixB = &ixB[0]
        nnz = X.shape[0]

    cdef double *ptr_U = NULL
    cdef double *ptr_Usp = NULL
    cdef int *ptr_Urow = NULL
    cdef int *ptr_Ucol = NULL
    cdef size_t nnz_U = 0
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif Usp.shape[0]:
        ptr_Usp = &Usp[0]
        ptr_Urow = &Urow[0]
        ptr_Ucol = &Ucol[0]
        nnz_U = Usp.shape[0]

    cdef double *ptr_I = NULL
    cdef double *ptr_Isp = NULL
    cdef int *ptr_Irow = NULL
    cdef int *ptr_Icol = NULL
    cdef size_t nnz_I = 0
    if II.shape[0]:
        ptr_I = &II[0,0]
    elif Isp.shape[0]:
        ptr_Isp = &Isp[0]
        ptr_Irow = &Irow[0]
        ptr_Icol = &Icol[0]
        nnz_I = Isp.shape[0]

    cdef double *ptr_Ub = NULL
    if Ub.shape[0]:
        ptr_Ub = &Ub[0,0]
    cdef double *ptr_Ib = NULL
    if Ib.shape[0]:
        ptr_Ib = &Ib[0,0]

    cdef int m_u = m
    cdef int m_ubin = m
    cdef int n_i = n
    cdef int n_ibin = n


    cdef double f = collective_fun_grad(
        &values[0], &grad[0],
        m, n, k,
        ptr_ixA, ptr_ixB, ptr_X, nnz,
        ptr_Xfull,
        NULL, NULL, NULL,
        NULL, NULL, NULL,
        ptr_weight, NULL, NULL,
        user_bias, item_bias,
        lam, ptr_lam_unique,
        ptr_U, m_u, p, 1,
        ptr_I, n_i, q, 1,
        ptr_Ub, m_ubin, pbin, 1,
        ptr_Ib, n_ibin, qbin, 1,
        ptr_Urow, ptr_Ucol, ptr_Usp, nnz_U,
        ptr_Irow, ptr_Icol, ptr_Isp, nnz_I,
        NULL, NULL, NULL,
        NULL, NULL, NULL,
        NULL, NULL, NULL,
        NULL, NULL, NULL,
        &buffer_double[0], <double*>NULL,
        k_main, k_user, k_item,
        w_main, w_user, w_item,
        nthreads
    )
    return f, grad


def py_fun_grad_collective_single(
        np.ndarray[double, ndim=1] a_vec,
        np.ndarray[double, ndim=1] grad,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[double, ndim=1] Xa,
        np.ndarray[double, ndim=1] Xfull,
        np.ndarray[int, ndim=1] ixB_U,
        np.ndarray[double, ndim=1] U_sp,
        np.ndarray[double, ndim=1] U,
        np.ndarray[double, ndim=1] Ubin,
        np.ndarray[double, ndim=2] B,
        np.ndarray[double, ndim=2] C,
        np.ndarray[double, ndim=2] Cb,
        np.ndarray[double, ndim=1] weight,
        int k, int k_main, int k_user, int k_item,
        double lam, double w_user, double w_main,
        np.ndarray[double, ndim=1] buffer_double,
    ):


    cdef int *ptr_ixB = NULL
    cdef double *ptr_Xa = NULL
    cdef size_t nnz = 0
    cdef double *ptr_Xfull = NULL
    if Xfull.shape[0]:
        ptr_Xfull = &Xfull[0]
    elif Xa.shape[0]:
        ptr_ixB = &ixB[0]
        ptr_Xa = &Xa[0]
        nnz = Xa.shape[0]

    cdef int *ptr_ixB_U = NULL
    cdef double *ptr_U_sp = NULL
    cdef size_t nnz_u_vec = 0
    cdef double *ptr_U = NULL
    if U.shape[0]:
        ptr_U = &U[0]
    elif U_sp.shape[0]:
        ptr_ixB_U = &ixB_U[0]
        ptr_U_sp = &U_sp[0]
        nnz_u_vec = U_sp.shape[0]

    cdef double *ptr_Ubin = NULL
    if Ubin.shape[0]:
        ptr_Ubin = &Ubin[0]

    cdef double *ptr_C = NULL
    cdef double *ptr_Cb = NULL
    cdef int p = 0
    cdef int pbin = 0
    if C.shape[0]:
        ptr_C = &C[0,0]
        p = C.shape[0]
    if Cb.shape[0]:
        ptr_Cb = &Cb[0,0]
        pbin = Cb.shape[0]

    cdef double *ptr_weight = NULL
    if weight.shape[0]:
        ptr_weight = &weight[0]

    cdef bint u_vec_has_NA = np.any(np.isnan(U))
    cdef bint u_bin_vec_has_NA = np.any(np.isnan(Ubin))

    cdef double f = collective_fun_grad_single(
        &a_vec[0], &grad[0],
        k, k_user, k_item, k_main,
        ptr_U, p,
        ptr_ixB_U, ptr_U_sp, nnz_u_vec,
        ptr_Ubin, pbin,
        u_vec_has_NA, u_bin_vec_has_NA,
        &B[0,0], B.shape[0],
        ptr_C, ptr_Cb,
        ptr_Xa, ptr_ixB, nnz,
        ptr_Xfull,
        ptr_weight,
        &buffer_double[0],
        lam, w_main, w_user, lam
    )
    return f, grad

def py_collective_cold_start(
        np.ndarray[int, ndim=1] ixB_U,
        np.ndarray[double, ndim=1] U_sp,
        np.ndarray[double, ndim=1] U,
        np.ndarray[double, ndim=1] Ubin,
        np.ndarray[double, ndim=2] C,
        np.ndarray[double, ndim=2] Cb,
        int k, int k_main, int k_user,
        double lam, double w_user,
        bint NA_as_zero_U,
        bint precompute
    ):
    
    cdef np.ndarray[double, ndim=1] outp = np.empty(k_user+k+k_main, dtype=ctypes.c_double)
    cdef double *ptr_TransCtCinvCt = NULL
    cdef double *ptr_CtCw = NULL
    cdef np.ndarray[double, ndim=2] TransCtCinvCt
    cdef np.ndarray[double, ndim=2] CtCw
    cdef int p = C.shape[0]
    cdef int pbin = Cb.shape[0]
    if precompute and C.shape[0]:
        CtCw = np.empty((k_user+k, k_user+k))
        TransCtCinvCt = np.empty((p, k_user+k))
        ptr_CtCw = &CtCw[0,0]
        ptr_TransCtCinvCt = &TransCtCinvCt[0,0]
        if np.isfortran(C):
            C = np.ascontiguousarray(C)

        precompute_collective_explicit(
            <double*>NULL, 0, 0, 0,
            &C[0,0], p,
            <double*>NULL, 0,
            <double*>NULL, 0., 0,
            k, k_user, 0, k_main,
            0, 0,
            lam, <double*>NULL,
            0, 0,
            1., w_user, 1.,
            <double*>NULL,
            <double*>NULL,
            <double*>NULL,
            <double*>NULL,
            <double*>NULL,
            <double*>NULL,
            ptr_TransCtCinvCt,
            ptr_CtCw
        )

        # CtCw = w_user * C.T.dot(C)
        # TransCtCinvCt = np.linalg.inv(C.T.dot(C) + (lam/w_user)*np.eye(k_user+k)).dot(C.T)
        # TransCtCinvCt = np.ascontiguousarray(TransCtCinvCt.T)
        # ptr_TransCtCinvCt = &TransCtCinvCt[0,0]
        # ptr_CtCw = &CtCw[0,0]

    cdef double *ptr_C = NULL
    cdef double *ptr_Cb = NULL
    if (C.shape[0]) and (U.shape[0] or U_sp.shape[0]):
        ptr_C = &C[0,0]
    if (Cb.shape[0]) and (Ubin.shape[0]):
        ptr_Cb = &Cb[0,0]

    cdef int *ptr_ixB_U = NULL
    cdef double *ptr_Usp = NULL
    cdef size_t nnz_u_vec = 0
    cdef double *ptr_U = NULL
    if U.shape[0]:
        ptr_U = &U[0]
    elif U_sp.shape[0]:
        ptr_ixB_U = &ixB_U[0]
        ptr_Usp = &U_sp[0]
        nnz_u_vec = U_sp.shape[0]

    cdef double *ptr_Ubin = NULL
    if Ubin.shape[0]:
        ptr_Ubin = &Ubin[0]
    
    collective_factors_cold(
        &outp[0],
        ptr_U, p,
        ptr_Usp, ptr_ixB_U, nnz_u_vec,
        ptr_Ubin, pbin,
        ptr_C, ptr_Cb,
        ptr_TransCtCinvCt, ptr_CtCw,
        <double*>NULL,
        k, k_user, k_main,
        lam, 0., 1., w_user,
        0,
        NA_as_zero_U,
        0
    )
    return outp

def py_optimizeA(
    np.ndarray[double, ndim=2] A,
    np.ndarray[double, ndim=2] B,
    int m, int n, int k,
    int lda, int ldb,
    np.ndarray[size_t, ndim=1] Xcsr_p,
    np.ndarray[int, ndim=1] Xcsr_i,
    np.ndarray[double, ndim=1] Xcsr,
    np.ndarray[double, ndim=2] Xfull,
    np.ndarray[double, ndim=1] weight,
    bint is_B,
    double lam, double w,
    bint NA_as_zero,
    bint as_near_dense,
    int nthreads,
    np.ndarray[double, ndim=1] buffer_double
    ):

    cdef size_t *ptr_indptr = NULL
    cdef int *ptr_indices = NULL
    cdef double *ptr_values = NULL
    cdef double *ptr_Xfull = NULL
    cdef int *ptr_cnt_NA = NULL
    cdef np.ndarray[int, ndim=1] cnt_NA
    cdef double *ptr_weight = NULL
    cdef bint full_dense = 0
    if Xfull.shape[0]:
        ptr_Xfull = &Xfull[0,0]
        if not is_B:
            cnt_NA = np.isnan(Xfull).sum(axis = 1).astype(ctypes.c_int)
        else:
            cnt_NA = np.isnan(Xfull).sum(axis = 0).astype(ctypes.c_int)
        ptr_cnt_NA = &cnt_NA[0]
        full_dense = cnt_NA.sum() == 0
        if full_dense:
            as_near_dense = 0
    else:
        ptr_indptr = &Xcsr_p[0]
        ptr_indices = &Xcsr_i[0]
        ptr_values = &Xcsr[0]
    if weight.shape[0]:
        ptr_weight = &weight[0]

    if w != 1.:
        lam /= w

    cdef c_bool ignore = 0
    
    optimizeA(
        &A[0,0], lda,
        &B[0,0], ldb,
        m, n, k,
        ptr_indptr, ptr_indices, ptr_values,
        ptr_Xfull, Xfull.shape[1], full_dense, as_near_dense,
        ptr_cnt_NA, ptr_weight, NA_as_zero,
        lam, lam,
        is_B, 1,
        nthreads,
        0, 0, 0, 0, <double*>NULL,
        <double*>NULL, <double*>NULL,
        0,
        <double*>NULL, &ignore,
        &buffer_double[0]
    )
    return A

def py_optimizeA_collective(
    np.ndarray[double, ndim=2] A,
    np.ndarray[double, ndim=2] B,
    np.ndarray[double, ndim=2] C,
    np.ndarray[double, ndim=2] Bi,
    int m, int n,
    int k, int k_user, int k_item, int k_main,
    int m_u, int p,
    np.ndarray[size_t, ndim=1] Xcsr_p,
    np.ndarray[int, ndim=1] Xcsr_i,
    np.ndarray[double, ndim=1] Xcsr,
    np.ndarray[double, ndim=2] Xfull,
    np.ndarray[double, ndim=1] weight,
    np.ndarray[size_t, ndim=1] U_csr_p,
    np.ndarray[int, ndim=1] U_csr_i,
    np.ndarray[double, ndim=1] U_csr,
    np.ndarray[double, ndim=2] U,
    bint is_B,
    double lam, double w_main, double w_user, double w_implicit,
    bint NA_as_zero_X, bint NA_as_zero_U,
    bint as_near_dense_x, bint as_near_dense_u,
    int nthreads,
    np.ndarray[double, ndim=1] buffer_double
    ):

    cdef np.ndarray[int, ndim=1] cnt_NA_x
    cdef bint full_dense = 0
    cdef int *ptr_cnt_NA_x = NULL
    cdef size_t *ptr_Xcsr_p = NULL
    cdef int *ptr_Xcsr_i = NULL
    cdef double *ptr_Xcsr = NULL
    cdef double *ptr_Xfull = NULL
    if Xfull.shape[0]:
        if not is_B:
            cnt_NA_x = np.isnan(Xfull).sum(axis = 1).astype(ctypes.c_int)
        else:
            cnt_NA_x = np.isnan(Xfull).sum(axis = 0).astype(ctypes.c_int)
        ptr_cnt_NA_x = &cnt_NA_x[0]
        if (cnt_NA_x.sum() == 0):
            full_dense = 1
            as_near_dense_x = 0
        ptr_Xfull = &Xfull[0,0]
    else:
        ptr_Xcsr_p = &Xcsr_p[0]
        ptr_Xcsr_i = &Xcsr_i[0]
        ptr_Xcsr = &Xcsr[0]

    cdef double *ptr_weight = NULL
    if weight.shape[0]:
        ptr_weight = &weight[0]

    cdef np.ndarray[int, ndim=1] cnt_NA_u
    cdef bint full_dense_u = 0
    cdef int *ptr_cnt_NA_u = NULL
    cdef size_t *ptr_Ucsr_p = NULL
    cdef int *ptr_Ucsr_i = NULL
    cdef double *ptr_Ucsr = NULL
    cdef double *ptr_U = NULL
    if U.shape[0]:
        cnt_NA_u = np.isnan(U).sum(axis = 1).astype(ctypes.c_int)
        ptr_cnt_NA_u = &cnt_NA_u[0]
        if (cnt_NA_u.sum() == 0):
            full_dense_u = 1
            as_near_dense_u = 0
        ptr_U = &U[0,0]
    else:
        ptr_Ucsr_p = &U_csr_p[0]
        ptr_Ucsr_i = &U_csr_i[0]
        ptr_Ucsr = &U_csr[0]

    if w_main != 1.:
        w_user /= w_main
        w_implicit /= w_main
        lam /= w_main

    cdef c_bool ignore1 = 0
    cdef c_bool ignore2 = 0
    cdef c_bool ignore3 = 0
    cdef c_bool ignore4 = 0

    cdef c_bool add_implicit_features = 0
    cdef double *ptr_Bi = NULL
    cdef double *ptr_X_ones = NULL
    cdef np.ndarray[double, ndim=1] Xones1
    cdef np.ndarray[double, ndim=2] Xones2
    if Bi.shape[0]:
        add_implicit_features = 1
        ptr_Bi = &Bi[0,0]
        if Xfull.shape[0]:
            Xones2 = (~np.isnan(Xfull)).astype("float64")
            if np.isfortran(Xfull) != np.isfortran(Xones2):
                if np.isfortran(Xfull):
                    Xones2 = np.asfortranarray(Xones2)
                else:
                    Xones2 = np.ascontiguousarray(Xones2)
            ptr_X_ones = &Xones2[0,0]
        else:
            Xones1 = np.ones(Xcsr.shape[0], dtype="float64")
            ptr_X_ones = &Xones1[0]

    optimizeA_collective(
        &A[0,0], A.shape[1], &B[0,0], B.shape[1], &C[0,0], ptr_Bi,
        m, m_u, n, p,
        k, k_main, k_user, k_item,
        ptr_Xcsr_p, ptr_Xcsr_i, ptr_Xcsr,
        ptr_Xfull, full_dense, as_near_dense_x, m if is_B else n,
        ptr_cnt_NA_x, ptr_weight, NA_as_zero_X,
        ptr_X_ones, k_main, m if is_B else n,
        add_implicit_features,
        ptr_Ucsr_p, ptr_Ucsr_i, ptr_Ucsr,
        ptr_U, ptr_cnt_NA_u,
        full_dense_u, as_near_dense_u, NA_as_zero_U,
        lam, w_user, w_implicit, lam,
        is_B,
        nthreads,
        0, 0, 1,
        0, 0, <double*>NULL,
        <double*>NULL, <double*>NULL,
        0, <double*>NULL, <double*>NULL, <double*>NULL, <double*>NULL,
        &ignore1, &ignore2, &ignore3, &ignore4,
        &buffer_double[0]
    )
    return A

def py_optimizeA_implicit(
    np.ndarray[double, ndim=2] A,
    np.ndarray[double, ndim=2] B,
    int m, int n, int k,
    np.ndarray[size_t, ndim=1] Xcsr_p,
    np.ndarray[int, ndim=1] Xcsr_i,
    np.ndarray[double, ndim=1] Xcsr,
    double lam, double alpha,
    int nthreads,
    np.ndarray[double, ndim=1] buffer_double
    ):
    
    cdef size_t lda = A.shape[1]
    cdef size_t ldb = B.shape[1]

    cdef np.ndarray[double, ndim=1] Xcsr_pass = Xcsr * alpha
    
    optimizeA_implicit(
        &A[0,0], lda,
        &B[0,0], ldb,
        m, n, k,
        &Xcsr_p[0], &Xcsr_i[0], &Xcsr_pass[0],
        lam,
        nthreads,
        0, 0, 1,
        0, 0, <double*>NULL,
        <double*>NULL,
        &buffer_double[0]
    )
    return A

def py_optimizeA_collective_implicit(
        np.ndarray[double, ndim=2] A,
        np.ndarray[double, ndim=2] B,
        np.ndarray[double, ndim=2] C,
        int m, int n,
        int k, int k_user, int k_item, int k_main,
        int m_u, int p,
        np.ndarray[size_t, ndim=1] Xcsr_p,
        np.ndarray[int, ndim=1] Xcsr_i,
        np.ndarray[double, ndim=1] Xcsr,
        np.ndarray[size_t, ndim=1] U_csr_p,
        np.ndarray[int, ndim=1] U_csr_i,
        np.ndarray[double, ndim=1] U_csr,
        np.ndarray[double, ndim=2] U,
        double lam, double alpha, double w_main, double w_user,
        bint NA_as_zero_U, bint as_near_dense_u,
        int nthreads,
        np.ndarray[double, ndim=1] buffer_double
    ):

    cdef size_t *ptr_Xcsr_p = NULL
    cdef int *ptr_Xcsr_i = NULL
    cdef double *ptr_Xcsr = NULL
    ptr_Xcsr_p = &Xcsr_p[0]
    ptr_Xcsr_i = &Xcsr_i[0]
    # ptr_Xcsr = &Xcsr[0]

    cdef np.ndarray[double, ndim=1] Xcsr_pass = alpha * Xcsr
    ptr_Xcsr = &Xcsr_pass[0]

    cdef np.ndarray[int, ndim=1] cnt_NA_u
    cdef bint full_dense_u = 0
    cdef int *ptr_cnt_NA_u = NULL
    cdef size_t *ptr_Ucsr_p = NULL
    cdef int *ptr_Ucsr_i = NULL
    cdef double *ptr_Ucsr = NULL
    cdef double *ptr_U = NULL
    if U.shape[0]:
        cnt_NA_u = np.isnan(U).sum(axis = 1).astype(ctypes.c_int)
        ptr_cnt_NA_u = &cnt_NA_u[0]
        if (cnt_NA_u.sum() == 0):
            full_dense_u = 1
            as_near_dense_u = 0
        ptr_U = &U[0,0]
    else:
        ptr_Ucsr_p = &U_csr_p[0]
        ptr_Ucsr_i = &U_csr_i[0]
        ptr_Ucsr = &U_csr[0]

    cdef c_bool ignore1 = 0
    cdef c_bool ignore2 = 0
    cdef c_bool ignore3 = 0

    optimizeA_collective_implicit(
        &A[0,0], &B[0,0], &C[0,0],
        m, m_u, n, p,
        k, k_main, k_user, k_item,
        ptr_Xcsr_p, ptr_Xcsr_i, ptr_Xcsr,
        ptr_Ucsr_p, ptr_Ucsr_i, ptr_Ucsr,
        ptr_U, ptr_cnt_NA_u,
        full_dense_u, as_near_dense_u, NA_as_zero_U,
        lam/w_main, w_user/w_main,
        nthreads,
        0, 0, 1,
        0, 0, <double*>NULL,
        <double*>NULL,
        <double*>NULL,
        <double*>NULL,
        <double*>NULL,
        &ignore1,
        &ignore2,
        &ignore3,
        &buffer_double[0]
    )

    return A

def py_fun_grad_offsets(
        np.ndarray[double, ndim=1] values,
        np.ndarray[double, ndim=1] grad,
        np.ndarray[double, ndim=2] Xfull,
        np.ndarray[int, ndim=1] ixA,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[double, ndim=1] X,
        np.ndarray[double, ndim=2] U,
        np.ndarray[size_t, ndim=1] U_csr_p,
        np.ndarray[int, ndim=1] U_csr_i,
        np.ndarray[double, ndim=1] U_csr,
        np.ndarray[size_t, ndim=1] U_csc_p,
        np.ndarray[int, ndim=1] U_csc_i,
        np.ndarray[double, ndim=1] U_csc,
        np.ndarray[double, ndim=2] II,
        np.ndarray[size_t, ndim=1] I_csr_p,
        np.ndarray[int, ndim=1] I_csr_i,
        np.ndarray[double, ndim=1] I_csr,
        np.ndarray[size_t, ndim=1] I_csc_p,
        np.ndarray[int, ndim=1] I_csc_i,
        np.ndarray[double, ndim=1] I_csc,
        np.ndarray[double, ndim=1] weight,
        int m, int n, int p, int q,
        int k, int k_main, int k_sec,
        double w_user, double w_item,
        double lam,
        bint user_bias, bint item_bias,
        bint add_intercepts,
        int nthreads,
        np.ndarray[double, ndim=1] buffer_double
    ):

    cdef double *ptr_weight = NULL
    if weight.shape[0]:
        ptr_weight = &weight[0]

    cdef double *ptr_X = NULL
    cdef int *ptr_ixA = NULL
    cdef int *ptr_ixB = NULL
    cdef size_t nnz = 0
    cdef double *ptr_Xfull = NULL
    cdef bint full_dense = 0
    if Xfull.shape[0]:
        ptr_Xfull = &Xfull[0,0]
        full_dense = np.sum(np.isnan(Xfull)) == 0
    elif X.shape[0]:
        ptr_X = &X[0]
        ptr_ixA = &ixA[0]
        ptr_ixB = &ixB[0]
        nnz = X.shape[0]

    cdef double *ptr_U = NULL
    cdef size_t *ptr_Ucsr_p = NULL
    cdef int *ptr_Ucsr_i = NULL
    cdef double *ptr_Ucsr = NULL
    cdef size_t *ptr_Ucsc_p = NULL
    cdef int *ptr_Ucsc_i = NULL
    cdef double *ptr_Ucsc = NULL
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_csr_p.shape[0]:
        ptr_Ucsr_p = &U_csr_p[0]
        ptr_Ucsr_i = &U_csr_i[0]
        ptr_Ucsr = &U_csr[0]
        ptr_Ucsc_p = &U_csc_p[0]
        ptr_Ucsc_i = &U_csc_i[0]
        ptr_Ucsc = &U_csc[0]

    cdef double *ptr_I = NULL
    cdef size_t *ptr_Icsr_p = NULL
    cdef int *ptr_Icsr_i = NULL
    cdef double *ptr_Icsr = NULL
    cdef size_t *ptr_Icsc_p = NULL
    cdef int *ptr_Icsc_i = NULL
    cdef double *ptr_Icsc = NULL
    if II.shape[0]:
        ptr_I = &II[0,0]
    elif I_csr_p.shape[0]:
        ptr_Icsr_p = &I_csr_p[0]
        ptr_Icsr_i = &I_csr_i[0]
        ptr_Icsr = &I_csr[0]
        ptr_Icsc_p = &I_csc_p[0]
        ptr_Icsc_i = &I_csc_i[0]
        ptr_Icsc = &I_csc[0]



    cdef double f = offsets_fun_grad(
        &values[0], &grad[0],
        ptr_ixA, ptr_ixB, ptr_X,
        nnz, m, n, k,
        ptr_Xfull, full_dense,
        <size_t*>NULL, <int*>NULL, <double*>NULL,
        <size_t*>NULL, <int*>NULL, <double*>NULL,
        ptr_weight, <double*>NULL, <double*>NULL,
        user_bias, item_bias,
        add_intercepts,
        lam, <double*>NULL,
        ptr_U, p,
        ptr_I, q,
        ptr_Ucsr_p, ptr_Ucsr_i, ptr_Ucsr,
        ptr_Ucsc_p, ptr_Ucsc_i, ptr_Ucsc,
        ptr_Icsr_p, ptr_Icsr_i, ptr_Icsr,
        ptr_Icsc_p, ptr_Icsc_i, ptr_Icsc,
        k_main, k_sec,
        w_user, w_item,
        nthreads,
        &buffer_double[0],
        <double*>NULL
    )
    return f, grad

def py_offsets_warm_start(
        np.ndarray[double, ndim=1] outp,
        np.ndarray[double, ndim=1] outp2,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[double, ndim=1] Xa,
        np.ndarray[double, ndim=1] Xfull,
        np.ndarray[double, ndim=1] weight,
        np.ndarray[int, ndim=1] ixB_U,
        np.ndarray[double, ndim=1] U_sp,
        np.ndarray[double, ndim=1] U,
        np.ndarray[double, ndim=2] Bm,
        np.ndarray[double, ndim=2] C,
        np.ndarray[double, ndim=1] C_bias,
        int k, int k_main, int k_sec,
        double lam, double w_user,
        bint precompute,
        bint exact
    ):

    cdef int *ptr_ixB = NULL
    cdef double *ptr_Xa = NULL
    cdef size_t nnz = 0
    cdef double *ptr_Xfull = NULL
    if Xfull.shape[0]:
        ptr_Xfull = &Xfull[0]
    elif Xa.shape[0]:
        ptr_ixB = &ixB[0]
        ptr_Xa = &Xa[0]
        nnz = Xa.shape[0]

    cdef int *ptr_ixB_U = NULL
    cdef double *ptr_U_sp = NULL
    cdef size_t nnz_u_vec = 0
    cdef double *ptr_U = NULL
    if U.shape[0]:
        ptr_U = &U[0]
    elif U_sp.shape[0]:
        ptr_ixB_U = &ixB_U[0]
        ptr_U_sp = &U_sp[0]
        nnz_u_vec = U_sp.shape[0]

    cdef double *ptr_weight = NULL
    if weight.shape[0]:
        ptr_weight = &weight[0]

    cdef double *ptr_outp2 = NULL
    if outp2.shape[0]:
        ptr_outp2 = &outp2[0]

    cdef double *ptr_C_bias = NULL
    if C_bias.shape[0]:
        ptr_C_bias = &C_bias[0]

    cdef np.ndarray[double, ndim=2] precomputedTransBtBinvBt
    cdef np.ndarray[double, ndim=2] precomputedBtBw
    cdef double *ptr_TransBtBinvBt = NULL
    cdef double *ptr_BtBw = NULL
    if precompute:
        precomputedBtBw = Bm.T.dot(Bm)
        precomputedTransBtBinvBt = np.linalg.inv(Bm.T.dot(Bm)).dot(Bm.T)
        ptr_TransBtBinvBt = &precomputedTransBtBinvBt[0,0]
        ptr_BtBw = &precomputedBtBw[0,0]

    offsets_factors_warm(
        &outp[0], <double*>NULL,
        ptr_U,
        ptr_ixB_U, ptr_U_sp, nnz_u_vec,
        ptr_ixB, ptr_Xa, nnz,
        ptr_Xfull, Bm.shape[0],
        ptr_weight,
        &Bm[0,0], &C[0,0],
        ptr_C_bias,
        0., <double*>NULL,
        k, k_sec, k_main,
        C.shape[0], w_user,
        lam, exact, 0.,
        0, 0.,
        ptr_TransBtBinvBt,
        ptr_BtBw,
        ptr_outp2,
        <double*>NULL
    )

    return outp, outp2

def py_initialize_biases(
        np.ndarray[double, ndim=1] biasA,
        np.ndarray[double, ndim=1] biasB,
        np.ndarray[double, ndim=2] Xfull,
        np.ndarray[int, ndim=1] ixA,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[double, ndim=1] X,
        np.ndarray[size_t, ndim=1] Xcsr_p,
        np.ndarray[int, ndim=1] Xcsr_i,
        np.ndarray[double, ndim=1] Xcsr,
        np.ndarray[size_t, ndim=1] Xcsc_p,
        np.ndarray[int, ndim=1] Xcsc_i,
        np.ndarray[double, ndim=1] Xcsc,
        int m, int n, bint user_bias, bint item_bias,
        bint has_trans,
        int nthreads
    ):

    cdef double *ptr_biasA = NULL
    cdef double *ptr_biasB = NULL
    if biasA.shape[0]:
        ptr_biasA = &biasA[0]
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]

    cdef double *ptr_Xfull = NULL
    cdef double *ptr_Xtrans = NULL
    cdef np.ndarray[double, ndim=2] Xtrans
    cdef size_t nnz = 0
    cdef int *ptr_ixA = NULL
    cdef int *ptr_ixB = NULL
    cdef double *ptr_X = NULL
    cdef size_t *ptr_Xcsr_p = NULL
    cdef int *ptr_Xcsr_i = NULL
    cdef double *ptr_Xcsr = NULL
    cdef size_t *ptr_Xcsc_p = NULL
    cdef int *ptr_Xcsc_i = NULL
    cdef double *ptr_Xcsc = NULL
    if Xfull.shape[0]:
        ptr_Xfull = &Xfull[0,0]
        if has_trans:
            Xtrans = np.ascontiguousarray(Xfull.T)
            ptr_Xtrans = &Xtrans[0,0]
    else:
        ptr_ixA = &ixA[0]
        ptr_ixB = &ixB[0]
        ptr_X = &X[0]
        nnz = X.shape[0]
        if Xcsr.shape[0]:
            ptr_Xcsr_p = &Xcsr_p[0]
            ptr_Xcsr_i = &Xcsr_i[0]
            ptr_Xcsr = &Xcsr[0]
        if Xcsc.shape[0]:
            ptr_Xcsc_p = &Xcsc_p[0]
            ptr_Xcsc_i = &Xcsc_i[0]
            ptr_Xcsc = &Xcsc[0]
    
    cdef double glob_mean
    initialize_biases(
        &glob_mean, ptr_biasA, ptr_biasB,
        user_bias, item_bias,
        0., 0., 0,
        m, n,
        m, n,
        ptr_ixA, ptr_ixB, ptr_X, nnz,
        ptr_Xfull, ptr_Xtrans,
        ptr_Xcsr_p, ptr_Xcsr_i, ptr_Xcsr,
        ptr_Xcsc_p, ptr_Xcsc_i, ptr_Xcsc,
        0,
        nthreads
    )
    return glob_mean, biasA, biasB

def py_center_by_cols(
        np.ndarray[double, ndim=1] col_means,
        np.ndarray[double, ndim=2] Xfull,
        np.ndarray[int, ndim=1] ixA,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[double, ndim=1] X,
        np.ndarray[size_t, ndim=1] Xcsr_p,
        np.ndarray[int, ndim=1] Xcsr_i,
        np.ndarray[double, ndim=1] Xcsr,
        np.ndarray[size_t, ndim=1] Xcsc_p,
        np.ndarray[int, ndim=1] Xcsc_i,
        np.ndarray[double, ndim=1] Xcsc,
        int m, int n,
        int nthreads
    ):

    cdef double *ptr_Xfull = NULL
    cdef size_t nnz = 0
    cdef int *ptr_ixA = NULL
    cdef int *ptr_ixB = NULL
    cdef double *ptr_X = NULL
    cdef size_t *ptr_Xcsr_p = NULL
    cdef int *ptr_Xcsr_i = NULL
    cdef double *ptr_Xcsr = NULL
    cdef size_t *ptr_Xcsc_p = NULL
    cdef int *ptr_Xcsc_i = NULL
    cdef double *ptr_Xcsc = NULL
    if Xfull.shape[0]:
        ptr_Xfull = &Xfull[0,0]
    else:
        ptr_ixA = &ixA[0]
        ptr_ixB = &ixB[0]
        ptr_X = &X[0]
        nnz = X.shape[0]
        if Xcsr.shape[0]:
            ptr_Xcsr_p = &Xcsr_p[0]
            ptr_Xcsr_i = &Xcsr_i[0]
            ptr_Xcsr = &Xcsr[0]
        if Xcsc.shape[0]:
            ptr_Xcsc_p = &Xcsc_p[0]
            ptr_Xcsc_i = &Xcsc_i[0]
            ptr_Xcsc = &Xcsc[0]

    center_by_cols(
        &col_means[0],
        ptr_Xfull, m, n,
        ptr_ixA, ptr_ixB, ptr_X, nnz,
        ptr_Xcsr_p, ptr_Xcsr_i, ptr_Xcsr,
        ptr_Xcsc_p, ptr_Xcsc_i, ptr_Xcsc,
        nthreads
    )
    return col_means

def py_topN(
        np.ndarray[double, ndim=1] a_vec,
        np.ndarray[double, ndim=2] B,
        np.ndarray[double, ndim=1] biasB,
        double glob_mean, double biasA,
        np.ndarray[int, ndim=1] include_ix,
        np.ndarray[int, ndim=1] exclude_ix,
        int n_top,
        int k, int k_user = 0, int k_item = 0, int k_main = 0,
        bint output_score = 1
    ):

    cdef double *ptr_biasB = NULL
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]

    cdef int *ptr_include = NULL
    cdef int *ptr_exclude = NULL
    if include_ix.shape[0]:
        ptr_include = &include_ix[0]
    if exclude_ix.shape[0]:
        ptr_exclude = &exclude_ix[0]

    cdef np.ndarray[double, ndim=1] outp_score = np.empty(0, dtype=ctypes.c_double)
    cdef double *ptr_outp_score = NULL
    if output_score:
        outp_score = np.empty(n_top, dtype=ctypes.c_double)
        ptr_outp_score = &outp_score[0]
    cdef np.ndarray[int, ndim=1] outp_ix = np.empty(n_top, dtype=ctypes.c_int)
    topN(
        &a_vec[0], k_user,
        &B[0,0], k_item,
        ptr_biasB,
        glob_mean, biasA,
        k, k_main,
        ptr_include, include_ix.shape[0],
        ptr_exclude, exclude_ix.shape[0],
        &outp_ix[0], ptr_outp_score,
        n_top, B.shape[0], 1
    )

    return outp_ix, outp_score
