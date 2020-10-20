import numpy as np
cimport numpy as np
import ctypes

# ctypedef double FPnum

###Uncomment code below to take BLAS and LAPACK from SciPy instead


# IF UNAME_SYSNAME != "Windows": ### for windows, will use package 'findblas'
#     from scipy.linalg.cython_blas cimport sdot as sdot_
#     from scipy.linalg.cython_blas cimport scopy as scopy_
#     from scipy.linalg.cython_blas cimport saxpy as saxpy_
#     from scipy.linalg.cython_blas cimport sscal as sscal_
#     from scipy.linalg.cython_blas cimport ssyr as ssyr_
#     from scipy.linalg.cython_blas cimport ssyrk as ssyrk_
#     from scipy.linalg.cython_blas cimport snrm2 as snrm2_
#     from scipy.linalg.cython_blas cimport sgemm as sgemm_
#     from scipy.linalg.cython_blas cimport sgemv as sgemv_
#     from scipy.linalg.cython_blas cimport ssymv as ssymv_

#     from scipy.linalg.cython_lapack cimport slacpy as slacpy_
#     from scipy.linalg.cython_lapack cimport sposv as sposv_
#     from scipy.linalg.cython_lapack cimport slarnv as slarnv_
#     from scipy.linalg.cython_lapack cimport spotrf as spotrf_
#     from scipy.linalg.cython_lapack cimport spotrs as spotrs_
#     from scipy.linalg.cython_lapack cimport sgels as sgels_

#     from scipy.linalg.cython_blas cimport ddot as ddot_
#     from scipy.linalg.cython_blas cimport dcopy as dcopy_
#     from scipy.linalg.cython_blas cimport daxpy as daxpy_
#     from scipy.linalg.cython_blas cimport dscal as dscal_
#     from scipy.linalg.cython_blas cimport dsyr as dsyr_
#     from scipy.linalg.cython_blas cimport dsyrk as dsyrk_
#     from scipy.linalg.cython_blas cimport dnrm2 as dnrm2_
#     from scipy.linalg.cython_blas cimport dgemm as dgemm_
#     from scipy.linalg.cython_blas cimport dgemv as dgemv_
#     from scipy.linalg.cython_blas cimport dsymv as dsymv_

#     from scipy.linalg.cython_lapack cimport dlacpy as dlacpy_
#     from scipy.linalg.cython_lapack cimport dposv as dposv_
#     from scipy.linalg.cython_lapack cimport dlarnv as dlarnv_
#     from scipy.linalg.cython_lapack cimport dpotrf as dpotrf_
#     from scipy.linalg.cython_lapack cimport dpotrs as dpotrs_
#     from scipy.linalg.cython_lapack cimport dgels as dgels_



### TODO: this module should move from doing operations in Python to
### using the new designated C functions for each type of prediction.


cdef extern from "cmfrec.h":
    int fit_collective_explicit_lbfgs_internal(
        FPnum *values, bint reset_values,
        FPnum *glob_mean,
        FPnum *U_colmeans, FPnum *I_colmeans,
        int m, int n, int k,
        int ixA[], int ixB[], FPnum *X, size_t nnz,
        FPnum *Xfull,
        FPnum *weight,
        bint user_bias, bint item_bias,
        FPnum lam, FPnum *lam_unique,
        FPnum *U, int m_u, int p,
        FPnum *II, int n_i, int q,
        FPnum *Ub, int m_ubin, int pbin,
        FPnum *Ib, int n_ibin, int qbin,
        int U_row[], int U_col[], FPnum *U_sp, size_t nnz_U,
        int I_row[], int I_col[], FPnum *I_sp, size_t nnz_I,
        int k_main, int k_user, int k_item,
        FPnum w_main, FPnum w_user, FPnum w_item,
        int n_corr_pairs, size_t maxiter, int seed,
        int nthreads, bint prefer_onepass,
        bint verbose, int print_every, bint handle_interrupt,
        int *niter, int *nfev,
        FPnum *B_plus_bias
    )

    int fit_offsets_explicit_lbfgs(
        FPnum *values, bint reset_values,
        FPnum *glob_mean,
        int m, int n, int k,
        int ixA[], int ixB[], FPnum *X, size_t nnz,
        FPnum *Xfull,
        FPnum *weight,
        bint user_bias, bint item_bias,
        bint add_intercepts,
        FPnum lam, FPnum *lam_unique,
        FPnum *U, int p,
        FPnum *II, int q,
        int U_row[], int U_col[], FPnum *U_sp, size_t nnz_U,
        int I_row[], int I_col[], FPnum *I_sp, size_t nnz_I,
        int k_main, int k_sec,
        FPnum w_user, FPnum w_item,
        int n_corr_pairs, size_t maxiter, int seed,
        int nthreads, bint prefer_onepass,
        bint verbose, int print_every, bint handle_interrupt,
        int *niter, int *nfev,
        FPnum *Am, FPnum *Bm,
        FPnum *B_plus_bias
    )

    int fit_collective_explicit_als(
        FPnum *biasA, FPnum *biasB,
        FPnum *A, FPnum *B,
        FPnum *C, FPnum *D,
        bint reset_values, int seed,
        FPnum *glob_mean,
        FPnum *U_colmeans, FPnum *I_colmeans,
        int m, int n, int k,
        int ixA[], int ixB[], FPnum *X, size_t nnz,
        FPnum *Xfull,
        FPnum *weight,
        bint user_bias, bint item_bias,
        FPnum lam, FPnum *lam_unique,
        FPnum *U, int m_u, int p,
        FPnum *II, int n_i, int q,
        int U_row[], int U_col[], FPnum *U_sp, size_t nnz_U,
        int I_row[], int I_col[], FPnum *I_sp, size_t nnz_I,
        bint NA_as_zero_X, bint NA_as_zero_U, bint NA_as_zero_I,
        int k_main, int k_user, int k_item,
        FPnum w_main, FPnum w_user, FPnum w_item,
        int niter, int nthreads, bint verbose, bint handle_interrupt,
        bint use_cg, int max_cg_steps, bint finalize_chol,
        bint precompute_for_predictions,
        FPnum *B_plus_bias,
        FPnum *precomputedBtB,
        FPnum *precomputedTransBtBinvBt,
        FPnum *precomputedBeTBeChol,
        FPnum *precomputedTransCtCinvCt,
        FPnum *precomputedCtCw
    )

    int fit_collective_implicit_als(
        FPnum *A, FPnum *B,
        FPnum *C, FPnum *D,
        bint reset_values, int seed,
        FPnum *U_colmeans, FPnum *I_colmeans,
        int m, int n, int k,
        int ixA[], int ixB[], FPnum *X, size_t nnz,
        FPnum lam, FPnum *lam_unique,
        FPnum *U, int m_u, int p,
        FPnum *II, int n_i, int q,
        int U_row[], int U_col[], FPnum *U_sp, size_t nnz_U,
        int I_row[], int I_col[], FPnum *I_sp, size_t nnz_I,
        bint NA_as_zero_U, bint NA_as_zero_I,
        int k_main, int k_user, int k_item,
        FPnum w_main, FPnum w_user, FPnum w_item,
        FPnum *w_main_multiplier,
        FPnum alpha, bint adjust_weight,
        int niter, int nthreads, bint verbose, bint handle_interrupt,
        bint use_cg, int max_cg_steps, bint finalize_chol,
        bint precompute_for_predictions,
        FPnum *precomputedBtB,
        FPnum *precomputedBeTBe,
        FPnum *precomputedBeTBeChol
    )

    int fit_offsets_als(
        FPnum *biasA, FPnum *biasB,
        FPnum *A, FPnum *B,
        FPnum *C, FPnum *C_bias,
        FPnum *D, FPnum *D_bias,
        bint reset_values, int seed,
        FPnum *glob_mean,
        int m, int n, int k,
        int ixA[], int ixB[], FPnum *X, size_t nnz,
        FPnum *Xfull,
        FPnum *weight,
        bint user_bias, bint item_bias, bint add_intercepts,
        FPnum lam,
        FPnum *U, int p,
        FPnum *II, int q,
        bint implicit, bint NA_as_zero_X, FPnum alpha,
        int niter,
        int nthreads, bint use_cg,
        int max_cg_steps, bint finalize_chol,
        bint verbose, bint handle_interrupt,
        bint precompute_for_predictions,
        FPnum *Am, FPnum *Bm,
        FPnum *Bm_plus_bias,
        FPnum *precomputedBtB,
        FPnum *precomputedTransBtBinvBt

    )

    int precompute_collective_explicit(
        FPnum *B, int n, int n_i, int n_ibin,
        FPnum *C, int p,
        int k, int k_user, int k_item, int k_main,
        bint user_bias,
        FPnum lam, FPnum *lam_unique,
        FPnum w_main, FPnum w_user,
        FPnum *B_plus_bias,
        FPnum *BtB,
        FPnum *TransBtBinvBt,
        FPnum *BeTBeChol,
        FPnum *TransCtCinvCt,
        FPnum *CtCw
    )

    int precompute_collective_implicit(
        FPnum *B, int n,
        FPnum *C, int p,
        int k, int k_user, int k_item, int k_main,
        FPnum lam, FPnum w_main, FPnum w_user, FPnum w_main_multiplier,
        bint extra_precision,
        FPnum *BtB,
        FPnum *BeTBe,
        FPnum *BeTBeChol
    )

    int collective_factors_cold(
        FPnum *a_vec,
        FPnum *u_vec, int p,
        FPnum *u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
        FPnum *u_bin_vec, int pbin,
        FPnum *C, FPnum *Cb,
        FPnum *TransCtCinvCt,
        FPnum *CtCw,
        FPnum *col_means,
        int k, int k_user, int k_main,
        FPnum lam, FPnum w_main, FPnum w_user,
        bint NA_as_zero_U
    )

    int collective_factors_cold_implicit(
        FPnum *a_vec,
        FPnum *u_vec, int p,
        FPnum *u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
        FPnum *B, int n,
        FPnum *C,
        FPnum *BeTBe,
        FPnum *BtB,
        FPnum *BeTBeChol,
        FPnum *col_means,
        int k, int k_user, int k_item, int k_main,
        FPnum lam, FPnum w_main, FPnum w_user, FPnum w_main_multiplier,
        bint NA_as_zero_U
    )

    int collective_factors_warm(
        FPnum *a_vec, FPnum *a_bias,
        FPnum *u_vec, int p,
        FPnum *u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
        FPnum *u_bin_vec, int pbin,
        FPnum *C, FPnum *Cb,
        FPnum glob_mean, FPnum *biasB,
        FPnum *col_means,
        FPnum *Xa, int ixB[], size_t nnz,
        FPnum *Xa_dense, int n,
        FPnum *weight,
        FPnum *B,
        int k, int k_user, int k_item, int k_main,
        FPnum lam, FPnum w_main, FPnum w_user, FPnum lam_bias,
        FPnum *TransBtBinvBt,
        FPnum *BtB,
        FPnum *BeTBeChol,
        FPnum *CtCw,
        bint NA_as_zero_U, bint NA_as_zero_X,
        FPnum *B_plus_bias
    )

    int collective_factors_warm_implicit(
        FPnum *a_vec,
        FPnum *u_vec, int p,
        FPnum *u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
        bint NA_as_zero_U,
        FPnum *col_means,
        FPnum *B, int n, FPnum *C,
        FPnum *Xa, int ixB[], size_t nnz,
        int k, int k_user, int k_item, int k_main,
        FPnum lam, FPnum alpha, FPnum w_main, FPnum w_user,
        FPnum w_main_multiplier,
        FPnum *BeTBe,
        FPnum *BtB,
        FPnum *BeTBeChol
    )

    int offsets_factors_cold(
        FPnum *a_vec,
        FPnum *u_vec,
        int u_vec_ixB[], FPnum *u_vec_sp, size_t nnz_u_vec,
        FPnum *C, int p,
        FPnum *C_bias,
        int k, int k_sec, int k_main,
        FPnum w_user
    )

    int offsets_factors_warm(
        FPnum *a_vec, FPnum *a_bias,
        FPnum *u_vec,
        int u_vec_ixB[], FPnum *u_vec_sp, size_t nnz_u_vec,
        int ixB[], FPnum *Xa, size_t nnz,
        FPnum *Xa_dense, int n,
        FPnum *weight,
        FPnum *Bm, FPnum *C,
        FPnum *C_bias,
        FPnum glob_mean, FPnum *biasB,
        int k, int k_sec, int k_main,
        int p, FPnum w_user,
        FPnum lam, bint exact, FPnum lam_bias,
        bint implicit, FPnum alpha,
        FPnum *precomputedTransBtBinvBt,
        FPnum *precomputedBtBw,
        FPnum *output_a,
        FPnum *Bm_plus_bias
    )

    void predict_multiple(
        FPnum *A, int k_user,
        FPnum *B, int k_item,
        FPnum *biasA, FPnum *biasB,
        FPnum glob_mean,
        int k, int k_main,
        int predA[], int predB[], size_t nnz,
        FPnum *outp,
        int nthreads
    )

    int topN(
        FPnum *a_vec, int k_user,
        FPnum *B, int k_item,
        FPnum *biasB,
        FPnum glob_mean, FPnum biasA,
        int k, int k_main,
        int *include_ix, int n_include,
        int *exclude_ix, int n_exclude,
        int *outp_ix, FPnum *outp_score,
        int n_top, int n, int nthreads
    )

    int topN_new_content_based(
        int k, int n_new,
        FPnum *u_vec, int p,
        FPnum *u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
        FPnum *II, int q,
        int I_row[], int I_col[], FPnum *I_sp, size_t nnz_I,
        size_t I_csr_p[], int I_csr_i[], FPnum *I_csr,
        FPnum *C, FPnum *C_bias,
        FPnum *D, FPnum *D_bias,
        FPnum glob_mean,
        int *outp_ix, FPnum *outp_score,
        int n_top, int nthreads
    )

    int fit_most_popular(
        FPnum *biasA, FPnum *biasB,
        FPnum *glob_mean,
        FPnum lam_bias, FPnum lam_item,
        FPnum alpha,
        int m, int n,
        int ixA[], int ixB[], FPnum *X, size_t nnz,
        FPnum *Xfull,
        FPnum *weight,
        bint implicit, bint adjust_weight,
        FPnum *w_main_multiplier,
        int nthreads
    )

    int impute_X_collective_explicit(
        int m, bint user_bias,
        FPnum *U, int m_u, int p,
        bint NA_as_zero_U,
        int U_row[], int U_col[], FPnum *U_sp, size_t nnz_U,
        size_t U_csr_p[], int U_csr_i[], FPnum *U_csr,
        FPnum *Ub, int m_ubin, int pbin,
        FPnum *C, FPnum *Cb,
        FPnum glob_mean, FPnum *biasB,
        FPnum *col_means,
        FPnum *Xfull, int n,
        FPnum *weight,
        FPnum *B,
        int k, int k_user, int k_item, int k_main,
        FPnum lam, FPnum *lam_unique,
        FPnum w_main, FPnum w_user,
        FPnum *TransBtBinvBt,
        FPnum *BtB,
        FPnum *BeTBeChol,
        FPnum *TransCtCinvCt,
        FPnum *CtCw,
        FPnum *B_plus_bias,
        int nthreads
    )

    int predict_X_old_content_based(
        FPnum *predicted, size_t n_predict,
        int m_new, int k,
        int row[],
        int col[],
        FPnum *U, int p,
        int U_row[], int U_col[], FPnum *U_sp, size_t nnz_U,
        size_t U_csr_p[], int U_csr_i[], FPnum *U_csr,
        FPnum *C, FPnum *C_bias,
        FPnum *Bm, FPnum *biasB,
        FPnum glob_mean,
        int nthreads
    )

    int predict_X_new_content_based(
        FPnum *predicted, size_t n_predict,
        int m_new, int n_new, int k,
        int row[], int col[],
        FPnum *U, int p,
        int U_row[], int U_col[], FPnum *U_sp, size_t nnz_U,
        size_t U_csr_p[], int U_csr_i[], FPnum *U_csr,
        FPnum *II, int q,
        int I_row[], int I_col[], FPnum *I_sp, size_t nnz_I,
        size_t I_csr_p[], int I_csr_i[], FPnum *I_csr,
        FPnum *C, FPnum *C_bias,
        FPnum *D, FPnum *D_bias,
        FPnum glob_mean,
        int nthreads
    )

    int factors_content_based_single(
        FPnum *a_vec, int k,
        FPnum *u_vec, int p,
        FPnum *u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
        FPnum *C, FPnum *C_bias
    )

    int fit_content_based_lbfgs(
        FPnum *biasA, FPnum *biasB,
        FPnum *C, FPnum *C_bias,
        FPnum *D, FPnum *D_bias,
        bint start_with_ALS, bint reset_values, int seed,
        FPnum *glob_mean,
        int m, int n, int k,
        int ixA[], int ixB[], FPnum *X, size_t nnz,
        FPnum *Xfull,
        FPnum *weight,
        bint user_bias, bint item_bias,
        bint add_intercepts,
        FPnum lam, FPnum *lam_unique,
        FPnum *U, int p,
        FPnum *II, int q,
        int U_row[], int U_col[], FPnum *U_sp, size_t nnz_U,
        int I_row[], int I_col[], FPnum *I_sp, size_t nnz_I,
        int n_corr_pairs, size_t maxiter,
        int nthreads, bint prefer_onepass,
        bint verbose, int print_every, bint handle_interrupt,
        int *niter, int *nfev,
        FPnum *Am, FPnum *Bm
    )

    int factors_collective_explicit_multiple(
        FPnum *A, FPnum *biasA, int m,
        FPnum *U, int m_u, int p,
        bint NA_as_zero_U, bint NA_as_zero_X,
        int U_row[], int U_col[], FPnum *U_sp, size_t nnz_U,
        size_t U_csr_p[], int U_csr_i[], FPnum *U_csr,
        FPnum *Ub, int m_ubin, int pbin,
        FPnum *C, FPnum *Cb,
        FPnum glob_mean, FPnum *biasB,
        FPnum *col_means,
        FPnum *X, int ixA[], int ixB[], size_t nnz,
        size_t *Xcsr_p, int *Xcsr_i, FPnum *Xcsr,
        FPnum *Xfull, int n,
        FPnum *weight,
        FPnum *B,
        int k, int k_user, int k_item, int k_main,
        FPnum lam, FPnum *lam_unique,
        FPnum w_main, FPnum w_user,
        FPnum *TransBtBinvBt,
        FPnum *BtB,
        FPnum *BeTBeChol,
        FPnum *TransCtCinvCt,
        FPnum *CtCw,
        FPnum *B_plus_bias,
        int nthreads
    )

    int factors_collective_implicit_multiple(
        FPnum *A, int m,
        FPnum *U, int m_u, int p,
        bint NA_as_zero_U,
        int U_row[], int U_col[], FPnum *U_sp, size_t nnz_U,
        size_t U_csr_p[], int U_csr_i[], FPnum *U_csr,
        FPnum *X, int ixA[], int ixB[], size_t nnz,
        size_t *Xcsr_p, int *Xcsr_i, FPnum *Xcsr,
        FPnum *B, int n,
        FPnum *C,
        FPnum *col_means,
        int k, int k_user, int k_item, int k_main,
        FPnum lam, FPnum alpha, FPnum w_main, FPnum w_user,
        FPnum w_main_multiplier,
        FPnum *BeTBe,
        FPnum *BtB,
        FPnum *BeTBeChol,
        int nthreads
    )

    int factors_offsets_explicit_multiple(
        FPnum *Am, FPnum *biasA,
        FPnum *A, int m,
        FPnum *U, int p,
        int U_row[], int U_col[], FPnum *U_sp, size_t nnz_U,
        size_t U_csr_p[], int U_csr_i[], FPnum *U_csr,
        FPnum *X, int ixA[], int ixB[], size_t nnz,
        size_t *Xcsr_p, int *Xcsr_i, FPnum *Xcsr,
        FPnum *Xfull, int n,
        FPnum *weight,
        FPnum *Bm, FPnum *C,
        FPnum *C_bias,
        FPnum glob_mean, FPnum *biasB,
        int k, int k_sec, int k_main,
        FPnum w_user,
        FPnum lam, FPnum *lam_unique, bint exact,
        FPnum *precomputedTransBtBinvBt,
        FPnum *precomputedBtB,
        FPnum *Bm_plus_bias,
        int nthreads
    )

    int factors_offsets_implicit_multiple(
        FPnum *Am, int m,
        FPnum *A,
        FPnum *U, int p,
        int U_row[], int U_col[], FPnum *U_sp, size_t nnz_U,
        size_t U_csr_p[], int U_csr_i[], FPnum *U_csr,
        FPnum *X, int ixA[], int ixB[], size_t nnz,
        size_t *Xcsr_p, int *Xcsr_i, FPnum *Xcsr,
        FPnum *Bm, FPnum *C,
        FPnum *C_bias,
        int k,
        FPnum lam, FPnum alpha,
        FPnum *precomputedBtB,
        int nthreads
    )


def call_fit_collective_explicit_lbfgs(
        np.ndarray[int, ndim=1] ixA,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[FPnum, ndim=1] X,
        np.ndarray[FPnum, ndim=1] W,
        np.ndarray[FPnum, ndim=2] Xfull,
        np.ndarray[FPnum, ndim=2] Wfull,
        np.ndarray[FPnum, ndim=2] U,
        np.ndarray[int, ndim=1] U_row,
        np.ndarray[int, ndim=1] U_col,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[FPnum, ndim=2] Ub,
        np.ndarray[FPnum, ndim=2] I,
        np.ndarray[int, ndim=1] I_row,
        np.ndarray[int, ndim=1] I_col,
        np.ndarray[FPnum, ndim=1] I_sp,
        np.ndarray[FPnum, ndim=2] Ib,
        int m, int n, int m_u, int n_i, int p, int q,
        int k=50, int k_user=0, int k_item=0, int k_main=0,
        FPnum w_main=1., FPnum w_user=1., FPnum w_item=1.,
        bint user_bias=1, bint item_bias=1,
        FPnum lam=1e2,
        np.ndarray[FPnum, ndim=1] lam_unique=np.empty(0, dtype=c_FPnum),
        bint verbose=1, int print_every=10,
        int n_corr_pairs=5, int maxiter=400,
        int nthreads=1, bint prefer_onepass=0,
        int seed=1, bint handle_interrupt=1
    ):

    cdef FPnum *ptr_Xfull = NULL
    cdef FPnum *ptr_weight = NULL
    cdef int *ptr_ixA = NULL
    cdef int *ptr_ixB = NULL
    cdef FPnum *ptr_X = NULL
    cdef size_t nnz = 0
    if Xfull.shape[0]:
        ptr_Xfull = &Xfull[0,0]
        if Wfull.shape[0]:
            ptr_weight = &Wfull[0,0]
    else:
        ptr_ixA = &ixA[0]
        ptr_ixB = &ixB[0]
        ptr_X = &X[0]
        nnz = X.shape[0]
        if W.shape[0]:
            ptr_weight = &W[0]

    cdef FPnum *ptr_U = NULL
    cdef int *ptr_U_row = NULL
    cdef int *ptr_U_col = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    cdef np.ndarray[FPnum, ndim=1] U_colmeans = np.empty(p, dtype=c_FPnum)
    cdef FPnum *ptr_U_colmeans = NULL
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]
    if U.shape[0] or U_sp.shape[0]:
        ptr_U_colmeans = &U_colmeans[0]

    cdef FPnum *ptr_I = NULL
    cdef int *ptr_I_row = NULL
    cdef int *ptr_I_col = NULL
    cdef FPnum *ptr_I_sp = NULL
    cdef size_t nnz_I = 0
    cdef np.ndarray[FPnum, ndim=1] I_colmeans = np.empty(q, dtype=c_FPnum)
    cdef FPnum *ptr_I_colmeans = NULL
    if I.shape[0]:
        ptr_I = &I[0,0]
    elif I_sp.shape[0]:
        ptr_I_row = &I_row[0]
        ptr_I_col = &I_col[0]
        ptr_I_sp = &I_sp[0]
        nnz_I = I_sp.shape[0]
    if I.shape[0] or I_sp.shape[0]:
        ptr_I_colmeans = &I_colmeans[0]

    cdef FPnum *ptr_Ub = NULL
    cdef int m_ubin = 0
    cdef int pbin = 0
    if Ub.shape[0]:
        ptr_Ub = &Ub[0,0]
        m_ubin = Ub.shape[0]
        pbin = Ub.shape[1]

    cdef FPnum *ptr_Ib = NULL
    cdef int n_ibin = 0
    cdef int qbin = 0
    if Ib.shape[0]:
        ptr_Ib = &Ib[0,0]
        n_ibin = Ib.shape[0]
        qbin = Ib.shape[1]

    cdef FPnum *ptr_lam_unique = NULL
    if lam_unique.shape[0]:
        ptr_lam_unique = &lam_unique[0]

    cdef size_t nvars = <size_t>max(m, m_u, m_ubin) * <size_t>(k_user+k+k_main) \
                        + <size_t>max(n, n_i, n_ibin) * <size_t>(k_item+k+k_main)
    if user_bias:
        nvars += max(m, m_u, m_ubin)
    if item_bias:
        nvars += max(n, n_i, n_ibin)
    if U.shape[0] or U_sp.shape[0]:
        nvars += <size_t>p * <size_t>(k_user + k)
    if I.shape[0] or I_sp.shape[0]:
        nvars += <size_t>q * <size_t>(k_item + k)
    if Ub.shape[0]:
        nvars += <size_t>pbin * <size_t>(k_user + k)
    if Ib.shape[0]:
        nvars += <size_t>qbin * <size_t>(k_item + k)
    rs = np.random.Generator(np.random.MT19937(seed = seed))
    cdef np.ndarray[FPnum, ndim=1] values = rs.standard_normal(size = nvars, dtype = c_FPnum)

    cdef np.ndarray[FPnum, ndim=2] B_plus_bias = np.empty((0,0), dtype=c_FPnum)
    cdef FPnum *ptr_B_plus_bias = NULL
    if user_bias:
        B_plus_bias = np.empty((max(n, n_i, n_ibin), k_item+k+k_main+1), dtype=c_FPnum)
        ptr_B_plus_bias = &B_plus_bias[0,0]

    cdef FPnum glob_mean
    cdef int niter, nfev
    cdef int retval = fit_collective_explicit_lbfgs_internal(
        &values[0], 0,
        &glob_mean,
        ptr_U_colmeans, ptr_I_colmeans,
        m, n, k,
        ptr_ixA, ptr_ixB, ptr_X, nnz,
        ptr_Xfull,
        ptr_weight,
        user_bias, item_bias,
        lam, ptr_lam_unique,
        ptr_U, m_u, p,
        ptr_I, n_i, q,
        ptr_Ub, m_ubin, pbin,
        ptr_Ib, n_ibin, qbin,
        ptr_U_row, ptr_U_col, ptr_U_sp, nnz_U,
        ptr_I_row, ptr_I_col, ptr_I_sp, nnz_I,
        k_main, k_user, k_item,
        w_main, w_user, w_item,
        n_corr_pairs, maxiter, 1,
        nthreads, prefer_onepass,
        verbose, print_every, handle_interrupt,
        &niter, &nfev,
        ptr_B_plus_bias
    )

    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return glob_mean, U_colmeans, I_colmeans, values, niter, nfev, B_plus_bias

def unpack_values_lbfgs_collective(
        np.ndarray[FPnum, ndim=1] values,
        bint user_bias, bint item_bias,
        size_t k, size_t k_user, size_t k_item, size_t k_main,
        size_t m, size_t n, size_t p, size_t q,
        size_t pbin, size_t qbin,
        size_t m_u, size_t n_i, size_t m_ubin, size_t n_ibin
    ):

    cdef np.ndarray[FPnum, ndim=1] biasA, biasB
    cdef np.ndarray[FPnum, ndim=2] A, B, C, Cbin, D, Dbin

    cdef size_t edge = 0
    if user_bias:
        biasA = values[:max([m, m_u, m_ubin])]
        edge += max([m, m_u, m_ubin])
    else:
        biasA = np.empty(0, dtype=c_FPnum)
    if item_bias:
        biasB = values[edge:edge + max([n, n_i, n_ibin])]
        edge += max([n, n_i, n_ibin])
    else:
        biasB = np.empty(0, dtype=c_FPnum)
    m = <size_t>max([m, m_u, m_ubin])
    n = <size_t>max([n, n_i, n_ibin])
    A = values[edge:edge + m*(k_user+k+k_main)].reshape((m, k_user+k+k_main))
    edge += m*(k_user+k+k_main)
    B = values[edge:edge + n*(k_item+k+k_main)].reshape((n, k_item+k+k_main))
    edge += n*(k_item+k+k_main)
    if p > 0:
        C = values[edge:edge + p*(k_user+k)].reshape((p, k_user+k))
        edge += p*(k_user+k)
    else:
        C = np.empty((0,0), dtype=c_FPnum)
    if pbin > 0:
        Cbin = values[edge:edge + pbin*(k_user+k)].reshape((pbin, k_user+k))
        edge += pbin*(k_user+k)
    else:
        Cbin = np.empty((0,0), dtype=c_FPnum)
    if q > 0:
        D = values[edge:edge + q*(k_item+k)].reshape((q, k_item+k))
        edge += q*(k_item+k)
    else:
        D = np.empty((0,0), dtype=c_FPnum)
    if qbin > 0:
        Dbin = values[edge:edge + qbin*(k_item+k)].reshape((qbin, k_item+k))
        edge += qbin*(k_item+k)
    else:
        Dbin = np.empty((0,0), dtype=c_FPnum)

    return biasA, biasB, A, B, C, Cbin, D, Dbin

def call_fit_offsets_explicit_lbfgs(
        np.ndarray[int, ndim=1] ixA,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[FPnum, ndim=1] X,
        np.ndarray[FPnum, ndim=1] W,
        np.ndarray[FPnum, ndim=2] Xfull,
        np.ndarray[FPnum, ndim=2] Wfull,
        np.ndarray[FPnum, ndim=2] U,
        np.ndarray[int, ndim=1] U_row,
        np.ndarray[int, ndim=1] U_col,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[FPnum, ndim=2] I,
        np.ndarray[int, ndim=1] I_row,
        np.ndarray[int, ndim=1] I_col,
        np.ndarray[FPnum, ndim=1] I_sp,
        int m, int n, int p, int q,
        int k=50, int k_sec=0, int k_main=0,
        FPnum w_user=1., FPnum w_item=1.,
        bint user_bias=1, bint item_bias=1,
        bint add_intercepts=1,
        FPnum lam=1e2,
        np.ndarray[FPnum, ndim=1] lam_unique=np.empty(0, dtype=c_FPnum),
        bint verbose=1, int print_every=10,
        int n_corr_pairs=5, int maxiter=400,
        int nthreads=1, bint prefer_onepass=0,
        int seed=1, bint handle_interrupt=1
    ):
    
    cdef FPnum *ptr_Xfull = NULL
    cdef FPnum *ptr_weight = NULL
    cdef int *ptr_ixA = NULL
    cdef int *ptr_ixB = NULL
    cdef FPnum *ptr_X = NULL
    cdef size_t nnz = 0
    if Xfull.shape[0]:
        ptr_Xfull = &Xfull[0,0]
        if Wfull.shape[0]:
            ptr_weight = &Wfull[0,0]
    else:
        ptr_ixA = &ixA[0]
        ptr_ixB = &ixB[0]
        ptr_X = &X[0]
        nnz = X.shape[0]
        if W.shape[0]:
            ptr_weight = &W[0]

    cdef FPnum *ptr_U = NULL
    cdef int *ptr_U_row = NULL
    cdef int *ptr_U_col = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]

    cdef FPnum *ptr_I = NULL
    cdef int *ptr_I_row = NULL
    cdef int *ptr_I_col = NULL
    cdef FPnum *ptr_I_sp = NULL
    cdef size_t nnz_I = 0
    if I.shape[0]:
        ptr_I = &I[0,0]
    elif I_sp.shape[0]:
        ptr_I_row = &I_row[0]
        ptr_I_col = &I_col[0]
        ptr_I_sp = &I_sp[0]
        nnz_I = I_sp.shape[0]

    cdef FPnum *ptr_lam_unique = NULL
    if lam_unique.shape[0]:
        ptr_lam_unique = &lam_unique[0]

    cdef size_t nvars = <size_t>m * <size_t>(k+k_main) \
                        + <size_t>n * <size_t>(k+k_main) \
                        + <size_t>p * <size_t>(k_sec+k) \
                        + <size_t>q * <size_t>(k_sec+k)
    if user_bias:
        nvars += m
    if item_bias:
        nvars += n
    if not (U.shape[0] or U_sp.shape[0]):
        nvars += <size_t>m * <size_t>k_sec
    if not (I.shape[0] or I_sp.shape[0]):
        nvars += <size_t>n * <size_t>k_sec
    if (add_intercepts) and (U.shape[0] or U_sp.shape[0]) and (k_sec or k):
        nvars += <size_t>(k_sec + k)
    if (add_intercepts) and (I.shape[0] or I_sp.shape[0]) and (k_sec or k):
        nvars += <size_t>(k_sec + k)

    cdef np.ndarray[FPnum, ndim=2] Am = np.empty((m, k_sec+k+k_main), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] Bm = np.empty((n, k_sec+k+k_main), dtype=c_FPnum)

    rs = np.random.Generator(np.random.MT19937(seed = seed))
    cdef np.ndarray[FPnum, ndim=1] values = rs.standard_normal(size = nvars, dtype = c_FPnum)

    cdef np.ndarray[FPnum, ndim=2] Bm_plus_bias = np.empty((0,0), dtype=c_FPnum)
    cdef FPnum *ptr_Bm_plus_bias = NULL
    if user_bias:
        Bm_plus_bias = np.empty((n, k_sec+k+k_main+1), dtype=c_FPnum)
        ptr_Bm_plus_bias = &Bm_plus_bias[0,0]

    cdef FPnum glob_mean
    cdef int niter, nfev
    cdef int retval = fit_offsets_explicit_lbfgs(
        &values[0], 0,
        &glob_mean,
        m, n, k,
        ptr_ixA, ptr_ixB, ptr_X, nnz,
        ptr_Xfull,
        ptr_weight,
        user_bias, item_bias,
        add_intercepts,
        lam, ptr_lam_unique,
        ptr_U, p,
        ptr_I, q,
        ptr_U_row, ptr_U_col, ptr_U_sp, nnz_U,
        ptr_I_row, ptr_I_col, ptr_I_sp, nnz_I,
        k_main, k_sec,
        w_user, w_item,
        n_corr_pairs, maxiter, 1,
        nthreads, prefer_onepass,
        verbose, print_every, handle_interrupt,
        &niter, &nfev,
        &Am[0,0], &Bm[0,0],
        ptr_Bm_plus_bias
    )

    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return glob_mean, Am, Bm, values, niter, nfev, Bm_plus_bias

def unpack_values_lbfgs_offsets(
        np.ndarray[FPnum, ndim=1] values,
        bint user_bias, bint item_bias,
        size_t k, size_t k_sec, size_t k_main,
        size_t m, size_t n, size_t p, size_t q,
        bint add_intercepts
    ):

    cdef np.ndarray[FPnum, ndim=1] biasA, biasB, C_bias, D_bias
    cdef np.ndarray[FPnum, ndim=2] A, B, C,  D

    cdef size_t edge = 0
    if user_bias:
        biasA = values[:m]
        edge += m
    else:
        biasA = np.empty(0, dtype=c_FPnum)
    if item_bias:
        biasB = values[edge:edge + n]
        edge += n
    else:
        biasB = np.empty(0, dtype=c_FPnum)
    if p > 0:
        A = values[edge:edge + m*(k+k_main)].reshape((m, k+k_main))
        edge += m*(k+k_main)
    else:
        A = values[edge:edge + m*(k_sec+k+k_main)].reshape((m, k_sec+k+k_main))
        edge += m*(k_sec+k+k_main)
    if q > 0:
        B = values[edge:edge + n*(k+k_main)].reshape((n, k+k_main))
        edge += n*(k+k_main)
    else:
        B = values[edge:edge + n*(k_sec+k+k_main)].reshape((n, k_sec+k+k_main))
        edge += n*(k_sec+k+k_main)
    if p > 0:
        C = values[edge:edge + p*(k_sec+k)].reshape((p, k_sec+k))
        edge += p*(k_sec+k)
    else:
        C = np.empty((0,0), dtype=c_FPnum)
    if (add_intercepts) and (p > 0):
        C_bias = values[edge:edge + (k_sec+k)]
        edge += (k_sec+k)
    else:
        C_bias = np.empty(0, dtype=c_FPnum)
    if q > 0:
        D = values[edge:edge + q*(k_sec+k)].reshape((q, k_sec+k))
        edge += q*(k_sec+k)
    else:
        D = np.empty((0,0), dtype=c_FPnum)
    if (add_intercepts) and (q > 0):
        D_bias = values[edge:edge + (k_sec+k)]
        edge += (k_sec+k)
    else:
        D_bias = np.empty(0, dtype=c_FPnum)

    return biasA, biasB, A, B, C, D, C_bias, D_bias

def call_fit_collective_explicit_als(
        np.ndarray[int, ndim=1] ixA,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[FPnum, ndim=1] X,
        np.ndarray[FPnum, ndim=1] W,
        np.ndarray[FPnum, ndim=2] Xfull,
        np.ndarray[FPnum, ndim=2] Wfull,
        np.ndarray[FPnum, ndim=2] U,
        np.ndarray[int, ndim=1] U_row,
        np.ndarray[int, ndim=1] U_col,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[FPnum, ndim=2] I,
        np.ndarray[int, ndim=1] I_row,
        np.ndarray[int, ndim=1] I_col,
        np.ndarray[FPnum, ndim=1] I_sp,
        bint NA_as_zero_X, bint NA_as_zero_U, bint NA_as_zero_I,
        int m, int n, int m_u, int n_i, int p, int q,
        int k=50, int k_user=0, int k_item=0, int k_main=0,
        FPnum w_main=1., FPnum w_user=1., FPnum w_item=1.,
        bint user_bias=1, bint item_bias=1,
        FPnum lam=1e2,
        np.ndarray[FPnum, ndim=1] lam_unique=np.empty(0, dtype=c_FPnum),
        bint verbose=1, int nthreads=1,
        bint use_cg = 0, int max_cg_steps=3,
        bint finalize_chol=0,
        int seed=1, int niter=5, bint handle_interrupt=1,
        bint precompute_for_predictions = 1
    ):

    cdef FPnum *ptr_Xfull = NULL
    cdef FPnum *ptr_weight = NULL
    cdef int *ptr_ixA = NULL
    cdef int *ptr_ixB = NULL
    cdef FPnum *ptr_X = NULL
    cdef size_t nnz = 0
    if Xfull.shape[0]:
        ptr_Xfull = &Xfull[0,0]
        if Wfull.shape[0]:
            ptr_weight = &Wfull[0,0]
    else:
        ptr_ixA = &ixA[0]
        ptr_ixB = &ixB[0]
        ptr_X = &X[0]
        nnz = X.shape[0]
        if W.shape[0]:
            ptr_weight = &W[0]

    cdef FPnum *ptr_U = NULL
    cdef int *ptr_U_row = NULL
    cdef int *ptr_U_col = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    cdef np.ndarray[FPnum, ndim=1] U_colmeans = np.empty(0, dtype=c_FPnum)
    cdef FPnum *ptr_U_colmeans = NULL
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]
    if U.shape[0] or (U_sp.shape[0] and not NA_as_zero_U):
        U_colmeans = np.empty(p, dtype=c_FPnum)
        ptr_U_colmeans = &U_colmeans[0]

    cdef FPnum *ptr_I = NULL
    cdef int *ptr_I_row = NULL
    cdef int *ptr_I_col = NULL
    cdef FPnum *ptr_I_sp = NULL
    cdef size_t nnz_I = 0
    cdef np.ndarray[FPnum, ndim=1] I_colmeans = np.empty(0, dtype=c_FPnum)
    cdef FPnum *ptr_I_colmeans = NULL
    if I.shape[0]:
        ptr_I = &I[0,0]
    elif I_sp.shape[0]:
        ptr_I_row = &I_row[0]
        ptr_I_col = &I_col[0]
        ptr_I_sp = &I_sp[0]
        nnz_I = I_sp.shape[0]
    if I.shape[0] or (I_sp.shape[0] and not NA_as_zero_I):
        I_colmeans = np.empty(q, dtype=c_FPnum)
        ptr_I_colmeans = &I_colmeans[0]

    cdef FPnum *ptr_lam_unique = NULL
    if lam_unique.shape[0]:
        ptr_lam_unique = &lam_unique[0]

    cdef np.ndarray[FPnum, ndim=1] biasA = np.zeros(0, dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=1] biasB = np.zeros(0, dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] A = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] B = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] C = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] D = np.zeros((0,0), dtype=c_FPnum)
    cdef FPnum *ptr_biasA = NULL
    cdef FPnum *ptr_biasB = NULL
    cdef FPnum *ptr_A = NULL
    cdef FPnum *ptr_B = NULL
    cdef FPnum *ptr_C = NULL
    cdef FPnum *ptr_D = NULL

    cdef size_t sizeA = <size_t>max(m, m_u) * <size_t>(k_user+k+k_main)
    cdef size_t sizeB = <size_t>max(n, n_i) * <size_t>(k_item+k+k_main)
    if (sizeA == 0) or (sizeB == 0):
        raise ValueError("Model cannot have empty 'A' or 'B' matrices.")

    if user_bias:
        biasA = np.zeros(max(m, m_u), dtype=c_FPnum)
        ptr_biasA = &biasA[0]
    if item_bias:
        biasB = np.zeros(max(n, n_i), dtype=c_FPnum)
        ptr_biasB = &biasB[0]

    rs = np.random.Generator(np.random.MT19937(seed = seed))
    A = rs.standard_normal(size = (max(m, m_u), k_user+k+k_main), dtype = c_FPnum)
    B = rs.standard_normal(size = (max(n, n_i), k_item+k+k_main), dtype = c_FPnum)
    ptr_A = &A[0,0]
    ptr_B = &B[0,0]
    if p:
        C = rs.standard_normal(size = (p, k_user + k), dtype = c_FPnum)
        if C.shape[0]:
            ptr_C = &C[0,0]
        else:
            raise ValueError("Unexpected error.")
    if q:
        D = rs.standard_normal(size = (q, k_item + k), dtype = c_FPnum)
        if D.shape[0]:
            ptr_D = &D[0,0]
        else:
            raise ValueError("Unexpected error.")

    cdef np.ndarray[FPnum, ndim=2] B_plus_bias = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] BtB = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] TransBtBinvBt = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] BeTBeChol = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] TransCtCinvCt = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] CtCw = np.zeros((0,0), dtype=c_FPnum)
    cdef FPnum *ptr_B_plus_bias = NULL
    cdef FPnum *ptr_BtB = NULL
    cdef FPnum *ptr_TransBtBinvBt = NULL
    cdef FPnum *ptr_BeTBeChol = NULL
    cdef FPnum *ptr_TransCtCinvCt = NULL
    cdef FPnum *ptr_CtCw = NULL

    if precompute_for_predictions:
        if user_bias:
            B_plus_bias = np.empty((B.shape[0],B.shape[1]), dtype=c_FPnum)
            ptr_B_plus_bias = &B_plus_bias[0,0]
        BtB = np.empty((k+k_main, k+k_main), dtype=c_FPnum)
        ptr_BtB = &BtB[0,0]
        TransBtBinvBt = np.empty((B.shape[0], k+k_main), dtype=c_FPnum)
        ptr_TransBtBinvBt = &TransBtBinvBt[0,0]
        if p:
            BeTBeChol = np.empty((B.shape[1], B.shape[1]), dtype=c_FPnum)
            ptr_BeTBeChol = &BeTBeChol[0,0]
            CtCw = np.empty((k_user+k, k_user+k), dtype=c_FPnum)
            if CtCw.shape[0]:
                ptr_CtCw = &CtCw[0,0]
            else:
                raise ValueError("Unexpected error.")
            TransCtCinvCt = np.empty((p, k_user+k), dtype=c_FPnum)
            if TransCtCinvCt.shape[0]:
                ptr_TransCtCinvCt = &TransCtCinvCt[0,0]
            else:
                raise ValueError("Unexpected error.")

    cdef FPnum glob_mean = 0

    cdef int retval = fit_collective_explicit_als(
        ptr_biasA, ptr_biasB,
        ptr_A, ptr_B, ptr_C, ptr_D,
        0, 0,
        &glob_mean,
        ptr_U_colmeans, ptr_I_colmeans,
        m, n, k,
        ptr_ixA, ptr_ixB, ptr_X, nnz,
        ptr_Xfull,
        ptr_weight,
        user_bias, item_bias,
        lam, ptr_lam_unique,
        ptr_U, m_u, p,
        ptr_I, n_i, q,
        ptr_U_row, ptr_U_col, ptr_U_sp, nnz_U,
        ptr_I_row, ptr_I_col, ptr_I_sp, nnz_I,
        NA_as_zero_X, NA_as_zero_U, NA_as_zero_I,
        k_main, k_user, k_item,
        w_main, w_user, w_item,
        niter, nthreads, verbose, handle_interrupt,
        use_cg, max_cg_steps, finalize_chol,
        precompute_for_predictions,
        ptr_B_plus_bias,
        ptr_BtB,
        ptr_TransBtBinvBt,
        ptr_BeTBeChol,
        ptr_TransCtCinvCt,
        ptr_CtCw
    )

    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return biasA, biasB, A, B, C, D, \
           glob_mean, U_colmeans, I_colmeans, \
           B_plus_bias, BtB, TransBtBinvBt, BeTBeChol, TransCtCinvCt, CtCw


def call_fit_collective_implicit_als(
        np.ndarray[int, ndim=1] ixA,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[FPnum, ndim=1] X,
        np.ndarray[FPnum, ndim=2] U,
        np.ndarray[int, ndim=1] U_row,
        np.ndarray[int, ndim=1] U_col,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[FPnum, ndim=2] I,
        np.ndarray[int, ndim=1] I_row,
        np.ndarray[int, ndim=1] I_col,
        np.ndarray[FPnum, ndim=1] I_sp,
        bint NA_as_zero_U, bint NA_as_zero_I,
        int m, int n, int m_u, int n_i, int p, int q,
        int k=50, int k_user=0, int k_item=0, int k_main=0,
        FPnum w_main=1., FPnum w_user=1., FPnum w_item=1.,
        FPnum lam=1e2, FPnum alpha=40., bint adjust_weight=1,
        np.ndarray[FPnum, ndim=1] lam_unique=np.empty(0, dtype=c_FPnum),
        bint verbose=1, int niter=5,
        int nthreads=1, bint use_cg=0,
        int max_cg_steps=3, bint finalize_chol=0,
        int seed=1, init="normal", bint handle_interrupt=1,
        bint precompute_for_predictions = 1
    ):
    
    cdef FPnum *ptr_U = NULL
    cdef int *ptr_U_row = NULL
    cdef int *ptr_U_col = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    cdef np.ndarray[FPnum, ndim=1] U_colmeans = np.empty(0, dtype=c_FPnum)
    cdef FPnum *ptr_U_colmeans = NULL
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]
    if U.shape[0] or (U_sp.shape[0] and not NA_as_zero_U):
        U_colmeans = np.empty(p, dtype=c_FPnum)
        ptr_U_colmeans = &U_colmeans[0]

    if X.shape[0] == 0:
        raise ValueError("Input data has no non-zero values.")

    cdef FPnum *ptr_I = NULL
    cdef int *ptr_I_row = NULL
    cdef int *ptr_I_col = NULL
    cdef FPnum *ptr_I_sp = NULL
    cdef size_t nnz_I = 0
    cdef np.ndarray[FPnum, ndim=1] I_colmeans = np.empty(0, dtype=c_FPnum)
    cdef FPnum *ptr_I_colmeans = NULL
    if I.shape[0]:
        ptr_I = &I[0,0]
    elif I_sp.shape[0]:
        ptr_I_row = &I_row[0]
        ptr_I_col = &I_col[0]
        ptr_I_sp = &I_sp[0]
        nnz_I = I_sp.shape[0]
    if I.shape[0] or (I_sp.shape[0] and not NA_as_zero_I):
        I_colmeans = np.empty(q, dtype=c_FPnum)
        ptr_I_colmeans = &I_colmeans[0]

    cdef FPnum *ptr_lam_unique = NULL
    if lam_unique.shape[0]:
        ptr_lam_unique = &lam_unique[0]


    cdef size_t sizeA = <size_t>max(m, m_u) * <size_t>(k_user+k+k_main)
    cdef size_t sizeB = <size_t>max(n, n_i) * <size_t>(k_item+k+k_main)
    if (sizeA == 0) or (sizeB == 0):
        raise ValueError("Error: model must have variables to optimize for both A and B.")
    cdef size_t sizeC = 0
    cdef size_t sizeD = 0
    if U.shape[0] or U_sp.shape[0]:
        sizeC = <size_t>p * <size_t>(k_user + k)
    if I.shape[0] or I_sp.shape[0]:
        sizeD = <size_t>q * <size_t>(k_item + k)
    
    rs = np.random.Generator(np.random.MT19937(seed = seed))

    cdef FPnum *ptr_A = NULL
    cdef FPnum *ptr_B = NULL
    cdef FPnum *ptr_C = NULL
    cdef FPnum *ptr_D = NULL
    cdef np.ndarray[FPnum, ndim=2] A = np.zeros((0,0), dtype = c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] B = np.zeros((0,0), dtype = c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] C = np.zeros((0,0), dtype = c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] D = np.zeros((0,0), dtype = c_FPnum)

    if init == "normal":
        A = rs.standard_normal(size = (max(m, m_u), (k_user+k+k_main)), dtype = c_FPnum)
        B = rs.standard_normal(size = (max(n, n_i), (k_item+k+k_main)), dtype = c_FPnum)
        if sizeC:
            C = rs.standard_normal(size = (p, k_user + k), dtype = c_FPnum)
        if sizeD:
            C = rs.standard_normal(size = (q, k_item + k), dtype = c_FPnum)
    else:
        A = rs.random(size = (max(m, m_u), (k_user+k+k_main)), dtype = c_FPnum)
        B = rs.random(size = (max(n, n_i), (k_item+k+k_main)), dtype = c_FPnum)
        if sizeC:
            C = rs.random(size = (p, k_user + k), dtype = c_FPnum)
        if sizeD:
            C = rs.random(size = (q, k_item + k), dtype = c_FPnum)
    
        if init == "gamma":
            A[:] = -np.log(A.clip(min=1e-6, max=20.))
            B[:] = -np.log(B.clip(min=1e-6, max=20.))
            if sizeC:
                C[:] = -np.log(C.clip(min=1e-6, max=20.))
            if sizeD:
                D[:] = -np.log(D.clip(min=1e-6, max=20.))
        elif init != "uniform":
            A[:] -= 0.5
            B[:] -= 0.5
            if sizeC:
                C[:] -= 0.5
            if sizeD:
                D[:] -= 0.5

    ptr_A = &A[0,0]
    ptr_B = &B[0,0]
    if sizeC:
        ptr_C = &C[0,0]
    if sizeD:
        ptr_D = &D[0,0]

    cdef np.ndarray[FPnum, ndim=2] precomputedBtB = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] precomputedBeTBe = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] precomputedBeTBeChol = np.zeros((0,0), dtype=c_FPnum)
    cdef FPnum *ptr_BtB = NULL
    cdef FPnum *ptr_BeTBe = NULL
    cdef FPnum *ptr_BeTBeChol = NULL
    if precompute_for_predictions:
        precomputedBtB = np.empty((k+k_main, k+k_main), dtype=c_FPnum)
        ptr_BtB = &precomputedBtB[0,0]
        if U.shape[0] or U_sp.shape[0]:
            precomputedBeTBe = np.empty((k_user+k+k_main, k_user+k+k_main), dtype=c_FPnum)
            precomputedBeTBeChol = np.empty((k_user+k+k_main, k_user+k+k_main), dtype=c_FPnum)
            ptr_BeTBe = &precomputedBeTBe[0,0]
            ptr_BeTBeChol = &precomputedBeTBeChol[0,0]




    cdef FPnum w_main_multiplier = 1.

    cdef int retval = fit_collective_implicit_als(
        ptr_A, ptr_B, ptr_C, ptr_D,
        0, 0,
        ptr_U_colmeans, ptr_I_colmeans,
        m, n, k,
        &ixA[0], &ixB[0], &X[0], X.shape[0],
        lam, ptr_lam_unique,
        ptr_U, m_u, p,
        ptr_I, n_i, q,
        ptr_U_row, ptr_U_col, ptr_U_sp, nnz_U,
        ptr_I_row, ptr_I_col, ptr_I_sp, nnz_I,
        NA_as_zero_U, NA_as_zero_I,
        k_main, k_user, k_item,
        w_main, w_user, w_item,
        &w_main_multiplier,
        alpha, adjust_weight,
        niter, nthreads, verbose, handle_interrupt,
        use_cg, max_cg_steps, finalize_chol,
        precompute_for_predictions,
        ptr_BtB, ptr_BeTBe, ptr_BeTBeChol
    )

    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return A, B, C, D, \
           U_colmeans, I_colmeans, w_main_multiplier, \
           precomputedBtB, precomputedBeTBe, precomputedBeTBeChol

def call_fit_offsets_explicit_als(
        np.ndarray[int, ndim=1] ixA,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[FPnum, ndim=1] X,
        np.ndarray[FPnum, ndim=1] W,
        np.ndarray[FPnum, ndim=2] Xfull,
        np.ndarray[FPnum, ndim=2] Wfull,
        np.ndarray[FPnum, ndim=2] U,
        np.ndarray[FPnum, ndim=2] I,
        bint NA_as_zero_X,
        int m, int n, int p, int q,
        int k=50,
        bint user_bias=1, bint item_bias=1,
        bint add_intercepts=1,
        FPnum lam=1e2,
        bint verbose=1, int nthreads=1,
        bint use_cg=0, int max_cg_steps=3,
        bint finalize_chol=0,
        int seed=1, int niter=5, bint handle_interrupt=1,
        bint precompute_for_predictions=1
    ):
    if k <= 0:
        raise ValueError("'k' must be a positive number.")
    if min(m,n) <= 0:
        raise ValueError("'X' must have positive dimensions.")

    cdef FPnum *ptr_Xfull = NULL
    cdef FPnum *ptr_weight = NULL
    cdef int *ptr_ixA = NULL
    cdef int *ptr_ixB = NULL
    cdef FPnum *ptr_X = NULL
    cdef size_t nnz = 0
    if Xfull.shape[0]:
        ptr_Xfull = &Xfull[0,0]
        if Wfull.shape[0]:
            ptr_weight = &Wfull[0,0]
    else:
        ptr_ixA = &ixA[0]
        ptr_ixB = &ixB[0]
        ptr_X = &X[0]
        nnz = X.shape[0]
        if W.shape[0]:
            ptr_weight = &W[0]

    cdef FPnum *ptr_U = NULL
    if U.shape[0]:
        ptr_U = &U[0,0]

    cdef FPnum *ptr_I = NULL
    if I.shape[0]:
        ptr_I = &I[0,0]

    cdef np.ndarray[FPnum, ndim=2] Am = np.empty((m, k), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] Bm = np.empty((n, k), dtype=c_FPnum)
    cdef FPnum *ptr_Am = &Am[0,0]
    cdef FPnum *ptr_Bm = &Bm[0,0]

    cdef np.ndarray[FPnum, ndim=2] Bm_plus_bias = np.empty((0,0), dtype=c_FPnum)
    cdef FPnum *ptr_Bm_plus_bias = NULL
    if user_bias:
        Bm_plus_bias = np.empty((n, k+1), dtype=c_FPnum)
        ptr_Bm_plus_bias = &Bm_plus_bias[0,0]

    rs = np.random.Generator(np.random.MT19937(seed = seed))
    cdef np.ndarray[FPnum, ndim=1] biasA = np.zeros(0, dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=1] biasB = np.zeros(0, dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] A = rs.standard_normal(size=(m,k), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] B = rs.standard_normal(size=(n,k), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] C = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] D = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=1] C_bias = np.zeros(0, dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=1] D_bias = np.zeros(0, dtype=c_FPnum)

    cdef FPnum *ptr_biasA = NULL
    cdef FPnum *ptr_biasB = NULL
    cdef FPnum *ptr_A = &A[0,0]
    cdef FPnum *ptr_B = &B[0,0]
    cdef FPnum *ptr_C = NULL
    cdef FPnum *ptr_D = NULL
    cdef FPnum *ptr_C_bias = NULL
    cdef FPnum *ptr_D_bias = NULL
    if p:
        C = np.empty((p,k), dtype=c_FPnum)
        if C.shape[0]:
            ptr_C = &C[0,0]
        if add_intercepts:
            C_bias = np.empty(k, dtype=c_FPnum)
            if C_bias.shape[0]:
                ptr_C_bias = &C_bias[0]
    if q:
        D = np.empty((q,k), dtype=c_FPnum)
        if D.shape[0]:
            ptr_D = &D[0,0]
        if add_intercepts:
            D_bias = np.empty(k, dtype=c_FPnum)
            if D_bias.shape[0]:
                ptr_D_bias = &D_bias[0]

    if user_bias:
        biasA = np.empty(k, dtype=c_FPnum)
        ptr_biasA = &biasA[0]
    if item_bias:
        biasB = np.empty(k, dtype=c_FPnum)
        ptr_biasB = &biasB[0]

    cdef np.ndarray[FPnum, ndim=2] BtB = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] TransBtBinvBt = np.zeros((0,0), dtype=c_FPnum)
    cdef FPnum *ptr_BtB = NULL
    cdef FPnum *ptr_TransBtBinvBt = NULL
    if precompute_for_predictions:
        BtB = np.empty((k,k), dtype=c_FPnum)
        ptr_BtB = &BtB[0,0]
        TransBtBinvBt = np.empty((n,k), dtype=c_FPnum)
        ptr_TransBtBinvBt = &TransBtBinvBt[0,0]

    cdef FPnum glob_mean = 0

    cdef int retval = fit_offsets_als(
        ptr_biasA, ptr_biasB,
        ptr_A, ptr_B,
        ptr_C, ptr_C_bias,
        ptr_D, ptr_D_bias,
        0, 0,
        &glob_mean,
        m, n, k,
        ptr_ixA, ptr_ixB, ptr_X, nnz,
        ptr_Xfull,
        ptr_weight,
        user_bias, item_bias, add_intercepts,
        lam,
        ptr_U, p,
        ptr_I, q,
        0, NA_as_zero_X, 0.,
        niter,
        nthreads,
        use_cg, max_cg_steps, finalize_chol,
        verbose, handle_interrupt,
        precompute_for_predictions,
        ptr_Am, ptr_Bm,
        ptr_Bm_plus_bias,
        ptr_BtB,
        ptr_TransBtBinvBt
    )

    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")
    elif retval == 2:
        raise ValueError("Invalid parameter combination.")

    return biasA, biasB, A, B, C, D, C_bias, D_bias, \
           Am, Bm, glob_mean, \
           Bm_plus_bias, BtB, TransBtBinvBt

def call_fit_offsets_implicit_als(
        np.ndarray[int, ndim=1] ixA,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[FPnum, ndim=1] X,
        np.ndarray[FPnum, ndim=2] U,
        np.ndarray[FPnum, ndim=2] I,
        int m, int n, int p, int q,
        int k=50, bint add_intercepts=1,
        FPnum lam=1e2, FPnum alpha=40.,
        bint verbose=1, int nthreads=1,
        bint use_cg=0, int max_cg_steps=3,
        bint finalize_chol=0,
        bint adjust_weight = 1,
        int seed=1, int niter=5, bint handle_interrupt=1,
        bint precompute_for_predictions=1
    ):
    if k <= 0:
        raise ValueError("'k' must be a positive integer.")
    if min(m,n) <= 0:
        raise ValueError("'X' must have positive dimensions.")
    cdef int *ptr_ixA = NULL
    cdef int *ptr_ixB = NULL
    cdef FPnum *ptr_X = NULL
    cdef size_t nnz = 0
    ptr_ixA = &ixA[0]
    ptr_ixB = &ixB[0]
    ptr_X = &X[0]
    nnz = X.shape[0]

    cdef FPnum *ptr_U = NULL
    if U.shape[0]:
        ptr_U = &U[0,0]

    cdef FPnum *ptr_I = NULL
    if I.shape[0]:
        ptr_I = &I[0,0]
    
    cdef np.ndarray[FPnum, ndim=2] Am = np.empty((m, k), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] Bm = np.empty((n, k), dtype=c_FPnum)
    cdef FPnum *ptr_Am = &Am[0,0]
    cdef FPnum *ptr_Bm = &Bm[0,0]

    rs = np.random.Generator(np.random.MT19937(seed = seed))
    cdef np.ndarray[FPnum, ndim=2] A = rs.standard_normal(size=(m,k), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] B = rs.standard_normal(size=(n,k), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] C = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] D = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=1] C_bias = np.zeros(0, dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=1] D_bias = np.zeros(0, dtype=c_FPnum)

    cdef FPnum *ptr_A = &A[0,0]
    cdef FPnum *ptr_B = &B[0,0]
    cdef FPnum *ptr_C = NULL
    cdef FPnum *ptr_D = NULL
    cdef FPnum *ptr_C_bias = NULL
    cdef FPnum *ptr_D_bias = NULL
    if p:
        C = np.empty((p,k), dtype=c_FPnum)
        if C.shape[0]:
            ptr_C = &C[0,0]
        if add_intercepts:
            C_bias = np.empty(k, dtype=c_FPnum)
            if C_bias.shape[0]:
                ptr_C_bias = &C_bias[0]
    if q:
        D = np.empty((q,k), dtype=c_FPnum)
        if D.shape[0]:
            ptr_D = &D[0,0]
        if add_intercepts:
            D_bias = np.empty(k, dtype=c_FPnum)
            if D_bias.shape[0]:
                ptr_D_bias = &D_bias[0]

    cdef np.ndarray[FPnum, ndim=2] BtB = np.zeros((0,0), dtype=c_FPnum)
    cdef FPnum *ptr_BtB = NULL
    if precompute_for_predictions:
        BtB = np.empty((k,k), dtype=c_FPnum)
        ptr_BtB = &BtB[0,0]

    cdef FPnum placeholder = 0

    cdef int retval = fit_offsets_als(
        <FPnum*>NULL, <FPnum*>NULL,
        ptr_A, ptr_B, ptr_C, ptr_C_bias, ptr_D, ptr_D_bias,
        0, 0,
        &placeholder,
        m, n, k,
        ptr_ixA, ptr_ixB, ptr_X, nnz,
        <FPnum*>NULL,
        <FPnum*>NULL,
        0, 0, add_intercepts,
        lam,
        ptr_U, p,
        ptr_I, q,
        1, 0, alpha,
        niter,
        nthreads, use_cg,
        max_cg_steps, finalize_chol,
        verbose, handle_interrupt,
        precompute_for_predictions,
        ptr_Am, ptr_Bm,
        <FPnum*>NULL,
        ptr_BtB,
        <FPnum*>NULL
    )

    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")
    elif retval == 2:
        raise ValueError("Invalid parameter combination.")

    return A, B, C, D, \
           Am, Bm, BtB

def precompute_matrices_collective_explicit(
        np.ndarray[FPnum, ndim=2] B,
        np.ndarray[FPnum, ndim=2] C,
        bint user_bias,
        int k, int k_user, int k_item, int k_main,
        FPnum lam, FPnum lam_bias, FPnum w_main, FPnum w_user,
    ):
    cdef int n = B.shape[0]
    cdef int p = C.shape[0]

    if n == 0:
        raise ValueError("'B' has no entries.")

    cdef FPnum *ptr_B = &B[0,0]
    cdef FPnum *ptr_C = NULL
    if p:
        ptr_C = &C[0,0]

    cdef np.ndarray[FPnum, ndim=2] B_plus_bias = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] BtB = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] TransBtBinvBt = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] BeTBeChol = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] TransCtCinvCt = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] CtCw = np.zeros((0,0), dtype=c_FPnum)
    cdef FPnum *ptr_B_plus_bias = NULL
    cdef FPnum *ptr_BtB = NULL
    cdef FPnum *ptr_TransBtBinvBt = NULL
    cdef FPnum *ptr_BeTBeChol = NULL
    cdef FPnum *ptr_TransCtCinvCt = NULL
    cdef FPnum *ptr_CtCw = NULL

    BtB = np.empty((k+k_main, k+k_main), dtype=c_FPnum)
    TransBtBinvBt = np.empty((B.shape[0], k+k_main), dtype=c_FPnum)
    ptr_BtB = &BtB[0,0]
    ptr_TransBtBinvBt = &TransBtBinvBt[0,0]
    if user_bias:
        B_plus_bias = np.empty((B.shape[0], B.shape[0]+1), dtype=c_FPnum)
        ptr_B_plus_bias = &B_plus_bias[0,0]
    else:
        B_plus_bias = B
    if p > 0:
        BeTBeChol = np.empty((k_user+k+k_main, k_user+k+k_main), dtype=c_FPnum)
        TransCtCinvCt = np.empty((C.shape[0], k_user+k), dtype=c_FPnum)
        CtCw = np.empty((k_user+k, k_user+k), dtype=c_FPnum)
        ptr_BeTBeChol = &BeTBeChol[0,0]
        ptr_TransCtCinvCt = &TransCtCinvCt[0,0]
        ptr_CtCw = &CtCw[0,0]

    cdef FPnum *ptr_lam_unique = NULL
    cdef np.ndarray[FPnum, ndim=1] lam_unique = np.zeros(6, dtype=c_FPnum)
    if lam_bias != lam:
        lam_unique[0] = lam_bias
        lam_unique[2] = lam
        ptr_lam_unique = &lam_unique[0]

    cdef int retval = precompute_collective_explicit(
        ptr_B, n, n, n,
        ptr_C, p,
        k, k_user, k_item, k_main,
        user_bias,
        lam, ptr_lam_unique,
        w_main, w_user,
        ptr_B_plus_bias,
        ptr_BtB,
        ptr_TransBtBinvBt,
        ptr_BeTBeChol,
        ptr_TransCtCinvCt,
        ptr_CtCw
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return B_plus_bias, BtB, TransBtBinvBt, BeTBeChol, TransCtCinvCt, CtCw

def precompute_matrices_collective_implicit(
        np.ndarray[FPnum, ndim=2] B,
        np.ndarray[FPnum, ndim=2] C,
        int k, int k_main, int k_user, int k_item,
        FPnum lam, FPnum w_main, FPnum w_user,
        FPnum w_main_multiplier
    ):
    cdef int n = B.shape[0]
    cdef np.ndarray[FPnum, ndim=2] BtB = np.empty((k+k_main, k+k_main), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] BeTBe = np.zeros((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] BeTBeChol = np.zeros((0,0), dtype=c_FPnum)
    if C.shape[0] and C.shape[1]:
        BeTBe = np.empty((k_user+k+k_main, k_user+k+k_main), dtype=c_FPnum)
        BeTBeChol = np.empty((k_user+k+k_main, k_user+k+k_main), dtype=c_FPnum)

    cdef FPnum *ptr_BtB = &BtB[0,0]
    cdef FPnum *ptr_BeTBe = NULL
    cdef FPnum *ptr_BeTBeChol = NULL
    if C.shape[0] and C.shape[1]:
        ptr_BeTBe = &BeTBe[0,0]
        ptr_BeTBeChol = &BeTBeChol[0,0]

    cdef FPnum *ptr_C = NULL
    cdef int p = 0
    if C.shape[0]:
        p = C.shape[0]
        ptr_C = &C[0,0]

    cdef int retval = precompute_collective_implicit(
        &B[0,0], B.shape[0],
        ptr_C, p,
        k, k_user, k_item, k_main,
        lam, w_main, w_user, w_main_multiplier,
        1,
        &BtB[0,0], &BeTBe[0,0], &BeTBeChol[0,0]
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return BtB, BeTBe, BeTBeChol

def call_factors_collective_cold(
        np.ndarray[FPnum, ndim=1] U,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[int, ndim=1] U_sp_i,
        np.ndarray[FPnum, ndim=1] U_bin,
        np.ndarray[FPnum, ndim=2] C,
        np.ndarray[FPnum, ndim=2] C_bin,
        np.ndarray[FPnum, ndim=2] TransCtCinvCt,
        np.ndarray[FPnum, ndim=2] CtC,
        np.ndarray[FPnum, ndim=1] U_colmeans,
        int p, int k,
        int k_user = 0, int k_main = 0,
        FPnum lam = 1e2, FPnum w_main = 1., FPnum w_user = 1.,
        bint NA_as_zero_U = 0
    ):
    cdef FPnum *ptr_U = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef int *ptr_U_sp_i = NULL
    if U.shape[0]:
        ptr_U = &U[0]
    elif U_sp.shape[0]:
        ptr_U_sp = &U_sp[0]
        ptr_U_sp_i = &U_sp_i[0]

    cdef FPnum *ptr_U_bin = NULL
    if U_bin.shape[0]:
        ptr_U_bin = &U_bin[0]

    cdef FPnum *ptr_C = NULL
    cdef FPnum *ptr_C_bin = NULL
    cdef FPnum *ptr_TransCtCinvCt = NULL
    cdef FPnum *ptr_CtC = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]
    if C_bin.shape[0]:
        ptr_C_bin = &C_bin[0,0]
    if TransCtCinvCt.shape[0]:
        ptr_TransCtCinvCt = &TransCtCinvCt[0,0]
    if CtC.shape[0]:
        ptr_CtC = &CtC[0,0]

    cdef FPnum *ptr_U_colmeans = NULL
    if U_colmeans.shape[0]:
        ptr_U_colmeans = &U_colmeans[0]

    cdef np.ndarray[FPnum, ndim=1] A = np.empty(k_user+k+k_main, dtype=c_FPnum)

    cdef int retval = collective_factors_cold(
        &A[0],
        ptr_U, p,
        ptr_U_sp, ptr_U_sp_i, U_sp.shape[0],
        ptr_U_bin, U_bin.shape[0],
        ptr_C, ptr_C_bin,
        ptr_TransCtCinvCt,
        ptr_CtC,
        ptr_U_colmeans,
        k, k_user, k_main,
        lam, w_main, w_user,
        NA_as_zero_U
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return A

def call_factors_collective_cold_implicit(
        np.ndarray[FPnum, ndim=1] U,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[int, ndim=1] U_sp_i,
        np.ndarray[FPnum, ndim=2] B,
        np.ndarray[FPnum, ndim=2] C,
        np.ndarray[FPnum, ndim=2] BeTBe,
        np.ndarray[FPnum, ndim=2] BtB,
        np.ndarray[FPnum, ndim=2] BeTBeChol,
        np.ndarray[FPnum, ndim=1] U_colmeans,
        int p, int k,
        int k_user = 0, int k_item = 0, int k_main = 0,
        FPnum lam = 1e2,
        FPnum w_main = 1., FPnum w_user = 1.,
        FPnum w_main_multiplier = 1.,
        bint NA_as_zero_U = 0
    ):
    cdef FPnum *ptr_U = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef int *ptr_U_sp_i = NULL
    if U.shape[0]:
        ptr_U = &U[0]
    elif U_sp.shape[0]:
        ptr_U_sp = &U_sp[0]
        ptr_U_sp_i = &U_sp_i[0]

    cdef FPnum *ptr_B = NULL
    if B.shape[0]:
        ptr_B = &B[0,0]
    
    cdef FPnum *ptr_C = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]

    cdef FPnum *ptr_BeTBe = NULL
    cdef FPnum *ptr_BtB = NULL
    cdef FPnum *ptr_BeTBeChol = NULL
    if BeTBe.shape[0]:
        ptr_BeTBe = &BeTBe[0,0]
    if BtB.shape[0]:
        ptr_BtB = &BtB[0,0]
    if BeTBeChol.shape[0]:
        ptr_BeTBeChol = &BeTBeChol[0,0]

    cdef FPnum *ptr_U_colmeans = NULL
    if U_colmeans.shape[0]:
        ptr_U_colmeans = &U_colmeans[0]

    cdef np.ndarray[FPnum, ndim=1] A = np.empty(k_user+k+k_main, dtype=c_FPnum)

    cdef int retval = collective_factors_cold_implicit(
        &A[0],
        ptr_U, p,
        ptr_U_sp, ptr_U_sp_i, <size_t> U_sp.shape[0],
        ptr_B, B.shape[0],
        ptr_C,
        ptr_BeTBe,
        ptr_BtB,
        ptr_BeTBeChol,
        ptr_U_colmeans,
        k, k_user, k_item, k_main,
        lam, w_main, w_user, w_main_multiplier,
        NA_as_zero_U
    )

    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return A

def call_factors_collective_warm_explicit(
        np.ndarray[FPnum, ndim=1] Xa_dense,
        np.ndarray[FPnum, ndim=1] W_dense,
        np.ndarray[FPnum, ndim=1] Xa,
        np.ndarray[int, ndim=1] Xa_i,
        np.ndarray[FPnum, ndim=1] W_sp,
        np.ndarray[FPnum, ndim=1] U,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[int, ndim=1] U_sp_i,
        np.ndarray[FPnum, ndim=1] U_bin,
        np.ndarray[FPnum, ndim=1] U_colmeans,
        np.ndarray[FPnum, ndim=1] biasB,
        np.ndarray[FPnum, ndim=2] B,
        np.ndarray[FPnum, ndim=2] B_plus_bias,
        np.ndarray[FPnum, ndim=2] C,
        np.ndarray[FPnum, ndim=2] C_bin,
        np.ndarray[FPnum, ndim=2] TransBtBinvBt,
        np.ndarray[FPnum, ndim=2] BtB,
        np.ndarray[FPnum, ndim=2] BeTBeChol,
        np.ndarray[FPnum, ndim=2] CtCw,
        FPnum glob_mean,
        int k, int k_user = 0, int k_item = 0, int k_main = 0,
        FPnum lam = 1e2, FPnum lam_bias = 1e2,
        FPnum w_user = 1., FPnum w_main = 1.,
        bint user_bias = 1,
        bint NA_as_zero_U = 0, bint NA_as_zero_X = 0
    ):
    
    cdef FPnum *ptr_Xa_dense = NULL
    cdef FPnum *ptr_Xa = NULL
    cdef int *ptr_Xa_i = NULL
    cdef FPnum *ptr_weight = NULL
    if Xa_dense.shape[0]:
        ptr_Xa_dense = &Xa_dense[0]
        if W_dense.shape[0]:
            ptr_weight = &W_dense[0]
    elif Xa.shape[0]:
        ptr_Xa = &Xa[0]
        ptr_Xa_i = &Xa_i[0]
        if W_sp.shape[0]:
            ptr_weight = &W_sp[0]

    cdef FPnum *ptr_U = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef int *ptr_U_sp_i = NULL
    if U.shape[0]:
        ptr_U = &U[0]
    elif U_sp.shape[0]:
        ptr_U_sp = &U_sp[0]
        ptr_U_sp_i = &U_sp_i[0]

    cdef FPnum *ptr_U_bin = NULL
    if U_bin.shape[0]:
        ptr_U_bin = &U_bin[0]

    cdef FPnum *ptr_U_colmeans = NULL
    if U_colmeans.shape[0]:
        ptr_U_colmeans = &U_colmeans[0]

    cdef FPnum *ptr_biasB = NULL
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]

    cdef FPnum *ptr_C = NULL
    cdef FPnum *ptr_C_bin = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]
    if C_bin.shape[0]:
        ptr_C_bin = &C_bin[0,0]

    cdef FPnum *ptr_TransBtBinvBt = NULL
    cdef FPnum *ptr_BtB = NULL
    cdef FPnum *ptr_BeTBeChol = NULL
    cdef FPnum *ptr_CtCw = NULL
    if TransBtBinvBt.shape[0]:
        ptr_TransBtBinvBt = &TransBtBinvBt[0,0]
    if BtB.shape[0]:
        ptr_BtB = &BtB[0,0]
    if BeTBeChol.shape[0]:
        ptr_BeTBeChol = &BeTBeChol[0,0]
    if CtCw.shape[0]:
        ptr_CtCw = &CtCw[0,0]
    
    cdef FPnum Amean = 0;
    cdef FPnum *ptr_Amean = NULL
    if user_bias:
        ptr_Amean = &Amean

    cdef FPnum *ptr_B_plus_bias = NULL
    if B_plus_bias.shape[0]:
        ptr_B_plus_bias = &B_plus_bias[0,0]

    cdef np.ndarray[FPnum, ndim=1] A = np.empty(k_user+k+k_main, dtype=c_FPnum)
    cdef int retval = collective_factors_warm(
        &A[0], ptr_Amean,
        ptr_U, C.shape[0],
        ptr_U_sp, ptr_U_sp_i, U_sp.shape[0],
        ptr_U_bin, C_bin.shape[0],
        ptr_C, ptr_C_bin,
        glob_mean, ptr_biasB,
        ptr_U_colmeans,
        ptr_Xa, ptr_Xa_i, Xa.shape[0],
        ptr_Xa_dense, B.shape[0],
        ptr_weight,
        &B[0,0],
        k, k_user, k_item, k_main,
        lam, w_main, w_user, lam_bias,
        ptr_TransBtBinvBt,
        ptr_BtB,
        ptr_BeTBeChol,
        ptr_CtCw,
        NA_as_zero_U, NA_as_zero_X,
        ptr_B_plus_bias
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return Amean, A

def call_factors_collective_warm_implicit(
        np.ndarray[FPnum, ndim=1] Xa,
        np.ndarray[int, ndim=1] Xa_i,
        np.ndarray[FPnum, ndim=1] U,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[int, ndim=1] U_sp_i,
        np.ndarray[FPnum, ndim=1] U_colmeans,
        np.ndarray[FPnum, ndim=2] B,
        np.ndarray[FPnum, ndim=2] C,
        np.ndarray[FPnum, ndim=2] BeTBe,
        np.ndarray[FPnum, ndim=2] BtB,
        np.ndarray[FPnum, ndim=2] BeTBeChol,
        int k, int k_user = 0, int k_item = 0, int k_main = 0,
        FPnum lam = 1e2, FPnum alpha = 40.,
        FPnum w_main_multiplier = 1.,
        FPnum w_user = 1., FPnum w_main = 1.,
        bint NA_as_zero_U = 0
    ):

    cdef FPnum *ptr_U = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef int *ptr_U_sp_i = NULL
    if U.shape[0]:
        ptr_U = &U[0]
    elif U_sp.shape[0]:
        ptr_U_sp = &U_sp[0]
        ptr_U_sp_i = &U_sp_i[0]

    cdef FPnum *ptr_U_colmeans = NULL
    if U_colmeans.shape[0]:
        ptr_U_colmeans = &U_colmeans[0]

    cdef FPnum *ptr_C = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]

    cdef FPnum *ptr_BeTBe = NULL
    cdef FPnum *ptr_BtB = NULL
    cdef FPnum *ptr_BeTBeChol = NULL
    if BeTBe.shape[0]:
        ptr_BeTBe = &BeTBe[0,0]
    if BtB.shape[0]:
        ptr_BtB = &BtB[0,0]
    if BeTBeChol.shape[0]:
        ptr_BeTBeChol = &BeTBeChol[0,0]
    
    cdef np.ndarray[FPnum, ndim=1] A = np.empty(k_user+k+k_main, dtype=c_FPnum)
    cdef int retval = collective_factors_warm_implicit(
        &A[0],
        ptr_U, C.shape[0],
        ptr_U_sp, ptr_U_sp_i, U_sp.shape[0],
        NA_as_zero_U,
        ptr_U_colmeans,
        &B[0,0], B.shape[0], ptr_C,
        &Xa[0], &Xa_i[0], Xa.shape[0],
        k, k_user, k_item, k_main,
        lam, alpha, w_main, w_user,
        w_main_multiplier,
        ptr_BeTBe,
        ptr_BtB,
        ptr_BeTBeChol
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return A

def call_factors_offsets_cold(
        np.ndarray[FPnum, ndim=1] U,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[int, ndim=1] U_sp_i,
        np.ndarray[FPnum, ndim=2] C,
        np.ndarray[FPnum, ndim=1] C_bias,
        int k,
        int k_sec = 0, int k_main = 0,
        FPnum w_user = 1.
    ):
    cdef FPnum *ptr_U = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef int *ptr_U_sp_i = NULL
    if U.shape[0]:
        ptr_U = &U[0]
    elif U_sp.shape[0]:
        ptr_U_sp = &U_sp[0]
        ptr_U_sp_i = &U_sp_i[0]

    cdef FPnum *ptr_C_bias = NULL
    if C_bias.shape[0]:
        ptr_C_bias = &C_bias[0]

    cdef np.ndarray[FPnum, ndim=1] A = np.empty(k_sec+k+k_main, dtype=c_FPnum)

    cdef int retval = offsets_factors_cold(
        &A[0],
        ptr_U,
        ptr_U_sp_i, ptr_U_sp, U_sp.shape[0],
        &C[0,0], C.shape[0],
        ptr_C_bias,
        k, k_sec, k_main,
        w_user
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return A

def call_factors_offsets_warm_explicit(
        np.ndarray[FPnum, ndim=1] Xa_dense,
        np.ndarray[FPnum, ndim=1] W_dense,
        np.ndarray[FPnum, ndim=1] Xa,
        np.ndarray[int, ndim=1] Xa_i,
        np.ndarray[FPnum, ndim=1] W_sp,
        np.ndarray[FPnum, ndim=1] U,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[int, ndim=1] U_sp_i,
        np.ndarray[FPnum, ndim=1] biasB,
        np.ndarray[FPnum, ndim=2] Bm,
        np.ndarray[FPnum, ndim=2] Bm_plus_bias,
        np.ndarray[FPnum, ndim=2] C,
        np.ndarray[FPnum, ndim=1] C_bias,
        np.ndarray[FPnum, ndim=2] TransBtBinvBt,
        np.ndarray[FPnum, ndim=2] BtB,
        FPnum glob_mean,
        int k, int k_sec = 0, int k_main = 0,
        FPnum lam = 1e2, FPnum lam_bias = 1e2,
        FPnum w_user = 1.,
        bint user_bias = 1,
        bint exact = 0, bint output_a = 1
    ):

    cdef FPnum *ptr_Xa_dense = NULL
    cdef FPnum *ptr_Xa = NULL
    cdef int *ptr_Xa_i = NULL
    cdef FPnum *ptr_weight = NULL
    if Xa_dense.shape[0]:
        ptr_Xa_dense = &Xa_dense[0]
        if W_dense.shape[0]:
            ptr_weight = &W_dense[0]
    elif Xa.shape[0]:
        ptr_Xa = &Xa[0]
        ptr_Xa_i = &Xa_i[0]
        if W_sp.shape[0]:
            ptr_weight = &W_sp[0]

    cdef FPnum *ptr_U = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef int *ptr_U_sp_i = NULL
    if U.shape[0]:
        ptr_U = &U[0]
    elif U_sp.shape[0]:
        ptr_U_sp = &U_sp[0]
        ptr_U_sp_i = &U_sp_i[0]

    cdef FPnum *ptr_C = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]

    cdef FPnum *ptr_biasB = NULL
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]

    cdef FPnum *ptr_TransBtBinvBt = NULL
    cdef FPnum *ptr_BtB = NULL
    if TransBtBinvBt.shape[0]:
        ptr_TransBtBinvBt = &TransBtBinvBt[0,0]
    if BtB.shape[0]:
        ptr_BtB = &BtB[0,0]
    
    cdef FPnum Amean = 0;
    cdef FPnum *ptr_Amean = NULL
    if user_bias:
        ptr_Amean = &Amean

    cdef np.ndarray[FPnum, ndim=1] A = np.empty(0, dtype=c_FPnum)
    cdef FPnum *ptr_A = NULL
    if output_a and (k or k_main):
        A = np.empty(k+k_main, dtype=c_FPnum)
        ptr_A = &A[0]

    cdef FPnum *ptr_Bm_plus_bias = NULL
    if Bm_plus_bias.shape[0]:
        ptr_Bm_plus_bias = &Bm_plus_bias[0,0]

    cdef FPnum *ptr_C_bias = NULL
    if C_bias.shape[0]:
        ptr_C_bias = &C_bias[0]

    cdef np.ndarray[FPnum, ndim=1] Am = np.empty(k_sec+k+k_main, dtype=c_FPnum)
    cdef int retval = offsets_factors_warm(
        &Am[0], ptr_Amean,
        ptr_U,
        ptr_U_sp_i, ptr_U_sp, U_sp.shape[0],
        ptr_Xa_i, ptr_Xa, Xa.shape[0],
        ptr_Xa_dense, Bm.shape[0],
        ptr_weight,
        &Bm[0,0], ptr_C,
        ptr_C_bias,
        glob_mean, ptr_biasB,
        k, k_sec, k_main,
        C.shape[0], w_user,
        lam, exact, lam_bias,
        0, 0.,
        ptr_TransBtBinvBt,
        ptr_BtB,
        ptr_A,
        ptr_Bm_plus_bias
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return Amean, Am, A

def call_factors_offsets_warm_implicit(
        np.ndarray[FPnum, ndim=1] Xa,
        np.ndarray[int, ndim=1] Xa_i,
        np.ndarray[FPnum, ndim=1] U,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[int, ndim=1] U_sp_i,
        np.ndarray[FPnum, ndim=2] Bm,
        np.ndarray[FPnum, ndim=2] C,
        np.ndarray[FPnum, ndim=1] C_bias,
        np.ndarray[FPnum, ndim=2] TransBtBinvBt,
        np.ndarray[FPnum, ndim=2] BtB,
        int k, int k_sec = 0, int k_main = 0,
        FPnum lam = 1e2, FPnum alpha = 40.,
        bint user_bias = 1,
        bint output_a = 1
    ):

    cdef FPnum *ptr_U = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef int *ptr_U_sp_i = NULL
    if U.shape[0]:
        ptr_U = &U[0]
    elif U_sp.shape[0]:
        ptr_U_sp = &U_sp[0]
        ptr_U_sp_i = &U_sp_i[0]

    cdef FPnum *ptr_C = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]

    cdef FPnum *ptr_TransBtBinvBt = NULL
    cdef FPnum *ptr_BtB = NULL
    if TransBtBinvBt.shape[0]:
        ptr_TransBtBinvBt = &TransBtBinvBt[0,0]
    if BtB.shape[0]:
        ptr_BtB = &BtB[0,0]

    cdef np.ndarray[FPnum, ndim=1] A = np.empty(0, dtype=c_FPnum)
    cdef FPnum *ptr_A = NULL
    if output_a:
        A = np.empty(k+k_main, dtype=c_FPnum)
        ptr_A = &A[0]

    cdef FPnum *ptr_C_bias = NULL
    if C_bias.shape[0]:
        ptr_C_bias = &C_bias[0]

    cdef np.ndarray[FPnum, ndim=1] Am = np.empty(k_sec+k+k_main, dtype=c_FPnum)
    cdef int retval = offsets_factors_warm(
        &Am[0], <FPnum*>NULL,
        ptr_U,
        ptr_U_sp_i, ptr_U_sp, U_sp.shape[0],
        &Xa_i[0], &Xa[0], Xa.shape[0],
        <FPnum*>NULL, Bm.shape[0],
        <FPnum*>NULL,
        &Bm[0,0], ptr_C,
        ptr_C_bias,
        0., <FPnum*>NULL,
        k, k_sec, k_main,
        C.shape[0], 1.,
        lam, 0, lam,
        1, alpha,
        ptr_TransBtBinvBt,
        ptr_BtB,
        ptr_A,
        <FPnum*>NULL
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return Am, A

def call_factors_content_based_single(
        np.ndarray[FPnum, ndim=1] U,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[int, ndim=1] U_sp_i,
        np.ndarray[FPnum, ndim=2] C,
        np.ndarray[FPnum, ndim=1] C_bias
    ):
    cdef FPnum *ptr_U = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef int *ptr_U_sp_i = NULL
    if U.shape[0]:
        ptr_U = &U[0]
    else:
        ptr_U_sp = &U_sp[0]
        ptr_U_sp_i = &U_sp_i[0]

    cdef FPnum *ptr_C_bias = NULL
    if C_bias.shape[0]:
        ptr_C_bias = &C_bias[0]
    
    cdef np.ndarray[FPnum, ndim=1] a_vec = np.empty(C.shape[1], dtype=c_FPnum)
    cdef int retval = factors_content_based_single(
        &a_vec[0], C.shape[1],
        ptr_U, C.shape[0],
        ptr_U_sp, ptr_U_sp_i, U_sp.shape[0],
        &C[0,0], ptr_C_bias
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")
    return a_vec

def call_predict_multiple(
        np.ndarray[FPnum, ndim=2] A,
        np.ndarray[FPnum, ndim=2] B,
        np.ndarray[FPnum, ndim=1] biasA,
        np.ndarray[FPnum, ndim=1] biasB,
        FPnum glob_mean,
        np.ndarray[int, ndim=1] predA,
        np.ndarray[int, ndim=1] predB,
        int k, int k_user = 0, int k_item = 0, int k_main = 0,
        int nthreads = 1
    ):
    cdef FPnum *ptr_biasA = NULL
    cdef FPnum *ptr_biasB = NULL
    if biasA.shape[0]:
        ptr_biasA = &biasA[0]
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]

    cdef np.ndarray[FPnum, ndim=1] outp = np.empty(predA.shape[0], dtype=c_FPnum)
    if outp.shape[0] == 0:
        return outp

    predict_multiple(
        &A[0,0], k_user,
        &B[0,0], k_item,
        ptr_biasA, ptr_biasB,
        glob_mean,
        k, k_main,
        &predA[0], &predB[0], predA.shape[0],
        &outp[0],
        nthreads
    )
    return outp

def call_topN(
        np.ndarray[FPnum, ndim=1] a_vec,
        np.ndarray[FPnum, ndim=2] B,
        np.ndarray[FPnum, ndim=1] biasB,
        FPnum glob_mean, FPnum biasA,
        np.ndarray[int, ndim=1] include_ix,
        np.ndarray[int, ndim=1] exclude_ix,
        int n_top,
        int k, int k_user = 0, int k_item = 0, int k_main = 0,
        bint output_score = 1,
        int nthreads = 1
    ):

    cdef FPnum *ptr_biasB = NULL
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]

    cdef int *ptr_include = NULL
    cdef int *ptr_exclude = NULL
    if include_ix.shape[0]:
        ptr_include = &include_ix[0]
    if exclude_ix.shape[0]:
        ptr_exclude = &exclude_ix[0]

    cdef np.ndarray[FPnum, ndim=1] outp_score = np.empty(0, dtype=c_FPnum)
    cdef FPnum *ptr_outp_score = NULL
    if output_score:
        outp_score = np.empty(n_top, dtype=c_FPnum)
        ptr_outp_score = &outp_score[0]
    cdef np.ndarray[int, ndim=1] outp_ix = np.empty(n_top, dtype=ctypes.c_int)
    cdef int retval = topN(
        &a_vec[0], k_user,
        &B[0,0], k_item,
        ptr_biasB,
        glob_mean, biasA,
        k, k_main,
        ptr_include, include_ix.shape[0],
        ptr_exclude, exclude_ix.shape[0],
        &outp_ix[0], ptr_outp_score,
        n_top, B.shape[0], nthreads
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return outp_ix, outp_score

def call_predict_X_old_content_based(
        np.ndarray[FPnum, ndim=2] U,
        np.ndarray[int, ndim=1] U_row,
        np.ndarray[int, ndim=1] U_col,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[size_t, ndim=1] U_csr_p,
        np.ndarray[int, ndim=1] U_csr_i,
        np.ndarray[FPnum, ndim=1] U_csr,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[FPnum, ndim=2] Bm,
        np.ndarray[FPnum, ndim=2] C,
        np.ndarray[FPnum, ndim=1] C_bias,
        np.ndarray[FPnum, ndim=1] biasB,
        int n_new,
        FPnum glob_mean,
        int nthreads
    ):

    cdef FPnum *ptr_U = NULL
    cdef int *ptr_U_row = NULL
    cdef int *ptr_U_col = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]

    cdef size_t *ptr_U_csr_p = NULL
    cdef int *ptr_U_csr_i = NULL
    cdef FPnum *ptr_U_csr = NULL
    if U_csr.shape[0]:
        ptr_U_csr_p = &U_csr_p[0]
        ptr_U_csr_i = &U_csr_i[0]
        ptr_U_csr = &U_csr[0]

    cdef FPnum *ptr_biasB = NULL
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]

    cdef FPnum *ptr_C_bias = NULL
    if C_bias.shape[0]:
        ptr_C_bias = &C_bias[0]

    cdef np.ndarray[FPnum, ndim=1] scores_new = np.empty(n_new, dtype=c_FPnum)
    cdef int k = C.shape[1]
    
    cdef int retval = predict_X_old_content_based(
        &scores_new[0], n_new, n_new, k,
        <int*>NULL, &ixB[0],
        ptr_U, C.shape[0],
        ptr_U_row, ptr_U_col, ptr_U_sp, nnz_U,
        ptr_U_csr_p, ptr_U_csr_i, ptr_U_csr,
        &C[0,0], ptr_C_bias,
        &Bm[0,0], ptr_biasB,
        glob_mean,
        nthreads
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return scores_new

def call_predict_X_new_content_based(
        np.ndarray[FPnum, ndim=2] U,
        np.ndarray[int, ndim=1] U_row,
        np.ndarray[int, ndim=1] U_col,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[size_t, ndim=1] U_csr_p,
        np.ndarray[int, ndim=1] U_csr_i,
        np.ndarray[FPnum, ndim=1] U_csr,
        np.ndarray[FPnum, ndim=2] I,
        np.ndarray[int, ndim=1] I_row,
        np.ndarray[int, ndim=1] I_col,
        np.ndarray[FPnum, ndim=1] I_sp,
        np.ndarray[size_t, ndim=1] I_csr_p,
        np.ndarray[int, ndim=1] I_csr_i,
        np.ndarray[FPnum, ndim=1] I_csr,
        np.ndarray[FPnum, ndim=2] C,
        np.ndarray[FPnum, ndim=1] C_bias,
        np.ndarray[FPnum, ndim=2] D,
        np.ndarray[FPnum, ndim=1] D_bias,
        int n_new,
        FPnum glob_mean,
        int nthreads
    ):

    cdef FPnum *ptr_U = NULL
    cdef int *ptr_U_row = NULL
    cdef int *ptr_U_col = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]

    cdef size_t *ptr_U_csr_p = NULL
    cdef int *ptr_U_csr_i = NULL
    cdef FPnum *ptr_U_csr = NULL
    if U_csr.shape[0]:
        ptr_U_csr_p = &U_csr_p[0]
        ptr_U_csr_i = &U_csr_i[0]
        ptr_U_csr = &U_csr[0]

    cdef FPnum *ptr_I = NULL
    cdef int *ptr_I_row = NULL
    cdef int *ptr_I_col = NULL
    cdef FPnum *ptr_I_sp = NULL
    cdef size_t nnz_I = 0
    if I.shape[0]:
        ptr_I = &I[0,0]
    elif I_sp.shape[0]:
        ptr_I_row = &I_row[0]
        ptr_I_col = &I_col[0]
        ptr_I_sp = &I_sp[0]
        nnz_I = I_sp.shape[0]

    cdef size_t *ptr_I_csr_p = NULL
    cdef int *ptr_I_csr_i = NULL
    cdef FPnum *ptr_I_csr = NULL
    if I_csr.shape[0]:
        ptr_I_csr_p = &I_csr_p[0]
        ptr_I_csr_i = &I_csr_i[0]
        ptr_I_csr = &I_csr[0]

    cdef FPnum *ptr_C_bias = NULL
    if C_bias.shape[0]:
        ptr_C_bias = &C_bias[0]

    cdef FPnum *ptr_D_bias = NULL
    if D_bias.shape[0]:
        ptr_D_bias = &D_bias[0]
    
    cdef np.ndarray[FPnum, ndim=1] scores_new = np.empty(n_new, dtype=c_FPnum)
    cdef int k = C.shape[1]
    cdef int retval = predict_X_new_content_based(
        &scores_new[0], n_new,
        n_new, n_new, k,
        <int*>NULL, <int*>NULL,
        ptr_U, C.shape[0],
        ptr_U_row, ptr_U_col, ptr_U_sp, nnz_U,
        ptr_U_csr_p, ptr_U_csr_i, ptr_U_csr,
        ptr_I, D.shape[0],
        ptr_I_row, ptr_I_col, ptr_I_sp, nnz_I,
        ptr_I_csr_p, ptr_I_csr_i, ptr_I_csr,
        &C[0,0], ptr_C_bias,
        &D[0,0], ptr_D_bias,
        glob_mean,
        nthreads
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return scores_new

def call_topN_new_content_based(
        np.ndarray[FPnum, ndim=1] U,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[int, ndim=1] U_sp_i,
        np.ndarray[FPnum, ndim=2] I,
        np.ndarray[int, ndim=1] I_row,
        np.ndarray[int, ndim=1] I_col,
        np.ndarray[FPnum, ndim=1] I_sp,
        np.ndarray[size_t, ndim=1] I_csr_p,
        np.ndarray[int, ndim=1] I_csr_i,
        np.ndarray[FPnum, ndim=1] I_csr,
        np.ndarray[FPnum, ndim=2] C,
        np.ndarray[FPnum, ndim=1] C_bias,
        np.ndarray[FPnum, ndim=2] D,
        np.ndarray[FPnum, ndim=1] D_bias,
        int n_new_I,
        FPnum glob_mean = 0.,
        int n_top = 10, bint output_score = 1,
        int nthreads = 1
    ):
    
    cdef FPnum *ptr_U = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef int *ptr_U_sp_i = NULL
    if U.shape[0]:
        ptr_U = &U[0]
    elif U_sp.shape[0]:
        ptr_U_sp = &U_sp[0]
        ptr_U_sp_i = &U_sp_i[0]

    cdef FPnum *ptr_I = NULL
    cdef int *ptr_I_row = NULL
    cdef int *ptr_I_col = NULL
    cdef FPnum *ptr_I_sp = NULL
    cdef size_t nnz_I = 0
    if I.shape[0]:
        ptr_I = &I[0,0]
    elif I_sp.shape[0]:
        ptr_I_row = &I_row[0]
        ptr_I_col = &I_col[0]
        ptr_I_sp = &I_sp[0]
        nnz_I = I_sp.shape[0]

    cdef size_t *ptr_I_csr_p = NULL
    cdef int *ptr_I_csr_i = NULL
    cdef FPnum *ptr_I_csr = NULL
    if I_csr.shape[0]:
        ptr_I_csr_p = &I_csr_p[0]
        ptr_I_csr_i = &I_csr_i[0]
        ptr_I_csr = &I_csr[0]

    cdef FPnum *ptr_C_bias = NULL
    if C_bias.shape[0]:
        ptr_C_bias = &C_bias[0]

    cdef FPnum *ptr_D_bias = NULL
    if D_bias.shape[0]:
        ptr_D_bias = &D_bias[0]

    cdef np.ndarray[FPnum, ndim=1] scores_new = np.empty(0, dtype=c_FPnum)
    cdef FPnum *ptr_scores_new = NULL
    if output_score:
        scores_new = np.empty(n_top, dtype=c_FPnum)
        ptr_scores_new = &scores_new[0]
    if n_top <= 0:
        raise ValueError("'n_top' must be a positive integer.")
    cdef np.ndarray[int, ndim=1] rank_new = np.empty(n_top, dtype=ctypes.c_int)
    cdef int k = C.shape[1]
    
    cdef int retval = topN_new_content_based(
        k, n_new_I,
        ptr_U, C.shape[0],
        ptr_U_sp, ptr_U_sp_i, U_sp.shape[0],
        ptr_I, D.shape[0],
        ptr_I_row, ptr_I_col, ptr_I_sp, nnz_I,
        ptr_I_csr_p, ptr_I_csr_i, ptr_I_csr,
        &C[0,0], ptr_C_bias,
        &D[0,0], ptr_D_bias,
        glob_mean,
        &rank_new[0], ptr_scores_new,
        n_top, nthreads
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return rank_new, scores_new

def call_fit_most_popular(
        np.ndarray[int, ndim=1] ixA,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[FPnum, ndim=1] X,
        np.ndarray[FPnum, ndim=1] W_sp,
        np.ndarray[FPnum, ndim=2] Xfull,
        np.ndarray[FPnum, ndim=2] W_dense,
        int m, int n,
        FPnum lam_user = 1e2, FPnum lam_item = 1e2,
        FPnum alpha = 40.,
        bint user_bias = 0,
        bint implicit = 0, bint adjust_weight = 1,
        int nthreads = 1
    ):
    cdef FPnum *ptr_Xfull = NULL
    cdef int *ptr_ixA = NULL
    cdef int *ptr_ixB = NULL
    cdef FPnum *ptr_X = NULL
    cdef size_t nnz = 0
    cdef FPnum glob_mean = 0
    cdef FPnum *ptr_weight = NULL
    if Xfull.shape[0]:
        ptr_Xfull = &Xfull[0,0]
        if W_sp.shape[0]:
            ptr_weight = &W_sp[0]
    else:
        ptr_ixA = &ixA[0]
        ptr_ixB = &ixB[0]
        ptr_X = &X[0]
        nnz = X.shape[0]
        if W_dense.shape[0]:
            ptr_weight = &W_dense[0,0]

    cdef FPnum *ptr_biasA = NULL
    cdef FPnum *ptr_biasB = NULL

    cdef np.ndarray[FPnum, ndim=1] values
    if user_bias:
        values = np.empty(m+n, dtype=c_FPnum)
        ptr_biasA = &values[0]
        ptr_biasB = &values[m]
    else:
        values = np.empty(n, dtype=c_FPnum)
        ptr_biasB = &values[0]

    cdef FPnum w_main_multiplier = 1.

    cdef int retval = fit_most_popular(
        ptr_biasA, ptr_biasB,
        &glob_mean,
        lam_user, lam_item,
        alpha,
        m, n,
        ptr_ixA, ptr_ixB, ptr_X, nnz,
        ptr_Xfull,
        ptr_weight,
        implicit, adjust_weight,
        &w_main_multiplier,
        nthreads
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    if user_bias:
        return glob_mean, values[:m], values[m:], w_main_multiplier
    else:
        return glob_mean, np.empty(0, dtype=c_FPnum), values, w_main_multiplier

def call_fit_content_based_lbfgs(
        np.ndarray[int, ndim=1] ixA,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[FPnum, ndim=1] X,
        np.ndarray[FPnum, ndim=1] W,
        np.ndarray[FPnum, ndim=2] Xfull,
        np.ndarray[FPnum, ndim=2] Wfull,
        np.ndarray[FPnum, ndim=2] U,
        np.ndarray[int, ndim=1] U_row,
        np.ndarray[int, ndim=1] U_col,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[FPnum, ndim=2] I,
        np.ndarray[int, ndim=1] I_row,
        np.ndarray[int, ndim=1] I_col,
        np.ndarray[FPnum, ndim=1] I_sp,
        int m, int n, int p, int q,
        int k=50,
        bint user_bias=1, bint item_bias=1,
        bint add_intercepts=1,
        FPnum lam=1e2,
        np.ndarray[FPnum, ndim=1] lam_unique=np.empty(0, dtype=c_FPnum),
        bint verbose=1, int print_every=10,
        int n_corr_pairs=5, int maxiter=400,
        int nthreads=1, bint prefer_onepass=0,
        int seed=1, bint handle_interrupt=1,
        bint start_with_ALS=1
    ):

    cdef FPnum *ptr_Xfull = NULL
    cdef FPnum *ptr_weight = NULL
    cdef int *ptr_ixA = NULL
    cdef int *ptr_ixB = NULL
    cdef FPnum *ptr_X = NULL
    cdef size_t nnz = 0
    if Xfull.shape[0]:
        ptr_Xfull = &Xfull[0,0]
        if Wfull.shape[0]:
            ptr_weight = &Wfull[0,0]
    else:
        ptr_ixA = &ixA[0]
        ptr_ixB = &ixB[0]
        ptr_X = &X[0]
        nnz = X.shape[0]
        if W.shape[0]:
            ptr_weight = &W[0]

    cdef FPnum *ptr_U = NULL
    cdef int *ptr_U_row = NULL
    cdef int *ptr_U_col = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]

    cdef FPnum *ptr_I = NULL
    cdef int *ptr_I_row = NULL
    cdef int *ptr_I_col = NULL
    cdef FPnum *ptr_I_sp = NULL
    cdef size_t nnz_I = 0
    if I.shape[0]:
        ptr_I = &I[0,0]
    elif I_sp.shape[0]:
        ptr_I_row = &I_row[0]
        ptr_I_col = &I_col[0]
        ptr_I_sp = &I_sp[0]
        nnz_I = I_sp.shape[0]

    cdef FPnum *ptr_lam_unique = NULL
    if lam_unique.shape[0]:
        ptr_lam_unique = &lam_unique[0]

    cdef np.ndarray[FPnum, ndim=2] Am = np.empty((m, k), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] Bm = np.empty((n, k), dtype=c_FPnum)

    rs = np.random.Generator(np.random.MT19937(seed = seed))
    cdef np.ndarray[FPnum, ndim=2] C = rs.standard_normal(size=(p,k), dtype = c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] D = rs.standard_normal(size=(q,k), dtype = c_FPnum)
    cdef np.ndarray[FPnum, ndim=1] C_bias = np.zeros(0, dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=1] D_bias = np.zeros(0, dtype=c_FPnum)
    cdef FPnum *ptr_C = &C[0,0]
    cdef FPnum *ptr_D = &D[0,0]
    cdef FPnum *ptr_C_bias = NULL
    cdef FPnum *ptr_D_bias = NULL
    if add_intercepts:
        ptr_C_bias = &C_bias[0]
        ptr_D_bias = &D_bias[0]

    cdef np.ndarray[FPnum, ndim=1] biasA = np.zeros(0, dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=1] biasB = np.zeros(0, dtype=c_FPnum)
    cdef FPnum *ptr_biasA = NULL
    cdef FPnum *ptr_biasB = NULL
    if user_bias:
        biasA = np.empty(m, dtype=c_FPnum)
        ptr_biasA = &biasA[0]
    if item_bias:
        biasB = np.empty(n, dtype=c_FPnum)
        ptr_biasB = &biasB[0]

    cdef FPnum glob_mean
    cdef int niter, nfev

    cdef int retval = fit_content_based_lbfgs(
        ptr_biasA, ptr_biasB,
        ptr_C, ptr_C_bias,
        ptr_D, ptr_D_bias,
        start_with_ALS, 0, 0,
        &glob_mean,
        m, n, k,
        ptr_ixA, ptr_ixB, ptr_X, nnz,
        ptr_Xfull,
        ptr_weight,
        user_bias, item_bias,
        add_intercepts,
        lam, ptr_lam_unique,
        ptr_U, p,
        ptr_I, q,
        ptr_U_row, ptr_U_col, ptr_U_sp, nnz_U,
        ptr_I_row, ptr_I_col, ptr_I_sp, nnz_I,
        n_corr_pairs, maxiter,
        nthreads, prefer_onepass,
        verbose, print_every, handle_interrupt,
        &niter, &nfev,
        &Am[0,0], &Bm[0,0]
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return biasA, biasB, C, D, C_bias, D_bias, Am, Bm, glob_mean, niter, nfev

def call_factors_collective_explicit_multiple(
        np.ndarray[int, ndim=1] ixA,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[FPnum, ndim=1] X,
        np.ndarray[size_t, ndim=1] Xcsr_p,
        np.ndarray[int, ndim=1] Xcsr_i,
        np.ndarray[FPnum, ndim=1] Xcsr,
        np.ndarray[FPnum, ndim=1] W,
        np.ndarray[FPnum, ndim=2] Xfull,
        np.ndarray[FPnum, ndim=2] Wfull,
        np.ndarray[FPnum, ndim=2] U,
        np.ndarray[int, ndim=1] U_row,
        np.ndarray[int, ndim=1] U_col,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[size_t, ndim=1] U_csr_p,
        np.ndarray[int, ndim=1] U_csr_i,
        np.ndarray[FPnum, ndim=1] U_csr,
        np.ndarray[FPnum, ndim=2] Ub,
        np.ndarray[FPnum, ndim=1] U_colmeans,
        np.ndarray[FPnum, ndim=1] biasB,
        np.ndarray[FPnum, ndim=2] B,
        np.ndarray[FPnum, ndim=2] B_plus_bias,
        np.ndarray[FPnum, ndim=2] C,
        np.ndarray[FPnum, ndim=2] C_bin,
        np.ndarray[FPnum, ndim=2] TransBtBinvBt,
        np.ndarray[FPnum, ndim=2] BtB,
        np.ndarray[FPnum, ndim=2] BeTBeChol,
        np.ndarray[FPnum, ndim=2] TransCtCinvCt,
        np.ndarray[FPnum, ndim=2] CtCw,
        int n, int m_u, int m_x,
        FPnum glob_mean,
        int k, int k_user = 0, int k_item = 0, int k_main = 0,
        FPnum lam = 1e2, FPnum lam_bias = 1e2,
        FPnum w_user = 1., FPnum w_main = 1.,
        bint user_bias = 1,
        bint NA_as_zero_U = 0, bint NA_as_zero_X = 0,
        int nthreads = 1
    ):
    cdef int m_ubin = Ub.shape[0]
    cdef int p = C.shape[0]
    cdef int pbin = C_bin.shape[0]

    cdef FPnum *ptr_Xfull = NULL
    cdef FPnum *ptr_weight = NULL
    cdef int *ptr_ixA = NULL
    cdef int *ptr_ixB = NULL
    cdef FPnum *ptr_X = NULL
    cdef size_t nnz = 0
    cdef size_t *ptr_Xcsr_p = NULL
    cdef int *ptr_Xcsr_i = NULL
    cdef FPnum *ptr_Xcsr = NULL
    if Xfull.shape[0]:
        ptr_Xfull = &Xfull[0,0]
        if Wfull.shape[0]:
            ptr_weight = &Wfull[0,0]
    elif Xcsr.shape[0]:
        ptr_Xcsr_p = &Xcsr_p[0]
        ptr_Xcsr_i = &Xcsr_i[0]
        ptr_Xcsr = &Xcsr[0]
        nnz = Xcsr.shape[0]
        if W.shape[0]:
            ptr_weight = &W[0]
    elif X.shape[0]:
        ptr_ixA = &ixA[0]
        ptr_ixB = &ixB[0]
        ptr_X = &X[0]
        nnz = X.shape[0]
        if W.shape[0]:
            ptr_weight = &W[0]

    cdef FPnum *ptr_U = NULL
    cdef int *ptr_U_row = NULL
    cdef int *ptr_U_col = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    cdef FPnum *ptr_U_colmeans = NULL
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]
    if U_colmeans.shape[0]:
        ptr_U_colmeans = &U_colmeans[0]

    cdef size_t *ptr_U_csr_p = NULL
    cdef int *ptr_U_csr_i = NULL
    cdef FPnum *ptr_U_csr = NULL
    if U_csr.shape[0]:
        ptr_U_csr_p = &U_csr_p[0]
        ptr_U_csr_i = &U_csr_i[0]
        ptr_U_csr = &U_csr[0]

    cdef FPnum *ptr_Ub = NULL
    if Ub.shape[0]:
        ptr_Ub = &Ub[0,0]

    cdef FPnum *ptr_biasB = NULL
    cdef FPnum *ptr_B_plus_bias = NULL
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]
    if B_plus_bias.shape[0]:
        ptr_B_plus_bias = &B_plus_bias[0,0]

    cdef FPnum *ptr_C = NULL
    cdef FPnum *ptr_C_bin = NULL
    cdef FPnum *ptr_TransCtCinvCt = NULL
    cdef FPnum *ptr_CtCw = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]
    if C_bin.shape[0]:
        ptr_C_bin = &C_bin[0,0]
    if TransCtCinvCt.shape[0]:
        ptr_TransCtCinvCt = &TransCtCinvCt[0,0]
    if CtCw.shape[0]:
        ptr_CtCw = &CtCw[0,0]

    cdef FPnum *ptr_TransBtBinvBt = NULL
    cdef FPnum *ptr_BtB = NULL
    cdef FPnum *ptr_BeTBeChol = NULL
    if TransBtBinvBt.shape[0]:
        ptr_TransBtBinvBt = &TransBtBinvBt[0,0]
    if BtB.shape[0]:
        ptr_BtB = &BtB[0,0]
    if BeTBeChol.shape[0]:
        ptr_BeTBeChol = &BeTBeChol[0,0]

    cdef int m = max([m_x, m_u, m_ubin])
    cdef np.ndarray[FPnum, ndim=2] A = np.empty((m, k_user+k+k_main), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=1] biasA = np.empty(0, dtype=c_FPnum)
    if m == 0:
        return A, biasA
    cdef FPnum *ptr_biasA = NULL
    if user_bias:
        biasA = np.empty(m, dtype=c_FPnum)
        ptr_biasA = &biasA[0]

    cdef np.ndarray[FPnum, ndim=1] lam_unique = np.zeros(6, dtype=c_FPnum)
    cdef FPnum *ptr_lam_unique = NULL
    if lam_bias != lam:
        lam_unique[0] = lam_bias
        lam_unique[2] = lam
        ptr_lam_unique = &lam_unique[0]

    cdef int retval = factors_collective_explicit_multiple(
        &A[0,0], ptr_biasA, m,
        ptr_U, m_u, p,
        NA_as_zero_U, NA_as_zero_X,
        ptr_U_row, ptr_U_col, ptr_U_sp, nnz_U,
        ptr_U_csr_p, ptr_U_csr_i, ptr_U_csr,
        ptr_Ub, m_ubin, pbin,
        ptr_C, ptr_C_bin,
        glob_mean, ptr_biasB,
        ptr_U_colmeans,
        ptr_X, ptr_ixA, ptr_ixB, nnz,
        ptr_Xcsr_p, ptr_Xcsr_i, ptr_Xcsr,
        ptr_Xfull, n,
        ptr_weight,
        &B[0,0],
        k, k_user, k_item, k_main,
        lam, ptr_lam_unique,
        w_main, w_user,
        ptr_TransBtBinvBt,
        ptr_BtB,
        ptr_BeTBeChol,
        ptr_TransCtCinvCt,
        ptr_CtCw,
        ptr_B_plus_bias,
        nthreads
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return A, biasA

def call_factors_collective_implicit_multiple(
        np.ndarray[int, ndim=1] ixA,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[FPnum, ndim=1] X,
        np.ndarray[size_t, ndim=1] Xcsr_p,
        np.ndarray[int, ndim=1] Xcsr_i,
        np.ndarray[FPnum, ndim=1] Xcsr,
        np.ndarray[FPnum, ndim=2] U,
        np.ndarray[int, ndim=1] U_row,
        np.ndarray[int, ndim=1] U_col,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[size_t, ndim=1] U_csr_p,
        np.ndarray[int, ndim=1] U_csr_i,
        np.ndarray[FPnum, ndim=1] U_csr,
        np.ndarray[FPnum, ndim=1] U_colmeans,
        np.ndarray[FPnum, ndim=2] B,
        np.ndarray[FPnum, ndim=2] C,
        np.ndarray[FPnum, ndim=2] BeTBe,
        np.ndarray[FPnum, ndim=2] BtB,
        np.ndarray[FPnum, ndim=2] BeTBeChol,
        int n, int m_u, int m_x,
        int k, int k_user = 0, int k_item = 0, int k_main = 0,
        FPnum lam = 1e2, FPnum alpha = 40.,
        FPnum w_main_multiplier = 1.,
        FPnum w_user = 1., FPnum w_main = 1.,
        bint NA_as_zero_U = 0,
        int nthreads = 1
    ):
    cdef int m = max([m_u, m_x])

    cdef int *ptr_ixA = NULL
    cdef int *ptr_ixB = NULL
    cdef FPnum *ptr_X = NULL
    cdef size_t nnz = 0
    cdef size_t *ptr_Xcsr_p = NULL
    cdef int *ptr_Xcsr_i = NULL
    cdef FPnum *ptr_Xcsr = NULL
    if Xcsr.shape[0]:
        ptr_Xcsr_p = &Xcsr_p[0]
        ptr_Xcsr_i = &Xcsr_i[0]
        ptr_Xcsr = &Xcsr[0]
        nnz = Xcsr.shape[0]
    else:
        ptr_ixA = &ixA[0]
        ptr_ixB = &ixB[0]
        ptr_X = &X[0]
        nnz = X.shape[0]

    cdef FPnum *ptr_U = NULL
    cdef int *ptr_U_row = NULL
    cdef int *ptr_U_col = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    cdef FPnum *ptr_U_colmeans = NULL
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]
    if U_colmeans.shape[0]:
        ptr_U_colmeans = &U_colmeans[0]

    cdef size_t *ptr_U_csr_p = NULL
    cdef int *ptr_U_csr_i = NULL
    cdef FPnum *ptr_U_csr = NULL
    if U_csr.shape[0]:
        ptr_U_csr_p = &U_csr_p[0]
        ptr_U_csr_i = &U_csr_i[0]
        ptr_U_csr = &U_csr[0]

    cdef FPnum *ptr_C = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]

    cdef FPnum *ptr_BeTBe = NULL
    cdef FPnum *ptr_BtB = NULL
    cdef FPnum *ptr_BeTBeChol = NULL
    if BeTBe.shape[0]:
        ptr_BeTBe = &BeTBe[0,0]
    if BtB.shape[0]:
        ptr_BtB = &BtB[0,0]
    if BeTBeChol.shape[0]:
        ptr_BeTBeChol = &BeTBeChol[0,0]

    cdef np.ndarray[FPnum, ndim=2] A = np.empty((m, k_user+k+k_main), dtype=c_FPnum)
    if m == 0:
        return A

    cdef int retval = factors_collective_implicit_multiple(
        &A[0,0], m,
        ptr_U, m_u, C.shape[0],
        NA_as_zero_U,
        ptr_U_row, ptr_U_col, ptr_U_sp, nnz_U,
        ptr_U_csr_p, ptr_U_csr_i, ptr_U_csr,
        ptr_X, ptr_ixA, ptr_ixB, nnz,
        ptr_Xcsr_p, ptr_Xcsr_i, ptr_Xcsr,
        &B[0,0], n,
        ptr_C,
        ptr_U_colmeans,
        k, k_user, k_item, k_main,
        lam, alpha, w_main, w_user,
        w_main_multiplier,
        ptr_BeTBe,
        ptr_BtB,
        ptr_BeTBeChol,
        nthreads
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return A

def call_factors_offsets_explicit_multiple(
        np.ndarray[int, ndim=1] ixA,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[FPnum, ndim=1] X,
        np.ndarray[size_t, ndim=1] Xcsr_p,
        np.ndarray[int, ndim=1] Xcsr_i,
        np.ndarray[FPnum, ndim=1] Xcsr,
        np.ndarray[FPnum, ndim=1] W,
        np.ndarray[FPnum, ndim=2] Xfull,
        np.ndarray[FPnum, ndim=2] Wfull,
        np.ndarray[FPnum, ndim=2] U,
        np.ndarray[int, ndim=1] U_row,
        np.ndarray[int, ndim=1] U_col,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[size_t, ndim=1] U_csr_p,
        np.ndarray[int, ndim=1] U_csr_i,
        np.ndarray[FPnum, ndim=1] U_csr,
        np.ndarray[FPnum, ndim=1] biasB,
        np.ndarray[FPnum, ndim=2] Bm,
        np.ndarray[FPnum, ndim=2] Bm_plus_bias,
        np.ndarray[FPnum, ndim=2] C,
        np.ndarray[FPnum, ndim=1] C_bias,
        np.ndarray[FPnum, ndim=2] TransBtBinvBt,
        np.ndarray[FPnum, ndim=2] BtB,
        FPnum glob_mean,
        int m, int n,
        int k, int k_sec = 0, int k_main = 0,
        FPnum lam = 1e2, FPnum lam_bias = 1e2,
        FPnum w_user = 1.,
        bint user_bias = 1,
        bint exact = 0, bint output_a = 1,
        int nthreads = 1
    ):

    cdef FPnum *ptr_Xfull = NULL
    cdef FPnum *ptr_weight = NULL
    cdef int *ptr_ixA = NULL
    cdef int *ptr_ixB = NULL
    cdef FPnum *ptr_X = NULL
    cdef size_t nnz = 0
    cdef size_t *ptr_Xcsr_p = NULL
    cdef int *ptr_Xcsr_i = NULL
    cdef FPnum *ptr_Xcsr = NULL
    if Xfull.shape[0]:
        ptr_Xfull = &Xfull[0,0]
        if Wfull.shape[0]:
            ptr_weight = &Wfull[0,0]
    elif Xcsr.shape[0]:
        ptr_Xcsr_p = &Xcsr_p[0]
        ptr_Xcsr_i = &Xcsr_i[0]
        ptr_Xcsr = &Xcsr[0]
        nnz = Xcsr.shape[0]
        if W.shape[0]:
            ptr_weight = &W[0]
    elif X.shape[0]:
        ptr_ixA = &ixA[0]
        ptr_ixB = &ixB[0]
        ptr_X = &X[0]
        nnz = X.shape[0]
        if W.shape[0]:
            ptr_weight = &W[0]

    cdef FPnum *ptr_U = NULL
    cdef int *ptr_U_row = NULL
    cdef int *ptr_U_col = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]

    cdef size_t *ptr_U_csr_p = NULL
    cdef int *ptr_U_csr_i = NULL
    cdef FPnum *ptr_U_csr = NULL
    if U_csr.shape[0]:
        ptr_U_csr_p = &U_csr_p[0]
        ptr_U_csr_i = &U_csr_i[0]
        ptr_U_csr = &U_csr[0]

    cdef FPnum *ptr_C = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]

    cdef FPnum *ptr_biasB = NULL
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]

    cdef FPnum *ptr_TransBtBinvBt = NULL
    cdef FPnum *ptr_BtB = NULL
    if TransBtBinvBt.shape[0]:
        ptr_TransBtBinvBt = &TransBtBinvBt[0,0]
    if BtB.shape[0]:
        ptr_BtB = &BtB[0,0]

    cdef FPnum *ptr_Bm_plus_bias = NULL
    if Bm_plus_bias.shape[0]:
        ptr_Bm_plus_bias = &Bm_plus_bias[0,0]

    cdef FPnum *ptr_C_bias = NULL
    if C_bias.shape[0]:
        ptr_C_bias = &C_bias[0]

    cdef np.ndarray[FPnum, ndim=2] Am = np.empty((m, k_sec+k+k_main), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] A = np.empty((0,0), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=1] biasA = np.empty(0, dtype=c_FPnum)
    if m == 0:
        return Am, biasA, A
    cdef FPnum *ptr_biasA = NULL
    cdef FPnum *ptr_A = NULL
    if user_bias:
        biasA = np.empty(m, dtype=c_FPnum)
        ptr_biasA = &biasA[0]
    if output_a:
        A = np.empty((m, k+k_main), dtype=c_FPnum)
        ptr_A = &A[0,0]

    cdef np.ndarray[FPnum, ndim=1] lam_unique = np.zeros(6, dtype=c_FPnum)
    cdef FPnum *ptr_lam_unique = NULL
    if lam != lam_bias:
        lam_unique[0] = lam_bias
        lam_unique[2] = lam
        ptr_lam_unique = &lam_unique[0]

    cdef int p = C.shape[0]

    cdef int retval =  factors_offsets_explicit_multiple(
        &Am[0,0], ptr_biasA,
        ptr_A, m,
        ptr_U, p,
        ptr_U_row, ptr_U_col, ptr_U_sp, nnz_U,
        ptr_U_csr_p, ptr_U_csr_i, ptr_U_csr,
        ptr_X, ptr_ixA, ptr_ixB, nnz,
        ptr_Xcsr_p, ptr_Xcsr_i, ptr_Xcsr,
        ptr_Xfull, n,
        ptr_weight,
        &Bm[0,0], ptr_C,
        ptr_C_bias,
        glob_mean, ptr_biasB,
        k, k_sec, k_main,
        w_user,
        lam, ptr_lam_unique, exact,
        ptr_TransBtBinvBt,
        ptr_BtB,
        ptr_Bm_plus_bias,
        nthreads
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return Am, A, biasA

def call_factors_offsets_implicit_multiple(
        np.ndarray[int, ndim=1] ixA,
        np.ndarray[int, ndim=1] ixB,
        np.ndarray[FPnum, ndim=1] X,
        np.ndarray[size_t, ndim=1] Xcsr_p,
        np.ndarray[int, ndim=1] Xcsr_i,
        np.ndarray[FPnum, ndim=1] Xcsr,
        np.ndarray[FPnum, ndim=2] U,
        np.ndarray[int, ndim=1] U_row,
        np.ndarray[int, ndim=1] U_col,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[size_t, ndim=1] U_csr_p,
        np.ndarray[int, ndim=1] U_csr_i,
        np.ndarray[FPnum, ndim=1] U_csr,
        np.ndarray[FPnum, ndim=2] Bm,
        np.ndarray[FPnum, ndim=2] C,
        np.ndarray[FPnum, ndim=1] C_bias,
        np.ndarray[FPnum, ndim=2] BtB,
        int m, int n,
        int k,
        FPnum lam = 1e2, FPnum alpha = 1.,
        bint output_a = 1,
        int nthreads = 1
    ):

    cdef int *ptr_ixA = NULL
    cdef int *ptr_ixB = NULL
    cdef FPnum *ptr_X = NULL
    cdef size_t nnz = 0
    cdef size_t *ptr_Xcsr_p = NULL
    cdef int *ptr_Xcsr_i = NULL
    cdef FPnum *ptr_Xcsr = NULL
    if Xcsr.shape[0]:
        ptr_Xcsr_p = &Xcsr_p[0]
        ptr_Xcsr_i = &Xcsr_i[0]
        ptr_Xcsr = &Xcsr[0]
        nnz = Xcsr.shape[0]
    elif X.shape[0]:
        ptr_ixA = &ixA[0]
        ptr_ixB = &ixB[0]
        ptr_X = &X[0]
        nnz = X.shape[0]

    cdef FPnum *ptr_U = NULL
    cdef int *ptr_U_row = NULL
    cdef int *ptr_U_col = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]

    cdef size_t *ptr_U_csr_p = NULL
    cdef int *ptr_U_csr_i = NULL
    cdef FPnum *ptr_U_csr = NULL
    if U_csr.shape[0]:
        ptr_U_csr_p = &U_csr_p[0]
        ptr_U_csr_i = &U_csr_i[0]
        ptr_U_csr = &U_csr[0]

    cdef FPnum *ptr_C = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]

    cdef FPnum *ptr_BtB = NULL
    if BtB.shape[0]:
        ptr_BtB = &BtB[0,0]

    cdef FPnum *ptr_C_bias = NULL
    if C_bias.shape[0]:
        ptr_C_bias = &C_bias[0]

    cdef np.ndarray[FPnum, ndim=2] Am = np.empty((m, k), dtype=c_FPnum)
    cdef np.ndarray[FPnum, ndim=2] A = np.empty((0,0), dtype=c_FPnum)
    if m == 0:
        return Am, A
    cdef FPnum *ptr_A = NULL
    if output_a:
        A = np.empty((m, k), dtype=c_FPnum)
        ptr_A = &A[0,0]

    cdef int p = C.shape[0]

    cdef int retval =  factors_offsets_implicit_multiple(
        &Am[0,0], m,
        ptr_A,
        ptr_U, p,
        ptr_U_row, ptr_U_col, ptr_U_sp, nnz_U,
        ptr_U_csr_p, ptr_U_csr_i, ptr_U_csr,
        ptr_X, ptr_ixA, ptr_ixB, nnz,
        ptr_Xcsr_p, ptr_Xcsr_i, ptr_Xcsr,
        &Bm[0,0], ptr_C,
        ptr_C_bias,
        k,
        lam, alpha,
        ptr_BtB,
        nthreads
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return Am, A
    
def call_impute_X_collective_explicit(
        np.ndarray[FPnum, ndim=2] Xfull,
        np.ndarray[FPnum, ndim=2] Wfull,
        np.ndarray[FPnum, ndim=2] U,
        np.ndarray[int, ndim=1] U_row,
        np.ndarray[int, ndim=1] U_col,
        np.ndarray[FPnum, ndim=1] U_sp,
        np.ndarray[size_t, ndim=1] U_csr_p,
        np.ndarray[int, ndim=1] U_csr_i,
        np.ndarray[FPnum, ndim=1] U_csr,
        np.ndarray[FPnum, ndim=2] Ub,
        np.ndarray[FPnum, ndim=1] U_colmeans,
        np.ndarray[FPnum, ndim=1] biasB,
        np.ndarray[FPnum, ndim=2] B,
        np.ndarray[FPnum, ndim=2] B_plus_bias,
        np.ndarray[FPnum, ndim=2] C,
        np.ndarray[FPnum, ndim=2] C_bin,
        np.ndarray[FPnum, ndim=2] TransBtBinvBt,
        np.ndarray[FPnum, ndim=2] BtB,
        np.ndarray[FPnum, ndim=2] BeTBeChol,
        np.ndarray[FPnum, ndim=2] TransCtCinvCt,
        np.ndarray[FPnum, ndim=2] CtCw,
        int m_u,
        FPnum glob_mean,
        int k, int k_user = 0, int k_item = 0, int k_main = 0,
        FPnum lam = 1e2, FPnum lam_bias = 1e2,
        FPnum w_user = 1., FPnum w_main = 1.,
        bint user_bias = 1,
        bint NA_as_zero_U = 0,
        int nthreads = 1
    ):
    
    cdef int n = B.shape[0]
    cdef int p = C.shape[0]
    cdef int pbin = C_bin.shape[0]
    cdef int m = Xfull.shape[0]
    if min(m, n) <= 0:
        raise ValueError("Invalid input dimensions.")
    cdef np.ndarray[FPnum, ndim=1] lam_unique = np.zeros(6, dtype=c_FPnum)
    cdef FPnum *ptr_lam_unique = NULL
    if (lam != lam_bias):
        lam_unique[0] = lam_bias
        lam_unique[2] = lam
        ptr_lam_unique = &lam_unique[0]

    cdef FPnum *ptr_Xfull = &Xfull[0,0]
    cdef FPnum *ptr_weight = NULL
    if Wfull.shape[0]:
        ptr_weight = &Wfull[0,0]

    cdef FPnum *ptr_U = NULL
    cdef int *ptr_U_row = NULL
    cdef int *ptr_U_col = NULL
    cdef FPnum *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    cdef FPnum *ptr_U_colmeans = NULL
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]
    if U_colmeans.shape[0]:
        ptr_U_colmeans = &U_colmeans[0]

    cdef size_t *ptr_U_csr_p = NULL
    cdef int *ptr_U_csr_i = NULL
    cdef FPnum *ptr_U_csr = NULL
    if U_csr.shape[0]:
        ptr_U_csr_p = &U_csr_p[0]
        ptr_U_csr_i = &U_csr_i[0]
        ptr_U_csr = &U_csr[0]

    cdef FPnum *ptr_Ub = NULL
    cdef int m_ubin = 0
    if Ub.shape[0]:
        ptr_Ub = &Ub[0,0]
        m_ubin = Ub.shape[0]

    cdef FPnum *ptr_B = &B[0,0]
    cdef FPnum *ptr_biasB = NULL
    cdef FPnum *ptr_B_plus_bias = NULL
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]
    if B_plus_bias.shape[0]:
        ptr_B_plus_bias = &B_plus_bias[0,0]

    cdef FPnum *ptr_C = NULL
    cdef FPnum *ptr_C_bin = NULL
    cdef FPnum *ptr_TransCtCinvCt = NULL
    cdef FPnum *ptr_CtCw = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]
    if C_bin.shape[0]:
        ptr_C_bin = &C_bin[0,0]
    if TransCtCinvCt.shape[0]:
        ptr_TransCtCinvCt = &TransCtCinvCt[0,0]
    if CtCw.shape[0]:
        ptr_CtCw = &CtCw[0,0]

    cdef FPnum *ptr_TransBtBinvBt = NULL
    cdef FPnum *ptr_BtB = NULL
    cdef FPnum *ptr_BeTBeChol = NULL
    if TransBtBinvBt.shape[0]:
        ptr_TransBtBinvBt = &TransBtBinvBt[0,0]
    if BtB.shape[0]:
        ptr_BtB = &BtB[0,0]
    if BeTBeChol.shape[0]:
        ptr_BeTBeChol = &BeTBeChol[0,0]

    cdef int retval = impute_X_collective_explicit(
        m, user_bias,
        ptr_U, m_u, p,
        NA_as_zero_U,
        ptr_U_row, ptr_U_col, ptr_U_sp, nnz_U,
        ptr_U_csr_p, ptr_U_csr_i, ptr_U_csr,
        ptr_Ub, m_ubin, pbin,
        ptr_C, ptr_C_bin,
        glob_mean, ptr_biasB,
        ptr_U_colmeans,
        ptr_Xfull, n,
        ptr_weight,
        ptr_B,
        k, k_user, k_item, k_main,
        lam, ptr_lam_unique,
        w_main, w_user,
        ptr_TransBtBinvBt,
        ptr_BtB,
        ptr_BeTBeChol,
        ptr_TransCtCinvCt,
        ptr_CtCw,
        ptr_B_plus_bias,
        nthreads
    )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return Xfull
