import numpy as np
cimport numpy as np
from cython cimport boundscheck, nonecheck, wraparound
import ctypes

ctypedef int int_t

# ctypedef double real_t

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
#     from scipy.linalg.cython_blas cimport sger as sger_

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
#     from scipy.linalg.cython_blas cimport dger as dger_

#     from scipy.linalg.cython_lapack cimport dlacpy as dlacpy_
#     from scipy.linalg.cython_lapack cimport dposv as dposv_
#     from scipy.linalg.cython_lapack cimport dlarnv as dlarnv_
#     from scipy.linalg.cython_lapack cimport dpotrf as dpotrf_
#     from scipy.linalg.cython_lapack cimport dpotrs as dpotrs_
#     from scipy.linalg.cython_lapack cimport dgels as dgels_



### TODO: this module should move from doing operations in Python to
### using the new designated C functions for each type of prediction.


cdef extern from "cmfrec.h":
    int_t fit_collective_explicit_lbfgs_internal(
        real_t *values, bint reset_values,
        real_t *glob_mean,
        real_t *U_colmeans, real_t *I_colmeans,
        int_t m, int_t n, int_t k,
        int_t ixA[], int_t ixB[], real_t *X, size_t nnz,
        real_t *Xfull,
        real_t *weight,
        bint user_bias, bint item_bias, bint center,
        real_t lam, real_t *lam_unique,
        real_t *U, int_t m_u, int_t p,
        real_t *II, int_t n_i, int_t q,
        real_t *Ub, int_t m_ubin, int_t pbin,
        real_t *Ib, int_t n_ibin, int_t qbin,
        int_t U_row[], int_t U_col[], real_t *U_sp, size_t nnz_U,
        int_t I_row[], int_t I_col[], real_t *I_sp, size_t nnz_I,
        int_t k_main, int_t k_user, int_t k_item,
        real_t w_main, real_t w_user, real_t w_item,
        int_t n_corr_pairs, size_t maxiter, int_t seed,
        int nthreads, bint prefer_onepass,
        bint verbose, int_t print_every, bint handle_interrupt,
        int_t *niter, int_t *nfev,
        real_t *B_plus_bias
    ) nogil

    int_t fit_offsets_explicit_lbfgs_internal(
        real_t *values, bint reset_values,
        real_t *glob_mean,
        int_t m, int_t n, int_t k,
        int_t ixA[], int_t ixB[], real_t *X, size_t nnz,
        real_t *Xfull,
        real_t *weight,
        bint user_bias, bint item_bias, bint center,
        bint add_intercepts,
        real_t lam, real_t *lam_unique,
        real_t *U, int_t p,
        real_t *II, int_t q,
        int_t U_row[], int_t U_col[], real_t *U_sp, size_t nnz_U,
        int_t I_row[], int_t I_col[], real_t *I_sp, size_t nnz_I,
        int_t k_main, int_t k_sec,
        real_t w_user, real_t w_item,
        int_t n_corr_pairs, size_t maxiter, int_t seed,
        int nthreads, bint prefer_onepass,
        bint verbose, int_t print_every, bint handle_interrupt,
        int_t *niter, int_t *nfev,
        real_t *Am, real_t *Bm,
        real_t *B_plus_bias
    ) nogil

    int_t fit_collective_explicit_als(
        real_t *biasA, real_t *biasB,
        real_t *A, real_t *B,
        real_t *C, real_t *D,
        real_t *Ai, real_t *Bi,
        bint add_implicit_features,
        bint reset_values, int_t seed,
        real_t *glob_mean,
        real_t *U_colmeans, real_t *I_colmeans,
        int_t m, int_t n, int_t k,
        int_t ixA[], int_t ixB[], real_t *X, size_t nnz,
        real_t *Xfull,
        real_t *weight,
        bint user_bias, bint item_bias, bint center,
        real_t lam, real_t *lam_unique,
        real_t l1_lam, real_t *l1_lam_unique,
        bint scale_lam, bint scale_lam_sideinfo, bint scale_bias_const,
        real_t *scaling_biasA, real_t *scaling_biasB,
        real_t *U, int_t m_u, int_t p,
        real_t *II, int_t n_i, int_t q,
        int_t U_row[], int_t U_col[], real_t *U_sp, size_t nnz_U,
        int_t I_row[], int_t I_col[], real_t *I_sp, size_t nnz_I,
        bint NA_as_zero_X, bint NA_as_zero_U, bint NA_as_zero_I,
        int_t k_main, int_t k_user, int_t k_item,
        real_t w_main, real_t w_user, real_t w_item, real_t w_implicit,
        int_t niter, int nthreads, bint verbose, bint handle_interrupt,
        bint use_cg, int_t max_cg_steps, bint finalize_chol,
        bint nonneg, int_t max_cd_steps, bint nonneg_C, bint nonneg_D,
        bint precompute_for_predictions,
        bint include_all_X,
        real_t *B_plus_bias,
        real_t *precomputedBtB,
        real_t *precomputedTransBtBinvBt,
        real_t *precomputedBtXbias,
        real_t *precomputedBeTBeChol,
        real_t *precomputedBiTBi,
        real_t *precomputedTransCtCinvCt,
        real_t *precomputedCtCw,
        real_t *precomputedCtUbias
    ) nogil

    int_t fit_collective_implicit_als(
        real_t *A, real_t *B,
        real_t *C, real_t *D,
        bint reset_values, int_t seed,
        real_t *U_colmeans, real_t *I_colmeans,
        int_t m, int_t n, int_t k,
        int_t ixA[], int_t ixB[], real_t *X, size_t nnz,
        real_t lam, real_t *lam_unique,
        real_t l1_lam, real_t *l1_lam_unique,
        real_t *U, int_t m_u, int_t p,
        real_t *II, int_t n_i, int_t q,
        int_t U_row[], int_t U_col[], real_t *U_sp, size_t nnz_U,
        int_t I_row[], int_t I_col[], real_t *I_sp, size_t nnz_I,
        bint NA_as_zero_U, bint NA_as_zero_I,
        int_t k_main, int_t k_user, int_t k_item,
        real_t w_main, real_t w_user, real_t w_item,
        real_t *w_main_multiplier,
        real_t alpha, bint adjust_weight, bint apply_log_transf,
        int_t niter, int nthreads, bint verbose, bint handle_interrupt,
        bint use_cg, int_t max_cg_steps, bint finalize_chol,
        bint nonneg, int_t max_cd_steps, bint nonneg_C, bint nonneg_D,
        bint precompute_for_predictions,
        real_t *precomputedBtB,
        real_t *precomputedBeTBe,
        real_t *precomputedBeTBeChol,
        real_t *precomputedCtUbias
    ) nogil

    int_t fit_offsets_als(
        real_t *biasA, real_t *biasB,
        real_t *A, real_t *B,
        real_t *C, real_t *C_bias,
        real_t *D, real_t *D_bias,
        bint reset_values, int_t seed,
        real_t *glob_mean,
        int_t m, int_t n, int_t k,
        int_t ixA[], int_t ixB[], real_t *X, size_t nnz,
        real_t *Xfull,
        real_t *weight,
        bint user_bias, bint item_bias, bint center, bint add_intercepts,
        real_t lam,
        real_t *U, int_t p,
        real_t *II, int_t q,
        bint implicit, bint NA_as_zero_X,
        real_t alpha, bint apply_log_transf,
        int_t niter,
        int nthreads, bint use_cg,
        int_t max_cg_steps, bint finalize_chol,
        bint verbose, bint handle_interrupt,
        bint precompute_for_predictions,
        real_t *Am, real_t *Bm,
        real_t *Bm_plus_bias,
        real_t *precomputedBtB,
        real_t *precomputedTransBtBinvBt
    ) nogil

    int_t precompute_collective_explicit(
        real_t *B, int_t n, int_t n_max, bint include_all_X,
        real_t *C, int_t p,
        real_t *Bi, bint add_implicit_features,
        real_t *biasB, real_t glob_mean, bint NA_as_zero_X,
        real_t *U_colmeans, bint NA_as_zero_U,
        int_t k, int_t k_user, int_t k_item, int_t k_main,
        bint user_bias,
        bint nonneg,
        real_t lam, real_t *lam_unique,
        bint scale_lam, bint scale_lam_sideinfo,
        bint scale_bias_const, real_t scaling_biasA,
        real_t w_main, real_t w_user, real_t w_item,
        real_t *B_plus_bias,
        real_t *BtB,
        real_t *TransBtBinvBt,
        real_t *BtXbias,
        real_t *BeTBeChol,
        real_t *BiTBi,
        real_t *TransCtCinvCt,
        real_t *CtCw,
        real_t *CtUbias
    ) nogil

    int_t precompute_collective_implicit(
        real_t *B, int_t n,
        real_t *C, int_t p,
        real_t *U_colmeans, bint NA_as_zero_U,
        int_t k, int_t k_user, int_t k_item, int_t k_main,
        real_t lam, real_t w_main, real_t w_user, real_t w_main_multiplier,
        bint nonneg,
        bint extra_precision,
        real_t *BtB,
        real_t *BeTBe,
        real_t *BeTBeChol,
        real_t *CtUbias
    ) nogil

    int_t offsets_factors_cold(
        real_t *a_vec,
        real_t *u_vec,
        int_t u_vec_ixB[], real_t *u_vec_sp, size_t nnz_u_vec,
        real_t *C, int_t p,
        real_t *C_bias,
        int_t k, int_t k_sec, int_t k_main,
        real_t w_user
    ) nogil

    void predict_multiple(
        real_t *A, int_t k_user,
        real_t *B, int_t k_item,
        real_t *biasA, real_t *biasB,
        real_t glob_mean,
        int_t k, int_t k_main,
        int_t m, int_t n,
        int_t predA[], int_t predB[], size_t nnz,
        real_t *outp,
        int nthreads
    ) nogil

    int_t predict_X_old_collective_explicit(
        int_t row[], int_t col[], real_t *predicted, size_t n_predict,
        real_t *A, real_t *biasA,
        real_t *B, real_t *biasB,
        real_t glob_mean,
        int_t k, int_t k_user, int_t k_item, int_t k_main,
        int_t m, int_t n_max,
        int nthreads
    ) nogil

    int_t topN(
        real_t *a_vec, int_t k_user,
        real_t *B, int_t k_item,
        real_t *biasB,
        real_t glob_mean, real_t biasA,
        int_t k, int_t k_main,
        int_t *include_ix, int_t n_include,
        int_t *exclude_ix, int_t n_exclude,
        int_t *outp_ix, real_t *outp_score,
        int_t n_top, int_t n, int nthreads
    ) nogil

    int_t topN_new_content_based(
        int_t k, int_t n_new,
        real_t *u_vec, int_t p,
        real_t *u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
        real_t *II, int_t q,
        int_t I_row[], int_t I_col[], real_t *I_sp, size_t nnz_I,
        size_t I_csr_p[], int_t I_csr_i[], real_t *I_csr,
        real_t *C, real_t *C_bias,
        real_t *D, real_t *D_bias,
        real_t glob_mean,
        int_t *outp_ix, real_t *outp_score,
        int_t n_top, int nthreads
    ) nogil

    int_t fit_most_popular(
        real_t *biasA, real_t *biasB,
        real_t *glob_mean,
        real_t lam_user, real_t lam_item,
        bint scale_lam, bint scale_bias_const,
        real_t alpha,
        int_t m, int_t n,
        int_t ixA[], int_t ixB[], real_t *X, size_t nnz,
        real_t *Xfull,
        real_t *weight,
        bint implicit, bint adjust_weight, bint apply_log_transf,
        bint nonneg, bint NA_as_zero,
        real_t *w_main_multiplier,
        int nthreads
    ) nogil

    int_t impute_X_collective_explicit(
        int_t m, bint user_bias,
        real_t *U, int_t m_u, int_t p,
        bint NA_as_zero_U,
        bint nonneg,
        int_t U_row[], int_t U_col[], real_t *U_sp, size_t nnz_U,
        size_t U_csr_p[], int_t U_csr_i[], real_t *U_csr,
        real_t *Ub, int_t m_ubin, int_t pbin,
        real_t *C, real_t *Cb,
        real_t glob_mean, real_t *biasB,
        real_t *U_colmeans,
        real_t *Xfull, int_t n,
        real_t *weight,
        real_t *B,
        real_t *Bi, bint add_implicit_features,
        int_t k, int_t k_user, int_t k_item, int_t k_main,
        real_t lam, real_t *lam_unique,
        real_t l1_lam, real_t *l1_lam_unique,
        bint scale_lam, bint scale_lam_sideinfo,
        bint scale_bias_const, real_t scaling_biasA,
        real_t w_main, real_t w_user, real_t w_implicit,
        int_t n_max, bint include_all_X,
        real_t *BtB,
        real_t *TransBtBinvBt,
        real_t *BeTBeChol,
        real_t *BiTBi,
        real_t *TransCtCinvCt,
        real_t *CtCw,
        real_t *CtUbias,
        real_t *B_plus_bias,
        int nthreads
    ) nogil

    int_t predict_X_old_content_based(
        real_t *predicted, size_t n_predict,
        int_t m_new, int_t k,
        int_t row[],
        int_t col[],
        int_t m_orig, int_t n_orig,
        real_t *U, int_t p,
        int_t U_row[], int_t U_col[], real_t *U_sp, size_t nnz_U,
        size_t U_csr_p[], int_t U_csr_i[], real_t *U_csr,
        real_t *C, real_t *C_bias,
        real_t *Bm, real_t *biasB,
        real_t glob_mean,
        int nthreads
    ) nogil

    int_t predict_X_new_content_based(
        real_t *predicted, size_t n_predict,
        int_t m_new, int_t n_new, int_t k,
        int_t row[], int_t col[],
        real_t *U, int_t p,
        int_t U_row[], int_t U_col[], real_t *U_sp, size_t nnz_U,
        size_t U_csr_p[], int_t U_csr_i[], real_t *U_csr,
        real_t *II, int_t q,
        int_t I_row[], int_t I_col[], real_t *I_sp, size_t nnz_I,
        size_t I_csr_p[], int_t I_csr_i[], real_t *I_csr,
        real_t *C, real_t *C_bias,
        real_t *D, real_t *D_bias,
        real_t glob_mean,
        int nthreads
    ) nogil

    int_t factors_content_based_single(
        real_t *a_vec, int_t k,
        real_t *u_vec, int_t p,
        real_t *u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
        real_t *C, real_t *C_bias
    ) nogil

    int_t fit_content_based_lbfgs(
        real_t *biasA, real_t *biasB,
        real_t *C, real_t *C_bias,
        real_t *D, real_t *D_bias,
        bint start_with_ALS, bint reset_values, int_t seed,
        real_t *glob_mean,
        int_t m, int_t n, int_t k,
        int_t ixA[], int_t ixB[], real_t *X, size_t nnz,
        real_t *Xfull,
        real_t *weight,
        bint user_bias, bint item_bias,
        bint add_intercepts,
        real_t lam, real_t *lam_unique,
        real_t *U, int_t p,
        real_t *II, int_t q,
        int_t U_row[], int_t U_col[], real_t *U_sp, size_t nnz_U,
        int_t I_row[], int_t I_col[], real_t *I_sp, size_t nnz_I,
        int_t n_corr_pairs, size_t maxiter,
        int nthreads, bint prefer_onepass,
        bint verbose, int_t print_every, bint handle_interrupt,
        int_t *niter, int_t *nfev,
        real_t *Am, real_t *Bm
    ) nogil

    int_t factors_collective_explicit_single(
        real_t *a_vec, real_t *a_bias,
        real_t *u_vec, int_t p,
        real_t *u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
        real_t *u_bin_vec, int_t pbin,
        bint NA_as_zero_U, bint NA_as_zero_X,
        bint nonneg,
        real_t *C, real_t *Cb,
        real_t glob_mean, real_t *biasB,
        real_t *U_colmeans,
        real_t *Xa, int_t ixB[], size_t nnz,
        real_t *Xa_dense, int_t n,
        real_t *weight,
        real_t *B,
        real_t *Bi, bint add_implicit_features,
        int_t k, int_t k_user, int_t k_item, int_t k_main,
        real_t lam, real_t *lam_unique,
        real_t l1_lam, real_t *l1_lam_unique,
        bint scale_lam, bint scale_lam_sideinfo,
        bint scale_bias_const, real_t scaling_biasA,
        real_t w_main, real_t w_user, real_t w_implicit,
        int_t n_max, bint include_all_X,
        real_t *BtB,
        real_t *TransBtBinvBt,
        real_t *BtXbias,
        real_t *BeTBeChol,
        real_t *BiTBi,
        real_t *CtCw,
        real_t *TransCtCinvCt,
        real_t *CtUbias,
        real_t *B_plus_bias
    ) nogil

    int_t factors_collective_implicit_single(
        real_t *a_vec,
        real_t *u_vec, int_t p,
        real_t *u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
        bint NA_as_zero_U,
        bint nonneg,
        real_t *U_colmeans,
        real_t *B, int_t n, real_t *C,
        real_t *Xa, int_t ixB[], size_t nnz,
        int_t k, int_t k_user, int_t k_item, int_t k_main,
        real_t lam, real_t l1_lam, real_t alpha, real_t w_main, real_t w_user,
        real_t w_main_multiplier,
        bint apply_log_transf,
        real_t *BeTBe,
        real_t *BtB,
        real_t *BeTBeChol,
        real_t *CtUbias
    ) nogil

    int_t factors_collective_explicit_multiple(
        real_t *A, real_t *biasA, int_t m,
        real_t *U, int_t m_u, int_t p,
        bint NA_as_zero_U, bint NA_as_zero_X,
        bint nonneg,
        int_t U_row[], int_t U_col[], real_t *U_sp, size_t nnz_U,
        size_t U_csr_p[], int_t U_csr_i[], real_t *U_csr,
        real_t *Ub, int_t m_ubin, int_t pbin,
        real_t *C, real_t *Cb,
        real_t glob_mean, real_t *biasB,
        real_t *U_colmeans,
        real_t *X, int_t ixA[], int_t ixB[], size_t nnz,
        size_t *Xcsr_p, int_t *Xcsr_i, real_t *Xcsr,
        real_t *Xfull, int_t n,
        real_t *weight,
        real_t *B,
        real_t *Bi, bint add_implicit_features,
        int_t k, int_t k_user, int_t k_item, int_t k_main,
        real_t lam, real_t *lam_unique,
        real_t l1_lam, real_t *l1_lam_unique,
        bint scale_lam, bint scale_lam_sideinfo,
        bint scale_bias_const, real_t scaling_biasA,
        real_t w_main, real_t w_user, real_t w_implicit,
        int_t n_max, bint include_all_X,
        real_t *BtB,
        real_t *TransBtBinvBt,
        real_t *BtXbias,
        real_t *BeTBeChol,
        real_t *BiTBi,
        real_t *TransCtCinvCt,
        real_t *CtCw,
        real_t *CtUbias,
        real_t *B_plus_bias,
        int nthreads
    ) nogil

    int_t factors_collective_implicit_multiple(
        real_t *A, int_t m,
        real_t *U, int_t m_u, int_t p,
        bint NA_as_zero_U,
        bint nonneg,
        int_t U_row[], int_t U_col[], real_t *U_sp, size_t nnz_U,
        size_t U_csr_p[], int_t U_csr_i[], real_t *U_csr,
        real_t *X, int_t ixA[], int_t ixB[], size_t nnz,
        size_t *Xcsr_p, int_t *Xcsr_i, real_t *Xcsr,
        real_t *B, int_t n,
        real_t *C,
        real_t *col_means,
        int_t k, int_t k_user, int_t k_item, int_t k_main,
        real_t lam, real_t l1_lam, real_t alpha, real_t w_main, real_t w_user,
        real_t w_main_multiplier,
        bint apply_log_transf,
        real_t *BeTBe,
        real_t *BtB,
        real_t *BeTBeChol,
        real_t *CtUbias,
        int nthreads
    ) nogil

    int_t factors_offsets_explicit_single(
        real_t *a_vec, real_t *a_bias, real_t *output_a,
        real_t *u_vec, int_t p,
        real_t *u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
        real_t *Xa, int_t ixB[], size_t nnz,
        real_t *Xa_dense, int_t n,
        real_t *weight,
        real_t *Bm, real_t *C,
        real_t *C_bias,
        real_t glob_mean, real_t *biasB,
        int_t k, int_t k_sec, int_t k_main,
        real_t w_user,
        real_t lam, real_t *lam_unique,
        bint exact,
        real_t *precomputedTransBtBinvBt,
        real_t *precomputedBtB,
        real_t *Bm_plus_bias
    ) nogil

    int_t factors_offsets_implicit_single(
        real_t *a_vec,
        real_t *u_vec, int_t p,
        real_t *u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
        real_t *Xa, int_t ixB[], size_t nnz,
        real_t *Bm, real_t *C,
        real_t *C_bias,
        int_t k, int_t n,
        real_t lam, real_t alpha,
        bint apply_log_transf,
        real_t *precomputedBtB,
        real_t *output_a
    ) nogil

    int_t factors_offsets_explicit_multiple(
        real_t *Am, real_t *biasA,
        real_t *A, int_t m,
        real_t *U, int_t p,
        int_t U_row[], int_t U_col[], real_t *U_sp, size_t nnz_U,
        size_t U_csr_p[], int_t U_csr_i[], real_t *U_csr,
        real_t *X, int_t ixA[], int_t ixB[], size_t nnz,
        size_t *Xcsr_p, int_t *Xcsr_i, real_t *Xcsr,
        real_t *Xfull, int_t n,
        real_t *weight,
        real_t *Bm, real_t *C,
        real_t *C_bias,
        real_t glob_mean, real_t *biasB,
        int_t k, int_t k_sec, int_t k_main,
        real_t w_user,
        real_t lam, real_t *lam_unique, bint exact,
        real_t *precomputedTransBtBinvBt,
        real_t *precomputedBtB,
        real_t *Bm_plus_bias,
        int nthreads
    ) nogil

    int_t factors_offsets_implicit_multiple(
        real_t *Am, int_t m,
        real_t *A,
        real_t *U, int_t p,
        int_t U_row[], int_t U_col[], real_t *U_sp, size_t nnz_U,
        size_t U_csr_p[], int_t U_csr_i[], real_t *U_csr,
        real_t *X, int_t ixA[], int_t ixB[], size_t nnz,
        size_t *Xcsr_p, int_t *Xcsr_i, real_t *Xcsr,
        real_t *Bm, real_t *C,
        real_t *C_bias,
        int_t k, int_t n,
        real_t lam, real_t alpha,
        bint apply_log_transf,
        real_t *precomputedBtB,
        int nthreads
    ) nogil


def call_fit_collective_explicit_lbfgs(
        np.ndarray[int_t, ndim=1] ixA,
        np.ndarray[int_t, ndim=1] ixB,
        np.ndarray[real_t, ndim=1] X,
        np.ndarray[real_t, ndim=1] W,
        np.ndarray[real_t, ndim=2] Xfull,
        np.ndarray[real_t, ndim=2] Wfull,
        np.ndarray[real_t, ndim=2] U,
        np.ndarray[int_t, ndim=1] U_row,
        np.ndarray[int_t, ndim=1] U_col,
        np.ndarray[real_t, ndim=1] U_sp,
        np.ndarray[real_t, ndim=2] Ub,
        np.ndarray[real_t, ndim=2] I,
        np.ndarray[int_t, ndim=1] I_row,
        np.ndarray[int_t, ndim=1] I_col,
        np.ndarray[real_t, ndim=1] I_sp,
        np.ndarray[real_t, ndim=2] Ib,
        int_t m, int_t n, int_t m_u, int_t n_i, int_t p, int_t q,
        int_t k=50, int_t k_user=0, int_t k_item=0, int_t k_main=0,
        real_t w_main=1., real_t w_user=1., real_t w_item=1.,
        bint user_bias=1, bint item_bias=1, bint center=1,
        real_t lam=1e2,
        np.ndarray[real_t, ndim=1] lam_unique=np.empty(0, dtype=c_real_t),
        bint verbose=1, int_t print_every=10,
        int_t n_corr_pairs=5, int_t maxiter=400,
        int nthreads=1, bint prefer_onepass=0,
        int_t seed=1, bint handle_interrupt=1
    ):

    cdef real_t *ptr_Xfull = NULL
    cdef real_t *ptr_weight = NULL
    cdef int_t *ptr_ixA = NULL
    cdef int_t *ptr_ixB = NULL
    cdef real_t *ptr_X = NULL
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

    cdef real_t *ptr_U = NULL
    cdef int_t *ptr_U_row = NULL
    cdef int_t *ptr_U_col = NULL
    cdef real_t *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    cdef np.ndarray[real_t, ndim=1] U_colmeans = np.empty(p, dtype=c_real_t)
    cdef real_t *ptr_U_colmeans = NULL
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]
    if U.shape[0] or U_sp.shape[0]:
        ptr_U_colmeans = &U_colmeans[0]

    cdef real_t *ptr_I = NULL
    cdef int_t *ptr_I_row = NULL
    cdef int_t *ptr_I_col = NULL
    cdef real_t *ptr_I_sp = NULL
    cdef size_t nnz_I = 0
    cdef np.ndarray[real_t, ndim=1] I_colmeans = np.empty(q, dtype=c_real_t)
    cdef real_t *ptr_I_colmeans = NULL
    if I.shape[0]:
        ptr_I = &I[0,0]
    elif I_sp.shape[0]:
        ptr_I_row = &I_row[0]
        ptr_I_col = &I_col[0]
        ptr_I_sp = &I_sp[0]
        nnz_I = I_sp.shape[0]
    if I.shape[0] or I_sp.shape[0]:
        ptr_I_colmeans = &I_colmeans[0]

    cdef real_t *ptr_Ub = NULL
    cdef int_t m_ubin = 0
    cdef int_t pbin = 0
    if Ub.shape[0]:
        ptr_Ub = &Ub[0,0]
        m_ubin = Ub.shape[0]
        pbin = Ub.shape[1]

    cdef real_t *ptr_Ib = NULL
    cdef int_t n_ibin = 0
    cdef int_t qbin = 0
    if Ib.shape[0]:
        ptr_Ib = &Ib[0,0]
        n_ibin = Ib.shape[0]
        qbin = Ib.shape[1]

    cdef real_t *ptr_lam_unique = NULL
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
    cdef np.ndarray[real_t, ndim=1] values = rs.standard_normal(size = nvars, dtype = c_real_t) / 100

    cdef np.ndarray[real_t, ndim=2] B_plus_bias = np.empty((0,0), dtype=c_real_t)
    cdef real_t *ptr_B_plus_bias = NULL
    if user_bias:
        B_plus_bias = np.empty((max(n, n_i, n_ibin), k_item+k+k_main+1), dtype=c_real_t)
        ptr_B_plus_bias = &B_plus_bias[0,0]

    cdef real_t glob_mean
    cdef int_t niter, nfev
    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = fit_collective_explicit_lbfgs_internal(
            &values[0], 0,
            &glob_mean,
            ptr_U_colmeans, ptr_I_colmeans,
            m, n, k,
            ptr_ixA, ptr_ixB, ptr_X, nnz,
            ptr_Xfull,
            ptr_weight,
            user_bias, item_bias, center,
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
    elif (retval == 3) and (not handle_interrupt):
        raise InterruptedError("Procedure was interrupted")

    return glob_mean, U_colmeans, I_colmeans, values, niter, nfev, B_plus_bias

def unpack_values_lbfgs_collective(
        np.ndarray[real_t, ndim=1] values,
        bint user_bias, bint item_bias,
        size_t k, size_t k_user, size_t k_item, size_t k_main,
        size_t m, size_t n, size_t p, size_t q,
        size_t pbin, size_t qbin,
        size_t m_u, size_t n_i, size_t m_ubin, size_t n_ibin
    ):

    cdef np.ndarray[real_t, ndim=1] biasA, biasB
    cdef np.ndarray[real_t, ndim=2] A, B, C, Cbin, D, Dbin

    cdef size_t edge = 0
    if user_bias:
        biasA = values[:max([m, m_u, m_ubin])]
        edge += max([m, m_u, m_ubin])
    else:
        biasA = np.empty(0, dtype=c_real_t)
    if item_bias:
        biasB = values[edge:edge + max([n, n_i, n_ibin])]
        edge += max([n, n_i, n_ibin])
    else:
        biasB = np.empty(0, dtype=c_real_t)
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
        C = np.empty((0,0), dtype=c_real_t)
    if pbin > 0:
        Cbin = values[edge:edge + pbin*(k_user+k)].reshape((pbin, k_user+k))
        edge += pbin*(k_user+k)
    else:
        Cbin = np.empty((0,0), dtype=c_real_t)
    if q > 0:
        D = values[edge:edge + q*(k_item+k)].reshape((q, k_item+k))
        edge += q*(k_item+k)
    else:
        D = np.empty((0,0), dtype=c_real_t)
    if qbin > 0:
        Dbin = values[edge:edge + qbin*(k_item+k)].reshape((qbin, k_item+k))
        edge += qbin*(k_item+k)
    else:
        Dbin = np.empty((0,0), dtype=c_real_t)

    return biasA, biasB, A, B, C, Cbin, D, Dbin

def call_fit_offsets_explicit_lbfgs_internal(
        np.ndarray[int_t, ndim=1] ixA,
        np.ndarray[int_t, ndim=1] ixB,
        np.ndarray[real_t, ndim=1] X,
        np.ndarray[real_t, ndim=1] W,
        np.ndarray[real_t, ndim=2] Xfull,
        np.ndarray[real_t, ndim=2] Wfull,
        np.ndarray[real_t, ndim=2] U,
        np.ndarray[int_t, ndim=1] U_row,
        np.ndarray[int_t, ndim=1] U_col,
        np.ndarray[real_t, ndim=1] U_sp,
        np.ndarray[real_t, ndim=2] I,
        np.ndarray[int_t, ndim=1] I_row,
        np.ndarray[int_t, ndim=1] I_col,
        np.ndarray[real_t, ndim=1] I_sp,
        int_t m, int_t n, int_t p, int_t q,
        int_t k=50, int_t k_sec=0, int_t k_main=0,
        real_t w_user=1., real_t w_item=1.,
        bint user_bias=1, bint item_bias=1, bint center=1,
        bint add_intercepts=1,
        real_t lam=1e2,
        np.ndarray[real_t, ndim=1] lam_unique=np.empty(0, dtype=c_real_t),
        bint verbose=1, int_t print_every=10,
        int_t n_corr_pairs=5, int_t maxiter=400,
        int nthreads=1, bint prefer_onepass=0,
        int_t seed=1, bint handle_interrupt=1
    ):
    
    cdef real_t *ptr_Xfull = NULL
    cdef real_t *ptr_weight = NULL
    cdef int_t *ptr_ixA = NULL
    cdef int_t *ptr_ixB = NULL
    cdef real_t *ptr_X = NULL
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

    cdef real_t *ptr_U = NULL
    cdef int_t *ptr_U_row = NULL
    cdef int_t *ptr_U_col = NULL
    cdef real_t *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]

    cdef real_t *ptr_I = NULL
    cdef int_t *ptr_I_row = NULL
    cdef int_t *ptr_I_col = NULL
    cdef real_t *ptr_I_sp = NULL
    cdef size_t nnz_I = 0
    if I.shape[0]:
        ptr_I = &I[0,0]
    elif I_sp.shape[0]:
        ptr_I_row = &I_row[0]
        ptr_I_col = &I_col[0]
        ptr_I_sp = &I_sp[0]
        nnz_I = I_sp.shape[0]

    cdef real_t *ptr_lam_unique = NULL
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

    cdef np.ndarray[real_t, ndim=2] Am = np.empty((m, k_sec+k+k_main), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] Bm = np.empty((n, k_sec+k+k_main), dtype=c_real_t)

    rs = np.random.Generator(np.random.MT19937(seed = seed))
    cdef np.ndarray[real_t, ndim=1] values = rs.standard_normal(size = nvars, dtype = c_real_t) / 100

    cdef np.ndarray[real_t, ndim=2] Bm_plus_bias = np.empty((0,0), dtype=c_real_t)
    cdef real_t *ptr_Bm_plus_bias = NULL
    if user_bias:
        Bm_plus_bias = np.empty((n, k_sec+k+k_main+1), dtype=c_real_t)
        ptr_Bm_plus_bias = &Bm_plus_bias[0,0]

    cdef real_t glob_mean
    cdef int_t niter, nfev
    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = fit_offsets_explicit_lbfgs_internal(
            &values[0], 0,
            &glob_mean,
            m, n, k,
            ptr_ixA, ptr_ixB, ptr_X, nnz,
            ptr_Xfull,
            ptr_weight,
            user_bias, item_bias, center,
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
    elif (retval == 3) and (not handle_interrupt):
        raise InterruptedError("Procedure was interrupted")

    return glob_mean, Am, Bm, values, niter, nfev, Bm_plus_bias

def unpack_values_lbfgs_offsets(
        np.ndarray[real_t, ndim=1] values,
        bint user_bias, bint item_bias,
        size_t k, size_t k_sec, size_t k_main,
        size_t m, size_t n, size_t p, size_t q,
        bint add_intercepts
    ):

    cdef np.ndarray[real_t, ndim=1] biasA, biasB, C_bias, D_bias
    cdef np.ndarray[real_t, ndim=2] A, B, C,  D

    cdef size_t edge = 0
    if user_bias:
        biasA = values[:m]
        edge += m
    else:
        biasA = np.empty(0, dtype=c_real_t)
    if item_bias:
        biasB = values[edge:edge + n]
        edge += n
    else:
        biasB = np.empty(0, dtype=c_real_t)
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
        C = np.empty((0,0), dtype=c_real_t)
    if (add_intercepts) and (p > 0):
        C_bias = values[edge:edge + (k_sec+k)]
        edge += (k_sec+k)
    else:
        C_bias = np.empty(0, dtype=c_real_t)
    if q > 0:
        D = values[edge:edge + q*(k_sec+k)].reshape((q, k_sec+k))
        edge += q*(k_sec+k)
    else:
        D = np.empty((0,0), dtype=c_real_t)
    if (add_intercepts) and (q > 0):
        D_bias = values[edge:edge + (k_sec+k)]
        edge += (k_sec+k)
    else:
        D_bias = np.empty(0, dtype=c_real_t)

    return biasA, biasB, A, B, C, D, C_bias, D_bias

def call_fit_collective_explicit_als(
        np.ndarray[int_t, ndim=1] ixA,
        np.ndarray[int_t, ndim=1] ixB,
        np.ndarray[real_t, ndim=1] X,
        np.ndarray[real_t, ndim=1] W,
        np.ndarray[real_t, ndim=2] Xfull,
        np.ndarray[real_t, ndim=2] Wfull,
        np.ndarray[real_t, ndim=2] U,
        np.ndarray[int_t, ndim=1] U_row,
        np.ndarray[int_t, ndim=1] U_col,
        np.ndarray[real_t, ndim=1] U_sp,
        np.ndarray[real_t, ndim=2] I,
        np.ndarray[int_t, ndim=1] I_row,
        np.ndarray[int_t, ndim=1] I_col,
        np.ndarray[real_t, ndim=1] I_sp,
        bint NA_as_zero_X, bint NA_as_zero_U, bint NA_as_zero_I,
        int_t m, int_t n, int_t m_u, int_t n_i, int_t p, int_t q,
        int_t k=50, int_t k_user=0, int_t k_item=0, int_t k_main=0,
        real_t w_main=1., real_t w_user=1., real_t w_item=1.,
        real_t w_implicit=0.5,
        bint user_bias=1, bint item_bias=1, bint center=1,
        real_t lam=1e2,
        np.ndarray[real_t, ndim=1] lam_unique=np.empty(0, dtype=c_real_t),
        real_t l1_lam=0.,
        np.ndarray[real_t, ndim=1] l1_lam_unique=np.empty(0, dtype=c_real_t),
        bint center_U=1, bint center_I=1,
        bint scale_lam=0, bint scale_lam_sideinfo=0,
        bint scale_bias_const=0,
        bint verbose=1, int nthreads=1,
        bint use_cg = 0, int_t max_cg_steps=3,
        bint finalize_chol=0,
        bint nonneg=0, bint nonneg_C=0, bint nonneg_D=0,
        size_t max_cd_steps=100,
        int_t seed=1, int_t niter=5, bint handle_interrupt=1,
        bint precompute_for_predictions = 1,
        bint add_implicit_features = 0,
        bint include_all_X = 1
    ):

    cdef real_t *ptr_Xfull = NULL
    cdef real_t *ptr_weight = NULL
    cdef int_t *ptr_ixA = NULL
    cdef int_t *ptr_ixB = NULL
    cdef real_t *ptr_X = NULL
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

    cdef real_t *ptr_U = NULL
    cdef int_t *ptr_U_row = NULL
    cdef int_t *ptr_U_col = NULL
    cdef real_t *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    cdef np.ndarray[real_t, ndim=1] U_colmeans = np.empty(0, dtype=c_real_t)
    cdef real_t *ptr_U_colmeans = NULL
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]
    if (U.shape[0] or U_sp.shape[0]) and center_U:
        U_colmeans = np.empty(p, dtype=c_real_t)
        ptr_U_colmeans = &U_colmeans[0]

    cdef real_t *ptr_I = NULL
    cdef int_t *ptr_I_row = NULL
    cdef int_t *ptr_I_col = NULL
    cdef real_t *ptr_I_sp = NULL
    cdef size_t nnz_I = 0
    cdef np.ndarray[real_t, ndim=1] I_colmeans = np.empty(0, dtype=c_real_t)
    cdef real_t *ptr_I_colmeans = NULL
    if I.shape[0]:
        ptr_I = &I[0,0]
    elif I_sp.shape[0]:
        ptr_I_row = &I_row[0]
        ptr_I_col = &I_col[0]
        ptr_I_sp = &I_sp[0]
        nnz_I = I_sp.shape[0]
    if (I.shape[0] or I_sp.shape[0]) and center_I:
        I_colmeans = np.empty(q, dtype=c_real_t)
        ptr_I_colmeans = &I_colmeans[0]

    cdef real_t *ptr_lam_unique = NULL
    cdef real_t *ptr_l1_lam_unique = NULL
    if lam_unique.shape[0]:
        ptr_lam_unique = &lam_unique[0]
    if l1_lam_unique.shape[0]:
        ptr_l1_lam_unique = &l1_lam_unique[0]

    cdef np.ndarray[real_t, ndim=1] biasA = np.zeros(0, dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=1] biasB = np.zeros(0, dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] A = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] B = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] C = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] D = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] Ai = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] Bi = np.zeros((0,0), dtype=c_real_t)
    cdef real_t *ptr_biasA = NULL
    cdef real_t *ptr_biasB = NULL
    cdef real_t *ptr_A = NULL
    cdef real_t *ptr_B = NULL
    cdef real_t *ptr_C = NULL
    cdef real_t *ptr_D = NULL
    cdef real_t *ptr_Ai = NULL
    cdef real_t *ptr_Bi = NULL

    cdef size_t sizeA = <size_t>max(m, m_u) * <size_t>(k_user+k+k_main)
    cdef size_t sizeB = <size_t>max(n, n_i) * <size_t>(k_item+k+k_main)
    if (sizeA == 0) or (sizeB == 0):
        raise ValueError("Model cannot have empty 'A' or 'B' matrices.")

    if user_bias:
        biasA = np.zeros(max(m, m_u), dtype=c_real_t)
        ptr_biasA = &biasA[0]
    if item_bias:
        biasB = np.zeros(max(n, n_i), dtype=c_real_t)
        ptr_biasB = &biasB[0]

    rs = np.random.Generator(np.random.MT19937(seed = seed))
    A = rs.standard_normal(size = (max(m, m_u), k_user+k+k_main), dtype = c_real_t) / 100
    B = rs.standard_normal(size = (max(n, n_i), k_item+k+k_main), dtype = c_real_t) / 100
    if nonneg:
        A[:,:] = np.abs(A)
        B[:,:] = np.abs(B)
    ptr_A = &A[0,0]
    ptr_B = &B[0,0]
    if p:
        C = rs.standard_normal(size = (p, k_user + k), dtype = c_real_t) / 100
        if nonneg_C:
            C[:,:] = np.abs(C)
        if C.shape[0]:
            ptr_C = &C[0,0]
        else:
            raise ValueError("Unexpected error.")
    if q:
        D = rs.standard_normal(size = (q, k_item + k), dtype = c_real_t) / 100
        if nonneg_D:
            D[:,:] = np.abs(D)
        if D.shape[0]:
            ptr_D = &D[0,0]
        else:
            raise ValueError("Unexpected error.")
    if add_implicit_features:
        Ai = rs.standard_normal(size = (max(m, m_u), k+k_main), dtype = c_real_t) / 100
        Bi = rs.standard_normal(size = (max(n, n_i), k+k_main), dtype = c_real_t) / 100
        if nonneg:
            Ai[:,:] = np.abs(Ai)
            Bi[:,:] = np.abs(Bi)
        ptr_Ai = &Ai[0,0]
        ptr_Bi = &Bi[0,0]

    cdef np.ndarray[real_t, ndim=2] B_plus_bias = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] BtB = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] TransBtBinvBt = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=1] BtXbias = np.zeros(0, dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] BeTBeChol = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] BiTBi = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] TransCtCinvCt = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] CtCw = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=1] CtUbias = np.zeros(0, dtype=c_real_t)
    cdef real_t *ptr_B_plus_bias = NULL
    cdef real_t *ptr_BtB = NULL
    cdef real_t *ptr_TransBtBinvBt = NULL
    cdef real_t *ptr_BtXbias = NULL
    cdef real_t *ptr_BeTBeChol = NULL
    cdef real_t *ptr_BiTBi = NULL
    cdef real_t *ptr_TransCtCinvCt = NULL
    cdef real_t *ptr_CtCw = NULL
    cdef real_t *ptr_CtUbias = NULL

    if precompute_for_predictions:
        if user_bias:
            B_plus_bias = np.empty((B.shape[0],B.shape[1]+1), dtype=c_real_t)
            ptr_B_plus_bias = &B_plus_bias[0,0]
        BtB = np.zeros((k+k_main+user_bias, k+k_main+user_bias), dtype=c_real_t)
        ptr_BtB = &BtB[0,0]
        if (not add_implicit_features) and (not nonneg):
            TransBtBinvBt = np.zeros((B.shape[0], k+k_main+user_bias), dtype=c_real_t)
            ptr_TransBtBinvBt = &TransBtBinvBt[0,0]
        if p:
            CtCw = np.zeros((k_user+k, k_user+k), dtype=c_real_t)
            if CtCw.shape[0]:
                ptr_CtCw = &CtCw[0,0]
            else:
                raise ValueError("Unexpected error.")
            if (not add_implicit_features) and (not nonneg):
                TransCtCinvCt = np.zeros((p, k_user+k), dtype=c_real_t)
                if TransCtCinvCt.shape[0]:
                    ptr_TransCtCinvCt = &TransCtCinvCt[0,0]
                else:
                    raise ValueError("Unexpected error.")
        if add_implicit_features:
            BiTBi = np.zeros((k+k_main, k+k_main), dtype=c_real_t)
            ptr_BiTBi = &BiTBi[0,0]

        if (p or add_implicit_features) and (not nonneg):
            BeTBeChol = np.zeros((k_user+k+k_main+user_bias, k_user+k+k_main+user_bias), dtype=c_real_t)
            ptr_BeTBeChol = &BeTBeChol[0,0]

        if (NA_as_zero_X) and ((center) or (user_bias)):
            BtXbias = np.empty(k+k_main+user_bias, dtype=c_real_t)
            ptr_BtXbias = &BtXbias[0]

        if (NA_as_zero_U) and (U_colmeans.shape[0]):
            CtUbias = np.empty(k_user+k, dtype=c_real_t)
            ptr_CtUbias = &CtUbias[0]

    cdef real_t glob_mean = 0
    cdef real_t scaling_biasA = 0
    cdef real_t scaling_biasB = 0

    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = fit_collective_explicit_als(
            ptr_biasA, ptr_biasB,
            ptr_A, ptr_B, ptr_C, ptr_D,
            ptr_Ai, ptr_Bi,
            add_implicit_features,
            0, 0,
            &glob_mean,
            ptr_U_colmeans, ptr_I_colmeans,
            m, n, k,
            ptr_ixA, ptr_ixB, ptr_X, nnz,
            ptr_Xfull,
            ptr_weight,
            user_bias, item_bias, center,
            lam, ptr_lam_unique,
            l1_lam, ptr_l1_lam_unique,
            scale_lam, scale_lam_sideinfo, scale_bias_const,
            &scaling_biasA, &scaling_biasB,
            ptr_U, m_u, p,
            ptr_I, n_i, q,
            ptr_U_row, ptr_U_col, ptr_U_sp, nnz_U,
            ptr_I_row, ptr_I_col, ptr_I_sp, nnz_I,
            NA_as_zero_X, NA_as_zero_U, NA_as_zero_I,
            k_main, k_user, k_item,
            w_main, w_user, w_item, w_implicit,
            niter, nthreads, verbose, handle_interrupt,
            use_cg, max_cg_steps, finalize_chol,
            nonneg, max_cd_steps, nonneg_C, nonneg_D,
            precompute_for_predictions,
            include_all_X,
            ptr_B_plus_bias,
            ptr_BtB,
            ptr_TransBtBinvBt,
            ptr_BtXbias,
            ptr_BeTBeChol,
            ptr_BiTBi,
            ptr_TransCtCinvCt,
            ptr_CtCw,
            ptr_CtUbias
        )

    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")
    elif (retval == 3) and (not handle_interrupt):
        raise InterruptedError("Procedure was interrupted")

    return biasA, biasB, A, B, C, D, Ai, Bi, \
           glob_mean, U_colmeans, I_colmeans, \
           B_plus_bias, BtB, TransBtBinvBt, BtXbias, \
           BeTBeChol, BiTBi, TransCtCinvCt, CtCw, \
           CtUbias, scaling_biasA, scaling_biasB


def call_fit_collective_implicit_als(
        np.ndarray[int_t, ndim=1] ixA,
        np.ndarray[int_t, ndim=1] ixB,
        np.ndarray[real_t, ndim=1] X,
        np.ndarray[real_t, ndim=2] U,
        np.ndarray[int_t, ndim=1] U_row,
        np.ndarray[int_t, ndim=1] U_col,
        np.ndarray[real_t, ndim=1] U_sp,
        np.ndarray[real_t, ndim=2] I,
        np.ndarray[int_t, ndim=1] I_row,
        np.ndarray[int_t, ndim=1] I_col,
        np.ndarray[real_t, ndim=1] I_sp,
        bint NA_as_zero_U, bint NA_as_zero_I,
        int_t m, int_t n, int_t m_u, int_t n_i, int_t p, int_t q,
        int_t k=50, int_t k_user=0, int_t k_item=0, int_t k_main=0,
        real_t w_main=1., real_t w_user=1., real_t w_item=1.,
        real_t lam=1e2, real_t alpha=1.,
        bint adjust_weight=1, bint apply_log_transf=0,
        np.ndarray[real_t, ndim=1] lam_unique=np.empty(0, dtype=c_real_t),
        real_t l1_lam=1e2,
        np.ndarray[real_t, ndim=1] l1_lam_unique=np.empty(0, dtype=c_real_t),
        bint center_U=1, bint center_I=1,
        bint verbose=1, int_t niter=10,
        int nthreads=1, bint use_cg=1,
        int_t max_cg_steps=3, bint finalize_chol=1,
        bint nonneg=0, bint nonneg_C=0, bint nonneg_D=0,
        int max_cd_steps=100,
        int_t seed=1, init="normal", bint handle_interrupt=1,
        bint precompute_for_predictions = 1
    ):
    
    cdef real_t *ptr_U = NULL
    cdef int_t *ptr_U_row = NULL
    cdef int_t *ptr_U_col = NULL
    cdef real_t *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    cdef np.ndarray[real_t, ndim=1] U_colmeans = np.empty(0, dtype=c_real_t)
    cdef real_t *ptr_U_colmeans = NULL
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]
    if (U.shape[0] or U_sp.shape[0]) and center_U:
        U_colmeans = np.empty(p, dtype=c_real_t)
        ptr_U_colmeans = &U_colmeans[0]

    if X.shape[0] == 0:
        raise ValueError("Input data has no non-zero values.")

    cdef real_t *ptr_I = NULL
    cdef int_t *ptr_I_row = NULL
    cdef int_t *ptr_I_col = NULL
    cdef real_t *ptr_I_sp = NULL
    cdef size_t nnz_I = 0
    cdef np.ndarray[real_t, ndim=1] I_colmeans = np.empty(0, dtype=c_real_t)
    cdef real_t *ptr_I_colmeans = NULL
    if I.shape[0]:
        ptr_I = &I[0,0]
    elif I_sp.shape[0]:
        ptr_I_row = &I_row[0]
        ptr_I_col = &I_col[0]
        ptr_I_sp = &I_sp[0]
        nnz_I = I_sp.shape[0]
    if (I.shape[0] or I_sp.shape[0]) and center_I:
        I_colmeans = np.empty(q, dtype=c_real_t)
        ptr_I_colmeans = &I_colmeans[0]

    cdef real_t *ptr_lam_unique = NULL
    cdef real_t *ptr_l1_lam_unique = NULL
    if lam_unique.shape[0]:
        ptr_lam_unique = &lam_unique[0]
    if l1_lam_unique.shape[0]:
        ptr_l1_lam_unique = &l1_lam_unique[0]


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

    cdef real_t *ptr_A = NULL
    cdef real_t *ptr_B = NULL
    cdef real_t *ptr_C = NULL
    cdef real_t *ptr_D = NULL
    cdef np.ndarray[real_t, ndim=2] A = np.zeros((0,0), dtype = c_real_t)
    cdef np.ndarray[real_t, ndim=2] B = np.zeros((0,0), dtype = c_real_t)
    cdef np.ndarray[real_t, ndim=2] C = np.zeros((0,0), dtype = c_real_t)
    cdef np.ndarray[real_t, ndim=2] D = np.zeros((0,0), dtype = c_real_t)

    if init == "normal":
        A = rs.standard_normal(size = (max(m, m_u), (k_user+k+k_main)), dtype = c_real_t) / 100
        B = rs.standard_normal(size = (max(n, n_i), (k_item+k+k_main)), dtype = c_real_t) / 100
        if sizeC:
            C = rs.standard_normal(size = (p, k_user + k), dtype = c_real_t) / 100
        if sizeD:
            D = rs.standard_normal(size = (q, k_item + k), dtype = c_real_t) / 100
    else:
        A = rs.random(size = (max(m, m_u), (k_user+k+k_main)), dtype = c_real_t) / 100
        B = rs.random(size = (max(n, n_i), (k_item+k+k_main)), dtype = c_real_t) / 100
        if sizeC:
            C = rs.random(size = (p, k_user + k), dtype = c_real_t) / 100
        if sizeD:
            D = rs.random(size = (q, k_item + k), dtype = c_real_t) / 100
    
        if init == "gamma":
            A[:,:] = -np.log(A.clip(min=1e-6, max=20.))
            B[:,:] = -np.log(B.clip(min=1e-6, max=20.))
            if sizeC:
                C[:,:] = -np.log(C.clip(min=1e-6, max=20.))
            if sizeD:
                D[:,:] = -np.log(D.clip(min=1e-6, max=20.))
        elif init != "uniform":
            A[:,:] -= 0.5
            B[:,:] -= 0.5
            if sizeC:
                C[:,:] -= 0.5
            if sizeD:
                D[:,:] -= 0.5

    if nonneg:
        A[:,:] = np.abs(A)
        B[:,:] = np.abs(B)
        if C.shape[0]:
            C[:,:] = np.abs(C)
        if D.shape[0]:
            D[:,:] = np.abs(D)

    ptr_A = &A[0,0]
    ptr_B = &B[0,0]
    if sizeC:
        ptr_C = &C[0,0]
    if sizeD:
        ptr_D = &D[0,0]

    cdef np.ndarray[real_t, ndim=2] precomputedBtB = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] precomputedBeTBe = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] precomputedBeTBeChol = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=1] precomputedCtUbias = np.zeros(0, dtype=c_real_t)
    cdef real_t *ptr_BtB = NULL
    cdef real_t *ptr_BeTBe = NULL
    cdef real_t *ptr_BeTBeChol = NULL
    cdef real_t *ptr_CtUbias = NULL
    if precompute_for_predictions:
        precomputedBtB = np.empty((k+k_main, k+k_main), dtype=c_real_t)
        ptr_BtB = &precomputedBtB[0,0]
        if U.shape[0] or U_sp.shape[0]:
            precomputedBeTBe = np.empty((k_user+k+k_main, k_user+k+k_main), dtype=c_real_t)
            ptr_BeTBe = &precomputedBeTBe[0,0]
            if (not nonneg):
                precomputedBeTBeChol = np.empty((k_user+k+k_main, k_user+k+k_main), dtype=c_real_t)
                ptr_BeTBeChol = &precomputedBeTBeChol[0,0]
        if (U_colmeans.shape[0]) and (NA_as_zero_U):
            precomputedCtUbias = np.empty(k_user+k, dtype=c_real_t)
            ptr_CtUbias = &precomputedCtUbias[0]


    cdef real_t w_main_multiplier = 1.

    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        reval = fit_collective_implicit_als(
            ptr_A, ptr_B, ptr_C, ptr_D,
            0, 0,
            ptr_U_colmeans, ptr_I_colmeans,
            m, n, k,
            &ixA[0], &ixB[0], &X[0], X.shape[0],
            lam, ptr_lam_unique,
            l1_lam, ptr_l1_lam_unique,
            ptr_U, m_u, p,
            ptr_I, n_i, q,
            ptr_U_row, ptr_U_col, ptr_U_sp, nnz_U,
            ptr_I_row, ptr_I_col, ptr_I_sp, nnz_I,
            NA_as_zero_U, NA_as_zero_I,
            k_main, k_user, k_item,
            w_main, w_user, w_item,
            &w_main_multiplier,
            alpha, adjust_weight, apply_log_transf,
            niter, nthreads, verbose, handle_interrupt,
            use_cg, max_cg_steps, finalize_chol,
            nonneg, max_cd_steps, nonneg_C, nonneg_D,
            precompute_for_predictions,
            ptr_BtB, ptr_BeTBe, ptr_BeTBeChol,
            ptr_CtUbias
        )

    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")
    elif (retval == 3) and (not handle_interrupt):
        raise InterruptedError("Procedure was interrupted")

    return A, B, C, D, \
           U_colmeans, I_colmeans, w_main_multiplier, \
           precomputedBtB, precomputedBeTBe, precomputedBeTBeChol, \
           precomputedCtUbias

def call_fit_offsets_explicit_als(
        np.ndarray[int_t, ndim=1] ixA,
        np.ndarray[int_t, ndim=1] ixB,
        np.ndarray[real_t, ndim=1] X,
        np.ndarray[real_t, ndim=1] W,
        np.ndarray[real_t, ndim=2] Xfull,
        np.ndarray[real_t, ndim=2] Wfull,
        np.ndarray[real_t, ndim=2] U,
        np.ndarray[real_t, ndim=2] I,
        bint NA_as_zero_X,
        int_t m, int_t n, int_t p, int_t q,
        int_t k=50,
        bint user_bias=1, bint item_bias=1, bint center=1,
        bint add_intercepts=1,
        real_t lam=1e2,
        bint verbose=1, int nthreads=1,
        bint use_cg=0, int_t max_cg_steps=3,
        bint finalize_chol=0,
        int_t seed=1, int_t niter=5, bint handle_interrupt=1,
        bint precompute_for_predictions=1
    ):
    if k <= 0:
        raise ValueError("'k' must be a positive number.")
    if min(m,n) <= 0:
        raise ValueError("'X' must have positive dimensions.")

    cdef real_t *ptr_Xfull = NULL
    cdef real_t *ptr_weight = NULL
    cdef int_t *ptr_ixA = NULL
    cdef int_t *ptr_ixB = NULL
    cdef real_t *ptr_X = NULL
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

    cdef real_t *ptr_U = NULL
    if U.shape[0]:
        ptr_U = &U[0,0]

    cdef real_t *ptr_I = NULL
    if I.shape[0]:
        ptr_I = &I[0,0]

    cdef np.ndarray[real_t, ndim=2] Am = np.empty((m, k), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] Bm = np.empty((n, k), dtype=c_real_t)
    cdef real_t *ptr_Am = &Am[0,0]
    cdef real_t *ptr_Bm = &Bm[0,0]

    cdef np.ndarray[real_t, ndim=2] Bm_plus_bias = np.empty((0,0), dtype=c_real_t)
    cdef real_t *ptr_Bm_plus_bias = NULL
    if user_bias:
        Bm_plus_bias = np.empty((n, k+1), dtype=c_real_t)
        ptr_Bm_plus_bias = &Bm_plus_bias[0,0]

    rs = np.random.Generator(np.random.MT19937(seed = seed))
    cdef np.ndarray[real_t, ndim=1] biasA = np.zeros(0, dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=1] biasB = np.zeros(0, dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] A = rs.standard_normal(size=(m,k), dtype=c_real_t) / 100
    cdef np.ndarray[real_t, ndim=2] B = rs.standard_normal(size=(n,k), dtype=c_real_t) / 100
    cdef np.ndarray[real_t, ndim=2] C = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] D = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=1] C_bias = np.zeros(0, dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=1] D_bias = np.zeros(0, dtype=c_real_t)

    cdef real_t *ptr_biasA = NULL
    cdef real_t *ptr_biasB = NULL
    cdef real_t *ptr_A = &A[0,0]
    cdef real_t *ptr_B = &B[0,0]
    cdef real_t *ptr_C = NULL
    cdef real_t *ptr_D = NULL
    cdef real_t *ptr_C_bias = NULL
    cdef real_t *ptr_D_bias = NULL
    if p:
        C = np.empty((p,k), dtype=c_real_t)
        if C.shape[0]:
            ptr_C = &C[0,0]
        if add_intercepts:
            C_bias = np.zeros(k, dtype=c_real_t)
            if C_bias.shape[0]:
                ptr_C_bias = &C_bias[0]
    if q:
        D = np.empty((q,k), dtype=c_real_t)
        if D.shape[0]:
            ptr_D = &D[0,0]
        if add_intercepts:
            D_bias = np.zeros(k, dtype=c_real_t)
            if D_bias.shape[0]:
                ptr_D_bias = &D_bias[0]

    if user_bias:
        biasA = np.empty(k, dtype=c_real_t)
        ptr_biasA = &biasA[0]
    if item_bias:
        biasB = np.empty(k, dtype=c_real_t)
        ptr_biasB = &biasB[0]

    cdef np.ndarray[real_t, ndim=2] BtB = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] TransBtBinvBt = np.zeros((0,0), dtype=c_real_t)
    cdef real_t *ptr_BtB = NULL
    cdef real_t *ptr_TransBtBinvBt = NULL
    if precompute_for_predictions:
        BtB = np.empty((k+user_bias,k+user_bias), dtype=c_real_t)
        ptr_BtB = &BtB[0,0]
        TransBtBinvBt = np.empty((n,k+user_bias), dtype=c_real_t)
        ptr_TransBtBinvBt = &TransBtBinvBt[0,0]

    cdef real_t glob_mean = 0

    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = fit_offsets_als(
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
            user_bias, item_bias, center, add_intercepts,
            lam,
            ptr_U, p,
            ptr_I, q,
            0, NA_as_zero_X, 0., 0,
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
    elif (retval == 3) and (not handle_interrupt):
        raise InterruptedError("Procedure was interrupted")

    return biasA, biasB, A, B, C, D, C_bias, D_bias, \
           Am, Bm, glob_mean, \
           Bm_plus_bias, BtB, TransBtBinvBt

def call_fit_offsets_implicit_als(
        np.ndarray[int_t, ndim=1] ixA,
        np.ndarray[int_t, ndim=1] ixB,
        np.ndarray[real_t, ndim=1] X,
        np.ndarray[real_t, ndim=2] U,
        np.ndarray[real_t, ndim=2] I,
        int_t m, int_t n, int_t p, int_t q,
        int_t k=50, bint add_intercepts=1,
        real_t lam=1e2, real_t alpha=40., bint apply_log_transf=0,
        bint verbose=1, int nthreads=1,
        bint use_cg=0, int_t max_cg_steps=3,
        bint finalize_chol=0,
        bint adjust_weight = 1,
        int_t seed=1, int_t niter=5, bint handle_interrupt=1,
        bint precompute_for_predictions=1
    ):
    if k <= 0:
        raise ValueError("'k' must be a positive integer.")
    if min(m,n) <= 0:
        raise ValueError("'X' must have positive dimensions.")
    cdef int_t *ptr_ixA = NULL
    cdef int_t *ptr_ixB = NULL
    cdef real_t *ptr_X = NULL
    cdef size_t nnz = 0
    ptr_ixA = &ixA[0]
    ptr_ixB = &ixB[0]
    ptr_X = &X[0]
    nnz = X.shape[0]

    cdef real_t *ptr_U = NULL
    if U.shape[0]:
        ptr_U = &U[0,0]

    cdef real_t *ptr_I = NULL
    if I.shape[0]:
        ptr_I = &I[0,0]
    
    cdef np.ndarray[real_t, ndim=2] Am = np.empty((m, k), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] Bm = np.empty((n, k), dtype=c_real_t)
    cdef real_t *ptr_Am = &Am[0,0]
    cdef real_t *ptr_Bm = &Bm[0,0]

    rs = np.random.Generator(np.random.MT19937(seed = seed))
    cdef np.ndarray[real_t, ndim=2] A = rs.standard_normal(size=(m,k), dtype=c_real_t) / 100
    cdef np.ndarray[real_t, ndim=2] B = rs.standard_normal(size=(n,k), dtype=c_real_t) / 100
    cdef np.ndarray[real_t, ndim=2] C = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] D = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=1] C_bias = np.zeros(0, dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=1] D_bias = np.zeros(0, dtype=c_real_t)

    cdef real_t *ptr_A = &A[0,0]
    cdef real_t *ptr_B = &B[0,0]
    cdef real_t *ptr_C = NULL
    cdef real_t *ptr_D = NULL
    cdef real_t *ptr_C_bias = NULL
    cdef real_t *ptr_D_bias = NULL
    if p:
        C = np.empty((p,k), dtype=c_real_t)
        if C.shape[0]:
            ptr_C = &C[0,0]
        if add_intercepts:
            C_bias = np.empty(k, dtype=c_real_t)
            if C_bias.shape[0]:
                ptr_C_bias = &C_bias[0]
    if q:
        D = np.empty((q,k), dtype=c_real_t)
        if D.shape[0]:
            ptr_D = &D[0,0]
        if add_intercepts:
            D_bias = np.empty(k, dtype=c_real_t)
            if D_bias.shape[0]:
                ptr_D_bias = &D_bias[0]

    cdef np.ndarray[real_t, ndim=2] BtB = np.zeros((0,0), dtype=c_real_t)
    cdef real_t *ptr_BtB = NULL
    if precompute_for_predictions:
        BtB = np.empty((k,k), dtype=c_real_t)
        ptr_BtB = &BtB[0,0]

    cdef real_t placeholder = 0

    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = fit_offsets_als(
            <real_t*>NULL, <real_t*>NULL,
            ptr_A, ptr_B, ptr_C, ptr_C_bias, ptr_D, ptr_D_bias,
            0, 0,
            &placeholder,
            m, n, k,
            ptr_ixA, ptr_ixB, ptr_X, nnz,
            <real_t*>NULL,
            <real_t*>NULL,
            0, 0, 0, add_intercepts,
            lam,
            ptr_U, p,
            ptr_I, q,
            1, 0,
            alpha, apply_log_transf,
            niter,
            nthreads, use_cg,
            max_cg_steps, finalize_chol,
            verbose, handle_interrupt,
            precompute_for_predictions,
            ptr_Am, ptr_Bm,
            <real_t*>NULL,
            ptr_BtB,
            <real_t*>NULL
        )

    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")
    elif retval == 2:
        raise ValueError("Invalid parameter combination.")
    elif (retval == 3) and (not handle_interrupt):
        raise InterruptedError("Procedure was interrupted")

    return A, B, C, D, \
           Am, Bm, BtB

def precompute_matrices_collective_explicit(
        np.ndarray[real_t, ndim=2] B,
        np.ndarray[real_t, ndim=2] C,
        np.ndarray[real_t, ndim=2] Bi,
        np.ndarray[real_t, ndim=1] biasB,
        np.ndarray[real_t, ndim=1] U_colmeans,
        bint user_bias, bint add_implicit_features,
        int_t n_orig,
        int_t k, int_t k_user, int_t k_item, int_t k_main,
        real_t lam, real_t lam_bias,
        real_t w_main, real_t w_user, real_t w_implicit,
        real_t glob_mean,
        bint scale_lam = 0, bint scale_lam_sideinfo = 0,
        bint scale_bias_const = 0, real_t scaling_biasA = 0.,
        bint NA_as_zero_X = 0,
        bint NA_as_zero_U = 0,
        bint nonneg = 0,
        bint include_all_X = 1
    ):
    cdef int_t n_max = B.shape[0]
    cdef int_t p = C.shape[0]

    if n_max == 0:
        raise ValueError("'B' has no entries.")
    if (add_implicit_features) and not max(Bi.shape[0], Bi.shape[1]):
        raise ValueError("'Bi' has no entries.")

    cdef real_t *ptr_B = &B[0,0]
    cdef real_t *ptr_C = NULL
    cdef real_t *ptr_Bi = NULL
    cdef real_t *ptr_biasB = NULL
    cdef real_t *ptr_U_colmeans = NULL
    if p:
        ptr_C = &C[0,0]
    if Bi.shape[0]:
        ptr_Bi = &Bi[0,0]
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]
    if U_colmeans.shape[0]:
        ptr_U_colmeans = &U_colmeans[0]

    cdef np.ndarray[real_t, ndim=2] B_plus_bias = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] BtB = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] TransBtBinvBt = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=1] BtXbias = np.zeros(0, dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] BeTBeChol = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] BiTBi = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] TransCtCinvCt = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] CtCw = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=1] CtUbias = np.zeros(0, dtype=c_real_t)
    cdef real_t *ptr_B_plus_bias = NULL
    cdef real_t *ptr_BtB = NULL
    cdef real_t *ptr_TransBtBinvBt = NULL
    cdef real_t *ptr_BtXbias = NULL
    cdef real_t *ptr_BeTBeChol = NULL
    cdef real_t *ptr_BiTBi = NULL
    cdef real_t *ptr_TransCtCinvCt = NULL
    cdef real_t *ptr_CtCw = NULL
    cdef real_t *ptr_CtUbias = NULL

    BtB = np.zeros((k+k_main+user_bias, k+k_main+user_bias), dtype=c_real_t)
    ptr_BtB = &BtB[0,0]
    if (not add_implicit_features) and (not nonneg):
        TransBtBinvBt = np.zeros((B.shape[0], k+k_main+user_bias), dtype=c_real_t)
        ptr_TransBtBinvBt = &TransBtBinvBt[0,0]
    if user_bias:
        B_plus_bias = np.empty((B.shape[0], B.shape[1]+1), dtype=c_real_t)
        ptr_B_plus_bias = &B_plus_bias[0,0]
    else:
        B_plus_bias = B
    if p > 0:
        CtCw = np.zeros((k_user+k, k_user+k), dtype=c_real_t)
        ptr_CtCw = &CtCw[0,0]

        if (not add_implicit_features) and (not nonneg):
            TransCtCinvCt = np.zeros((C.shape[0], k_user+k), dtype=c_real_t)
            ptr_TransCtCinvCt = &TransCtCinvCt[0,0]
    if add_implicit_features:
        BiTBi = np.zeros((k+k_main, k+k_main), dtype=c_real_t)
        ptr_BiTBi = &BiTBi[0,0]

    if ((p) or (add_implicit_features)) and (not nonneg):
        BeTBeChol = np.zeros((k_user+k+k_main+user_bias, k_user+k+k_main+user_bias), dtype=c_real_t)
        ptr_BeTBeChol = &BeTBeChol[0,0]

    if (NA_as_zero_X) and (biasB.shape[0] or glob_mean):
        BtXbias = np.zeros(k+k_main+user_bias, dtype=c_real_t)
        ptr_BtXbias = &BtXbias[0]

    if (NA_as_zero_U) and (U_colmeans.shape[0]):
        CtUbias = np.zeros(k_user+k, dtype=c_real_t)
        ptr_CtUbias = &CtUbias[0]

    cdef real_t *ptr_lam_unique = NULL
    cdef np.ndarray[real_t, ndim=1] lam_unique = np.zeros(6, dtype=c_real_t)
    if lam_bias != lam:
        lam_unique[0] = lam_bias
        lam_unique[2] = lam
        ptr_lam_unique = &lam_unique[0]

    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = precompute_collective_explicit(
            ptr_B, n_orig, n_max, include_all_X,
            ptr_C, p,
            ptr_Bi, add_implicit_features,
            ptr_biasB, glob_mean, NA_as_zero_X,
            ptr_U_colmeans, NA_as_zero_U,
            k, k_user, k_item, k_main,
            user_bias,
            nonneg,
            lam, ptr_lam_unique,
            scale_lam, scale_lam_sideinfo,
            scale_bias_const, scaling_biasA,
            w_main, w_user, w_implicit,
            ptr_B_plus_bias,
            ptr_BtB,
            ptr_TransBtBinvBt,
            ptr_BtXbias,
            ptr_BeTBeChol,
            ptr_BiTBi,
            ptr_TransCtCinvCt,
            ptr_CtCw,
            ptr_CtUbias
        )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return B_plus_bias, BtB, TransBtBinvBt, BtXbias, \
           BeTBeChol, BiTBi, TransCtCinvCt, CtCw, CtUbias

def precompute_matrices_collective_implicit(
        np.ndarray[real_t, ndim=2] B,
        np.ndarray[real_t, ndim=2] C,
        np.ndarray[real_t, ndim=1] U_colmeans,
        int_t k, int_t k_main, int_t k_user, int_t k_item,
        real_t lam, real_t w_main, real_t w_user,
        real_t w_main_multiplier,
        bint nonneg = 0,
        bint NA_as_zero_U = 0
    ):
    cdef int_t n = B.shape[0]
    cdef np.ndarray[real_t, ndim=2] BtB = np.empty((k+k_main, k+k_main), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] BeTBe = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] BeTBeChol = np.zeros((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=1] CtUbias = np.zeros(0, dtype=c_real_t)
    if C.shape[0] and C.shape[1]:
        BeTBe = np.empty((k_user+k+k_main, k_user+k+k_main), dtype=c_real_t)
        if not nonneg:
            BeTBeChol = np.empty((k_user+k+k_main, k_user+k+k_main), dtype=c_real_t)

    cdef real_t *ptr_BtB = &BtB[0,0]
    cdef real_t *ptr_BeTBe = NULL
    cdef real_t *ptr_BeTBeChol = NULL
    if BeTBe.shape[0] and BeTBe.shape[1]:
        ptr_BeTBe = &BeTBe[0,0]
    if BeTBeChol.shape[0] and BeTBeChol.shape[1]:
        ptr_BeTBeChol = &BeTBeChol[0,0]

    cdef real_t *ptr_C = NULL
    cdef int_t p = 0
    if C.shape[0]:
        p = C.shape[0]
        ptr_C = &C[0,0]

    cdef real_t *ptr_U_colmeans = NULL
    cdef real_t *ptr_CtUbias = NULL
    if (U_colmeans.shape[0]) and (NA_as_zero_U):
        ptr_U_colmeans = &U_colmeans[0]
        CtUbias = np.zeros(k_user+k, dtype=c_real_t)
        ptr_CtUbias = &CtUbias[0]

    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = precompute_collective_implicit(
            &B[0,0], B.shape[0],
            ptr_C, p,
            ptr_U_colmeans, NA_as_zero_U,
            k, k_user, k_item, k_main,
            lam, w_main, w_user, w_main_multiplier,
            nonneg,
            1,
            ptr_BtB, ptr_BeTBe, ptr_BeTBeChol, ptr_CtUbias
        )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return BtB, BeTBe, BeTBeChol, CtUbias

def call_factors_collective_explicit_single(
        np.ndarray[real_t, ndim=1] Xa_dense,
        np.ndarray[real_t, ndim=1] W_dense,
        np.ndarray[real_t, ndim=1] Xa,
        np.ndarray[int_t, ndim=1] Xa_i,
        np.ndarray[real_t, ndim=1] W_sp,
        np.ndarray[real_t, ndim=1] U,
        np.ndarray[real_t, ndim=1] U_sp,
        np.ndarray[int_t, ndim=1] U_sp_i,
        np.ndarray[real_t, ndim=1] U_bin,
        np.ndarray[real_t, ndim=1] U_colmeans,
        np.ndarray[real_t, ndim=1] biasB,
        np.ndarray[real_t, ndim=2] B,
        np.ndarray[real_t, ndim=2] B_plus_bias,
        np.ndarray[real_t, ndim=2] C,
        np.ndarray[real_t, ndim=2] C_bin,
        np.ndarray[real_t, ndim=2] Bi,
        np.ndarray[real_t, ndim=2] BtB,
        np.ndarray[real_t, ndim=2] TransBtBinvBt,
        np.ndarray[real_t, ndim=1] BtXbias,
        np.ndarray[real_t, ndim=2] BeTBeChol,
        np.ndarray[real_t, ndim=2] BiTBi,
        np.ndarray[real_t, ndim=2] CtCw,
        np.ndarray[real_t, ndim=2] TransCtCinvCt,
        np.ndarray[real_t, ndim=1] CtUbias,
        real_t glob_mean,
        int_t n_orig,
        int_t k, int_t k_user = 0, int_t k_item = 0, int_t k_main = 0,
        real_t lam = 1e2, real_t lam_bias = 1e2,
        real_t l1_lam = 0., real_t l1_lam_bias = 0.,
        bint scale_lam = 0, bint scale_lam_sideinfo = 0,
        bint scale_bias_const = 0, real_t scaling_biasA = 0.,
        real_t w_user = 1., real_t w_main = 1., real_t w_implicit = 0.5,
        bint user_bias = 1,
        bint NA_as_zero_U = 0, bint NA_as_zero_X = 0,
        bint nonneg = 0,
        bint add_implicit_features = 0,
        bint include_all_X = 1
    ):


    cdef real_t *ptr_Xa_dense = NULL
    cdef real_t *ptr_Xa = NULL
    cdef int_t *ptr_Xa_i = NULL
    cdef real_t *ptr_weight = NULL
    if Xa_dense.shape[0]:
        ptr_Xa_dense = &Xa_dense[0]
        if W_dense.shape[0]:
            ptr_weight = &W_dense[0]
    elif Xa.shape[0]:
        ptr_Xa = &Xa[0]
        ptr_Xa_i = &Xa_i[0]
        if W_sp.shape[0]:
            ptr_weight = &W_sp[0]

    cdef real_t *ptr_U = NULL
    cdef real_t *ptr_U_sp = NULL
    cdef int_t *ptr_U_sp_i = NULL
    if U.shape[0]:
        ptr_U = &U[0]
    elif U_sp.shape[0]:
        ptr_U_sp = &U_sp[0]
        ptr_U_sp_i = &U_sp_i[0]

    cdef real_t *ptr_U_bin = NULL
    if U_bin.shape[0]:
        ptr_U_bin = &U_bin[0]

    cdef real_t *ptr_U_colmeans = NULL
    if U_colmeans.shape[0]:
        ptr_U_colmeans = &U_colmeans[0]

    cdef real_t *ptr_biasB = NULL
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]

    cdef real_t *ptr_C = NULL
    cdef real_t *ptr_C_bin = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]
    if C_bin.shape[0]:
        ptr_C_bin = &C_bin[0,0]

    cdef real_t *ptr_Bi = NULL
    if Bi.shape[0]:
        ptr_Bi = &Bi[0,0]

    cdef real_t *ptr_TransBtBinvBt = NULL
    cdef real_t *ptr_BtB = NULL
    cdef real_t *ptr_BtXbias = NULL
    cdef real_t *ptr_BeTBeChol = NULL
    cdef real_t *ptr_BiTBi = NULL
    cdef real_t *ptr_CtCw = NULL
    cdef real_t *ptr_TransCtCinvCt = NULL
    cdef real_t *ptr_CtUbias = NULL
    if TransBtBinvBt.shape[0]:
        ptr_TransBtBinvBt = &TransBtBinvBt[0,0]
    if BtB.shape[0]:
        ptr_BtB = &BtB[0,0]
    if BtXbias.shape[0]:
        ptr_BtXbias = &BtXbias[0]
    if BeTBeChol.shape[0]:
        ptr_BeTBeChol = &BeTBeChol[0,0]
    if BiTBi.shape[0]:
        ptr_BiTBi = &BiTBi[0,0]
    if CtCw.shape[0]:
        ptr_CtCw = &CtCw[0,0]
    if TransCtCinvCt.shape[0]:
        ptr_TransCtCinvCt = &TransCtCinvCt[0,0]
    if CtUbias.shape[0]:
        ptr_CtUbias = &CtUbias[0]
    
    cdef real_t Abias = 0;
    cdef real_t *ptr_Abias = NULL
    if user_bias:
        ptr_Abias = &Abias

    cdef real_t *ptr_B_plus_bias = NULL
    if B_plus_bias.shape[0]:
        ptr_B_plus_bias = &B_plus_bias[0,0]

    cdef int_t n_max = B.shape[0]

    if Xa_dense.shape[0] and (Xa_dense.shape[0] < n_orig):
        n_orig = Xa_dense.shape[0]

    cdef np.ndarray[real_t, ndim=1] lam_unique = np.zeros(6, dtype=c_real_t)
    cdef real_t *ptr_lam_unique = NULL
    if lam_bias != lam:
        lam_unique[0] = lam_bias
        lam_unique[2] = lam
        ptr_lam_unique = &lam_unique[0]

    cdef np.ndarray[real_t, ndim=1] l1_lam_unique = np.zeros(6, dtype=c_real_t)
    cdef real_t *ptr_l1_lam_unique = NULL
    if l1_lam_bias != l1_lam:
        l1_lam_unique[0] = l1_lam_bias
        l1_lam_unique[2] = l1_lam
        ptr_l1_lam_unique = &l1_lam_unique[0]

    cdef np.ndarray[real_t, ndim=1] A = np.empty(k_user+k+k_main, dtype=c_real_t)
    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = factors_collective_explicit_single(
            &A[0], ptr_Abias,
            ptr_U, C.shape[0],
            ptr_U_sp, ptr_U_sp_i, U_sp.shape[0],
            ptr_U_bin, C_bin.shape[0],
            NA_as_zero_U, NA_as_zero_X,
            nonneg,
            ptr_C, ptr_C_bin,
            glob_mean, ptr_biasB,
            ptr_U_colmeans,
            ptr_Xa, ptr_Xa_i, Xa.shape[0],
            ptr_Xa_dense, n_orig,
            ptr_weight,
            &B[0,0],
            ptr_Bi, add_implicit_features,
            k, k_user, k_item, k_main,
            lam, ptr_lam_unique,
            l1_lam, ptr_l1_lam_unique,
            scale_lam, scale_lam_sideinfo,
            scale_bias_const, scaling_biasA,
            w_main, w_user, w_implicit,
            n_max, include_all_X,
            ptr_BtB,
            ptr_TransBtBinvBt,
            ptr_BtXbias,
            ptr_BeTBeChol,
            ptr_BiTBi,
            ptr_CtCw,
            ptr_TransCtCinvCt,
            ptr_CtUbias,
            ptr_B_plus_bias
        )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return Abias, A

def call_factors_collective_implicit_single(
        np.ndarray[real_t, ndim=1] Xa,
        np.ndarray[int_t, ndim=1] Xa_i,
        np.ndarray[real_t, ndim=1] U,
        np.ndarray[real_t, ndim=1] U_sp,
        np.ndarray[int_t, ndim=1] U_sp_i,
        np.ndarray[real_t, ndim=1] U_colmeans,
        np.ndarray[real_t, ndim=2] B,
        np.ndarray[real_t, ndim=2] C,
        np.ndarray[real_t, ndim=2] BeTBe,
        np.ndarray[real_t, ndim=2] BtB,
        np.ndarray[real_t, ndim=2] BeTBeChol,
        np.ndarray[real_t, ndim=1] CtUbias,
        int_t k, int_t k_user = 0, int_t k_item = 0, int_t k_main = 0,
        real_t lam = 1e0, real_t l1_lam = 0., real_t alpha = 40.,
        real_t w_main_multiplier = 1.,
        real_t w_user = 1., real_t w_main = 1.,
        bint apply_log_transf = 0,
        bint NA_as_zero_U = 0,
        bint nonneg = 0
    ):

    cdef real_t *ptr_U = NULL
    cdef real_t *ptr_U_sp = NULL
    cdef int_t *ptr_U_sp_i = NULL
    cdef int p = C.shape[0]
    if U.shape[0]:
        ptr_U = &U[0]
        p = U.shape[0]
    elif U_sp.shape[0]:
        ptr_U_sp = &U_sp[0]
        ptr_U_sp_i = &U_sp_i[0]

    cdef real_t *ptr_U_colmeans = NULL
    if U_colmeans.shape[0]:
        ptr_U_colmeans = &U_colmeans[0]

    cdef real_t *ptr_C = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]

    cdef real_t *ptr_BeTBe = NULL
    cdef real_t *ptr_BtB = NULL
    cdef real_t *ptr_BeTBeChol = NULL
    cdef real_t *ptr_CtUbias = NULL
    if BeTBe.shape[0]:
        ptr_BeTBe = &BeTBe[0,0]
    if BtB.shape[0]:
        ptr_BtB = &BtB[0,0]
    if BeTBeChol.shape[0]:
        ptr_BeTBeChol = &BeTBeChol[0,0]
    if CtUbias.shape[0]:
        ptr_CtUbias = &CtUbias[0]

    cdef real_t *ptr_Xa = NULL
    cdef int_t *ptr_Xa_i = NULL
    if Xa.shape[0]:
        ptr_Xa = &Xa[0]
        ptr_Xa_i = &Xa_i[0]

    cdef np.ndarray[real_t, ndim=1] A = np.empty(k_user+k+k_main, dtype=c_real_t)
    cdef int retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = factors_collective_implicit_single(
            &A[0],
            ptr_U, p,
            ptr_U_sp, ptr_U_sp_i, U_sp.shape[0],
            NA_as_zero_U,
            nonneg,
            ptr_U_colmeans,
            &B[0,0], B.shape[0], ptr_C,
            ptr_Xa, ptr_Xa_i, Xa.shape[0],
            k, k_user, k_item, k_main,
            lam, l1_lam, alpha, w_main, w_user,
            w_main_multiplier,
            apply_log_transf,
            ptr_BeTBe,
            ptr_BtB,
            ptr_BeTBeChol,
            ptr_CtUbias
        )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return A

def call_factors_offsets_cold(
        np.ndarray[real_t, ndim=1] U,
        np.ndarray[real_t, ndim=1] U_sp,
        np.ndarray[int_t, ndim=1] U_sp_i,
        np.ndarray[real_t, ndim=2] C,
        np.ndarray[real_t, ndim=1] C_bias,
        int_t k,
        int_t k_sec = 0, int_t k_main = 0,
        real_t w_user = 1.
    ):
    cdef real_t *ptr_U = NULL
    cdef real_t *ptr_U_sp = NULL
    cdef int_t *ptr_U_sp_i = NULL
    if U.shape[0]:
        ptr_U = &U[0]
    elif U_sp.shape[0]:
        ptr_U_sp = &U_sp[0]
        ptr_U_sp_i = &U_sp_i[0]

    cdef real_t *ptr_C_bias = NULL
    if C_bias.shape[0]:
        ptr_C_bias = &C_bias[0]

    cdef np.ndarray[real_t, ndim=1] A = np.empty(k_sec+k+k_main, dtype=c_real_t)

    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = offsets_factors_cold(
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

def call_factors_offsets_explicit_single(
        np.ndarray[real_t, ndim=1] Xa_dense,
        np.ndarray[real_t, ndim=1] W_dense,
        np.ndarray[real_t, ndim=1] Xa,
        np.ndarray[int_t, ndim=1] Xa_i,
        np.ndarray[real_t, ndim=1] W_sp,
        np.ndarray[real_t, ndim=1] U,
        np.ndarray[real_t, ndim=1] U_sp,
        np.ndarray[int_t, ndim=1] U_sp_i,
        np.ndarray[real_t, ndim=1] biasB,
        np.ndarray[real_t, ndim=2] Bm,
        np.ndarray[real_t, ndim=2] Bm_plus_bias,
        np.ndarray[real_t, ndim=2] C,
        np.ndarray[real_t, ndim=1] C_bias,
        np.ndarray[real_t, ndim=2] TransBtBinvBt,
        np.ndarray[real_t, ndim=2] BtB,
        real_t glob_mean,
        int_t k, int_t k_sec = 0, int_t k_main = 0,
        real_t lam = 1e2, real_t lam_bias = 1e2,
        real_t w_user = 1.,
        bint user_bias = 1,
        bint exact = 0, bint output_a = 1
    ):

    cdef real_t *ptr_Xa_dense = NULL
    cdef real_t *ptr_Xa = NULL
    cdef int_t *ptr_Xa_i = NULL
    cdef real_t *ptr_weight = NULL
    if Xa_dense.shape[0]:
        ptr_Xa_dense = &Xa_dense[0]
        if W_dense.shape[0]:
            ptr_weight = &W_dense[0]
    elif Xa.shape[0]:
        ptr_Xa = &Xa[0]
        ptr_Xa_i = &Xa_i[0]
        if W_sp.shape[0]:
            ptr_weight = &W_sp[0]

    cdef real_t *ptr_U = NULL
    cdef real_t *ptr_U_sp = NULL
    cdef int_t *ptr_U_sp_i = NULL
    if U.shape[0]:
        ptr_U = &U[0]
    elif U_sp.shape[0]:
        ptr_U_sp = &U_sp[0]
        ptr_U_sp_i = &U_sp_i[0]

    cdef real_t *ptr_C = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]

    cdef real_t *ptr_biasB = NULL
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]

    cdef real_t *ptr_TransBtBinvBt = NULL
    cdef real_t *ptr_BtB = NULL
    if TransBtBinvBt.shape[0]:
        ptr_TransBtBinvBt = &TransBtBinvBt[0,0]
    if BtB.shape[0]:
        ptr_BtB = &BtB[0,0]
    
    cdef real_t Amean = 0;
    cdef real_t *ptr_Amean = NULL
    if user_bias:
        ptr_Amean = &Amean

    cdef np.ndarray[real_t, ndim=1] A = np.empty(0, dtype=c_real_t)
    cdef real_t *ptr_A = NULL
    if output_a and (k or k_main):
        A = np.empty(k+k_main, dtype=c_real_t)
        ptr_A = &A[0]

    cdef real_t *ptr_Bm_plus_bias = NULL
    if Bm_plus_bias.shape[0]:
        ptr_Bm_plus_bias = &Bm_plus_bias[0,0]

    cdef real_t *ptr_C_bias = NULL
    if C_bias.shape[0]:
        ptr_C_bias = &C_bias[0]

    cdef int_t p = 0
    if C.shape[0]:
        p = C.shape[0]

    cdef np.ndarray[real_t, ndim=1] lam_unique
    cdef real_t *ptr_lam_unique = NULL
    if lam != lam_bias:
        lam_unique = np.zeros(6, dtype=c_real_t)
        lam_unique[0] = lam_bias
        lam_unique[2] = lam
        ptr_lam_unique = &lam_unique[0]

    cdef np.ndarray[real_t, ndim=1] Am = np.empty(k_sec+k+k_main, dtype=c_real_t)
    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = factors_offsets_explicit_single(
            &Am[0], ptr_Amean, ptr_A,
            ptr_U, p,
            ptr_U_sp, ptr_U_sp_i, U_sp.shape[0],
            ptr_Xa, ptr_Xa_i, Xa.shape[0],
            ptr_Xa_dense, Bm.shape[0],
            ptr_weight,
            &Bm[0,0], ptr_C,
            ptr_C_bias,
            glob_mean, ptr_biasB,
            k, k_sec, k_main,
            w_user,
            lam, ptr_lam_unique,
            exact,
            ptr_TransBtBinvBt,
            ptr_BtB,
            ptr_Bm_plus_bias
        )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return Amean, Am, A

def call_factors_offsets_implicit_single(
        np.ndarray[real_t, ndim=1] Xa,
        np.ndarray[int_t, ndim=1] Xa_i,
        np.ndarray[real_t, ndim=1] U,
        np.ndarray[real_t, ndim=1] U_sp,
        np.ndarray[int_t, ndim=1] U_sp_i,
        np.ndarray[real_t, ndim=2] Bm,
        np.ndarray[real_t, ndim=2] C,
        np.ndarray[real_t, ndim=1] C_bias,
        np.ndarray[real_t, ndim=2] TransBtBinvBt,
        np.ndarray[real_t, ndim=2] BtB,
        int_t k,
        real_t lam = 1e2, real_t alpha = 40.,
        bint apply_log_transf = 1,
        bint output_a = 1
    ):

    cdef real_t *ptr_Xa = NULL
    cdef int_t *ptr_Xa_i = NULL
    if Xa.shape[0]:
        ptr_Xa = &Xa[0]
        ptr_Xa_i = &Xa_i[0]

    cdef real_t *ptr_U = NULL
    cdef real_t *ptr_U_sp = NULL
    cdef int_t *ptr_U_sp_i = NULL
    if U.shape[0]:
        ptr_U = &U[0]
    elif U_sp.shape[0]:
        ptr_U_sp = &U_sp[0]
        ptr_U_sp_i = &U_sp_i[0]

    cdef real_t *ptr_C = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]

    cdef real_t *ptr_TransBtBinvBt = NULL
    cdef real_t *ptr_BtB = NULL
    if TransBtBinvBt.shape[0]:
        ptr_TransBtBinvBt = &TransBtBinvBt[0,0]
    if BtB.shape[0]:
        ptr_BtB = &BtB[0,0]

    cdef np.ndarray[real_t, ndim=1] A = np.empty(0, dtype=c_real_t)
    cdef real_t *ptr_A = NULL
    if output_a:
        A = np.empty(k, dtype=c_real_t)
        ptr_A = &A[0]

    cdef real_t *ptr_C_bias = NULL
    if C_bias.shape[0]:
        ptr_C_bias = &C_bias[0]

    cdef int_t p = 0
    if C.shape[0]:
        p = C.shape[0]

    cdef np.ndarray[real_t, ndim=1] Am = np.empty(k, dtype=c_real_t)
    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = factors_offsets_implicit_single(
            &Am[0],
            ptr_U, p,
            ptr_U_sp, ptr_U_sp_i, U_sp.shape[0],
            ptr_Xa, ptr_Xa_i, Xa.shape[0],
            &Bm[0,0], ptr_C,
            ptr_C_bias,
            Am.shape[1], Bm.shape[0],
            lam, alpha,
            apply_log_transf,
            ptr_BtB,
            ptr_A
        )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return Am, A

def call_factors_content_based_single(
        np.ndarray[real_t, ndim=1] U,
        np.ndarray[real_t, ndim=1] U_sp,
        np.ndarray[int_t, ndim=1] U_sp_i,
        np.ndarray[real_t, ndim=2] C,
        np.ndarray[real_t, ndim=1] C_bias
    ):
    cdef real_t *ptr_U = NULL
    cdef real_t *ptr_U_sp = NULL
    cdef int_t *ptr_U_sp_i = NULL
    if U.shape[0]:
        ptr_U = &U[0]
    else:
        ptr_U_sp = &U_sp[0]
        ptr_U_sp_i = &U_sp_i[0]

    cdef real_t *ptr_C_bias = NULL
    if C_bias.shape[0]:
        ptr_C_bias = &C_bias[0]
    
    cdef np.ndarray[real_t, ndim=1] a_vec = np.empty(C.shape[1], dtype=c_real_t)
    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = factors_content_based_single(
            &a_vec[0], C.shape[1],
            ptr_U, C.shape[0],
            ptr_U_sp, ptr_U_sp_i, U_sp.shape[0],
            &C[0,0], ptr_C_bias
        )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")
    return a_vec

def call_predict_multiple(
        np.ndarray[real_t, ndim=2] A,
        np.ndarray[real_t, ndim=2] B,
        np.ndarray[real_t, ndim=1] biasA,
        np.ndarray[real_t, ndim=1] biasB,
        real_t glob_mean,
        np.ndarray[int_t, ndim=1] predA,
        np.ndarray[int_t, ndim=1] predB,
        int_t k, int_t k_user = 0, int_t k_item = 0, int_t k_main = 0,
        int nthreads = 1
    ):
    cdef real_t *ptr_biasA = NULL
    cdef real_t *ptr_biasB = NULL
    if biasA.shape[0]:
        ptr_biasA = &biasA[0]
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]

    cdef np.ndarray[real_t, ndim=1] outp = np.empty(predA.shape[0], dtype=c_real_t)
    if outp.shape[0] == 0:
        return outp

    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        predict_multiple(
            &A[0,0], k_user,
            &B[0,0], k_item,
            ptr_biasA, ptr_biasB,
            glob_mean,
            k, k_main,
            A.shape[0], B.shape[0],
            &predA[0], &predB[0], predA.shape[0],
            &outp[0],
            nthreads
        )
    return outp

def call_predict_X_old_collective_explicit(
        np.ndarray[real_t, ndim=2] A,
        np.ndarray[real_t, ndim=2] B,
        np.ndarray[real_t, ndim=1] biasA,
        np.ndarray[real_t, ndim=1] biasB,
        real_t glob_mean,
        np.ndarray[int_t, ndim=1] predA,
        np.ndarray[int_t, ndim=1] predB,
        int_t k, int_t k_user = 0, int_t k_item = 0, int_t k_main = 0,
        int nthreads = 1
    ):
    cdef real_t *ptr_biasA = NULL
    cdef real_t *ptr_biasB = NULL
    if biasA.shape[0]:
        ptr_biasA = &biasA[0]
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]

    cdef np.ndarray[real_t, ndim=1] outp = np.empty(predA.shape[0], dtype=c_real_t)
    if outp.shape[0] == 0:
        return outp

    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        predict_X_old_collective_explicit(
            &predA[0], &predB[0], &outp[0], predA.shape[0],
            &A[0,0], ptr_biasA,
            &B[0,0], ptr_biasB,
            glob_mean,
            k, k_user, k_item, k_main,
            A.shape[0], B.shape[0],
            nthreads
        )
    return outp
    

def call_topN(
        np.ndarray[real_t, ndim=1] a_vec,
        np.ndarray[real_t, ndim=2] B,
        np.ndarray[real_t, ndim=1] biasB,
        real_t glob_mean, real_t biasA,
        np.ndarray[int_t, ndim=1] include_ix,
        np.ndarray[int_t, ndim=1] exclude_ix,
        int_t n_top,
        int_t k, int_t k_user = 0, int_t k_item = 0, int_t k_main = 0,
        bint output_score = 1,
        int nthreads = 1
    ):

    cdef real_t *ptr_biasB = NULL
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]

    cdef int_t *ptr_include = NULL
    cdef int_t *ptr_exclude = NULL
    if include_ix.shape[0]:
        ptr_include = &include_ix[0]
    if exclude_ix.shape[0]:
        ptr_exclude = &exclude_ix[0]

    cdef np.ndarray[real_t, ndim=1] outp_score = np.empty(0, dtype=c_real_t)
    cdef real_t *ptr_outp_score = NULL
    if output_score:
        outp_score = np.empty(n_top, dtype=c_real_t)
        ptr_outp_score = &outp_score[0]
    cdef np.ndarray[int_t, ndim=1] outp_ix = np.empty(n_top, dtype=ctypes.c_int)
    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = topN(
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
        np.ndarray[real_t, ndim=2] U,
        np.ndarray[int_t, ndim=1] U_row,
        np.ndarray[int_t, ndim=1] U_col,
        np.ndarray[real_t, ndim=1] U_sp,
        np.ndarray[size_t, ndim=1] U_csr_p,
        np.ndarray[int_t, ndim=1] U_csr_i,
        np.ndarray[real_t, ndim=1] U_csr,
        np.ndarray[int_t, ndim=1] ixB,
        np.ndarray[real_t, ndim=2] Bm,
        np.ndarray[real_t, ndim=2] C,
        np.ndarray[real_t, ndim=1] C_bias,
        np.ndarray[real_t, ndim=1] biasB,
        int_t n_new,
        real_t glob_mean,
        int nthreads
    ):

    cdef real_t *ptr_U = NULL
    cdef int_t *ptr_U_row = NULL
    cdef int_t *ptr_U_col = NULL
    cdef real_t *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]

    cdef size_t *ptr_U_csr_p = NULL
    cdef int_t *ptr_U_csr_i = NULL
    cdef real_t *ptr_U_csr = NULL
    if U_csr.shape[0]:
        ptr_U_csr_p = &U_csr_p[0]
        ptr_U_csr_i = &U_csr_i[0]
        ptr_U_csr = &U_csr[0]

    cdef real_t *ptr_biasB = NULL
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]

    cdef real_t *ptr_C_bias = NULL
    if C_bias.shape[0]:
        ptr_C_bias = &C_bias[0]

    cdef np.ndarray[real_t, ndim=1] scores_new = np.empty(n_new, dtype=c_real_t)
    cdef int_t k = C.shape[1]
    
    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = predict_X_old_content_based(
            &scores_new[0], n_new, n_new, k,
            <int_t*>NULL, &ixB[0],
            0, Bm.shape[0],
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
        np.ndarray[real_t, ndim=2] U,
        np.ndarray[int_t, ndim=1] U_row,
        np.ndarray[int_t, ndim=1] U_col,
        np.ndarray[real_t, ndim=1] U_sp,
        np.ndarray[size_t, ndim=1] U_csr_p,
        np.ndarray[int_t, ndim=1] U_csr_i,
        np.ndarray[real_t, ndim=1] U_csr,
        np.ndarray[real_t, ndim=2] I,
        np.ndarray[int_t, ndim=1] I_row,
        np.ndarray[int_t, ndim=1] I_col,
        np.ndarray[real_t, ndim=1] I_sp,
        np.ndarray[size_t, ndim=1] I_csr_p,
        np.ndarray[int_t, ndim=1] I_csr_i,
        np.ndarray[real_t, ndim=1] I_csr,
        np.ndarray[real_t, ndim=2] C,
        np.ndarray[real_t, ndim=1] C_bias,
        np.ndarray[real_t, ndim=2] D,
        np.ndarray[real_t, ndim=1] D_bias,
        int_t n_new,
        real_t glob_mean,
        int nthreads
    ):

    cdef real_t *ptr_U = NULL
    cdef int_t *ptr_U_row = NULL
    cdef int_t *ptr_U_col = NULL
    cdef real_t *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]

    cdef size_t *ptr_U_csr_p = NULL
    cdef int_t *ptr_U_csr_i = NULL
    cdef real_t *ptr_U_csr = NULL
    if U_csr.shape[0]:
        ptr_U_csr_p = &U_csr_p[0]
        ptr_U_csr_i = &U_csr_i[0]
        ptr_U_csr = &U_csr[0]

    cdef real_t *ptr_I = NULL
    cdef int_t *ptr_I_row = NULL
    cdef int_t *ptr_I_col = NULL
    cdef real_t *ptr_I_sp = NULL
    cdef size_t nnz_I = 0
    if I.shape[0]:
        ptr_I = &I[0,0]
    elif I_sp.shape[0]:
        ptr_I_row = &I_row[0]
        ptr_I_col = &I_col[0]
        ptr_I_sp = &I_sp[0]
        nnz_I = I_sp.shape[0]

    cdef size_t *ptr_I_csr_p = NULL
    cdef int_t *ptr_I_csr_i = NULL
    cdef real_t *ptr_I_csr = NULL
    if I_csr.shape[0]:
        ptr_I_csr_p = &I_csr_p[0]
        ptr_I_csr_i = &I_csr_i[0]
        ptr_I_csr = &I_csr[0]

    cdef real_t *ptr_C_bias = NULL
    if C_bias.shape[0]:
        ptr_C_bias = &C_bias[0]

    cdef real_t *ptr_D_bias = NULL
    if D_bias.shape[0]:
        ptr_D_bias = &D_bias[0]
    
    cdef np.ndarray[real_t, ndim=1] scores_new = np.empty(n_new, dtype=c_real_t)
    cdef int_t k = C.shape[1]
    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = predict_X_new_content_based(
            &scores_new[0], n_new,
            n_new, n_new, k,
            <int_t*>NULL, <int_t*>NULL,
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
        np.ndarray[real_t, ndim=1] U,
        np.ndarray[real_t, ndim=1] U_sp,
        np.ndarray[int_t, ndim=1] U_sp_i,
        np.ndarray[real_t, ndim=2] I,
        np.ndarray[int_t, ndim=1] I_row,
        np.ndarray[int_t, ndim=1] I_col,
        np.ndarray[real_t, ndim=1] I_sp,
        np.ndarray[size_t, ndim=1] I_csr_p,
        np.ndarray[int_t, ndim=1] I_csr_i,
        np.ndarray[real_t, ndim=1] I_csr,
        np.ndarray[real_t, ndim=2] C,
        np.ndarray[real_t, ndim=1] C_bias,
        np.ndarray[real_t, ndim=2] D,
        np.ndarray[real_t, ndim=1] D_bias,
        int_t n_new_I,
        real_t glob_mean = 0.,
        int_t n_top = 10, bint output_score = 1,
        int nthreads = 1
    ):
    
    cdef real_t *ptr_U = NULL
    cdef real_t *ptr_U_sp = NULL
    cdef int_t *ptr_U_sp_i = NULL
    if U.shape[0]:
        ptr_U = &U[0]
    elif U_sp.shape[0]:
        ptr_U_sp = &U_sp[0]
        ptr_U_sp_i = &U_sp_i[0]

    cdef real_t *ptr_I = NULL
    cdef int_t *ptr_I_row = NULL
    cdef int_t *ptr_I_col = NULL
    cdef real_t *ptr_I_sp = NULL
    cdef size_t nnz_I = 0
    if I.shape[0]:
        ptr_I = &I[0,0]
    elif I_sp.shape[0]:
        ptr_I_row = &I_row[0]
        ptr_I_col = &I_col[0]
        ptr_I_sp = &I_sp[0]
        nnz_I = I_sp.shape[0]

    cdef size_t *ptr_I_csr_p = NULL
    cdef int_t *ptr_I_csr_i = NULL
    cdef real_t *ptr_I_csr = NULL
    if I_csr.shape[0]:
        ptr_I_csr_p = &I_csr_p[0]
        ptr_I_csr_i = &I_csr_i[0]
        ptr_I_csr = &I_csr[0]

    cdef real_t *ptr_C_bias = NULL
    if C_bias.shape[0]:
        ptr_C_bias = &C_bias[0]

    cdef real_t *ptr_D_bias = NULL
    if D_bias.shape[0]:
        ptr_D_bias = &D_bias[0]

    cdef np.ndarray[real_t, ndim=1] scores_new = np.empty(0, dtype=c_real_t)
    cdef real_t *ptr_scores_new = NULL
    if output_score:
        scores_new = np.empty(n_top, dtype=c_real_t)
        ptr_scores_new = &scores_new[0]
    if n_top <= 0:
        raise ValueError("'n_top' must be a positive integer.")
    cdef np.ndarray[int_t, ndim=1] rank_new = np.empty(n_top, dtype=ctypes.c_int)
    cdef int_t k = C.shape[1]
    
    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = topN_new_content_based(
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
        np.ndarray[int_t, ndim=1] ixA,
        np.ndarray[int_t, ndim=1] ixB,
        np.ndarray[real_t, ndim=1] X,
        np.ndarray[real_t, ndim=1] W_sp,
        np.ndarray[real_t, ndim=2] Xfull,
        np.ndarray[real_t, ndim=2] W_dense,
        int_t m, int_t n,
        real_t lam_user = 1e2, real_t lam_item = 1e2,
        real_t alpha = 40.,
        bint user_bias = 0,
        bint implicit = 0, bint adjust_weight = 1,
        bint scale_lam = 0,
        bint scale_bias_const = 0,
        bint apply_log_transf = 0,
        bint nonneg = 0,
        bint center = 1,
        bint NA_as_zero = 0,
        int nthreads = 1
    ):
    cdef real_t *ptr_Xfull = NULL
    cdef int_t *ptr_ixA = NULL
    cdef int_t *ptr_ixB = NULL
    cdef real_t *ptr_X = NULL
    cdef size_t nnz = 0
    cdef real_t glob_mean = 0
    cdef real_t *ptr_weight = NULL
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

    cdef real_t *ptr_biasA = NULL
    cdef real_t *ptr_biasB = NULL

    cdef np.ndarray[real_t, ndim=1] values
    if user_bias:
        values = np.empty(m+n, dtype=c_real_t)
        ptr_biasA = &values[0]
        ptr_biasB = &values[m]
    else:
        values = np.empty(n, dtype=c_real_t)
        ptr_biasB = &values[0]

    cdef real_t w_main_multiplier = 1.

    cdef real_t *ptr_glob_mean = NULL
    if (center):
        ptr_glob_mean = &glob_mean

    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = fit_most_popular(
            ptr_biasA, ptr_biasB,
            ptr_glob_mean,
            lam_user, lam_item,
            scale_lam, scale_bias_const,
            alpha,
            m, n,
            ptr_ixA, ptr_ixB, ptr_X, nnz,
            ptr_Xfull,
            ptr_weight,
            implicit, adjust_weight, apply_log_transf,
            nonneg, NA_as_zero,
            &w_main_multiplier,
            nthreads
        )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    if user_bias:
        return glob_mean, values[:m], values[m:], w_main_multiplier
    else:
        return glob_mean, np.empty(0, dtype=c_real_t), values, w_main_multiplier

def call_fit_content_based_lbfgs(
        np.ndarray[int_t, ndim=1] ixA,
        np.ndarray[int_t, ndim=1] ixB,
        np.ndarray[real_t, ndim=1] X,
        np.ndarray[real_t, ndim=1] W,
        np.ndarray[real_t, ndim=2] Xfull,
        np.ndarray[real_t, ndim=2] Wfull,
        np.ndarray[real_t, ndim=2] U,
        np.ndarray[int_t, ndim=1] U_row,
        np.ndarray[int_t, ndim=1] U_col,
        np.ndarray[real_t, ndim=1] U_sp,
        np.ndarray[real_t, ndim=2] I,
        np.ndarray[int_t, ndim=1] I_row,
        np.ndarray[int_t, ndim=1] I_col,
        np.ndarray[real_t, ndim=1] I_sp,
        int_t m, int_t n, int_t p, int_t q,
        int_t k=50,
        bint user_bias=1, bint item_bias=1,
        bint add_intercepts=1,
        real_t lam=1e2,
        np.ndarray[real_t, ndim=1] lam_unique=np.empty(0, dtype=c_real_t),
        bint verbose=1, int_t print_every=10,
        int_t n_corr_pairs=5, int_t maxiter=400,
        int nthreads=1, bint prefer_onepass=0,
        int_t seed=1, bint handle_interrupt=1,
        bint start_with_ALS=1
    ):

    cdef real_t *ptr_Xfull = NULL
    cdef real_t *ptr_weight = NULL
    cdef int_t *ptr_ixA = NULL
    cdef int_t *ptr_ixB = NULL
    cdef real_t *ptr_X = NULL
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

    cdef real_t *ptr_U = NULL
    cdef int_t *ptr_U_row = NULL
    cdef int_t *ptr_U_col = NULL
    cdef real_t *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]

    cdef real_t *ptr_I = NULL
    cdef int_t *ptr_I_row = NULL
    cdef int_t *ptr_I_col = NULL
    cdef real_t *ptr_I_sp = NULL
    cdef size_t nnz_I = 0
    if I.shape[0]:
        ptr_I = &I[0,0]
    elif I_sp.shape[0]:
        ptr_I_row = &I_row[0]
        ptr_I_col = &I_col[0]
        ptr_I_sp = &I_sp[0]
        nnz_I = I_sp.shape[0]

    cdef real_t *ptr_lam_unique = NULL
    if lam_unique.shape[0]:
        ptr_lam_unique = &lam_unique[0]

    cdef np.ndarray[real_t, ndim=2] Am = np.empty((m, k), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] Bm = np.empty((n, k), dtype=c_real_t)

    rs = np.random.Generator(np.random.MT19937(seed = seed))
    cdef np.ndarray[real_t, ndim=2] C = rs.standard_normal(size=(p,k), dtype = c_real_t) / 100
    cdef np.ndarray[real_t, ndim=2] D = rs.standard_normal(size=(q,k), dtype = c_real_t) / 100
    cdef np.ndarray[real_t, ndim=1] C_bias = np.zeros(0, dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=1] D_bias = np.zeros(0, dtype=c_real_t)
    cdef real_t *ptr_C = &C[0,0]
    cdef real_t *ptr_D = &D[0,0]
    cdef real_t *ptr_C_bias = NULL
    cdef real_t *ptr_D_bias = NULL
    if add_intercepts:
        C_bias = np.zeros(k, dtype=c_real_t)
        D_bias = np.zeros(k, dtype=c_real_t)
        ptr_C_bias = &C_bias[0]
        ptr_D_bias = &D_bias[0]

    cdef np.ndarray[real_t, ndim=1] biasA = np.zeros(0, dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=1] biasB = np.zeros(0, dtype=c_real_t)
    cdef real_t *ptr_biasA = NULL
    cdef real_t *ptr_biasB = NULL
    if user_bias:
        biasA = np.empty(m, dtype=c_real_t)
        ptr_biasA = &biasA[0]
    if item_bias:
        biasB = np.empty(n, dtype=c_real_t)
        ptr_biasB = &biasB[0]

    cdef real_t glob_mean
    cdef int_t niter, nfev

    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = fit_content_based_lbfgs(
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
    elif (retval == 3) and (not handle_interrupt):
        raise InterruptedError("Procedure was interrupted")

    return biasA, biasB, C, D, C_bias, D_bias, Am, Bm, glob_mean, niter, nfev

def call_factors_collective_explicit_multiple(
        np.ndarray[int_t, ndim=1] ixA,
        np.ndarray[int_t, ndim=1] ixB,
        np.ndarray[real_t, ndim=1] X,
        np.ndarray[size_t, ndim=1] Xcsr_p,
        np.ndarray[int_t, ndim=1] Xcsr_i,
        np.ndarray[real_t, ndim=1] Xcsr,
        np.ndarray[real_t, ndim=1] W,
        np.ndarray[real_t, ndim=2] Xfull,
        np.ndarray[real_t, ndim=2] Wfull,
        np.ndarray[real_t, ndim=2] U,
        np.ndarray[int_t, ndim=1] U_row,
        np.ndarray[int_t, ndim=1] U_col,
        np.ndarray[real_t, ndim=1] U_sp,
        np.ndarray[size_t, ndim=1] U_csr_p,
        np.ndarray[int_t, ndim=1] U_csr_i,
        np.ndarray[real_t, ndim=1] U_csr,
        np.ndarray[real_t, ndim=2] Ub,
        np.ndarray[real_t, ndim=1] U_colmeans,
        np.ndarray[real_t, ndim=1] biasB,
        np.ndarray[real_t, ndim=2] B,
        np.ndarray[real_t, ndim=2] B_plus_bias,
        np.ndarray[real_t, ndim=2] Bi,
        np.ndarray[real_t, ndim=2] C,
        np.ndarray[real_t, ndim=2] C_bin,
        np.ndarray[real_t, ndim=2] BtB,
        np.ndarray[real_t, ndim=2] TransBtBinvBt,
        np.ndarray[real_t, ndim=1] BtXbias,
        np.ndarray[real_t, ndim=2] BeTBeChol,
        np.ndarray[real_t, ndim=2] BiTBi,
        np.ndarray[real_t, ndim=2] TransCtCinvCt,
        np.ndarray[real_t, ndim=2] CtCw,
        np.ndarray[real_t, ndim=1] CtUbias,
        int_t m_u, int_t m_x,
        real_t glob_mean,
        int_t n_orig,
        int_t k, int_t k_user = 0, int_t k_item = 0, int_t k_main = 0,
        real_t lam = 1e2, real_t lam_bias = 1e2,
        real_t l1_lam = 0., real_t l1_lam_bias = 0.,
        bint scale_lam = 0, bint scale_lam_sideinfo = 0,
        bint scale_bias_const = 0, real_t scaling_biasA = 0,
        real_t w_user = 1., real_t w_main = 1., real_t w_implicit = 0.5,
        bint user_bias = 1,
        bint NA_as_zero_U = 0, bint NA_as_zero_X = 0,
        bint nonneg = 0,
        bint add_implicit_features = 0,
        bint include_all_X = 1,
        int nthreads = 1
    ):
    if add_implicit_features and not max(Bi.shape[0], Bi.shape[1], BiTBi.shape[0], BiTBi.shape[1]):
        raise ValueError("Cannot use 'add_implicit_features' with empty 'Bi'.")
    
    cdef int_t m_ubin = Ub.shape[0]
    cdef int_t p = C.shape[0]
    cdef int_t pbin = C_bin.shape[0]

    cdef real_t *ptr_Xfull = NULL
    cdef real_t *ptr_weight = NULL
    cdef int_t *ptr_ixA = NULL
    cdef int_t *ptr_ixB = NULL
    cdef real_t *ptr_X = NULL
    cdef size_t nnz = 0
    cdef size_t *ptr_Xcsr_p = NULL
    cdef int_t *ptr_Xcsr_i = NULL
    cdef real_t *ptr_Xcsr = NULL
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

    cdef real_t *ptr_U = NULL
    cdef int_t *ptr_U_row = NULL
    cdef int_t *ptr_U_col = NULL
    cdef real_t *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    cdef real_t *ptr_U_colmeans = NULL
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
    cdef int_t *ptr_U_csr_i = NULL
    cdef real_t *ptr_U_csr = NULL
    if U_csr.shape[0]:
        ptr_U_csr_p = &U_csr_p[0]
        ptr_U_csr_i = &U_csr_i[0]
        ptr_U_csr = &U_csr[0]

    cdef real_t *ptr_Ub = NULL
    if Ub.shape[0]:
        ptr_Ub = &Ub[0,0]

    cdef real_t *ptr_biasB = NULL
    cdef real_t *ptr_B_plus_bias = NULL
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]
    if B_plus_bias.shape[0]:
        ptr_B_plus_bias = &B_plus_bias[0,0]

    cdef real_t *ptr_Bi = NULL
    if Bi.shape[0]:
        ptr_Bi = &Bi[0,0]

    cdef real_t *ptr_C = NULL
    cdef real_t *ptr_C_bin = NULL
    cdef real_t *ptr_TransCtCinvCt = NULL
    cdef real_t *ptr_CtCw = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]
    if C_bin.shape[0]:
        ptr_C_bin = &C_bin[0,0]
    if TransCtCinvCt.shape[0]:
        ptr_TransCtCinvCt = &TransCtCinvCt[0,0]
    if CtCw.shape[0]:
        ptr_CtCw = &CtCw[0,0]

    cdef real_t *ptr_TransBtBinvBt = NULL
    cdef real_t *ptr_BtB = NULL
    cdef real_t *ptr_BtXbias = NULL
    cdef real_t *ptr_BeTBeChol = NULL
    cdef real_t *ptr_BiTBi = NULL
    cdef real_t *ptr_CtUbias = NULL
    if TransBtBinvBt.shape[0]:
        ptr_TransBtBinvBt = &TransBtBinvBt[0,0]
    if BtB.shape[0]:
        ptr_BtB = &BtB[0,0]
    if BtXbias.shape[0]:
        ptr_BtXbias = &BtXbias[0]
    if BeTBeChol.shape[0]:
        ptr_BeTBeChol = &BeTBeChol[0,0]
    if BiTBi.shape[0]:
        ptr_BiTBi = &BiTBi[0,0]
    if CtUbias.shape[0]:
        ptr_CtUbias = &CtUbias[0]

    cdef int_t m = max([m_x, m_u, m_ubin])
    cdef np.ndarray[real_t, ndim=2] A = np.empty((m, k_user+k+k_main), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=1] biasA = np.empty(0, dtype=c_real_t)
    if m == 0:
        return A, biasA
    cdef real_t *ptr_biasA = NULL
    if user_bias:
        biasA = np.empty(m, dtype=c_real_t)
        ptr_biasA = &biasA[0]

    cdef np.ndarray[real_t, ndim=1] lam_unique = np.zeros(6, dtype=c_real_t)
    cdef real_t *ptr_lam_unique = NULL
    if lam_bias != lam:
        lam_unique[0] = lam_bias
        lam_unique[2] = lam
        ptr_lam_unique = &lam_unique[0]

    cdef np.ndarray[real_t, ndim=1] l1_lam_unique = np.zeros(6, dtype=c_real_t)
    cdef real_t *ptr_l1_lam_unique = NULL
    if l1_lam_bias != l1_lam:
        l1_lam_unique[0] = l1_lam_bias
        l1_lam_unique[2] = l1_lam
        ptr_l1_lam_unique = &l1_lam_unique[0]

    if (Xfull.shape[1]) and (Xfull.shape[1] != n_orig):
        n_orig = Xfull.shape[1]

    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = factors_collective_explicit_multiple(
            &A[0,0], ptr_biasA, m,
            ptr_U, m_u, p,
            NA_as_zero_U, NA_as_zero_X,
            nonneg,
            ptr_U_row, ptr_U_col, ptr_U_sp, nnz_U,
            ptr_U_csr_p, ptr_U_csr_i, ptr_U_csr,
            ptr_Ub, m_ubin, pbin,
            ptr_C, ptr_C_bin,
            glob_mean, ptr_biasB,
            ptr_U_colmeans,
            ptr_X, ptr_ixA, ptr_ixB, nnz,
            ptr_Xcsr_p, ptr_Xcsr_i, ptr_Xcsr,
            ptr_Xfull, n_orig,
            ptr_weight,
            &B[0,0],
            ptr_Bi, add_implicit_features,
            k, k_user, k_item, k_main,
            lam, ptr_lam_unique,
            l1_lam, ptr_l1_lam_unique,
            scale_lam, scale_lam_sideinfo,
            scale_bias_const, scaling_biasA,
            w_main, w_user, w_implicit,
            B.shape[0], include_all_X,
            ptr_BtB,
            ptr_TransBtBinvBt,
            ptr_BtXbias,
            ptr_BeTBeChol,
            ptr_BiTBi,
            ptr_TransCtCinvCt,
            ptr_CtCw,
            ptr_B_plus_bias,
            ptr_CtUbias,
            nthreads
        )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return A, biasA

def call_factors_collective_implicit_multiple(
        np.ndarray[int_t, ndim=1] ixA,
        np.ndarray[int_t, ndim=1] ixB,
        np.ndarray[real_t, ndim=1] X,
        np.ndarray[size_t, ndim=1] Xcsr_p,
        np.ndarray[int_t, ndim=1] Xcsr_i,
        np.ndarray[real_t, ndim=1] Xcsr,
        np.ndarray[real_t, ndim=2] U,
        np.ndarray[int_t, ndim=1] U_row,
        np.ndarray[int_t, ndim=1] U_col,
        np.ndarray[real_t, ndim=1] U_sp,
        np.ndarray[size_t, ndim=1] U_csr_p,
        np.ndarray[int_t, ndim=1] U_csr_i,
        np.ndarray[real_t, ndim=1] U_csr,
        np.ndarray[real_t, ndim=1] U_colmeans,
        np.ndarray[real_t, ndim=2] B,
        np.ndarray[real_t, ndim=2] C,
        np.ndarray[real_t, ndim=2] BeTBe,
        np.ndarray[real_t, ndim=2] BtB,
        np.ndarray[real_t, ndim=2] BeTBeChol,
        np.ndarray[real_t, ndim=1] CtUbias,
        int_t n, int_t m_u, int_t m_x,
        int_t k, int_t k_user = 0, int_t k_item = 0, int_t k_main = 0,
        real_t lam = 1e2, real_t l1_lam = 0., real_t alpha = 40.,
        real_t w_main_multiplier = 1.,
        real_t w_user = 1., real_t w_main = 1.,
        bint apply_log_transf = 0,
        bint NA_as_zero_U = 0,
        bint nonneg = 0,
        int nthreads = 1
    ):
    cdef int_t m = max([m_u, m_x])

    cdef int_t *ptr_ixA = NULL
    cdef int_t *ptr_ixB = NULL
    cdef real_t *ptr_X = NULL
    cdef size_t nnz = 0
    cdef size_t *ptr_Xcsr_p = NULL
    cdef int_t *ptr_Xcsr_i = NULL
    cdef real_t *ptr_Xcsr = NULL
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

    cdef real_t *ptr_U = NULL
    cdef int_t *ptr_U_row = NULL
    cdef int_t *ptr_U_col = NULL
    cdef real_t *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    cdef real_t *ptr_U_colmeans = NULL
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
    cdef int_t *ptr_U_csr_i = NULL
    cdef real_t *ptr_U_csr = NULL
    if U_csr.shape[0]:
        ptr_U_csr_p = &U_csr_p[0]
        ptr_U_csr_i = &U_csr_i[0]
        ptr_U_csr = &U_csr[0]

    cdef real_t *ptr_C = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]

    cdef real_t *ptr_BeTBe = NULL
    cdef real_t *ptr_BtB = NULL
    cdef real_t *ptr_BeTBeChol = NULL
    cdef real_t *ptr_CtUbias = NULL
    if BeTBe.shape[0]:
        ptr_BeTBe = &BeTBe[0,0]
    if BtB.shape[0]:
        ptr_BtB = &BtB[0,0]
    if BeTBeChol.shape[0]:
        ptr_BeTBeChol = &BeTBeChol[0,0]
    if CtUbias.shape[0]:
        ptr_CtUbias = &CtUbias[0]

    cdef np.ndarray[real_t, ndim=2] A = np.empty((m, k_user+k+k_main), dtype=c_real_t)
    if m == 0:
        return A

    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = factors_collective_implicit_multiple(
            &A[0,0], m,
            ptr_U, m_u, C.shape[0],
            NA_as_zero_U,
            nonneg,
            ptr_U_row, ptr_U_col, ptr_U_sp, nnz_U,
            ptr_U_csr_p, ptr_U_csr_i, ptr_U_csr,
            ptr_X, ptr_ixA, ptr_ixB, nnz,
            ptr_Xcsr_p, ptr_Xcsr_i, ptr_Xcsr,
            &B[0,0], n,
            ptr_C,
            ptr_U_colmeans,
            k, k_user, k_item, k_main,
            lam, l1_lam, alpha, w_main, w_user,
            w_main_multiplier,
            apply_log_transf,
            ptr_BeTBe,
            ptr_BtB,
            ptr_BeTBeChol,
            ptr_CtUbias,
            nthreads
        )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return A

def call_factors_offsets_explicit_multiple(
        np.ndarray[int_t, ndim=1] ixA,
        np.ndarray[int_t, ndim=1] ixB,
        np.ndarray[real_t, ndim=1] X,
        np.ndarray[size_t, ndim=1] Xcsr_p,
        np.ndarray[int_t, ndim=1] Xcsr_i,
        np.ndarray[real_t, ndim=1] Xcsr,
        np.ndarray[real_t, ndim=1] W,
        np.ndarray[real_t, ndim=2] Xfull,
        np.ndarray[real_t, ndim=2] Wfull,
        np.ndarray[real_t, ndim=2] U,
        np.ndarray[int_t, ndim=1] U_row,
        np.ndarray[int_t, ndim=1] U_col,
        np.ndarray[real_t, ndim=1] U_sp,
        np.ndarray[size_t, ndim=1] U_csr_p,
        np.ndarray[int_t, ndim=1] U_csr_i,
        np.ndarray[real_t, ndim=1] U_csr,
        np.ndarray[real_t, ndim=1] biasB,
        np.ndarray[real_t, ndim=2] Bm,
        np.ndarray[real_t, ndim=2] Bm_plus_bias,
        np.ndarray[real_t, ndim=2] C,
        np.ndarray[real_t, ndim=1] C_bias,
        np.ndarray[real_t, ndim=2] TransBtBinvBt,
        np.ndarray[real_t, ndim=2] BtB,
        real_t glob_mean,
        int_t m, int_t n,
        int_t k, int_t k_sec = 0, int_t k_main = 0,
        real_t lam = 1e2, real_t lam_bias = 1e2,
        real_t w_user = 1.,
        bint user_bias = 1,
        bint exact = 0, bint output_a = 1,
        int nthreads = 1
    ):

    cdef real_t *ptr_Xfull = NULL
    cdef real_t *ptr_weight = NULL
    cdef int_t *ptr_ixA = NULL
    cdef int_t *ptr_ixB = NULL
    cdef real_t *ptr_X = NULL
    cdef size_t nnz = 0
    cdef size_t *ptr_Xcsr_p = NULL
    cdef int_t *ptr_Xcsr_i = NULL
    cdef real_t *ptr_Xcsr = NULL
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

    cdef real_t *ptr_U = NULL
    cdef int_t *ptr_U_row = NULL
    cdef int_t *ptr_U_col = NULL
    cdef real_t *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]

    cdef size_t *ptr_U_csr_p = NULL
    cdef int_t *ptr_U_csr_i = NULL
    cdef real_t *ptr_U_csr = NULL
    if U_csr.shape[0]:
        ptr_U_csr_p = &U_csr_p[0]
        ptr_U_csr_i = &U_csr_i[0]
        ptr_U_csr = &U_csr[0]

    cdef real_t *ptr_C = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]

    cdef real_t *ptr_biasB = NULL
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]

    cdef real_t *ptr_TransBtBinvBt = NULL
    cdef real_t *ptr_BtB = NULL
    if TransBtBinvBt.shape[0]:
        ptr_TransBtBinvBt = &TransBtBinvBt[0,0]
    if BtB.shape[0]:
        ptr_BtB = &BtB[0,0]

    cdef real_t *ptr_Bm_plus_bias = NULL
    if Bm_plus_bias.shape[0]:
        ptr_Bm_plus_bias = &Bm_plus_bias[0,0]

    cdef real_t *ptr_C_bias = NULL
    if C_bias.shape[0]:
        ptr_C_bias = &C_bias[0]

    cdef np.ndarray[real_t, ndim=2] Am = np.empty((m, k_sec+k+k_main), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] A = np.empty((0,0), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=1] biasA = np.empty(0, dtype=c_real_t)
    if m == 0:
        return Am, biasA, A
    cdef real_t *ptr_biasA = NULL
    cdef real_t *ptr_A = NULL
    if user_bias:
        biasA = np.empty(m, dtype=c_real_t)
        ptr_biasA = &biasA[0]
    if output_a:
        A = np.empty((m, k+k_main), dtype=c_real_t)
        ptr_A = &A[0,0]

    cdef np.ndarray[real_t, ndim=1] lam_unique = np.zeros(6, dtype=c_real_t)
    cdef real_t *ptr_lam_unique = NULL
    if lam != lam_bias:
        lam_unique[0] = lam_bias
        lam_unique[2] = lam
        ptr_lam_unique = &lam_unique[0]

    cdef int_t p = C.shape[0]

    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = factors_offsets_explicit_multiple(
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
        np.ndarray[int_t, ndim=1] ixA,
        np.ndarray[int_t, ndim=1] ixB,
        np.ndarray[real_t, ndim=1] X,
        np.ndarray[size_t, ndim=1] Xcsr_p,
        np.ndarray[int_t, ndim=1] Xcsr_i,
        np.ndarray[real_t, ndim=1] Xcsr,
        np.ndarray[real_t, ndim=2] U,
        np.ndarray[int_t, ndim=1] U_row,
        np.ndarray[int_t, ndim=1] U_col,
        np.ndarray[real_t, ndim=1] U_sp,
        np.ndarray[size_t, ndim=1] U_csr_p,
        np.ndarray[int_t, ndim=1] U_csr_i,
        np.ndarray[real_t, ndim=1] U_csr,
        np.ndarray[real_t, ndim=2] Bm,
        np.ndarray[real_t, ndim=2] C,
        np.ndarray[real_t, ndim=1] C_bias,
        np.ndarray[real_t, ndim=2] BtB,
        int_t m, int_t n,
        int_t k,
        real_t lam = 1e2, real_t alpha = 1.,
        bint apply_log_transf = 0,
        bint output_a = 1,
        int nthreads = 1
    ):

    cdef int_t *ptr_ixA = NULL
    cdef int_t *ptr_ixB = NULL
    cdef real_t *ptr_X = NULL
    cdef size_t nnz = 0
    cdef size_t *ptr_Xcsr_p = NULL
    cdef int_t *ptr_Xcsr_i = NULL
    cdef real_t *ptr_Xcsr = NULL
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

    cdef real_t *ptr_U = NULL
    cdef int_t *ptr_U_row = NULL
    cdef int_t *ptr_U_col = NULL
    cdef real_t *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    if U.shape[0]:
        ptr_U = &U[0,0]
    elif U_sp.shape[0]:
        ptr_U_row = &U_row[0]
        ptr_U_col = &U_col[0]
        ptr_U_sp = &U_sp[0]
        nnz_U = U_sp.shape[0]

    cdef size_t *ptr_U_csr_p = NULL
    cdef int_t *ptr_U_csr_i = NULL
    cdef real_t *ptr_U_csr = NULL
    if U_csr.shape[0]:
        ptr_U_csr_p = &U_csr_p[0]
        ptr_U_csr_i = &U_csr_i[0]
        ptr_U_csr = &U_csr[0]

    cdef real_t *ptr_C = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]

    cdef real_t *ptr_BtB = NULL
    if BtB.shape[0]:
        ptr_BtB = &BtB[0,0]

    cdef real_t *ptr_C_bias = NULL
    if C_bias.shape[0]:
        ptr_C_bias = &C_bias[0]

    cdef np.ndarray[real_t, ndim=2] Am = np.empty((m, k), dtype=c_real_t)
    cdef np.ndarray[real_t, ndim=2] A = np.empty((0,0), dtype=c_real_t)
    if m == 0:
        return Am, A
    cdef real_t *ptr_A = NULL
    if output_a:
        A = np.empty((m, k), dtype=c_real_t)
        ptr_A = &A[0,0]

    cdef int_t p = C.shape[0]

    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = factors_offsets_implicit_multiple(
            &Am[0,0], m,
            ptr_A,
            ptr_U, p,
            ptr_U_row, ptr_U_col, ptr_U_sp, nnz_U,
            ptr_U_csr_p, ptr_U_csr_i, ptr_U_csr,
            ptr_X, ptr_ixA, ptr_ixB, nnz,
            ptr_Xcsr_p, ptr_Xcsr_i, ptr_Xcsr,
            &Bm[0,0], ptr_C,
            ptr_C_bias,
            k, n,
            lam, alpha,
            apply_log_transf,
            ptr_BtB,
            nthreads
        )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return Am, A
    
def call_impute_X_collective_explicit(
        np.ndarray[real_t, ndim=2] Xfull,
        np.ndarray[real_t, ndim=2] Wfull,
        np.ndarray[real_t, ndim=2] U,
        np.ndarray[int_t, ndim=1] U_row,
        np.ndarray[int_t, ndim=1] U_col,
        np.ndarray[real_t, ndim=1] U_sp,
        np.ndarray[size_t, ndim=1] U_csr_p,
        np.ndarray[int_t, ndim=1] U_csr_i,
        np.ndarray[real_t, ndim=1] U_csr,
        np.ndarray[real_t, ndim=2] Ub,
        np.ndarray[real_t, ndim=1] U_colmeans,
        np.ndarray[real_t, ndim=1] biasB,
        np.ndarray[real_t, ndim=2] B,
        np.ndarray[real_t, ndim=2] B_plus_bias,
        np.ndarray[real_t, ndim=2] Bi,
        np.ndarray[real_t, ndim=2] C,
        np.ndarray[real_t, ndim=2] C_bin,
        np.ndarray[real_t, ndim=2] BtB,
        np.ndarray[real_t, ndim=2] TransBtBinvBt,
        np.ndarray[real_t, ndim=2] BeTBeChol,
        np.ndarray[real_t, ndim=2] BiTBi,
        np.ndarray[real_t, ndim=2] TransCtCinvCt,
        np.ndarray[real_t, ndim=2] CtCw,
        np.ndarray[real_t, ndim=1] CtUbias,
        int_t m_u,
        real_t glob_mean,
        int_t n_orig,
        int_t k, int_t k_user = 0, int_t k_item = 0, int_t k_main = 0,
        real_t lam = 1e2, real_t lam_bias = 1e2,
        real_t l1_lam = 0., real_t l1_lam_bias = 0.,
        bint scale_lam = 0, bint scale_lam_sideinfo = 0,
        bint scale_bias_const = 0, real_t scaling_biasA = 0,
        real_t w_user = 1., real_t w_main = 1., real_t w_implicit = 0.5,
        bint user_bias = 1,
        bint NA_as_zero_U = 0,
        bint nonneg = 0,
        bint add_implicit_features = 0,
        bint include_all_X = 1,
        int nthreads = 1
    ):
    
    cdef int_t n_max = B.shape[0]
    cdef int_t p = C.shape[0]
    cdef int_t pbin = C_bin.shape[0]
    cdef int_t m = Xfull.shape[0]
    if min(m, Xfull.shape[1]) <= 0:
        raise ValueError("Invalid input dimensions.")
    if add_implicit_features and not max(Bi.shape[0], Bi.shape[1], BiTBi.shape[0], BiTBi.shape[1]):
        raise ValueError("Cannot use 'add_implicit_features' without 'Bi'.")
    
    cdef np.ndarray[real_t, ndim=1] lam_unique = np.zeros(6, dtype=c_real_t)
    cdef real_t *ptr_lam_unique = NULL
    if (lam != lam_bias):
        lam_unique[0] = lam_bias
        lam_unique[2] = lam
        ptr_lam_unique = &lam_unique[0]

    cdef np.ndarray[real_t, ndim=1] l1_lam_unique = np.zeros(6, dtype=c_real_t)
    cdef real_t *ptr_l1_lam_unique = NULL
    if (l1_lam != l1_lam_bias):
        l1_lam_unique[0] = l1_lam_bias
        l1_lam_unique[2] = l1_lam
        ptr_l1_lam_unique = &l1_lam_unique[0]

    cdef real_t *ptr_Xfull = &Xfull[0,0]
    cdef real_t *ptr_weight = NULL
    if Wfull.shape[0]:
        ptr_weight = &Wfull[0,0]

    cdef real_t *ptr_U = NULL
    cdef int_t *ptr_U_row = NULL
    cdef int_t *ptr_U_col = NULL
    cdef real_t *ptr_U_sp = NULL
    cdef size_t nnz_U = 0
    cdef real_t *ptr_U_colmeans = NULL
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
    cdef int_t *ptr_U_csr_i = NULL
    cdef real_t *ptr_U_csr = NULL
    if U_csr.shape[0]:
        ptr_U_csr_p = &U_csr_p[0]
        ptr_U_csr_i = &U_csr_i[0]
        ptr_U_csr = &U_csr[0]

    cdef real_t *ptr_Ub = NULL
    cdef int_t m_ubin = 0
    if Ub.shape[0]:
        ptr_Ub = &Ub[0,0]
        m_ubin = Ub.shape[0]

    cdef real_t *ptr_B = &B[0,0]
    cdef real_t *ptr_biasB = NULL
    cdef real_t *ptr_B_plus_bias = NULL
    cdef real_t *ptr_Bi = NULL
    if biasB.shape[0]:
        ptr_biasB = &biasB[0]
    if B_plus_bias.shape[0]:
        ptr_B_plus_bias = &B_plus_bias[0,0]
    if Bi.shape[0]:
        ptr_Bi = &Bi[0,0]

    cdef real_t *ptr_C = NULL
    cdef real_t *ptr_C_bin = NULL
    cdef real_t *ptr_TransCtCinvCt = NULL
    cdef real_t *ptr_CtCw = NULL
    cdef real_t *ptr_CtUbias = NULL
    if C.shape[0]:
        ptr_C = &C[0,0]
    if C_bin.shape[0]:
        ptr_C_bin = &C_bin[0,0]
    if TransCtCinvCt.shape[0]:
        ptr_TransCtCinvCt = &TransCtCinvCt[0,0]
    if CtCw.shape[0]:
        ptr_CtCw = &CtCw[0,0]
    if CtUbias.shape[0]:
        ptr_CtUbias = &CtUbias[0]

    cdef real_t *ptr_TransBtBinvBt = NULL
    cdef real_t *ptr_BtB = NULL
    cdef real_t *ptr_BeTBeChol = NULL
    cdef real_t *ptr_BiTBi = NULL
    if TransBtBinvBt.shape[0]:
        ptr_TransBtBinvBt = &TransBtBinvBt[0,0]
    if BtB.shape[0]:
        ptr_BtB = &BtB[0,0]
    if BeTBeChol.shape[0]:
        ptr_BeTBeChol = &BeTBeChol[0,0]
    if BiTBi.shape[0]:
        ptr_BiTBi = &BiTBi[0,0]

    if (Xfull.shape[1]) and (Xfull.shape[1] != n_orig):
        n_orig = Xfull.shape[1]

    cdef int_t retval = 0
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        retval = impute_X_collective_explicit(
            m, user_bias,
            ptr_U, m_u, p,
            NA_as_zero_U,
            nonneg,
            ptr_U_row, ptr_U_col, ptr_U_sp, nnz_U,
            ptr_U_csr_p, ptr_U_csr_i, ptr_U_csr,
            ptr_Ub, m_ubin, pbin,
            ptr_C, ptr_C_bin,
            glob_mean, ptr_biasB,
            ptr_U_colmeans,
            ptr_Xfull, n_orig,
            ptr_weight,
            ptr_B,
            ptr_Bi, add_implicit_features,
            k, k_user, k_item, k_main,
            lam, ptr_lam_unique,
            l1_lam, ptr_l1_lam_unique,
            scale_lam, scale_lam_sideinfo,
            scale_bias_const, scaling_biasA,
            w_main, w_user, w_implicit,
            n_max, include_all_X,
            ptr_BtB,
            ptr_TransBtBinvBt,
            ptr_BeTBeChol,
            ptr_BiTBi,
            ptr_TransCtCinvCt,
            ptr_CtCw,
            ptr_CtUbias,
            ptr_B_plus_bias,
            nthreads
        )
    if retval == 1:
        raise MemoryError("Could not allocate sufficient memory.")

    return Xfull
