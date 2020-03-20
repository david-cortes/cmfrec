/*******************************************************************************
    Collective Matrix Factorization
    -------------------------------
    
    This is a module for multi-way factorization of sparse and dense matrices
    intended to be used for recommender system with explicit feedback data plus
    side information about users and/or items.

    The reference papers are:
        (a) Cortes, David.
            "Cold-start recommendations in Collective Matrix Factorization."
            arXiv preprint arXiv:1809.00366 (2018).
        (b) Singh, Ajit P., and Geoffrey J. Gordon.
            "Relational learning via collective matrix factorization."
            Proceedings of the 14th ACM SIGKDD international conference on
            Knowledge discovery and data mining. 2008.
        (c) Hu, Yifan, Yehuda Koren, and Chris Volinsky.
            "Collaborative filtering for implicit feedback datasets."
            2008 Eighth IEEE International Conference on Data Mining.
            Ieee, 2008.

    For information about the models offered here and how they are fit to
    the data, see the files 'collective.c' and 'offsets.c'.

    Written for C99 standard and OpenMP version 2.0 or higher, and aimed to be
    used either as a stand-alone program, or wrapped into scripting languages
    such as Python and R.
    <https://www.github.com/david-cortes/cmfrec>

    

    MIT License:

    Copyright (c) 2020 David Cortes

    All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.

*******************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <signal.h>
#ifndef _FOR_R
    #include <stdio.h>
#endif
#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() 0
#endif

#ifdef _FOR_PYTHON
    #if defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
        #include "findblas.h" /* https://www.github.com/david-cortes/findblas */
    #endif
#elif defined(_FOR_R)
    #include <R.h>
    #include <Rinternals.h>
    #include <R_ext/Print.h>
    #include <R_ext/BLAS.h>
    #include <R_ext/Lapack.h>
    #define USE_DOUBLE
    #define printf Rprintf
    #define fprintf(f, message) REprintf(message)
#else
    #include "cblas.h"
    #include "lapack.h"
#endif

/* Aliasing for compiler optimizations */
#ifdef __cplusplus
    #if defined(__GNUG__) || defined(__GNUC__) || defined(_MSC_VER) || defined(__clang__) || defined(__INTEL_COMPILER)
        #define restrict __restrict
    #else
        #define restrict 
    #endif
#elif defined(_MSC_VER)
    #define restrict __restrict
#elif !defined(__STDC_VERSION__) || (__STDC_VERSION__ < 199901L)
    #define restrict 
#endif

/*    OpenMP < 3.0 (e.g. MSVC as of 2020) does not support parallel for's with unsigned iterators,
    and does not support declaring the iterator type in the loop itself */
#ifdef _OPENMP
    #if (_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64) /* OpenMP < 3.0 */
        #define size_t_for 
    #else
        #define size_t_for size_t
    #endif
#else
    #define size_t_for size_t
#endif

#ifndef isnan
    #ifdef _isnan
        #define isnan _isnan
    #else
        #define isnan(x) ( (x) != (x) )
    #endif
#endif

#ifdef USE_DOUBLE
    #define LBFGS_FLOAT 64
    #define FPnum double
    #define exp_t exp
    #define cblas_tdot cblas_ddot
    #define cblas_tcopy cblas_dcopy
    #define cblas_taxpy cblas_daxpy
    #define cblas_tscal cblas_dscal
    #define cblas_tsyr cblas_dsyr
    #define cblas_tsyrk cblas_dsyrk
    #define cblas_tnrm2 cblas_dnrm2
    #define cblas_tgemm cblas_dgemm
    #define cblas_tgemv cblas_dgemv
    #define cblas_tsymv cblas_dsymv
    #define tlacpy_ dlacpy_
    #define tposv_ dposv_
    #define tlarnv_ dlarnv_
    #define tpotrf_ dpotrf_
    #define tpotrs_ dpotrs_
    #define tgels_ dgels_
#else
    #define LBFGS_FLOAT 32
    #define FPnum float
    #define exp_t expf
    #define cblas_tdot cblas_sdot
    #define cblas_tcopy cblas_scopy
    #define cblas_taxpy cblas_saxpy
    #define cblas_tscal cblas_sscal
    #define cblas_tsyr cblas_ssyr
    #define cblas_tsyrk cblas_ssyrk
    #define cblas_tnrm2 cblas_snrm2
    #define cblas_tgemm cblas_sgemm
    #define cblas_tgemv cblas_sgemv
    #define cblas_tsymv cblas_ssymv
    #define tlacpy_ slacpy_
    #define tposv_ sposv_
    #define tlarnv_ slarnv_
    #define tpotrf_ spotrf_
    #define tpotrs_ spotrs_
    #define tgels_ sgels_
#endif
#if !defined(_FOR_R) && !defined(LAPACK_H)
void tposv_(const char*, const int*, const int*, const FPnum*, const int*, const FPnum*, const int*, const int*);
void tlacpy_(const char*, const int*, const int*, const FPnum*, const int*, const FPnum*, const int*);
void tlarnv_(const int*, const int*, const int*, const FPnum*);
void tpotrf_(const char*, const int*, const FPnum*, const int*, const int*);
void tpotrs_(const char*, const int*, const int*, const FPnum*, const int*, const FPnum*, const int*, const int*);
void tgels_(const char*, const int*, const int*, const int*,
            const FPnum*, const int*, const FPnum*, const int*,
            const FPnum*, const int*, const int*);
#endif

#if !defined(_WIN32) && !defined(_WIN64) && !defined(_MSC_VER)
#ifndef CBLAS_H
typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef CBLAS_ORDER CBLAS_LAYOUT;

FPnum  cblas_tdot(const int n, const FPnum  *x, const int incx, const FPnum  *y, const int incy);
void cblas_tcopy(const int n, const FPnum *x, const int incx, FPnum *y, const int incy);
void cblas_taxpy(const int n, const FPnum alpha, const FPnum *x, const int incx, FPnum *y, const int incy);
void cblas_tscal(const int N, const FPnum alpha, FPnum *X, const int incX);
void cblas_tsyr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const int N, const FPnum alpha, const FPnum *X, const int incX, FPnum *A, const int lda);
void cblas_tsyrk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans,
         const int N, const int K, const FPnum alpha, const FPnum *A, const int lda, const FPnum beta, FPnum *C, const int ldc);
FPnum  cblas_tnrm2 (const int N, const FPnum  *X, const int incX);
void cblas_tgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
         const FPnum alpha, const FPnum *A, const int lda, const FPnum *B, const int ldb, const FPnum beta, FPnum *C, const int ldc);
void cblas_tgemv(const CBLAS_ORDER order,  const CBLAS_TRANSPOSE trans,  const int m, const int n,
         const FPnum alpha, const FPnum  *a, const int lda,  const FPnum  *x, const int incx,  const FPnum beta,  FPnum  *y, const int incy);
void cblas_tsymv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const int N, const FPnum alpha, const FPnum *A,
                 const int lda, const FPnum *X, const int incX, const FPnum beta, FPnum *Y, const int incY);
#endif
#endif

#include "lbfgs.h"

#define square(x) ( (x) * (x) )
#define cap_to_4(x) (((x) > 4)? 4 : (x))
#define max2(a, b) ((a) >= ((b))? (a) : (b))
#define min2(a, b) ((a) <= ((b))? (a) : (b))


/* helpers.c */
void set_to_zero(FPnum *arr, const size_t n, int nthreads);
void copy_arr(FPnum *restrict src, FPnum *restrict dest, size_t n, int nthreads);
int count_NAs(FPnum arr[], size_t n, int nthreads);
void count_NAs_by_row
(
    FPnum *restrict arr, int m, int n,
    int *restrict cnt_NA, int nthreads,
    bool *restrict full_dense, bool *restrict near_dense
);
void count_NAs_by_col
(
    FPnum *restrict arr, int m, int n,
    int *restrict cnt_NA,
    bool *restrict full_dense, bool *restrict near_dense
);
void sum_by_rows(FPnum *restrict A, FPnum *restrict outp, int m, int n, int nthreads);
void sum_by_cols(FPnum *restrict A, FPnum *restrict outp, int m, int n, size_t lda, int nthreads);
void mat_plus_rowvec(FPnum *restrict A, FPnum *restrict b, int m, int n, int nthreads);
void mat_plus_colvec(FPnum *restrict A, FPnum *restrict b, FPnum alpha, int m, int n, size_t lda, int nthreads);
void mat_minus_rowvec2
(
    FPnum *restrict Xfull,
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    FPnum *restrict b, int m, int n, int nthreads
);
void mat_minus_colvec2
(
    FPnum *restrict Xfull,
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    FPnum *restrict b, int m, int n, int nthreads
);
void nan_to_zero(FPnum *restrict arr, FPnum *restrict comp, size_t n, int nthreads);
void mult_if_non_nan(FPnum *restrict arr, FPnum *restrict comp, FPnum *restrict w, size_t n, int nthreads);
void mult_elemwise(FPnum *restrict inout, FPnum *restrict other, size_t n, int nthreads);
FPnum sum_squares(FPnum *restrict arr, size_t n, int nthreads);
void saxpy_large(FPnum *restrict A, FPnum x, FPnum *restrict Y, size_t n, int nthreads);
void sscal_large(FPnum *restrict arr, FPnum alpha, size_t n, int nthreads);
int rnorm(FPnum *restrict arr, size_t n, int seed, int nthreads);
void reduce_mat_sum(FPnum *restrict outp, size_t lda, FPnum *restrict inp,
                    int m, int n, int nthreads);
void exp_neg_x(FPnum *restrict arr, size_t n, int nthreads);
void add_to_diag(FPnum *restrict A, FPnum val, int n);
FPnum sum_sq_div_w(FPnum *restrict arr, FPnum *restrict w, size_t n, bool compensated, int nthreads);
void sgemm_sp_dense
(
    int m, int n, FPnum alpha,
    long indptr[], int indices[], FPnum values[],
    FPnum DenseMat[], size_t ldb,
    FPnum OutputMat[], size_t ldc,
    int nthreads
);
void sgemv_dense_sp
(
    int m, int n,
    FPnum alpha, FPnum DenseMat[], size_t lda,
    int ixB[], FPnum vec_sp[], size_t nnz,
    FPnum OutputVec[]
);
void sgemv_dense_sp_weighted
(
    int m, int n,
    FPnum alpha[], FPnum DenseMat[], size_t lda,
    int ixB[], FPnum vec_sp[], size_t nnz,
    FPnum OutputVec[]
);
void sgemv_dense_sp_weighted2
(
    int m, int n,
    FPnum alpha[], FPnum alpha2, FPnum DenseMat[], size_t lda,
    int ixB[], FPnum vec_sp[], size_t nnz,
    FPnum OutputVec[]
);
void copy_mat
(
    int m, int n,
    FPnum *restrict A, int lda,
    FPnum *restrict B, int ldb
);
void sum_mat
(
    int m, int n,
    FPnum *restrict A, int lda,
    FPnum *restrict B, int ldb
);
void AtAinvAt_plus_chol(FPnum *restrict A, int lda, int offset,
                        FPnum *restrict AtAinvAt_out,
                        FPnum *restrict AtAw_out,
                        FPnum *restrict AtAchol_out,
                        FPnum lam, FPnum lam_last, int m, int n, FPnum w,
                        FPnum *restrict buffer_FPnum,
                        bool no_reg_to_AtA);
void transpose_mat(FPnum *restrict A, int m, int n, FPnum *restrict buffer_FPnum);
void transpose_mat2(FPnum *restrict A, int m, int n, FPnum *restrict outp);
int coo_to_csr_plus_alloc
(
    int *restrict Xrow, int *restrict Xcol, FPnum *restrict Xval,
    FPnum *restrict W,
    int m, int n, size_t nnz,
    long *restrict *csr_p, int *restrict *csr_i, FPnum *restrict *csr_v,
    FPnum *restrict *csr_w
);
void coo_to_csr
(
    int *restrict Xrow, int *restrict Xcol, FPnum *restrict Xval,
    FPnum *restrict W,
    int m, int n, size_t nnz,
    long *restrict csr_p, int *restrict csr_i, FPnum *restrict csr_v,
    FPnum *restrict csr_w
);
void coo_to_csr_and_csc
(
    int *restrict Xrow, int *restrict Xcol, FPnum *restrict Xval,
    FPnum *restrict W, int m, int n, size_t nnz,
    long *restrict csr_p, int *restrict csr_i, FPnum *restrict csr_v,
    long *restrict csc_p, int *restrict csc_i, FPnum *restrict csc_v,
    FPnum *restrict csr_w, FPnum *restrict csc_w,
    int nthreads
);
void row_means_csr(long indptr[], FPnum *restrict values,
                   FPnum *restrict output, int m, int nthreads);
extern bool should_stop_procedure;
void set_interrup_global_variable(int s);
int lbfgs_printer_collective
(
    void *instance,
    const lbfgsFPnumval_t *x,
    const lbfgsFPnumval_t *g,
    const lbfgsFPnumval_t fx,
    const lbfgsFPnumval_t xnorm,
    const lbfgsFPnumval_t gnorm,
    const lbfgsFPnumval_t step,
    size_t n,
    int k,
    int ls
);
int lbfgs_printer_offsets
(
    void *instance,
    const lbfgsFPnumval_t *x,
    const lbfgsFPnumval_t *g,
    const lbfgsFPnumval_t fx,
    const lbfgsFPnumval_t xnorm,
    const lbfgsFPnumval_t gnorm,
    const lbfgsFPnumval_t step,
    size_t n,
    int k,
    int ls
);
bool check_is_sorted(int arr[], int n);
void qs_argpartition(int arr[], FPnum values[], int n, int k);
void solve_conj_grad
(
    FPnum *restrict A, FPnum *restrict b, int m,
    FPnum *restrict buffer_FPnum
);
void append_ones_last_col
(
    FPnum *restrict orig, int m, int n,
    FPnum *restrict outp
);


/* common.c */
FPnum fun_grad_cannonical_form
(
    FPnum *restrict A, int lda, FPnum *restrict B, int ldb,
    FPnum *restrict g_A, FPnum *restrict g_B,
    int m, int n, int k,
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    FPnum *restrict Xfull, bool full_dense,
    long Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    long Xcsc_p[], int Xcsc_i[], FPnum *restrict Xcsc,
    bool user_bias, bool item_bias,
    FPnum *restrict biasA, FPnum *restrict biasB,
    FPnum *restrict g_biasA, FPnum *restrict g_biasB,
    FPnum *restrict weight, FPnum *restrict weightR, FPnum *restrict weightC,
    FPnum scaling,
    FPnum *restrict buffer_FPnum,
    FPnum *restrict buffer_mt,
    bool overwrite_grad,
    int nthreads
);
void factors_closed_form
(
    FPnum *restrict a_vec, int k,
    FPnum *restrict B, int n, int ldb,
    FPnum *restrict Xa_dense, bool full_dense,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    FPnum *restrict weight,
    FPnum *restrict buffer_FPnum,
    FPnum lam, FPnum w, FPnum lam_last,
    FPnum *restrict precomputedBtBinvBt,
    FPnum *restrict precomputedBtBw, int cnt_NA, int strideBtB,
    FPnum *restrict precomputedBtBchol, bool NA_as_zero, bool use_cg,
    bool force_add_diag
);
void factors_implicit
(
    FPnum *restrict a_vec, int k,
    FPnum *restrict B, size_t ldb,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    FPnum lam, FPnum alpha,
    FPnum *restrict precomputedBtBw, int strideBtB,
    bool zero_out, bool use_cg,
    FPnum *restrict buffer_FPnum,
    bool force_add_diag
);
FPnum fun_grad_Adense
(
    FPnum *restrict g_A,
    FPnum *restrict A, int lda,
    FPnum *restrict B, int ldb,
    int m, int n, int k,
    FPnum *restrict Xfull, FPnum *restrict weight,
    FPnum lam, FPnum w, FPnum lam_last,
    bool do_B, bool reset_grad,
    int nthreads,
    FPnum *restrict buffer_FPnum
);
void add_lam_to_grad_and_fun
(
    FPnum *restrict fun,
    FPnum *restrict grad,
    FPnum *restrict A,
    int m, int k, int lda,
    FPnum lam, int nthreads
);
typedef struct data_fun_grad_Adense {
    int lda;
    FPnum *B; int ldb;
    int m; int n; int k;
    FPnum *Xfull; FPnum *weight;
    FPnum lam; FPnum w; FPnum lam_last;
    int nthreads;
    FPnum *buffer_FPnum;
} data_fun_grad_Adense;
typedef struct data_fun_grad_Bdense {
    FPnum *A; int lda;
    int ldb;
    int m; int n; int k;
    FPnum *Xfull; FPnum *weight;
    FPnum lam; FPnum w; FPnum lam_last;
    int nthreads;
    FPnum *buffer_FPnum;
} data_fun_grad_Bdense;
FPnum wrapper_fun_grad_Adense
(
    void *instance,
    lbfgsFPnumval_t *x,
    lbfgsFPnumval_t *g,
    const size_t n,
    const lbfgsFPnumval_t step
);
FPnum wrapper_fun_grad_Bdense
(
    void *instance,
    lbfgsFPnumval_t *x,
    lbfgsFPnumval_t *g,
    const size_t n,
    const lbfgsFPnumval_t step
);
void buffer_size_optimizeA
(
    size_t *buffer_size, size_t *buffer_lbfgs_size,
    int m, int n, int k, int lda, int nthreads,
    bool do_B, bool NA_as_zero, bool use_cg,
    bool full_dense, bool near_dense,
    bool has_dense, bool has_weight
);
void optimizeA
(
    FPnum *restrict A, int lda,
    FPnum *restrict B, int ldb,
    int m, int n, int k,
    long Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    FPnum *restrict Xfull, bool full_dense, bool near_dense,
    int cnt_NA[], FPnum *restrict weight, bool NA_as_zero,
    FPnum lam, FPnum w, FPnum lam_last,
    bool do_B,
    int nthreads,
    bool use_cg,
    FPnum *restrict buffer_FPnum,
    iteration_data_t *buffer_lbfgs_iter
);
void optimizeA_implicit
(
    FPnum *restrict A, size_t lda,
    FPnum *restrict B, size_t ldb,
    int m, int n, int k,
    long Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    FPnum lam, FPnum alpha,
    int nthreads,
    bool use_cg,
    FPnum *restrict buffer_FPnum
);
int initialize_biases
(
    FPnum *restrict glob_mean, FPnum *restrict biasA, FPnum *restrict biasB,
    bool user_bias, bool item_bias,
    FPnum lam_user, FPnum lam_item,
    int m, int n,
    int m_bias, int n_bias,
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    FPnum *restrict Xfull, FPnum *restrict Xtrans,
    long Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    long Xcsc_p[], int Xcsc_i[], FPnum *restrict Xcsc,
    int nthreads
);
int center_by_cols
(
    FPnum *restrict col_means,
    FPnum *restrict Xfull, int m, int n,
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    long Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    long Xcsc_p[], int Xcsc_i[], FPnum *restrict Xcsc,
    int nthreads
);
void predict_multiple
(
    FPnum *restrict A, int k_user,
    FPnum *restrict B, int k_item,
    FPnum *restrict biasA, FPnum *restrict biasB,
    FPnum glob_mean,
    int k, int k_main,
    int predA[], int predB[], size_t nnz,
    FPnum *restrict outp,
    int nthreads
);
int cmp_int(const void *a, const void *b);
extern FPnum *ptr_FPnum_glob;
#if !defined(_WIN32) && !defined(_WIN64)
#pragma omp threadprivate(ptr_FPnum_glob)
/* Note: will not be used inside OMP, this is a precausion just in case */
#endif
int cmp_argsort(const void *a, const void *b);
int topN
(
    FPnum *restrict a_vec, int k_user,
    FPnum *restrict B, int k_item,
    FPnum *restrict biasB,
    FPnum glob_mean, FPnum biasA,
    int k, int k_main,
    int *restrict include_ix, int n_include,
    int *restrict exclude_ix, int n_exclude,
    int *restrict outp_ix, FPnum *restrict outp_score,
    int n_top, int n, int nthreads
);
int fit_most_popular
(
    FPnum *restrict biasA, FPnum *restrict biasB,
    FPnum *restrict glob_mean,
    FPnum lam_user, FPnum lam_item,
    FPnum alpha,
    int m, int n,
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    FPnum *restrict Xfull,
    FPnum *restrict weight,
    bool implicit, bool adjust_weight,
    FPnum *restrict w_main_multiplier,
    int nthreads
);

/* collective.c */
void nvars_collective_fun_grad
(
    size_t m, size_t n, size_t m_u, size_t n_i, size_t m_ubin, size_t n_ibin,
    size_t p, size_t q, size_t pbin, size_t qbin,
    size_t nnz, size_t nnz_U, size_t nnz_I,
    size_t k, size_t k_main, size_t k_user, size_t k_item,
    bool user_bias, bool item_bias, size_t nthreads,
    FPnum *X, FPnum *Xfull, FPnum *Xcsr,
    FPnum *U, FPnum *Ub, FPnum *II, FPnum *Ib,
    FPnum *U_sp, FPnum *U_csr, FPnum *I_sp, FPnum *I_csr,
    size_t *nvars, size_t *nbuffer, size_t *nbuffer_mt
);
FPnum collective_fun_grad
(
    FPnum *restrict values, FPnum *restrict grad,
    int m, int n, int k,
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    FPnum *restrict Xfull,
    long Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    long Xcsc_p[], int Xcsc_i[], FPnum *restrict Xcsc,
    FPnum *restrict weight, FPnum *restrict weightR, FPnum *restrict weightC,
    bool user_bias, bool item_bias,
    FPnum lam, FPnum *restrict lam_unique,
    FPnum *restrict U, int m_u, int p, bool U_has_NA,
    FPnum *restrict II, int n_i, int q, bool I_has_NA,
    FPnum *restrict Ub, int m_ubin, int pbin, bool Ub_has_NA,
    FPnum *restrict Ib, int n_ibin, int qbin, bool Ib_has_NA,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    int I_row[], int I_col[], FPnum *restrict I_sp, size_t nnz_I,
    long U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    long U_csc_p[], int U_csc_i[], FPnum *restrict U_csc,
    long I_csr_p[], int I_csr_i[], FPnum *restrict I_csr,
    long I_csc_p[], int I_csc_i[], FPnum *restrict I_csc,
    FPnum *restrict buffer_FPnum, FPnum *restrict buffer_mt,
    int k_main, int k_user, int k_item,
    FPnum w_main, FPnum w_user, FPnum w_item,
    int nthreads
);
FPnum collective_fun_grad_bin
(
    FPnum *restrict A, int lda, FPnum *restrict Cb, int ldc,
    FPnum *restrict g_A, FPnum *restrict g_Cb,
    FPnum *restrict Ub,
    int m, int pbin, int k,
    bool Ub_has_NA, double w_user,
    FPnum *restrict buffer_FPnum,
    int nthreads
);
FPnum collective_fun_grad_single
(
    FPnum *restrict a_vec, FPnum *restrict g_A,
    int k, int k_user, int k_item, int k_main,
    FPnum *restrict u_vec, int p,
    int u_vec_ixB[], FPnum *restrict u_vec_sp, size_t nnz_u_vec,
    FPnum *restrict u_bin_vec, int pbin,
    bool u_vec_has_NA, bool u_bin_vec_has_NA,
    FPnum *restrict B, int n,
    FPnum *restrict C, FPnum *restrict Cb,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    FPnum *restrict Xa_dense,
    FPnum *restrict weight,
    FPnum *restrict buffer_FPnum,
    FPnum lam, FPnum w_main, FPnum w_user, FPnum lam_last
);
typedef struct data_factors_fun_grad {
    int k; int k_user; int k_item; int k_main;
    FPnum *u_vec; int p;
    int *u_vec_ixB; FPnum *u_vec_sp; size_t nnz_u_vec;
    FPnum *u_bin_vec; int pbin;
    bool u_vec_has_NA; bool u_bin_vec_has_NA;
    FPnum *B; int n;
    FPnum *C; FPnum *Cb;
    FPnum *Xa; int *ixB; FPnum *weight; size_t nnz;
    FPnum *Xa_dense;
    FPnum *buffer_FPnum;
    FPnum lam; FPnum w_main; FPnum w_user; FPnum lam_last;
} data_factors_fun_grad;
FPnum wrapper_factors_fun_grad
(
    void *instance,
    lbfgsFPnumval_t *x,
    lbfgsFPnumval_t *g,
    const size_t n,
    const lbfgsFPnumval_t step
);
int collective_factors_lbfgs
(
    FPnum *restrict a_vec,
    int k, int k_user, int k_item, int k_main,
    FPnum *restrict u_vec, int p,
    int u_vec_ixB[], FPnum *restrict u_vec_sp, size_t nnz_u_vec,
    FPnum *restrict u_bin_vec, int pbin,
    bool u_vec_has_NA, bool u_bin_vec_has_NA,
    FPnum *restrict B, int n,
    FPnum *restrict C, FPnum *restrict Cb,
    FPnum *restrict Xa, int ixB[], FPnum *restrict weight, size_t nnz,
    FPnum *restrict Xa_dense,
    FPnum *restrict buffer_FPnum,
    FPnum lam, FPnum w_main, FPnum w_user, FPnum lam_last
);
void collective_closed_form_block
(
    FPnum *restrict a_vec,
    int k, int k_user, int k_item, int k_main, int k_item_BtB, int padding,
    FPnum *restrict Xa_dense,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    int u_vec_ixB[], FPnum *restrict u_vec_sp, size_t nnz_u_vec,
    FPnum *restrict u_vec,
    bool NA_as_zero_X, bool NA_as_zero_U,
    FPnum *restrict B, int n,
    FPnum *restrict C, int p,
    FPnum *restrict weight,
    FPnum lam, FPnum w_user, FPnum w_main, FPnum lam_last,
    FPnum *restrict precomputedBtBw, int cnt_NA_x,
    FPnum *restrict precomputedCtCw, int cnt_NA_u,
    bool add_X, bool add_U, bool use_cg,
    FPnum *restrict buffer_FPnum
);
void collective_closed_form_block_implicit
(
    FPnum *restrict a_vec,
    int k, int k_user, int k_item, int k_main,
    FPnum *restrict B, int n, FPnum *restrict C, int p,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    FPnum *restrict u_vec, int cnt_NA_u,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    bool NA_as_zero_U,
    FPnum lam, FPnum alpha, FPnum w_main, FPnum w_user,
    FPnum *restrict precomputedBeTBe,
    FPnum *restrict precomputedBtB,
    bool add_U, bool shapes_match, bool use_cg,
    FPnum *restrict buffer_FPnum
);
void optimizeA_collective_implicit
(
    FPnum *restrict A, FPnum *restrict B, FPnum *restrict C,
    int m, int m_u, int n, int p,
    int k, int k_main, int k_user, int k_item,
    long Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    long U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict U, int cnt_NA_u[],
    bool full_dense_u, bool near_dense_u, bool NA_as_zero_U,
    FPnum lam, FPnum alpha, FPnum w_main, FPnum w_user,
    int nthreads,
    bool use_cg,
    FPnum *restrict buffer_FPnum,
    iteration_data_t *buffer_lbfgs_iter
);
int collective_factors_cold
(
    FPnum *restrict a_vec,
    FPnum *restrict u_vec, int p,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    FPnum *restrict u_bin_vec, int pbin,
    FPnum *restrict C, FPnum *restrict Cb,
    FPnum *restrict CtCinvCt,
    FPnum *restrict CtCw,
    FPnum *restrict CtCchol,
    FPnum *restrict col_means,
    int k, int k_user, int k_main,
    FPnum lam, FPnum w_user,
    bool NA_as_zero_U
);
int collective_factors_warm
(
    FPnum *restrict a_vec, FPnum *restrict a_bias,
    FPnum *restrict u_vec, int p,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    FPnum *restrict u_bin_vec, int pbin,
    FPnum *restrict C, FPnum *restrict Cb,
    FPnum glob_mean, FPnum *restrict biasB,
    FPnum *restrict col_means,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    FPnum *restrict Xa_dense, int n,
    FPnum *restrict weight,
    FPnum *restrict B,
    int k, int k_user, int k_item, int k_main,
    FPnum lam, FPnum w_user, FPnum w_main, FPnum lam_bias,
    FPnum *restrict BtBinvBt,
    FPnum *restrict BtBw,
    FPnum *restrict BtBchol,
    FPnum *restrict CtCw,
    int k_item_BtB,
    bool NA_as_zero_U, bool NA_as_zero_X,
    FPnum *restrict B_plus_bias
);
int collective_factors_warm_implicit
(
    FPnum *restrict a_vec,
    FPnum *restrict u_vec, int p,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    bool NA_as_zero_U,
    FPnum *restrict col_means,
    FPnum *restrict B, int n, FPnum *restrict C,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    int k, int k_user, int k_item, int k_main,
    FPnum lam, FPnum alpha, FPnum w_user, FPnum w_main,
    FPnum w_main_multiplier,
    FPnum *restrict precomputedBeTBe,
    FPnum *restrict precomputedBtB,
    FPnum *restrict precomputedBtB_shrunk,
    int k_item_BtB
);
FPnum fun_grad_A_collective
(
    FPnum *restrict A, FPnum *restrict g_A,
    FPnum *restrict B, FPnum *restrict C,
    int m, int m_u, int n, int p,
    int k, int k_main, int k_user, int k_item, int padding,
    FPnum *restrict Xfull, bool full_dense,
    long Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    FPnum *restrict weight,
    long U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict U, bool full_dense_u,
    FPnum lam, FPnum w_main, FPnum w_user, FPnum lam_last,
    bool do_B,
    int nthreads,
    FPnum *restrict buffer_FPnum
);
typedef struct data_fun_grad_Adense_col {
    FPnum *B; FPnum *C;
    int m; int m_u; int n; int p;
    int k; int k_main; int k_user; int k_item; int padding;
    FPnum *Xfull; bool full_dense;
    long *Xcsr_p; int *Xcsr_i; FPnum *Xcsr;
    FPnum *weight;
    long *U_csr_p; int *U_csr_i; FPnum *U_csr;
    FPnum *U; bool full_dense_u;
    FPnum lam; FPnum w_main; FPnum w_user; FPnum lam_last;
    bool do_B;
    int nthreads;
    FPnum *buffer_FPnum;
} data_fun_grad_Adense_col;
FPnum wrapper_fun_grad_Adense_col
(
    void *instance,
    lbfgsFPnumval_t *x,
    lbfgsFPnumval_t *g,
    const size_t n,
    const lbfgsFPnumval_t step
);
void buffer_size_optimizeA_collective
(
    size_t *buffer_size, size_t *buffer_lbfgs_size,
    int m, int n, int k, int k_user, int k_main, int padding,
    int m_u, int p, int nthreads,
    bool do_B, bool NA_as_zero_X, bool NA_as_zero_U, bool use_cg,
    bool full_dense, bool near_dense,
    bool has_dense, bool has_weight,
    bool full_dense_u, bool near_dense_u, bool has_dense_u
);
void optimizeA_collective
(
    FPnum *restrict A, FPnum *restrict B, FPnum *restrict C,
    int m, int m_u, int n, int p,
    int k, int k_main, int k_user, int k_item, int padding,
    long Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    FPnum *restrict Xfull, bool full_dense, bool near_dense,
    int cnt_NA_x[], FPnum *restrict weight, bool NA_as_zero_X,
    long U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict U, int cnt_NA_u[],
    bool full_dense_u, bool near_dense_u, bool NA_as_zero_U,
    FPnum lam, FPnum w_main, FPnum w_user, FPnum lam_last,
    bool do_B,
    int nthreads,
    bool use_cg,
    FPnum *restrict buffer_FPnum,
    iteration_data_t *buffer_lbfgs_iter
);
void build_BeTBe
(
    FPnum *restrict bufferBeTBe,
    FPnum *restrict B, FPnum *restrict C,
    int k, int k_user, int k_main, int k_item, int padding,
    int n, int p,
    FPnum lam, FPnum w_main, FPnum w_user
);
void build_BtB_CtC
(
    FPnum *restrict BtB, FPnum *restrict CtC,
    FPnum *restrict B, int n,
    FPnum *restrict C, int p,
    int k, int k_user, int k_main, int k_item, int padding,
    FPnum w_main, FPnum w_user,
    FPnum *restrict weight
);
void build_XBw
(
    FPnum *restrict A, int lda,
    FPnum *restrict B, int ldb,
    FPnum *restrict Xfull,
    int m, int n, int k, FPnum w,
    bool do_B, bool overwrite
);
void preprocess_vec
(
    FPnum *restrict vec_full, int n,
    int *restrict ix_vec, FPnum *restrict vec_sp, size_t nnz,
    FPnum glob_mean, FPnum lam,
    FPnum *restrict col_means,
    FPnum *restrict vec_mean,
    int *restrict cnt_NA
);
int convert_sparse_X
(
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    long **Xcsr_p, int **Xcsr_i, FPnum *restrict *Xcsr,
    long **Xcsc_p, int **Xcsc_i, FPnum *restrict *Xcsc,
    FPnum *restrict weight, FPnum *restrict *weightR, FPnum *restrict *weightC,
    int m, int n, int nthreads
);
int preprocess_sideinfo_matrix
(
    FPnum *U, int m_u, int p,
    int U_row[], int U_col[], FPnum *U_sp, size_t nnz_U,
    FPnum *U_colmeans, FPnum *restrict *Utrans, FPnum *restrict *U_orig,
    long **U_csr_p, int **U_csr_i, FPnum *restrict *U_csr,
    long **U_csc_p, int **U_csc_i, FPnum *restrict *U_csc,
    int *restrict *cnt_NA_u_byrow, int *restrict *cnt_NA_u_bycol,
    bool *full_dense_u, bool *near_dense_u_row, bool *near_dense_u_col,
    bool NA_as_zero_U, int nthreads
);
int precompute_matrices_collective
(
    FPnum *restrict B, int n,
    FPnum *restrict BtBinvBt, /* explicit, no side info, no NAs */
    FPnum *restrict BtBw,     /* explicit, few NAs */
    FPnum *restrict BtBchol,  /* explicit, NA as zero */
    int k, int k_main, int k_user, int k_item,
    FPnum *restrict C, int p,
    FPnum *restrict CtCinvCt,   /* cold-start, no NAs, no binaries */
    FPnum *restrict CtC,        /* cold-start, few NAs, no bin. */
    FPnum *restrict CtCchol,    /* cold-start, NA as zero, no bin. */
    FPnum *restrict BeTBe,      /* implicit, warm-start, few NAs or NA as zero*/
    FPnum *restrict BtB_padded, /* implicit, warm-start, many NAs */
    FPnum *restrict BtB_shrunk, /* implicit, no side info */
    FPnum lam, FPnum w_main, FPnum w_user, FPnum lam_last,
    FPnum w_main_multiplier,
    bool has_U, bool has_U_bin, bool implicit
);
lbfgsFPnumval_t wrapper_collective_fun_grad
(
    void *instance,
    lbfgsFPnumval_t *x,
    lbfgsFPnumval_t *g,
    const size_t n,
    const lbfgsFPnumval_t step
);
typedef struct data_collective_fun_grad {
    int m; int n; int k;
    int *ixA; int *ixB; FPnum *X; size_t nnz;
    FPnum *Xfull;
    long *Xcsr_p; int *Xcsr_i; FPnum *Xcsr;
    long *Xcsc_p; int *Xcsc_i; FPnum *Xcsc;
    FPnum *weight; FPnum *weightR; FPnum *weightC;
    bool user_bias; bool item_bias;
    FPnum lam; FPnum *lam_unique;
    FPnum *U; int m_u; int p; bool U_has_NA;
    FPnum *II; int n_i; int q; bool I_has_NA;
    FPnum *Ub; int m_ubin; int pbin; bool Ub_has_NA;
    FPnum *Ib; int n_ibin; int qbin; bool Ib_has_NA;
    int *U_row; int *U_col; FPnum *U_sp; size_t nnz_U;
    int *I_row; int *I_col; FPnum *I_sp; size_t nnz_I;
    long *U_csr_p; int *U_csr_i; FPnum *U_csr;
    long *U_csc_p; int *U_csc_i; FPnum *U_csc;
    long *I_csr_p; int *I_csr_i; FPnum *I_csr;
    long *I_csc_p; int *I_csc_i; FPnum *I_csc;
    FPnum *buffer_FPnum; FPnum *buffer_mt;
    int k_main; int k_user; int k_item;
    FPnum w_main; FPnum w_user; FPnum w_item;
    int nthreads;
    int print_every; int nfev; int niter;
} data_collective_fun_grad;
int fit_collective_explicit_lbfgs
(
    FPnum *restrict values, bool reset_values,
    FPnum *restrict glob_mean,
    FPnum *restrict U_colmeans, FPnum *restrict I_colmeans,
    int m, int n, int k,
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    FPnum *restrict Xfull,
    FPnum *restrict weight,
    bool user_bias, bool item_bias,
    FPnum lam, FPnum *restrict lam_unique,
    FPnum *restrict U, int m_u, int p,
    FPnum *restrict II, int n_i, int q,
    FPnum *restrict Ub, int m_ubin, int pbin,
    FPnum *restrict Ib, int n_ibin, int qbin,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    int I_row[], int I_col[], FPnum *restrict I_sp, size_t nnz_I,
    int k_main, int k_user, int k_item,
    FPnum w_main, FPnum w_user, FPnum w_item,
    int n_corr_pairs, size_t maxiter, int seed,
    int nthreads, bool prefer_onepass,
    bool verbose, int print_every,
    int *restrict niter, int *restrict nfev,
    FPnum *restrict B_plus_bias
);
int fit_collective_explicit_als
(
    FPnum *restrict values, bool reset_values,
    FPnum *restrict glob_mean,
    FPnum *restrict U_colmeans, FPnum *restrict I_colmeans,
    int m, int n, int k,
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    FPnum *restrict Xfull,
    FPnum *restrict weight,
    bool user_bias, bool item_bias,
    FPnum lam, FPnum *restrict lam_unique,
    FPnum *restrict U, int m_u, int p,
    FPnum *restrict II, int n_i, int q,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    int I_row[], int I_col[], FPnum *restrict I_sp, size_t nnz_I,
    bool NA_as_zero_X, bool NA_as_zero_U, bool NA_as_zero_I,
    int k_main, int k_user, int k_item,
    FPnum w_main, FPnum w_user, FPnum w_item,
    int niter, int nthreads, int seed, bool verbose, bool use_cg,
    FPnum *restrict B_plus_bias
);
int fit_collective_implicit_als
(
    FPnum *restrict values, bool reset_values,
    FPnum *restrict U_colmeans, FPnum *restrict I_colmeans,
    int m, int n, int k,
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    FPnum lam, FPnum *restrict lam_unique,
    FPnum *restrict U, int m_u, int p,
    FPnum *restrict II, int n_i, int q,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    int I_row[], int I_col[], FPnum *restrict I_sp, size_t nnz_I,
    bool NA_as_zero_U, bool NA_as_zero_I,
    int k_main, int k_user, int k_item,
    FPnum w_main, FPnum w_user, FPnum w_item,
    FPnum *restrict w_main_multiplier,
    FPnum alpha, bool adjust_weight,
    int niter, int nthreads, int seed, bool verbose, bool use_cg
);
int collective_factors_cold_multiple
(
    FPnum *restrict A, int m,
    FPnum *restrict U, int m_u, int p,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    long U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict Ub, int m_ubin, int pbin,
    FPnum *restrict C, FPnum *restrict Cb,
    FPnum *restrict CtCinvCt,
    FPnum *restrict CtCw,
    FPnum *restrict CtCchol,
    FPnum *restrict col_means,
    int k, int k_user, int k_main,
    FPnum lam, FPnum w_user,
    bool NA_as_zero_U,
    int nthreads
);
int collective_factors_warm_multiple
(
    FPnum *restrict A, FPnum *restrict biasA, int m, int m_x,
    FPnum *restrict U, int m_u, int p,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    long U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict Ub, int m_ubin, int pbin,
    FPnum *restrict C, FPnum *restrict Cb,
    FPnum glob_mean, FPnum *restrict biasB,
    FPnum *restrict col_means,
    FPnum *restrict X, int ixA[], int ixB[], size_t nnz,
    long *restrict Xcsr_p, int *restrict Xcsr_i, FPnum *restrict Xcsr,
    FPnum *restrict Xfull, int n,
    FPnum *restrict weight,
    FPnum *restrict B,
    int k, int k_user, int k_item, int k_main,
    FPnum lam, FPnum w_user, FPnum w_main, FPnum lam_bias,
    FPnum *restrict BtBinvBt,
    FPnum *restrict BtBw,
    FPnum *restrict BtBchol,
    FPnum *restrict CtCinvCt,
    FPnum *restrict CtCw,
    FPnum *restrict CtCchol,
    int k_item_BtB,
    bool NA_as_zero_U, bool NA_as_zero_X,
    FPnum *restrict B_plus_bias,
    int nthreads
);
int collective_factors_warm_implicit_multiple
(
    FPnum *restrict A, int m, int m_x,
    FPnum *restrict U, int m_u, int p,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    long U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    bool NA_as_zero_U,
    FPnum *restrict col_means,
    FPnum *restrict B, int n, FPnum *restrict C,
    FPnum *restrict X, int ixA[], int ixB[], size_t nnz,
    long *restrict Xcsr_p, int *restrict Xcsr_i, FPnum *restrict Xcsr,
    int k, int k_user, int k_item, int k_main,
    FPnum lam, FPnum alpha, FPnum w_user, FPnum w_main,
    FPnum w_main_multiplier,
    FPnum *restrict precomputedBeTBe,
    FPnum *restrict precomputedBtB,
    FPnum *restrict precomputedBtB_shrunk,
    FPnum *restrict CtCinvCt,
    FPnum *restrict CtCw,
    FPnum *restrict CtCchol,
    int k_item_BtB,
    int nthreads
);

/* offsets.c */
FPnum offsets_fun_grad
(
    FPnum *restrict values, FPnum *restrict grad,
    int ixA[], int ixB[], FPnum *restrict X,
    size_t nnz, int m, int n, int k,
    FPnum *restrict Xfull, bool full_dense,
    long Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    long Xcsc_p[], int Xcsc_i[], FPnum *restrict Xcsc,
    FPnum *restrict weight, FPnum *restrict weightR, FPnum *restrict weightC,
    bool user_bias, bool item_bias,
    bool add_intercepts,
    FPnum lam, FPnum *restrict lam_unique,
    FPnum *restrict U, int p,
    FPnum *restrict II, int q,
    long U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    long U_csc_p[], int U_csc_i[], FPnum *restrict U_csc,
    long I_csr_p[], int I_csr_i[], FPnum *restrict I_csr,
    long I_csc_p[], int I_csc_i[], FPnum *restrict I_csc,
    int k_main, int k_sec,
    FPnum w_user, FPnum w_item,
    int nthreads,
    FPnum *restrict buffer_FPnum,
    FPnum *restrict buffer_mt
);
void construct_Am
(
    FPnum *restrict Am, FPnum *restrict A,
    FPnum *restrict C, FPnum *restrict C_bias,
    bool add_intercepts,
    FPnum *restrict U, int m, int p,
    long U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    int k, int k_sec, int k_main,
    FPnum w_user, int nthreads
);
void assign_gradients
(
    FPnum *restrict bufferA, FPnum *restrict g_A, FPnum *restrict g_C,
    bool add_intercepts, FPnum *restrict g_C_bias,
    FPnum *restrict U,
    long U_csc_p[], int U_csc_i[], FPnum *restrict U_csc,
    int m, int p, int k, int k_sec, int k_main,
    FPnum w_user, int nthreads
);
int offsets_factors_cold
(
    FPnum *restrict a_vec,
    FPnum *restrict u_vec,
    int u_vec_ixB[], FPnum *restrict u_vec_sp, size_t nnz_u_vec,
    FPnum *restrict C, int p,
    FPnum *restrict C_bias,
    int k, int k_sec, int k_main,
    FPnum w_user
);
int offsets_factors_warm
(
    FPnum *restrict a_vec, FPnum *restrict a_bias,
    FPnum *restrict u_vec,
    int u_vec_ixB[], FPnum *restrict u_vec_sp, size_t nnz_u_vec,
    int ixB[], FPnum *restrict Xa, size_t nnz,
    FPnum *restrict Xa_dense, int n,
    FPnum *restrict weight,
    FPnum *restrict Bm, FPnum *restrict C,
    FPnum *restrict C_bias,
    FPnum glob_mean, FPnum *restrict biasB,
    int k, int k_sec, int k_main,
    int p, FPnum w_user,
    FPnum lam, bool exact, FPnum lam_bias,
    bool implicit, FPnum alpha,
    FPnum w_main_multiplier,
    FPnum *restrict precomputedBtBinvBt,
    FPnum *restrict precomputedBtBw,
    FPnum *restrict output_a,
    FPnum *restrict Bm_plus_bias
);
int precompute_matrices_offsets
(
    FPnum *restrict A, int m,
    FPnum *restrict B, int n,
    FPnum *restrict C, int p,
    FPnum *restrict D, int q,
    FPnum *restrict C_bias, FPnum *restrict D_bias,
    bool add_intercepts,
    FPnum *restrict U,
    long U_csr_p[], int U_csc_i[], FPnum *restrict U_csr,
    FPnum *restrict II,
    long I_csr_p[], int I_csc_i[], FPnum *restrict I_csr,
    FPnum *restrict Am,       /* used for existing cases */
    FPnum *restrict Bm,       /* warm-start, with item info */
    FPnum *restrict BtBinvBt, /* explicit, no side info, no NAs */
    FPnum *restrict BtBw,     /* implicit + explicit, few NAs */
    FPnum *restrict BtBchol,  /* explicit, NA as zero */
    int k, int k_main, int k_sec,
    FPnum lam, FPnum w_user, FPnum w_item, FPnum lam_last,
    bool implicit,
    int nthreads
);
typedef struct data_offsets_fun_grad {
    int *ixA; int *ixB; FPnum *X;
    size_t nnz; int m; int n; int k;
    FPnum *Xfull; bool full_dense;
    long *Xcsr_p; int *Xcsr_i; FPnum *Xcsr;
    long *Xcsc_p; int *Xcsc_i; FPnum *Xcsc;
    FPnum *weight; FPnum *weightR; FPnum *weightC;
    bool user_bias; bool item_bias;
    bool add_intercepts;
    FPnum lam; FPnum *lam_unique;
    FPnum *U; int p;
    FPnum *II; int q;
    long *U_csr_p; int *U_csr_i; FPnum *U_csr;
    long *U_csc_p; int *U_csc_i; FPnum *U_csc;
    long *I_csr_p; int *I_csr_i; FPnum *I_csr;
    long *I_csc_p; int *I_csc_i; FPnum *I_csc;
    int k_main; int k_sec;
    FPnum w_user; FPnum w_item;
    int nthreads;
    FPnum *buffer_FPnum;
    FPnum *buffer_mt;
    int print_every; int nfev; int niter;
} data_offsets_fun_grad;
lbfgsFPnumval_t wrapper_offsets_fun_grad
(
    void *instance,
    lbfgsFPnumval_t *x,
    lbfgsFPnumval_t *g,
    const size_t n,
    const lbfgsFPnumval_t step
);
int fit_offsets_explicit_lbfgs
(
    FPnum *restrict values, bool reset_values,
    FPnum *restrict glob_mean,
    int m, int n, int k,
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    FPnum *restrict Xfull,
    FPnum *restrict weight,
    bool user_bias, bool item_bias,
    bool add_intercepts,
    FPnum lam, FPnum *restrict lam_unique,
    FPnum *restrict U, int p,
    FPnum *restrict II, int q,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    int I_row[], int I_col[], FPnum *restrict I_sp, size_t nnz_I,
    int k_main, int k_sec,
    FPnum w_user, FPnum w_item,
    int n_corr_pairs, size_t maxiter, int seed,
    int nthreads, bool prefer_onepass,
    bool verbose, int print_every,
    int *restrict niter, int *restrict nfev,
    FPnum *restrict Am, FPnum *restrict Bm,
    FPnum *restrict Bm_plus_bias
);
int fit_offsets_als
(
    FPnum *restrict values, bool reset_values,
    FPnum *restrict glob_mean,
    FPnum *restrict Am, FPnum *restrict Bm,
    int m, int n, int k,
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    FPnum *restrict Xfull,
    FPnum *restrict weight,
    bool user_bias, bool item_bias, bool add_intercepts,
    FPnum lam,
    FPnum *restrict U, int p,
    FPnum *restrict II, int q,
    bool implicit, bool NA_as_zero_X, FPnum alpha,
    bool adjust_weight, FPnum *restrict w_main_multiplier,
    int niter, int seed,
    int nthreads, bool use_cg,
    bool verbose,
    FPnum *restrict Bm_plus_bias
);
void factors_content_based
(
    FPnum *restrict a_vec, int k_sec,
    FPnum *restrict u_vec, int p,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    FPnum *restrict C, FPnum *restrict C_bias
);
int matrix_content_based
(
    FPnum *restrict Am_new,
    int n_new, int k_sec,
    FPnum *restrict U, int p,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    long U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict C, FPnum *restrict C_bias,
    int nthreads
);
int predict_content_based_new
(
    FPnum *restrict scores_new, int n_new, int k_sec,
    FPnum *restrict U, int p,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    long U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict II, int q,
    int I_row[], int I_col[], FPnum *restrict I_sp, size_t nnz_I,
    long I_csr_p[], int I_csr_i[], FPnum *restrict I_csr,
    FPnum *restrict C, FPnum *restrict C_bias,
    FPnum *restrict D, FPnum *restrict D_bias,
    FPnum glob_mean,
    int nthreads
);
int predict_content_based_old
(
    FPnum *restrict scores_new, int n_new, int k_sec,
    FPnum *restrict U, int p,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    long U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict C, FPnum *restrict C_bias,
    FPnum *restrict Bm, FPnum *restrict biasB, int ixB[],
    FPnum glob_mean,
    int nthreads
);
int rank_content_based_new
(
    FPnum *restrict scores_new, int *restrict rank_new,
    int n_new, int k_sec, int n_top,
    FPnum *restrict u_vec, int p,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    FPnum *restrict II, int q,
    int I_row[], int I_col[], FPnum *restrict I_sp, size_t nnz_I,
    long I_csr_p[], int I_csr_i[], FPnum *restrict I_csr,
    FPnum *restrict C, FPnum *restrict C_bias,
    FPnum *restrict D, FPnum *restrict D_bias,
    FPnum glob_mean,
    int nthreads
);
int offsets_factors_cold_multiple
(
    FPnum *restrict A, int m,
    FPnum *restrict U, int p,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    long U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict C, FPnum *restrict C_bias,
    int k, int k_sec, int k_main,
    FPnum w_user,
    int nthreads
);
int offsets_factors_warm_multiple
(
    FPnum *restrict A, FPnum *restrict biasA, int m,
    FPnum *restrict U, int p,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    long U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict X, int ixA[], int ixB[], size_t nnz,
    long *restrict Xcsr_p, int *restrict Xcsr_i, FPnum *restrict Xcsr,
    FPnum *restrict Xfull, int n,
    FPnum *restrict weight,
    FPnum *restrict Bm, FPnum *restrict C,
    FPnum *restrict C_bias,
    FPnum glob_mean, FPnum *restrict biasB,
    int k, int k_sec, int k_main,
    FPnum w_user,
    FPnum lam, bool exact, FPnum lam_bias,
    bool implicit, FPnum alpha,
    FPnum w_main_multiplier,
    FPnum *restrict precomputedBtBinvBt,
    FPnum *restrict precomputedBtBw,
    FPnum *restrict Bm_plus_bias,
    FPnum *restrict output_A,
    int nthreads
);

#ifdef __cplusplus
}
#endif
