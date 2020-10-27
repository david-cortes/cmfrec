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
    #define omp_get_thread_num() (0)
#endif

#ifdef _FOR_PYTHON
    /* This contains the standard cblas.h header */
    #include "findblas.h" /* https://www.github.com/david-cortes/findblas */
#elif defined(_FOR_R)
    #include <R.h>
    #include <Rinternals.h>
    #include <R_ext/Print.h>
    #include <R_ext/BLAS.h>
    #include <R_ext/Lapack.h>
    #define USE_DOUBLE
    #define FORCE_NO_NAN_PROPAGATION
    #define printf Rprintf
    #define fprintf(f, message) REprintf(message)
#elif defined(MKL_ILP64)
    #include "mkl.h"
#endif
/* Here one may also include the standard headers "cblas.h" and "lapack.h",
   if one wants to use a non-standard version such as ILP64 (-DMKL_ILP64). */

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
#ifdef _FOR_R
    #define NAN_ NA_REAL
#else
    #define NAN_ NAN
#endif

#if !defined(USE_FLOAT)
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
    #define tgelsd_ dgelsd_
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
    #define tgelsd_ sgelsd_
#endif

#if !defined(LAPACK_H) && !defined(_FOR_R)
void tposv_(const char*, const int*, const int*, const FPnum*, const int*, const FPnum*, const int*, const int*);
void tlacpy_(const char*, const int*, const int*, const FPnum*, const int*, const FPnum*, const int*);
void tlarnv_(const int*, const int*, const int*, const FPnum*);
void tpotrf_(const char*, const int*, const FPnum*, const int*, const int*);
void tpotrs_(const char*, const int*, const int*, const FPnum*, const int*, const FPnum*, const int*, const int*);
void tgelsd_(const int*, const int*, const int*,
             const FPnum*, const int*,
             const FPnum*, const int*,
             const FPnum*, const FPnum*, const int*, const FPnum*,
             const int*, const int*, const int*);
#endif

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
void taxpy_large(FPnum *restrict A, FPnum x, FPnum *restrict Y, size_t n, int nthreads);
void tscal_large(FPnum *restrict arr, FPnum alpha, size_t n, int nthreads);
int rnorm(FPnum *restrict arr, size_t n, int seed, int nthreads);
void rnorm_preserve_seed(FPnum *restrict arr, size_t n, int seed_arr[4]);
void process_seed_for_larnv(int seed_arr[4]);
void reduce_mat_sum(FPnum *restrict outp, size_t lda, FPnum *restrict inp,
                    int m, int n, int nthreads);
void exp_neg_x(FPnum *restrict arr, size_t n, int nthreads);
void add_to_diag(FPnum *restrict A, FPnum val, size_t n);
FPnum sum_sq_div_w(FPnum *restrict arr, FPnum *restrict w, size_t n, bool compensated, int nthreads);
void tgemm_sp_dense
(
    int m, int n, FPnum alpha,
    size_t indptr[], int indices[], FPnum values[],
    FPnum DenseMat[], size_t ldb,
    FPnum OutputMat[], size_t ldc,
    int nthreads
);
void tgemv_dense_sp
(
    int m, int n,
    FPnum alpha, FPnum DenseMat[], size_t lda,
    int ixB[], FPnum vec_sp[], size_t nnz,
    FPnum OutputVec[]
);
void tgemv_dense_sp_weighted
(
    int m, int n,
    FPnum alpha[], FPnum DenseMat[], size_t lda,
    int ixB[], FPnum vec_sp[], size_t nnz,
    FPnum OutputVec[]
);
void tgemv_dense_sp_weighted2
(
    int m, int n,
    FPnum alpha[], FPnum alpha2, FPnum DenseMat[], size_t lda,
    int ixB[], FPnum vec_sp[], size_t nnz,
    FPnum OutputVec[]
);
void tgemv_dense_sp_notrans
(
    int m, int n,
    FPnum DenseMat[], int lda,
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
    size_t m, size_t n,
    FPnum *restrict A, size_t lda,
    FPnum *restrict B, size_t ldb
);
void transpose_mat(FPnum *restrict A, size_t m, size_t n, FPnum *restrict buffer_FPnum);
void transpose_mat2(FPnum *restrict A, size_t m, size_t n, FPnum *restrict outp);
void transpose_mat3
(
    FPnum *restrict A, size_t lda,
    size_t m, size_t n,
    FPnum *restrict outp, size_t ldb
);
int coo_to_csr_plus_alloc
(
    int *restrict Xrow, int *restrict Xcol, FPnum *restrict Xval,
    FPnum *restrict W,
    int m, int n, size_t nnz,
    size_t *restrict *csr_p, int *restrict *csr_i, FPnum *restrict *csr_v,
    FPnum *restrict *csr_w
);
void coo_to_csr
(
    int *restrict Xrow, int *restrict Xcol, FPnum *restrict Xval,
    FPnum *restrict W,
    int m, int n, size_t nnz,
    size_t *restrict csr_p, int *restrict csr_i, FPnum *restrict csr_v,
    FPnum *restrict csr_w
);
void coo_to_csr_and_csc
(
    int *restrict Xrow, int *restrict Xcol, FPnum *restrict Xval,
    FPnum *restrict W, int m, int n, size_t nnz,
    size_t *restrict csr_p, int *restrict csr_i, FPnum *restrict csr_v,
    size_t *restrict csc_p, int *restrict csc_i, FPnum *restrict csc_v,
    FPnum *restrict csr_w, FPnum *restrict csc_w,
    int nthreads
);
void row_means_csr(size_t indptr[], FPnum *restrict values,
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
void append_ones_last_col
(
    FPnum *restrict orig, size_t m, size_t n,
    FPnum *restrict outp
);
void fill_lower_triangle(FPnum A[], size_t n, size_t lda);
void print_oom_message(void);
#ifdef _FOR_R
void R_nan_to_C_nan(FPnum arr[], size_t n);
#endif


/* common.c */
FPnum fun_grad_cannonical_form
(
    FPnum *restrict A, int lda, FPnum *restrict B, int ldb,
    FPnum *restrict g_A, FPnum *restrict g_B,
    int m, int n, int k,
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    FPnum *restrict Xfull, bool full_dense,
    size_t Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    size_t Xcsc_p[], int Xcsc_i[], FPnum *restrict Xcsc,
    bool user_bias, bool item_bias,
    FPnum *restrict biasA, FPnum *restrict biasB,
    FPnum *restrict g_biasA, FPnum *restrict g_biasB,
    FPnum *restrict weight, FPnum *restrict weightR, FPnum *restrict weightC,
    FPnum scaling,
    FPnum *restrict buffer_FPnum,
    FPnum *restrict buffer_mt,
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
    FPnum lam, FPnum lam_last,
    FPnum *restrict precomputedTransBtBinvBt,
    FPnum *restrict precomputedBtB, int cnt_NA, int ld_BtB,
    bool BtB_has_diag, bool BtB_is_scaled, FPnum scale_BtB, int n_BtB,
    FPnum *restrict precomputedBtBchol, bool NA_as_zero,
    bool use_cg, int max_cg_steps, /* <- 'cg' should not be used for new data */
    bool force_add_diag
);
void factors_explicit_cg
(
    FPnum *restrict a_vec, int k,
    FPnum *restrict B, int n, int ldb,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    FPnum *restrict weight,
    FPnum *restrict buffer_FPnum,
    FPnum lam, FPnum lam_last,
    int max_cg_steps
);
void factors_explicit_cg_NA_as_zero_weighted
(
    FPnum *restrict a_vec, int k,
    FPnum *restrict B, int n, int ldb,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    FPnum *restrict weight,
    FPnum *restrict precomputedBtB, int ld_BtB,
    FPnum *restrict buffer_FPnum,
    FPnum lam, FPnum lam_last,
    int max_cg_steps
);
void factors_explicit_cg_dense
(
    FPnum *restrict a_vec, int k,
    FPnum *restrict B, int n, int ldb,
    FPnum *restrict Xa_dense, int cnt_NA,
    FPnum *restrict weight,
    FPnum *restrict precomputedBtB, int ld_BtB,
    FPnum *restrict buffer_FPnum,
    FPnum lam, FPnum lam_last,
    int max_cg_steps
);
void factors_implicit_cg
(
    FPnum *restrict a_vec, int k,
    FPnum *restrict B, size_t ldb,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    FPnum lam,
    FPnum *restrict precomputedBtB, int ld_BtB,
    int max_cg_steps,
    FPnum *restrict buffer_FPnum
);
void factors_implicit_chol
(
    FPnum *restrict a_vec, int k,
    FPnum *restrict B, size_t ldb,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    FPnum lam,
    FPnum *restrict precomputedBtB, int ld_BtB,
    bool zero_out,
    FPnum *restrict buffer_FPnum,
    bool force_add_diag
);
void factors_implicit
(
    FPnum *restrict a_vec, int k,
    FPnum *restrict B, size_t ldb,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    FPnum lam,
    FPnum *restrict precomputedBtB, int ld_BtB,
    bool zero_out, bool use_cg, int max_cg_steps,
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
size_t buffer_size_optimizeA
(
    size_t n, bool full_dense, bool near_dense, bool do_B,
    bool has_dense, bool has_weights, bool NA_as_zero,
    size_t k, size_t nthreads,
    bool pass_allocated_BtB, bool keep_precomputedBtB,
    bool use_cg, bool finalize_chol
);
size_t buffer_size_optimizeA_implicit
(
    size_t k, size_t nthreads,
    bool pass_allocated_BtB,
    bool use_cg, bool finalize_chol
);
void optimizeA
(
    FPnum *restrict A, int lda,
    FPnum *restrict B, int ldb,
    int m, int n, int k,
    size_t Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    FPnum *restrict Xfull, int ldX, bool full_dense, bool near_dense,
    int cnt_NA[], FPnum *restrict weight, bool NA_as_zero,
    FPnum lam, FPnum lam_last,
    bool do_B, bool is_first_iter,
    int nthreads,
    bool use_cg, int max_cg_steps,
    bool keep_precomputedBtB,
    FPnum *restrict precomputedBtB, bool *filled_BtB,
    FPnum *restrict buffer_FPnum
);
void optimizeA_implicit
(
    FPnum *restrict A, size_t lda,
    FPnum *restrict B, size_t ldb,
    int m, int n, int k,
    size_t Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    FPnum lam,
    int nthreads,
    bool use_cg, int max_cg_steps, bool force_set_to_zero,
    bool keep_precomputedBtB,
    FPnum *restrict precomputedBtB,
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
    size_t Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    size_t Xcsc_p[], int Xcsc_i[], FPnum *restrict Xcsc,
    int nthreads
);
int center_by_cols
(
    FPnum *restrict col_means,
    FPnum *restrict Xfull, int m, int n,
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    size_t Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    size_t Xcsc_p[], int Xcsc_i[], FPnum *restrict Xcsc,
    int nthreads
);
bool check_sparse_indices
(
    int n, int p,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    FPnum *restrict Xa, int ixB[], size_t nnz
);
void predict_multiple
(
    FPnum *restrict A, int k_user,
    FPnum *restrict B, int k_item,
    FPnum *restrict biasA, FPnum *restrict biasB,
    FPnum glob_mean,
    int k, int k_main,
    int m, int n,
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
int topN_old_most_popular
(
    bool user_bias,
    FPnum a_bias,
    FPnum *restrict biasA, int row_index,
    FPnum *restrict B,
    FPnum *restrict biasB,
    FPnum glob_mean,
    int *restrict include_ix, int n_include,
    int *restrict exclude_ix, int n_exclude,
    int *restrict outp_ix, FPnum *restrict outp_score,
    int n_top, int n
);
int predict_X_old_most_popular
(
    int row[], int col[], FPnum *restrict predicted, size_t n_predict,
    FPnum *restrict biasA, FPnum *restrict biasB,
    FPnum glob_mean,
    int m, int n
);

/* collective.c */
void nvars_collective_fun_grad
(
    size_t m, size_t n, size_t m_u, size_t n_i, size_t m_ubin, size_t n_ibin,
    size_t p, size_t q, size_t pbin, size_t qbin,
    size_t nnz, size_t nnz_U, size_t nnz_I,
    size_t k, size_t k_main, size_t k_user, size_t k_item,
    bool user_bias, bool item_bias, size_t nthreads,
    FPnum *X, FPnum *Xfull,
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
    size_t Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    size_t Xcsc_p[], int Xcsc_i[], FPnum *restrict Xcsc,
    FPnum *restrict weight, FPnum *restrict weightR, FPnum *restrict weightC,
    bool user_bias, bool item_bias,
    FPnum lam, FPnum *restrict lam_unique,
    FPnum *restrict U, int m_u, int p, bool U_has_NA,
    FPnum *restrict II, int n_i, int q, bool I_has_NA,
    FPnum *restrict Ub, int m_ubin, int pbin, bool Ub_has_NA,
    FPnum *restrict Ib, int n_ibin, int qbin, bool Ib_has_NA,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    int I_row[], int I_col[], FPnum *restrict I_sp, size_t nnz_I,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    size_t U_csc_p[], int U_csc_i[], FPnum *restrict U_csc,
    size_t I_csr_p[], int I_csr_i[], FPnum *restrict I_csr,
    size_t I_csc_p[], int I_csc_i[], FPnum *restrict I_csc,
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
    int k, int k_user, int k_item, int k_main,
    FPnum *restrict Xa_dense,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    int u_vec_ixB[], FPnum *restrict u_vec_sp, size_t nnz_u_vec,
    FPnum *restrict u_vec,
    bool NA_as_zero_X, bool NA_as_zero_U,
    FPnum *restrict B, int n, int ldb,
    FPnum *restrict C, int p,
    FPnum *restrict weight,
    FPnum lam, FPnum w_user, FPnum lam_last,
    FPnum *restrict precomputedBtB, int cnt_NA_x,
    FPnum *restrict precomputedCtCw, int cnt_NA_u,
    FPnum *restrict precomputedBeTBeChol, int n_BtB,
    bool add_X, bool add_U,
    bool use_cg, int max_cg_steps, /* <- 'cg' should not be used for new data */
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
    FPnum lam, FPnum w_user,
    FPnum *restrict precomputedBeTBe,
    FPnum *restrict precomputedBtB, /* for cg, should NOT have lambda added */
    FPnum *restrict precomputedBeTBeChol,
    FPnum *restrict precomputedCtCw,
    bool add_U, bool shapes_match,
    bool use_cg, int max_cg_steps,
    FPnum *restrict buffer_FPnum
);
void collective_block_cg
(
    FPnum *restrict a_vec,
    int k, int k_user, int k_item, int k_main,
    FPnum *restrict Xa_dense,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    int u_vec_ixB[], FPnum *restrict u_vec_sp, size_t nnz_u_vec,
    FPnum *restrict u_vec,
    bool NA_as_zero_X, bool NA_as_zero_U,
    FPnum *restrict B, int n, int ldb,
    FPnum *restrict C, int p,
    FPnum *restrict weight,
    FPnum lam, FPnum w_user, FPnum lam_last,
    int cnt_NA_x, int cnt_NA_u,
    FPnum *restrict precomputedBtB,
    FPnum *restrict precomputedCtC, /* should NOT be multiplied by 'w_user' */
    int max_cg_steps,
    FPnum *restrict buffer_FPnum
);
void collective_block_cg_implicit
(
    FPnum *restrict a_vec,
    int k, int k_user, int k_item, int k_main,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    int u_vec_ixB[], FPnum *restrict u_vec_sp, size_t nnz_u_vec,
    FPnum *restrict u_vec,
    bool NA_as_zero_U,
    FPnum *restrict B, int n,
    FPnum *restrict C, int p,
    FPnum lam, FPnum w_user,
    int cnt_NA_u,
    int max_cg_steps,
    FPnum *restrict precomputedBtB,
    FPnum *restrict precomputedCtC,
    FPnum *restrict buffer_FPnum
);
void optimizeA_collective_implicit
(
    FPnum *restrict A, FPnum *restrict B, FPnum *restrict C,
    int m, int m_u, int n, int p,
    int k, int k_main, int k_user, int k_item,
    size_t Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict U, int cnt_NA_u[],
    bool full_dense_u, bool near_dense_u, bool NA_as_zero_U,
    FPnum lam, FPnum w_user,
    int nthreads,
    bool use_cg, int max_cg_steps, bool is_first_iter,
    bool keep_precomputedBtB,
    FPnum *restrict precomputedBtB, /* will not have lambda with CG */
    FPnum *restrict precomputedBeTBe,
    FPnum *restrict precomputedBeTBeChol,
    FPnum *restrict precomputedCtC,
    bool *filled_BeTBe,
    bool *filled_BeTBeChol,
    bool *filled_CtC,
    FPnum *restrict buffer_FPnum
);
int collective_factors_cold
(
    FPnum *restrict a_vec,
    FPnum *restrict u_vec, int p,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    FPnum *restrict u_bin_vec, int pbin,
    FPnum *restrict C, FPnum *restrict Cb,
    FPnum *restrict TransCtCinvCt,
    FPnum *restrict CtCw,
    FPnum *restrict col_means,
    int k, int k_user, int k_main,
    FPnum lam, FPnum w_main, FPnum w_user,
    bool NA_as_zero_U
);
int collective_factors_cold_implicit
(
    FPnum *restrict a_vec,
    FPnum *restrict u_vec, int p,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    FPnum *restrict B, int n,
    FPnum *restrict C,
    FPnum *restrict BeTBe,
    FPnum *restrict BtB,
    FPnum *restrict BeTBeChol,
    FPnum *restrict col_means,
    int k, int k_user, int k_item, int k_main,
    FPnum lam, FPnum w_main, FPnum w_user, FPnum w_main_multiplier,
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
    FPnum lam, FPnum w_main, FPnum w_user, FPnum lam_bias,
    int n_max, bool include_all_X,
    FPnum *restrict TransBtBinvBt,
    FPnum *restrict BtB,
    FPnum *restrict BeTBeChol,
    FPnum *restrict CtCw,
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
    FPnum lam, FPnum alpha, FPnum w_main, FPnum w_user,
    FPnum w_main_multiplier,
    FPnum *restrict BeTBe,
    FPnum *restrict BtB,
    FPnum *restrict BeTBeChol
);
FPnum fun_grad_A_collective
(
    FPnum *restrict A, FPnum *restrict g_A,
    FPnum *restrict B, FPnum *restrict C,
    int m, int m_u, int n, int p,
    int k, int k_main, int k_user, int k_item, int padding,
    FPnum *restrict Xfull, bool full_dense,
    size_t Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    FPnum *restrict weight,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
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
    size_t *Xcsr_p; int *Xcsr_i; FPnum *Xcsr;
    FPnum *weight;
    size_t *U_csr_p; int *U_csr_i; FPnum *U_csr;
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
size_t buffer_size_optimizeA_collective
(
    size_t m, size_t m_u, size_t n, size_t p,
    size_t k, size_t k_main, size_t k_user,
    bool full_dense, bool near_dense, bool do_B,
    bool has_dense, bool has_sparse, bool has_weights, bool NA_as_zero_X,
    bool has_dense_U, bool has_sparse_U,
    bool full_dense_u, bool near_dense_u, bool NA_as_zero_U,
    size_t nthreads,
    bool use_cg, bool finalize_chol,
    bool keep_precomputed,
    bool pass_allocated_BtB,
    bool pass_allocated_BeTBeChol,
    bool pass_allocated_CtCw
);
size_t buffer_size_optimizeA_collective_implicit
(
    size_t m, size_t m_u, size_t p,
    size_t k, size_t k_main, size_t k_user,
    bool has_sparse_U,
    bool NA_as_zero_U,
    size_t nthreads,
    bool use_cg,
    bool pass_allocated_BtB,
    bool pass_allocated_BeTBe,
    bool pass_allocated_BeTBeChol,
    bool pass_allocated_CtC,
    bool finalize_chol
);
void optimizeA_collective
(
    FPnum *restrict A, int lda, FPnum *restrict B, int ldb,
    FPnum *restrict C,
    int m, int m_u, int n, int p,
    int k, int k_main, int k_user, int k_item,
    size_t Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    FPnum *restrict Xfull, bool full_dense, bool near_dense,
    int cnt_NA_x[], FPnum *restrict weight, bool NA_as_zero_X,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict U, int cnt_NA_u[],
    bool full_dense_u, bool near_dense_u, bool NA_as_zero_U,
    FPnum lam, FPnum w_user, FPnum lam_last,
    bool do_B,
    int nthreads,
    bool use_cg, int max_cg_steps, bool is_first_iter,
    bool keep_precomputed,
    FPnum *restrict precomputedBtB,
    FPnum *restrict precomputedCtCw,
    FPnum *restrict precomputedBeTBeChol,
    bool *filled_BtB, bool *filled_CtCw, bool *filled_BeTBeChol,
    FPnum *restrict buffer_FPnum
);
void build_BeTBe
(
    FPnum *restrict bufferBeTBe,
    FPnum *restrict B, int ldb, FPnum *restrict C,
    int k, int k_user, int k_main, int k_item,
    int n, int p,
    FPnum lam, FPnum w_user
);
void build_BtB_CtC
(
    FPnum *restrict BtB, FPnum *restrict CtC,
    FPnum *restrict B, int n, int ldb,
    FPnum *restrict C, int p,
    int k, int k_user, int k_main, int k_item,
    FPnum w_user,
    FPnum *restrict weight
);
void build_XBw
(
    FPnum *restrict A, int lda,
    FPnum *restrict B, int ldb,
    FPnum *restrict Xfull, int ldX,
    int m, int n, int k,
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
    size_t **Xcsr_p, int **Xcsr_i, FPnum *restrict *Xcsr,
    size_t **Xcsc_p, int **Xcsc_i, FPnum *restrict *Xcsc,
    FPnum *restrict weight, FPnum *restrict *weightR, FPnum *restrict *weightC,
    int m, int n, int nthreads
);
int preprocess_sideinfo_matrix
(
    FPnum *U, int m_u, int p,
    int U_row[], int U_col[], FPnum *U_sp, size_t nnz_U,
    FPnum *U_colmeans, FPnum *restrict *Utrans,
    size_t **U_csr_p, int **U_csr_i, FPnum *restrict *U_csr,
    size_t **U_csc_p, int **U_csc_i, FPnum *restrict *U_csc,
    int *restrict *cnt_NA_u_byrow, int *restrict *cnt_NA_u_bycol,
    bool *full_dense_u, bool *near_dense_u_row, bool *near_dense_u_col,
    bool NA_as_zero_U, int nthreads
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
    size_t *Xcsr_p; int *Xcsr_i; FPnum *Xcsr;
    size_t *Xcsc_p; int *Xcsc_i; FPnum *Xcsc;
    FPnum *weight; FPnum *weightR; FPnum *weightC;
    bool user_bias; bool item_bias;
    FPnum lam; FPnum *lam_unique;
    FPnum *U; int m_u; int p; bool U_has_NA;
    FPnum *II; int n_i; int q; bool I_has_NA;
    FPnum *Ub; int m_ubin; int pbin; bool Ub_has_NA;
    FPnum *Ib; int n_ibin; int qbin; bool Ib_has_NA;
    int *U_row; int *U_col; FPnum *U_sp; size_t nnz_U;
    int *I_row; int *I_col; FPnum *I_sp; size_t nnz_I;
    size_t *U_csr_p; int *U_csr_i; FPnum *U_csr;
    size_t *U_csc_p; int *U_csc_i; FPnum *U_csc;
    size_t *I_csr_p; int *I_csr_i; FPnum *I_csr;
    size_t *I_csc_p; int *I_csc_i; FPnum *I_csc;
    FPnum *buffer_FPnum; FPnum *buffer_mt;
    int k_main; int k_user; int k_item;
    FPnum w_main; FPnum w_user; FPnum w_item;
    int nthreads;
    int print_every; int nfev; int niter;
    bool handle_interrupt;
} data_collective_fun_grad;
int fit_collective_explicit_lbfgs_internal
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
    bool verbose, int print_every, bool handle_interrupt,
    int *restrict niter, int *restrict nfev,
    FPnum *restrict B_plus_bias
);
int fit_collective_explicit_lbfgs
(
    FPnum *restrict biasA, FPnum *restrict biasB,
    FPnum *restrict A, FPnum *restrict B,
    FPnum *restrict C, FPnum *restrict Cb,
    FPnum *restrict D, FPnum *restrict Db,
    bool reset_values, int seed,
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
    int n_corr_pairs, size_t maxiter,
    int nthreads, bool prefer_onepass,
    bool verbose, int print_every, bool handle_interrupt,
    int *restrict niter, int *restrict nfev,
    bool precompute_for_predictions,
    FPnum *restrict B_plus_bias,
    FPnum *restrict precomputedBtB,
    FPnum *restrict precomputedTransBtBinvBt,
    FPnum *restrict precomputedBeTBeChol,
    FPnum *restrict precomputedTransCtCinvCt,
    FPnum *restrict precomputedCtCw
);
int fit_collective_explicit_als
(
    FPnum *restrict biasA, FPnum *restrict biasB,
    FPnum *restrict A, FPnum *restrict B,
    FPnum *restrict C, FPnum *restrict D,
    bool reset_values, int seed,
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
    int niter, int nthreads, bool verbose, bool handle_interrupt,
    bool use_cg, int max_cg_steps, bool finalize_chol,
    bool precompute_for_predictions,
    bool include_all_X,
    FPnum *restrict B_plus_bias,
    FPnum *restrict precomputedBtB,
    FPnum *restrict precomputedTransBtBinvBt,
    FPnum *restrict precomputedBeTBeChol,
    FPnum *restrict precomputedTransCtCinvCt,
    FPnum *restrict precomputedCtCw
);
int fit_collective_implicit_als
(
    FPnum *restrict A, FPnum *restrict B,
    FPnum *restrict C, FPnum *restrict D,
    bool reset_values, int seed,
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
    int niter, int nthreads, bool verbose, bool handle_interrupt,
    bool use_cg, int max_cg_steps, bool finalize_chol,
    bool precompute_for_predictions,
    FPnum *restrict precomputedBtB,
    FPnum *restrict precomputedBeTBe,
    FPnum *restrict precomputedBeTBeChol
);
int precompute_collective_explicit
(
    FPnum *restrict B, int n, int n_max, bool include_all_X,
    FPnum *restrict C, int p,
    int k, int k_user, int k_item, int k_main,
    bool user_bias,
    FPnum lam, FPnum *restrict lam_unique,
    FPnum w_main, FPnum w_user,
    FPnum *restrict B_plus_bias,
    FPnum *restrict BtB,
    FPnum *restrict TransBtBinvBt,
    FPnum *restrict BeTBeChol,
    FPnum *restrict TransCtCinvCt,
    FPnum *restrict CtCw
);
int precompute_collective_implicit
(
    FPnum *restrict B, int n,
    FPnum *restrict C, int p,
    int k, int k_user, int k_item, int k_main,
    FPnum lam, FPnum w_main, FPnum w_user, FPnum w_main_multiplier,
    bool extra_precision,
    FPnum *restrict BtB,
    FPnum *restrict BeTBe,
    FPnum *restrict BeTBeChol
);
int factors_collective_explicit_single
(
    FPnum *restrict a_vec, FPnum *restrict a_bias,
    FPnum *restrict u_vec, int p,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    FPnum *restrict u_bin_vec, int pbin,
    bool NA_as_zero_U, bool NA_as_zero_X,
    FPnum *restrict C, FPnum *restrict Cb,
    FPnum glob_mean, FPnum *restrict biasB,
    FPnum *restrict col_means,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    FPnum *restrict Xa_dense, int n,
    FPnum *restrict weight,
    FPnum *restrict B,
    int k, int k_user, int k_item, int k_main,
    FPnum lam, FPnum *restrict lam_unique,
    FPnum w_main, FPnum w_user,
    int n_max, bool include_all_X,
    FPnum *restrict TransBtBinvBt,
    FPnum *restrict BtB,
    FPnum *restrict BeTBeChol,
    FPnum *restrict CtCw,
    FPnum *restrict TransCtCinvCt,
    FPnum *restrict B_plus_bias
);
int factors_collective_implicit_single
(
    FPnum *restrict a_vec,
    FPnum *restrict u_vec, int p,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    bool NA_as_zero_U,
    FPnum *restrict col_means,
    FPnum *restrict B, int n, FPnum *restrict C,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    int k, int k_user, int k_item, int k_main,
    FPnum lam, FPnum alpha, FPnum w_main, FPnum w_user,
    FPnum w_main_multiplier,
    FPnum *restrict BeTBe,
    FPnum *restrict BtB,
    FPnum *restrict BeTBeChol
);
int factors_collective_explicit_multiple
(
    FPnum *restrict A, FPnum *restrict biasA, int m,
    FPnum *restrict U, int m_u, int p,
    bool NA_as_zero_U, bool NA_as_zero_X,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict Ub, int m_ubin, int pbin,
    FPnum *restrict C, FPnum *restrict Cb,
    FPnum glob_mean, FPnum *restrict biasB,
    FPnum *restrict col_means,
    FPnum *restrict X, int ixA[], int ixB[], size_t nnz,
    size_t *restrict Xcsr_p, int *restrict Xcsr_i, FPnum *restrict Xcsr,
    FPnum *restrict Xfull, int n,
    FPnum *restrict weight,
    FPnum *restrict B,
    int k, int k_user, int k_item, int k_main,
    FPnum lam, FPnum *restrict lam_unique,
    FPnum w_main, FPnum w_user,
    int n_max, bool include_all_X,
    FPnum *restrict TransBtBinvBt,
    FPnum *restrict BtB,
    FPnum *restrict BeTBeChol,
    FPnum *restrict TransCtCinvCt,
    FPnum *restrict CtCw,
    FPnum *restrict B_plus_bias,
    int nthreads
);
int factors_collective_implicit_multiple
(
    FPnum *restrict A, int m,
    FPnum *restrict U, int m_u, int p,
    bool NA_as_zero_U,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict X, int ixA[], int ixB[], size_t nnz,
    size_t *restrict Xcsr_p, int *restrict Xcsr_i, FPnum *restrict Xcsr,
    FPnum *restrict B, int n,
    FPnum *restrict C,
    FPnum *restrict col_means,
    int k, int k_user, int k_item, int k_main,
    FPnum lam, FPnum alpha, FPnum w_main, FPnum w_user,
    FPnum w_main_multiplier,
    FPnum *restrict BeTBe,
    FPnum *restrict BtB,
    FPnum *restrict BeTBeChol,
    int nthreads
);
int impute_X_collective_explicit
(
    int m, bool user_bias,
    FPnum *restrict U, int m_u, int p,
    bool NA_as_zero_U,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict Ub, int m_ubin, int pbin,
    FPnum *restrict C, FPnum *restrict Cb,
    FPnum glob_mean, FPnum *restrict biasB,
    FPnum *restrict col_means,
    FPnum *restrict Xfull, int n,
    FPnum *restrict weight,
    FPnum *restrict B,
    int k, int k_user, int k_item, int k_main,
    FPnum lam, FPnum *restrict lam_unique,
    FPnum w_main, FPnum w_user,
    int n_max, bool include_all_X,
    FPnum *restrict TransBtBinvBt,
    FPnum *restrict BtB,
    FPnum *restrict BeTBeChol,
    FPnum *restrict TransCtCinvCt,
    FPnum *restrict CtCw,
    FPnum *restrict B_plus_bias,
    int nthreads
);
int topN_old_collective_explicit
(
    FPnum *restrict a_vec, FPnum a_bias,
    FPnum *restrict A, FPnum *restrict biasA, int row_index,
    FPnum *restrict B,
    FPnum *restrict biasB,
    FPnum glob_mean,
    int k, int k_user, int k_item, int k_main,
    int *restrict include_ix, int n_include,
    int *restrict exclude_ix, int n_exclude,
    int *restrict outp_ix, FPnum *restrict outp_score,
    int n_top, int n, int n_max, bool include_all_X, int nthreads
);
int topN_old_collective_implicit
(
    FPnum *restrict a_vec,
    FPnum *restrict A, int row_index,
    FPnum *restrict B,
    int k, int k_user, int k_item, int k_main,
    int *restrict include_ix, int n_include,
    int *restrict exclude_ix, int n_exclude,
    int *restrict outp_ix, FPnum *restrict outp_score,
    int n_top, int n, int nthreads
);
int topN_new_collective_explicit
(
    /* inputs for the factors */
    bool user_bias,
    FPnum *restrict u_vec, int p,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    FPnum *restrict u_bin_vec, int pbin,
    bool NA_as_zero_U, bool NA_as_zero_X,
    FPnum *restrict C, FPnum *restrict Cb,
    FPnum glob_mean, FPnum *restrict biasB,
    FPnum *restrict col_means,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    FPnum *restrict Xa_dense, int n,
    FPnum *restrict weight,
    FPnum *restrict B,
    int k, int k_user, int k_item, int k_main,
    FPnum lam, FPnum *restrict lam_unique,
    FPnum w_main, FPnum w_user,
    int n_max, bool include_all_X,
    FPnum *restrict TransBtBinvBt,
    FPnum *restrict BtB,
    FPnum *restrict BeTBeChol,
    FPnum *restrict CtCw,
    FPnum *restrict TransCtCinvCt,
    FPnum *restrict B_plus_bias,
    /* inputs for topN */
    int *restrict include_ix, int n_include,
    int *restrict exclude_ix, int n_exclude,
    int *restrict outp_ix, FPnum *restrict outp_score,
    int n_top, int nthreads
);
int topN_new_collective_implicit
(
    /* inputs for the factors */
    int n,
    FPnum *restrict u_vec, int p,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    bool NA_as_zero_U,
    FPnum *restrict col_means,
    FPnum *restrict B, FPnum *restrict C,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    int k, int k_user, int k_item, int k_main,
    FPnum lam, FPnum alpha, FPnum w_main, FPnum w_user,
    FPnum w_main_multiplier,
    FPnum *restrict BeTBe,
    FPnum *restrict BtB,
    FPnum *restrict BeTBeChol,
    /* inputs for topN */
    int *restrict include_ix, int n_include,
    int *restrict exclude_ix, int n_exclude,
    int *restrict outp_ix, FPnum *restrict outp_score,
    int n_top, int nthreads
);
int predict_X_old_collective_explicit
(
    int row[], int col[], FPnum *restrict predicted, size_t n_predict,
    FPnum *restrict A, FPnum *restrict biasA,
    FPnum *restrict B, FPnum *restrict biasB,
    FPnum glob_mean,
    int k, int k_user, int k_item, int k_main,
    int m, int n_max,
    int nthreads
);
int predict_X_old_collective_implicit
(
    int row[], int col[], FPnum *restrict predicted, size_t n_predict,
    FPnum *restrict A,
    FPnum *restrict B,
    int k, int k_user, int k_item, int k_main,
    int m, int n,
    int nthreads
);
int predict_X_new_collective_explicit
(
    /* inputs for predictions */
    int m_new,
    int row[], int col[], FPnum *restrict predicted, size_t n_predict,
    int nthreads,
    /* inputs for factors */
    bool user_bias,
    FPnum *restrict U, int m_u, int p,
    bool NA_as_zero_U, bool NA_as_zero_X,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict Ub, int m_ubin, int pbin,
    FPnum *restrict C, FPnum *restrict Cb,
    FPnum glob_mean, FPnum *restrict biasB,
    FPnum *restrict col_means,
    FPnum *restrict X, int ixA[], int ixB[], size_t nnz,
    size_t *restrict Xcsr_p, int *restrict Xcsr_i, FPnum *restrict Xcsr,
    FPnum *restrict Xfull, int n,
    FPnum *restrict weight,
    FPnum *restrict B,
    int k, int k_user, int k_item, int k_main,
    FPnum lam, FPnum *restrict lam_unique,
    FPnum w_main, FPnum w_user,
    int n_max, bool include_all_X,
    FPnum *restrict TransBtBinvBt,
    FPnum *restrict BtB,
    FPnum *restrict BeTBeChol,
    FPnum *restrict TransCtCinvCt,
    FPnum *restrict CtCw,
    FPnum *restrict B_plus_bias
);
int predict_X_new_collective_implicit
(
    /* inputs for predictions */
    int m_new,
    int row[], int col[], FPnum *restrict predicted, size_t n_predict,
    int nthreads,
    /* inputs for factors */
    FPnum *restrict U, int m_u, int p,
    bool NA_as_zero_U,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict X, int ixA[], int ixB[], size_t nnz,
    size_t *restrict Xcsr_p, int *restrict Xcsr_i, FPnum *restrict Xcsr,
    FPnum *restrict B, int n,
    FPnum *restrict C,
    FPnum *restrict col_means,
    int k, int k_user, int k_item, int k_main,
    FPnum lam, FPnum alpha, FPnum w_main, FPnum w_user,
    FPnum w_main_multiplier,
    FPnum *restrict BeTBe,
    FPnum *restrict BtB,
    FPnum *restrict BeTBeChol
);

/* offsets.c */
FPnum offsets_fun_grad
(
    FPnum *restrict values, FPnum *restrict grad,
    int ixA[], int ixB[], FPnum *restrict X,
    size_t nnz, int m, int n, int k,
    FPnum *restrict Xfull, bool full_dense,
    size_t Xcsr_p[], int Xcsr_i[], FPnum *restrict Xcsr,
    size_t Xcsc_p[], int Xcsc_i[], FPnum *restrict Xcsc,
    FPnum *restrict weight, FPnum *restrict weightR, FPnum *restrict weightC,
    bool user_bias, bool item_bias,
    bool add_intercepts,
    FPnum lam, FPnum *restrict lam_unique,
    FPnum *restrict U, int p,
    FPnum *restrict II, int q,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    size_t U_csc_p[], int U_csc_i[], FPnum *restrict U_csc,
    size_t I_csr_p[], int I_csr_i[], FPnum *restrict I_csr,
    size_t I_csc_p[], int I_csc_i[], FPnum *restrict I_csc,
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
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    int k, int k_sec, int k_main,
    FPnum w_user, int nthreads
);
void assign_gradients
(
    FPnum *restrict bufferA, FPnum *restrict g_A, FPnum *restrict g_C,
    bool add_intercepts, FPnum *restrict g_C_bias,
    FPnum *restrict U,
    size_t U_csc_p[], int U_csc_i[], FPnum *restrict U_csc,
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
    FPnum *restrict precomputedTransBtBinvBt,
    FPnum *restrict precomputedBtBw,
    FPnum *restrict output_a,
    FPnum *restrict Bm_plus_bias
);
int precompute_offsets_both
(
    FPnum *restrict A, int m,
    FPnum *restrict B, int n,
    FPnum *restrict C, int p,
    FPnum *restrict D, int q,
    FPnum *restrict C_bias, FPnum *restrict D_bias,
    bool user_bias, bool add_intercepts, bool implicit,
    int k, int k_main, int k_sec,
    FPnum lam, FPnum *restrict lam_unique,
    FPnum w_user, FPnum w_item, 
    FPnum *restrict U,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict II,
    size_t I_csr_p[], int I_csr_i[], FPnum *restrict I_csr,
    int I_row[], int I_col[], FPnum *restrict I_sp, size_t nnz_I,
    FPnum *restrict Am,
    FPnum *restrict Bm,
    FPnum *restrict Bm_plus_bias,
    FPnum *restrict BtB,
    FPnum *restrict TransBtBinvBt
);
int precompute_offsets_explicit
(
    FPnum *restrict A, int m,
    FPnum *restrict B, int n,
    FPnum *restrict C, int p,
    FPnum *restrict D, int q,
    FPnum *restrict C_bias, FPnum *restrict D_bias,
    bool user_bias, bool add_intercepts,
    int k, int k_main, int k_sec,
    FPnum lam, FPnum *restrict lam_unique,
    FPnum w_user, FPnum w_item, 
    FPnum *restrict U,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict II,
    size_t I_csr_p[], int I_csr_i[], FPnum *restrict I_csr,
    int I_row[], int I_col[], FPnum *restrict I_sp, size_t nnz_I,
    FPnum *restrict Am,
    FPnum *restrict Bm,
    FPnum *restrict Bm_plus_bias,
    FPnum *restrict BtB,
    FPnum *restrict TransBtBinvBt
);
int precompute_offsets_implicit
(
    FPnum *restrict A, int m,
    FPnum *restrict B, int n,
    FPnum *restrict C, int p,
    FPnum *restrict D, int q,
    FPnum *restrict C_bias, FPnum *restrict D_bias,
    bool add_intercepts,
    int k,
    FPnum lam,
    FPnum *restrict U,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict II,
    size_t I_csr_p[], int I_csr_i[], FPnum *restrict I_csr,
    int I_row[], int I_col[], FPnum *restrict I_sp, size_t nnz_I,
    FPnum *restrict Am,
    FPnum *restrict Bm,
    FPnum *restrict BtB
);
typedef struct data_offsets_fun_grad {
    int *ixA; int *ixB; FPnum *X;
    size_t nnz; int m; int n; int k;
    FPnum *Xfull; bool full_dense;
    size_t *Xcsr_p; int *Xcsr_i; FPnum *Xcsr;
    size_t *Xcsc_p; int *Xcsc_i; FPnum *Xcsc;
    FPnum *weight; FPnum *weightR; FPnum *weightC;
    bool user_bias; bool item_bias;
    bool add_intercepts;
    FPnum lam; FPnum *lam_unique;
    FPnum *U; int p;
    FPnum *II; int q;
    size_t *U_csr_p; int *U_csr_i; FPnum *U_csr;
    size_t *U_csc_p; int *U_csc_i; FPnum *U_csc;
    size_t *I_csr_p; int *I_csr_i; FPnum *I_csr;
    size_t *I_csc_p; int *I_csc_i; FPnum *I_csc;
    int k_main; int k_sec;
    FPnum w_user; FPnum w_item;
    int nthreads;
    FPnum *buffer_FPnum;
    FPnum *buffer_mt;
    int print_every; int nfev; int niter;
    bool handle_interrupt;
} data_offsets_fun_grad;
lbfgsFPnumval_t wrapper_offsets_fun_grad
(
    void *instance,
    lbfgsFPnumval_t *x,
    lbfgsFPnumval_t *g,
    const size_t n,
    const lbfgsFPnumval_t step
);
int fit_offsets_explicit_lbfgs_internal
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
    bool verbose, int print_every, bool handle_interrupt,
    int *restrict niter, int *restrict nfev,
    FPnum *restrict Am, FPnum *restrict Bm,
    FPnum *restrict Bm_plus_bias
);
int fit_offsets_explicit_lbfgs
(
    FPnum *restrict biasA, FPnum *restrict biasB,
    FPnum *restrict A, FPnum *restrict B,
    FPnum *restrict C, FPnum *restrict C_bias,
    FPnum *restrict D, FPnum *restrict D_bias,
    bool reset_values, int seed,
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
    int n_corr_pairs, size_t maxiter,
    int nthreads, bool prefer_onepass,
    bool verbose, int print_every, bool handle_interrupt,
    int *restrict niter, int *restrict nfev,
    bool precompute_for_predictions,
    FPnum *restrict Am, FPnum *restrict Bm,
    FPnum *restrict Bm_plus_bias,
    FPnum *restrict precomputedBtB,
    FPnum *restrict precomputedTransBtBinvBt
);
int fit_offsets_als
(
    FPnum *restrict biasA, FPnum *restrict biasB,
    FPnum *restrict A, FPnum *restrict B,
    FPnum *restrict C, FPnum *restrict C_bias,
    FPnum *restrict D, FPnum *restrict D_bias,
    bool reset_values, int seed,
    FPnum *restrict glob_mean,
    int m, int n, int k,
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    FPnum *restrict Xfull,
    FPnum *restrict weight,
    bool user_bias, bool item_bias, bool add_intercepts,
    FPnum lam,
    FPnum *restrict U, int p,
    FPnum *restrict II, int q,
    bool implicit, bool NA_as_zero_X, FPnum alpha,
    int niter,
    int nthreads, bool use_cg,
    int max_cg_steps, bool finalize_chol,
    bool verbose, bool handle_interrupt,
    bool precompute_for_predictions,
    FPnum *restrict Am, FPnum *restrict Bm,
    FPnum *restrict Bm_plus_bias,
    FPnum *restrict precomputedBtB,
    FPnum *restrict precomputedTransBtBinvBt

);
int fit_offsets_explicit_als
(
    FPnum *restrict biasA, FPnum *restrict biasB,
    FPnum *restrict A, FPnum *restrict B,
    FPnum *restrict C, FPnum *restrict C_bias,
    FPnum *restrict D, FPnum *restrict D_bias,
    bool reset_values, int seed,
    FPnum *restrict glob_mean,
    int m, int n, int k,
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    FPnum *restrict Xfull,
    FPnum *restrict weight,
    bool user_bias, bool item_bias, bool add_intercepts,
    FPnum lam,
    FPnum *restrict U, int p,
    FPnum *restrict II, int q,
    bool NA_as_zero_X,
    int niter,
    int nthreads, bool use_cg,
    int max_cg_steps, bool finalize_chol,
    bool verbose, bool handle_interrupt,
    bool precompute_for_predictions,
    FPnum *restrict Am, FPnum *restrict Bm,
    FPnum *restrict Bm_plus_bias,
    FPnum *restrict precomputedBtB,
    FPnum *restrict precomputedTransBtBinvBt
);
int fit_offsets_implicit_als
(
    FPnum *restrict A, FPnum *restrict B,
    FPnum *restrict C, FPnum *restrict C_bias,
    FPnum *restrict D, FPnum *restrict D_bias,
    bool reset_values, int seed,
    int m, int n, int k,
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    bool add_intercepts,
    FPnum lam,
    FPnum *restrict U, int p,
    FPnum *restrict II, int q,
    FPnum alpha,
    int niter,
    int nthreads, bool use_cg,
    int max_cg_steps, bool finalize_chol,
    bool verbose, bool handle_interrupt,
    bool precompute_for_predictions,
    FPnum *restrict Am, FPnum *restrict Bm,
    FPnum *restrict Bm_plus_bias,
    FPnum *restrict precomputedBtB,
    FPnum *restrict precomputedTransBtBinvBt
);
int matrix_content_based
(
    FPnum *restrict Am_new,
    int n_new, int k_sec,
    FPnum *restrict U, int p,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict C, FPnum *restrict C_bias,
    int nthreads
);
int factors_offsets_explicit_single
(
    FPnum *restrict a_vec, FPnum *restrict a_bias, FPnum *restrict output_a,
    FPnum *restrict u_vec, int p,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    FPnum *restrict Xa_dense, int n,
    FPnum *restrict weight,
    FPnum *restrict Bm, FPnum *restrict C,
    FPnum *restrict C_bias,
    FPnum glob_mean, FPnum *restrict biasB,
    int k, int k_sec, int k_main,
    FPnum w_user,
    FPnum lam, FPnum *restrict lam_unique,
    bool exact,
    FPnum *restrict precomputedTransBtBinvBt,
    FPnum *restrict precomputedBtB,
    FPnum *restrict Bm_plus_bias
);
int factors_offsets_implicit_single
(
    FPnum *restrict a_vec,
    FPnum *restrict u_vec, int p,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    FPnum *restrict Bm, FPnum *restrict C,
    FPnum *restrict C_bias,
    int k, int n,
    FPnum lam, FPnum alpha,
    FPnum *restrict precomputedBtB,
    FPnum *restrict output_a
);
int factors_offsets_explicit_multiple
(
    FPnum *restrict Am, FPnum *restrict biasA,
    FPnum *restrict A, int m,
    FPnum *restrict U, int p,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict X, int ixA[], int ixB[], size_t nnz,
    size_t *restrict Xcsr_p, int *restrict Xcsr_i, FPnum *restrict Xcsr,
    FPnum *restrict Xfull, int n,
    FPnum *restrict weight,
    FPnum *restrict Bm, FPnum *restrict C,
    FPnum *restrict C_bias,
    FPnum glob_mean, FPnum *restrict biasB,
    int k, int k_sec, int k_main,
    FPnum w_user,
    FPnum lam, FPnum *restrict lam_unique, bool exact,
    FPnum *restrict precomputedTransBtBinvBt,
    FPnum *restrict precomputedBtB,
    FPnum *restrict Bm_plus_bias,
    int nthreads
);
int factors_offsets_implicit_multiple
(
    FPnum *restrict Am, int m,
    FPnum *restrict A,
    FPnum *restrict U, int p,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict X, int ixA[], int ixB[], size_t nnz,
    size_t *restrict Xcsr_p, int *restrict Xcsr_i, FPnum *restrict Xcsr,
    FPnum *restrict Bm, FPnum *restrict C,
    FPnum *restrict C_bias,
    int k, int n,
    FPnum lam, FPnum alpha,
    FPnum *restrict precomputedBtB,
    int nthreads
);
int topN_old_offsets_explicit
(
    FPnum *restrict a_vec, FPnum a_bias,
    FPnum *restrict Am, FPnum *restrict biasA, int row_index,
    FPnum *restrict Bm,
    FPnum *restrict biasB,
    FPnum glob_mean,
    int k, int k_sec, int k_main,
    int *restrict include_ix, int n_include,
    int *restrict exclude_ix, int n_exclude,
    int *restrict outp_ix, FPnum *restrict outp_score,
    int n_top, int n, int nthreads
);
int topN_old_offsets_implicit
(
    FPnum *restrict a_vec,
    FPnum *restrict Am, int row_index,
    FPnum *restrict Bm,
    int k,
    int *restrict include_ix, int n_include,
    int *restrict exclude_ix, int n_exclude,
    int *restrict outp_ix, FPnum *restrict outp_score,
    int n_top, int n, int nthreads
);
int topN_new_offsets_explicit
(
    /* inputs for factors */
    bool user_bias, int n,
    FPnum *restrict u_vec, int p,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    FPnum *restrict Xa_dense,
    FPnum *restrict weight,
    FPnum *restrict Bm, FPnum *restrict C,
    FPnum *restrict C_bias,
    FPnum glob_mean, FPnum *restrict biasB,
    int k, int k_sec, int k_main,
    FPnum w_user,
    FPnum lam, FPnum *restrict lam_unique,
    bool exact,
    FPnum *restrict precomputedTransBtBinvBt,
    FPnum *restrict precomputedBtB,
    FPnum *restrict Bm_plus_bias,
    /* inputs for topN */
    int *restrict include_ix, int n_include,
    int *restrict exclude_ix, int n_exclude,
    int *restrict outp_ix, FPnum *restrict outp_score,
    int n_top, int nthreads
);
int topN_new_offsets_implicit
(
    /* inputs for factors */
    FPnum *restrict u_vec, int p,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    FPnum *restrict Xa, int ixB[], size_t nnz,
    FPnum *restrict Bm, FPnum *restrict C,
    FPnum *restrict C_bias,
    int k,
    FPnum lam, FPnum alpha,
    FPnum *restrict precomputedBtB,
    /* inputs for topN */
    int *restrict include_ix, int n_include,
    int *restrict exclude_ix, int n_exclude,
    int *restrict outp_ix, FPnum *restrict outp_score,
    int n_top, int n, int nthreads
);
int predict_X_old_offsets_explicit
(
    int row[], int col[], FPnum *restrict predicted, size_t n_predict,
    FPnum *restrict Am, FPnum *restrict biasA,
    FPnum *restrict Bm, FPnum *restrict biasB,
    FPnum glob_mean,
    int k, int k_sec, int k_main,
    int m, int n,
    int nthreads
);
int predict_X_old_offsets_implicit
(
    int row[], int col[], FPnum *restrict predicted, size_t n_predict,
    FPnum *restrict Am,
    FPnum *restrict Bm,
    int k,
    int m, int n,
    int nthreads
);
int predict_X_new_offsets_explicit
(
    /* inputs for predictions */
    int m_new, bool user_bias,
    int row[], int col[], FPnum *restrict predicted, size_t n_predict,
    int nthreads,
    /* inputs for factors */
    FPnum *restrict U, int p,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict X, int ixA[], int ixB[], size_t nnz,
    size_t *restrict Xcsr_p, int *restrict Xcsr_i, FPnum *restrict Xcsr,
    FPnum *restrict Xfull, int n, /* <- 'n' MUST be passed */
    FPnum *restrict weight,
    FPnum *restrict Bm, FPnum *restrict C,
    FPnum *restrict C_bias,
    FPnum glob_mean, FPnum *restrict biasB,
    int k, int k_sec, int k_main,
    FPnum w_user,
    FPnum lam, FPnum *restrict lam_unique, bool exact,
    FPnum *restrict precomputedTransBtBinvBt,
    FPnum *restrict precomputedBtB,
    FPnum *restrict Bm_plus_bias
);
int predict_X_new_offsets_implicit
(
    /* inputs for predictions */
    int m_new,
    int row[], int col[], FPnum *restrict predicted, size_t n_predict,
    int n_orig,
    int nthreads,
    /* inputs for factors */
    FPnum *restrict U, int p,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict X, int ixA[], int ixB[], size_t nnz,
    size_t *restrict Xcsr_p, int *restrict Xcsr_i, FPnum *restrict Xcsr,
    FPnum *restrict Bm, FPnum *restrict C,
    FPnum *restrict C_bias,
    int k,
    FPnum lam, FPnum alpha,
    FPnum *restrict precomputedBtB
);
int fit_content_based_lbfgs
(
    FPnum *restrict biasA, FPnum *restrict biasB,
    FPnum *restrict C, FPnum *restrict C_bias,
    FPnum *restrict D, FPnum *restrict D_bias,
    bool start_with_ALS, bool reset_values, int seed,
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
    int n_corr_pairs, size_t maxiter,
    int nthreads, bool prefer_onepass,
    bool verbose, int print_every, bool handle_interrupt,
    int *restrict niter, int *restrict nfev,
    FPnum *restrict Am, FPnum *restrict Bm
);
int factors_content_based_single
(
    FPnum *restrict a_vec, int k,
    FPnum *restrict u_vec, int p,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    FPnum *restrict C, FPnum *restrict C_bias
);
int factors_content_based_mutliple
(
    FPnum *restrict Am, int m_new, int k,
    FPnum *restrict C, FPnum *restrict C_bias,
    FPnum *restrict U, int p,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    int nthreads
);
int topN_old_content_based
(
    FPnum *restrict a_vec, FPnum a_bias,
    FPnum *restrict Am, FPnum *restrict biasA, int row_index,
    FPnum *restrict Bm,
    FPnum *restrict biasB,
    FPnum glob_mean,
    int k,
    int *restrict include_ix, int n_include,
    int *restrict exclude_ix, int n_exclude,
    int *restrict outp_ix, FPnum *restrict outp_score,
    int n_top, int n, int nthreads
);
int topN_new_content_based
(
    /* inputs for the factors */
    int k, int n_new,
    FPnum *restrict u_vec, int p,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    FPnum *restrict II, int q,
    int I_row[], int I_col[], FPnum *restrict I_sp, size_t nnz_I,
    size_t I_csr_p[], int I_csr_i[], FPnum *restrict I_csr,
    FPnum *restrict C, FPnum *restrict C_bias,
    FPnum *restrict D, FPnum *restrict D_bias,
    FPnum glob_mean,
    /* inputs for topN */
    int *restrict outp_ix, FPnum *restrict outp_score,
    int n_top, int nthreads
);
int predict_X_old_content_based
(
    FPnum *restrict predicted, size_t n_predict,
    int m_new, int k,
    int row[], /* <- optional */
    int col[],
    int m_orig, int n_orig,
    FPnum *restrict U, int p,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict C, FPnum *restrict C_bias,
    FPnum *restrict Bm, FPnum *restrict biasB,
    FPnum glob_mean,
    int nthreads
);
int predict_X_new_content_based
(
    FPnum *restrict predicted, size_t n_predict,
    int m_new, int n_new, int k,
    int row[], int col[], /* <- optional */
    FPnum *restrict U, int p,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict II, int q,
    int I_row[], int I_col[], FPnum *restrict I_sp, size_t nnz_I,
    size_t I_csr_p[], int I_csr_i[], FPnum *restrict I_csr,
    FPnum *restrict C, FPnum *restrict C_bias,
    FPnum *restrict D, FPnum *restrict D_bias,
    FPnum glob_mean,
    int nthreads
);

#ifdef __cplusplus
}
#endif
