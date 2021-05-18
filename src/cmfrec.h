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
        (d) Takacs, Gabor, Istvan Pilaszy, and Domonkos Tikk.
            "Applications of the conjugate gradient method for
            implicit feedback collaborative filtering."
            Proceedings of the fifth ACM conference on
            Recommender systems. 2011.
        (e) Rendle, Steffen, Li Zhang, and Yehuda Koren.
            "On the difficulty of evaluating baselines:
            A study on recommender systems."
            arXiv preprint arXiv:1905.01395 (2019).
        (f) Franc, Vojtech, Vaclav Hlavac, and Mirko Navara.
            "Sequential coordinate-wise algorithm for the
            non-negative least squares problem."
            International Conference on Computer Analysis of Images
            and Patterns. Springer, Berlin, Heidelberg, 2005.
        (g) Zhou, Yunhong, et al.
            "Large-scale parallel collaborative filtering for
             the netflix prize."
            International conference on algorithmic applications in management.
            Springer, Berlin, Heidelberg, 2008.

    For information about the models offered here and how they are fit to
    the data, see the files 'collective.c' and 'offsets.c'.

    Written for C99 standard and OpenMP version 2.0 or higher, and aimed to be
    used either as a stand-alone program, or wrapped into scripting languages
    such as Python and R.
    <https://www.github.com/david-cortes/cmfrec>

    

    MIT License:

    Copyright (c) 2020-2021 David Cortes

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
#include <float.h>
#ifndef _FOR_R
    #include <stdio.h>
#endif
#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() (0)
#endif
#include <signal.h>
typedef void (*sig_t_)(int);

#ifdef _FOR_PYTHON
    /* This contains the standard cblas.h header */
    #ifdef USE_FINDBLAS
        #include "findblas.h" /* https://www.github.com/david-cortes/findblas */
    #endif
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

/*  OpenMP < 3.0 (e.g. MSVC as of 2020) does not support parallel for's with unsigned iterators,
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
    #define real_t double
    #define exp_t exp
    #define log_t log
    #define fabs_t fabs
    #define fmax_t fmax
    #define sqrt_t sqrt
    #define EPSILON_T DBL_EPSILON
    #define HUGE_VAL_T HUGE_VAL
    #define cblas_tdot cblas_ddot
    #define cblas_tcopy cblas_dcopy
    #define cblas_taxpy cblas_daxpy
    #define cblas_tscal cblas_dscal
    #define cblas_tsyr cblas_dsyr
    #define cblas_tsyrk cblas_dsyrk
    #define cblas_tnrm2 cblas_dnrm2
    #define cblas_tgemm cblas_dgemm
    #define cblas_tgemv cblas_dgemv
    #define cblas_tger cblas_dger
    #define cblas_tsymv cblas_dsymv
    #define tlacpy_ dlacpy_
    #define tposv_ dposv_
    #define tlarnv_ dlarnv_
    #define tpotrf_ dpotrf_
    #define tpotrs_ dpotrs_
    #define tgelsd_ dgelsd_
#else
    #define LBFGS_FLOAT 32
    #define real_t float
    #define exp_t expf
    #define log_t logf
    #define fmax_t fmaxf
    #define fabs_t fabsf
    #define sqrt_t sqrtf
    #define EPSILON_T FLT_EPSILON
    #define HUGE_VAL_T HUGE_VALF
    #define cblas_tdot cblas_sdot
    #define cblas_tcopy cblas_scopy
    #define cblas_taxpy cblas_saxpy
    #define cblas_tscal cblas_sscal
    #define cblas_tsyr cblas_ssyr
    #define cblas_tsyrk cblas_ssyrk
    #define cblas_tnrm2 cblas_snrm2
    #define cblas_tgemm cblas_sgemm
    #define cblas_tgemv cblas_sgemv
    #define cblas_tger cblas_sger
    #define cblas_tsymv cblas_ssymv
    #define tlacpy_ slacpy_
    #define tposv_ sposv_
    #define tlarnv_ slarnv_
    #define tpotrf_ spotrf_
    #define tpotrs_ spotrs_
    #define tgelsd_ sgelsd_
#endif

#ifndef isfinite
    #define isfinite(x) ((x) > (-HUGE_VAL_T) && (x) < HUGE_VAL_T)
#endif

#if !defined(USE_INT64) && !defined(MKL_ILP64)
    #define int_t int
#else
    #define ILP64 
    #include <inttypes.h>
    #define int_t int64_t
#endif

#if !defined(LAPACK_H) && !defined(_FOR_R)
void tposv_(const char*, const int_t*, const int_t*, const real_t*, const int_t*, const real_t*, const int_t*, const int_t*);
void tlacpy_(const char*, const int_t*, const int_t*, const real_t*, const int_t*, const real_t*, const int_t*);
void tlarnv_(const int_t*, const int_t*, const int_t*, const real_t*);
void tpotrf_(const char*, const int_t*, const real_t*, const int_t*, const int_t*);
void tpotrs_(const char*, const int_t*, const int_t*, const real_t*, const int_t*, const real_t*, const int_t*, const int_t*);
void tgelsd_(const int_t*, const int_t*, const int_t*,
             const real_t*, const int_t*,
             const real_t*, const int_t*,
             const real_t*, const real_t*, const int_t*, const real_t*,
             const int_t*, const int_t*, const int_t*);
#endif

#ifndef CBLAS_H
typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef CBLAS_ORDER CBLAS_LAYOUT;

real_t  cblas_tdot(const int_t n, const real_t  *x, const int_t incx, const real_t  *y, const int_t incy);
void cblas_tcopy(const int_t n, const real_t *x, const int_t incx, real_t *y, const int_t incy);
void cblas_taxpy(const int_t n, const real_t alpha, const real_t *x, const int_t incx, real_t *y, const int_t incy);
void cblas_tscal(const int_t N, const real_t alpha, real_t *X, const int_t incX);
void cblas_tsyr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const int_t N, const real_t alpha, const real_t *X, const int_t incX, real_t *A, const int_t lda);
void cblas_tsyrk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans,
         const int_t N, const int_t K, const real_t alpha, const real_t *A, const int_t lda, const real_t beta, real_t *C, const int_t ldc);
real_t  cblas_tnrm2 (const int_t N, const real_t  *X, const int_t incX);
void cblas_tgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int_t M, const int_t N, const int_t K,
         const real_t alpha, const real_t *A, const int_t lda, const real_t *B, const int_t ldb, const real_t beta, real_t *C, const int_t ldc);
void cblas_tgemv(const CBLAS_ORDER order,  const CBLAS_TRANSPOSE trans,  const int_t m, const int_t n,
         const real_t alpha, const real_t  *a, const int_t lda,  const real_t  *x, const int_t incx,  const real_t beta,  real_t  *y, const int_t incy);
void cblas_tsymv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const int_t N, const real_t alpha, const real_t *A,
                 const int_t lda, const real_t *X, const int_t incX, const real_t beta, real_t *Y, const int_t incY);
void cblas_tger(const CBLAS_ORDER order, const int_t m, const int_t n, const real_t alpha,
                const real_t *x, const int_t incx, const real_t *y, const int_t incy, real_t *a, const int_t lda);
#endif


#include "lbfgs.h"

#define square(x) ( (x) * (x) )
#define cap_to_4(x) (((x) > 4)? 4 : (x))
#define max2(a, b) ((a) >= ((b))? (a) : (b))
#define min2(a, b) ((a) <= ((b))? (a) : (b))
#define set_to_zero(arr, n) memset((arr), 0, (size_t)(n)*sizeof(real_t))
#define copy_arr(from, to, n) memcpy((to), (from), (size_t)(n)*sizeof(real_t))


/* helpers.c */
void set_to_zero_(real_t *arr, const size_t n, int nthreads);
void copy_arr_(real_t *restrict src, real_t *restrict dest, size_t n, int nthreads);
int_t count_NAs(real_t arr[], size_t n, int nthreads);
void count_NAs_by_row
(
    real_t *restrict arr, int_t m, int_t n,
    int_t *restrict cnt_NA, int nthreads,
    bool *restrict full_dense, bool *restrict near_dense
);
void count_NAs_by_col
(
    real_t *restrict arr, int_t m, int_t n,
    int_t *restrict cnt_NA,
    bool *restrict full_dense, bool *restrict near_dense
);
void sum_by_rows(real_t *restrict A, real_t *restrict outp, int_t m, int_t n, int nthreads);
void sum_by_cols(real_t *restrict A, real_t *restrict outp, int_t m, int_t n, size_t lda, int nthreads);
void mat_plus_rowvec(real_t *restrict A, real_t *restrict b, int_t m, int_t n, int nthreads);
void mat_plus_colvec(real_t *restrict A, real_t *restrict b, real_t alpha, int_t m, int_t n, size_t lda, int nthreads);
void mat_minus_rowvec2
(
    real_t *restrict Xfull,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t *restrict b, int_t m, int_t n, int nthreads
);
void mat_minus_colvec2
(
    real_t *restrict Xfull,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t *restrict b, int_t m, int_t n, int nthreads
);
void nan_to_zero(real_t *restrict arr, real_t *restrict comp, size_t n, int nthreads);
void mult_if_non_nan(real_t *restrict arr, real_t *restrict comp, real_t *restrict w, size_t n, int nthreads);
void mult_elemwise(real_t *restrict inout, real_t *restrict other, size_t n, int nthreads);
real_t sum_squares(real_t *restrict arr, size_t n, int nthreads);
void taxpy_large(real_t *restrict A, real_t x, real_t *restrict Y, size_t n, int nthreads);
void tscal_large(real_t *restrict arr, real_t alpha, size_t n, int nthreads);
int_t rnorm(real_t *restrict arr, size_t n, int_t seed, int nthreads);
void rnorm_preserve_seed(real_t *restrict arr, size_t n, int_t seed_arr[4]);
void process_seed_for_larnv(int_t seed_arr[4]);
void reduce_mat_sum(real_t *restrict outp, size_t lda, real_t *restrict inp,
                    int_t m, int_t n, int nthreads);
void exp_neg_x(real_t *restrict arr, size_t n, int nthreads);
void add_to_diag(real_t *restrict A, real_t val, size_t n);
real_t sum_sq_div_w(real_t *restrict arr, real_t *restrict w, size_t n, bool compensated, int nthreads);
void tgemm_sp_dense
(
    int_t m, int_t n, real_t alpha,
    size_t indptr[], int_t indices[], real_t values[],
    real_t DenseMat[], size_t ldb,
    real_t OutputMat[], size_t ldc,
    int nthreads
);
void tgemv_dense_sp
(
    int_t m, int_t n,
    real_t alpha, real_t DenseMat[], size_t lda,
    int_t ixB[], real_t vec_sp[], size_t nnz,
    real_t OutputVec[]
);
void tgemv_dense_sp_weighted
(
    int_t m, int_t n,
    real_t alpha[], real_t DenseMat[], size_t lda,
    int_t ixB[], real_t vec_sp[], size_t nnz,
    real_t OutputVec[]
);
void tgemv_dense_sp_weighted2
(
    int_t m, int_t n,
    real_t alpha[], real_t alpha2, real_t DenseMat[], size_t lda,
    int_t ixB[], real_t vec_sp[], size_t nnz,
    real_t OutputVec[]
);
void tgemv_dense_sp_notrans
(
    int_t m, int_t n,
    real_t DenseMat[], int_t lda,
    int_t ixB[], real_t vec_sp[], size_t nnz,
    real_t OutputVec[]
);
void copy_mat
(
    int_t m, int_t n,
    real_t *restrict A, int_t lda,
    real_t *restrict B, int_t ldb
);
void sum_mat
(
    size_t m, size_t n,
    real_t *restrict A, size_t lda,
    real_t *restrict B, size_t ldb
);
void transpose_mat(real_t *restrict A, size_t m, size_t n, real_t *restrict buffer_real_t);
void transpose_mat2(real_t *restrict A, size_t m, size_t n, real_t *restrict outp);
void transpose_mat3
(
    real_t *restrict A, size_t lda,
    size_t m, size_t n,
    real_t *restrict outp, size_t ldb
);
int_t coo_to_csr_plus_alloc
(
    int_t *restrict Xrow, int_t *restrict Xcol, real_t *restrict Xval,
    real_t *restrict W,
    int_t m, int_t n, size_t nnz,
    size_t *restrict *csr_p, int_t *restrict *csr_i, real_t *restrict *csr_v,
    real_t *restrict *csr_w
);
void coo_to_csr
(
    int_t *restrict Xrow, int_t *restrict Xcol, real_t *restrict Xval,
    real_t *restrict W,
    int_t m, int_t n, size_t nnz,
    size_t *restrict csr_p, int_t *restrict csr_i, real_t *restrict csr_v,
    real_t *restrict csr_w
);
void coo_to_csr_and_csc
(
    int_t *restrict Xrow, int_t *restrict Xcol, real_t *restrict Xval,
    real_t *restrict W, int_t m, int_t n, size_t nnz,
    size_t *restrict csr_p, int_t *restrict csr_i, real_t *restrict csr_v,
    size_t *restrict csc_p, int_t *restrict csc_i, real_t *restrict csc_v,
    real_t *restrict csr_w, real_t *restrict csc_w,
    int nthreads
);
void row_means_csr(size_t indptr[], real_t *restrict values,
                   real_t *restrict output, int_t m, int nthreads);
extern bool should_stop_procedure;
extern bool handle_is_locked;
void set_interrup_global_variable(int_t s);
int_t lbfgs_printer_collective
(
    void *instance,
    const real_t *x,
    const real_t *g,
    const real_t fx,
    const real_t xnorm,
    const real_t gnorm,
    const real_t step,
    size_t n,
    int_t k,
    int_t ls
);
int_t lbfgs_printer_offsets
(
    void *instance,
    const real_t *x,
    const real_t *g,
    const real_t fx,
    const real_t xnorm,
    const real_t gnorm,
    const real_t step,
    size_t n,
    int_t k,
    int_t ls
);
bool check_is_sorted(int_t arr[], int_t n);
void qs_argpartition(int_t arr[], real_t values[], int_t n, int_t k);
void append_ones_last_col
(
    real_t *restrict orig, size_t m, size_t n,
    real_t *restrict outp
);
void fill_lower_triangle(real_t A[], size_t n, size_t lda);
void print_err_msg(const char *msg);
void print_oom_message(void);
void act_on_interrupt(int retval, bool handle_interrupt, bool print_msg);
#ifdef _FOR_R
void R_nan_to_C_nan(real_t arr[], size_t n);
#endif
long double compensated_sum(real_t *arr, size_t n);
long double compensated_sum_product(real_t *restrict arr1, real_t *restrict arr2, size_t n);
void custom_syr(const int_t n, const real_t alpha, const real_t *restrict x, real_t *restrict A, const int_t lda);


/* common.c */
real_t fun_grad_cannonical_form
(
    real_t *restrict A, int_t lda, real_t *restrict B, int_t ldb,
    real_t *restrict g_A, real_t *restrict g_B,
    int_t m, int_t n, int_t k,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t *restrict Xfull, bool full_dense,
    size_t Xcsr_p[], int_t Xcsr_i[], real_t *restrict Xcsr,
    size_t Xcsc_p[], int_t Xcsc_i[], real_t *restrict Xcsc,
    bool user_bias, bool item_bias,
    real_t *restrict biasA, real_t *restrict biasB,
    real_t *restrict g_biasA, real_t *restrict g_biasB,
    real_t *restrict weight, real_t *restrict weightR, real_t *restrict weightC,
    real_t scaling,
    real_t *restrict buffer_real_t,
    real_t *restrict buffer_mt,
    int nthreads
);
void factors_closed_form
(
    real_t *restrict a_vec, int_t k,
    real_t *restrict B, int_t n, int_t ldb,
    real_t *restrict Xa_dense, bool full_dense,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    real_t *restrict weight,
    real_t *restrict buffer_real_t,
    real_t lam, real_t lam_last,
    real_t l1_lam, real_t l1_lam_last,
    bool scale_lam, bool scale_bias_const, real_t wsum,
    real_t *restrict precomputedTransBtBinvBt,
    real_t *restrict precomputedBtB, int_t cnt_NA, int_t ld_BtB,
    bool BtB_has_diag, bool BtB_is_scaled, real_t scale_BtB, int_t n_BtB,
    real_t *restrict precomputedBtBchol, bool NA_as_zero,
    bool use_cg, int_t max_cg_steps,/* <- 'cg' should not be used for new data*/
    bool nonneg, int_t max_cd_steps,
    real_t *restrict bias_BtX, real_t *restrict bias_X, real_t bias_X_glob,
    real_t multiplier_bias_BtX,
    bool force_add_diag
);
void factors_explicit_cg
(
    real_t *restrict a_vec, int_t k,
    real_t *restrict B, int_t n, int_t ldb,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    real_t *restrict weight,
    real_t *restrict buffer_real_t,
    real_t lam, real_t lam_last,
    int_t max_cg_steps
);
void factors_explicit_cg_NA_as_zero_weighted
(
    real_t *restrict a_vec, int_t k,
    real_t *restrict B, int_t n, int_t ldb,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    real_t *restrict weight,
    real_t *restrict precomputedBtB, int_t ld_BtB,
    real_t *restrict bias_BtX, real_t *restrict bias_X, real_t bias_X_glob,
    real_t multiplier_bias_BtX,
    real_t *restrict buffer_real_t,
    real_t lam, real_t lam_last,
    int_t max_cg_steps
);
void factors_explicit_cg_dense
(
    real_t *restrict a_vec, int_t k,
    real_t *restrict B, int_t n, int_t ldb,
    real_t *restrict Xa_dense, int_t cnt_NA,
    real_t *restrict weight,
    real_t *restrict precomputedBtB, int_t ld_BtB,
    real_t *restrict buffer_real_t,
    real_t lam, real_t lam_last,
    int_t max_cg_steps
);
void factors_implicit_cg
(
    real_t *restrict a_vec, int_t k,
    real_t *restrict B, size_t ldb,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    real_t lam,
    real_t *restrict precomputedBtB, int_t ld_BtB,
    int_t max_cg_steps,
    real_t *restrict buffer_real_t
);
void factors_implicit_chol
(
    real_t *restrict a_vec, int_t k,
    real_t *restrict B, size_t ldb,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    real_t lam, real_t l1_lam,
    real_t *restrict precomputedBtB, int_t ld_BtB,
    bool nonneg, int_t max_cd_steps,
    real_t *restrict buffer_real_t
);
void solve_nonneg
(
    real_t *restrict BtB,
    real_t *restrict BtX, /* <- solution will be here */
    real_t *restrict buffer_real_t,
    int_t k,
    real_t l1_lam, real_t l1_lam_last,
    size_t max_cd_steps,
    bool fill_lower
);
void solve_nonneg_batch
(
    real_t *restrict BtB,
    real_t *restrict BtX, /* <- solution will be here */
    real_t *restrict buffer_real_t,
    int_t m, int_t k, size_t lda,
    real_t l1_lam, real_t l1_lam_last,
    size_t max_cd_steps,
    int nthreads
);
void solve_elasticnet
(
    real_t *restrict BtB,
    real_t *restrict BtX, /* <- solution will be here */
    real_t *restrict buffer_real_t,
    int_t k,
    real_t l1_lam, real_t l1_lam_last,
    size_t max_cd_steps,
    bool fill_lower
);
void solve_elasticnet_batch
(
    real_t *restrict BtB,
    real_t *restrict BtX, /* <- solution will be here */
    real_t *restrict buffer_real_t,
    int_t m, int_t k, size_t lda,
    real_t l1_lam, real_t l1_lam_last,
    size_t max_cd_steps,
    int nthreads
);
real_t fun_grad_Adense
(
    real_t *restrict g_A,
    real_t *restrict A, int_t lda,
    real_t *restrict B, int_t ldb,
    int_t m, int_t n, int_t k,
    real_t *restrict Xfull, real_t *restrict weight,
    real_t lam, real_t w, real_t lam_last,
    bool do_B, bool reset_grad,
    int nthreads,
    real_t *restrict buffer_real_t
);
void add_lam_to_grad_and_fun
(
    real_t *restrict fun,
    real_t *restrict grad,
    real_t *restrict A,
    int_t m, int_t k, int_t lda,
    real_t lam, int nthreads
);
typedef struct data_fun_grad_Adense {
    int_t lda;
    real_t *B; int_t ldb;
    int_t m; int_t n; int_t k;
    real_t *Xfull; real_t *weight;
    real_t lam; real_t w; real_t lam_last;
    int nthreads;
    real_t *buffer_real_t;
} data_fun_grad_Adense;
typedef struct data_fun_grad_Bdense {
    real_t *A; int_t lda;
    int_t ldb;
    int_t m; int_t n; int_t k;
    real_t *Xfull; real_t *weight;
    real_t lam; real_t w; real_t lam_last;
    int nthreads;
    real_t *buffer_real_t;
} data_fun_grad_Bdense;
real_t wrapper_fun_grad_Adense
(
    void *instance,
    real_t *x,
    real_t *g,
    const size_t n,
    const real_t step
);
real_t wrapper_fun_grad_Bdense
(
    void *instance,
    real_t *x,
    real_t *g,
    const size_t n,
    const real_t step
);
size_t buffer_size_optimizeA
(
    size_t n, bool full_dense, bool near_dense, bool do_B,
    bool has_dense, bool has_weights, bool NA_as_zero,
    bool nonneg, bool has_l1,
    size_t k, size_t nthreads,
    bool has_bias_static,
    bool pass_allocated_BtB, bool keep_precomputedBtB,
    bool use_cg, bool finalize_chol
);
size_t buffer_size_optimizeA_implicit
(
    size_t k, size_t nthreads,
    bool pass_allocated_BtB,
    bool nonneg, bool has_l1,
    bool use_cg, bool finalize_chol
);
void optimizeA
(
    real_t *restrict A, int_t lda,
    real_t *restrict B, int_t ldb,
    int_t m, int_t n, int_t k,
    size_t Xcsr_p[], int_t Xcsr_i[], real_t *restrict Xcsr,
    real_t *restrict Xfull, int_t ldX, bool full_dense, bool near_dense,
    int_t cnt_NA[], real_t *restrict weight, bool NA_as_zero,
    real_t lam, real_t lam_last,
    real_t l1_lam, real_t l1_lam_last,
    bool scale_lam, bool scale_bias_const, real_t *restrict wsumA,
    bool do_B, bool is_first_iter,
    int nthreads,
    bool use_cg, int_t max_cg_steps,
    bool nonneg, int_t max_cd_steps,
    real_t *restrict bias_restore,
    real_t *restrict bias_BtX, real_t *restrict bias_X, real_t bias_X_glob,
    real_t *restrict bias_static, real_t multiplier_bias_BtX,
    bool keep_precomputedBtB,
    real_t *restrict precomputedBtB, bool *filled_BtB,
    real_t *restrict buffer_real_t
);
void optimizeA_implicit
(
    real_t *restrict A, size_t lda,
    real_t *restrict B, size_t ldb,
    int_t m, int_t n, int_t k,
    size_t Xcsr_p[], int_t Xcsr_i[], real_t *restrict Xcsr,
    real_t lam, real_t l1_lam,
    int nthreads,
    bool use_cg, int_t max_cg_steps, bool force_set_to_zero,
    bool nonneg, int_t max_cd_steps,
    real_t *restrict precomputedBtB, /* <- will be calculated if not passed */
    real_t *restrict buffer_real_t
);
void calc_mean_and_center
(
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t *restrict Xfull, real_t *restrict Xtrans,
    int_t m, int_t n,
    size_t Xcsr_p[], int_t Xcsr_i[], real_t *restrict Xcsr,
    size_t Xcsc_p[], int_t Xcsc_i[], real_t *restrict Xcsc,
    real_t *restrict weight,
    bool NA_as_zero, bool nonneg, bool center, int nthreads,
    real_t *restrict glob_mean
);
int_t initialize_biases
(
    real_t *restrict glob_mean, real_t *restrict biasA, real_t *restrict biasB,
    bool user_bias, bool item_bias, bool center,
    real_t lam_user, real_t lam_item,
    bool scale_lam, bool scale_bias_const,
    bool force_calc_user_scale, bool force_calc_item_scale,
    real_t *restrict scaling_biasA, real_t *restrict scaling_biasB,
    int_t m, int_t n,
    int_t m_bias, int_t n_bias,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t *restrict Xfull, real_t *restrict Xtrans,
    size_t Xcsr_p[], int_t Xcsr_i[], real_t *restrict Xcsr,
    size_t Xcsc_p[], int_t Xcsc_i[], real_t *restrict Xcsc,
    real_t *restrict weight, real_t *restrict Wtrans,
    real_t *restrict weightR, real_t *restrict weightC,
    bool nonneg,
    int nthreads
);
int_t initialize_biases_onesided
(
    real_t *restrict Xfull, int_t m, int_t n, bool do_B, int_t *restrict cnt_NA,
    size_t Xcsr_p[], int_t Xcsr_i[], real_t *restrict Xcsr,
    real_t *restrict weight, real_t *restrict weightR,
    real_t glob_mean, bool NA_as_zero, bool nonneg,
    real_t lam, bool scale_lam,
    real_t *restrict wsumA,
    real_t *restrict biasA,
    int nthreads
);
int_t initialize_biases_twosided
(
    real_t *restrict Xfull, real_t *restrict Xtrans,
    int_t *restrict cnt_NA_byrow, int_t *restrict cnt_NA_bycol,
    int_t m, int_t n,
    bool NA_as_zero, bool nonneg, double glob_mean,
    size_t *restrict Xcsr_p, int_t *restrict Xcsr_i, real_t *Xcsr,
    size_t *restrict Xcsc_p, int_t *restrict Xcsc_i, real_t *Xcsc,
    real_t *restrict weight, real_t *restrict Wtrans,
    real_t *restrict weightR, real_t *restrict weightC,
    real_t lam_user, real_t lam_item, bool scale_lam,
    real_t *restrict wsumA, real_t *restrict wsumB,
    real_t *restrict biasA, real_t *restrict biasB,
    int nthreads
);
int_t center_by_cols
(
    real_t *restrict col_means,
    real_t *restrict Xfull, int_t m, int_t n,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    size_t Xcsr_p[], int_t Xcsr_i[], real_t *restrict Xcsr,
    size_t Xcsc_p[], int_t Xcsc_i[], real_t *restrict Xcsc,
    int nthreads
);
bool check_sparse_indices
(
    int_t n, int_t p,
    real_t *restrict u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
    real_t *restrict Xa, int_t ixB[], size_t nnz
);
void predict_multiple
(
    real_t *restrict A, int_t k_user,
    real_t *restrict B, int_t k_item,
    real_t *restrict biasA, real_t *restrict biasB,
    real_t glob_mean,
    int_t k, int_t k_main,
    int_t m, int_t n,
    int_t predA[], int_t predB[], size_t nnz,
    real_t *restrict outp,
    int nthreads
);
int_t cmp_int(const void *a, const void *b);
extern real_t *ptr_real_t_glob;
#if !defined(_WIN32) && !defined(_WIN64)
#pragma omp threadprivate(ptr_real_t_glob)
/* Note: will not be used inside OMP, this is a precausion just in case */
#endif
int_t cmp_argsort(const void *a, const void *b);
int_t topN
(
    real_t *restrict a_vec, int_t k_user,
    real_t *restrict B, int_t k_item,
    real_t *restrict biasB,
    real_t glob_mean, real_t biasA,
    int_t k, int_t k_main,
    int_t *restrict include_ix, int_t n_include,
    int_t *restrict exclude_ix, int_t n_exclude,
    int_t *restrict outp_ix, real_t *restrict outp_score,
    int_t n_top, int_t n, int nthreads
);
int_t fit_most_popular
(
    real_t *restrict biasA, real_t *restrict biasB,
    real_t *restrict glob_mean,
    real_t lam_user, real_t lam_item,
    bool scale_lam, bool scale_bias_const,
    real_t alpha,
    int_t m, int_t n,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t *restrict Xfull,
    real_t *restrict weight,
    bool implicit, bool adjust_weight, bool apply_log_transf,
    bool nonneg, bool NA_as_zero,
    real_t *restrict w_main_multiplier,
    int nthreads
);
int_t fit_most_popular_internal
(
    real_t *restrict biasA, real_t *restrict biasB,
    real_t *restrict glob_mean, bool center,
    real_t lam_user, real_t lam_item,
    bool scale_lam, bool scale_bias_const,
    real_t alpha,
    int_t m, int_t n,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t *restrict Xfull,
    real_t *restrict weight,
    bool implicit, bool adjust_weight, bool apply_log_transf,
    bool nonneg,
    real_t *restrict w_main_multiplier,
    int nthreads
);
int_t topN_old_most_popular
(
    bool user_bias,
    real_t a_bias,
    real_t *restrict biasA, int_t row_index,
    real_t *restrict biasB,
    real_t glob_mean,
    int_t *restrict include_ix, int_t n_include,
    int_t *restrict exclude_ix, int_t n_exclude,
    int_t *restrict outp_ix, real_t *restrict outp_score,
    int_t n_top, int_t n
);
int_t predict_X_old_most_popular
(
    int_t row[], int_t col[], real_t *restrict predicted, size_t n_predict,
    real_t *restrict biasA, real_t *restrict biasB,
    real_t glob_mean,
    int_t m, int_t n
);

/* collective.c */
void nvars_collective_fun_grad
(
    size_t m, size_t n, size_t m_u, size_t n_i, size_t m_ubin, size_t n_ibin,
    size_t p, size_t q, size_t pbin, size_t qbin,
    size_t nnz, size_t nnz_U, size_t nnz_I,
    size_t k, size_t k_main, size_t k_user, size_t k_item,
    bool user_bias, bool item_bias, size_t nthreads,
    real_t *X, real_t *Xfull,
    real_t *U, real_t *Ub, real_t *II, real_t *Ib,
    real_t *U_sp, real_t *U_csr, real_t *I_sp, real_t *I_csr,
    size_t *nvars, size_t *nbuffer, size_t *nbuffer_mt
);
real_t collective_fun_grad
(
    real_t *restrict values, real_t *restrict grad,
    int_t m, int_t n, int_t k,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t *restrict Xfull,
    size_t Xcsr_p[], int_t Xcsr_i[], real_t *restrict Xcsr,
    size_t Xcsc_p[], int_t Xcsc_i[], real_t *restrict Xcsc,
    real_t *restrict weight, real_t *restrict weightR, real_t *restrict weightC,
    bool user_bias, bool item_bias,
    real_t lam, real_t *restrict lam_unique,
    real_t *restrict U, int_t m_u, int_t p, bool U_has_NA,
    real_t *restrict II, int_t n_i, int_t q, bool I_has_NA,
    real_t *restrict Ub, int_t m_ubin, int_t pbin, bool Ub_has_NA,
    real_t *restrict Ib, int_t n_ibin, int_t qbin, bool Ib_has_NA,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    int_t I_row[], int_t I_col[], real_t *restrict I_sp, size_t nnz_I,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    size_t U_csc_p[], int_t U_csc_i[], real_t *restrict U_csc,
    size_t I_csr_p[], int_t I_csr_i[], real_t *restrict I_csr,
    size_t I_csc_p[], int_t I_csc_i[], real_t *restrict I_csc,
    real_t *restrict buffer_real_t, real_t *restrict buffer_mt,
    int_t k_main, int_t k_user, int_t k_item,
    real_t w_main, real_t w_user, real_t w_item,
    int nthreads
);
real_t collective_fun_grad_bin
(
    real_t *restrict A, int_t lda, real_t *restrict Cb, int_t ldc,
    real_t *restrict g_A, real_t *restrict g_Cb,
    real_t *restrict Ub,
    int_t m, int_t pbin, int_t k,
    bool Ub_has_NA, double w_user,
    real_t *restrict buffer_real_t,
    int nthreads
);
real_t collective_fun_grad_single
(
    real_t *restrict a_vec, real_t *restrict g_A,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t *restrict u_vec, int_t p,
    int_t u_vec_ixB[], real_t *restrict u_vec_sp, size_t nnz_u_vec,
    real_t *restrict u_bin_vec, int_t pbin,
    bool u_vec_has_NA, bool u_bin_vec_has_NA,
    real_t *restrict B, int_t n,
    real_t *restrict C, real_t *restrict Cb,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    real_t *restrict Xa_dense,
    real_t *restrict weight,
    real_t *restrict buffer_real_t,
    real_t lam, real_t w_main, real_t w_user, real_t lam_last
);
typedef struct data_factors_fun_grad {
    int_t k; int_t k_user; int_t k_item; int_t k_main;
    real_t *u_vec; int_t p;
    int_t *u_vec_ixB; real_t *u_vec_sp; size_t nnz_u_vec;
    real_t *u_bin_vec; int_t pbin;
    bool u_vec_has_NA; bool u_bin_vec_has_NA;
    real_t *B; int_t n;
    real_t *C; real_t *Cb;
    real_t *Xa; int_t *ixB; real_t *weight; size_t nnz;
    real_t *Xa_dense;
    real_t *buffer_real_t;
    real_t lam; real_t w_main; real_t w_user; real_t lam_last;
} data_factors_fun_grad;
real_t wrapper_factors_fun_grad
(
    void *instance,
    real_t *x,
    real_t *g,
    const size_t n,
    const real_t step
);
int_t collective_factors_lbfgs
(
    real_t *restrict a_vec,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t *restrict u_vec, int_t p,
    int_t u_vec_ixB[], real_t *restrict u_vec_sp, size_t nnz_u_vec,
    real_t *restrict u_bin_vec, int_t pbin,
    bool u_vec_has_NA, bool u_bin_vec_has_NA,
    real_t *restrict B, int_t n,
    real_t *restrict C, real_t *restrict Cb,
    real_t *restrict Xa, int_t ixB[], real_t *restrict weight, size_t nnz,
    real_t *restrict Xa_dense,
    real_t *restrict buffer_real_t,
    real_t lam, real_t w_main, real_t w_user, real_t lam_last
);
void collective_closed_form_block
(
    real_t *restrict a_vec,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t *restrict Xa_dense,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    int_t u_vec_ixB[], real_t *restrict u_vec_sp, size_t nnz_u_vec,
    real_t *restrict u_vec,
    bool NA_as_zero_X, bool NA_as_zero_U,
    real_t *restrict B, int_t n, int_t ldb,
    real_t *restrict C, int_t p,
    real_t *restrict Bi, int_t k_main_i, bool add_implicit_features,
    real_t *restrict Xones, int_t incXones,
    real_t *restrict weight,
    real_t lam, real_t w_user, real_t w_implicit, real_t lam_last,
    real_t l1_lam, real_t l1_lam_bias,
    bool scale_lam, bool scale_lam_sideinfo, bool scale_bias_const, real_t wsum,
    real_t *restrict precomputedBtB, int_t cnt_NA_x,
    real_t *restrict precomputedCtCw, int_t cnt_NA_u,
    real_t *restrict precomputedBeTBeChol, int_t n_BtB,
    real_t *restrict precomputedBiTBi,
    bool add_X, bool add_U,
    bool use_cg, int_t max_cg_steps,/* <- 'cg' should not be used for new data*/
    bool nonneg, int_t max_cd_steps,
    real_t *restrict bias_BtX, real_t *restrict bias_X, real_t bias_X_glob,
    real_t *restrict bias_CtU,
    real_t *restrict buffer_real_t
);
void collective_closed_form_block_implicit
(
    real_t *restrict a_vec,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t *restrict B, int_t n, real_t *restrict C, int_t p,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    real_t *restrict u_vec, int_t cnt_NA_u,
    real_t *restrict u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
    bool NA_as_zero_U,
    real_t lam, real_t l1_lam, real_t w_user,
    real_t *restrict bias_CtU,
    real_t *restrict precomputedBeTBe,
    real_t *restrict precomputedBtB, /* for cg, should NOT have lambda added */
    real_t *restrict precomputedBeTBeChol,
    real_t *restrict precomputedCtCw,
    bool add_U, bool shapes_match,
    bool use_cg, int_t max_cg_steps,/* <- 'cg' should not be used for new data*/
    bool nonneg, int_t max_cd_steps,
    real_t *restrict buffer_real_t
);
void collective_block_cg
(
    real_t *restrict a_vec,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t *restrict Xa_dense,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    int_t u_vec_ixB[], real_t *restrict u_vec_sp, size_t nnz_u_vec,
    real_t *restrict u_vec,
    bool NA_as_zero_X, bool NA_as_zero_U,
    real_t *restrict B, int_t n, int_t ldb,
    real_t *restrict C, int_t p,
    bool add_implicit_features,
    real_t *restrict Xones, int_t incXones,
    real_t * restrict Bi, real_t *restrict precomputedBiTBi, int_t k_main_i,
    real_t *restrict weight,
    real_t lam, real_t w_user, real_t w_implicit, real_t lam_last,
    int_t cnt_NA_x, int_t cnt_NA_u,
    real_t *restrict precomputedBtB,
    real_t *restrict precomputedCtC, /* should NOT be multiplied by 'w_user' */
    int_t max_cg_steps,
    real_t *restrict bias_BtX, real_t *restrict bias_X, real_t bias_X_glob,
    real_t *restrict bias_CtU,
    real_t *restrict buffer_real_t
);
void collective_block_cg_implicit
(
    real_t *restrict a_vec,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    int_t u_vec_ixB[], real_t *restrict u_vec_sp, size_t nnz_u_vec,
    real_t *restrict u_vec,
    bool NA_as_zero_U,
    real_t *restrict B, int_t n,
    real_t *restrict C, int_t p,
    real_t lam, real_t w_user,
    int_t cnt_NA_u,
    int_t max_cg_steps,
    real_t *restrict bias_CtU,
    real_t *restrict precomputedBtB,
    real_t *restrict precomputedCtC, /* should NOT be multiplied by weight */
    real_t *restrict buffer_real_t
);
void optimizeA_collective_implicit
(
    real_t *restrict A, real_t *restrict B, real_t *restrict C,
    int_t m, int_t m_u, int_t n, int_t p,
    int_t k, int_t k_main, int_t k_user, int_t k_item,
    size_t Xcsr_p[], int_t Xcsr_i[], real_t *restrict Xcsr,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict U, int_t cnt_NA_u[], real_t *restrict U_colmeans,
    bool full_dense_u, bool near_dense_u, bool NA_as_zero_U,
    real_t lam, real_t l1_lam, real_t w_user,
    int nthreads,
    bool use_cg, int_t max_cg_steps, bool is_first_iter,
    bool nonneg, int_t max_cd_steps,
    real_t *restrict precomputedBtB, /* will not have lambda with CG */
    real_t *restrict precomputedBeTBe,
    real_t *restrict precomputedBeTBeChol,
    real_t *restrict precomputedCtC,
    real_t *restrict precomputedCtUbias,
    bool *filled_BeTBe,
    bool *filled_BeTBeChol,
    bool *filled_CtC,
    bool *filled_CtUbias,
    real_t *restrict buffer_real_t
);
int_t collective_factors_cold
(
    real_t *restrict a_vec,
    real_t *restrict u_vec, int_t p,
    real_t *restrict u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
    real_t *restrict u_bin_vec, int_t pbin,
    real_t *restrict C, real_t *restrict Cb,
    real_t *restrict TransCtCinvCt,
    real_t *restrict CtCw,
    real_t *restrict col_means,
    real_t *restrict CtUbias,
    int_t k, int_t k_user, int_t k_main,
    real_t lam, real_t l1_lam, real_t w_main, real_t w_user,
    bool scale_lam_sideinfo,
    bool NA_as_zero_U,
    bool nonneg
);
int_t collective_factors_cold_implicit
(
    real_t *restrict a_vec,
    real_t *restrict u_vec, int_t p,
    real_t *restrict u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
    real_t *restrict B, int_t n,
    real_t *restrict C,
    real_t *restrict BeTBe,
    real_t *restrict BtB,
    real_t *restrict BeTBeChol,
    real_t *restrict col_means,
    real_t *restrict CtUbias,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t lam, real_t l1_lam,
    real_t w_main, real_t w_user, real_t w_main_multiplier,
    bool NA_as_zero_U,
    bool nonneg
);
int_t collective_factors_warm
(
    real_t *restrict a_vec, real_t *restrict a_bias,
    real_t *restrict u_vec, int_t p,
    real_t *restrict u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
    real_t *restrict u_bin_vec, int_t pbin,
    real_t *restrict C, real_t *restrict Cb,
    real_t glob_mean, real_t *restrict biasB,
    real_t *restrict col_means,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    real_t *restrict Xa_dense, int_t n,
    real_t *restrict weight,
    real_t *restrict B,
    real_t *restrict Bi, bool add_implicit_features,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t lam, real_t w_main, real_t w_user, real_t w_implicit,real_t lam_bias,
    real_t l1_lam, real_t l1_lam_bias,
    bool scale_lam, bool scale_lam_sideinfo, bool scale_bias_const,
    int_t n_max, bool include_all_X,
    real_t *restrict TransBtBinvBt,
    real_t *restrict BtXbias,
    real_t *restrict BtB,
    real_t *restrict BeTBeChol,
    real_t *restrict BiTBi,
    real_t *restrict CtCw,
    real_t *restrict CtUbias,
    bool NA_as_zero_U, bool NA_as_zero_X,
    bool nonneg,
    real_t *restrict B_plus_bias
);
int_t collective_factors_warm_implicit
(
    real_t *restrict a_vec,
    real_t *restrict u_vec, int_t p,
    real_t *restrict u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
    bool NA_as_zero_U,
    bool nonneg,
    real_t *restrict col_means,
    real_t *restrict B, int_t n, real_t *restrict C,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t lam, real_t l1_lam, real_t alpha, real_t w_main, real_t w_user,
    real_t w_main_multiplier,
    real_t *restrict BeTBe,
    real_t *restrict BtB,
    real_t *restrict BeTBeChol,
    real_t *restrict CtUbias
);
real_t fun_grad_A_collective
(
    real_t *restrict A, real_t *restrict g_A,
    real_t *restrict B, real_t *restrict C,
    int_t m, int_t m_u, int_t n, int_t p,
    int_t k, int_t k_main, int_t k_user, int_t k_item, int_t padding,
    real_t *restrict Xfull, bool full_dense,
    size_t Xcsr_p[], int_t Xcsr_i[], real_t *restrict Xcsr,
    real_t *restrict weight,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict U, bool full_dense_u,
    real_t lam, real_t w_main, real_t w_user, real_t lam_last,
    bool do_B,
    int nthreads,
    real_t *restrict buffer_real_t
);
typedef struct data_fun_grad_Adense_col {
    real_t *B; real_t *C;
    int_t m; int_t m_u; int_t n; int_t p;
    int_t k; int_t k_main; int_t k_user; int_t k_item; int_t padding;
    real_t *Xfull; bool full_dense;
    size_t *Xcsr_p; int_t *Xcsr_i; real_t *Xcsr;
    real_t *weight;
    size_t *U_csr_p; int_t *U_csr_i; real_t *U_csr;
    real_t *U; bool full_dense_u;
    real_t lam; real_t w_main; real_t w_user; real_t lam_last;
    bool do_B;
    int nthreads;
    real_t *buffer_real_t;
} data_fun_grad_Adense_col;
real_t wrapper_fun_grad_Adense_col
(
    void *instance,
    real_t *x,
    real_t *g,
    const size_t n,
    const real_t step
);
size_t buffer_size_optimizeA_collective
(
    size_t m, size_t m_u, size_t n, size_t p,
    size_t k, size_t k_main, size_t k_user,
    bool full_dense, bool near_dense, bool do_B,
    bool has_dense, bool has_sparse, bool has_weights, bool NA_as_zero_X,
    bool has_dense_U, bool has_sparse_U,
    bool full_dense_u, bool near_dense_u, bool NA_as_zero_U,
    bool add_implicit_features, size_t k_main_i,
    size_t nthreads,
    bool use_cg, bool finalize_chol,
    bool nonneg, bool has_l1,
    bool keep_precomputed,
    bool pass_allocated_BtB,
    bool pass_allocated_CtCw,
    bool pass_allocated_BeTBeChol,
    bool pass_allocated_BiTBi
);
size_t buffer_size_optimizeA_collective_implicit
(
    size_t m, size_t m_u, size_t p,
    size_t k, size_t k_main, size_t k_user,
    bool has_sparse_U,
    bool NA_as_zero_U,
    size_t nthreads,
    bool use_cg,
    bool nonneg, bool has_l1,
    bool pass_allocated_BtB,
    bool pass_allocated_BeTBe,
    bool pass_allocated_BeTBeChol,
    bool pass_allocated_CtC,
    bool finalize_chol
);
void optimizeA_collective
(
    real_t *restrict A, int_t lda, real_t *restrict B, int_t ldb,
    real_t *restrict C,
    real_t *restrict Bi,
    int_t m, int_t m_u, int_t n, int_t p,
    int_t k, int_t k_main, int_t k_user, int_t k_item,
    size_t Xcsr_p[], int_t Xcsr_i[], real_t *restrict Xcsr,
    real_t *restrict Xfull, bool full_dense, bool near_dense, int_t ldX,
    int_t cnt_NA_x[], real_t *restrict weight, bool NA_as_zero_X,
    real_t *restrict Xones, int_t k_main_i, int_t ldXones,
    bool add_implicit_features,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict U, int_t cnt_NA_u[], real_t *restrict U_colmeans,
    bool full_dense_u, bool near_dense_u, bool NA_as_zero_U,
    real_t lam, real_t w_user, real_t w_implicit, real_t lam_last,
    real_t l1_lam, real_t l1_lam_bias,
    bool scale_lam, bool scale_lam_sideinfo,
    bool scale_bias_const, real_t *restrict wsumA,
    bool do_B,
    int nthreads,
    bool use_cg, int_t max_cg_steps, bool is_first_iter,
    bool nonneg, int_t max_cd_steps,
    real_t *restrict bias_restore,
    real_t *restrict bias_BtX, real_t *restrict bias_X, real_t bias_X_glob,
    bool keep_precomputed,
    real_t *restrict precomputedBtB,
    real_t *restrict precomputedCtCw,
    real_t *restrict precomputedBeTBeChol,
    real_t *restrict precomputedBiTBi,
    real_t *restrict precomputedCtUbias,
    bool *filled_BtB, bool *filled_CtCw,
    bool *filled_BeTBeChol, bool *filled_CtUbias,
    bool *CtC_is_scaled,
    real_t *restrict buffer_real_t
);
void build_BeTBe
(
    real_t *restrict bufferBeTBe,
    real_t *restrict B, int_t ldb, real_t *restrict C,
    int_t k, int_t k_user, int_t k_main, int_t k_item,
    int_t n, int_t p,
    real_t lam, real_t w_user
);
void build_BtB_CtC
(
    real_t *restrict BtB, real_t *restrict CtC,
    real_t *restrict B, int_t n, int_t ldb,
    real_t *restrict C, int_t p,
    int_t k, int_t k_user, int_t k_main, int_t k_item,
    real_t w_user,
    real_t *restrict weight
);
void build_XBw
(
    real_t *restrict A, int_t lda,
    real_t *restrict B, int_t ldb,
    real_t *restrict Xfull, int_t ldX,
    int_t m, int_t n, int_t k,
    real_t w,
    bool do_B, bool overwrite
);
void preprocess_vec
(
    real_t *restrict vec_full, int_t n,
    int_t *restrict ix_vec, real_t *restrict vec_sp, size_t nnz,
    real_t glob_mean, real_t lam,
    real_t *restrict col_means,
    real_t *restrict vec_mean,
    int_t *restrict cnt_NA
);
int_t convert_sparse_X
(
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    size_t **Xcsr_p, int_t **Xcsr_i, real_t *restrict *Xcsr,
    size_t **Xcsc_p, int_t **Xcsc_i, real_t *restrict *Xcsc,
    real_t *restrict weight, real_t *restrict *weightR, real_t *restrict *weightC,
    int_t m, int_t n, int nthreads
);
int_t preprocess_sideinfo_matrix
(
    real_t *U, int_t m_u, int_t p,
    int_t U_row[], int_t U_col[], real_t *U_sp, size_t nnz_U,
    real_t *U_colmeans, real_t *restrict *Utrans,
    size_t **U_csr_p, int_t **U_csr_i, real_t *restrict *U_csr,
    size_t **U_csc_p, int_t **U_csc_i, real_t *restrict *U_csc,
    int_t *restrict *cnt_NA_u_byrow, int_t *restrict *cnt_NA_u_bycol,
    bool *full_dense_u, bool *near_dense_u_row, bool *near_dense_u_col,
    bool NA_as_zero_U, bool nonneg, int nthreads
);
real_t wrapper_collective_fun_grad
(
    void *instance,
    real_t *x,
    real_t *g,
    const size_t n,
    const real_t step
);
typedef struct data_collective_fun_grad {
    int_t m; int_t n; int_t k;
    int_t *ixA; int_t *ixB; real_t *X; size_t nnz;
    real_t *Xfull;
    size_t *Xcsr_p; int_t *Xcsr_i; real_t *Xcsr;
    size_t *Xcsc_p; int_t *Xcsc_i; real_t *Xcsc;
    real_t *weight; real_t *weightR; real_t *weightC;
    bool user_bias; bool item_bias;
    real_t lam; real_t *lam_unique;
    real_t *U; int_t m_u; int_t p; bool U_has_NA;
    real_t *II; int_t n_i; int_t q; bool I_has_NA;
    real_t *Ub; int_t m_ubin; int_t pbin; bool Ub_has_NA;
    real_t *Ib; int_t n_ibin; int_t qbin; bool Ib_has_NA;
    int_t *U_row; int_t *U_col; real_t *U_sp; size_t nnz_U;
    int_t *I_row; int_t *I_col; real_t *I_sp; size_t nnz_I;
    size_t *U_csr_p; int_t *U_csr_i; real_t *U_csr;
    size_t *U_csc_p; int_t *U_csc_i; real_t *U_csc;
    size_t *I_csr_p; int_t *I_csr_i; real_t *I_csr;
    size_t *I_csc_p; int_t *I_csc_i; real_t *I_csc;
    real_t *buffer_real_t; real_t *buffer_mt;
    int_t k_main; int_t k_user; int_t k_item;
    real_t w_main; real_t w_user; real_t w_item;
    int nthreads;
    int_t print_every; int_t nfev; int_t niter;
} data_collective_fun_grad;
int_t fit_collective_explicit_lbfgs_internal
(
    real_t *restrict values, bool reset_values,
    real_t *restrict glob_mean,
    real_t *restrict U_colmeans, real_t *restrict I_colmeans,
    int_t m, int_t n, int_t k,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t *restrict Xfull,
    real_t *restrict weight,
    bool user_bias, bool item_bias, bool center,
    real_t lam, real_t *restrict lam_unique,
    real_t *restrict U, int_t m_u, int_t p,
    real_t *restrict II, int_t n_i, int_t q,
    real_t *restrict Ub, int_t m_ubin, int_t pbin,
    real_t *restrict Ib, int_t n_ibin, int_t qbin,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    int_t I_row[], int_t I_col[], real_t *restrict I_sp, size_t nnz_I,
    int_t k_main, int_t k_user, int_t k_item,
    real_t w_main, real_t w_user, real_t w_item,
    int_t n_corr_pairs, size_t maxiter, int_t seed,
    int nthreads, bool prefer_onepass,
    bool verbose, int_t print_every, bool handle_interrupt,
    int_t *restrict niter, int_t *restrict nfev,
    real_t *restrict B_plus_bias
);
int_t fit_collective_explicit_lbfgs
(
    real_t *restrict biasA, real_t *restrict biasB,
    real_t *restrict A, real_t *restrict B,
    real_t *restrict C, real_t *restrict Cb,
    real_t *restrict D, real_t *restrict Db,
    bool reset_values, int_t seed,
    real_t *restrict glob_mean,
    real_t *restrict U_colmeans, real_t *restrict I_colmeans,
    int_t m, int_t n, int_t k,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t *restrict Xfull,
    real_t *restrict weight,
    bool user_bias, bool item_bias, bool center,
    real_t lam, real_t *restrict lam_unique,
    real_t *restrict U, int_t m_u, int_t p,
    real_t *restrict II, int_t n_i, int_t q,
    real_t *restrict Ub, int_t m_ubin, int_t pbin,
    real_t *restrict Ib, int_t n_ibin, int_t qbin,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    int_t I_row[], int_t I_col[], real_t *restrict I_sp, size_t nnz_I,
    int_t k_main, int_t k_user, int_t k_item,
    real_t w_main, real_t w_user, real_t w_item,
    int_t n_corr_pairs, size_t maxiter,
    int nthreads, bool prefer_onepass,
    bool verbose, int_t print_every, bool handle_interrupt,
    int_t *restrict niter, int_t *restrict nfev,
    bool precompute_for_predictions,
    bool include_all_X,
    real_t *restrict B_plus_bias,
    real_t *restrict precomputedBtB,
    real_t *restrict precomputedTransBtBinvBt,
    real_t *restrict precomputedBeTBeChol,
    real_t *restrict precomputedTransCtCinvCt,
    real_t *restrict precomputedCtCw
);
int_t fit_collective_explicit_als
(
    real_t *restrict biasA, real_t *restrict biasB,
    real_t *restrict A, real_t *restrict B,
    real_t *restrict C, real_t *restrict D,
    real_t *restrict Ai, real_t *restrict Bi,
    bool add_implicit_features,
    bool reset_values, int_t seed,
    real_t *restrict glob_mean,
    real_t *restrict U_colmeans, real_t *restrict I_colmeans,
    int_t m, int_t n, int_t k,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t *restrict Xfull,
    real_t *restrict weight,
    bool user_bias, bool item_bias, bool center,
    real_t lam, real_t *restrict lam_unique,
    real_t l1_lam, real_t *restrict l1_lam_unique,
    bool scale_lam, bool scale_lam_sideinfo, bool scale_bias_const,
    real_t *restrict scaling_biasA, real_t *restrict scaling_biasB,
    real_t *restrict U, int_t m_u, int_t p,
    real_t *restrict II, int_t n_i, int_t q,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    int_t I_row[], int_t I_col[], real_t *restrict I_sp, size_t nnz_I,
    bool NA_as_zero_X, bool NA_as_zero_U, bool NA_as_zero_I,
    int_t k_main, int_t k_user, int_t k_item,
    real_t w_main, real_t w_user, real_t w_item, real_t w_implicit,
    int_t niter, int nthreads, bool verbose, bool handle_interrupt,
    bool use_cg, int_t max_cg_steps, bool finalize_chol,
    bool nonneg, int_t max_cd_steps, bool nonneg_C, bool nonneg_D,
    bool precompute_for_predictions,
    bool include_all_X,
    real_t *restrict B_plus_bias,
    real_t *restrict precomputedBtB,
    real_t *restrict precomputedTransBtBinvBt,
    real_t *restrict precomputedBtXbias,
    real_t *restrict precomputedBeTBeChol,
    real_t *restrict precomputedBiTBi,
    real_t *restrict precomputedTransCtCinvCt,
    real_t *restrict precomputedCtCw,
    real_t *restrict precomputedCtUbias
);
int_t fit_collective_implicit_als
(
    real_t *restrict A, real_t *restrict B,
    real_t *restrict C, real_t *restrict D,
    bool reset_values, int_t seed,
    real_t *restrict U_colmeans, real_t *restrict I_colmeans,
    int_t m, int_t n, int_t k,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t lam, real_t *restrict lam_unique,
    real_t l1_lam, real_t *restrict l1_lam_unique,
    real_t *restrict U, int_t m_u, int_t p,
    real_t *restrict II, int_t n_i, int_t q,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    int_t I_row[], int_t I_col[], real_t *restrict I_sp, size_t nnz_I,
    bool NA_as_zero_U, bool NA_as_zero_I,
    int_t k_main, int_t k_user, int_t k_item,
    real_t w_main, real_t w_user, real_t w_item,
    real_t *restrict w_main_multiplier,
    real_t alpha, bool adjust_weight, bool apply_log_transf,
    int_t niter, int nthreads, bool verbose, bool handle_interrupt,
    bool use_cg, int_t max_cg_steps, bool finalize_chol,
    bool nonneg, int_t max_cd_steps, bool nonneg_C, bool nonneg_D,
    bool precompute_for_predictions,
    real_t *restrict precomputedBtB,
    real_t *restrict precomputedBeTBe,
    real_t *restrict precomputedBeTBeChol,
    real_t *restrict precomputedCtUbias
);
int_t precompute_collective_explicit
(
    real_t *restrict B, int_t n, int_t n_max, bool include_all_X,
    real_t *restrict C, int_t p,
    real_t *restrict Bi, bool add_implicit_features,
    real_t *restrict biasB, real_t glob_mean, bool NA_as_zero_X,
    real_t *restrict U_colmeans, bool NA_as_zero_U,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    bool user_bias,
    bool nonneg,
    real_t lam, real_t *restrict lam_unique,
    bool scale_lam, bool scale_lam_sideinfo,
    bool scale_bias_const, real_t scaling_biasA,
    real_t w_main, real_t w_user, real_t w_implicit,
    real_t *restrict B_plus_bias,
    real_t *restrict BtB,
    real_t *restrict TransBtBinvBt,
    real_t *restrict BtXbias,
    real_t *restrict BeTBeChol,
    real_t *restrict BiTBi,
    real_t *restrict TransCtCinvCt,
    real_t *restrict CtCw,
    real_t *restrict CtUbias
);
int_t precompute_collective_implicit
(
    real_t *restrict B, int_t n,
    real_t *restrict C, int_t p,
    real_t *restrict U_colmeans, bool NA_as_zero_U,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t lam, real_t w_main, real_t w_user, real_t w_main_multiplier,
    bool nonneg,
    bool extra_precision,
    real_t *restrict BtB,
    real_t *restrict BeTBe,
    real_t *restrict BeTBeChol,
    real_t *restrict CtUbias
);
int_t factors_collective_explicit_single
(
    real_t *restrict a_vec, real_t *restrict a_bias,
    real_t *restrict u_vec, int_t p,
    real_t *restrict u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
    real_t *restrict u_bin_vec, int_t pbin,
    bool NA_as_zero_U, bool NA_as_zero_X,
    bool nonneg,
    real_t *restrict C, real_t *restrict Cb,
    real_t glob_mean, real_t *restrict biasB,
    real_t *restrict U_colmeans,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    real_t *restrict Xa_dense, int_t n,
    real_t *restrict weight,
    real_t *restrict B,
    real_t *restrict Bi, bool add_implicit_features,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t lam, real_t *restrict lam_unique,
    real_t l1_lam, real_t *restrict l1_lam_unique,
    bool scale_lam, bool scale_lam_sideinfo,
    bool scale_bias_const, real_t scaling_biasA,
    real_t w_main, real_t w_user, real_t w_implicit,
    int_t n_max, bool include_all_X,
    real_t *restrict BtB,
    real_t *restrict TransBtBinvBt,
    real_t *restrict BtXbias,
    real_t *restrict BeTBeChol,
    real_t *restrict BiTBi,
    real_t *restrict CtCw,
    real_t *restrict TransCtCinvCt,
    real_t *restrict CtUbias,
    real_t *restrict B_plus_bias
);
int_t factors_collective_implicit_single
(
    real_t *restrict a_vec,
    real_t *restrict u_vec, int_t p,
    real_t *restrict u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
    bool NA_as_zero_U,
    bool nonneg,
    real_t *restrict U_colmeans,
    real_t *restrict B, int_t n, real_t *restrict C,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t lam, real_t l1_lam, real_t alpha, real_t w_main, real_t w_user,
    real_t w_main_multiplier,
    bool apply_log_transf,
    real_t *restrict BeTBe,
    real_t *restrict BtB,
    real_t *restrict BeTBeChol,
    real_t *restrict CtUbias
);
int_t factors_collective_explicit_multiple
(
    real_t *restrict A, real_t *restrict biasA, int_t m,
    real_t *restrict U, int_t m_u, int_t p,
    bool NA_as_zero_U, bool NA_as_zero_X,
    bool nonneg,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict Ub, int_t m_ubin, int_t pbin,
    real_t *restrict C, real_t *restrict Cb,
    real_t glob_mean, real_t *restrict biasB,
    real_t *restrict U_colmeans,
    real_t *restrict X, int_t ixA[], int_t ixB[], size_t nnz,
    size_t *restrict Xcsr_p, int_t *restrict Xcsr_i, real_t *restrict Xcsr,
    real_t *restrict Xfull, int_t n,
    real_t *restrict weight,
    real_t *restrict B,
    real_t *restrict Bi, bool add_implicit_features,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t lam, real_t *restrict lam_unique,
    real_t l1_lam, real_t *restrict l1_lam_unique,
    bool scale_lam, bool scale_lam_sideinfo,
    bool scale_bias_const, real_t scaling_biasA,
    real_t w_main, real_t w_user, real_t w_implicit,
    int_t n_max, bool include_all_X,
    real_t *restrict BtB,
    real_t *restrict TransBtBinvBt,
    real_t *restrict BtXbias,
    real_t *restrict BeTBeChol,
    real_t *restrict BiTBi,
    real_t *restrict TransCtCinvCt,
    real_t *restrict CtCw,
    real_t *restrict CtUbias,
    real_t *restrict B_plus_bias,
    int nthreads
);
int_t factors_collective_implicit_multiple
(
    real_t *restrict A, int_t m,
    real_t *restrict U, int_t m_u, int_t p,
    bool NA_as_zero_U,
    bool nonneg,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict X, int_t ixA[], int_t ixB[], size_t nnz,
    size_t *restrict Xcsr_p, int_t *restrict Xcsr_i, real_t *restrict Xcsr,
    real_t *restrict B, int_t n,
    real_t *restrict C,
    real_t *restrict U_colmeans,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t lam, real_t l1_lam, real_t alpha, real_t w_main, real_t w_user,
    real_t w_main_multiplier,
    bool apply_log_transf,
    real_t *restrict BeTBe,
    real_t *restrict BtB,
    real_t *restrict BeTBeChol,
    real_t *restrict CtUbias,
    int nthreads
);
int_t impute_X_collective_explicit
(
    int_t m, bool user_bias,
    real_t *restrict U, int_t m_u, int_t p,
    bool NA_as_zero_U,
    bool nonneg,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict Ub, int_t m_ubin, int_t pbin,
    real_t *restrict C, real_t *restrict Cb,
    real_t glob_mean, real_t *restrict biasB,
    real_t *restrict U_colmeans,
    real_t *restrict Xfull, int_t n,
    real_t *restrict weight,
    real_t *restrict B,
    real_t *restrict Bi, bool add_implicit_features,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t lam, real_t *restrict lam_unique,
    real_t l1_lam, real_t *restrict l1_lam_unique,
    bool scale_lam, bool scale_lam_sideinfo,
    bool scale_bias_const, real_t scaling_biasA,
    real_t w_main, real_t w_user, real_t w_implicit,
    int_t n_max, bool include_all_X,
    real_t *restrict BtB,
    real_t *restrict TransBtBinvBt,
    real_t *restrict BeTBeChol,
    real_t *restrict BiTBi,
    real_t *restrict TransCtCinvCt,
    real_t *restrict CtCw,
    real_t *restrict CtUbias,
    real_t *restrict B_plus_bias,
    int nthreads
);
int_t topN_old_collective_explicit
(
    real_t *restrict a_vec, real_t a_bias,
    real_t *restrict A, real_t *restrict biasA, int_t row_index,
    real_t *restrict B,
    real_t *restrict biasB,
    real_t glob_mean,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    int_t *restrict include_ix, int_t n_include,
    int_t *restrict exclude_ix, int_t n_exclude,
    int_t *restrict outp_ix, real_t *restrict outp_score,
    int_t n_top, int_t n, int_t n_max, bool include_all_X, int nthreads
);
int_t topN_old_collective_implicit
(
    real_t *restrict a_vec,
    real_t *restrict A, int_t row_index,
    real_t *restrict B,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    int_t *restrict include_ix, int_t n_include,
    int_t *restrict exclude_ix, int_t n_exclude,
    int_t *restrict outp_ix, real_t *restrict outp_score,
    int_t n_top, int_t n, int nthreads
);
int_t topN_new_collective_explicit
(
    /* inputs for the factors */
    bool user_bias,
    real_t *restrict u_vec, int_t p,
    real_t *restrict u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
    real_t *restrict u_bin_vec, int_t pbin,
    bool NA_as_zero_U, bool NA_as_zero_X,
    bool nonneg,
    real_t *restrict C, real_t *restrict Cb,
    real_t glob_mean, real_t *restrict biasB,
    real_t *restrict U_colmeans,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    real_t *restrict Xa_dense, int_t n,
    real_t *restrict weight,
    real_t *restrict B,
    real_t *restrict Bi, bool add_implicit_features,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t lam, real_t *restrict lam_unique,
    real_t l1_lam, real_t *restrict l1_lam_unique,
    bool scale_lam, bool scale_lam_sideinfo,
    bool scale_bias_const, real_t scaling_biasA,
    real_t w_main, real_t w_user, real_t w_implicit,
    int_t n_max, bool include_all_X,
    real_t *restrict BtB,
    real_t *restrict TransBtBinvBt,
    real_t *restrict BtXbias,
    real_t *restrict BeTBeChol,
    real_t *restrict BiTBi,
    real_t *restrict CtCw,
    real_t *restrict TransCtCinvCt,
    real_t *restrict CtUbias,
    real_t *restrict B_plus_bias,
    /* inputs for topN */
    int_t *restrict include_ix, int_t n_include,
    int_t *restrict exclude_ix, int_t n_exclude,
    int_t *restrict outp_ix, real_t *restrict outp_score,
    int_t n_top, int nthreads
);
int_t topN_new_collective_implicit
(
    /* inputs for the factors */
    int_t n,
    real_t *restrict u_vec, int_t p,
    real_t *restrict u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
    bool NA_as_zero_U,
    bool nonneg,
    real_t *restrict U_colmeans,
    real_t *restrict B, real_t *restrict C,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t lam, real_t l1_lam, real_t alpha, real_t w_main, real_t w_user,
    real_t w_main_multiplier,
    bool apply_log_transf,
    real_t *restrict BeTBe,
    real_t *restrict BtB,
    real_t *restrict BeTBeChol,
    real_t *restrict CtUbias,
    /* inputs for topN */
    int_t *restrict include_ix, int_t n_include,
    int_t *restrict exclude_ix, int_t n_exclude,
    int_t *restrict outp_ix, real_t *restrict outp_score,
    int_t n_top, int nthreads
);
int_t predict_X_old_collective_explicit
(
    int_t row[], int_t col[], real_t *restrict predicted, size_t n_predict,
    real_t *restrict A, real_t *restrict biasA,
    real_t *restrict B, real_t *restrict biasB,
    real_t glob_mean,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    int_t m, int_t n_max,
    int nthreads
);
int_t predict_X_old_collective_implicit
(
    int_t row[], int_t col[], real_t *restrict predicted, size_t n_predict,
    real_t *restrict A,
    real_t *restrict B,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    int_t m, int_t n,
    int nthreads
);
int_t predict_X_new_collective_explicit
(
    /* inputs for predictions */
    int_t m_new,
    int_t row[], int_t col[], real_t *restrict predicted, size_t n_predict,
    int nthreads,
    /* inputs for factors */
    bool user_bias,
    real_t *restrict U, int_t m_u, int_t p,
    bool NA_as_zero_U, bool NA_as_zero_X,
    bool nonneg,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict Ub, int_t m_ubin, int_t pbin,
    real_t *restrict C, real_t *restrict Cb,
    real_t glob_mean, real_t *restrict biasB,
    real_t *restrict U_colmeans,
    real_t *restrict X, int_t ixA[], int_t ixB[], size_t nnz,
    size_t *restrict Xcsr_p, int_t *restrict Xcsr_i, real_t *restrict Xcsr,
    real_t *restrict Xfull, int_t n,
    real_t *restrict weight,
    real_t *restrict B,
    real_t *restrict Bi, bool add_implicit_features,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t lam, real_t *restrict lam_unique,
    real_t l1_lam, real_t *restrict l1_lam_unique,
    bool scale_lam, bool scale_lam_sideinfo,
    bool scale_bias_const, real_t scaling_biasA,
    real_t w_main, real_t w_user, real_t w_implicit,
    int_t n_max, bool include_all_X,
    real_t *restrict BtB,
    real_t *restrict TransBtBinvBt,
    real_t *restrict BtXbias,
    real_t *restrict BeTBeChol,
    real_t *restrict BiTBi,
    real_t *restrict TransCtCinvCt,
    real_t *restrict CtCw,
    real_t *restrict CtUbias,
    real_t *restrict B_plus_bias
);
int_t predict_X_new_collective_implicit
(
    /* inputs for predictions */
    int_t m_new,
    int_t row[], int_t col[], real_t *restrict predicted, size_t n_predict,
    int nthreads,
    /* inputs for factors */
    real_t *restrict U, int_t m_u, int_t p,
    bool NA_as_zero_U,
    bool nonneg,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict X, int_t ixA[], int_t ixB[], size_t nnz,
    size_t *restrict Xcsr_p, int_t *restrict Xcsr_i, real_t *restrict Xcsr,
    real_t *restrict B, int_t n,
    real_t *restrict C,
    real_t *restrict U_colmeans,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t lam, real_t l1_lam, real_t alpha, real_t w_main, real_t w_user,
    real_t w_main_multiplier,
    bool apply_log_transf,
    real_t *restrict BeTBe,
    real_t *restrict BtB,
    real_t *restrict BeTBeChol,
    real_t *restrict CtUbias
);

/* offsets.c */
real_t offsets_fun_grad
(
    real_t *restrict values, real_t *restrict grad,
    int_t ixA[], int_t ixB[], real_t *restrict X,
    size_t nnz, int_t m, int_t n, int_t k,
    real_t *restrict Xfull, bool full_dense,
    size_t Xcsr_p[], int_t Xcsr_i[], real_t *restrict Xcsr,
    size_t Xcsc_p[], int_t Xcsc_i[], real_t *restrict Xcsc,
    real_t *restrict weight, real_t *restrict weightR, real_t *restrict weightC,
    bool user_bias, bool item_bias,
    bool add_intercepts,
    real_t lam, real_t *restrict lam_unique,
    real_t *restrict U, int_t p,
    real_t *restrict II, int_t q,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    size_t U_csc_p[], int_t U_csc_i[], real_t *restrict U_csc,
    size_t I_csr_p[], int_t I_csr_i[], real_t *restrict I_csr,
    size_t I_csc_p[], int_t I_csc_i[], real_t *restrict I_csc,
    int_t k_main, int_t k_sec,
    real_t w_user, real_t w_item,
    int nthreads,
    real_t *restrict buffer_real_t,
    real_t *restrict buffer_mt
);
void construct_Am
(
    real_t *restrict Am, real_t *restrict A,
    real_t *restrict C, real_t *restrict C_bias,
    bool add_intercepts,
    real_t *restrict U, int_t m, int_t p,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    int_t k, int_t k_sec, int_t k_main,
    real_t w_user, int nthreads
);
void assign_gradients
(
    real_t *restrict bufferA, real_t *restrict g_A, real_t *restrict g_C,
    bool add_intercepts, real_t *restrict g_C_bias,
    real_t *restrict U,
    size_t U_csc_p[], int_t U_csc_i[], real_t *restrict U_csc,
    int_t m, int_t p, int_t k, int_t k_sec, int_t k_main,
    real_t w_user, int nthreads
);
int_t offsets_factors_cold
(
    real_t *restrict a_vec,
    real_t *restrict u_vec,
    int_t u_vec_ixB[], real_t *restrict u_vec_sp, size_t nnz_u_vec,
    real_t *restrict C, int_t p,
    real_t *restrict C_bias,
    int_t k, int_t k_sec, int_t k_main,
    real_t w_user
);
int_t offsets_factors_warm
(
    real_t *restrict a_vec, real_t *restrict a_bias,
    real_t *restrict u_vec,
    int_t u_vec_ixB[], real_t *restrict u_vec_sp, size_t nnz_u_vec,
    int_t ixB[], real_t *restrict Xa, size_t nnz,
    real_t *restrict Xa_dense, int_t n,
    real_t *restrict weight,
    real_t *restrict Bm, real_t *restrict C,
    real_t *restrict C_bias,
    real_t glob_mean, real_t *restrict biasB,
    int_t k, int_t k_sec, int_t k_main,
    int_t p, real_t w_user,
    real_t lam, bool exact, real_t lam_bias,
    bool implicit, real_t alpha,
    real_t *restrict precomputedTransBtBinvBt,
    real_t *restrict precomputedBtBw,
    real_t *restrict output_a,
    real_t *restrict Bm_plus_bias
);
int_t precompute_offsets_both
(
    real_t *restrict A, int_t m,
    real_t *restrict B, int_t n,
    real_t *restrict C, int_t p,
    real_t *restrict D, int_t q,
    real_t *restrict C_bias, real_t *restrict D_bias,
    real_t *restrict biasB, real_t glob_mean, bool NA_as_zero_X,
    bool user_bias, bool add_intercepts, bool implicit,
    int_t k, int_t k_main, int_t k_sec,
    real_t lam, real_t *restrict lam_unique,
    real_t w_user, real_t w_item, 
    real_t *restrict U,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict II,
    size_t I_csr_p[], int_t I_csr_i[], real_t *restrict I_csr,
    int_t I_row[], int_t I_col[], real_t *restrict I_sp, size_t nnz_I,
    real_t *restrict Am,
    real_t *restrict Bm,
    real_t *restrict Bm_plus_bias,
    real_t *restrict BtB,
    real_t *restrict TransBtBinvBt
);
int_t precompute_offsets_explicit
(
    real_t *restrict A, int_t m,
    real_t *restrict B, int_t n,
    real_t *restrict C, int_t p,
    real_t *restrict D, int_t q,
    real_t *restrict C_bias, real_t *restrict D_bias,
    bool user_bias, bool add_intercepts,
    int_t k, int_t k_main, int_t k_sec,
    real_t lam, real_t *restrict lam_unique,
    real_t w_user, real_t w_item, 
    real_t *restrict U,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict II,
    size_t I_csr_p[], int_t I_csr_i[], real_t *restrict I_csr,
    int_t I_row[], int_t I_col[], real_t *restrict I_sp, size_t nnz_I,
    real_t *restrict Am,
    real_t *restrict Bm,
    real_t *restrict Bm_plus_bias,
    real_t *restrict BtB,
    real_t *restrict TransBtBinvBt
);
int_t precompute_offsets_implicit
(
    real_t *restrict A, int_t m,
    real_t *restrict B, int_t n,
    real_t *restrict C, int_t p,
    real_t *restrict D, int_t q,
    real_t *restrict C_bias, real_t *restrict D_bias,
    bool add_intercepts,
    int_t k,
    real_t lam,
    real_t *restrict U,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict II,
    size_t I_csr_p[], int_t I_csr_i[], real_t *restrict I_csr,
    int_t I_row[], int_t I_col[], real_t *restrict I_sp, size_t nnz_I,
    real_t *restrict Am,
    real_t *restrict Bm,
    real_t *restrict BtB
);
typedef struct data_offsets_fun_grad {
    int_t *ixA; int_t *ixB; real_t *X;
    size_t nnz; int_t m; int_t n; int_t k;
    real_t *Xfull; bool full_dense;
    size_t *Xcsr_p; int_t *Xcsr_i; real_t *Xcsr;
    size_t *Xcsc_p; int_t *Xcsc_i; real_t *Xcsc;
    real_t *weight; real_t *weightR; real_t *weightC;
    bool user_bias; bool item_bias;
    bool add_intercepts;
    real_t lam; real_t *lam_unique;
    real_t *U; int_t p;
    real_t *II; int_t q;
    size_t *U_csr_p; int_t *U_csr_i; real_t *U_csr;
    size_t *U_csc_p; int_t *U_csc_i; real_t *U_csc;
    size_t *I_csr_p; int_t *I_csr_i; real_t *I_csr;
    size_t *I_csc_p; int_t *I_csc_i; real_t *I_csc;
    int_t k_main; int_t k_sec;
    real_t w_user; real_t w_item;
    int nthreads;
    real_t *buffer_real_t;
    real_t *buffer_mt;
    int_t print_every; int_t nfev; int_t niter;
} data_offsets_fun_grad;
real_t wrapper_offsets_fun_grad
(
    void *instance,
    real_t *x,
    real_t *g,
    const size_t n,
    const real_t step
);
int_t fit_offsets_explicit_lbfgs_internal
(
    real_t *restrict values, bool reset_values,
    real_t *restrict glob_mean,
    int_t m, int_t n, int_t k,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t *restrict Xfull,
    real_t *restrict weight,
    bool user_bias, bool item_bias, bool center,
    bool add_intercepts,
    real_t lam, real_t *restrict lam_unique,
    real_t *restrict U, int_t p,
    real_t *restrict II, int_t q,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    int_t I_row[], int_t I_col[], real_t *restrict I_sp, size_t nnz_I,
    int_t k_main, int_t k_sec,
    real_t w_user, real_t w_item,
    int_t n_corr_pairs, size_t maxiter, int_t seed,
    int nthreads, bool prefer_onepass,
    bool verbose, int_t print_every, bool handle_interrupt,
    int_t *restrict niter, int_t *restrict nfev,
    real_t *restrict Am, real_t *restrict Bm,
    real_t *restrict Bm_plus_bias
);
int_t fit_offsets_explicit_lbfgs
(
    real_t *restrict biasA, real_t *restrict biasB,
    real_t *restrict A, real_t *restrict B,
    real_t *restrict C, real_t *restrict C_bias,
    real_t *restrict D, real_t *restrict D_bias,
    bool reset_values, int_t seed,
    real_t *restrict glob_mean,
    int_t m, int_t n, int_t k,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t *restrict Xfull,
    real_t *restrict weight,
    bool user_bias, bool item_bias, bool center,
    bool add_intercepts,
    real_t lam, real_t *restrict lam_unique,
    real_t *restrict U, int_t p,
    real_t *restrict II, int_t q,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    int_t I_row[], int_t I_col[], real_t *restrict I_sp, size_t nnz_I,
    int_t k_main, int_t k_sec,
    real_t w_user, real_t w_item,
    int_t n_corr_pairs, size_t maxiter,
    int nthreads, bool prefer_onepass,
    bool verbose, int_t print_every, bool handle_interrupt,
    int_t *restrict niter, int_t *restrict nfev,
    bool precompute_for_predictions,
    real_t *restrict Am, real_t *restrict Bm,
    real_t *restrict Bm_plus_bias,
    real_t *restrict precomputedBtB,
    real_t *restrict precomputedTransBtBinvBt
);
int_t fit_offsets_als
(
    real_t *restrict biasA, real_t *restrict biasB,
    real_t *restrict A, real_t *restrict B,
    real_t *restrict C, real_t *restrict C_bias,
    real_t *restrict D, real_t *restrict D_bias,
    bool reset_values, int_t seed,
    real_t *restrict glob_mean,
    int_t m, int_t n, int_t k,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t *restrict Xfull,
    real_t *restrict weight,
    bool user_bias, bool item_bias, bool center, bool add_intercepts,
    real_t lam,
    real_t *restrict U, int_t p,
    real_t *restrict II, int_t q,
    bool implicit, bool NA_as_zero_X,
    real_t alpha, bool apply_log_transf,
    int_t niter,
    int nthreads, bool use_cg,
    int_t max_cg_steps, bool finalize_chol,
    bool verbose, bool handle_interrupt,
    bool precompute_for_predictions,
    real_t *restrict Am, real_t *restrict Bm,
    real_t *restrict Bm_plus_bias,
    real_t *restrict precomputedBtB,
    real_t *restrict precomputedTransBtBinvBt
);
int_t fit_offsets_explicit_als
(
    real_t *restrict biasA, real_t *restrict biasB,
    real_t *restrict A, real_t *restrict B,
    real_t *restrict C, real_t *restrict C_bias,
    real_t *restrict D, real_t *restrict D_bias,
    bool reset_values, int_t seed,
    real_t *restrict glob_mean,
    int_t m, int_t n, int_t k,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t *restrict Xfull,
    real_t *restrict weight,
    bool user_bias, bool item_bias, bool center, bool add_intercepts,
    real_t lam,
    real_t *restrict U, int_t p,
    real_t *restrict II, int_t q,
    bool NA_as_zero_X,
    int_t niter,
    int nthreads, bool use_cg,
    int_t max_cg_steps, bool finalize_chol,
    bool verbose, bool handle_interrupt,
    bool precompute_for_predictions,
    real_t *restrict Am, real_t *restrict Bm,
    real_t *restrict Bm_plus_bias,
    real_t *restrict precomputedBtB,
    real_t *restrict precomputedTransBtBinvBt
);
int_t fit_offsets_implicit_als
(
    real_t *restrict A, real_t *restrict B,
    real_t *restrict C, real_t *restrict C_bias,
    real_t *restrict D, real_t *restrict D_bias,
    bool reset_values, int_t seed,
    int_t m, int_t n, int_t k,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    bool add_intercepts,
    real_t lam,
    real_t *restrict U, int_t p,
    real_t *restrict II, int_t q,
    real_t alpha, bool apply_log_transf,
    int_t niter,
    int nthreads, bool use_cg,
    int_t max_cg_steps, bool finalize_chol,
    bool verbose, bool handle_interrupt,
    bool precompute_for_predictions,
    real_t *restrict Am, real_t *restrict Bm,
    real_t *restrict precomputedBtB
);
int_t matrix_content_based
(
    real_t *restrict Am_new,
    int_t m_new, int_t k,
    real_t *restrict U, int_t p,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict C, real_t *restrict C_bias,
    int nthreads
);
int_t factors_offsets_explicit_single
(
    real_t *restrict a_vec, real_t *restrict a_bias, real_t *restrict output_a,
    real_t *restrict u_vec, int_t p,
    real_t *restrict u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    real_t *restrict Xa_dense, int_t n,
    real_t *restrict weight,
    real_t *restrict Bm, real_t *restrict C,
    real_t *restrict C_bias,
    real_t glob_mean, real_t *restrict biasB,
    int_t k, int_t k_sec, int_t k_main,
    real_t w_user,
    real_t lam, real_t *restrict lam_unique,
    bool exact,
    real_t *restrict precomputedTransBtBinvBt,
    real_t *restrict precomputedBtB,
    real_t *restrict Bm_plus_bias
);
int_t factors_offsets_implicit_single
(
    real_t *restrict a_vec,
    real_t *restrict u_vec, int_t p,
    real_t *restrict u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    real_t *restrict Bm, real_t *restrict C,
    real_t *restrict C_bias,
    int_t k, int_t n,
    real_t lam, real_t alpha,
    bool apply_log_transf,
    real_t *restrict precomputedBtB,
    real_t *restrict output_a
);
int_t factors_offsets_explicit_multiple
(
    real_t *restrict Am, real_t *restrict biasA,
    real_t *restrict A, int_t m,
    real_t *restrict U, int_t p,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict X, int_t ixA[], int_t ixB[], size_t nnz,
    size_t *restrict Xcsr_p, int_t *restrict Xcsr_i, real_t *restrict Xcsr,
    real_t *restrict Xfull, int_t n,
    real_t *restrict weight,
    real_t *restrict Bm, real_t *restrict C,
    real_t *restrict C_bias,
    real_t glob_mean, real_t *restrict biasB,
    int_t k, int_t k_sec, int_t k_main,
    real_t w_user,
    real_t lam, real_t *restrict lam_unique, bool exact,
    real_t *restrict precomputedTransBtBinvBt,
    real_t *restrict precomputedBtB,
    real_t *restrict Bm_plus_bias,
    int nthreads
);
int_t factors_offsets_implicit_multiple
(
    real_t *restrict Am, int_t m,
    real_t *restrict A,
    real_t *restrict U, int_t p,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict X, int_t ixA[], int_t ixB[], size_t nnz,
    size_t *restrict Xcsr_p, int_t *restrict Xcsr_i, real_t *restrict Xcsr,
    real_t *restrict Bm, real_t *restrict C,
    real_t *restrict C_bias,
    int_t k, int_t n,
    real_t lam, real_t alpha,
    bool apply_log_transf,
    real_t *restrict precomputedBtB,
    int nthreads
);
int_t topN_old_offsets_explicit
(
    real_t *restrict a_vec, real_t a_bias,
    real_t *restrict Am, real_t *restrict biasA, int_t row_index,
    real_t *restrict Bm,
    real_t *restrict biasB,
    real_t glob_mean,
    int_t k, int_t k_sec, int_t k_main,
    int_t *restrict include_ix, int_t n_include,
    int_t *restrict exclude_ix, int_t n_exclude,
    int_t *restrict outp_ix, real_t *restrict outp_score,
    int_t n_top, int_t n, int nthreads
);
int_t topN_old_offsets_implicit
(
    real_t *restrict a_vec,
    real_t *restrict Am, int_t row_index,
    real_t *restrict Bm,
    int_t k,
    int_t *restrict include_ix, int_t n_include,
    int_t *restrict exclude_ix, int_t n_exclude,
    int_t *restrict outp_ix, real_t *restrict outp_score,
    int_t n_top, int_t n, int nthreads
);
int_t topN_new_offsets_explicit
(
    /* inputs for factors */
    bool user_bias, int_t n,
    real_t *restrict u_vec, int_t p,
    real_t *restrict u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    real_t *restrict Xa_dense,
    real_t *restrict weight,
    real_t *restrict Bm, real_t *restrict C,
    real_t *restrict C_bias,
    real_t glob_mean, real_t *restrict biasB,
    int_t k, int_t k_sec, int_t k_main,
    real_t w_user,
    real_t lam, real_t *restrict lam_unique,
    bool exact,
    real_t *restrict precomputedTransBtBinvBt,
    real_t *restrict precomputedBtB,
    real_t *restrict Bm_plus_bias,
    /* inputs for topN */
    int_t *restrict include_ix, int_t n_include,
    int_t *restrict exclude_ix, int_t n_exclude,
    int_t *restrict outp_ix, real_t *restrict outp_score,
    int_t n_top, int nthreads
);
int_t topN_new_offsets_implicit
(
    /* inputs for factors */
    real_t *restrict u_vec, int_t p,
    real_t *restrict u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    real_t *restrict Bm, real_t *restrict C,
    real_t *restrict C_bias,
    int_t k,
    real_t lam, real_t alpha,
    bool apply_log_transf,
    real_t *restrict precomputedBtB,
    /* inputs for topN */
    int_t *restrict include_ix, int_t n_include,
    int_t *restrict exclude_ix, int_t n_exclude,
    int_t *restrict outp_ix, real_t *restrict outp_score,
    int_t n_top, int_t n, int nthreads
);
int_t predict_X_old_offsets_explicit
(
    int_t row[], int_t col[], real_t *restrict predicted, size_t n_predict,
    real_t *restrict Am, real_t *restrict biasA,
    real_t *restrict Bm, real_t *restrict biasB,
    real_t glob_mean,
    int_t k, int_t k_sec, int_t k_main,
    int_t m, int_t n,
    int nthreads
);
int_t predict_X_old_offsets_implicit
(
    int_t row[], int_t col[], real_t *restrict predicted, size_t n_predict,
    real_t *restrict Am,
    real_t *restrict Bm,
    int_t k,
    int_t m, int_t n,
    int nthreads
);
int_t predict_X_new_offsets_explicit
(
    /* inputs for predictions */
    int_t m_new, bool user_bias,
    int_t row[], int_t col[], real_t *restrict predicted, size_t n_predict,
    int nthreads,
    /* inputs for factors */
    real_t *restrict U, int_t p,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict X, int_t ixA[], int_t ixB[], size_t nnz,
    size_t *restrict Xcsr_p, int_t *restrict Xcsr_i, real_t *restrict Xcsr,
    real_t *restrict Xfull, int_t n, /* <- 'n' MUST be passed */
    real_t *restrict weight,
    real_t *restrict Bm, real_t *restrict C,
    real_t *restrict C_bias,
    real_t glob_mean, real_t *restrict biasB,
    int_t k, int_t k_sec, int_t k_main,
    real_t w_user,
    real_t lam, real_t *restrict lam_unique, bool exact,
    real_t *restrict precomputedTransBtBinvBt,
    real_t *restrict precomputedBtB,
    real_t *restrict Bm_plus_bias
);
int_t predict_X_new_offsets_implicit
(
    /* inputs for predictions */
    int_t m_new,
    int_t row[], int_t col[], real_t *restrict predicted, size_t n_predict,
    int_t n_orig,
    int nthreads,
    /* inputs for factors */
    real_t *restrict U, int_t p,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict X, int_t ixA[], int_t ixB[], size_t nnz,
    size_t *restrict Xcsr_p, int_t *restrict Xcsr_i, real_t *restrict Xcsr,
    real_t *restrict Bm, real_t *restrict C,
    real_t *restrict C_bias,
    int_t k,
    real_t lam, real_t alpha,
    bool apply_log_transf,
    real_t *restrict precomputedBtB
);
int_t fit_content_based_lbfgs
(
    real_t *restrict biasA, real_t *restrict biasB,
    real_t *restrict C, real_t *restrict C_bias,
    real_t *restrict D, real_t *restrict D_bias,
    bool start_with_ALS, bool reset_values, int_t seed,
    real_t *restrict glob_mean,
    int_t m, int_t n, int_t k,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t *restrict Xfull,
    real_t *restrict weight,
    bool user_bias, bool item_bias,
    bool add_intercepts,
    real_t lam, real_t *restrict lam_unique,
    real_t *restrict U, int_t p,
    real_t *restrict II, int_t q,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    int_t I_row[], int_t I_col[], real_t *restrict I_sp, size_t nnz_I,
    int_t n_corr_pairs, size_t maxiter,
    int nthreads, bool prefer_onepass,
    bool verbose, int_t print_every, bool handle_interrupt,
    int_t *restrict niter, int_t *restrict nfev,
    real_t *restrict Am, real_t *restrict Bm
);
int_t factors_content_based_single
(
    real_t *restrict a_vec, int_t k,
    real_t *restrict u_vec, int_t p,
    real_t *restrict u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
    real_t *restrict C, real_t *restrict C_bias
);
int_t factors_content_based_mutliple
(
    real_t *restrict Am, int_t m_new, int_t k,
    real_t *restrict C, real_t *restrict C_bias,
    real_t *restrict U, int_t p,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    int nthreads
);
int_t topN_old_content_based
(
    real_t *restrict a_vec, real_t a_bias,
    real_t *restrict Am, real_t *restrict biasA, int_t row_index,
    real_t *restrict Bm,
    real_t *restrict biasB,
    real_t glob_mean,
    int_t k,
    int_t *restrict include_ix, int_t n_include,
    int_t *restrict exclude_ix, int_t n_exclude,
    int_t *restrict outp_ix, real_t *restrict outp_score,
    int_t n_top, int_t n, int nthreads
);
int_t topN_new_content_based
(
    /* inputs for the factors */
    int_t k, int_t n_new,
    real_t *restrict u_vec, int_t p,
    real_t *restrict u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
    real_t *restrict II, int_t q,
    int_t I_row[], int_t I_col[], real_t *restrict I_sp, size_t nnz_I,
    size_t I_csr_p[], int_t I_csr_i[], real_t *restrict I_csr,
    real_t *restrict C, real_t *restrict C_bias,
    real_t *restrict D, real_t *restrict D_bias,
    real_t glob_mean,
    /* inputs for topN */
    int_t *restrict outp_ix, real_t *restrict outp_score,
    int_t n_top, int nthreads
);
int_t predict_X_old_content_based
(
    real_t *restrict predicted, size_t n_predict,
    int_t m_new, int_t k,
    int_t row[], /* <- optional */
    int_t col[],
    int_t m_orig, int_t n_orig,
    real_t *restrict U, int_t p,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict C, real_t *restrict C_bias,
    real_t *restrict Bm, real_t *restrict biasB,
    real_t glob_mean,
    int nthreads
);
int_t predict_X_new_content_based
(
    real_t *restrict predicted, size_t n_predict,
    int_t m_new, int_t n_new, int_t k,
    int_t row[], int_t col[], /* <- optional */
    real_t *restrict U, int_t p,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict II, int_t q,
    int_t I_row[], int_t I_col[], real_t *restrict I_sp, size_t nnz_I,
    size_t I_csr_p[], int_t I_csr_i[], real_t *restrict I_csr,
    real_t *restrict C, real_t *restrict C_bias,
    real_t *restrict D, real_t *restrict D_bias,
    real_t glob_mean,
    int nthreads
);

#ifdef __cplusplus
}
#endif
