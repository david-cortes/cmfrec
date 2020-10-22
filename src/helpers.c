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
#include "cmfrec.h"

/* Note: in x86_64 computers, there's hardly any speed up from having > 2
   threads zeroing out an array */
void set_to_zero(FPnum *arr, size_t n, int nthreads)
{
    if (n == 0) return;
    #if defined(_OPENMP)
    nthreads = (nthreads > 1)? 2 : 1;
    size_t chunk_size = n / (size_t)nthreads;
    size_t remainder = n % (size_t)nthreads;
    int i = 0;
    if (nthreads > 1 && n > (size_t)1e7)
    {
        #pragma omp parallel for schedule(static, 1) \
                firstprivate(arr, chunk_size, nthreads) num_threads(nthreads)
        for (i = 0; i < nthreads; i++)
            memset(arr + i * chunk_size, 0, chunk_size*sizeof(FPnum));
        if (remainder > 0)
            memset(arr + nthreads * chunk_size, 0, remainder*sizeof(FPnum));
    } else
    #endif
    {
        memset(arr, 0, n*sizeof(FPnum));
    }
}

/* Note: in x86_64 computers, there's hardly any speed up from having > 4
   threads copying arrays */
void copy_arr(FPnum *restrict src, FPnum *restrict dest, size_t n, int nthreads)
{
    /* Note: don't use BLAS scopy as it's actually much slower */
    if (n == 0) return;
    #if defined(_OPENMP)
    nthreads = cap_to_4(nthreads);
    size_t chunk_size = n / (size_t)nthreads;
    size_t remainder = n % (size_t)nthreads;
    int i = 0;
    if (nthreads > 1 && n > (size_t)5e6)
    {
        #pragma omp parallel for schedule(static, 1) \
                firstprivate(src, dest, chunk_size, nthreads) num_threads(nthreads)
        for (i = 0; i < nthreads; i++)
            memcpy(dest + i * chunk_size, src + i * chunk_size, chunk_size*sizeof(FPnum));
        if (remainder > 0)
            memcpy(dest + nthreads*chunk_size, src + nthreads*chunk_size, remainder*sizeof(FPnum));
    }  else 
    #endif
    {
        memcpy(dest, src, n*sizeof(FPnum));
    }
}

int count_NAs(FPnum arr[], size_t n, int nthreads)
{
    int cnt_NA = 0;
    nthreads = cap_to_4(nthreads);

    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif

    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(arr, n) reduction(+:cnt_NA)
    for (size_t_for ix = 0; ix < n; ix++)
        cnt_NA += isnan(arr[ix]);
    if (cnt_NA < 0) cnt_NA = INT_MAX; /* <- overflow */
    return cnt_NA;
}

void count_NAs_by_row
(
    FPnum *restrict arr, int m, int n,
    int *restrict cnt_NA, int nthreads,
    bool *restrict full_dense, bool *restrict near_dense
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long row;
    #endif
    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(m, n, arr, cnt_NA)
    for (size_t_for row = 0; row < (size_t)m; row++)
        for (size_t col = 0; col < (size_t)n; col++)
            cnt_NA[row] += isnan(arr[col + row*n]);

    *full_dense = true;
    for (int ix = 0; ix < m; ix++) {
        if (cnt_NA[ix]) {
            *full_dense = false;
            break;
        }
    }

    /* Will be considered near-dense if at least 70% of the rows have
       no missing values.
       This is used later in order to decide whether to use a gradient-
       based approach or closed-form when optimizing a matrix in isolation */
    *near_dense = false;
    int cnt_rows_w_NA = 0;
    if (!full_dense)
    {
        for (int ix = 0; ix < m; ix++)
            cnt_rows_w_NA += (cnt_NA[ix] > 0);
        if ((m - cnt_rows_w_NA) >= (int)(0.7 * (double)m))
            *near_dense = true;
    }
}

void count_NAs_by_col
(
    FPnum *restrict arr, int m, int n,
    int *restrict cnt_NA,
    bool *restrict full_dense, bool *restrict near_dense
)
{
    for (size_t row = 0; row < (size_t)m; row++)
        for (size_t col = 0; col < (size_t)n; col++)
            cnt_NA[col] += isnan(arr[col + row*n]);

    *full_dense = true;
    for (int ix = 0; ix < n; ix++) {
        if (cnt_NA[ix]) {
            *full_dense = false;
            break;
        }
    }

    *near_dense = false;
    int cnt_rows_w_NA = 0;
    if (!full_dense)
    {
        for (int ix = 0; ix < n; ix++)
            cnt_rows_w_NA += (cnt_NA[ix] > 0);
        if ((n - cnt_rows_w_NA) >= (int)(0.7 * (double)n))
            *near_dense = true;
    }
}

void sum_by_rows(FPnum *restrict A, FPnum *restrict outp, int m, int n, int nthreads)
{
    nthreads = cap_to_4(nthreads);
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long row;
    #endif
    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(m, n, A, outp)
    for (size_t_for row = 0; row < (size_t)m; row++)
        for (size_t col = 0; col < (size_t)n; col++)
            outp[row] += A[col + row*(size_t)n];
}

void sum_by_cols(FPnum *restrict A, FPnum *restrict outp, int m, int n, size_t lda, int nthreads)
{
    #ifdef _OPENMP
    /* Note: GCC and CLANG do a poor optimization when the array to sum has many
       rows and few columns, which is the most common use-case for this */
    if ((FPnum)n > 1e3*(FPnum)m && nthreads > 4) /* this assumes there's many columns, in which case there's a speedup */
    {
        #if defined(_OPENMP) && \
                    ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                      || defined(_WIN32) || defined(_WIN64) \
                    )
        long long col;
        #endif
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(A, outp, m, n, lda)
        for (size_t_for col = 0; col < (size_t)n; col++)
            for (size_t row = 0; row < (size_t)m; row++)
                outp[col] += A[col + row*lda];
    }

    else
    #endif
    {
        for (size_t row = 0; row < (size_t)m; row++)
            for (size_t col = 0; col < (size_t)n; col++)
                outp[col] += A[col + row*lda];
    }
}

void mat_plus_rowvec(FPnum *restrict A, FPnum *restrict b, int m, int n, int nthreads)
{
    nthreads = cap_to_4(nthreads);
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long row;
    #endif
    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(A, b, m, n)
    for (size_t_for row = 0; row < (size_t)m; row++)
        for (size_t col = 0; col < (size_t)n; col++)
            A[col + (size_t)row*n] += b[row];
}

void mat_plus_colvec(FPnum *restrict A, FPnum *restrict b, FPnum alpha, int m, int n, size_t lda, int nthreads)
{
    nthreads = cap_to_4(nthreads);
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long row;
    #endif

    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(m, n, lda, A, b)
    for (size_t_for row = 0; row < (size_t)m; row++)
        cblas_taxpy(n, alpha, b, 1, A + row*lda, 1);
}

void mat_minus_rowvec2
(
    FPnum *restrict Xfull,
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    FPnum *restrict b, int m, int n, int nthreads
)
{
    nthreads = cap_to_4(nthreads);
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long row, ix;
    #endif

    if (Xfull != NULL)
    {
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(Xfull, m, n, b)
        for (size_t_for row = 0; row < (size_t)m; row++)
            for (size_t col = 0; col < (size_t)n; col++)
                Xfull[col + row*(size_t)n] -= b[row];
    }

    else
    {
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(X, b, ixA, nnz)
        for (size_t_for ix = 0; ix < nnz; ix++)
            X[ix] -= b[ixA[ix]];
    }
}

void mat_minus_colvec2
(
    FPnum *restrict Xfull,
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    FPnum *restrict b, int m, int n, int nthreads
)
{
    nthreads = cap_to_4(nthreads);
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif

    if (Xfull != NULL)
    {
        for (size_t row = 0; row < (size_t)m; row++)
            for (size_t col = 0; col < (size_t)n; col++)
                Xfull[col + row*(size_t)n] -= b[col];
    }

    else
    {
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(X, b, ixA, nnz)
        for (size_t_for ix = 0; ix < nnz; ix++)
            X[ix] -= b[ixB[ix]];
    }
}

void nan_to_zero(FPnum *restrict arr, FPnum *restrict comp, size_t n, int nthreads)
{
    nthreads = cap_to_4(nthreads);
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif
    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(arr, comp, n)
    for (size_t_for ix = 0; ix < n; ix++)
        arr[ix] = (!isnan(comp[ix]))? arr[ix] : 0;
}

void mult_if_non_nan(FPnum *restrict arr, FPnum *restrict comp, FPnum *restrict w, size_t n, int nthreads)
{
    nthreads = cap_to_4(nthreads);
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif
    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(arr, w, n)
    for (size_t_for ix = 0; ix < n; ix++)
        arr[ix] = (!isnan(arr[ix]))? (w[ix] * arr[ix]) : (0);
}

void mult_elemwise(FPnum *restrict inout, FPnum *restrict other, size_t n, int nthreads)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif
    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(inout, other, n)
    for (size_t_for ix = 0; ix < n; ix++)
        inout[ix] *= other[ix];
}

FPnum sum_squares(FPnum *restrict arr, size_t n, int nthreads)
{
    double res = 0;
    if (n < (size_t)INT_MAX)
        return cblas_tdot((int)n, arr, 1, arr, 1);
    else {
        nthreads = cap_to_4(nthreads);
        #if defined(_OPENMP) && \
                    ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                      || defined(_WIN32) || defined(_WIN64) \
                    )
        long long ix;
        #endif
        #pragma omp parallel for schedule(static) num_threads(nthreads) shared(arr, n) reduction(+:res)
        for (size_t_for ix = 0; ix < n; ix++)
            res += square(arr[ix]);
    }
    return (FPnum)res;
}

void taxpy_large(FPnum *restrict A, FPnum x, FPnum *restrict Y, size_t n, int nthreads)
{
    if (n < (size_t)INT_MAX)
        cblas_taxpy((int)n, x, A, 1, Y, 1);
    else {
        nthreads = cap_to_4(nthreads);
        #if defined(_OPENMP) && \
                    ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                      || defined(_WIN32) || defined(_WIN64) \
                    )
        long long ix;
        #endif
        if (x == 1.)
            #pragma omp parallel for schedule(static) num_threads(nthreads) shared(A, Y, n)
            for (size_t_for ix = 0; ix < n; ix++)
                Y[ix] += A[ix];
        else
            #pragma omp parallel for schedule(static) num_threads(nthreads) shared(A, x, Y, n)
            for (size_t_for ix = 0; ix < n; ix++)
                Y[ix] += x*A[ix];
    }
}

void tscal_large(FPnum *restrict arr, FPnum alpha, size_t n, int nthreads)
{
    if (alpha == 1.)
        return;
    if (n < (size_t)INT_MAX)
        cblas_tscal((int)n, alpha, arr, 1);
    else {
        nthreads = cap_to_4(nthreads);
        #if defined(_OPENMP) && \
                    ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                      || defined(_WIN32) || defined(_WIN64) \
                    )
        long long ix;
        #endif
        #pragma omp parallel for schedule(static) num_threads(nthreads) shared(arr, alpha, n)
        for (size_t_for ix = 0; ix < n; ix++)
            arr[ix] *= alpha;
    }
}

int rnorm(FPnum *restrict arr, size_t n, int seed, int nthreads)
{
    #ifndef _FOR_R
    int three = 3;
    int seed_arr[4] = {seed, seed, seed, seed};
    process_seed_for_larnv(seed_arr);
    if (n < (size_t)INT_MAX)
    {
        int n_int = (int)n;
        tlarnv_(&three, seed_arr, &n_int, arr);
    }

    else
    {
        #if defined(_OPENMP) && \
                    ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                      || defined(_WIN32) || defined(_WIN64) \
                    )
        long long chunk;
        #endif
        int chunk_size = (int)INT_MAX;
        size_t chunks = n / (size_t)INT_MAX;
        int remainder = n - (size_t)INT_MAX * chunks;
        int *restrict mt_seed_arr = (int*)malloc(4*nthreads*sizeof(int));
        int *restrict thread_seed;
        if (mt_seed_arr == NULL) return 1;

        #pragma omp parallel for schedule(static, 1) num_threads(nthreads) \
                shared(arr, three, chunk_size, chunks, seed) \
                private(thread_seed)
        for (size_t_for chunk = 0; chunk < chunks; chunk++) {
            thread_seed = mt_seed_arr + 4*omp_get_thread_num();
            thread_seed[0] = seed; thread_seed[1] = seed;
            thread_seed[2] = seed; thread_seed[3] = seed;
            tlarnv_(&three, thread_seed, &chunk_size,
                    arr + chunk*(size_t)chunk_size);
        }
        if (remainder)
            tlarnv_(&three, seed_arr, &remainder, arr + (size_t)INT_MAX * chunks);
        free(mt_seed_arr);
    }
    #else
    GetRNGstate();
    for (size_t ix = 0; ix < n; ix++)
        arr[ix] = norm_rand();
    PutRNGstate();
    #endif
    return 0;
}

void rnorm_preserve_seed(FPnum *restrict arr, size_t n, int seed_arr[4])
{
    #ifndef _FOR_R
    process_seed_for_larnv(seed_arr);
    int three = 3;

    if (n < (size_t)INT_MAX){
        int n_int = (int)n;
        tlarnv_(&three, seed_arr, &n_int, arr);
    }

    else {
        size_t remainder = n;
        int size_chunk = 0;
        while (remainder)
        {
            if (remainder >= (size_t)INT_MAX)
                size_chunk = INT_MAX;
            else
                size_chunk = remainder;
            remainder -= (size_t)size_chunk;
            tlarnv_(&three, seed_arr, &size_chunk, arr);
            arr += size_chunk;
        }
    }
    #else
    GetRNGstate();
    for (size_t ix = 0; ix < n; ix++)
        arr[ix] = norm_rand();
    PutRNGstate();
    #endif
}

void process_seed_for_larnv(int seed_arr[4])
{
    for (int ix = 0; ix < 4; ix++)
    {
        seed_arr[ix] = min2(seed_arr[ix], 4095);
        seed_arr[ix] = max2(seed_arr[ix], 0);
        if (ix == 3 && (seed_arr[ix] % 2) == 0)
        {
            if ((seed_arr[ix] + 1) <= 4095 && (seed_arr[ix] + 1) >= 0)
                seed_arr[ix]++;
            else if ((seed_arr[ix] - 1) <= 4095 && (seed_arr[ix] - 1) >= 0)
                seed_arr[ix]--;
            else
                seed_arr[ix] = 1;
        }
    }
}

void reduce_mat_sum(FPnum *restrict outp, size_t lda, FPnum *restrict inp,
                    int m, int n, int nthreads)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long row;
    #endif
    size_t m_by_n = m * n;
    if (n > 1 || lda > 0)
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(outp, inp, m, n, nthreads)
        for (size_t_for row = 0; row < (size_t)m; row++)
            for (size_t tid = 0; tid < (size_t)nthreads; tid++)
                for (size_t col = 0; col < (size_t)n; col++)
                    outp[col + row*lda] += inp[tid*m_by_n + col + row*n];
    else
        for (size_t tid = 0; tid < (size_t)nthreads; tid++)
            for (size_t row = 0; row < (size_t)m; row++)
                outp[row] += inp[tid*m_by_n + row];
}

void exp_neg_x(FPnum *restrict arr, size_t n, int nthreads)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif
    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(arr, n)
    for (size_t_for ix = 0; ix < n; ix++)
        arr[ix] = exp_t(-arr[ix]);
}

void add_to_diag(FPnum *restrict A, FPnum val, size_t n)
{
    for (size_t ix = 0; ix < n; ix++)
        A[ix + ix*n] += val;
}

FPnum sum_sq_div_w(FPnum *restrict arr, FPnum *restrict w, size_t n, bool compensated, int nthreads)
{
    FPnum res = 0;
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif
    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(arr, w, n) reduction(+:res)
    for (size_t_for ix = 0; ix < n; ix++)
        res += square(arr[ix]) / w[ix];
    return res;
}

/* X <- alpha*A*B + X | A(m,k) is sparse CSR, B(k,n) is dense */
void tgemm_sp_dense
(
    int m, int n, FPnum alpha,
    size_t indptr[], int indices[], FPnum values[],
    FPnum DenseMat[], size_t ldb,
    FPnum OutputMat[], size_t ldc,
    int nthreads
)
{
    FPnum *ptr_col;
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long row;
    #endif

    if (alpha != 1.)
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                shared(m, n, alpha, ldb, ldc, OutputMat, DenseMat, indptr, indices, values) \
                private(ptr_col)
        for (size_t_for row = 0; row < (size_t)m; row++) {
            ptr_col = OutputMat + row*ldc;
            for (size_t col = indptr[row]; col < indptr[row+1]; col++) {
                cblas_taxpy(n, alpha*values[col], DenseMat + (size_t)indices[col]*ldb, 1, ptr_col, 1);
            }
        }
    else
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                shared(m, n, ldb, ldc, OutputMat, DenseMat, indptr, indices, values) \
                private(ptr_col)
        for (size_t_for row = 0; row < (size_t)m; row++) {
            ptr_col = OutputMat + row*ldc;
            for (size_t col = indptr[row]; col < indptr[row+1]; col++) {
                cblas_taxpy(n, values[col], DenseMat + (size_t)indices[col]*ldb, 1, ptr_col, 1);
            }
        }
}

/* x <- alpha*t(A)*v + x | A[m,n] is dense, v[m] is sparse, x[n] is dense */
void tgemv_dense_sp
(
    int m, int n,
    FPnum alpha, FPnum DenseMat[], size_t lda,
    int ixB[], FPnum vec_sp[], size_t nnz,
    FPnum OutputVec[]
)
{
    if (alpha != 1.)
        for (size_t ix = 0; ix < nnz; ix++)
            cblas_taxpy(n, alpha*vec_sp[ix], DenseMat + (size_t)ixB[ix]*lda, 1, OutputVec, 1);
    else
        for (size_t ix = 0; ix < nnz; ix++)
            cblas_taxpy(n, vec_sp[ix], DenseMat + (size_t)ixB[ix]*lda, 1, OutputVec, 1);
}

/* Same but with an array of weights */
void tgemv_dense_sp_weighted
(
    int m, int n,
    FPnum alpha[], FPnum DenseMat[], size_t lda,
    int ixB[], FPnum vec_sp[], size_t nnz,
    FPnum OutputVec[]
)
{
    for (size_t ix = 0; ix < nnz; ix++)
        cblas_taxpy(n, alpha[ix]*vec_sp[ix], DenseMat + (size_t)ixB[ix]*lda, 1, OutputVec, 1);
}

/* Same, but with both array of weights and scalar weight */
void tgemv_dense_sp_weighted2
(
    int m, int n,
    FPnum alpha[], FPnum alpha2, FPnum DenseMat[], size_t lda,
    int ixB[], FPnum vec_sp[], size_t nnz,
    FPnum OutputVec[]
)
{
    for (size_t ix = 0; ix < nnz; ix++)
        cblas_taxpy(n, alpha2*alpha[ix]*vec_sp[ix], DenseMat + (size_t)ixB[ix]*lda, 1, OutputVec, 1);
}

void tgemv_dense_sp_notrans
(
    int m, int n,
    FPnum DenseMat[], int lda,
    int ixB[], FPnum vec_sp[], size_t nnz,
    FPnum OutputVec[]
)
{
    for (size_t ix = 0; ix < nnz; ix++)
        cblas_taxpy(m, vec_sp[ix],
                    DenseMat + ixB[ix], lda,
                    OutputVec, 1);
}

/* B[:m,:n] := A[:m,:n] */
void copy_mat
(
    int m, int n,
    FPnum *restrict A, int lda,
    FPnum *restrict B, int ldb
)
{
    char uplo = '?';
    if (m == 0 && n == 0) return;

    if (ldb == n && lda == n)
        memcpy(B, A, (size_t)m*(size_t)n*sizeof(FPnum));
    else
        tlacpy_(&uplo, &n, &m, A, &lda, B, &ldb);
}

/* B[:m,:n] = A[:m,:n] + B[:m,:n] */
void sum_mat
(
    size_t m, size_t n,
    FPnum *restrict A, size_t lda,
    FPnum *restrict B, size_t ldb
)
{
    int n_int = (int)n;
    if (lda == n && ldb == n)
        taxpy_large(A, 1., B, m*n, 1);
    else
        for (size_t row = 0; row < m; row++)
            cblas_taxpy(n_int, 1., A + row*lda, 1, B + row*ldb, 1);
}

void transpose_mat(FPnum *restrict A, size_t m, size_t n, FPnum *restrict buffer_FPnum)
{
    memcpy(buffer_FPnum, A, m*n*sizeof(FPnum));
    for (size_t row = 0; row < m; row++)
        for (size_t col = 0; col < n; col++)
            A[row + col*m] = buffer_FPnum[col + row*n];
}

void transpose_mat2(FPnum *restrict A, size_t m, size_t n, FPnum *restrict outp)
{
    for (size_t row = 0; row < m; row++)
        for (size_t col = 0; col < n; col++)
            outp[row + col*m] = A[col + row*n];
}

int coo_to_csr_plus_alloc
(
    int *restrict Xrow, int *restrict Xcol, FPnum *restrict Xval,
    FPnum *restrict W,
    int m, int n, size_t nnz,
    size_t *restrict *csr_p, int *restrict *csr_i, FPnum *restrict *csr_v,
    FPnum *restrict *csr_w
)
{
    *csr_p = (size_t*)malloc(((size_t)m+(size_t)1)*sizeof(size_t));
    *csr_i = (int*)malloc(nnz*sizeof(int));
    *csr_v = (FPnum*)malloc(nnz*sizeof(FPnum));
    if (csr_p == NULL || csr_i == NULL || csr_v == NULL)
        return 1;

    if (W != NULL) {
        *csr_w = (FPnum*)malloc(nnz*sizeof(FPnum));
        if (csr_w == NULL) return 1;
    }

    coo_to_csr(
        Xrow, Xcol, Xval,
        W,
        m, n, nnz,
        *csr_p, *csr_i, *csr_v,
        (W == NULL)? ((FPnum*)NULL) : (*csr_w)
    );
    return 0;
}

void coo_to_csr
(
    int *restrict Xrow, int *restrict Xcol, FPnum *restrict Xval,
    FPnum *restrict W,
    int m, int n, size_t nnz,
    size_t *restrict csr_p, int *restrict csr_i, FPnum *restrict csr_v,
    FPnum *restrict csr_w
)
{
    bool has_mem = true;
    int *cnt_byrow = NULL;

    produce_p:
    {
        memset(csr_p, 0, ((size_t)m+(size_t)1)*sizeof(size_t));
        for (size_t ix = 0; ix < nnz; ix++)
            csr_p[Xrow[ix]+(size_t)1]++;
        for (int row = 0; row < m; row++)
            csr_p[row+(size_t)1] += csr_p[row];
    }

    if (!has_mem) goto cleanup;

    cnt_byrow = (int*)calloc(m, sizeof(int));

    if (cnt_byrow != NULL)
    {
        if (W == NULL)
            for (size_t ix = 0; ix < nnz; ix++) {
                csr_v[csr_p[Xrow[ix]] + cnt_byrow[Xrow[ix]]] = Xval[ix];
                csr_i[csr_p[Xrow[ix]] + cnt_byrow[Xrow[ix]]++] = Xcol[ix];
            }
        else
            for (size_t ix = 0; ix < nnz; ix++) {
                csr_w[csr_p[Xrow[ix]] + cnt_byrow[Xrow[ix]]] = W[ix];
                csr_v[csr_p[Xrow[ix]] + cnt_byrow[Xrow[ix]]] = Xval[ix];
                csr_i[csr_p[Xrow[ix]] + cnt_byrow[Xrow[ix]]++] = Xcol[ix];
            }
        goto cleanup;
    }

    else
    {
        if (W == NULL)
            for (size_t ix = 0; ix < nnz; ix++) {
                csr_i[--csr_p[Xrow[ix]+(size_t)1]] = Xcol[ix];
                csr_v[csr_p[Xrow[ix]+(size_t)1]] = Xval[ix];
            }
        else
            for (size_t ix = 0; ix < nnz; ix++) {
                csr_i[--csr_p[Xrow[ix]+(size_t)1]] = Xcol[ix];
                csr_v[csr_p[Xrow[ix]+(size_t)1]] = Xval[ix];
                csr_w[csr_p[Xrow[ix]+(size_t)1]] = W[ix];
            }
        has_mem = false;
        goto produce_p;
    }

    cleanup:
        free(cnt_byrow);
}

void coo_to_csr_and_csc
(
    int *restrict Xrow, int *restrict Xcol, FPnum *restrict Xval,
    FPnum *restrict W, int m, int n, size_t nnz,
    size_t *restrict csr_p, int *restrict csr_i, FPnum *restrict csr_v,
    size_t *restrict csc_p, int *restrict csc_i, FPnum *restrict csc_v,
    FPnum *restrict csr_w, FPnum *restrict csc_w,
    int nthreads
)
{
    bool has_mem = true;
    nthreads = (nthreads > 2)? 2 : 1;
    int *cnt_byrow = NULL;
    int *cnt_bycol = NULL;

    produce_p:
    {
        memset(csr_p, 0, ((size_t)m+(size_t)1)*sizeof(size_t));
        memset(csc_p, 0, ((size_t)n+(size_t)1)*sizeof(size_t));
        for (size_t ix = 0; ix < nnz; ix++) {
            csr_p[Xrow[ix]+(size_t)1]++;
            csc_p[Xcol[ix]+(size_t)1]++;
        }
        for (int row = 0; row < m; row++)
            csr_p[row+(size_t)1] += csr_p[row];
        for (int col = 0; col < n; col++)
            csc_p[col+(size_t)1] += csc_p[col];
    }


    if (!has_mem) goto cleanup;

    cnt_byrow = (int*)calloc(m, sizeof(int));
    cnt_bycol = (int*)calloc(n, sizeof(int));

    #if defined(_OPENMP) && (_OPENMP > 201305) /* OpenMP >= 4.0 */
    omp_set_max_active_levels(2);
    #endif

    if (cnt_byrow != NULL && cnt_bycol != NULL) {
        #pragma omp parallel sections num_threads(nthreads)
        {
            #pragma omp section
            {
                if (W == NULL)
                    for (size_t ix = 0; ix < nnz; ix++) {
                        csr_v[csr_p[Xrow[ix]] + cnt_byrow[Xrow[ix]]] = Xval[ix];
                        csr_i[csr_p[Xrow[ix]] + cnt_byrow[Xrow[ix]]++] = Xcol[ix];
                    }
                else
                    for (size_t ix = 0; ix < nnz; ix++) {
                        csr_w[csr_p[Xcol[ix]] + cnt_byrow[Xcol[ix]]] = W[ix];
                        csr_v[csr_p[Xcol[ix]] + cnt_byrow[Xcol[ix]]] = Xval[ix];
                        csr_i[csr_p[Xcol[ix]] + cnt_byrow[Xcol[ix]]++] = Xrow[ix];
                    }

            }

            #pragma omp section
            {
                if (W == NULL)
                    for (size_t ix = 0; ix < nnz; ix++) {
                        csc_v[csc_p[Xcol[ix]] + cnt_bycol[Xcol[ix]]] = Xval[ix];
                        csc_i[csc_p[Xcol[ix]] + cnt_bycol[Xcol[ix]]++] = Xrow[ix];
                    }
                else
                    for (size_t ix = 0; ix < nnz; ix++) {
                        csc_w[csc_p[Xcol[ix]] + cnt_bycol[Xcol[ix]]] = W[ix];
                        csc_v[csc_p[Xcol[ix]] + cnt_bycol[Xcol[ix]]] = Xval[ix];
                        csc_i[csc_p[Xcol[ix]] + cnt_bycol[Xcol[ix]]++] = Xrow[ix];
                    }
            }
        }
        goto cleanup;
    }

    else {
        #pragma omp parallel sections num_threads(nthreads)
        {
            #pragma omp section
            {
                if (W == NULL)
                    for (size_t ix = 0; ix < nnz; ix++) {
                        csr_i[--csr_p[Xrow[ix]+(size_t)1]] = Xcol[ix];
                        csr_v[csr_p[Xrow[ix]+(size_t)1]] = Xval[ix];
                    }
                else
                    for (size_t ix = 0; ix < nnz; ix++) {
                        csr_i[--csr_p[Xrow[ix]+(size_t)1]] = Xcol[ix];
                        csr_v[csr_p[Xrow[ix]+(size_t)1]] = Xval[ix];
                        csr_w[csr_p[Xrow[ix]+(size_t)1]] = W[ix];
                    }
            }

            #pragma omp section
            {
                if (W == NULL)
                    for (size_t ix = 0; ix < nnz; ix++) {
                        csc_i[--csc_p[Xcol[ix]+(size_t)1]] = Xrow[ix];
                        csc_v[csc_p[Xcol[ix]+(size_t)1]] = Xval[ix];
                    }
                else
                    for (size_t ix = 0; ix < nnz; ix++) {
                        csc_i[--csc_p[Xcol[ix]+(size_t)1]] = Xrow[ix];
                        csc_v[csc_p[Xcol[ix]+(size_t)1]] = Xval[ix];
                        csc_w[csc_p[Xcol[ix]+(size_t)1]] = W[ix];
                    }
            }
        }
        has_mem = false;
        goto produce_p;
    }

    cleanup:
        free(cnt_byrow);
        free(cnt_bycol);
}

void row_means_csr(size_t indptr[], FPnum *restrict values,
                   FPnum *restrict output, int m, int nthreads)
{
    int row = 0;
    set_to_zero(values, m, nthreads);
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
            shared(indptr, values, output, m)
    for (row = 0; row < m; row++)
        for (size_t ix = indptr[row]; ix < indptr[row+(size_t)1]; ix++)
            output[row] += values[ix];
    nthreads = cap_to_4(nthreads);
    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(indptr, output, m)
    for (row = 0; row < m; row++)
        output[row] /= (FPnum)(indptr[row+(size_t)1] - indptr[row]);
}

bool should_stop_procedure = false;
void set_interrup_global_variable(int s)
{
    fprintf(stderr, "Error: procedure was interrupted\n");
    #if !defined(_FOR_R)
    fflush(stderr);
    #endif
    should_stop_procedure = true;
}

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
)
{
    ((data_collective_fun_grad*)instance)->niter = k;
    int print_every = ((data_collective_fun_grad*)instance)->print_every;
    if ((k % print_every) == 0 && print_every > 0) {
        printf("Iteration %3d - f(x)=%7.03g - ||g(x)||=%5.03g - ls=%d\n",
               k, fx, gnorm, ls);
        #if !defined(_FOR_R)
        fflush(stdout);
        #endif
    }
    if (((data_collective_fun_grad*)instance)->handle_interrupt)
        signal(SIGINT, set_interrup_global_variable);
    if (should_stop_procedure) {
        should_stop_procedure = false;
        return 1;
    }
    return 0;
}

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
)
{
    ((data_offsets_fun_grad*)instance)->niter = k;
    int print_every = ((data_offsets_fun_grad*)instance)->print_every;
    if ((k % print_every) == 0 && print_every > 0) {
        printf("Iteration %3d - f(x)=%7.03g - ||g(x)||=%5.03g - ls=%d\n",
               k, fx, gnorm, ls);
        #if !defined(_FOR_R)
        fflush(stdout);
        #endif
    }
    if (((data_offsets_fun_grad*)instance)->handle_interrupt)
        signal(SIGINT, set_interrup_global_variable);
    if (should_stop_procedure) {
        should_stop_procedure = false;
        return 1;
    }
    return 0;
}

bool check_is_sorted(int arr[], int n)
{
    if (n <= 1) return true;
    for (int ix = 0; ix < n-1; ix++)
        if (arr[ix] > arr[ix+1]) return false;
    return true;
}

/* https://www.stat.cmu.edu/~ryantibs/median/quickselect.c */
/* Some sample C code for the quickselect algorithm, 
   taken from Numerical Recipes in C. */
#define SWAP(a,b) temp=(a);(a)=(b);(b)=temp;
void qs_argpartition(int arr[], FPnum values[], int n, int k)
{
    int i,ir,j,l,mid;
    int a,temp;

    l=0;
    ir=n-1;
    for(;;) {
        if (ir <= l+1) { 
            if (ir == l+1 && values[arr[ir]] > values[arr[l]]) {
                SWAP(arr[l],arr[ir]);
            }
            return;
        }
        else {
            mid=(l+ir) >> 1; 
            SWAP(arr[mid],arr[l+1]);
            if (values[arr[l]] < values[arr[ir]]) {
                SWAP(arr[l],arr[ir]);
            }
            if (values[arr[l+1]] < values[arr[ir]]) {
                SWAP(arr[l+1],arr[ir]);
            }
            if (values[arr[l]] < values[arr[l+1]]) {
                SWAP(arr[l],arr[l+1]);
            }
            i=l+1; 
            j=ir;
            a=arr[l+1]; 
            for (;;) { 
                do i++; while (values[arr[i]] > values[a]); 
                do j--; while (values[arr[j]] < values[a]); 
                if (j < i) break; 
                SWAP(arr[i],arr[j]);
            } 
            arr[l+1]=arr[j]; 
            arr[j]=a;
            if (j >= k) ir=j-1; 
            if (j <= k) l=i;
        }
    }
}


void append_ones_last_col
(
    FPnum *restrict orig, size_t m, size_t n,
    FPnum *restrict outp
)
{
    copy_mat(m, n,
             orig, n,
             outp, n+1);
    for (size_t ix = 0; ix < m; ix++)
        outp[n + ix*(n+(size_t)1)] = 1.;
}

void fill_lower_triangle(FPnum A[], size_t n, size_t lda)
{
    for (size_t row = 1; row < n; row++)
        for (size_t col = 0; col < row; col++)
            A[col + row*lda] = A[row + col*lda];
}

void print_oom_message(void)
{
    fprintf(stderr, "Error: could not allocate enough memory.\n");
    #ifndef _FOR_R
    fflush(stderr);
    #endif
}
