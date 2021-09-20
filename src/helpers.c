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
#include "cmfrec.h"

/* Note: in x86_64 computers, there's hardly any speed up from having > 2
   threads zeroing out an array */
void set_to_zero_(real_t *arr, size_t n, int nthreads)
{
    if (n == 0) return;
    #if defined(_OPENMP)
    nthreads = (nthreads > 1)? 2 : 1;
    size_t chunk_size = n / (size_t)nthreads;
    size_t remainder = n % (size_t)nthreads;
    int_t i = 0;
    if (nthreads > 1 && n > (size_t)1e8)
    {
        #pragma omp parallel for schedule(static, 1) \
                firstprivate(arr, chunk_size, nthreads) num_threads(nthreads)
        for (i = 0; i < nthreads; i++)
            memset(arr + i * chunk_size, 0, chunk_size*sizeof(real_t));
        if (remainder > 0)
            memset(arr + nthreads * chunk_size, 0, remainder*sizeof(real_t));
    } else
    #endif
    {
        memset(arr, 0, n*sizeof(real_t));
    }
}

/* Note: in x86_64 computers, there's hardly any speed up from having > 4
   threads copying arrays */
void copy_arr_(real_t *restrict src, real_t *restrict dest, size_t n, int nthreads)
{
    /* Note: don't use BLAS scopy as it's actually much slower */
    if (n == 0) return;
    #if defined(_OPENMP)
    if (nthreads > 1 && n > (size_t)1e8)
    {
        nthreads = cap_to_4(nthreads);
        size_t chunk_size = n / (size_t)nthreads;
        size_t remainder = n % (size_t)nthreads;
        int_t i = 0;

        #pragma omp parallel for schedule(static, 1) \
                firstprivate(src, dest, chunk_size, nthreads) num_threads(nthreads)
        for (i = 0; i < nthreads; i++)
            memcpy(dest + i * chunk_size, src + i * chunk_size, chunk_size*sizeof(real_t));
        if (remainder > 0)
            memcpy(dest + nthreads*chunk_size, src + nthreads*chunk_size, remainder*sizeof(real_t));
    }  else 
    #endif
    {
        memcpy(dest, src, n*sizeof(real_t));
    }
}

/* Note: the C99 standard only guarantes that isnan(NAN)!=0, and some compilers
   like mingw64 will NOT make isnan(NAN)==1. */
int_t count_NAs(real_t arr[], size_t n, int nthreads)
{
    int_t cnt_NA = 0;
    nthreads = cap_to_4(nthreads);

    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif

    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(arr, n) reduction(+:cnt_NA)
    for (size_t_for ix = 0; ix < n; ix++)
        cnt_NA += isnan(arr[ix]) != 0;
    if (cnt_NA < 0) cnt_NA = INT_MAX; /* <- overflow */
    return cnt_NA;
}

void count_NAs_by_row
(
    real_t *restrict arr, int_t m, int_t n,
    int_t *restrict cnt_NA, int nthreads,
    bool *restrict full_dense, bool *restrict near_dense,
    bool *restrict some_full
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
    {
        int_t cnt = 0;
        for (size_t col = 0; col < (size_t)n; col++)
            cnt += isnan(arr[col + row*n]) != 0;
        cnt_NA[row] = cnt;
    }

    *full_dense = true;
    for (int_t ix = 0; ix < m; ix++) {
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
    int_t cnt_rows_w_NA = 0;
    if (!(*full_dense))
    {
        for (int_t ix = 0; ix < m; ix++)
            cnt_rows_w_NA += (cnt_NA[ix] > 0);
        if ((m - cnt_rows_w_NA) >= (int)(0.75 * (double)m))
            *near_dense = true;
    }

    *some_full = *full_dense;
    if (!(*full_dense))
    {
        for (int_t ix = 0; ix < m; ix++)
        {
            if (cnt_NA[ix] == 0) {
                *some_full = true;
                break;
            }
        }
    }
}

void count_NAs_by_col
(
    real_t *restrict arr, int_t m, int_t n,
    int_t *restrict cnt_NA,
    bool *restrict full_dense, bool *restrict near_dense,
    bool *restrict some_full
)
{
    for (size_t row = 0; row < (size_t)m; row++)
        for (size_t col = 0; col < (size_t)n; col++)
            cnt_NA[col] += isnan(arr[col + row*n]) != 0;

    *full_dense = true;
    for (int_t ix = 0; ix < n; ix++) {
        if (cnt_NA[ix]) {
            *full_dense = false;
            break;
        }
    }

    *near_dense = false;
    int_t cnt_rows_w_NA = 0;
    if (!(*full_dense))
    {
        for (int_t ix = 0; ix < n; ix++)
            cnt_rows_w_NA += (cnt_NA[ix] > 0);
        if ((n - cnt_rows_w_NA) >= (int_t)(0.75 * (real_t)n))
            *near_dense = true;
    }

    *some_full = *full_dense;
    if (!(*full_dense))
    {
        for (int_t ix = 0; ix < n; ix++)
        {
            if (cnt_NA[ix] == 0) {
                *some_full = true;
                break;
            }
        }
    }
}

void sum_by_rows(real_t *restrict A, real_t *restrict outp, int_t m, int_t n, int nthreads)
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
    {
        double rsum = 0;
        for (size_t col = 0; col < (size_t)n; col++)
            rsum += A[col + row*(size_t)n];
        outp[row] = rsum;
    }
}

void sum_by_cols(real_t *restrict A, real_t *restrict outp, int_t m, int_t n, size_t lda, int nthreads)
{
    #ifdef _OPENMP
    /* Note: GCC and CLANG do a poor optimization when the array to sum has many
       rows and few columns, which is the most common use-case for this */
    if ((real_t)n > 1e3*(real_t)m && nthreads > 4) /* this assumes there's many columns, in which case there's a speedup */
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
        {
            double csum = 0;
            for (size_t row = 0; row < (size_t)m; row++)
                csum += A[col + row*lda];
            outp[col] = csum;
        }
    }

    else
    #endif
    {
        for (size_t row = 0; row < (size_t)m; row++)
            for (size_t col = 0; col < (size_t)n; col++)
                outp[col] += A[col + row*lda];
    }
}

void mat_plus_rowvec(real_t *restrict A, real_t *restrict b, int_t m, int_t n, int nthreads)
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

void mat_plus_colvec(real_t *restrict A, real_t *restrict b, real_t alpha, int_t m, int_t n, size_t lda, int nthreads)
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
    real_t *restrict Xfull,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t *restrict b, int_t m, int_t n, int nthreads
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
    real_t *restrict Xfull,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t *restrict b, int_t m, int_t n, int nthreads
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

void nan_to_zero(real_t *restrict arr, real_t *restrict comp, size_t n, int nthreads)
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

void mult_if_non_nan(real_t *restrict arr, real_t *restrict comp, real_t *restrict w, size_t n, int nthreads)
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

void mult_elemwise(real_t *restrict inout, real_t *restrict other, size_t n, int nthreads)
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

real_t sum_squares(real_t *restrict arr, size_t n, int nthreads)
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
    return (real_t)res;
}

void taxpy_large(real_t *restrict A, real_t x, real_t *restrict Y, size_t n, int nthreads)
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
                Y[ix] = fma_t(x, A[ix], Y[ix]);
    }
}

void tscal_large(real_t *restrict arr, real_t alpha, size_t n, int nthreads)
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

/* Xoshiro256++ and Xoshiro128++
   https://prng.di.unimi.it */
static inline uint64_t splitmix64(const uint64_t seed)
{
    uint64_t z = (seed + 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}
#ifndef USE_XOSHIRO128
static inline uint64_t rotl64(const uint64_t x, const int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoshiro256pp(uint64_t state[4])
{
    const uint64_t result = rotl64(state[0] + state[3], 23) + state[0];
    const uint64_t t = state[1] << 17;
    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];
    state[2] ^= t;
    state[3] = rotl64(state[3], 45);
    return result;
}

static inline void xoshiro256pp_jump(uint64_t state[4])
{
    const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
                              0xa9582618e03fc9aa, 0x39abdc4529b1661c };
    uint64_t s0 = 0;
    uint64_t s1 = 0;
    uint64_t s2 = 0;
    uint64_t s3 = 0;
    for (int i = 0; i < (int)(sizeof (JUMP) / sizeof (*JUMP)); i++)
    {
        for (int b = 0; b < 64; b++)
        {
            if (JUMP[i] & UINT64_C(1) << b)
            {
                s0 ^= state[0];
                s1 ^= state[1];
                s2 ^= state[2];
                s3 ^= state[3];
            }
            xoshiro256pp(state);
        }
    }
        
    state[0] = s0;
    state[1] = s1;
    state[2] = s2;
    state[3] = s3;
}
#else

static inline uint32_t rotl32(const uint32_t x, const int k) {
    return (x << k) | (x >> (32 - k));
}

static inline uint32_t xoshiro128pp(uint32_t state[4])
{
    const uint32_t result = rotl32(state[0] + state[3], 7) + state[0];
    const uint32_t t = state[1] << 9;
    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];
    state[2] ^= t;
    state[3] = rotl32(state[3], 11);
    return result;
}

static inline void xoshiro128pp_jump(uint32_t state[4])
{
    const uint32_t JUMP[] = { 0x8764000b, 0xf542d2d3,
                              0x6fa035c3, 0x77f2db5b };
    uint32_t s0 = 0;
    uint32_t s1 = 0;
    uint32_t s2 = 0;
    uint32_t s3 = 0;
    for(int i = 0; i < (int)(sizeof (JUMP) / sizeof (*JUMP)); i++)
    {
        for(int b = 0; b < 32; b++)
        {
            if (JUMP[i] & UINT32_C(1) << b)
            {
                s0 ^= state[0];
                s1 ^= state[1];
                s2 ^= state[2];
                s3 ^= state[3];
            }
            xoshiro128pp(state);
        }
    }
        
    state[0] = s0;
    state[1] = s1;
    state[2] = s2;
    state[3] = s3;
}
#endif

/* Note: for double precision, this uses the Box-Muller transform
   with raw form, which is less efficient than the polar form.
   Nevertheless, from some experiments, this seems to give slightly better
   end results when using double precision, even though it is slower and
   loses more numeric precision by boxing to [0, 1] instead of [-1, 1].
   For single precision, the polar form tended to give better results.

   Note: if generating a uniform random number ~ (0,1), dividing
   a random draw by the maximum will not result in a uniform
   distribution, as the upper possible numbers are not evenly-spaced.
   In these cases, it's necessary to take something up to 2^53 as
   this is the interval that's evenly-representable. */
#if defined(USE_DOUBLE) || !(defined(USE_FLOAT) && defined(USE_XOSHIRO128))
void rnorm_xoshiro(real_t *seq, const size_t n, rng_state_t state[4])
{
    #ifndef USE_XOSHIRO128
    const uint64_t two53_i = (UINT64_C(1) << 53) - UINT64_C(1);
    #endif
    const double twoPI = 2. * M_PI;
    uint64_t rnd1, rnd2;
    #ifdef USE_XOSHIRO128
    uint32_t rnd11, rnd12, rnd21, rnd22;
    const uint32_t two21_i = (UINT32_C(1) << 21) - UINT32_C(1);
    const uint32_t ONE = 1;
    const bool is_little_endian = *((unsigned char*)&ONE) != 0;
    #endif
    double u, v;
    size_t n_ = n / (size_t)2;
    for (size_t ix = 0; ix < n_; ix++)
    {
        do
        {
            #ifdef USE_XOSHIRO128
            rnd11 = xoshiro128pp(state);
            rnd12 = xoshiro128pp(state);
            rnd21 = xoshiro128pp(state);
            rnd22 = xoshiro128pp(state);
            #else
            rnd1 = xoshiro256pp(state);
            rnd2 = xoshiro256pp(state);
            #endif

            #if defined(DBL_MANT_DIG) && (DBL_MANT_DIG == 53) &&(FLT_RADIX == 2)
            #ifdef USE_XOSHIRO128
            if (is_little_endian) {
                rnd12 = rnd12 & two21_i;
                rnd22 = rnd22 & two21_i;
            } else {
                rnd11 = rnd11 & two21_i;
                rnd21 = rnd21 & two21_i;
            }
            memcpy((char*)&rnd1, &rnd11, sizeof(uint32_t));
            memcpy((char*)&rnd1 + sizeof(uint32_t), &rnd12, sizeof(uint32_t));
            memcpy((char*)&rnd2, &rnd21, sizeof(uint32_t));
            memcpy((char*)&rnd2 + sizeof(uint32_t), &rnd22, sizeof(uint32_t));
            u = ldexp((double)rnd1, -53);
            v = ldexp((double)rnd2, -53);
            #else
            u = ldexp((double)(rnd1 & two53_i), -53);
            v = ldexp((double)(rnd2 & two53_i), -53);
            #endif
            #else
            u = (double)rnd1 / (double)UINT64_MAX;
            v = (double)rnd2 / (double)UINT64_MAX;
            #endif
        }
        while (u == 0 || v == 0);

        u = sqrt(-2. * log(u));
        seq[(size_t)2*ix] = (real_t)ldexp(cos(twoPI * v) * u, -7);
        seq[(size_t)2*ix + (size_t)1] = (real_t)ldexp(sin(twoPI * v) * u, -7);
    }

    if ((n % (size_t)2) != 0)
    {
        do
        {
            #ifdef USE_XOSHIRO128
            rnd11 = xoshiro128pp(state);
            rnd12 = xoshiro128pp(state);
            rnd21 = xoshiro128pp(state);
            rnd22 = xoshiro128pp(state);
            #else
            rnd1 = xoshiro256pp(state);
            rnd2 = xoshiro256pp(state);
            #endif

            #if defined(DBL_MANT_DIG) && (DBL_MANT_DIG == 53) &&(FLT_RADIX == 2)
            #ifdef USE_XOSHIRO128
            if (is_little_endian) {
                rnd12 = rnd12 & two21_i;
                rnd22 = rnd22 & two21_i;
            } else {
                rnd11 = rnd11 & two21_i;
                rnd21 = rnd21 & two21_i;
            }
            memcpy((char*)&rnd1, &rnd11, sizeof(uint32_t));
            memcpy((char*)&rnd1 + sizeof(uint32_t), &rnd12, sizeof(uint32_t));
            memcpy((char*)&rnd2, &rnd21, sizeof(uint32_t));
            memcpy((char*)&rnd2 + sizeof(uint32_t), &rnd22, sizeof(uint32_t));
            u = ldexp((double)rnd1, -53);
            v = ldexp((double)rnd2, -53);
            #else
            u = ldexp((double)(rnd1 & two53_i), -53);
            v = ldexp((double)(rnd2 & two53_i), -53);
            #endif
            #else
            u = (double)rnd1 / (double)UINT64_MAX;
            v = (double)rnd2 / (double)UINT64_MAX;
            #endif
        }
        while (u == 0 || v == 0);

        u = sqrt(-2. * log(u));
        seq[n - (size_t)1] = (real_t)ldexp(cos(twoPI * v) * u, -7);
    }
}
#else
void rnorm_xoshiro(float *seq, const size_t n, rng_state_t state[4])
{
    const uint32_t two25_i = (UINT32_C(1) << 25) - UINT32_C(1);
    const int32_t two24_i = (UINT32_C(1) << 24);
    uint32_t rnd1, rnd2;
    #ifndef USE_XOSHIRO128
    uint64_t rnd0;
    #endif
    float u, v, s;
    size_t n_ = n / (size_t)2;
    for (size_t ix = 0; ix < n_; ix++)
    {
        do
        {
            #ifdef USE_XOSHIRO128
            rnd1 = xoshiro128pp(state);
            rnd2 = xoshiro128pp(state);
            #else
            rnd0 = xoshiro256pp(state);
            memcpy(&rnd1, (char*)&rnd0, sizeof(uint32_t));
            memcpy(&rnd2, (char*)&rnd0 + sizeof(uint32_t), sizeof(uint32_t));
            #endif

            #if defined(FLT_MANT_DIG) && (FLT_MANT_DIG == 24) &&(FLT_RADIX == 2)
            u = ldexpf((float)((int32_t)(rnd1 & two25_i) - two24_i), -24);
            v = ldexpf((float)((int32_t)(rnd2 & two25_i) - two24_i), -24);
            #else
            u = (float)rnd1 / (float)INT32_MAX;
            v = (float)rnd2 / (float)INT32_MAX;
            #endif

            s = square(u) + square(v);
        }
        while (s == 0 || s >= 1);

        s = sqrtf((-2.0f / s) * logf(s));
        seq[(size_t)2*ix] = ldexpf(u * s, -7);
        seq[(size_t)2*ix + (size_t)1] = ldexpf(v * s, -7);
    }

    if ((n % (size_t)2) != 0)
    {
        do
        {
            #ifdef USE_XOSHIRO128
            rnd1 = xoshiro128pp(state);
            rnd2 = xoshiro128pp(state);
            #else
            rnd0 = xoshiro256pp(state);
            memcpy(&rnd1, (char*)&rnd0, sizeof(uint32_t));
            memcpy(&rnd2, (char*)&rnd0 + sizeof(uint32_t), sizeof(uint32_t));
            #endif

            #if defined(FLT_MANT_DIG) && (FLT_MANT_DIG == 24) &&(FLT_RADIX == 2)
            u = ldexpf((float)((int32_t)(rnd1 & two25_i) - two24_i), -24);
            v = ldexpf((float)((int32_t)(rnd2 & two25_i) - two24_i), -24);
            #else
            u = (float)rnd1 / (float)INT32_MAX;
            v = (float)rnd2 / (float)INT32_MAX;
            #endif

            s = square(u) + square(v);
        }
        while (s == 0 || s >= 1);

        s = sqrtf((-2.0f / s) * logf(s));
        seq[n - (size_t)1] = ldexpf(u * s, -7);
    }
}
#endif

void seed_state(int_t seed, rng_state_t state[4])
{
    #ifdef USE_XOSHIRO128
    uint64_t s1 = splitmix64(seed);
    uint64_t s2 = splitmix64(s1);
    memcpy(state, &s1, sizeof(uint64_t));
    memcpy(&state[2], &s2, sizeof(uint64_t));
    #else
    state[0] = splitmix64(seed);
    state[1] = splitmix64(state[0]);
    state[2] = splitmix64(state[1]);
    state[3] = splitmix64(state[2]);
    #endif
}

void fill_rnorm_buckets
(
    const size_t n_buckets, real_t *arr, const size_t n,
    real_t **ptr_bucket, size_t *sz_bucket, const size_t BUCKET_SIZE
)
{
    if (n_buckets == 0 || n == 0) return;
    for (size_t bucket = 0; bucket < n_buckets; bucket++)
    {
        ptr_bucket[bucket] = arr;
        arr += BUCKET_SIZE;
    }
    sz_bucket[n_buckets-(size_t)1] = n - BUCKET_SIZE*(n_buckets-(size_t)1);
}

void rnorm_singlethread(ArraysToFill arrays, rng_state_t state[4])
{
    if (arrays.sizeA)
        rnorm_xoshiro(arrays.A, arrays.sizeA, state);
    if (arrays.sizeB)
        rnorm_xoshiro(arrays.B, arrays.sizeB, state);
}

/* This function generates random normal numbers in parallel, but dividing the
   arrays to fill into buckets of up to 250k each. It uses the jumping technique
   from the Xorshiro family in order to ensure that the generated numbers will
   not overlap. */
int_t rnorm_parallel(ArraysToFill arrays, int_t seed, int nthreads)
{
    #ifdef USE_R_RNG
    GetRNGstate();
    for (size_t ix = 0; ix < arrays.sizeA; ix++)
        arrays.A[ix] = norm_rand();
    for (size_t ix = 0; ix < arrays.sizeB; ix++)
        arrays.B[ix] = norm_rand();
    PutRNGstate();
    return 0;
    #endif
    
    const size_t BUCKET_SIZE = (size_t)250000;
    rng_state_t initial_state[4];
    seed_state(seed, initial_state);
    if (arrays.sizeA + arrays.sizeB < BUCKET_SIZE)
    {
        rnorm_singlethread(arrays, initial_state);
        return 0;
    }

    const size_t buckA  = arrays.sizeA  / BUCKET_SIZE + (arrays.sizeA %  BUCKET_SIZE) != 0;
    const size_t buckB  = arrays.sizeB  / BUCKET_SIZE + (arrays.sizeB %  BUCKET_SIZE) != 0;
    const size_t tot_buckets = buckA + buckB;

    real_t **ptr_bucket = (real_t**)malloc(tot_buckets*sizeof(real_t*));
    size_t *sz_bucket   =  (size_t*)malloc(tot_buckets*sizeof(size_t));
    rng_state_t *states = (rng_state_t*)malloc((size_t)4*tot_buckets*sizeof(rng_state_t));

    if (ptr_bucket == NULL || sz_bucket == NULL || states == NULL)
    {
        free(ptr_bucket);
        free(sz_bucket);
        free(states);
        return 1;
    }

    for (size_t ix = 0; ix < tot_buckets; ix++)
        sz_bucket[ix] = BUCKET_SIZE;

    memcpy(states, initial_state, 4*sizeof(rng_state_t));
    for (size_t ix = 1; ix < tot_buckets; ix++)
    {
        memcpy(states + (size_t)4*ix, states + (size_t)4*(ix-(size_t)1), 4*sizeof(rng_state_t));
        #ifdef USE_XOSHIRO128
        xoshiro128pp_jump(states + 4*ix);
        #else
        xoshiro256pp_jump(states + 4*ix);
        #endif
    }

    real_t ** const ptr_bucket_ = ptr_bucket;
    size_t *  const sz_bucket_  = sz_bucket;
    
    fill_rnorm_buckets(
        buckA, arrays.A, arrays.sizeA,
        ptr_bucket, sz_bucket, BUCKET_SIZE
    );
    ptr_bucket += buckA; sz_bucket += buckA;
    fill_rnorm_buckets(
        buckB, arrays.B, arrays.sizeB,
        ptr_bucket, sz_bucket, BUCKET_SIZE
    );

    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif

    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(states)
    for (size_t_for ix = 0; ix < tot_buckets; ix++)
    {
        rng_state_t state[] = {states[(size_t)4*ix],
                               states[(size_t)4*ix + (size_t)1],
                               states[(size_t)4*ix + (size_t)2],
                               states[(size_t)4*ix + (size_t)3]};
        rnorm_xoshiro(ptr_bucket_[ix], sz_bucket_[ix], state);
    }

    free(ptr_bucket_);
    free(sz_bucket_);
    free(states);
    return 0;
}

void reduce_mat_sum(real_t *restrict outp, size_t lda, real_t *restrict inp,
                    int_t m, int_t n, int nthreads)
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

void exp_neg_x(real_t *restrict arr, size_t n, int nthreads)
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

void add_to_diag(real_t *restrict A, real_t val, size_t n)
{
    for (size_t ix = 0; ix < n; ix++)
        A[ix + ix*n] += val;
}

real_t sum_sq_div_w(real_t *restrict arr, real_t *restrict w, size_t n, bool compensated, int nthreads)
{
    real_t res = 0;
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
    int_t m, int_t n, real_t alpha,
    size_t indptr[], int_t indices[], real_t values[],
    real_t DenseMat[], size_t ldb,
    real_t OutputMat[], size_t ldc,
    int nthreads
)
{
    if (m <= 0 || indptr[0] == indptr[m])
        return;
    real_t *row_ptr;
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long row;
    #endif

    if (alpha != 1.)
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                shared(m, n, alpha, ldb, ldc, OutputMat, DenseMat, indptr, indices, values) \
                private(row_ptr)
        for (size_t_for row = 0; row < (size_t)m; row++) {
            row_ptr = OutputMat + row*ldc;
            for (size_t col = indptr[row]; col < indptr[row+1]; col++)
                cblas_taxpy(n, alpha*values[col], DenseMat + (size_t)indices[col]*ldb, 1, row_ptr, 1);
        }
    else
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                shared(m, n, ldb, ldc, OutputMat, DenseMat, indptr, indices, values) \
                private(row_ptr)
        for (size_t_for row = 0; row < (size_t)m; row++) {
            row_ptr = OutputMat + row*ldc;
            for (size_t col = indptr[row]; col < indptr[row+1]; col++)
                cblas_taxpy(n, values[col], DenseMat + (size_t)indices[col]*ldb, 1, row_ptr, 1);
        }
}

/* x <- alpha*t(A)*v + x | A[m,n] is dense, v[m] is sparse, x[n] is dense */
void tgemv_dense_sp
(
    int_t m, int_t n,
    real_t alpha, real_t DenseMat[], size_t lda,
    int_t ixB[], real_t vec_sp[], size_t nnz,
    real_t OutputVec[]
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
    int_t m, int_t n,
    real_t alpha[], real_t DenseMat[], size_t lda,
    int_t ixB[], real_t vec_sp[], size_t nnz,
    real_t OutputVec[]
)
{
    for (size_t ix = 0; ix < nnz; ix++)
        cblas_taxpy(n, alpha[ix]*vec_sp[ix], DenseMat + (size_t)ixB[ix]*lda, 1, OutputVec, 1);
}

/* Same, but with both array of weights and scalar weight */
void tgemv_dense_sp_weighted2
(
    int_t m, int_t n,
    real_t alpha[], real_t alpha2, real_t DenseMat[], size_t lda,
    int_t ixB[], real_t vec_sp[], size_t nnz,
    real_t OutputVec[]
)
{
    for (size_t ix = 0; ix < nnz; ix++)
        cblas_taxpy(n, alpha2*alpha[ix]*vec_sp[ix], DenseMat + (size_t)ixB[ix]*lda, 1, OutputVec, 1);
}

void tgemv_dense_sp_notrans
(
    int_t m, int_t n,
    real_t DenseMat[], int_t lda,
    int_t ixB[], real_t vec_sp[], size_t nnz,
    real_t OutputVec[]
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
    int_t m, int_t n,
    real_t *restrict A, int_t lda,
    real_t *restrict B, int_t ldb
)
{
    char uplo = '?';
    if (m == 0 && n == 0) return;

    if (ldb == n && lda == n)
        memcpy(B, A, (size_t)m*(size_t)n*sizeof(real_t));
    else
        tlacpy_(&uplo, &n, &m, A, &lda, B, &ldb);
}

/* B[:m,:n] = A[:m,:n] + B[:m,:n] */
void sum_mat
(
    size_t m, size_t n,
    real_t *restrict A, size_t lda,
    real_t *restrict B, size_t ldb
)
{
    /* Note1: do NOT change this to axpy, it gets a huge slow-down when
       used with MKL for some reason. OpenBLAS still works fine though */
    /* Note2: in most cases it is expected that m >> n */
    for (size_t row = 0; row < m; row++)
        for (size_t col = 0; col < n; col++)
            B[col + row*ldb] += A[col + row*lda];
}

void transpose_mat(real_t *restrict A, size_t m, size_t n, real_t *restrict buffer_real_t)
{
    memcpy(buffer_real_t, A, m*n*sizeof(real_t));
    for (size_t row = 0; row < m; row++)
        for (size_t col = 0; col < n; col++)
            A[row + col*m] = buffer_real_t[col + row*n];
}

void transpose_mat2(real_t *restrict A, size_t m, size_t n, real_t *restrict outp)
{
    for (size_t row = 0; row < m; row++)
        for (size_t col = 0; col < n; col++)
            outp[row + col*m] = A[col + row*n];
}

void transpose_mat3
(
    real_t *restrict A, size_t lda,
    size_t m, size_t n,
    real_t *restrict outp, size_t ldb
)
{
    for (size_t row = 0; row < m; row++)
        for (size_t col = 0; col < n; col++)
            outp[row + col*ldb] = A[col + row*lda];
}

int_t coo_to_csr_plus_alloc
(
    int_t *restrict Xrow, int_t *restrict Xcol, real_t *restrict Xval,
    real_t *restrict W,
    int_t m, int_t n, size_t nnz,
    size_t *restrict *csr_p, int_t *restrict *csr_i, real_t *restrict *csr_v,
    real_t *restrict *csr_w
)
{
    *csr_p = (size_t*)malloc(((size_t)m+(size_t)1)*sizeof(size_t));
    *csr_i = (int_t*)malloc(nnz*sizeof(int_t));
    *csr_v = (real_t*)malloc(nnz*sizeof(real_t));
    if (*csr_p == NULL || *csr_i == NULL || *csr_v == NULL)
        return 1;

    if (W != NULL) {
        *csr_w = (real_t*)malloc(nnz*sizeof(real_t));
        if (*csr_w == NULL) return 1;
    }

    coo_to_csr(
        Xrow, Xcol, Xval,
        W,
        m, n, nnz,
        *csr_p, *csr_i, *csr_v,
        (W == NULL)? ((real_t*)NULL) : (*csr_w)
    );
    return 0;
}

void coo_to_csr
(
    int_t *restrict Xrow, int_t *restrict Xcol, real_t *restrict Xval,
    real_t *restrict W,
    int_t m, int_t n, size_t nnz,
    size_t *restrict csr_p, int_t *restrict csr_i, real_t *restrict csr_v,
    real_t *restrict csr_w
)
{
    bool has_mem = true;
    int_t *cnt_byrow = NULL;

    produce_p:
    {
        memset(csr_p, 0, ((size_t)m+(size_t)1)*sizeof(size_t));
        for (size_t ix = 0; ix < nnz; ix++)
            csr_p[Xrow[ix]+(size_t)1]++;
        for (int_t row = 0; row < m; row++)
            csr_p[row+(size_t)1] += csr_p[row];
    }

    if (!has_mem) goto cleanup;

    cnt_byrow = (int_t*)calloc(m, sizeof(int_t));

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
    int_t *restrict Xrow, int_t *restrict Xcol, real_t *restrict Xval,
    real_t *restrict W, int_t m, int_t n, size_t nnz,
    size_t *restrict csr_p, int_t *restrict csr_i, real_t *restrict csr_v,
    size_t *restrict csc_p, int_t *restrict csc_i, real_t *restrict csc_v,
    real_t *restrict csr_w, real_t *restrict csc_w,
    int nthreads
)
{
    bool has_mem = true;
    nthreads = (nthreads > 2)? 2 : 1;
    int_t *cnt_byrow = NULL;
    int_t *cnt_bycol = NULL;

    produce_p:
    {
        memset(csr_p, 0, ((size_t)m+(size_t)1)*sizeof(size_t));
        memset(csc_p, 0, ((size_t)n+(size_t)1)*sizeof(size_t));
        for (size_t ix = 0; ix < nnz; ix++) {
            csr_p[Xrow[ix]+(size_t)1]++;
            csc_p[Xcol[ix]+(size_t)1]++;
        }
        for (int_t row = 0; row < m; row++)
            csr_p[row+(size_t)1] += csr_p[row];
        for (int_t col = 0; col < n; col++)
            csc_p[col+(size_t)1] += csc_p[col];
    }


    if (!has_mem) goto cleanup;

    cnt_byrow = (int_t*)calloc(m, sizeof(int_t));
    cnt_bycol = (int_t*)calloc(n, sizeof(int_t));

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
                        csr_w[csr_p[Xrow[ix]] + cnt_byrow[Xrow[ix]]] = W[ix];
                        csr_v[csr_p[Xrow[ix]] + cnt_byrow[Xrow[ix]]] = Xval[ix];
                        csr_i[csr_p[Xrow[ix]] + cnt_byrow[Xrow[ix]]++] = Xcol[ix];
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

void row_means_csr(size_t indptr[], real_t *restrict values,
                   real_t *restrict output, int_t m, int nthreads)
{
    int_t row = 0;
    set_to_zero(values, m);
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
            shared(indptr, values, output, m)
    for (row = 0; row < m; row++)
    {
        double rsum = 0;
        for (size_t ix = indptr[row]; ix < indptr[row+(size_t)1]; ix++)
            rsum += values[ix];
        output[row] = rsum;
    }
    nthreads = cap_to_4(nthreads);
    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(indptr, output, m)
    for (row = 0; row < m; row++)
        output[row] /= (real_t)(indptr[row+(size_t)1] - indptr[row]);
}

bool should_stop_procedure = false;
bool handle_is_locked = false;
void set_interrup_global_variable(int_t s)
{
    #pragma omp critical
    {
        should_stop_procedure = true;
    }
}

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
)
{
    ((data_collective_fun_grad*)instance)->niter = k;
    int_t print_every = ((data_collective_fun_grad*)instance)->print_every;
    if ((k % print_every) == 0 && print_every > 0) {
        printf("Iteration %-4d - f(x)= %-8.03g - ||g(x)||= %-8.03g - ls=% 2d\n",
               k, fx, gnorm, ls);
        fflush(stdout);
    }
    if (should_stop_procedure)
        return 1;
    return 0;
}

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
)
{
    ((data_offsets_fun_grad*)instance)->niter = k;
    int_t print_every = ((data_offsets_fun_grad*)instance)->print_every;
    if ((k % print_every) == 0 && print_every > 0) {
        printf("Iteration %-5d - f(x)= %-8.03g - ||g(x)||= %-8.03g - ls=% 2d\n",
               k, fx, gnorm, ls);
        fflush(stdout);
    }
    if (should_stop_procedure)
        return 1;
    return 0;
}

bool check_is_sorted(int_t arr[], int_t n)
{
    if (n <= 1) return true;
    for (int_t ix = 0; ix < n-1; ix++)
        if (arr[ix] > arr[ix+1]) return false;
    return true;
}

/* https://www.stat.cmu.edu/~ryantibs/median/quickselect.c */
/* Some sample C code for the quickselect algorithm, 
   taken from Numerical Recipes in C. */
#define SWAP(a,b) temp=(a);(a)=(b);(b)=temp;
void qs_argpartition(int_t arr[], real_t values[], int_t n, int_t k)
{
    int_t i,ir,j,l,mid;
    int_t a,temp;

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
    real_t *restrict orig, size_t m, size_t n,
    real_t *restrict outp
)
{
    copy_mat(m, n,
             orig, n,
             outp, n+1);
    for (size_t ix = 0; ix < m; ix++)
        outp[n + ix*(n+(size_t)1)] = 1.;
}

void fill_lower_triangle(real_t A[], size_t n, size_t lda)
{
    for (size_t row = 1; row < n; row++)
        for (size_t col = 0; col < row; col++)
            A[col + row*lda] = A[row + col*lda];
}

void print_err_msg(const char *msg)
{
    #ifndef _FOR_R
    fprintf(stderr, "%s", msg);
    #else
    fprintf(stderr, msg);
    #endif
    fflush(stderr);
}

void print_oom_message(void)
{
    print_err_msg("Error: could not allocate enough memory.\n");
}

#ifdef _FOR_PYTHON
#define PY_MSG_MAX_LENGTH 256
void py_printf(const char *fmt, ...)
{
    char msg[PY_MSG_MAX_LENGTH];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, PY_MSG_MAX_LENGTH, fmt, args);
    va_end(args);
    cy_printf(msg);
}

void py_errprintf(void *ignored, const char *fmt, ...)
{
    char msg[PY_MSG_MAX_LENGTH];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, PY_MSG_MAX_LENGTH, fmt, args);
    va_end(args);
    cy_errprintf(msg);
}

void python_printmsg(char *msg)
{
    PySys_WriteStdout("%s", msg);
}

void python_printerrmsg(char *msg)
{
    PySys_WriteStderr("%s", msg);
}
#endif

void act_on_interrupt(int retval, bool handle_interrupt, bool print_msg)
{
    if (retval == 3)
    {
        if (print_msg)
            print_err_msg(" Error: procedure was interrupted.\n");
        if (!handle_interrupt)
            raise(SIGINT);
    }
}

#ifdef _FOR_R
void R_nan_to_C_nan(real_t arr[], size_t n)
{
    for (size_t ix = 0; ix < n; ix++)
        arr[ix] = ISNAN(arr[ix])? NAN : arr[ix];
}
#endif

long double compensated_sum(real_t *arr, size_t n)
{
    long double err = 0.;
    long double diff = 0.;
    long double temp;
    long double res = 0;

    for (size_t ix = 0; ix < n; ix++)
    {
        diff = arr[ix] - err;
        temp = res + diff;
        err = (temp - res) - diff;
        res = temp;
    }

    return res;
}

long double compensated_sum_product(real_t *restrict arr1, real_t *restrict arr2, size_t n)
{
    long double err = 0.;
    long double diff = 0.;
    long double temp;
    long double res = 0;

    for (size_t ix = 0; ix < n; ix++)
    {
        diff = fmal(arr1[ix], arr2[ix], -err);
        temp = res + diff;
        err = (temp - res) - diff;
        res = temp;
    }

    return res;
}

#ifdef AVOID_BLAS_SYR
/* https://github.com/xianyi/OpenBLAS/issues/3237 */
void custom_syr(const int_t n, const real_t alpha, const real_t *restrict x, real_t *restrict A, const int_t lda)
{
    real_t temp;
    real_t *restrict Arow;
    for (int i = 0; i < n; i++) {
        temp = alpha*x[i];
        Arow = A + (size_t)i*(size_t)lda;
        for (int j = i; j < n; j++)
            Arow[j] = fma_t(temp, x[j], Arow[j]);
    }
}
#endif

void set_blas_threads(int nthreads_set, int *nthreads_curr)
{
    #ifdef _FOR_R
    /* https://gist.github.com/KRD1/2503984 */
    if (!has_RhpcBLASctl || ptr_glob_lst == NULL || ptr_nthreads == NULL)
        return;
    int errinfo = 0;
    if (nthreads_curr != NULL) {
        SEXP nthreads_curr_R = R_tryEvalSilent(VECTOR_ELT(*ptr_glob_lst, 5),
                                               VECTOR_ELT(*ptr_glob_lst, 0),
                                               &errinfo);
        if (!errinfo) {
            *nthreads_curr = Rf_asInteger(nthreads_curr_R);
        }
        *nthreads_curr = max2(*nthreads_curr, 1);
    }
    *ptr_nthreads = nthreads_set;
    errinfo = 0;
    R_tryEvalSilent(VECTOR_ELT(*ptr_glob_lst, 4),
                    VECTOR_ELT(*ptr_glob_lst, 0),
                    &errinfo);
    

    #elif defined(_FOR_PYTHON) && !defined(IS_PY_TEST)
    if (nthreads_curr != NULL) {
        *nthreads_curr = py_get_threads();
    }
    py_set_threads(nthreads_set);
    #if defined(HAS_OPENBLAS)
    openblas_set_num_threads(nthreads_set);
    #endif
    

    #elif defined(HAS_OPENBLAS)
    if (nthreads_curr != NULL) {
        *nthreads_curr = openblas_get_num_threads();
        *nthreads_curr = max2(*nthreads_curr, 1);
    }
    openblas_set_num_threads(nthreads_set);
    

    #elif defined(_OPENMP) && !defined(MKL_H) && !defined(HAS_MKL)
    if (nthreads_curr != NULL) {
        *nthreads_curr = omp_get_num_threads();
        *nthreads_curr = max2(*nthreads_curr, 1);
    }
    omp_set_num_threads(nthreads_set);
    #endif
}


#if defined(_FOR_R) && defined(WRAPPED_GELSD) && !defined(USE_FLOAT)
SEXP wrapper_GELSD(void *data)
{
    Args_to_GELSD *data_ = (Args_to_GELSD*)data;
    tgelsd_(data_->m, data_->n, data_->nrhs,
            data_->A, data_->lda, data_->B, data_->ldb,
            data_->S, data_->rcond, data_->rank,
            data_->work, data_->lwork, data_->iwork,
            data_->info);
    return R_NilValue;
}

void clean_after_GELSD(void *cdata, Rboolean jump)
{
    if (jump)
    {
        PointersToFree *cdata_ = (PointersToFree*)cdata;
        for (size_t ix = 0; ix < cdata_->n_pointers; ix++)
            free(cdata_->pointers[ix]);
        GELSD_free_inputs = false;
    }
}
#endif

bool get_has_openmp(void)
{
    #ifdef _OPENMP
    return true;
    #else
    return false;
    #endif
}
