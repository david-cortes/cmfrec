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

    Copyright (c) 2020-2022 David Cortes

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
    int i = 0;
    if (nthreads > 1 && n > (size_t)1e8)
    {
        #pragma omp parallel for schedule(static, 1) \
                firstprivate(arr, chunk_size, nthreads) num_threads(nthreads)
        for (i = 0; i < nthreads; i++)
            memset(arr + (size_t)i * chunk_size, 0, chunk_size*sizeof(real_t));
        if (remainder > 0)
            memset(arr + (size_t)nthreads * (size_t)chunk_size, 0, remainder*sizeof(real_t));
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
        int i = 0;

        #pragma omp parallel for schedule(static, 1) \
                firstprivate(src, dest, chunk_size, nthreads) num_threads(nthreads)
        for (i = 0; i < nthreads; i++)
            memcpy(dest + (size_t)i * (size_t)chunk_size, src + (size_t)i * chunk_size, chunk_size*sizeof(real_t));
        if (remainder > 0)
            memcpy(dest + (size_t)nthreads*chunk_size, src + (size_t)nthreads*chunk_size, remainder*sizeof(real_t));
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
#ifdef __GNUC__
__attribute__((always_inline))
#endif
static inline uint64_t rotl64(const uint64_t x, const int k)
{
    return (x << k) | (x >> (64 - k));
}

#ifdef __GNUC__
__attribute__((always_inline))
#endif
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

#ifdef __GNUC__
__attribute__((always_inline))
#endif
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

#ifdef __GNUC__
__attribute__((always_inline))
#endif
static inline uint32_t rotl32(const uint32_t x, const int k)
{
    return (x << k) | (x >> (32 - k));
}

#ifdef __GNUC__
__attribute__((always_inline))
#endif
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

#ifdef __GNUC__
__attribute__((always_inline))
#endif
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

/* Note: this samples from a **truncated** normal distribution, excluding
   the ~0.7% most extreme values. It does so by using the ziggurat method,
   but instead of sampling from the tails when needed, it simply restarts
   so as to never give extreme values. This is only used for initialization
   of parameters so extreme values are not desired here. As a bonus, this
   makes the procedure much faster than a proper ziggurat from a full
   normal distribution. The tables are pre-calculated and taken from NumPy.

   Note that this involves drawing random numbers ~U(0,1). A naive procedure
   would just draw an integer and divide by the maximum, but the interval is
   not evenly representable as floating point numbers, so one may instead divide
   it in chunks of length 2^-53 to represent the mantissa. If one instead wants
   a number in a closed (0,1) interval (excluding both endpoints), it's possible
   to generate one by using 52-bits of mantissa (an integer in the range
   [0,2^52-1], including both endpoints), adding +0.5 (or 2^-1, another bit),
   and then dividing by 2^52 (can do the math with all numbers expressed as
   power of 2 to make it easier to see).

   Given that a single random draw involves 64-bits, it's possible to choose the
   rectangle (last 8 bits, for 2^8-1=255 rectangles), the sign (next 1 bit), and
   the uniformly-drawn real from the same random 64-bit random draw by taking it
   in chunks for each required part.  */
#include "ziggurat.h"
#if defined(USE_DOUBLE) || !defined(USE_FLOAT)
void rnorm_xoshiro(double *seq, const size_t n, rng_state_t state[4])
{
    uint64_t rnd;
    uint8_t rectangle;
    uint8_t sign;
    double rnorm;
    double runif;

    #ifdef USE_XOSHIRO128
    uint32_t rnd1, rnd2;
    #endif
    
    size_t ix = 0;
    while (ix < n)
    {
        #ifndef USE_XOSHIRO128
        rnd = xoshiro256pp(state);
        #else
        rnd1 = xoshiro128pp(state);
        rnd2 = xoshiro128pp(state);
        memcpy((char*)&rnd, &rnd1, sizeof(uint32_t));
        memcpy((char*)&rnd + sizeof(uint32_t), &rnd2, sizeof(uint32_t));
        #endif
        rectangle = rnd & 255; /* <- number of rectangles (took 8 bits) */
        rnd >>= 8;
        sign = rnd & 1; /* <- took 1 bit */
        /* there's currently 56 bits left, already used 1 for the sign, need to
           take 52 for for the uniform draw, so can chop off 3 more than what
           was taken to get there faster. */
        rnd >>= 4;
        rnorm = rnd * wi_double[rectangle];
        if (rnd < ki_double[rectangle])
        {
            seq[ix++] = sign? rnorm : -rnorm;
        }

        else if (rectangle != 0)
        {
            #ifndef USE_XOSHIRO128
            rnd = xoshiro256pp(state);
            #else
            rnd1 = xoshiro128pp(state);
            rnd2 = xoshiro128pp(state);
            memcpy((char*)&rnd, &rnd1, sizeof(uint32_t));
            memcpy((char*)&rnd + sizeof(uint32_t), &rnd2, sizeof(uint32_t));
            #endif
            #ifdef SUPPORTS_HEXFLOAT
            runif = ((double)(rnd  >> 12) + 0.5) * 0x1.0p-52;
            #else
            runif = ((double)(rnd >> 12) + 0.5);
            runif = ldexp(runif, -52);
            #endif
            if (runif * (fi_double[rectangle-1] - fi_double[rectangle])
                    <
                exp(-0.5 * rnorm * rnorm) - fi_double[rectangle])
            {
                seq[ix++] = sign? rnorm : -rnorm;
            }
        }
    }

    #ifdef SUPPORTS_HEXFLOAT
    for (size_t ix = 0; ix < n; ix++)
        seq[ix] *= 0x1.0p-7;
    #else
    for (size_t ix = 0; ix < n; ix++)
        seq[ix] = ldexp(seq[ix], -7);
    #endif
}

void runif_xoshiro(double *seq, const size_t n, rng_state_t state[4])
{
    #ifndef USE_XOSHIRO128
    #ifdef SUPPORTS_HEXFLOAT
    for (size_t ix = 0; ix < n; ix++)
        seq[ix] = ((double)(xoshiro256pp(state) >> 12) + 0.5) * 0x1.0p-59;
    #else
    for (size_t ix = 0; ix < n; ix++)
        seq[ix] = ldexp((double)(xoshiro256pp(state) >> 12) + 0.5, -59);
    #endif
    #else
    uint64_t rnd;
    uint32_t rnd1, rnd2;
    for (size_t ix = 0; ix < n; ix++) {
        rnd1 = xoshiro128pp(state);
        rnd2 = xoshiro128pp(state);
        memcpy((char*)&rnd, &rnd1, sizeof(uint32_t));
        memcpy((char*)&rnd + sizeof(uint32_t), &rnd2, sizeof(uint32_t));
        #ifdef SUPPORTS_HEXFLOAT
        seq[ix] = ((double)(rnd >> 12) + 0.5) * 0x1.0p-59;
        #else
        seq[ix] = ldexp((double)(rnd >> 12) + 0.5, 59);
        #endif
    }
    #endif
}
#else
void rnorm_xoshiro(float *seq, const size_t n, rng_state_t state[4])
{
    uint32_t rnd;
    uint8_t sign;
    uint8_t rectangle;
    float rnorm;
    float runif;

    #ifndef USE_XOSHIRO128
    uint64_t rnd_big;
    bool reuse_draw = false;
    #endif

    size_t ix = 0;
    while (ix < n)
    {
        #ifndef USE_XOSHIRO128
        if (reuse_draw) {
            reuse_draw = false;
            rnd = rnd_big;
        }
        else {
            rnd_big = xoshiro256pp(state);
            reuse_draw = true;
            rnd = rnd_big & 0xffffffff;
            rnd_big >>= 32;
        }
        #else
        rnd = xoshiro128pp(state);
        #endif

        rectangle = rnd & 255; /* <- number of rectangles (took 8 bits) */
        rnd >>= 8;
        sign = rnd & 1; /* <- took 1 bit */
        rnd >>= 1;

        /* For 32-bit floats, mantissa is 24 bits, so we need 23 bits to get
           a closed-interval uniform. We drew 32 bits, took 8 for the rectangle,
           and 1 for the sign, leaving exactly 23 bits, so no need to chop off
           more of them like for float-64. */
        
        rnorm = rnd * wi_float[rectangle];
        if (rnd < ki_float[rectangle])
        {
            seq[ix++] = sign? rnorm : -rnorm;
        }

        else
        {
            #ifndef USE_XOSHIRO128
            if (reuse_draw) {
                reuse_draw = false;
                rnd = rnd_big;
            }
            else {
                rnd_big = xoshiro256pp(state);
                reuse_draw = true;
                rnd = rnd_big & 0xffffffff;
                rnd_big >>= 32;
            }
            #else
            rnd = xoshiro128pp(state);
            #endif
            
            #ifdef SUPPORTS_HEXFLOAT
            runif = ((float)(rnd >> 9) + 0.5f) * 0x1.0p-23f;
            #else
            runif = ((float)(rnd  >> 9) + 0.5f);
            runif = ldexpf(runif, -23);
            #endif
            if (runif * (fi_float[rectangle-1] - fi_float[rectangle])
                    <
                expf(-0.5f * rnorm * rnorm) - fi_float[rectangle])
            {
                seq[ix++] = sign? rnorm : -rnorm;
            }
        }
    }

    #ifdef SUPPORTS_HEXFLOAT
    for (size_t ix = 0; ix < n; ix++)
        seq[ix] *= 0x1.0p-7f;
    #else
    for (size_t ix = 0; ix < n; ix++)
        seq[ix] = ldexpf(seq[ix], -7);
    #endif
}

void runif_xoshiro(float *seq, const size_t n, rng_state_t state[4])
{
    #ifndef USE_XOSHIRO128
    uint64_t rnd;
    size_t ix_;
    size_t lim = n >> 1;
    for (size_t ix = 0; ix < lim; ix++)
    {
        rnd = xoshiro256pp(state);
        ix_ = ix << 1;
        #ifdef SUPPORTS_HEXFLOAT
        seq[ix_] = ((float)(rnd & 0x7fffff) + 0.5f) * 0x1.0p-30f;
        seq[ix_ + 1] = ((float)(rnd >> 41) + 0.5f) * 0x1.0p-30f;
        #else
        seq[ix_] = ldexpf((float)(rnd & 0x7fffff) + 0.5f, -30);
        seq[ix_ + 1] = ldexpf((float)(rnd >> 41) + 0.5f, -30);
        #endif
    }
    if ((lim << 1) < n)
    {
        rnd = xoshiro256pp(state);
        #ifdef SUPPORTS_HEXFLOAT
        seq[lim-1] = ((float)(rnd & 0x7fffff) + 0.5f) * 0x1.0p-30f;
        #else
        seq[lim-1] = ldexpf((float)(rnd & 0x7fffff) + 0.5f, -30);
        #endif
    }
    #else
    #ifdef SUPPORTS_HEXFLOAT
    for (size_t ix = 0; ix < n; ix++)
        seq[ix] = ((float)(xoshiro128pp(state) >> 9) + 0.5f) * 0x1.0p-30f;
    #else
    for (size_t ix = 0; ix < n; ix++)
        seq[ix] = ldexpf((float)(xoshiro128pp(state) >> 9) + 0.5f, -30);
    #endif
    #endif
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

void runif_singlethread(ArraysToFill arrays, rng_state_t state[4])
{
    if (arrays.sizeA)
        runif_xoshiro(arrays.A, arrays.sizeA, state);
    if (arrays.sizeB)
        runif_xoshiro(arrays.B, arrays.sizeB, state);
}

/* This function generates random normal numbers in parallel, but dividing the
   arrays to fill into buckets of up to 250k each. It uses the jumping technique
   from the Xorshiro family in order to ensure that the generated numbers will
   not overlap. */
int_t random_parallel(ArraysToFill arrays, int_t seed, bool normal, int nthreads)
{
    #ifdef USE_R_RNG
    GetRNGstate();
    if (normal)
    {
        #ifdef SUPPORTS_HEXFLOAT
        for (size_t ix = 0; ix < arrays.sizeA; ix++)
            arrays.A[ix] = norm_rand() * 0x1.0p-7;
        for (size_t ix = 0; ix < arrays.sizeB; ix++)
            arrays.B[ix] = norm_rand() * 0x1.0p-7;
        #else
        for (size_t ix = 0; ix < arrays.sizeA; ix++)
            arrays.A[ix] = ldexp(norm_rand(), -7);
        for (size_t ix = 0; ix < arrays.sizeB; ix++)
            arrays.B[ix] = ldexp(norm_rand(), -7);
        #endif
    }

    else
    {
        #ifdef SUPPORTS_HEXFLOAT
        for (size_t ix = 0; ix < arrays.sizeA; ix++)
            arrays.A[ix] = unif_rand() * 0x1.0p-7;
        for (size_t ix = 0; ix < arrays.sizeB; ix++)
            arrays.B[ix] = unif_rand() * 0x1.0p-7;
        #else
        for (size_t ix = 0; ix < arrays.sizeA; ix++)
            arrays.A[ix] = ldexp(unif_rand(), -7);
        for (size_t ix = 0; ix < arrays.sizeB; ix++)
            arrays.B[ix] = ldexp(unif_rand(), -7);
        #endif
    }
    PutRNGstate();
    return 0;
    #endif
    
    const size_t BUCKET_SIZE = (size_t)1 << 18; /* <- a bit over 250k */
    rng_state_t initial_state[4];
    seed_state(seed, initial_state);
    if (arrays.sizeA + arrays.sizeB <= BUCKET_SIZE)
    {
        rnorm_singlethread(arrays, initial_state);
        return 0;
    }

    const size_t buckA  = arrays.sizeA  / BUCKET_SIZE + (arrays.sizeA %  BUCKET_SIZE) != 0;
    const size_t buckB  = arrays.sizeB  / BUCKET_SIZE + (arrays.sizeB %  BUCKET_SIZE) != 0;
    const size_t tot_buckets = max2(1, buckA + buckB);
    /* Note: the condition above is not needed, but GCC12 complains otherwise
       and by extension CRAN complains too */

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
            shared(states, normal)
    for (size_t_for ix = 0; ix < tot_buckets; ix++)
    {
        rng_state_t state[] = {states[(size_t)4*ix],
                               states[(size_t)4*ix + (size_t)1],
                               states[(size_t)4*ix + (size_t)2],
                               states[(size_t)4*ix + (size_t)3]};
        if (normal)
            rnorm_xoshiro(ptr_bucket_[ix], sz_bucket_[ix], state);
        else
            runif_xoshiro(ptr_bucket_[ix], sz_bucket_[ix], state);
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
    #ifndef _WIN32
    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(arr, n)
    #endif
    for (size_t_for ix = 0; ix < n; ix++)
        arr[ix] = exp_t(-arr[ix]);
}

void add_to_diag(real_t *restrict A, real_t val, size_t n)
{
    for (size_t ix = 0; ix < n; ix++)
        A[ix + ix*n] += val;
}

void add_to_diag2(real_t *restrict A, real_t val, size_t n, real_t val_last)
{
    for (size_t ix = 0; ix < n-1; ix++)
        A[ix + ix*n] += val;
    A[square(n)-1] += val_last;
}

void fma_extra(real_t *restrict a, real_t w, real_t *restrict b, int_t n)
{
    #if defined(CLANG_FP_REASSOCIATE) && defined(__clang__)
    #pragma clang fp reassociate(on)
    #endif
    for (int_t ix = 0; ix < n; ix++)
        a[ix] += w * b[ix] * b[ix];
}

void mult2(real_t *restrict out, real_t *restrict a, real_t *restrict b, int_t n)
{
    for (int_t ix = 0; ix < n; ix++)
        out[ix] = a[ix] * b[ix];
}

void recipr(real_t *restrict x, int_t n)
{
    for (int_t ix = 0; ix < n; ix++)
        x[ix] = (real_t)1 / x[ix];
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
    #ifndef _WIN32
    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(arr, w, n) reduction(+:res)
    #endif
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
        #pragma omp parallel for schedule(guided) num_threads(nthreads) \
                shared(m, n, alpha, ldb, ldc, OutputMat, DenseMat, indptr, indices, values) \
                private(row_ptr)
        for (size_t_for row = 0; row < (size_t)m; row++) {
            row_ptr = OutputMat + row*ldc;
            for (size_t col = indptr[row]; col < indptr[row+1]; col++)
                cblas_taxpy(n, alpha*values[col], DenseMat + (size_t)indices[col]*ldb, 1, row_ptr, 1);
        }
    else
        #pragma omp parallel for schedule(guided) num_threads(nthreads) \
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
    copy_mat((int_t)m, (int_t)n,
             orig, (int_t)n,
             outp, (int_t)(n+1));
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

double compensated_sum(real_t *arr, size_t n)
{
    double err = 0.;
    double diff = 0.;
    double temp;
    double res = 0;

    for (size_t ix = 0; ix < n; ix++)
    {
        diff = (double)arr[ix] - err;
        temp = res + diff;
        err = (temp - res) - diff;
        res = temp;
    }

    return res;
}

double compensated_sum_product(real_t *restrict arr1, real_t *restrict arr2, size_t n)
{
    double err = 0.;
    double diff = 0.;
    double temp;
    double res = 0;

    for (size_t ix = 0; ix < n; ix++)
    {
        diff = fma((double)arr1[ix], (double)arr2[ix], -err);
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
    for (int_t i = 0; i < n; i++) {
        temp = alpha*x[i];
        Arow = A + (size_t)i*(size_t)lda;
        for (int_t j = i; j < n; j++)
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
