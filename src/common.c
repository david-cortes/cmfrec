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

/* Note: the descriptions about input parameters of the functions might be
   outdated and might not match with the actual code anymore. */

/*******************************************************************************
    Function and Gradient for cannonical form
    -----------------------------------------

    This function computes the gradients in the cannonical-form problem:
        min  0.5 * scaling * ||M . W . (X - A*t(B) - bias1 - bias2 - mu)||^2
    (Note that it will not add regularization into the formula)
    See the files "collective.c" and "offsets.c" for details on what the
    formula represents.

    The gradients are given as:
        E = scaling * (M . W . (A*t(B) - X))
        grad(bias1) = sum_cols(E)
        grad(bias2) = sum_rows(E)
        grad(A) =   E  * B
        grad(B) = t(E) * A

    Since the formulas here ignore missing entries, when the matrices are
    sparse, the gradients can be calculated more easily as follows:
        grad(A) := 0
        grad(B) := 0
        For i..nnz:
            err = <A[row], B[col]> - X[row,col]
            grad(A[row]) += B[col] * err
            grad(B[col]) += A[row] * err

    In order to speed things up, this loop can be parallelized in two ways:
    - Let each thread/worker have separate matrices grad(A), grad(B), to which
      each adds its own portions based on the non-zero entries it is assigned,
      then sum them up into a common matrix (this part can also be parallelized
      by rows). This approach is the fastest, but letting each thread/worker
      have a full copy of grad(A), grad(B) can require a lot of memory.
    - Have copies of X in CSR and CSC formats, then iterate **twice** over it:
      (a) first by rows (using the CSR matrix), letting each thread write into
          a different row at a time (no race conditions).
      (b) then by columns (using the CSC matrix), letting each thread write into
         a different column at a time.
      This approach is slower, as it requires iterating twice, thus computing
      twice as many dot products, but it requires less memory as each thread
      writes into the final matrices grad(A), grad(B).

    This module implements both approaches, but it is suggested to use the
    first one if memory constraints allow for it.

    If passing a dense matrix X as input, need to pass a temporary buffer
    of dimensions m * n regardless.

    Note that the arrays for the gradients must be passed already initialized:
    that is, they must be set to all-zeros to obtain the full gradient, or to
    the already-obtained gradients from the main factorization if using it for
    the collective model.

    Parameters
    ----------
    A[m * lda]
        Array with the A parameters. Only the portion A(:m,:k) will be taken.
    lda
        Leading dimension of the array A (>= k).
    B[n * ldb]
        Array with the B parameters. Only the portion B(:n,:k) will be taken.
    ldb
        Leading dimension of the array B (>= k).
    g_A[m * lda], g_B[n * ldb] (out)
        Arrays to which to sum the computed gradients for A and B.
    m
        Number of rows in X.
    n
        Number of columns in X.
    k
        Dimensionality of the A and B matries (a.k.a. latent factors)
    ixA[nnz], ixB[nnz], X[nnz], nnz
        Matrix X in triplets format (row,col,x). Should only pass one of
        'X', 'Xfull', 'Xcsr/Xcsc'. If 'Xfull' is passed, it will be preferred.
        Pass NULL if 'X' will be passed in a different format.
    Xfull[m * n]
        Matrix X in dense form, with missing entries as NAN. Should only pass
        one of 'X', 'Xfull', 'Xcsr/Xcsc'. If 'Xfull' is passed, it will be
        preferred. Pass NULL if 'X' will be passed in a different format.
    full_dense
        Whether the X matrix contains no missing values.
    Xcsr_p[m+1], Xcsr_i[nnz], Xcsr[nnz], Xcsc_p[n], Xcsc_i[nnz], Xcsc[nnz]
        Matrix X in both CSR and CSC formats, in wich array 'p' indicates the
        starting position of a row for CSR and column for CSC in the Xcsr array,
        and array 'i' indicates the corresponding column for CSR and row for CSC
        for that entry. Should only pass one of 'X', 'Xfull', 'Xcsr/Xcsc'. If
        'Xfull' is passed, it will be preferred. Pass NULL if 'X' will be passed
         in a different format.
    user_bias, item_bias
        Whether user/row and/or item/column biases are to be used. Ignored if
        passing overwrite_grad' = 'false'
    biasA[m], biasB[n]
        Arrays containing the row/column biases. Pass NULL if not used.
    g_biasA[m], g_biasB[n] (out)
        Arrays in which to sum the gradient calculations for the biases.
    weight[nnz or m*n], weightR[nnz], weightC[nnz]
        Observation weights for X. Must match the shape of X - that is, either
        'nnz' for sparse, or 'm*n' for dense. If passing CSR/CSC matrices for X,
        must pass these in variables 'weightR' and 'weightC', otherwise, should
        be passed in 'weight'. Pass NULL if not used.
    scaling
        Scaling to add to the objective function. Pass '1' for no scaling.
    buffer_real_t[m*n]
        If passing a dense matrix, temporary array which will be overwritten.
        Not required for sparse matrices.
    buffer_mt[nthreads * k * (m+n+1)]
        If passing X and nthreads > 1, will be used as a temporary array into
        which to copy thread-local values for the one-pass parallelization
        strategy described at the top. Not required if passing Xcsr/Xcsc,
        Xfull, or nthreads = 1.
    overwrite_grad
        Whether it is allowed to overwrite the gradient arrays. If passing
        'false', will ignore the biases. If passing 'true', will assume that
        all the gradient arrays are continuous in memory. The reasoning behind
        is to allow for a more numerically precise calculation when there is
        some scaling parameter.
    nthreads
        Number of parallel threads to use. Note that BLAS and LAPACK threads
        are not controlled through this function.

    Returns
    -------
    f : The evaluated function value.


*******************************************************************************/

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
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix, ia, ib;
    #else
    size_t ia = 0, ib = 0;
    #endif

    double f = 0;
    double corr = 0;
    double tempf = 0;
    double fsum = 0;

    /* these will be de-referenced, but not updated */
    if (!user_bias)
    {
        if (!item_bias)
            g_biasA = g_A;
        else if (g_A == g_biasB + n)
            g_biasA = g_biasB;
        else
            g_biasA = g_A;
    }

    if (!item_bias)
    {
        if (user_bias && (g_A == g_biasA + m || g_biasA == g_A))
            g_biasB = g_biasA;
        else if (!user_bias &&
                 g_B == g_A + (size_t)m*(size_t)lda - (size_t)(lda-k))
            g_biasB = g_A;
        else
            g_biasB = g_B;
    }

    real_t err;
    size_t m_by_n = (size_t)m * (size_t)n;

    bool parallel_onepass = (Xfull == NULL && nthreads > 1 &&
                             Xcsr == NULL  && buffer_mt != NULL);
    if (parallel_onepass)
        set_to_zero_(buffer_mt, (size_t)nthreads
                                * (  (size_t)(k + (int)user_bias) * (size_t)m
                                    +(size_t)(k + (int)item_bias) * (size_t)n),
                    nthreads);

    if (Xfull == NULL)  /* sparse input with NAs - this is the expected case */
    {
        /* This is the loop as explained at the top.
           Note about the differentiation here: if using it for the main
           factorization (which allows biases), it's not a problem to overwrite
           the gradient matrices over their full leading dimension, and these
           are expected to be all continuous in a memory layout. When there is
           some scaling applied, it is more numerically  precise to apply the
           scaling after all the variables have already been summed, but if the
           gradients already have some information from a previous factorization
           to which these should add instead of overwrite them, it's necessary
           to apply the scaling observation-by-observation instead */
        #ifdef _OPENMP
        if (    nthreads <= 1 ||
                (Xcsr == NULL && !parallel_onepass) ||
                (Xcsr != NULL && nthreads <= 2) )
        #endif
        {
            for (size_t ix = 0; ix < nnz; ix++)
            {
                ia = (size_t)ixA[ix];
                ib = (size_t)ixB[ix];
                err = cblas_tdot(k, A + ia*(size_t)lda, 1,
                                 B + ib*(size_t)ldb, 1)
                       - X[ix];
                err += (user_bias? biasA[ia] : 0.)
                     + (item_bias? biasB[ib] : 0.);

                tempf = square(err)*((weight==NULL)? 1. : weight[ix])-corr;
                fsum = f + tempf;
                corr = (fsum - f) - tempf;
                f = fsum;

                err *= ((weight == NULL)? 1. : weight[ix]);

                g_biasA[ia] += user_bias? err : 0.;
                g_biasB[ib] += item_bias? err : 0.;
                cblas_taxpy(k, err, B + ib*(size_t)ldb, 1,
                            g_A + ia*(size_t)lda, 1);
                cblas_taxpy(k, err, A + ia*(size_t)lda, 1,
                            g_B + ib*(size_t)ldb, 1);
            }
        }

        #ifdef _OPENMP
        else if (parallel_onepass)
        {
            size_t thr_szA = (size_t)m*(size_t)k;
            size_t thr_szB = (size_t)n*(size_t)k;
            real_t *restrict g_A_t = buffer_mt;
            real_t *restrict g_B_t = g_A_t + (size_t)nthreads*thr_szA;
            real_t *restrict g_biasA_t = g_B_t + (size_t)nthreads*thr_szB;
            real_t *restrict g_biasB_t = g_biasA_t
                                         + (user_bias?
                                            ((size_t)nthreads * (size_t)m)
                                            : (0));

            #pragma omp parallel for schedule(static) num_threads(nthreads)\
                    reduction(+:f) private(ia, ib, err) \
                    shared(ixA, ixB, X, nnz, A, B, lda, ldb, k, weight, \
                           scaling, thr_szA, thr_szB, g_biasA_t, g_biasB_t,\
                           biasA, biasB, user_bias, item_bias, m, n)
            for (size_t_for ix = 0; ix < nnz; ix++)
            {
                ia = (size_t)ixA[ix];
                ib = (size_t)ixB[ix];
                err = cblas_tdot(k, A + ia*(size_t)lda, 1,
                                 B + ib*(size_t)ldb, 1)
                       + (user_bias? biasA[ia] : 0.)
                       + (item_bias? biasB[ib] : 0.)
                       - X[ix];

                f += square(err) * ((weight == NULL)? 1. : weight[ix]);
                err *= ((weight == NULL)? 1. : weight[ix]);

                g_biasA_t[ia + (size_t)m*(size_t)(omp_get_thread_num())]
                    += user_bias? err : 0.;
                g_biasB_t[ib + (size_t)n*(size_t)(omp_get_thread_num())]
                    += item_bias? err : 0.;

                cblas_taxpy(k, err, B + ib*(size_t)ldb, 1,
                            g_A_t + (size_t)(omp_get_thread_num())*thr_szA
                                  + ia*(size_t)lda, 1);
                cblas_taxpy(k, err, A + ia*(size_t)lda, 1,
                            g_B_t + (size_t)(omp_get_thread_num())*thr_szB
                                  + ib*(size_t)ldb, 1);
            }

            reduce_mat_sum(g_A, lda, g_A_t,
                           m, k, nthreads);
            reduce_mat_sum(g_B, ldb, g_B_t,
                           n, k, nthreads);
            if (user_bias)
                reduce_mat_sum(g_biasA, 1, g_biasA_t,
                               m, 1, nthreads);
            if (item_bias)
                reduce_mat_sum(g_biasB, 1, g_biasB_t,
                               n, 1, nthreads);
        }

        else
        {
            #pragma omp parallel for schedule(dynamic) reduction(+:f) \
                    num_threads(nthreads) private(err, ib, tempf) \
                    shared(m, k, A, B, lda, ldb, Xcsr, Xcsr_p, Xcsr_i, \
                           scaling, weightR, g_A, user_bias, item_bias, \
                           biasA, biasB, g_biasA)
            for (ia = 0; ia < (size_t)m; ia++)
            {
                tempf = 0;
                for (size_t ix = (size_t)Xcsr_p[ia];
                            ix < (size_t)Xcsr_p[ia+(size_t)1]; ix++)
                {
                    ib = (size_t)Xcsr_i[ix];
                    err = cblas_tdot(k, A + ia*(size_t)lda, 1,
                                     B + ib*(size_t)ldb, 1)
                         + (user_bias? biasA[ia] : 0.)
                         + (item_bias? biasB[ib] : 0.)
                         - Xcsr[ix];

                    tempf += square(err)
                              * ((weightR == NULL)? 1. : weightR[ix]);
                    err *= ((weightR == NULL)? 1. : weightR[ix]);

                    g_biasA[ia] += user_bias? err : 0.;
                    cblas_taxpy(k, err, B + ib*(size_t)ldb, 1,
                                g_A + ia*(size_t)lda, 1);
                }
                f += tempf;
            }


            #pragma omp parallel for schedule(dynamic) \
                    num_threads(nthreads) private(err, ia) \
                    shared(n, k, A, B, lda, ldb, Xcsc, Xcsc_p, Xcsc_i, \
                           scaling, weightC, g_B, user_bias, item_bias, \
                           biasA, biasB, g_biasB)
            for (ib = 0; ib < (size_t)n; ib++)
            {
                for (size_t ix = (size_t)Xcsc_p[ib];
                            ix < (size_t)Xcsc_p[ib+(size_t)1]; ix++)
                {
                    ia = (size_t)Xcsc_i[ix];
                    err = cblas_tdot(k, A + ia*(size_t)lda, 1,
                                     B + ib*(size_t)ldb, 1)
                           + (user_bias? biasA[ia] : 0.)
                           + (item_bias? biasB[ib] : 0.)
                           - Xcsc[ix];
                    err *= ((weightC == NULL)? 1. : weightC[ix]);

                    g_biasB[ib] += item_bias? err : 0.;
                    cblas_taxpy(k, err, A + ia*(size_t)lda, 1,
                                g_B + ib*(size_t)ldb, 1);
                }
            }
        }
        #endif

        #pragma omp barrier
        if (scaling != 1.)
        {
            /* Note: the gradients should be contiguous in memory in the
               following order: biasA, biasB, A, B - hence these conditions.
               If passing discontiguous arrays (e.g. when the gradient is
               a zeroed-out array and later summed to the previous
               gradient), there should be no biases, and the biases
               should already be assigned to the same memory locations as
               the gradient arrays. Otherwise should pass
               'overwrite_grad=false', which should not used within this
               module */
            if (g_B == g_biasA
                            + (size_t)(user_bias? m : 0)
                            + (size_t)(item_bias? n : 0)
                            + ((size_t)m*(size_t)lda  - (size_t)(lda-k)))
            {
                tscal_large(g_biasA, scaling,
                            ((size_t)m*(size_t)lda + (size_t)n*(size_t)ldb)
                            + (size_t)(user_bias? m : 0)
                            + (size_t)(item_bias? n : 0)
                            - (size_t)(lda-k),
                            nthreads);
            }

            else if (!user_bias && !item_bias &&
                     g_B == g_A + (size_t)m*(size_t)lda- (size_t)(lda-k))
            {
                tscal_large(g_A, scaling,
                            ((size_t)m*(size_t)lda + (size_t)n*(size_t)ldb),
                            nthreads);
            }

            else
            {
                if (user_bias)
                    cblas_tscal(m, scaling, g_biasA, 1);
                if (item_bias)
                    cblas_tscal(n, scaling, g_biasB, 1);
                tscal_large(g_A, scaling,
                            ((size_t)m*(size_t)lda) - (size_t)(lda-k),
                            nthreads);
                tscal_large(g_B, scaling,
                            ((size_t)n*(size_t)ldb) - (size_t)(ldb-k),
                            nthreads);
            }
        }
    }

    else /* dense input - this is usually not optimal, but still supported */
    {
        /* Buffer = X */
        copy_arr_(Xfull, buffer_real_t, m_by_n, nthreads);
        /* Buffer = A*t(B) - Buffer */
        cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m, n, k,
                    1., A, lda, B, ldb,
                    -1., buffer_real_t, n);
        /* Buffer += biasA[m,1] + biasB[1,n] */
        if (user_bias)
            mat_plus_rowvec(buffer_real_t, biasA, m, n, nthreads);
        if (item_bias)
            mat_plus_colvec(buffer_real_t, biasB, 1., m, n, (size_t)n,nthreads);

        /* Buffer *= W  (Now buffer becomes E without the scaling) */
        if (full_dense) {
            if (weight != NULL)
                mult_elemwise(buffer_real_t, weight, m_by_n, nthreads);
        } else {
            if (weight == NULL)
                nan_to_zero(buffer_real_t, Xfull, m_by_n, nthreads);
            else
                mult_if_non_nan(buffer_real_t, Xfull, weight, m_by_n, nthreads);
        }

        /* f = ||E||^2 */
        if (weight == NULL)
            f = sum_squares(buffer_real_t, m_by_n, nthreads);
        else
            f = sum_sq_div_w(buffer_real_t, weight, m_by_n, true, nthreads);

        /* grad(bias1) = scaling * sum_rows(E) */
        if (user_bias) {
            sum_by_rows(buffer_real_t, g_biasA, m, n, nthreads);
            if (scaling != 1)
                cblas_tscal(m, scaling, g_biasA, 1);
        }
        /* grad(bias2) = scaling * sum_cols(E) */
        if (item_bias) {
            sum_by_cols(buffer_real_t, g_biasB, m, n, (size_t)n, nthreads);
            if (scaling != 1)
                cblas_tscal(n, scaling, g_biasB, 1);
        }

        /* grad(A) =  scaling * E * B */
        cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, k, n,
                    scaling, buffer_real_t, n, B, ldb,
                    0., g_A, lda);
        /* grad(B) = scaling * t(E) * A */
        cblas_tgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    n, k, m,
                    scaling, buffer_real_t, n, A, lda,
                    0., g_B, ldb);
        /* Note: don't apply the scaling earlier as otherwise it
           loses precision, even if it manages to save some operations */
    }

    return (real_t)((scaling / 2.) * f);
}

/*******************************************************************************
    Closed-form solution for cannonical form
    ----------------------------------------

    This function uses the closed form to obtain the least-squares minimizer
    for a single row of the A matrix:
        min ||M . W . (X - A*t(B)) ||^2

    The formula is given as follows:
        Aopt[1,k] = inv(t(B'*W) * B' + diag(lambda)) * (t(B' * W) * Xa[1,n])

    Where B'[n,k] is a matrix constructed from B which has zeros in every row
    for which the corresponding entry for row 'a' in Xa[1,n] is missing.

    Since the matrix whose inverse is required is symmetric positive-definite,
    it's possible to solve this procedure quite efficiently with specialized
    methods.

    Note that this function can accomodate weights and regulatization, but not
    biases. In order to determine the bias for the given row 'a' of X, the B
    matrix should get a column of all-ones appended at the end. Doing this
    also requires passing a matrix 'X' with the bias already-subtracted from
	it when solving for B.

    This function is not meant to exploit multi-threading, but it still calls
    BLAS and LAPACK functions, which set their number of threads externally.

    Parameters
    ----------
    a_vec[k] (out)
        The optimal optimal values for A[1,k].
    k
        The dimensionality of the factorization (a.k.a. latent factors).
    B[n*k]
        The B matrix with which A is multiplied to approximate X.
    n
        Number of rows in B, number of columns in X.
    ldb
        Leading dimension of matrix B (>= k).
    Xa_dense[n]
        Row of X in dense format. Missing values should appear as NAN.
        Will be modified in-place if there are any missing entries. Pass NULL
        if X is sparse.
    full_dense
        Whether the Xa_dense matrix has only non-missing entries.
    Xa[nnz], ixB[nnz], nnz
        Row of the X matrix in sparse format, with ixB denoting the indices
        of the non-missing entries and Xa the values. Pass NULL if X is
        given in dense format.
    weight[nnz or n]
        Observation weights for each non-missing entry in X. Must match the
        shape of X passed - that is, if Xa_dense is passed, must have lenght
        'n', if Xa is passed, must have length 'nnz'. Pass NULL if the weights
        are uniform.
    buffer_real_t[k^2 or k^2 + n*k]
        Array in which to write temporary values. For sparse X and dense X
        with full_dense=true, must have space for k^2 elements. For dense X
        with full_dense=false, must have space for k^2 + n*k elements.
    lam
        Regularization parameter applied on the A matrix.
    precomputedTransBtBinvBt
        Precomputed matrix inv(t(B)*B + diag(lam))*t(B). Can only be used
        if passing 'Xa_dense' + 'full_dense=true' + 'weight=NULL'. This is
        used in order to speed up computations, but if not passed, will not be
        used. When passed, there is no requirement for any buffer.
    precomputedBtBw
        Precomputed matrix t(B)*W*B + diag(lam). Will only be used if either:
        (a) passing 'Xa_dense' + 'full_dense=false', and the proportion of
        missing entries in 'Xa_dense' is <= 10% of the total; or
        (b) passing 'Xa_dense=NULL' + 'NA_as_zero=true'.
        This is only used to speed up computations, and if not passed, will
        not be used, unless passing 'NA_as_zero=true'.
    cnt_NA
        Number of missing entries in 'Xa_dense'. Only used if passing
        'precomputedBtBw' and 'full_dense=false'.
    NA_as_zero
        Whether to take missing entries from 'Xa' as zero, and assume that
        there are no missing entries. This implies a different model in which
        the squared error is computed over all values of X. If passing 'true',
        must also pass 'precomputedBtBw'. This is ignored if passing 'Xa_dense'.


*******************************************************************************/
#ifdef AVOID_BLAS_SYR
#undef cblas_tsyr
#define cblas_tsyr(order, Uplo, N, alpha, X, incX, A, lda) \
        custom_syr(N, alpha, X, A, lda)
#endif
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
)
{
    /* TODO: here should add a parameter 'incX' for dense vectors */
    real_t *restrict bufferBtB = buffer_real_t;
    if (ldb == 0) ldb = k;
    bool add_diag = true;
    char lo = 'L';
    int_t one = 1;
    int_t ignore;
    if (n_BtB == 0) n_BtB = n;
    bool prefer_BtB = (cnt_NA + (n_BtB-n) < 2*k) ||
                      (nnz > (size_t)(2*k));
    if (precomputedBtB == NULL)
        prefer_BtB = false;

    /* Note: if passing 'NA_as_zero', 'n' and 'n_BtB' cannot be different */

    /* Potential bad inputs */
    if ((   (Xa_dense != NULL && cnt_NA == n) ||
            (Xa_dense == NULL && nnz == 0)  )
        && !(NA_as_zero && bias_BtX != NULL))
    {
        zero_out:
        set_to_zero(a_vec, k);
        return;
    }

    if (scale_lam)
    {
        real_t multiplier_lam = 1.;

        if (weight == NULL)
        {
            if (Xa_dense != NULL)
                multiplier_lam = (real_t)(n - (full_dense? 0 : cnt_NA));
            else if (NA_as_zero)
                multiplier_lam = (real_t)n;
            else
                multiplier_lam = (real_t)nnz;
        }

        else
        {
            if (wsum <= 0)
            {
                wsum = 0.;
                if (Xa_dense != NULL) {
                    for (int_t ix = 0; ix < n; ix++)
                        wsum += isnan(Xa_dense[ix])? 0. : weight[ix];
                }

                else {
                    for (size_t ix = 0; ix < nnz; ix++)
                        wsum += weight[ix];
                }

                if (NA_as_zero && Xa_dense == NULL)
                    wsum += (real_t)(n - (int_t)nnz);
            }

            if (fabs_t(wsum) < EPSILON_T && bias_BtX == NULL) goto zero_out;

            multiplier_lam = wsum;
        }
        
        lam *= multiplier_lam;
        l1_lam *= multiplier_lam;
        if (!scale_bias_const) {
            lam_last *= multiplier_lam;
            l1_lam_last *= multiplier_lam;
        }
    }

    if (nonneg) use_cg = false;

    #ifdef TEST_CG
    if (l1_lam || l1_lam_last)
        use_cg = false;
    #endif

    /* If inv(t(B)*B + diag(lam))*B is already given, use it as a shortcut.
       The intended use-case for this is for cold-start recommendations
       for the collective model with no missing values, given that the
       C matrix is already fixed and is the same for all users. */
    if (precomputedTransBtBinvBt != NULL && weight == NULL && !nonneg &&
        ((full_dense && Xa_dense != NULL && n_BtB == n) ||
         (Xa_dense == NULL && NA_as_zero && bias_BtX == NULL)) &&
        (l1_lam == 0. && l1_lam_last == 0.))
    {
        if (Xa_dense != NULL)
            cblas_tgemv(CblasRowMajor, CblasTrans,
                        n, k,
                        1., precomputedTransBtBinvBt, k,
                        Xa_dense, 1,
                        0., a_vec, 1);
        else
        {
            set_to_zero(a_vec, k);
            tgemv_dense_sp(
                n, k, /* <- 'n' doesn't get used*/
                1., precomputedTransBtBinvBt, k,
                ixB, Xa, nnz,
                a_vec
            );
        }
        return;
    }

    /* If t(B*w)*B + diag(lam) is given, and there are very few mising
       values, can still be used as a shortcut by substracting from it */
    else if (Xa_dense != NULL && precomputedBtB != NULL && weight == NULL &&
             prefer_BtB)
    {
        add_diag = false;
        copy_mat(k, k,
                 precomputedBtB, ld_BtB,
                 bufferBtB, k);
        if (BtB_is_scaled && scale_BtB != 1.)
            cblas_tscal(square(k), 1./scale_BtB, bufferBtB, 1);
        add_diag = !BtB_has_diag;

        set_to_zero(a_vec, k);
        for (size_t ix = 0; ix < (size_t)n; ix++)
        {
            if (isnan(Xa_dense[ix]))
                cblas_tsyr(CblasRowMajor, CblasUpper, k,
                           -1., B + ix*(size_t)ldb, 1,
                           bufferBtB, k);
            else
                cblas_taxpy(k, Xa_dense[ix], B + ix*(size_t)ldb, 1, a_vec, 1);
        }
        for (size_t ix = (size_t)n; ix < (size_t)n_BtB; ix++)
            cblas_tsyr(CblasRowMajor, CblasUpper, k,
                       -1., B + ix*(size_t)ldb, 1,
                       bufferBtB, k);
    }

    /* If the input is sparse and it's assumed that the non-present
       entries are zero, with no missing values, it's still possible
       to use the precomputed and pre-factorized matrix. */
    else if (NA_as_zero && weight == NULL && !nonneg &&
             Xa_dense == NULL && precomputedBtBchol != NULL &&
             l1_lam == 0. && l1_lam_last == 0.)
    {
        set_to_zero(a_vec, k);
        tgemv_dense_sp(n, k,
                       1., B, (size_t)ldb,
                       ixB, Xa, nnz,
                       a_vec);
        if (bias_BtX != NULL)
            cblas_taxpy(k, 1./multiplier_bias_BtX, bias_BtX, 1, a_vec, 1);
        tpotrs_(&lo, &k, &one,
                precomputedBtBchol, &k,
                a_vec, &k,
                &ignore);
        return;
    }

    /* In some cases, the sparse matrices might hold zeros instead
       of NAs - here the already-factorized BtBchol should be passed,
       but if for some reason it wasn't, will construct the usual
       matrices on-the-fly.
       If it has weights, could still use the precomputed matrix,
       then adjust by substracting from it again. */
    else if (Xa_dense == NULL && NA_as_zero && !use_cg)
    {
        set_to_zero(a_vec, k);
        set_to_zero(bufferBtB, square(k));

        if (weight != NULL)
        {
            for (size_t ix = 0; ix < nnz; ix++)
            {
                cblas_tsyr(CblasRowMajor, CblasUpper,
                           k, (weight[ix] - 1.),
                           B + (size_t)ixB[ix]*(size_t)ldb, 1,
                           bufferBtB, k);
                cblas_taxpy(k,
                            (weight[ix] * Xa[ix])
                                -
                            (weight[ix]-1.)
                                *
                            (bias_X_glob + ((bias_X == NULL)?
                                                0 : bias_X[ixB[ix]])),
                            B + (size_t)ixB[ix]*(size_t)ldb, 1,
                            a_vec, 1);
            }
        }

        else {
            tgemv_dense_sp(n, k,
                           1., B, (size_t)ldb,
                           ixB, Xa, nnz,
                           a_vec);
        }


        if (precomputedBtB == NULL)
            cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        k, max2(n, n_BtB),
                        1., B, ldb,
                        1., bufferBtB, k);
        else {
            if (BtB_is_scaled && scale_BtB != 1.)
                cblas_taxpy(square(k), 1./scale_BtB, precomputedBtB, 1,
                            bufferBtB, 1);
            else
                sum_mat(k, k,
                        precomputedBtB, ld_BtB,
                        bufferBtB, k);
            add_diag = !BtB_has_diag;
        }

    }

    /* If none of the above apply, it's faster to get an approximate
       solution using the conjugate gradient method.
       In this case, will exit the function afterwards as it will
       not need to calculate the Cholesky. */
    /* Note: 'multiplier_bias_BtX' will only be encountered when solving for
       'A' given 'U' alone in the prediction functions, thus it's not necessary
       to include here. */
    else if (use_cg)
    {
        if (Xa_dense != NULL)
            factors_explicit_cg_dense(
                a_vec, k,
                B, n, ldb,
                Xa_dense, cnt_NA,
                weight,
                precomputedBtB, ld_BtB,
                buffer_real_t,
                lam, lam_last,
                max_cg_steps
            );
        else if (NA_as_zero && weight != NULL)
            factors_explicit_cg_NA_as_zero_weighted(
                a_vec, k,
                B, n, ldb,
                Xa, ixB, nnz,
                weight,
                precomputedBtB, ld_BtB,
                bias_BtX, bias_X, bias_X_glob,
                multiplier_bias_BtX,
                buffer_real_t,
                lam, lam_last,
                max_cg_steps
            );
        else
            factors_explicit_cg(
                a_vec, k,
                B, n, ldb,
                Xa, ixB, nnz,
                weight,
                buffer_real_t,
                lam, lam_last,
                max_cg_steps
            );

        return;
    }

    /* In more general cases, need to construct the following matrices:
        - t(B)*B + diag(lam), with only the rows of B which have a
          non-missing entry in X.
        - t(B)*t(X), with missing entries in X set to zero.
       If X is dense, this can be accomplished by following the steps verbatim.
       If X is sparse, it's more efficient to obtain the first one through
       SR1 updates to an empty matrix, while the second one can be obtained
       with a dense-sparse matrix multiplication */

    else if (Xa_dense == NULL)
    {
        /* Sparse X - obtain t(B)*B through SR1 updates, avoiding a full
           matrix-matrix multiplication. This is the expected scenario for
           most use-cases. */
        set_to_zero(bufferBtB, square(k));
        for (size_t ix = 0; ix < nnz; ix++)
            cblas_tsyr(CblasRowMajor, CblasUpper, k,
                       (weight == NULL)? (1.) : (weight[ix]),
                       B + (size_t)ixB[ix]*(size_t)ldb, 1,
                       bufferBtB, k);

        /* Now obtain t(B)*t(X) from a dense_matrix-sparse_vector product,
           avoid again a full matrix-vector multiply. Note that this is
           stored in 'a_vec', despite the name */
        set_to_zero(a_vec, k);
        /* Note: in theory, should pass max2(n, n_BtB) to these functions,
           but 'n' is not used so it doesn't matter. */
        if (weight == NULL) {
            tgemv_dense_sp(n, k,
                           1., B, (size_t)ldb,
                           ixB, Xa, nnz,
                           a_vec);
        }

        else {
            tgemv_dense_sp_weighted(n, k,
                                    weight, B, (size_t)ldb,
                                    ixB, Xa, nnz,
                                    a_vec);
        }
    }

    else
    {
        /* Dense X - in this case, re-construct t(B)*B without the missing
           entries - at once if possible, or one-by-one if not.
           Note that, if B0 is the B matrix with the rows set to zero
           when the entries of X are missing, then the following
           equalities will apply:
            t(B0)*B == t(B)*B0 == t(B0)*B0 */
        if (full_dense && weight == NULL)
        {
            /* this will only be encountered when calculating factors
               after having called 'fit_*'. */
            cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        k, n,
                        1., B, ldb,
                        0., bufferBtB, k);
            cblas_tgemv(CblasRowMajor, CblasTrans,
                        n, k,
                        1., B, ldb,
                        Xa_dense, 1,
                        0., a_vec, 1);
        }

        else
        {
            set_to_zero(a_vec, k);
            set_to_zero(bufferBtB, square(k));
            for (size_t ix = 0; ix < (size_t)n; ix++) {
                if (!isnan(Xa_dense[ix])) {
                    cblas_tsyr(CblasRowMajor, CblasUpper,
                               k, (weight == NULL)? (1.) : (weight[ix]),
                               B + ix*(size_t)ldb, 1,
                               bufferBtB, k);
                    cblas_taxpy(k,
                                ((weight == NULL)? (1.) : (weight[ix]))
                                    * Xa_dense[ix],
                                B + ix*(size_t)ldb, 1,
                                a_vec, 1);
                }
            }
        }
    }

    /* Finally, obtain closed-form through Cholesky factorization,
       exploiting the fact that t(B)*B+diag(lam) is symmetric PSD */
    if (add_diag || force_add_diag) {
        add_to_diag(bufferBtB, lam, k);
        if (lam_last != lam) bufferBtB[square(k)-1] += (lam_last - lam);
    }
    if (bias_BtX != NULL && NA_as_zero)
        cblas_taxpy(k, 1./multiplier_bias_BtX, bias_BtX, 1, a_vec, 1);

    if (!nonneg && l1_lam == 0. && l1_lam_last == 0.)
        tposv_(&lo, &k, &one,
               bufferBtB, &k,
               a_vec, &k,
               &ignore);
    else if (!nonneg)
        solve_elasticnet(
            bufferBtB,
            a_vec,
            buffer_real_t + square(k),
            k,
            l1_lam, l1_lam_last,
            max_cd_steps,
            true
        );
    else
        solve_nonneg(
            bufferBtB,
            a_vec,
            buffer_real_t + square(k),
            k,
            l1_lam, l1_lam_last,
            max_cd_steps,
            true
        );
    /* Note: Function 'posv' is taken from LAPACK for FORTRAN.
       If using LAPACKE for C with with Row-Major parameter,
       some implementations will copy the matrix to transpose it
       and pass it to FORTRAN-posv. */
}

/* https://en.wikipedia.org/wiki/Conjugate_gradient_method */
void factors_explicit_cg
(
    real_t *restrict a_vec, int_t k,
    real_t *restrict B, int_t n, int_t ldb,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    real_t *restrict weight,
    real_t *restrict buffer_real_t,
    real_t lam, real_t lam_last,
    int_t max_cg_steps
)
{
    real_t *restrict Ap = buffer_real_t;
    real_t *restrict p  = Ap + k;
    real_t *restrict r  = p  + k;
    set_to_zero(r, k);
    real_t coef;
    real_t a;
    real_t r_old, r_new;

    for (size_t ix = 0; ix < nnz; ix++) {
        coef  = cblas_tdot(k, B + (size_t)ixB[ix]*(size_t)ldb, 1, a_vec, 1);
        coef -= Xa[ix];
        coef *= (weight == NULL)? 1. : weight[ix];
        cblas_taxpy(k, -coef, B + (size_t)ixB[ix]*ldb, 1, r, 1);
    }
    cblas_taxpy(k, -lam, a_vec, 1, r, 1);
    if (lam != lam_last)
        r[k-1] -= (lam_last-lam) * a_vec[k-1];

    copy_arr(r, p, k);
    r_old = cblas_tdot(k, r, 1, r, 1);

    #ifdef TEST_CG
    if (r_old <= 1e-15)
        return;
    #else
    if (r_old <= 1e-12)
        return;
    #endif

    for (int_t cg_step = 0; cg_step < max_cg_steps; cg_step++)
    {
        set_to_zero(Ap, k);
        for (size_t ix = 0; ix < nnz; ix++) {
            coef = cblas_tdot(k, B + (size_t)ixB[ix]*(size_t)ldb, 1, p, 1);
            coef *= (weight == NULL)? 1. : weight[ix];
            cblas_taxpy(k, coef, B + (size_t)ixB[ix]*ldb, 1, Ap, 1);
        }
        cblas_taxpy(k, lam, p, 1, Ap, 1);
        if (lam != lam_last)
            Ap[k-1] += (lam_last-lam) * p[k-1];

        a = r_old / cblas_tdot(k, p, 1, Ap, 1);
        cblas_taxpy(k,  a,  p, 1, a_vec, 1);
        cblas_taxpy(k, -a, Ap, 1, r, 1);

        r_new = cblas_tdot(k, r, 1, r, 1);
        #ifdef TEST_CG
        if (r_new <= 1e-15)
            break;
        #else
        if (r_new <= 1e-8)
            break;
        #endif

        cblas_tscal(k, r_new / r_old, p, 1);
        cblas_taxpy(k, 1., r, 1, p, 1);
        r_old = r_new;
    }
}

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
)
{
    real_t *restrict Ap = buffer_real_t;
    real_t *restrict p  = Ap + k;
    real_t *restrict r  = p  + k;
    real_t *restrict wr = r  + k; /* length is 'n' */
    set_to_zero(r, k);
    real_t a;
    real_t r_old, r_new, coef;

    bool prefer_BtB = k < n;
    if (precomputedBtB == NULL)
        prefer_BtB = false;

    if (prefer_BtB)
    {
        cblas_tsymv(CblasRowMajor, CblasUpper, k,
                    -1., precomputedBtB, ld_BtB,
                    a_vec, 1,
                    0., r, 1);
        for (size_t ix = 0; ix < nnz; ix++)
        {
            coef = cblas_tdot(k, B + (size_t)ixB[ix]*(size_t)ldb, 1, a_vec, 1);
            cblas_taxpy(k,
                        -(weight[ix]-1.)
                            *
                        (coef + bias_X_glob + ((bias_X == NULL)?
                                                0: bias_X[ixB[ix]]))
                            +
                        (weight[ix] * Xa[ix]),
                        B + (size_t)ixB[ix]*(size_t)ldb, 1,
                        r, 1);
        }
    }

    else
    {
        cblas_tgemv(CblasRowMajor, CblasNoTrans,
                    n, k,
                    -1., B, ldb,
                    a_vec, 1,
                    0., wr, 1);
        for (size_t ix = 0; ix < nnz; ix++)
            wr[ixB[ix]] = weight[ix] * (wr[ixB[ix]] + Xa[ix]);
        if (bias_X != NULL) {
            for (size_t ix = 0; ix < nnz; ix++)
                wr[ixB[ix]] -= (weight[ix] - 1.)
                                    *
                               (bias_X[ixB[ix]] + bias_X_glob);
        }
        else if (bias_X_glob) {
            for (size_t ix = 0; ix < nnz; ix++)
                wr[ixB[ix]] -= (weight[ix] - 1.) * bias_X_glob;
        }
        cblas_tgemv(CblasRowMajor, CblasTrans,
                    n, k,
                    1., B, ldb,
                    wr, 1,
                    1., r, 1);
    }

    if (bias_BtX != NULL)
    {
        cblas_taxpy(k, 1./multiplier_bias_BtX, bias_BtX, 1, r, 1);
    }

    cblas_taxpy(k, -lam, a_vec, 1, r, 1);
    if (lam != lam_last)
        r[k-1] -= (lam_last-lam) * a_vec[k-1];

    copy_arr(r, p, k);
    r_old = cblas_tdot(k, r, 1, r, 1);

    #ifdef TEST_CG
    if (r_old <= 1e-15)
        return;
    #else
    if (r_old <= 1e-12)
        return;
    #endif

    for (int_t cg_step = 0; cg_step < max_cg_steps; cg_step++)
    {
        if (precomputedBtB != NULL && prefer_BtB)
        {
            cblas_tsymv(CblasRowMajor, CblasUpper, k,
                        1., precomputedBtB, ld_BtB,
                        p, 1,
                        0., Ap, 1);
            for (size_t ix = 0; ix < nnz; ix++) {
                coef = cblas_tdot(k,
                                  B + (size_t)ixB[ix]*(size_t)ldb, 1,
                                  p, 1);
                cblas_taxpy(k, (weight[ix] - 1.) * coef,
                            B + (size_t)ixB[ix]*(size_t)ldb, 1,
                            Ap, 1);
            }
        }

        else
        {
            cblas_tgemv(CblasRowMajor, CblasNoTrans,
                        n, k,
                        1., B, ldb,
                        p, 1,
                        0., wr, 1);
            for (size_t ix = 0; ix < nnz; ix++)
                wr[ixB[ix]] *= weight[ix];
            cblas_tgemv(CblasRowMajor, CblasTrans,
                        n, k,
                        1., B, ldb,
                        wr, 1,
                        0., Ap, 1);
        }

        cblas_taxpy(k, lam, p, 1, Ap, 1);
        if (lam != lam_last)
            Ap[k-1] += (lam_last-lam) * p[k-1];

        a = r_old / cblas_tdot(k, p, 1, Ap, 1);
        cblas_taxpy(k,  a,  p, 1, a_vec, 1);
        cblas_taxpy(k, -a, Ap, 1, r, 1);

        r_new = cblas_tdot(k, r, 1, r, 1);
        #ifdef TEST_CG
        if (r_new <= 1e-15)
            break;
        #else
        if (r_new <= 1e-8)
            break;
        #endif

        cblas_tscal(k, r_new / r_old, p, 1);
        cblas_taxpy(k, 1., r, 1, p, 1);
        r_old = r_new;
    }
}

void factors_explicit_cg_dense
(
    real_t *restrict a_vec, int_t k,
    real_t *restrict B, int_t n, int_t ldb,
    real_t *restrict Xa_dense, int_t cnt_NA,
    real_t *restrict weight,
    real_t *restrict precomputedBtB,
    int_t ld_BtB,
    real_t *restrict buffer_real_t,
    real_t lam, real_t lam_last,
    int_t max_cg_steps
)
{
    real_t *restrict Ap = buffer_real_t;
    real_t *restrict p  = Ap + k;
    real_t *restrict r  = p  + k;
    real_t r_new, r_old;
    real_t a, coef, w_this;

    bool prefer_BtB = cnt_NA < k && precomputedBtB != NULL && weight == NULL;
    if (!prefer_BtB)
        set_to_zero(r, k);

    if (prefer_BtB)
    {
        cblas_tsymv(CblasRowMajor, CblasUpper, k,
                    -1., precomputedBtB, ld_BtB,
                    a_vec, 1,
                    0., r, 1);
        for (size_t ix = 0; ix < (size_t)n; ix++)
        {
            if (isnan(Xa_dense[ix])) {
                coef = cblas_tdot(k, B + ix*(size_t)ldb, 1, a_vec, 1);
                cblas_taxpy(k, coef, B + ix*(size_t)ldb, 1, r, 1);
            }
            
            else {
                cblas_taxpy(k, Xa_dense[ix], B + ix*(size_t)ldb, 1, r, 1);
            }
        }
    }

    else
    {
        for (size_t ix = 0; ix < (size_t)n; ix++)
        {
            if (!isnan(Xa_dense[ix]))
            {
                coef = cblas_tdot(k, B + ix*(size_t)ldb, 1, a_vec, 1);
                cblas_taxpy(k,
                            (-coef + Xa_dense[ix])
                                *
                            ((weight == NULL)? 1. : weight[ix]),
                            B + ix*(size_t)ldb, 1, r, 1);
            }
        }
    }
    
    cblas_taxpy(k, -lam, a_vec, 1, r, 1);
    if (lam != lam_last)
        r[k-1] -= (lam_last-lam) * a_vec[k-1];

    copy_arr(r, p, k);
    r_old = cblas_tdot(k, r, 1, r, 1);

    #ifdef TEST_CG
    if (r_old <= 1e-15)
        return;
    #else
    if (r_old <= 1e-12)
        return;
    #endif

    for (int_t cg_step = 0; cg_step < max_cg_steps; cg_step++)
    {
        if (prefer_BtB)
        {
            cblas_tsymv(CblasRowMajor, CblasUpper, k,
                        1., precomputedBtB, ld_BtB,
                        p, 1,
                        0., Ap, 1);
            for (size_t ix = 0; ix < (size_t)n; ix++)
                if (isnan(Xa_dense[ix])) {
                    coef = cblas_tdot(k,  B + ix*(size_t)ldb, 1, p,  1);
                    cblas_taxpy(k, -coef, B + ix*(size_t)ldb, 1, Ap, 1);
                }
        }

        else
        {
            set_to_zero(Ap, k);
            for (size_t ix = 0; ix < (size_t)n; ix++)
            {
                if (!isnan(Xa_dense[ix]))
                {
                    w_this = (weight == NULL)? 1. : weight[ix];
                    coef = cblas_tdot(k, B + ix*(size_t)ldb, 1, p, 1);
                    cblas_taxpy(k, w_this * coef, B + ix*(size_t)ldb, 1, Ap, 1);
                }
            }
        }

        cblas_taxpy(k, lam, p, 1, Ap, 1);
        if (lam != lam_last)
            Ap[k-1] += (lam_last-lam) * p[k-1];

        a = r_old / cblas_tdot(k, p, 1, Ap, 1);
        cblas_taxpy(k,  a,  p, 1, a_vec, 1);
        cblas_taxpy(k, -a, Ap, 1, r, 1);
        r_new = cblas_tdot(k, r, 1, r, 1);
        #ifdef TEST_CG
        if (r_new <= 1e-15)
            break;
        #else
        if (r_new <= 1e-8)
            break;
        #endif

        cblas_tscal(k, r_new / r_old, p, 1);
        cblas_taxpy(k, 1., r, 1, p, 1);
        r_old = r_new;
    }
}


/* https://www.benfrederickson.com/fast-implicit-matrix-factorization/ */
void factors_implicit_cg
(
    real_t *restrict a_vec, int_t k,
    real_t *restrict B, size_t ldb,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    real_t lam,
    real_t *restrict precomputedBtB, int_t ld_BtB,
    int_t max_cg_steps,
    real_t *restrict buffer_real_t
)
{
    real_t *restrict Ap = buffer_real_t;
    real_t *restrict r  = Ap + k;
    real_t *restrict p  = r  + k;
    real_t coef;
    real_t r_old, r_new;
    real_t a;

    cblas_tsymv(CblasRowMajor, CblasUpper, k,
                -1., precomputedBtB, ld_BtB,
                a_vec, 1,
                0., r, 1);
    for (size_t ix = 0; ix < nnz; ix++) {
        coef = cblas_tdot(k, B + (size_t)ixB[ix]*ldb, 1, a_vec, 1);
        cblas_taxpy(k,
                    -(coef - 1.) * Xa[ix] - coef,
                    B + (size_t)ixB[ix]*ldb, 1,
                    r, 1);
    }
    cblas_taxpy(k, -lam, a_vec, 1, r, 1);

    copy_arr(r, p, k);
    r_old = cblas_tdot(k, r, 1, r, 1);

    #ifdef TEST_CG
    if (r_old <= 1e-15)
        return;
    #else
    if (r_old <= 1e-12)
        return;
    #endif

    for (int_t cg_step = 0; cg_step < max_cg_steps; cg_step++)
    {
        cblas_tsymv(CblasRowMajor, CblasUpper, k,
                    1., precomputedBtB, ld_BtB,
                    p, 1,
                    0., Ap, 1);
        for (size_t ix = 0; ix < nnz; ix++) {
            coef = cblas_tdot(k, B + (size_t)ixB[ix]*ldb, 1, p, 1);
            cblas_taxpy(k,
                        coef * (Xa[ix] - 1.) + coef,
                        B + (size_t)ixB[ix]*ldb, 1,
                        Ap, 1);
        }
        cblas_taxpy(k, lam, p, 1, Ap, 1);

        a = r_old / cblas_tdot(k, Ap, 1, p, 1);
        cblas_taxpy(k,  a,  p, 1, a_vec, 1);
        cblas_taxpy(k, -a, Ap, 1, r, 1);
        r_new = cblas_tdot(k, r, 1, r, 1);
        #ifdef TEST_CG
        if (r_new <= 1e-15)
            break;
        #else
        if (r_new <= 1e-8)
            break;
        #endif
        cblas_tscal(k, r_new / r_old, p, 1);
        cblas_taxpy(k, 1., r, 1, p, 1);
        r_old = r_new;
    }
}


void factors_implicit_chol
(
    real_t *restrict a_vec, int_t k,
    real_t *restrict B, size_t ldb,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    real_t lam, real_t l1_lam,
    real_t *restrict precomputedBtB, int_t ld_BtB,
    bool nonneg, int_t max_cd_steps,
    real_t *restrict buffer_real_t
)
{
    char lo = 'L';
    int_t one = 1;
    int_t ignore;
    if (nnz == 0) {
        set_to_zero(a_vec, k);
        return;
    }

    real_t *restrict BtB = buffer_real_t;
    buffer_real_t += square(k);

    for (size_t ix = 0; ix < nnz; ix++) {
        cblas_taxpy(k, Xa[ix] + 1.,
                    B + (size_t)ixB[ix]*ldb, 1, a_vec, 1);
    }

    set_to_zero(BtB, square(k));
    for (size_t ix = 0; ix < nnz; ix++) {
        cblas_tsyr(CblasRowMajor, CblasUpper, k,
                   Xa[ix], B + (size_t)ixB[ix]*ldb, 1,
                   BtB, k);
    }

    sum_mat(k, k,
            precomputedBtB, ld_BtB,
            BtB, k);

    if (!nonneg && !l1_lam)
        tposv_(&lo, &k, &one,
               BtB, &k,
               a_vec, &k,
               &ignore);
    else if (!nonneg)
        solve_elasticnet(
            BtB,
            a_vec,
            buffer_real_t,
            k,
            l1_lam, l1_lam,
            max_cd_steps,
            false
        );
    else
        solve_nonneg(
            BtB,
            a_vec,
            buffer_real_t,
            k,
            l1_lam, l1_lam,
            max_cd_steps,
            true
        );
}

void solve_nonneg
(
    real_t *restrict BtB,
    real_t *restrict BtX, /* <- solution will be here */
    real_t *restrict buffer_real_t,
    int_t k,
    real_t l1_lam, real_t l1_lam_last,
    size_t max_cd_steps,
    bool fill_lower
)
{
    real_t diff_iter = 0.;
    real_t diff_val = 0.;
    real_t newval = 0;
    if (fill_lower)
        fill_lower_triangle(BtB, k, k);

    if (l1_lam != 0.)
    {
        for (int_t ix = 0; ix < k; ix++)
            BtX[ix] -= l1_lam;
        if (l1_lam_last != l1_lam)
            BtX[k-1] -= (l1_lam_last - l1_lam);
    }

    real_t *restrict a_prev = buffer_real_t;
    set_to_zero(a_prev, k);
    
    if (max_cd_steps == 0) max_cd_steps = INT_MAX;
    size_t incr = max_cd_steps > 0;
    for (size_t iter = 0; iter < max_cd_steps; iter += incr)
    {
        diff_iter = 0;
        for (int_t ix = 0; ix < k; ix++)
        {
            newval = a_prev[ix] + BtX[ix] / BtB[ix + ix*k];
            newval = max2(newval, 0.);
            diff_val = newval - a_prev[ix];
            if (fabs_t(diff_val) > 1e-8) {
                diff_iter += fabs_t(diff_val);
                cblas_taxpy(k, -diff_val, BtB + ix*k, 1, BtX, 1);
                a_prev[ix] = newval;
            }
        }
        if (isnan(diff_iter) || !isfinite(diff_iter) || diff_iter < 1e-8)
            break;
    }
    copy_arr(a_prev, BtX, k);
}

void solve_nonneg_batch
(
    real_t *restrict BtB,
    real_t *restrict BtX, /* <- solution will be here */
    real_t *restrict buffer_real_t,
    int_t m, int_t k, size_t lda,
    real_t l1_lam, real_t l1_lam_last,
    size_t max_cd_steps,
    int nthreads
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif

    fill_lower_triangle(BtB, k, k);

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
            shared(BtB, BtX, buffer_real_t, m, lda, k, \
                   l1_lam, l1_lam_last, max_cd_steps)
    for (size_t_for ix = 0; ix < (size_t)m; ix++)
    {
        for (size_t ii = 0; ii < (size_t)k; ii++) {
            if (isnan(BtX[ii + ix*lda]))
                continue;
        }
        solve_nonneg(
            BtB,
            BtX + ix*lda,
            buffer_real_t + (size_t)k*(size_t)omp_get_thread_num(),
            k,
            l1_lam, l1_lam_last,
            max_cd_steps,
            false
        );
    }
}

void solve_elasticnet
(
    real_t *restrict BtB,
    real_t *restrict BtX, /* <- solution will be here */
    real_t *restrict buffer_real_t,
    int_t k,
    real_t l1_lam, real_t l1_lam_last,
    size_t max_cd_steps,
    bool fill_lower
)
{
    real_t diff_iter = 0.;
    real_t diff_val = 0.;
    real_t newval = 0;
    if (fill_lower)
        fill_lower_triangle(BtB, k, k);

    real_t *restrict BtX_neg = buffer_real_t;
    buffer_real_t += k;
    real_t *restrict a_prev = buffer_real_t;
    buffer_real_t += k;
    real_t *restrict a_neg = buffer_real_t;

    set_to_zero(a_prev, (size_t)2*(size_t)k);
    for (int_t ix = 0; ix < k; ix++)
        BtX_neg[ix] = -BtX[ix] - l1_lam;
    for (int_t ix = 0; ix < k; ix++)
        BtX[ix] -= l1_lam;
    if (l1_lam != l1_lam_last) {
        BtX[k-1] -= (l1_lam_last - l1_lam);
        BtX_neg[k-1] -= (l1_lam_last - l1_lam);
    }
    
    if (max_cd_steps == 0) max_cd_steps = INT_MAX;
    size_t incr = max_cd_steps > 0;
    for (size_t iter = 0; iter < max_cd_steps; iter += incr)
    {
        diff_iter = 0;

        for (int_t ix = 0; ix < k; ix++)
        {
            newval = a_prev[ix] + BtX[ix] / BtB[ix + ix*k];
            newval = max2(newval, 0.);
            diff_val = newval - a_prev[ix];
            if (fabs_t(diff_val) > 1e-8) {
                diff_iter += fabs_t(diff_val);
                cblas_taxpy(k,  diff_val, BtB + ix*k, 1, BtX_neg, 1);
                cblas_taxpy(k, -diff_val, BtB + ix*k, 1, BtX, 1);
                a_prev[ix] = newval;
            }
        }

        for (int_t ix = 0; ix < k; ix++)
        {
            newval = a_neg[ix] + BtX_neg[ix] / BtB[ix + ix*k];
            newval = max2(newval, 0.);
            diff_val = newval - a_neg[ix];
            if (fabs_t(diff_val) > 1e-8) {
                diff_iter += fabs_t(diff_val);
                cblas_taxpy(k,  diff_val, BtB + ix*k, 1, BtX, 1);
                cblas_taxpy(k, -diff_val, BtB + ix*k, 1, BtX_neg, 1);
                a_neg[ix] = newval;
            }
        }

        if (isnan(diff_iter) || !isfinite(diff_iter) || diff_iter < 1e-8)
            break;
    }

    for (int_t ix = 0; ix < k; ix++)
        BtX[ix] = a_prev[ix] - a_neg[ix];
}

void solve_elasticnet_batch
(
    real_t *restrict BtB,
    real_t *restrict BtX, /* <- solution will be here */
    real_t *restrict buffer_real_t,
    int_t m, int_t k, size_t lda,
    real_t l1_lam, real_t l1_lam_last,
    size_t max_cd_steps,
    int nthreads
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif

    fill_lower_triangle(BtB, k, k);

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
            shared(BtB, BtX, buffer_real_t, m, lda, k, \
                   l1_lam, l1_lam_last, max_cd_steps)
    for (size_t_for ix = 0; ix < (size_t)m; ix++)
    {
        for (size_t ii = 0; ii < (size_t)k; ii++) {
            if (isnan(BtX[ii + ix*lda]))
                continue;
        }
        solve_elasticnet(
            BtB,
            BtX + ix*lda,
            buffer_real_t + (size_t)3*(size_t)k*(size_t)omp_get_thread_num(),
            k,
            l1_lam, l1_lam_last,
            max_cd_steps,
            false
        );
    }
}


/* TODO: these functions for Adense or Bdense are no longer used,
   but should be used in 'factors_multiple' so do not delete yet. */
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
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif
    real_t *g_B = NULL;
    if (do_B) g_B = g_A;
    real_t f = 0.;
    size_t m_by_n = (size_t)m * (size_t)n;

    copy_arr_(Xfull, buffer_real_t, m_by_n, nthreads);
    cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k,
                1., A, lda, B, ldb,
                -1., buffer_real_t, n);
    if (weight == NULL) {
        nan_to_zero(buffer_real_t, Xfull, m_by_n, nthreads);
        f = w * sum_squares(buffer_real_t, m_by_n, nthreads);
    }
    else {
        /* TODO: make it compensated summation */
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(buffer_real_t, m, n, weight) reduction(+:f)
        for (size_t_for ix = 0; ix < m_by_n; ix++)
            f += (!isnan(buffer_real_t[ix]))?
                  (square(buffer_real_t[ix]) * w*weight[ix]) : (0);
        mult_if_non_nan(buffer_real_t, Xfull, weight, m_by_n, nthreads);
    }
    if (!do_B)
        cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, k, n,
                    w, buffer_real_t, n, B, ldb,
                    reset_grad? 0. : 1., g_A, lda);
    else
        cblas_tgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    n, k, m,
                    w, buffer_real_t, n, A, lda,
                    reset_grad? 0. : 1., g_B, ldb);
    if (lam != 0)
    {
        if (!do_B)
            add_lam_to_grad_and_fun(&f, g_A, A, m, k, lda, lam, nthreads);
        else
            add_lam_to_grad_and_fun(&f, g_B, B, n, k, ldb, lam, nthreads);
    }
    if (lam != 0. && lam_last != lam && k >= 1) {
        if (!do_B) {
            cblas_taxpy(m, lam_last-lam, A + k-1, lda, g_A + k-1, lda);
            f += (lam_last-lam) * cblas_tdot(m, A + k-1, lda, A + k-1, lda);
        }
        else {
            cblas_taxpy(n, lam_last-lam, B + k-1, ldb, g_B + k-1, ldb);
            f += (lam_last-lam) * cblas_tdot(n, B + k-1, ldb, B + k-1, ldb);
        }
    }
    return f / 2.;
}

void add_lam_to_grad_and_fun
(
    real_t *restrict fun,
    real_t *restrict grad,
    real_t *restrict A,
    int_t m, int_t k, int_t lda,
    real_t lam, int nthreads
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long row;
    #endif
    if (lda == k)
    {
        taxpy_large(A, lam, grad, (size_t)m*(size_t)k, nthreads);
        *fun += lam * sum_squares(A, (size_t)m*(size_t)k, nthreads);
    }

    else
    {
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(m, k, A, grad, lam, lda)
        for (size_t_for row = 0; row < (size_t)m; row++)
            for (size_t col = 0; col < (size_t)k; col++)
                grad[col + row*lda] += lam * A[col + row*lda];
        real_t reg = 0;
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(m, k, A, lda) reduction(+:reg)
        for (size_t_for row = 0; row < (size_t)m; row++)
            for (size_t col = 0; col < (size_t)k; col++)
                reg += square(A[col + row*lda]);
        *fun += lam * reg;
    }
}

real_t wrapper_fun_grad_Adense
(
    void *instance,
    real_t *x,
    real_t *g,
    const size_t n,
    const real_t step
)
{
    data_fun_grad_Adense *data = (data_fun_grad_Adense*)instance;
    return  fun_grad_Adense(
                g,
                x, data->lda,
                data->B, data->ldb,
                data->m, data->n, data->k,
                data->Xfull, data->weight,
                data->lam, data->w, data->lam_last,
                false, true,
                data->nthreads,
                data->buffer_real_t
            );
}

real_t wrapper_fun_grad_Bdense
(
    void *instance,
    real_t *x,
    real_t *g,
    const size_t n,
    const real_t step
)
{
    data_fun_grad_Bdense *data = (data_fun_grad_Bdense*)instance;
    return  fun_grad_Adense(
                g,
                data->A, data->lda,
                x, data->ldb,
                data->m, data->n, data->k,
                data->Xfull, data->weight,
                data->lam, data->w, data->lam_last,
                true, true,
                data->nthreads,
                data->buffer_real_t
            );
}

size_t buffer_size_optimizeA
(
    size_t n, bool full_dense, bool near_dense, bool do_B,
    bool has_dense, bool has_weights, bool NA_as_zero,
    bool nonneg, bool has_l1,
    size_t k, size_t nthreads,
    bool has_bias_static,
    bool pass_allocated_BtB, bool keep_precomputedBtB,
    bool use_cg, bool finalize_chol
)
{
    if (finalize_chol && use_cg)
    {
        return max2(
                buffer_size_optimizeA(
                        n, full_dense, near_dense, do_B,
                        has_dense, has_weights, NA_as_zero,
                        nonneg, has_l1,
                        k, nthreads,
                        has_bias_static,
                        pass_allocated_BtB, keep_precomputedBtB,
                        true, false
                ),
                buffer_size_optimizeA(
                    n, full_dense, near_dense, do_B,
                    has_dense, has_weights, NA_as_zero,
                    nonneg, has_l1,
                    k, nthreads,
                    has_bias_static,
                    pass_allocated_BtB, keep_precomputedBtB,
                    false, false
                )
        );
    }

    if (!has_dense)
        do_B = false;
    if (has_l1 || nonneg)
        use_cg = false;

    size_t buffer_size = 0;
    bool assigned_to_BtB_copy = false;
    /* case 1 */
    if (has_dense && (full_dense || near_dense) && !has_weights)
    {
        if (near_dense)
        {
            if (pass_allocated_BtB && !keep_precomputedBtB)
            {
                assigned_to_BtB_copy = true;
                buffer_size += 0;
            }
            else {
                buffer_size += square(k);
            }
        }
        if (pass_allocated_BtB && !keep_precomputedBtB && !near_dense &&
            !assigned_to_BtB_copy)
            buffer_size += 0;
        else {
            buffer_size += square(k);
        }
        if (do_B)
            buffer_size += n*nthreads;

        if (!full_dense)
        {
            size_t size_thread_buffer = square(k);
            if (use_cg)
                size_thread_buffer = max2(size_thread_buffer, (3 * k));
            if (nonneg)
                size_thread_buffer += k;
            else if (has_l1)
                size_thread_buffer += 3*k;
            buffer_size += nthreads * size_thread_buffer;
        }

        else {
            if (nonneg)
                buffer_size += k*nthreads;
            else if (has_l1)
                buffer_size += (size_t)3*k*nthreads;
        }
        return buffer_size;
    }

    /* case 2 */
    else if (has_dense)
    {
        if (!has_weights) {
            if (pass_allocated_BtB && !keep_precomputedBtB)
                buffer_size += 0;
            else {
                buffer_size += square(k);
            }
        }
        if (do_B) {
            buffer_size += n * nthreads;
        }
        if (do_B && has_weights) {
            buffer_size += n * nthreads;
        }
        size_t size_thread_buffer = square(k) + (use_cg? (3*k) : 0);
        if (nonneg)
            size_thread_buffer += k;
        else if (has_l1)
            size_thread_buffer += (size_t)3*k;
        return buffer_size + nthreads * size_thread_buffer;
    }

    /* case 3 */
    else if (!has_dense && NA_as_zero && !has_weights)
    {
        if (pass_allocated_BtB && !keep_precomputedBtB)
            buffer_size += 0;
        else
            buffer_size += square(k);
        if (nonneg)
            buffer_size += k*nthreads;
        else if (has_l1)
            buffer_size += (size_t)3*k*nthreads;
        if (has_bias_static)
            buffer_size += k;
        return buffer_size;
    }

    /* case 4 */
    else
    {
        bool add_diag_to_BtB = !(use_cg && !has_dense && NA_as_zero);
        if (!has_dense && NA_as_zero && (!use_cg || has_weights))
        {
            if (pass_allocated_BtB &&
                (!keep_precomputedBtB || !add_diag_to_BtB))
            {
                buffer_size += 0;
            }
            else
            {
                buffer_size += square(k);
            }
        }

        size_t size_thread_buffer = square(k);
        if (use_cg)
            size_thread_buffer = max2(size_thread_buffer, (3 * k));
        if (use_cg && !has_dense)
            size_thread_buffer = (3 * k) + ((NA_as_zero && k >= n)? n : 0);
        if (nonneg)
            size_thread_buffer += k;
        else if (has_l1)
            size_thread_buffer += (size_t)3*k;
        return buffer_size + nthreads * size_thread_buffer;
    }
}

size_t buffer_size_optimizeA_implicit
(
    size_t k, size_t nthreads,
    bool pass_allocated_BtB,
    bool nonneg, bool has_l1,
    bool use_cg, bool finalize_chol
)
{
    if (finalize_chol && use_cg)
    {
        return max2(
            buffer_size_optimizeA_implicit(
                k, nthreads,
                pass_allocated_BtB,
                nonneg, has_l1,
                false, false
            ),
            buffer_size_optimizeA_implicit(
                k, nthreads,
                pass_allocated_BtB,
                nonneg, has_l1,
                true, false
            )
        );
    }

    size_t size_thread_buffer = use_cg? (3 * k) : (square(k));
    if (nonneg)
        size_thread_buffer += k;
    else if (has_l1)
        size_thread_buffer += (size_t)3*k;
    return (pass_allocated_BtB? (size_t)0 : square(k))
            + nthreads * size_thread_buffer;
}

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
)
{
    /* Note: the BtB produced here has diagonal, the one from
       collective doesn't. */
    /* TODO: handle case of all-missing when the values are not reset to zero */
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif
    char uplo = 'L';
    int_t ignore;
    *filled_BtB = false;


    if (Xfull == NULL) do_B = false;
    if (l1_lam || l1_lam_last || nonneg) use_cg = false;


    /* TODO: in many cases here, it's possible to shrink the buffer
       size for the threads when using 'use_cg'. */

    /* Case 1: X is full dense with few or no missing values.
       Here can apply the closed-form solution with only
       one multiplication by B for all rows at once.
       If there is a small amount of observations with missing
       values, can do a post-hoc pass over them to obtain their
       solutions individually. */
    if (Xfull != NULL && (full_dense || near_dense) && weight == NULL)
    {
        real_t *restrict bufferBtBcopy = NULL;
        if (near_dense)
        {
            if (precomputedBtB != NULL && !keep_precomputedBtB)
                bufferBtBcopy = precomputedBtB;
            else {
                bufferBtBcopy = buffer_real_t;
                buffer_real_t += square(k);
            }
        }
        real_t *restrict bufferBtB = NULL;
        if (precomputedBtB != NULL && !keep_precomputedBtB && !near_dense &&
            bufferBtBcopy != precomputedBtB)
        {
            bufferBtB = precomputedBtB;
        }
        else {
            bufferBtB = buffer_real_t;
            buffer_real_t += square(k);
        }
        /* TODO: this function should do away with 'bufferX', replace it instead
           with an 'incX' parameter. */
        real_t *restrict bufferX = buffer_real_t;
        if (do_B)
            buffer_real_t += (size_t)n*(size_t)nthreads;
        
        /* t(B)*B + diag(lam) */
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k, n,
                    1., B, ldb,
                    0., bufferBtB, k);
        if (precomputedBtB != NULL && keep_precomputedBtB) {
            copy_arr(bufferBtB, precomputedBtB, square(k));
            *filled_BtB = true;
        }
        add_to_diag(bufferBtB, scale_lam? (lam*(real_t)n) : (lam), k);
        if (lam_last != lam)
        {
            if (!scale_lam)
                bufferBtB[square(k)-1] += (lam_last - lam);
            else
                bufferBtB[square(k)-1] += (lam_last - lam) * (real_t)n;
        }
        /* Here will also need t(B)*B + diag(lambda) alone (no Cholesky) */
        if (bufferBtBcopy != NULL)
            memcpy(bufferBtBcopy, bufferBtB, (size_t)square(k)*sizeof(real_t));
        
        /* t(B)*t(X)
           Note: this will be passed to LAPACK function which assumes
           column-major order, thus must pass the transpose.
           Note2: if passing 'do_B=true', the inputs will all be
           swapped, so what here says 'A' will be 'B', what says
           'm' will be 'n', and so on. But the matrix 'X' remains
           the same, so the inputs need to be transposed for B. */
        if (!do_B)
            cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, k, n,
                        1., Xfull, n, B, ldb,
                        0., A, lda);
        else
            cblas_tgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m, k, n,
                        1., Xfull, ldX, B, ldb,
                        0., A, lda);

        
        #ifdef FORCE_NO_NAN_PROPAGATION
        if (!full_dense && !nonneg && !l1_lam && !l1_lam_last)
            #pragma omp parallel for schedule(static) \
                    num_threads(min2(4, nthreads)) \
                    shared(A, m, lda)
            for (size_t_for ix = 0;
                 ix < (size_t)m*(size_t)lda - (size_t)(lda-k);
                 ix++)
                A[ix] = isnan(A[ix])? 0. : A[ix];
        #endif
        /* A = t( inv(t(B)*B + diag(lam)) * t(B)*t(X) )
           Note: don't try to flip the equation as the 'posv'
           function assumes only the LHS is symmetric. */
        if (!nonneg && !l1_lam && !l1_lam_last)
            tposv_(&uplo, &k, &m,
                   bufferBtB, &k,
                   A, &lda,
                   &ignore);
        else if (!nonneg)
            solve_elasticnet_batch(
                bufferBtB,
                A,
                buffer_real_t,
                m, k, lda,
                scale_lam? (l1_lam*n) : (l1_lam),
                scale_lam? (l1_lam_last*n) : (l1_lam_last),
                max_cd_steps,
                nthreads
            );
        else
            solve_nonneg_batch(
                bufferBtB,
                A,
                buffer_real_t,
                m, k, lda,
                scale_lam? (l1_lam*n) : (l1_lam),
                scale_lam? (l1_lam_last*n) : (l1_lam_last),
                max_cd_steps,
                nthreads
            );

        /* If there are some few rows with missing values, now do a
           post-hoc pass over them only */
        if (!full_dense)
        {
            size_t size_buffer = square(k);
            if (use_cg)
                size_buffer = max2(size_buffer, (size_t)(3 * k));
            if (nonneg)
                size_buffer += k;
            else if (l1_lam || l1_lam_last)
                size_buffer += (size_t)3*(size_t)k;

            #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                    shared(A, lda, B, ldb, ldX, m, n, k, \
                           scale_lam, scale_bias_const, \
                           lam, lam_last, l1_lam, l1_lam_last, weight,\
                           cnt_NA, Xfull, buffer_real_t, bufferBtBcopy, \
                           nthreads, use_cg, nonneg, max_cd_steps) \
                    firstprivate(bufferX)
            for (size_t_for ix = 0; ix < (size_t)m; ix++)
                if (cnt_NA[ix] > 0)
                {
                    if (cnt_NA[ix] == n) {
                        set_to_zero(A + ix*(size_t)lda, k);
                        continue;
                    }

                    if (!do_B)
                        bufferX = Xfull + ix*(size_t)n;
                    else
                        cblas_tcopy(n, Xfull + ix, ldX,
                                    bufferX
                            + ((size_t)n*(size_t)omp_get_thread_num()), 1);

                    if (use_cg)
                    {
                        set_to_zero(A + ix*(size_t)lda, k);
                        if (bias_restore != NULL)
                            A[ix*(size_t)lda + (size_t)(k-1)] =bias_restore[ix];
                    }

                    factors_closed_form(
                        A + ix*(size_t)lda, k,
                        B, n, ldb,
                        bufferX + (do_B?
                                    ((size_t)n*(size_t)omp_get_thread_num())
                                        :
                                    ((size_t)0)), false,
                        (real_t*)NULL, (int_t*)NULL, (size_t)0,
                        (real_t*)NULL,
                        buffer_real_t
                         + size_buffer * (size_t)omp_get_thread_num(),
                        lam, lam_last,
                        l1_lam, l1_lam_last,
                        scale_lam, scale_bias_const, 0.,
                        (real_t*)NULL,
                        bufferBtBcopy, cnt_NA[ix], k,
                        true, false, 1., n,
                        (real_t*)NULL, false,
                        use_cg, k, /* <- A was reset to zero, need more steps */
                        nonneg, max_cd_steps,
                        (real_t*)NULL, (real_t*)NULL, 0., 1.,
                        false
                    );
                }
        }
    }


    /* Case 2: X is dense, but has many missing values or has weights.
       Here will do them all individually, pre-calculating only
         t(B)*B + diag(lam)
       in case some rows have few missing values. */
    else if (Xfull != NULL)
    {
        real_t *restrict bufferBtB = NULL;
        if (weight == NULL) {
            if (precomputedBtB != NULL && !keep_precomputedBtB)
                bufferBtB = precomputedBtB;
            else {
                bufferBtB = buffer_real_t;
                buffer_real_t += square(k);
            }
        }
        real_t *restrict bufferX = NULL;
        if (do_B) {
            bufferX = buffer_real_t;
            buffer_real_t += (size_t)n * (size_t)nthreads;
        }
        real_t *restrict bufferW = NULL;
        if (do_B && weight != NULL) {
            bufferW = buffer_real_t;
            buffer_real_t += (size_t)n * (size_t)nthreads;
        }
        real_t *restrict buffer_remainder = buffer_real_t;


        if (bufferBtB != NULL)
        {
            cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        k, n,
                        1., B, ldb,
                        0., bufferBtB, k);
            if (keep_precomputedBtB && precomputedBtB != NULL)
            {
                copy_arr(bufferBtB, precomputedBtB, square(k));
                *filled_BtB = true;
            }
            add_to_diag(bufferBtB, lam, k);
            if (lam_last != lam) bufferBtB[square(k)-1] += (lam_last - lam);
        }

        size_t size_buffer = (size_t)square(k) + (size_t)(use_cg? (3*k) : 0);

        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                shared(Xfull, weight, do_B, m, n, k, A, lda, B, ldb, ldX, \
                       lam, lam_last, l1_lam, l1_lam_last, \
                       scale_lam, scale_bias_const, wsumA, \
                       bufferBtB, cnt_NA, buffer_remainder, \
                       use_cg, max_cg_steps, nonneg, max_cd_steps) \
                firstprivate(bufferX, bufferW)
        for (size_t_for ix = 0; ix < (size_t)m; ix++)
        {
            if (!do_B) {
                bufferX = Xfull + ix*(size_t)n;
                if (weight != NULL)
                    bufferW = weight + ix*(size_t)n;
            }
            else {
                cblas_tcopy(n, Xfull + ix, ldX,
                            bufferX
                                + ((size_t)n*(size_t)omp_get_thread_num()), 1);
                if (weight != NULL)
                    cblas_tcopy(n, weight + ix, ldX,
                                bufferW
                            + ((size_t)n*(size_t)omp_get_thread_num()), 1);
            }

            factors_closed_form(
                A + ix*(size_t)lda, k,
                B, n, ldb,
                bufferX + (do_B? ((size_t)n*(size_t)omp_get_thread_num())
                                    : ((size_t)0)),
                cnt_NA[ix] == 0,
                (real_t*)NULL, (int_t*)NULL, (size_t)0,
                (weight != NULL)?
                    (bufferW + (do_B? ((size_t)n*(size_t)omp_get_thread_num())
                                        : ((size_t)0)))
                      :
                    ((real_t*)NULL),
                buffer_remainder + size_buffer*(size_t)omp_get_thread_num(),
                lam, lam_last,
                l1_lam, l1_lam_last,
                scale_lam, scale_bias_const,
                (weight == NULL || wsumA == NULL)? 0. : wsumA[ix],
                (real_t*)NULL,
                bufferBtB, cnt_NA[ix], k,
                true, false, 1., n,
                (real_t*)NULL, false,
                use_cg, max_cg_steps,
                nonneg, max_cd_steps,
                (real_t*)NULL, (real_t*)NULL, 0., 1.,
                false
            );
        }
    }

    /* Case 3: X is sparse, with missing-as-zero, and no weights.
       Here can also use one Cholesky for all rows at once. */
    else if (Xfull == NULL && NA_as_zero && weight == NULL)
    {
        real_t *restrict bufferBtB = NULL;
        if (precomputedBtB != NULL && !keep_precomputedBtB)
            bufferBtB = precomputedBtB;
        else {
            bufferBtB = buffer_real_t;
            buffer_real_t += square(k);
        }

        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k, n,
                    1., B, ldb,
                    0., bufferBtB, k);
        if (keep_precomputedBtB && precomputedBtB != NULL)
        {
            copy_arr(bufferBtB, precomputedBtB, square(k));
            *filled_BtB = true;
        }
        add_to_diag(bufferBtB, scale_lam? (lam*(real_t)n) : (lam), k);
        if (lam_last != lam)
        {
            if (!scale_lam)
                bufferBtB[square(k)-1] += (lam_last - lam);
            else
                bufferBtB[square(k)-1] += (lam_last - lam) * (real_t)n;
        }
        if (lda == k)
            set_to_zero_(A, (size_t)m*(size_t)k, nthreads);
        else
            for (size_t row = 0; row < (size_t)m; row++)
                memset(A + row*(size_t)lda, 0, (size_t)k*sizeof(real_t));
        tgemm_sp_dense(
            m, k, 1.,
            Xcsr_p, Xcsr_i, Xcsr,
            B, (size_t)ldb,
            A, (size_t)lda,
            nthreads
        );
        if (bias_BtX != NULL) {
            multiplier_bias_BtX = 1./multiplier_bias_BtX;
            for (size_t row = 0; row < (size_t)m; row++)
                cblas_taxpy(k, multiplier_bias_BtX,
                            bias_BtX, 1, A + row*(size_t)lda, 1);
        }
        else if (bias_static != NULL) {
            /* Note: this should only be encountered when optimizing the side
               info matrices, thus there are no extra routes for the other
               cases, as it can only happen in this particular situation. */
            real_t *restrict buffer_colsumsB = buffer_real_t;
            buffer_real_t += k;

            set_to_zero(buffer_colsumsB, k);
            sum_by_cols(B, buffer_colsumsB, n, k, ldb, nthreads);
            cblas_tger(CblasRowMajor, m, k,
                       1., bias_static, 1,
                       buffer_colsumsB, 1,
                       A, lda);
        }

        if (!nonneg && !l1_lam && !l1_lam_last)
            tposv_(&uplo, &k, &m,
                   bufferBtB, &k,
                   A, &lda,
                   &ignore);
        else if (!nonneg)
            solve_elasticnet_batch(
                bufferBtB,
                A,
                buffer_real_t,
                m, k, lda,
                scale_lam? (l1_lam*(real_t)n) : (l1_lam),
                scale_lam? (l1_lam_last*(real_t)n) : (l1_lam_last),
                max_cd_steps,
                nthreads
            );
        else
            solve_nonneg_batch(
                bufferBtB,
                A,
                buffer_real_t,
                m, k, lda,
                scale_lam? (l1_lam*(real_t)n) : (l1_lam),
                scale_lam? (l1_lam_last*(real_t)n) : (l1_lam_last),
                max_cd_steps,
                nthreads
            );
    }

    /* Case 4: X is sparse, with non-present as NA, or with weights.
       This is the expected case for most situations. */
    else
    {
        if (is_first_iter)
        {
            set_to_zero_(A, (size_t)m*(size_t)lda - (size_t)(lda-k), nthreads);
            if (use_cg && bias_restore != NULL)
                cblas_tcopy(m, bias_restore, 1, A + (k-1), lda);
        }

        /* When NAs are treated as zeros, can use a precomputed t(B)*B */
        real_t *restrict bufferBtB = NULL;
        bool add_diag_to_BtB = !(use_cg && Xfull == NULL && NA_as_zero) &&
                               !scale_lam;
        if (Xfull == NULL && NA_as_zero && (!use_cg || weight != NULL))
        {
            if (precomputedBtB != NULL &&
                (!keep_precomputedBtB || !add_diag_to_BtB))
            {
                bufferBtB = precomputedBtB;
            }
            else
            {
                bufferBtB = buffer_real_t;
                buffer_real_t += square(k);
            }
            cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        k, n,
                        1., B, ldb,
                        0., bufferBtB, k);
            if (precomputedBtB != NULL && keep_precomputedBtB &&
                bufferBtB != precomputedBtB)
            {
                copy_arr(bufferBtB, precomputedBtB, square(k));
            }
            if (add_diag_to_BtB)
            {
                add_to_diag(bufferBtB, lam, k);
                if (lam_last != lam) bufferBtB[square(k)-1] += (lam_last - lam);
            }
            if (precomputedBtB != NULL)
                *filled_BtB = true;
        }

        size_t size_buffer = square(k);
        if (use_cg)
            size_buffer = max2(size_buffer, (size_t)(3 * k));
        if (use_cg && Xfull == NULL)
            size_buffer = (size_t)(3 * k)
                        + (size_t)((NA_as_zero && k >= n)? n : 0);
        if (nonneg)
            size_buffer += k;
        else if (l1_lam || l1_lam_last)
            size_buffer += (size_t)3*(size_t)k;

        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                shared(A, lda, B, ldb, m, n, k, \
                       scale_lam, scale_bias_const, wsumA, \
                       lam, lam_last, l1_lam, l1_lam_last, weight, cnt_NA, \
                       Xcsr_p, Xcsr_i, Xcsr, buffer_real_t, NA_as_zero, \
                       bufferBtB, size_buffer, use_cg, \
                       nonneg, max_cd_steps, bias_BtX, bias_X, \
                       bias_X_glob, multiplier_bias_BtX)
        for (size_t_for ix = 0; ix < (size_t)m; ix++)
        {
            if ((Xcsr_p[ix+(size_t)1] > Xcsr_p[ix]) ||
                (NA_as_zero && bias_BtX != NULL))
            {
                factors_closed_form(
                    A + ix*(size_t)lda, k,
                    B, n, ldb,
                    (real_t*)NULL, false,
                    Xcsr +  Xcsr_p[ix], Xcsr_i +  Xcsr_p[ix],
                    Xcsr_p[ix+(size_t)1] - Xcsr_p[ix],
                    (weight == NULL)? ((real_t*)NULL) : (weight + Xcsr_p[ix]),
                    buffer_real_t + ((size_t)omp_get_thread_num()*size_buffer),
                    lam, lam_last,
                    l1_lam, l1_lam_last,
                    scale_lam, scale_bias_const,
                    (weight == NULL || wsumA == NULL)? 0. : wsumA[ix],
                    (real_t*)NULL,
                    bufferBtB, 0, k,
                    add_diag_to_BtB, false, 1., n,
                    (real_t*)NULL, NA_as_zero,
                    use_cg, max_cg_steps,
                    nonneg, max_cd_steps,
                    bias_BtX, bias_X, bias_X_glob, multiplier_bias_BtX,
                    false
                );
            }
        }
    }
}

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
)
{
    if (nonneg || l1_lam) use_cg = false;
    if (precomputedBtB == NULL)
    {
        precomputedBtB = buffer_real_t;
        buffer_real_t += square(k);
    }
    /* TODO: keep it as a parameter whether the BtB comes precomputed or not,
       so as to incorporate this function later into the 'factors_multiple'. */
    cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                k, n,
                1., B, ldb,
                0., precomputedBtB, k);
    if (!use_cg)
        add_to_diag(precomputedBtB, lam, k);
    if (!use_cg || force_set_to_zero)
        set_to_zero_(A, (size_t)m*(size_t)k - (lda-(size_t)k), nthreads);
    size_t size_buffer = use_cg? (3 * k) : (square(k));
    if (nonneg)
        size_buffer += k;
    else if (l1_lam)
        size_buffer += (size_t)3*(size_t)k;


    int_t ix = 0;

    if (use_cg)
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                shared(A, B, lda, ldb, m, n, k, lam, \
                       Xcsr, Xcsr_i, Xcsr_p, precomputedBtB, buffer_real_t, \
                       max_cg_steps, size_buffer)
        for (ix = 0; ix < m; ix++)
            if (Xcsr_p[ix+(size_t)1] > Xcsr_p[ix])
                factors_implicit_cg(
                    A + (size_t)ix*lda, k,
                    B, ldb,
                    Xcsr + Xcsr_p[ix], Xcsr_i + Xcsr_p[ix],
                    Xcsr_p[ix+(size_t)1] - Xcsr_p[ix],
                    lam,
                    precomputedBtB, k,
                    max_cg_steps,
                    buffer_real_t + ((size_t)omp_get_thread_num() * size_buffer)
                );
    }
    
    else
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                shared(A, B, lda, ldb, m, n, k, lam, l1_lam, \
                       Xcsr, Xcsr_i, Xcsr_p, precomputedBtB, buffer_real_t, \
                       size_buffer, nonneg, max_cd_steps)
        for (ix = 0; ix < m; ix++)
            if (Xcsr_p[ix+(size_t)1] > Xcsr_p[ix])
                factors_implicit_chol(
                    A + (size_t)ix*lda, k,
                    B, ldb,
                    Xcsr + Xcsr_p[ix], Xcsr_i + Xcsr_p[ix],
                    Xcsr_p[ix+(size_t)1] - Xcsr_p[ix],
                    lam, l1_lam,
                    precomputedBtB, k,
                    nonneg, max_cd_steps,
                    buffer_real_t + ((size_t)omp_get_thread_num() * size_buffer)
                );
    }
}

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
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif
    if (glob_mean == NULL)
        return;
    if (!center)
    {
        *glob_mean = 0;
        return;
    }

    size_t m_by_n = (Xfull == NULL)? 0 : ((size_t)m * (size_t)n);

    double xsum = 0.;
    double wsum = 0.;
    size_t cnt = 0;

    if (weight == NULL)
    {
        if (Xfull != NULL)
        {
            #ifdef _OPENMP
            if (nthreads >= 8)
            {
                #pragma omp parallel for schedule(static) \
                        num_threads(nthreads) \
                        reduction(+:xsum,cnt) shared(Xfull, m_by_n)
                for (size_t_for ix = 0; ix < m_by_n; ix++) {
                    xsum += (!isnan(Xfull[ix]))? Xfull[ix] : 0;
                    cnt += !isnan(Xfull[ix]);
                }
                *glob_mean = (real_t)(xsum / (double)cnt);
            }

            else
            #endif
            {
                for (size_t ix = 0; ix < m_by_n; ix++) {
                    xsum += isnan(Xfull[ix])?
                              0 : ((Xfull[ix] - xsum) / (double)(++cnt));
                }
                *glob_mean = xsum;
            }

            if (!cnt)
                fprintf(stderr, "Warning: 'X' has all entries missing.\n");
        }

        else
        {
            #ifdef _OPENMP
            if (nthreads >= 8)
            {
                #pragma omp parallel for schedule(static) \
                        num_threads(nthreads) \
                        reduction(+:xsum) shared(X, nnz)
                for (size_t_for ix = 0; ix < nnz; ix++)
                    xsum += X[ix];
                *glob_mean = (real_t)(xsum / (double)nnz);
            }

            else
            #endif
            {
                for (size_t ix = 0; ix < nnz; ix++)
                    xsum += (X[ix] - xsum) / (double)(++cnt);
                *glob_mean = xsum;
            }

            if (!xsum)
                fprintf(stderr, "Warning: 'X' has only zeros.\n");

            if (NA_as_zero)
                *glob_mean = (long double)(*glob_mean)
                                *
                             ((long double)nnz
                                    / 
                              ((long double)m * (long double)n));
        }
    }

    else /* <- has weights */
    {
        if (Xfull != NULL)
        {
            #ifdef _OPENMP
            if (nthreads >= 8)
            {
                #pragma omp parallel for schedule(static) \
                        num_threads(nthreads) \
                        reduction(+:xsum,wsum) shared(Xfull, weight, m_by_n)
                for (size_t_for ix = 0; ix < m_by_n; ix++) {
                    xsum += (!isnan(Xfull[ix]))? (Xfull[ix]) : (0);
                    wsum += isnan(Xfull[ix])? 0. : weight[ix];
                }
                *glob_mean = (real_t)(xsum / wsum);
            }

            else
            #endif
            {
                wsum = DBL_EPSILON;
                for (size_t ix = 0; ix < m_by_n; ix++) {
                    xsum += isnan(Xfull[ix])?
                              0 : (((Xfull[ix] - xsum) * weight[ix])
                                        /
                                   (wsum += weight[ix]));
                }
                *glob_mean = xsum;
            }
        }

        else
        {
            #ifdef _OPENMP
            if (nthreads >= 8)
            {
                #pragma omp parallel for schedule(static) \
                        num_threads(nthreads) \
                        reduction(+:xsum,wsum) shared(X, weight, nnz)
                for (size_t_for ix = 0; ix < nnz; ix++) {
                    xsum += X[ix];
                    wsum += weight[ix];
                }
                *glob_mean = (real_t)(xsum / wsum);
            }

            else
            #endif
            {
                wsum = DBL_EPSILON;
                for (size_t ix = 0; ix < nnz; ix++) {
                    xsum += ((X[ix] - xsum) * weight[ix])
                                /
                            (wsum += weight[ix]);
                }
                *glob_mean = xsum;
            }

            if (NA_as_zero)
            {
                long double wsum_l = compensated_sum(weight, nnz);
                *glob_mean = (long double)(*glob_mean)
                                /
                             (wsum_l / (wsum_l + ((long double)m*(long double)n
                                                   - (long double)nnz)));
            }
        }

        if (wsum <= 0)
            fprintf(stderr, "Warning: weights are not positive.\n");
    }

    if (nonneg)
        *glob_mean = fmax_t(*glob_mean, 0);

    /* If the obtained mean is too small, simply ignore it */
    if (fabs_t(*glob_mean) < sqrt_t(EPSILON_T))
        *glob_mean = 0;

    /* Now center X in-place */
    if (*glob_mean != 0 && !(Xfull == NULL && NA_as_zero))
    {
        if (Xfull != NULL) {
            for (size_t_for ix = 0; ix < m_by_n; ix++)
                Xfull[ix] = isnan(Xfull[ix])?
                              (NAN_) : (Xfull[ix] - (*glob_mean));
            if (Xtrans != NULL) {
                for (size_t_for ix = 0; ix < m_by_n; ix++)
                    Xtrans[ix] = isnan(Xtrans[ix])?
                                   (NAN_) : (Xtrans[ix] - (*glob_mean));
            }
        } else if (Xcsr != NULL) {
            for (size_t_for ix = 0; ix < nnz; ix++) {
                Xcsr[ix] -= *glob_mean;
                Xcsc[ix] -= *glob_mean;
            }
        } else {
            for (size_t_for ix = 0; ix < nnz; ix++)
                X[ix] -= *glob_mean;
        }
    }
}

/* TODO: factor out this function */
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
)
{
    int_t retval = 0;
    size_t m_by_n = (Xfull == NULL)? (size_t)0 : ((size_t)m * (size_t)n);
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long row, col;
    #endif

    size_t *restrict buffer_cnt = NULL;
    double *restrict buffer_w = NULL;

    size_t cnt = 0;
    long double wsum = 0;

    if (user_bias || item_bias)
    {
        size_t dim_buffer = (user_bias && item_bias)?
                                max2(m_bias, n_bias)
                                    :
                                (user_bias? m_bias : n_bias);
        if (weight == NULL)
        {
            buffer_cnt = (size_t*)calloc(dim_buffer, sizeof(size_t));
            if (buffer_cnt == NULL) goto throw_oom;
        }

        else
        {
            buffer_w = (double*)calloc(dim_buffer, sizeof(double));
            if (buffer_w == NULL) goto throw_oom;
        }
    }


    /* First calculate the global mean */
    if (center)
    {
        calc_mean_and_center(
            ixA, ixB, X, nnz,
            Xfull, Xtrans,
            m, n,
            Xcsr_p, Xcsr_i, Xcsr,
            Xcsc_p, Xcsc_i, Xcsc,
            weight,
            false, nonneg, center, nthreads,
            glob_mean
        );
    }

    /* If not centering, might still need to know the number of non-missing
       entries in order to calculate the scaling of the biases. */
    else if ((Xfull != NULL || weight != NULL) &&
             (user_bias || item_bias ||
              force_calc_user_scale || force_calc_item_scale) &&
             scale_lam && scale_bias_const)
    {
        if (weight == NULL)
        {
            if (Xfull != NULL)
                for (size_t ix = 0; ix < m_by_n; ix++)
                    cnt += !isnan(Xfull[ix]);
        }

        else
        {
            if (Xfull != NULL)
                for (size_t ix = 0; ix < m_by_n; ix++)
                    wsum += isnan(Xfull[ix])? 0. : weight[ix];
            else
                for (size_t ix = 0; ix < nnz; ix++)
                    wsum += weight[ix];
        }
    }

    if ((user_bias || item_bias ||
         force_calc_user_scale || force_calc_item_scale) &&
        scale_lam && scale_bias_const)
    {
        if (weight == NULL)
        {
            if (Xfull != NULL)
            {
                if (user_bias || force_calc_user_scale)
                    *scaling_biasA = (long double)cnt / (long double)m;
                if (item_bias || force_calc_item_scale)
                    *scaling_biasB = (long double)cnt / (long double)n;
            }

            else
            {
                if (user_bias || force_calc_user_scale)
                    *scaling_biasA = (long double)nnz / (long double)m;
                if (item_bias || force_calc_item_scale)
                    *scaling_biasB = (long double)nnz / (long double)n;
            }
        }

        else
        {
            if (user_bias || force_calc_user_scale)
                *scaling_biasA = (long double)wsum / (long double)m;
            if (item_bias || force_calc_item_scale)
                *scaling_biasB = (long double)wsum / (long double)n;
        }

        if (user_bias || force_calc_user_scale)
            lam_user *= *scaling_biasA;
        if (item_bias || force_calc_item_scale)
            lam_item *= *scaling_biasB;
        scale_lam = false;
        scale_bias_const = false;
    }


    /* Note: the original papers suggested starting these values by
       obtaining user biases first, then item biases, but I've found
       that doing it the other way around leads to better results
       with the ALS method **when** the ALS method updates the main matrices
       in that same order (which this software does, unlike most other
       implementations). Thus, both the ALS procedure and this function
       make updates over items first and users later. */

    /* Calculate item biases, but don't apply them to X */
    if (item_bias)
    {
        if (Xtrans != NULL)
        {
            if (weight == NULL)
            {
                #pragma omp parallel for schedule(static) \
                        num_threads(cap_to_4(nthreads)) \
                        shared(m, n_bias, Xtrans, biasB, buffer_cnt)
                for (size_t_for col = 0; col < (size_t)n_bias; col++)
                {
                    double bsum = 0;
                    size_t cnt = 0;
                    for (size_t row = 0; row < (size_t)m; row++)
                    {
                        bsum += (!isnan(Xtrans[row + col*(size_t)m]))?
                                    Xtrans[row + col*(size_t)m] : 0.;
                        cnt += !isnan(Xtrans[row + col*(size_t)m]);
                    }
                    biasB[col] = bsum;
                    buffer_cnt[col] = cnt;
                }
            }

            else
            {
                #pragma omp parallel for schedule(static) \
                        num_threads(cap_to_4(nthreads)) \
                        shared(m, n_bias, Xtrans, Wtrans, biasB, buffer_w)
                for (size_t_for col = 0; col < (size_t)n_bias; col++)
                {
                    double bsum = 0;
                    double wsum = 0;
                    for (size_t row = 0; row < (size_t)m; row++)
                    {
                        bsum += (!isnan(Xtrans[row + col*(size_t)m]))?
                                    Xtrans[row + col*(size_t)m] : 0.;
                        wsum += (!isnan(Xtrans[row + col*(size_t)m]))?
                                    Wtrans[row + col*(size_t)m] : 0.;
                    }
                    biasB[col] = bsum;
                    buffer_w[col] = wsum;
                }
            }

        }

        else if (Xfull != NULL)
        {
            if (weight == NULL)
            {
                for (size_t row = 0; row < (size_t)m; row++)
                    for (size_t col = 0; col < (size_t)n_bias; col++) {
                        biasB[col] += (!isnan(Xfull[col + row*(size_t)n]))?
                                       (Xfull[col + row*(size_t)n]) : (0.);
                        buffer_cnt[col] += !isnan(Xfull[col + row*(size_t)n]);
                    }
            }

            else
            {
                for (size_t row = 0; row < (size_t)m; row++)
                    for (size_t col = 0; col < (size_t)n_bias; col++) {
                        biasB[col] += (!isnan(Xfull[col + row*(size_t)n]))?
                                       (Xfull[col + row*(size_t)n]) : (0.);
                        buffer_w[col] += (!isnan(Xfull[col + row*(size_t)n]))?
                                          (weight[col + row*(size_t)n]) : (0.);
                    }
            }
        }

        else if (Xcsc != NULL)
        {
            if (weight == NULL)
            {
                #pragma omp parallel for schedule(dynamic) \
                        num_threads(cap_to_4(nthreads)) \
                        shared(n_bias, Xcsc_p, Xcsc, biasB, buffer_cnt)
                for (size_t_for col = 0; col < (size_t)n_bias; col++)
                {
                    buffer_cnt[col] = Xcsc_p[col+(size_t)1] - Xcsc_p[col];
                    double bsum = 0;
                    for (size_t ix=Xcsc_p[col]; ix < Xcsc_p[col+(size_t)1];ix++)
                    {
                        bsum += Xcsc[ix];
                    }
                    biasB[col] = bsum;
                }
            }

            else
            {
                #pragma omp parallel for schedule(dynamic) \
                        num_threads(cap_to_4(nthreads)) \
                        shared(n_bias, Xcsc_p, Xcsc, biasB, weightC, buffer_w)
                for (size_t_for col = 0; col < (size_t)n_bias; col++)
                {
                    double bsum = 0;
                    double wsum = 0;
                    for (size_t ix=Xcsc_p[col]; ix < Xcsc_p[col+(size_t)1];ix++)
                    {
                        bsum += Xcsc[ix];
                        wsum += weightC[ix];
                    }
                    biasB[col] = bsum;
                    buffer_w[col] = wsum;
                }
            }
        }

        else
        {
            if (weight == NULL)
            {
                for (size_t ix = 0; ix < nnz; ix++) {
                    biasB[ixB[ix]] += X[ix];
                    buffer_cnt[ixB[ix]]++;
                }
            }

            else
            {
                for (size_t ix = 0; ix < nnz; ix++) {
                    biasB[ixB[ix]] += X[ix];
                    buffer_w[ixB[ix]] += weight[ix];
                }
            }
        }

        if (weight == NULL)
        {
            for (int_t ix = 0; ix < n_bias; ix++)
                biasB[ix] /= ((double)buffer_cnt[ix]
                                + (lam_item
                                    *
                                   (scale_lam? (double)buffer_cnt[ix] : 1.)));
        }

        else
        {
            for (int_t ix = 0; ix < n_bias; ix++)
                biasB[ix] /= (buffer_w[ix]
                                + (lam_item
                                    *
                                   (scale_lam? buffer_w[ix] : 1.)));
        }

        for (int_t ix = 0; ix < n_bias; ix++)
            biasB[ix] = (!isnan(biasB[ix]))? biasB[ix] : 0.;

        if (nonneg)
        {
            for (int_t ix = 0; ix < n_bias; ix++)
                biasB[ix] = (biasB[ix] >= 0.)? biasB[ix] : 0.;
        }
    }

    /* Finally, user biases */
    if (user_bias)
    {
        if (item_bias)
        {
            if (buffer_cnt != NULL)
                memset(buffer_cnt, 0, (size_t)m_bias*sizeof(size_t));
            if (buffer_w != NULL)
                memset(buffer_w, 0, (size_t)m_bias*sizeof(double));
        }

        if (Xfull != NULL)
        {
            if (weight == NULL)
            {
                #pragma omp parallel for schedule(static) \
                        num_threads(cap_to_4(nthreads)) \
                        shared(m_bias, n, Xfull, biasA, biasB, buffer_cnt)
                for (size_t_for row = 0; row < (size_t)m_bias; row++)
                {
                    double asum = 0;
                    size_t cnt = 0;
                    for (size_t col = 0; col < (size_t)n; col++) {
                        asum += (!isnan(Xfull[col + row*(size_t)n]))?
                                    (Xfull[col + row*(size_t)n]
                                       - (item_bias? biasB[col] : 0.))
                                     : (0.);
                        cnt += !isnan(Xfull[col + row*(size_t)n]);
                    }
                    biasA[row] = asum;
                    buffer_cnt[row] = cnt;
                }
            }

            else
            {
                #pragma omp parallel for schedule(static) \
                        num_threads(cap_to_4(nthreads)) \
                        shared(m_bias, n, Xfull, biasA, biasB, buffer_w)
                for (size_t_for row = 0; row < (size_t)m_bias; row++)
                {
                    double asum = 0;
                    double wsum = 0;
                    for (size_t col = 0; col < (size_t)n; col++) {
                        asum += (!isnan(Xfull[col + row*(size_t)n]))?
                                    (Xfull[col + row*(size_t)n]
                                      - (item_bias? biasB[col] : 0.))
                                    : (0.);
                        wsum += (!isnan(Xfull[col + row*(size_t)n]))?
                                    weight[col + row*(size_t)n] : 0.;
                    }
                    biasA[row] = asum;
                    buffer_w[row] = wsum;
                }
            }
        }

        else if (Xcsr != NULL)
        {
            if (weight == NULL)
            {
                #pragma omp parallel for schedule(dynamic) \
                        num_threads(nthreads) \
                        shared(m_bias, Xcsr_p, Xcsr_i, Xcsr, biasA, biasB, \
                               item_bias, buffer_cnt)
                for (size_t_for row = 0; row < (size_t)m_bias; row++)
                {
                    buffer_cnt[row] = Xcsr_p[row+(size_t)1] - Xcsr_p[row];
                    double asum = 0;
                    for (size_t ix=Xcsr_p[row]; ix < Xcsr_p[row+(size_t)1];ix++)
                    {
                        asum += Xcsr[ix] - (item_bias? biasB[Xcsr_i[ix]] : 0.);
                    }
                    biasA[row] = asum;
                }
            }

            else
            {
                #pragma omp parallel for schedule(dynamic) \
                        num_threads(cap_to_4(nthreads)) \
                        shared(m_bias, Xcsr_p, Xcsr_i, Xcsr, biasA, biasB, \
                               item_bias, buffer_w, weightR)
                for (size_t_for row = 0; row < (size_t)m_bias; row++)
                {
                    double asum = 0;
                    double wsum = 0;
                    for (size_t ix=Xcsr_p[row]; ix < Xcsr_p[row+(size_t)1];ix++)
                    {
                        asum += Xcsr[ix] - (item_bias? biasB[Xcsr_i[ix]] : 0.);
                        wsum += weightR[ix];
                    }
                    biasA[row] = asum;
                    buffer_w[row] = wsum;
                }
            }
        }

        else
        {
            if (weight == NULL)
            {
                for (size_t ix = 0; ix < nnz; ix++) {
                    biasA[ixA[ix]]
                        +=
                    X[ix] - (item_bias? (biasB[ixB[ix]]) : (0.));
                    buffer_cnt[ixA[ix]]++;
                }
            }

            else
            {
                for (size_t ix = 0; ix < nnz; ix++) {
                    biasA[ixA[ix]]
                        +=
                    X[ix] - (item_bias? (biasB[ixB[ix]]) : (0.));
                    buffer_w[ixA[ix]] += weight[ix];
                }
            }
        }

        if (weight == NULL)
        {
            for (int_t ix = 0; ix < m_bias; ix++)
                biasA[ix] /= ((double)buffer_cnt[ix]
                                + (lam_user
                                    *
                                   (scale_lam? (double)buffer_cnt[ix] : 1.)));
        }

        else
        {
            for (int_t ix = 0; ix < m_bias; ix++)
                biasA[ix] /= (buffer_w[ix]
                                + (lam_user
                                    *
                                   (scale_lam? buffer_w[ix] : 1.)));
        }

        for (int_t ix = 0; ix < m_bias; ix++)
            biasA[ix] = (!isnan(biasA[ix]))? biasA[ix] : 0.;

        if (nonneg)
        {
            for (int_t ix = 0; ix < m_bias; ix++)
                biasA[ix] = (biasA[ix] >= 0.)? biasA[ix] : 0.;
        }
    }

    cleanup:
        free(buffer_cnt);
        free(buffer_w);
    return retval;

    throw_oom:
        retval = 1;
        goto cleanup;
}

/* Here, 'X' should already be centered, unless using 'NA_as_zero'. */
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
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long row;
    #endif

    if (fabs_t(lam) < EPSILON_T)
        lam = EPSILON_T;

    if (Xfull != NULL && weight == NULL)
    {
        if (!do_B)
        {
            #pragma omp parallel for schedule(static) \
                    num_threads(cap_to_4(nthreads)) \
                    shared(Xfull, m, n, biasA, wsumA, scale_lam, lam)
            for (size_t_for row = 0; row < (size_t)m; row++)
            {
                double bmean = 0;
                int_t cnt = 0;
                for (size_t col = 0; col < (size_t)n; col++)
                {
                    bmean += (isnan(Xfull[col + row*n]))?
                                (0) : ((Xfull[col + row*n] - bmean)
                                            /
                                       (double)(++cnt));
                }
                bmean *= (double)cnt
                            /
                         ((double)cnt
                            +
                          (lam * ((wsumA != NULL)?
                                    ((double)wsumA[row])
                                        :
                                    (scale_lam? (double)max2(cnt,1) : 1.))));
                biasA[row] = bmean;
            }
        }

        else /* <- in this case, X is transposed */
        {
            set_to_zero(biasA, m);
            int_t tmp = m; m = n; n = tmp;
            real_t *restrict biasB = biasA;
            for (size_t row = 0; row < (size_t)m; row++)
                for (size_t col = 0; col < (size_t)n; col++)
                    biasB[col] += (isnan(Xfull[col + row*n]))?
                                    0 : Xfull[col + row*n];
            for (int_t col = 0; col < n; col++)
                biasB[col] /= (double)(n - cnt_NA[col])
                                +
                              (lam * ((wsumA != NULL)?
                                  ((double)wsumA[col])
                                      :
                                  (scale_lam?
                                    (double)max2(n - cnt_NA[col], 1) : 1.)));
        }
    }

    else if (Xfull != NULL) /* <- has weights */
    {
        if (!do_B)
        {
            #pragma omp parallel for schedule(static) \
                    num_threads(cap_to_4(nthreads)) \
                    shared(Xfull, m, n, biasA, weight, wsumA, scale_lam, lam)
            for (size_t_for row = 0; row < (size_t)m; row++)
            {
                double bmean = 0;
                double wsum = DBL_EPSILON;
                for (size_t col = 0; col < (size_t)n; col++)
                {
                    bmean += (isnan(Xfull[col + row*n]))?
                                (0) : ( ( (Xfull[col + row*n] - bmean)
                                           * weight[col + row*n])
                                                /
                                        (double)(wsum += weight[col + row*n]) );
                }
                
                if (m > cnt_NA[row]) {
                    bmean *= wsum
                                /
                             (wsum
                                +
                              (lam * ((wsumA != NULL)?
                                        (wsumA[row])
                                            :
                                        (scale_lam? wsum : (real_t)1.))));
                }
                biasA[row] = bmean;
            }
        }

        else /* <- in this case, X is transposed */
        {
            set_to_zero(biasA, m);
            int_t tmp = m; m = n; n = tmp;
            real_t *restrict biasB = biasA;
            real_t *restrict wsum = (real_t*)calloc(n, sizeof(real_t));
            if (wsum == NULL) return 1;
            for (size_t row = 0; row < (size_t)m; row++)
            {
                for (size_t col = 0; col < (size_t)n; col++)
                {
                    biasB[col] += (isnan(Xfull[col + row*n]))?
                                    0 : Xfull[col + row*n];
                    wsum[col]  += (isnan(Xfull[col + row*n]))?
                                    0 : weight[col + row*n];
                }
            }
            for (int_t col = 0; col < n; col++)
                wsum[col] = (cnt_NA[col] < n)? wsum[col] : 1;
            for (int_t col = 0; col < n; col++)
                biasB[col] /= wsum[col]
                                +
                              (lam * ((wsumA != NULL)?
                                  ((double)wsumA[col])
                                      :
                                  (scale_lam? wsum[col] : (real_t)1.)));
            free(wsum);
        }
    }

    else if (weightR == NULL && !NA_as_zero)
    {
        #pragma omp parallel for schedule(dynamic) \
                num_threads(cap_to_4(nthreads)) \
                shared(Xcsr_p, Xcsr, m, biasA, wsumA, scale_lam, lam)
        for (size_t_for row = 0; row < (size_t)m; row++)
        {
            double bmean = 0;
            for (size_t ix = Xcsr_p[row]; ix < Xcsr_p[row+1]; ix++)
                bmean += (Xcsr[ix] - bmean) / (double)(ix - Xcsr_p[row] + 1);
            bmean *= (double)(Xcsr_p[row+1] - Xcsr_p[row])
                        /
                     ((double)(Xcsr_p[row+1] - Xcsr_p[row])
                        + lam * ((wsumA != NULL)?
                                    (double)wsumA[row]
                                        :
                                    (scale_lam?
                                        (double)max2(
                                                    Xcsr_p[row+1] - Xcsr_p[row],
                                                    1)
                                            :
                                        1.)));
            biasA[row] = bmean;
        }
    }

    else if (!NA_as_zero) /* <- has weights */
    {
        #pragma omp parallel for schedule(dynamic) \
                num_threads(cap_to_4(nthreads)) \
                shared(Xcsr_p, Xcsr, weightR, m, biasA, wsumA, scale_lam, lam)
        for (size_t_for row = 0; row < (size_t)m; row++)
        {
            double bmean = 0;
            double wsum = DBL_EPSILON;
            for (size_t ix = Xcsr_p[row]; ix < Xcsr_p[row+1]; ix++)
                bmean += ((Xcsr[ix] - bmean) * weightR[ix])
                            /
                         (wsum += weightR[ix]);
            wsum = (Xcsr_p[row+1] > Xcsr_p[row])? wsum : 0;
            bmean *= wsum
                        /
                     (wsum
                        + lam * ((wsumA != NULL)?
                                    (double)wsumA[row]
                                        :
                                    (scale_lam? max2(wsum, DBL_EPSILON) : 1.)));
            biasA[row] = bmean;
        }
    }

    else if (weightR == NULL) /* <- has NA_as_zero */
    {
        #pragma omp parallel for schedule(dynamic) \
                num_threads(cap_to_4(nthreads)) \
                shared(Xcsr_p, Xcsr, m, biasA, wsumA, n, \
                       scale_lam, lam, glob_mean)
        for (size_t_for row = 0; row < (size_t)m; row++)
        {
            double bmean = 0;
            for (size_t ix = Xcsr_p[row]; ix < Xcsr_p[row+1]; ix++)
                bmean += (Xcsr[ix] - bmean) / (double)(ix - Xcsr_p[row] + 1);
            bmean -= glob_mean
                        / 
                     ((double)(Xcsr_p[row+1] - Xcsr_p[row]) / (double)n);
            bmean *= (double)(Xcsr_p[row+1] - Xcsr_p[row])
                        /
                     ((double)n
                        + lam * ((wsumA != NULL)?
                                    (double)wsumA[row]
                                        :
                                    (scale_lam? (double)n : 1.)));
            biasA[row] = bmean;
        }
    }

    else  /* <- has NA_as_zero and weights */
    {
        #pragma omp parallel for schedule(dynamic) \
                num_threads(cap_to_4(nthreads)) \
                shared(Xcsr_p, Xcsr, m, biasA, wsumA, n, scale_lam, \
                       weightR, glob_mean)
        for (size_t_for row = 0; row < (size_t)m; row++)
        {
            double bmean = 0;
            double wsum = DBL_EPSILON;
            for (size_t ix = Xcsr_p[row]; ix < Xcsr_p[row+1]; ix++)
                bmean += ((Xcsr[ix] - bmean) * weightR[ix])
                            /
                         (wsum += weightR[ix]);
            wsum = (Xcsr_p[row+1] > Xcsr_p[row])? wsum : 0;
            bmean -= glob_mean
                        /
                     (wsum / ((double)(n
                                       - (int_t)(Xcsr_p[row+1] - Xcsr_p[row]))
                                       + wsum));
            bmean *= wsum
                        /
                     (((double)(n - (Xcsr_p[row+1] - Xcsr_p[row])) + wsum)
                        + lam * ((wsumA != NULL)?
                                    (double)wsumA[row]
                                        :
                                    (scale_lam?
                                        ((double)(n
                                                  -(int_t)(Xcsr_p[row+1]
                                                           -Xcsr_p[row]))
                                          + wsum)
                                            :
                                        1.)));
            biasA[row] = bmean;
        }
    }

    if (nonneg)
    {
        for (int_t row = 0; row < m; row++)
            biasA[row] = (biasA[row] >= 0)? biasA[row] : 0;
    }

    return 0;
}

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
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long row, col;
    #endif

    if (fabs_t(lam_user) < EPSILON_T)
        lam_user = EPSILON_T;
    if (fabs_t(lam_item) < EPSILON_T)
        lam_item = EPSILON_T;

    int_t retval = 0;
    double *restrict meanA = NULL;
    double *restrict meanB = NULL;
    double *restrict adjA = NULL;
    double *restrict adjB = NULL;
    real_t *restrict wsum_B = NULL;

    int niter = nonneg? 15 : 5;

    if (NA_as_zero)
    {
        meanA = (double*)malloc((size_t)m*sizeof(double));
        meanB = (double*)malloc((size_t)n*sizeof(double));
        if (meanA == NULL || meanB == NULL) goto throw_oom;

        if (weight != NULL)
        {
            adjA = (double*)malloc((size_t)m*sizeof(double));
            adjB = (double*)malloc((size_t)n*sizeof(double));
            if (adjA == NULL || adjB == NULL) goto throw_oom;
        }

        if (weight == NULL)
        {
            for (int_t row = 0; row < m; row++) {
                double xmean = 0;
                for (size_t ix = Xcsr_p[row]; ix < Xcsr_p[row+1]; ix++)
                    xmean += (Xcsr[ix] - xmean) / (double)(ix - Xcsr_p[row] +1);
                xmean *= (double)(Xcsr_p[row+1] - Xcsr_p[row]) / (double)n;
                meanA[row] = xmean;
            }

            for (int_t col = 0; col < n; col++) {
                double xmean = 0;
                for (size_t ix = Xcsc_p[col]; ix < Xcsc_p[col+1]; ix++)
                    xmean += (Xcsc[ix] - xmean) / (double)(ix - Xcsc_p[col] +1);
                xmean *= (double)(Xcsc_p[col+1] - Xcsc_p[col]) / (double)m;
                meanB[col] = xmean;
            }
        }

        else
        {
            #pragma omp parallel for schedule(dynamic) \
                    num_threads(cap_to_4(nthreads)) \
                    shared(Xcsr_p, Xcsr, weightR, m, n, meanA, \
                           adjA, scale_lam, wsumA, lam_user)
            for (size_t_for row = 0; row < (size_t)m; row++)
            {
                double xmean = 0;
                double wsum = DBL_EPSILON;
                for (size_t ix = Xcsr_p[row]; ix < Xcsr_p[row+1]; ix++)
                    xmean += (weightR[ix] * (Xcsr[ix] - xmean))
                              / (wsum += weightR[ix]);
                wsum = (Xcsr_p[row+1] > Xcsr_p[row])? wsum : 0;
                xmean *= wsum
                         / (wsum + (double)(n - (Xcsr_p[row+1] - Xcsr_p[row])));
                wsum += (double)(n - (Xcsr_p[row+1] - Xcsr_p[row]));
                wsum /= (wsum + lam_user * ((wsumA != NULL)?
                                                (wsumA[row])
                                                    :
                                                (scale_lam? wsum : 1.)));
                meanA[row] = xmean;
                adjA[row] = wsum;
            }

            #pragma omp parallel for schedule(dynamic) \
                    num_threads(cap_to_4(nthreads)) \
                    shared(Xcsc_p, Xcsc, weightC, m, n, meanB, \
                           adjB, scale_lam, wsumB, lam_item)
            for (size_t_for col = 0; col < (size_t)n; col++)
            {
                double xmean = 0;
                double wsum = DBL_EPSILON;
                for (size_t ix = Xcsc_p[col]; ix < Xcsc_p[col+1]; ix++)
                    xmean += (weightC[ix] * (Xcsc[ix] - xmean))
                              / (wsum += weightC[ix]);
                wsum = (Xcsc_p[col+1] > Xcsc_p[col])? wsum : 0;
                xmean *= wsum
                         / (wsum + (double)(m - (Xcsc_p[col+1] - Xcsc_p[col])));
                wsum += (double)(m - (Xcsc_p[col+1] - Xcsc_p[col]));
                wsum /= (wsum + lam_item * ((wsumB != NULL)?
                                                (wsumB[col])
                                                    :
                                                (scale_lam? wsum : 1.)));
                meanB[col] = xmean;
                adjB[col] = wsum;
            }
        }
    }

    if (Xtrans == NULL && Xfull != NULL && weight != NULL)
    {
        wsum_B = (real_t*)malloc((size_t)n*sizeof(int_t));
        if (wsum_B == NULL) goto throw_oom;
    }

    set_to_zero(biasA, m);
    set_to_zero(biasB, n);

    for (int iter = 0; iter < niter; iter++)
    {
        /* First by items */

        if (Xtrans != NULL && weight == NULL)
        {
            #pragma omp parallel for schedule(dynamic) \
                    num_threads(cap_to_4(nthreads)) \
                    shared(Xtrans, m, n, biasA, biasB, lam_item)
            for (size_t_for col = 0; col < (size_t)n; col++)
            {
                double xmean = 0;
                int_t cnt = 0;
                for (size_t row = 0; row < (size_t)m; row++)
                    xmean += (isnan(Xtrans[row + col*(size_t)m]))?
                                0 : ((Xtrans[row + col*(size_t)m]
                                      - biasA[row]
                                      - xmean)
                                      / (double)(++cnt));
                xmean *= (double)cnt
                         / ((double)cnt
                                + lam_item * ((wsumB != NULL)?
                                                wsumB[col]
                                                    :
                                                (scale_lam?
                                                    (double)max2(cnt,1) : 1.)));
                biasB[col] = xmean;
                             
            }
        }

        else if (Xtrans != NULL) /* <- has weights */
        {
            #pragma omp parallel for schedule(dynamic) \
                    num_threads(cap_to_4(nthreads)) \
                    shared(Xtrans, m, n, biasA, biasB, Wtrans, lam_item)
            for (size_t_for col = 0; col < (size_t)n; col++)
            {
                double xmean = 0;
                double wsum = DBL_EPSILON;
                for (size_t row = 0; row < (size_t)m; row++)
                    xmean += (isnan(Xtrans[row + col*(size_t)m]))?
                                0 : ((Wtrans[row + col*(size_t)m]
                                      * (Xtrans[row + col*(size_t)m]
                                         - biasA[row]
                                         - xmean))
                                      / (wsum += Wtrans[row + col*(size_t)m]));
                if (cnt_NA_bycol[col] < m)
                    xmean *= wsum
                             / (wsum + lam_item * ((wsumB != NULL)?
                                                        wsumB[col]
                                                            :
                                                        (scale_lam? wsum :1.)));
                biasB[col] = xmean;
            }
        }

        else if (Xfull != NULL && weight == NULL)
        {
            set_to_zero(biasB, n);
            for (size_t row = 0; row < (size_t)m; row++)
            {
                for (size_t col = 0; col < (size_t)n; col++)
                {
                    biasB[col] += (isnan(Xfull[col + row*(size_t)n]))?
                                     0 : Xfull[col + row*(size_t)n];
                }
            }
            for (int_t col = 0; col < n; col++)
                biasB[col] /= (real_t)(m - cnt_NA_bycol[col])
                              + lam_item * ((wsumB != NULL)?
                                                wsumB[col]
                                                    :
                                                (scale_lam?
                                                   (real_t)max2(
                                                            m-cnt_NA_bycol[col],
                                                            1)
                                                    :
                                                    1.));
        }

        else if (Xfull != NULL) /* <- has weights */
        {
            set_to_zero(biasB, n);
            set_to_zero(wsum_B, n);
            for (size_t row = 0; row < (size_t)m; row++)
            {
                for (size_t col = 0; col < (size_t)n; col++)
                {
                    biasB[col] += (isnan(Xfull[col + row*(size_t)n]))?
                                     0 : Xfull[col + row*(size_t)n];
                    wsum_B[col] += (isnan(Xfull[col + row*(size_t)n]))?
                                      0 : weight[col + row*(size_t)n];
                }
            }
            for (int_t col = 0; col < n; col++)
                wsum_B[col] = (cnt_NA_bycol[col] == m)? wsum_B[col] : 1;
            for (int_t col = 0; col < n; col++)
                biasB[col] /= wsum_B[col]
                              + lam_item * ((wsumB != NULL)?
                                                wsumB[col]
                                                    :
                                                (scale_lam? wsum_B[col] : 1.));
        }

        else if (!NA_as_zero && weight == NULL)
        {
            #pragma omp parallel for schedule(dynamic) \
                    num_threads(cap_to_4(nthreads)) \
                    shared(Xcsc_p, Xcsc_i, Xcsc, biasA, biasB, \
                           n, wsumB, scale_lam, lam_item)
            for (size_t_for col = 0; col < (size_t)n; col++)
            {
                double bmean = 0;
                for (size_t ix = Xcsc_p[col]; ix < Xcsc_p[col+1]; ix++)
                    bmean += (Xcsc[ix] - biasA[Xcsc_i[ix]] - bmean)
                             / (double)(ix - Xcsc_p[col] + 1);
                bmean *= (double)(Xcsc_p[col+1] - Xcsc_p[col])
                            /
                         ((double)(Xcsc_p[col+1] - Xcsc_p[col])
                            + lam_item * ((wsumB != NULL)?
                                            ((double)wsumB[col])
                                              :
                                            (scale_lam?
                                             (double)max2(
                                                      Xcsc_p[col+1]-Xcsc_p[col],
                                                      1)
                                               :
                                             1.)));
                biasB[col] = bmean;
            }
        }

        else if (!NA_as_zero) /* <- has weights */
        {
            #pragma omp parallel for schedule(dynamic) \
                    num_threads(cap_to_4(nthreads)) \
                    shared(Xcsc_p, Xcsc_i, Xcsc, weightC, biasA, biasB, \
                           m, n, wsumB, scale_lam, lam_item)
            for (size_t_for col = 0; col < (size_t)n; col++)
            {
                double bmean = 0;
                double wsum = DBL_EPSILON;
                for (size_t ix = Xcsc_p[col]; ix < Xcsc_p[col+1]; ix++)
                    bmean += (weightC[ix] * (Xcsc[ix]-biasA[Xcsc_i[ix]]-bmean))
                              / (wsum += weightC[ix]);
               if (Xcsc_p[col+1] > Xcsc_p[col])
                    bmean *= wsum / (wsum + lam_item * ((wsumB != NULL)?
                                                            wsumB[col]
                                                              :
                                                            (scale_lam?
                                                                wsum : 1.)));
                biasB[col] = bmean;
            }
        }

        else if (weight == NULL) /* <- has 'NA_as_zero' */
        {
            double bmean = 0;
            if (iter > 0) {
                for (int_t row = 0; row < n; row++)
                    bmean += (biasA[row] - bmean) / (double)(row+1);
            }

            for (int_t col = 0; col < n; col++)
                biasB[col] = (meanB[col] - bmean - glob_mean)
                             * ( (double)m
                                    /
                                ((double)m + lam_item * ((wsumB != NULL)?
                                                            wsumB[col]
                                                              :
                                                            (scale_lam?
                                                                (double)m
                                                                    :
                                                                1.))) );
        }

        else /* <- has 'NA_as_zero' and weights */
        {
            double bmean = 0;
            if (iter > 0) {
                for (int_t row = 0; row < n; row++)
                    bmean += (biasA[row] - bmean) / (double)(row+1);
            }

            #pragma omp parallel for schedule(dynamic) \
                    num_threads(cap_to_4(nthreads)) \
                    shared(Xcsc_p, Xcsc_i, m, n, weightC, bmean, \
                           biasB, meanB, adjB, glob_mean)
            for (size_t_for col = 0; col < (size_t)n; col++)
            {
                double wsum = m;
                double bmean_this = bmean;
                for (size_t ix = Xcsc_p[col]; ix < Xcsc_p[col+1]; ix++)
                    bmean_this += (  (weightC[ix] - 1)
                                     * (biasB[Xcsc_i[ix]] - bmean_this)  )
                                  / (wsum += (weightC[ix] - 1));
                biasB[col] = (meanB[col] - bmean_this - glob_mean) * adjB[col];
            }
        }

        if (nonneg)
            for (int_t col = 0; col < n; col++)
                biasB[col] = (biasB[col] >= (real_t)0)? biasB[col] : (real_t)0;
        

        /* Then by users */

        if (Xfull != NULL && weight == NULL)
        {
            #pragma omp parallel for schedule(dynamic) \
                    num_threads(cap_to_4(nthreads)) \
                    shared(Xfull, m, n, biasA, biasB, lam_user)
            for (size_t_for row = 0; row < (size_t)m; row++)
            {
                double xmean = 0;
                int_t cnt = 0;
                for (size_t col = 0; col < (size_t)n; col++)
                    xmean += (isnan(Xfull[col + row*(size_t)n]))?
                                0 : ((Xfull[col + row*(size_t)n]
                                      - biasB[col]
                                      - xmean)
                                      / (double)(++cnt));
                xmean *= (double)cnt
                         / ((double)cnt + lam_user * ((wsumA != NULL)?
                                                        wsumA[row]
                                                            :
                                                        (scale_lam?
                                                            (double)max2(cnt,1)
                                                            :
                                                            1.)));
                biasA[row] = xmean;
            }
        }

        else if (Xfull != NULL) /* <- has weights */
        {
            #pragma omp parallel for schedule(dynamic) \
                    num_threads(cap_to_4(nthreads)) \
                    shared(Xfull, m, n, biasA, biasB, weight, lam_user)
            for (size_t_for row = 0; row < (size_t)m; row++)
            {
                double xmean = 0;
                double wsum = DBL_EPSILON;
                for (size_t col = 0; col < (size_t)n; col++)
                    xmean += (isnan(Xfull[col + row*(size_t)n]))?
                                0 : ((weight[col + row*(size_t)n]
                                      * (Xfull[col + row*(size_t)n]
                                         - biasB[col]
                                         - xmean))
                                      / (wsum += weight[col + row*(size_t)n]));
                if (cnt_NA_byrow[row] < n)
                    xmean *= wsum
                             / (wsum + lam_user * ((wsumA != NULL)?
                                                        wsumA[row]
                                                            :
                                                        (scale_lam? wsum :1.)));
                biasA[row] = xmean;
            }
        }

        else if (!NA_as_zero && weight == NULL)
        {
            #pragma omp parallel for schedule(dynamic) \
                    num_threads(cap_to_4(nthreads)) \
                    shared(Xcsr_p, Xcsr_i, Xcsr, biasA, biasB, \
                           m, wsumA, scale_lam, lam_user)
            for (size_t_for row = 0; row < (size_t)m; row++)
            {
                double bmean = 0;
                for (size_t ix = Xcsr_p[row]; ix < Xcsr_p[row+1]; ix++)
                    bmean += (Xcsr[ix] - biasB[Xcsr_i[ix]] - bmean)
                             / (double)(ix - Xcsr_p[row] + 1);
                if (Xcsr_p[row+1] > Xcsr_p[row])
                    bmean *= (double)(Xcsr_p[row+1] - Xcsr_p[row])
                                /
                             ((double)(Xcsr_p[row+1] - Xcsr_p[row])
                                + lam_user * ((wsumA != NULL)?
                                                ((double)wsumA[row])
                                                    :
                                                (scale_lam?
                                                    (double)(Xcsr_p[row+1]
                                                             -Xcsr_p[row])
                                                    :
                                                    1.)));
                biasA[row] = bmean;
            }
        }

        else if (!NA_as_zero) /* <- has weights */
        {
            #pragma omp parallel for schedule(dynamic) \
                    num_threads(cap_to_4(nthreads)) \
                    shared(Xcsr_p, Xcsr_i, Xcsr, weightR, biasA, biasB, \
                           m, n, wsumA, scale_lam, lam_user)
            for (size_t_for row = 0; row < (size_t)m; row++)
            {
                double bmean = 0;
                double wsum = DBL_EPSILON;
                for (size_t ix = Xcsr_p[row]; ix < Xcsr_p[row+1]; ix++)
                    bmean += (weightR[ix] * (Xcsr[ix]-biasB[Xcsr_i[ix]]-bmean))
                              / (wsum += weightR[ix]);
                if (Xcsr_p[row+1] > Xcsr_p[row])
                    bmean *= wsum
                             / (wsum + lam_user * ((wsumA != NULL)?
                                                        wsumA[row]
                                                            :
                                                        (scale_lam? wsum :1.)));
                biasA[row] = bmean;
            }
        }

        else if (weight == NULL) /* <- has 'NA_as_zero' */
        {
            double bmean = 0;
            if (iter > 0) {
                for (int_t col = 0; col < n; col++)
                    bmean += (biasB[col] - bmean) / (double)(col+1);
            }

            for (int_t row = 0; row < m; row++)
                biasA[row] = (meanA[row] - bmean - glob_mean)
                             * ( (double)n
                                    /
                                ((double)n + lam_user * ((wsumA != NULL)?
                                                            wsumA[row]
                                                                :
                                                            (scale_lam?
                                                                (double)n
                                                                    :
                                                                1.))) );
        }

        else /* <- has weights and 'NA_as_zero' */
        {
            double bmean = 0;
            for (int_t col = 0; col < n; col++)
                bmean += (biasB[col] - bmean) / (double)(col+1);

            #pragma omp parallel for schedule(dynamic) \
                    num_threads(cap_to_4(nthreads)) \
                    shared(Xcsr_p, Xcsr_i, m, n, weightR, bmean, \
                           biasA, meanA, adjA, glob_mean)
            for (size_t_for row = 0; row < (size_t)m; row++)
            {
                double wsum = n;
                double bmean_this = bmean;
                for (size_t ix = Xcsr_p[row]; ix < Xcsr_p[row+1]; ix++)
                    bmean_this += (  (weightR[ix] - 1)
                                     * (biasB[Xcsr_i[ix]] - bmean_this)  )
                                  / (wsum += (weightR[ix] - 1));
                biasA[row] = (meanA[row] - bmean_this - glob_mean) * adjA[row];
            }
        }

        if (nonneg)
            for (int_t row = 0; row < m; row++)
                biasA[row] = (biasA[row] >= (real_t)0)? biasA[row] : (real_t)0;

    }

    cleanup:
        free(meanA);
        free(meanB);
        free(adjA);
        free(adjB);
        free(wsum_B);
    return retval;
    throw_oom:
        retval = 1;
        goto cleanup;
}

int_t center_by_cols
(
    real_t *restrict col_means,
    real_t *restrict Xfull, int_t m, int_t n,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    size_t Xcsr_p[], int_t Xcsr_i[], real_t *restrict Xcsr,
    size_t Xcsc_p[], int_t Xcsc_i[], real_t *restrict Xcsc,
    int nthreads
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix, ib;
    #endif
    int_t *restrict cnt_by_col = NULL;
    if (Xfull != NULL || Xcsc == NULL) {
        cnt_by_col = (int_t*)calloc(n, sizeof(int_t));
        if (cnt_by_col == NULL) return 1;
    }
    set_to_zero(col_means, n);

    if (Xfull != NULL)
    {
        for (size_t row = 0; row < (size_t)m; row++)
            for (size_t col = 0; col < (size_t)n; col++) {
                col_means[col] += (!isnan(Xfull[col + row*(size_t)n]))?
                                   (Xfull[col + row*(size_t)n]) : (0.);
                cnt_by_col[col] += !isnan(Xfull[col + row*(size_t)n]);
            }
    }

    else if (Xcsc != NULL)
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                shared(n, Xcsc, Xcsc_p, col_means)
        for (size_t_for ib = 0; ib < (size_t)n; ib++)
        {
            double csum = 0;
            for (size_t ix = Xcsc_p[ib]; ix < Xcsc_p[ib+(size_t)1]; ix++)
                csum += Xcsc[ix];
            col_means[ib] = csum;
        }
    }

    else if (Xcsr != NULL && X == NULL)
    {

        for (size_t ia = 0; ia < (size_t)m; ia++)
            for (size_t ix = Xcsr_p[ia]; ix < Xcsr_p[ia+(size_t)1]; ix++)
                col_means[Xcsr_i[ix]] += Xcsr[ix];
    }

    else
    {
        for (size_t ix = 0; ix < nnz; ix++) {
            col_means[ixB[ix]] += X[ix];
            cnt_by_col[ixB[ix]]++;
        }
    }

    /* -------- */
    if (Xfull != NULL || Xcsc == NULL)
        for (size_t ix = 0; ix < (size_t)n; ix++)
            col_means[ix] /= (double)cnt_by_col[ix];
    else
        for (size_t ix = 0; ix < (size_t)n; ix++)
            col_means[ix] /= (double)(Xcsc_p[ix+(size_t)1] - Xcsc_p[ix]);
    /* -------- */

    if (Xfull != NULL)
    {
        for (size_t row = 0; row < (size_t)m; row++)
            for (size_t col = 0; col < (size_t)n; col++)
                Xfull[col + row*(size_t)n] -= col_means[col];
    }

    else if (Xcsc != NULL || Xcsr != NULL)
    {
        if (Xcsc != NULL)
        {
            #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                    shared(Xcsc, Xcsc_p, n, col_means)
            for (size_t_for ib = 0; ib < (size_t)n; ib++)
                for (size_t ix = Xcsc_p[ib]; ix < Xcsc_p[ib+(size_t)1]; ix++)
                    Xcsc[ix] -= col_means[ib];
        }

        if (Xcsr != NULL)
        {
            if (X != NULL)
                for (size_t ix = 0; ix < nnz; ix++)
                    Xcsr[ix] -= col_means[Xcsr_i[ix]];
            else
                for (size_t ia = 0; ia < (size_t)m; ia++)
                    for (size_t ix = Xcsr_p[ia]; ix < Xcsr_p[ia+(size_t)1];ix++)
                        Xcsr[ix] -= col_means[Xcsr_i[ix]];
        }
    }

    else
    {
        nthreads = cap_to_4(nthreads);
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(nnz, X, col_means, ixB)
        for (size_t_for ix = 0; ix < nnz; ix++)
            X[ix] -= col_means[ixB[ix]];

    }

    free(cnt_by_col);
    return 0;
}

bool check_sparse_indices
(
    int_t n, int_t p,
    real_t *restrict u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
    real_t *restrict Xa, int_t ixB[], size_t nnz
)
{
    if (nnz) {
        for (size_t ix = 0; ix < nnz; ix++) {
            if (ixB[ix] < 0 || ixB[ix] >= ((n > 0)? n : INT_MAX)) {
                return true;
            }
        }
    }
    if (nnz_u_vec) {
        for (size_t ix = 0; ix < nnz_u_vec; ix++) {
            if (u_vec_ixB[ix] < 0 || u_vec_ixB[ix] >= ((p > 0)? p : INT_MAX)) {
                return true;
            }
        }
    }

    return false;
}

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
)
{
    size_t lda = (size_t)k_user + (size_t)k + (size_t)k_main;
    size_t ldb = (size_t)k_item + (size_t)k + (size_t)k_main;
    A += k_user;
    B += k_item;
    if (m == 0) m = INT_MAX;
    if (n == 0) n = INT_MAX;
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif
    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(A, B, outp, nnz, predA, predB, lda, ldb, k)
    for (size_t_for ix = 0; ix < nnz; ix++)
        outp[ix] = (predA[ix] >= m || predA[ix] < 0 ||
                    predB[ix] >= n || predB[ix] < 0)?
                        (NAN_)
                        :
                    (cblas_tdot(k, A + (size_t)predA[ix]*lda, 1,
                                   B + (size_t)predB[ix]*ldb, 1)
                     + ((biasA != NULL)? biasA[predA[ix]] : 0.)
                     + ((biasB != NULL)? biasB[predB[ix]] : 0.)
                     + glob_mean);
}

int_t cmp_int(const void *a, const void *b)
{
    return *((int_t*)a) - *((int_t*)b);
}

real_t *ptr_real_t_glob = NULL;
// #pragma omp threadprivate(ptr_real_t_glob)
// ptr_real_t_glob = NULL;
int_t cmp_argsort(const void *a, const void *b)
{
    real_t v1 = ptr_real_t_glob[*((int_t*)a)];
    real_t v2 = ptr_real_t_glob[*((int_t*)b)];
    return (v1 == v2)? 0 : ((v1 < v2)? 1 : -1);
}

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
)
{
    int_t retval = 0;
    if (include_ix != NULL && exclude_ix != NULL) {
        fprintf(stderr, "Cannot pass both 'include_ix' and 'exclude_ix'.\n");
        retval = 2;
    }
    if (n_top == 0) {
        fprintf(stderr, "'n_top' must be greater than zero.\n");
        retval = 2;
    }
    if (n_exclude > n-n_top) {
        fprintf(stderr, "Number of rankeable entities is less than 'n_top'\n");
        retval = 2;
    }
    if (n_include > n) {
        fprintf(stderr, "Number of entities to include is larger than 'n'.\n");
        retval = 2;
    }

    if (include_ix != NULL)
    {
        for (int_t ix = 0; ix < n_include; ix++)
            if (include_ix[ix] < 0 || include_ix[ix] >= n)
            {
                fprintf(stderr, "'include_ix' contains invalid entries\n");
                retval = 2;
                break;
            }
    }
    if (exclude_ix != NULL)
    {
        for (int_t ix = 0; ix < n_exclude; ix++)
            if (exclude_ix[ix] < 0 || exclude_ix[ix] >= n)
            {
                fprintf(stderr, "'exclude_ix' contains invalid entries\n");
                retval = 2;
                break;
            }
    }
    for (int_t ix = 0; ix < k_user+k+k_main; ix++)
    {
        if (isnan(a_vec[ix])) {
            fprintf(stderr, "The latent factors contain NAN values\n");
            retval = 2;
            break;
        }
    }
    if (isnan(biasA)) {
        fprintf(stderr, "The bias is a NAN value\n");
        retval = 2;
    }

    if (retval == 2)
    {
        #ifndef _FOR_R
        fflush(stderr);
        #endif
        return retval;
    }

    int_t ix = 0;

    int_t k_pred = k + k_main;
    int_t k_totB = k_item + k + k_main;
    size_t n_take = (include_ix != NULL)?
                     (size_t)n_include :
                     ((exclude_ix == NULL)? (size_t)n : (size_t)(n-n_exclude) );

    real_t *restrict buffer_scores = NULL;
    int_t *restrict buffer_ix = NULL;
    int_t *restrict buffer_mask = NULL;
    a_vec += k_user;

    if (include_ix != NULL) {
        buffer_ix = include_ix;
    }

    else {
        buffer_ix = (int_t*)malloc((size_t)n*sizeof(int_t));
        if (buffer_ix == NULL) goto throw_oom;
        for (int_t ix = 0; ix < n; ix++) buffer_ix[ix] = ix;
    }

    if (exclude_ix != NULL)
    {
        int_t move_to = n-1;
        int_t temp;
        if (!check_is_sorted(exclude_ix, n_exclude))
            qsort(exclude_ix, n_exclude, sizeof(int_t), cmp_int);

        for (int_t ix = n_exclude-1; ix >= 0; ix--) {
            temp = buffer_ix[move_to];
            buffer_ix[move_to] = exclude_ix[ix];
            buffer_ix[exclude_ix[ix]] = temp;
            move_to--;
        }
    }

    /* Case 1: there is a potentially small number of items to include.
       Here can produce predictons only for those, then make
       an argsort with doubly-masked indices. */
    if (include_ix != NULL)
    {
        buffer_scores = (real_t*)malloc((size_t)n_include*sizeof(real_t));
        buffer_mask = (int_t*)malloc((size_t)n_include*sizeof(int_t));
        if (buffer_scores == NULL || buffer_mask == NULL) goto throw_oom;
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(a_vec, B, k_pred, k_item, n_include, k_totB, \
                       include_ix, biasB, buffer_scores)
        for (ix = 0; ix < n_include; ix++) {
            buffer_scores[ix] = cblas_tdot(k_pred, a_vec, 1,
                                           B + k_item + (size_t)include_ix[ix]
                                                * (size_t)k_totB, 1)
                                + ((biasB != NULL)? biasB[include_ix[ix]] : 0.);
        }
        for (int_t ix = 0; ix < n_include; ix++)
            buffer_mask[ix] = ix;
    }

    /* Case 2: there is a large number of items to exclude.
       Here can also produce predictions only for the included ones
       and then make a full or partial argsort. */
    else if (exclude_ix != NULL && (double)n_exclude > (double)n/20)
    {
        buffer_scores = (real_t*)malloc(n_take*sizeof(real_t));
        buffer_mask = (int_t*)malloc(n_take*sizeof(int_t));
        if (buffer_scores == NULL || buffer_mask == NULL) goto throw_oom;
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(a_vec, B, k_pred, k_item, n_take, k_totB, \
                       buffer_ix, biasB, buffer_scores)
        for (ix = 0; ix < (int_t)n_take; ix++)
            buffer_scores[ix] = cblas_tdot(k_pred, a_vec, 1,
                                           B + k_item + (size_t)buffer_ix[ix]
                                                * (size_t)k_totB, 1)
                                + ((biasB != NULL)? biasB[buffer_ix[ix]] : 0.);
        for (int_t ix = 0; ix < (int_t)n_take; ix++)
            buffer_mask[ix] = ix;
    }

    /* General case: make predictions for all the entries, then
       a partial argsort (this is faster since it makes use of
       optimized BLAS gemv, but it's not memory-efficient) */
    else
    {
        buffer_scores = (real_t*)malloc((size_t)n*sizeof(real_t));
        if (buffer_scores == NULL) goto throw_oom;
        cblas_tgemv(CblasRowMajor, CblasNoTrans,
                    n, k_pred,
                    1., B + k_item, k_totB,
                    a_vec, 1,
                    0., buffer_scores, 1);
        if (biasB != NULL)
            cblas_taxpy(n, 1., biasB, 1, buffer_scores, 1);
    }

    /* If there is no double-mask for indices, do a partial argsort */
    ptr_real_t_glob = buffer_scores;
    if (buffer_mask == NULL)
    {
        /* If the number of elements is very small, it's faster to
           make a full argsort, taking advantage of qsort's optimizations */
        if (n_take <= 50 || n_take >= (double)n*0.75)
        {
            qsort(buffer_ix, n_take, sizeof(int_t), cmp_argsort);
        }

        /* Otherwise, do a proper partial sort */
        else
        {
            qs_argpartition(buffer_ix, buffer_scores, n_take, n_top);
            qsort(buffer_ix, n_top, sizeof(int_t), cmp_argsort);
        }

        memcpy(outp_ix, buffer_ix, (size_t)n_top*sizeof(int_t));
    }

    /* Otherwise, do a partial argsort with doubly-indexed arrays */
    else
    {
        if (n_take <= 50 || n_take >= (double)n*0.75)
        {
            qsort(buffer_mask, n_take, sizeof(int_t), cmp_argsort);
        }

        else
        {
            qs_argpartition(buffer_mask, buffer_scores, n_take, n_top);
            qsort(buffer_mask, n_top, sizeof(int_t), cmp_argsort);
        }

        for (int_t ix = 0; ix < n_top; ix++)
                outp_ix[ix] = buffer_ix[buffer_mask[ix]];
    }
    ptr_real_t_glob = NULL;

    /* If scores were requested, need to also output those */
    if (outp_score != NULL)
    {
        glob_mean += biasA;
        if (buffer_mask == NULL)
            for (int_t ix = 0; ix < n_top; ix++)
                outp_score[ix] = buffer_scores[outp_ix[ix]] + glob_mean;
        else
            for (int_t ix = 0; ix < n_top; ix++)
                outp_score[ix] = buffer_scores[buffer_mask[ix]] + glob_mean;
    }

    cleanup:
        free(buffer_scores);
        if (include_ix == NULL)
            free(buffer_ix);
        free(buffer_mask);
    return retval;

    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}

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
)
{
    if (implicit)
    {
        if (NA_as_zero) {
            fprintf(stderr,
                    "Warning: 'NA_as_zero' ignored with 'implicit=true'.\n");
            NA_as_zero = false;
        }

        if (scale_lam) {
            fprintf(stderr,
                    "Warning: 'scale_lam' ignored with 'implicit=true'.\n");
            scale_lam = false;
        }

        if (weight != NULL) {
            fprintf(stderr,
                    "Warning: 'weight' ignored with 'implicit=true'.\n");
            weight = NULL;
        }

        if (Xfull != NULL) {
            fprintf(stderr,
                    "Error: cannot pass dense 'X' with 'implicit=true'.\n");
            return 2;
        }
    }

    else
    {
        if (adjust_weight) {
            fprintf(stderr,
                "Warning: 'adjust_weight' ignored with 'implicit=false'.\n");
            adjust_weight = false;
        }

        if (apply_log_transf) {
            fprintf(stderr,
                "Warning: 'apply_log_transf' ignored with 'implicit=false'.\n");
            apply_log_transf = false;
        }
    }

    if (biasB == NULL)
    {
        fprintf(stderr, "Error: must pass 'biasB'.\n");
        return 2;
    }

    if (!scale_lam) scale_bias_const = false;

    int_t retval = 0;
    if (glob_mean != NULL)
        *glob_mean = 0;

    size_t *restrict Xcsr_p = NULL;
    int_t *restrict Xcsr_i = NULL;
    real_t *restrict Xcsr = NULL;
    size_t *restrict Xcsc_p = NULL;
    int_t *restrict Xcsc_i = NULL;
    real_t *restrict Xcsc = NULL;
    real_t *restrict weightR = NULL;
    real_t *restrict weightC = NULL;
    real_t *restrict ones = NULL;
    real_t *restrict wsumA = NULL;
    real_t *restrict wsumB = NULL;

    if (NA_as_zero && Xfull == NULL)
    {
        if (glob_mean != NULL)
            calc_mean_and_center(
                ixA, ixB, X, nnz,
                (real_t*)NULL, (real_t*)NULL,
                m, n,
                (size_t*)NULL, (int_t*)NULL, (real_t*)NULL,
                (size_t*)NULL, (int_t*)NULL, (real_t*)NULL,
                weight,
                NA_as_zero, nonneg, true, nthreads,
                glob_mean
            );

        if (biasA != NULL)
        {
            retval = coo_to_csr_plus_alloc(
                ixA, ixB, X,
                weight,
                m, n, nnz,
                &Xcsr_p, &Xcsr_i, &Xcsr,
                &weightR
            );
            if (retval == 1) goto throw_oom;
            if (retval != 0) goto cleanup;
        }

        if (biasB != NULL)
        {
            retval = coo_to_csr_plus_alloc(
                ixB, ixA, X,
                weight,
                n, m, nnz,
                &Xcsc_p, &Xcsc_i, &Xcsc,
                &weightC
            );
            if (retval == 1) goto throw_oom;
            if (retval != 0) goto cleanup;
        }

        if (scale_lam && weight != NULL)
        {
            if (biasA != NULL)
            {
                wsumA = (real_t*)calloc(m, sizeof(real_t));
                if (wsumA == NULL) goto throw_oom;

                for (int_t row = 0; row < m; row++) {
                    for (size_t ix = Xcsr_p[row]; ix < Xcsr_p[row+1]; ix++)
                        wsumA[row] += weightR[ix];
                    wsumA[row] /= wsumA[row]
                                  + (real_t)(n - (Xcsr_p[row+1] - Xcsr_p[row]));
                }
            }

            if (biasB != NULL)
            {
                wsumB = (real_t*)calloc(n, sizeof(real_t));
                if (wsumB == NULL) goto throw_oom;

                for (int_t col = 0; col < n; col++) {
                    for (size_t ix = Xcsc_p[col]; ix < Xcsc_p[col+1]; ix++)
                        wsumB[col] += weightC[ix];
                    wsumB[col] /= wsumB[col]
                                  + (real_t)(m - (Xcsc_p[col+1] - Xcsc_p[col]));
                }
            }
        }

        if (scale_bias_const)
        {
            scale_bias_const = false;
            scale_lam = false;
            if (weight == NULL)
            {
                lam_user *= n;
                lam_item *= m;
            }

            else
            {
                double wmean = 0;
                if (biasA != NULL)
                {
                    for (int_t row = 0; row < m; row++)
                        wmean += (wsumA[row] - wmean) / (double)(row+1);
                    lam_user *= wmean;
                }

                wmean = 0;
                if (biasB != NULL)
                {
                    for (int_t col = 0; col < n; col++)
                        wmean += (wsumB[col] - wmean) / (double)(col+1);
                    lam_item *= wmean;
                }

                free(wsumA); wsumA = NULL;
                free(wsumB); wsumB = NULL;
            }
        }

        if (biasA != NULL && biasB != NULL)
            retval = initialize_biases_twosided(
                (real_t*)NULL, (real_t*)NULL,
                (int_t*)NULL, (int_t*)NULL,
                m, n,
                NA_as_zero, nonneg, (glob_mean == NULL)? 0. : (*glob_mean),
                Xcsr_p, Xcsr_i, Xcsr,
                Xcsc_p, Xcsc_i, Xcsc,
                weight, (real_t*)NULL,
                weightR, weightC,
                lam_user, lam_item, scale_lam,
                wsumA, wsumB,
                biasA, biasB,
                nthreads
            );
        else
            initialize_biases_onesided(
                (real_t*)NULL, n, m, false, (int_t*)NULL,
                Xcsc_p, Xcsc_i, Xcsc,
                weight, weightC,
                (glob_mean == NULL)? 0. : (*glob_mean), NA_as_zero, nonneg,
                lam_item, scale_lam,
                wsumB,
                biasB,
                nthreads
            );

        if (retval == 1) goto throw_oom;
        goto cleanup;
    }

    if (implicit && biasA != NULL)
    {
        for (size_t ix = 0; ix < nnz; ix++)
            X[ix] += 1;
        if (apply_log_transf)
            for (size_t ix = 0; ix < nnz; ix++)
                X[ix] = log_t(X[ix]);

        retval = coo_to_csr_plus_alloc(
            ixA, ixB, X,
            (real_t*)NULL,
            m, n, nnz,
            &Xcsr_p, &Xcsr_i, &Xcsr,
            (real_t**)NULL
        );
        if (retval == 1) goto throw_oom;
        if (retval != 0) goto cleanup;

        retval = coo_to_csr_plus_alloc(
            ixB, ixA, X,
            (real_t*)NULL,
            n, m, nnz,
            &Xcsc_p, &Xcsc_i, &Xcsc,
            (real_t**)NULL
        );
        if (retval == 1) goto throw_oom;
        if (retval != 0) goto cleanup;

        ones = (real_t*)malloc(nnz*sizeof(real_t));
        if (ones == NULL) goto throw_oom;
        for (size_t ix = 0; ix < nnz; ix++) ones[ix] = 1;

        if (adjust_weight)
        {
            *w_main_multiplier
                =
            (long double)nnz / ((long double)m * (long double)n);
            lam_item /= *w_main_multiplier;
            lam_user /= *w_main_multiplier;
        }

        retval = initialize_biases_twosided(
            (real_t*)NULL, (real_t*)NULL,
            (int_t*)NULL, (int_t*)NULL,
            m, n,
            true, nonneg, 0.,
            Xcsr_p, Xcsr_i, ones,
            Xcsc_p, Xcsc_i, ones,
            X, (real_t*)NULL,
            Xcsr, Xcsc,
            lam_user, lam_item, false,
            (real_t*)NULL, (real_t*)NULL,
            biasA, biasB,
            nthreads
        );
        if (retval == 1) goto throw_oom;
        goto cleanup;
    }

    return fit_most_popular_internal(
        biasA, biasB,
        glob_mean, glob_mean != NULL,
        lam_user, lam_item,
        scale_lam, scale_bias_const,
        alpha,
        m, n,
        ixA, ixB, X, nnz,
        Xfull,
        weight,
        implicit, adjust_weight, apply_log_transf,
        nonneg,
        w_main_multiplier,
        nthreads
    );

    cleanup:
        free(Xcsr_p);
        free(Xcsr_i);
        free(Xcsr);
        free(Xcsc_p);
        free(Xcsc_i);
        free(Xcsc);
        free(weightR);
        free(weightC);
        free(ones);
        free(wsumA);
        free(wsumB);
    return retval;
    throw_oom:
        retval = 1;
        print_oom_message();
        goto cleanup;
}


/* TODO: factor out this function */
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
)
{
    int_t retval = 0;
    int_t *restrict cnt_by_col = NULL;
    int_t *restrict cnt_by_row = NULL;
    real_t *restrict sum_by_col = NULL;
    real_t *restrict sum_by_row = NULL;
    int_t maxiter = 5;


    if (implicit)
    {
        cnt_by_col = (int_t*)calloc((size_t)n, sizeof(int_t));
        sum_by_col = (real_t*)calloc((size_t)n, sizeof(real_t));
        if (cnt_by_col == NULL || sum_by_col == NULL) goto throw_oom;

        if (apply_log_transf)
        {
            if (Xfull != NULL) {
                for (size_t row = 0; row < (size_t)m; row++)
                    for (size_t col = 0; col < (size_t)n; col++)
                        Xfull[col + row*(size_t)n]
                            =
                        log_t(Xfull[col + row*(size_t)n]);
            }

            else {
                for (size_t ix = 0; ix < nnz; ix++)
                    X[ix] = log_t(X[ix]);
            }
        }

        if (Xfull != NULL)
        {
            for (size_t row = 0; row < (size_t)m; row++) {
                for (size_t col = 0; col < (size_t)n; col++) {
                    cnt_by_col[col] += !isnan(Xfull[col + row*(size_t)n]);
                    sum_by_col[col] += (!isnan(Xfull[col + row*(size_t)n]))?
                                        (Xfull[col + row*(size_t)n] +1.) : (0.);
                }
            }
        }

        else
        {
            for (size_t ix = 0; ix < nnz; ix++) {
                cnt_by_col[ixB[ix]]++;
                sum_by_col[ixB[ix]] += X[ix] + 1.;
            }
        }

        if (adjust_weight) {
            nnz = 0;
            for (int_t ix = 0; ix < n; ix++)
                nnz += (size_t)cnt_by_col[ix];
            *w_main_multiplier
                =
            (long double)nnz / ((long double)m * (long double)n);
            lam_item /= *w_main_multiplier;
        }

        for (int_t ix = 0; ix < n; ix++)
            biasB[ix] = alpha * sum_by_col[ix]
                                / (alpha * sum_by_col[ix]
                                    + (double)(m - cnt_by_col[ix])
                                    + lam_item);

        goto cleanup;
    }


    if (biasA != NULL) {

        if (weight == NULL || scale_lam) {
            cnt_by_col = (int_t*)calloc((size_t)n, sizeof(int_t));
            cnt_by_row = (int_t*)calloc((size_t)m, sizeof(int_t));
            if (cnt_by_col == NULL || cnt_by_row == NULL)
                goto throw_oom;
        }
        if (weight != NULL) {
            sum_by_col = (real_t*)calloc((size_t)n, sizeof(real_t));
            sum_by_row = (real_t*)calloc((size_t)m, sizeof(real_t));
            if (sum_by_col == NULL || sum_by_row == NULL)
                goto throw_oom;
        }
    }

    retval = initialize_biases(
        glob_mean, biasA, biasB,
        false, biasA == NULL, center,
        lam_user, lam_item,
        scale_lam, scale_bias_const,
        biasA != NULL, biasB != NULL,
        (real_t*)NULL, (real_t*)NULL,
        m, n,
        m, n,
        ixA, ixB, X, nnz,
        Xfull, (real_t*)NULL,
        (size_t*)NULL, (int_t*)NULL, (real_t*)NULL,
        (size_t*)NULL, (int_t*)NULL, (real_t*)NULL,
        weight, (real_t*)NULL,
        (real_t*)NULL, (real_t*)NULL,
        nonneg,
        nthreads
    );
    if (retval == 1) goto throw_oom;


    if (biasA == NULL && !implicit)
    {
        goto cleanup;
    }

    if (Xfull != NULL)
    {
        if (weight == NULL || scale_lam)
            for (size_t row = 0; row < (size_t)m; row++) {
                for (size_t col = 0; col < (size_t)n; col++) {
                    cnt_by_row[row] += !isnan(Xfull[col + row*(size_t)n]);
                    cnt_by_col[col] += !isnan(Xfull[col + row*(size_t)n]);
                }
            }
        if (weight != NULL)
            for (size_t row = 0; row < (size_t)m; row++) {
                for (size_t col = 0; col < (size_t)n; col++) {
                    sum_by_row[row] += (!isnan(Xfull[col + row*(size_t)n]))?
                                        (weight[col + row*(size_t)n]) : (0.);
                    sum_by_col[col] += (!isnan(Xfull[col + row*(size_t)n]))?
                                        (weight[col + row*(size_t)n]) : (0.);
                }
            }
    }

    else
    {
        if (weight == NULL || scale_lam)
            for (size_t ix = 0; ix < nnz; ix++) {
                cnt_by_row[ixA[ix]]++;
                cnt_by_col[ixB[ix]]++;
            }
        if (weight != NULL)
            for (size_t ix = 0; ix < nnz; ix++) {
                sum_by_row[ixA[ix]] += weight[ix];
                sum_by_col[ixB[ix]] += weight[ix];
            }
    }

    if (scale_lam && scale_bias_const)
    {
        if (weight != NULL)
        {
            double wmean = 0;
            for (int_t row = 0; row < m; row++)
                wmean += (sum_by_row[row] - wmean) / (double)(row+1);
            lam_user *= wmean;

            wmean = 0;
            for (int_t col = 0; col < n; col++)
                wmean += (sum_by_col[col] - wmean) / (double)(col+1);
            lam_item *= wmean;
        }
        
        else
        {
            double cmean = 0;
            for (int_t row = 0; row < m; row++)
                cmean += ((double)cnt_by_row[row] - cmean) / (double)(row+1);
            lam_user *= cmean;

            cmean = 0;
            for (int_t col = 0; col < n; col++)
                cmean += ((double)cnt_by_col[col] - cmean) / (double)(col+1);
            lam_item *= cmean;
        }
        
        scale_bias_const = false;
        scale_lam = false;
    }

    set_to_zero(biasA, m);
    set_to_zero(biasB, n);

    maxiter = nonneg? 15 : 5;
    for (int_t iter = 0; iter <= maxiter; iter++)
    {
        if (Xfull != NULL)
        {
            if (iter > 0)
                set_to_zero(biasB, n);

            if (weight == NULL)
            {
                for (size_t row = 0; row < (size_t)m; row++)
                    for (size_t col = 0; col < (size_t)n; col++)
                        biasB[col] += (!isnan(Xfull[col + row*(size_t)n]))?
                                       (Xfull[col + row*(size_t)n] - biasA[row])
                                       : (0.);
                for (int_t ix = 0; ix < n; ix++)
                    biasB[ix] /= ((double)cnt_by_col[ix]
                                    + (lam_item
                                        *
                                       (scale_lam?
                                            (double)cnt_by_col[ix] : 1.)));
            }

            else
            {
                for (size_t row = 0; row < (size_t)m; row++)
                    for (size_t col = 0; col < (size_t)n; col++)
                        biasB[col] += (!isnan(Xfull[col + row*(size_t)n]))?
         weight[col + row*(size_t)n] * (Xfull[col + row*(size_t)n] - biasA[row])
                                       : (0.);
                for (int_t ix = 0; ix < n; ix++)
                    biasB[ix] /= (sum_by_col[ix]
                                    + (lam_item
                                        *
                                       (scale_lam?
                                            (double)sum_by_col[ix] : 1.)));
            }

            for (int_t ix = 0; ix < n; ix++)
                biasB[ix] = (!isnan(biasB[ix]))? biasB[ix] : 0.;

            if (nonneg)
                for (int_t ix = 0; ix < n; ix++)
                    biasB[ix] = (biasB[ix] >= 0.)? biasB[ix] : 0.;

            if (iter > 0)
                set_to_zero(biasA, m);

            if (weight == NULL)
            {
                for (size_t row = 0; row < (size_t)m; row++)
                    for (size_t col = 0; col < (size_t)n; col++)
                        biasA[row] += (!isnan(Xfull[col + row*(size_t)n]))?
                                       (Xfull[col + row*(size_t)n] - biasB[col])
                                       : (0.);
                for (int_t ix = 0; ix < m; ix++)
                    biasA[ix] /= ((double)cnt_by_row[ix]
                                    + (lam_user
                                        *
                                       (scale_lam?
                                            (double)cnt_by_row[ix] : 1.)));
            }

            else
            {
                for (size_t row = 0; row < (size_t)m; row++)
                    for (size_t col = 0; col < (size_t)n; col++)
                        biasA[row] += (!isnan(Xfull[col + row*(size_t)n]))?
         weight[col + row*(size_t)n] * (Xfull[col + row*(size_t)n] - biasB[col])
                                       : (0.);
                for (int_t ix = 0; ix < m; ix++)
                    biasA[ix] /= (sum_by_row[ix]
                                    + (lam_user
                                        *
                                       (scale_lam?
                                            (double)sum_by_row[ix] : 1.)));
            }

            for (int_t ix = 0; ix < m; ix++)
                biasA[ix] = (!isnan(biasA[ix]))? biasA[ix] : 0.;

            if (nonneg)
                for (int_t ix = 0; ix < m; ix++)
                    biasA[ix] = (biasA[ix] >= 0.)? biasA[ix] : 0.;
        }

        else
        {
            if (iter > 0)
                set_to_zero(biasB, n);

            if (weight == NULL)
            {
                for (size_t ix = 0; ix < nnz; ix++)
                    biasB[ixB[ix]] += (X[ix] - biasA[ixA[ix]]);
                for (int_t ix = 0; ix < n; ix++)
                    biasB[ix] /= ((double)cnt_by_col[ix]
                                    + (lam_item
                                        *
                                       (scale_lam?
                                            (double)cnt_by_col[ix] : 1.)));
            }

            else
            {
                for (size_t ix = 0; ix < nnz; ix++)
                    biasB[ixB[ix]] += weight[ix] * (X[ix] - biasA[ixA[ix]]);
                for (int_t ix = 0; ix < n; ix++)
                    biasB[ix] /= (sum_by_col[ix]
                                    + (lam_item
                                        *
                                       (scale_lam?
                                            (double)sum_by_col[ix] : 1.)));
            }

            for (int_t ix = 0; ix < n; ix++)
                biasB[ix] = (!isnan(biasB[ix]))? biasB[ix] : 0.;

            if (nonneg)
                for (int_t ix = 0; ix < n; ix++)
                    biasB[ix] = (biasB[ix] >= 0.)? biasB[ix] : 0.;

            if (iter > 0)
                set_to_zero(biasA, m);

            if (weight == NULL)
            {
                for (size_t ix = 0; ix < nnz; ix++)
                    biasA[ixA[ix]] += (X[ix] - biasB[ixB[ix]]);
                for (int_t ix = 0; ix < m; ix++)
                    biasA[ix] /= ((double)cnt_by_row[ix]
                                    + (lam_user
                                        *
                                       (scale_lam?
                                            (double)cnt_by_row[ix] : 1.)));
            }

            else
            {
                for (size_t ix = 0; ix < nnz; ix++)
                    biasA[ixA[ix]] += weight[ix] * (X[ix] - biasB[ixB[ix]]);
                for (int_t ix = 0; ix < m; ix++)
                    biasA[ix] /= (sum_by_row[ix]
                                    + (lam_user
                                        *
                                       (scale_lam?
                                            (double)sum_by_row[ix] : 1.)));
            }

            for (int_t ix = 0; ix < m; ix++)
                biasA[ix] = (!isnan(biasA[ix]))? biasA[ix] : 0.;

            if (nonneg)
                for (int_t ix = 0; ix < m; ix++)
                    biasA[ix] = (biasA[ix] >= 0.)? biasA[ix] : 0.;
        }
    }

    cleanup:
        free(cnt_by_col);
        free(cnt_by_row);
        free(sum_by_col);
        free(sum_by_row);
    return retval;
    
    throw_oom:
    {
        retval = 1;
        print_oom_message();
        goto cleanup;
    }
}

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
)
{
    int_t retval = 0;
    real_t one = 1.;
    real_t *restrict ones = (real_t*)malloc((size_t)n*sizeof(real_t));
    if (ones == NULL) goto throw_oom;
    for (int_t ix = 0; ix < n; ix++)
        ones[ix] = 1.;
    if (biasA != NULL && user_bias)
        a_bias = biasA[row_index];
    if (!user_bias)
        a_bias = 0.;

    retval = topN(
        &one, 0,
        biasB, 0,
        (real_t*)NULL,
        glob_mean, a_bias,
        1, 0,
        include_ix, n_include,
        exclude_ix, n_exclude,
        outp_ix, outp_score,
        n_top, n, 1
    );

    cleanup:
        free(ones);
        return retval;
    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}

int_t predict_X_old_most_popular
(
    int_t row[], int_t col[], real_t *restrict predicted, size_t n_predict,
    real_t *restrict biasA, real_t *restrict biasB,
    real_t glob_mean,
    int_t m, int_t n
)
{
    if (m == 0) m = INT_MAX;
    if (n == 0) n = INT_MAX;
    bool user_bias = biasA != NULL;
    for (size_t ix = 0; ix < n_predict; ix++)
        predicted[ix] =   (row[ix] >= m || row[ix] < 0 ||
                           col[ix] >= n || col[ix] < 0)?
                                (NAN_)
                                  :
                                (biasB[col[ix]]
                                 + (user_bias? biasA[row[ix]] : 0.)
                                 + glob_mean);
    return 0;
}
