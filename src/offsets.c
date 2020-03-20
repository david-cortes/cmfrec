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

/*******************************************************************************
    Offsets Model
    -------------

    This is the model optimized for cold-start recommendations decribed in:
        Cortes, David.
        "Cold-start recommendations in Collective Matrix Factorization."
        arXiv preprint arXiv:1809.00366 (2018).


    ===========================      COMMON PART     ===========================

    A note about the mathematical notation used through these comments:
    - Variables with capital letters denote 2D matrices - e.g. 'X'.
    - Smaller-letter versions of a 2D matrix denote a single row of it.
    - X[m,n] denotes that matrix 'X' has dimensions (m,n)
    - x[n] denotes that vector 'x' has dimension 'm'.
    - X(:m,:n) denotes taking the first 'm' rows and 'n' columns of 'X'.
    - X(m:p,n:q) denotes thaking the rows from 'm' through 'p' and columns
      from 'n' to 'q'.
    - t(X) denotes the transpose of 'X', inv(X) the inverse.
    - ||X|| denotes the L2 norm of X (sum of squared entries).
    - sigm(X) denotes a sigmoid elementwise transformation: 1 / (1 + exp(-X))
    - Matrices followed by a small 'b' denote binary (0/1) entries only.
    - Matrices followed by small 'u', 'i', 'm', 's', denote that only the
      columns corresponding to certain components are taken.
    - Ae denotes an extended block matrix.
    - [Au, As, Ak] denotes a block matrix comprised by the column union of
      the above 3 matrices.
    - [[A1, A1, A3],  denotes a block matrix with sub-blocks arranged by rows
       [A4, A5, A6]]  and by columns like that.


    General idea is to factorize a sparse input matrix X[m,n] into the product
    of two lower dimensional matrices A[m,k] and B[n,k], in such a way that the
    squared error is minimized on the non-missing entries in X, given by binary
    matrix M[m,n], i.e.
        min || M * (X - A*t(B)) ||^2

    As some small improvements, the matrix 'X' is centered by substracting the
    mean from it, and additionally subtracting row and column biases, which
    are model parameters too, while imposing a regularization penalty on the
    magnitude of the parameters (given by L2 norm):
        min ||M * (X - A*t(B) - mu[1] - b1[m,1] - b2[1,n])||^2
            + lambda*(||A||^2 + ||B||^2 + ||b1||^2 + ||b2||^2)

    The intended purpose is to use this as a recommender system model, in which
    'X' is a matrix comprising the ratings that users give to items, with each
    row corresponding to a user, each column to an item, and each non-missing
    entry to the observed rating or explicit evaluation from the user.

    =======================     END OF COMMON PART     =========================

    This idea however (a) does not take into account any side information
    (attributes) about the users and/or items, (b) is slow to compute the latent
    factors for new users which were not in the training data.

    The model here extends it in such a way that the factorizing matrices are
    obtained by a linear combination of the user/item attributes U[m,p], I[n,q],
    plus a free offset which is not dependent on the attributes, i.e.
        Am = A + U*C
        Bm = B + I*D
        min ||M*(X - Am*t(Bm))||^2

    Compared to the collective model (see file 'collective.c'), this model is
    aimed at making better recommendations for new users, which it tends to do
    better, but the model is harder to fit, as the problem becomes a lot more
    non-linear. The approaches used to fit it are the same as for the collective
    model: gradient-based approach with the L-BFGS solver, and closed-form
    alternating least-squares procedure.

    This module allows the following modifications to the base formula:
    - Different regularizaton parameter for each matrix.
    - A scaling for the attribute-based components - i.e.
        Am = A + w_user * U*C
        Bm = B + w_item * I*D
    - Observation weights W[m,n] for each entry in X.
    - Having biases only for users and/or only for items, or for neither.
    - The U and I matrices are centered column-by-column, but these column
      biases are not model parameters.
    - Can have independent components (k_main, k_sec) for either A or for C - 
      i.e.
        Am = [ [0, Ak, As] + [w_user*U*Cs, w_user*U*Ck, 0] ]
        Ak = A(:,:k)      ,  As = A(:,k:k+k_main)
        Cs = C(:,:k_sec)  ,  Ck = C(:,k_sec:k_sec+k)
      (In this case note that, if only information about one of users/items
       is present while the other is missing, the other matrix must not have
       any independent components)
    And allows working with the inputs either as sparse or as dense matrices.

    For the gradient-based solution, the gradients can be calculated as:
        E[m,n]  = M * W * (Am*t(Bm) - X - b1 - b2)
        grad(A) =   E  * Bm(:,k_sec:) + lambda*A
        grad(B) = t(E) * Am(:,k_sec:) + lambda*B
        grad(C) = w_user * t(U) *   E  * Bm(:,:k_sec+k) + lambda*C
        grad(D) = w_item * t(I) * t(E) * Am(:,:k_sec+k) + lambda*D
    
    For the closed-form  alternating least-squares solution, one could also
    first obtain the optimal Am and Bm matrices after many iterations, then
    obtain C and D as the least-squares minimizers for those matrices, i.e.
        - Obtain Am and Bm through regular ALS.
        - Set
            Copt = argmin ||Am - U*C||^2
            Dopt = argmin ||Bm - I*D||^2
            Aopt = Am - U*Copt
            Bopt = Bm - I*Dopt

    Note however that this approach would apply the regularization term to
    the Am and Bm matrices rather than to the original ones. If the closed-form
    that follows the exact formulation is desired, it would be necessary to
    have a dense matrix X, from which Xdiff = X - w_user*U*C could be computed,
    in order to build the closed-form on that, and then C would still need to
    somehow get a correction according to the regularization to account for
    being multiplied by U and w_user.

    As can be seen from the formulas, calculating the factors for new users is
    easier for both the cold-start (based only on user attributes Ua[1,p]) and
    the warm-start case (based on both user attributes Ua[1,p] and on
    ratings Xa[1,n]).

    For the cold-start case, they can be obtained as:
        Am = [w_user*t(C)*t(Xa), 0[k_sec]]

    For the warm-start case, a similar closed-form solution can be taken, but
    note that when k_sec > 0, it's necessary to calculate Xdiff and solve for
    it instead.

    
*******************************************************************************/

#include "cmfrec.h"

/*******************************************************************************
    Function and Gradient Calculation
    ---------------------------------
    
    This function calculates the gradient as explained at the beginning. It
    can receive the X, U, I, matrices either as sparse (COO or CSR+CSC
    depending on parallelization strategy for X - see file 'common.c' for
    details, for U and I must be in both CSR+CSC regardless of parallelization
    strategy), but be sure to pass only ONE form (sparse/dense) of each.

    For sparse X matrix, non-present values will not be accounted for into the
    function and gradient calculations, while for dense matrices, missing values
    (as 'NAN') will not be counted. The U and I matrices cannot have missing
    values or else the function and gradient will become NAN.

    If passing observation weights, these must match the shape of the X matrix
     - that is, if X is dense, weights must be an array of dimensions (m, n),
    if X is sparse, must be an array of dimensions (nnz), and if parallelizing
    by independent X matrices, must pass it twice, each matching to a given
    format of X.


*******************************************************************************/

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
)
{
    int k_totA = k_sec + k + k_main;
    int k_totB = k_totA;
    int k_szA = k + k_main;
    int k_szB = k_szA;
    size_t dimC = (size_t)(k_sec + k) * (size_t)p;
    size_t dimD = (size_t)(k_sec + k) * (size_t)q;

    bool has_U = (U  != NULL || U_csr != NULL);
    bool has_I = (II != NULL || I_csr != NULL);

    if (!has_U) k_szA = k_totA;
    if (!has_I) k_szB = k_totB;

    FPnum f = 0;

    size_t nvars = (size_t)m * (size_t)k_szA + (size_t)n * (size_t)k_szB;
    if (user_bias) nvars += m;
    if (item_bias) nvars += n;
    if (has_U) nvars += dimC;
    if (has_I) nvars += dimD;
    if (add_intercepts && has_U) nvars += (size_t)(k_sec + k);
    if (add_intercepts && has_I) nvars += (size_t)(k_sec + k);
    set_to_zero(grad, nvars, nthreads);

    FPnum *restrict biasA = values;
    FPnum *restrict biasB = biasA + (user_bias? m : 0);
    FPnum *restrict A = biasB + (item_bias? n : 0);
    FPnum *restrict B = A + (size_t)m * (size_t)k_szA;
    FPnum *restrict C = B + (size_t)n * (size_t)k_szB;
    FPnum *restrict C_bias = C + (has_U? dimC : (size_t)0);
    FPnum *restrict D = C_bias + ((add_intercepts && has_U)?
                                  (size_t)(k_sec+k) : (size_t)0);
    FPnum *restrict D_bias = D + (has_I? dimD : (size_t)0);

    FPnum *restrict g_biasA = grad;
    FPnum *restrict g_biasB = g_biasA + (user_bias? m : 0);
    FPnum *restrict g_A = g_biasB + (item_bias? n : 0);
    FPnum *restrict g_B = g_A + (size_t)m * (size_t)k_szA;
    FPnum *restrict g_C = g_B + (size_t)n * (size_t)k_szB;
    FPnum *restrict g_C_bias = g_C + (has_U? dimC : (size_t)0);
    FPnum *restrict g_D = g_C_bias + ((add_intercepts && has_U)?
                                      (size_t)(k_sec+k) : (size_t)0);
    FPnum *restrict g_D_bias = g_D + (has_I? dimD : (size_t)0);

    FPnum *restrict Am = buffer_FPnum;
    FPnum *restrict Bm = Am + ((has_U && (k_sec || k))?
                               ((size_t)m*(size_t)k_totA) : (0));
    FPnum *restrict bufferA = Bm + ((has_I && (k_sec || k))?
                                    ((size_t)n*(size_t)k_totB) : 0);
    FPnum *restrict bufferB = bufferA
                                + (((k_main || k_sec) && has_U)?
                                   ((size_t)m * (size_t)k_totA) : (0));
    FPnum *restrict buffer_remainder = bufferB
                                        + (((k_main || k_sec) && has_I)?
                                           ((size_t)n * (size_t)k_totB) : (0));
    if ((k_main || k_sec) && (has_U || has_I))
    {
        if (has_U && has_I) {
            set_to_zero(bufferA,
                        (size_t)m*(size_t)k_totA + (size_t)n*(size_t)k_totB,
                        nthreads);
        } else if (has_U && !has_I) {
            set_to_zero(bufferA, (size_t)m*(size_t)k_totA, nthreads);
            bufferB = g_B;
        } else if (!has_U && has_I) {
            set_to_zero(bufferB, (size_t)n*(size_t)k_totB, nthreads);
            bufferA = g_A;
        }
    }
    else {
        bufferA = g_A;
        bufferB = g_B;
    }

    /* Pre-multiply and sum the factor matrices to obtain the factors
       to use in a linear comb. */
    if (has_U && (k_sec || k))
        construct_Am(
            Am, A,
            C, C_bias,
            add_intercepts,
            U, m, p,
            U_csr_p, U_csr_i, U_csr,
            k, k_sec, k_main,
            w_user, nthreads
        );
    else
        Am = A;

    if (has_I && (k_sec || k))
        construct_Am(
            Bm, B,
            D, D_bias,
            add_intercepts,
            II, n, q,
            I_csr_p, I_csr_i, I_csr,
            k, k_sec, k_main,
            w_item, nthreads
        );
    else
        Bm = B;

    f = fun_grad_cannonical_form(
        Am, k_totA, Bm, k_totB,
        bufferA, bufferB,
        m, n, k_sec+k+k_main,
        ixA, ixB, X, nnz,
        Xfull, full_dense,
        Xcsr_p, Xcsr_i, Xcsr,
        Xcsc_p, Xcsc_i, Xcsc,
        user_bias, item_bias,
        biasA, biasB,
        g_biasA, g_biasB,
        weight, weightR, weightC,
        1.,
        buffer_remainder,
        buffer_mt,
        true,
        nthreads
    );

    if (has_U)
        assign_gradients(
            bufferA, g_A, g_C,
            add_intercepts, g_C_bias,
            U, U_csc_p, U_csc_i, U_csc,
            m, p, k, k_sec, k_main,
            w_user, nthreads
        );
    if (has_I)
        assign_gradients(
            bufferB, g_B, g_D,
            add_intercepts, g_D_bias,
            II, I_csc_p, I_csc_i, I_csc,
            n, q, k, k_sec, k_main,
            w_item, nthreads
        );

    /* If all matrices have the same regulatization, can do it in one pass */
    if (lam_unique == NULL) {
        saxpy_large(values, lam, grad, nvars, nthreads);
        f += (lam / 2.) * sum_squares(values, nvars, nthreads);
    }

    else {
        long double freg = 0;
        if (user_bias) cblas_taxpy(m, lam_unique[0], biasA, 1, g_biasA, 1);
        if (item_bias) cblas_taxpy(n, lam_unique[1], biasB, 1, g_biasB, 1);
        saxpy_large(A, lam_unique[2], g_A, (size_t)m*(size_t)k_szA, nthreads);
        saxpy_large(B, lam_unique[3], g_B, (size_t)n*(size_t)k_szB, nthreads);

        if (has_U)
            saxpy_large(C, lam_unique[4], g_C,
                        (size_t)(p + (int)add_intercepts)*(size_t)(k_sec+k),
                        nthreads);
        if (has_I)
            saxpy_large(D, lam_unique[5], g_D,
                        (size_t)(q + (int)add_intercepts)*(size_t)(k_sec+k),
                        nthreads);

        if (user_bias)
            freg += (lam_unique[0] / 2.) * cblas_tdot(m, biasA, 1, biasA, 1);
        if (item_bias)
            freg += (lam_unique[1] / 2.) * cblas_tdot(n, biasB, 1, biasB, 1);
        freg += (lam_unique[2] / 2.) * sum_squares(A, (size_t)m*(size_t)k_szA,
                                                   nthreads);
        freg += (lam_unique[3] / 2.) * sum_squares(B, (size_t)n*(size_t)k_szB,
                                                   nthreads);
        if (has_U)
            freg += (lam_unique[4] / 2.)
                     * sum_squares(C, (size_t)(p + (int)add_intercepts)
                                       * (size_t)(k_sec+k), nthreads);
        if (has_I)
            freg += (lam_unique[5] / 2.)
                     * sum_squares(D, (size_t)(q + (int)add_intercepts)
                                       * (size_t)(k_sec+k),nthreads);
        f += (FPnum)freg;
    }

    return (FPnum) f;
}

void construct_Am
(
    FPnum *restrict Am, FPnum *restrict A,
    FPnum *restrict C, FPnum *restrict C_bias,
    bool add_intercepts,
    FPnum *restrict U, int m, int p,
    long U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    int k, int k_sec, int k_main,
    FPnum w_user, int nthreads
)
{
    /* Matrix dimensions are:
       - A[m, k+k_main]
       - C[p, k_sec+k]
       - C_bias[k_sec+k]
       - Am[m, k_sec+k+k_main] */

    size_t k_totA = k_sec + k + k_main;
    size_t k_szA = ((U == NULL && U_csr == NULL)? k_sec : 0) + k + k_main;

    if (k_sec == 0 && k_main == 0) {
        copy_arr(A, Am, (size_t)m*k_totA, nthreads);
    } else {
        /* Am[:,:] = 0; Am[:,k_sec:] = A[:,:] */
        set_to_zero(Am, (size_t)m*k_totA, nthreads);
        copy_mat(m, k+k_main, A, k_szA, Am + k_sec, (int)k_totA);
    }

    /* Am[:,:k_sec+k] += U * C */
    if (U != NULL)
        cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, k_sec+k, p,
                    w_user, U, p, C, k_sec+k,
                    1., Am, k_totA);
    else if (U_csr != NULL)
        sgemm_sp_dense(m, k+k_sec, w_user,
                       U_csr_p, U_csr_i, U_csr,
                       C, k_sec+k,
                       Am, k_totA,
                       nthreads);

    if (add_intercepts)
        mat_plus_colvec(Am, C_bias, w_user, m, k_sec+k, k_totA, nthreads);
}

void assign_gradients
(
    FPnum *restrict bufferA, FPnum *restrict g_A, FPnum *restrict g_C,
    bool add_intercepts, FPnum *restrict g_C_bias,
    FPnum *restrict U,
    long U_csc_p[], int U_csc_i[], FPnum *restrict U_csc,
    int m, int p, int k, int k_sec, int k_main,
    FPnum w_user, int nthreads
)
{
    if (U != NULL && (k || k_sec))
        cblas_tgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    p, k_sec+k, m,
                    w_user, U, p, bufferA, k_sec+k+k_main,
                    0., g_C, k_sec+k);
    else if (U_csc != NULL && (k || k_sec))
        sgemm_sp_dense(p, k_sec+k, w_user,
                       U_csc_p, U_csc_i, U_csc,
                       bufferA, k_sec+k+k_main,
                       g_C, k_sec+k,
                       nthreads);
    if (bufferA != g_A && (k || k_main))
        copy_mat(m, k+k_main,
                 bufferA + k_sec, k_sec+k+k_main,
                 g_A, k+k_main);

    if (add_intercepts) {
        sum_by_cols(bufferA, g_C_bias, m, k_sec+k,
                    (size_t)(k_sec+k+k_main), nthreads);
        if (w_user != 1.)
            cblas_tscal(k_sec+k, w_user, g_C_bias, 1);
    }

}

int offsets_factors_cold
(
    FPnum *restrict a_vec,
    FPnum *restrict u_vec,
    int u_vec_ixB[], FPnum *restrict u_vec_sp, size_t nnz_u_vec,
    FPnum *restrict C, int p,
    FPnum *restrict C_bias,
    int k, int k_sec, int k_main,
    FPnum w_user
)
{
    /* a_vec[:k_sec+k] := t(C)*u_vec
       a_vec[k_sec+k:] := 0 */
    int k_pred = k_sec + k;
    if (u_vec_sp != NULL)
        set_to_zero(a_vec, k_sec+k+k_main, 1);
    else if (k_main != 0)
        set_to_zero(a_vec + (k_sec+k), k_main, 1);

    if (u_vec != NULL)
        cblas_tgemv(CblasRowMajor, CblasTrans,
                    p, k_sec+k,
                    w_user, C, k_sec+k,
                    u_vec, 1,
                    0., a_vec, 1);
    else {
        sgemv_dense_sp(p, k_pred,
                       w_user, C, k_sec+k,
                       u_vec_ixB, u_vec_sp, nnz_u_vec,
                       a_vec);
        if (w_user != 1)
            cblas_tscal(k_pred, w_user, a_vec, 1);
    }

    if (C_bias != NULL)
        cblas_taxpy(k_sec+k, 1., C_bias, 1, a_vec, 1);

    return 0;
}

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
)
{
    /* TODO: add functionality for obtaining an exact solution through
       the L-BFGS solver for a single row of A, like it was done for
       the collective model. */
    int retval = 0;
    FPnum *restrict buffer_FPnum = NULL;
    size_t size_buffer = 0;
    int cnt_NA = 0;
    bool append_bias = (Bm_plus_bias != NULL && a_bias != NULL);
    if (!implicit)
        preprocess_vec(Xa_dense, n, ixB, Xa, nnz,
                       glob_mean, lam_bias, biasB,
                       (Bm_plus_bias == NULL)? a_bias : (FPnum*)NULL,
                       &cnt_NA);

    FPnum *restrict a_plus_bias = NULL;
    if (append_bias) {
        a_plus_bias = (FPnum*)malloc((k_sec+k+k_main+1)*sizeof(FPnum));
        if (a_plus_bias == NULL) { retval = 1; goto cleanup; }
    }

    if (implicit) lam /= w_main_multiplier;

    if ((!exact && k_sec == 0) || implicit)
    {
        if (precomputedBtBinvBt == NULL || Xa_dense == NULL ||
            cnt_NA > 0 || weight != NULL || implicit)
        {
            size_buffer = square(k_sec + k + k_main + append_bias);
            if (Xa_dense != NULL) {
                if (cnt_NA > 0 || weight != NULL)
                    size_buffer += (size_t)n
                                    * (size_t)(k_sec+k+k_main + append_bias);
            }
            buffer_FPnum = (FPnum*)malloc(size_buffer*sizeof(FPnum));
            if (buffer_FPnum == NULL) return 1;
        }

        /* Determine a_vec[:] through closed-form, ignore u_vec for them
           as it's possible to get
             Am := a_vec + t(C)*u_vec
           straight away and then subtract t(C)*u_vec if needed - this is
           however not exact as it is applying the regularization term to 'Am'
           instead of 'a_vec', but the difference should be small. This is
           faster as it avoids re-creating Xa-U*C*t(Bm). */
        if (!implicit) {
            if (!append_bias)
                factors_closed_form(a_vec, k_sec+k+k_main,
                                    Bm, n, k_sec+k+k_main,
                                    Xa_dense, cnt_NA==0,
                                    Xa, ixB, nnz,
                                    weight,
                                    buffer_FPnum,
                                    lam, 1., lam,
                                    precomputedBtBinvBt,
                                    precomputedBtBw, cnt_NA, 0,
                                    (FPnum*)NULL, false, false,
                                    false);
            else {
                factors_closed_form(a_plus_bias, k_sec+k+k_main+1,
                                    Bm_plus_bias, n, k_sec+k+k_main+1,
                                    Xa_dense, cnt_NA==0,
                                    Xa, ixB, nnz,
                                    weight,
                                    buffer_FPnum,
                                    lam, 1., lam_bias,
                                    precomputedBtBinvBt,
                                    precomputedBtBw, cnt_NA, 0,
                                    (FPnum*)NULL, false, false,
                                    false);
                memcpy(a_vec, a_plus_bias, (k_sec+k+k_main)*sizeof(FPnum));
                *a_bias = a_plus_bias[k_sec+k+k_main];
            }
        }
        else
            factors_implicit(
                a_vec, k_sec+k+k_main,
                Bm, k_sec+k+k_main,
                Xa, ixB, nnz,
                lam, alpha,
                precomputedBtBw, 0,
                true, false,
                buffer_FPnum,
                false
            );

        /* If A is required instead of just Am, then calculate as
             A := Am - U*C */
        if (output_a != NULL)
        {
            offsets_factors_cold(output_a, u_vec,
                                 u_vec_ixB, u_vec_sp, nnz_u_vec,
                                 C, p,
                                 C_bias,
                                 k, k_sec, k_main, w_user);
            for (int ix = k_sec; ix < k_sec+k+k_main; ix++)
                output_a[ix - k_sec] -= w_user * a_vec[ix];
        }

    }

    else
    {
        size_buffer = square(k_sec+k+k_main+append_bias) + n + k_sec+k;
        if (weight != NULL)
            size_buffer += (size_t)n * (size_t)(k_sec+k+k_main + append_bias);
        if (weight != NULL && Xa_dense == NULL)
            size_buffer += n;
        buffer_FPnum = (FPnum*)malloc(size_buffer*sizeof(FPnum));
        if (buffer_FPnum == NULL) return 1;

        FPnum *restrict bufferX = buffer_FPnum;
        FPnum *restrict bufferW = bufferX + n;
        FPnum *restrict buffer_uc = bufferW
                                     + ((weight != NULL && Xa_dense == NULL)?
                                        (n) : (0));
        FPnum *restrict buffer_remainder = buffer_uc + k_sec+k;

        /* If the exact solution is desired (regularization applied to A, not
           to Am), or if there are factors from only the side info (k_sec),
           then it can be obtained through the closed form by transforming the
           X matrix:
             X' := X - U*C*t(Bm),
           and determining optimal A from it given Bm.
           Am is then obtained as
             Am := Aopt + U*C. */

        /* a_vec = w*U*C */
        if (u_vec != NULL || u_vec_sp != NULL)
            offsets_factors_cold(a_vec, u_vec,
                                 u_vec_ixB, u_vec_sp, nnz_u_vec,
                                 C, p,
                                 C_bias,
                                 k, k_sec, 0, w_user);
        else
            set_to_zero(a_vec, k_sec+k+k_main, 1);

        /* buffer_uc = w*U*C (save for later) */
        memcpy(buffer_uc, a_vec, (k_sec+k)*sizeof(FPnum));
        /* BufferX = -w*U*C*t(Bm) */
        if (u_vec != NULL || u_vec_sp != NULL)
            cblas_tgemv(CblasRowMajor, CblasNoTrans,
                        n, k_sec+k,
                        -1., Bm, k_sec+k+k_main,
                        a_vec, 1,
                        0., bufferX, 1);
        else
            set_to_zero(bufferX, n, 1);

        /* BufferX += X */
        if (Xa_dense == NULL)
        {
            for (size_t ix = 0; ix < nnz; ix++)
                bufferX[ixB[ix]] += Xa[ix];
            if (weight != NULL) {
                for (int ix = 0; ix < n; ix++) bufferW[ix] = 1.;
                for (size_t ix = 0; ix < nnz; ix++)
                    bufferW[ixB[ix]] = weight[ix];
                weight = bufferW;
            }
        }
        else
        {
            for (int ix = 0; ix < n; ix++)
                bufferX[ix] += isnan(Xa_dense[ix])? 0 : Xa_dense[ix];
        }

        /* Solve lsq(A, X - w*U*C*t(Bm)) */
        if (k || k_main) {
            if (!append_bias)
                factors_closed_form(a_vec + k_sec, k+k_main,
                                    Bm + k_sec, n, k_sec+k+k_main,
                                    bufferX, true,
                                    (FPnum*)NULL, (int*)NULL, (size_t)0,
                                    weight,
                                    buffer_remainder,
                                    lam, 1., lam,
                                    (FPnum*)NULL,
                                    (FPnum*)NULL, 0, 0,
                                    (FPnum*)NULL, false, false,
                                    false);
            else {
                factors_closed_form(a_plus_bias + k_sec, k+k_main+1,
                                    Bm_plus_bias + k_sec, n, k_sec+k+k_main+1,
                                    bufferX, true,
                                    (FPnum*)NULL, (int*)NULL, (size_t)0,
                                    weight,
                                    buffer_remainder,
                                    lam, 1., lam_bias,
                                    (FPnum*)NULL,
                                    (FPnum*)NULL, 0, 0,
                                    (FPnum*)NULL, false, false,
                                    false);
                memcpy(a_vec + k_sec, a_plus_bias + k_sec,
                       (k+k_main)*sizeof(FPnum));
                *a_bias = a_plus_bias[k_sec+k+k_main];
            }
        }
        /* Save A (as opposed to Am) */
        if (output_a != NULL && (k || k_main))
            memcpy(output_a, a_vec + k_sec, (k+k_main)*sizeof(FPnum));

        /* Am = A + w*U*C */
        for (int ix = 0; ix < k_sec+k; ix++)
            a_vec[ix] += buffer_uc[ix];
    }

    cleanup:
        free(buffer_FPnum);
        free(a_plus_bias);
    return retval;
}

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
)
{
    /* Shapes are as follows:
       - Am[m, k_sec+k+k_main]
       - Bm[n, k_sec+k+k_main]
       - BtBinvBt[k_sec+k+k_main, n]
       - BtBw[k_sec+k+k_main, k_sec+k+k_main]
       - BtBchol[k_sec+k+k_main, k_sec+k+k_main]
       - CtCinvCt[k_sec+k, p]
       - CtC[k_sec+k, k_sec+k]
       - CtCchol[k_sec+k, k_sec+k] */
    bool has_U = (U  != NULL || U_csr != NULL);
    bool has_I = (II != NULL || I_csr != NULL);
    size_t size_buffer = (size_t)n * (size_t)(k_sec+k+k_main);
    if (has_U)
        size_buffer = max2(size_buffer, (size_t)p * (size_t)(k_sec+k));
    FPnum *restrict buffer_FPnum = (FPnum*)malloc(size_buffer*sizeof(FPnum));
    if (buffer_FPnum == NULL) return 1;

    if (has_U && Am != NULL)
        construct_Am(
            Am, A,
            C, C_bias,
            add_intercepts,
            U, m, p,
            U_csr_p, U_csc_i, U_csr,
            k, k_sec, k_main,
            w_user, nthreads
        );
    else
        Am = A;

    if (has_I && Bm != NULL)
        construct_Am(
            Bm, B,
            D, D_bias,
            add_intercepts,
            II, n, q,
            I_csr_p, I_csc_i, I_csr,
            k, k_sec, k_main,
            w_item, nthreads
        );
    else
        Bm = B;

    if (!implicit)
        AtAinvAt_plus_chol(Bm, k_sec+k+k_main, 0,
                           BtBinvBt,
                           BtBw,
                           BtBchol,
                           lam, lam_last, n, k_sec+k+k_main, 1.,
                           buffer_FPnum,
                           false);
    else
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k_sec+k+k_main, n,
                    1., B, k_sec+k+k_main,
                    0., BtBw, k_sec+k+k_main);

    free(buffer_FPnum);
    return 0;
}

lbfgsFPnumval_t wrapper_offsets_fun_grad
(
    void *instance,
    lbfgsFPnumval_t *x,
    lbfgsFPnumval_t *g,
    const size_t n,
    const lbfgsFPnumval_t step
)
{
    data_offsets_fun_grad *data = (data_offsets_fun_grad*)instance;
    (data->nfev)++;
    return offsets_fun_grad(
        x, g,
        data->ixA, data->ixB, data->X,
        data->nnz, data->m, data->n, data->k,
        data->Xfull, data->full_dense,
        data->Xcsr_p, data->Xcsr_i, data->Xcsr,
        data->Xcsc_p, data->Xcsc_i, data->Xcsc,
        data->weight, data->weightR, data->weightC,
        data->user_bias, data->item_bias,
        data->add_intercepts,
        data->lam, data->lam_unique,
        data->U, data->p,
        data->II, data->q,
        data->U_csr_p, data->U_csr_i, data->U_csr,
        data->U_csc_p, data->U_csc_i, data->U_csc,
        data->I_csr_p, data->I_csr_i, data->I_csr,
        data->I_csc_p, data->I_csc_i, data->I_csc,
        data->k_main, data->k_sec,
        data->w_user, data->w_item,
        data->nthreads,
        data->buffer_FPnum,
        data->buffer_mt
    );
}

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
)
{
    FPnum *restrict buffer_FPnum = NULL;
    FPnum *restrict buffer_mt = NULL;
    int retval = 0;
    size_t nvars = (size_t)m*(size_t)(k+k_main) + (size_t)n*(size_t)(k+k_main)
                   + (size_t)(user_bias? m : 0) + (size_t)(item_bias? n : 0)
                   + (size_t)p*(size_t)(k_sec+k) + (size_t)q*(size_t)(k_sec+k);
    size_t size_buffer = 0;
    if (Xfull != NULL) size_buffer = (size_t)m * (size_t)n;
    if (U != NULL || U_sp != NULL || II != NULL || I_sp != NULL ||
        k_sec != 0 || k_main != 0)
    {
        if (U !=  NULL || U_sp != NULL) {
            if (k_sec || k)
                size_buffer += (size_t)m * (size_t)(k_sec+k+k_main);
            if (k_main || k_sec)
                size_buffer += (size_t)m * (size_t)(k_sec+k+k_main);
            if (add_intercepts)
                nvars += (size_t)(k_sec + k);
        } else {
            nvars += (size_t)m * (size_t)k_sec;
        }
        if (II != NULL || I_sp != NULL) {
            if (k_sec || k)
                size_buffer += (size_t)n * (size_t)(k_sec+k+k_main);
            if (k_main || k_sec)
                size_buffer += (size_t)n * (size_t)(k_sec+k+k_main);
            if (add_intercepts)
                nvars += (size_t)(k_sec + k);
        } else {
            nvars += (size_t)n * (size_t)k_sec;
        }
    }
    if (size_buffer) {
        buffer_FPnum = (FPnum*)malloc(size_buffer*sizeof(FPnum));
        if (buffer_FPnum == NULL) return 1;
    }

    FPnum *restrict biasA = values;
    FPnum *restrict biasB = biasA + (user_bias? m : 0);
    FPnum *restrict A = biasB + (item_bias? n : 0);
    FPnum *restrict B = A + ((size_t)m
                             * (size_t)((U != NULL  || U_row != NULL)?
                                        (k+k_main) : (k_sec+k+k_main)));
    FPnum *restrict C = B + ((size_t)n
                             * (size_t)((II != NULL || I_row != NULL)?
                                        (k+k_main) : (k_sec+k+k_main)));
    FPnum *restrict C_bias = C + ((U != NULL || U_sp != NULL)?
                                  ((size_t)p * (size_t)(k_sec+k)) : (0));
    FPnum *restrict D = C_bias + (
                                (add_intercepts && (U != NULL || U_sp != NULL))?
                                (size_t)(k_sec+k) : (size_t)0
                                );
    FPnum *restrict D_bias = D + ((II != NULL || I_sp != NULL)?
                                  ((size_t)q * (size_t)(k_sec+k)) : (0));

    lbfgs_parameter_t lbfgs_params;
    data_offsets_fun_grad data;
    lbfgsFPnumval_t funval;

    long *Xcsr_p = NULL;
    int *Xcsr_i = NULL;
    FPnum *restrict Xcsr = NULL;
    FPnum *restrict weightR = NULL;
    long *Xcsc_p = NULL;
    int *Xcsc_i = NULL;
    FPnum *restrict Xcsc = NULL;
    FPnum *restrict weightC = NULL;
    long *U_csr_p = NULL;
    int *U_csr_i = NULL;
    FPnum *restrict U_csr = NULL;
    long *U_csc_p = NULL;
    int *U_csc_i = NULL;
    FPnum *restrict U_csc = NULL;
    long *I_csr_p = NULL;
    int *I_csr_i = NULL;
    FPnum *restrict I_csr = NULL;
    long *I_csc_p = NULL;
    int *I_csc_i = NULL;
    FPnum *restrict I_csc = NULL;

    bool full_dense = false;
    if (Xfull != NULL)
        full_dense = (count_NAs(Xfull, (size_t)m*(size_t)n, nthreads)) == 0;
    
    #ifdef _OPENMP
    if (nthreads > 1 && Xfull == NULL)
    {
        if (prefer_onepass)
        {
            size_t size_mt = (size_t)(m + n) * (size_t)(k_sec + k + k_main);
            if (user_bias) size_mt += (size_t)m;
            if (item_bias) size_mt += (size_t)n;
            buffer_mt = (FPnum*)malloc((size_t)nthreads*size_mt*sizeof(FPnum));
            if (buffer_mt == NULL)
            {
                retval = 1;
                goto cleanup;
            }
        }

        else
        {
            retval = convert_sparse_X(
                        ixA, ixB, X, nnz,
                        &Xcsr_p, &Xcsr_i, &Xcsr,
                        &Xcsc_p, &Xcsc_i, &Xcsc,
                        weight, &weightR, &weightC,
                        m, n, nthreads
                    );
            if (retval != 0) goto cleanup;
        }
    }
    #endif

    if (U_sp != NULL)
    {
        retval = preprocess_sideinfo_matrix(
            (FPnum*)NULL, m, p,
            U_row, U_col, U_sp, nnz_U,
            (FPnum*)NULL, (FPnum**)NULL, (FPnum**)NULL,
            &U_csr_p, &U_csr_i, &U_csr,
            &U_csc_p, &U_csc_i, &U_csc,
            (int**)NULL, (int**)NULL,
            (bool*)NULL, (bool*)NULL, (bool*)NULL,
            false, nthreads
        );
        if (retval != 0) goto cleanup;
    }

    if (I_sp != NULL)
    {
        retval = preprocess_sideinfo_matrix(
            (FPnum*)NULL, n, q,
            I_row, I_col, I_sp, nnz_I,
            (FPnum*)NULL, (FPnum**)NULL, (FPnum**)NULL,
            &I_csr_p, &I_csr_i, &I_csr,
            &I_csc_p, &I_csc_i, &I_csc,
            (int**)NULL, (int**)NULL,
            (bool*)NULL, (bool*)NULL, (bool*)NULL,
            false, nthreads
        );
        if (retval != 0) goto cleanup;
    }

    retval = initialize_biases(
        glob_mean, values, values + (user_bias? m : 0),
        user_bias, item_bias,
        (lam_unique == NULL)? (lam) : (lam_unique[0]),
        (lam_unique == NULL)? (lam) : (lam_unique[1]),
        m, n,
        m, n,
        ixA, ixB, X, nnz,
        Xfull, (FPnum*)NULL,
        Xcsr_p, Xcsr_i, Xcsr,
        Xcsc_p, Xcsc_i, Xcsc,
        nthreads
    );
    if (retval != 0) goto cleanup;

    if (reset_values)
        retval = rnorm(values + (user_bias? m : 0) + (item_bias? n : 0),
                       nvars - (size_t)(user_bias? m : 0)
                             - (size_t)(item_bias? n : 0),
                       seed, nthreads);
    if (retval != 0) goto cleanup;

    lbfgs_params = 
                    #ifndef __cplusplus
                    (lbfgs_parameter_t)
                    #endif 
                                        {
        (size_t)n_corr_pairs, 1e-5, 0, 1e-5,
        maxiter, LBFGS_LINESEARCH_DEFAULT, 40,
        1e-20, 1e20, 1e-4, 0.9, 0.9, 1.0e-16,
        0.0, 0, -1,
    };
    data = 
            #ifndef __cplusplus
            (data_offsets_fun_grad) 
            #endif
                                    {
        ixA, ixB, X,
        nnz, m, n, k,
        Xfull, full_dense,
        Xcsr_p, Xcsr_i, Xcsr,
        Xcsc_p, Xcsc_i, Xcsc,
        weight, weightR, weightC,
        user_bias, item_bias,
        add_intercepts,
        lam, lam_unique,
        U, p,
        II, q,
        U_csr_p, U_csr_i, U_csr,
        U_csc_p, U_csc_i, U_csc,
        I_csr_p, I_csr_i, I_csr,
        I_csc_p, I_csc_i, I_csc,
        k_main, k_sec,
        w_user, w_item,
        nthreads,
        buffer_FPnum,
        buffer_mt,
        print_every, 0, 0,
    };

    signal(SIGINT, set_interrup_global_variable);
    if (should_stop_procedure)
    {
        should_stop_procedure = false;
        fprintf(stderr, "Procedure terminated before starting optimization\n");
        goto cleanup;
    }

    retval = lbfgs(
        nvars,
        values,
        &funval,
        wrapper_offsets_fun_grad,
        (verbose)? (lbfgs_printer_offsets) : (NULL),
        (void*) &data,
        &lbfgs_params,
        (lbfgsFPnumval_t*)NULL,
        (iteration_data_t*)NULL
    );
    if (verbose) {
        printf("\n\nOptimization terminated\n");
        printf("\t%s\n", lbfgs_strerror(retval));
        printf("\tniter:%3d, nfev:%3d\n", data.niter, data.nfev);
        fflush(stdout);
    }
    if (retval == LBFGSERR_OUTOFMEMORY)
        retval = 1;
    else
        retval = 0;
    *niter = data.niter;
    *nfev = data.nfev;

    if (retval != 1)
    {
        if ((U != NULL || U_csr != NULL) && Am != NULL)
            construct_Am(
                Am,
                A,
                C, C_bias,
                add_intercepts,
                U, m, p,
                U_csr_p, U_csr_i, U_csr,
                k, k_sec, k_main,
                w_user, nthreads
            );

        if ((II != NULL || I_csr != NULL) && Bm != NULL)
            construct_Am(
                Bm,
                B,
                D, D_bias,
                add_intercepts,
                II, n, q,
                I_csr_p, I_csr_i, I_csr,
                k, k_sec, k_main,
                w_item, nthreads
            );

        if (Bm_plus_bias != NULL && user_bias)
            append_ones_last_col(
                Bm,
                n, k_sec+k+k_main,
                Bm_plus_bias
            );
    }

    cleanup:
        free(buffer_FPnum);
        free(buffer_mt);
        free(Xcsr_p);
        free(Xcsr_i);
        free(Xcsr);
        free(weightR);
        free(Xcsc_p);
        free(Xcsc_i);
        free(Xcsc);
        free(weightC);
        free(U_csr_p);
        free(U_csr_i);
        free(U_csr);
        free(U_csc_p);
        free(U_csc_i);
        free(U_csc);
        free(I_csr_p);
        free(I_csr_i);
        free(I_csr);
        free(I_csc_p);
        free(I_csc_i);
        free(I_csc);
    if (retval == 1) return 1;
    return 0;
}

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
)
{
    if (p > m || q > n || k > m || k > n)
        return 2;
    if (implicit && (NA_as_zero_X || weight != NULL || Xfull != NULL))
        return 2;

    int retval;
    if (!implicit)
        retval = fit_collective_explicit_als(
                    values, reset_values,
                    glob_mean,
                    (FPnum*)NULL, (FPnum*)NULL,
                    m, n, k,
                    ixA, ixB, X, nnz,
                    Xfull,
                    weight,
                    user_bias, item_bias,
                    lam, (FPnum*)NULL,
                    (FPnum*)NULL, 0, 0,
                    (FPnum*)NULL, 0, 0,
                    (int*)NULL, (int*)NULL, (FPnum*)NULL, 0,
                    (int*)NULL, (int*)NULL, (FPnum*)NULL, 0,
                    NA_as_zero_X, false, false,
                    0, 0, 0,
                    1., 1., 1.,
                    niter, nthreads, seed, verbose, use_cg,
                    Bm_plus_bias
                );
    else
        retval = fit_collective_implicit_als(
                    values, reset_values,
                    (FPnum*)NULL, (FPnum*)NULL,
                    m, n, k,
                    ixA, ixB, X, nnz,
                    lam, (FPnum*)NULL,
                    (FPnum*)NULL, 0, 0,
                    (FPnum*)NULL, 0, 0,
                    (int*)NULL, (int*)NULL, (FPnum*)NULL, 0,
                    (int*)NULL, (int*)NULL, (FPnum*)NULL, 0,
                    false, false,
                    0, 0, 0,
                    1., 1., 1.,
                    w_main_multiplier,
                    alpha, adjust_weight,
                    niter, nthreads, seed, verbose, use_cg
                );
    if (retval == 1) return 1;

    FPnum *restrict biasA = values;
    FPnum *restrict biasB = biasA + (user_bias? m : 0);
    FPnum *restrict A = biasB + (item_bias? n : 0);
    FPnum *restrict B = A + (size_t)m * (size_t)k;
    FPnum *restrict C = B + (size_t)n * (size_t)k;
    FPnum *restrict C_bias = C + ((U != NULL)?
                                  ((size_t)p*(size_t)k) : ((size_t)0));
    FPnum *restrict D = C_bias + ((add_intercepts && U != NULL)? (k) : (0));
    FPnum *restrict MatTrans = NULL;
    FPnum *restrict MatTransSec = NULL;
    FPnum *restrict buffer_FPnum = NULL;

    int minus_one = -1;
    char trans = 'T';
    char uplo = '?';
    int ignore;
    FPnum temp = 0;
    int temp_intA;
    int temp_intB;

    int p_plus_bias = p + (int)add_intercepts;
    int q_plus_bias = q + (int)add_intercepts;
    FPnum *restrict U_plus_bias = NULL;
    FPnum *restrict I_plus_bias = NULL;

    if (U == NULL || !add_intercepts)
        U_plus_bias = U;
    else {
        U_plus_bias = (FPnum*)malloc((size_t)m*(size_t)(p+1)*sizeof(FPnum));
        if (U_plus_bias == NULL) { retval = 1; goto cleanup; }
        append_ones_last_col(
            U, m, p,
            U_plus_bias
        );
    }

    if (II == NULL || !add_intercepts)
        I_plus_bias = II;
    else {
        I_plus_bias = (FPnum*)malloc((size_t)n*(size_t)(q+1)*sizeof(FPnum));
        if (I_plus_bias == NULL) { retval = 1; goto cleanup; }
        append_ones_last_col(
            II, n, q,
            I_plus_bias
        );
    }

    if (U != NULL && II != NULL) {
        MatTrans = (FPnum*)malloc((size_t)max2(m,n)*(size_t)k*sizeof(FPnum));
        MatTransSec = (FPnum*)malloc((size_t)max2(p_plus_bias,q_plus_bias)
                                      * (size_t)k * sizeof(FPnum));
    }
    else if (U != NULL) {
        MatTrans = (FPnum*)malloc((size_t)m*(size_t)k*sizeof(FPnum));
        MatTransSec = (FPnum*)malloc((size_t)p_plus_bias
                                      * (size_t)k*sizeof(FPnum));
    }
    else if (II != NULL) {
        MatTrans = (FPnum*)malloc((size_t)n*(size_t)k*sizeof(FPnum));
        MatTransSec = (FPnum*)malloc((size_t)q_plus_bias
                                      * (size_t)k*sizeof(FPnum));
    }

    if ((U != NULL || II != NULL) && (MatTrans == NULL || MatTransSec == NULL))
    {
        retval = 1;
        goto cleanup;
    }

    if (Am != NULL)
        copy_arr(A, Am, (size_t)m*(size_t)k, nthreads);
    if (Bm != NULL)
        copy_arr(B, Bm, (size_t)n*(size_t)k, nthreads);

    if (U != NULL)
    {
        /* Determine the size of the working array */
        tgels_(&trans, &p_plus_bias, &m, &k,
               U_plus_bias, &p_plus_bias, MatTrans, &m,
               &temp, &minus_one, &ignore);
        temp_intA = (int)temp;
        buffer_FPnum = (FPnum*)malloc((size_t)temp_intA*sizeof(FPnum));
        if (buffer_FPnum == NULL) { retval = 1; goto cleanup; }

        /* Obtain t(C) */
        transpose_mat2(A, m, k, MatTrans);
        tgels_(&trans, &p_plus_bias, &m, &k,
               U_plus_bias, &p_plus_bias, MatTrans, &m,
               buffer_FPnum, &temp_intA, &ignore);
        tlacpy_(&uplo, &k, &p_plus_bias, MatTrans, &m, MatTransSec, &k);


        /* Now obtain A and C */
        cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m, k, p_plus_bias,
                    -1., U_plus_bias, p_plus_bias, MatTransSec, p_plus_bias,
                    1., A, k);
        transpose_mat2(MatTransSec, k, p_plus_bias, C);
    }

    if (II != NULL)
    {
        /* Determine the size of the working array */
        tgels_(&trans, &q_plus_bias, &n, &k,
               I_plus_bias, &q_plus_bias, MatTrans, &n,
               &temp, &minus_one, &ignore);
        temp_intB = (int)temp;

        if (temp_intB > temp_intA)
            free(buffer_FPnum);
        if (buffer_FPnum == NULL)
            buffer_FPnum = (FPnum*)malloc((size_t)temp_intB*sizeof(FPnum));
        if (buffer_FPnum == NULL) { retval = 1; goto cleanup; }

        /* Obtain t(D) */
        transpose_mat2(B, n, k, MatTrans);
        tgels_(&trans, &q_plus_bias, &n, &k,
               I_plus_bias, &q_plus_bias, MatTrans, &n,
               buffer_FPnum, &temp_intB, &ignore);
        tlacpy_(&uplo, &k, &q_plus_bias, MatTrans, &n, MatTransSec, &k);

        /* Now obtain B and D */
        cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n, k, q_plus_bias,
                    -1., I_plus_bias, q_plus_bias, MatTransSec, q_plus_bias,
                    1., B, k);
        transpose_mat2(MatTransSec, k, q_plus_bias, D);
    }


    cleanup:
        free(MatTrans);
        free(MatTransSec);
        free(buffer_FPnum);
        if (U_plus_bias != U)
            free(U_plus_bias);
        if (I_plus_bias != II)
            free(I_plus_bias);
    if (retval == 1) return 1;
    return 0;
}

void factors_content_based
(
    FPnum *restrict a_vec, int k_sec,
    FPnum *restrict u_vec, int p,
    FPnum *restrict u_vec_sp, int u_vec_ixB[], size_t nnz_u_vec,
    FPnum *restrict C, FPnum *restrict C_bias
)
{
    if (a_vec != NULL)
    {
        cblas_tgemv(CblasRowMajor, CblasTrans,
                    p, k_sec,
                    1., C, k_sec,
                    u_vec, 1,
                    0., a_vec, 1);
    }

    else
    {
        set_to_zero(a_vec, k_sec, 1);
        sgemv_dense_sp(
            p, k_sec,
            1., C, (size_t)k_sec,
            u_vec_ixB, u_vec_sp, nnz_u_vec,
            a_vec
        );
    }

    if (C_bias != NULL)
        cblas_taxpy(k_sec, 1., C_bias, 1, a_vec, 1);
}

int matrix_content_based
(
    FPnum *restrict Am_new,
    int n_new, int k_sec,
    FPnum *restrict U, int p,
    int U_row[], int U_col[], FPnum *restrict U_sp, size_t nnz_U,
    long U_csr_p[], int U_csr_i[], FPnum *restrict U_csr,
    FPnum *restrict C, FPnum *restrict C_bias,
    int nthreads
)
{
    if (U != NULL)
    {
        cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n_new, k_sec, p,
                    1., U, p, C, k_sec,
                    0., Am_new, k_sec);
    }

    else
    {
        int retval = 0;
        long *restrict U_csr_p_use = NULL;
        int *restrict U_csr_i_use = NULL;
        FPnum *restrict U_csr_use = NULL;
        if (U_sp != NULL && U_csr == NULL) {
            retval = coo_to_csr_plus_alloc(
                        U_row, U_col, U_sp, (FPnum*)NULL,
                        n_new, p, nnz_U,
                        &U_csr_p_use, &U_csr_i_use, &U_csr_use, (FPnum**)NULL
                    );
            if (retval != 0) goto cleanup;
        }
        else {
            U_csr_p_use = U_csr_p;
            U_csr_i_use = U_csr_i;
            U_csr_use = U_csr;
        }

        set_to_zero(Am_new, (size_t)n_new*(size_t)k_sec, nthreads);
        sgemm_sp_dense(
            n_new, k_sec, 1.,
            U_csr_p_use, U_csr_i_use, U_csr_use,
            C, (size_t)k_sec,
            Am_new, (size_t)k_sec,
            nthreads
        );

        cleanup:
            if (U_csr_p_use != U_csr_p)
                free(U_csr_p_use);
            if (U_csr_i_use != U_csr_i)
                free(U_csr_i_use);
            if (U_csr_use != U_csr)
                free(U_csr_use);
        if (retval != 0) return retval;
    }

    if (C_bias != NULL)
        mat_plus_colvec(Am_new, C_bias, 1., n_new,
                        k_sec, (size_t)k_sec, nthreads);

    return 0;
}

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
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif

    FPnum *restrict Am = (FPnum*)malloc(n_new*(size_t)k_sec*sizeof(FPnum));
    FPnum *restrict Bm = (FPnum*)malloc(n_new*(size_t)k_sec*sizeof(FPnum));
    int retval = 0;
    if (Am == NULL || Bm == NULL)
    {
        retval = 1;
        goto cleanup;
    }

    retval = matrix_content_based(
        Am,
        n_new, k_sec,
        U, p,
        U_row, U_col, U_sp, nnz_U,
        U_csr_p, U_csr_i, U_csr,
        C, C_bias,
        nthreads
    );
    if (retval == 1) goto cleanup;

    retval = matrix_content_based(
        Bm,
        n_new, k_sec,
        II, q,
        I_row, I_col, I_sp, nnz_I,
        I_csr_p, I_csr_i, I_csr,
        D, D_bias,
        nthreads
    );
    if (retval == 1) goto cleanup;

    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(scores_new, n_new, k_sec, Am, Bm, glob_mean)
    for (size_t_for ix = 0; ix < (size_t)n_new; ix++)
        scores_new[ix] = cblas_tdot(k_sec, Am + ix*(size_t)k_sec, 1,
                                    Bm + ix*(size_t)k_sec, 1)
                          + glob_mean;

    cleanup:
        free(Am);
        free(Bm);
    return retval;
}

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
)
{
    int ix = 0;
    FPnum *restrict Am = (FPnum*)malloc(n_new*(size_t)k_sec*sizeof(FPnum));
    int retval = 0;
    if (Am == NULL)
    {
        retval = 1;
        goto cleanup;
    }

    retval = matrix_content_based(
        Am,
        n_new, k_sec,
        U, p,
        U_row, U_col, U_sp, nnz_U,
        U_csr_p, U_csr_i, U_csr,
        C, C_bias,
        nthreads
    );
    if (retval == 1) goto cleanup;

    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(n_new, scores_new, k_sec, Am, Bm, ixB, biasB, glob_mean)
    for (ix = 0; ix < n_new; ix++)
        scores_new[ix] = cblas_tdot(k_sec, Am + (size_t)ix*(size_t)k_sec, 1,
                                    Bm + (size_t)ixB[ix]*(size_t)k_sec, 1)
                          + ((biasB != NULL)? biasB[ixB[ix]] : 0.)
                          + glob_mean;

    cleanup:
        free(Am);
    return retval;
}

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
)
{
    int retval = 0;
    FPnum *restrict a_vec = (FPnum*)malloc((size_t)k_sec*sizeof(FPnum));
    FPnum *restrict Bm = (FPnum*)malloc((size_t)n_new * (size_t)k_sec
                                                      * sizeof(FPnum));
    FPnum *restrict scores_copy = (FPnum*)malloc((size_t)n_new*sizeof(FPnum));
    
    int *restrict buffer_ix = NULL;
    if (n_top == 0 || n_top == n_new)
        buffer_ix = rank_new;
    else
        buffer_ix = (int*)malloc((size_t)n_new*sizeof(int));

    if (a_vec == NULL || Bm == NULL || scores_copy == NULL || buffer_ix == NULL)
    {
        retval = 1;
        goto cleanup;
    }

    if (n_top == 0) n_top = n_new;

    factors_content_based(
        a_vec, k_sec,
        u_vec, p,
        u_vec_sp, u_vec_ixB, nnz_u_vec,
        C, C_bias
    );

    retval = matrix_content_based(
        Bm,
        n_new, k_sec,
        II, q,
        I_row, I_col, I_sp, nnz_I,
        I_csr_p, I_csr_i, I_csr,
        D, D_bias,
        nthreads
    );
    if (retval == 1) goto cleanup;

    cblas_tgemv(CblasRowMajor, CblasNoTrans,
                n_new, k_sec,
                1., Bm, k_sec,
                a_vec, 1,
                0., scores_copy, 1);

    for (int ix = 0; ix < n_new; ix++)
        buffer_ix[ix] = ix;

    ptr_FPnum_glob = scores_copy;
    if (n_top <= 50 || n_top >= (double)n_new*0.75)
    {
        qsort(buffer_ix, n_new, sizeof(int), cmp_argsort);
    }

    else
    {
        qs_argpartition(buffer_ix, scores_copy, n_new, n_top);
        qsort(buffer_ix, n_top, sizeof(int), cmp_argsort);
    }

    if (buffer_ix != rank_new)
        memcpy(rank_new, buffer_ix, (size_t)n_top*sizeof(int));

    if (scores_new != NULL)
        for (int ix = 0; ix < n_top; ix++)
            scores_new[ix] = scores_copy[buffer_ix[ix]] + glob_mean;

    cleanup:
        free(a_vec);
        free(Bm);
        free(scores_copy);
        if (buffer_ix != rank_new)
            free(buffer_ix);
    return retval;
}

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
)
{
    nthreads = cap_to_4(nthreads);
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long row;
    #endif

    size_t k_totA = k_sec+k+k_main;
    set_to_zero(A, (size_t)m*k_totA, nthreads);

    long *restrict U_csr_p_use = NULL;
    int *restrict U_csr_i_use = NULL;
    FPnum *restrict U_csr_use = NULL;
    int retval = 0;

    if (U != NULL)
    {
        cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, k_sec+k, p,
                    w_user, U, p, C, k_sec+k,
                    0., A, k_sec+k+k_main);
    }

    else
    {
        if (U_sp != NULL && U_csr == NULL) {
            retval = coo_to_csr_plus_alloc(
                        U_row, U_col, U_sp, (FPnum*)NULL,
                        m, p, nnz_U,
                        &U_csr_p_use, &U_csr_i_use, &U_csr_use, (FPnum**)NULL
                    );
            if (retval != 0) goto cleanup;
        }
        else {
            U_csr_p_use = U_csr_p;
            U_csr_i_use = U_csr_i;
            U_csr_use = U_csr;
        }

        sgemm_sp_dense(
            m, k_sec+k, w_user,
            U_csr_p_use, U_csr_i_use, U_csr_use,
            C, (size_t)(k_sec+k),
            A, (size_t)(k_sec+k+k_main),
            nthreads
        );
    }

    if (C_bias != NULL)
        mat_plus_colvec(A, C_bias, w_user, m, k_sec+k, k_totA, nthreads);

    cleanup:
        if (U_csr_p_use != U_csr_p)
            free(U_csr_p_use);
        if (U_csr_i_use != U_csr_i)
            free(U_csr_i_use);
        if (U_csr_use != U_csr)
            free(U_csr_use);
    return retval;
}

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
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif

    int k_totA = k_sec + k + k_main;

    long *restrict Xcsr_p_use = NULL;
    int *restrict Xcsr_i_use = NULL;
    FPnum *restrict Xcsr_use = NULL;
    FPnum *restrict weightR = NULL;

    long *restrict U_csr_p_use = NULL;
    int *restrict U_csr_i_use = NULL;
    FPnum *restrict U_csr_use = NULL;

    int retval = 0;
    int *ret = (int*)malloc(m*sizeof(int));
    if (ret == NULL) { retval = 1; goto cleanup; }

    if (X != NULL && Xfull == NULL) {
        retval = coo_to_csr_plus_alloc(
                    ixA, ixB, X, weight,
                    m, n, nnz,
                    &Xcsr_p_use, &Xcsr_i_use, &Xcsr_use, &weightR
                );
        if (retval != 0) goto cleanup;
    }
    else if (Xcsr != NULL) {
        Xcsr_p_use = Xcsr_p;
        Xcsr_i_use = Xcsr_i;
        Xcsr_use = Xcsr;
        weightR = weight;
    }

    if (U_sp != NULL && U_csr == NULL) {
        retval = coo_to_csr_plus_alloc(
                    U_row, U_col, U_sp, (FPnum*)NULL,
                    m, p, nnz_U,
                    &U_csr_p_use, &U_csr_i_use, &U_csr_use, (FPnum**)NULL
                );
        if (retval != 0) goto cleanup;
    }
    else {
        U_csr_p_use = U_csr_p;
        U_csr_i_use = U_csr_i;
        U_csr_use = U_csr;
    }

    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(A, biasA, Bm, biasB, C, C_bias, m, n, k_totA, U, p, \
                   U_csr_p_use, U_csr_i_use, U_csr_use, \
                   Xcsr_p_use, Xcsr_i_use, Xcsr_use, Xfull, weight, weightR, \
                   Bm_plus_bias, output_A, glob_mean, k, k_sec, k_main, \
                   w_user, w_main_multiplier, lam, exact, lam_bias, \
                   implicit, alpha, precomputedBtBw, precomputedBtBinvBt)
    for (size_t_for ix = 0; ix < (size_t)m; ix++)
        ret[ix] = offsets_factors_warm(
                    A + ix*(size_t)k_totA,
                    (biasA != NULL)? (biasA + ix) : ((FPnum*)NULL),
                    (U == NULL)?
                      ((FPnum*)NULL)
                      : (U + ix*(size_t)p),
                    (U_csr_i_use == NULL)?
                      ((int*)NULL)
                      : (U_csr_i_use + U_csr_p_use[ix]),
                    (U_csr_use == NULL)?
                      ((FPnum*)NULL)
                      : (U_csr_use + U_csr_p_use[ix]),
                    (U_csr_p_use == NULL)?
                      (0) : (U_csr_p_use[ix+1] - U_csr_p_use[ix]),
                    (Xcsr_i_use == NULL)?
                      ((int*)NULL)
                      : (Xcsr_i_use + Xcsr_p_use[ix]),
                    (Xcsr_use == NULL)?
                      ((FPnum*)NULL)
                      : (Xcsr_use + Xcsr_p_use[ix]),
                    (Xcsr_p_use == NULL)?
                      (0) : (Xcsr_p_use[ix+1] - Xcsr_p_use[ix]),
                    (Xfull == NULL)? ((FPnum*)NULL) : (Xfull + ix*(size_t)n),
                    n,
                    (weight == NULL)?
                      ((FPnum*)NULL)
                      : ( (weightR != NULL)?
                            (weightR + Xcsr_p_use[ix])
                            : (weight + ix*(size_t)n) ),
                    Bm, C,
                    C_bias,
                    glob_mean, biasB,
                    k, k_sec, k_main,
                    p, w_user,
                    lam, exact, lam_bias,
                    implicit, alpha,
                    w_main_multiplier,
                    precomputedBtBinvBt,
                    precomputedBtBw,
                    (output_A == NULL)?
                      ((FPnum*)NULL)
                      : (output_A + ix*(size_t)(k+k_main)),
                    Bm_plus_bias
                );

    cleanup:
        free(ret);
        if (weightR != weight) free(weightR);
        if (Xcsr_p_use != Xcsr_p) free(Xcsr_p_use);
        if (Xcsr_i_use != Xcsr_i) free(Xcsr_i_use);
        if (Xcsr_use != Xcsr) free(Xcsr_use);
        if (U_csr_p_use != U_csr_p) free(U_csr_p_use);
        if (U_csr_i_use != U_csr_i) free(U_csr_i_use);
        if (U_csr_use != U_csr) free(U_csr_use);
    return retval;
}
