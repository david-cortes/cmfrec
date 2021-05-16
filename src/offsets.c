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

/*******************************************************************************
    Offsets Model
    -------------

    This is the model optimized for cold-start recommendations decribed in:
        Cortes, David.
        "Cold-start recommendations in Collective Matrix Factorization."
        arXiv preprint_t arXiv:1809.00366 (2018).


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
    - '.' denotes element-wise multiplication
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
        min || M . (X - A*t(B)) ||^2

    As some small improvements, the matrix 'X' is centered by substracting the
    mean from it, and additionally subtracting row and column biases, which
    are model parameters too, while imposing a regularization penalty on the
    magnitude of the parameters (given by L2 norm):
        min ||M . (X - A*t(B) - mu[1] - b1[m,1] - b2[1,n])||^2
            + lambda*(||A||^2 + ||B||^2 + ||b1||^2 + ||b2||^2)

    The intended purpose is to use this as a recommender system model, in which
    'X' is a matrix comprising the ratings that users give to items, with each
    row corresponding to a user, each column to an item, and each non-missing
    entry to the observed rating or explicit evaluation from the user.

    For the case of recommender systems, there is also the so-called
    'implicit-feedback' model, in which the entries of 'X' are assumed to all
    be zeros or ones (i.e. the matrix is full with no missing values), but with
    a weight given by the actual values and a confidence score multiplier:
        min ||sqrt(alpha*X + 1) . (M - A*t(B))||^2 + lambda*(||A||^2 + ||B||^2)

    =======================     END OF COMMON PART     =========================

    This idea however (a) does not take into account any side information
    (attributes) about the users and/or items, (b) is slow to compute the latent
    factors for new users which were not in the training data.

    The model here extends it in such a way that the factorizing matrices are
    obtained by a linear combination of the user/item attributes U[m,p], I[n,q],
    plus a free offset which is not dependent on the attributes, i.e.
        Am = A + U*C
        Bm = B + I*D
        min ||M . (X - Am*t(Bm))||^2

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
        E[m,n]  = M . W . (Am*t(Bm) - X - b1 - b2)
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
    being multiplied by U and w_user. See the file 'collective.c' for an
    explanation of how the biases are obtained.

    As can be seen from the formulas, calculating the factors for new users is
    easier for both the cold-start (based only on user attributes Ua[1,p]) and
    the warm-start case (based on both user attributes Ua[1,p] and on
    ratings Xa[1,n]).

    For the cold-start case, they can be obtained as:
        Am = [w_user*t(C)*t(Xa), 0[k_sec]]

    Note however that, for the implicit-feedback model, the expected values of
    the 'A' matrix are not all zeros like in the explicit feedback case, so
    it might make more sense to add to Am an average by column of 'A'.

    For the warm-start case, a similar closed-form solution can be taken, but
    note that when k_sec > 0, it's necessary to calculate Xdiff and solve for
    it instead.

    As yet another alternative, it's possible to fit a purely content-based
    model with the same logic if one sets 'k' and 'k_main' to zero here while
    using 'k_sec' greater than zero.

    
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
)
{
    int_t k_totA = k_sec + k + k_main;
    int_t k_totB = k_totA;
    int_t k_szA = k + k_main;
    int_t k_szB = k_szA;
    size_t dimC = (size_t)(k_sec + k) * (size_t)p;
    size_t dimD = (size_t)(k_sec + k) * (size_t)q;

    bool has_U = (U  != NULL || U_csr != NULL);
    bool has_I = (II != NULL || I_csr != NULL);

    if (!has_U) k_szA = k_totA;
    if (!has_I) k_szB = k_totB;

    real_t f = 0;

    size_t nvars = (size_t)m * (size_t)k_szA + (size_t)n * (size_t)k_szB;
    if (user_bias) nvars += m;
    if (item_bias) nvars += n;
    if (has_U) nvars += dimC;
    if (has_I) nvars += dimD;
    if (add_intercepts && has_U) nvars += (size_t)(k_sec + k);
    if (add_intercepts && has_I) nvars += (size_t)(k_sec + k);
    set_to_zero_(grad, nvars, nthreads);

    real_t *restrict biasA = values;
    real_t *restrict biasB = biasA + (user_bias? m : 0);
    real_t *restrict A = biasB + (item_bias? n : 0);
    real_t *restrict B = A + (size_t)m * (size_t)k_szA;
    real_t *restrict C = B + (size_t)n * (size_t)k_szB;
    real_t *restrict C_bias = C + (has_U? dimC : (size_t)0);
    real_t *restrict D = C_bias + ((add_intercepts && has_U)?
                                  (size_t)(k_sec+k) : (size_t)0);
    real_t *restrict D_bias = D + (has_I? dimD : (size_t)0);

    real_t *restrict g_biasA = grad;
    real_t *restrict g_biasB = g_biasA + (user_bias? m : 0);
    real_t *restrict g_A = g_biasB + (item_bias? n : 0);
    real_t *restrict g_B = g_A + (size_t)m * (size_t)k_szA;
    real_t *restrict g_C = g_B + (size_t)n * (size_t)k_szB;
    real_t *restrict g_C_bias = g_C + (has_U? dimC : (size_t)0);
    real_t *restrict g_D = g_C_bias + ((add_intercepts && has_U)?
                                      (size_t)(k_sec+k) : (size_t)0);
    real_t *restrict g_D_bias = g_D + (has_I? dimD : (size_t)0);

    real_t *restrict Am = buffer_real_t;
    real_t *restrict Bm = Am + ((has_U && (k_sec || k))?
                               ((size_t)m*(size_t)k_totA) : (0));
    real_t *restrict bufferA = Bm + ((has_I && (k_sec || k))?
                                    ((size_t)n*(size_t)k_totB) : 0);
    real_t *restrict bufferB = bufferA
                                + (((k_main || k_sec) && has_U)?
                                   ((size_t)m * (size_t)k_totA) : (0));
    real_t *restrict buffer_remainder = bufferB
                                        + (((k_main || k_sec) && has_I)?
                                           ((size_t)n * (size_t)k_totB) : (0));
    if ((k_main || k_sec) && (has_U || has_I))
    {
        if (has_U && has_I) {
            set_to_zero_(bufferA,
                         (size_t)m*(size_t)k_totA + (size_t)n*(size_t)k_totB,
                         nthreads);
        } else if (has_U && !has_I) {
            set_to_zero_(bufferA, (size_t)m*(size_t)k_totA, nthreads);
            bufferB = g_B;
        } else if (!has_U && has_I) {
            set_to_zero_(bufferB, (size_t)n*(size_t)k_totB, nthreads);
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
        taxpy_large(values, lam, grad, nvars, nthreads);
        f += (lam / 2.) * sum_squares(values, nvars, nthreads);
    }

    else {
        long double freg = 0;
        if (user_bias) cblas_taxpy(m, lam_unique[0], biasA, 1, g_biasA, 1);
        if (item_bias) cblas_taxpy(n, lam_unique[1], biasB, 1, g_biasB, 1);
        taxpy_large(A, lam_unique[2], g_A, (size_t)m*(size_t)k_szA, nthreads);
        taxpy_large(B, lam_unique[3], g_B, (size_t)n*(size_t)k_szB, nthreads);

        if (has_U)
            taxpy_large(C, lam_unique[4], g_C,
                        (size_t)(p + (int)add_intercepts)*(size_t)(k_sec+k),
                        nthreads);
        if (has_I)
            taxpy_large(D, lam_unique[5], g_D,
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
        f += (real_t)freg;
    }

    return (real_t) f;
}

void construct_Am
(
    real_t *restrict Am, real_t *restrict A,
    real_t *restrict C, real_t *restrict C_bias,
    bool add_intercepts,
    real_t *restrict U, int_t m, int_t p,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    int_t k, int_t k_sec, int_t k_main,
    real_t w_user, int nthreads
)
{
    /* Matrix dimensions are:
       - A[m, k+k_main]
       - C[p, k_sec+k]
       - C_bias[k_sec+k]
       - Am[m, k_sec+k+k_main] */

    size_t k_totA = k_sec + k + k_main;
    size_t k_szA = ((U == NULL && U_csr_p == NULL)? k_sec : 0) + k + k_main;

    if (k_sec == 0 && k_main == 0) {
        copy_arr_(A, Am, (size_t)m*k_totA, nthreads);
    } else {
        /* Am[:,:] = 0; Am[:,k_sec:] = A[:,:] */
        set_to_zero_(Am, (size_t)m*k_totA, nthreads);
        copy_mat(m, k+k_main, A, k_szA, Am + k_sec, k_totA);
    }

    /* Am[:,:k_sec+k] += U * C */
    if (U != NULL)
        cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, k_sec+k, p,
                    w_user, U, p, C, k_sec+k,
                    1., Am, k_totA);
    else if (U_csr_p != NULL)
        tgemm_sp_dense(m, k+k_sec, w_user,
                       U_csr_p, U_csr_i, U_csr,
                       C, k_sec+k,
                       Am, k_totA,
                       nthreads);

    if (add_intercepts)
        mat_plus_colvec(Am, C_bias, w_user, m, k_sec+k, k_totA, nthreads);
}

void assign_gradients
(
    real_t *restrict bufferA, real_t *restrict g_A, real_t *restrict g_C,
    bool add_intercepts, real_t *restrict g_C_bias,
    real_t *restrict U,
    size_t U_csc_p[], int_t U_csc_i[], real_t *restrict U_csc,
    int_t m, int_t p, int_t k, int_t k_sec, int_t k_main,
    real_t w_user, int nthreads
)
{
    if (U != NULL && (k || k_sec))
        cblas_tgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    p, k_sec+k, m,
                    w_user, U, p, bufferA, k_sec+k+k_main,
                    0., g_C, k_sec+k);
    else if (U_csc != NULL && (k || k_sec))
        tgemm_sp_dense(p, k_sec+k, w_user,
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

int_t offsets_factors_cold
(
    real_t *restrict a_vec,
    real_t *restrict u_vec,
    int_t u_vec_ixB[], real_t *restrict u_vec_sp, size_t nnz_u_vec,
    real_t *restrict C, int_t p,
    real_t *restrict C_bias,
    int_t k, int_t k_sec, int_t k_main,
    real_t w_user
)
{
    /* a_vec[:k_sec+k] := t(C)*u_vec
       a_vec[k_sec+k:] := 0 */
    int_t k_pred = k_sec + k;
    if (u_vec_sp != NULL)
        set_to_zero(a_vec, k_sec+k+k_main);
    else if (k_main != 0)
        set_to_zero(a_vec + (k_sec+k), k_main);

    if (u_vec != NULL)
        cblas_tgemv(CblasRowMajor, CblasTrans,
                    p, k_sec+k,
                    w_user, C, k_sec+k,
                    u_vec, 1,
                    0., a_vec, 1);
    else {
        tgemv_dense_sp(p, k_pred,
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
    real_t *restrict precomputedBtB,
    real_t *restrict output_a,
    real_t *restrict Bm_plus_bias
)
{
    /* TODO: add functionality for obtaining an exact solution through
       the L-BFGS solver for a single row of A, like it was done for
       the collective model. */
    /* TODO: do something about 'NA_as_zero' */
    int_t retval = 0;
    real_t *restrict buffer_real_t = NULL;
    size_t size_buffer = 0;
    int_t cnt_NA = 0;
    bool append_bias = (Bm_plus_bias != NULL && a_bias != NULL);

    real_t *restrict bufferX = NULL;
    real_t *restrict bufferW = NULL;
    real_t *restrict buffer_uc = NULL;
    real_t *restrict buffer_remainder = NULL;
    real_t *restrict bufferBtB = NULL;

    if (!implicit)
        preprocess_vec(Xa_dense, n, ixB, Xa, nnz,
                       glob_mean, lam_bias, biasB,
                       (Bm_plus_bias == NULL)? a_bias : (real_t*)NULL,
                       &cnt_NA);
    else if (alpha != 1.) {
        if (Xa != NULL)
            tscal_large(Xa, alpha, nnz, 1);
        else
            cblas_tscal(n, alpha, Xa_dense, 1); /* should not be reached */
    }

    real_t *restrict a_plus_bias = NULL;
    if (append_bias) {
        a_plus_bias = (real_t*)malloc((size_t)(k_sec+k+k_main+1)
                                       * sizeof(real_t));
        if (a_plus_bias == NULL) goto throw_oom;
    }

    if ((!exact && k_sec == 0) || implicit)
    {
        if (precomputedTransBtBinvBt == NULL || Xa_dense == NULL ||
            cnt_NA > 0 || weight != NULL || implicit)
        {
            size_buffer = square(k_sec + k + k_main + append_bias);
            buffer_real_t = (real_t*)malloc(size_buffer*sizeof(real_t));
            if (buffer_real_t == NULL) goto throw_oom;
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
                                    buffer_real_t,
                                    lam, lam, 0., 0., false, false, 0.,
                                    precomputedTransBtBinvBt,
                                    precomputedBtB, cnt_NA, k_sec+k+k_main,
                                    false, false, 1., n,
                                    (real_t*)NULL, false,
                                    false, 0, false, 0,
                                    (real_t*)NULL, (real_t*)NULL, 0., 1.,false);
            else {
                factors_closed_form(a_plus_bias, k_sec+k+k_main+1,
                                    Bm_plus_bias, n, k_sec+k+k_main+1,
                                    Xa_dense, cnt_NA==0,
                                    Xa, ixB, nnz,
                                    weight,
                                    buffer_real_t,
                                    lam, lam_bias, 0., 0., false, false, 0.,
                                    precomputedTransBtBinvBt,
                                    precomputedBtB, cnt_NA, k_sec+k+k_main+1,
                                    false, false, 1., n,
                                    (real_t*)NULL, false,
                                    false, 0, false, 0,
                                    (real_t*)NULL, (real_t*)NULL, 0., 1.,false);
                memcpy(a_vec, a_plus_bias,
                       (size_t)(k_sec+k+k_main)*sizeof(real_t));
                *a_bias = a_plus_bias[k_sec+k+k_main];
            }
        }
        else {

            if (precomputedBtB == NULL) {
                bufferBtB = (real_t*)malloc((size_t)square(k_sec+k+k_main)
                                             * sizeof(double));
                if (bufferBtB == NULL) goto throw_oom;
                cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                            k_sec+k+k_main, n,
                            1., Bm, k_sec+k+k_main,
                            0., bufferBtB, k_sec+k+k_main);
                precomputedBtB = bufferBtB;
            }
            set_to_zero(a_vec, k_sec+k+k_main);

            factors_implicit_chol(
                a_vec, k_sec+k+k_main,
                Bm, k_sec+k+k_main,
                Xa, ixB, nnz,
                lam, 0.,
                precomputedBtB, k_sec+k+k_main,
                false, 0,
                buffer_real_t
            );
        }

        /* If A is required instead of just Am, then calculate as
             A := Am - U*C */
        if (output_a != NULL)
        {
            offsets_factors_cold(output_a, u_vec,
                                 u_vec_ixB, u_vec_sp, nnz_u_vec,
                                 C, p,
                                 C_bias,
                                 k, k_sec, k_main, w_user);
            for (int_t ix = k_sec; ix < k_sec+k+k_main; ix++)
                output_a[ix - k_sec] -= w_user * a_vec[ix];
        }

    }

    else
    {
        size_buffer =   (size_t)square(k_sec+k+k_main+append_bias)
                      + (size_t)n
                      + (size_t)(k_sec+k);
        if (weight != NULL && Xa_dense == NULL)
            size_buffer += n;
        buffer_real_t = (real_t*)malloc(size_buffer*sizeof(real_t));
        if (buffer_real_t == NULL) goto throw_oom;

        bufferX = buffer_real_t;
        bufferW = bufferX + n;
        buffer_uc = bufferW + ((weight != NULL && Xa_dense == NULL)?
                                (n) : (0));
        buffer_remainder = buffer_uc + k_sec+k;

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
            set_to_zero(a_vec, k_sec+k+k_main);

        /* buffer_uc = w*U*C (save for later) */
        memcpy(buffer_uc, a_vec, (size_t)(k_sec+k)*sizeof(real_t));
        /* BufferX = -w*U*C*t(Bm) */
        if (u_vec != NULL || u_vec_sp != NULL)
            cblas_tgemv(CblasRowMajor, CblasNoTrans,
                        n, k_sec+k,
                        -1., Bm, k_sec+k+k_main,
                        a_vec, 1,
                        0., bufferX, 1);
        else
            set_to_zero(bufferX, n);

        /* BufferX += X */
        if (Xa_dense == NULL)
        {
            for (size_t ix = 0; ix < nnz; ix++)
                bufferX[ixB[ix]] += Xa[ix];
            if (weight != NULL) {
                for (int_t ix = 0; ix < n; ix++) bufferW[ix] = 1.;
                for (size_t ix = 0; ix < nnz; ix++)
                    bufferW[ixB[ix]] = weight[ix];
                weight = bufferW;
            }
        }
        else
        {
            for (int_t ix = 0; ix < n; ix++)
                bufferX[ix] += isnan(Xa_dense[ix])? 0 : Xa_dense[ix];
        }

        /* Solve lsq(A, X' - w*U*C*t(Bm)) */
        if (k || k_main) {
            if (!append_bias)
                factors_closed_form(a_vec + k_sec, k+k_main,
                                    Bm + k_sec, n, k_sec+k+k_main,
                                    bufferX, true,
                                    (real_t*)NULL, (int_t*)NULL, (size_t)0,
                                    weight,
                                    buffer_remainder,
                                    lam, lam, 0., 0., false, false, 0.,
                                    (real_t*)NULL,
                                    (real_t*)NULL, 0, 0,
                                    false, false, 1., n,
                                    (real_t*)NULL, false,
                                    false, 0, false, 0,
                                    (real_t*)NULL, (real_t*)NULL, 0., 1.,false);
            else {
                factors_closed_form(a_plus_bias + k_sec, k+k_main+1,
                                    Bm_plus_bias + k_sec, n, k_sec+k+k_main+1,
                                    bufferX, true,
                                    (real_t*)NULL, (int_t*)NULL, (size_t)0,
                                    weight,
                                    buffer_remainder,
                                    lam, lam_bias, 0., 0., false, false, 0.,
                                    (real_t*)NULL,
                                    (real_t*)NULL, 0, 0,
                                    false, false, 1., n,
                                    (real_t*)NULL, false,
                                    false, 0, false, 0,
                                    (real_t*)NULL, (real_t*)NULL, 0., 1.,false);
                memcpy(a_vec + k_sec, a_plus_bias + k_sec,
                       (size_t)(k+k_main)*sizeof(real_t));
                *a_bias = a_plus_bias[k_sec+k+k_main];
            }
        }
        /* Save A (as opposed to Am) */
        if (output_a != NULL && (k || k_main))
            memcpy(output_a, a_vec + k_sec, (size_t)(k+k_main)*sizeof(real_t));

        /* Am = A + w*U*C */
        for (int_t ix = 0; ix < k_sec+k; ix++)
            a_vec[ix] += buffer_uc[ix];
    }

    cleanup:
        free(buffer_real_t);
        free(a_plus_bias);
        free(bufferBtB);
        return retval;
    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}

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
)
{
    int_t retval = 0;
    bool free_U_csr = false;
    bool free_I_csr = false;
    if (nnz_U && U_csr_p == NULL)
    {
        free_U_csr = true;
        U_csr_p = (size_t*)malloc(((size_t)m+(size_t)1)*sizeof(size_t));
        U_csr_i = (int_t*)malloc(nnz_U*sizeof(int_t));
        U_csr = (real_t*)malloc(nnz_U*sizeof(real_t));
        if (U_csr_p == NULL || U_csr_i == NULL || U_csr == NULL)
            goto throw_oom;
        coo_to_csr(
            U_row, U_col, U_sp,
            (real_t*)NULL,
            m, p, nnz_U,
            U_csr_p, U_csr_i, U_csr,
            (real_t*)NULL
        );
    }

    if (nnz_I && I_csr_p == NULL)
    {
        free_I_csr = true;
        I_csr_p = (size_t*)malloc(((size_t)n+(size_t)1)*sizeof(size_t));
        I_csr_i = (int_t*)malloc(nnz_I*sizeof(int_t));
        I_csr = (real_t*)malloc(nnz_I*sizeof(real_t));
        if (I_csr_p == NULL || I_csr_i == NULL || I_csr == NULL)
            goto throw_oom;
        coo_to_csr(
            I_row, I_col, I_sp,
            (real_t*)NULL,
            n, q, nnz_I,
            I_csr_p, I_csr_i, I_csr,
            (real_t*)NULL
        );
    }

    if (U  != NULL || U_csr_p != NULL)
        construct_Am(
            Am, A,
            C, C_bias,
            add_intercepts,
            U, m, p,
            U_csr_p, U_csr_i, U_csr,
            k, k_sec, k_main,
            w_user, 1
        );
    else
        Am = A;

    if (II != NULL || I_csr_p != NULL)
        construct_Am(
            Bm, B,
            D, D_bias,
            add_intercepts,
            II, n, q,
            I_csr_p, I_csr_i, I_csr,
            k, k_sec, k_main,
            w_item, 1
        );
    else
        Bm = B;

    if (!implicit)
        retval = precompute_collective_explicit(
            Bm, n, 0, 0,
            (real_t*)NULL, 0,
            (real_t*)NULL, false,
            biasB, glob_mean, NA_as_zero_X,
            (real_t*)NULL, false,
            k_sec+k+k_main, 0, 0, 0,
            user_bias,
            false,
            lam, lam_unique,
            false, false,
            false, 0.,
            1., 1., 1.,
            Bm_plus_bias,
            BtB,
            TransBtBinvBt,
            (real_t*)NULL,
            (real_t*)NULL,
            (real_t*)NULL,
            (real_t*)NULL,
            (real_t*)NULL,
            (real_t*)NULL
        );
    else
        retval = precompute_collective_implicit(
            Bm, n,
            (real_t*)NULL, 0,
            (real_t*)NULL, false,
            k, 0, 0, 0, /* <- cannot have 'k_sec' or 'k_main' with 'implicit' */
            lam, 1., 1., 1., /* <- cannot have different 'lambda' either */
            false,
            true,
            BtB,
            (real_t*)NULL,
            (real_t*)NULL,
            (real_t*)NULL
        );
    if (retval == 1) goto throw_oom;

    cleanup:
        if (free_U_csr) {
            free(U_csr);
            free(U_csr_i);
            free(U_csr_p);
        }
        if (free_I_csr) {
            free(I_csr);
            free(I_csr_i);
            free(I_csr_p);
        }
        return retval;

    throw_oom:
    {
        retval = 1;
        print_oom_message();
        goto cleanup;
    }
}

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
)
{
    return precompute_offsets_both(
        A, m,
        B, n,
        C, p,
        D, q,
        C_bias, D_bias,
        (real_t*)NULL, 0., false,
        user_bias, add_intercepts, false,
        k, k_main, k_sec,
        lam, lam_unique,
        w_user, w_item, 
        U,
        U_row, U_col, U_sp, nnz_U,
        U_csr_p, U_csr_i, U_csr,
        II,
        I_csr_p, I_csr_i, I_csr,
        I_row, I_col, I_sp, nnz_I,
        Am,
        Bm,
        Bm_plus_bias,
        BtB,
        TransBtBinvBt
    );
}

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
)
{
    return precompute_offsets_both(
        A, m,
        B, n,
        C, p,
        D, q,
        C_bias, D_bias,
        (real_t*)NULL, 0., false,
        false, add_intercepts, true,
        k, 0, 0,
        lam, (real_t*)NULL,
        1., 1., 
        U,
        U_row, U_col, U_sp, nnz_U,
        U_csr_p, U_csr_i, U_csr,
        II,
        I_csr_p, I_csr_i, I_csr,
        I_row, I_col, I_sp, nnz_I,
        Am,
        Bm,
        (real_t*)NULL,
        BtB,
        (real_t*)NULL
    );
}

real_t wrapper_offsets_fun_grad
(
    void *instance,
    real_t *x,
    real_t *g,
    const size_t n,
    const real_t step
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
        data->buffer_real_t,
        data->buffer_mt
    );
}

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
)
{
    if (k_sec > 0 && U == NULL && !nnz_U && II == NULL && !nnz_I)
    {
        if (verbose) {
            fprintf(stderr, "Cannot pass 'k_sec' without 'U' or 'I'.\n");
            #ifndef _FOR_R
            fflush(stderr);
            #endif
        }
        return 2;
    }

    real_t *restrict buffer_real_t = NULL;
    real_t *restrict buffer_mt = NULL;
    int_t retval = 0;
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
        buffer_real_t = (real_t*)malloc(size_buffer*sizeof(real_t));
        if (buffer_real_t == NULL) return 1;
    }

    real_t *restrict biasA = values;
    real_t *restrict biasB = biasA + (user_bias? m : 0);
    real_t *restrict A = biasB + (item_bias? n : 0);
    real_t *restrict B = A + ((size_t)m
                             * (size_t)((U != NULL  || U_row != NULL)?
                                        (k+k_main) : (k_sec+k+k_main)));
    real_t *restrict C = B + ((size_t)n
                             * (size_t)((II != NULL || I_row != NULL)?
                                        (k+k_main) : (k_sec+k+k_main)));
    real_t *restrict C_bias = C + ((U != NULL || U_sp != NULL)?
                                  ((size_t)p * (size_t)(k_sec+k)) : (0));
    real_t *restrict D = C_bias + (
                                (add_intercepts && (U != NULL || U_sp != NULL))?
                                (size_t)(k_sec+k) : (size_t)0
                                );
    real_t *restrict D_bias = D + ((II != NULL || I_sp != NULL)?
                                  ((size_t)q * (size_t)(k_sec+k)) : (0));

    lbfgs_parameter_t lbfgs_params;
    data_offsets_fun_grad data;
    real_t funval;

    size_t *Xcsr_p = NULL;
    int_t *Xcsr_i = NULL;
    real_t *restrict Xcsr = NULL;
    real_t *restrict weightR = NULL;
    size_t *Xcsc_p = NULL;
    int_t *Xcsc_i = NULL;
    real_t *restrict Xcsc = NULL;
    real_t *restrict weightC = NULL;
    size_t *U_csr_p = NULL;
    int_t *U_csr_i = NULL;
    real_t *restrict U_csr = NULL;
    size_t *U_csc_p = NULL;
    int_t *U_csc_i = NULL;
    real_t *restrict U_csc = NULL;
    size_t *I_csr_p = NULL;
    int_t *I_csr_i = NULL;
    real_t *restrict I_csr = NULL;
    size_t *I_csc_p = NULL;
    int_t *I_csc_i = NULL;
    real_t *restrict I_csc = NULL;

    bool full_dense = false;
    if (Xfull != NULL)
        full_dense = (count_NAs(Xfull, (size_t)m*(size_t)n, nthreads)) == 0;

    sig_t_ old_interrupt_handle = NULL;
    bool has_lock_on_handle = false;
    #pragma omp critical
    {
        if (!handle_is_locked)
        {
            handle_is_locked = true;
            has_lock_on_handle = true;
            should_stop_procedure = false;
            old_interrupt_handle = signal(SIGINT, set_interrup_global_variable);
        }
    }
    
    #ifdef _OPENMP
    if (nthreads > 1 && Xfull == NULL)
    {
        if (prefer_onepass)
        {
            size_t size_mt = (size_t)(m + n) * (size_t)(k_sec + k + k_main);
            if (user_bias) size_mt += (size_t)m;
            if (item_bias) size_mt += (size_t)n;
            buffer_mt = (real_t*)malloc((size_t)nthreads * size_mt
                                         * sizeof(real_t));
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
            (real_t*)NULL, m, p,
            U_row, U_col, U_sp, nnz_U,
            (real_t*)NULL, (real_t**)NULL,
            &U_csr_p, &U_csr_i, &U_csr,
            &U_csc_p, &U_csc_i, &U_csc,
            (int_t**)NULL, (int_t**)NULL,
            (bool*)NULL, (bool*)NULL, (bool*)NULL,
            false, false, nthreads
        );
        if (retval != 0) goto cleanup;
    }

    if (I_sp != NULL)
    {
        retval = preprocess_sideinfo_matrix(
            (real_t*)NULL, n, q,
            I_row, I_col, I_sp, nnz_I,
            (real_t*)NULL, (real_t**)NULL,
            &I_csr_p, &I_csr_i, &I_csr,
            &I_csc_p, &I_csc_i, &I_csc,
            (int_t**)NULL, (int_t**)NULL,
            (bool*)NULL, (bool*)NULL, (bool*)NULL,
            false, false, nthreads
        );
        if (retval != 0) goto cleanup;
    }

    *glob_mean = 0;
    retval = initialize_biases(
        glob_mean, values, values + (user_bias? m : 0),
        user_bias, item_bias, center,
        (lam_unique == NULL)? (lam) : (lam_unique[0]),
        (lam_unique == NULL)? (lam) : (lam_unique[1]),
        false, false,
        false, false,
        (real_t*)NULL, (real_t*)NULL,
        m, n,
        m, n,
        ixA, ixB, X, nnz,
        Xfull, (real_t*)NULL,
        Xcsr_p, Xcsr_i, Xcsr,
        Xcsc_p, Xcsc_i, Xcsc,
        weight, (real_t*)NULL,
        weightR, weightC,
        false,
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
        buffer_real_t,
        buffer_mt,
        print_every, 0, 0
    };

    if (should_stop_procedure)
    {
        fprintf(stderr, "Procedure terminated before starting optimization\n");
        #if !defined(_FOR_R)
        fflush(stderr);
        #endif
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
        (real_t*)NULL,
        (iteration_data_t*)NULL
    );
    if (verbose) {
        printf("\n\nOptimization terminated\n");
        printf("\t%s\n", lbfgs_strerror(retval));
        printf("\tniter:%3d, nfev:%3d\n", data.niter, data.nfev);
        #if !defined(_FOR_R)
        fflush(stdout);
        #endif
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
        else if (Am != NULL)
            copy_arr_(A, Am, (size_t)m*(size_t)(k_sec+k+k_main), nthreads);

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
        else if (Bm != NULL)
            copy_arr_(B, Bm, (size_t)n*(size_t)(k_sec+k+k_main), nthreads);

        if (Bm_plus_bias != NULL && user_bias && Bm != NULL)
            append_ones_last_col(
                Bm,
                n, k_sec+k+k_main,
                Bm_plus_bias
            );
    }

    cleanup:
        free(buffer_real_t);
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
        #pragma omp critical
        {
            if (has_lock_on_handle && handle_is_locked)
            {
                handle_is_locked = false;
                signal(SIGINT, old_interrupt_handle);
            }
            if (should_stop_procedure)
            {
                act_on_interrupt(3, handle_interrupt, true);
                if (retval != 1) retval = 3;
            }
        }
    if (retval == 1)
    {
        if (verbose)
            print_oom_message();
    }
    return retval;
}

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
)
{
    int_t retval = 0;
    size_t edge = 0;
    real_t *restrict values = NULL;

    int_t k_totA = k_sec + k + k_main;
    int_t k_totB = k_totA;
    int_t k_szA = k + k_main;
    int_t k_szB = k_szA;
    size_t dimC = (size_t)(k_sec + k) * (size_t)p;
    size_t dimD = (size_t)(k_sec + k) * (size_t)q;

    bool has_U = (U  != NULL || nnz_U);
    bool has_I = (II != NULL || nnz_I);

    if (!has_U) k_szA = k_totA;
    if (!has_I) k_szB = k_totB;

    size_t nvars = (size_t)m * (size_t)k_szA + (size_t)n * (size_t)k_szB;
    if (user_bias) nvars += m;
    if (item_bias) nvars += n;
    if (has_U) nvars += dimC;
    if (has_I) nvars += dimD;
    if (add_intercepts && has_U) nvars += (size_t)(k_sec + k);
    if (add_intercepts && has_I) nvars += (size_t)(k_sec + k);
    if (!has_U) k_szA = k_totA;
    if (!has_I) k_szB = k_totB;

    values = (real_t*)malloc(nvars*sizeof(real_t));
    if (values == NULL) goto throw_oom;

    if (!reset_values)
    {
        edge = 0;
        if (user_bias) {
            copy_arr(biasA, values + edge, m);
            edge += m;
        }
        if (item_bias) {
            copy_arr(biasB, values + edge, n);
            edge += n;
        }
        copy_arr_(A, values + edge, (size_t)m*k_szA, nthreads);
        edge += (size_t)m*k_szA;
        copy_arr_(B, values + edge, (size_t)n*k_szB, nthreads);
        edge += (size_t)n*k_szB;
        if (p) {
            copy_arr_(C, values + edge, (size_t)p*(size_t)(k_sec+k), nthreads);
            edge += (size_t)p*(size_t)(k_sec+k);
            if (add_intercepts) {
                copy_arr(C_bias, values + edge, k_sec+k);
                edge += k_sec+k;
            }
        }
        if (q) {
            copy_arr_(D, values + edge, (size_t)q*(size_t)(k_sec+k), nthreads);
            edge += (size_t)q*(size_t)(k_sec+k);
            if (add_intercepts) {
                copy_arr(D_bias, values + edge, k_sec+k);
                edge += k_sec+k;
            }
        }
    }

    retval = fit_offsets_explicit_lbfgs_internal(
        values, reset_values,
        glob_mean,
        m, n, k,
        ixA, ixB, X, nnz,
        Xfull,
        weight,
        user_bias, item_bias, center,
        add_intercepts,
        lam, lam_unique,
        U, p,
        II, q,
        U_row, U_col, U_sp, nnz_U,
        I_row, I_col, I_sp, nnz_I,
        k_main, k_sec,
        w_user, w_item,
        n_corr_pairs, maxiter, seed,
        nthreads, prefer_onepass,
        verbose, print_every, true,
        niter, nfev,
        Am, Bm,
        Bm_plus_bias
    );
    if ((retval != 0 && retval != 3) || (retval == 3 && !handle_interrupt))
        goto cleanup;

    if (true)
    {
        edge = 0;
        if (user_bias) {
            copy_arr(values + edge, biasA, m);
            edge += m;
        }
        if (item_bias) {
            copy_arr(values + edge, biasB, n);
            edge += n;
        }
        copy_arr_(values + edge, A, (size_t)m*k_szA, nthreads);
        edge += (size_t)m*k_szA;
        copy_arr_(values + edge, B, (size_t)n*k_szB, nthreads);
        edge += (size_t)n*k_szB;
        if (p) {
            copy_arr_(values + edge, C, (size_t)p*(size_t)(k_sec+k), nthreads);
            edge += (size_t)p*(size_t)(k_sec+k);
            if (add_intercepts) {
                copy_arr(values + edge, C_bias, k_sec+k);
                edge += k_sec+k;
            }
        }
        if (q) {
            copy_arr_(values + edge, D, (size_t)q*(size_t)(k_sec+k), nthreads);
            edge += (size_t)q*(size_t)(k_sec+k);
            if (add_intercepts) {
                copy_arr(values + edge, D_bias, k_sec+k);
                edge += k_sec+k;
            }
        }
        free(values); values = NULL;
    }

    if (precompute_for_predictions)
    {
        #pragma omp critical
        {
            if (retval == 3) should_stop_procedure = true;
        }
        retval = precompute_collective_explicit(
            Bm, n, n, true,
            (real_t*)NULL, 0,
            (real_t*)NULL, false,
            biasB, *glob_mean, false,
            (real_t*)NULL, false,
            k_sec+k+k_main, 0, 0, 0,
            user_bias,
            false,
            lam, lam_unique,
            false, false,
            false, 0.,
            1., 1., 1.,
            Bm_plus_bias,
            precomputedBtB,
            precomputedTransBtBinvBt,
            (real_t*)NULL,
            (real_t*)NULL,
            (real_t*)NULL,
            (real_t*)NULL,
            (real_t*)NULL,
            (real_t*)NULL
        );
        #pragma omp critical
        {
            if (should_stop_procedure && retval == 0) {
                retval = 3;
            }
        }
    }

    cleanup:
        free(values);
        act_on_interrupt(retval, handle_interrupt, false);
        return retval;
    throw_oom:
    {
        if (verbose)
            print_oom_message();
        retval = 1;
        goto cleanup;
    }
}

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
)
{
    int_t retval = 0;
    if (p > m || q > n || k > m || k > n) {
        if (verbose) {
            if (k > m || k > n)
                fprintf(stderr, "'k' cannot be greater than 'm' or 'n'.\n");
            else
                fprintf(stderr, "Side info has larger dimension than 'X'\n");
        }
        retval = 2;
    }
    if (implicit && (NA_as_zero_X || weight != NULL || Xfull != NULL)) {
        if (verbose) {
            fprintf(stderr, "Combination of inputs invalid for 'implicit'.\n");
        }
        retval = 2;
    }
    if (NA_as_zero_X && Xfull != NULL) {
        if (verbose) {
            fprintf(stderr, "Cannot use 'NA_as_zero' with dense inputs.\n");
        }
        retval = 2;
    }

    if (retval != 0) {
        if (verbose) {
            #ifndef _FOR_R
            fflush(stderr);
            #endif
        }
        return retval;
    }

    int_t minus_one = -1;
    int_t ignore = 0;
    real_t temp = 0;
    int_t temp_intA = 0;
    int_t ldb = 0;
    real_t placeholder = 0;

    real_t threshold_svd = 1e-5;
    int_t rank = 0;
    int_t sz_iwork = 0;
    real_t *restrict sv = NULL;


    int_t p_plus_bias = p + (int)add_intercepts;
    int_t q_plus_bias = q + (int)add_intercepts;
    real_t *U_plus_bias = NULL;
    real_t *I_plus_bias = NULL;
    real_t *restrict MatTrans = NULL;
    real_t *restrict buffer_real_t = NULL;
    int_t *restrict buffer_iwork = NULL;

    if (!implicit)
        retval = fit_collective_explicit_als(
                    biasA, biasB, A, B, (real_t*)NULL, (real_t*)NULL,
                    (real_t*)NULL, (real_t*)NULL, false,
                    reset_values, seed,
                    glob_mean,
                    (real_t*)NULL, (real_t*)NULL,
                    m, n, k,
                    ixA, ixB, X, nnz,
                    Xfull,
                    weight,
                    user_bias, item_bias, center,
                    lam, (real_t*)NULL,
                    0., (real_t*)NULL,
                    false, false, false,
                    (real_t*)NULL, (real_t*)NULL,
                    (real_t*)NULL, 0, 0,
                    (real_t*)NULL, 0, 0,
                    (int_t*)NULL, (int_t*)NULL, (real_t*)NULL, 0,
                    (int_t*)NULL, (int_t*)NULL, (real_t*)NULL, 0,
                    NA_as_zero_X, false, false,
                    0, 0, 0,
                    1., 1., 1., 1.,
                    niter, nthreads, verbose, true,
                    use_cg, max_cg_steps, finalize_chol,
                    false, 0, false, false,
                    precompute_for_predictions,
                    true,
                    Bm_plus_bias,
                    precomputedBtB,
                    precomputedTransBtBinvBt,
                    (real_t*)NULL,
                    (real_t*)NULL,
                    (real_t*)NULL,
                    (real_t*)NULL,
                    (real_t*)NULL,
                    (real_t*)NULL
                );
    else
        retval = fit_collective_implicit_als(
                    A, B, (real_t*)NULL, (real_t*)NULL,
                    reset_values, seed,
                    (real_t*)NULL, (real_t*)NULL,
                    m, n, k,
                    ixA, ixB, X, nnz,
                    lam, (real_t*)NULL,
                    0., (real_t*)NULL,
                    (real_t*)NULL, 0, 0,
                    (real_t*)NULL, 0, 0,
                    (int_t*)NULL, (int_t*)NULL, (real_t*)NULL, 0,
                    (int_t*)NULL, (int_t*)NULL, (real_t*)NULL, 0,
                    false, false,
                    0, 0, 0,
                    1., 1., 1.,
                    &placeholder,
                    alpha, false, apply_log_transf,
                    niter, nthreads, verbose, true,
                    use_cg, max_cg_steps, finalize_chol,
                    false, 0, false, false,
                    precompute_for_predictions,
                    precomputedBtB,
                    (real_t*)NULL, (real_t*)NULL, (real_t*)NULL
                );
    if (retval == 1)
        goto throw_oom;
    else if (retval == 3) {
        if (!handle_interrupt)
            goto cleanup;
    }
    else if (retval != 0) {
        if (verbose) {
            fprintf(stderr, "Unexpected error\n");
            #ifndef _FOR_R
            fflush(stderr);
            #endif
        }
        return retval;
    }

    if (Am != NULL)
        copy_arr_(A, Am, (size_t)m*(size_t)k, nthreads);
    if (Bm != NULL)
        copy_arr_(B, Bm, (size_t)n*(size_t)k, nthreads);

    if (U != NULL)
    {
        ldb = max2(m, p_plus_bias);
        MatTrans = (real_t*)malloc((size_t)ldb*(size_t)k*sizeof(real_t));
        U_plus_bias = (real_t*)malloc((size_t)m*(size_t)p_plus_bias
                                      * sizeof(real_t));
        sv = (real_t*)malloc((size_t)min2(m,p_plus_bias)*sizeof(real_t));
        if (MatTrans == NULL || U_plus_bias == NULL || sv == NULL)
            goto throw_oom;

        tgelsd_(&m, &p_plus_bias, &k,
                U_plus_bias, &m, MatTrans, &ldb,
                sv, &threshold_svd, &rank,
                &temp, &minus_one, &sz_iwork, &ignore);
        temp_intA = (int_t)temp;

        buffer_real_t = (real_t*)malloc((size_t)temp_intA*sizeof(real_t));
        buffer_iwork = (int_t*)malloc((size_t)(sz_iwork+1)*sizeof(int_t));
        if (buffer_real_t == NULL || buffer_iwork == NULL)
            goto throw_oom;

        
        transpose_mat3(
            A, k,
            m, k,
            MatTrans, ldb
        );

        transpose_mat3(
            U, p,
            m, p,
            U_plus_bias, m
        );
        if (add_intercepts)
            for (size_t ix = 0; ix < (size_t)m; ix++)
                U_plus_bias[(size_t)m*(size_t)p + ix] = 1.;

        tgelsd_(&m, &p_plus_bias, &k,
                U_plus_bias, &m, MatTrans, &ldb,
                sv, &threshold_svd, &rank,
                buffer_real_t, &temp_intA, buffer_iwork, &ignore);

        if (add_intercepts)
            append_ones_last_col(
                U, m, p,
                U_plus_bias
            );
        else
            U_plus_bias = U;
        transpose_mat3(
            MatTrans, ldb,
            k, p,
            C, k
        );
        if (add_intercepts)
            cblas_tcopy(k, MatTrans + p, ldb, C_bias, 1);

        cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m, k, p_plus_bias,
                    -1., U_plus_bias, p_plus_bias, MatTrans, ldb,
                    1., A, k);

        free(MatTrans); MatTrans = NULL;
        free(buffer_real_t); buffer_real_t = NULL;
        if (U_plus_bias != U) {
            free(U_plus_bias);
            U_plus_bias = NULL;
        }
        free(sv); sv = NULL;
        free(buffer_iwork); buffer_iwork = NULL;
    }

    if (II != NULL)
    {
        ldb = max2(n, q_plus_bias);
        MatTrans = (real_t*)malloc((size_t)ldb*(size_t)k*sizeof(real_t));
        I_plus_bias = (real_t*)malloc((size_t)n*(size_t)q_plus_bias
                                      * sizeof(real_t));
        sv = (real_t*)malloc((size_t)min2(n,q_plus_bias)*sizeof(real_t));
        if (MatTrans == NULL || I_plus_bias == NULL || sv == NULL)
            goto throw_oom;

        tgelsd_(&n, &q_plus_bias, &k,
                I_plus_bias, &m, MatTrans, &ldb,
                sv, &threshold_svd, &rank,
                &temp, &minus_one, &sz_iwork, &ignore);
        temp_intA = (int_t)temp;

        buffer_real_t = (real_t*)malloc((size_t)temp_intA*sizeof(real_t));
        buffer_iwork = (int_t*)malloc((size_t)(sz_iwork+1)*sizeof(int_t));
        if (buffer_real_t == NULL || buffer_iwork == NULL)
            goto throw_oom;
        
        transpose_mat3(
            B, k,
            n, k,
            MatTrans, ldb
        );

        transpose_mat3(
            II, q,
            n, q,
            I_plus_bias, n
        );
        if (add_intercepts)
            for (size_t ix = 0; ix < (size_t)n; ix++)
                I_plus_bias[(size_t)n*(size_t)q + ix] = 1.;

        tgelsd_(&n, &q_plus_bias, &k,
                I_plus_bias, &n, MatTrans, &ldb,
                sv, &threshold_svd, &rank,
                buffer_real_t, &temp_intA, buffer_iwork, &ignore);

        if (add_intercepts)
            append_ones_last_col(
                II, n, q,
                I_plus_bias
            );
        else
            I_plus_bias = II;
        transpose_mat3(
            MatTrans, ldb,
            k, q,
            D, k
        );
        if (add_intercepts)
            cblas_tcopy(k, MatTrans + q, ldb, D_bias, 1);

        cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n, k, q_plus_bias,
                    -1., I_plus_bias, q_plus_bias, MatTrans, ldb,
                    1., B, k);

        free(MatTrans); MatTrans = NULL;
        free(buffer_real_t); buffer_real_t = NULL;
        if (I_plus_bias != II) {
            free(I_plus_bias);
            I_plus_bias = NULL;
        }
        free(sv); sv = NULL;
        free(buffer_iwork); buffer_iwork = NULL;
    }

    cleanup:
        free(MatTrans);
        free(buffer_real_t);
        if (U_plus_bias != U)
            free(U_plus_bias);
        if (I_plus_bias != II)
            free(I_plus_bias);
        free(sv);
        free(buffer_iwork);
        act_on_interrupt(retval, handle_interrupt, true);
    return retval;

    throw_oom:
    {
        if (verbose)
            print_oom_message();
        retval = 1;
        goto cleanup;
    }
}

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
)
{
    return fit_offsets_als(
        biasA, biasB,
        A, B,
        C, C_bias,
        D, D_bias,
        reset_values, seed,
        glob_mean,
        m, n, k,
        ixA, ixB, X, nnz,
        Xfull,
        weight,
        user_bias, item_bias, center, add_intercepts,
        lam,
        U, p,
        II, q,
        false, NA_as_zero_X, 1., false,
        niter,
        nthreads, use_cg,
        max_cg_steps, finalize_chol,
        verbose, handle_interrupt,
        precompute_for_predictions,
        Am, Bm,
        Bm_plus_bias,
        precomputedBtB,
        precomputedTransBtBinvBt
    );
}

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
)
{
    return fit_offsets_als(
        (real_t*)NULL, (real_t*)NULL,
        A, B,
        C, C_bias,
        D, D_bias,
        reset_values, seed,
        (real_t*)NULL,
        m, n, k,
        ixA, ixB, X, nnz,
        (real_t*)NULL,
        (real_t*)NULL,
        false, false, false, add_intercepts,
        lam,
        U, p,
        II, q,
        true, false,
        alpha, apply_log_transf,
        niter,
        nthreads, use_cg,
        max_cg_steps, finalize_chol,
        verbose, handle_interrupt,
        precompute_for_predictions,
        Am, Bm,
        (real_t*)NULL,
        precomputedBtB,
        (real_t*)NULL
    );
}

int_t matrix_content_based
(
    real_t *restrict Am_new,
    int_t m_new, int_t k,
    real_t *restrict U, int_t p,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict C, real_t *restrict C_bias,
    int nthreads
)
{
    if (U != NULL)
    {
        cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m_new, k, p,
                    1., U, p, C, k,
                    0., Am_new, k);
    }

    else
    {
        int_t retval = 0;
        size_t *restrict U_csr_p_use = NULL;
        int_t *restrict U_csr_i_use = NULL;
        real_t *restrict U_csr_use = NULL;
        if (U_sp != NULL && U_csr_p == NULL) {
            retval = coo_to_csr_plus_alloc(
                        U_row, U_col, U_sp, (real_t*)NULL,
                        m_new, p, nnz_U,
                        &U_csr_p_use, &U_csr_i_use, &U_csr_use, (real_t**)NULL
                    );
            if (retval != 0) goto cleanup;
        }
        else {
            U_csr_p_use = U_csr_p;
            U_csr_i_use = U_csr_i;
            U_csr_use = U_csr;
        }

        set_to_zero_(Am_new, (size_t)m_new*(size_t)k, nthreads);
        tgemm_sp_dense(
            m_new, k, 1.,
            U_csr_p_use, U_csr_i_use, U_csr_use,
            C, (size_t)k,
            Am_new, (size_t)k,
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
        mat_plus_colvec(Am_new, C_bias, 1., m_new,
                        k, (size_t)k, nthreads);

    return 0;
}

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
)
{
    int_t retval = 0;

    bool set_to_nan = check_sparse_indices(
        n, p,
        u_vec_sp, u_vec_ixB, nnz_u_vec,
        Xa, ixB, nnz
    );
    if (set_to_nan) {
        for (int_t ix = 0; ix < k_sec+k+k_main; ix++)
            a_vec[ix] = NAN_;
        if (a_bias != NULL) *a_bias = NAN_;
        if (output_a != NULL)
            for (int_t ix = 0; ix < k+k_main; ix++)
                output_a[ix] = NAN_;
        return 0;
    }

    bool free_Bm_plus_bias = false;
    real_t lam_bias = lam;
    if (lam_unique != NULL)
    {
        lam_bias = lam_unique[(a_bias != NULL)? 0 : 2];
        lam = lam_unique[2];
    }

    if (nnz && (Bm_plus_bias == NULL && a_bias != NULL))
    {
        if (a_bias != NULL)
        {
            free_Bm_plus_bias = true;
            Bm_plus_bias = (real_t*)malloc((size_t)n*(size_t)(k_sec+k+k_main+1)
                                           * sizeof(real_t));
            if (Bm_plus_bias == NULL) goto throw_oom;
            append_ones_last_col(
                Bm, n, k_sec+k+k_main,
                Bm_plus_bias
            );
        }
    }


    if (!nnz)
    {
        if (a_bias != NULL)
            *a_bias = 0;
        retval = offsets_factors_cold(
            a_vec,
            u_vec,
            u_vec_ixB, u_vec_sp, nnz_u_vec,
            C, p,
            C_bias,
            k, k_sec, k_main,
            w_user
        );
    }

    else
        retval = offsets_factors_warm(
            a_vec, a_bias,
            u_vec,
            u_vec_ixB, u_vec_sp, nnz_u_vec,
            ixB, Xa, nnz,
            Xa_dense, n,
            weight,
            Bm, C,
            C_bias,
            glob_mean, biasB,
            k, k_sec, k_main,
            p, w_user,
            lam, exact, lam_bias,
            false, 1.,
            precomputedTransBtBinvBt,
            precomputedBtB,
            output_a,
            Bm_plus_bias
        );

    cleanup:
        if (free_Bm_plus_bias)
            free(Bm_plus_bias);
        return retval;

    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}

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
)
{
    bool set_to_nan = check_sparse_indices(
        n, p,
        u_vec_sp, u_vec_ixB, nnz_u_vec,
        Xa, ixB, nnz
    );
    if (set_to_nan) {
        for (int_t ix = 0; ix < k; ix++)
            a_vec[ix] = NAN_;
        return 0;
    }

    if (!nnz)
        return offsets_factors_cold(
            a_vec,
            u_vec,
            u_vec_ixB, u_vec_sp, nnz_u_vec,
            C, p,
            C_bias,
            k, 0, 0,
            1.
        );
    else {
        if (apply_log_transf)
            for (size_t ix = 0; ix < nnz; ix++)
                Xa[ix] = log_t(Xa[ix]);

        return offsets_factors_warm(
            a_vec, (real_t*)NULL,
            u_vec,
            u_vec_ixB, u_vec_sp, nnz_u_vec,
            ixB, Xa, nnz,
            (real_t*)NULL, 0,
            (real_t*)NULL,
            Bm, C,
            C_bias,
            0., (real_t*)NULL,
            k, 0, 0,
            p, 1.,
            lam, false, lam,
            true, alpha,
            (real_t*)NULL,
            precomputedBtB,
            output_a,
            (real_t*)NULL
        );
    }
}

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
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif
    int_t retval = 0;

    bool free_Bm_plus_bias = false;
    bool free_U_csr = false;
    bool free_X_csr = false;
    size_t lda = k_sec + k + k_main;
    size_t ld_A_orig = k + k_main;
    real_t *restrict weightR = NULL;
    int_t *restrict ret = (int_t*)malloc(m*sizeof(int_t));
    if (ret == NULL) goto throw_oom;

    if (!nnz && (Bm_plus_bias == NULL && biasA != NULL))
    {
        if (biasA != NULL)
        {
            free_Bm_plus_bias = true;
            Bm_plus_bias = (real_t*)malloc((size_t)n*(size_t)(k_sec+k+k_main+1)
                                           * sizeof(real_t));
            if (Bm_plus_bias == NULL) goto throw_oom;
            append_ones_last_col(
                Bm, n, k_sec+k+k_main,
                Bm_plus_bias
            );
        }
    }

    if (Xfull != NULL) {
        Xcsr_p = NULL;
        nnz = 0;
    }

    if (Xfull == NULL && Xcsr == NULL && nnz)
    {
        free_X_csr = true;
        Xcsr_p = (size_t*)malloc(((size_t)m + (size_t)1) * sizeof(size_t));
        Xcsr_i = (int_t*)malloc(nnz*sizeof(int_t));
        Xcsr = (real_t*)malloc(nnz*sizeof(real_t));
        if (Xcsr_p == NULL || Xcsr_i == NULL || Xcsr == NULL)
            goto throw_oom;
        if (weight != NULL) {
            weightR = (real_t*)malloc(nnz*sizeof(real_t));
            if (weightR == NULL) goto throw_oom;
        }
        coo_to_csr(
            ixA, ixB, X,
            weight,
            m, n, nnz,
            Xcsr_p, Xcsr_i, Xcsr,
            weightR
        );
    }

    else if (Xfull == NULL && Xcsr_p != NULL && weight != NULL) {
        weightR = weight;
    }

    if (U != NULL) {
        nnz_U = 0;
        U_csr_p = NULL;
    }

    if (U == NULL && nnz_U && U_csr_p == NULL)
    {
        free_U_csr = true;
        U_csr_p = (size_t*)malloc(((size_t)m + (size_t)1) * sizeof(size_t));
        U_csr_i = (int_t*)malloc(nnz_U*sizeof(int_t));
        U_csr = (real_t*)malloc(nnz_U*sizeof(real_t));
        if (U_csr_p == NULL || U_csr_i == NULL || U_csr == NULL)
            goto throw_oom;
        coo_to_csr(
            U_row, U_col, U_sp,
            (real_t*)NULL,
            m, p, nnz_U,
            U_csr_p, U_csr_i, U_csr,
            (real_t*)NULL
        );
    }

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
            shared(A, Am, Bm, C, C_bias, biasA, biasB, lda, ld_A_orig, \
                   U, U_csr, U_csr_i, U_csr_p, p, \
                   Xfull, Xcsr, Xcsr_i, Xcsr_p, n, weight, weightR, glob_mean, \
                   k, k_sec, k_main, w_user, lam, lam_unique, \
                   exact, precomputedTransBtBinvBt, precomputedBtB,Bm_plus_bias)
    for (size_t_for ix = 0; ix < (size_t)m; ix++)
        ret[ix] = factors_offsets_explicit_single(
            Am + ix*lda,
            (biasA == NULL)? ((real_t*)NULL) : (biasA + ix),
            (A == NULL)? ((real_t*)NULL) : (A + ix*ld_A_orig),
            (U == NULL)?
                ((real_t*)NULL) : (U + ix*(size_t)p),
            p,
            (U_csr_p == NULL)?
                ((real_t*)NULL) : (U_csr + U_csr_p[ix]),
            (U_csr_p == NULL)?
                ((int_t*)NULL) : (U_csr_i + U_csr_p[ix]),
            (U_csr_p == NULL)?
                ((size_t)0) : (U_csr_p[ix+1] - U_csr_p[ix]),
            (Xcsr_p != NULL)?
                (Xcsr + Xcsr_p[ix]) : ((real_t*)NULL),
            (Xcsr_p != NULL)?
                (Xcsr_i + Xcsr_p[ix]) : ((int_t*)NULL),
            (Xcsr_p != NULL)?
                (Xcsr_p[ix+1] - Xcsr_p[ix]) : ((size_t)0),
            (Xfull == NULL)?
                ((real_t*)NULL) : (Xfull + ix*(size_t)n),
            n,
            (weight == NULL)?
                ((real_t*)NULL)
                    :
                ((Xfull != NULL)?
                    (weight + ix*(size_t)n) : (weightR + Xcsr_p[ix])),
            Bm, C,
            C_bias,
            glob_mean, biasB,
            k, k_sec, k_main,
            w_user,
            lam, lam_unique,
            exact,
            precomputedTransBtBinvBt,
            precomputedBtB,
            Bm_plus_bias
        );


    for (size_t ix = 0; ix < (size_t)m; ix++)
        retval = max2(retval, ret[ix]);
    if (retval == 1) goto throw_oom;


    cleanup:
        if (free_U_csr) {
            free(U_csr);
            free(U_csr_i);
            free(U_csr_p);
        }
        if (free_X_csr) {
            free(Xcsr);
            free(Xcsr_p);
            free(Xcsr_i);
            free(weightR);
        }
        if (free_Bm_plus_bias)
            free(Bm_plus_bias);
        free(ret);
        return retval;

    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }

    return 0;
}

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
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif
    int_t retval = 0;
    
    bool free_U_csr = false;
    bool free_X_csr = false;
    bool free_BtB = false;
    int_t *restrict ret = (int_t*)malloc(m*sizeof(int_t));
    if (ret == NULL) goto throw_oom;

    if (Xcsr == NULL && nnz)
    {
        free_X_csr = true;
        Xcsr_p = (size_t*)malloc(((size_t)m + (size_t)1) * sizeof(size_t));
        Xcsr_i = (int_t*)malloc(nnz*sizeof(int_t));
        Xcsr = (real_t*)malloc(nnz*sizeof(real_t));
        if (Xcsr_p == NULL || Xcsr_i == NULL || Xcsr == NULL)
            goto throw_oom;
        coo_to_csr(
            ixA, ixB, X,
            (real_t*)NULL,
            m, 0, nnz,
            Xcsr_p, Xcsr_i, Xcsr,
            (real_t*)NULL
        );
    }

    if (U != NULL) {
        nnz_U = 0;
        U_csr_p = NULL;
    }

    if (U == NULL && nnz_U && U_csr_p == NULL)
    {
        free_U_csr = true;
        U_csr_p = (size_t*)malloc(((size_t)m + (size_t)1) * sizeof(size_t));
        U_csr_i = (int_t*)malloc(nnz_U*sizeof(int_t));
        U_csr = (real_t*)malloc(nnz_U*sizeof(real_t));
        if (U_csr_p == NULL || U_csr_i == NULL || U_csr == NULL)
            goto throw_oom;
        coo_to_csr(
            U_row, U_col, U_sp,
            (real_t*)NULL,
            m, p, nnz_U,
            U_csr_p, U_csr_i, U_csr,
            (real_t*)NULL
        );
    }

    if (precomputedBtB == NULL && Xcsr_p != NULL)
    {
        free_BtB = true;
        precomputedBtB = (real_t*)malloc((size_t)square(k)*sizeof(real_t));
        if (precomputedBtB == NULL) goto throw_oom;
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k, n,
                    1., Bm, k,
                    0., precomputedBtB, k);
    }

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
            shared(Am, Bm, C, C_bias, A, \
                   Xcsr, Xcsr_i, Xcsr_p, \
                   U, U_csr, U_csr_i, U_csr_p, p, \
                   k, lam, alpha, precomputedBtB)
    for (size_t_for ix = 0; ix < (size_t)m; ix++)
        ret[ix] = factors_offsets_implicit_single(
            Am + ix*(size_t)k,
            (U == NULL)?
                ((real_t*)NULL) : (U + ix*(size_t)p),
            p,
            (U_csr_p == NULL)?
                ((real_t*)NULL) : (U_csr + U_csr_p[ix]),
            (U_csr_p == NULL)?
                ((int_t*)NULL) : (U_csr_i + U_csr_p[ix]),
            (U_csr_p == NULL)?
                ((size_t)0) : (U_csr_p[ix+1] - U_csr_p[ix]),
            (Xcsr_p != NULL)? (Xcsr + Xcsr_p[ix]) : ((real_t*)NULL),
            (Xcsr_p != NULL)? (Xcsr_i + Xcsr_p[ix]) : ((int_t*)NULL),
            (Xcsr_p != NULL)? (Xcsr_p[ix+1] - Xcsr_p[ix]) : ((size_t)0),
            Bm, C,
            C_bias,
            k, n,
            lam, alpha,
            apply_log_transf,
            precomputedBtB,
            (A == NULL)? ((real_t*)NULL) : (A + ix*(size_t)k)
        );
    
    for (size_t ix = 0; ix < (size_t)m; ix++)
        retval = max2(retval, ret[ix]);
    if (retval == 1) goto throw_oom;

    cleanup:
        if (free_U_csr) {
            free(U_csr);
            free(U_csr_i);
            free(U_csr_p);
        }
        if (free_X_csr) {
            free(Xcsr);
            free(Xcsr_p);
            free(Xcsr_i);
        }
        if (free_BtB) {
            free(precomputedBtB);
        }
        free(ret);
        return retval;

    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}

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
)
{
    if (a_vec != NULL)
        return topN(
            a_vec, 0,
            Bm, 0,
            biasB,
            glob_mean, a_bias,
            k_sec+k+k_main, 0,
            include_ix, n_include,
            exclude_ix, n_exclude,
            outp_ix, outp_score,
            n_top, n, nthreads
        );

    else
        return topN(
            Am + (size_t)row_index*(size_t)(k_sec+k+k_main), 0,
            Bm, 0,
            biasB,
            glob_mean, (biasA == NULL)? (0.) : (biasA[row_index]),
            k_sec+k+k_main, 0,
            include_ix, n_include,
            exclude_ix, n_exclude,
            outp_ix, outp_score,
            n_top, n, nthreads
        );
}

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
)
{
    if (a_vec != NULL)
        return topN(
            a_vec, 0,
            Bm, 0,
            (real_t*)NULL,
            0., 0.,
            k, 0,
            include_ix, n_include,
            exclude_ix, n_exclude,
            outp_ix, outp_score,
            n_top, n, nthreads
        );

    else
        return topN(
            Am + (size_t)row_index*(size_t)k, 0,
            Bm, 0,
            (real_t*)NULL,
            0., 0.,
            k, 0,
            include_ix, n_include,
            exclude_ix, n_exclude,
            outp_ix, outp_score,
            n_top, n, nthreads
        );
}

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
)
{
    int_t retval = 0;
    real_t *restrict a_vec = (real_t*)malloc((size_t)(k_sec+k+k_main)
                                            * sizeof(real_t));
    real_t a_bias = 0.;
    if (a_vec == NULL) goto throw_oom;

    retval = factors_offsets_explicit_single(
        a_vec, user_bias? &a_bias : (real_t*)NULL, (real_t*)NULL,
        u_vec, p,
        u_vec_sp, u_vec_ixB, nnz_u_vec,
        Xa, ixB, nnz,
        Xa_dense, n,
        weight,
        Bm, C,
        C_bias,
        glob_mean, biasB,
        k, k_sec, k_main,
        w_user,
        lam, lam_unique,
        exact,
        precomputedTransBtBinvBt,
        precomputedBtB,
        Bm_plus_bias
    );

    if (retval == 1)
        goto throw_oom;
    else if (retval != 0)
        goto cleanup;

    retval = topN_old_offsets_explicit(
        a_vec, a_bias,
        (real_t*)NULL, (real_t*)NULL, 0,
        Bm,
        biasB,
        glob_mean,
        k, k_sec, k_main,
        include_ix, n_include,
        exclude_ix, n_exclude,
        outp_ix, outp_score,
        n_top, n, nthreads
    );

    cleanup:
        free(a_vec);
        return retval;
    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}

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
)
{
    int_t retval = 0;
    real_t *restrict a_vec = (real_t*)malloc((size_t)k * sizeof(real_t));
    if (a_vec == NULL) goto throw_oom;

    retval = factors_offsets_implicit_single(
        a_vec,
        u_vec, p,
        u_vec_sp, u_vec_ixB, nnz_u_vec,
        Xa, ixB, nnz,
        Bm, C,
        C_bias,
        k, n,
        lam, alpha,
        apply_log_transf,
        precomputedBtB,
        (real_t*)NULL
    );

    if (retval == 1)
        goto throw_oom;
    else if (retval != 0)
        goto cleanup;

    retval = topN_old_offsets_implicit(
        a_vec,
        (real_t*)NULL, 0,
        Bm,
        k,
        include_ix, n_include,
        exclude_ix, n_exclude,
        outp_ix, outp_score,
        n_top, n, nthreads
    );
    
    cleanup:
        free(a_vec);
        return retval;
    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}

int_t predict_X_old_offsets_explicit
(
    int_t row[], int_t col[], real_t *restrict predicted, size_t n_predict,
    real_t *restrict Am, real_t *restrict biasA,
    real_t *restrict Bm, real_t *restrict biasB,
    real_t glob_mean,
    int_t k, int_t k_sec, int_t k_main,
    int_t m, int_t n,
    int nthreads
)
{
    predict_multiple(
        Am, 0,
        Bm, 0,
        biasA, biasB,
        glob_mean,
        k_sec+k+k_main, 0,
        m, n,
        row, col, n_predict,
        predicted,
        nthreads
    );

    return 0;
}

int_t predict_X_old_offsets_implicit
(
    int_t row[], int_t col[], real_t *restrict predicted, size_t n_predict,
    real_t *restrict Am,
    real_t *restrict Bm,
    int_t k,
    int_t m, int_t n,
    int nthreads
)
{
    predict_multiple(
        Am, 0,
        Bm, 0,
        (real_t*)NULL, (real_t*)NULL,
        0.,
        k, 0,
        m, n,
        row, col, n_predict,
        predicted,
        nthreads
    );

    return 0;
}

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
)
{
    int_t retval = 0;
    real_t *restrict biasA = NULL;
    real_t *restrict Am = (real_t*)malloc(   (size_t)m_new
                                           * (size_t)(k_sec+k+k_main)
                                           * sizeof(real_t));
    if (Am == NULL) goto throw_oom;
    if (user_bias) {
        biasA = (real_t*)malloc((size_t)m_new * sizeof(real_t));
        if (biasA == NULL) goto throw_oom;
    }


    retval = factors_offsets_explicit_multiple(
        Am, biasA,
        (real_t*)NULL, m_new,
        U, p,
        U_row, U_col, U_sp, nnz_U,
        U_csr_p, U_csr_i, U_csr,
        X, ixA, ixB, nnz,
        Xcsr_p, Xcsr_i, Xcsr,
        Xfull, n,
        weight,
        Bm, C,
        C_bias,
        glob_mean, biasB,
        k, k_sec, k_main,
        w_user,
        lam, lam_unique, exact,
        precomputedTransBtBinvBt,
        precomputedBtB,
        Bm_plus_bias,
        nthreads
    );

    if (retval != 0)
        goto cleanup;

    retval = predict_X_old_offsets_explicit(
        row, col, predicted, n_predict,
        Am, biasA,
        Bm, biasB,
        glob_mean,
        k, k_sec, k_main,
        m_new, n,
        nthreads
    );

    cleanup:
        free(Am);
        free(biasA);
        return retval;
    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}

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
)
{
    int_t retval = 0;
    real_t *restrict Am = (real_t*)malloc((size_t)m_new * (size_t)k
                                         * sizeof(real_t));
    if (Am == NULL) goto throw_oom;


    retval = factors_offsets_implicit_multiple(
        Am, m_new,
        (real_t*)NULL,
        U, p,
        U_row, U_col, U_sp, nnz_U,
        U_csr_p, U_csr_i, U_csr,
        X, ixA, ixB, nnz,
        Xcsr_p, Xcsr_i, Xcsr,
        Bm, C,
        C_bias,
        k, n_orig,
        lam, alpha,
        apply_log_transf,
        precomputedBtB,
        nthreads
    );

    if (retval != 0)
        goto cleanup;

    retval = predict_X_old_offsets_implicit(
        row, col, predicted, n_predict,
        Am,
        Bm,
        k,
        m_new, n_orig,
        nthreads
    );

    cleanup:
        free(Am);
        return retval;
    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}


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
)
{
    int_t retval = 0;
    real_t *restrict tempA = NULL;
    real_t *restrict tempB = NULL;
    real_t *restrict Xorig = NULL;
    real_t *values = NULL;
    size_t edge = 0;
    /* If the values are all in contiguous order, can avoid allocating and
       copying them before and after. */
    bool free_values = false;
    if (add_intercepts && C_bias != C + (size_t)p*(size_t)k)
        free_values = true;
    if (add_intercepts && D_bias != D + (size_t)q*(size_t)k)
        free_values = true;
    if (D != C + (size_t)(p+add_intercepts)*(size_t)k)
        free_values =  true;
    if (item_bias && C != biasB + n)
        free_values = true;
    if (user_bias && item_bias && biasB != biasA + m)
        free_values = true;
    else if (user_bias && !item_bias && C != biasA + m)
        free_values = true;

    if (U == NULL || II == NULL)
        start_with_ALS = false;

    if (start_with_ALS)
    {
        tempA = (real_t*)malloc((size_t)m*(size_t)k*sizeof(real_t));
        tempB = (real_t*)malloc((size_t)n*(size_t)k*sizeof(real_t));
        if (Xfull != NULL)
            Xorig = (real_t*)malloc((size_t)m*(size_t)n*sizeof(real_t));
        else
            Xorig = (real_t*)malloc(nnz*sizeof(real_t));
        if (tempA == NULL || tempB == NULL || Xorig == NULL)
            goto throw_oom;

        if (Xfull != NULL)
            copy_arr_(Xfull, Xorig, (size_t)m*(size_t)n, nthreads);
        else
            copy_arr_(X, Xorig, nnz, nthreads);

        if (!reset_values)
        {
            retval = matrix_content_based(
                tempA,
                m, k,
                U, p,
                U_row, U_col, U_sp, nnz_U,
                (size_t*)NULL, (int_t*)NULL, (real_t*)NULL,
                C, C_bias,
                nthreads
            );
            if (retval == 1)
                goto throw_oom;
            else if (retval != 0)
                goto cleanup;

            retval = matrix_content_based(
                tempB,
                n, k,
                II, q,
                I_row, I_col, I_sp, nnz_I,
                (size_t*)NULL, (int_t*)NULL, (real_t*)NULL,
                D, D_bias,
                nthreads
            );
            if (retval == 1)
                goto throw_oom;
            else if (retval != 0)
                goto cleanup;
        }

        retval = fit_offsets_explicit_als(
            biasA, biasB,
            tempA, tempB,
            C, C_bias,
            D, D_bias,
            reset_values, seed,
            glob_mean,
            m, n, k,
            ixA, ixB, (X == NULL)? ((real_t*)NULL) : Xorig, nnz,
            (Xfull == NULL)? ((real_t*)NULL) : Xorig,
            weight,
            user_bias, item_bias, true, add_intercepts,
            lam,
            U, p,
            II, q,
            false,
            10,
            nthreads, true,
            3, false,
            verbose, true,
            false,
            (real_t*)NULL, (real_t*)NULL,
            (real_t*)NULL,
            (real_t*)NULL,
            (real_t*)NULL
        );
        if ((retval != 0 && retval != 3) || (retval == 3 && !handle_interrupt))
            goto cleanup;

        free(tempA); tempA = NULL;
        free(tempB); tempB = NULL;
        free(Xorig); Xorig = NULL;
    }

    if (free_values)
    {
        values = (real_t*)malloc((  (size_t)p
                                   +(size_t)q
                                   +(size_t)(2*add_intercepts))
                                 * (size_t)k
                                 * sizeof(real_t));
        if (values == NULL) goto throw_oom;

        if (!reset_values  || start_with_ALS)
        {
            edge = 0;
            if (user_bias) {
                copy_arr(biasA, values + edge, m);
                edge += m;
            }
            if (item_bias) {
                copy_arr(biasB, values + edge, n);
                edge += n;
            }
            copy_arr_(C, values + edge, (size_t)p*(size_t)k, nthreads);
            edge += (size_t)p*(size_t)k;
            if (add_intercepts) {
                copy_arr(C_bias, values + edge, k);
                edge += k;
            }
            copy_arr_(D, values + edge, (size_t)q*(size_t)k, nthreads);
            edge += (size_t)q*(size_t)k;
            if (add_intercepts) {
                copy_arr(D_bias, values + edge, k);
                edge += k;
            }
        }
    } else {
        values = user_bias? biasA : (item_bias? biasB : C);
    }

    if (retval != 3)
        retval = fit_offsets_explicit_lbfgs_internal(
            values, reset_values,
            glob_mean,
            m, n, 0,
            ixA, ixB, X, nnz,
            Xfull,
            weight,
            user_bias, item_bias, true,
            add_intercepts,
            lam, lam_unique,
            U, p,
            II, q,
            U_row, U_col, U_sp, nnz_U,
            I_row, I_col, I_sp, nnz_I,
            0, k,
            1., 1.,
            n_corr_pairs, maxiter, seed,
            nthreads, prefer_onepass,
            verbose, print_every, true,
            niter, nfev,
            Am, Bm,
            (real_t*)NULL
        );
    if ((retval != 0 && retval != 3) || (retval == 3 && !handle_interrupt))
        goto cleanup;

    if (free_values)
    {
        edge = 0;
        if (user_bias) {
            copy_arr(values + edge, biasA, m);
            edge += m;
        }
        if (item_bias) {
            copy_arr(values + edge, biasB, n);
            edge += n;
        }
        copy_arr_(values + edge, C, (size_t)p*(size_t)k, nthreads);
        edge += (size_t)p*(size_t)k;
        if (add_intercepts) {
            copy_arr(values + edge, C_bias, k);
            edge += k;
        }
        copy_arr_(values + edge, D, (size_t)q*(size_t)k, nthreads);
        edge += (size_t)q*(size_t)k;
        if (add_intercepts) {
            copy_arr(values + edge, D_bias, k);
            edge += k;
        }
    }

    cleanup:
        if (free_values)
            free(values);
        free(tempA);
        free(tempB);
        free(Xorig);
        act_on_interrupt(retval, handle_interrupt, false);
        return retval;
    throw_oom:
    {
        if (verbose)
            print_oom_message();
        retval = 1;
        goto cleanup;
    }
}

int_t factors_content_based_single
(
    real_t *restrict a_vec, int_t k,
    real_t *restrict u_vec, int_t p,
    real_t *restrict u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
    real_t *restrict C, real_t *restrict C_bias
)
{
    if (u_vec != NULL)
    {
        cblas_tgemv(CblasRowMajor, CblasTrans,
                    p, k,
                    1., C, k,
                    u_vec, 1,
                    0., a_vec, 1);
    }

    else
    {
        set_to_zero(a_vec, k);
        tgemv_dense_sp(
            p, k,
            1., C, (size_t)k,
            u_vec_ixB, u_vec_sp, nnz_u_vec,
            a_vec
        );
    }

    if (C_bias != NULL)
        cblas_taxpy(k, 1., C_bias, 1, a_vec, 1);

    return 0;
}

int_t factors_content_based_mutliple
(
    real_t *restrict Am, int_t m_new, int_t k,
    real_t *restrict C, real_t *restrict C_bias,
    real_t *restrict U, int_t p,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    int nthreads
)
{
    return matrix_content_based(
        Am,
        m_new, k,
        U, p,
        U_row, U_col, U_sp, nnz_U,
        U_csr_p, U_csr_i, U_csr,
        C, C_bias,
        nthreads
    );
}

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
)
{
    return topN_old_collective_explicit(
        a_vec, a_bias,
        Am, biasA, row_index,
        Bm,
        biasB,
        glob_mean,
        k, 0, 0, 0,
        include_ix, n_include,
        exclude_ix, n_exclude,
        outp_ix, outp_score,
        n_top, n, n, false, nthreads
    );
}

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
)
{
    int_t retval = 0;
    real_t *restrict a_vec = (real_t*)malloc((size_t)k*sizeof(real_t));
    real_t *restrict Bm = (real_t*)malloc((size_t)n_new * (size_t)k
                                                      * sizeof(real_t));
    real_t *restrict scores_copy = (real_t*)malloc((size_t)n_new
                                                    * sizeof(real_t));
    
    int_t *restrict buffer_ix = NULL;
    if (n_top == 0 || n_top == n_new)
        buffer_ix = outp_ix;
    else
        buffer_ix = (int_t*)malloc((size_t)n_new*sizeof(int_t));

    if (a_vec == NULL || Bm == NULL || scores_copy == NULL || buffer_ix == NULL)
    {
        retval = 1;
        goto cleanup;
    }

    if (n_top == 0) n_top = n_new;

    retval = factors_content_based_single(
        a_vec, k,
        u_vec, p,
        u_vec_sp, u_vec_ixB, nnz_u_vec,
        C, C_bias
    );
    if (retval != 0) goto cleanup;

    retval = matrix_content_based(
        Bm,
        n_new, k,
        II, q,
        I_row, I_col, I_sp, nnz_I,
        I_csr_p, I_csr_i, I_csr,
        D, D_bias,
        nthreads
    );
    if (retval != 0) goto cleanup;

    cblas_tgemv(CblasRowMajor, CblasNoTrans,
                n_new, k,
                1., Bm, k,
                a_vec, 1,
                0., scores_copy, 1);

    for (int_t ix = 0; ix < n_new; ix++)
        buffer_ix[ix] = ix;


    ptr_real_t_glob = scores_copy;
    if (n_top <= 50 || n_top >= (double)n_new*0.75)
    {
        qsort(buffer_ix, n_new, sizeof(int_t), cmp_argsort);
    }

    else
    {
        qs_argpartition(buffer_ix, scores_copy, n_new, n_top);
        qsort(buffer_ix, n_top, sizeof(int_t), cmp_argsort);
    }

    if (buffer_ix != outp_ix)
        memcpy(outp_ix, buffer_ix, (size_t)n_top*sizeof(int_t));

    if (outp_score != NULL)
        for (int_t ix = 0; ix < n_top; ix++)
            outp_score[ix] = scores_copy[buffer_ix[ix]] + glob_mean;

    cleanup:
        free(a_vec);
        free(Bm);
        free(scores_copy);
        if (buffer_ix != outp_ix)
            free(buffer_ix);
    return retval;
}

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
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif
    if (m_orig == 0) m_orig = INT_MAX;
    if (n_orig == 0) n_orig = INT_MAX;
    real_t *restrict Am = (real_t*)malloc((size_t)n_predict *
                                        (size_t)k *
                                        sizeof(real_t));
    int_t retval = 0;
    if (Am == NULL) goto throw_oom;

    retval = matrix_content_based(
        Am,
        m_new, k,
        U, p,
        U_row, U_col, U_sp, nnz_U,
        U_csr_p, U_csr_i, U_csr,
        C, C_bias,
        nthreads
    );
    if (retval != 0) goto cleanup;

    if (row == NULL)
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(predicted, n_predict, col, k, Am, Bm, biasB, glob_mean)
        for (size_t_for ix = 0; ix < n_predict; ix++)
            predicted[ix] = (col[ix] >= n_orig || col[ix] < 0)?
                                (NAN_)
                                  :
                                (cblas_tdot(k, Am + ix*(size_t)k, 1,
                                               Bm + (size_t)col[ix]*(size_t)k,1)
                                 + ((biasB != NULL)? biasB[col[ix]] : 0.)
                                 + glob_mean);
    else
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(predicted, n_predict, row, col, k, Am, Bm, \
                       biasB, glob_mean)
        for (size_t_for ix = 0; ix < n_predict; ix++)
            predicted[ix] = (row[ix] >= m_orig || row[ix] < 0 ||
                             col[ix] >= n_orig || col[ix] < 0)?
                                (NAN_)
                                  :
                                (cblas_tdot(k, Am + row[ix]*(size_t)k, 1,
                                               Bm + (size_t)col[ix]*(size_t)k,1)
                                 + ((biasB != NULL)? biasB[col[ix]] : 0.)
                                 + glob_mean);

    cleanup:
        free(Am);
        return retval;
    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}

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
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif

    real_t *restrict Am = (real_t*)malloc((size_t)n_new *
                                          (size_t)k *
                                          sizeof(real_t));
    real_t *restrict Bm = (real_t*)malloc((size_t)n_new *
                                          (size_t)k *
                                          sizeof(real_t));
    int_t retval = 0;
    if (Am == NULL || Bm == NULL) goto throw_oom;

    retval = matrix_content_based(
        Am,
        m_new, k,
        U, p,
        U_row, U_col, U_sp, nnz_U,
        U_csr_p, U_csr_i, U_csr,
        C, C_bias,
        nthreads
    );
    if (retval != 0) goto cleanup;

    retval = matrix_content_based(
        Bm,
        n_new, k,
        II, q,
        I_row, I_col, I_sp, nnz_I,
        I_csr_p, I_csr_i, I_csr,
        D, D_bias,
        nthreads
    );
    if (retval != 0) goto cleanup;

    if (row == NULL || col == NULL)
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(predicted, n_predict, k, Am, Bm, glob_mean)
        for (size_t_for ix = 0; ix < n_predict; ix++)
            predicted[ix] = cblas_tdot(k, Am + ix*(size_t)k, 1,
                                          Bm + ix*(size_t)k, 1)
                             + glob_mean;
    else
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(predicted, n_predict, row, col, k, Am, Bm, glob_mean)
        for (size_t_for ix = 0; ix < n_predict; ix++)
            predicted[ix] = (row[ix] >= m_new || row[ix] < 0 ||
                             col[ix] >= n_new || col[ix] < 0)?
                                (NAN_)
                                  :
                                (cblas_tdot(k,Am + (size_t)row[ix]*(size_t)k, 1,
                                              Bm + (size_t)col[ix]*(size_t)k, 1)
                                 + glob_mean);

    cleanup:
        free(Am);
        free(Bm);
        return retval;
    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}
