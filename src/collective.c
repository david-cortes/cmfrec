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
#include "cmfrec.h"

/*******************************************************************************
    Collective Model
    ----------------

    This is a generalization of the model described in
        Singh, Ajit P., and Geoffrey J. Gordon.
        "Relational learning via collective matrix factorization."
        Proceedings of the 14th ACM SIGKDD international conference on
        Knowledge discovery and data mining. 2008.

    
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

    The basic model is complemented with side information about the users and
    the items in the form of matrices U[m,p], I[n,q], which are also factorized
    using the same A and B matrices as before, which are multiplied by new
    matrices C[p,k] and D[q,k] - e.g.:
        min ||M * (X - A*t(B))||^2 + ||U - A*t(C)||^2 + ||I - B*t(D)||^2

    This idea is further extended by:
    - Letting some of the 'k' components of each matrix be used only
      for one factorization but not for the other (these are: 'k' for the
      shared ones, 'k_main' for those that apply to approximate 'X',
      'k_user' for those that apply to approximate U, 'k_item' for I).
    - Applying a sigmoid transformation on the obtained approximate matrices
      for the columns that are binary in the user/item side information, e.g.
        min ||Ub - sigm(A*t(C))||^2

    The model can be fit either through a gradient-based approach using the
    L-BFGS solver, or (when there are no binary variables requiring a sigmoid
    transformation) through an alternating least-squares - note that when
    all but one matrices are fixed, there is a closed-form solution for the
    variable matrix which can be computed for each row in parallel.

    This module allows some yet additional distinctions in the formulas:
    - Different regularizaton parameter for each matrix.
    - Different weights for each factorization (w_main, w_user, w_item).
    - Observation weights W[m,n] for each entry in X (these have the same
      effect as repeating 'w_main' by 'n' times).
    - Having biases only for users and/or only for items, or for neither.
    - The U and I matrices are centered column-by-column, but these column
      biases are not model parameters.
    - U and I can also have missing values.
    And allows working with the inputs either as sparse or as dense matrices.

    For the gradient-based solution, the gradients can be calculated as:
        grad(A) = (W[m,n] * M[m,n] * (A*t(B) - X - b1 - b2 - mu)) * B
                  + (Mu[m,p] * (A*t(C) - U))*C
                  + (Mb*(Ub-sigm(A*t(Cb)))*exp(-A*t(Cb))/(exp(-A*t(Cb))+1)^2)*Cb
                  + lamda * A
    (The function value needs to be divided by 2 to match with the gradient
     calculated like that)

    For the closed-form solution with no binary variables, assuming that the
    matrix A is a block matrix composed of independent components in this order
    (first user-independent, then shared, then rating-independent):
        Ae = [Au, As, Am]
    The solution can be obtained **for each row 'a'** by factorizing an
    extended X matrix like this:
        Xe = [Xa[1,n], Ua[1,p]]
    in which [Au[1,k_user], As[1,k], Am[1,k_main]] is multiplied by another
    extended block matrix:
        Be = [[0,  Bs, Bm],
              [Cu, Cs, 0 ]]
    Where each row of [Bi, Bs, Bm] has the values of B if the entry for that
    column in the corresponding row Xa of X is present, and zeros if it's
    missing, i.e.
        [Bs, Bm, Bi] = B * t(M[1,n])
    Since the closed-form solution allows no 'w_user' and 'w_main', if these
    are present, they have to be transformed into a concatenated W[1,n+p]
        We[1,n+p] = [W[1,n] * w_main, 1[p] * w_user]
    The solution is then given by:
        A* = inv(t(Be*W)*Be + diag(lambda)) * (t(Be*W)*t(Xa))
    Note that since the left-hand side is, by definition, a symmetric
    positive-semi definite matrix, this this computation can be done with
    specialized procedures based on e.g. a Cholesky factorization, rather than
    a more general linear solver or matrix inversion.

    To obtain the bias, **before** applying this closed form solution, first
    center Xa and set that mean as the bias for that row - no regularization is
    possible, but it is still a model parameter which becomes updated after
    each iteration.


    Both the gradient-based approach and the closed-form solution with these
    formulas can be used for any of the 4 matrices by substituting the matrices
    accordingly - i.e.
        For matrix 'B', replace:
            A->B, C->D, U->I
        For matrix 'C', replace:
            A<->C, X<->U
        For matrix 'D', replace:
            A->D, X->I, C->A, U->X
    

    In order to obtain the factor matrices for new users, in a cold-start
    scenario (based on user attributes alone), it's only necessary to obtain
    the closed form for A assuming X is zero, while for a warm-start scenario
    (based on both user attributes and ratings), the closed form on block
    matrices can be applied. If there are binary variables, there is no
    closed-form solution, but can still be obtained in a reasonable time with
    a gradient-based approach.
    
    
*******************************************************************************/


/*******************************************************************************
    Function and Gradient Calculation
    ---------------------------------
    
    This function calculates the gradient as explained at the beginning. It
    can receive the X, U, I, matrices either as sparse (COO or CSR+CSC depending
    on parallelization strategy - see file 'common.c' for details), but be sure
    to pass only ONE (dense/sparse) of them. If they have binary variables
    (Ub, Ib), these must be passed as dense.

    For sparse matrices, non-present values will not be accounted for into the
    function and gradient calculations, while for dense matrices, missing values
    (as 'NAN') will not be counted.

    The X matrix should have already undergone global centering (subtracting the
    mean), while the U and I matrices, should be centered column-wise as they
    don't have biases as parameters.

    If passing observation weights, these must match the shape of the X matrix
     - that is, if X is dense, weights must be an array of dimensions (m, n),
    if X is sparse, must be an array of dimensions (nnz), and if parallelizing
    by independent X matrices, must pass it twice, each matching to a given
    formar of X.

    In order to know how many variables will the model have and/or how much
    space is required for temporary variables, use the function
    'nvars_collective_fun_grad' beforehand.

    Parameters
    ----------
    values[nvars]
        The current values for which the function and gradient will
        be evaluated.
    grad[nvars]
        Array in which to write the gradient evauates at 'values'.
    m
        Number of rows in X, A, U, Ub.
    n
        Number of columns in X and number of rows in B, I, Ib.
    k
        Dimensionality of the factorizing matrices (a.k.a. latent factors),
        denoting only the columns/factors that are shared between two
        factorizations (so A has number of columns equal to k_user + k + k_main,
        B has number of columns equal to k_item + k + k_main, and they arranged
        in that order - i.e. the first 'k_user' columns in A are factors used
        only for factorizing U).
    ixA[nnz], ixB[nnz], X[nnz], nnz
        The X matrix in sparse triplets (a.k.a. COO) format. Pass NULL if
        X will be provided in a different format.
    Xfull[m * n]
        The X matrix in dense format, with missing entries set as NAN. Pass
        NULL if X will be provided in a different format. If X is passed
        in multiple formats, the dense one will be required. Note that working
        with dense matrices will require extra memory of the same sizefor
        temporary values.
    Xcsr_p[m+1], Xcsr_i[nnz], Xcsr[nnz], Xcsc_p[n], Xcsc_i[nnz], Xcsc[nnz]
        The X matrix in sparse CSR and CSC formats. Only used if nthreads>1,
        otherwise pass X as COO. These are used if parallelizing according to
        a 2-pass strategy (see file "common.c" for details). Pass NULL if
        not applicable.
    weight[nnz or m*n], weightR[nnz], weightC[nnz]
        The observation weights for each entry of X. Must match with the shape
        of X - that is, if passing Xfull, must have size 'm*n', if passing X,
        must have size 'nnz', if passing Xcsr and Xcsc, must pass weightR
        and weightC instead. Pass NULL for uniform weights.
    user_bias, item_bias
        Whether the model should include these biases as parameters.
    lam
        Regularization parameter to apply to all matrices.
    lam_unique[6]
        Regularization parameters for each matrix used in the model. If passed,
        each one will have different regulatization, while if ommited (passing
        NULL), will apply the same regulatization to all. The entries must be
        in this order: (1) user bias, (2) item bias, (3) row/user factors,
        (4) column/item factors, (5) user attribute factors, (6) item attribute
        factors.
    U[m_u*p], U_row[nnz_U], U_col[nnz_U], U_sp[nnz_U],
    U_csr_p[m], U_csr_i[nnz_U], U_csr[nnz_U],
    U_csc_p[p], U_csc_i[nnz_U], U_csc[nnz_U]
        The user/row attribute matrix - same guidelines as for X apply. Note
        that it's possible to pass combinations of sparse X, dense U, sparse I,
        etc. without problems.
    II[n_i*q], I_row[nnz_I], I_col[nnz_I], I_sp[nnz_I],
    I_csr_p[n+1], I_csr_i[nnz_I], I_csr[nnz_I],
    I_csc_p[q+1], I_csc_i[nnz_I], I_csc[nnz_I]
        The item/column attribute matrix - same guidelines as for X apply. Note
        that it's possible to pass combinations of sparse X, dense U, sparse I,
        etc. without problems.
    Ub[m*pbin]
        The binary columns of the user/row attribute matrix. Must be passed as
        dense, but can contain missing values. The non-missing entries should
        all be either zero or one. Note that it's still possible to pass only
        one of U or Ubin, or neither.
    m_u
        Number of rows in the U matrix, in case not all rows in A have
        a corresponding entry in U. If this differs from 'm', the topmost rows
        of A and X will be assumed to match to the rows of U. Ignored when
        passing sparse U.
    m_ubin
        Same as above, but for Ub matrix.
    p
        Number of columns in the U matrix (i.e. number of user attributes).
    pbin
        Number of columns in the Ub matrix (i.e. number of binary attributes).
    Ib[n*qbin]
        The binary columns of the item/column attribute matrix. Must be passed
        as dense, but can contain missing values. The non-missing entries should
        all be either zero or one. Note that it's still possible to pass only
        one of I or Ibin, or neither.
    n_i
        Number of rows in the II matrix, in case not all rows in B have a
        corresponding entry in II. If this differs from 'n', the topmost rows
        of B and columns of X will be assumed to match to the rows of II.
        Ignored when passing sparse U.
    n_ibin
        Same as above, but for Ib matrix.
    q
        Number of columns in the I matrix (i.e. number of item attributes).
    qbin
        Number of columns in the Ib matrix (i.e. number of binary attributes).
    U_has_NA, I_has_NA, Ub_has_NA, Ib_has_NA
        Whether these matrices contain any missing values. Ignored when the
        matrices are passed as sparse.
    buffer_FPnum[temp]
        Temporary array in which to write values. Only used when passing at
        least one matrix as dense. Pass NULL if not applicable. The required
        size can be obtained from function 'nvars_collective_fun_grad', but
        as a guideline it needs to be able to hold the largest dense matrix
        that is passed.
    buffer_mt
        Temporary array in which to write thread-local values when using the
        one-pass parallelization strategy with sparse X matrix in COO format.
        Will not be used if passing Xcsr/Xcsc or Xfull or nthreads=1. The
        required size can be obtained through function
        'nvars_collective_gradient', but as a guideline it needs to hold the
        largest combination of gradient matrices used for a given factorization,
        multiplied by the number of threads. Pass NULL if not applicable, or if
        no parallelization is desired for a sparse X matrix in COO format.
    k_main
        Factors of A and B which are used only for factorizing the X matrix and
        not the other matrices. These will be available in the last columns
        of both matrices.
    k_user
        Factors of A which are used only for factorizing the U and Ub matrices.
        These will be available in the first columns of A.
    k_item
        Factors of B which are used only for factorizing the I and Ib matrices.
        These will be available in the first columns of B.
    w_main
        Weight given to the squared error in the factorization of the X matrix.
    w_user
        Weight given to the squared error in the factorization of the U and
        Ub matrices.
    w_item
        Weight given to the squared error in the factorization of the II and
        Ib matrices.
    nthreads
        Number of parallel threads to use. Note that (a) this function relies on
        BLAS and LAPACK functions, which set their number of threads externally,
        (b) Depending on the number of threads relative to the data size and the
        parallelization strategy (one-pass or two-pass), adding more threads
        might result in a slow down, and if using the one-pass strategy, will
        require extra memory.


*******************************************************************************/

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
)
{
    size_t m_max = max2(max2(m, m_u), m_ubin);
    size_t n_max = max2(max2(n, n_i), n_ibin);
    *nvars =   m_max * (k_user + k + k_main)
             + n_max * (k_item + k + k_main)
             + (p + pbin) * (k + k_user)
             + (q + qbin) * (k + k_item);
    if (user_bias) *nvars += m_max;
    if (item_bias) *nvars += n_max;

    *nbuffer = 0;
    if (Xfull != NULL) *nbuffer = m * n;
    if (U != NULL)  *nbuffer = max2(*nbuffer, m_u * p);
    if (II != NULL) *nbuffer = max2(*nbuffer, n_i * q);
    if (Ub != NULL) *nbuffer = max2(*nbuffer, m_ubin * pbin);
    if (Ib != NULL) *nbuffer = max2(*nbuffer, n_ibin * qbin);

    *nbuffer_mt = 0;
    if (nthreads > 1) {
        if (Xfull == NULL && X != NULL)
            *nbuffer_mt = (k_user + k + k_main + 1) * (m + n);
        if (U == NULL && U_sp != NULL)
            *nbuffer_mt = max2(*nbuffer_mt, (k_user + k + k_main) * m_u);
        if (II == NULL && I_sp != NULL)
            *nbuffer_mt = max2(*nbuffer_mt, (k_user + k + k_main) * n_i);
        *nbuffer_mt *= nthreads;
    }
}

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
)
{
    /* Shorthands to use later */
    int k_totA = k_user + k + k_main;
    int k_totB = k_item + k + k_main;
    int m_max = max2(max2(m, m_u), m_ubin);
    int n_max = max2(max2(n, n_i), n_ibin);
    long double f = 0;

    /* set the gradients to zero - first need to know how many entries to set */
    size_t nvars, ignored, mtvars;
    nvars_collective_fun_grad(
        (size_t)m, (size_t)n, (size_t)m_u, (size_t)n_i,
        (size_t)m_ubin, (size_t)n_ibin,
        (size_t)p, (size_t)q, (size_t)pbin, (size_t)qbin,
        nnz, nnz_U, nnz_I,
        (size_t)k, (size_t)k_main, (size_t)k_user, (size_t)k_item,
        user_bias, item_bias, (size_t)nthreads,
        X, Xfull, Xcsr,
        U, Ub, II, Ib,
        U_sp, U_csr, I_sp, I_csr,
        &nvars, &ignored, &mtvars
    );
    set_to_zero(grad, nvars, nthreads);
    if (mtvars && buffer_mt != NULL) set_to_zero(buffer_mt, mtvars, nthreads);

    /* unravel the arrays */
    FPnum *restrict biasA = values;
    FPnum *restrict biasB = biasA + (user_bias? m_max : 0);
    FPnum *restrict A = biasB + (item_bias? n_max : 0);
    FPnum *restrict B = A + (size_t)m_max * (size_t)k_totA;
    FPnum *restrict C = B + (size_t)n_max * (size_t)k_totB;
    FPnum *restrict Cb = C + (size_t)(k + k_user) * (size_t)p;
    FPnum *restrict D  = Cb + (size_t)(k + k_user) * (size_t)pbin;
    FPnum *restrict Db = D + (size_t)(k + k_item) * (size_t)q;


    FPnum *restrict g_biasA = grad;
    FPnum *restrict g_biasB = g_biasA + (user_bias? m_max : 0);
    FPnum *restrict g_A = g_biasB + (item_bias? n_max : 0);
    FPnum *restrict g_B = g_A + (size_t)m_max * (size_t)k_totA;
    FPnum *restrict g_C = g_B + (size_t)n_max * (size_t)k_totB;
    FPnum *restrict g_Cb = g_C + (size_t)(k + k_user) * (size_t)p;
    FPnum *restrict g_D  = g_Cb + (size_t)(k + k_user) * (size_t)pbin;
    FPnum *restrict g_Db = g_D + (size_t)(k + k_item) * (size_t)q;

    /* first the main factorization */
    f = fun_grad_cannonical_form(
        A + k_user, k_totA, B + k_item, k_totB,
        g_A + k_user, g_B + k_item,
        m, n, k + k_main,
        ixA, ixB, X, nnz,
        Xfull, false,
        Xcsr_p, Xcsr_i, Xcsr,
        Xcsc_p, Xcsc_i, Xcsc,
        user_bias, item_bias,
        biasA, biasB,
        g_biasA, g_biasB,
        weight, weightR, weightC,
        w_main,
        buffer_FPnum,
        buffer_mt,
        true,
        nthreads
    );

    /* then user non-binary factorization */
    if (U != NULL || U_sp != NULL)
        f += fun_grad_cannonical_form(
            A, k_totA, C, k_user + k,
            g_A, g_C,
            (U != NULL)? m_u : m, p, k_user + k,
            U_row, U_col, U_sp, nnz_U,
            U, !U_has_NA,
            U_csr_p, U_csr_i, U_csr,
            U_csc_p, U_csc_i, U_csc,
            false, false,
            (FPnum*)NULL, (FPnum*)NULL,
            (FPnum*)NULL, (FPnum*)NULL,
            (FPnum*)NULL, (FPnum*)NULL, (FPnum*)NULL,
            w_user,
            buffer_FPnum,
            buffer_mt,
            false,
            nthreads
        );

    /* then item non-binary factorization */
    if (II != NULL || I_sp != NULL)
        f += fun_grad_cannonical_form(
            B, k_totB, D, k_item + k,
            g_B, g_D,
            (II != NULL)? n_i : n, q, k_item + k,
            I_row, I_col, I_sp, nnz_I,
            II, !I_has_NA,
            I_csr_p, I_csr_i, I_csr,
            I_csc_p, I_csc_i, I_csc,
            false, false,
            (FPnum*)NULL, (FPnum*)NULL,
            (FPnum*)NULL, (FPnum*)NULL,
            (FPnum*)NULL, (FPnum*)NULL, (FPnum*)NULL,
            w_item,
            buffer_FPnum,
            buffer_mt,
            false,
            nthreads
        );

    /* if there are binary matrices with sigmoid transformation, need a
       different formula for the gradients */

    if (Ub != NULL)
        f += collective_fun_grad_bin(
            A, k_totA, Cb, k_user + k,
            g_A, g_Cb,
            Ub,
            m_ubin, pbin, k_user + k,
            !Ub_has_NA, w_user,
            buffer_FPnum,
            nthreads
        );

    if (Ib != NULL)
        f += collective_fun_grad_bin(
            B, k_totB, Db, k_item + k,
            g_B, g_Db,
            Ib,
            n_ibin, qbin, k_item + k,
            !Ib_has_NA, w_item,
            buffer_FPnum,
            nthreads
        );

    /* Now account for the regulatization parameter
        grad = grad + lambda * var
        f = f + (lambda / 2) * || var ||^2 */

    /* If all matrices have the same regulatization, can do it in one pass */
    if (lam_unique == NULL) {
        saxpy_large(values, lam, grad, nvars, nthreads);
        f += (lam / 2.) * sum_squares(values, nvars, nthreads);
    }

    /* otherwise, add it one by one */ 
    else {
        long double freg = 0;

        /* Note: Cbin is in memory right next to C, so there's not need to
           account for it separately - can be passed extra elements to C */

        if (user_bias) cblas_taxpy(m_max, lam_unique[0], biasA, 1, g_biasA, 1);
        if (item_bias) cblas_taxpy(n_max, lam_unique[1], biasB, 1, g_biasB, 1);
        saxpy_large(A, lam_unique[2],g_A,(size_t)m_max*(size_t)k_totA,nthreads);
        saxpy_large(B, lam_unique[3],g_B,(size_t)n_max*(size_t)k_totB,nthreads);

        if (U != NULL || U_sp != NULL || U_csr != NULL || Ub != NULL)
            saxpy_large(C, lam_unique[4], g_C,
                        (size_t)(p+pbin)*(size_t)(k_user+k), nthreads);
        if (II != NULL || I_sp != NULL || I_csr != NULL || Ib != NULL)
            saxpy_large(D, lam_unique[5], g_D,
                        (size_t)(q+qbin)*(size_t)(k_item+k), nthreads);

        if (user_bias)
            freg += (lam_unique[0] / 2.) * cblas_tdot(m_max, biasA, 1, biasA,1);
        if (item_bias)
            freg += (lam_unique[1] / 2.) * cblas_tdot(n_max, biasB, 1, biasB,1);
        freg += (lam_unique[2]/2.) * sum_squares(A,(size_t)m_max*(size_t)k_totA,
                                                 nthreads);
        freg += (lam_unique[3]/2.) * sum_squares(B,(size_t)n_max*(size_t)k_totB,
                                                 nthreads);
        if (U != NULL || U_sp != NULL || U_csr != NULL || Ub != NULL)
            freg += (lam_unique[4] / 2.)
                     * sum_squares(C, (size_t)(p+pbin)*(size_t)(k_user+k),
                                   nthreads);
        if (II != NULL || I_sp != NULL || I_csr != NULL || Ib != NULL)
            freg += (lam_unique[5] / 2.)
                     * sum_squares(D, (size_t)(q+qbin)*(size_t)(k_item+k),
                                   nthreads);
        f += (FPnum)freg;
    }

    return (FPnum) f;
}

/* This function calculates the gradient for squared error on
   sigmoid-transformed approximations */
FPnum collective_fun_grad_bin
(
    FPnum *restrict A, int lda, FPnum *restrict Cb, int ldc,
    FPnum *restrict g_A, FPnum *restrict g_Cb,
    FPnum *restrict Ub,
    int m, int pbin, int k,
    bool Ub_has_NA, double w_user,
    FPnum *restrict buffer_FPnum,
    int nthreads
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif
    double f = 0;

    /* Buffer = exp(-A * t(Cb)) */
    cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, pbin, k,
                1, A, lda, Cb, ldc,
                0, buffer_FPnum, pbin);
    exp_neg_x(buffer_FPnum, (size_t)m * (size_t)pbin, nthreads);

    /* f = sum_sq(Ub - 1/(1+Buffer))
       See explanation at the top for the gradient formula */
    if (Ub_has_NA)
    {
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(Ub, buffer_FPnum) reduction(+:f)
        for (size_t_for ix = 0; ix < (size_t)m*(size_t)pbin; ix++)
            f += (!isnan(Ub[ix]))?
                  square(Ub[ix] - 1./(1.+buffer_FPnum[ix])) : (0);
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(buffer_FPnum, m, pbin, Ub)
        for (size_t_for ix = 0; ix < (size_t)m*(size_t)pbin; ix++)
            buffer_FPnum[ix] = (!isnan(Ub[ix]))?
                                ( (1./(1.+buffer_FPnum[ix]) - Ub[ix])
                                  * buffer_FPnum[ix]
                                  / square(buffer_FPnum[ix]+1)
                                ) : (0);
    }

    else
    {
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(Ub, buffer_FPnum) reduction(+:f)
        for (size_t_for ix = 0; ix < (size_t)m*(size_t)pbin; ix++)
            f += square(Ub[ix] - 1./(1.+buffer_FPnum[ix]));
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(buffer_FPnum, m, pbin, Ub)
        for (size_t_for ix = 0; ix < (size_t)m*(size_t)pbin; ix++)
            buffer_FPnum[ix] = (
                                 (1./(1.+buffer_FPnum[ix]) - Ub[ix])
                                  * buffer_FPnum[ix]
                                  / square(buffer_FPnum[ix]+1)
                                );
    }

    f *= (w_user / 2);

    cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, k, pbin,
                w_user, buffer_FPnum, pbin, Cb, ldc,
                1., g_A, lda);
    cblas_tgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                pbin, k, m,
                w_user, buffer_FPnum, pbin, A, lda,
                0., g_Cb, ldc);

    return f;
}

/*******************************************************************************
    Function and Gradient for Single Row
    ------------------------------------

    This is a shorter version of the main function and gradient function
    applied to only one row of the A matrix. The purpose of this function
    is to be used for determining the optimal values of A through the L-BFGS
    solver in cases in which the closed-form is not possible to obtain.

    The formula is the exact same as in the larger case, with the only
    difference that the bias is not taken into consideration here, as it can be
    obtained by a simple average on the non-missing entries of X for this
    particular row, assuming no regularization applied on that bias.

    The X matrix can be passed either as a dense vector or as a sparse vector,
    and must have al ready been centered according to previously-fit model
    biases and the user/row bias to determine for this data.

    Note that this function does not attempt to exploit any parallelization
    as calculating the gradient for a single row is supposed to be very fast
    for typical problem sizes.
    
    Parameters
    ----------
    a_vec[k_user + k + k_main]
        The current values of the factors/variables for this row.
    g_A[k_user + k + k_main] (out)
        Array into which to write the gradient evaluated at 'a_vec'.
    k, k_user, k_item, k_main
        Dimensionality of the factorizing matrix. See description at the top
        for a better idea.
    u_vec[p]
        User attributes for this row, in dense format. If passing them in both
        dense and sparse formats, the dense one will be preferred. Pass NULL
        if no data is available or is passed as sparse.
    p
        The number of attributes for users.
    u_vec_ixB[nnz_u_vec], u_vec_sp[nnz_u_vec], nnz_u_vec
        User attributes for this row, in sparse format. Pass NULL if not
        available or if passed as dense.
    u_bin_vec[pbin]
        User binary attributes for this row. Pass NULL if the model had no
        binary user attributes.
    B[n*(k_item + k + k_main)]
        The B matrices with which a_vec is multiplied to approximate X.
    n
        The number of columns in X and number of rows in B.
    C[p * (k_user + k)]
        The C matrix used to approximate the U matrix in the model.
    Cb[pbin * (k_user + k)]
        The portion of the C matrix used to approximate the binary columns
        in the U matrix (Ub here).
    Xa[nnz], ixB[nnz], nnz
        The row of the X matrix for this row to evaluate the gradient, as
        a sparse vector. Pass NULL if not applicable.
    Xa_dense[n]
        The row of the X matrix in dense format, with missing entries as NAN.
        If passing it in both sparse and dense formats, the dense one will
        be preferred. Pass NULL if not applicable.
    weight[nnz or n]
        Observation weights for each non-missing entry in X. Must have the same
        shape as the X matrix that was passed - that is, if Xa is passed, must
        have 'nnz' entries, if Xa_dense is passed, must have 'n' entries.
        Pass NULL for uniform weights.
    buffer_FPnum[n or p or pbin]
        Temporary array into which to write values. Must be able to hold the
        largest dense vector which is passed. Pass NULL if all the input
        vectors are sparse.
    lam
        Regularization parameter applied to A.
    w_main
        Weight of the squared error of the X factorization.
    w_user
        Weight of the squared error of the U and Ub factorizations.

    Returns
    -------
    f : the function value evaluated at 'a_vec'.


*******************************************************************************/
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
)
{
    int ldb = k_item + k + k_main;
    int k_pred = k + k_main;
    set_to_zero(g_A, k_user + k + k_main, 1);
    FPnum f = 0;


    FPnum err;
    FPnum *restrict a_vec_pred = a_vec + k_user;
    FPnum *restrict g_A_pred = g_A + k_user;
    FPnum *restrict Bm = B + k_item;
    if (Xa_dense == NULL && Xa != NULL) /* sparse X */
    {
        for (size_t ix = 0; ix < nnz; ix++) {
            err = cblas_tdot(k_pred, a_vec_pred, 1,
                             Bm + (size_t)ixB[ix]*(size_t)ldb, 1)
                  - Xa[ix];
            f += square(err) * ((weight != NULL)? weight[ix] : 1);
            err *= (weight != NULL)? weight[ix] : 1;
            cblas_taxpy(k_pred, err, Bm + (size_t)ixB[ix]*(size_t)ldb, 1,
                        g_A_pred, 1);
        }
        f *= w_main / 2.;
        if (w_main != 1.)
            cblas_tscal(k_pred, w_main, g_A_pred, 1);
    }

    else if (Xa_dense != NULL) /* dense X */
    {
        memcpy(buffer_FPnum, Xa_dense, n*sizeof(FPnum));
        cblas_tgemv(CblasRowMajor, CblasNoTrans,
                    n, k + k_main,
                    1, B + k_item, ldb,
                    a_vec + k_user, 1,
                    -1, buffer_FPnum, 1);

        if (weight != NULL)
            mult_if_non_nan(buffer_FPnum, Xa_dense, weight, n, 1);
        else
            nan_to_zero(buffer_FPnum, Xa_dense, n, 1);

        cblas_tgemv(CblasRowMajor, CblasTrans,
                    n, k + k_main,
                    w_main, B + k_item, ldb,
                    buffer_FPnum, 1,
                    0, g_A + k_user, 1);

        if (weight == NULL)
            f = (w_main / 2.) * cblas_tdot(n, buffer_FPnum, 1, buffer_FPnum, 1);
        else
            f = (w_main / 2.) * sum_sq_div_w(buffer_FPnum, weight, n, false, 1);
    }

    if (u_vec != NULL)
    {
        memcpy(buffer_FPnum, u_vec, p*sizeof(FPnum));
        cblas_tgemv(CblasRowMajor, CblasNoTrans,
                    p, k_user + k,
                    1., C, k_user + k,
                    a_vec, 1,
                    -1., buffer_FPnum, 1);
        if (u_vec_has_NA)
            nan_to_zero(buffer_FPnum, u_vec, p, 1);

        cblas_tgemv(CblasRowMajor, CblasTrans,
                    p, k_user + k,
                    w_user, C, k_user + k,
                    buffer_FPnum, 1,
                    1., g_A, 1);

        f += (w_user / 2.) * cblas_tdot(p, buffer_FPnum, 1, buffer_FPnum, 1);
    }

    else if (u_vec_sp != NULL)
    {
        FPnum err_sp = 0;
        k_pred = k_user + k;
        for (size_t ix = 0; ix < nnz_u_vec; ix++) {
            err = cblas_tdot(k_pred, a_vec, 1,
                             C + (size_t)u_vec_ixB[ix]*(size_t)k_pred, 1)
                  - u_vec_sp[ix];
            err_sp += square(err);
            cblas_taxpy(k_pred, w_user*err,
                        C + (size_t)u_vec_ixB[ix]*(size_t)k_pred, 1, g_A,1);
        }
        f += (w_user / 2.) * err_sp;
    }

    if (u_bin_vec != NULL)
    {
        FPnum err_bin = 0;
        cblas_tgemv(CblasRowMajor, CblasNoTrans,
                    pbin, k_user + k,
                    1, Cb, k_user + k,
                    a_vec, 1,
                    0, buffer_FPnum, 1);
        exp_neg_x(buffer_FPnum, (size_t)pbin, 1);

        if (u_bin_vec_has_NA)
        {
            for (int ix = 0; ix < pbin; ix++)
                err_bin += (!isnan(u_bin_vec[ix]))?
                            square(1./(1.+buffer_FPnum[ix]) - u_bin_vec[ix])
                            : (0);
            for (int ix = 0; ix < pbin; ix++)
                buffer_FPnum[ix] = (!isnan(u_bin_vec[ix]))?
                                    ( (1./(1.+buffer_FPnum[ix]) - u_bin_vec[ix])
                                       * buffer_FPnum[ix]
                                       / square(buffer_FPnum[ix]+1) )
                                    : (0);
        }

        else
        {
            for (int ix = 0; ix < pbin; ix++)
                err_bin += square(1./(1.+buffer_FPnum[ix]) - u_bin_vec[ix]);
            for (int ix = 0; ix < pbin; ix++)
                buffer_FPnum[ix] = (1./(1.+buffer_FPnum[ix]) - u_bin_vec[ix])
                                    * buffer_FPnum[ix] 
                                    / square(buffer_FPnum[ix]+1);
        }

        cblas_tgemv(CblasRowMajor, CblasTrans,
                    pbin, k_user + k,
                    w_user, Cb, k_user + k,
                    buffer_FPnum, 1,
                    1, g_A, 1);
        f += (w_user / 2.) * err_bin;
    }


    f += (lam / 2.) * cblas_tdot(k_user+k+k_main, a_vec, 1, a_vec, 1);
    cblas_taxpy(k_user+k+k_main, lam, a_vec, 1, g_A, 1);
    if (lam_last != lam && k_main) {
        f += (lam_last-lam)/2. * square(a_vec[k_user+k+k_main-1]);
        g_A[k_user+k+k_main-1] += (lam_last-lam) * a_vec[k_user+k+k_main-1];
    }
    return (FPnum) f;
}

/* These functions find the optimal values for a single row of A using the
   gradient function above, passing it to the L-BFGS solver */
FPnum wrapper_factors_fun_grad
(
    void *instance,
    lbfgsFPnumval_t *x,
    lbfgsFPnumval_t *g,
    const size_t n,
    const lbfgsFPnumval_t step
)
{
    data_factors_fun_grad *data = (data_factors_fun_grad*)instance;
    return collective_fun_grad_single(
        x, g,
        data->k, data->k_user, data->k_item, data->k_main,
        data->u_vec, data->p,
        data->u_vec_ixB, data->u_vec_sp, data->nnz_u_vec,
        data->u_bin_vec, data->pbin,
        data->u_vec_has_NA, data->u_bin_vec_has_NA,
        data->B, data->n,
        data->C, data->Cb,
        data->Xa, data->ixB, data->nnz,
        data->Xa_dense,
        data->weight,
        data->buffer_FPnum,
        data->lam, data->w_main, data->w_user, data->lam_last
    );
}

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
)
{
    data_factors_fun_grad data = {
        k, k_user, k_item, k_main,
        u_vec, p,
        u_vec_ixB, u_vec_sp, nnz_u_vec,
        u_bin_vec, pbin,
        u_vec_has_NA, u_bin_vec_has_NA,
        B, n,
        C, Cb,
        Xa, ixB, weight, nnz,
        Xa_dense,
        buffer_FPnum,
        lam, w_main, w_user, lam_last
    };

    lbfgs_parameter_t lbfgs_params = {
        5, 1e-5, 0, 1e-5,
        250, LBFGS_LINESEARCH_MORETHUENTE, 20,
        1e-20, 1e20, 1e-4, 0.9, 0.9, 1.0e-16,
        0.0, 0, -1,
    };

    lbfgs_progress_t callback = (lbfgs_progress_t)NULL;
    size_t nvars = k_user + k + k_main;

    /* Starting point can be set to zero, since the other matrices
       are already at their local optima. */
    set_to_zero(a_vec, nvars, 1);

    int retval = lbfgs(
        nvars,
        a_vec,
        (FPnum*)NULL,
        wrapper_factors_fun_grad,
        callback,
        (void*) &data,
        &lbfgs_params,
        (lbfgsFPnumval_t*)NULL,
        (iteration_data_t*)NULL
    );

    if (retval == LBFGSERR_OUTOFMEMORY)
        return 1;
    else
        return 0;
}


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
)
{
    int k_tot = k_user + k + k_main;
    int k_totB = k_item + k + k_main + padding;
    FPnum *restrict bufferBeTBe = buffer_FPnum;
    FPnum *restrict bufferBw = bufferBeTBe + square(k_tot);
    FPnum *restrict bufferCw = bufferBeTBe + (square(k_tot)
                                + ((Xa_dense != NULL && weight == NULL)?
                                   ((size_t)n*(size_t)(k+k_main)) : 0));
    set_to_zero(bufferBeTBe, square(k_tot), 1);
    if (add_X && add_U)
        set_to_zero(a_vec, k_tot, 1);
    else if (add_U)
        set_to_zero(a_vec, k_user, 1);
    else if (add_X)
        set_to_zero(a_vec + k_user+k, k_main, 1);

    /* =================== Part 1 =====================
       Constructing t(Be)*Be, upper-left square (from C) */
    if (u_vec != NULL || NA_as_zero_U) /* Dense u_vec */
    {
        /* If it's full or near full, can use the precomputed matrix
           and subtract missing entries from it if necessary */
        if ((FPnum)cnt_NA_u < (FPnum)p*0.1 || NA_as_zero_U)
        {
            if (precomputedCtCw != NULL) {
                sum_mat(k_user+k, k_user+k,
                        precomputedCtCw, k_user+k,
                        bufferBeTBe, k_tot);
            } else {
                cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                            k_user+k, p,
                            w_user, C, k_user+k,
                            0., bufferBeTBe, k_tot);
            }
            if (cnt_NA_u > 0 && u_vec != NULL)
                for (size_t ix = 0; ix < (size_t)p; ix++)
                    if (isnan(u_vec[ix]))
                    {
                        cblas_tsyr(CblasRowMajor, CblasUpper,
                                   k_user+k, -w_user,
                                   C + ix*(size_t)(k_user+k), 1,
                                   bufferBeTBe, k_tot);
                        u_vec[ix] = 0;
                    }
        }

        /* Otherwise, will need to construct it one-by-one */
        else
        {
            nnz_u_vec = 0;
            for (size_t ix = 0; ix < (size_t)p; ix++)
                if (isnan(u_vec[ix]))
                    u_vec[ix] = 0;
                else
                    memcpy(bufferCw + (nnz_u_vec++)*((size_t)(k_user+k)),
                           C + ix*(size_t)(k_user+k),
                           (k_user+k)*sizeof(FPnum));
            cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        k_user+k, (int)nnz_u_vec,
                        w_user, bufferCw, k_user+k,
                        0., bufferBeTBe, k_tot);
        }
    }

    else /* Sparse u_vec */
    {
        for (size_t ix = 0; ix < nnz_u_vec; ix++)
            cblas_tsyr(CblasRowMajor, CblasUpper,
                       k_user+k, w_user,
                       C + (size_t)u_vec_ixB[ix]*(size_t)(k_user+k), 1,
                       bufferBeTBe, k_tot);
    }

    /* =================== Part 2 ======================
       Constructing t(Be)*Be, lower-right square (from B) */
    if (
        (Xa_dense != NULL || NA_as_zero_X) &&
        (Xa_dense == NULL || weight == NULL) &&
        ((Xa_dense == NULL && NA_as_zero_X) || (FPnum)cnt_NA_x < (FPnum)n*0.1)
        )
    {
        if (precomputedBtBw != NULL) {
            sum_mat(k+k_main, k+k_main,
                    precomputedBtBw
                      + k_item_BtB 
                      + k_item_BtB*(k_item_BtB + k + k_main),
                    k_item_BtB + k + k_main,
                    bufferBeTBe + k_user + k_user*k_tot, k_tot);
        } else {
            cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        k+k_main, n,
                        w_main, B + k_item, k_totB,
                        1., bufferBeTBe + k_user + k_user*k_tot, k_tot);
        }

        if (cnt_NA_x > 0 && Xa_dense != NULL) {
            for (size_t ix = 0; ix < (size_t)n; ix++)
                if (isnan(Xa_dense[ix]))
                {
                    cblas_tsyr(CblasRowMajor, CblasUpper,
                               k+k_main, -w_main,
                               B + (size_t)k_item + ix*(size_t)k_totB, 1,
                               bufferBeTBe + k_user + k_user*k_tot, k_tot);
                    Xa_dense[ix] = 0;
                }
        }

        else if (Xa_dense == NULL && NA_as_zero_X && weight != NULL) {
            for (size_t ix = 0; ix < nnz; ix++)
                cblas_tsyr(CblasRowMajor, CblasUpper,
                           k+k_main, w_main * (weight[ix]-1.),
                           B + (size_t)k_item +(size_t)ixB[ix]*(size_t)k_totB,1,
                           bufferBeTBe + k_user + k_user*k_tot, k_tot);
        }
    }

    else if (Xa_dense != NULL)
    {
        if (weight == NULL) {
            nnz = 0;
            for (size_t ix = 0; ix < (size_t)n; ix++)
                if (isnan(Xa_dense[ix]))
                    Xa_dense[ix] = 0;
                else
                    memcpy(bufferBw + (nnz++)*((size_t)(k+k_main)),
                           B + (size_t)k_item + ix*(size_t)k_totB,
                           (k+k_main)*sizeof(FPnum));
            cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        k+k_main, (int)nnz,
                        w_main, bufferBw, k+k_main,
                        1., bufferBeTBe + k_user + k_user*k_tot, k_tot);
        }

        else {
            for (size_t ix = 0; ix < (size_t)n; ix++)
                if (isnan(Xa_dense[ix]))
                    Xa_dense[ix] = 0;
                else {
                    Xa_dense[ix] *= weight[ix];
                    cblas_tsyr(CblasRowMajor, CblasUpper,
                               k+k_main, weight[ix]*w_main,
                               B + (size_t)k_item + ix*(size_t)k_totB, 1,
                               bufferBeTBe + k_user + k_user*k_tot, k_tot);
                }
        }
    }

    else /* Sparse Xa - this is the expected scenario for most use-cases */
    {
        for (size_t ix = 0; ix < nnz; ix++)
            cblas_tsyr(CblasRowMajor, CblasUpper,
                       k+k_main,
                       (weight == NULL)? w_main : w_main*weight[ix],
                       B + (size_t)k_item + (size_t)ixB[ix]*(size_t)k_totB, 1,
                       bufferBeTBe + k_user + k_user*k_tot, k_tot);
    }


    /* ================ Part 3 =================
       Constructing Be*t(Xe), upper part (from X) */
    if (add_X)
    {
        if (Xa_dense != NULL)
        {
            cblas_tgemv(CblasRowMajor, CblasTrans,
                        n, k+k_main,
                        w_main, B + k_item, k_totB, Xa_dense, 1,
                        add_U? 0. : 1., a_vec + k_user, 1);
            /* Note: 'Xa_dense' was already multiplied by the weights earlier */
        }

        else
        {
            if (weight == NULL) {
                sgemv_dense_sp(n, k+k_main,
                               w_main, B + k_item, (size_t)k_totB,
                               ixB, Xa, nnz,
                               a_vec + k_user);
            }

            else {
                sgemv_dense_sp_weighted2(n, k+k_main,
                                         weight, w_main,
                                         B + k_item, (size_t)k_totB,
                                         ixB, Xa, nnz,
                                         a_vec + k_user);
            }
        }
    }

    /* ================ Part 4 =================
       Constructing Be*t(Xe), lower part (from U) */
    if (add_U)
    {
        if (u_vec != NULL)
            cblas_tgemv(CblasRowMajor, CblasTrans,
                        p, k_user+k,
                        w_user, C, k_user+k, u_vec, 1,
                        1., a_vec, 1);
        else
            sgemv_dense_sp(p, k_user+k,
                           w_user, C, (size_t)k_user+(size_t)k,
                           u_vec_ixB, u_vec_sp, nnz_u_vec,
                           a_vec);
    }

    /* =================== Part 5 ======================
       Solving A = inv(t(Be)*Be + diag(lam)) * (Be*t(Xe)) */

    add_to_diag(bufferBeTBe, lam, k_tot);
    if (lam_last != lam) bufferBeTBe[square(k_tot)-1] += (lam_last - lam);
    char uplo = 'L';
    int one = 1;
    int ignore;

    if (!use_cg)
        tposv_(&uplo, &k_tot, &one,
               bufferBeTBe, &k_tot,
               a_vec, &k_tot,
               &ignore);
    else
        solve_conj_grad(
            bufferBeTBe, a_vec, k_tot,
            bufferBeTBe + square(k_tot)
        );
}

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
)
{
    char uplo = 'L';
    int one = 1;
    int ignore;
    int k_totA = k_user + k + k_main;
    size_t k_totB = k_item + k + k_main;
    int k_totC = k_user + k;
    alpha *= w_main;
    bool few_NAs = (u_vec != NULL && (FPnum)cnt_NA_u < 0.1*(FPnum)p);
    if (add_U || cnt_NA_u > 0) set_to_zero(a_vec, k_totA, 1);


    FPnum *restrict BtB = buffer_FPnum;
    bool add_C = false;
    if ((u_vec != NULL && few_NAs) || (u_vec == NULL && NA_as_zero_U)) {
        if (precomputedBeTBe != NULL)
            memcpy(BtB, precomputedBeTBe, square(k_totA)*sizeof(FPnum));
        else {
            set_to_zero(BtB, square(k_totA), 1);
            cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        k+k_main, n,
                        w_main, B + k_item, k_totB,
                        0., BtB + k_user + k_user*k_totA, k_totA);
            cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        k_user+k, p,
                        w_user, C, k_totC,
                        1., BtB, k_totA);
            add_to_diag(BtB, lam, k_totA);
        }
    }
    else {
        add_C = true;
        if (shapes_match && precomputedBtB != NULL)
            memcpy(BtB, precomputedBtB, square(k_totA)*sizeof(FPnum));
        else if (precomputedBtB != NULL) {
            set_to_zero(BtB, square(k_totA), 1);
            copy_mat(k+k_main, k+k_main,
                     precomputedBtB + k_item + k_item*k_totB, k_totB,
                     BtB + k_user + k_user*k_totA, k_totA);
            add_to_diag(BtB, lam, k_totA);
        } else {
            cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        k+k_main, n,
                        w_main, B + k_item, k_totB,
                        0., BtB + k_user + k_user*k_totA, k_totA);
            add_to_diag(BtB, lam, k_totA);
        }
    }

    /* t(Be)*Be, upper-left square (from C)
            AND
       Be*t(Xe), lower part (from U)  */
    if (u_vec == NULL)
    {
        if (add_C)
            for (size_t ix = 0; ix < nnz_u_vec; ix++)
                cblas_tsyr(CblasRowMajor, CblasUpper,
                           k_user+k, w_user,
                           C + (size_t)u_vec_ixB[ix]*(size_t)k_totC, 1,
                           BtB, k_totA);
        if (add_U)
            sgemv_dense_sp(p, k_user+k,
                           w_user, C, (size_t)k_totC,
                           u_vec_ixB, u_vec_sp, nnz_u_vec,
                           a_vec);
    }

    else
    {
        if (few_NAs && cnt_NA_u > 0 && !add_C)
            for (size_t ix = 0; ix < (size_t)p; ix++) {
                if (isnan(u_vec[ix])) {
                    cblas_tsyr(CblasRowMajor, CblasUpper,
                               k_totC, -w_user,
                               C + ix*(size_t)k_totC, 1,
                               BtB, k_totA);
                    u_vec[ix] = 0;
                }
            }
        else if (add_C)
            for (size_t ix = 0; ix < (size_t)p; ix++) {
                if (!isnan(u_vec[ix]))
                    cblas_tsyr(CblasRowMajor, CblasUpper,
                               k_totC, w_user,
                               C + ix*(size_t)k_totC, 1,
                               BtB, k_totA);
                else
                    u_vec[ix] = 0;
            }
        else if (cnt_NA_u > 0)
            for (int ix = 0; ix < p; ix++)
                u_vec[ix] = (!isnan(u_vec[ix]))? (u_vec[ix]) : (0);

        if (add_U || cnt_NA_u > 0)
            cblas_tgemv(CblasRowMajor, CblasTrans,
                        p, k_user+k,
                        w_user, C, k_user+k, u_vec, 1,
                        0., a_vec, 1);
    }

    /* t(Be)*Be, lower-right square (from B)
            AND
       Be*t(Xe), upper part (from X) */
    for (size_t ix = 0; ix < nnz; ix++) {
        cblas_tsyr(CblasRowMajor, CblasUpper, k+k_main,
                   alpha*Xa[ix],
                   B + (size_t)k_item + (size_t)ixB[ix]*k_totB, 1,
                   BtB + k_user + k_user*k_totA, k_totA);
        cblas_taxpy(k + k_main, alpha*Xa[ix] + w_main,
                    B + (size_t)k_item + (size_t)ixB[ix]*k_totB, 1,
                    a_vec + k_user, 1);
    }

    if (!use_cg)
        tposv_(&uplo, &k_totA, &one,
               BtB, &k_totA,
               a_vec, &k_totA,
               &ignore);
    else
        solve_conj_grad(
            BtB, a_vec, k_totA,
            buffer_FPnum + square(k_totA)
        );
}

/*******************************************************************************
    Cold-Start Predictions
    ----------------------

    This function aims at determining the optimal values of a single row of the
    A matrix given only information about U and/or Ubin, with no data for X.
    The same function works for both implicit and explicit feedback cases
    (given that there's no X vector).

    The intended purpose is for cold-start recommendations, which are then
    obtained by multiplying the obtained vector with the B matrix.

    If there are no binary variables, it's possible to use the closed form
    solution as explained at the top of this file, otherwise it's posssible
    to use a gradient-based approach with the function that calculates the
    gradient for a single observation.

    Note that the values for U passed to this function must already be centered
    by columns (if this step was performed when fitting the model).

    See documentation of the single-row gradient function for details
    about the input parameters.

    This function can be sped-up using precomputed multiplications of C:
    (a) inv(t(C)*C + diag(lam))*t(C), if passing a full u_vec with no NAs.
    (b) t(C)*C+diag(lam), if passing u_vec with <= 10% NAs.
    (c) t(C)*C+diag(lam), if passing sparse u_vec with 'NA_as_zero_U=true'
        (this is a different model formulation from the others)

    Will return  0 (zero) if everything went correctly, one (1) if it
    ran out of memory, (2) if the parameters were invalid (basically, cannot
    have 'NA_as_zero_U=true' if there's u_bin_vec and/or Cb.

    The obtained factors will be available in 'a_vec', while the obtained bias
    will be available in 'a_bias'.


*******************************************************************************/
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
)
{
    if (NA_as_zero_U && u_bin_vec != NULL) return 2;
    int cnt_NA_u_vec = 0;
    int cnt_NA_u_bin_vec = 0;
    if (u_vec != NULL || (u_vec_sp != NULL && !NA_as_zero_U))
        preprocess_vec(u_vec, p, u_vec_ixB, u_vec_sp, nnz_u_vec,
                       0., 0., col_means, (FPnum*)NULL, &cnt_NA_u_vec);
    if (u_bin_vec != NULL)
        cnt_NA_u_bin_vec = count_NAs(u_bin_vec, (size_t)pbin, 1);

    if (k_main > 0)
        set_to_zero(a_vec + k_user+k, k_main, 1);

    FPnum *restrict buffer_FPnum = NULL;
    size_t size_buffer = 0;

    /* If there are no binary variables, solution can be obtained through
       closed form */
    if (u_bin_vec == NULL)
    {
        if (CtCinvCt == NULL || u_vec == NULL || cnt_NA_u_vec > 0) {
            size_buffer = square(k_user + k);
            if (u_vec != NULL && cnt_NA_u_vec > 0)
                size_buffer += (size_t)p * (size_t)(k_user + k);
            buffer_FPnum = (FPnum*)malloc(size_buffer * sizeof(FPnum));
            if (buffer_FPnum == NULL) return 1;
        }

        factors_closed_form(a_vec, k_user + k,
                            C, p, k_user + k,
                            u_vec, cnt_NA_u_vec==0,
                            u_vec_sp, u_vec_ixB, nnz_u_vec,
                            (FPnum*)NULL,
                            buffer_FPnum, lam, w_user, lam,
                            CtCinvCt, CtCw, cnt_NA_u_vec, 0,
                            CtCchol, NA_as_zero_U, false,
                            true);
        if (buffer_FPnum != NULL) free(buffer_FPnum);
        return 0;
    }

    else
    {
        /* Otherwise, need to take a gradient-based approach with a solver. */
        buffer_FPnum = (FPnum*)malloc(max2(p, pbin)*sizeof(FPnum));
        if (buffer_FPnum == NULL) return 1;

        int retval = collective_factors_lbfgs(
            a_vec,
            k, k_user, 0, 0,
            u_vec, p,
            u_vec_ixB, u_vec_sp, nnz_u_vec,
            u_bin_vec, pbin,
            cnt_NA_u_vec>0, cnt_NA_u_bin_vec>0,
            (FPnum*)NULL, 0,
            C, Cb,
            (FPnum*)NULL, (int*)NULL, (FPnum*)NULL, 0,
            (FPnum*)NULL,
            buffer_FPnum,
            lam, 1., w_user, lam
        );
        free(buffer_FPnum);
        return retval;
    }
}

/*******************************************************************************
    Warm-Start Predictions
    ----------------------

    Note that the values for U passed to this function must already be centered
    by columns (if this step was performed when fitting the model).

    See documentation of the single-row gradient function for details
    about the input parameters.

    Additionally:
    - If using user bias, must pass a pointer to a single value in which to
      output it ('a_bias').
    - Must pass the column means for 'U' if it is passed, as it will center
      them here.
    - The values for X and U will be modified in-place.

    Will return  0 (zero) if everything went correctly, one (1) if it ran
    out of memory, two (2) if the parameter combinations were invalid
    (basically, cannot pass NAs as zero if there is Ub or Cb).


*******************************************************************************/
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
)
{
    if (u_bin_vec != NULL && (NA_as_zero_X || NA_as_zero_U))
        return 2;

    int cnt_NA_u_vec = 0;
    int cnt_NA_u_bin_vec = 0;
    int cnt_NA_x = 0;
    bool append_bias = (B_plus_bias != NULL && a_bias != NULL);
    if (u_bin_vec != NULL)
        cnt_NA_u_bin_vec = count_NAs(u_bin_vec, (size_t)pbin, 1);
    if (u_vec != NULL || (u_vec_sp != NULL && !NA_as_zero_U))
        preprocess_vec(u_vec, p, u_vec_ixB, u_vec_sp, nnz_u_vec,
                       0., 0., col_means, (FPnum*)NULL, &cnt_NA_u_vec);
    if (!NA_as_zero_X || Xa_dense != NULL)
        preprocess_vec(Xa_dense, n, ixB, Xa, nnz,
                       glob_mean, lam_bias, biasB,
                       (B_plus_bias == NULL)? a_bias : (FPnum*)NULL,
                       &cnt_NA_x);

    FPnum *restrict buffer_FPnum = NULL;
    size_t size_buffer;
    int retval = 0;

    FPnum *restrict a_plus_bias = NULL;
    if (append_bias) {
        a_plus_bias = (FPnum*)malloc((k_user+k+k_main+1)*sizeof(FPnum));
        if (a_plus_bias == NULL) { retval = 1; goto cleanup; }
    }

    /* If there's no side info, just need to apply the closed-form
       on the X data */
    if (u_vec == NULL && u_vec_sp == NULL && u_bin_vec == NULL)
    {
        if (BtBinvBt == NULL || Xa_dense == NULL ||
            cnt_NA_x > 0 || weight != NULL || NA_as_zero_X)
        {
            size_buffer = square(k + k_main + append_bias);
            if (Xa_dense != NULL || (NA_as_zero_X && weight != NULL)) {
                if (cnt_NA_x > 0 || k_item != 0 || weight != NULL)
                    size_buffer += (size_t)n * (size_t)(k + k_main+append_bias);
            }
            buffer_FPnum = (FPnum*)malloc(size_buffer*sizeof(FPnum));
            if (buffer_FPnum == NULL) return 1;
        }

        /* Small detail: the closed-form doesn't allow a general weight,
           only the regulatization is controlled. If scaling everything
           so that the main weight is 1, then the closed form will match
           with the desired hyperparameters. */
        // lam /= w_main;
        if (k_user > 0) {
            if (a_plus_bias == NULL)
                set_to_zero(a_vec, k_user, 1);
            else
                set_to_zero(a_plus_bias, k_user, 1);
        }
        if (!append_bias)
            factors_closed_form(a_vec + k_user, k+k_main,
                                B + k_item, n, k_item+k+k_main,
                                Xa_dense, cnt_NA_x==0,
                                Xa, ixB, nnz,
                                weight,
                                buffer_FPnum,
                                lam, w_main, lam,
                                BtBinvBt, BtBw, cnt_NA_x, k_item_BtB,
                                BtBchol, NA_as_zero_X, false,
                                true);
        else
            factors_closed_form(a_plus_bias + k_user, k+k_main+1,
                                B_plus_bias + k_item, n, k_item+k+k_main+1,
                                Xa_dense, cnt_NA_x==0,
                                Xa, ixB, nnz,
                                weight,
                                buffer_FPnum,
                                lam, w_main, lam_bias,
                                BtBinvBt, BtBw, cnt_NA_x, k_item_BtB,
                                BtBchol, NA_as_zero_X, false,
                                true);
        retval = 0;
    }

    /* If there are binary variables, there's no closed form solution,
       so it will follow a gradient-based optimization approach with
       the L-BFGS solver */
    else if (u_bin_vec != NULL)
    {
        size_t size_buffer = max2(p, pbin);
        if (Xa_dense != NULL)
            size_buffer = max2(size_buffer, (size_t)n);
        buffer_FPnum = (FPnum*)malloc(size_buffer*sizeof(FPnum));
        if (buffer_FPnum == NULL) return 1;

        if (!append_bias)
            retval = collective_factors_lbfgs(
                a_vec,
                k, k_user, k_item, k_main,
                u_vec, p,
                u_vec_ixB, u_vec_sp, nnz_u_vec,
                u_bin_vec, pbin,
                cnt_NA_u_vec>0, cnt_NA_u_bin_vec>0,
                B, n,
                C, Cb,
                Xa, ixB, weight, nnz,
                Xa_dense,
                buffer_FPnum,
                lam, w_main, w_user, lam
            );
        else
            retval = collective_factors_lbfgs(
                a_plus_bias,
                k, k_user, k_item, k_main+1,
                u_vec, p,
                u_vec_ixB, u_vec_sp, nnz_u_vec,
                u_bin_vec, pbin,
                cnt_NA_u_vec>0, cnt_NA_u_bin_vec>0,
                B_plus_bias, n,
                C, Cb,
                Xa, ixB, weight, nnz,
                Xa_dense,
                buffer_FPnum,
                lam, w_main, w_user, lam_bias
            );
    }

    /* If there's no binary data, can apply the closed form on extended block
       matrices Xe and Be, whose composition differs according to the
       independent components */
    else
    {
        size_buffer = square(k_user+k+k_main+(int)append_bias);
        bool alloc_B = false;
        bool alloc_C = false;
        if (Xa_dense != NULL && weight == NULL) alloc_B = true;
        if (u_vec != NULL && (FPnum)cnt_NA_u_vec >= (FPnum)p*0.1) alloc_C=true;
        size_buffer += (k+k_main)
                        * ((alloc_B && alloc_C)?
                            max2(p, n) : (alloc_B? n : (alloc_C? p : 0)));
        buffer_FPnum = (FPnum*)malloc(size_buffer*sizeof(FPnum));
        if (buffer_FPnum == NULL) return 1;

        if (!append_bias)
            collective_closed_form_block(
                a_vec,
                k, k_user, k_item, k_main, k_item_BtB, 0,
                Xa_dense,
                Xa, ixB, nnz,
                u_vec_ixB, u_vec_sp, nnz_u_vec,
                u_vec,
                NA_as_zero_X, NA_as_zero_U,
                B, n,
                C, p,
                weight,
                lam, w_user, w_main, lam,
                BtBw, cnt_NA_x,
                CtCw, cnt_NA_u_vec,
                true, true, false,
                buffer_FPnum
            );
        else
            collective_closed_form_block(
                a_plus_bias,
                k, k_user, k_item, k_main+1, k_item_BtB, 0,
                Xa_dense,
                Xa, ixB, nnz,
                u_vec_ixB, u_vec_sp, nnz_u_vec,
                u_vec,
                NA_as_zero_X, NA_as_zero_U,
                B_plus_bias, n,
                C, p,
                weight,
                lam, w_user, w_main, lam_bias,
                BtBw, cnt_NA_x,
                CtCw, cnt_NA_u_vec,
                true, true, false,
                buffer_FPnum
            );
        retval = 0;
    }

    if (append_bias) {
        memcpy(a_vec, a_plus_bias, (k_user+k+k_main)*sizeof(FPnum));
        *a_bias = a_plus_bias[k_user+k+k_main];
    }

    cleanup:
        free(buffer_FPnum);
        free(a_plus_bias);
    return retval;
}

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
)
{
    int k_totA = k_user + k + k_item;
    FPnum *restrict buffer_FPnum = (FPnum*)malloc(square(k_totA)*sizeof(FPnum));
    if (buffer_FPnum == NULL) return 1;

    w_main /= w_main_multiplier;

    int cnt_NA_u_vec = 0;
    if (u_vec != NULL || (u_vec_sp != NULL && !NA_as_zero_U)) {
        preprocess_vec(u_vec, p, u_vec_ixB, u_vec_sp, nnz_u_vec,
                       0., 0., col_means, (FPnum*)NULL, &cnt_NA_u_vec);

        collective_closed_form_block_implicit(
            a_vec,
            k, k_user, k_item, k_main,
            B, n, C, p,
            Xa, ixB, nnz,
            u_vec, cnt_NA_u_vec,
            u_vec_sp, u_vec_ixB, nnz_u_vec,
            NA_as_zero_U,
            lam, alpha, w_main, w_user,
            precomputedBeTBe,
            precomputedBtB,
            true, true, false,
            buffer_FPnum
        );
    }

    else {
        set_to_zero(a_vec, k_user, 1);
        factors_implicit(
            a_vec + k_user, k+k_main,
            B + k_item, (size_t)(k_item+k+k_main),
            Xa, ixB, nnz,
            lam/w_main, alpha,
            precomputedBtB_shrunk, 0,
            true, false,
            buffer_FPnum,
            true
        );
    }

    free(buffer_FPnum);
    return 0;
}

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
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ia;
    #endif
    int k_totA = k_user + k + k_main + padding;
    int k_totB = k_item + k + k_main + padding;
    int k_totC = k_user + k;
    int k_totX = k + k_main;
    int m_max = max2(m, m_u);

    FPnum f = 0.;
    FPnum err;
    size_t ib;

    set_to_zero(g_A, (size_t)m_max*(size_t)k_totA, nthreads);

    if (Xfull != NULL)
    {
        if (!do_B)
            f = fun_grad_Adense(
                    g_A + k_user,
                    A + k_user, k_totA,
                    B + k_item, k_totB,
                    m, n, k + k_main,
                    Xfull, weight,
                    0., w_main, 0.,
                    false, true,
                    nthreads,
                    buffer_FPnum
                );
        else
            f = fun_grad_Adense(
                    g_A + k_user,
                    B + k_item, k_totB,
                    A + k_user, k_totA,
                    n, m, k + k_main,
                    Xfull, weight,
                    0., w_main, 0.,
                    true, true,
                    nthreads,
                    buffer_FPnum
                );
    }

    else
    {
        FPnum *restrict Ax = A + k_user;
        FPnum *restrict Bx = B + k_item;
        FPnum *restrict g_Ax = g_A + k_user;
        FPnum err_row = 0;
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                shared(Xcsr_p, Xcsr_i, Xcsr, Ax, Bx, g_Ax, \
                       k_totA, k_totB, weight, w_main) \
                private(ib, err) firstprivate(err_row) reduction(+:f)
        for (size_t_for ia = 0; ia < (size_t)m; ia++)
        {
            err_row = 0;
            for (int ix = Xcsr_p[ia]; ix < Xcsr_p[ia+1]; ix++)
            {
                ib = (size_t)Xcsr_i[ix];
                err = cblas_tdot(k_totX, Ax + ia*(size_t)k_totA, 1,
                                 Bx + ib*(size_t)k_totB, 1)
                       - Xcsr[ix];
                err_row += square(err) * ((weight == NULL)? 1. : weight[ix]);
                err *= w_main * ((weight == NULL)? 1. : weight[ix]);
                cblas_taxpy(k_totX, err, Bx + ib*(size_t)k_totB, 1,
                            g_Ax + ia*(size_t)k_totA,1);
            }
            f += err_row;
        }
        f *= w_main / 2.;
    }

    if (U != NULL)
    {
        f += fun_grad_Adense(
                    g_A,
                    A, k_totA,
                    C, k_totC,
                    m_u, p, k_user + k,
                    U, (FPnum*)NULL,
                    0., w_user, 0.,
                    false, false,
                    nthreads,
                    buffer_FPnum
             );
    }

    else
    {
        FPnum f_user = 0;
        FPnum err_row = 0;
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                shared(U_csr_p, U_csr_i, U_csr, A, C, \
                       g_A, k_totA, k_totC, w_user) \
                private(ib, err, err_row) reduction(+:f_user)
        for (size_t_for ia = 0; ia < (size_t)m_u; ia++)
        {
            err_row = 0;
            for (int ix = U_csr_p[ia]; ix < U_csr_p[ia+1]; ix++)
            {
                ib = (size_t)U_csr_i[ix];
                err = cblas_tdot(k_totC, A + ia*(size_t)k_totA, 1,
                                 C + ib*(size_t)k_totC, 1)
                       - U_csr[ix];
                err_row += square(err);
                cblas_taxpy(k_totC, err * w_user,
                            C + ib*(size_t)k_totC, 1,
                            g_A + ia*(size_t)k_totA, 1);
            }
            f_user += err_row;
        }
        f += (w_user / 2.) * f_user;
    }

    FPnum f_reg = 0;
    add_lam_to_grad_and_fun(&f_reg, g_A, A, m_max, k_totA,
                            k_totA, lam, nthreads);
    if (lam != 0. && lam_last != lam && k_main >= 1) {
        cblas_taxpy(m, lam_last-lam, A + k_user + k + k_main, k_totA,
                    g_A + k_user + k + k_main, k_totA);
        f += (lam_last-lam) * cblas_tdot(m, A + k_user + k + k_main, k_totA,
                                         A + k_user + k + k_main, k_totA);
    }
    return f + (f_reg / 2.);
}

FPnum wrapper_fun_grad_Adense_col
(
    void *instance,
    lbfgsFPnumval_t *x,
    lbfgsFPnumval_t *g,
    const size_t n,
    const lbfgsFPnumval_t step
)
{
    data_fun_grad_Adense_col *data = (data_fun_grad_Adense_col*)instance;
    return  fun_grad_A_collective(
                x, g,
                data->B, data->C,
                data->m, data->m_u, data->n, data->p,
                data->k, data->k_main, data->k_user, data->k_item,data->padding,
                data->Xfull, data->full_dense,
                data->Xcsr_p, data->Xcsr_i, data->Xcsr,
                data->weight,
                data->U_csr_p, data->U_csr_i, data->U_csr,
                data->U, data->full_dense_u,
                data->lam, data->w_main, data->w_user, data->lam_last,
                data->do_B,
                data->nthreads,
                data->buffer_FPnum
            );
}

void buffer_size_optimizeA_collective
(
    size_t *buffer_size, size_t *buffer_lbfgs_size,
    int m, int n, int k, int k_user, int k_main, int padding,
    int m_u, int p, int nthreads,
    bool do_B, bool NA_as_zero_X, bool NA_as_zero_U, bool use_cg,
    bool full_dense, bool near_dense,
    bool has_dense, bool has_weight,
    bool full_dense_u, bool near_dense_u, bool has_dense_u
)
{
    if (has_dense && has_dense_u && !has_weight &&
        (full_dense || near_dense) && (full_dense_u || near_dense_u))
    {
        *buffer_size = square(k_user+k+k_main);
        if (!full_dense || !full_dense_u) {
            *buffer_size += (size_t)square(k_user+k)
                            + (size_t)square(k+k_main)
                            + (size_t)nthreads
                                        * ((size_t)square(k_user+k+k_main)
                                            + (size_t)n*(size_t)(k+k_main)
                                            + (size_t)p*(size_t)(k_user+k));
            if (do_B) *buffer_size += (size_t)n*(size_t)nthreads;

        }
    }

    else if (((has_dense && ((!full_dense && !near_dense) || has_weight))
                ||
             (has_dense_u && !full_dense_u && !near_dense_u &&
              p > n && (double)m_u >= .5 * (double)m))
                &&
             (has_dense || !NA_as_zero_X) && (has_dense_u || !NA_as_zero_U))
    {
        *buffer_lbfgs_size = 4;
        *buffer_size = (size_t)min2(m, m_u) * (size_t)(k_user+k+k_main+padding);
        *buffer_size *= (size_t)13;
        *buffer_size += max2((has_dense?
                                ((size_t)min2(m, m_u)*(size_t)n) : 0),
                             (has_dense_u?
                                ((size_t)min2(m, m_u)*(size_t)p) : 0));
    }

    else if (!has_dense && !has_dense_u &&
             NA_as_zero_X && NA_as_zero_U &&
             !has_weight)
    {
        *buffer_size = square(k_user+k+k_main);
    }

    else
    {
        *buffer_size = square(k_user+k+k_main);
        if (has_dense && !has_weight)
            *buffer_size += (size_t)n*(size_t)(k+k_main);
        if (has_dense_u && !full_dense_u)
            *buffer_size += (size_t)p*(size_t)(k_user+k);
        *buffer_size *= (size_t)nthreads;

        *buffer_size += square(k_user+k);
        if (!has_weight) *buffer_size += square(k+k_main);
        if (do_B) *buffer_size += (size_t)n*(size_t)nthreads;
        if (do_B && has_weight) *buffer_size += (size_t)n*(size_t)(nthreads+1);
    }

    if (use_cg)
        *buffer_size += (size_t)nthreads + (size_t)6*(size_t)(k_user+k+k_main);

    if (m > m_u)
    {
        size_t alt_size = 0;
        size_t alt_lbfgs = 0;
        buffer_size_optimizeA(
            &alt_size, &alt_lbfgs,
            m-m_u, n, k+k_main, k_user+k+k_main+padding, nthreads,
            do_B, NA_as_zero_X, use_cg,
            full_dense, near_dense,
            has_dense, has_weight
        );
        *buffer_size = max2(*buffer_size, alt_size);
        *buffer_lbfgs_size = max2(*buffer_lbfgs_size, alt_lbfgs);
    }

    else if (m_u > m)
    {
        size_t alt_size = 0;
        size_t alt_lbfgs = 0;
        buffer_size_optimizeA(
            &alt_size, &alt_lbfgs,
            m_u-m, p, k_user+k, k_user+k+k_main+padding, nthreads,
            false, NA_as_zero_U, use_cg,
            full_dense_u, near_dense_u,
            has_dense_u, false
        );
        *buffer_size = max2(*buffer_size, alt_size);
        *buffer_lbfgs_size = max2(*buffer_lbfgs_size, alt_lbfgs);
    }
}

/* Note: cannot use 'do_B' with dense matrix X if m>m_u */
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
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif
    char uplo = 'L';
    int ignore;

    int k_totA = k_user + k + k_main + padding;
    int k_totB = k_item + k + k_main + padding;
    int k_totC = k_user + k;
    int k_pred = k_user + k + k_main;
    int m_min = min2(m, m_u);
    int m_x = m; /* 'm' will be overwritten later */
    /* TODO: here should only need to set straight away the lower half,
       and only when there are un-even entries in each matrix */
    set_to_zero(A, (size_t)max2(m, m_u)*(size_t)k_totA, nthreads);

    /* If one of the matrices has more rows than the other, the rows
       for the larger matrix will be independent and can be obtained
       from the single-matrix formula instead. */
    if (m > m_u) {
        int m_diff = m - m_u;
        optimizeA(
            A + (size_t)k_user + (size_t)m_u*(size_t)k_totA, k_totA,
            B + k_item, k_totB,
            m_diff, n, k + k_main,
            (Xfull != NULL)? ((long*)NULL) : (Xcsr_p + m_u),
            (Xfull != NULL)? ((int*)NULL) : Xcsr_i,
            (Xfull != NULL)? ((FPnum*)NULL) : Xcsr,
            (Xfull == NULL)? ((FPnum*)NULL) : (Xfull + (size_t)m_u*(size_t)n),
            full_dense, near_dense,
            (Xfull == NULL)? ((int*)NULL) : (cnt_NA_x + m_u),
            (weight == NULL)? ((FPnum*)NULL)
                            : ( (Xfull == NULL)?
                                  (weight) : (weight + (size_t)m_u*(size_t)n) ),
            NA_as_zero_X,
            lam, w_main, lam_last,
            false,
            nthreads,
            use_cg,
            buffer_FPnum,
            buffer_lbfgs_iter
        );
    }

    else if (m_u > m) {
        int m_diff = m_u - m;
        optimizeA(
            A + (size_t)m*(size_t)k_totA, k_totA,
            C, k_totC,
            m_diff, p, k_user + k,
            (U != NULL)? ((long*)NULL) : (U_csr_p + m),
            (U != NULL)? ((int*)NULL) : U_csr_i,
            (U != NULL)? ((FPnum*)NULL) : U_csr,
            (U == NULL)? ((FPnum*)NULL) : (U + (size_t)m*(size_t)p),
            full_dense_u, near_dense_u,
            (U == NULL)? ((int*)NULL) : (cnt_NA_u + m),
            (FPnum*)NULL,
            NA_as_zero_U,
            lam, w_user, lam,
            false,
            nthreads,
            use_cg,
            buffer_FPnum,
            buffer_lbfgs_iter
        );
    }

    m = m_min;

    /* Case 1: both matrices are dense with few missing values and no weights.
       Here can use the closed-form solution on all the observations
       at once, and then do corrections one-by-one if there are any
       missing values. */
    if (Xfull != NULL && U != NULL && weight == NULL &&
        (full_dense || near_dense) && (full_dense_u || near_dense_u))
    {
        FPnum *restrict bufferBeTBe = buffer_FPnum;
        build_BeTBe(
            bufferBeTBe,
            B, C,
            k, k_user, k_main, k_item, padding,
            n, p,
            lam, w_main, w_user
        );
        if (lam_last != lam)
            bufferBeTBe[square(k_user+k+k_main)-1] += (lam_last - lam);
        build_XBw(
            A + k_user, k_totA,
            B + k_item, k_totB,
            Xfull,
            m, n, k + k_main, w_main,
            do_B, true
        );
        cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, k_user + k, p,
                    w_user, U, p, C, k_user + k,
                    1., A, k_totA);

        tposv_(&uplo, &k_pred, &m,
               bufferBeTBe, &k_pred,
               A, &k_totA,
               &ignore);

        if (!full_dense || !full_dense_u)
        {
            /* The LHS matrices can be precomputed and then sum or subtract
               from them as more efficient according to the number of NAs. */
            FPnum *restrict bufferBtB = buffer_FPnum;
            FPnum *restrict bufferCtC = bufferBtB + square(k+k_main);
            build_BtB_CtC(
                bufferBtB, bufferCtC,
                B, n,
                C, p,
                k, k_user, k_main, k_item, padding,
                w_main, w_user,
                weight
            );

            /* When doing the B matrix, the X matrix will be transposed
               and need to make a copy of the column for each observation,
               whereas the U matrix will be in the right order. */
            FPnum *restrict bufferX = bufferCtC + square(k_user+k);
            FPnum *restrict buffer_remainder = bufferX
                                                + (do_B? (n*nthreads) : (0));
            size_t size_buffer = square(k_totA)
                                  + ((size_t)n * (size_t)(k+k_main))
                                  + ((size_t)p * (size_t)(k_user+k));
            if (use_cg) size_buffer += (size_t)6 * (size_t)k_totA;

            #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                    shared(A, k_totA, B, k_totB, C, k, k_user, k_item, k_main, \
                           m, n, p, lam, lam_last, w_main, w_user, \
                           Xfull, cnt_NA_x, full_dense, \
                           U, cnt_NA_u, full_dense_u, \
                           buffer_remainder, size_buffer, m_x, do_B, \
                           bufferBtB, bufferCtC, nthreads, use_cg) \
                    firstprivate(bufferX)
            for (size_t_for ix = 0; ix < (size_t)m; ix++)
            {
                if (cnt_NA_x[ix] || cnt_NA_u[ix])
                {
                    if (!do_B)
                        bufferX = Xfull + ix*(size_t)n;
                    else
                        cblas_tcopy(n, Xfull + ix, m_x,
                                    bufferX + n*omp_get_thread_num(), 1);

                    collective_closed_form_block(
                        A + ix*(size_t)k_totA,
                        k, k_user, k_item, k_main, 0, padding,
                        bufferX + (do_B? (n*omp_get_thread_num()) : (0)),
                        (FPnum*)NULL, (int*)NULL, (size_t)0,
                        (int*)NULL, (FPnum*)NULL, (size_t)0,
                        U + ix*(size_t)p,
                        false, false,
                        B, n,
                        C, p,
                        (FPnum*)NULL,
                        lam, w_user, w_main, lam_last,
                        bufferBtB, cnt_NA_x[ix],
                        bufferCtC, cnt_NA_u[ix],
                        true, true, use_cg,
                        buffer_remainder
                          + (size_buffer*(size_t)omp_get_thread_num())
                    );
                }
            }
        }
    }

    /* Case 2: both matrices are dense, but with many missing values,
       or at least one of them has many missing values, or there are weights.
       In this case, it's faster to use a gradient-based approach. */
    else if (
             ((Xfull != NULL && ((!full_dense && !near_dense) || weight!=NULL))
                ||
             (U != NULL && !full_dense_u && !near_dense_u &&
              p > n && (double)m_u >= .5 * (double)m_x))
                &&
             (Xfull != NULL || !NA_as_zero_X) && (U != NULL || !NA_as_zero_U)
            )
    {
        size_t nvars = (size_t)m * (size_t)(k_user + k + k_main + padding);
        size_t m_lbfgs = 4;
        size_t past = 0;
        FPnum *restrict buffer_lbfgs = buffer_FPnum + (
                                        max2(
                                            ((Xfull != NULL)?
                                                ((size_t)m*(size_t)n) : 0),
                                            ((U != NULL)?
                                                ((size_t)m*(size_t)p) : 0)
                                            )
                                        );
        lbfgs_parameter_t lbfgs_params = {
            m_lbfgs, 1e-5, past, 1e-5,
            100, LBFGS_LINESEARCH_MORETHUENTE, 30,
            1e-20, 1e20, 1e-4, 0.9, 0.9, 1.0e-16,
            0.0, 0, -1,
        };
        lbfgs_progress_t callback = (lbfgs_progress_t)NULL;
        data_fun_grad_Adense_col data = {
            B, C,
            m, m, n, p,
            k, k_main, k_user, k_item, padding,
            Xfull, full_dense,
            Xcsr_p, Xcsr_i, Xcsr,
            weight,
            U_csr_p, U_csr_i, U_csr,
            U, full_dense_u,
            lam, w_main, w_user, lam_last,
            do_B,
            nthreads,
            buffer_FPnum
        };
        lbfgs(
            nvars,
            A,
            (FPnum*)NULL,
            wrapper_fun_grad_Adense_col,
            callback,
            (void*) &data,
            &lbfgs_params,
            buffer_lbfgs,
            buffer_lbfgs_iter
        );

    }

    /* Case 3: both matrices are sparse, missing as zero, no weights.
       Here can also use the closed-form solution for all rows at once. */
    else if (Xfull == NULL && U == NULL &&
             NA_as_zero_X && NA_as_zero_U &&
             weight == NULL)
    {
        FPnum *restrict bufferBeTBe = buffer_FPnum;
        build_BeTBe(
            bufferBeTBe,
            B, C,
            k, k_user, k_main, k_item, padding,
            n, p,
            lam, w_main, w_user
        );
        if (lam_last != lam)
            bufferBeTBe[square(k_user+k+k_main)-1] += (lam_last - lam);

        sgemm_sp_dense(
            m, k+k_main, w_main,
            Xcsr_p, Xcsr_i, Xcsr,
            B + k_item, (size_t)k_totB,
            A + k_user, (size_t)k_totA,
            nthreads
        );
        sgemm_sp_dense(
            m, k_user+k, w_user,
            U_csr_p, U_csr_i, U_csr,
            C, (size_t)k_totC,
            A, (size_t)k_totA,
            nthreads
        );

        tposv_(&uplo, &k_pred, &m,
               bufferBeTBe, &k_pred,
               A, &k_totA,
               &ignore);
    }

    /* General case - construct one-by-one, use precomputed matrices
       when beneficial, determined on a case-by-case basis. */
    else
    {
        FPnum *restrict bufferCtC = buffer_FPnum;
        FPnum *restrict bufferBtB = bufferCtC + square(k_user+k);
        FPnum *restrict bufferX = bufferBtB + (
                                        (weight == NULL)? square(k+k_main) : 0
                                        );
        FPnum *restrict bufferX_zeros = bufferX + (do_B?
                                                   ((size_t)n*(size_t)nthreads)
                                                   : (0));
        FPnum *restrict bufferX_orig = bufferX;
        FPnum *restrict bufferW = bufferX_zeros + ((do_B && weight != NULL)?
                                                    (n) : (0));
        FPnum *restrict buffer_remainder = bufferW + (
                                                (do_B && weight != NULL)?
                                                ((size_t)n*(size_t)nthreads)
                                                : (0));
        if (weight == NULL) bufferW = NULL;
        bool add_X = true;
        bool add_U = true;

        build_BtB_CtC(
            bufferBtB, bufferCtC,
            B, n,
            C, p,
            k, k_user, k_main, k_item, padding,
            w_main, w_user,
            weight
        );
        if (weight != NULL)
            bufferBtB = NULL;

        if (Xfull != NULL && (full_dense || near_dense) && weight == NULL) {
            add_X = false;
            build_XBw(
                A + k_user, k_totA,
                B + k_item, k_totB,
                Xfull,
                m, n, k + k_main, w_main,
                do_B, true
            );

            if (!full_dense && do_B)
                set_to_zero(bufferX_zeros, n, 1); /*still needs a placeholder*/
        }

        else if (Xfull == NULL && weight == NULL && NA_as_zero_X) {
            add_X = false;
            sgemm_sp_dense(
                m, k+k_main, w_main,
                Xcsr_p, Xcsr_i, Xcsr,
                B + k_item, (size_t)k_totB,
                A + k_user, (size_t)k_totA,
                nthreads
            );
        }

        else if (U != NULL && (full_dense_u || near_dense_u)) {
            add_U = false;
            cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, k_user + k, p,
                        w_user, U, p, C, k_user + k,
                        0., A, k_totA);
        }

        else if (U == NULL && NA_as_zero_U) {
            add_U = false;
            sgemm_sp_dense(
                m, k_user+k, w_user,
                U_csr_p, U_csr_i, U_csr,
                C, (size_t)k_totC,
                A, (size_t)k_totA,
                nthreads
            );
        }

        size_t size_buffer = square(k_totA)
                              + ((Xfull != NULL && weight == NULL)?
                                  ((size_t)n*(size_t)(k+k_main)) : (0))
                              + ((U != NULL && !full_dense_u)?
                                    ((size_t)p*(size_t)(k_user+k)) : (0));
        if (use_cg) size_buffer += (size_t)6*(size_t)k_totA;

        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                shared(A, k_totA, B, k_totB, C, k, k_user, k_item, k_main, \
                       m, n, p, lam, lam_last, w_user, w_main, \
                       NA_as_zero_X, NA_as_zero_U, add_X, add_U, weight, \
                       Xfull, Xcsr_p, Xcsr_i, Xcsr, cnt_NA_x, \
                       U, U_csr_p, U_csr_i, U_csr, cnt_NA_u, \
                       bufferBtB, bufferCtC, buffer_remainder, size_buffer, \
                       m_x, do_B, nthreads, use_cg) \
                firstprivate(bufferX, bufferW)
        for (size_t_for ix = 0; ix < (size_t)m; ix++)
        {
            if (Xfull != NULL) {
                if (!do_B)
                    bufferX = Xfull + ix*(size_t)n;
                else if (add_X || cnt_NA_x[ix]) {
                    cblas_tcopy(n, Xfull + ix, m_x,
                                bufferX_orig + n*omp_get_thread_num(), 1);
                    bufferX = bufferX_orig;
                }
                else
                    bufferX = bufferX_zeros - n*omp_get_thread_num();

                if (weight != NULL) {
                    if (!do_B)
                        bufferW = weight + ix*(size_t)n;
                    else
                        cblas_tcopy(n, weight + ix, m_x,
                                    bufferW + n*omp_get_thread_num(), 1);
                }
            }

            collective_closed_form_block(
                A + ix*(size_t)k_totA,
                k, k_user, k_item, k_main, 0, padding,
                (Xfull == NULL)? ((FPnum*)NULL)
                                 : (bufferX
                                    + (do_B? (n*omp_get_thread_num()) : (0))),
                (Xfull != NULL)? ((FPnum*)NULL) : (Xcsr + Xcsr_p[ix]),
                (Xfull != NULL)? ((int*)NULL) : (Xcsr_i + Xcsr_p[ix]),
                (Xfull != NULL)? (size_t)0 : (size_t)(Xcsr_p[ix+1] -Xcsr_p[ix]),
                (U != NULL)? ((int*)NULL) : (U_csr_i + U_csr_p[ix]),
                (U != NULL)? ((FPnum*)NULL) : (U_csr + U_csr_p[ix]),
                (U != NULL)? (size_t)0 : (size_t)(U_csr_p[ix+1] - U_csr_p[ix]),
                (U == NULL)? ((FPnum*)NULL) : (U + ix*(size_t)p),
                NA_as_zero_X, NA_as_zero_U,
                B, n,
                C, p,
                (weight == NULL)? ((FPnum*)NULL)
                                : ( (Xfull != NULL)?
                                      (bufferW
                                       + (do_B? (n*omp_get_thread_num()) : (0)))
                                      : (weight + Xcsr_p[ix]) ),
                lam, w_user, w_main, lam_last,
                bufferBtB, (Xfull != NULL)? cnt_NA_x[ix] : 0,
                bufferCtC, (U != NULL)? cnt_NA_u[ix] : 0,
                (Xfull == NULL)? (add_X) : (add_X || cnt_NA_x[ix] > 0),
                (U == NULL)? (add_U) : (add_U || cnt_NA_u[ix] > 0),
                use_cg,
                buffer_remainder + (size_buffer*(size_t)omp_get_thread_num())
            );
        }

    }
}

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
)
{
    int k_totA = k_user + k + k_main;
    int k_totB = k_item + k + k_main;
    int k_totC = k_user + k;
    int m_x = m; /* <- later gets overwritten */
    set_to_zero(A, (size_t)max2(m, m_u)*(size_t)k_totA, nthreads);

    /* If the X matrix has more rows, the extra rows will be independent
       from U and can be obtained from the single-matrix formula instead.
       However, if the U matrix has more rows, those still need to be
       considered as having a value of zero in X. */
    if (m > m_u) {
        int m_diff = m - m_u;
        optimizeA_implicit(
            A + (size_t)k_user + (size_t)m_u * (size_t)k_totA, (size_t)k_totA,
            B + k_item, (size_t)k_totB,
            m_diff, n, k + k_main,
            Xcsr_p + m_u, Xcsr_i, Xcsr,
            lam/w_main, alpha,
            nthreads, use_cg,
            buffer_FPnum
        );
        m = m_u;
    }

    /* Lower-right square of Be */
    FPnum *restrict precomputedBeTBe = buffer_FPnum;
    FPnum *restrict precomputedBtB = precomputedBeTBe + square(k_totA);
    FPnum *restrict buffer_remainder = precomputedBtB + square(k_totA);

    set_to_zero(precomputedBeTBe, square(k_totA), 1);
    cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                k+k_main, n,
                w_main, B + k_item, k_totB,
                0., precomputedBeTBe + k_user + k_user*k_totA, k_totA);
    add_to_diag(precomputedBeTBe, lam, k_totA);
    memcpy(precomputedBtB, precomputedBeTBe, square(k_totA)*sizeof(FPnum));

    /* Upper-left square of Be if possible */
    if (U != NULL || NA_as_zero_U)
    {
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k_user+k, p,
                    w_user, C, k_totC,
                    1., precomputedBeTBe, k_totA);
    }

    /* Lower half of Xe (reuse if possible) */
    if (U != NULL) {
        cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, k_user + k, p,
                    w_user, U, p, C, k_totC,
                    0., A, k_totA);
    }
    else {
        sgemm_sp_dense(
            m, k_user + k, w_user,
            U_csr_p, U_csr_i, U_csr,
            C, k_totC,
            A, k_totA,
            nthreads
        );
    }

    size_t size_buffer = square(k_totA);
    if (use_cg) size_buffer += (size_t)6 *(size_t)k_totA;

    int ix = 0;
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
            shared(A, B, C, m, n, p, k, k_user, k_item, k_main, lam, alpha, \
                   Xcsr, Xcsr_p, Xcsr_i, U, U_csr, U_csr_i, U_csr_p, \
                   NA_as_zero_U, cnt_NA_u, precomputedBeTBe, precomputedBtB, \
                   k_totA, buffer_remainder, use_cg, m_x)
    for (ix = 0; ix < m; ix++)
        collective_closed_form_block_implicit(
            A + (size_t)ix*(size_t)k_totA,
            k, k_user, k_item, k_main,
            B, n, C, p,
            (ix < m_x)? (Xcsr + Xcsr_p[ix]) : ((FPnum*)NULL),
            (ix < m_x)? (Xcsr_i + Xcsr_p[ix]) : ((int*)NULL),
            (ix < m_x)? (Xcsr_p[ix+1] - Xcsr_p[ix]) : (0),
            (U == NULL)? ((FPnum*)NULL) : (U + (size_t)ix*(size_t)p),
            (U == NULL)? (0) : (cnt_NA_u[ix]),
            (U == NULL)? (U_csr + U_csr_p[ix]) : ((FPnum*)NULL),
            (U == NULL)? (U_csr_i + U_csr_p[ix]) : ((int*)NULL),
            (U == NULL)? (size_t)(U_csr_p[ix+1] - U_csr_p[ix]) : ((size_t)0),
            NA_as_zero_U,
            lam, alpha, w_main, w_user,
            precomputedBeTBe,
            precomputedBtB,
            false, true, use_cg,
            buffer_remainder + ((size_t)omp_get_thread_num() * size_buffer)
        );

}

void build_BeTBe
(
    FPnum *restrict bufferBeTBe,
    FPnum *restrict B, FPnum *restrict C,
    int k, int k_user, int k_main, int k_item, int padding,
    int n, int p,
    FPnum lam, FPnum w_main, FPnum w_user
)
{
    int k_totA = k_user + k + k_main;
    int k_totB = k_item + k + k_main + padding;
    set_to_zero(bufferBeTBe, square(k_totA), 1);
    cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                k_user + k, p,
                w_user, C, k_user + k,
                0., bufferBeTBe, k_totA);
    cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                k + k_main, n,
                w_main, B + k_item, k_totB,
                1., bufferBeTBe + k_user + k_user*k_totA, k_totA);
    add_to_diag(bufferBeTBe, lam, k_totA);
}

void build_BtB_CtC
(
    FPnum *restrict BtB, FPnum *restrict CtC,
    FPnum *restrict B, int n,
    FPnum *restrict C, int p,
    int k, int k_user, int k_main, int k_item, int padding,
    FPnum w_main, FPnum w_user,
    FPnum *restrict weight
)
{
    int k_totB = k_item + k + k_main + padding;
    if (weight == NULL) {
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k + k_main, n,
                    w_main, B + k_item, k_totB,
                    0., BtB, k+k_main);
    }
    cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                k_user + k, p,
                w_user, C, k_user + k,
                0., CtC, k_user + k);
}

void build_XBw
(
    FPnum *restrict A, int lda,
    FPnum *restrict B, int ldb,
    FPnum *restrict Xfull,
    int m, int n, int k, FPnum w,
    bool do_B, bool overwrite
)
{
    if (!do_B)
        cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, k, n,
                    w, Xfull, n, B, ldb,
                    overwrite? 0. : 1., A, lda);
    else
        cblas_tgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    m, k, n,
                    w, Xfull, m, B, ldb,
                    overwrite? 0. : 1., A, lda);
}

void preprocess_vec
(
    FPnum *restrict vec_full, int n,
    int *restrict ix_vec, FPnum *restrict vec_sp, size_t nnz,
    FPnum glob_mean, FPnum lam,
    FPnum *restrict col_means,
    FPnum *restrict vec_mean,
    int *restrict cnt_NA
)
{
    if (col_means != NULL)
    {
        if (vec_full != NULL) {
            for (int ix = 0; ix < n; ix++)
                vec_full[ix] -= col_means[ix] + glob_mean;
        }

        else {
            for (size_t ix = 0; ix < nnz; ix++)
                vec_sp[ix] -= col_means[ix_vec[ix]] + glob_mean;
        }
    }

    else if (glob_mean != 0.)
    {
        if (vec_full != NULL) {
            for (int ix = 0; ix < n; ix++)
                vec_full[ix] -= glob_mean;
        }

        else {
            for (size_t ix = 0; ix < nnz; ix++)
                vec_sp[ix] -= glob_mean;
        }
    }

    if (vec_full != NULL)
        *cnt_NA = count_NAs(vec_full, (size_t)n, 1);

    /* Note: this is a heuristic to obtain the user bias when making
       warm start predictions. It tends to assing higher weights to the
       bias and lower weights to the actual coefficients.
       This is not used in the final code as when there are user biases,
       it will be multiplied against a B matrix with ones appended as the
       last column, from which the bias will be obtained. */
    if (vec_mean != NULL)
    {
        *vec_mean = 0;

        if (vec_full == NULL) {
            for (size_t ix = 0; ix < nnz; ix++)
                *vec_mean += vec_sp[ix];
            *vec_mean /= ((double)nnz + lam);
            for (size_t ix = 0; ix < nnz; ix++)
                vec_sp[ix] -= *vec_mean;
        }

        else {
            if (*cnt_NA > 0) {
                for (int ix = 0; ix < n; ix++) {
                    *vec_mean += (!isnan(vec_full[ix]))? vec_full[ix] : 0;
                }
                *vec_mean /= ((double)(n - *cnt_NA) + lam);
            }

            else {
                for (int ix = 0; ix < n; ix++)
                    *vec_mean += vec_full[ix];
                *vec_mean /= ((double)n + lam);
            }
            
            for (int ix = 0; ix < n; ix++)
                vec_full[ix] -= *vec_mean;
        }

    }
}

int convert_sparse_X
(
    int ixA[], int ixB[], FPnum *restrict X, size_t nnz,
    long **Xcsr_p, int **Xcsr_i, FPnum *restrict *Xcsr,
    long **Xcsc_p, int **Xcsc_i, FPnum *restrict *Xcsc,
    FPnum *restrict weight, FPnum *restrict *weightR, FPnum *restrict *weightC,
    int m, int n, int nthreads
)
{
    *Xcsr_p = (long*)malloc((size_t)(m+1)*sizeof(long));
    *Xcsr_i = (int*)malloc(nnz*sizeof(int));
    *Xcsr = (FPnum*)malloc(nnz*sizeof(FPnum));
    if (weight != NULL)
        *weightR = (FPnum*)malloc(nnz*sizeof(FPnum));
    *Xcsc_p = (long*)malloc((size_t)(n+1)*sizeof(long));
    *Xcsc_i = (int*)malloc(nnz*sizeof(int));
    *Xcsc = (FPnum*)malloc(nnz*sizeof(FPnum));
    if (weight != NULL)
        *weightC = (FPnum*)malloc(nnz*sizeof(FPnum));
    if (*Xcsr_p == NULL || *Xcsr_i == NULL || *Xcsr == NULL ||
        *Xcsc_p == NULL || *Xcsc_i == NULL || *Xcsc == NULL ||
        (weight != NULL && (*weightR == NULL || *weightC == NULL)))
        return 1;
    coo_to_csr_and_csc(
        ixA, ixB, X,
        weight, m, n, nnz,
        *Xcsr_p, *Xcsr_i, *Xcsr,
        *Xcsc_p, *Xcsc_i, *Xcsc,
        *weightR, *weightC,
        nthreads
    );
    return 0;
}

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
)
{
    int retval = 0;
    if (U != NULL)
    {
        *cnt_NA_u_byrow = (int*)calloc(m_u, sizeof(int));
        *cnt_NA_u_bycol = (int*)calloc(p, sizeof(int));
        if (*cnt_NA_u_byrow == NULL || *cnt_NA_u_bycol == NULL)
            return 1;
        count_NAs_by_row(U, m_u, p, *cnt_NA_u_byrow, nthreads,
                         full_dense_u, near_dense_u_row);
        count_NAs_by_col(U, m_u, p, *cnt_NA_u_bycol,
                         full_dense_u, near_dense_u_col);
    }

    else
    {
        *U_csr_p = (long*)malloc((size_t)(m_u+1)*sizeof(long));
        *U_csr_i = (int*)malloc(nnz_U*sizeof(int));
        *U_csr = (FPnum*)malloc(nnz_U*sizeof(FPnum));
        *U_csc_p = (long*)malloc((size_t)(p+1)*sizeof(long));
        *U_csc_i = (int*)malloc(nnz_U*sizeof(int));
        *U_csc = (FPnum*)malloc(nnz_U*sizeof(FPnum));
        if (*U_csr_p == NULL || *U_csr_i == NULL || *U_csr == NULL ||
            *U_csc_p == NULL || *U_csc_i == NULL || *U_csc == NULL)
            return 1;
        coo_to_csr_and_csc(
            U_row, U_col, U_sp,
            (FPnum*)NULL, m_u, p, nnz_U,
            *U_csr_p, *U_csr_i, *U_csr,
            *U_csc_p, *U_csc_i, *U_csc,
            (FPnum*)NULL, (FPnum*)NULL,
            nthreads
        );
    }

    if ((U != NULL || !NA_as_zero_U) && U_colmeans != NULL)
        retval = center_by_cols(
            U_colmeans,
            U, m_u, p,
            U_row, U_col, U_sp, nnz_U,
            *U_csr_p, *U_csr_i, *U_csr,
            *U_csc_p, *U_csc_i, *U_csc,
            nthreads
        );
    if (retval != 0) return 1;

    if (U != NULL && near_dense_u_col && Utrans != NULL)
    {
        *Utrans = (FPnum*)malloc((size_t)m_u*(size_t)p*sizeof(FPnum));
        if (*Utrans == NULL)
            return 1;
        transpose_mat2(U, m_u, p, *Utrans);
    }

    if (U != NULL && !full_dense_u && U_orig != NULL)
    {
        *U_orig = malloc((size_t)m_u*(size_t)p*sizeof(FPnum));
        if (*U_orig == NULL)
            return 1;
        copy_arr(U, *U_orig, (size_t)m_u*(size_t)p, nthreads);
    }

    return 0;
}

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
)
{
    /* Dimensions for each input matrix:
       - B: [n, k_item+k+k_main]
       - C: [p, k_user+k]

       Dimensions for each output matrix:
       - BtBinvBt: [k+k_main, n]
       - BtBw: [k+k_main, k+k_main]
       - BtBchol: [k+k_main, k+k_main]
       - CtCinvCt: [k_user+k, p]
       - CtCchol: [k_user+k, k_user+k]
       - BeTBe: [k_user+k+k_main, k_user+k+k_main]
       - BtB_padded: [k_user+k+k_main, k_user+k+k_main]
       - BtB_shrunk: [k+k_main, k+k_main]
    */
    FPnum *restrict buffer_FPnum = NULL;
    size_t size_buffer = 0;
    if (has_U && (!has_U_bin || implicit))
        size_buffer = (size_t)p * (size_t)(k_user+k);
    if (!implicit)
        size_buffer = max2(size_buffer, (size_t)n * (size_t)(k+k_main));

    buffer_FPnum = (FPnum*)malloc(size_buffer*sizeof(FPnum));
    if (buffer_FPnum == NULL) return 1;

    /* For cold-start predictions */
    if (has_U && (!has_U_bin || implicit))
        AtAinvAt_plus_chol(C, k_user+k, 0,
                           CtCinvCt,
                           CtC,
                           CtCchol,
                           lam, lam, p, k_user+k, w_user,
                           buffer_FPnum,
                           true);

    if (implicit)
    {
        w_main *= w_main_multiplier;
        int k_totA = k_user + k + k_main;
        set_to_zero(BeTBe, square(k_totA), 1);
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k+k_main, n,
                    w_main, B + k_item, k_item+k+k_main,
                    0., BeTBe + k_user + k_user*k_totA, k_totA);

        if (has_U) {
            memcpy(BtB_padded, BeTBe, square(k_totA)*sizeof(FPnum));
            copy_mat(k+k_main, k+k_main, BeTBe, k_totA, BtB_shrunk, k+k_main);
            add_to_diag(BtB_shrunk, lam, k+k_main);

            sum_mat(k_user+k, k_user+k, CtC, k_user+k, BeTBe, k_totA);
            for (int ix = k_user+k; ix < k_totA; ix++)
                BeTBe[ix + ix*k_totA] += lam;
        } else {
            add_to_diag(BeTBe, lam, k_totA);
        }
    }

    else
    {
        /* For warm-start predictions */
        AtAinvAt_plus_chol(B, k_item+k+k_main, k_item,
                           BtBinvBt,
                           BtBw,
                           BtBchol,
                           lam, lam_last, n, k+k_main, w_main,
                           buffer_FPnum,
                           true);
    }

    free(buffer_FPnum);
    return 0;
}

lbfgsFPnumval_t wrapper_collective_fun_grad
(
    void *instance,
    lbfgsFPnumval_t *x,
    lbfgsFPnumval_t *g,
    const size_t n,
    const lbfgsFPnumval_t step
)
{
    data_collective_fun_grad *data = (data_collective_fun_grad*)instance;
    (data->nfev)++;
    return collective_fun_grad(
            x, g,
            data->m, data->n, data->k,
            data->ixA, data->ixB, data->X, data->nnz,
            data->Xfull,
            data->Xcsr_p, data->Xcsr_i, data->Xcsr,
            data->Xcsc_p, data->Xcsc_i, data->Xcsc,
            data->weight, data->weightR, data->weightC,
            data->user_bias, data->item_bias,
            data->lam, data->lam_unique,
            data->U, data->m_u, data->p, data->U_has_NA,
            data->II, data->n_i, data->q, data->I_has_NA,
            data->Ub, data->m_ubin, data->pbin, data->Ub_has_NA,
            data->Ib, data->n_ibin, data->qbin, data->Ib_has_NA,
            data->U_row, data->U_col, data->U_sp, data->nnz_U,
            data->I_row, data->I_col, data->I_sp, data->nnz_I,
            data->U_csr_p, data->U_csr_i, data->U_csr,
            data->U_csc_p, data->U_csc_i, data->U_csc,
            data->I_csr_p, data->I_csr_i, data->I_csr,
            data->I_csc_p, data->I_csc_i, data->I_csc,
            data->buffer_FPnum, data->buffer_mt,
            data->k_main, data->k_user, data->k_item,
            data->w_main, data->w_user, data->w_item,
            data->nthreads
        );
}

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
)
{
    FPnum *restrict buffer_FPnum = NULL;
    FPnum *restrict buffer_mt = NULL;
    int retval = 0;

    size_t nvars, size_buffer, size_mt;
    nvars_collective_fun_grad(
        m, n, m_u, n_i, m_ubin, n_ibin,
        p, q, pbin, qbin,
        nnz, nnz_U, nnz_I,
        k, k_main, k_user, k_item,
        user_bias, item_bias, nthreads,
        X, Xfull, X,
        U, Ub, II, Ib,
        U_sp, U_sp, I_sp, I_sp,
        &nvars, &size_buffer, &size_mt
    );

    if (size_buffer) {
        buffer_FPnum = (FPnum*)malloc(size_buffer*sizeof(FPnum));
        if (buffer_FPnum == NULL) return 1;
    }

    int m_max = max2(max2(m, m_u), m_ubin);
    int n_max = max2(max2(n, n_i), n_ibin);

    bool U_has_NA = false;
    bool I_has_NA = false;
    bool Ub_has_NA = false;
    bool Ib_has_NA = false;

    lbfgsFPnumval_t funval;
    lbfgs_parameter_t lbfgs_params;
    data_collective_fun_grad data;

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

    #ifdef _OPENMP
    if (nthreads > 1 && (Xfull == NULL || U_sp != NULL || I_sp != NULL))
    {
        if (prefer_onepass)
        {
            buffer_mt = (FPnum*)malloc(size_mt*sizeof(FPnum));
            if (buffer_mt == NULL) {
                retval = 1;
                goto cleanup;
            }
        }

        else if (Xfull == NULL || U_sp != NULL || I_sp != NULL)
        {
            if (Xfull == NULL)
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

            if (U_sp != NULL)
            {
                retval = preprocess_sideinfo_matrix(
                    (FPnum*)NULL, m_u, p,
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
                    (FPnum*)NULL, n_i, q,
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
        }
    }
    #endif

    retval = initialize_biases(
        glob_mean, values, values + (user_bias? m_max : 0),
        user_bias, item_bias,
        (lam_unique == NULL)? (lam) : (lam_unique[0]),
        (lam_unique == NULL)? (lam) : (lam_unique[1]),
        m, n,
        m_max, n_max,
        ixA, ixB, X, nnz,
        Xfull, (FPnum*)NULL,
        Xcsr_p, Xcsr_i, Xcsr,
        Xcsc_p, Xcsc_i, Xcsc,
        nthreads
    );
    if (retval != 0) goto cleanup;

    if (U != NULL || U_sp != NULL) {
        retval = center_by_cols(
            U_colmeans,
            U, m_u, p,
            U_row, U_col, U_sp, nnz_U,
            U_csr_p, U_csr_i, U_csr,
            U_csc_p, U_csc_i, U_csc,
            nthreads
        );
        if (retval != 0) goto cleanup;
    }

    if (II != NULL || I_sp != NULL) {
        retval = center_by_cols(
            I_colmeans,
            II, n_i, q,
            I_row, I_col, I_sp, nnz_I,
            I_csr_p, I_csr_i, I_csr,
            I_csc_p, I_csc_i, I_csc,
            nthreads
        );
        if (retval != 0) goto cleanup;
    }


    if (reset_values)
        retval = rnorm(values + (user_bias? m_max : 0) + (item_bias? n_max : 0),
                       nvars - (size_t)(user_bias? m_max : 0)
                             - (size_t)(item_bias? n_max : 0),
                       seed, nthreads);
    if (retval != 0) goto cleanup;

    if (U != NULL)
        U_has_NA = (bool)count_NAs(U, (size_t)m_u*(size_t)p, nthreads);
    if (II != NULL)
        I_has_NA = (bool)count_NAs(II, (size_t)n_i*(size_t)q, nthreads);
    if (Ub != NULL)
        Ub_has_NA = (bool)count_NAs(Ub, (size_t)m_ubin*(size_t)p, nthreads);
    if (Ib != NULL)
        Ib_has_NA = (bool)count_NAs(Ib, (size_t)n_ibin*(size_t)q, nthreads);

    lbfgs_params = 
                    #ifndef __cplusplus
                    (lbfgs_parameter_t)
                    #endif 
                                        {
        (size_t)n_corr_pairs, 1e-5, 0, 1e-5,
        maxiter, LBFGS_LINESEARCH_DEFAULT, 20,
        1e-20, 1e20, 1e-4, 0.9, 0.9, 1.0e-16,
        0.0, 0, -1,
    };
    data = 
            #ifndef __cplusplus
            (data_collective_fun_grad) 
            #endif
                                        {
        m, n, k,
        ixA, ixB, X, nnz,
        Xfull,
        Xcsr_p, Xcsr_i, Xcsr,
        Xcsc_p, Xcsc_i, Xcsc,
        weight, weightR, weightC,
        user_bias, item_bias,
        lam, lam_unique,
        U, m_u, p, U_has_NA,
        II, n_i, q, I_has_NA,
        Ub, m_ubin, pbin, Ub_has_NA,
        Ib, n_ibin, qbin, Ib_has_NA,
        U_row, U_col, U_sp, nnz_U,
        I_row, I_col, I_sp, nnz_I,
        U_csr_p, U_csr_i, U_csr,
        U_csc_p, U_csc_i, U_csc,
        I_csr_p, I_csr_i, I_csr,
        I_csc_p, I_csc_i, I_csc,
        buffer_FPnum, buffer_mt,
        k_main, k_user, k_item,
        w_main, w_user, w_item,
        nthreads, print_every, 0, 0
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
        wrapper_collective_fun_grad,
        (verbose)? (lbfgs_printer_collective) : (NULL),
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

    if (B_plus_bias != NULL && user_bias)
        append_ones_last_col(
            values + (
                  (user_bias? (size_t)m_max : (size_t)0)
                + (item_bias? (size_t)n_max : (size_t)0)
                + ((size_t)m_max * (size_t)(k_user+k+k_main))
                ),
            n_max, k_item+k+k_main,
            B_plus_bias
        );

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
)
{
    if ((NA_as_zero_X && Xfull == NULL) && (user_bias || item_bias))
        return 2;

    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long row, col, ix;
    #endif

    int k_totA = k_user + k + k_main;
    int k_totB = k_item + k + k_main;
    int paddingA = (!user_bias  &&  item_bias);
    int paddingB = ( user_bias  && !item_bias);
    int m_max = max2(m, m_u);
    int n_max = max2(n, n_i);

    FPnum *restrict biasA = values;
    FPnum *restrict biasB = biasA + (user_bias? m_max : 0);
    FPnum *restrict A = biasB + (item_bias? n_max : 0);
    FPnum *restrict B = A + (size_t)m_max * (size_t)k_totA;
    FPnum *restrict C = B + (size_t)n_max * (size_t)k_totB;
    FPnum *restrict D = C + (size_t)(k_user + k) * (size_t)p;
    size_t nvars = (D + (size_t)(k_item + k) * (size_t)q) - values;

    FPnum *restrict A_bias = NULL;
    FPnum *restrict B_bias = NULL;
    FPnum *restrict Xcsr_orig = NULL;
    FPnum *restrict Xcsc_orig = NULL;
    FPnum *restrict Xfull_orig = NULL;
    FPnum *restrict Xtrans_orig = NULL;
    FPnum *restrict U_orig = NULL;
    FPnum *restrict I_orig = NULL;

    int retval = 0;
    FPnum *restrict buffer_FPnum = NULL;
    iteration_data_t *buffer_lbfgs_iter = NULL;
    size_t size_bufferA = 0;
    size_t size_bufferB = 0;
    size_t size_bufferC = 0;
    size_t size_bufferD = 0;
    size_t size_buffer = 0;
    size_t size_buffer_lbfgs = 0;

    FPnum *restrict Xtrans = NULL;
    FPnum *restrict Wtrans = NULL;
    long *Xcsr_p = NULL;
    int *Xcsr_i = NULL;
    FPnum *restrict Xcsr = NULL;
    FPnum *restrict weightR = NULL;
    long *Xcsc_p = NULL;
    int *Xcsc_i = NULL;
    FPnum *restrict Xcsc = NULL;
    FPnum *restrict weightC = NULL;
    FPnum *restrict Utrans = NULL;
    long *U_csr_p = NULL;
    int *U_csr_i = NULL;
    FPnum *restrict U_csr = NULL;
    long *U_csc_p = NULL;
    int *U_csc_i = NULL;
    FPnum *restrict U_csc = NULL;
    FPnum *restrict Itrans = NULL;
    long *I_csr_p = NULL;
    int *I_csr_i = NULL;
    FPnum *restrict I_csr = NULL;
    long *I_csc_p = NULL;
    int *I_csc_i = NULL;
    FPnum *restrict I_csc = NULL;
    int *restrict cnt_NA_byrow = NULL;
    int *restrict cnt_NA_bycol = NULL;
    int *restrict cnt_NA_u_byrow = NULL;
    int *restrict cnt_NA_u_bycol = NULL;
    int *restrict cnt_NA_i_byrow = NULL;
    int *restrict cnt_NA_i_bycol = NULL;
    bool full_dense = false;
    bool near_dense_row = false;
    bool near_dense_col = false;
    bool full_dense_u = false;
    bool near_dense_u_row = false;
    bool near_dense_u_col = false;
    bool full_dense_i = false;
    bool near_dense_i_row = false;
    bool near_dense_i_col = false;

    if (Xfull != NULL || !NA_as_zero_X)
    {
        retval = initialize_biases(
            glob_mean, values, values + (user_bias? m_max : 0),
            user_bias, item_bias,
            (lam_unique == NULL)? (lam) : (lam_unique[0]),
            (lam_unique == NULL)? (lam) : (lam_unique[1]),
            m, n,
            m_max, n_max,
            ixA, ixB, X, nnz,
            Xfull, (FPnum*)NULL,
            (long*)NULL, (int*)NULL, (FPnum*)NULL,
            (long*)NULL, (int*)NULL, (FPnum*)NULL,
            nthreads
        );
        if (retval != 0) goto cleanup;
    }


    if (Xfull != NULL)
    {
        cnt_NA_byrow = (int*)calloc(m, sizeof(int));
        cnt_NA_bycol = (int*)calloc(n, sizeof(int));
        if (cnt_NA_byrow == NULL || cnt_NA_bycol == NULL)
        {
            retval = 1;
            goto cleanup;
        }

        count_NAs_by_row(Xfull, m, n, cnt_NA_byrow, nthreads,
                         &full_dense, &near_dense_row);
        count_NAs_by_col(Xfull, m, n, cnt_NA_bycol,
                         &full_dense, &near_dense_col);
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

    if (Xfull != NULL && ((!full_dense && near_dense_col) || m > m_u))
    {
        Xtrans = (FPnum*)malloc((size_t)m*(size_t)n*sizeof(FPnum));
        if (Xtrans == NULL)
        {
            retval = 1;
            goto cleanup;
        }
        transpose_mat2(Xfull, m, n, Xtrans);

        if (weight != NULL)
        {
            Wtrans = (FPnum*)malloc((size_t)m*(size_t)n*sizeof(FPnum));
            if (Wtrans == NULL)
            {
                retval = 1;
                goto cleanup;
            }
            transpose_mat2(weight, m, n, Wtrans);
        }
    }

    /* For the biases, will do the trick by subtracting the bias from
       all entries before optimizing a given matrix. */
    if (user_bias || item_bias)
    {
        A_bias = (FPnum*)malloc((size_t)m_max*(size_t)(k_totA+1)*sizeof(FPnum));
        if (B_plus_bias == NULL)
            B_bias = (FPnum*)malloc((size_t)n_max * (size_t)(k_totB+1)
                                                  * sizeof(FPnum));
        else
            B_bias = B_plus_bias;
        if (A_bias == NULL || B_bias == NULL) { retval = 1; goto cleanup; }
        
        if (Xcsr != NULL)
        {
            if (item_bias) {
                Xcsr_orig = (FPnum*)malloc(nnz*sizeof(FPnum));
                if (Xcsr_orig == NULL) { retval = 1; goto cleanup; }
                copy_arr(Xcsr, Xcsr_orig, nnz, nthreads);
            }
            if (user_bias) {
                Xcsc_orig = (FPnum*)malloc(nnz*sizeof(FPnum));
                if (Xcsc_orig == NULL) { retval = 1; goto cleanup; }
                copy_arr(Xcsc, Xcsc_orig, nnz, nthreads);
            }    
        }

        if (Xfull != NULL && (item_bias || Xtrans == NULL))
        {
            Xfull_orig = (FPnum*)malloc((size_t)m*(size_t)n*sizeof(FPnum));
            if (Xfull_orig == NULL) {
                retval = 1;
                goto cleanup;
            }
            copy_arr(Xfull, Xfull_orig, (size_t)m*(size_t)n, nthreads);
        }

        if (Xtrans != NULL && user_bias)
        {
            Xtrans_orig = (FPnum*)malloc((size_t)m*(size_t)n*sizeof(FPnum));
            if (Xtrans_orig == NULL) {
                retval = 1;
                goto cleanup;
            }
            copy_arr(Xtrans, Xtrans_orig, (size_t)m*(size_t)n, nthreads);
        }
    }

    else {
        A_bias = A;
        B_bias = B;

        if (Xfull != NULL && weight == NULL)
        {
            Xfull_orig = (FPnum*)malloc((size_t)m*(size_t)n*sizeof(FPnum));
            if (Xfull_orig == NULL) { retval = 1; goto cleanup; }
            copy_arr(Xfull, Xfull_orig, (size_t)m*(size_t)n, nthreads);
        }

        if (Xtrans != NULL && weight == NULL)
        {
            Xtrans_orig = (FPnum*)malloc((size_t)m*(size_t)n*sizeof(FPnum));
            if (Xtrans_orig == NULL) { retval = 1; goto cleanup; }
            copy_arr(Xtrans, Xtrans_orig, (size_t)m*(size_t)n, nthreads);
        }
    }

    if (U != NULL || U_sp != NULL)
    {
        retval = preprocess_sideinfo_matrix(
            U, m_u, p,
            U_row, U_col, U_sp, nnz_U,
            U_colmeans, &Utrans, &U_orig,
            &U_csr_p, &U_csr_i, &U_csr,
            &U_csc_p, &U_csc_i, &U_csc,
            &cnt_NA_u_byrow, &cnt_NA_u_bycol,
            &full_dense_u, &near_dense_u_row, &near_dense_u_col,
            NA_as_zero_U, nthreads
        );
        if (retval != 0) goto cleanup;
    }

    if (II != NULL || I_sp != NULL)
    {
        retval = preprocess_sideinfo_matrix(
            II, n_i, q,
            I_row, I_col, I_sp, nnz_I,
            I_colmeans, &Itrans, &I_orig,
            &I_csr_p, &I_csr_i, &I_csr,
            &I_csc_p, &I_csc_i, &I_csc,
            &cnt_NA_i_byrow, &cnt_NA_i_bycol,
            &full_dense_i, &near_dense_i_row, &near_dense_i_col,
            NA_as_zero_U, nthreads
        );
        if (retval != 0) goto cleanup;
    }

    /* Sizes of the temporary arrays */
    if (U != NULL || U_sp != NULL)
        buffer_size_optimizeA(
            &size_bufferC, &size_buffer_lbfgs,
            p, m_u, k_user+k, k_user+k, nthreads,
            false, NA_as_zero_U, use_cg,
            full_dense_u, near_dense_u_col,
            U != NULL, false
        );
    if (II != NULL || I_sp != NULL)
        buffer_size_optimizeA(
            &size_bufferD, &size_buffer_lbfgs,
            q, n_i, k_item+k, k_item+k, nthreads,
            false, NA_as_zero_I, use_cg,
            full_dense_i, near_dense_i_col,
            II != NULL, false
        );

    if (U != NULL || U_sp != NULL)
        buffer_size_optimizeA_collective(
            &size_bufferA, &size_buffer_lbfgs,
            m, n, k, k_user, k_main+(int)user_bias, paddingA,
            m_u, p, nthreads,
            false, NA_as_zero_X, NA_as_zero_U, use_cg,
            full_dense, near_dense_row,
            Xfull != NULL, weight != NULL,
            full_dense_u, near_dense_u_row, U != NULL
        );
    else
        buffer_size_optimizeA(
            &size_bufferA, &size_buffer_lbfgs,
            m, n, k+k_main+(int)user_bias,
            k_user+k+k_main+(int)(user_bias||item_bias),
            nthreads,
            false, NA_as_zero_X, use_cg,
            full_dense, near_dense_row,
            Xfull != NULL, weight != NULL
        );

    if (II != NULL || I_sp != NULL)
        buffer_size_optimizeA_collective(
            &size_bufferB, &size_buffer_lbfgs,
            n, m, k, k_item, k_main+(int)item_bias, paddingB,
            n_i, q, nthreads,
            (Xfull != NULL && Xtrans == NULL),
            NA_as_zero_X, NA_as_zero_I, use_cg,
            full_dense, near_dense_col,
            Xfull != NULL, weight != NULL,
            full_dense_i, near_dense_i_row, II != NULL
        );
    else
        buffer_size_optimizeA(
            &size_bufferB, &size_buffer_lbfgs,
            n, m, k+k_main+(int)item_bias,
            k_item+k+k_main+(int)(user_bias||item_bias),
            nthreads,
            (Xfull != NULL && Xtrans == NULL), NA_as_zero_X, use_cg,
            full_dense, near_dense_col,
            Xfull != NULL, weight != NULL
        );

    size_buffer = max2(max2(size_bufferA, size_bufferB),
                       max2(size_bufferC, size_bufferD));
    buffer_FPnum = (FPnum*)malloc(size_buffer * sizeof(FPnum));
    if (buffer_FPnum == NULL)
    {
        retval = 1;
        goto cleanup;
    }

    if (size_buffer_lbfgs) {
        buffer_lbfgs_iter = (iteration_data_t*)
                             malloc(size_buffer_lbfgs*sizeof(iteration_data_t));
        if (buffer_lbfgs_iter == NULL)
        {
            retval = 1;
            goto cleanup;
        }
    }

    if (reset_values)
        retval = rnorm(values, nvars, seed, nthreads);
    if (retval != 0) goto cleanup;

    if (user_bias || item_bias)
    {
        copy_mat(m_max,  k_user+k+k_main,
                 A,      k_user+k+k_main,
                 A_bias, k_user+k+k_main + 1);
        copy_mat(n_max,  k_item+k+k_main,
                 B,      k_item+k+k_main,
                 B_bias, k_item+k+k_main + 1);

        /* TODO: one of these two is probably redundant depending on
           parameters, find out and eliminate it. */
        if (user_bias)
            cblas_tcopy(m_max, biasA, 1,
                        A_bias + k_user+k+k_main, k_user+k+k_main + 1);
        else
            for (size_t ix = 0; ix < (size_t)m_max; ix++)
                A_bias[(size_t)(k_user+k+k_main)
                        + ix*(size_t)(k_user+k+k_main + 1)]
                 = 1.;

        if (item_bias)
            cblas_tcopy(n_max, biasB, 1,
                        B_bias + k_item+k+k_main, k_item+k+k_main + 1);
        else
            for (size_t ix = 0; ix < (size_t)n_max; ix++)
                B_bias[(size_t)(k_item+k+k_main)
                        + ix*(size_t)(k_item+k+k_main + 1)]
                 = 1.;
    }

    if (verbose) {
        printf("Starting ALS optimization routine\n\n");
        fflush(stdout);
    }

    for (int iter = 0; iter < niter; iter++)
    {
        /* Optimize C and D (they are independent of each other) */
        signal(SIGINT, set_interrup_global_variable);
        if (should_stop_procedure) goto check_interrupt;
        if (U != NULL || U_sp != NULL) {
            if (verbose) { printf("Updating C..."); fflush(stdout); }

            if (U_orig != NULL)
                copy_arr(U, U_orig, (size_t)m_u*(size_t)p, nthreads);

            optimizeA(
                C, k_user+k,
                A_bias, k_user+k+k_main+(int)(user_bias||item_bias),
                p, m_u, k_user+k,
                U_csc_p, U_csc_i, U_csc,
                (U != NULL && near_dense_u_col)? (Utrans) : (U),
                full_dense_u, near_dense_u_col,
                cnt_NA_u_bycol, (FPnum*)NULL, NA_as_zero_U,
                (lam_unique == NULL)? (lam) : (lam_unique[4]), w_user,
                (lam_unique == NULL)? (lam) : (lam_unique[4]),
                (U != NULL && near_dense_u_col)? (false) : (true),
                nthreads, use_cg,
                buffer_FPnum,
                buffer_lbfgs_iter
            );
            if (verbose) { printf(" done\n"); fflush(stdout); }
        }

        signal(SIGINT, set_interrup_global_variable);
        if (should_stop_procedure) goto check_interrupt;
        if (II != NULL || I_sp != NULL) {
            if (verbose) { printf("Updating D..."); fflush(stdout); }

            if (I_orig != NULL)
                copy_arr(II, I_orig, (size_t)n_i*(size_t)q, nthreads);

            optimizeA(
                D, k_item+k,
                B_bias, k_item+k+k_main+(int)(user_bias||item_bias),
                q, n_i, k_item+k,
                I_csc_p, I_csc_i, I_csc,
                (II != NULL && near_dense_u_col)? (Itrans) : (II),
                full_dense_i, near_dense_i_col,
                cnt_NA_i_bycol, (FPnum*)NULL, NA_as_zero_I,
                (lam_unique == NULL)? (lam) : (lam_unique[5]), w_item,
                (lam_unique == NULL)? (lam) : (lam_unique[5]),
                (II != NULL && near_dense_i_col)? (false) : (true),
                nthreads, use_cg,
                buffer_FPnum,
                buffer_lbfgs_iter
            );
            if (verbose) { printf(" done\n"); fflush(stdout); }
        }

        /* Apply bias beforehand, as its column will be fixed */
        if (user_bias)
        {
            if (Xtrans != NULL) {
                #pragma omp parallel for schedule(static) \
                        num_threads(cap_to_4(nthreads)) \
                        shared(Xtrans, m, n, biasA)
                for (size_t_for col = 0; col < (size_t)n; col++)
                    for (size_t row = 0; row < (size_t)m; row++)
                        Xtrans[row + col*m] = Xtrans_orig[row + col*m]
                                               - biasA[row];
            }
            else if (Xfull != NULL) {
                for (size_t row = 0; row < (size_t)m; row++)
                    for (size_t col = 0; col < (size_t)n; col++)
                        Xfull[col + row*n] = Xfull_orig[col + row*n]
                                              -  biasA[row];
            }
            else {
                #pragma omp parallel for schedule(static) \
                        num_threads(cap_to_4(nthreads)) \
                        shared(nnz, Xcsc, Xcsc_i, biasA)
                for (size_t_for ix = 0; ix < nnz; ix++)
                    Xcsc[ix] = Xcsc_orig[ix] - biasA[Xcsc_i[ix]];
            }

            if (item_bias)
                for (int ix = 0; ix < m_max; ix++)
                    A_bias[(size_t)(k_user+k+k_main)
                            + ix*(size_t)(k_user+k+k_main + 1)] = 1.;
        }

        else if (Xfull_orig != NULL || Xtrans_orig != NULL)
        {
            if (Xtrans_orig != NULL)
                copy_arr(Xtrans, Xtrans_orig, (size_t)m*(size_t)n, nthreads);
            else
                copy_arr(Xfull, Xfull_orig, (size_t)m*(size_t)n, nthreads);
        }

        /* Optimize B */
        signal(SIGINT, set_interrup_global_variable);
        if (should_stop_procedure) goto check_interrupt;
        if (verbose) { printf("Updating B..."); fflush(stdout); }
        if (II != NULL || I_sp != NULL)
        {
            if (I_orig != NULL)
                copy_arr(II, I_orig, (size_t)n_i*(size_t)q, nthreads);

            optimizeA_collective(
                B_bias, A_bias, D,
                n, n_i, m, q,
                k, k_main+(int)item_bias, k_item, k_user, paddingB,
                Xcsc_p, Xcsc_i, Xcsc,
                (Xtrans != NULL)? (Xtrans) : (Xfull),
                full_dense, near_dense_col,
                cnt_NA_bycol,
                (Xfull == NULL)? (weightC) : (weight),
                NA_as_zero_X,
                I_csr_p, I_csr_i, I_csr,
                II, cnt_NA_i_byrow,
                full_dense_i, near_dense_i_row, NA_as_zero_I,
                (lam_unique == NULL)? (lam) : (lam_unique[3]),
                w_main, w_item,
                (lam_unique == NULL)? (lam) : (lam_unique[1]),
                Xfull != NULL && Xtrans == NULL,
                nthreads, use_cg,
                buffer_FPnum,
                buffer_lbfgs_iter
            );
        }
        else
            optimizeA(
                B_bias + k_item, k_item+k+k_main+(int)(user_bias||item_bias),
                A_bias + k_user, k_user+k+k_main+(int)(user_bias||item_bias),
                n, m, k+k_main+(int)item_bias,
                Xcsc_p, Xcsc_i, Xcsc,
                (Xtrans != NULL)? (Xtrans) : (Xfull),
                full_dense, near_dense_col,
                cnt_NA_bycol, (FPnum*)NULL, NA_as_zero_X,
                (lam_unique == NULL)? (lam) : (lam_unique[3]), w_main,
                (lam_unique == NULL)? (lam) : (lam_unique[1]),
                Xfull != NULL && Xtrans == NULL,
                nthreads, use_cg,
                buffer_FPnum,
                buffer_lbfgs_iter
            );
        if (verbose) { printf(" done\n"); fflush(stdout); }

        if (item_bias)
            cblas_tcopy(n_max, B_bias + k_item+k+k_main, k_item+k+k_main + 1,
                        biasB, 1);

        /* Apply bias beforehand, as its column will be fixed */
        if (item_bias)
        {
            if (Xfull != NULL) {
                #pragma omp parallel for schedule(static) \
                        num_threads(cap_to_4(nthreads)) \
                        shared(Xfull, m, n, biasB)
                for (size_t_for row = 0; row < (size_t)m; row++)
                    for (size_t col = 0; col < (size_t)n; col++)
                        Xfull[col + row*n] = Xfull_orig[col + row*n]
                                              - biasB[col];
            }
            else {
                #pragma omp parallel for schedule(static) \
                        num_threads(cap_to_4(nthreads)) \
                        shared(nnz, Xcsr, Xcsr_i, biasB)
                for (size_t_for ix = 0; ix < nnz; ix++)
                    Xcsr[ix] = Xcsr_orig[ix] - biasB[Xcsr_i[ix]];
            }

            if (user_bias)
                for (int ix = 0; ix < n_max; ix++)
                    B_bias[(size_t)(k_item+k+k_main)
                            + ix*(size_t)(k_item+k+k_main + 1)] = 1.;
        }

        else if (Xfull_orig != NULL)
        {
            copy_arr(Xfull, Xfull_orig, (size_t)m*(size_t)n, nthreads);
        }

        /* Optimize A */
        signal(SIGINT, set_interrup_global_variable);
        if (should_stop_procedure) goto check_interrupt;
        if (verbose) { printf("Updating A..."); fflush(stdout); }
        if (U != NULL || U_sp != NULL)
        {
            if (U_orig != NULL)
                copy_arr(U, U_orig, (size_t)m_u*(size_t)p, nthreads);

            optimizeA_collective(
                A_bias, B_bias, C,
                m, m_u, n, p,
                k, k_main+(int)user_bias, k_user, k_item, paddingA,
                Xcsr_p, Xcsr_i, Xcsr,
                Xfull, full_dense, near_dense_row,
                cnt_NA_byrow,
                (Xfull == NULL)? (weightR) : (weight),
                NA_as_zero_X,
                U_csr_p, U_csr_i, U_csr,
                U, cnt_NA_u_byrow,
                full_dense_u, near_dense_u_row, NA_as_zero_U,
                (lam_unique == NULL)? (lam) : (lam_unique[2]),
                w_main, w_user,
                (lam_unique == NULL)? (lam) : (lam_unique[0]),
                false,
                nthreads, use_cg,
                buffer_FPnum,
                buffer_lbfgs_iter
            );
        }
        else
            optimizeA(
                A_bias + k_user, k_user+k+k_main+(int)(user_bias||item_bias),
                B_bias + k_item, k_item+k+k_main+(int)(user_bias||item_bias),
                m, n, k+k_main+(int)user_bias,
                Xcsr_p, Xcsr_i, Xcsr,
                Xfull,
                full_dense, near_dense_row,
                cnt_NA_byrow, (FPnum*)NULL, NA_as_zero_X,
                (lam_unique == NULL)? (lam) : (lam_unique[2]), w_main,
                (lam_unique == NULL)? (lam) : (lam_unique[0]),
                false,
                nthreads, use_cg,
                buffer_FPnum,
                buffer_lbfgs_iter
            );
        if (verbose) { printf(" done\n"); fflush(stdout); }

        if (user_bias)
            cblas_tcopy(m_max, A_bias + k_user+k+k_main, k_user+k+k_main + 1,
                        biasA, 1);

        if (verbose) {
            printf("\tCompleted ALS iteration %2d\n\n", iter+1);
            fflush(stdout);
        }
        check_interrupt:
            if (should_stop_procedure) {
                should_stop_procedure = false;
                goto cleanup;
            }
    }

    if (verbose) {
        if (!isnan(A[k_user]))
            printf("ALS procedure terminated successfully\n");
        else
            printf("ALS procedure failed\n");
        fflush(stdout);
    }

    if (user_bias || item_bias)
    {
        copy_mat(
            m_max,  k_user+k+k_main,
            A_bias, k_user+k+k_main + 1,
            A,      k_user+k+k_main
        );
        copy_mat(
            n_max,  k_item+k+k_main,
            B_bias, k_item+k+k_main + 1,
            B,      k_item+k+k_main
        );
    }

    cleanup:
        free(buffer_FPnum);
        free(buffer_lbfgs_iter);
        free(Xtrans);
        free(Wtrans);
        free(Xcsr_p);
        free(Xcsr_i);
        free(Xcsr);
        free(weightR);
        free(Xcsc_p);
        free(Xcsc_i);
        free(Xcsc);
        free(weightC);
        free(Utrans);
        free(U_csr_p);
        free(U_csr_i);
        free(U_csr);
        free(U_csc_p);
        free(U_csc_i);
        free(U_csc);
        free(Itrans);
        free(I_csr_p);
        free(I_csr_i);
        free(I_csr);
        free(I_csc_p);
        free(I_csc_i);
        free(I_csc);
        free(cnt_NA_byrow);
        free(cnt_NA_bycol);
        free(cnt_NA_u_byrow);
        free(cnt_NA_u_bycol);
        free(cnt_NA_i_byrow);
        free(cnt_NA_i_bycol);
        if (user_bias || item_bias) {
            free(A_bias);
            if (B_plus_bias != B_bias)
                free(B_bias);
            free(Xcsr_orig);
            free(Xcsc_orig);
        }
        free(Xfull_orig);
        free(Xtrans_orig);
        free(U_orig);
        free(I_orig);
    if (retval == 1) return 1;

    return 0;
}

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
)
{
    int k_totA = k_user + k + k_main;
    int k_totB = k_item + k + k_main;
    int m_max = max2(m, m_u);
    int n_max = max2(n, n_i);

    FPnum *restrict A = values;
    FPnum *restrict B = A + (size_t)m_max * (size_t)k_totA;
    FPnum *restrict C = B + (size_t)n_max * (size_t)k_totB;
    FPnum *restrict D  = C + (size_t)(k_user + k) * (size_t)p;
    size_t nvars = (D + (size_t)(k_item + k) * (size_t)q) - values;

    int retval = 0;
    FPnum *restrict buffer_FPnum = NULL;
    iteration_data_t *buffer_lbfgs_iter = NULL;
    size_t size_bufferA = (size_t)(nthreads+2)*(size_t)square(k_user+k+k_main);
    size_t size_bufferB = (size_t)(nthreads+2)*(size_t)square(k_item+k+k_main);
    size_t size_bufferC = 0;
    size_t size_bufferD = 0;
    size_t size_buffer = 0;
    size_t size_buffer_lbfgs = 0;
    size_t alt_size = 0;
    size_t alt_lbfgs = 0;

    long *Xcsr_p = (long*)malloc((size_t)(m+1)*sizeof(long));
    int *Xcsr_i = (int*)malloc(nnz*sizeof(int));
    FPnum *restrict Xcsr = (FPnum*)malloc(nnz*sizeof(FPnum));
    long *Xcsc_p = (long*)malloc((size_t)(n+1)*sizeof(long));
    int *Xcsc_i = (int*)malloc(nnz*sizeof(int));
    FPnum *restrict Xcsc = (FPnum*)malloc(nnz*sizeof(FPnum));
    FPnum *restrict Utrans = NULL;
    long *U_csr_p = NULL;
    int *U_csr_i = NULL;
    FPnum *restrict U_csr = NULL;
    long *U_csc_p = NULL;
    int *U_csc_i = NULL;
    FPnum *restrict U_csc = NULL;
    FPnum *restrict Itrans = NULL;
    long *I_csr_p = NULL;
    int *I_csr_i = NULL;
    FPnum *restrict I_csr = NULL;
    long *I_csc_p = NULL;
    int *I_csc_i = NULL;
    FPnum *restrict I_csc = NULL;
    int *restrict cnt_NA_u_byrow = NULL;
    int *restrict cnt_NA_u_bycol = NULL;
    int *restrict cnt_NA_i_byrow = NULL;
    int *restrict cnt_NA_i_bycol = NULL;
    FPnum *restrict U_orig = NULL;
    FPnum *restrict I_orig = NULL;
    bool full_dense_u = false;
    bool near_dense_u_row = false;
    bool near_dense_u_col = false;
    bool full_dense_i = false;
    bool near_dense_i_row = false;
    bool near_dense_i_col = false;

    if (Xcsr_p == NULL || Xcsr_i == NULL || Xcsr == NULL ||
        Xcsc_p == NULL || Xcsc_i == NULL || Xcsc == NULL)
    {
        retval = 1;
        goto cleanup;
    }
    coo_to_csr_and_csc(
        ixA, ixB, X,
        (FPnum*)NULL, m, n, nnz,
        Xcsr_p, Xcsr_i, Xcsr,
        Xcsc_p, Xcsc_i, Xcsc,
        (FPnum*)NULL, (FPnum*)NULL,
        nthreads
    );

    retval = preprocess_sideinfo_matrix(
        U, m_u, p,
        U_row, U_col, U_sp, nnz_U,
        U_colmeans, &Utrans, &U_orig,
        &U_csr_p, &U_csr_i, &U_csr,
        &U_csc_p, &U_csc_i, &U_csc,
        &cnt_NA_u_byrow, &cnt_NA_u_bycol,
        &full_dense_u, &near_dense_u_row, &near_dense_u_col,
        NA_as_zero_U, nthreads
    );
    if (retval != 0) goto cleanup;
    if (U != NULL || U_sp != NULL)
        buffer_size_optimizeA(
            &size_bufferC, &size_buffer_lbfgs,
            p, m_u, k_user+k, k_user+k, nthreads,
            false, NA_as_zero_U, use_cg,
            full_dense_u, near_dense_u_col,
            U != NULL, false
        );

    
    retval = preprocess_sideinfo_matrix(
        II, n_i, q,
        I_row, I_col, I_sp, nnz_I,
        I_colmeans, &Itrans, &I_orig,
        &I_csr_p, &I_csr_i, &I_csr,
        &I_csc_p, &I_csc_i, &I_csc,
        &cnt_NA_i_byrow, &cnt_NA_i_bycol,
        &full_dense_i, &near_dense_i_row, &near_dense_i_col,
        NA_as_zero_I, nthreads
    );
    if (retval != 0) goto cleanup;
    if (II != NULL || I_sp != NULL)
        buffer_size_optimizeA(
            &size_bufferD, &size_buffer_lbfgs,
            q, n_i, k_item+k, k_item+k, nthreads,
            false, NA_as_zero_I, use_cg,
            full_dense_i, near_dense_i_col,
            II != NULL, false
        );

    if (use_cg) {
        size_bufferA += (size_t)nthreads*(size_t)6*(size_t)(k_user+k+k_main);
        size_bufferB += (size_t)nthreads*(size_t)6*(size_t)(k_item+k+k_main);
    }

    if (m_u > m) {
        buffer_size_optimizeA(
            &alt_size, &alt_lbfgs,
            m_u-m, p, k, k_user+k+k_main, nthreads,
            false, NA_as_zero_U, use_cg,
            full_dense_u, near_dense_u_row,
            U != NULL, false
        );
        size_bufferA = max2(size_bufferA, alt_size);
        size_buffer_lbfgs = max2(size_buffer_lbfgs, alt_lbfgs);
    }

    if (n_i > n) {
        buffer_size_optimizeA(
            &alt_size, &alt_lbfgs,
            n_i-n, q, k, k_item+k+k_main, nthreads,
            false, NA_as_zero_I, use_cg,
            full_dense_i, near_dense_i_col,
            II != NULL, false
        );
        size_bufferA = max2(size_bufferA, alt_size);
        size_buffer_lbfgs = max2(size_buffer_lbfgs, alt_lbfgs);
    }

    size_buffer = max2(max2(size_bufferA, size_bufferB),
                       max2(size_bufferC, size_bufferD));
    buffer_FPnum = (FPnum*)malloc(size_buffer * sizeof(FPnum));
    if (buffer_FPnum == NULL)
    {
        retval = 1;
        goto cleanup;
    }

    if (size_buffer_lbfgs) {
        buffer_lbfgs_iter = (iteration_data_t*)
                             malloc(size_buffer_lbfgs*sizeof(iteration_data_t));
        if (buffer_lbfgs_iter == NULL)
        {
            retval = 1;
            goto cleanup;
        }
    }

    if (reset_values)
        retval = rnorm(values, nvars, seed, nthreads);
    if (retval != 0) goto cleanup;

    *w_main_multiplier = 1.;
    if (adjust_weight)
    {
        *w_main_multiplier = (double)nnz / (double)((size_t)m * (size_t)n);
        w_main *= *w_main_multiplier;
    }
    

    if (verbose) {
        printf("Starting ALS optimization routine\n\n");
        fflush(stdout);
    }

    for (int iter = 0; iter < niter; iter++)
    {
        /* Optimize C and D (they are independent of each other) */
        signal(SIGINT, set_interrup_global_variable);
        if (should_stop_procedure) goto check_interrupt;
        if (U != NULL || U_sp != NULL) {
            if (verbose) { printf("Updating C..."); fflush(stdout); }

            if (U_orig != NULL)
                copy_arr(U, U_orig, (size_t)m_u*(size_t)p, nthreads);

            optimizeA(
                C, k_user+k,
                A, k_user+k+k_main,
                p, m_u, k_user+k,
                U_csc_p, U_csc_i, U_csc,
                (U != NULL && near_dense_u_col)? (Utrans) : (U),
                full_dense_u, near_dense_u_col,
                cnt_NA_u_bycol, (FPnum*)NULL, NA_as_zero_U,
                (lam_unique == NULL)? (lam) : (lam_unique[4]), w_user,
                (lam_unique == NULL)? (lam) : (lam_unique[4]),
                (U != NULL && near_dense_u_col)? (false) : (true),
                nthreads, use_cg,
                buffer_FPnum,
                buffer_lbfgs_iter
            );
            if (verbose) { printf(" done\n"); fflush(stdout); }
        }

        signal(SIGINT, set_interrup_global_variable);
        if (should_stop_procedure) goto check_interrupt;
        if (II != NULL || I_sp != NULL) {
            if (verbose) { printf("Updating D..."); fflush(stdout); }

            if (I_orig != NULL)
                copy_arr(II, I_orig, (size_t)n_i*(size_t)q, nthreads);

            optimizeA(
                D, k_item+k,
                B, k_item+k+k_main,
                q, n_i, k_item+k,
                I_csc_p, I_csc_i, I_csc,
                (II != NULL && near_dense_u_col)? (Itrans) : (II),
                full_dense_i, near_dense_i_col,
                cnt_NA_i_bycol, (FPnum*)NULL, NA_as_zero_I,
                (lam_unique == NULL)? (lam) : (lam_unique[5]), w_item,
                (lam_unique == NULL)? (lam) : (lam_unique[5]),
                (II != NULL && near_dense_i_col)? (false) : (true),
                nthreads, use_cg,
                buffer_FPnum,
                buffer_lbfgs_iter
            );
            if (verbose) { printf(" done\n"); fflush(stdout); }
        }

        /* Optimize B */
        signal(SIGINT, set_interrup_global_variable);
        if (should_stop_procedure) goto check_interrupt;
        if (verbose) { printf("Updating B..."); fflush(stdout); }
        if (II != NULL || I_sp != NULL)
        {
            if (I_orig != NULL)
                copy_arr(II, I_orig, (size_t)n_i*(size_t)q, nthreads);

            optimizeA_collective_implicit(
                B, A, D,
                n, n_i, m, q,
                k, k_main, k_item, k_user,
                Xcsc_p, Xcsc_i, Xcsc,
                I_csr_p, I_csr_i, I_csr,
                II, cnt_NA_i_byrow,
                full_dense_i, near_dense_i_row, NA_as_zero_I,
                (lam_unique == NULL)? (lam) : (lam_unique[3]),
                alpha, w_main, w_item,
                nthreads, use_cg,
                buffer_FPnum,
                buffer_lbfgs_iter
            );
        }
        else
            optimizeA_implicit(
                B + k_item, k_item+k+k_main,
                A + k_user, k_user+k+k_main,
                n, m, k+k_main,
                Xcsc_p, Xcsc_i, Xcsc,
                (lam_unique == NULL)? (lam/w_main) : (lam_unique[3]/w_main),
                alpha,
                nthreads, use_cg,
                buffer_FPnum
            );
        if (verbose) { printf(" done\n"); fflush(stdout); }

        /* Optimize A */
        signal(SIGINT, set_interrup_global_variable);
        if (should_stop_procedure) goto check_interrupt;
        if (verbose) { printf("Updating A..."); fflush(stdout); }
        if (U != NULL || U_sp != NULL)
        {
            if (U_orig != NULL)
                copy_arr(U, U_orig, (size_t)m_u*(size_t)p, nthreads);

            optimizeA_collective_implicit(
                A, B, C,
                m, m_u, n, p,
                k, k_main, k_user, k_item,
                Xcsr_p, Xcsr_i, Xcsr,
                U_csr_p, U_csr_i, U_csr,
                U, cnt_NA_u_byrow,
                full_dense_u, near_dense_u_row, NA_as_zero_U,
                (lam_unique == NULL)? (lam) : (lam_unique[2]),
                alpha, w_main, w_user,
                nthreads, use_cg,
                buffer_FPnum,
                buffer_lbfgs_iter
            );
        }
        else
            optimizeA_implicit(
                A + k_user, k_user+k+k_main,
                B + k_item, k_item+k+k_main,
                m, n, k+k_main,
                Xcsr_p, Xcsr_i, Xcsr,
                (lam_unique == NULL)? (lam/w_main) : (lam_unique[2]/w_main),
                alpha,
                nthreads, use_cg,
                buffer_FPnum
            );
        if (verbose) { printf(" done\n"); fflush(stdout); }

        
        if (verbose) {
            printf("\tCompleted ALS iteration %2d\n\n", iter+1);
            fflush(stdout);
        }
        check_interrupt:
            if (should_stop_procedure) {
                should_stop_procedure = false;
                goto cleanup;
            }

    }

    if (verbose) {
        if (!isnan(A[k_user]))
            printf("ALS procedure terminated successfully\n");
        else
            printf("ALS procedure failed\n");
        fflush(stdout);
    }

    cleanup:
        free(buffer_FPnum);
        free(buffer_lbfgs_iter);
        free(Xcsr_p);
        free(Xcsr_i);
        free(Xcsr);
        free(Xcsc_p);
        free(Xcsc_i);
        free(Xcsc);
        free(Utrans);
        free(U_csr_p);
        free(U_csr_i);
        free(U_csr);
        free(U_csc_p);
        free(U_csc_i);
        free(U_csc);
        free(Itrans);
        free(I_csr_p);
        free(I_csr_i);
        free(I_csr);
        free(I_csc_p);
        free(I_csc_i);
        free(I_csc);
        free(cnt_NA_u_byrow);
        free(cnt_NA_u_bycol);
        free(cnt_NA_i_byrow);
        free(cnt_NA_i_bycol);
        free(U_orig);
        free(I_orig);
    if (retval == 1) return 1;

    return 0;
}

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
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif

    size_t k_totA = k_user + k + k_main;
    long *restrict U_csr_p_use = NULL;
    int *restrict U_csr_i_use = NULL;
    FPnum *restrict U_csr_use = NULL;

    int retval = 0;
    int *ret = (int*)malloc(m*sizeof(int));
    if (ret == NULL) { retval = 1; goto cleanup; }

    if (U_sp != NULL && U_csr == NULL) {
        retval = coo_to_csr_plus_alloc(
                    U_row, U_col, U_sp, (FPnum*)NULL,
                    m_u, p, nnz_U,
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
            shared(ret, m, m_u, m_ubin, A, k_totA, p, pbin, \
                   U, Ub, U_csr_use, U_csr_i_use, U_csr_p_use, col_means, \
                   k, k_user, k_main, lam, w_user, NA_as_zero_U, \
                   C, Cb, CtCinvCt, CtCw, CtCchol)
    for (size_t_for ix = 0; ix < (size_t)m; ix++)
        ret[ix] = collective_factors_cold(
                    A + ix*k_totA,
                    (U == NULL || (int)ix >= m_u)?
                      ((FPnum*)NULL)
                      : (U + ix*(size_t)p),
                    p,
                    (U_csr_use == NULL || (int)ix >= m_u)?
                      ((FPnum*)NULL)
                      : (U_csr_use + U_csr_p_use[ix]),
                    (U_csr_i_use == NULL || (int)ix >= m_u)?
                      ((int*)NULL)
                      : (U_csr_i_use + U_csr_p_use[ix]),
                    (U_csr_p_use == NULL)?
                      (0) : (U_csr_p_use[ix+1] - U_csr_p_use[ix]),
                    (Ub == NULL || (int)ix >= m_ubin)?
                      ((FPnum*)NULL)
                      : (Ub + ix*(size_t)pbin),
                    pbin,
                    C, Cb,
                    CtCinvCt,
                    CtCw,
                    CtCchol,
                    col_means,
                    k, k_user, k_main,
                    lam, w_user,
                    NA_as_zero_U
                );

    for (int ix = 0; ix < m; ix++)
        retval = (ret[ix] != 0)? 1 : retval;
    cleanup:
        free(ret);
        if (U_csr_p_use != U_csr_p)
            free(U_csr_p_use);
        if (U_csr_i_use != U_csr_i)
            free(U_csr_i_use);
        if (U_csr_use != U_csr)
            free(U_csr_use);
    return retval;
}

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
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif

    size_t k_totA = k_user + k + k_main;
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
                    m_x, n, nnz,
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
                    m_u, p, nnz_U,
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
            shared(A, biasA, k_totA, m, n, ret, m_u, m_x, m_ubin, p, pbin, \
                   glob_mean, biasB, col_means, weight, weightR, \
                   Xfull, Xcsr_p_use, Xcsr_i_use, Xcsr_use, X, \
                   U, Ub, U_csr_use, U_csr_i_use, U_csr_p_use, \
                   B, B_plus_bias, C, Cb, \
                   k, k_user, k_item, k_main, \
                   lam, w_user, w_main, lam_bias, \
                   BtBinvBt, BtBw, BtBchol, CtCw, \
                   k_item_BtB, NA_as_zero_U, NA_as_zero_X)
    for (size_t_for ix = 0; ix < (size_t)m_x; ix++)
        ret[ix] = collective_factors_warm(
                    A + ix*k_totA,
                    (biasA == NULL)? ((FPnum*)NULL) : (biasA + ix),
                    (U == NULL || (int)ix >= m_u)?
                      ((FPnum*)NULL)
                      : (U + ix*(size_t)p),
                    p,
                    (U_csr_use == NULL || (int)ix >= m_u)?
                      ((FPnum*)NULL)
                      : (U_csr_use + U_csr_p_use[ix]),
                    (U_csr_i_use == NULL || (int)ix >= m_u)?
                      ((int*)NULL)
                      : (U_csr_i_use + U_csr_p_use[ix]),
                    (U_csr_p_use == NULL)?
                      (0) : (U_csr_p_use[ix+1] - U_csr_p_use[ix]),
                    (Ub == NULL || (int)ix >= m_ubin)?
                      ((FPnum*)NULL)
                      : (Ub + ix*(size_t)pbin),
                    pbin,
                    C, Cb,
                    glob_mean, biasB,
                    col_means,
                    (Xcsr_use == NULL)?
                      ((FPnum*)NULL)
                      : (Xcsr_use + Xcsr_p_use[ix]),
                    (Xcsr_i_use == NULL)?
                      ((int*)NULL)
                      : (Xcsr_i_use + Xcsr_p_use[ix]),
                    (Xcsr_p_use == NULL)?
                      (0) : (Xcsr_p_use[ix+1] - Xcsr_p_use[ix]),
                    (Xfull != NULL)? (Xfull + ix*(size_t)n) : ((FPnum*)NULL),
                    n,
                    (Xfull != NULL)? weight : weightR,
                    B,
                    k, k_user, k_item, k_main,
                    lam, w_user, w_main, lam_bias,
                    BtBinvBt,
                    BtBw,
                    BtBchol,
                    CtCw,
                    k_item_BtB,
                    NA_as_zero_U, NA_as_zero_X,
                    B_plus_bias
                );

    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(ret, m, m_u, m_ubin, A, k_totA, p, pbin, \
                   U, Ub, U_csr_use, U_csr_i_use, U_csr_p_use, col_means, \
                   k, k_user, k_main, lam, w_user, NA_as_zero_U, \
                   C, Cb, CtCinvCt, CtCw, CtCchol)
    for (size_t_for ix = m_x; ix < (size_t)m; ix++) {
        ret[ix] = collective_factors_cold(
                    A + ix*k_totA,
                    (U == NULL || (int)ix >= m_u)?
                      ((FPnum*)NULL)
                      : (U + ix*(size_t)p),
                    p,
                    (U_csr_use == NULL || (int)ix >= m_u)?
                      ((FPnum*)NULL)
                      : (U_csr_use + U_csr_p_use[ix]),
                    (U_csr_i_use == NULL || (int)ix >= m_u)?
                      ((int*)NULL)
                      : (U_csr_i_use + U_csr_p_use[ix]),
                    (U_csr_p_use == NULL)?
                      (0) : (U_csr_p_use[ix+1] - U_csr_p_use[ix]),
                    (Ub == NULL || (int)ix >= m_ubin)?
                      ((FPnum*)NULL)
                      : (Ub + ix*(size_t)pbin),
                    pbin,
                    C, Cb,
                    CtCinvCt,
                    CtCw,
                    CtCchol,
                    col_means,
                    k, k_user, k_main,
                    lam, w_user,
                    NA_as_zero_U
                );
        if (biasA != NULL) biasA[ix] = 0;
    }


    for (int ix = 0; ix < m; ix++)
        retval = (ret[ix] != 0)? 1 : retval;
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
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif

    size_t k_totA = k_user + k + k_main;
    long *restrict Xcsr_p_use = NULL;
    int *restrict Xcsr_i_use = NULL;
    FPnum *restrict Xcsr_use = NULL;

    long *restrict U_csr_p_use = NULL;
    int *restrict U_csr_i_use = NULL;
    FPnum *restrict U_csr_use = NULL;

    int retval = 0;
    int *ret = (int*)malloc(m*sizeof(int));
    if (ret == NULL) { retval = 1; goto cleanup; }

    
    if (Xcsr == NULL) {
        retval = coo_to_csr_plus_alloc(
                    ixA, ixB, X, (FPnum*)NULL,
                    m_x, n, nnz,
                    &Xcsr_p_use, &Xcsr_i_use, &Xcsr_use, (FPnum**)NULL
                );
        if (retval != 0) goto cleanup;
    }
    else {
        Xcsr_p_use = Xcsr_p;
        Xcsr_i_use = Xcsr_i;
        Xcsr_use = Xcsr;
    }

    if (U_sp != NULL && U_csr == NULL) {
        retval = coo_to_csr_plus_alloc(
                    U_row, U_col, U_sp, (FPnum*)NULL,
                    m_u, p, nnz_U,
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
            shared(ret, A, B, C, m, m_x, m_u, n, p, \
                   U, U_csr_p_use, U_csr_i_use, U_csr_use, \
                   Xcsr_p_use, Xcsr_i_use, Xcsr_use, \
                   col_means, NA_as_zero_U, \
                   k, k_user, k_item, k_main, lam, alpha, w_user, w_main, \
                   w_main_multiplier, k_item_BtB, \
                   precomputedBeTBe, precomputedBtB, precomputedBtB_shrunk)
    for (size_t_for ix = 0; ix < (size_t)m_x; ix++)
        ret[ix] = collective_factors_warm_implicit(
                    A + ix*k_totA,
                    (U == NULL || (int)ix >= m_u)?
                      ((FPnum*)NULL)
                      : (U + ix*(size_t)p),
                    p,
                    (U_csr_use == NULL || (int)ix >= m_u)?
                      ((FPnum*)NULL)
                      : (U_csr_use + U_csr_p_use[ix]),
                    (U_csr_i_use == NULL || (int)ix >= m_u)?
                      ((int*)NULL)
                      : (U_csr_i_use + U_csr_p_use[ix]),
                    (U_csr_p_use == NULL)?
                      (0) : (U_csr_p_use[ix+1] - U_csr_p_use[ix]),
                    NA_as_zero_U,
                    col_means,
                    B, n, C,
                    Xcsr_use + Xcsr_p_use[ix],
                    Xcsr_i_use + Xcsr_p_use[ix],
                    Xcsr_p_use[ix+1] - Xcsr_p_use[ix],
                    k, k_user, k_item, k_main,
                    lam, alpha, w_user, w_main,
                    w_main_multiplier,
                    precomputedBeTBe,
                    precomputedBtB,
                    precomputedBtB_shrunk,
                    k_item_BtB
                );

    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(ret, A, C, m, m_x, m_u, p, \
                   U, U_csr_p_use, U_csr_i_use, U_csr_use, \
                   col_means, NA_as_zero_U, \
                   k, k_user, k_main, lam, w_user, \
                   CtCinvCt, CtCw, CtCchol)
    for (size_t_for ix = m_x; ix < (size_t)m; ix++)
        ret[ix] = collective_factors_cold(
                    A + ix*k_totA,
                    (U == NULL || (int)ix >= m_u)?
                      ((FPnum*)NULL)
                      : (U + ix*(size_t)p),
                    p,
                    (U_csr_use == NULL || (int)ix >= m_u)?
                      ((FPnum*)NULL)
                      : (U_csr_use + U_csr_p_use[ix]),
                    (U_csr_i_use == NULL || (int)ix >= m_u)?
                      ((int*)NULL)
                      : (U_csr_i_use + U_csr_p_use[ix]),
                    (U_csr_p_use == NULL)?
                      (0) : (U_csr_p_use[ix+1] - U_csr_p_use[ix]),
                    (FPnum*)NULL,
                    0,
                    C, (FPnum*)NULL,
                    CtCinvCt,
                    CtCw,
                    CtCchol,
                    col_means,
                    k, k_user, k_main,
                    lam, w_user,
                    NA_as_zero_U
                );


    for (int ix = 0; ix < m; ix++)
        retval = (ret[ix] != 0)? 1 : retval;
    cleanup:
        free(ret);
        if (Xcsr_p_use != Xcsr_p) free(Xcsr_p_use);
        if (Xcsr_i_use != Xcsr_i) free(Xcsr_i_use);
        if (Xcsr_use != Xcsr) free(Xcsr_use);
        if (U_csr_p_use != U_csr_p) free(U_csr_p_use);
        if (U_csr_i_use != U_csr_i) free(U_csr_i_use);
        if (U_csr_use != U_csr) free(U_csr_use);
    return retval;
}
