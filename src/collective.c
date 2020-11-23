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

    The basic model is complemented with side information about the users and
    the items in the form of matrices U[m,p], I[n,q], which are also factorized
    using the same A and B matrices as before, which are multiplied by new
    matrices C[p,k] and D[q,k] - e.g.:
        min ||M . (X - A*t(B))||^2 + ||U - A*t(C)||^2 + ||I - B*t(D)||^2

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
        grad(A) = (W[m,n] . M[m,n] . (A*t(B) - X - b1 - b2 - mu)) * B
                  + (Mu[m,p] . (A*t(C) - U))*C
                  + (Mb.(Ub-sigm(A*t(Cb)))*exp(-A*t(Cb))/(exp(-A*t(Cb))+1)^2)*Cb
                  + lambda * A
    (The function value needs to be divided by 2 to match with the gradient
     calculated like that)

    For the closed-form solution with no binary variables, assuming that the
    matrix A is a block matrix composed of independent components in this order
    (first user-independent, then shared, then rating-independent):
        Ae = [Au, As, Am]
    The solution can be obtained **for each row 'a' of 'A'** by factorizing an
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

    The solution is then given by:
        A* = inv(t(Be.W)*Be + diag(lambda)) * (t(Be.W)*t(Xa))
    
    Note that since the left-hand side is, by definition, a symmetric
    positive-semi definite matrix, this computation can be done with
    specialized procedures based on e.g. a Cholesky factorization, rather than
    a more general linear solver or matrix inversion.

    Also note that 'w_main' and 'w_user' can be incorporated efficiently by
    rescaling the rows once they sum into t(Be)*Be - that is, it can be updated
    like this:
        T := 0[k_user+k+k_main, k_user+k+k_main]
        T(k_user:,k_user:)      += w_main*t(B)*B
        T(:k_user+k, :k_user+k) += w_user*t(C)*C
        <T is now equal to t(Be)*Be>
    What's more, it's possible to simplify out one of the 3 weights by dividing
    the other two and the regularization by it. Here, 'w_main' is the one that
    gets simplified out (this also allows simpler functions for the
    non-collective factors) by dividing the others (w_user, w_main, lambda)
    by it.

    
    As an alternative to the Cholesky method, can also use a Conjugate Gradient
    method which follows an iterative procedure for each row a[n] of A,
    taking the corresponding vectors u[p] from 'U', x[n] from 'X',
    **assuming that all rows in 'B' and 'C' for which the corresponding value
    in x[n] or u[p] is missing are set to zero**, iterating as follows:
        r[k_user+k+k_main] := 0
        r(k_user:)   += w_main * (t(B)*x - t(B)*B*a(k_user:))
        r(:k_user+k) += w_user * (t(C)*u - t(C)*C*a(:k_user+k))
        r(:) += lambda * a
        pp[k_user+k+k_main] := r(:)
        ap[k_user+k+k_main] := 0
        r_old = ||r||^2
        for i..s:
            ap(:) := 0
            ap(k_user:)   += w_main * t(B)*B*pp(k_user:)
            ap(:k_user+k) += w_user * t(C)*C*pp(:k_user+k)
            ap(:) += lambda * pp
            a(:) += (r_old / <pp, ap>) * pp
            r(:) -= (r_old / <pp, ap>) * ap
            r_new := ||r||^2
            <Terminate if r_new is small enough>
            pp(:) := (r_new / r_old) * pp + r
            r_old := r_new
    The key for this approach is: if there are few non-missing values of 'X',
    it's faster to compute t(B)*B*v as t(B)*( B*v ) several times than to
    compute t(B)*B as required for the Cholesky. The CG method is mathematically
    guaranteed to reach the optimum solution in no more steps than the dimension
    of 'Be' (here: k_user+k+k_main), assuming infinite numerical precision, but
    in practice, since it is not required to reach the optimal solution at each
    iteration, can run it for only a small number of steps (e.g. 2 or 3) during
    each update of 'A' (since next time the other matrices will be updated too,
    makes sense not to spend too much time on a perfect solution for a single
    matrix vs. spending little time on worse solutions but doing more updates
    to all matrices).

    
    Note that, for both the Cholesky and the CG method, there are some extra
    potential shortcuts that can be taken - for example:
        - If 'X' has no missing values, it's possible to precompute
          t(B)*B to use it for all rows at once (same for 'U' and t(C)*C).
          If there are few missing values, can compute it for all rows at once
          and then subtract from it to obtain the required values for
          a given row.
        - In the Cholesky case, if there are no missing values, can use the same
          Cholesky for all rows at once, and if there are few missing values,
          can at first compute the solution for all rows (ignoring potential
          missing values for the rows that have missing values), and then do
          a post-hoc pass over the rows that have missing values computing their
          actual solutions through either the Cholesky or the CG method.
        - If t(B)*B or t(C)*C are computed from all rows in B/C, when iterating
          over a given row which has missing values in only one of 'X' or 'U',
          it's still possible to use the precomputed matrix for one side while
          making the necessary computations for the other.
        - If 'X' has weights and is sparse with missing-as-zero (e.g. the
          implicit-feedback model), it's possible for both methods to use a
          t(B)*B computed from all rows, and then add the non-zero entries as
          necessary, taking into consideration that they were previously added
          with a weight of one and thus their new weight needs to be decreased.


    The row and column biases (b1/b2) for the factorization of 'X' are obtained
    as follows: first, all the X[m,n] data are centered by subtracting their
    global mean (mu). Then, the biases are initialized by an alternating
    optimization procedure (between row/column biases), in which at each
    iteration, the optimal biases are calculated in closed-form.
    For a given row 'a', the closed-form minimizer of the bias is:
        bias(a) = sum(X(a,:) - b2[1,n] - mu[1]) / (N_non_NA(X(a,:) + lambda))
    At the beginning of each ALS iteration, assuming the 'A' matrix is being
    optimized, the procedure then subtracts from the already-centered X the
    current biases:
        X_iter[m,n] = X[m,n] - b1[m,1] - mu[1])
    and adds an extra column of all-ones at the end of the other factorizing
    matrix:
        B_iter[n+p, k_user+k+k_main+1] := [[Be(:n, :),  1[n,1] ],
                                           [Be(n:, :),  0[p,1] ]]
    The A matrix being solved for is also extended by 1 column.
    The procedure is then run as usual with the altered X and Be matrices, and
    after solving for A, the new values for b1[m,1] will correspond to its last
    column, which will not be used any further when optimizing for the other
    matrices. The 'X' matrix then needs to be restored to its original values,
    and the all-ones column in 'Be' is also ignored.
    Note that it is possible to set a different regulatization parameter for the
    biases by simply changing the last element of diag(lambda) that is added
    to the matrices (or the last element of ap/pp for the CG method)
    accordingly.


    Both the gradient-based approach and the closed-form solution with these
    formulas can be used for any of the 4 matrices by substituting the matrices
    accordingly - i.e.
        For matrix 'B', replace:
            A->B, C->D, X->t(X), U->I
        For matrix 'C', replace:
            A->C, C->NULL, X->U, U->NULL
        For matrix 'D', replace:
            A->D, C->NULL, X->I, U->NULL


    In addition, it's also possible to fit the weighted-implicit model in which:
    - There are no biases and the global mean is not used.
    - The values of 'X' are all zeros or ones.
    - There is an associated weight to each non-zero value of 'X',  given by
      W = alpha*X + 1.
    Using the same procedures as explained earlier. The module includes
    specialized functions for those which make fewer checks on the data. 


    As a special case of 'side information', the model an also add so-called
    'implicit-features', which are binarized matrices telling whether each
    entry in 'X' is non-zero or not. These usually result in a very small
    improvement in recommender systems, with or without side information.
    Since these are considered separate from the actual side information, it's
    still possible to add external features. The procedure for adding these
    features is the same as for the real side information, but in this case
    assuming the missing entries are actual zeros (see reference (d)).
    

    In order to obtain the factor matrices for new users, in a cold-start
    scenario (based on user attributes alone), it's only necessary to obtain
    the closed form for A assuming X is zero, while for a warm-start scenario
    (based on both user attributes and ratings), the closed form on block
    matrices can be applied. If there are binary variables, there is no
    closed-form solution, but can still be obtained in a reasonable time with
    a gradient-based approach.
    
    
*******************************************************************************/

/* Note: the descriptions about input parameters of the functions might be
   outdated and might not match with the actual code anymore. */


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
    buffer_real_t[temp]
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
    real_t *X, real_t *Xfull,
    real_t *U, real_t *Ub, real_t *II, real_t *Ib,
    real_t *U_sp, real_t *U_csr, real_t *I_sp, real_t *I_csr,
    size_t *nvars, size_t *nbuffer, size_t *nbuffer_mt
)
{
    size_t m_max = max2(max2(m, m_u), m_ubin);
    size_t n_max = max2(max2(n, n_i), n_ibin);
    size_t sizeA = m_max * (k_user + k + k_main);
    size_t sizeB = n_max * (k_item + k + k_main);
    *nvars =   m_max * (k_user + k + k_main)
             + n_max * (k_item + k + k_main)
             + (p + pbin) * (k + k_user)
             + (q + qbin) * (k + k_item);
    if (user_bias) *nvars += m_max;
    if (item_bias) *nvars += n_max;

    *nbuffer = 0;
    if (Xfull != NULL) *nbuffer = m * n;
    if (U != NULL)  *nbuffer = max2(*nbuffer, m_u * p + sizeA);
    if (II != NULL) *nbuffer = max2(*nbuffer, n_i * q + sizeB);
    if (Ub != NULL) *nbuffer = max2(*nbuffer, m_ubin * pbin + sizeA);
    if (Ib != NULL) *nbuffer = max2(*nbuffer, n_ibin * qbin + sizeB);
    if (U_csr != NULL || U_sp != NULL) *nbuffer = max2(*nbuffer, sizeA);
    if (I_csr != NULL || U_sp != NULL) *nbuffer = max2(*nbuffer, sizeB);

    *nbuffer_mt = 0;
    if (nthreads > 1)
    {
        if (Xfull == NULL && X != NULL)
            *nbuffer_mt = (k + k_main) * (m + n)
                            + (user_bias? m : 0)
                            + (item_bias? n : 0);
        if (U == NULL && U_sp != NULL)
            *nbuffer_mt = max2(*nbuffer_mt, (k_user + k) * (m_u + p));
        if (II == NULL && I_sp != NULL)
            *nbuffer_mt = max2(*nbuffer_mt, (k_item + k) * (n_i + q));
        *nbuffer_mt *= nthreads;
    }
}

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
    int_t nthreads
)
{
    /* TODO: implement option 'NA_as_zero' here too. */

    /* Shorthands to use later */
    int_t k_totA = k_user + k + k_main;
    int_t k_totB = k_item + k + k_main;
    int_t m_max = max2(max2(m, m_u), m_ubin);
    int_t n_max = max2(max2(n, n_i), n_ibin);
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
        X, Xfull,
        U, Ub, II, Ib,
        U_sp, U_csr, I_sp, I_csr,
        &nvars, &ignored, &mtvars
    );
    set_to_zero_(grad, nvars, nthreads);
    if (mtvars && buffer_mt != NULL) set_to_zero_(buffer_mt, mtvars, nthreads);

    /* unravel the arrays */
    real_t *restrict biasA = values;
    real_t *restrict biasB = biasA + (user_bias? m_max : 0);
    real_t *restrict A = biasB + (item_bias? n_max : 0);
    real_t *restrict B = A + (size_t)m_max * (size_t)k_totA;
    real_t *restrict C = B + (size_t)n_max * (size_t)k_totB;
    real_t *restrict Cb = C + (size_t)(k + k_user) * (size_t)p;
    real_t *restrict D  = Cb + (size_t)(k + k_user) * (size_t)pbin;
    real_t *restrict Db = D + (size_t)(k + k_item) * (size_t)q;


    real_t *restrict g_biasA = grad;
    real_t *restrict g_biasB = g_biasA + (user_bias? m_max : 0);
    real_t *restrict g_A = g_biasB + (item_bias? n_max : 0);
    real_t *restrict g_B = g_A + (size_t)m_max * (size_t)k_totA;
    real_t *restrict g_C = g_B + (size_t)n_max * (size_t)k_totB;
    real_t *restrict g_Cb = g_C + (size_t)(k + k_user) * (size_t)p;
    real_t *restrict g_D  = g_Cb + (size_t)(k + k_user) * (size_t)pbin;
    real_t *restrict g_Db = g_D + (size_t)(k + k_item) * (size_t)q;

    size_t sizeA = m_max * (size_t)k_totA;
    size_t sizeB = n_max * (size_t)k_totB;

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
        buffer_real_t,
        buffer_mt,
        nthreads
    );

    /* then user non-binary factorization */
    if (U != NULL || U_sp != NULL || U_csr != NULL)
    {
        set_to_zero_(buffer_real_t, sizeA, nthreads);
        f += fun_grad_cannonical_form(
            A, k_totA, C, k_user + k,
            buffer_real_t,  g_C,
            (U != NULL)? m_u : m, p, k_user + k,
            U_row, U_col, U_sp, nnz_U,
            U, !U_has_NA,
            U_csr_p, U_csr_i, U_csr,
            U_csc_p, U_csc_i, U_csc,
            false, false,
            (real_t*)NULL, (real_t*)NULL,
            (real_t*)NULL, (real_t*)NULL,
            (real_t*)NULL, (real_t*)NULL, (real_t*)NULL,
            w_user,
            buffer_real_t + sizeA, 
            buffer_mt,
            nthreads
        );
        taxpy_large(buffer_real_t, 1., g_A, sizeA, nthreads);
    }

    /* then item non-binary factorization */
    if (II != NULL || I_sp != NULL || I_csr != NULL)
    {
        set_to_zero_(buffer_real_t, sizeB, nthreads);
        f += fun_grad_cannonical_form(
            B, k_totB, D, k_item + k,
            buffer_real_t, g_D,
            (II != NULL)? n_i : n, q, k_item + k,
            I_row, I_col, I_sp, nnz_I,
            II, !I_has_NA,
            I_csr_p, I_csr_i, I_csr,
            I_csc_p, I_csc_i, I_csc,
            false, false,
            (real_t*)NULL, (real_t*)NULL,
            (real_t*)NULL, (real_t*)NULL,
            (real_t*)NULL, (real_t*)NULL, (real_t*)NULL,
            w_item,
            buffer_real_t + sizeB,
            buffer_mt,
            nthreads
        );
        taxpy_large(buffer_real_t, 1., g_B, sizeB, nthreads);
    }

    /* if there are binary matrices with sigmoid transformation, need a
       different formula for the gradients */

    if (Ub != NULL)
    {
        set_to_zero_(buffer_real_t, sizeA, nthreads);
        f += collective_fun_grad_bin(
            A, k_totA, Cb, k_user + k,
            buffer_real_t, g_Cb,
            Ub,
            m_ubin, pbin, k_user + k,
            !Ub_has_NA, w_user,
            buffer_real_t + sizeA,
            nthreads
        );
        taxpy_large(buffer_real_t, 1., g_A, sizeA, nthreads);
    }

    if (Ib != NULL)
    {
        set_to_zero_(buffer_real_t, sizeB, nthreads);
        f += collective_fun_grad_bin(
            B, k_totB, Db, k_item + k,
            buffer_real_t, g_Db,
            Ib,
            n_ibin, qbin, k_item + k,
            !Ib_has_NA, w_item,
            buffer_real_t + sizeB,
            nthreads
        );
        taxpy_large(buffer_real_t, 1., g_B, sizeB, nthreads);
    }

    /* Now account for the regulatization parameter
        grad = grad + lambda * var
        f = f + (lambda / 2) * || var ||^2 */

    /* If all matrices have the same regulatization, can do it in one pass */
    if (lam_unique == NULL) {
        taxpy_large(values, lam, grad, nvars, nthreads);
        f += (lam / 2.) * sum_squares(values, nvars, nthreads);
    }

    /* otherwise, add it one by one */ 
    else {
        long double freg = 0;

        /* Note: Cbin is in memory right next to C, so there's not need to
           account for it separately - can be passed extra elements to C */

        if (user_bias) cblas_taxpy(m_max, lam_unique[0], biasA, 1, g_biasA, 1);
        if (item_bias) cblas_taxpy(n_max, lam_unique[1], biasB, 1, g_biasB, 1);
        taxpy_large(A, lam_unique[2],g_A,(size_t)m_max*(size_t)k_totA,nthreads);
        taxpy_large(B, lam_unique[3],g_B,(size_t)n_max*(size_t)k_totB,nthreads);

        if (U != NULL || U_sp != NULL || U_csr != NULL || Ub != NULL)
            taxpy_large(C, lam_unique[4], g_C,
                        (size_t)(p+pbin)*(size_t)(k_user+k), nthreads);
        if (II != NULL || I_sp != NULL || I_csr != NULL || Ib != NULL)
            taxpy_large(D, lam_unique[5], g_D,
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
        f += (real_t)freg;
    }

    return (real_t) f;
}

/* This function calculates the gradient for squared error on
   sigmoid-transformed approximations */
real_t collective_fun_grad_bin
(
    real_t *restrict A, int_t lda, real_t *restrict Cb, int_t ldc,
    real_t *restrict g_A, real_t *restrict g_Cb,
    real_t *restrict Ub,
    int_t m, int_t pbin, int_t k,
    bool Ub_has_NA, double w_user,
    real_t *restrict buffer_real_t,
    int_t nthreads
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
                1., A, lda, Cb, ldc,
                0., buffer_real_t, pbin);
    exp_neg_x(buffer_real_t, (size_t)m * (size_t)pbin, nthreads);

    /* f = sum_sq(Ub - 1/(1+Buffer))
       See explanation at the top for the gradient formula */
    if (Ub_has_NA)
    {
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(Ub, buffer_real_t) reduction(+:f)
        for (size_t_for ix = 0; ix < (size_t)m*(size_t)pbin; ix++)
            f += (!isnan(Ub[ix]))?
                  square(Ub[ix] - 1./(1.+buffer_real_t[ix])) : (0);
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(buffer_real_t, m, pbin, Ub)
        for (size_t_for ix = 0; ix < (size_t)m*(size_t)pbin; ix++)
            buffer_real_t[ix] = (!isnan(Ub[ix]))?
                                ( (1./(1.+buffer_real_t[ix]) - Ub[ix])
                                  * buffer_real_t[ix]
                                  / square(buffer_real_t[ix]+1.)
                                ) : (0);
    }

    else
    {
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(Ub, buffer_real_t) reduction(+:f)
        for (size_t_for ix = 0; ix < (size_t)m*(size_t)pbin; ix++)
            f += square(Ub[ix] - 1./(1.+buffer_real_t[ix]));
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(buffer_real_t, m, pbin, Ub)
        for (size_t_for ix = 0; ix < (size_t)m*(size_t)pbin; ix++)
            buffer_real_t[ix] = (
                                 (1./(1.+buffer_real_t[ix]) - Ub[ix])
                                  * buffer_real_t[ix]
                                  / square(buffer_real_t[ix]+1.)
                                );
    }

    f *= (w_user / 2);

    cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, k, pbin,
                w_user, buffer_real_t, pbin, Cb, ldc,
                0., g_A, lda);
    cblas_tgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                pbin, k, m,
                w_user, buffer_real_t, pbin, A, lda,
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
    buffer_real_t[n or p or pbin]
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
)
{
    int_t ldb = k_item + k + k_main;
    int_t k_pred = k + k_main;
    set_to_zero(g_A, k_user + k + k_main);
    real_t f = 0;


    real_t err;
    real_t *restrict a_vec_pred = a_vec + k_user;
    real_t *restrict g_A_pred = g_A + k_user;
    real_t *restrict Bm = B + k_item;
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
        memcpy(buffer_real_t, Xa_dense, (size_t)n*sizeof(real_t));
        cblas_tgemv(CblasRowMajor, CblasNoTrans,
                    n, k + k_main,
                    1, B + k_item, ldb,
                    a_vec + k_user, 1,
                    -1, buffer_real_t, 1);

        if (weight != NULL)
            mult_if_non_nan(buffer_real_t, Xa_dense, weight, n, 1);
        else
            nan_to_zero(buffer_real_t, Xa_dense, n, 1);

        cblas_tgemv(CblasRowMajor, CblasTrans,
                    n, k + k_main,
                    w_main, B + k_item, ldb,
                    buffer_real_t, 1,
                    0, g_A + k_user, 1);

        if (weight == NULL)
            f = (w_main / 2.) * cblas_tdot(n, buffer_real_t, 1,buffer_real_t,1);
        else
            f = (w_main / 2.) * sum_sq_div_w(buffer_real_t, weight, n, false,1);
    }

    if (u_vec != NULL)
    {
        memcpy(buffer_real_t, u_vec, (size_t)p*sizeof(real_t));
        cblas_tgemv(CblasRowMajor, CblasNoTrans,
                    p, k_user + k,
                    1., C, k_user + k,
                    a_vec, 1,
                    -1., buffer_real_t, 1);
        if (u_vec_has_NA)
            nan_to_zero(buffer_real_t, u_vec, p, 1);

        cblas_tgemv(CblasRowMajor, CblasTrans,
                    p, k_user + k,
                    w_user, C, k_user + k,
                    buffer_real_t, 1,
                    1., g_A, 1);

        f += (w_user / 2.) * cblas_tdot(p, buffer_real_t, 1, buffer_real_t, 1);
    }

    else if (u_vec_sp != NULL)
    {
        real_t err_sp = 0;
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
        real_t err_bin = 0;
        cblas_tgemv(CblasRowMajor, CblasNoTrans,
                    pbin, k_user + k,
                    1, Cb, k_user + k,
                    a_vec, 1,
                    0, buffer_real_t, 1);
        exp_neg_x(buffer_real_t, (size_t)pbin, 1);

        if (u_bin_vec_has_NA)
        {
            for (int_t ix = 0; ix < pbin; ix++)
                err_bin += (!isnan(u_bin_vec[ix]))?
                            square(1./(1.+buffer_real_t[ix]) - u_bin_vec[ix])
                            : (0);
            for (int_t ix = 0; ix < pbin; ix++)
                buffer_real_t[ix] = (!isnan(u_bin_vec[ix]))?
                                    ( (1./(1.+buffer_real_t[ix]) -u_bin_vec[ix])
                                       * buffer_real_t[ix]
                                       / square(buffer_real_t[ix]+1.) )
                                    : (0);
        }

        else
        {
            for (int_t ix = 0; ix < pbin; ix++)
                err_bin += square(1./(1.+buffer_real_t[ix]) - u_bin_vec[ix]);
            for (int_t ix = 0; ix < pbin; ix++)
                buffer_real_t[ix] = (1./(1.+buffer_real_t[ix]) - u_bin_vec[ix])
                                    * buffer_real_t[ix] 
                                    / square(buffer_real_t[ix]+1.);
        }

        cblas_tgemv(CblasRowMajor, CblasTrans,
                    pbin, k_user + k,
                    w_user, Cb, k_user + k,
                    buffer_real_t, 1,
                    1, g_A, 1);
        f += (w_user / 2.) * err_bin;
    }


    f += (lam / 2.) * cblas_tdot(k_user+k+k_main, a_vec, 1, a_vec, 1);
    cblas_taxpy(k_user+k+k_main, lam, a_vec, 1, g_A, 1);
    if (lam_last != lam && k_main) {
        f += (lam_last-lam)/2. * square(a_vec[k_user+k+k_main-1]);
        g_A[k_user+k+k_main-1] += (lam_last-lam) * a_vec[k_user+k+k_main-1];
    }
    return (real_t) f;
}

/* These functions find the optimal values for a single row of A using the
   gradient function above, passing it to the L-BFGS solver */
real_t wrapper_factors_fun_grad
(
    void *instance,
    real_t *x,
    real_t *g,
    const size_t n,
    const real_t step
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
        data->buffer_real_t,
        data->lam, data->w_main, data->w_user, data->lam_last
    );
}

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
        buffer_real_t,
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

    /* Starting point_t can be set to zero, since the other matrices
       are already at their local optima. */
    set_to_zero(a_vec, nvars);

    int_t retval = lbfgs(
        nvars,
        a_vec,
        (real_t*)NULL,
        wrapper_factors_fun_grad,
        callback,
        (void*) &data,
        &lbfgs_params,
        (real_t*)NULL,
        (iteration_data_t*)NULL
    );

    if (retval == LBFGSERR_OUTOFMEMORY)
        return 1;
    else
        return 0;
}

/* TODO: for better numerical precision in the Cholesky method, could keep
   some temporary arrays matching to t(Be)*Be and 'a_vec', zero them out,
   sum the parts from X and U separately, then add them to the other arrays.
   That way could also take the 'w_user' scaling after having already summed.
   Maybe add it as an extra parameter 'extra_precision' or so. */
/* TODO: for an even more efficient version with dense inputs, could have
   an array that keeps track of which values are missing, set the missing ones
   to zero in the original array, then use 'gemv' at once and subtract the
   missing ones later. */
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
    real_t *restrict precomputedBtB, int_t cnt_NA_x,
    real_t *restrict precomputedCtCw, int_t cnt_NA_u,
    real_t *restrict precomputedBeTBeChol, int_t n_BtB,
    real_t *restrict precomputedBiTBi,
    bool add_X, bool add_U,
    bool use_cg, int_t max_cg_steps,/* <- 'cg' should not be used for new data*/
    bool nonneg, int_t max_cd_steps, real_t *restrict a_prev,
    real_t *restrict buffer_real_t
)
{
    /* Note: for CG, CtC should not be scaled by 'w_user', but for Cholesky
       it should be. In neither case should it have lambda added.
       BtB should also never have lambda added. */
    
    /* Potential bad inputs - should not reach this point_t */
    if (  (  (Xa_dense != NULL && cnt_NA_x == n) ||
             (Xa_dense == NULL && nnz == 0)  )
                &&
          (  (u_vec != NULL && cnt_NA_u == p) ||
             (u_vec == NULL && nnz_u_vec == 0)  ) )
    {
        set_to_zero(a_vec, k_user + k + k_main);
        return;
    }
    if (Xa_dense != NULL && cnt_NA_x == n) {
        Xa_dense = NULL;
        nnz = 0;
        NA_as_zero_X = false;
    }
    if (u_vec != NULL && cnt_NA_u == p) {
        u_vec = NULL;
        nnz_u_vec = 0;
        NA_as_zero_U = false;
    }

    if (cnt_NA_x || cnt_NA_u) {
        add_X = true;
        add_U = true;
    }

    
    int_t k_totA = k_user + k + k_main;
    size_t k_totC = k_user + k;
    size_t offset_square = (size_t)k_user + (size_t)k_user*(size_t)k_totA;
    int_t ld_BtB = k + k_main;

    char lo = 'L';
    int_t one = 1;
    int_t ignore;

    if (n_BtB == 0) n_BtB = n;

    if (precomputedBeTBeChol != NULL && !nonneg &&
        (   (Xa_dense != NULL && cnt_NA_x == 0 &&
             weight == NULL && n_BtB == n) ||
            (Xa_dense == NULL && NA_as_zero_X &&
             (weight == NULL || nnz == 0))  ) &&
        (   (u_vec != NULL && cnt_NA_u == 0) ||
            (u_vec == NULL && NA_as_zero_U)  ))
    {
        if (add_U && p)
        {
            if (u_vec != NULL) {
                cblas_tgemv(CblasRowMajor, CblasTrans,
                            p, k_user+k,
                            add_X? (1.) : (w_user), C, k_user+k,
                            u_vec, 1,
                            add_X? 0. : 1., a_vec, 1);
                if (k_main && add_X)
                    set_to_zero(a_vec + k_user+k, k_main);
            }
            else {
                if (add_X)
                    set_to_zero(a_vec, k_user+k+k_main);
                else if (k_user)
                    set_to_zero(a_vec, k_user);
                tgemv_dense_sp(
                    p, k_user+k,
                    add_X? (1.) : (w_user), C, k_user+k,
                    u_vec_ixB, u_vec_sp, nnz_u_vec,
                    a_vec
                );
            }
            if (w_user != 1. && add_X && (u_vec != NULL || nnz_u_vec))
                cblas_tscal(k_user+k, w_user, a_vec, 1);
        }

        if (add_X)
        {
            if (!add_U || (!p && !nnz_u_vec))
                set_to_zero(a_vec + k_user+k, k_main);
            if (Xa_dense != NULL)
                cblas_tgemv(CblasRowMajor, CblasTrans,
                            n, k+k_main,
                            1., B + k_item, ldb,
                            Xa_dense, 1,
                            1., a_vec + k_user, 1);
            else
                tgemv_dense_sp(
                    n, k+k_main,
                    1., B + k_item, (size_t)ldb,
                    ixB, Xa, nnz,
                    a_vec + k_user
                );

            if (add_implicit_features)
            {
                if (Xa_dense != NULL)
                    cblas_tgemv(CblasRowMajor, CblasTrans,
                                n, k+k_main_i,
                                w_implicit, Bi, k+k_main_i,
                                Xones, incXones,
                                1., a_vec + k_user, 1);
                else
                    tgemv_dense_sp(
                        n, k+k_main_i,
                        w_implicit, Bi, k+k_main_i,
                        ixB, Xones, nnz,
                        a_vec + k_user
                    );
            }
        }
        
        tpotrs_(&lo, &k_totA, &one,
                precomputedBeTBeChol, &k_totA,
                a_vec, &k_totA,
                &ignore);
        return;
    }


    if (!p && !nnz_u_vec && !NA_as_zero_U)
        precomputedCtCw = NULL;

    #ifdef TEST_CG
    use_cg = true;
    max_cg_steps = 10000;
    if (precomputedBiTBi != NULL && add_implicit_features)
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k+k_main_i, n,
                    1.,
                    Bi, k+k_main_i,
                    0., precomputedBiTBi, k+k_main_i);
    #endif

    if (add_implicit_features && precomputedBiTBi == NULL)
    {
        precomputedBiTBi = buffer_real_t;
        buffer_real_t += square(k+k_main_i);
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k+k_main_i, n,
                    use_cg? (1.) : (w_implicit),
                    Bi, k+k_main_i,
                    0., precomputedBiTBi, k+k_main_i);
    }

    if (use_cg && add_X && add_U)
    {
        collective_block_cg(
            a_vec,
            k, k_user, k_item, k_main,
            Xa_dense,
            Xa, ixB, nnz,
            u_vec_ixB, u_vec_sp, nnz_u_vec,
            u_vec,
            NA_as_zero_X, NA_as_zero_U,
            B, n, ldb,
            C, p,
            add_implicit_features,
            Xones, incXones,
            Bi, precomputedBiTBi, k_main_i,
            weight,
            lam, w_user, w_implicit, lam_last,
            cnt_NA_x, cnt_NA_u,
            precomputedBtB, precomputedCtCw,
            max_cg_steps,
            buffer_real_t
        );
        return;
    }

    real_t *restrict bufferBeTBe = buffer_real_t;
    set_to_zero(bufferBeTBe, square(k_totA));
    
    if (add_X && add_U)
        set_to_zero(a_vec, k_totA);
    else if (add_U)
        set_to_zero(a_vec, k_user);
    else if (add_X)
        set_to_zero(a_vec + k_user+k, k_main);

    bool prefer_BtB = (cnt_NA_x + (n_BtB-n) < 2*(k+k_main+1)) ||
                      (nnz > (size_t)(2*(k+k_main+1))) ||
                      (NA_as_zero_X);
    bool prefer_CtC = (cnt_NA_u < 2*(k+k_user)) ||
                      (nnz_u_vec > (size_t)(2*(k+k_user))) ||
                      (NA_as_zero_U);
    if (precomputedBtB == NULL)
        prefer_BtB = false;
    if (precomputedCtCw == NULL)
        prefer_CtC = false;

    /* TODO: for better precision, here could do the parts from matrix B
       before the parts from matrix C, then rescale according to 'w_main',
       instead of multiplying each row by 'w_main'. */

    /* =================== Part 1 =====================
       Constructing t(Be)*Be, upper-left square (from C) */
    if (u_vec != NULL || NA_as_zero_U) /* Dense u_vec */
    {
        /* If it's full or near full, can use the precomputed matrix
           and subtract missing entries from it if necessary */
        if (prefer_CtC || NA_as_zero_U)
        {
            if (precomputedCtCw != NULL) {
                copy_mat(k_user+k, k_user+k,
                         precomputedCtCw, k_user+k,
                         bufferBeTBe, k_totA);
            } else {
                cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                            k_user+k, p,
                            (cnt_NA_u > 0 && u_vec != NULL)? (1.) : (w_user),
                            C, k_user+k,
                            0., bufferBeTBe, k_totA);
            }
            
            if (cnt_NA_u > 0 && u_vec != NULL)
            {
                for (size_t ix = 0; ix < (size_t)p; ix++)
                    if (isnan(u_vec[ix]))
                    {
                        cblas_tsyr(CblasRowMajor, CblasUpper,
                                   k_user+k,
                                   (precomputedCtCw != NULL)? (-w_user) : (-1.),
                                   C + ix*k_totC, 1,
                                   bufferBeTBe, k_totA);
                    }
            }
            if (precomputedCtCw == NULL && w_user != 1. &&
                (cnt_NA_u > 0 && u_vec != NULL) && p)
            {
                cblas_tscal(square(k_totA) - k_main*k_totA - k_main,
                            w_user, bufferBeTBe, 1);
            }
        }

        /* Otherwise, will need to construct it one-by-one */
        else
        {
            for (size_t ix = 0; ix < (size_t)p; ix++)
                if (!isnan(u_vec[ix]))
                    cblas_tsyr(CblasRowMajor, CblasUpper,
                               k_user+k, 1.,
                               C + ix*k_totC, 1,
                               bufferBeTBe, k_totA);
            if (w_user != 1.)
                cblas_tscal(square(k_totA) - k_main*k_totA - k_main,
                            w_user, bufferBeTBe, 1);
        }
    }

    else if (nnz_u_vec) /* Sparse u_vec */
    {
        for (size_t ix = 0; ix < nnz_u_vec; ix++)
            cblas_tsyr(CblasRowMajor, CblasUpper,
                       k_user+k, 1.,
                       C + (size_t)u_vec_ixB[ix]*k_totC, 1,
                       bufferBeTBe, k_totA);
        if (w_user != 1.)
            cblas_tscal(square(k_totA) - k_main*k_totA - k_main,
                        w_user, bufferBeTBe, 1);
    }



    /* =================== Part 2 ======================
       Constructing t(Be)*Be, lower-right square (from B) */
    if ((Xa_dense != NULL && weight == NULL && prefer_BtB) ||
        (Xa_dense == NULL && NA_as_zero_X))
    {
        if (precomputedBtB != NULL) {
            sum_mat(k+k_main, k+k_main,
                    precomputedBtB, ld_BtB,
                    bufferBeTBe + offset_square, k_totA);
        } else {
            cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        k+k_main, n,
                        1., B + k_item, ldb,
                        1., bufferBeTBe + offset_square, k_totA);
        }

        
        if ((cnt_NA_x > 0 || n_BtB > n) && Xa_dense != NULL) {
            for (size_t ix = 0; ix < (size_t)n; ix++)
                if (isnan(Xa_dense[ix]))
                    cblas_tsyr(CblasRowMajor, CblasUpper,
                               k+k_main, -1.,
                               B + (size_t)k_item + ix*(size_t)ldb, 1,
                               bufferBeTBe + offset_square, k_totA);
            for (size_t ix = (size_t)n; ix < (size_t)n_BtB; ix++)
                cblas_tsyr(CblasRowMajor, CblasUpper,
                           k+k_main, -1.,
                           B + (size_t)k_item + ix*(size_t)ldb, 1,
                           bufferBeTBe + offset_square, k_totA);
        }

        else if (Xa_dense == NULL && NA_as_zero_X && weight != NULL) {
            for (size_t ix = 0; ix < nnz; ix++)
                cblas_tsyr(CblasRowMajor, CblasUpper,
                           k+k_main, weight[ix]-1.,
                           B + (size_t)k_item +(size_t)ixB[ix]*(size_t)ldb, 1,
                           bufferBeTBe + offset_square, k_totA);
        }

        /* Note: nothing extra is required when having 'NA_as_zero_X' without
           weights, hence the if-else chain stops without checking for it. */
    }

    else if (Xa_dense != NULL)
    {
        if (weight == NULL) {
            for (size_t ix = 0; ix < (size_t)n; ix++)
                if (!isnan(Xa_dense[ix]))
                    cblas_tsyr(CblasRowMajor, CblasUpper,
                               k+k_main, 1.,
                               B + (size_t)k_item + ix*(size_t)ldb, 1,
                               bufferBeTBe + offset_square, k_totA);
        }

        else {
            for (size_t ix = 0; ix < (size_t)n; ix++)
                if (!isnan(Xa_dense[ix]))
                    cblas_tsyr(CblasRowMajor, CblasUpper,
                               k+k_main, weight[ix],
                               B + (size_t)k_item + ix*(size_t)ldb, 1,
                               bufferBeTBe + offset_square, k_totA);
        }
    }

    else /* Sparse Xa - this is the expected scenario for most use-cases */
    {
        for (size_t ix = 0; ix < nnz; ix++)
            cblas_tsyr(CblasRowMajor, CblasUpper,
                       k+k_main,
                       (weight == NULL)? (1.) : (weight[ix]),
                       B + (size_t)k_item + (size_t)ixB[ix]*(size_t)ldb, 1,
                       bufferBeTBe + offset_square, k_totA);
    }

    /* TODO: could add BiTBi to BtB beforehand and save one operation that way,
       along with decreasing memory usage. */
    if (add_implicit_features)
        sum_mat(k+k_main_i, k+k_main_i,
                precomputedBiTBi, k+k_main_i,
                bufferBeTBe + offset_square, k_totA);


    /* ================ Part 3 =================
       Constructing Be*t(Xe), upper part (from X) */
    if (add_X)
    {
        if (Xa_dense != NULL)
        {
            if (weight == NULL && cnt_NA_x == 0)
            {
                cblas_tgemv(CblasRowMajor, CblasTrans,
                            n, k+k_main,
                            1., B + k_item, ldb, Xa_dense, 1,
                            add_U? 0. : 1., a_vec + k_user, 1);
            }
            
            else
            {
                for (size_t ix = 0; ix < (size_t)n; ix++)
                    if  (!isnan(Xa_dense[ix]))
                        cblas_taxpy(k+k_main,
                                    ((weight == NULL)?
                                        1. : weight[ix]) * Xa_dense[ix],
                                    B + (size_t)k_item + ix*(size_t)ldb, 1,
                                    a_vec + k_user, 1);
            }
        }

        else
        {
            if (weight == NULL)
                tgemv_dense_sp(n, k+k_main,
                               1., B + k_item, (size_t)ldb,
                               ixB, Xa, nnz,
                               a_vec + k_user);
            else
                tgemv_dense_sp_weighted(n, k+k_main, weight,
                                        B + k_item, (size_t)ldb,
                                        ixB, Xa, nnz,
                                        a_vec + k_user);
        }


        if (add_implicit_features)
        {
            if (Xa_dense != NULL)
                cblas_tgemv(CblasRowMajor, CblasTrans,
                            n, k+k_main_i,
                            w_implicit, Bi, k+k_main_i,
                            Xones, incXones,
                            1., a_vec + k_user, 1);
            else
                tgemv_dense_sp(
                    n, k+k_main_i,
                    w_implicit, Bi, k+k_main_i,
                    ixB, Xones, nnz,
                    a_vec + k_user
                );
        }
    }

    /* TODO: maybe this part should be moved before Part 3, so that it can
       scale by 'w_user' in a post-hoc pass for more numerical precision. */
    
    /* ================ Part 4 =================
       Constructing Be*t(Xe), lower part (from U) */
    if (add_U && p)
    {
        if (u_vec != NULL && cnt_NA_u == 0)
        {
            cblas_tgemv(CblasRowMajor, CblasTrans,
                        p, k_user+k,
                        w_user, C, k_user+k, u_vec, 1,
                        1., a_vec, 1);
        }
        
        else if (u_vec != NULL) 
        {
            for (size_t ix = 0; ix < (size_t)p; ix++)
                if (!isnan(u_vec[ix]))
                    cblas_taxpy(k_user+k, w_user * u_vec[ix],
                                C + ix*k_totC, 1,
                                a_vec, 1);
        }

        else
        {
            tgemv_dense_sp(p, k_user+k,
                           w_user, C, k_totC,
                           u_vec_ixB, u_vec_sp, nnz_u_vec,
                           a_vec);
        }
    }


    /* =================== Part 5 ======================
       Solving A = inv(t(Be)*Be + diag(lam)) * (Be*t(Xe)) */

    add_to_diag(bufferBeTBe, lam, k_totA);
    if (lam_last != lam) bufferBeTBe[square(k_totA)-1] += (lam_last - lam);

    if (!nonneg)
        tposv_(&lo, &k_totA, &one,
               bufferBeTBe, &k_totA,
               a_vec, &k_totA,
               &ignore);
    else
        solve_nonneg(
            bufferBeTBe,
            a_vec,
            a_prev,
            k_totA,
            max_cd_steps,
            true
        );
}

void collective_closed_form_block_implicit
(
    real_t *restrict a_vec,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t *restrict B, int_t n, real_t *restrict C, int_t p,
    real_t *restrict Xa, int_t ixB[], size_t nnz,
    real_t *restrict u_vec, int_t cnt_NA_u,
    real_t *restrict u_vec_sp, int_t u_vec_ixB[], size_t nnz_u_vec,
    bool NA_as_zero_U,
    real_t lam, real_t w_user,
    real_t *restrict precomputedBeTBe,
    real_t *restrict precomputedBtB, /* for cg, should NOT have lambda added */
    real_t *restrict precomputedBeTBeChol,
    real_t *restrict precomputedCtCw,
    bool add_U, bool shapes_match,
    bool use_cg, int_t max_cg_steps,/* <- 'cg' should not be used for new data*/
    bool nonneg, int_t max_cd_steps, real_t *restrict a_prev,
    real_t *restrict buffer_real_t
)
{
    /* Potential bad inputs - should not reach this point_t */
    if (u_vec != NULL && cnt_NA_u == p) {
        u_vec = NULL;
        nnz_u_vec = 0;
    }
    if ( (nnz == 0)
            &&
          ((u_vec != NULL && cnt_NA_u == p) ||
           (u_vec == NULL && nnz_u_vec == 0))
        )
    {
        set_to_zero(a_vec, k_user + k + k_main);
        return;
    }

    char lo = 'L';
    int_t one = 1;
    int_t ignore;
    int_t k_totA = k_user + k + k_main;
    size_t k_totB = k_item + k + k_main;
    int_t k_totC = k_user + k;
    size_t offset_square = k_user + k_user*k_totA;
    int_t ld_BtB = k + k_main;
    bool few_NAs = (u_vec != NULL && cnt_NA_u < k_user+k);
    if (cnt_NA_u)
        add_U = true;
    if ((add_U || cnt_NA_u > 0) && !use_cg)
        set_to_zero(a_vec, k_totA);

    real_t *restrict BtB = buffer_real_t;
    bool add_C = false;

    if (nnz == 0 && ((u_vec != NULL && cnt_NA_u == 0) || NA_as_zero_U) &&
        precomputedBeTBeChol != NULL && (!use_cg || p < k_totA) && !nonneg)
    {
        if (use_cg || add_U)
        {
            if (u_vec != NULL) {
                cblas_tgemv(CblasRowMajor, CblasTrans,
                            p, k_user+k,
                            1., C, k_user+k,
                            u_vec, 1,
                            0., a_vec, 1);
                if (k_main)
                    set_to_zero(a_vec + k_user+k, k_main);
            }
            else {
                set_to_zero(a_vec, k_totA);
                tgemv_dense_sp(
                    p, k_user+k,
                    1., C, k_user+k,
                    u_vec_ixB, u_vec_sp, nnz_u_vec,
                    a_vec
                );
            }
            if (w_user != 1.)
                cblas_tscal(k_user+k, w_user, a_vec, 1);
        }
        tpotrs_(&lo, &k_totA, &one,
                precomputedBeTBeChol, &k_totA,
                a_vec, &k_totA,
                &ignore);
        return;
    }

    else if (use_cg)
    {
        collective_block_cg_implicit(
                a_vec,
                k, k_user, k_item, k_main,
                Xa, ixB, nnz,
                u_vec_ixB, u_vec_sp, nnz_u_vec,
                u_vec,
                NA_as_zero_U,
                B, n,
                C, p,
                lam, w_user,
                cnt_NA_u,
                max_cg_steps,
                precomputedBtB,
                precomputedCtCw,
                buffer_real_t
            );
        return;
    }
    

    if ((u_vec != NULL && few_NAs) || (u_vec == NULL && NA_as_zero_U))
    {
        if (precomputedBeTBe != NULL)
            memcpy(BtB, precomputedBeTBe,(size_t)square(k_totA)*sizeof(real_t));
        else {
            set_to_zero(BtB, square(k_totA));
            if (precomputedCtCw == NULL)
                cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                            k_user+k, p,
                            w_user, C, k_totC,
                            0., BtB, k_totA);
            else
                copy_mat(k_user+k, k_user+k,
                         precomputedCtCw, k_totC,
                         BtB, k_totA);
            if (precomputedBtB  == NULL) {
                cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                            k+k_main, n,
                            1., B + k_item, k_totB,
                            1., BtB + offset_square, k_totA);
                add_to_diag(BtB, lam, k_totA);
            } else {
                sum_mat(k+k_main, k+k_main,
                        precomputedBtB, k+k_main,
                        BtB + offset_square, k_totA);
                for (size_t ix = 0; ix < (size_t)k_user; ix++)
                    BtB[ix + ix*k_totA] += lam;
            }
        }
    }
    
    else
    {
        add_C = true;
        if (ld_BtB != k_totA)
            set_to_zero(BtB, square(k_totA));
        if (precomputedBtB != NULL) {
            copy_mat(ld_BtB, ld_BtB,
                     precomputedBtB, ld_BtB,
                     BtB + offset_square, k_totA);
            if (ld_BtB != k_totA)
                for (int_t ix = 0; ix < k_user; ix++)
                    BtB[ix + ix*k_totA] += lam;
        }
        else {
            cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        k+k_main, n,
                        1., B + k_item, k_totB,
                        0., BtB + offset_square, k_totA);
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
            tgemv_dense_sp(p, k_user+k,
                           w_user, C, (size_t)k_totC,
                           u_vec_ixB, u_vec_sp, nnz_u_vec,
                           a_vec);
    }

    else
    {
        if (few_NAs && cnt_NA_u > 0)
        {
            for (size_t ix = 0; ix < (size_t)p; ix++) {
                if (isnan(u_vec[ix])) {
                    cblas_tsyr(CblasRowMajor, CblasUpper,
                               k_totC, -w_user,
                               C + ix*(size_t)k_totC, 1,
                               BtB, k_totA);
                }
            }
        }
        else if (add_C)
        {
            for (size_t ix = 0; ix < (size_t)p; ix++) {
                if (!isnan(u_vec[ix]))
                    cblas_tsyr(CblasRowMajor, CblasUpper,
                               k_totC, w_user,
                               C + ix*(size_t)k_totC, 1,
                               BtB, k_totA);
            }
        }

        if (add_U || cnt_NA_u > 0)
        {
            if (cnt_NA_u == 0)
            {
                cblas_tgemv(CblasRowMajor, CblasTrans,
                            p, k_user+k,
                            w_user, C, k_user+k, u_vec, 1,
                            0., a_vec, 1);
            }

            else
            {
                set_to_zero(a_vec, k_user+k);
                for (size_t ix = 0; ix < (size_t)p; ix++)
                    if (!isnan(u_vec[ix]))
                        cblas_taxpy(k_user+k, u_vec[ix],
                                    C + ix*(size_t)k_totC, 1,
                                    a_vec, 1);
                if (w_user != 1.)
                    cblas_tscal(k_user+k, w_user, a_vec, 1);
            }
        }
    }

    /* t(Be)*Be, lower-right square (from B)
            AND
       Be*t(Xe), upper part (from X) */
    for (size_t ix = 0; ix < nnz; ix++) {
        cblas_tsyr(CblasRowMajor, CblasUpper, k+k_main,
                   Xa[ix],
                   B + (size_t)k_item + (size_t)ixB[ix]*k_totB, 1,
                   BtB + offset_square, k_totA);
        cblas_taxpy(k + k_main, Xa[ix] + 1.,
                    B + (size_t)k_item + (size_t)ixB[ix]*k_totB, 1,
                    a_vec + k_user, 1);
    }

    if (!nonneg)
        tposv_(&lo, &k_totA, &one,
               BtB, &k_totA,
               a_vec, &k_totA,
               &ignore);
    else
        solve_nonneg(
            BtB,
            a_vec,
            a_prev,
            k_totA,
            max_cd_steps,
            true
        );
}

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
    real_t *restrict buffer_real_t
)
{
    /* TODO: when using BtB or CtC, should sum from the loop first, then
       call symv, for higher numeric precision */
    int_t k_totA = k_user + k + k_main;
    int_t k_totC = k_user + k;
    int_t ld_BtB = k + k_main;
    real_t *restrict Ap = buffer_real_t;
    real_t *restrict pp = Ap + k_totA;
    real_t *restrict r  = pp + k_totA;
    real_t *restrict wr = NULL;
    if (Xa_dense == NULL && NA_as_zero_X && weight != NULL)
        wr = r + k_totA;  /* has length 'n' */
    set_to_zero(r, k_totA);
    real_t r_new, r_old;
    real_t a, coef, w_this;

    if (u_vec != NULL && cnt_NA_u == p) {
        u_vec = NULL;
        nnz_u_vec = 0;
        NA_as_zero_U = false;
    }

    if (Xa_dense != NULL && cnt_NA_x == n) {
        Xa_dense = NULL;
        nnz = 0;
        NA_as_zero_X = false;
    }

    bool prefer_BtB = (cnt_NA_x < 2*(k+k_main)) ||
                      (nnz > (size_t)(2*(k+k_main))) ||
                      (NA_as_zero_X);
    bool prefer_CtC = (cnt_NA_u < 2*(k+k_user)) ||
                      (nnz_u_vec > (size_t)(2*(k+k_user))) ||
                      (NA_as_zero_U);
        prefer_CtC = true;
    if (precomputedBtB == NULL)
        prefer_BtB = false;
    if (precomputedCtC == NULL || (!p && !nnz_u_vec))
        prefer_CtC = false;

    /* TODO: this function can be simplified. Many of the code paths are
       redundant and/or do not provide a speed up - these are a result of
       earlier experimentation which was not porperly cleaned up later. */

    /* TODO: here should do the parts from C first as then they can be rescaled
       after being summed. */


    /* t(B)*t(X) - t(B)*B*a */
    if (Xa_dense != NULL && cnt_NA_x == 0 && weight == NULL)
    {
        cblas_tgemv(CblasRowMajor, CblasTrans,
                    n, k+k_main,
                    1., B + k_item, ldb,
                    Xa_dense, 1,
                    0., r + k_user, 1);
        if (prefer_BtB)
            cblas_tsymv(CblasRowMajor, CblasUpper, k+k_main,
                        -1., precomputedBtB, ld_BtB,
                        a_vec + k_user, 1,
                        1., r + k_user, 1);
        else
            for (size_t ix = 0; ix < (size_t)n; ix++) {
                coef = cblas_tdot(k + k_main,
                                  B + (size_t)k_item + ix*(size_t)ldb, 1,
                                  a_vec + k_user, 1);
                cblas_taxpy(k + k_main, -coef,
                            B + (size_t)k_item + ix*(size_t)ldb, 1,
                            r + k_user, 1);
        }
    }

    else if (Xa_dense != NULL)
    {
        if (weight == NULL && prefer_BtB)
        {
            for (size_t ix = 0; ix < (size_t)n; ix++)
            {
                if (isnan(Xa_dense[ix])) {
                    coef = cblas_tdot(k + k_main,
                                      B + (size_t)k_item + ix*(size_t)ldb, 1,
                                      a_vec + k_user, 1);
                    cblas_taxpy(k + k_main, coef,
                                B + (size_t)k_item + ix*(size_t)ldb, 1,
                                r + k_user, 1);
                }
                
                else {
                    cblas_taxpy(k + k_main,
                                Xa_dense[ix],
                                B + (size_t)k_item + ix*(size_t)ldb, 1,
                                r + k_user, 1);
                }
            }
            cblas_tsymv(CblasRowMajor, CblasUpper, k+k_main,
                        -1., precomputedBtB, ld_BtB,
                        a_vec + k_user, 1,
                        1., r + k_user, 1);
        }

        else
            for (size_t ix = 0; ix < (size_t)n; ix++)
                if (!isnan(Xa_dense[ix])) {
                    w_this = (weight == NULL)? 1. : weight[ix];
                    cblas_taxpy(k + k_main,
                                w_this * Xa_dense[ix],
                                B + (size_t)k_item + ix*(size_t)ldb, 1,
                                r + k_user, 1);
                    coef = cblas_tdot(k + k_main,
                                      B + (size_t)k_item + ix*(size_t)ldb, 1,
                                      a_vec + k_user, 1);
                    cblas_taxpy(k + k_main, -w_this * coef,
                                B + (size_t)k_item + ix*(size_t)ldb, 1,
                                r + k_user, 1);
                }
    }

    else if (NA_as_zero_X)
    {
        if (weight == NULL)
            tgemv_dense_sp(
                n, k+k_main,
                1., B + k_item, (size_t)ldb,
                ixB, Xa, nnz,
                r + k_user
            );
        else {
            tgemv_dense_sp_weighted(
                n, k+k_main,
                weight, B + k_item, (size_t)ldb,
                ixB, Xa, nnz,
                r + k_user
            );
        }
        
        if (weight == NULL)
        {
            if (prefer_BtB)
                cblas_tsymv(CblasRowMajor, CblasUpper, k+k_main,
                            -1., precomputedBtB, ld_BtB,
                            a_vec + k_user, 1,
                            1., r + k_user, 1);
            else
                for (size_t ix = 0; ix < (size_t)n; ix++) {
                    coef = cblas_tdot(k + k_main,
                                      B + (size_t)k_item + ix*(size_t)ldb, 1,
                                      a_vec + k_user, 1);
                    cblas_taxpy(k + k_main, -coef,
                                B + (size_t)k_item + ix*(size_t)ldb, 1,
                                r + k_user, 1);
                }
        }
        
        else
        {
            if (prefer_BtB)
            {
                cblas_tsymv(CblasRowMajor, CblasUpper, k+k_main,
                            -1., precomputedBtB, ld_BtB,
                            a_vec + k_user, 1,
                            1., r + k_user, 1);
                for (size_t ix = 0; ix < nnz; ix++)
                {
                    coef = cblas_tdot(k + k_main,
                                      B
                                        + (size_t)k_item
                                        + (size_t)ixB[ix]*(size_t)ldb, 1,
                                      a_vec + k_user, 1);
                    cblas_taxpy(k + k_main, -((weight[ix]-1.) * coef),
                                B + (size_t)k_item + ix*(size_t)ldb, 1,
                                r + k_user, 1);
                }
            }

            else
            {
                cblas_tgemv(CblasRowMajor, CblasNoTrans,
                            n, k+k_main,
                            -1., B + k_item, ldb,
                            a_vec + k_user, 1,
                            0., wr, 1);
                for (size_t ix = 0; ix < nnz; ix++)
                    wr[ixB[ix]] *= weight[ix];
                cblas_tgemv(CblasRowMajor, CblasTrans,
                            n, k+k_main,
                            1., B + k_item, ldb,
                            wr, 1,
                            1., r + k_user, 1);
            }
        }
    }

    else
    {
        for (size_t ix = 0; ix < nnz; ix++)
        {
            w_this = (weight == NULL)? 1. : weight[ix];
            cblas_taxpy(k + k_main,
                        w_this * Xa[ix],
                        B + (size_t)k_item + (size_t)ixB[ix]*(size_t)ldb, 1,
                        r + k_user, 1);
            coef = cblas_tdot(k + k_main,
                              B
                                + (size_t)k_item
                                + (size_t)ixB[ix]*(size_t)ldb,
                              1,
                              a_vec + k_user, 1);
            cblas_taxpy(k + k_main, -w_this * coef,
                        B + (size_t)k_item + (size_t)ixB[ix]*(size_t)ldb, 1,
                        r + k_user, 1);
        }
    }

    /* t(C)*t(U) - t(C)*C*a */
    if (u_vec != NULL && cnt_NA_u == 0)
    {
        cblas_tgemv(CblasRowMajor, CblasTrans,
                    p, k_user+k,
                    w_user, C, k_totC,
                    u_vec, 1,
                    1., r, 1);
        if (prefer_CtC)
            cblas_tsymv(CblasRowMajor, CblasUpper, k_totC,
                        -w_user, precomputedCtC, k_totC,
                        a_vec, 1,
                        1., r, 1);
        else
            for (size_t ix = 0; ix < (size_t)p; ix++) {
                coef = cblas_tdot(k_user+k,
                                  C + ix*(size_t)k_totC, 1,
                                  a_vec, 1);
                cblas_taxpy(k, -w_user * coef,
                            C + ix*(size_t)k_totC, 1,
                            r, 1);
            }
    }

    else if (u_vec != NULL)
    {
        if (prefer_CtC)
        {
            cblas_tsymv(CblasRowMajor, CblasUpper, k_totC,
                        -w_user, precomputedCtC, k_totC,
                        a_vec, 1,
                        1., r, 1);
            for (size_t ix = 0; ix < (size_t)p; ix++)
            {
                if (isnan(u_vec[ix])) {
                    coef = cblas_tdot(k_user+k,
                                      C + ix*(size_t)k_totC, 1,
                                      a_vec, 1);
                    cblas_taxpy(k, w_user * coef,
                                C + ix*(size_t)k_totC, 1,
                                r, 1);
                }

                else {
                    cblas_taxpy(k_user+k,
                                w_user * u_vec[ix],
                                C + ix*(size_t)k_totC, 1,
                                r, 1);
                }
            }
        }

        else
            for (size_t ix = 0; ix < (size_t)p; ix++)
                if (!isnan(u_vec[ix])) {
                    cblas_taxpy(k_user+k,
                                w_user * u_vec[ix],
                                C + ix*(size_t)k_totC, 1,
                                r, 1);
                    coef = cblas_tdot(k_user+k,
                                      C + ix*(size_t)k_totC, 1,
                                      a_vec, 1);
                    cblas_taxpy(k, -w_user * coef,
                                C + ix*(size_t)k_totC, 1,
                                r, 1);
                }
    }

    else if (NA_as_zero_U)
    {
        tgemv_dense_sp(
                p, k_user+k,
                w_user, C, k_totC,
                u_vec_ixB, u_vec_sp, nnz_u_vec,
                r
            );

        if (prefer_CtC)
            cblas_tsymv(CblasRowMajor, CblasUpper, k_totC,
                        -w_user, precomputedCtC, k_totC,
                        a_vec, 1,
                        1., r, 1);
        else
            for (size_t ix = 0; ix < (size_t)p; ix++) {
                coef = cblas_tdot(k_user+k,
                                  C + ix*(size_t)k_totC, 1,
                                  a_vec, 1);
                cblas_taxpy(k, -w_user * coef,
                            C + ix*(size_t)k_totC, 1,
                            r, 1);
            }
    }

    else if (u_vec_sp != NULL)
    {
        for (size_t ix = 0; ix < nnz_u_vec; ix++)
        {
            cblas_taxpy(k_user+k,
                        w_user * u_vec_sp[ix],
                        C + (size_t)u_vec_ixB[ix]*(size_t)k_totC, 1,
                        r, 1);
            coef = cblas_tdot(k_user+k,
                              C + (size_t)u_vec_ixB[ix]*(size_t)k_totC, 1,
                              a_vec, 1);
            cblas_taxpy(k, -w_user * coef,
                        C + (size_t)u_vec_ixB[ix]*(size_t)k_totC, 1,
                        r, 1);
        }
    }

    /* t(Bi)*t(Xi) - t(Bi)*Bi*a */
    if (add_implicit_features)
    {
        cblas_tsymv(CblasRowMajor, CblasUpper, k+k_main_i,
                    -w_implicit, precomputedBiTBi, k+k_main_i,
                    a_vec + k_user, 1,
                    1., r + k_user, 1);

        if (Xa_dense != NULL)
            cblas_tgemv(CblasRowMajor, CblasTrans,
                        n, k+k_main_i,
                        w_implicit, Bi, k+k_main_i,
                        Xones, incXones,
                        1., r + k_user, 1);
            
        else
            tgemv_dense_sp(
                n, k+k_main_i,
                w_implicit, Bi, k+k_main_i,
                ixB, Xones, nnz,
                r + k_user
            );
    }

    /* diag(lam) */
    cblas_taxpy(k_totA, -lam, a_vec, 1, r, 1);
    if (lam != lam_last)
        r[k_totA-1] -= (lam_last-lam) * a_vec[k_totA-1];

    /* p := r */
    copy_arr(r, pp, k_totA);
    r_old = cblas_tdot(k_totA, r, 1, r, 1);

    #ifdef TEST_CG
    if (r_old <= 1e-15)
        return;
    #else
    if (r_old <= 1e-12)
        return;
    #endif

    for (int_t cg_step = 0; cg_step < max_cg_steps; cg_step++)
    {
        set_to_zero(Ap, k_totA);

        /* t(B)*B*p */
        if ((Xa_dense != NULL && cnt_NA_x == 0) ||
            (Xa_dense == NULL && NA_as_zero_X && weight == NULL))
        {
            if (weight == NULL && prefer_BtB)
                cblas_tsymv(CblasRowMajor, CblasUpper, k+k_main,
                            1., precomputedBtB, ld_BtB,
                            pp + k_user, 1,
                            0., Ap + k_user, 1);
            else
                for (size_t ix = 0; ix < (size_t)n; ix++)
                {
                    w_this = (weight == NULL)? 1. : weight[ix];
                    coef = cblas_tdot(k+k_main,
                                      B + (size_t)k_item + ix*(size_t)ldb, 1,
                                      pp + k_user, 1);
                    cblas_taxpy(k+k_main, w_this * coef,
                                B + (size_t)k_item + ix*(size_t)ldb, 1,
                                Ap + k_user, 1);
                }
        }

        else if (Xa_dense == NULL && NA_as_zero_X && weight != NULL)
        {
            if (prefer_BtB)
            {
                cblas_tsymv(CblasRowMajor, CblasUpper, k+k_main,
                            1., precomputedBtB, ld_BtB,
                            pp + k_user, 1,
                            0., Ap + k_user, 1);
                for (size_t ix = 0; ix < nnz; ix++) {
                    coef = cblas_tdot(k+k_main,
                                      B
                                        + (size_t)k_item
                                        + (size_t)ixB[ix]*(size_t)ldb,
                                      1,
                                      pp + k_user, 1);
                    cblas_taxpy(k+k_main, (weight[ix] - 1.) * coef,
                                B
                                  + (size_t)k_item
                                  + (size_t)ixB[ix]*(size_t)ldb,
                                1,
                                Ap + k_user, 1);
                }
            }

            else
            {
                cblas_tgemv(CblasRowMajor, CblasNoTrans,
                            n, k+k_main,
                            1., B + k_item, ldb,
                            pp + k_user, 1,
                            0., wr, 1);
                for (size_t ix = 0; ix < nnz; ix++)
                    wr[ixB[ix]] *= weight[ix];
                cblas_tgemv(CblasRowMajor, CblasTrans,
                            n, k+k_main,
                            1., B + k_item, ldb,
                            wr, 1,
                            0., Ap + k_user, 1);
            }
        }

        else if (Xa_dense != NULL)
        {
            if (weight == NULL && prefer_BtB)
            {
                cblas_tsymv(CblasRowMajor, CblasUpper, k+k_main,
                            1., precomputedBtB, ld_BtB,
                            pp + k_user, 1,
                            0., Ap + k_user, 1);
                for (size_t ix = 0; ix < (size_t)n; ix++)
                    if (isnan(Xa_dense[ix])) {
                        coef = cblas_tdot(k+k_main,
                                          B
                                            + (size_t)k_item
                                            + ix*(size_t)ldb, 1,
                                          pp + k_user, 1);
                        cblas_taxpy(k+k_main, -coef,
                                    B + (size_t)k_item + ix*(size_t)ldb, 1,
                                    Ap + k_user, 1);
                    }
            }

            else
                for (size_t ix = 0; ix < (size_t)n; ix++)
                {
                    if (!isnan(Xa_dense[ix])) {
                        w_this = (weight == NULL)? 1. : weight[ix];
                        coef = cblas_tdot(k+k_main,
                                          B
                                            + (size_t)k_item
                                            + ix*(size_t)ldb, 1,
                                          pp + k_user, 1);
                        cblas_taxpy(k+k_main, w_this * coef,
                                    B + (size_t)k_item + ix*(size_t)ldb, 1,
                                    Ap + k_user, 1);
                    }
                }
        }

        else
        {
            for (size_t ix = 0; ix < nnz; ix++)
            {
                w_this = (weight == NULL)? 1. : weight[ix];
                coef = cblas_tdot(k+k_main,
                                  B
                                    + (size_t)k_item
                                    + (size_t)ixB[ix]*(size_t)ldb,
                                  1,
                                  pp + k_user, 1);
                cblas_taxpy(k+k_main, w_this * coef,
                            B
                              + (size_t)k_item
                              + (size_t)ixB[ix]*(size_t)ldb,
                            1,
                            Ap + k_user, 1);
            }
        }

        /* t(C)*C*p */
        if ((u_vec != NULL && cnt_NA_u == 0) ||
            (u_vec == NULL && NA_as_zero_U))
        {
            if (prefer_CtC)
                cblas_tsymv(CblasRowMajor, CblasUpper, k_totC,
                            w_user, precomputedCtC, k_totC,
                            pp, 1,
                            1., Ap, 1);
            else
                for (size_t ix = 0; ix < (size_t)p; ix++) {
                    coef = cblas_tdot(k_user+k,
                                      C + ix*(size_t)k_totC, 1,
                                      pp, 1);
                    cblas_taxpy(k_user+k, w_user * coef,
                                C + ix*(size_t)k_totC, 1,
                                Ap, 1);
                }
        }

        else if (u_vec != NULL)
        {
            if (prefer_CtC)
            {
                cblas_tsymv(CblasRowMajor, CblasUpper, k_user+k,
                            w_user, precomputedCtC, k_user+k,
                            pp, 1,
                            1., Ap, 1);
                for (size_t ix = 0; ix < (size_t)p; ix++)
                    if (isnan(u_vec[ix])) {
                        coef = cblas_tdot(k_user+k,
                                          C + ix*(size_t)k_totC, 1,
                                          pp, 1);
                        cblas_taxpy(k_user+k, -w_user * coef,
                                    C + ix*(size_t)k_totC, 1,
                                    Ap, 1);
                    }
            }

            else
                for (size_t ix = 0; ix < (size_t)p; ix++)
                {
                    if (!isnan(u_vec[ix])) {
                        coef = cblas_tdot(k_user+k,
                                          C + ix*(size_t)k_totC, 1,
                                          pp, 1);
                        cblas_taxpy(k_user+k, w_user * coef,
                                    C + ix*(size_t)k_totC, 1,
                                    Ap, 1);
                    }
                }
        }

        else if (u_vec_sp != NULL)
        {
            for (size_t ix = 0; ix < nnz_u_vec; ix++)
            {
                coef = cblas_tdot(k_user+k,
                                  C + (size_t)u_vec_ixB[ix]*(size_t)k_totC, 1,
                                  pp, 1);
                cblas_taxpy(k_user+k, w_user * coef,
                            C + (size_t)u_vec_ixB[ix]*(size_t)k_totC, 1,
                            Ap, 1);
            }
        }

        /* t(Bi)*Bi*p */
        if (add_implicit_features)
        {
            cblas_tsymv(CblasRowMajor, CblasUpper, k+k_main_i,
                        w_implicit, precomputedBiTBi, k+k_main_i,
                        pp + k_user, 1,
                        1., Ap + k_user, 1);
        }

        /* diag(lam) */
        cblas_taxpy(k_totA, lam, pp, 1, Ap, 1);
        if (lam != lam_last)
            Ap[k_totA-1] += (lam_last-lam) * pp[k_totA-1];

        /* rest of the procedure */
        a = r_old / cblas_tdot(k_totA, pp, 1, Ap, 1);
        cblas_taxpy(k_totA,  a, pp, 1, a_vec, 1);
        cblas_taxpy(k_totA, -a, Ap, 1, r, 1);
        r_new = cblas_tdot(k_totA, r, 1, r, 1);
        #ifdef TEST_CG
        if (r_new <= 1e-15)
            break;
        #else
        if (r_new <= 1e-8)
            break;
        #endif

        cblas_tscal(k_totA, r_new / r_old, pp, 1);
        cblas_taxpy(k_totA, 1., r, 1, pp, 1);
        r_old = r_new;
    }
}


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
    real_t *restrict precomputedBtB,
    real_t *restrict precomputedCtC, /* should NOT be multiplied by weight */
    real_t *restrict buffer_real_t
)
{
    int_t k_totA = k_user + k + k_main;
    int_t k_totC = k_user + k;
    size_t ldb = k_item + k + k_main;
    int_t ld_BtB = k + k_main;
    real_t *restrict Ap = buffer_real_t;
    real_t *restrict pp = Ap + k_totA;
    real_t *restrict r  = pp + k_totA;
    set_to_zero(r, k_user);
    real_t r_new, r_old;
    real_t a, coef;

    bool prefer_CtC = (cnt_NA_u < 2*(k+k_user)) ||
                      (nnz_u_vec > (size_t)(2*(k+k_user)));
    if (NA_as_zero_U)
        prefer_CtC = true;
    if (precomputedCtC == NULL)
        prefer_CtC = false;

    if (u_vec != NULL && cnt_NA_u == p) {
        u_vec = NULL;
        nnz_u_vec = 0;
        NA_as_zero_U = false;
    }

    /* t(B)*t(X) - t(B)*B*a */
    cblas_tsymv(CblasRowMajor, CblasUpper, k+k_main,
                -1., precomputedBtB, ld_BtB,
                a_vec + k_user, 1,
                0., r + k_user, 1);
    for (size_t ix = 0; ix < nnz; ix++)
    {
        coef = cblas_tdot(k+k_main,
                          B + (size_t)k_item + (size_t)ixB[ix]*ldb, 1,
                          a_vec + k_user, 1);
        cblas_taxpy(k+k_main,
                    -(coef - 1.) * Xa[ix] - coef,
                    B + (size_t)k_item + (size_t)ixB[ix]*ldb, 1,
                    r + k_user, 1);
    }

    /* t(C)*t(U) - t(C)*C*a */
    if (u_vec != NULL && cnt_NA_u == 0)
    {
        cblas_tgemv(CblasRowMajor, CblasTrans,
                    p, k_user+k,
                    w_user, C, k_totC,
                    u_vec, 1,
                    1., r, 1);
        if (prefer_CtC)
            cblas_tsymv(CblasRowMajor, CblasUpper, k_totC,
                        -w_user, precomputedCtC, k_totC,
                        a_vec, 1,
                        1., r, 1);
        else
            for (size_t ix = 0; ix < (size_t)p; ix++) {
                coef = cblas_tdot(k_user+k,
                                  C + ix*(size_t)k_totC, 1,
                                  a_vec, 1);
                cblas_taxpy(k, -w_user * coef,
                            C + ix*(size_t)k_totC, 1,
                            r, 1);
            }
    }

    else if (u_vec != NULL)
    {
        if (prefer_CtC)
        {
            cblas_tsymv(CblasRowMajor, CblasUpper, k_totC,
                        -w_user, precomputedCtC, k_totC,
                        a_vec, 1,
                        1., r, 1);
            for (size_t ix = 0; ix < (size_t)p; ix++)
            {
                if (isnan(u_vec[ix])) {
                    coef = cblas_tdot(k_user+k,
                                      C + ix*(size_t)k_totC, 1,
                                      a_vec, 1);
                    cblas_taxpy(k, w_user * coef,
                                C + ix*(size_t)k_totC, 1,
                                r, 1);
                }

                else {
                    cblas_taxpy(k_user+k,
                                w_user * u_vec[ix],
                                C + ix*(size_t)k_totC, 1,
                                r, 1);
                }
            }
        }

        else
            for (size_t ix = 0; ix < (size_t)p; ix++)
                if (!isnan(u_vec[ix])) {
                    cblas_taxpy(k_user+k,
                                w_user * u_vec[ix],
                                C + ix*(size_t)k_totC, 1,
                                r, 1);
                    coef = cblas_tdot(k_user+k,
                                      C + ix*(size_t)k_totC, 1,
                                      a_vec, 1);
                    cblas_taxpy(k, -w_user * coef,
                                C + ix*(size_t)k_totC, 1,
                                r, 1);
                }
    }

    else if (u_vec == NULL && NA_as_zero_U)
    {
        tgemv_dense_sp(
                p, k_user+k,
                w_user, C, k_totC,
                u_vec_ixB, u_vec_sp, nnz_u_vec,
                r
            );

        if (prefer_CtC)
            cblas_tsymv(CblasRowMajor, CblasUpper, k_totC,
                        -w_user, precomputedCtC, k_totC,
                        a_vec, 1,
                        1., r, 1);
        else
            for (size_t ix = 0; ix < (size_t)p; ix++) {
                coef = cblas_tdot(k_user+k,
                                  C + ix*(size_t)k_totC, 1,
                                  a_vec, 1);
                cblas_taxpy(k, -w_user * coef,
                            C + ix*(size_t)k_totC, 1,
                            r, 1);
            }
    }

    else if (u_vec_sp != NULL)
    {
        for (size_t ix = 0; ix < nnz_u_vec; ix++)
        {
            cblas_taxpy(k_user+k,
                        w_user * u_vec_sp[ix],
                        C + (size_t)u_vec_ixB[ix]*(size_t)k_totC, 1,
                        r, 1);
            coef = cblas_tdot(k_user+k,
                              C + (size_t)u_vec_ixB[ix]*(size_t)k_totC, 1,
                              a_vec, 1);
            cblas_taxpy(k, -w_user * coef,
                        C + (size_t)u_vec_ixB[ix]*(size_t)k_totC, 1,
                        r, 1);
        }
    }

    /* diag(lam) */
    cblas_taxpy(k_totA, -lam, a_vec, 1, r, 1);

    /* p := r */
    copy_arr(r, pp, k_totA);
    r_old = cblas_tdot(k_totA, r, 1, r, 1);

    #ifdef TEST_CG
    if (r_old <= 1e-15)
        return;
    #else
    if (r_old <= 1e-12)
        return;
    #endif

    for (int_t cg_step = 0; cg_step < max_cg_steps; cg_step++)
    {
        /* t(B)*B*p */
        if (k_user)
            set_to_zero(Ap, k_user);
        cblas_tsymv(CblasRowMajor, CblasUpper, k+k_main,
                    1., precomputedBtB, ld_BtB,
                    pp + k_user, 1,
                    0., Ap + k_user, 1);
        for (size_t ix = 0; ix < nnz; ix++) {
            coef = cblas_tdot(k+k_main,
                              B + (size_t)k_item + (size_t)ixB[ix]*ldb, 1,
                              pp + k_user, 1);
            cblas_taxpy(k+k_main,
                        coef * (Xa[ix] - 1.) + coef,
                        B + (size_t)k_item + (size_t)ixB[ix]*ldb, 1,
                        Ap + k_user, 1);
        }

        /* t(C)*C*p */
        if ((u_vec != NULL && cnt_NA_u == 0) ||
            (u_vec == NULL && NA_as_zero_U))
        {
            if (prefer_CtC)
                cblas_tsymv(CblasRowMajor, CblasUpper, k_totC,
                            w_user, precomputedCtC, k_totC,
                            pp, 1,
                            1., Ap, 1);
            else
                for (size_t ix = 0; ix < (size_t)p; ix++) {
                    coef = cblas_tdot(k_user+k,
                                      C + ix*(size_t)k_totC, 1,
                                      pp, 1);
                    cblas_taxpy(k_user+k, w_user * coef,
                                C + ix*(size_t)k_totC, 1,
                                Ap, 1);
                }
        }

        else if (u_vec != NULL)
        {
            if (prefer_CtC)
            {
                cblas_tsymv(CblasRowMajor, CblasUpper, k_totC,
                            w_user, precomputedCtC, k_totC,
                            pp, 1,
                            1., Ap, 1);
                for (size_t ix = 0; ix < (size_t)p; ix++)
                    if (isnan(u_vec[ix])) {
                        coef = cblas_tdot(k_user+k,
                                          C + ix*(size_t)k_totC, 1,
                                          pp, 1);
                        cblas_taxpy(k_user+k, -w_user * coef,
                                    C + ix*(size_t)k_totC, 1,
                                    Ap, 1);
                    }
            }

            else
                for (size_t ix = 0; ix < (size_t)p; ix++)
                {
                    if (!isnan(u_vec[ix])) {
                        coef = cblas_tdot(k_user+k,
                                          C + ix*(size_t)k_totC, 1,
                                          pp, 1);
                        cblas_taxpy(k_user+k, w_user * coef,
                                    C + ix*(size_t)k_totC, 1,
                                    Ap, 1);
                    }
                }
        }

        else if (u_vec_sp != NULL)
        {
            for (size_t ix = 0; ix < nnz_u_vec; ix++)
            {
                coef = cblas_tdot(k_user+k,
                                  C + (size_t)u_vec_ixB[ix]*(size_t)k_totC, 1,
                                  pp, 1);
                cblas_taxpy(k_user+k, w_user * coef,
                            C + (size_t)u_vec_ixB[ix]*(size_t)k_totC, 1,
                            Ap, 1);
            }
        }

        /* diag(lam) */
        cblas_taxpy(k_totA, lam, pp, 1, Ap, 1);

        /* rest of the procedure */
        a = r_old / cblas_tdot(k_totA, Ap, 1, pp, 1);
        cblas_taxpy(k_totA,  a, pp, 1, a_vec, 1);
        cblas_taxpy(k_totA, -a, Ap, 1, r, 1);
        r_new = cblas_tdot(k_totA, r, 1, r, 1);
        #ifdef TEST_CG
        if (r_new <= 1e-15)
            break;
        #else
        if (r_new <= 1e-8)
            break;
        #endif
        cblas_tscal(k_totA, r_new / r_old, pp, 1);
        cblas_taxpy(k_totA, 1., r, 1, pp, 1);
        r_old = r_new;
    }

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
    int_t k, int_t k_user, int_t k_main,
    real_t lam, real_t w_main, real_t w_user,
    bool NA_as_zero_U,
    bool nonneg
)
{
    if (NA_as_zero_U && u_bin_vec != NULL) {
        fprintf(stderr, "Cannot use 'NA_as_zero_U' when there is 'u_bin'\n");
        #ifndef _FOR_R
        fflush(stderr);
        #endif
        return 2;
    }
    int_t retval = 0;
    int_t cnt_NA_u_vec = 0;
    int_t cnt_NA_u_bin_vec = 0;
    if (u_vec != NULL || (u_vec_sp != NULL && !NA_as_zero_U))
        preprocess_vec(u_vec, p, u_vec_ixB, u_vec_sp, nnz_u_vec,
                       0., 0., col_means, (real_t*)NULL, &cnt_NA_u_vec);
    if (u_bin_vec != NULL)
        cnt_NA_u_bin_vec = count_NAs(u_bin_vec, (size_t)pbin, 1);

    if (k_main > 0)
        set_to_zero(a_vec + k_user+k, k_main);

    real_t *restrict a_prev = NULL;

    real_t *restrict buffer_real_t = NULL;
    size_t size_buffer = 0;

    if (w_main != 1.) {
        lam /= w_main;
        w_user /= w_main;
    }

    /* If there is no data, just return zeros */
    if (    ((u_vec != NULL && cnt_NA_u_vec == p) ||
             (u_vec == NULL && nnz_u_vec == 0))
                    &&
            (u_bin_vec == NULL || cnt_NA_u_bin_vec == pbin)  )
    {
        set_to_zero(a_vec, k_user+k+k_main);
        goto cleanup;
    }

    if (nonneg) {
        a_prev = (real_t*)calloc(k_user+k+k_main, sizeof(real_t));
        if (a_prev == NULL) goto throw_oom;
    }

    /* If there are no binary variables, solution can be obtained through
       closed form */
    if (u_bin_vec == NULL)
    {
        size_buffer = square(k_user + k);
        if (TransCtCinvCt != NULL && !nonneg &&
            ((cnt_NA_u_vec == 0 && u_vec != NULL) ||
             (u_vec == NULL && NA_as_zero_U)))
        {
            size_buffer = 0;
        }
        if (size_buffer) {
            buffer_real_t = (real_t*)malloc(size_buffer * sizeof(real_t));
            if (buffer_real_t == NULL) goto throw_oom;
        }

        factors_closed_form(a_vec, k_user + k,
                            C, p, k_user + k,
                            u_vec, cnt_NA_u_vec==0,
                            u_vec_sp, u_vec_ixB, nnz_u_vec,
                            (real_t*)NULL,
                            buffer_real_t, lam/w_user, lam/w_user,
                            TransCtCinvCt, CtCw, cnt_NA_u_vec, k_user + k,
                            false, true, w_user, p,
                            (real_t*)NULL, NA_as_zero_U,
                            false, 0,
                            nonneg, max2(k_user+k, (int_t)10*(k_user+k)),a_prev,
                            true);
    }

    else
    {
        /* Otherwise, need to take a gradient-based approach with a solver. */
        buffer_real_t = (real_t*)malloc((size_t)max2(p, pbin)*sizeof(real_t));
        if (buffer_real_t == NULL) goto throw_oom;

        retval = collective_factors_lbfgs(
            a_vec,
            k, k_user, 0, 0,
            u_vec, p,
            u_vec_ixB, u_vec_sp, nnz_u_vec,
            u_bin_vec, pbin,
            cnt_NA_u_vec>0, cnt_NA_u_bin_vec>0,
            (real_t*)NULL, 0,
            C, Cb,
            (real_t*)NULL, (int_t*)NULL, (real_t*)NULL, 0,
            (real_t*)NULL,
            buffer_real_t,
            lam, 1., w_user, lam
        );
    }

    cleanup:
        free(buffer_real_t);
        free(a_prev);
        return retval;
    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}

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
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t lam, real_t w_main, real_t w_user, real_t w_main_multiplier,
    bool NA_as_zero_U,
    bool nonneg
)
{
    int_t retval = 0;
    int_t k_totA = k_user + k + k_main;
    int_t cnt_NA_u_vec = 0;
    real_t *restrict a_prev = NULL;
    real_t *restrict buffer_real_t = (real_t*)malloc((size_t)square(k_totA)
                                                      *
                                                     sizeof(real_t));
    if (buffer_real_t == NULL) goto throw_oom;

    if (u_vec != NULL || (u_vec_sp != NULL && !NA_as_zero_U))
        preprocess_vec(u_vec, p, u_vec_ixB, u_vec_sp, nnz_u_vec,
                       0., 0., col_means, (real_t*)NULL, &cnt_NA_u_vec);

    if ((u_vec != NULL && cnt_NA_u_vec == p)
          ||
        (u_vec_sp != NULL && nnz_u_vec == 0))
    {
        set_to_zero(a_vec, k_totA);
        goto cleanup;
    }

    if (nonneg) {
        a_prev = (real_t*)calloc(k_totA, sizeof(real_t));
        if (a_prev == NULL) goto throw_oom;
    }

    if (w_main_multiplier != 1.)
        w_main *= w_main_multiplier;

    if (w_main != 1.) {
        w_user /= w_main;
        lam /= w_main;
    }

    collective_closed_form_block_implicit(
        a_vec,
        k, k_user, 0, k_main,
        B, n, C, p,
        (real_t*)NULL, (int_t*)NULL, (size_t)0,
        u_vec, cnt_NA_u_vec,
        u_vec_sp, u_vec_ixB, nnz_u_vec,
        NA_as_zero_U,
        lam, w_user,
        BeTBe,
        BtB,
        BeTBeChol,
        (real_t*)NULL,
        true, true, false, 0,
        nonneg, max2(k_totA, (int_t)10*k_totA), a_prev,
        buffer_real_t
    );

    cleanup:
        free(buffer_real_t);
        free(a_prev);
        return retval;
    throw_oom:
    {
        retval = 1;
        goto cleanup;
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
    int_t n_max, bool include_all_X,
    real_t *restrict TransBtBinvBt,
    real_t *restrict BtB,
    real_t *restrict BeTBeChol,
    real_t *restrict BiTBi,
    real_t *restrict CtCw,
    bool NA_as_zero_U, bool NA_as_zero_X,
    bool nonneg,
    real_t *restrict B_plus_bias
)
{
    if (u_bin_vec != NULL && (NA_as_zero_X || NA_as_zero_U)) {
        fprintf(stderr, "Cannot use 'NA_as_zero' when there is 'u_bin'\n");
        #ifndef _FOR_R
        fflush(stderr);
        #endif
        return 2;
    }
    if (u_bin_vec != NULL && add_implicit_features) {
        fprintf(stderr, "Cannot use implicit features when there is 'u_bin'\n");
        #ifndef _FOR_R
        fflush(stderr);
        #endif
        return 2;
    }

    int_t retval = 0;

    int_t cnt_NA_u_vec = 0;
    int_t cnt_NA_u_bin_vec = 0;
    int_t cnt_NA_x = 0;
    bool append_bias = (B_plus_bias != NULL && a_bias != NULL);
    if (u_bin_vec != NULL)
        cnt_NA_u_bin_vec = count_NAs(u_bin_vec, (size_t)pbin, 1);
    if (u_vec != NULL || (u_vec_sp != NULL && !NA_as_zero_U))
        preprocess_vec(u_vec, p, u_vec_ixB, u_vec_sp, nnz_u_vec,
                       0., 0., col_means, (real_t*)NULL, &cnt_NA_u_vec);
    if (!NA_as_zero_X || Xa_dense != NULL)
        preprocess_vec(Xa_dense, n, ixB, Xa, nnz,
                       glob_mean, lam_bias, biasB,
                       (B_plus_bias == NULL)? a_bias : (real_t*)NULL,
                       &cnt_NA_x);

    /* If there is no data, can just set it to zero */
    if (
        ((Xa_dense != NULL && cnt_NA_x == n) || (Xa_dense == NULL && nnz == 0))
            &&
        (   (u_vec != NULL && cnt_NA_u_vec == p)
                ||
            (u_vec == NULL && nnz_u_vec == 0)
        )
            &&
        (u_bin_vec == NULL || cnt_NA_u_bin_vec == 0)
        )
    {
        if (append_bias) *a_bias = 0;
        set_to_zero(a_vec, k_user + k + k_main);
        return 0;
    }

    /* If there is no 'X' data but there is 'U', should call the cold version */
    else if (
                !add_implicit_features &&
                ((Xa_dense != NULL && cnt_NA_x == n) ||
                 (Xa_dense == NULL && nnz == 0 && !NA_as_zero_X))
            )
    {
        if (append_bias) *a_bias = 0;
        return collective_factors_cold(
            a_vec,
            u_vec, p,
            u_vec_sp, u_vec_ixB, nnz_u_vec,
            u_bin_vec, pbin,
            C, Cb,
            (real_t*)NULL,
            CtCw,
            col_means,
            k, k_user, k_main,
            lam, w_main, w_user,
            NA_as_zero_U,
            nonneg
        );
    }

    /* Otherwise (expected case), calculate the 'warm' factors */
    real_t *restrict Xones = NULL;
    if (add_implicit_features)
    {
        if (Xa_dense != NULL)
            Xones = (real_t*)malloc((size_t)n*sizeof(real_t));
        else
            Xones = (real_t*)malloc(nnz*sizeof(real_t));
        if (Xones == NULL) return 1;

        if (Xa_dense != NULL)
            for (int_t ix = 0; ix < n; ix++)
                Xones[ix] = isnan(Xa_dense[ix])? 0. : 1.;
        else
            for (size_t ix = 0; ix < nnz; ix++)
                Xones[ix] = 1.;
    }

    real_t *restrict buffer_real_t = NULL;
    size_t size_buffer;

    real_t *restrict a_prev = NULL;
    
    real_t *restrict a_plus_bias = NULL;
    if (append_bias) {
        a_plus_bias = (real_t*)malloc((size_t)(k_user+k+k_main+1)
                                       * sizeof(real_t));
        if (a_plus_bias == NULL) goto throw_oom;
    }

    if (nonneg) {
        a_prev = (real_t*)calloc(k_user+k+k_main+append_bias, sizeof(real_t));
        if (a_prev == NULL) goto throw_oom;
    }

    if (w_main != 1.) {
        w_user /= w_main;
        w_implicit /= w_main;
        lam /= w_main;
        lam_bias /= w_main;
        w_main = 1.;
    }

    /* If there's no side info, just need to apply the closed-form
       on the X data */
    if (u_vec == NULL && (nnz_u_vec == 0 && !NA_as_zero_U) &&
        u_bin_vec == NULL && !add_implicit_features)
    {
        size_buffer = square(k + k_main + (int)append_bias);
        if (TransBtBinvBt != NULL && weight == NULL && !nonneg &&
            ((cnt_NA_x == 0 && Xa_dense != NULL) ||
             (Xa_dense == NULL && NA_as_zero_X)) )
        {
            size_buffer = 0;
        }
        if (size_buffer) {
            buffer_real_t = (real_t*)malloc(size_buffer*sizeof(real_t));
            if (buffer_real_t == NULL) goto throw_oom;
        }

        if (k_user > 0) {
            if (a_plus_bias == NULL)
                set_to_zero(a_vec, k_user);
            else
                set_to_zero(a_plus_bias, k_user);
        }

        if (!append_bias)
            factors_closed_form(a_vec + k_user, k+k_main,
                                B + k_item, n, k_item+k+k_main,
                                Xa_dense, cnt_NA_x==0,
                                Xa, ixB, nnz,
                                weight,
                                buffer_real_t,
                                lam, lam,
                                TransBtBinvBt, BtB,
                                cnt_NA_x, k+k_main,
                                false, false, 1., include_all_X? n_max : n,
                                (real_t*)NULL, NA_as_zero_X,
                                false, 0,
                                nonneg, max2(k+k_main, (int_t)10*(k+k_main)),
                                a_prev,
                                true);
        else
            factors_closed_form(a_plus_bias + k_user, k+k_main+1,
                                B_plus_bias + k_item, n, k_item+k+k_main+1,
                                Xa_dense, cnt_NA_x==0,
                                Xa, ixB, nnz,
                                weight,
                                buffer_real_t,
                                lam, lam_bias,
                                TransBtBinvBt, BtB,
                                cnt_NA_x, k+k_main+1,
                                false, false, 1., include_all_X? n_max : n,
                                (real_t*)NULL, NA_as_zero_X,
                                false, 0,
                                nonneg, max2(k+k_main+1,(int_t)10*(k+k_main+1)),
                                a_prev,
                                true);
    }

    /* If there are binary variables, there's no closed form solution,
       so it will follow a gradient-based optimization approach with
       the L-BFGS solver */
    else if (u_bin_vec != NULL)
    {
        size_t size_buffer = max2(p, pbin);
        if (Xa_dense != NULL)
            size_buffer = max2(size_buffer, (size_t)n);
        buffer_real_t = (real_t*)malloc(size_buffer*sizeof(real_t));
        if (buffer_real_t == NULL) goto throw_oom;

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
                buffer_real_t,
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
                buffer_real_t,
                lam, w_main, w_user, lam_bias
            );
    }

    /* If there's no binary data, can apply the closed form on extended block
       matrices Xe and Be, whose composition differs according to the
       independent components */
    else
    {
        size_buffer = square(k_user+k+k_main+(int)append_bias);
        if (add_implicit_features && BiTBi == NULL)
            size_buffer += square(k+k_main);
        buffer_real_t = (real_t*)malloc(size_buffer*sizeof(real_t));
        if (buffer_real_t == NULL) goto throw_oom;

        if (!append_bias)
            collective_closed_form_block(
                a_vec,
                k, k_user, k_item, k_main,
                Xa_dense,
                Xa, ixB, nnz,
                u_vec_ixB, u_vec_sp, nnz_u_vec,
                u_vec,
                NA_as_zero_X, NA_as_zero_U,
                B, n, k_item+k+k_main,
                C, p,
                Bi, k_main, add_implicit_features,
                Xones, 1,
                weight,
                lam, w_user, w_implicit, lam,
                BtB, cnt_NA_x,
                CtCw, cnt_NA_u_vec,
                BeTBeChol, include_all_X? n_max : n,
                BiTBi,
                true, true, false, 0,
                nonneg, max2(k_user+k+k_main, (int_t)10*(k_user+k+k_main)),
                a_prev,
                buffer_real_t
            );
        else
            collective_closed_form_block(
                a_plus_bias,
                k, k_user, k_item, k_main+1,
                Xa_dense,
                Xa, ixB, nnz,
                u_vec_ixB, u_vec_sp, nnz_u_vec,
                u_vec,
                NA_as_zero_X, NA_as_zero_U,
                B_plus_bias, n, k_item+k+k_main+1,
                C, p,
                Bi, k_main, add_implicit_features,
                Xones, 1,
                weight,
                lam, w_user, w_implicit, lam_bias,
                BtB, cnt_NA_x,
                CtCw, cnt_NA_u_vec,
                BeTBeChol, include_all_X? n_max : n,
                BiTBi,
                true, true, false, 0,
                nonneg, max2(k_user+k+k_main+1, (int_t)10*(k_user+k+k_main+1)),
                a_prev,
                buffer_real_t
            );
        retval = 0;
    }

    if (append_bias) {
        memcpy(a_vec, a_plus_bias, (size_t)(k_user+k+k_main)*sizeof(real_t));
        *a_bias = a_plus_bias[k_user+k+k_main];
    }

    cleanup:
        free(buffer_real_t);
        free(a_plus_bias);
        free(a_prev);
        free(Xones);
        return retval;
    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}

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
    real_t lam, real_t alpha, real_t w_main, real_t w_user,
    real_t w_main_multiplier,
    real_t *restrict BeTBe,
    real_t *restrict BtB,
    real_t *restrict BeTBeChol
)
{
    int_t retval = 0;
    int_t cnt_NA_u_vec = 0;
    int_t k_totA = k_user + k + k_main;
    real_t *restrict buffer_real_t = (real_t*)malloc((size_t)square(k_totA)
                                                      *
                                                     sizeof(real_t));
    if (buffer_real_t == NULL) return 1;

    real_t *restrict a_prev = NULL;
    if (nonneg) {
        a_prev = (real_t*)calloc(k_totA, sizeof(real_t));
        if (a_prev == NULL) goto throw_oom;
    }

    w_main *= w_main_multiplier;
    if (w_main != 1.) {
        lam /= w_main;
        w_user /= w_main;
    }

    if (alpha != 1.)
        tscal_large(Xa, alpha, nnz, 1);

    if (u_vec != NULL || nnz_u_vec || NA_as_zero_U) {
        if (u_vec != NULL || nnz_u_vec)
            preprocess_vec(u_vec, p, u_vec_ixB, u_vec_sp, nnz_u_vec,
                           0., 0., col_means, (real_t*)NULL, &cnt_NA_u_vec);

        collective_closed_form_block_implicit(
            a_vec,
            k, k_user, k_item, k_main,
            B, n, C, p,
            Xa, ixB, nnz,
            u_vec, cnt_NA_u_vec,
            u_vec_sp, u_vec_ixB, nnz_u_vec,
            NA_as_zero_U,
            lam, w_user,
            BeTBe,
            BtB,
            BeTBeChol,
            (real_t*)NULL,
            true, true, false, 0,
            nonneg, max2(k_totA, (int_t)10*k_totA), a_prev,
            buffer_real_t
        );
    }

    else {
        set_to_zero(a_vec, k_user+k+k_main);
        factors_implicit_chol(
            a_vec + k_user, k+k_main,
            B + k_item, (size_t)(k_item+k+k_main),
            Xa, ixB, nnz,
            lam,
            BtB, k+k_main,
            nonneg, max2(k_totA, (int_t)10*k_totA), a_prev,
            buffer_real_t
        );
    }

    cleanup:
        free(buffer_real_t);
        free(a_prev);
        return retval;
    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}

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
    int_t nthreads,
    real_t *restrict buffer_real_t
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ia;
    #endif
    int_t k_totA = k_user + k + k_main + padding;
    int_t k_totB = k_item + k + k_main + padding;
    int_t k_totC = k_user + k;
    int_t k_totX = k + k_main;
    int_t m_max = max2(m, m_u);

    real_t f = 0.;
    real_t err;
    size_t ib;

    set_to_zero_(g_A, (size_t)m_max*(size_t)k_totA, nthreads);

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
                    buffer_real_t
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
                    buffer_real_t
                );
    }

    else
    {
        real_t *restrict Ax = A + k_user;
        real_t *restrict Bx = B + k_item;
        real_t *restrict g_Ax = g_A + k_user;
        real_t err_row = 0;
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                shared(Xcsr_p, Xcsr_i, Xcsr, Ax, Bx, g_Ax, \
                       k_totA, k_totB, weight, w_main) \
                private(ib, err) firstprivate(err_row) reduction(+:f)
        for (size_t_for ia = 0; ia < (size_t)m; ia++)
        {
            err_row = 0;
            for (size_t ix = Xcsr_p[ia]; ix < Xcsr_p[ia+(size_t)1]; ix++)
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
                    U, (real_t*)NULL,
                    0., w_user, 0.,
                    false, false,
                    nthreads,
                    buffer_real_t
             );
    }

    else
    {
        real_t f_user = 0;
        real_t err_row = 0;
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                shared(U_csr_p, U_csr_i, U_csr, A, C, \
                       g_A, k_totA, k_totC, w_user) \
                private(ib, err, err_row) reduction(+:f_user)
        for (size_t_for ia = 0; ia < (size_t)m_u; ia++)
        {
            err_row = 0;
            for (size_t ix = U_csr_p[ia]; ix < U_csr_p[ia+(size_t)1]; ix++)
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

    real_t f_reg = 0;
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

real_t wrapper_fun_grad_Adense_col
(
    void *instance,
    real_t *x,
    real_t *g,
    const size_t n,
    const real_t step
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
                data->buffer_real_t
            );
}

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
    bool nonneg,
    bool keep_precomputed,
    bool pass_allocated_BtB,
    bool pass_allocated_CtCw,
    bool pass_allocated_BeTBeChol,
    bool pass_allocated_BiTBi
)
{
    if (finalize_chol && use_cg)
    {
        return max2(
            buffer_size_optimizeA_collective(
                m, m_u, n, p,
                k, k_main, k_user,
                full_dense, near_dense, do_B,
                has_dense, has_sparse, has_weights, NA_as_zero_X,
                has_dense_U, has_sparse_U,
                full_dense_u, near_dense_u, NA_as_zero_U,
                add_implicit_features, k_main_i,
                nthreads,
                true, false,
                nonneg,
                keep_precomputed,
                pass_allocated_BtB,
                pass_allocated_CtCw,
                pass_allocated_BeTBeChol,
                pass_allocated_BiTBi
            ),
            buffer_size_optimizeA_collective(
                m, m_u, n, p,
                k, k_main, k_user,
                full_dense, near_dense, do_B,
                has_dense, has_sparse, has_weights, NA_as_zero_X,
                has_dense_U, has_sparse_U,
                full_dense_u, near_dense_u, NA_as_zero_U,
                add_implicit_features, k_main_i,
                nthreads,
                false, false,
                nonneg,
                keep_precomputed,
                pass_allocated_BtB,
                pass_allocated_CtCw,
                pass_allocated_BeTBeChol,
                pass_allocated_BiTBi
            )
        );
    }


    size_t m_x = m;
    size_t k_totA = k_user + k + k_main;
    size_t size_optimizeA = 0;
    size_t buffer_size = 0;
    size_t buffer_thread = 0;
    size_t size_alt = 0;
    size_t min_size = 0;

    bool will_use_BtB_here = false;
    bool will_use_CtC_here = false;

    if (!has_dense) do_B = false;

    if (m_x > m_u && m_u > 0)
    {
        if (
                !has_weights &&
                (   (has_dense && (full_dense || near_dense)) ||
                    (!has_dense && has_sparse && NA_as_zero_X) ) &&
                (   (has_dense_U && (full_dense_u || near_dense_u)) ||
                    (!has_dense_U && has_sparse_U && NA_as_zero_U) )
            )
        {
            will_use_BtB_here = true;
        }
        else {
            will_use_BtB_here = true;
            if (!has_dense && has_sparse && !NA_as_zero_X)
                will_use_BtB_here = false;
            if (has_dense && has_weights)
                will_use_BtB_here = false;
        }

        if (will_use_BtB_here && !pass_allocated_BtB)
        {
            min_size += square(k+k_main);
            pass_allocated_BtB = true;
        }

        if (add_implicit_features && !pass_allocated_BiTBi) {
            pass_allocated_BiTBi = true;
            min_size += square(k+k_main_i);
        }

        if (!add_implicit_features)
            size_optimizeA = buffer_size_optimizeA(
                n, full_dense, near_dense, do_B,
                has_dense, has_weights, NA_as_zero_X,
                k+k_main, nthreads,
                pass_allocated_BtB, keep_precomputed || will_use_BtB_here,
                use_cg, finalize_chol
            );
        else
            size_optimizeA = buffer_size_optimizeA_collective(
                m_x - m_u, 0, n, 0,
                k, k_main, k_user,
                full_dense, near_dense, do_B,
                has_dense, has_sparse, has_weights, NA_as_zero_X,
                false, false,
                false, false, false,
                add_implicit_features, k_main_i,
                nthreads,
                use_cg, finalize_chol,
                nonneg,
                keep_precomputed,
                pass_allocated_BtB,
                false,
                pass_allocated_BeTBeChol,
                pass_allocated_BiTBi
            );
    }

    else if (m_u > m_x)
    {
        if (
                !has_weights &&
                (   (has_dense && (full_dense || near_dense)) ||
                    (!has_dense && has_sparse && NA_as_zero_X) ) &&
                (   (has_dense_U && (full_dense_u || near_dense_u)) ||
                    (!has_dense_U && has_sparse_U && NA_as_zero_U) )
            )
        {
            will_use_CtC_here = true;
        }
        else {
            will_use_CtC_here = true;
            if (!has_dense_U && has_sparse_U && !NA_as_zero_U)
                will_use_CtC_here = false;
        }

        if (will_use_CtC_here && !pass_allocated_CtCw) {
            min_size += square(k_user+k);
            pass_allocated_CtCw = true;
        }

        size_optimizeA = buffer_size_optimizeA(
            p, full_dense_u, near_dense_u, false,
            has_dense_U, false, NA_as_zero_U,
            k_user+k, nthreads,
            pass_allocated_CtCw, keep_precomputed || will_use_CtC_here,
            use_cg, finalize_chol
        );
    }


    if (
            !has_weights &&
            (   (has_dense && (full_dense || near_dense)) ||
                (!has_dense && has_sparse && NA_as_zero_X) ) &&
            (   (has_dense_U && (full_dense_u || near_dense_u)) ||
                (!has_dense_U && has_sparse_U && NA_as_zero_U) ||
                (!p) )
        )
    {
        /* TODO: here can decrease memory usage by determining when will
           the BtB and CtC matrices be filled from 'optimizeA' */
        // bool filled_BtB = false || will_use_BtB_here;
        // bool filled_CtCw = false || will_use_CtC_here;
        bool filled_BtB = true;
        bool filled_CtCw = true;

        if (add_implicit_features)
        {
            if (!pass_allocated_BiTBi)
                buffer_size += square(k+k_main_i);
        }

        if (    ((has_dense && full_dense) ||
                 (!has_dense && NA_as_zero_X)) &&
                ((has_dense_U && full_dense_u) ||
                 (!has_dense_U && NA_as_zero_U)) &&
                !(filled_BtB) && !(filled_CtCw) && !keep_precomputed    )
        {
            if (pass_allocated_BeTBeChol)
                buffer_size += 0;
            else
                buffer_size += square(k_user+k+k_main);
        }

        else
        {
            if (pass_allocated_BtB)
                buffer_size += 0;
            else {
               buffer_size += square(k+k_main);
            }
            if (pass_allocated_CtCw)
                buffer_size += 0;
            else if (p) {
                buffer_size += square(k_user+k);
            }

            if (pass_allocated_BeTBeChol)
                size_alt += 0;
            else
                size_alt += square(k_user+k+k_main);
        }

        if (do_B)
            buffer_thread += n;
        buffer_thread += use_cg? (3*k_totA) : (square(k_totA));
        buffer_thread *= nthreads;
        buffer_thread = max2(buffer_thread, size_alt);

        buffer_size += buffer_thread;
    }

    else
    {
        bool prefer_BtB = true;
        bool prefer_CtC = true;
        if (!has_dense && has_sparse && !NA_as_zero_X)
            prefer_BtB = false;
        if (has_dense && has_weights)
            prefer_BtB = false;
        if (!has_dense_U && has_sparse_U && !NA_as_zero_U)
            prefer_CtC = false;

        if (add_implicit_features)
        {
            if (!pass_allocated_BiTBi)
                buffer_size += square(k+k_main_i);
        }

        if (prefer_BtB)
        {
            if (pass_allocated_BtB)
                buffer_size += 0;
            else {
                buffer_size += square(k+k_main);
            }
            
            if (!nonneg
                    &&
                prefer_CtC
                    &&
                (NA_as_zero_X || ((has_dense && !has_weights) &&
                                  (near_dense || full_dense))    )
                    &&
                ((has_dense_U && (near_dense_u || full_dense_u)) ||
                 (!has_dense_U && NA_as_zero_U) || (!p)))
            {
                if (pass_allocated_BeTBeChol)
                    buffer_size += 0;
                else {
                    buffer_size += square(k_user+k+k_main);
                }
            }
        }
        if (prefer_CtC)
        {
            if (pass_allocated_CtCw)
                buffer_size += 0;
            else {
                buffer_size += square(k_user+k);
            }
        }

        if (do_B)
            buffer_size += n * nthreads;
        if (do_B && has_weights)
            buffer_size += n * (nthreads+1);

        buffer_thread += use_cg? (3*k_totA) : (square(k_totA));
        if (use_cg && !has_dense && NA_as_zero_X && has_weights)
            buffer_thread += n;

        buffer_size += buffer_thread * nthreads;
    }


    return max2(buffer_size, size_optimizeA) + min_size;
}

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
)
{
    if (finalize_chol)
    {
        return max2(
                buffer_size_optimizeA_collective_implicit(
                    m, m_u, p,
                    k, k_main, k_user,
                    has_sparse_U,
                    NA_as_zero_U,
                    nthreads,
                    true,
                    pass_allocated_BtB,
                    pass_allocated_BeTBe,
                    pass_allocated_BeTBeChol,
                    pass_allocated_CtC,
                    false
                ),
                buffer_size_optimizeA_collective_implicit(
                    m, m_u, p,
                    k, k_main, k_user,
                    has_sparse_U,
                    NA_as_zero_U,
                    nthreads,
                    false,
                    pass_allocated_BtB,
                    pass_allocated_BeTBe,
                    pass_allocated_BeTBeChol,
                    pass_allocated_CtC,
                    false
                )
            );
    }

    size_t k_totA = k_user + k + k_main;

    size_t size_buffer = 0;
    if (!pass_allocated_BtB)
        size_buffer += square(k+k_main);

    size_t size_from_single = 0;
    if (m > m_u)
        size_from_single = buffer_size_optimizeA_implicit(
                                k + k_main, nthreads,
                                true,
                                use_cg, finalize_chol
                            );

    bool precomputedBeTBeChol_is_NULL = true;
    if (m_u > m &&
        !(has_sparse_U && !NA_as_zero_U) &&
        (!use_cg || p > 2 * k_totA))
    {
        precomputedBeTBeChol_is_NULL = false;
        if (!pass_allocated_BeTBeChol)
            size_buffer += square(k_totA);
    }

    bool prefer_CtC = !(has_sparse_U && !NA_as_zero_U);
    if (use_cg && prefer_CtC)
    {
        if (!pass_allocated_CtC)
            size_buffer += square(k_user+k);
    }

    if (!use_cg || !precomputedBeTBeChol_is_NULL)
    {
        if (!pass_allocated_BeTBe)
            size_buffer += square(k_totA);
    }

    size_t size_buffer_thread = use_cg? ((size_t)3 * k_totA) : (square(k_totA));
    size_buffer += nthreads * size_buffer_thread;

    size_buffer = max2(size_buffer, size_from_single);
    return size_buffer;
}

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
    real_t *restrict U, int_t cnt_NA_u[],
    bool full_dense_u, bool near_dense_u, bool NA_as_zero_U,
    real_t lam, real_t w_user, real_t w_implicit, real_t lam_last,
    bool do_B,
    int_t nthreads,
    bool use_cg, int_t max_cg_steps, bool is_first_iter,
    bool nonneg, int_t max_cd_steps, real_t *restrict A_prev,
    bool keep_precomputed,
    real_t *restrict precomputedBtB,
    real_t *restrict precomputedCtCw,
    real_t *restrict precomputedBeTBeChol,
    real_t *restrict precomputedBiTBi,
    bool *filled_BtB, bool *filled_CtCw, bool *filled_BeTBeChol,
    real_t *restrict buffer_real_t
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif
    char lo = 'L';
    int_t ignore = 0;
    bool ignore_bool = false;
    bool ignore_bool2 = false;

    #ifdef TEST_CG
    use_cg = true;
    max_cg_steps = 10000;
    is_first_iter = true;
    #endif

    *filled_BtB = false;
    *filled_CtCw = false;
    *filled_BeTBeChol = false;
    bool filled_BiTBi = false;

    /* TODO: could reduce number of operations and save memory by determining
       when the BiTBi matrix could be added to BtB and when not. */


    int_t k_totA = k_user + k + k_main;
    int_t k_totC = k_user + k;
    int_t k_pred = k_user + k + k_main;
    int_t m_x = m; /* 'm' will be overwritten later */
    size_t offset_square = k_user + k_user*(size_t)k_totA;
    if (Xfull == NULL) do_B = false;
    /* TODO: here should only need to set straight away the lower half,
       and only when there are un-even entries in each matrix */
    if (!use_cg || is_first_iter)
        set_to_zero_(A, (size_t)max2(m, m_u)*(size_t)lda - (size_t)(lda-k_totA),
                     nthreads);
    if (nonneg)
        set_to_zero_(A_prev,
                     (size_t)max2(m, m_u)*(size_t)lda - (size_t)(lda-k_totA),
                     nthreads);

    /* If one of the matrices has more rows than the other, the rows
       for the larger matrix will be independent and can be obtained
       from the single-matrix formula instead.
    
       Note: if using 'NA_as_zero_X', m_x >= m_u,
       whereas if using 'NA_as_zero_U', m_x <= m_u,
       and if using both, then m_x == m_u.  */
    /* TODO: refactor this, maybe move it after the end */
    if (m_x > m_u && m_u > 0)
    {
        bool will_use_BtB_here = false;

        if (
                weight == NULL &&
                (   (Xfull != NULL && (full_dense || near_dense)) ||
                    (Xfull == NULL && Xcsr_p != NULL && NA_as_zero_X) ) &&
                (   (U != NULL && (full_dense_u || near_dense_u)) ||
                    (U == NULL && U_csr_p != NULL && NA_as_zero_U) )
            )
        {
            will_use_BtB_here = true;
        }
        else {
            will_use_BtB_here = true;
            if (Xfull == NULL && Xcsr_p != NULL && !NA_as_zero_X)
                will_use_BtB_here = false;
            if (Xfull != NULL && weight != NULL)
                will_use_BtB_here = false;
        }

        if (will_use_BtB_here && precomputedBtB == NULL)
        {
            precomputedBtB = buffer_real_t;
            buffer_real_t += square(k+k_main);
        }

        if (add_implicit_features && precomputedBiTBi == NULL)
        {
            precomputedBiTBi = buffer_real_t;
            buffer_real_t += square(k+k_main_i);
        }

        int_t m_diff = m - m_u;
        if (!add_implicit_features)
            optimizeA(
                A + (size_t)k_user + (size_t)m_u*(size_t)lda, lda,
                B + k_item, ldb,
                m_diff, n, k + k_main,
                (Xfull != NULL)? ((size_t*)NULL) : (Xcsr_p + m_u),
                (Xfull != NULL)? ((int_t*)NULL) : Xcsr_i,
                (Xfull != NULL)? ((real_t*)NULL) : Xcsr,
                (Xfull == NULL)?
                    ((real_t*)NULL)
                        :
                    (do_B? (Xfull + m_u) : (Xfull + (size_t)m_u*(size_t)n)),
                ldX,
                full_dense, near_dense,
                (Xfull == NULL)? ((int_t*)NULL) : (cnt_NA_x + m_u),
                (weight == NULL)? ((real_t*)NULL)
                                : ( (Xfull == NULL)?
                                      (weight)
                                        :
                                      (do_B?
                                        (weight + m_u)
                                            :
                                        (weight + (size_t)m_u*(size_t)n)) ),
                NA_as_zero_X,
                lam, lam_last,
                false, false,
                nthreads,
                use_cg, max_cg_steps,
                nonneg, max_cd_steps,
                (A_prev == NULL)?
                    ((real_t*)NULL)
                        :
                    (A_prev + (size_t)k_user + (size_t)m_u*(size_t)lda),
                keep_precomputed || will_use_BtB_here,
                precomputedBtB,
                (keep_precomputed || will_use_BtB_here)?
                    filled_BtB : &ignore_bool,
                buffer_real_t
            );
        else
            optimizeA_collective(
                A + (size_t)m_u*(size_t)lda, lda, B, ldb,
                (real_t*)NULL,
                Bi,
                m_diff, 0, n, 0,
                k, k_main, k_user, k_item,
                (Xfull != NULL)? ((size_t*)NULL) : (Xcsr_p + m_u),
                (Xfull != NULL)? ((int_t*)NULL) : Xcsr_i,
                (Xfull != NULL)? ((real_t*)NULL) : Xcsr,
                (Xfull == NULL)?
                    ((real_t*)NULL)
                        :
                    (do_B? (Xfull + m_u) : (Xfull + (size_t)m_u*(size_t)n)),
                full_dense, near_dense, ldX,
                (Xfull == NULL)? ((int_t*)NULL) : (cnt_NA_x + m_u),
                (weight == NULL)? ((real_t*)NULL)
                                : ( (Xfull == NULL)?
                                      (weight)
                                        :
                                    (do_B?
                                        (weight + m_u)
                                            :
                                        (weight + (size_t)m_u*(size_t)n)) ),
                NA_as_zero_X,
                (Xfull == NULL)?
                    (Xones)
                        :
                    (do_B? (Xones + m_u) : (Xones + (size_t)m_u*(size_t)n)),
                k_main_i, ldXones, add_implicit_features,
                (size_t*)NULL, (int_t*)NULL, (real_t*)NULL,
                (real_t*)NULL, (int_t*)NULL,
                false, false, false,
                lam, w_user, w_implicit, lam_last,
                do_B,
                nthreads,
                use_cg, max_cg_steps, is_first_iter,
                nonneg, max_cd_steps,
                (A_prev == NULL)?
                    ((real_t*)NULL) : (A_prev + (size_t)m_u*(size_t)lda),
                keep_precomputed,
                precomputedBtB,
                (real_t*)NULL,
                precomputedBeTBeChol,
                precomputedBiTBi,
                filled_BtB, &ignore_bool, &ignore_bool2,
                buffer_real_t
            );

        if (add_implicit_features)
        {
            filled_BiTBi = true;
            if (use_cg && w_implicit != 1.)
                filled_BiTBi = false;
        }

        m_x = m_u;
    }

    else if (m_u > m_x)
    {
        bool will_use_CtC_here = false;

        if (
                weight == NULL &&
                (   (Xfull != NULL && (full_dense || near_dense)) ||
                    (Xfull == NULL && Xcsr_p != NULL && NA_as_zero_X) ) &&
                (   (U != NULL && (full_dense_u || near_dense_u)) ||
                    (U == NULL && U_csr_p != NULL && NA_as_zero_U) )
            )
        {
            will_use_CtC_here = true;
        }
        else {
            will_use_CtC_here = true;
            if (U == NULL && U_csr_p != NULL && !NA_as_zero_U)
                will_use_CtC_here = false;
        }

        if (will_use_CtC_here && precomputedCtCw == NULL) {
            precomputedCtCw = buffer_real_t;
            buffer_real_t += square(k_user+k);
        }

        int_t m_diff = m_u - m;
        optimizeA(
            A + (size_t)m*(size_t)lda, lda,
            C, k_totC,
            m_diff, p, k_user + k,
            (U != NULL)? ((size_t*)NULL) : (U_csr_p + m),
            (U != NULL)? ((int_t*)NULL) : U_csr_i,
            (U != NULL)? ((real_t*)NULL) : U_csr,
            (U == NULL)? ((real_t*)NULL) : (U + (size_t)m*(size_t)p),
            p,
            full_dense_u, near_dense_u,
            (U == NULL)? ((int_t*)NULL) : (cnt_NA_u + m),
            (real_t*)NULL,
            NA_as_zero_U,
            lam/w_user, lam/w_user,
            false, false,
            nthreads,
            use_cg, max_cg_steps,
            nonneg, max_cd_steps,
            (A_prev == NULL)?
                ((real_t*)NULL) : (A_prev + (size_t)m*(size_t)lda),
            keep_precomputed || will_use_CtC_here,
            precomputedCtCw,
            (keep_precomputed || will_use_CtC_here)? filled_CtCw : &ignore_bool,
            buffer_real_t
        );

        m_u = m_x;
    }

    m = max2(m_x, m_u);


    /* Case 1: both matrices are either (a) dense with few missing values and
       no weights, or (b) sparse with missing-as-zero.
       Here can use the closed-form solution on all the observations
       at once, and then do corrections one-by-one if there are any
       missing values. */
    if (
            weight == NULL &&
            (   (Xfull != NULL && (full_dense || near_dense)) ||
                (Xfull == NULL && Xcsr_p != NULL && NA_as_zero_X)   ) &&
            (   (U != NULL && (full_dense_u || near_dense_u)) ||
                (U == NULL && U_csr_p != NULL && NA_as_zero_U) ||
                (!p && U_csr_p == NULL)    )
        )
    {
        real_t *restrict bufferBtB = NULL;
        real_t *restrict bufferCtC = NULL;
        real_t *restrict bufferBeTBeChol = NULL;
        real_t *restrict bufferBiTBi = NULL;

        if (add_implicit_features)
        {
            /* Note: this won't be needed if next condition is met */
            if (precomputedBiTBi == NULL) {
                bufferBiTBi = buffer_real_t;
                buffer_real_t += square(k+k_main_i);
            } else {
                bufferBiTBi = precomputedBiTBi;
            }
        }

        if (    ((Xfull != NULL && full_dense) ||
                 (Xfull == NULL && NA_as_zero_X)) &&
                ((U != NULL && full_dense_u) ||
                 (U == NULL && NA_as_zero_U) || (!p && U_csr_p == NULL)) &&
                !(*filled_BtB) && !(*filled_CtCw) && !filled_BiTBi &&
                !keep_precomputed    )
        {
            if (add_implicit_features &&
                bufferBiTBi == buffer_real_t - square(k+k_main_i))
            {
                buffer_real_t -= square(k+k_main_i);
            }

            if (precomputedBeTBeChol != NULL)
                bufferBeTBeChol = precomputedBeTBeChol;
            else
                bufferBeTBeChol = buffer_real_t;
            /* Note: buffer_real_t won't be used any further if reaching here */
            build_BeTBe(
                bufferBeTBeChol,
                B, ldb, C,
                k, k_user, k_main, k_item,
                n, p,
                lam, w_user
            );
            if (add_implicit_features)
                cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                            k+k_main_i, n,
                            w_implicit, Bi, k+k_main_i,
                            1., bufferBeTBeChol + offset_square, k_totA);
            if (lam_last != lam)
                bufferBeTBeChol[square(k_totA)-1] += (lam_last - lam);
        }

        else
        {
            if (precomputedBtB != NULL)
                bufferBtB = precomputedBtB;
            else {
               bufferBtB = buffer_real_t;
               buffer_real_t += square(k+k_main);
            }
            if (precomputedCtCw != NULL)
                bufferCtC = precomputedCtCw;
            else if (p) {
                bufferCtC = buffer_real_t;
                buffer_real_t += square(k_user+k);
            }

            if (precomputedBeTBeChol != NULL)
                bufferBeTBeChol = precomputedBeTBeChol;
            else
                bufferBeTBeChol = buffer_real_t;
            /* Note: the Cholesky won't not be needed any more after this, and
               in such case the memory will get reused */

            if (add_implicit_features && !filled_BiTBi)
                cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                            k+k_main_i, n,
                            w_implicit, Bi, k+k_main_i,
                            0., bufferBiTBi, k+k_main_i);
            
            build_BtB_CtC(
                (*filled_BtB)? (real_t*)NULL : bufferBtB,
                (*filled_CtCw)? (real_t*)NULL : bufferCtC,
                B, n, ldb,
                C, p,
                k, k_user, k_main, k_item,
                1.,
                (real_t*)NULL
            );

            if (k_user || k_main || (!p && U_csr_p == NULL))
                set_to_zero(bufferBeTBeChol, square(k_totA));
            if (p || U_csr_p != NULL)
                copy_mat(k_totC, k_totC,
                         bufferCtC, k_totC,
                         bufferBeTBeChol, k_totA);
            if (w_user != 1. && p)
                cblas_tscal(square(k_totA) - k_main - k_main*k_totA, w_user,
                            bufferBeTBeChol, 1);
            sum_mat(k+k_main, k+k_main,
                    bufferBtB, k+k_main,
                    bufferBeTBeChol + offset_square, k_totA);
            if (add_implicit_features)
                sum_mat(k+k_main_i, k+k_main_i,
                        bufferBiTBi, k+k_main_i,
                        bufferBeTBeChol + offset_square, k_totA);

            add_to_diag(bufferBeTBeChol, lam, k_totA);
            if (lam_last != lam)
                bufferBeTBeChol[square(k_totA)-1] += (lam_last - lam);

            if (w_user != 1. && !use_cg && p &&
                (keep_precomputed || ((Xfull != NULL && !full_dense) ||
                                      (U != NULL && !full_dense_u))))
            {
                cblas_tscal(square(k_totC), w_user, bufferCtC, 1);
            }

            *filled_BtB = true;
            *filled_CtCw = true;
            *filled_BeTBeChol = true;
        }

        /* Note: this messes up the current values when there are NAs and using
           conjugate gradient, so it will reset them later. Could alternaively
           keep another matrix with the values before this override to restore
           them later. */
        /* TODO: keep track of older values when using CG method */
        if (use_cg && !is_first_iter && (k_user || Xfull == NULL))
        {
            set_to_zero_(A,
                         (size_t)max2(m, m_u)*(size_t)lda -(size_t)(lda-k_totA),
                         nthreads);
        }

        if (Xfull != NULL)
            build_XBw(
                A + k_user, lda,
                B + k_item, ldb,
                Xfull, ldX,
                m_x, n, k + k_main,
                1.,
                do_B, true
            );
        else if (Xcsr_p != NULL)
            tgemm_sp_dense(
                m_x, k+k_main, 1.,
                Xcsr_p, Xcsr_i, Xcsr,
                B + k_item, (size_t)ldb,
                A + k_user, (size_t)lda,
                nthreads
            );

        if (U != NULL)
            cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m_u, k_user + k, p,
                        w_user, U, p, C, k_user + k,
                        1., A, lda);
        else if (U_csr_p != NULL)
            tgemm_sp_dense(
                m_u, k_user+k, w_user,
                U_csr_p, U_csr_i, U_csr,
                C, (size_t)k_totC,
                A, (size_t)lda,
                nthreads
            );

        if (add_implicit_features)
        {
            if (Xfull != NULL)
                build_XBw(
                    A + k_user, lda,
                    Bi, k+k_main_i,
                    Xones, ldXones,
                    m_x, n, k + k_main_i,
                    w_implicit,
                    do_B, false
                );
            else
                tgemm_sp_dense(
                    m_x, k+k_main_i, w_implicit,
                    Xcsr_p, Xcsr_i, Xones,
                    Bi, k+k_main_i,
                    A + k_user, (size_t)lda,
                    nthreads
                );
        }

        #ifdef FORCE_NO_NAN_PROPAGATION
        if (!nonneg) {
            if ((Xfull != NULL && !full_dense) || (U != NULL && !full_dense_u))
                #pragma omp parallel for schedule(static) \
                        num_threads(min2(4, nthreads)) \
                        shared(A, m, k_totA, lda)
                for (size_t_for ix = 0;
                     ix < (size_t)m*(size_t)lda - (size_t)(lda - k_totA);
                     ix++)
                    A[ix] = isnan(A[ix])? (0.) : (A[ix]);
        }
        #endif

        if (!nonneg)
            tposv_(&lo, &k_pred, &m,
                   bufferBeTBeChol, &k_pred,
                   A, &lda,
                   &ignore);
        else
            solve_nonneg_batch(
                bufferBeTBeChol,
                A,
                A_prev,
                m, k_pred, lda,
                max_cd_steps,
                nthreads
            );

        if ((Xfull != NULL && !full_dense) || (U != NULL && !full_dense_u))
        {
            if (add_implicit_features && use_cg && w_implicit != 1.)
                cblas_tscal(square(k+k_main_i), 1./w_implicit, bufferBiTBi, 1);

            /* When doing the B matrix, the X matrix will be transposed
               and need to make a copy of the column for each observation,
               whereas the U matrix will be in the right order. */
            if (Xfull == NULL)
                do_B = false;
            /* TODO: do away with the 'bufferX', replace it instead with an
               'incX' parameter */
            real_t *restrict bufferX = buffer_real_t;
            real_t *restrict buffer_remainder = bufferX
                                                + (do_B?
                                                    ((size_t)n*(size_t)nthreads)
                                                    : (0));
            size_t size_buffer = use_cg? (3*k_totA) : (square(k_totA));

            #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                    shared(A, k_totA, B, C, k, k_user, k_item, k_main, \
                           m, m_x, m_u, n, p, lda, ldb, lam, lam_last, w_user, \
                           Xfull, cnt_NA_x, ldX, full_dense, \
                           Xcsr, Xcsr_i, Xcsr_p, \
                           add_implicit_features, Xones, w_implicit, k_main_i, \
                           U, cnt_NA_u, full_dense_u, \
                           U_csr, U_csr_i, U_csr_p, \
                           buffer_remainder, size_buffer, do_B, \
                           bufferBtB, bufferCtC, nthreads, use_cg, \
                           nonneg, max_cd_steps, A_prev) \
                    firstprivate(bufferX)
            for (size_t_for ix = 0; ix < (size_t)m; ix++)
            {
                if ((Xfull != NULL && cnt_NA_x[ix]) ||
                    (U != NULL && cnt_NA_u[ix]))
                {
                    if (Xfull != NULL)
                    {
                        if (!do_B)
                            bufferX = Xfull + ix*(size_t)n;
                        else
                            cblas_tcopy(n, Xfull + ix, ldX,
                                        bufferX
                                        +(size_t)n*(size_t)omp_get_thread_num(),
                                        1);
                    }

                    if (use_cg)
                        set_to_zero(A + ix*(size_t)lda, k_totA);
                        /* this is compensated by higher 'max_cg_steps' below */

                    collective_closed_form_block(
                        A + ix*(size_t)lda,
                        k, k_user, k_item, k_main,
                        (Xfull == NULL)?
                            ((real_t*)NULL)
                                :
                            (bufferX
                                + (do_B?
                                    ((size_t)n*(size_t)omp_get_thread_num())
                                        :
                                    ((size_t)0))),
                        (Xfull != NULL)? ((real_t*)NULL) : (Xcsr + Xcsr_p[ix]),
                        (Xfull != NULL)? ((int_t*)NULL) : (Xcsr_i + Xcsr_p[ix]),
                        (Xfull != NULL)?
                            (size_t)0 : (Xcsr_p[ix+(size_t)1] - Xcsr_p[ix]),
                        (U != NULL || U_csr_p == NULL)?
                            ((int_t*)NULL) : (U_csr_i + U_csr_p[ix]),
                        (U != NULL || U_csr_p == NULL)?
                            ((real_t*)NULL) : (U_csr + U_csr_p[ix]),
                        (U != NULL || U_csr_p == NULL)?
                            (size_t)0 : (U_csr_p[ix+(size_t)1] - U_csr_p[ix]),
                        (U == NULL)? ((real_t*)NULL) : (U + ix*(size_t)p),
                        NA_as_zero_X, NA_as_zero_U,
                        B, n, ldb,
                        C, p,
                        Bi, k_main_i, add_implicit_features,
                        (Xfull == NULL)?
                            (Xones) : (Xones + (do_B? ix : (ix*(size_t)(n)))),
                        (Xfull == NULL)? ((int_t)1) : (do_B? ldXones : (int_t)1),
                        (real_t*)NULL,
                        lam, w_user, w_implicit, lam_last,
                        bufferBtB,
                        (Xfull == NULL)? (int_t)0 : cnt_NA_x[ix],
                        bufferCtC,
                        (U == NULL)? (int_t)0 : cnt_NA_u[ix],
                        (real_t*)NULL, n,
                        bufferBiTBi,
                        true, true,
                        use_cg,
                        is_first_iter?
                            max_cg_steps
                            : k_pred, /* <- more steps to reach optimum */
                        nonneg, max_cd_steps,
                        (A_prev == NULL)?
                            ((real_t*)NULL) : (A_prev + ix*(size_t)lda),
                        buffer_remainder
                          + (size_buffer*(size_t)omp_get_thread_num())
                    );
                }
            }
        }
    }


    /* General case - construct one-by-one, use precomputed matrices
       when beneficial, determined on a case-by-case basis. */
    else
    {
        bool prefer_BtB = true;
        bool prefer_CtC = true;
        if (Xfull == NULL && Xcsr_p != NULL && !NA_as_zero_X)
            prefer_BtB = false;
        if (Xfull != NULL && weight != NULL)
            prefer_BtB = false;
        if (U == NULL && U_csr_p != NULL && !NA_as_zero_U)
            prefer_CtC = false;

        real_t *restrict bufferBtB = NULL;
        real_t *restrict bufferBeTBeChol = NULL;
        real_t *restrict bufferCtC = NULL;
        real_t *restrict bufferBiTBi = NULL;

        if (add_implicit_features)
        {
            if (precomputedBiTBi == NULL) {
                bufferBiTBi = buffer_real_t;
                buffer_real_t += square(k+k_main_i);
            }
            else {
                bufferBiTBi = precomputedBiTBi;
            }
        }

        if (prefer_BtB)
        {
            if (precomputedBtB != NULL)
                bufferBtB = precomputedBtB;
            else {
                bufferBtB = buffer_real_t;
                buffer_real_t += square(k+k_main);
            }
            
            if (!nonneg
                    &&
                prefer_CtC
                    &&
                (NA_as_zero_X || ((Xfull != NULL && weight == NULL) &&
                                  (near_dense || full_dense))    )
                    &&
                ((U != NULL && (near_dense_u || full_dense_u)) ||
                 (U == NULL && NA_as_zero_U) || (!p && U_csr_p == NULL)))
            {
                if (precomputedBeTBeChol != NULL)
                    bufferBeTBeChol = precomputedBeTBeChol;
                else {
                    bufferBeTBeChol = buffer_real_t;
                    buffer_real_t += square(k_user+k+k_main);
                }
            }
        }
        if (prefer_CtC && p)
        {
            if (precomputedCtCw != NULL)
                bufferCtC = precomputedCtCw;
            else {
                bufferCtC = buffer_real_t;
                buffer_real_t += square(k_user+k);
            }
        }

        real_t *restrict bufferX = buffer_real_t;
        real_t *restrict bufferX_zeros = bufferX + (do_B?
                                                   ((size_t)n*(size_t)nthreads)
                                                   : ((size_t)0));
        real_t *restrict bufferX_orig = bufferX;
        real_t *restrict bufferW = bufferX_zeros + ((do_B && weight != NULL)?
                                                    (n) : (0));
        real_t *restrict buffer_remainder = bufferW + (
                                                (do_B && weight != NULL)?
                                                ((size_t)n*(size_t)nthreads)
                                                : ((size_t)0));
        if (weight == NULL) bufferW = NULL;
        bool add_X = true;
        bool add_U = true;

        if (*filled_CtCw && w_user != 1. && !use_cg && p)
            cblas_tscal(square(k_totC), w_user, bufferCtC, 1);

        build_BtB_CtC(
            *filled_BtB? ((real_t*)NULL) : bufferBtB,
            *filled_CtCw? ((real_t*)NULL) : bufferCtC,
            B, n, ldb,
            C, p,
            k, k_user, k_main, k_item,
            use_cg? 1. : w_user,
            (use_cg && NA_as_zero_X && Xfull == NULL && weight != NULL)?
                ((real_t*)NULL) : (NA_as_zero_X? (real_t*)NULL : weight)
        );
        if (bufferBtB == precomputedBtB) *filled_BtB = true;
        if (bufferCtC == precomputedCtCw) *filled_CtCw = true;

        if (add_implicit_features && !filled_BiTBi)
        {
            cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        k+k_main_i, n,
                        w_implicit, Bi, k+k_main_i,
                        0., bufferBiTBi, k+k_main_i);
        }

        if (bufferBeTBeChol != NULL)
        {
            if (k_user || k_main)
                set_to_zero(bufferBeTBeChol, square(k_totA));
            if (p)
                copy_mat(k_user+k, k_user+k,
                         bufferCtC, k_user+k,
                         bufferBeTBeChol, k_totA);
            if (use_cg && w_user != 1. && p)
                cblas_tscal(square(k_totA) - k_main - k_main*k_totA, w_user,
                            bufferBeTBeChol, 1);
            sum_mat(k+k_main, k+k_main,
                    bufferBtB, k+k_main,
                    bufferBeTBeChol + offset_square, k_totA);
            if (add_implicit_features)
                sum_mat(k+k_main_i, k+k_main_i,
                        bufferBiTBi, k+k_main_i,
                        bufferBeTBeChol + offset_square, k_totA);
            add_to_diag(bufferBeTBeChol, lam, k_totA);
            if (lam_last != lam)
                bufferBeTBeChol[square(k_totA)-1] += (lam_last - lam);
            tpotrf_(&lo, &k_totA, bufferBeTBeChol, &k_totA, &ignore);
            if (bufferBeTBeChol == precomputedBeTBeChol)
                *filled_BeTBeChol = true;
        }

        if (add_implicit_features && use_cg && w_implicit != 1.)
            cblas_tscal(square(k+k_main_i), 1./w_implicit, bufferBiTBi, 1);

        if (use_cg) goto skip_chol_simplifications;

        if (Xfull != NULL && (full_dense || near_dense) && weight == NULL)
        {
            add_X = false;
            build_XBw(
                A + k_user, lda,
                B + k_item, ldb,
                Xfull, ldX,
                m_x, n, k + k_main,
                1.,
                do_B, true
            );

            /* TODO: what's the point_t of this 'bufferX_zeros'? */
            if (!full_dense && do_B)
                set_to_zero(bufferX_zeros, n); /*still needs a placeholder*/
        }

        else if (Xfull == NULL && weight == NULL && NA_as_zero_X) {
            add_X = false;
            tgemm_sp_dense(
                m_x, k+k_main, 1.,
                Xcsr_p, Xcsr_i, Xcsr,
                B + k_item, (size_t)ldb,
                A + k_user, (size_t)lda,
                nthreads
            );
        }

        if (U != NULL && (full_dense_u || near_dense_u)) {
            add_U = false;
            cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m_u, k_user + k, p,
                        w_user, U, p, C, k_user + k,
                        add_X? 0. : 1., A, lda);
        }

        else if (U == NULL && NA_as_zero_U) {
            add_U = false;
            tgemm_sp_dense(
                m_u, k_user+k, w_user,
                U_csr_p, U_csr_i, U_csr,
                C, (size_t)k_totC,
                A, (size_t)lda,
                nthreads
            );
        }

        if (add_implicit_features && !add_X)
        {
            if (Xfull != NULL)
                build_XBw(
                    A + k_user, k_totA,
                    Bi, k+k_main_i,
                    Xones, ldXones,
                    m_x, n, k+k_main_i,
                    w_implicit,
                    do_B, false
                );
            else
                tgemm_sp_dense(
                    m_x, k+k_main_i, w_implicit,
                    Xcsr_p, Xcsr_i, Xones,
                    Bi, k+k_main_i,
                    A + k_user, (size_t)lda,
                    nthreads
                );
        }

        skip_chol_simplifications:
            {};
        size_t size_buffer = use_cg? (3*k_totA) : (square(k_totA));
        if (use_cg && Xfull == NULL && NA_as_zero_X && weight != NULL)
            size_buffer += n;

        if (!p && U_csr_p == NULL) {
           if (!use_cg) add_U = false;
        }

        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
                shared(A, k_totA, B, C, Bi, k, k_user, k_item, k_main,k_main_i,\
                       m, m_x, m_u, n, p, lam, lam_last, w_user, w_implicit, \
                       NA_as_zero_X, NA_as_zero_U, add_implicit_features, \
                       add_X, add_U, weight, \
                       Xfull, Xcsr_p, Xcsr_i, Xcsr, cnt_NA_x, ldX, \
                       U, U_csr_p, U_csr_i, U_csr, cnt_NA_u, \
                       bufferBtB, bufferCtC, bufferBiTBi, bufferBeTBeChol, \
                       buffer_remainder, size_buffer, \
                       do_B, nthreads, use_cg, nonneg, max_cd_steps, A_prev) \
                firstprivate(bufferX, bufferW)
        for (size_t_for ix = 0; ix < (size_t)m; ix++)
        {
            /* TODO: do away with the 'bufferX', replace it instead with an
               'incX' parameter */
            if (Xfull != NULL)
            {
                if (!do_B)
                    bufferX = Xfull + ix*(size_t)n;
                else if (add_X || cnt_NA_x[ix] || (U != NULL && cnt_NA_u[ix]))
                {
                    cblas_tcopy(n, Xfull + ix, ldX,
                                bufferX_orig
                                    + (size_t)n*(size_t)omp_get_thread_num(),1);
                    bufferX = bufferX_orig;
                }
                else
                    bufferX = bufferX_zeros;

                if (weight != NULL) {
                    if (!do_B)
                        bufferW = weight + ix*(size_t)n;
                    else
                        cblas_tcopy(n, weight + ix, ldX,
                                    bufferW
                                    + (size_t)n*(size_t)omp_get_thread_num(),1);
                }
            }


            collective_closed_form_block(
                A + ix*(size_t)lda,
                k, k_user, k_item, k_main,
                (Xfull == NULL)? ((real_t*)NULL)
                                 : (bufferX
                                    + (do_B?
                                        ((size_t)n*(size_t)omp_get_thread_num())
                                            :
                                        ((size_t)0))),
                (Xfull != NULL)? ((real_t*)NULL) : (Xcsr + Xcsr_p[ix]),
                (Xfull != NULL)? ((int_t*)NULL) : (Xcsr_i + Xcsr_p[ix]),
                (Xfull != NULL)? (size_t)0 : (Xcsr_p[ix+(size_t)1] -Xcsr_p[ix]),
                (U != NULL || U_csr_p == NULL)?
                    ((int_t*)NULL) : (U_csr_i + U_csr_p[ix]),
                (U != NULL || U_csr_p == NULL)?
                    ((real_t*)NULL) : (U_csr + U_csr_p[ix]),
                (U != NULL || U_csr_p == NULL)?
                    (size_t)0 : (U_csr_p[ix+(size_t)1] - U_csr_p[ix]),
                (U == NULL)? ((real_t*)NULL) : (U + ix*(size_t)p),
                NA_as_zero_X, NA_as_zero_U,
                B, n, ldb,
                C, p,
                Bi, k_main_i, add_implicit_features,
                (Xfull == NULL)?
                    (Xones) : (Xones + (do_B? ix : (ix*(size_t)n))),
                (Xfull == NULL)? ((int_t)1) : (do_B? ldXones : (int_t)1),
                (weight == NULL)? ((real_t*)NULL)
                                : ( (Xfull != NULL)?
                                      (bufferW
                                       + (do_B?
                                    ((size_t)n*(size_t)omp_get_thread_num())
                                        : ((size_t)0)))
                                      : (weight + Xcsr_p[ix]) ),
                lam, w_user, w_implicit, lam_last,
                bufferBtB, (Xfull != NULL)? cnt_NA_x[ix] : (int_t)0,
                bufferCtC,
                (U == NULL)? (int_t)0 : cnt_NA_u[ix],
                bufferBeTBeChol, n,
                bufferBiTBi,
                (Xfull == NULL)? (add_X) : (add_X || cnt_NA_x[ix] > 0),
                (U == NULL)? (add_U) : (add_U || cnt_NA_u[ix] > 0),
                use_cg, max_cg_steps,
                nonneg, max_cd_steps,
                (A_prev == NULL)?
                    ((real_t*)NULL) : (A_prev + ix*(size_t)lda),
                buffer_remainder + (size_buffer*(size_t)omp_get_thread_num())
            );
        }

    }
}

void optimizeA_collective_implicit
(
    real_t *restrict A, real_t *restrict B, real_t *restrict C,
    int_t m, int_t m_u, int_t n, int_t p,
    int_t k, int_t k_main, int_t k_user, int_t k_item,
    size_t Xcsr_p[], int_t Xcsr_i[], real_t *restrict Xcsr,
    size_t U_csr_p[], int_t U_csr_i[], real_t *restrict U_csr,
    real_t *restrict U, int_t cnt_NA_u[],
    bool full_dense_u, bool near_dense_u, bool NA_as_zero_U,
    real_t lam, real_t w_user,
    int_t nthreads,
    bool use_cg, int_t max_cg_steps, bool is_first_iter,
    bool nonneg, int_t max_cd_steps, real_t *restrict A_prev,
    real_t *restrict precomputedBtB, /* will not have lambda with CG */
    real_t *restrict precomputedBeTBe,
    real_t *restrict precomputedBeTBeChol,
    real_t *restrict precomputedCtC,
    bool *filled_BeTBe,
    bool *filled_BeTBeChol,
    bool *filled_CtC,
    real_t *restrict buffer_real_t
)
{
    int_t k_totA = k_user + k + k_main;
    int_t k_totB = k_item + k + k_main;
    int_t k_totC = k_user + k;
    int_t ld_BtB = k + k_main;
    int_t m_x = m; /* <- 'm' later gets overwritten */

    *filled_BeTBe = false;
    *filled_BeTBeChol = false;
    *filled_CtC = false;

    int_t ix = 0;


    #ifdef TEST_CG
    use_cg = true;
    max_cg_steps = 10000;
    is_first_iter = true;
    #endif

    if (!use_cg || is_first_iter)
        set_to_zero_(A, (size_t)max2(m, m_u)*(size_t)k_totA, nthreads);
    if (nonneg)
        set_to_zero_(A_prev, (size_t)max2(m, m_u)*(size_t)k_totA, nthreads);

    /* TODO: BtB can be skipped when using NA_as_zero_U */
    if (precomputedBtB == NULL)
    {
        precomputedBtB = buffer_real_t;
        buffer_real_t += square(ld_BtB);
    }


    /* TODO: should get rid of the tsymv, replacing them with tgemv as it's
       faster, by filling up the lower half of the precomputed matrices. */

    /* If the X matrix has more rows, the extra rows will be independent
       from U and can be obtained from the single-matrix formula instead.
       However, if the U matrix has more rows, those still need to be
       considered as having a value of zero in X. */
    if (m > m_u)
    {
        int_t m_diff = m - m_u;
        if (Xcsr_p[m_u] < Xcsr_p[m])
            optimizeA_implicit(
                A + (size_t)k_user + (size_t)m_u*(size_t)k_totA, (size_t)k_totA,
                B + k_item, (size_t)k_totB,
                m_diff, n, k + k_main,
                Xcsr_p + m_u, Xcsr_i, Xcsr,
                lam,
                nthreads, use_cg, max_cg_steps, false,
                nonneg, max_cd_steps,
                (A_prev == NULL)?
                    ((real_t*)NULL)
                        :
                    (A_prev + (size_t)k_user + (size_t)m_u*(size_t)k_totA),
                precomputedBtB,
                buffer_real_t
            );
        m = m_u;
    }

    else {
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k+k_main, n,
                    1., B + k_item, k_totB,
                    0., precomputedBtB, ld_BtB);
        if (!use_cg)
            add_to_diag(precomputedBtB, lam, ld_BtB);
    }


    if (m_u > m_x && !nonneg &&
        !(U == NULL && U_csr_p != NULL && !NA_as_zero_U) &&
        (!use_cg || p > 2 * k_totA))
    {
        if (precomputedBeTBeChol == NULL)
        {
            precomputedBeTBeChol = buffer_real_t;
            buffer_real_t += square(k_totA);
        }
    }
    else {
        precomputedBeTBeChol = NULL;
    }

    bool prefer_CtC = !(U == NULL && U_csr_p != NULL && !NA_as_zero_U);
    if (use_cg && prefer_CtC)
    {
        if (precomputedCtC == NULL)
        {
            precomputedCtC = buffer_real_t;
            buffer_real_t += square(k_user+k);
        }
    }
    else {
        precomputedCtC = NULL;
    }

    if (use_cg && prefer_CtC)
    {
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k_user+k, p,
                    1., C, k_user+k,
                    0., precomputedCtC, k_user+k);
        *filled_CtC = true;
    }

    /* Lower-right square of Be */
    if (!use_cg || precomputedBeTBeChol != NULL)
    {
        if (precomputedBeTBe == NULL)
        {
            precomputedBeTBe = buffer_real_t;
            buffer_real_t += square(k_totA);
        }
    }
    else {
        precomputedBeTBe = NULL;
    }
    real_t *restrict buffer_remainder = buffer_real_t;

    if (precomputedBeTBe != NULL)
    {
        if (ld_BtB != k_totA)
            set_to_zero(precomputedBeTBe, square(k_totA));
        copy_mat(k+k_main, k+k_main,
                 precomputedBtB, k+k_main,
                 precomputedBeTBe + k_user + k_user*k_totA, k_totA);
        if (use_cg)
            add_to_diag(precomputedBeTBe, lam, k_totA);
        else
            for (int_t ix = 0; ix < k_user; ix++)
                precomputedBeTBe[ix + ix*k_totA] += lam;
        *filled_BeTBe = true;
    }

    /* Upper-left square of Be if possible */
    if (precomputedBeTBe != NULL && (U != NULL || NA_as_zero_U))
    {
        if (precomputedCtC != NULL)
        {
            if (w_user == 1.)
                sum_mat(
                    k_user+k, k_user+k,
                    precomputedCtC, k_user+k,
                    precomputedBeTBe, k_totA
                );
            else {
                for (size_t row = 0; row < (size_t)(k_user+k); row++)
                    for (size_t col = 0; col < (size_t)(k_user+k); col++)
                        precomputedBeTBe[col + row*(size_t)k_totA]
                            +=
                        w_user * precomputedCtC[col + row*(size_t)k_totC];
            }
        }
        else
            cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        k_user+k, p,
                        w_user, C, k_totC,
                        1., precomputedBeTBe, k_totA);
    }

    /* Lower half of Xe (reuse if possible) */
    if (!use_cg)
    {
        if (U != NULL) {
            cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m_u, k_user + k, p,
                        w_user, U, p, C, k_totC,
                        0., A, k_totA);
        }
        else {
            tgemm_sp_dense(
                m_u, k_user + k, w_user,
                U_csr_p, U_csr_i, U_csr,
                C, k_totC,
                A, k_totA,
                nthreads
            );
        }
    }

    /* If there are no positive entries for some X and no missing values
       in U, can reuse a single Cholesky factorization for them. */
    if (precomputedBeTBeChol != NULL)
    {
        copy_arr(precomputedBeTBe, precomputedBeTBeChol, square(k_totA));
        char lo = 'L';
        tpotrf_(&lo, &k_totA, precomputedBeTBeChol, &k_totA, &ix);
        *filled_BeTBeChol = true;
    }

    m = max2(m, m_u);

    size_t size_buffer = use_cg? (3 * k_totA) : (square(k_totA));

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
            shared(A, B, C, m, n, p, k, k_user, k_item, k_main, lam, \
                   Xcsr, Xcsr_p, Xcsr_i, U, U_csr, U_csr_i, U_csr_p, \
                   NA_as_zero_U, cnt_NA_u, \
                   precomputedBeTBe, precomputedBtB, precomputedBeTBeChol, \
                   k_totA, buffer_remainder, use_cg, m_x, \
                   nonneg, max_cd_steps, A_prev)
    for (ix = 0; ix < m; ix++)
        collective_closed_form_block_implicit(
            A + (size_t)ix*(size_t)k_totA,
            k, k_user, k_item, k_main,
            B, n, C, p,
            (ix < m_x)? (Xcsr + Xcsr_p[ix]) : ((real_t*)NULL),
            (ix < m_x)? (Xcsr_i + Xcsr_p[ix]) : ((int_t*)NULL),
            (ix < m_x)? (Xcsr_p[ix+(size_t)1] - Xcsr_p[ix]) : ((size_t)0),
            (U == NULL)? ((real_t*)NULL) : (U + (size_t)ix*(size_t)p),
            (U == NULL)? (0) : (cnt_NA_u[ix]),
            (U == NULL)? (U_csr + U_csr_p[ix]) : ((real_t*)NULL),
            (U == NULL)? (U_csr_i + U_csr_p[ix]) : ((int_t*)NULL),
            (U == NULL)? (U_csr_p[ix+(size_t)1] - U_csr_p[ix]) : ((size_t)0),
            NA_as_zero_U,
            lam, w_user,
            precomputedBeTBe,
            precomputedBtB,
            precomputedBeTBeChol,
            precomputedCtC,
            false, true, use_cg, max_cg_steps,
            nonneg, max_cd_steps,
            (A_prev == NULL)?
                ((real_t*)NULL) : (A_prev + (size_t)ix*(size_t)k_totA),
            buffer_remainder + ((size_t)omp_get_thread_num() * size_buffer)
        );

}

void build_BeTBe
(
    real_t *restrict bufferBeTBe,
    real_t *restrict B, int_t ldb, real_t *restrict C,
    int_t k, int_t k_user, int_t k_main, int_t k_item,
    int_t n, int_t p,
    real_t lam, real_t w_user
)
{
    int_t k_totA = k_user + k + k_main;
    set_to_zero(bufferBeTBe, square(k_totA));
    if (p)
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k_user + k, p,
                    w_user, C, k_user + k,
                    0., bufferBeTBe, k_totA);
    cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                k + k_main, n,
                1., B + k_item, ldb,
                1., bufferBeTBe + k_user + k_user*k_totA, k_totA);
    add_to_diag(bufferBeTBe, lam, k_totA);
}

void build_BtB_CtC
(
    real_t *restrict BtB, real_t *restrict CtC,
    real_t *restrict B, int_t n, int_t ldb,
    real_t *restrict C, int_t p,
    int_t k, int_t k_user, int_t k_main, int_t k_item,
    real_t w_user,
    real_t *restrict weight
)
{
    if (weight == NULL && BtB != NULL) {
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k + k_main, n,
                    1., B + k_item, ldb,
                    0., BtB, k+k_main);
    }
    if (CtC != NULL && p)
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k_user + k, p,
                    w_user, C, k_user + k,
                    0., CtC, k_user + k);
}

void build_XBw
(
    real_t *restrict A, int_t lda,
    real_t *restrict B, int_t ldb,
    real_t *restrict Xfull, int_t ldX,
    int_t m, int_t n, int_t k,
    real_t w,
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
                    w, Xfull, ldX, B, ldb,
                    overwrite? 0. : 1., A, lda);
}

void preprocess_vec
(
    real_t *restrict vec_full, int_t n,
    int_t *restrict ix_vec, real_t *restrict vec_sp, size_t nnz,
    real_t glob_mean, real_t lam,
    real_t *restrict col_means,
    real_t *restrict vec_mean,
    int_t *restrict cnt_NA
)
{
    if (col_means != NULL)
    {
        if (vec_full != NULL) {
            for (int_t ix = 0; ix < n; ix++)
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
            for (int_t ix = 0; ix < n; ix++)
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
                for (int_t ix = 0; ix < n; ix++) {
                    *vec_mean += (!isnan(vec_full[ix]))? vec_full[ix] : 0;
                }
                *vec_mean /= ((double)(n - *cnt_NA) + lam);
            }

            else {
                for (int_t ix = 0; ix < n; ix++)
                    *vec_mean += vec_full[ix];
                *vec_mean /= ((double)n + lam);
            }
            
            for (int_t ix = 0; ix < n; ix++)
                vec_full[ix] -= *vec_mean;
        }

    }
}

int_t convert_sparse_X
(
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    size_t **Xcsr_p, int_t **Xcsr_i, real_t *restrict *Xcsr,
    size_t **Xcsc_p, int_t **Xcsc_i, real_t *restrict *Xcsc,
    real_t *restrict weight,
    real_t *restrict *weightR, real_t *restrict *weightC,
    int_t m, int_t n, int_t nthreads
)
{
    *Xcsr_p = (size_t*)malloc(((size_t)m+(size_t)1)*sizeof(size_t));
    *Xcsr_i = (int_t*)malloc(nnz*sizeof(int_t));
    *Xcsr = (real_t*)malloc(nnz*sizeof(real_t));
    if (weight != NULL)
        *weightR = (real_t*)malloc(nnz*sizeof(real_t));
    *Xcsc_p = (size_t*)malloc(((size_t)n+(size_t)1)*sizeof(size_t));
    *Xcsc_i = (int_t*)malloc(nnz*sizeof(int_t));
    *Xcsc = (real_t*)malloc(nnz*sizeof(real_t));
    if (weight != NULL)
        *weightC = (real_t*)malloc(nnz*sizeof(real_t));
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

int_t preprocess_sideinfo_matrix
(
    real_t *U, int_t m_u, int_t p,
    int_t U_row[], int_t U_col[], real_t *U_sp, size_t nnz_U,
    real_t *U_colmeans, real_t *restrict *Utrans,
    size_t **U_csr_p, int_t **U_csr_i, real_t *restrict *U_csr,
    size_t **U_csc_p, int_t **U_csc_i, real_t *restrict *U_csc,
    int_t *restrict *cnt_NA_u_byrow, int_t *restrict *cnt_NA_u_bycol,
    bool *full_dense_u, bool *near_dense_u_row, bool *near_dense_u_col,
    bool NA_as_zero_U, bool nonneg, int_t nthreads
)
{
    int_t retval = 0;

    *full_dense_u = false;
    *near_dense_u_row = false;
    *near_dense_u_col = false;
    if (U != NULL)
    {
        *cnt_NA_u_byrow = (int_t*)calloc(m_u, sizeof(int_t));
        *cnt_NA_u_bycol = (int_t*)calloc(p, sizeof(int_t));
        if (*cnt_NA_u_byrow == NULL || *cnt_NA_u_bycol == NULL)
            return 1;
        count_NAs_by_row(U, m_u, p, *cnt_NA_u_byrow, nthreads,
                         full_dense_u, near_dense_u_row);
        count_NAs_by_col(U, m_u, p, *cnt_NA_u_bycol,
                         full_dense_u, near_dense_u_col);
    }

    if ((U != NULL || !NA_as_zero_U) && U_colmeans != NULL && !nonneg)
        retval = center_by_cols(
            U_colmeans,
            U, m_u, p,
            U_row, U_col, U_sp, nnz_U,
            *U_csr_p, *U_csr_i, *U_csr,
            *U_csc_p, *U_csc_i, *U_csc,
            nthreads
        );
    if (retval != 0) return 1;

    else
    {
        *U_csr_p = (size_t*)malloc(((size_t)m_u+(size_t)1)*sizeof(size_t));
        *U_csr_i = (int_t*)malloc(nnz_U*sizeof(int_t));
        *U_csr = (real_t*)malloc(nnz_U*sizeof(real_t));
        *U_csc_p = (size_t*)malloc(((size_t)p+(size_t)1)*sizeof(size_t));
        *U_csc_i = (int_t*)malloc(nnz_U*sizeof(int_t));
        *U_csc = (real_t*)malloc(nnz_U*sizeof(real_t));
        if (*U_csr_p == NULL || *U_csr_i == NULL || *U_csr == NULL ||
            *U_csc_p == NULL || *U_csc_i == NULL || *U_csc == NULL)
            return 1;
        coo_to_csr_and_csc(
            U_row, U_col, U_sp,
            (real_t*)NULL, m_u, p, nnz_U,
            *U_csr_p, *U_csr_i, *U_csr,
            *U_csc_p, *U_csc_i, *U_csc,
            (real_t*)NULL, (real_t*)NULL,
            nthreads
        );
    }

    if (U != NULL && Utrans != NULL && !full_dense_u && !near_dense_u_col)
    {
        *Utrans = (real_t*)malloc((size_t)m_u*(size_t)p*sizeof(real_t));
        if (*Utrans == NULL)
            return 1;
        transpose_mat2(U, m_u, p, *Utrans);
    }

    if ((NA_as_zero_U || nonneg) && U_colmeans != NULL)
        set_to_zero(U_colmeans, p);

    return 0;
}


real_t wrapper_collective_fun_grad
(
    void *instance,
    real_t *x,
    real_t *g,
    const size_t n,
    const real_t step
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
            data->buffer_real_t, data->buffer_mt,
            data->k_main, data->k_user, data->k_item,
            data->w_main, data->w_user, data->w_item,
            data->nthreads
        );
}

int_t fit_collective_explicit_lbfgs_internal
(
    real_t *restrict values, bool reset_values,
    real_t *restrict glob_mean,
    real_t *restrict U_colmeans, real_t *restrict I_colmeans,
    int_t m, int_t n, int_t k,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t *restrict Xfull,
    real_t *restrict weight,
    bool user_bias, bool item_bias,
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
    int_t nthreads, bool prefer_onepass,
    bool verbose, int_t print_every, bool handle_interrupt,
    int_t *restrict niter, int_t *restrict nfev,
    real_t *restrict B_plus_bias
)
{
    real_t *restrict buffer_real_t = NULL;
    real_t *restrict buffer_mt = NULL;
    int_t retval = 0;

    size_t nvars, size_buffer, size_mt;
    nvars_collective_fun_grad(
        m, n, m_u, n_i, m_ubin, n_ibin,
        p, q, pbin, qbin,
        nnz, nnz_U, nnz_I,
        k, k_main, k_user, k_item,
        user_bias, item_bias, nthreads,
        X, Xfull,
        U, Ub, II, Ib,
        U_sp, U_sp, I_sp, I_sp,
        &nvars, &size_buffer, &size_mt
    );

    if (size_buffer) {
        buffer_real_t = (real_t*)malloc(size_buffer*sizeof(real_t));
        if (buffer_real_t == NULL) return 1;
    }

    int_t m_max = max2(max2(m, m_u), m_ubin);
    int_t n_max = max2(max2(n, n_i), n_ibin);

    bool U_has_NA = false;
    bool I_has_NA = false;
    bool Ub_has_NA = false;
    bool Ib_has_NA = false;

    real_t funval;
    lbfgs_parameter_t lbfgs_params;
    data_collective_fun_grad data;

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

    #ifdef _FOR_R
    if (Xfull != NULL) R_nan_to_C_nan(Xfull, (size_t)m*(size_t)n);
    if (U != NULL) R_nan_to_C_nan(U, (size_t)m_u*(size_t)p);
    if (II != NULL) R_nan_to_C_nan(II, (size_t)n_i*(size_t)q);
    if (Ub != NULL) R_nan_to_C_nan(Ub, (size_t)m_ubin*(size_t)pbin);
    if (Ib != NULL) R_nan_to_C_nan(Ib, (size_t)n_ibin*(size_t)qbin);
    #endif

    #ifdef _OPENMP
    bool ignore = false;
    if (nthreads > 1 && (Xfull == NULL || U_sp != NULL || I_sp != NULL))
    {
        if (prefer_onepass)
        {
            buffer_mt = (real_t*)malloc(size_mt*sizeof(real_t));
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
                    (real_t*)NULL, m_u, p,
                    U_row, U_col, U_sp, nnz_U,
                    (real_t*)NULL, (real_t**)NULL,
                    &U_csr_p, &U_csr_i, &U_csr,
                    &U_csc_p, &U_csc_i, &U_csc,
                    (int_t**)NULL, (int_t**)NULL,
                    &ignore, &ignore, &ignore,
                    false, false, nthreads
                );
                if (retval != 0) goto cleanup;
            }

            if (I_sp != NULL)
            {
                retval = preprocess_sideinfo_matrix(
                    (real_t*)NULL, n_i, q,
                    I_row, I_col, I_sp, nnz_I,
                    (real_t*)NULL, (real_t**)NULL,
                    &I_csr_p, &I_csr_i, &I_csr,
                    &I_csc_p, &I_csc_i, &I_csc,
                    (int_t**)NULL, (int_t**)NULL,
                    &ignore, &ignore, &ignore,
                    false, false, nthreads
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
        Xfull, (real_t*)NULL,
        Xcsr_p, Xcsr_i, Xcsr,
        Xcsc_p, Xcsc_i, Xcsc,
        false,
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
        buffer_real_t, buffer_mt,
        k_main, k_user, k_item,
        w_main, w_user, w_item,
        nthreads, print_every, 0, 0,
        handle_interrupt
    };

    if (handle_interrupt)
        signal(SIGINT, set_interrup_global_variable);
    if (should_stop_procedure)
    {
        should_stop_procedure = false;
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
        wrapper_collective_fun_grad,
        (verbose)? (lbfgs_printer_collective) : (NULL),
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
    if (retval == 1)
    {
        if (verbose)
            print_oom_message();
        return 1;
    }

    return 0;
}

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
    bool user_bias, bool item_bias,
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
    int_t nthreads, bool prefer_onepass,
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
)
{
    int_t retval = 0;

    int_t k_totA = k_user + k + k_main;
    int_t k_totB = k_item + k + k_main;
    int_t m_max = max2(max2(m, m_u), m_ubin);
    int_t n_max = max2(max2(n, n_i), n_ibin);
    size_t nvars, ignored, ignored2 = 0;
    nvars_collective_fun_grad(
        (size_t)m, (size_t)n, (size_t)m_u, (size_t)n_i,
        (size_t)m_ubin, (size_t)n_ibin,
        (size_t)p, (size_t)q, (size_t)pbin, (size_t)qbin,
        nnz, nnz_U, nnz_I,
        (size_t)k, (size_t)k_main, (size_t)k_user, (size_t)k_item,
        user_bias, item_bias, (size_t)nthreads,
        X, Xfull,
        U, Ub, II, Ib,
        U_sp, U_sp, I_sp, I_sp,
        &nvars, &ignored, &ignored2
    );
    size_t edge = 0;
    real_t *restrict values = (real_t*)malloc(nvars*sizeof(real_t));
    if (values == NULL) goto throw_oom;
    if (!reset_values)
    {
        edge = 0;
        if (user_bias) {
            copy_arr(biasA, values + edge, m_max);
            edge += m_max;
        }
        if (item_bias) {
            copy_arr(biasB, values + edge, n_max);
            edge += n_max;
        }
        copy_arr_(A, values + edge, (size_t)m_max*(size_t)k_totA, nthreads);
        edge += (size_t)m_max*(size_t)k_totA;
        copy_arr_(B, values + edge, (size_t)n_max*(size_t)k_totB, nthreads);
        edge += (size_t)n_max*(size_t)k_totB;
        if (p) {
            copy_arr_(C, values + edge, (size_t)p*(size_t)(k_user+k), nthreads);
            edge += (size_t)p*(size_t)(k_user+k);
        }
        if (pbin) {
            copy_arr_(Cb, values+edge,(size_t)pbin*(size_t)(k_user+k),nthreads);
            edge += (size_t)pbin*(size_t)(k_user+k);
        }
        if (q) {
            copy_arr_(D, values + edge, (size_t)q*(size_t)(k_item+k), nthreads);
            edge += (size_t)q*(size_t)(k_item+k);
        }
        if (qbin) {
            copy_arr_(Db, values+edge,(size_t)qbin*(size_t)(k_item+k),nthreads);
            edge += (size_t)qbin*(size_t)(k_item+k);
        }
    }

    retval = fit_collective_explicit_lbfgs_internal(
        values, reset_values,
        glob_mean,
        U_colmeans, I_colmeans,
        m, n, k,
        ixA, ixB, X, nnz,
        Xfull,
        weight,
        user_bias, item_bias,
        lam, lam_unique,
        U, m_u, p,
        II, n_i, q,
        Ub, m_ubin, pbin,
        Ib, n_ibin, qbin,
        U_row, U_col, U_sp, nnz_U,
        I_row, I_col, I_sp, nnz_I,
        k_main, k_user, k_item,
        w_main, w_user, w_item,
        n_corr_pairs, maxiter, seed,
        nthreads, prefer_onepass,
        verbose, print_every, handle_interrupt,
        niter, nfev,
        B_plus_bias
    );
    if (retval != 0)
        goto cleanup;


    if (true)
    {
        edge = 0;
        if (user_bias) {
            copy_arr(values + edge, biasA, m_max);
            edge += m_max;
        }
        if (item_bias) {
            copy_arr(values + edge, biasB, n_max);
            edge += n_max;
        }
        copy_arr_(values + edge, A, (size_t)m_max*(size_t)k_totA, nthreads);
        edge += (size_t)m_max*(size_t)k_totA;
        copy_arr_(values + edge, B, (size_t)n_max*(size_t)k_totB, nthreads);
        edge += (size_t)n_max*(size_t)k_totB;
        if (p) {
            copy_arr_(values + edge, C, (size_t)p*(size_t)(k_user+k), nthreads);
            edge += (size_t)p*(size_t)(k_user+k);
        }
        if (pbin) {
            copy_arr_(values+edge,Cb, (size_t)pbin*(size_t)(k_user+k),nthreads);
            edge += (size_t)pbin*(size_t)(k_user+k);
        }
        if (q) {
            copy_arr_(values + edge, D, (size_t)q*(size_t)(k_item+k), nthreads);
            edge += (size_t)q*(size_t)(k_item+k);
        }
        if (qbin) {
            copy_arr_(values+edge,Db, (size_t)qbin*(size_t)(k_item+k),nthreads);
            edge += (size_t)qbin*(size_t)(k_item+k);
        }
    }

    if (precompute_for_predictions)
    {
        retval = precompute_collective_explicit(
            B, n, n_max, include_all_X,
            C, p,
            (real_t*)NULL, false,
            k, k_user, k_item, k_main,
            user_bias,
            false,
            lam, lam_unique,
            w_main, w_user, 1.,
            B_plus_bias,
            precomputedBtB,
            precomputedTransBtBinvBt,
            precomputedBeTBeChol,
            (real_t*)NULL,
            precomputedTransCtCinvCt,
            precomputedCtCw
        );
    }

    cleanup:
        free(values);
        return retval;
    throw_oom:
    {
        if (verbose)
            print_oom_message();
        retval = 1;
        goto cleanup;
    }
}

/* TODO: it's no longer necessary or beneficial to have separate functions
   for 'optimizeA' - could instead make calls only to 'optimizeA_collective',
   and can also replace the 'optimizeA_implicit' with calls to
   'optimizeA' + 'NA_as_zero_X', so as to simplify the code. */
/* TODO: should have the option of passing the matrices either in row-major
   or in column-major order, as it needs to have both in any case. */
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
    bool user_bias, bool item_bias,
    real_t lam, real_t *restrict lam_unique,
    real_t *restrict U, int_t m_u, int_t p,
    real_t *restrict II, int_t n_i, int_t q,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    int_t I_row[], int_t I_col[], real_t *restrict I_sp, size_t nnz_I,
    bool NA_as_zero_X, bool NA_as_zero_U, bool NA_as_zero_I,
    int_t k_main, int_t k_user, int_t k_item,
    real_t w_main, real_t w_user, real_t w_item, real_t w_implicit,
    int_t niter, int_t nthreads, bool verbose, bool handle_interrupt,
    bool use_cg, int_t max_cg_steps, bool finalize_chol,
    bool nonneg, int_t max_cd_steps, bool nonneg_C, bool nonneg_D,
    bool precompute_for_predictions,
    bool include_all_X,
    real_t *restrict B_plus_bias,
    real_t *restrict precomputedBtB,
    real_t *restrict precomputedTransBtBinvBt,
    real_t *restrict precomputedBeTBeChol,
    real_t *restrict precomputedBiTBi,
    real_t *restrict precomputedTransCtCinvCt,
    real_t *restrict precomputedCtCw
)
{
    int_t retval = 0;
    if ((NA_as_zero_X && Xfull == NULL) && (user_bias || item_bias)) {
        if (verbose)
            fprintf(stderr, "Cannot calculate biases with 'NA_as_zero_X'.\n");
        retval = 2;
    }

    if (k_user && U == NULL && nnz_U == 0) {
        if (verbose)
            fprintf(stderr, "Cannot pass 'k_user' without U data.\n");
        retval = 2;
    }

    if (k_item && II == NULL && nnz_I == 0) {
        if (verbose)
            fprintf(stderr, "Cannot pass 'k_item' without I data.\n");
        retval = 2;
    }

    if (k_main && Xfull == NULL && nnz == 0) {
        if (verbose)
            fprintf(stderr, "Cannot pass 'k_main' without X data.\n");
        retval = 2;
    }

    if ((Xfull != NULL && NA_as_zero_X) ||
        (U != NULL && NA_as_zero_U) ||
        (II != NULL && NA_as_zero_I))
    {
        if (verbose)
            fprintf(stderr, "Cannot pass 'NA_as_zero' with dense data.\n");
        retval = 2;
    }

    if (retval == 2)
    {
        if (verbose) {
            #ifndef _FOR_R
            fflush(stderr);
            #endif
        }
        return retval;
    }

    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long row, col, ix;
    #endif

    int_t k_totA = k_user + k + k_main;
    int_t k_totB = k_item + k + k_main;
    int_t has_bias = user_bias || item_bias;
    int_t m_max = max2(m, m_u);
    int_t n_max = max2(n, n_i);

    real_t *restrict A_bias = NULL;
    real_t *restrict B_bias = NULL;
    real_t *restrict Xcsr_orig = NULL;
    real_t *restrict Xcsc_orig = NULL;
    real_t *restrict Xfull_orig = NULL;
    real_t *restrict Xtrans_orig = NULL;

    real_t *restrict A_prev = NULL;
    real_t *restrict B_prev = NULL;
    real_t *restrict C_prev = NULL;
    real_t *restrict D_prev = NULL;
    real_t *restrict Ai_prev = NULL;
    real_t *restrict Bi_prev = NULL;

    real_t *restrict buffer_real_t = NULL;
    size_t size_bufferA = 0;
    size_t size_bufferB = 0;
    size_t size_bufferC = 0;
    size_t size_bufferD = 0;
    size_t size_bufferAi = 0;
    size_t size_bufferBi = 0;
    size_t size_buffer = 0;

    real_t *restrict Xtrans = NULL;
    real_t *restrict Wtrans = NULL;
    size_t *Xcsr_p = NULL;
    int_t *Xcsr_i = NULL;
    real_t *restrict Xcsr = NULL;
    real_t *restrict weightR = NULL;
    size_t *Xcsc_p = NULL;
    int_t *Xcsc_i = NULL;
    real_t *restrict Xcsc = NULL;
    real_t *restrict weightC = NULL;
    real_t *restrict Xones = NULL;
    real_t *restrict Utrans = NULL;
    size_t *U_csr_p = NULL;
    int_t *U_csr_i = NULL;
    real_t *restrict U_csr = NULL;
    size_t *U_csc_p = NULL;
    int_t *U_csc_i = NULL;
    real_t *restrict U_csc = NULL;
    real_t *restrict Itrans = NULL;
    size_t *I_csr_p = NULL;
    int_t *I_csr_i = NULL;
    real_t *restrict I_csr = NULL;
    size_t *I_csc_p = NULL;
    int_t *I_csc_i = NULL;
    real_t *restrict I_csc = NULL;
    int_t *restrict cnt_NA_byrow = NULL;
    int_t *restrict cnt_NA_bycol = NULL;
    int_t *restrict cnt_NA_u_byrow = NULL;
    int_t *restrict cnt_NA_u_bycol = NULL;
    int_t *restrict cnt_NA_i_byrow = NULL;
    int_t *restrict cnt_NA_i_bycol = NULL;
    int_t *restrict zeros_m = NULL;
    int_t *restrict zeros_n = NULL;
    bool full_dense = false;
    bool near_dense_row = false;
    bool near_dense_col = false;
    bool full_dense_u = false;
    bool near_dense_u_row = false;
    bool near_dense_u_col = false;
    bool full_dense_i = false;
    bool near_dense_i_row = false;
    bool near_dense_i_col = false;

    bool filled_BtB = false;
    bool filled_CtCw = false;
    bool filled_BeTBeChol = false;
    bool filled_BiTBi = false;
    bool ignore = false;
    bool back_to_precompute = false;

    bool finished_TransBtBinvBt = false;
    bool finished_TransCtCinvCt = false;
    char lo = 'L';
    int_t ignore_int = 0;
    int_t k_pred = 0;
    bool free_BiTBi = false;
    bool free_arr_use = false;
    real_t *arr_use = NULL;

    int_t *restrict seed_arr = NULL;
    real_t *restrict lam_unique_copy = NULL;

    bool use_cg_A = use_cg;
    bool use_cg_B = use_cg;
    if (NA_as_zero_X && weight == NULL && NA_as_zero_U && U == NULL)
        use_cg_A = false;
    if (NA_as_zero_X && weight == NULL && NA_as_zero_I && II == NULL)
        use_cg_B = false;

    if (!use_cg) finalize_chol = false;

    /* This avoids differences in the scaling of the precomputed matrices */
    if (w_main != 1.)
    {
        lam /= w_main;
        w_user /= w_main;
        w_item /= w_main;
        w_implicit /= w_main;
        if (lam_unique != NULL)
        {
            lam_unique_copy = (real_t*)malloc(6*sizeof(real_t));
            if (lam_unique_copy == NULL) goto throw_oom;
            for (int_t ix = 0; ix < 6; ix++)
                lam_unique_copy[ix] = lam_unique[ix] / w_main;
            lam_unique = lam_unique_copy;
        }
        w_main = 1.;
    }

    if (add_implicit_features && precomputedBiTBi == NULL)
    {
        free_BiTBi = true;
        precomputedBiTBi = (real_t*)malloc((size_t)square(k+k_main)
                                            * sizeof(real_t));
        if (precomputedBiTBi == NULL) goto throw_oom;
    }

    if (nonneg)
    {
        precomputedTransBtBinvBt = NULL;
        precomputedBeTBeChol = NULL;
        precomputedTransCtCinvCt = NULL;
        A_prev = (real_t*)calloc((size_t)m_max
                                 *(size_t)(k_user+k+k_main+has_bias),
                                 sizeof(real_t));
        B_prev = (real_t*)calloc((size_t)n_max
                                 *(size_t)(k_item+k+k_main+has_bias),
                                 sizeof(real_t));
        if (A_prev == NULL || B_prev == NULL) goto throw_oom;
        if (add_implicit_features)
        {
            Ai_prev = (real_t*)calloc((size_t)m*(size_t)(k+k_main),
                                      sizeof(real_t));
            Bi_prev = (real_t*)calloc((size_t)n*(size_t)(k+k_main),
                                      sizeof(real_t));
            if (Ai_prev == NULL || Bi_prev == NULL) goto throw_oom;
        }
    }

    if (nonneg_C)
    {
        C_prev = (real_t*)calloc((size_t)m_u*(size_t)(k_user+k),sizeof(real_t));
        if (C_prev == NULL) goto throw_oom;
    }
    if (nonneg_D)
    {
        D_prev = (real_t*)calloc((size_t)n_i*(size_t)(k_item+k),sizeof(real_t));
        if (D_prev == NULL) goto throw_oom;
    }

    #ifdef _FOR_R
    if (Xfull != NULL) R_nan_to_C_nan(Xfull, (size_t)m*(size_t)n);
    if (U != NULL) R_nan_to_C_nan(U, (size_t)m_u*(size_t)p);
    if (II != NULL) R_nan_to_C_nan(II, (size_t)n_i*(size_t)q);
    #endif

    if (Xfull != NULL || !NA_as_zero_X)
    {
        if (user_bias && item_bias)
        {
            retval = fit_most_popular(
                biasA, biasB,
                glob_mean,
                (lam_unique == NULL)? (lam) : (lam_unique[0]),
                (lam_unique == NULL)? (lam) : (lam_unique[1]),
                1.,
                m, n,
                ixA, ixB, X, nnz,
                Xfull,
                weight,
                false, false, false,
                nonneg,
                (real_t*)NULL,
                nthreads
            );
            if (retval != 0) goto cleanup;
        }
        
        else
        {
            retval = initialize_biases(
                glob_mean, biasA, biasB,
                user_bias, item_bias,
                (lam_unique == NULL)? (lam) : (lam_unique[0]),
                (lam_unique == NULL)? (lam) : (lam_unique[1]),
                m, n,
                m_max, n_max,
                ixA, ixB, X, nnz,
                Xfull, (real_t*)NULL,
                (size_t*)NULL, (int_t*)NULL, (real_t*)NULL,
                (size_t*)NULL, (int_t*)NULL, (real_t*)NULL,
                nonneg,
                nthreads
            );
            if (retval != 0) goto throw_oom;
        }
    }


    if (Xfull != NULL)
    {
        cnt_NA_byrow = (int_t*)calloc(m, sizeof(int_t));
        cnt_NA_bycol = (int_t*)calloc(n, sizeof(int_t));
        if (cnt_NA_byrow == NULL || cnt_NA_bycol == NULL)
            goto throw_oom;

        count_NAs_by_row(Xfull, m, n, cnt_NA_byrow, nthreads,
                         &full_dense, &near_dense_row);
        count_NAs_by_col(Xfull, m, n, cnt_NA_bycol,
                         &full_dense, &near_dense_col);
    }

    else
    {
        if (NA_as_zero_X)
        {
            if (U != NULL  || nnz_U)
                m = max2(m, m_u);
            if (II != NULL || nnz_I)
                n = max2(n, n_i);
            *glob_mean = 0.;
        }
        retval = convert_sparse_X(
                    ixA, ixB, X, nnz,
                    &Xcsr_p, &Xcsr_i, &Xcsr,
                    &Xcsc_p, &Xcsc_i, &Xcsc,
                    weight, &weightR, &weightC,
                    m, n, nthreads
                );
        if (retval != 0) goto throw_oom;
    }

    if (Xfull != NULL && ((!full_dense && near_dense_col) || m > m_u))
    {
        Xtrans = (real_t*)malloc((size_t)m*(size_t)n*sizeof(real_t));
        if (Xtrans == NULL) goto throw_oom;
        transpose_mat2(Xfull, m, n, Xtrans);

        if (weight != NULL)
        {
            Wtrans = (real_t*)malloc((size_t)m*(size_t)n*sizeof(real_t));
            if (Wtrans == NULL) goto throw_oom;
            transpose_mat2(weight, m, n, Wtrans);
        }
    }

    if (add_implicit_features)
    {
        if (Xfull == NULL) {
            Xones = (real_t*)malloc(nnz*sizeof(real_t));
            if (Xones == NULL) goto throw_oom;
            for (size_t ix = 0; ix < nnz; ix++)
                Xones[ix] = 1.;
        }
        else {
            Xones = (real_t*)malloc((size_t)m*(size_t)n*sizeof(real_t));
            zeros_m = (int_t*)calloc(m, sizeof(int_t));
            zeros_n = (int_t*)calloc(n, sizeof(int_t));
            if (Xones == NULL || zeros_m == NULL || zeros_n == NULL)
                goto throw_oom;
            
            for (size_t row = 0; row < (size_t)m; row++)
                for (size_t col = 0; col < (size_t)n; col++)
                    Xones[col + row*(size_t)m]
                        =
                    isnan(Xfull[col + row*(size_t)m])? 0. : 1.;
        }
    }

    /* For the biases, will do the trick by subtracting the bias from
       all entries before optimizing a given matrix. */
    if (user_bias || item_bias)
    {
        A_bias = (real_t*)malloc((size_t)m_max*(size_t)(k_totA+1)
                                  * sizeof(real_t));
        /* Note: 'B_plus_bias' might be part of the desired outputs, in which
           case it is to be passed already allocated. If not, will allocate it
           here instead */
        if (B_plus_bias == NULL)
            B_bias = (real_t*)malloc((size_t)n_max * (size_t)(k_totB+1)
                                                   * sizeof(real_t));
        else
            B_bias = B_plus_bias;
        if (A_bias == NULL || B_bias == NULL) goto throw_oom;
        
        if (Xcsr != NULL)
        {
            if (item_bias) {
                Xcsr_orig = (real_t*)malloc(nnz*sizeof(real_t));
                if (Xcsr_orig == NULL) goto throw_oom;
                copy_arr_(Xcsr, Xcsr_orig, nnz, nthreads);
            }
            if (user_bias) {
                Xcsc_orig = (real_t*)malloc(nnz*sizeof(real_t));
                if (Xcsc_orig == NULL) goto throw_oom;
                copy_arr_(Xcsc, Xcsc_orig, nnz, nthreads);
            }    
        }

        if (Xfull != NULL && (item_bias || Xtrans == NULL))
        {
            Xfull_orig = (real_t*)malloc((size_t)m*(size_t)n*sizeof(real_t));
            if (Xfull_orig == NULL) goto throw_oom;
            copy_arr_(Xfull, Xfull_orig, (size_t)m*(size_t)n, nthreads);
        }

        if (Xtrans != NULL && user_bias)
        {
            Xtrans_orig = (real_t*)malloc((size_t)m*(size_t)n*sizeof(real_t));
            if (Xtrans_orig == NULL) goto throw_oom;
            copy_arr_(Xtrans, Xtrans_orig, (size_t)m*(size_t)n, nthreads);
        }
    }

    else {
        /* these are only used as place-holders, do not get overwritten */
        A_bias = A;
        B_bias = B;
    }

    if (U != NULL || nnz_U)
    {
        if (U == NULL  && NA_as_zero_U)
        {
            m_u = max2(m, m_u);
        }

        retval = preprocess_sideinfo_matrix(
            U, m_u, p,
            U_row, U_col, U_sp, nnz_U,
            U_colmeans, &Utrans,
            &U_csr_p, &U_csr_i, &U_csr,
            &U_csc_p, &U_csc_i, &U_csc,
            &cnt_NA_u_byrow, &cnt_NA_u_bycol,
            &full_dense_u, &near_dense_u_row, &near_dense_u_col,
            NA_as_zero_U, nonneg_C, nthreads
        );
        if (retval != 0) goto throw_oom;
    }

    if (II != NULL || nnz_I)
    {
        if (II == NULL && NA_as_zero_U)
        {
            n_i = max2(n, n_i);
        }

        retval = preprocess_sideinfo_matrix(
            II, n_i, q,
            I_row, I_col, I_sp, nnz_I,
            I_colmeans, &Itrans,
            &I_csr_p, &I_csr_i, &I_csr,
            &I_csc_p, &I_csc_i, &I_csc,
            &cnt_NA_i_byrow, &cnt_NA_i_bycol,
            &full_dense_i, &near_dense_i_row, &near_dense_i_col,
            NA_as_zero_I, nonneg_D, nthreads
        );
        if (retval != 0) goto throw_oom;
    }

    /* Sizes of the temporary arrays */
    if (U != NULL || nnz_U)
        size_bufferC = buffer_size_optimizeA(
            m_u, full_dense_u,
            near_dense_u_col,
            Utrans == NULL,
            U != NULL, false, NA_as_zero_U,
            k_user+k, nthreads,
            ((size_t)n*(size_t)(k_user+k+k_main+user_bias)
                >=
             (size_t)square(k_user+k) )?
                (precomputedTransBtBinvBt != NULL)
                    :
                (precomputedBeTBeChol != NULL),
            false,
            use_cg && !nonneg_C, finalize_chol
        );
    if (II != NULL || nnz_I)
        size_bufferD = buffer_size_optimizeA(
            n_i, full_dense_i,
            near_dense_i_col,
            Itrans == NULL,
            II != NULL, false, NA_as_zero_I,
            k_item+k, nthreads,
            ((size_t)n*(size_t)(k_user+k+k_main+user_bias)
                >=
             (size_t)square(k_item+k) )?
                (precomputedTransBtBinvBt != NULL)
                    :
                ((k_item <= k_user + user_bias)?
                    (precomputedBeTBeChol != NULL) : (false)),
            false,
            use_cg && !nonneg_D, finalize_chol
        );

    if (add_implicit_features)
        size_bufferAi = buffer_size_optimizeA(
            n, true, false, false,
            Xfull != NULL, false, Xfull == NULL,
            k+k_main, nthreads,
            precomputedBtB != NULL, false,
            false, false
        );
    if (add_implicit_features)
        size_bufferBi = buffer_size_optimizeA(
            m, true, false, Xfull != NULL,
            Xfull != NULL, false, Xfull == NULL,
            k+k_main, nthreads,
            precomputedBtB != NULL, false,
            false, false
        );

    if (U != NULL || nnz_U || add_implicit_features)
        size_bufferA = buffer_size_optimizeA_collective(
            m, m_u, n, p,
            k, k_main + (int)user_bias, k_user,
            full_dense,
            near_dense_row, false,
            Xfull != NULL, Xcsr_p != NULL, weight != NULL, NA_as_zero_X,
            U != NULL, U_csr_p != NULL,
            full_dense_u, near_dense_u_row, NA_as_zero_U,
            add_implicit_features, k_main,
            nthreads,
            use_cg_A && !nonneg, finalize_chol,
            nonneg,
            true,
            precomputedBtB != NULL,
            precomputedCtCw != NULL,
            precomputedBeTBeChol != NULL,
            true
        );
    else
        size_bufferA = buffer_size_optimizeA(
            n, full_dense, near_dense_row, false,
            Xfull != NULL, weight != NULL, NA_as_zero_X,
            k+k_main+(int)user_bias, nthreads,
            precomputedBtB != NULL, precompute_for_predictions,
            use_cg && !nonneg, finalize_chol
        );

    if (II != NULL || nnz_I || add_implicit_features)
        size_bufferB = buffer_size_optimizeA_collective(
            n, n_i, m, q,
            k, k_main + (int)item_bias, k_item,
            full_dense,
            near_dense_col,
            (Xtrans != NULL)? false : true,
            Xfull != NULL, Xcsc_p != NULL, weight != NULL, NA_as_zero_X,
            II != NULL, I_csr_p != NULL,
            full_dense_i, near_dense_i_row, NA_as_zero_I,
            add_implicit_features, k_main,
            nthreads,
            use_cg_B && !nonneg, finalize_chol,
            nonneg,
            false,
            (item_bias <= user_bias)?
                (precomputedBtB != NULL) : (false),
            (k_item + item_bias <= k_user + user_bias)?
                (precomputedCtCw != NULL) : (false),
            (k_item + item_bias <= k_user + user_bias)?
                (precomputedBeTBeChol != NULL) : (false),
            true
        );
    else
        size_bufferB = buffer_size_optimizeA(
            m, full_dense, near_dense_col, Xtrans != NULL,
            Xfull != NULL, weight != NULL, NA_as_zero_X,
            k+k_main+(int)item_bias, nthreads,
            ((size_t)n*(size_t)(k_user+k+k_main+user_bias)
                    >=
                 (size_t)square(k+k_main+item_bias) )?
                    (precomputedTransBtBinvBt != NULL)
                        :
                    ((k_item + item_bias <= k_user + user_bias)?
                        (precomputedBeTBeChol != NULL) : (false)),
            false,
            use_cg && !nonneg, finalize_chol
        );

    size_buffer = max2(max2(size_bufferA, size_bufferB),
                       max2(size_bufferC, size_bufferD));
    size_buffer = max2(size_buffer, max2(size_bufferAi, size_bufferBi));
    buffer_real_t = (real_t*)malloc(size_buffer * sizeof(real_t));
    if (buffer_real_t == NULL) goto throw_oom;


    if (reset_values)
    {
        seed_arr = (int_t*)malloc(4*sizeof(int_t));
        if (seed_arr == NULL) goto throw_oom;
        for (int_t ix = 0; ix < 4; ix++) seed_arr[ix] = seed;
        rnorm_preserve_seed(A, (size_t)m_max*(size_t)k_totA, seed_arr);
        rnorm_preserve_seed(B, (size_t)n_max*(size_t)k_totB, seed_arr);
        if (U != NULL || U_csr_p != NULL)
            rnorm_preserve_seed(C, (size_t)p*(size_t)(k_user+k), seed_arr);
        if (II != NULL || I_csr_p != NULL)
            rnorm_preserve_seed(D, (size_t)q*(size_t)(k_item+k), seed_arr);
        if (add_implicit_features) {
            rnorm_preserve_seed(Ai, (size_t)m*(size_t)(k+k_main), seed_arr);
            rnorm_preserve_seed(Bi, (size_t)n*(size_t)(k+k_main), seed_arr);
        }

        free(seed_arr); seed_arr = NULL;

        if (nonneg)
        {
            for (size_t ix = 0; ix < (size_t)m_max*(size_t)k_totA; ix++)
                A[ix] = fabs_t(A[ix]);
            for (size_t ix = 0; ix < (size_t)n_max*(size_t)k_totB; ix++)
                B[ix] = fabs_t(B[ix]);
            if (add_implicit_features)
            {
                for (size_t ix = 0; ix < (size_t)m*(size_t)(k+k_main); ix++)
                    Ai[ix] = fabs_t(Ai[ix]);
                for (size_t ix = 0; ix < (size_t)n*(size_t)(k+k_main); ix++)
                    Bi[ix] = fabs_t(Bi[ix]);
            }
        }

        if (nonneg_C)
        {
            for (size_t ix = 0; ix < (size_t)p*(size_t)(k_user+k); ix++)
                C[ix] = fabs_t(C[ix]);
        }

        if (nonneg_D)
        {
            for (size_t ix = 0; ix < (size_t)q*(size_t)(k_item+k); ix++)
                D[ix] = fabs_t(D[ix]);
        }
    }

    if (include_all_X && add_implicit_features && n_max > n)
    {
        set_to_zero_(Bi + (size_t)n*(size_t)(k+k_main),
                     (size_t)(n_max-n)*(size_t)(k+k_main),
                     nthreads);
    }

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
        if (user_bias) {
            if (m_max > m)
                set_to_zero(biasA + m, m_max - m);
            cblas_tcopy(m_max, biasA, 1,
                        A_bias + k_user+k+k_main, k_user+k+k_main + 1);
        }
        else
            for (size_t ix = 0; ix < (size_t)m_max; ix++)
                A_bias[(size_t)(k_user+k+k_main)
                        + ix*(size_t)(k_user+k+k_main + 1)]
                 = 1.;

        if (item_bias) {
            if (n_max > n)
                set_to_zero(biasB + n, n_max - n);
            cblas_tcopy(n_max, biasB, 1,
                        B_bias + k_item+k+k_main, k_item+k+k_main + 1);
        }
        else
            for (size_t ix = 0; ix < (size_t)n_max; ix++)
                B_bias[(size_t)(k_item+k+k_main)
                        + ix*(size_t)(k_item+k+k_main + 1)]
                 = 1.;
    }

    if (verbose) {
        printf("Starting ALS optimization routine\n\n");
        #if !defined(_FOR_R)
        fflush(stdout);
        #endif
    }

    for (int_t iter = 0; iter < niter; iter++)
    {
        if (iter == niter - 1 && use_cg && finalize_chol) {
            use_cg = false;
            use_cg_A = false;
            use_cg_B = false;
        }

        /* Optimize C and D (they are independent of each other) */
        if (handle_interrupt)
            signal(SIGINT, set_interrup_global_variable);
        if (should_stop_procedure) goto check_interrupt;
        if (U != NULL || nnz_U) {
            if (verbose) {
                printf("Updating C ...");
                #if !defined(_FOR_R)
                fflush(stdout);
                #endif
            }

            if ((size_t)n*(size_t)(k_user+k+k_main+user_bias)
                    <
                (size_t)square(k_user+k))
            {
                filled_BeTBeChol = false;
            }

            optimizeA(
                C, k_user+k,
                A_bias, k_user+k+k_main+(int)(user_bias||item_bias),
                p, m_u, k_user+k,
                U_csc_p, U_csc_i, U_csc,
                (Utrans != NULL)? (Utrans) : (U),
                (Utrans != NULL)? m_u : p,
                full_dense_u,
                near_dense_u_col,
                cnt_NA_u_bycol, (real_t*)NULL, NA_as_zero_U,
                (lam_unique == NULL)? (lam/w_user) : (lam_unique[4]/w_user),
                (lam_unique == NULL)? (lam/w_user) : (lam_unique[4]/w_user),
                Utrans == NULL,
                iter == 0,
                nthreads,
                use_cg && !nonneg_C, max_cg_steps,
                nonneg_C, max_cd_steps, C_prev,
                false,
                ((size_t)n*(size_t)(k_user+k+k_main+user_bias)
                    >=
                 (size_t)square(k_user+k) )?
                    (precomputedTransBtBinvBt) : (precomputedBeTBeChol),
                &ignore,
                buffer_real_t
            );
            if (verbose) {
                printf(" done\n");
                #if !defined(_FOR_R)
                fflush(stdout);
                #endif
            }
        }

        if (handle_interrupt)
            signal(SIGINT, set_interrup_global_variable);
        if (should_stop_procedure) goto check_interrupt;
        if (II != NULL || nnz_I) {
            if (verbose) {
                printf("Updating D ...");
                #if !defined(_FOR_R)
                fflush(stdout);
                #endif
            }

            if ((size_t)n*(size_t)(k_user+k+k_main+user_bias)
                    <
                (size_t)square(k_item+k))
            {
                if (k_item > k_user + user_bias)
                    filled_BeTBeChol = false;
            }

            optimizeA(
                D, k_item+k,
                B_bias, k_item+k+k_main+(int)(user_bias||item_bias),
                q, n_i, k_item+k,
                I_csc_p, I_csc_i, I_csc,
                (Itrans != NULL)? (Itrans) : (II),
                (Itrans != NULL)? n_i : q,
                full_dense_i,
                near_dense_i_col,
                cnt_NA_i_bycol, (real_t*)NULL, NA_as_zero_I,
                (lam_unique == NULL)? (lam/w_item) : (lam_unique[5]/w_item),
                (lam_unique == NULL)? (lam/w_item) : (lam_unique[5]/w_item),
                Itrans == NULL,
                iter == 0,
                nthreads,
                use_cg && !nonneg_D, max_cg_steps,
                nonneg_D, max_cd_steps, D_prev,
                false,
                ((size_t)n*(size_t)(k_user+k+k_main+user_bias)
                    >=
                 (size_t)square(k_item+k) )?
                    (precomputedTransBtBinvBt)
                        :
                    ((k_item <= k_user + user_bias)?
                        (precomputedBeTBeChol) : ((real_t*)NULL)),
                &ignore,
                buffer_real_t
            );
            if (verbose) {
                printf(" done\n");
                #if !defined(_FOR_R)
                fflush(stdout);
                #endif
            }
        }

        /* Optimizing implicit-features matrices (also independent) */
        if (add_implicit_features)
        {
            if (handle_interrupt)
                signal(SIGINT, set_interrup_global_variable);
            if (should_stop_procedure) goto check_interrupt;
            if (verbose) {
                printf("Updating Bi...");
                #if !defined(_FOR_R)
                fflush(stdout);
                #endif
            }

            filled_BtB = false;
            optimizeA(
                Bi, k+k_main,
                A_bias + k_user, k_user+k+k_main+(user_bias||item_bias),
                n, m, k+k_main,
                Xcsc_p, Xcsc_i, (Xfull == NULL)? (Xones) : ((real_t*)NULL),
                (Xfull == NULL)? ((real_t*)NULL) : (Xones),
                m,
                Xfull != NULL, false,
                (Xfull == NULL)? ((int_t*)NULL) : (zeros_n),
                (real_t*)NULL, Xfull == NULL,
                (lam_unique == NULL)?
                    (lam/w_implicit) : (lam_unique[3]/w_implicit),
                (lam_unique == NULL)?
                    (lam/w_implicit) : (lam_unique[3]/w_implicit),
                Xfull != NULL, niter == 0,
                nthreads,
                false, 0,
                nonneg, max_cd_steps, Bi_prev,
                false,
                precomputedBtB, &ignore,
                buffer_real_t
            );
            if (verbose) {
                printf(" done\n");
                #if !defined(_FOR_R)
                fflush(stdout);
                #endif
            }


            if (handle_interrupt)
                signal(SIGINT, set_interrup_global_variable);
            if (should_stop_procedure) goto check_interrupt;
            if (verbose) {
                printf("Updating Ai...");
                #if !defined(_FOR_R)
                fflush(stdout);
                #endif
            }

            optimizeA(
                Ai, k+k_main,
                B_bias + k_item, k_item+k+k_main+(user_bias||item_bias),
                m, n, k+k_main,
                Xcsr_p, Xcsr_i, (Xfull == NULL)? (Xones) : ((real_t*)NULL),
                (Xfull == NULL)? ((real_t*)NULL) : (Xones),
                n,
                Xfull != NULL, false,
                (Xfull == NULL)? ((int_t*)NULL) : (zeros_m),
                (real_t*)NULL, Xfull == NULL,
                (lam_unique == NULL)?
                    (lam/w_implicit) : (lam_unique[2]/w_implicit),
                (lam_unique == NULL)?
                    (lam/w_implicit) : (lam_unique[2]/w_implicit),
                false, niter == 0,
                nthreads,
                false, 0,
                nonneg, max_cd_steps, Bi_prev,
                false,
                precomputedBtB, &ignore,
                buffer_real_t
            );

            filled_BiTBi = true;

            if (verbose) {
                printf(" done\n");
                #if !defined(_FOR_R)
                fflush(stdout);
                #endif
            }
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
                for (int_t ix = 0; ix < m_max; ix++)
                    A_bias[(size_t)(k_user+k+k_main)
                            + ix*(size_t)(k_user+k+k_main + 1)] = 1.;
        }

        else if (Xfull_orig != NULL || Xtrans_orig != NULL)
        {
            if (Xtrans_orig != NULL)
                copy_arr_(Xtrans, Xtrans_orig, (size_t)m*(size_t)n, nthreads);
            else
                copy_arr_(Xfull, Xfull_orig, (size_t)m*(size_t)n, nthreads);
        }

        /* Optimize B */
        if (handle_interrupt)
            signal(SIGINT, set_interrup_global_variable);
        if (should_stop_procedure) goto check_interrupt;
        if (verbose) {
            printf("Updating B ...");
            #if !defined(_FOR_R)
            fflush(stdout);
            #endif
        }

        if (k_item + item_bias <= k_user + user_bias)
        {
            filled_CtCw = false;
            filled_BeTBeChol = false;
        }

        if (item_bias <= user_bias)
        {
            filled_BtB = false;
            filled_BeTBeChol = false;
        }
        filled_BiTBi = false;

        /* TODO: it's possible to use more buffers here for the
           case of k_item > k_user, avoid extra memory usage */
        if (II != NULL || nnz_I || add_implicit_features)
            optimizeA_collective(
                B_bias, k_totB + has_bias, A_bias, k_totA + has_bias, D, Ai,
                n, n_i, m, q,
                k, k_main+(int)item_bias, k_item, k_user,
                Xcsc_p, Xcsc_i, Xcsc,
                (Xtrans != NULL)? (Xtrans) : (Xfull),
                full_dense, near_dense_col, (Xtrans == NULL)? (m) : (n),
                cnt_NA_bycol,
                (Xtrans != NULL)? (Wtrans) : ((Xfull == NULL)? weight:weightC),
                NA_as_zero_X,
                Xones, k_main, m,
                add_implicit_features,
                I_csr_p, I_csr_i, I_csr,
                II, cnt_NA_i_byrow,
                full_dense_i, near_dense_i_row, NA_as_zero_I,
                (lam_unique == NULL)? (lam) : (lam_unique[3]),
                w_item, w_implicit,
                (lam_unique == NULL)? (lam) : (lam_unique[item_bias? 1 : 3]),
                Xfull != NULL && Xtrans == NULL,
                nthreads,
                use_cg_B && !nonneg, max_cg_steps, iter == 0,
                nonneg, max_cd_steps, B_prev,
                false,
                (item_bias <= user_bias)?
                    (precomputedBtB) : ((real_t*)NULL),
                (k_item + item_bias <= k_user + user_bias)?
                    (precomputedCtCw) : ((real_t*)NULL),
                (k_item + item_bias <= k_user + user_bias)?
                    (precomputedBeTBeChol) : ((real_t*)NULL),
                precomputedBiTBi,
                &ignore, &ignore, &ignore,
                buffer_real_t
            );
        else
            optimizeA(
                B_bias + k_item, k_item+k+k_main+(int)(user_bias||item_bias),
                A_bias + k_user, k_user+k+k_main+(int)(user_bias||item_bias),
                n, m, k+k_main+(int)item_bias,
                Xcsc_p, Xcsc_i, Xcsc,
                (Xtrans != NULL)? (Xtrans) : (Xfull),
                (Xtrans == NULL)? m : n,
                full_dense, near_dense_col,
                cnt_NA_bycol,
                (Xtrans != NULL)? (Wtrans) : ((Xfull == NULL)? weight:weightC),
                NA_as_zero_X,
                (lam_unique == NULL)? (lam) : (lam_unique[3]),
                (lam_unique == NULL)? (lam) : (lam_unique[item_bias? 1 : 3]),
                Xfull != NULL && Xtrans == NULL, iter == 0,
                nthreads,
                use_cg && !nonneg, max_cg_steps,
                nonneg, max_cd_steps, B_prev,
                false,
                ((size_t)n*(size_t)(k_user+k+k_main+user_bias)
                    >=
                 (size_t)square(k+k_main+item_bias) )?
                    (precomputedTransBtBinvBt)
                        :
                    ((item_bias <= k_user + user_bias)?
                        (precomputedBeTBeChol) : ((real_t*)NULL)),
                &ignore,
                buffer_real_t
            );
        if (verbose) {
            printf(" done\n");
            #if !defined(_FOR_R)
            fflush(stdout);
            #endif
        }

        if (item_bias)
            cblas_tcopy(n, B_bias + k_item+k+k_main, k_item+k+k_main + 1,
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
                for (int_t ix = 0; ix < n_max; ix++)
                    B_bias[(size_t)(k_item+k+k_main)
                            + ix*(size_t)(k_item+k+k_main + 1)] = 1.;
        }

        else if (Xfull_orig != NULL)
        {
            copy_arr_(Xfull, Xfull_orig, (size_t)m*(size_t)n, nthreads);
        }

        /* Optimize A */
        filled_BtB = false;
        filled_CtCw = false;
        filled_BeTBeChol = false;
        if (handle_interrupt)
            signal(SIGINT, set_interrup_global_variable);
        if (should_stop_procedure) goto check_interrupt;
        if (verbose) {
            printf("Updating A ...");
            #if !defined(_FOR_R)
            fflush(stdout);
            #endif
        }
        if (U != NULL || nnz_U || add_implicit_features)
            optimizeA_collective(
                A_bias, k_totA + has_bias, B_bias, k_totB + has_bias, C, Bi,
                m, m_u, n, p,
                k, k_main+(int)user_bias, k_user, k_item,
                Xcsr_p, Xcsr_i, Xcsr,
                Xfull, full_dense, near_dense_row, n,
                cnt_NA_byrow,
                (Xfull == NULL)? (weightR) : (weight),
                NA_as_zero_X,
                Xones, k_main, n,
                add_implicit_features,
                U_csr_p, U_csr_i, U_csr,
                U, cnt_NA_u_byrow,
                full_dense_u, near_dense_u_row, NA_as_zero_U,
                (lam_unique == NULL)? (lam) : (lam_unique[2]),
                w_user, w_implicit,
                (lam_unique == NULL)? (lam) : (lam_unique[user_bias? 0 : 2]),
                false,
                nthreads,
                use_cg_B && !nonneg, max_cg_steps, iter == 0,
                nonneg, max_cd_steps, A_prev,
                precompute_for_predictions,
                precomputedBtB, precomputedCtCw, precomputedBeTBeChol,
                precomputedBiTBi,
                &filled_BtB, &filled_CtCw, &filled_BeTBeChol,
                buffer_real_t
            );
        else
            optimizeA(
                A_bias + k_user, k_user+k+k_main+(int)(user_bias||item_bias),
                B_bias + k_item, k_item+k+k_main+(int)(user_bias||item_bias),
                m, n, k+k_main+(int)user_bias,
                Xcsr_p, Xcsr_i, Xcsr,
                Xfull, n,
                full_dense, near_dense_row,
                cnt_NA_byrow,
                (Xfull == NULL)? (weightR) : (weight),
                NA_as_zero_X,
                (lam_unique == NULL)? (lam) : (lam_unique[2]),
                (lam_unique == NULL)? (lam) : (lam_unique[user_bias? 0 : 2]),
                false, iter == 0,
                nthreads,
                use_cg && !nonneg, max_cg_steps,
                nonneg, max_cd_steps, A_prev,
                iter == niter - 1,
                precomputedBtB, &filled_BtB,
                buffer_real_t
            );
        if (verbose) {
            printf(" done\n");
            #if !defined(_FOR_R)
            fflush(stdout);
            #endif
        }

        if (user_bias)
            cblas_tcopy(m, A_bias + k_user+k+k_main, k_user+k+k_main + 1,
                        biasA, 1);

        if (verbose) {
            printf("\tCompleted ALS iteration %2d\n\n", iter+1);
            #if !defined(_FOR_R)
            fflush(stdout);
            #endif
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
        #if !defined(_FOR_R)
        fflush(stdout);
        #endif
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

        if (m_max > m && user_bias)
            set_to_zero(biasA + m, m_max-m);
        if (n_max > n && item_bias)
            set_to_zero(biasB + n, n_max-n);
    }

    precompute:
    if (precompute_for_predictions)
    {
        if (add_implicit_features && !filled_BiTBi)
        {
            cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        k+k_main, n,
                        w_implicit, Bi, k+k_main,
                        0., precomputedBiTBi, k+k_main);
            filled_BiTBi = true;
        }
        else if (add_implicit_features && use_cg && w_implicit != 1. &&
                 !back_to_precompute && !free_BiTBi)
        {
            cblas_tscal(square(k+k_main), 1./w_implicit, precomputedBiTBi, 1);
        }
        if (!filled_BtB)
        {
            cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        k + k_main + user_bias, include_all_X? n_max : n,
                        1., B_bias + k_item, k_item+k+k_main+has_bias,
                        0., precomputedBtB, k+k_main+user_bias);
            filled_BtB = true;
        }
        else if (include_all_X && n != n_max &&
                 buffer_real_t != NULL && !back_to_precompute)
            cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        k + k_main + user_bias, n_max - n,
                        1., B_bias
                                + k_item
                                + (size_t)n*(size_t)(k_item+k+k_main+has_bias),
                        k_item+k+k_main+has_bias,
                        1., precomputedBtB, k+k_main+user_bias);

        if (precomputedTransBtBinvBt != NULL && !finished_TransBtBinvBt &&
            !add_implicit_features && !nonneg)
        {
            k_pred = k + k_main + (int)user_bias;
            free_arr_use = false;

             /* This is in case it needs an extra malloc, in which case will
                first free up all the memory it allocated before, which is
                no longer needed at this point. */
            if (!back_to_precompute && buffer_real_t != NULL)
            {
                if (precomputedBeTBeChol != NULL && !filled_BeTBeChol)
                    arr_use = precomputedBeTBeChol;
                else if (precomputedTransCtCinvCt != NULL &&
                         (size_t)p*(size_t)(k_user+k) >= (size_t)square(k_pred))
                    arr_use = precomputedTransCtCinvCt;
                else if (precomputedCtCw != NULL && !filled_CtCw &&
                         k_user >= k_main + user_bias)
                    arr_use = precomputedCtCw;
                else if (size_buffer >= (size_t)square(k_pred))
                    arr_use = buffer_real_t;
                else {
                    back_to_precompute = true;
                    goto cleanup;
                }
            }

            else
            {
                free_arr_use = true;
                back_to_precompute = false;
                arr_use = (real_t*)malloc((size_t)square(k_pred)
                                           * sizeof(real_t));
                if (arr_use == NULL) goto throw_oom;
            }

            copy_mat(include_all_X? n_max : n, k+k_main+user_bias,
                     B_bias + k_item, k_item+k+k_main+has_bias,
                     precomputedTransBtBinvBt, k+k_main+user_bias);
            copy_arr(precomputedBtB, arr_use, square(k_pred));
            add_to_diag(arr_use,
                        (lam_unique == NULL)? (lam) : (lam_unique[2]),
                        k_pred);
            if (lam_unique != NULL && user_bias && lam_unique[0]!=lam_unique[2])
                arr_use[square(k_pred)-1] += (lam_unique[0]-lam_unique[2]);
            tposv_(&lo, &k_pred, include_all_X? &n_max : &n,
                   arr_use, &k_pred,
                   precomputedTransBtBinvBt, &k_pred, &ignore_int);

            if (free_arr_use) free(arr_use);
            arr_use = NULL;
            finished_TransBtBinvBt = true;
        }

        if (p > 0)
        {
            if (precomputedTransCtCinvCt != NULL && !finished_TransCtCinvCt &&
                !add_implicit_features && !nonneg)
            {
                k_pred = k_user + k;
                free_arr_use = false;
                arr_use = NULL;

                if (!back_to_precompute && buffer_real_t != NULL)
                {
                    if (precomputedBeTBeChol != NULL && !filled_BeTBeChol)
                        arr_use = precomputedBeTBeChol;
                    else if (size_buffer > (size_t)square(k_user+k))
                        arr_use = buffer_real_t;
                    else {
                        back_to_precompute = true;
                        goto cleanup;
                    }
                }

                else
                {
                    free_arr_use = true;
                    back_to_precompute = false;
                    arr_use = (real_t*)malloc((size_t)square(k_pred)
                                              * sizeof(real_t));
                    if (arr_use == NULL) goto throw_oom;
                }

                copy_arr_(C, precomputedTransCtCinvCt,
                          (size_t)p*(size_t)(k_user+k), nthreads);
                if (!filled_CtCw) {
                    cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                                k_pred, p,
                                1., C, k_pred,
                                0., precomputedCtCw, k_pred);
                    copy_arr(precomputedCtCw, arr_use, square(k_pred));
                    add_to_diag(arr_use,
                                ((lam_unique == NULL)? (lam) : (lam_unique[2]))
                                    / w_user,
                                k_pred);
                    if (w_user != 1.)
                        cblas_tscal(square(k_pred), w_user, precomputedCtCw, 1);
                    filled_CtCw = true;
                }
                else {
                    copy_arr(precomputedCtCw, arr_use, square(k_pred));
                    if (w_user != 1. && !use_cg)
                        cblas_tscal(square(k_pred), 1./w_user, arr_use, 1);
                    add_to_diag(arr_use,
                                ((lam_unique == NULL)? (lam) : (lam_unique[2]))
                                    / w_user,
                                k_pred);
                }
                tposv_(&lo, &k_pred, &p,
                       arr_use, &k_pred,
                       precomputedTransCtCinvCt, &k_pred, &ignore_int);

                if (free_arr_use) free(arr_use);
                arr_use = NULL;
                finished_TransCtCinvCt = true;
            }


            if (precomputedCtCw != NULL && filled_CtCw && use_cg && w_user !=1.)
                cblas_tscal(square(k_user+k), w_user, precomputedCtCw, 1);
            else if (!filled_CtCw && precomputedCtCw != NULL) {
                cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                            k_user+k, p,
                            w_user, C, k_user+k,
                            0., precomputedCtCw, k_user+k);
                filled_CtCw = true;
            }
        }

        if (precomputedBeTBeChol != NULL && !filled_BeTBeChol &&
            (p || add_implicit_features) && !nonneg)
        {
            k_pred = k_user + k + k_main + (int)user_bias;
            set_to_zero(precomputedBeTBeChol, square(k_pred));
            
            copy_mat(k+k_main+user_bias, k+k_main+user_bias,
                     precomputedBtB, k+k_main+user_bias,
                     precomputedBeTBeChol + k_user + k_user*k_pred, k_pred);
            if (p) {
                if (filled_CtCw) {
                    sum_mat(k_user+k, k_user+k,
                            precomputedCtCw, k_user+k,
                            precomputedBeTBeChol, k_pred);
                }
                else {
                    if (precomputedCtCw != NULL) {
                        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                                    k_user+k, p,
                                    w_user, C, k_user+k,
                                    0., precomputedCtCw, k_user+k);
                        sum_mat(k_user+k, k_user+k,
                                precomputedCtCw, k_user+k,
                                precomputedBeTBeChol, k_pred);
                        if (w_user != 1.)
                            cblas_tscal(square(k_user+k), w_user,
                                        precomputedCtCw, 1);
                        filled_CtCw = true;
                    } else {
                        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                                    k_user+k, p,
                                    w_user, C, k_user+k,
                                    1., precomputedBeTBeChol, k_pred);
                    }
                }
            }
            if (add_implicit_features)
                sum_mat(k+k_main, k+k_main,
                        precomputedBiTBi, k+k_main,
                        precomputedBeTBeChol + k_user + k_user*k_pred, k_pred);

            add_to_diag(precomputedBeTBeChol,
                        (lam_unique == NULL)? (lam) : (lam_unique[2]),
                        k_user+k+k_main+user_bias);
            if (lam_unique != NULL && user_bias && lam_unique[0]!=lam_unique[2])
                precomputedBeTBeChol[square(k_pred)-1]
                    +=
                (lam_unique[0]-lam_unique[2]);

            tpotrf_(&lo, &k_pred, precomputedBeTBeChol,&k_pred,&ignore_int);
            filled_BeTBeChol = true;
        }

        back_to_precompute = false;
    }

    cleanup:
        free(buffer_real_t); buffer_real_t = NULL;
        free(A_prev); A_prev = NULL;
        free(B_prev); B_prev = NULL;
        free(C_prev); C_prev = NULL;
        free(D_prev); D_prev = NULL;
        free(Ai_prev); Ai_prev = NULL;
        free(Bi_prev); Bi_prev = NULL;
        free(Xtrans); Xtrans = NULL;
        free(Wtrans); Wtrans = NULL;
        free(Xcsr_p); Xcsr_p = NULL;
        free(Xcsr_i); Xcsr_i = NULL;
        free(Xcsr); Xcsr = NULL;
        free(weightR); weightR = NULL;
        free(Xcsc_p); Xcsc_p = NULL;
        free(Xcsc_i); Xcsc_i = NULL;
        free(Xcsc); Xcsc = NULL;
        free(weightC); weightC = NULL;
        free(Xones); Xones = NULL;
        free(Utrans); Utrans = NULL;
        free(U_csr_p); U_csr_p = NULL;
        free(U_csr_i); U_csr_i = NULL;
        free(U_csr); U_csr = NULL;
        free(U_csc_p); U_csc_p = NULL;
        free(U_csc_i); U_csc_i = NULL;
        free(U_csc); U_csc = NULL;
        free(Itrans); Itrans = NULL;
        free(I_csr_p); I_csr_p = NULL;
        free(I_csr_i); I_csr_i = NULL;
        free(I_csr); I_csr = NULL;
        free(I_csc_p); I_csc_p = NULL;
        free(I_csc_i); I_csc_i = NULL;
        free(I_csc); I_csc = NULL;
        free(cnt_NA_byrow); cnt_NA_byrow = NULL;
        free(cnt_NA_bycol); cnt_NA_bycol = NULL;
        free(cnt_NA_u_byrow); cnt_NA_u_byrow = NULL;
        free(cnt_NA_u_bycol); cnt_NA_u_bycol = NULL;
        free(cnt_NA_i_byrow); cnt_NA_i_byrow = NULL;
        free(cnt_NA_i_bycol); cnt_NA_i_bycol = NULL;
        free(zeros_m); zeros_m = NULL;
        free(zeros_n); zeros_n = NULL;
        if (user_bias || item_bias) {
            free(A_bias); A_bias = NULL;
            if (B_plus_bias != B_bias && B_bias != B) {
                free(B_bias); B_bias = NULL;
            }
            free(Xcsr_orig); Xcsr_orig = NULL;
            free(Xcsc_orig); Xcsc_orig = NULL;
        }
        if (Xfull_orig != NULL) {
            free(Xfull_orig); Xfull_orig = NULL;
        }
        if (Xtrans_orig != NULL) {
            free(Xtrans_orig); Xtrans_orig = NULL;
        }
        free(seed_arr); seed_arr = NULL;
        if (back_to_precompute) goto precompute;
        free(lam_unique_copy); lam_unique_copy = NULL;
        if (free_BiTBi) {
            free(precomputedBiTBi); precomputedBiTBi = NULL;
        }
    return retval;

    throw_oom:
    {
        retval = 1;
        back_to_precompute = false;
        if (verbose)
            print_oom_message();
        goto cleanup;
    }
}

/* TODO: the separation between implicit/explicit is no longer needed,
   as the explicit one can now imitate the implicit. Should instead make
   this function call the explicit one. */
int_t fit_collective_implicit_als
(
    real_t *restrict A, real_t *restrict B,
    real_t *restrict C, real_t *restrict D,
    bool reset_values, int_t seed,
    real_t *restrict U_colmeans, real_t *restrict I_colmeans,
    int_t m, int_t n, int_t k,
    int_t ixA[], int_t ixB[], real_t *restrict X, size_t nnz,
    real_t lam, real_t *restrict lam_unique,
    real_t *restrict U, int_t m_u, int_t p,
    real_t *restrict II, int_t n_i, int_t q,
    int_t U_row[], int_t U_col[], real_t *restrict U_sp, size_t nnz_U,
    int_t I_row[], int_t I_col[], real_t *restrict I_sp, size_t nnz_I,
    bool NA_as_zero_U, bool NA_as_zero_I,
    int_t k_main, int_t k_user, int_t k_item,
    real_t w_main, real_t w_user, real_t w_item,
    real_t *restrict w_main_multiplier,
    real_t alpha, bool adjust_weight, bool apply_log_transf,
    int_t niter, int_t nthreads, bool verbose, bool handle_interrupt,
    bool use_cg, int_t max_cg_steps, bool finalize_chol,
    bool nonneg, int_t max_cd_steps, bool nonneg_C, bool nonneg_D,
    bool precompute_for_predictions,
    real_t *restrict precomputedBtB,
    real_t *restrict precomputedBeTBe,
    real_t *restrict precomputedBeTBeChol
)
{
    int_t retval = 0;
    if (k_user && U == NULL && nnz_U == 0) {
        if (verbose)
            fprintf(stderr, "Cannot pass 'k_user' without U data.\n");
        retval = 2;
    }

    if (k_item && II == NULL && nnz_I == 0) {
        if (verbose)
            fprintf(stderr, "Cannot pass 'k_item' without I data.\n");
        retval = 2;
    }

    if (k_main && nnz == 0) {
        if (verbose)
            fprintf(stderr, "Cannot pass 'k_main' without X data.\n");
        retval = 2;
    }

    if ((U != NULL && NA_as_zero_U) ||
        (II != NULL && NA_as_zero_I))
    {
        if (verbose)
            fprintf(stderr, "Cannot pass 'NA_as_zero' with dense data.\n");
        retval = 2;
    }

    if (retval == 2)
    {
        if (verbose) {
            #ifndef _FOR_R
            fflush(stderr);
            #endif
        }
        return retval;
    }
    
    int_t k_totA = k_user + k + k_main;
    int_t k_totB = k_item + k + k_main;
    int_t m_max = max2(m, m_u);
    int_t n_max = max2(n, n_i);

    real_t *restrict buffer_real_t = NULL;
    size_t size_bufferA = 0;
    size_t size_bufferB = 0;
    size_t size_bufferC = 0;
    size_t size_bufferD = 0;
    size_t size_buffer = 0;

    real_t *restrict A_prev = NULL;
    real_t *restrict B_prev = NULL;
    real_t *restrict C_prev = NULL;
    real_t *restrict D_prev = NULL;

    size_t *Xcsr_p = (size_t*)malloc(((size_t)m+(size_t)1)*sizeof(size_t));
    int_t *Xcsr_i = (int_t*)malloc(nnz*sizeof(int_t));
    real_t *restrict Xcsr = (real_t*)malloc(nnz*sizeof(real_t));
    size_t *Xcsc_p = (size_t*)malloc(((size_t)n+(size_t)1)*sizeof(size_t));
    int_t *Xcsc_i = (int_t*)malloc(nnz*sizeof(int_t));
    real_t *restrict Xcsc = (real_t*)malloc(nnz*sizeof(real_t));
    real_t *restrict Utrans = NULL;
    size_t *U_csr_p = NULL;
    int_t *U_csr_i = NULL;
    real_t *restrict U_csr = NULL;
    size_t *U_csc_p = NULL;
    int_t *U_csc_i = NULL;
    real_t *restrict U_csc = NULL;
    real_t *restrict Itrans = NULL;
    size_t *I_csr_p = NULL;
    int_t *I_csr_i = NULL;
    real_t *restrict I_csr = NULL;
    size_t *I_csc_p = NULL;
    int_t *I_csc_i = NULL;
    real_t *restrict I_csc = NULL;
    int_t *restrict cnt_NA_u_byrow = NULL;
    int_t *restrict cnt_NA_u_bycol = NULL;
    int_t *restrict cnt_NA_i_byrow = NULL;
    int_t *restrict cnt_NA_i_bycol = NULL;
    bool full_dense_u = false;
    bool near_dense_u_row = false;
    bool near_dense_u_col = false;
    bool full_dense_i = false;
    bool near_dense_i_row = false;
    bool near_dense_i_col = false;

    bool filled_BtB = false;
    bool filled_BeTBe = false;
    bool filled_BeTBeChol = false;
    bool filled_CtC = false;
    bool allocated_CtC = false;
    bool ignore = false;
    real_t *restrict precomputedCtC = NULL;

    int_t *restrict seed_arr = NULL;
    real_t *restrict lam_unique_copy = NULL;

    if (!use_cg) finalize_chol = false;

    if (Xcsr_p == NULL || Xcsr_i == NULL || Xcsr == NULL ||
        Xcsc_p == NULL || Xcsc_i == NULL || Xcsc == NULL)
    {
        goto throw_oom;
    }

    if (!precompute_for_predictions)
    {
        precomputedBtB = (real_t*)malloc((size_t)square(k+k_main)
                                         * sizeof(real_t));
        if (precomputedBtB == NULL) goto throw_oom;
    }

    if (precompute_for_predictions && (U != NULL || nnz_U) && use_cg)
    {
        /* This one may be reused when solving for A or B. */
        precomputedCtC = (real_t*)malloc((size_t)square(max2(k_user, k_item)+k)
                                          * sizeof(real_t));
        if (precomputedCtC == NULL) goto throw_oom;
        allocated_CtC = true;
    }

    if (nonneg)
    {
        precomputedBeTBeChol = NULL;
        A_prev = (real_t*)calloc((size_t)m_max*(size_t)(k_user+k+k_main),
                                 sizeof(real_t));
        B_prev = (real_t*)calloc((size_t)n_max*(size_t)(k_item+k+k_main),
                                 sizeof(real_t));
        if (A_prev == NULL || B_prev == NULL) goto throw_oom;
    }

    if (nonneg_C)
    {
        C_prev = (real_t*)calloc((size_t)m_u*(size_t)(k_user+k),sizeof(real_t));
        if (C_prev == NULL) goto throw_oom;
    }
    if (nonneg_D)
    {
        D_prev = (real_t*)calloc((size_t)n_i*(size_t)(k_item+k),sizeof(real_t));
        if (D_prev == NULL) goto throw_oom;
    }

    #ifdef _FOR_R
    if (U != NULL) R_nan_to_C_nan(U, (size_t)m_u*(size_t)p);
    if (II != NULL) R_nan_to_C_nan(II, (size_t)n_i*(size_t)q);
    #endif

    if (apply_log_transf)
        for (size_t ix = 0; ix < nnz; ix++)
            X[ix] = log_t(X[ix]);
    if (alpha != 1.)
        tscal_large(X, alpha, nnz, nthreads);
    coo_to_csr_and_csc(
        ixA, ixB, X,
        (real_t*)NULL, m, n, nnz,
        Xcsr_p, Xcsr_i, Xcsr,
        Xcsc_p, Xcsc_i, Xcsc,
        (real_t*)NULL, (real_t*)NULL,
        nthreads
    );

    if (U == NULL && NA_as_zero_U)
        m_u = m_max;
    retval = preprocess_sideinfo_matrix(
        U, m_u, p,
        U_row, U_col, U_sp, nnz_U,
        U_colmeans, &Utrans,
        &U_csr_p, &U_csr_i, &U_csr,
        &U_csc_p, &U_csc_i, &U_csc,
        &cnt_NA_u_byrow, &cnt_NA_u_bycol,
        &full_dense_u, &near_dense_u_row, &near_dense_u_col,
        NA_as_zero_U, nonneg_C, nthreads
    );
    if (retval != 0) goto throw_oom;
    if (U != NULL || U_csr_p != NULL)
        size_bufferC = buffer_size_optimizeA(
            m_u, full_dense_u,
            near_dense_u_col,
            Utrans == NULL,
            U != NULL, false, NA_as_zero_U,
            k_user+k, nthreads,
            precomputedBeTBeChol != NULL,
            false,
            use_cg && !nonneg_C, finalize_chol
        );
    
    if (II == NULL && NA_as_zero_I)
        n_i = n_max;
    retval = preprocess_sideinfo_matrix(
        II, n_i, q,
        I_row, I_col, I_sp, nnz_I,
        I_colmeans, &Itrans,
        &I_csr_p, &I_csr_i, &I_csr,
        &I_csc_p, &I_csc_i, &I_csc,
        &cnt_NA_i_byrow, &cnt_NA_i_bycol,
        &full_dense_i, &near_dense_i_row, &near_dense_i_col,
        NA_as_zero_I, nonneg_D, nthreads
    );
    if (retval != 0) goto throw_oom;
    if (II != NULL || I_csc_p != NULL)
        size_bufferD = buffer_size_optimizeA(
            n_i, full_dense_i,
            near_dense_i_col,
            Itrans == NULL,
            II != NULL, false, NA_as_zero_I,
            k_item+k, nthreads,
            precomputedBeTBeChol != NULL && k_item+k <= k_user+k+k_main,
            false,
            use_cg && !nonneg_D, finalize_chol
        );

    if (U != NULL || U_csr_p != NULL)
        size_bufferA = buffer_size_optimizeA_collective_implicit(
                            m, m_u, p,
                            k, k_main, k_user,
                            U == NULL && U_csr_p != NULL,
                            NA_as_zero_U,
                            nthreads,
                            use_cg && !nonneg,
                            true,
                            precompute_for_predictions,
                            precompute_for_predictions,
                            allocated_CtC,
                            finalize_chol
                        );
    else
        size_bufferA = buffer_size_optimizeA_implicit(
                            k+k_main, nthreads,
                            true,
                            use_cg && !nonneg, finalize_chol
                        );

    if (II != NULL || I_csr_p != NULL)
        size_bufferB = buffer_size_optimizeA_collective_implicit(
                            n, n_i, q,
                            k, k_main, k_item,
                            II == NULL && I_csr_p != NULL,
                            NA_as_zero_I,
                            nthreads,
                            use_cg && !nonneg,
                            true,
                            precompute_for_predictions && k_item <= k_user,
                            precompute_for_predictions && k_item <= k_user,
                            allocated_CtC,
                            finalize_chol
                        );
    else
        size_bufferB = buffer_size_optimizeA_implicit(
                            k+k_main, nthreads,
                            true,
                            use_cg && !nonneg, finalize_chol
                        );


    size_buffer = max2(max2(size_bufferA, size_bufferB),
                       max2(size_bufferC, size_bufferD));
    buffer_real_t = (real_t*)malloc(size_buffer * sizeof(real_t));
    if (buffer_real_t == NULL) goto throw_oom;

    if (reset_values)
    {
        seed_arr = (int_t*)malloc(4*sizeof(int_t));
        if (seed_arr == NULL) goto throw_oom;
        for (int_t ix = 0; ix < 4; ix++) seed_arr[ix] = seed;
        rnorm_preserve_seed(A, (size_t)k_totA*(size_t)m_max, seed_arr);
        rnorm_preserve_seed(B, (size_t)k_totB*(size_t)n_max, seed_arr);
        if (U != NULL || U_csr_p != NULL)
            rnorm_preserve_seed(C, (size_t)(k_user+k)*(size_t)p, seed_arr);
        if (II != NULL || I_csr_p != NULL)
            rnorm_preserve_seed(D, (size_t)(k_item+k)*(size_t)q, seed_arr);
        free(seed_arr); seed_arr = NULL;

        if (nonneg)
        {
            for (size_t ix = 0; ix < (size_t)m_max*(size_t)k_totA; ix++)
                A[ix] = fabs_t(A[ix]);
            for (size_t ix = 0; ix < (size_t)n_max*(size_t)k_totB; ix++)
                B[ix] = fabs_t(B[ix]);
        }

        if (nonneg_C)
        {
            for (size_t ix = 0; ix < (size_t)p*(size_t)(k_user+k); ix++)
                C[ix] = fabs_t(C[ix]);
        }

        if (nonneg_D)
        {
            for (size_t ix = 0; ix < (size_t)q*(size_t)(k_item+k); ix++)
                D[ix] = fabs_t(D[ix]);
        }
    }

    *w_main_multiplier = 1.;
    if (adjust_weight)
    {
        *w_main_multiplier = (long double)nnz
                                /
                             (long double)((size_t)m * (size_t)n);
        w_main *= *w_main_multiplier;
    }

    /* This avoids differences in the scaling of the precomputed matrices */
    if (w_main != 1.)
    {
        lam /= w_main;
        w_user /= w_main;
        w_item /= w_main;
        if (lam_unique != NULL)
        {
            lam_unique_copy = (real_t*)malloc(6*sizeof(real_t));
            if (lam_unique_copy == NULL) goto throw_oom;
            for (int_t ix = 2; ix < 6; ix++) {
                lam_unique_copy[ix] = lam_unique[ix] / w_main;
            }
            lam_unique = lam_unique_copy;
        }
        w_main = 1.;
    }
    

    if (verbose) {
        printf("Starting ALS optimization routine\n\n");
        #if !defined(_FOR_R)
        fflush(stdout);
        #endif
    }

    for (int_t iter = 0; iter < niter; iter++)
    {
        if (iter == niter - 1 && use_cg && finalize_chol)
            use_cg = false;

        /* Optimize C and D (they are independent of each other) */
        if (handle_interrupt)
            signal(SIGINT, set_interrup_global_variable);
        if (should_stop_procedure) goto check_interrupt;
        if (U != NULL || nnz_U) {
            if (verbose) {
                printf("Updating C...");
                #if !defined(_FOR_R)
                fflush(stdout);
                #endif
            }

            if (k_item+k <= k_user+k+k_main)
                filled_BeTBeChol = false;
            
            optimizeA(
                C, k_user+k,
                A, k_user+k+k_main,
                p, m_u, k_user+k,
                U_csc_p, U_csc_i, U_csc,
                (Utrans != NULL)? (Utrans) : (U),
                (Utrans != NULL)? m_u : p,
                full_dense_u,
                near_dense_u_col,
                cnt_NA_u_bycol, (real_t*)NULL, NA_as_zero_U,
                (lam_unique == NULL)? (lam/w_user) : (lam_unique[4]/w_user),
                (lam_unique == NULL)? (lam/w_user) : (lam_unique[4]/w_user),
                (Utrans != NULL)? (false) : (true),
                iter == 0,
                nthreads,
                use_cg && !nonneg_C, max_cg_steps,
                nonneg_C, max_cd_steps, C_prev,
                false,
                precomputedBeTBeChol,
                &ignore,
                buffer_real_t
            );
            if (verbose) {
                printf(" done\n");
                #if !defined(_FOR_R)
                fflush(stdout);
                #endif
            }
        }

        if (handle_interrupt)
            signal(SIGINT, set_interrup_global_variable);
        if (should_stop_procedure) goto check_interrupt;
        if (II != NULL || nnz_I) {
            if (verbose) {
                printf("Updating D...");
                #if !defined(_FOR_R)
                fflush(stdout);
                #endif
            }

            filled_BeTBeChol = false;

            optimizeA(
                D, k_item+k,
                B, k_item+k+k_main,
                q, n_i, k_item+k,
                I_csc_p, I_csc_i, I_csc,
                (Itrans != NULL)? (Itrans) : (II),
                (Itrans != NULL)? n_i : q,
                full_dense_i,
                near_dense_i_col,
                cnt_NA_i_bycol, (real_t*)NULL, NA_as_zero_I,
                (lam_unique == NULL)? (lam/w_item) : (lam_unique[5]/w_item),
                (lam_unique == NULL)? (lam/w_item) : (lam_unique[5]/w_item),
                (Itrans != NULL)? (false) : (true),
                iter == 0,
                nthreads,
                use_cg && !nonneg_D, max_cg_steps,
                nonneg_D, max_cd_steps, D_prev,
                false,
                (k_item+k <= k_user+k+k_main)?
                    (precomputedBeTBeChol) : ((real_t*)NULL),
                &ignore,
                buffer_real_t
            );
            if (verbose) {
                printf(" done\n");
                #if !defined(_FOR_R)
                fflush(stdout);
                #endif
            }
        }

        /* Optimize B */
        if (handle_interrupt)
            signal(SIGINT, set_interrup_global_variable);
        if (should_stop_procedure) goto check_interrupt;
        if (verbose) {
            printf("Updating B...");
            #if !defined(_FOR_R)
            fflush(stdout);
            #endif
        }
        /* Precomputed matrices might get overwritten when solving for A,
           or the procedure may get interrupted */
        filled_BtB = false;
        filled_BeTBe = false;
        filled_BeTBeChol = false;
        filled_CtC = false;
        if (II != NULL || nnz_I)
            optimizeA_collective_implicit(
                B, A, D,
                n, n_i, m, q,
                k, k_main, k_item, k_user,
                Xcsc_p, Xcsc_i, Xcsc,
                I_csr_p, I_csr_i, I_csr,
                II, cnt_NA_i_byrow,
                full_dense_i, near_dense_i_row, NA_as_zero_I,
                (lam_unique == NULL)? (lam) : (lam_unique[3]),
                w_item,
                nthreads, use_cg && !nonneg, max_cg_steps, iter == 0,
                nonneg, max_cd_steps, B_prev,
                precomputedBtB,
                (k_item <= k_user)? (precomputedBeTBe) : ((real_t*)NULL),
                (k_item <= k_user)? (precomputedBeTBeChol) : ((real_t*)NULL),
                precomputedCtC,
                &ignore, &ignore, &ignore,
                buffer_real_t
            );
        else
            optimizeA_implicit(
                B + k_item, k_item+k+k_main,
                A + k_user, k_user+k+k_main,
                n, m, k+k_main,
                Xcsc_p, Xcsc_i, Xcsc,
                (lam_unique == NULL)? (lam) : (lam_unique[3]),
                nthreads, use_cg && !nonneg, max_cg_steps, iter == 0,
                nonneg, max_cd_steps, B_prev,
                precomputedBtB,
                buffer_real_t
            );
        if (verbose) {
            printf(" done\n");
            #if !defined(_FOR_R)
            fflush(stdout);
            #endif
        }

        /* Optimize A */
        if (handle_interrupt)
            signal(SIGINT, set_interrup_global_variable);
        if (should_stop_procedure) goto check_interrupt;
        if (verbose) {
            printf("Updating A...");
            #if !defined(_FOR_R)
            fflush(stdout);
            #endif
        }
        if (U != NULL || nnz_U)
            optimizeA_collective_implicit(
                A, B, C,
                m, m_u, n, p,
                k, k_main, k_user, k_item,
                Xcsr_p, Xcsr_i, Xcsr,
                U_csr_p, U_csr_i, U_csr,
                U, cnt_NA_u_byrow,
                full_dense_u, near_dense_u_row, NA_as_zero_U,
                (lam_unique == NULL)? (lam) : (lam_unique[2]),
                w_user,
                nthreads, use_cg && !nonneg, max_cg_steps, iter == 0,
                nonneg, max_cd_steps, A_prev,
                precomputedBtB,
                precomputedBeTBe,
                precomputedBeTBeChol,
                precomputedCtC,
                &filled_BeTBe, &filled_BeTBeChol,
                &filled_CtC,
                buffer_real_t
            );
        else
            optimizeA_implicit(
                A + k_user, k_user+k+k_main,
                B + k_item, k_item+k+k_main,
                m, n, k+k_main,
                Xcsr_p, Xcsr_i, Xcsr,
                (lam_unique == NULL)? (lam) : (lam_unique[2]),
                nthreads,
                use_cg && !nonneg, max_cg_steps, iter == 0,
                nonneg, max_cd_steps, A_prev,
                precomputedBtB,
                buffer_real_t
            );
        if (verbose) {
            printf(" done\n");
            #if !defined(_FOR_R)
            fflush(stdout);
            #endif
        }
        filled_BtB = true;

        
        if (verbose) {
            printf("\tCompleted ALS iteration %2d\n\n", iter+1);
            #if !defined(_FOR_R)
            fflush(stdout);
            #endif
        }
        check_interrupt:
            if (should_stop_procedure) {
                should_stop_procedure = false;
                if (precompute_for_predictions)
                {
                    printf("Finishing precomputed matrices...");
                    #if !defined(_FOR_R)
                    fflush(stdout);
                    #endif
                }
                goto cleanup;
            }

    }

    if (verbose) {
        if (!isnan(A[k_user]))
            printf("ALS procedure terminated successfully\n");
        else
            printf("ALS procedure failed\n");
        #if !defined(_FOR_R)
        fflush(stdout);
        #endif
    }

    if (precompute_for_predictions)
    {
        if (verbose) {
            printf("Finishing precomputed matrices...");
            #if !defined(_FOR_R)
            fflush(stdout);
            #endif
        }

        if (!filled_BtB)
        {
            cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        k+k_main, n,
                        1., B + k_item, k_totB,
                        0., precomputedBtB, k+k_main);
            add_to_diag(precomputedBtB, lam, k+k_main);
        }

        else if (use_cg) {
            add_to_diag(precomputedBtB, lam, k_totA);
        }

        if (!filled_BeTBe && (U != NULL || nnz_U))
        {
            set_to_zero(precomputedBeTBe, square(k_totA));
            
            if (filled_CtC)
            {
                if (w_user == 1.)
                    cblas_tscal(square(k_user+k), w_user, precomputedCtC, 1);

                copy_mat(k_user+k, k_user+k,
                         precomputedCtC, k_user+k,
                         precomputedBeTBe, k_totA);
            }

            else
            {
                cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                            k_user+k, p,
                            w_user, C, k_user+k,
                            0., precomputedBeTBe, k_totA);
            }

            sum_mat(k+k_main, k+k_main,
                    precomputedBtB, k+k_main,
                    precomputedBeTBe + k_user + k_user*k_totA, k_totA);

            for (int_t ix = 0; ix < k_user; ix++)
                precomputedBeTBe[ix + ix*k_totA] += lam;
        }

        if (!filled_BeTBeChol && (U != NULL || nnz_U) &&
            precomputedBeTBeChol != NULL)
        {
            copy_arr(precomputedBeTBe, precomputedBeTBeChol, square(k_totA));
            char lo = 'L';
            int_t ignore_int = 0;
            tpotrf_(&lo, &k_totA, precomputedBeTBeChol, &k_totA, &ignore_int);
        }
        

        if (verbose) {
            printf("  done\n");
            #if !defined(_FOR_R)
            fflush(stdout);
            #endif
        }
    }

    cleanup:
        free(buffer_real_t);
        free(A_prev);
        free(B_prev);
        free(C_prev);
        free(D_prev);
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
        if (!precompute_for_predictions)
            free(precomputedBtB);
        free(precomputedCtC);
        free(seed_arr);
        free(lam_unique_copy);
    return retval;

    throw_oom:
    {
        retval = 1;
        if (verbose)
            print_oom_message();
        goto cleanup;
    }
}

int_t precompute_collective_explicit
(
    real_t *restrict B, int_t n, int_t n_max, bool include_all_X,
    real_t *restrict C, int_t p,
    real_t *restrict Bi, bool add_implicit_features,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    bool user_bias,
    bool nonneg,
    real_t lam, real_t *restrict lam_unique,
    real_t w_main, real_t w_user, real_t w_implicit,
    real_t *restrict B_plus_bias,
    real_t *restrict BtB,
    real_t *restrict TransBtBinvBt,
    real_t *restrict BeTBeChol,
    real_t *restrict BiTBi,
    real_t *restrict TransCtCinvCt,
    real_t *restrict CtCw
)
{
    int_t retval = 0;
    char lo = 'L';
    int_t ignore = 0;
    if (n_max == 0) n_max = n;
    if (include_all_X) n = n_max;
    int k_main_i = k_main;
    int_t k_pred = 0;
    real_t lam_last = lam;
    if (lam_unique != NULL)
    {
        lam_last = lam_unique[user_bias? 0 : 2];
        lam = lam_unique[2];
    }

    if (w_main != 1.)
    {
        lam /= w_main;
        lam_last /= w_main;
        w_user /= w_main;
        w_implicit /= w_main;
    }

    real_t *arr_use = NULL;
    bool free_B_plus_bias = false;
    if (user_bias && B != NULL)
    {
        if (B_plus_bias == NULL)
        {
            free_B_plus_bias = true;
            B_plus_bias = (real_t*)malloc(  (size_t)n_max
                                         * (size_t)(k_item+k+k_main+1)
                                         * sizeof(real_t));
            if (B_plus_bias == NULL) goto throw_oom;
        }

        append_ones_last_col(
            B, n_max, k_item+k+k_main,
            B_plus_bias
        );
    }

    if (user_bias)
    {
        k_main++;
        B = B_plus_bias;
    }

    if (B != NULL)
    {
        set_to_zero(BtB, square(k+k_main));
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k+k_main, n,
                    1., B + k_item, k_item+k+k_main,
                    0., BtB, k+k_main);
    }

    if (Bi != NULL && add_implicit_features)
    {
        set_to_zero(BiTBi, square(k+k_main_i));
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k+k_main_i, n,
                    w_implicit, Bi, k+k_main_i,
                    0., BiTBi, k+k_main_i);
    }
    
    if (TransBtBinvBt != NULL && B != NULL && !nonneg && !add_implicit_features)
    {
        k_pred = k + k_main;
        if (BeTBeChol != NULL)
            arr_use = BeTBeChol; /* temporary */
        else {
            arr_use = (real_t*)malloc((size_t)square(k+k_main)*sizeof(real_t));
            if (arr_use == NULL) goto throw_oom;
        }

        copy_arr(BtB, arr_use, square(k+k_main));
        add_to_diag(arr_use, lam, k+k_main);
        if (lam != lam_last)
            arr_use[square(k+k_main)-1] += (lam_last - lam);
        copy_mat(n, k+k_main,
                 B + k_item, k_item+k+k_main,
                 TransBtBinvBt, k+k_main);
        tposv_(&lo, &k_pred, &n,
               arr_use, &k_pred,
               TransBtBinvBt, &k_pred, &ignore);

        if (arr_use != BeTBeChol)
        {
            free(arr_use);
            arr_use = NULL;
        }
    }

    if (p > 0 && C != NULL && CtCw != NULL)
    {
        set_to_zero(CtCw, square(k_user+k));
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k_user+k, p,
                    1., C, k_user+k,
                    0., CtCw, k_user+k);

        if (TransCtCinvCt != NULL && !add_implicit_features && !nonneg)
        {
            k_pred = k_user + k;
            copy_arr(C, TransCtCinvCt, (size_t)p*(size_t)(k_user+k));
            if (BeTBeChol != NULL)
                arr_use = BeTBeChol; /* temporary */
            else {
                arr_use = (real_t*)malloc((size_t)square(k_user+k)
                                          * sizeof(real_t));
                if (arr_use == NULL) goto throw_oom;
            }

            copy_arr(CtCw, arr_use, square(k_user+k));
            add_to_diag(arr_use, lam/w_user, k_user+k);
            tposv_(&lo, &k_pred, &p,
                   arr_use, &k_pred,
                   TransCtCinvCt, &k_pred, &ignore);

            if (arr_use != BeTBeChol)
            {
                free(arr_use);
                arr_use = NULL;
            }
        }

        if (w_user != 1.)
            cblas_tscal(square(k_user+k), w_user, CtCw, 1);
    }

    if (BeTBeChol != NULL && B != NULL &&
        (C != NULL || add_implicit_features) &&
        !nonneg)
    {
        set_to_zero(BeTBeChol, square(k_user+k+k_main));
        int_t k_totA = k_user + k + k_main;
        
        if (CtCw != NULL)
        {
            copy_mat(k+k_main, k+k_main,
                     BtB, k+k_main,
                     BeTBeChol + k_user + k_user*k_totA, k_totA);
            sum_mat(k_user+k, k_user+k,
                    CtCw, k_user+k,
                    BeTBeChol, k_totA);
        }

        else
        {
            if (p)
                cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                            k_user+k, p,
                            w_user, C, k_user+k,
                            0., BeTBeChol, k_totA);
            sum_mat(k+k_main, k+k_main,
                    BtB, k+k_main,
                    BeTBeChol + k_user + k_user*k_totA, k_totA);
        }

        if (add_implicit_features)
            sum_mat(k+k_main_i, k+k_main_i,
                    BiTBi, k+k_main_i,
                    BeTBeChol + k_user + k_user*k_totA, k_totA);

        add_to_diag(BeTBeChol, lam, k_user+k+k_main);
        if (lam != lam_last)
            BeTBeChol[square(k_user+k+k_main)-1] += (lam_last-lam);

        tpotrf_(&lo, &k_totA, BeTBeChol, &k_totA, &ignore);
    }

    cleanup:
        if (free_B_plus_bias)
            free(B_plus_bias);
        if (arr_use != BeTBeChol)
            free(arr_use);
        return retval;

    throw_oom:
    {
        retval = 1;
        print_oom_message();
        goto cleanup;
    }
}

int_t precompute_collective_implicit
(
    real_t *restrict B, int_t n,
    real_t *restrict C, int_t p,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    real_t lam, real_t w_main, real_t w_user, real_t w_main_multiplier,
    bool nonneg,
    bool extra_precision,
    real_t *restrict BtB,
    real_t *restrict BeTBe,
    real_t *restrict BeTBeChol
)
{
    if (w_main_multiplier != 1.)
        w_main *= w_main_multiplier;
    if (w_main != 1.)
    {
        lam /= w_main;
        w_user /= w_main;
    }

    /* BtB */
    set_to_zero(BtB, square(k+k_main));
    cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                k+k_main, n,
                1., B + k_item, k_item+k+k_main,
                0., BtB, k+k_main);
    add_to_diag(BtB, lam, k+k_main);

    if (!p)
        return 0;

    /* BeTBe */
    int_t k_totA = k_user + k + k_main;
    set_to_zero(BeTBe, square(k_totA));
    copy_mat(k+k_main, k+k_main,
             BtB, k+k_main,
             BeTBe + k_user + k_user*k_totA, k_totA);
    if (extra_precision)
    {
        real_t *restrict CtC = (real_t*)calloc((size_t)square(k_user+k),
                                               sizeof(real_t));
        if (CtC == NULL) return 1;

        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k_user+k, p,
                    w_user, C, k_user+k,
                    0., CtC, k_user+k);
        sum_mat(k_user+k, k_user+k,
                CtC, k_user+k,
                BeTBe, k_totA);
        free(CtC);
    }

    else
    {
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k_user+k, p,
                    w_user, C, k_user+k,
                    1., BeTBe, k_totA);
    }

    for (int_t ix = 0; ix < k_user; ix++)
            BeTBe[ix + ix*k_totA] += lam;

    /* BeTBeChol */
    if (BeTBeChol != NULL && !nonneg)
    {
        copy_arr(BeTBe, BeTBeChol, square(k_totA));
        char lo = 'L';
        int_t ignore = 0;
        tpotrf_(&lo, &k_totA, BeTBeChol, &k_totA, &ignore);
    }

    return 0;
}

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
    real_t w_main, real_t w_user, real_t w_implicit,
    int_t n_max, bool include_all_X,
    real_t *restrict TransBtBinvBt,
    real_t *restrict BtB,
    real_t *restrict BeTBeChol,
    real_t *restrict BiTBi,
    real_t *restrict CtCw,
    real_t *restrict TransCtCinvCt,
    real_t *restrict B_plus_bias
)
{
    int_t retval = 0;
    real_t lam_bias = lam;
    if (lam_unique != NULL)
    {
        lam_bias = lam_unique[(a_bias != NULL)? 0 : 2];
        lam = lam_unique[2];
    }

    bool set_to_nan = check_sparse_indices(
        (include_all_X || n == 0)? n_max : n, p,
        u_vec_sp, u_vec_ixB, nnz_u_vec,
        Xa, ixB, nnz
    );
    if (set_to_nan) {
        for (int_t ix = 0; ix < k_user+k+k_main; ix++)
            a_vec[ix] = NAN_;
        if (a_bias != NULL) *a_bias = NAN_;
        return 0;
    }

    #ifdef _FOR_R
    if (u_vec != NULL) R_nan_to_C_nan(u_vec, p);
    if (Xa_dense != NULL) R_nan_to_C_nan(Xa_dense, n);
    #endif

    if (u_vec == NULL && !nnz_u_vec && !NA_as_zero_U)
        p = 0;

    bool user_bias = (a_bias != NULL);
    bool free_B_plus_bias = false;
    if (user_bias && B_plus_bias == NULL)
    {
        free_B_plus_bias = true;
        B_plus_bias = (real_t*)malloc((size_t)(include_all_X? n_max : n)
                                      *(size_t)(k_item+k+k_main+1)
                                      * sizeof(real_t));
        if (B_plus_bias == NULL) goto throw_oom;
        append_ones_last_col(
            B, include_all_X? n_max : n, k_item+k+k_main,
            B_plus_bias
        );
    }

    if (!nnz && Xa_dense == NULL && !NA_as_zero_X && !add_implicit_features)
    {
        if (a_bias != NULL)
            *a_bias = 0.;
        retval = collective_factors_cold(
            a_vec,
            u_vec, p,
            u_vec_sp, u_vec_ixB, nnz_u_vec,
            u_bin_vec, pbin,
            C, Cb,
            TransCtCinvCt,
            CtCw,
            U_colmeans,
            k, k_user, k_main,
            lam, w_main, w_user,
            NA_as_zero_U,
            nonneg
        );
    }
    else
        retval = collective_factors_warm(
            a_vec, a_bias,
            u_vec, p,
            u_vec_sp, u_vec_ixB, nnz_u_vec,
            u_bin_vec, pbin,
            C, Cb,
            glob_mean, biasB,
            U_colmeans,
            Xa, ixB, nnz,
            Xa_dense, n,
            weight,
            B,
            Bi, add_implicit_features,
            k, k_user, k_item, k_main,
            lam, w_main, w_user, w_implicit, lam_bias,
            n_max, include_all_X,
            TransBtBinvBt,
            BtB,
            BeTBeChol,
            BiTBi,
            CtCw,
            NA_as_zero_U, NA_as_zero_X,
            nonneg,
            B_plus_bias
        );

    cleanup:
        if (free_B_plus_bias)
            free(B_plus_bias);
        return retval;
    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}

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
    real_t lam, real_t alpha, real_t w_main, real_t w_user,
    real_t w_main_multiplier,
    bool apply_log_transf,
    real_t *restrict BeTBe,
    real_t *restrict BtB,
    real_t *restrict BeTBeChol
)
{
    bool set_to_nan = check_sparse_indices(
        n, p,
        u_vec_sp, u_vec_ixB, nnz_u_vec,
        Xa, ixB, nnz
    );
    if (set_to_nan) {
        for (int_t ix = 0; ix < k_user+k+k_main; ix++)
            a_vec[ix] = NAN_;
        return 0;
    }

    #ifdef _FOR_R
    if (u_vec != NULL) R_nan_to_C_nan(u_vec, p);
    #endif

    int retval = 0;
    bool free_BtB = false;
    if (BtB == NULL) {
        free_BtB = true;
        BtB = (real_t*)malloc((size_t)square(k+k_main)*sizeof(real_t));
        if (BtB == NULL) return 1;
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k+k_main, n,
                    1., B + k_item, k_item+k+k_main,
                    0., BtB, k+k_main);
        add_to_diag(BtB, lam, k+k_main);
    }

    if (apply_log_transf)
        for (size_t ix = 0; ix < nnz; ix++)
            Xa[ix] = log_t(Xa[ix]);

    if (nnz)
        retval = collective_factors_warm_implicit(
            a_vec,
            u_vec, p,
            u_vec_sp, u_vec_ixB, nnz_u_vec,
            NA_as_zero_U,
            nonneg,
            U_colmeans,
            B, n, C,
            Xa, ixB, nnz,
            k, k_user, k_item, k_main,
            lam, alpha, w_main, w_user,
            w_main_multiplier,
            BeTBe,
            BtB,
            BeTBeChol
        );
    else
        retval = collective_factors_cold_implicit(
            a_vec,
            u_vec, p,
            u_vec_sp, u_vec_ixB, nnz_u_vec,
            B, n,
            C,
            BeTBe,
            BtB,
            BeTBeChol,
            U_colmeans,
            k, k_user, k_item, k_main,
            lam, w_main, w_user, w_main_multiplier,
            NA_as_zero_U,
            nonneg
        );

    if (free_BtB)
        free(BtB);
    return retval;
}

/* TODO: these functions should call 'optimizeA' instead */
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
    real_t w_main, real_t w_user, real_t w_implicit,
    int_t n_max, bool include_all_X,
    real_t *restrict TransBtBinvBt,
    real_t *restrict BtB,
    real_t *restrict BeTBeChol,
    real_t *restrict BiTBi,
    real_t *restrict TransCtCinvCt,
    real_t *restrict CtCw,
    real_t *restrict B_plus_bias,
    int_t nthreads
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif
    size_t lda = k_user+k+k_main;

    int_t retval = 0;
    size_t m_max = max2(m, m_u);
    if (NA_as_zero_U && U == NULL) m_u = m_max;
    if (NA_as_zero_X && Xfull == NULL) m = m_max;
    if (U == NULL && (!nnz_U && U_csr_p == NULL) && !NA_as_zero_U) m_u = 0;
    if (Xfull == NULL && (!nnz && Xcsr_p == NULL) && !NA_as_zero_X) m = 0;
    bool user_bias = (biasA != NULL);
    bool free_B_plus_bias = false;

    real_t *restrict weightR = NULL;
    bool free_U_csr = false;
    bool free_X_csr = false;
    bool free_BiTBi = false;
    int_t *restrict ret = (int_t*)malloc(m_max*sizeof(int_t));
    if (ret == NULL) goto throw_oom;

    if (user_bias && B_plus_bias == NULL)
    {
        free_B_plus_bias = true;
        B_plus_bias = (real_t*)malloc((size_t)n*(size_t)(k_item+k+k_main+1)
                                      * sizeof(real_t));
        if (B_plus_bias == NULL) goto throw_oom;
        append_ones_last_col(
            B, n, k_item+k+k_main,
            B_plus_bias
        );
    }

    if (Xfull == NULL && (nnz || NA_as_zero_X) && Xcsr_p == NULL)
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

    if (U == NULL && (nnz_U || NA_as_zero_U) && U_csr_p == NULL)
    {
        free_U_csr = true;
        U_csr_p = (size_t*)malloc(((size_t)m_u + (size_t)1) * sizeof(size_t));
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

    if (add_implicit_features && BiTBi == NULL)
    {
        free_BiTBi = true;
        BiTBi = (real_t*)malloc((size_t)square(k+k_main)*sizeof(real_t));
        if (BiTBi == NULL) goto throw_oom;
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k+k_main, n,
                    w_implicit,
                    Bi, k+k_main,
                    0., BiTBi, k+k_main);
    }

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
            shared(A, B, C, Cb, biasA, glob_mean, U_colmeans, \
                   Bi, add_implicit_features, \
                   Xfull, weight, Xcsr, Xcsr_p, Xcsr_i, weightR, m, n, \
                   U, U_csr, U_csr_p, U_csr_i, p, Ub, pbin, m_u, m_ubin, \
                   NA_as_zero_X, NA_as_zero_U, nonneg, m_max, \
                   lam, lam_unique, w_main, w_user, w_implicit, \
                   k, k_user, k_item, k_main, \
                   TransBtBinvBt, BtB, BeTBeChol, CtCw, TransCtCinvCt, \
                   B_plus_bias, BiTBi)
    for (size_t_for ix = 0; ix < m_max; ix++)
        ret[ix] = factors_collective_explicit_single(
            A + ix*lda,
            user_bias? (biasA + ix) : ((real_t*)NULL),
            (U == NULL || ix >= (size_t)m_u)?
                ((real_t*)NULL) : (U + ix*(size_t)p),
            (ix < (size_t)m_u)? p : 0,
            (ix < (size_t)m_u && U_csr_p != NULL)?
                (U_csr + U_csr_p[ix]) : ((real_t*)NULL),
            (ix < (size_t)m_u && U_csr_p != NULL)?
                (U_csr_i + U_csr_p[ix]) : ((int_t*)NULL),
            (ix < (size_t)m_u && U_csr_p != NULL)?
                (U_csr_p[ix+1] - U_csr_p[ix]) : ((size_t)0),
            (Ub == NULL || ix >= (size_t)m_ubin)?
                ((real_t*)NULL) : (Ub + ix*(size_t)pbin),
            (ix < (size_t)m_ubin)? pbin : 0,
            NA_as_zero_U, NA_as_zero_X,
            nonneg,
            C, Cb,
            glob_mean, biasB,
            U_colmeans,
            (ix < (size_t)m && Xcsr_p != NULL)?
                (Xcsr + Xcsr_p[ix]) : ((real_t*)NULL),
            (ix < (size_t)m && Xcsr_p != NULL)?
                (Xcsr_i + Xcsr_p[ix]) : ((int_t*)NULL),
            (ix < (size_t)m && Xcsr_p != NULL)?
                (Xcsr_p[ix+1] - Xcsr_p[ix]) : ((size_t)0),
            (Xfull == NULL || ix >= (size_t)m)?
                ((real_t*)NULL) : (Xfull + ix*(size_t)n),
            (ix < (size_t)m)? n : 0,
            (weight == NULL || ix >= (size_t)m)?
                ((real_t*)NULL)
                    :
                ((Xfull != NULL)?
                    (weight + ix*(size_t)n) : (weightR + Xcsr_p[ix])),
            B,
            Bi, add_implicit_features,
            k, k_user, k_item, k_main,
            lam, lam_unique,
            w_main, w_user, w_implicit,
            n_max, include_all_X,
            TransBtBinvBt,
            BtB,
            BeTBeChol,
            BiTBi,
            CtCw,
            TransCtCinvCt,
            B_plus_bias
        );

    for (size_t ix = 0; ix < m_max; ix++)
        retval = max2(retval, ret[ix]);
    if (retval == 1)
        goto throw_oom;
    else if (retval != 0)
        goto cleanup;

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
        if (free_B_plus_bias)
            free(B_plus_bias);
        if (free_BiTBi)
            free(BiTBi);
        free(ret);
        return retval;

    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}

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
    real_t lam, real_t alpha, real_t w_main, real_t w_user,
    real_t w_main_multiplier,
    bool apply_log_transf,
    real_t *restrict BeTBe,
    real_t *restrict BtB,
    real_t *restrict BeTBeChol,
    int_t nthreads
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long ix;
    #endif
    size_t lda = k_user+k+k_main;

    int_t retval = 0;
    m = max2(m, m_u);
    if (NA_as_zero_U && U == NULL) m_u = m;
    if (U == NULL && (!nnz_U && U_csr_p == NULL) && !NA_as_zero_U) m_u = 0;

    
    bool free_U_csr = false;
    bool free_X_csr = false;
    bool free_BtB = false;
    int_t *restrict ret = (int_t*)malloc(m*sizeof(int_t));
    if (ret == NULL) goto throw_oom;

    if (Xcsr_p == NULL)
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
            m, n, nnz,
            Xcsr_p, Xcsr_i, Xcsr,
            (real_t*)NULL
        );
    }

    if (U == NULL && (nnz_U || NA_as_zero_U) && U_csr_p == NULL)
    {
        free_U_csr = true;
        U_csr_p = (size_t*)malloc(((size_t)m_u + (size_t)1) * sizeof(size_t));
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

    if (BtB == NULL && m > 1 && BeTBeChol == NULL)
    {
        free_BtB = true;
        BtB = (real_t*)malloc((size_t)square(k+k_main)*sizeof(real_t));
        if (BtB == NULL) goto throw_oom;
        cblas_tsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    k+k_main, n,
                    1., B + k_item, k_item+k+k_main,
                    0., BtB, k+k_main);
        add_to_diag(BtB, lam, k+k_main);
    }


    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(A, B, C, m, m_u, U_colmeans, n, \
                   U, U_csr, U_csr_i, U_csr_p, NA_as_zero_U, nonneg, \
                   Xcsr, Xcsr_i, Xcsr_p, \
                   lam, alpha, w_main, w_user, w_main_multiplier, \
                   k, k_user, k_item, k_main, \
                   BtB, BeTBe, BeTBeChol)
    for (size_t_for ix = 0; ix < (size_t)m; ix++)
        ret[ix] = factors_collective_implicit_single(
            A + ix*lda,
            (U == NULL || ix >= (size_t)m_u)?
                ((real_t*)NULL) : (U + ix*(size_t)p),
            (ix < (size_t)m_u)? p : 0,
            (ix < (size_t)m_u && U_csr_p != NULL)?
                (U_csr + U_csr_p[ix]) : ((real_t*)NULL),
            (ix < (size_t)m_u && U_csr_p != NULL)?
                (U_csr_i + U_csr_p[ix]) : ((int_t*)NULL),
            (ix < (size_t)m_u && U_csr_p != NULL)?
                (U_csr_p[ix+1] - U_csr_p[ix]) : ((size_t)0),
            NA_as_zero_U,
            nonneg,
            U_colmeans,
            B, n, C,
            Xcsr + Xcsr_p[ix],
            Xcsr_i + Xcsr_p[ix],
            Xcsr_p[ix+1] - Xcsr_p[ix],
            k, k_user, k_item, k_main,
            lam, alpha, w_main, w_user,
            w_main_multiplier,
            apply_log_transf,
            BeTBe,
            BtB,
            BeTBeChol
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
            free(BtB);
        }
        free(ret);
        return retval;

    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}

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
    real_t w_main, real_t w_user, real_t w_implicit,
    int_t n_max, bool include_all_X,
    real_t *restrict TransBtBinvBt,
    real_t *restrict BtB,
    real_t *restrict BeTBeChol,
    real_t *restrict BiTBi,
    real_t *restrict TransCtCinvCt,
    real_t *restrict CtCw,
    real_t *restrict B_plus_bias,
    int_t nthreads
)
{
    #if defined(_OPENMP) && \
                ( (_OPENMP < 200801)  /* OpenMP < 3.0 */ \
                  || defined(_WIN32) || defined(_WIN64) \
                )
    long long row;
    #endif

    /* TODO: this function should first check which rows have missing values,
       and calculate the factors only for them. The imputation loop should
       also make a pre-check on the whole row to see if it has missing values.*/

    int_t retval = 0;
    size_t m_by_n = (size_t)m*(size_t)n;
    size_t lda = k_user + k + k_main;
    size_t ldb = k_item + k + k_main;
    size_t cnt_NA = 0;
    bool free_B_plus_bias = false;
    bool dont_produce_full_X = false;
    real_t *restrict Xpred = NULL;
    real_t *restrict A = (real_t*)malloc(  (size_t)max2(m, m_u)
                                         * (size_t)lda
                                         * sizeof(real_t));
    real_t *restrict biasA = NULL;
    if (user_bias) biasA = (real_t*)calloc((size_t)max2(m, m_u),sizeof(real_t));

    if (A == NULL || (biasA == NULL && user_bias))
        goto throw_oom;

    if (user_bias && B_plus_bias == NULL)
    {
        free_B_plus_bias = true;
        B_plus_bias = (real_t*)malloc((size_t)n*(size_t)(ldb + (size_t)1)
                                      * sizeof(real_t));
        if (B_plus_bias == NULL) goto throw_oom;
        append_ones_last_col(
            B, n, k_item+k+k_main,
            B_plus_bias
        );
    }

    for (size_t ix = 0; ix < m_by_n; ix++)
        cnt_NA += isnan(Xfull[ix]);
    dont_produce_full_X = (cnt_NA <= m_by_n / (size_t)10);
    if (cnt_NA == 0) goto cleanup;

    Xpred = (real_t*)malloc(m_by_n*sizeof(real_t));
    if (Xpred == NULL) goto throw_oom;
    copy_arr_(Xfull, Xpred, m_by_n, nthreads);

    retval = factors_collective_explicit_multiple(
        A, biasA, m,
        U, m_u, p,
        NA_as_zero_U, false,
        nonneg,
        U_row, U_col, U_sp, nnz_U,
        U_csr_p, U_csr_i, U_csr,
        Ub, m_ubin, pbin,
        C, Cb,
        glob_mean, biasB,
        U_colmeans,
        (real_t*)NULL, (int_t*)NULL, (int_t*)NULL, (size_t)0,
        (size_t*)NULL, (int_t*)NULL, (real_t*)NULL,
        Xpred, n,
        weight,
        B,
        Bi, add_implicit_features,
        k, k_user, k_item, k_main,
        lam, lam_unique,
        w_main, w_user, w_implicit,
        n_max, include_all_X,
        TransBtBinvBt,
        BtB,
        BeTBeChol,
        BiTBi,
        TransCtCinvCt,
        CtCw,
        B_plus_bias,
        nthreads
    );
    if (retval == 1)
        goto throw_oom;
    else if (retval != 0)
        goto cleanup;

    if (dont_produce_full_X)
    {
        #pragma omp parallel for collapse(2) \
                schedule(dynamic) num_threads(nthreads) \
                shared(m, n, Xfull, k, k_user, k_item, k_main, lda, ldb, \
                       glob_mean, user_bias, biasA, biasB, A, B)
        for (size_t_for row = 0; row < (size_t)m; row++)
            for (size_t col = 0; col < (size_t)n; col++)
                Xfull[col + row*(size_t)n]
                    =
                #ifndef _FOR_R
                (!isnan(Xfull[col + row*(size_t)n]))?
                #else
                (!isnan(Xfull[col + row*(size_t)n]) &&
                 !ISNAN(Xfull[col + row*(size_t)n]))?
                #endif
                    (Xfull[col + row*(size_t)n])
                        :
                    (
                        cblas_tdot(k+k_main,
                                   A + row*lda + (size_t)k_user, 1,
                                   B + col*ldb + (size_t)k_item, 1)
                        + glob_mean
                        + (user_bias? biasA[row] : 0.)
                        + ((biasB != NULL)? biasB[col] : 0.)
                    );
    }

    else
    {
        cblas_tgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m, n, k+k_main,
                    1., A + k_user, lda, B + k_item, ldb,
                    0., Xpred, n);
        #pragma omp parallel for collapse(2) \
                schedule(dynamic) num_threads(nthreads) \
                shared(m, n, Xfull, Xpred, glob_mean, user_bias, biasA, biasB)
        for (size_t_for row = 0; row < (size_t)m; row++)
            for (size_t col = 0; col < (size_t)n; col++)
                Xfull[col + row*(size_t)n]
                    =
                #ifndef _FOR_R
                (!isnan(Xfull[col + row*(size_t)n]))?
                #else
                (!isnan(Xfull[col + row*(size_t)n]) &&
                 !ISNAN(Xfull[col + row*(size_t)n]))?
                #endif
                    (Xfull[col + row*(size_t)n])
                        :
                    (Xpred[col + row*(size_t)n]
                        + glob_mean
                        + (user_bias? biasA[row] : 0.)
                        + ((biasB != NULL)? biasB[col] : 0.));
    }

    cleanup:
        if (Xpred != Xfull)
            free(Xpred);
        free(A);
        free(biasA);
        if (free_B_plus_bias)
            free(B_plus_bias);
        return retval;
    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}

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
    int_t n_top, int_t n, int_t n_max, bool include_all_X, int_t nthreads
)
{
    if (include_all_X || n == 0)
        n = n_max;
    if (a_vec != NULL)
        return topN(
            a_vec, k_user,
            B, k_item,
            biasB,
            glob_mean, a_bias,
            k, k_main,
            include_ix, n_include,
            exclude_ix, n_exclude,
            outp_ix, outp_score,
            n_top, n, nthreads
        );

    else
        return topN(
            A + (size_t)row_index*(size_t)(k_user+k+k_main), k_user,
            B, k_item,
            biasB,
            glob_mean, (biasA == NULL)? (0.) : (biasA[row_index]),
            k, k_main,
            include_ix, n_include,
            exclude_ix, n_exclude,
            outp_ix, outp_score,
            n_top, n, nthreads
        );
}

int_t topN_old_collective_implicit
(
    real_t *restrict a_vec,
    real_t *restrict A, int_t row_index,
    real_t *restrict B,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    int_t *restrict include_ix, int_t n_include,
    int_t *restrict exclude_ix, int_t n_exclude,
    int_t *restrict outp_ix, real_t *restrict outp_score,
    int_t n_top, int_t n, int_t nthreads
)
{
    return topN_old_collective_explicit(
        a_vec, 0.,
        A, (real_t*)NULL, row_index,
        B,
        (real_t*)NULL,
        0.,
        k, k_user, k_item, k_main,
        include_ix, n_include,
        exclude_ix, n_exclude,
        outp_ix, outp_score,
        n_top, n, n, false, nthreads
    );
}

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
    real_t w_main, real_t w_user, real_t w_implicit,
    int_t n_max, bool include_all_X,
    real_t *restrict TransBtBinvBt,
    real_t *restrict BtB,
    real_t *restrict BeTBeChol,
    real_t *restrict BiTBi,
    real_t *restrict CtCw,
    real_t *restrict TransCtCinvCt,
    real_t *restrict B_plus_bias,
    /* inputs for topN */
    int_t *restrict include_ix, int_t n_include,
    int_t *restrict exclude_ix, int_t n_exclude,
    int_t *restrict outp_ix, real_t *restrict outp_score,
    int_t n_top, int_t nthreads
)
{
    int_t retval = 0;
    real_t *restrict a_vec = (real_t*)malloc((size_t)(k_user+k+k_main)
                                             * sizeof(real_t));
    real_t a_bias = 0.;
    if (a_vec == NULL) goto throw_oom;

    retval = factors_collective_explicit_single(
        a_vec, user_bias? &a_bias : (real_t*)NULL,
        u_vec, p,
        u_vec_sp, u_vec_ixB, nnz_u_vec,
        u_bin_vec, pbin,
        NA_as_zero_U, NA_as_zero_X,
        nonneg,
        C, Cb,
        glob_mean, biasB,
        U_colmeans,
        Xa, ixB, nnz,
        Xa_dense, n,
        weight,
        B,
        Bi, add_implicit_features,
        k, k_user, k_item, k_main,
        lam, lam_unique,
        w_main, w_user, w_implicit,
        n_max, include_all_X,
        TransBtBinvBt,
        BtB,
        BeTBeChol,
        BiTBi,
        CtCw,
        TransCtCinvCt,
        B_plus_bias
    );
    if (retval == 1)
        goto throw_oom;
    else if (retval != 0)
        goto cleanup;

    retval = topN_old_collective_explicit(
        a_vec, a_bias,
        (real_t*)NULL, (real_t*)NULL, 0,
        B,
        biasB,
        glob_mean,
        k, k_user, k_item, k_main,
        include_ix, n_include,
        exclude_ix, n_exclude,
        outp_ix, outp_score,
        n_top, n, n_max, include_all_X, nthreads
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
    real_t lam, real_t alpha, real_t w_main, real_t w_user,
    real_t w_main_multiplier,
    bool apply_log_transf,
    real_t *restrict BeTBe,
    real_t *restrict BtB,
    real_t *restrict BeTBeChol,
    /* inputs for topN */
    int_t *restrict include_ix, int_t n_include,
    int_t *restrict exclude_ix, int_t n_exclude,
    int_t *restrict outp_ix, real_t *restrict outp_score,
    int_t n_top, int_t nthreads
)
{
    int_t retval = 0;
    real_t *restrict a_vec = (real_t*)malloc((size_t)(k_user+k+k_main)
                                             * sizeof(real_t));
    if (a_vec == NULL) goto throw_oom;

    retval = factors_collective_implicit_single(
        a_vec,
        u_vec, p,
        u_vec_sp, u_vec_ixB, nnz_u_vec,
        NA_as_zero_U,
        nonneg,
        U_colmeans,
        B, n, C,
        Xa, ixB, nnz,
        k, k_user, k_item, k_main,
        lam, alpha, w_main, w_user,
        w_main_multiplier,
        apply_log_transf,
        BeTBe,
        BtB,
        BeTBeChol
    );
    if (retval == 1)
        goto throw_oom;
    else if (retval != 0)
        goto cleanup;

    retval = topN_old_collective_implicit(
        a_vec,
        (real_t*)NULL, 0,
        B,
        k, k_user, k_item, k_main,
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

int_t predict_X_old_collective_explicit
(
    int_t row[], int_t col[], real_t *restrict predicted, size_t n_predict,
    real_t *restrict A, real_t *restrict biasA,
    real_t *restrict B, real_t *restrict biasB,
    real_t glob_mean,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    int_t m, int_t n_max,
    int_t nthreads
)
{
    predict_multiple(
        A, k_user,
        B, k_item,
        biasA, biasB,
        glob_mean,
        k, k_main,
        m, n_max,
        row, col, n_predict,
        predicted,
        nthreads
    );
    return 0;
}

int_t predict_X_old_collective_implicit
(
    int_t row[], int_t col[], real_t *restrict predicted, size_t n_predict,
    real_t *restrict A,
    real_t *restrict B,
    int_t k, int_t k_user, int_t k_item, int_t k_main,
    int_t m, int_t n,
    int_t nthreads
)
{
    predict_multiple(
        A, k_user,
        B, k_item,
        (real_t*)NULL, (real_t*)NULL,
        0.,
        k, k_main,
        m, n,
        row, col, n_predict,
        predicted,
        nthreads
    );
    return 0;
}

int_t predict_X_new_collective_explicit
(
    /* inputs for predictions */
    int_t m_new,
    int_t row[], int_t col[], real_t *restrict predicted, size_t n_predict,
    int_t nthreads,
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
    real_t w_main, real_t w_user, real_t w_implicit,
    int_t n_max, bool include_all_X,
    real_t *restrict TransBtBinvBt,
    real_t *restrict BtB,
    real_t *restrict BeTBeChol,
    real_t *restrict BiTBi,
    real_t *restrict TransCtCinvCt,
    real_t *restrict CtCw,
    real_t *restrict B_plus_bias
)
{
    int_t retval = 0;
    size_t m_max = max2(m_new, m_u);
    real_t *restrict biasA = NULL;
    real_t *restrict A = (real_t*)malloc(m_max * (size_t)(k_user+k+k_main)
                                         * sizeof(real_t));
    if (A == NULL) goto throw_oom;
    if (user_bias) {
        biasA = (real_t*)malloc(m_max * sizeof(real_t));
        if (biasA == NULL) goto throw_oom;
    }

    retval = factors_collective_explicit_multiple(
        A, biasA, m_new,
        U, m_u, p,
        NA_as_zero_U, NA_as_zero_X,
        nonneg,
        U_row, U_col, U_sp, nnz_U,
        U_csr_p, U_csr_i, U_csr,
        Ub, m_ubin, pbin,
        C, Cb,
        glob_mean, biasB,
        U_colmeans,
        X, ixA, ixB, nnz,
        Xcsr_p, Xcsr_i, Xcsr,
        Xfull, n,
        weight,
        B,
        Bi, add_implicit_features,
        k, k_user, k_item, k_main,
        lam, lam_unique,
        w_main, w_user, w_implicit,
        n_max, include_all_X,
        TransBtBinvBt,
        BtB,
        BeTBeChol,
        BiTBi,
        TransCtCinvCt,
        CtCw,
        B_plus_bias,
        nthreads
    );

    if (retval != 0)
        goto cleanup;

    retval = predict_X_old_collective_explicit(
        row, col, predicted, n_predict,
        A, biasA,
        B, biasB,
        glob_mean,
        k, k_user, k_item, k_main,
        m_max, n_max,
        nthreads
    );

    cleanup:
        free(A);
        free(biasA);
        return retval;
    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}

int_t predict_X_new_collective_implicit
(
    /* inputs for predictions */
    int_t m_new,
    int_t row[], int_t col[], real_t *restrict predicted, size_t n_predict,
    int_t nthreads,
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
    real_t lam, real_t alpha, real_t w_main, real_t w_user,
    real_t w_main_multiplier,
    bool apply_log_transf,
    real_t *restrict BeTBe,
    real_t *restrict BtB,
    real_t *restrict BeTBeChol
)
{
    int_t retval = 0;
    size_t m_max = max2(m_new, m_u);
    real_t *restrict A = (real_t*)malloc(m_max * (size_t)(k_user+k+k_main)
                                         * sizeof(real_t));
    if (A == NULL) goto throw_oom;

    retval = factors_collective_implicit_multiple(
        A, m_new,
        U, m_u, p,
        NA_as_zero_U,
        nonneg,
        U_row, U_col, U_sp, nnz_U,
        U_csr_p, U_csr_i, U_csr,
        X, ixA, ixB, nnz,
        Xcsr_p, Xcsr_i, Xcsr,
        B, n,
        C,
        U_colmeans,
        k, k_user, k_item, k_main,
        lam, alpha, w_main, w_user,
        w_main_multiplier,
        apply_log_transf,
        BeTBe,
        BtB,
        BeTBeChol,
        nthreads
    );

    if (retval == 1)
        goto throw_oom;
    else if (retval != 0)
        goto cleanup;

    retval = predict_X_old_collective_implicit(
        row, col, predicted, n_predict,
        A,
        B,
        k, k_user, k_item, k_main,
        m_max, n,
        nthreads
    );

    cleanup:
        free(A);
        return retval;
    throw_oom:
    {
        retval = 1;
        goto cleanup;
    }
}
