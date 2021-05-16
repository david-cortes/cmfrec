/*******************************************************************************
    Collective Matrix Factorization
    -------------------------------
    
    This is a module for multi-way factorization of sparse and dense matrices
    int_tended to be used for recommender system with explicit feedback data plus
    side information about users and/or items.

    The reference papers are:
        (a) Cortes, David.
            "Cold-start recommendations in Collective Matrix Factorization."
            arXiv preprint arXiv:1809.00366 (2018).
        (b) Singh, Ajit P., and Geoffrey J. Gordon.
            "Relational learning via collective matrix factorization."
            Proceedings of the 14th ACM SIGKDD int_ternational conference on
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
    used either as a stand-alone program, or wrapped int_to scripting languages
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

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CBLAS_H

#if !defined(USE_FLOAT)
    #define tdot_ ddot_
    #define tcopy_ dcopy_
    #define taxpy_ daxpy_
    #define tscal_ dscal_
    #define tsyr_ dsyr_
    #define tsyrk_ dsyrk_
    #define tnrm2_ dnrm2_
    #define tgemm_ dgemm_
    #define tgemv_ dgemv_
    #define tsymv_ dsymv_
    #define tger_ dger_
#else
    #define tdot_ sdot_
    #define tcopy_ scopy_
    #define taxpy_ saxpy_
    #define tscal_ sscal_
    #define tsyr_ ssyr_
    #define tsyrk_ ssyrk_
    #define tnrm2_ snrm2_
    #define tgemm_ sgemm_
    #define tgemv_ sgemv_
    #define tsymv_ ssymv_
    #define tger_ sger_
#endif

#ifndef _FOR_R
real_t tdot_(const int_t*, const real_t*, const int_t*, const real_t*, const int_t*);
void tcopy_(const int_t*, const real_t*, const int_t*, const real_t*, const int_t*);
void taxpy_(const int_t*, const real_t*, const real_t*, const int_t*, const real_t*, const int_t*);
void tscal_(const int_t*, const real_t*, const real_t*, const int_t*);
void tsyr_(const char*, const int_t*, const real_t*, const real_t*, const int_t*, const real_t*, const int_t*);
void tsyrk_(const char*, const char*, const int_t*, const int_t*, const real_t*, const real_t*, const int_t*, const real_t*, const real_t*, const int_t*);
real_t tnrm2_(const int_t*, const real_t*, const int_t*);
void tgemm_(const char*, const char*, const int_t*, const int_t*, const int_t*, const real_t*, const real_t*, const int_t*, const real_t*, const int_t*, const real_t*, const real_t*, const int_t*);
void tgemv_(const char*, const int_t*, const int_t*, const real_t*, const real_t*, const int_t*, const real_t*, const int_t*, const real_t*, const real_t*, const int_t*);
void tsymv_(const char*, const int_t*, const real_t*, const real_t*, const int_t*, const real_t*, const int_t*, const real_t*, const real_t*, const int_t*);
void tger_(const int*, const int*, const real_t*, const real_t*, const int*, const real_t*, const int*, const real_t*, const int*);
#endif


real_t  cblas_tdot(const int_t n, const real_t  *x, const int_t incx, const real_t  *y, const int_t incy)
{
    return tdot_(&n, x, &incx, y, &incy);
}
void cblas_tcopy(const int_t n, const real_t *x, const int_t incx, real_t *y, const int_t incy)
{
    tcopy_(&n, x, &incx, y, &incy);
}
void cblas_taxpy(const int_t n, const real_t alpha, const real_t *x, const int_t incx, real_t *y, const int_t incy)
{
    taxpy_(&n, &alpha, x, &incx, y, &incy);
}
void cblas_tscal(const int_t N, const real_t alpha, real_t *X, const int_t incX)
{
    tscal_(&N, &alpha, X, &incX);
}
void cblas_tsyr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const int_t N, const real_t alpha, const real_t *X, const int_t incX, real_t *A, const int_t lda)
{
    char uplo = '\0';
    if (order == CblasColMajor)
    {
        if (Uplo == CblasLower)
            uplo = 'L';
        else
            uplo = 'U';
    }

    else
    {
        if (Uplo == CblasLower)
            uplo = 'U';
        else
            uplo = 'L';
    }
    tsyr_(&uplo, &N, &alpha, X, &incX, A, &lda);
}
void cblas_tsyrk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans,
         const int_t N, const int_t K, const real_t alpha, const real_t *A, const int_t lda, const real_t beta, real_t *C, const int_t ldc)
{
    char uplo = '\0';
    char trans = '\0';
    if (Order == CblasColMajor)
    {
        if (Uplo == CblasUpper)
            uplo = 'U';
        else
            uplo = 'L';

        if (Trans == CblasTrans)
            trans = 'T';
        else if (Trans == CblasConjTrans)
            trans = 'C';
        else
            trans = 'N';
    }

    else
    {
        if (Uplo == CblasUpper)
            uplo = 'L';
        else
            uplo = 'U';

        if (Trans == CblasTrans)
            trans = 'N';
        else if (Trans == CblasConjTrans)
            trans = 'N';
        else
            trans = 'T';
    }

    tsyrk_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc);
}
real_t  cblas_tnrm2 (const int_t N, const real_t  *X, const int_t incX)
{
    return tnrm2_(&N, X, &incX);
}
void cblas_tgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int_t M, const int_t N, const int_t K,
         const real_t alpha, const real_t *A, const int_t lda, const real_t *B, const int_t ldb, const real_t beta, real_t *C, const int_t ldc)
{
    char transA = '\0';
    char transB = '\0';
    if (Order == CblasColMajor)
    {
        if (TransA == CblasTrans)
            transA = 'T';
        else if (TransA == CblasConjTrans)
            transA = 'C';
        else
            transA = 'N';

        if (TransB == CblasTrans)
            transB = 'T';
        else if (TransB == CblasConjTrans)
            transB = 'C';
        else
            transB = 'N';

        tgemm_(&transA, &transB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
    }

    else
    {
        if (TransA == CblasTrans)
            transB = 'T';
        else if (TransA == CblasConjTrans)
            transB = 'C';
        else
            transB = 'N';

        if (TransB == CblasTrans)
            transA = 'T';
        else if (TransB == CblasConjTrans)
            transA = 'C';
        else
            transA = 'N';

        tgemm_(&transA, &transB, &N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
    }
}
void cblas_tgemv(const CBLAS_ORDER order,  const CBLAS_TRANSPOSE TransA,  const int_t m, const int_t n,
         const real_t alpha, const real_t  *a, const int_t lda,  const real_t  *x, const int_t incx,  const real_t beta,  real_t  *y, const int_t incy)
{
    char trans = '\0';
    if (order == CblasColMajor)
    {
        if (TransA == CblasNoTrans)
            trans = 'N';
        else if (TransA == CblasTrans)
            trans = 'T';
        else
            trans = 'C';

        tgemv_(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }

    else
    {
        if (TransA == CblasNoTrans)
            trans = 'T';
        else if (TransA == CblasTrans)
            trans = 'N';
        else
            trans = 'N';

        tgemv_(&trans, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }
}
void cblas_tsymv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const int_t N, const real_t alpha, const real_t *A,
                 const int_t lda, const real_t *X, const int_t incX, const real_t beta, real_t *Y, const int_t incY)
{
    char uplo = '\0';
    if (order == CblasColMajor)
    {
        if (Uplo == CblasUpper)
            uplo = 'U';
        else
            uplo = 'L';
    }

    else
    {
        if (Uplo == CblasUpper)
            uplo = 'L';
        else
            uplo = 'U';
    }

    tsymv_(&uplo, &N, &alpha, A, &lda, X, &incX, &beta, Y, &incY);
}

void cblas_tger(const CBLAS_ORDER order, const int_t m, const int_t n, const real_t alpha,
                const real_t *x, const int_t incx, const real_t *y, const int_t incy, real_t *a, const int_t lda)
{
    if (order == CblasColMajor)
    {
        tger_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
    }

    else
    {
        tger_(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
    }
}


#endif

#ifdef __cplusplus
}
#endif
