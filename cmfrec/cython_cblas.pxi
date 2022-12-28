from scipy.linalg.cython_blas cimport (
    ddot, dcopy, daxpy, dscal, dsyr, dsyrk, dnrm2, dgemm, dgemv, dsymv, dger,
    sdot, scopy, saxpy, sscal, ssyr, ssyrk, snrm2, sgemm, sgemv, ssymv, sger
)
from scipy.linalg.cython_lapack cimport (
    dlacpy, dposv, dpotrf, dpotrs, dgelsd,
    slacpy, sposv, spotrf, spotrs, sgelsd
)

ctypedef double (*ddot_)(const int*, const double*, const int*, const double*, const int*) nogil
ctypedef void (*dcopy_)(const int*, const double*, const int*, const double*, const int*) nogil
ctypedef void (*daxpy_)(const int*, const double*, const double*, const int*, double*, const int*) nogil
ctypedef void (*dscal_)(const int*, const double*, double*, const int*) nogil
ctypedef void (*dsyr_)(const char*, const int*, const double*, const double*, const int*, double*, const int*) nogil
ctypedef void (*dsyrk_)(const char*, const char*, const int*, const int*, const double*, const double*, const int*, const double*, double*, const int*) nogil
ctypedef double (*dnrm2_)(const int*, const double*, const int*) nogil
ctypedef void (*dgemm_)(const char*, const char*, const int*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*) nogil
ctypedef void (*dgemv_)(const char*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*) nogil
ctypedef void (*dsymv_)(const char*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*) nogil
ctypedef void (*dger_)(const int*, const int*, const double*, const double*, const int*, const double*, const int*, double*, const int*) nogil

ctypedef void (*dposv__)(const char*, const int*, const int*, double*, const int*, double*, const int*, int*) nogil
ctypedef void (*dlacpy__)(const char*, const int*, const int*, const double*, const int*, double*, const int*) nogil
ctypedef void (*dpotrf__)(const char*, const int*, double*, const int*, int*) nogil
ctypedef void (*dpotrs__)(const char*, const int*, const int*, const double*, const int*, double*, const int*, int*) nogil
ctypedef void (*dgelsd__)(const int*, const int*, const int*,
             double*, const int*,
             double*, const int*,
             double*, const double*, int*, double*,
             const int*, int*, int*) nogil


ctypedef float (*sdot_)(const int*, const float*, const int*, const float*, const int*) nogil
ctypedef void (*scopy_)(const int*, const float*, const int*, const float*, const int*) nogil
ctypedef void (*saxpy_)(const int*, const float*, const float*, const int*, float*, const int*) nogil
ctypedef void (*sscal_)(const int*, const float*, float*, const int*) nogil
ctypedef void (*ssyr_)(const char*, const int*, const float*, const float*, const int*, float*, const int*) nogil
ctypedef void (*ssyrk_)(const char*, const char*, const int*, const int*, const float*, const float*, const int*, const float*, float*, const int*) nogil
ctypedef float (*snrm2_)(const int*, const float*, const int*) nogil
ctypedef void (*sgemm_)(const char*, const char*, const int*, const int*, const int*, const float*, const float*, const int*, const float*, const int*, const float*, float*, const int*) nogil
ctypedef void (*sgemv_)(const char*, const int*, const int*, const float*, const float*, const int*, const float*, const int*, const float*, float*, const int*) nogil
ctypedef void (*ssymv_)(const char*, const int*, const float*, const float*, const int*, const float*, const int*, const float*, float*, const int*) nogil
ctypedef float (*sger_)(const int*, const int*, const float*, const float*, const int*, const float*, const int*, float*, const int*) nogil

ctypedef void (*sposv__)(const char*, const int*, const int*, float*, const int*, float*, const int*, int*) nogil
ctypedef void (*slacpy__)(const char*, const int*, const int*, const float*, const int*, float*, const int*) nogil
ctypedef void (*spotrf__)(const char*, const int*, float*, const int*, int*) nogil
ctypedef void (*spotrs__)(const char*, const int*, const int*, const float*, const int*, float*, const int*, int*) nogil
ctypedef void (*sgelsd__)(const int*, const int*, const int*,
             float*, const int*,
             float*, const int*,
             float*, const float*, int*, float*,
             const int*, int*, int*) nogil

ctypedef enum CBLAS_ORDER:
    CblasRowMajor = 101
    CblasColMajor = 102

ctypedef CBLAS_ORDER CBLAS_LAYOUT

ctypedef enum cblas_TRANSPOSE:
    CblasNoTrans=111
    CblasTrans=112
    CblasConjTrans=113
    CblasConjNoTrans=114

ctypedef enum CBLAS_UPLO:
    CblasUpper=121
    CblasLower=122

ctypedef enum CBLAS_DIAG:
    CblasNonUnit=131
    CblasUnit=132

ctypedef enum CBLAS_SIDE:
    CblasLeft=141
    CblasRight=142

cdef public double cblas_ddot(const int n, const double  *x, const int incx, const double  *y, const int incy) nogil:
    return (<ddot_>ddot)(&n, x, &incx, y, &incy)

cdef public void cblas_dcopy(const int n, const double *x, const int incx, double *y, const int incy) nogil:
    (<dcopy_>dcopy)(&n, x, &incx, y, &incy)

cdef public void cblas_daxpy(const int n, const double alpha, const double *x, const int incx, double *y, const int incy) nogil:
    (<daxpy_>daxpy)(&n, &alpha, x, &incx, y, &incy)

cdef public void cblas_dscal(const int N, const double alpha, double *X, const int incX) nogil:
    (<dscal_>dscal)(&N, &alpha, X, &incX)

cdef public void cblas_dsyr(const int order, const int Uplo, const int N, const double alpha, const double *X, const int incX, double *A, const int lda) nogil:
    cdef char uplo = 0#'\0'
    if (order == CblasColMajor):
        if (Uplo == CblasLower):
            uplo = 76#'L'
        else:
            uplo = 85#'U'

    else:
        if (Uplo == CblasLower):
            uplo = 85#'U'
        else:
            uplo = 76#'L'
    (<dsyr_>dsyr)(&uplo, &N, &alpha, X, &incX, A, &lda)

cdef public void cblas_dsyrk(const int Order, const int Uplo, const int Trans,
         const int N, const int K, const double alpha, const double *A, const int lda, const double beta, double *C, const int ldc) nogil:
    cdef char uplo = 0#'\0'
    cdef char trans = 0#'\0'
    if (Order == CblasColMajor):
        if (Uplo == CblasUpper):
            uplo = 85#'U'
        else:
            uplo = 76#'L'

        if (Trans == CblasTrans):
            trans = 84#'T'
        elif (Trans == CblasConjTrans):
            trans = 67#'C'
        else:
            trans = 78#'N'

    else:
        if (Uplo == CblasUpper):
            uplo = 76#'L'
        else:
            uplo = 85#'U'

        if (Trans == CblasTrans):
            trans = 78#'N'
        elif (Trans == CblasConjTrans):
            trans = 78#'N'
        else:
            trans = 84#'T'

    (<dsyrk_>dsyrk)(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc)

cdef public double cblas_dnrm2 (const int N, const double  *X, const int incX) nogil:
    return (<dnrm2_>dnrm2)(&N, X, &incX)

cdef public void cblas_dgemm(const int Order, const int TransA, const int TransB, const int M, const int N, const int K,
         const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc) nogil:
    cdef char transA = 0#'\0'
    cdef char transB = 0#'\0'
    if (Order == CblasColMajor):
        if (TransA == CblasTrans):
            transA = 84#'T'
        elif (TransA == CblasConjTrans):
            transA = 67#'C'
        else:
            transA = 78#'N'

        if (TransB == CblasTrans):
            transB = 84#'T'
        elif (TransB == CblasConjTrans):
            transB = 67#'C'
        else:
            transB = 78#'N'

        (<dgemm_>dgemm)(&transA, &transB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

    else:
        if (TransA == CblasTrans):
            transB = 84#'T'
        elif (TransA == CblasConjTrans):
            transB = 67#'C'
        else:
            transB = 78#'N'

        if (TransB == CblasTrans):
            transA = 84#'T'
        elif (TransB == CblasConjTrans):
            transA = 67#'C'
        else:
            transA = 78#'N'

        (<dgemm_>dgemm)(&transA, &transB, &N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C, &ldc)

cdef public void cblas_dgemv(const int order,  const int TransA,  const int m, const int n,
         const double alpha, const double  *a, const int lda,  const double  *x, const int incx,  const double beta,  double  *y, const int incy) nogil:
    cdef char trans = 0#'\0'
    if (order == CblasColMajor):
        if (TransA == CblasNoTrans):
            trans = 78#'N'
        elif (TransA == CblasTrans):
            trans = 84#'T'
        else:
            trans = 67#'C'

        (<dgemv_>dgemv)(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);

    else:
        if (TransA == CblasNoTrans):
            trans = 84#'T'
        elif (TransA == CblasTrans):
            trans = 78#'N'
        else:
            trans = 78#'N'

        (<dgemv_>dgemv)(&trans, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);

cdef public void cblas_dsymv(const int order, const int Uplo, const int N, const double alpha, const double *A,
                 const int lda, const double *X, const int incX, const double beta, double *Y, const int incY) nogil:
    cdef char uplo = 0#'\0';
    if (order == CblasColMajor):
        if (Uplo == CblasUpper):
            uplo = 85#'U'
        else:
            uplo = 76#'L'

    else:
        if (Uplo == CblasUpper):
            uplo = 76#'L'
        else:
            uplo = 85#'U'

    (<dsymv_>dsymv)(&uplo, &N, &alpha, A, &lda, X, &incX, &beta, Y, &incY)

cdef public void cblas_dger(const int order, const int m, const int n, const double alpha,
                const double *x, const int incx, const double *y, const int incy, double *a, const int lda) nogil:
    if (order == CblasColMajor):
        (<dger_>dger)(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);

    else:
        (<dger_>dger)(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);

cdef public void dposv_(const char* uplo, const int* m, const int* n, double* x, const int* ldx, double* y, const int* ldy, int* info) nogil:
    (<dposv__>dposv)(uplo, m, n, x, ldx, y, ldy, info)

cdef public void dlacpy_(const char* uplo, const int* m, const int* n, const double* x, const int* ldx, double* y, const int* ldy) nogil:
    (<dlacpy__>dlacpy)(uplo, m, n, x, ldx, y, ldy)

cdef public void dpotrf_(const char* a1, const int* a2, double* a3, const int* a4, int* a5) nogil:
    (<dpotrf__>dpotrf)(a1, a2, a3, a4, a5)

cdef public void dpotrs_(const char* a1, const int* a2, const int* a3, const double* a4, const int* a5, double* a6, const int* a7, int* a8) nogil:
    (<dpotrs__>dpotrs)(a1, a2, a3, a4, a5, a6, a7, a8)

cdef public void dgelsd_(const int* a1, const int* a2, const int* a3,
             double* a4, const int* a5,
             double* a6, const int* a7,
             double* a8, const double* a9, int* a10, double* a11,
             const int* a12, int* a13, int* a14) nogil:
    (<dgelsd__>dgelsd)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14)

########################

cdef public float cblas_sdot(const int n, const float  *x, const int incx, const float  *y, const int incy) nogil:
    return (<sdot_>sdot)(&n, x, &incx, y, &incy)

cdef public void cblas_scopy(const int n, const float *x, const int incx, float *y, const int incy) nogil:
    (<scopy_>scopy)(&n, x, &incx, y, &incy)

cdef public void cblas_saxpy(const int n, const float alpha, const float *x, const int incx, float *y, const int incy) nogil:
    (<saxpy_>saxpy)(&n, &alpha, x, &incx, y, &incy)

cdef public void cblas_sscal(const int N, const float alpha, float *X, const int incX) nogil:
    (<sscal_>sscal)(&N, &alpha, X, &incX)

cdef public void cblas_ssyr(const int order, const int Uplo, const int N, const float alpha, const float *X, const int incX, float *A, const int lda) nogil:
    cdef char uplo = 0#'\0'
    if (order == CblasColMajor):
        if (Uplo == CblasLower):
            uplo = 76#'L'
        else:
            uplo = 85#'U'

    else:
        if (Uplo == CblasLower):
            uplo = 85#'U'
        else:
            uplo = 76#'L'
    (<ssyr_>ssyr)(&uplo, &N, &alpha, X, &incX, A, &lda)

cdef public void cblas_ssyrk(const int Order, const int Uplo, const int Trans,
         const int N, const int K, const float alpha, const float *A, const int lda, const float beta, float *C, const int ldc) nogil:
    cdef char uplo = 0#'\0'
    cdef char trans = 0#'\0'
    if (Order == CblasColMajor):
        if (Uplo == CblasUpper):
            uplo = 85#'U'
        else:
            uplo = 76#'L'

        if (Trans == CblasTrans):
            trans = 84#'T'
        elif (Trans == CblasConjTrans):
            trans = 67#'C'
        else:
            trans = 78#'N'

    else:
        if (Uplo == CblasUpper):
            uplo = 76#'L'
        else:
            uplo = 85#'U'

        if (Trans == CblasTrans):
            trans = 78#'N'
        elif (Trans == CblasConjTrans):
            trans = 78#'N'
        else:
            trans = 84#'T'

    (<ssyrk_>ssyrk)(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc)

cdef public float cblas_snrm2 (const int N, const float  *X, const int incX) nogil:
    return (<snrm2_>snrm2)(&N, X, &incX)

cdef public void cblas_sgemm(const int Order, const int TransA, const int TransB, const int M, const int N, const int K,
         const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc) nogil:
    cdef char transA = 0#'\0'
    cdef char transB = 0#'\0'
    if (Order == CblasColMajor):
        if (TransA == CblasTrans):
            transA = 84#'T'
        elif (TransA == CblasConjTrans):
            transA = 67#'C'
        else:
            transA = 78#'N'

        if (TransB == CblasTrans):
            transB = 84#'T'
        elif (TransB == CblasConjTrans):
            transB = 67#'C'
        else:
            transB = 78#'N'

        (<sgemm_>sgemm)(&transA, &transB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

    else:
        if (TransA == CblasTrans):
            transB = 84#'T'
        elif (TransA == CblasConjTrans):
            transB = 67#'C'
        else:
            transB = 78#'N'

        if (TransB == CblasTrans):
            transA = 84#'T'
        elif (TransB == CblasConjTrans):
            transA = 67#'C'
        else:
            transA = 78#'N'

        (<sgemm_>sgemm)(&transA, &transB, &N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C, &ldc)

cdef public void cblas_sgemv(const int order,  const int TransA,  const int m, const int n,
         const float alpha, const float  *a, const int lda,  const float  *x, const int incx,  const float beta,  float  *y, const int incy) nogil:
    cdef char trans = 0#'\0'
    if (order == CblasColMajor):
        if (TransA == CblasNoTrans):
            trans = 78#'N'
        elif (TransA == CblasTrans):
            trans = 84#'T'
        else:
            trans = 67#'C'

        (<sgemv_>sgemv)(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);

    else:
        if (TransA == CblasNoTrans):
            trans = 84#'T'
        elif (TransA == CblasTrans):
            trans = 78#'N'
        else:
            trans = 78#'N'

        (<sgemv_>sgemv)(&trans, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);

cdef public void cblas_ssymv(const int order, const int Uplo, const int N, const float alpha, const float *A,
                 const int lda, const float *X, const int incX, const float beta, float *Y, const int incY) nogil:
    cdef char uplo = 0#'\0';
    if (order == CblasColMajor):
        if (Uplo == CblasUpper):
            uplo = 85#'U'
        else:
            uplo = 76#'L'

    else:
        if (Uplo == CblasUpper):
            uplo = 76#'L'
        else:
            uplo = 85#'U'

    (<ssymv_>ssymv)(&uplo, &N, &alpha, A, &lda, X, &incX, &beta, Y, &incY)

cdef public void cblas_sger(const int order, const int m, const int n, const float alpha,
                const float *x, const int incx, const float *y, const int incy, float *a, const int lda) nogil:
    if (order == CblasColMajor):
        (<sger_>sger)(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);

    else:
        (<sger_>sger)(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);

cdef public void sposv_(const char* uplo, const int* m, const int* n, float* x, const int* ldx, float* y, const int* ldy, int* info) nogil:
    (<sposv__>sposv)(uplo, m, n, x, ldx, y, ldy, info)

cdef public void slacpy_(const char* uplo, const int* m, const int* n, const float* x, const int* ldx, float* y, const int* ldy) nogil:
    (<slacpy__>slacpy)(uplo, m, n, x, ldx, y, ldy)

cdef public void spotrf_(const char* a1, const int* a2, float* a3, const int* a4, int* a5) nogil:
    (<spotrf__>spotrf)(a1, a2, a3, a4, a5)

cdef public void spotrs_(const char* a1, const int* a2, const int* a3, const float* a4, const int* a5, float* a6, const int* a7, int* a8) nogil:
    (<spotrs__>spotrs)(a1, a2, a3, a4, a5, a6, a7, a8)

cdef public void sgelsd_(const int* a1, const int* a2, const int* a3,
             float* a4, const int* a5,
             float* a6, const int* a7,
             float* a8, const float* a9, int* a10, float* a11,
             const int* a12, int* a13, int* a14) nogil:
    (<sgelsd__>sgelsd)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14)

