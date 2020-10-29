/* Short example usage of the cmfrec C library.
   
   This example factorizes a small random sparse matrix
   X = [[.  , 3  , .  , .  , 3  , 4  , 5  , .  ],
        [.  , .  , 2  , .  , .  , .  , .  , .  ],
        [1  , .  , .  , 5  , .  , 1  , .  , .  ],
        [.  , .  , .  , .  , 4  , 2  , .  , 3  ],
        [.  , 3  , .  , .  , .  , .  , .  , .  ],
        [1  , .  , 3  , 2  , .  , .  , 5  , .  ]]

   (Dimension is [6, 8])

   Into the product of two smaller matrices A[6,3] and B[8,3],
   by minimizing the squared error with respect to the non-zero
   entries.

   The "X" matrix is centered within the procedure by substracting its mean.

   The example then prints the obtained factor matrices and the values of
   "X" that are approximates with them.

   To run the example, build the library through the CMake system, and then
   compile this file linking to it. Examples:
     gcc c_example.c -lcmfrec -std=c99 (from 'example/', system wide install)
     gcc example/c_example.c -std=c99 \
        -I./build -L./build -l:libcmfrec.so -Wl,-rpath,./build
        (form root folder, assuming the package was built under 'build/''
         but not installed)
   Then execute the produced file - e.g.
     ./a.out
 */ 
#include "cmfrec.h"
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>

void print_sparse_matrix(double X[], int Xrow[], int Xcol[], int m, int n, size_t nnz);
void print_dense_matrix(double X[], int m, int n);

int main()
{
    printf("CMFREC short example\n\n\n");

    /* Constructing the X matrix in COO/triplets format */
    double X[] = {3, 3, 4, 5,
                  2,
                  1, 5, 1,
                  4, 2, 3,
                  3,
                  1, 3, 2, 5};
    int Xrow[] = {0, 0, 0, 0,
                  1,
                  2, 2, 2,
                  3, 3, 3,
                  4,
                  5, 5, 5, 5};
    int Xcol[] = {1, 4, 5, 6,
                  2,
                  0, 3, 5,
                  4, 5, 7,
                  1,
                  0, 2, 3, 6};
    int m = 6;
    int n = 8;
    size_t nnz = 16;

    printf("Input matrix X[%d, %d]:\n", m, n);
    print_sparse_matrix(X, Xrow, Xcol, m, n, nnz);
    printf("\nWill be factorized into the product:\n");
    printf("\tX ~ A * t(B) + mean\n\n");

    /* Now factorize it */
    int k = 3;
    double *A = (double*)malloc(m*k*sizeof(double));
    double *B = (double*)malloc(n*k*sizeof(double));
    double glob_mean;
    int seed = 123;
    bool reset_values = true;
    double regularization = 1e-1;
    bool user_bias = false;
    bool item_bias = false;
    bool use_cg = true;
    int max_cg_steps = 3;
    int niter = 10;
    bool finalize_chol = true;

    /* Important: the "X" data will be modified in-place while
       the model is being fit. The values on exit will be
       different than on input */
    fit_collective_explicit_als(
        NULL, NULL,
        A, B,
        NULL, NULL,
        reset_values, seed,
        &glob_mean,
        NULL, NULL,
        m, n, k,
        Xrow, Xcol, X, nnz,
        NULL,
        NULL,
        user_bias, item_bias,
        regularization, NULL,
        NULL, 0, 0,
        NULL, 0, 0,
        NULL, NULL, NULL, 0,
        NULL, NULL, NULL, 0,
        false, false, false,
        0, 0, 0,
        1., 0., 0.,
        10, 1, false, false,
        use_cg, max_cg_steps, finalize_chol,
        false,
        false,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL
    );

    /* Taking a look at the results */
    printf("\nObtained factor matrices:\n\n");

    printf("Matrix A[%d, %d]:\n", m, k);
    print_dense_matrix(A, m, k);

    printf("\nMatrix B[%d, %d]:\n", n, k);
    print_dense_matrix(B, n, k);

    printf("\nGlobal mean: %.2f\n", glob_mean);


    /* See what the approximation A*t(B) + mu for 'X' is */
    double *X_predicted = (double*)malloc(nnz*sizeof(double));
    predict_X_old_collective_explicit(
        Xrow, Xcol, X_predicted, nnz,
        A, NULL,
        B, NULL,
        glob_mean,
        k, 0, 0, 0,
        m, n,
        1
    );

    printf("\nApproximation of X: A*t(B) + mean:\n");
    print_sparse_matrix(X_predicted, Xrow, Xcol, m, n, nnz);
    printf("\n");

    printf("(Should be very close to the real X)\n");
    printf("\nEND OF EXAMPLE\n");

    free(A);
    free(B);
    free(X_predicted);
    return 0;
}

/* Helpers */
void print_sparse_matrix(double X[], int Xrow[], int Xcol[], int m, int n, size_t nnz)
{
    /* This is an inefficient function which will create a dense matrix
       and then fill it, in order to print it more easily */
    double *Xfull = (double*)calloc(m*n, sizeof(double));
    for (size_t ix = 0; ix < nnz; ix++) {
        Xfull[Xcol[ix] + Xrow[ix]*(size_t)n] = X[ix];
    }

    printf("[");
    for (int row = 0; row < m; row++) {
        if (row > 0)
            printf(" ");
        printf("[ ");

        for (int col = 0; col < n; col++) {
            if (Xfull[col + row*n] == 0.)
                printf("  .  ");
            else
                printf("% 2.1f ", Xfull[col + row*n]);
        }

        printf("]");
        if (row != m-1)
            printf(",\n");
    }
    printf("]\n");
}

void print_dense_matrix(double X[], int m, int n)
{
    printf("[");

    for (int row = 0; row < m; row++) {
        if (row > 0)
            printf(" ");
        printf("[ ");


        for (int col = 0; col < n; col++) {
            if (col != n-1)
                printf("% 5.2f,  ", X[col + row*n]);
            else
                printf("% 5.2f ", X[col + row*n]);
        }


        printf("]");
        if (row != m-1)
            printf(",\n");
    }

    printf("]\n");
}
