#' @export
#' @title Impute missing entries in `X` data
#' @description Replace `NA`/`NaN` values in new `X` data according to the model predictions,
#' given that same `X` data and optionally `U` data.
#' 
#' Note: this function will not perform any internal re-indexing for the data.
#' If the `X` to which the data was fit was a `data.frame`, the numeration of the
#' items will be under `model$info$item_mapping`. There is also a function
#' \link{predict_new} which will let the model do the appropriate reindexing.
#' @details If using the matrix factorization model as a general missing-value imputer,
#' it's recommended to:\itemize{
#' \item Fit a model without user biases.
#' \item Set a lower regularization for the item biases than for the matrices.
#' \item Tune the regularization parameter(s) very well.
#' }
#' In general, matrix factorization works better for imputation of selected entries
#' of sparse-and-wide matrices, whereas for dense matrices, the method is unlikely
#' to provide better results than mean/median imputation, but it is nevertheless
#' provided for experimentation purposes.
#' @param model A collective matrix factorization model as output by function
#' \link{CMF}. This functionality is not available for the other model classes.
#' @param X New `X` data with missing values which will be imputed.
#' Must be passed as a dense matrix from base R (class `matrix`).
#' @param weight Associated observation weights for entries in `X`. If passed, must
#' have the same shape as `X`.
#' @param U New `U` data, with rows matching to those of `X`.
#' Can be passed in the following formats:\itemize{
#' \item A sparse COO/triplets matrix, either from package
#' `Matrix` (class `dgTMatrix`), or from package `SparseM` (class `matrix.coo`).
#' \item A sparse matrix in CSR format, either from package
#' `Matrix` (class `dgRMatrix`), or from package `SparseM` (class `matrix.csr`).
#' Passing the input as CSR is faster than COO as it will be converted internally.
#' \item A sparse row vector from package `Matrix` (class `dsparseVector`).
#' \item A dense matrix from base R (class `matrix`), with missing entries set as `NA`/`NaN`.
#' \item A dense row vector from base R (class `numeric`).
#' \item A `data.frame`.
#' }
#' @param U_bin New binary columns of `U` (rows matching to those of `X`).
#' Must be passed as a dense matrix from base R or as a `data.frame`.
#' @return The `X` matrix with its missing values imputed according to the
#' model predictions.
#' @examples
#' library(cmfrec)
#' 
#' ### Simplest example
#' SeqMat <- matrix(1:50, nrow=10)
#' SeqMat[2,1] <- NaN
#' SeqMat[8,3] <- NaN
#' set.seed(123)
#' m <- CMF(SeqMat, k=1, lambda=1e-10, nthreads=1L, verbose=FALSE)
#' imputeX(m, SeqMat)
#' 
#' 
#' ### Better example with multivariate normal data
#' if (require("MASS")) {
#'     ### Generate random data, set some values as NA
#'     set.seed(1)
#'     n_rows <- 100
#'     n_cols <- 50
#'     mu <- rnorm(n_cols)
#'     S <- matrix(rnorm(n_cols^2), nrow = n_cols)
#'     S <- t(S) %*% S + diag(1, n_cols)
#'     X <- MASS::mvrnorm(n_rows, mu, S)
#'     X_na <- X
#'     values_NA <- matrix(runif(n_rows*n_cols) < .25, nrow=n_rows)
#'     X_na[values_NA] <- NaN
#'     
#'     ### In the event that any column is fully missing
#'     if (any(colSums(is.na(X_na)) == n_rows)) {
#'         cols_remove <- colSums(is.na(X_na)) == n_rows
#'         X_na <- X_na[, !cols_remove, drop=FALSE]
#'         values_NA <- values_NA[, !cols_remove, drop=FALSE]
#'     }
#'     
#'     ### Impute missing values with model
#'     set.seed(1)
#'     model <- CMF(X_na, k=15, lambda=50, user_bias=FALSE,
#'                  verbose=FALSE, nthreads=1L)
#'     X_imputed <- imputeX(model, X_na)
#'     cat(sprintf("RMSE for imputed values w/model: %f\n",
#'                 sqrt(mean((X[values_NA] - X_imputed[values_NA])^2))))
#'     
#'     ### Compare against simple mean imputation
#'     X_means <- apply(X_na, 2, mean, na.rm=TRUE)
#'     X_imp_mean <- X_na
#'     for (cl in 1:n_cols)
#'         X_imp_mean[values_NA[,cl], cl] <- X_means[cl]
#'     cat(sprintf("RMSE for imputed values w/means: %f\n",
#'                 sqrt(mean((X[values_NA] - X_imp_mean[values_NA])^2))))
#' }
imputeX <- function(model, X, weight = NULL, U = NULL, U_bin = NULL) {
    if (!("CMF" %in% class(model)))
        stop("Method is only applicable to 'CMF' model.")
    if (!("matrix" %in% class(X)))
        stop("'X' must be a matrix with NAN values.")
    if (!anyNA(X))
        return(X)
    
    inputs <- process.data.factors("CMF", model,
                                   X = X, weight = weight,
                                   U = U, U_bin = U_bin,
                                   matched_shapes=TRUE)
    ret_code <- .Call("call_impute_X_collective_explicit",
                      inputs$processed_X$m, as.logical(NROW(model$matrices$user_bias)),
                      inputs$processed_U$Uarr, inputs$processed_U$m, inputs$processed_U$p,
                      model$info$NA_as_zero_user,
                      model$info$nonneg,
                      inputs$processed_U$Urow, inputs$processed_U$Ucol, inputs$processed_U$Uval,
                      inputs$processed_U$Ucsr_p, inputs$processed_U$Ucsr_i, inputs$processed_U$Ucsr,
                      inputs$processed_U_bin$Uarr, inputs$processed_U_bin$m, inputs$processed_U_bin$p,
                      model$matrices$C, model$matrices$Cb,
                      model$matrices$glob_mean, model$matrices$item_bias,
                      model$matrices$U_colmeans,
                      inputs$processed_X$Xarr, inputs$processed_X$n,
                      inputs$processed_X$Warr,
                      model$matrices$B,
                      model$matrices$Bi, model$info$add_implicit_features,
                      model$info$k, model$info$k_user, model$info$k_item, model$info$k_main,
                      model$info$lambda,
                      model$info$w_main, model$info$w_user, model$info$w_implicit,
                      NCOL(model$matrices$B), model$info$include_all_X,
                      model$precomputed$TransBtBinvBt,
                      model$precomputed$BtB,
                      model$precomputed$BeTBeChol,
                      model$precomputed$BiTBi,
                      model$precomputed$TransCtCinvCt,
                      model$precomputed$CtC,
                      model$precomputed$B_plus_bias,
                      model$info$nthreads)
    check.ret.code(ret_code)
    inputs$processed_X$Xarr <- matrix(inputs$processed_X$Xarr, ncol = nrow(X), nrow = ncol(X))
    if (!is.null(rownames(X)))
        colnames(inputs$processed_X$Xarr) <- rownames(X)
    if (!is.null(colnames(X)))
        rownames(inputs$processed_X$Xarr) <- colnames(X)
    X <- t(inputs$processed_X$Xarr)
    return(X)
}
