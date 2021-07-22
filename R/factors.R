#' @export
#' @title Calculate latent factors on new data
#' @rdname factors
#' @description Determine latent factors for new user(s)/row(s), given either `X` data
#' (a.k.a. "warm-start"), or `U` data (a.k.a. "cold-start"), or both.
#' 
#' If passing both types of data (`X` and `U`), and the number of rows in them
#' differs, will be assumed that the shorter matrix has only missing values
#' for the unmatched entries in the other matrix.
#' 
#' Note: this function will not perform any internal re-indexing for the data.
#' If the `X` to which the data was fit was a `data.frame`, the numeration of the
#' items will be under `model$info$item_mapping`. There is also a function
#' \link{factors_single} which will let the model do the appropriate reindexing.
#' 
#' For example usage, see the main section \link{fit_models}.
#' @param model A collective matrix factorization model from this package - see
#' \link{fit_models} for details.
#' @param X New `X` data, with rows denoting new users.
#' Can be passed in the following formats:\itemize{
#' \item A sparse COO/triplets matrix, either from package
#' `Matrix` (class `dgTMatrix`), or from package `SparseM` (class `matrix.coo`).
#' \item A sparse matrix in CSR format, either from package
#' `Matrix` (class `dgRMatrix`), or from package `SparseM` (class `matrix.csr`).
#' Passing the input as CSR is faster than COO as it will be converted internally.
#' \item A sparse row vector from package `Matrix` (class `dsparseVector`).
#' \item A dense matrix from base R (class `matrix`), with missing entries set as `NA`/`NaN`.
#' \item A dense row vector from base R (class `numeric`), with missing entries set as `NA`/`NaN`.
#' }
#' 
#' Dense `X` data is not supported for `CMF_implicit` or `OMF_implicit`.
#' @param weight Associated observation weights for entries in `X`. If passed, must
#' have the same shape as `X` - that is, if `X` is a sparse matrix, should be a
#' numeric vector with length equal to the non-missing elements (or a sparse matrix in
#' the same format, but will not make any checks on the indices), if `X` is a dense
#' matrix, should also be a dense matrix with the same number of rows and columns.
#' @param U New `U` data, with rows denoting new users.
#' Can be passed in the same formats as `X`, or additionally
#' as a `data.frame`, which will be internally converted to a matrix.
#' @param U_bin New binary columns of `U`. Must be passed as a dense matrix from
#' base R or as a `data.frame`.
#' @param output_bias Whether to also return the user bias determined by the model
#' given the data in `X`.
#' @param output_A Whether to return the raw `A` factors (the free offset).
#' @param exact (In the `OMF_explicit` model)
#' Whether to calculate `A` and `Am` with the regularization applied
#' to `A` instead of to `Am` (if using the L-BFGS method, this is how the model
#' was fit). This is usually a slower procedure.
#' Only relevant when passing `X` data.
#' @param ... Not used.
#' @details Note that, regardless of whether the model was fit with the L-BFGS or
#' ALS method with CG or Cholesky solver, the new factors will be determined through the
#' Cholesky method or through the precomputed matrices (e.g. a simple matrix-matrix multiply
#' for the `ContentBased` model), unless passing `U_bin` in which case they will be
#' determined through the same L-BFGS method with which the model was fit.
#' @return If passing `output_bias=FALSE`, `output_A=FALSE`, and for the
#' implicit-feedback models, will return a matrix with the obtained latent
#' factors for each row/user given the `X` and/or `U` data (number of rows is
#' `max(nrow(X), nrow(U), nrow(U_bin))`).
#' If passing any of the above options,
#' will return a list with the following elements: \itemize{
#' \item `factors`: The obtained latent factors (a matrix).
#' \item `bias`: (If passing `output_bias=TRUE`)
#' A vector with the obtained biases for each row/user.
#' \item `A`: (If passing `output_A=TRUE`) The raw `A` factors matrix
#' (which is added to the factors determined from user attributes in order to
#' obtain the factorization parameters).
#' }
#' @seealso \link{factors_single}
factors <- function(model, ...) {
    UseMethod("factors")
}

process.data.factors <- function(model, obj, X=NULL, weight=NULL,
                                 U=NULL, U_bin=NULL,
                                 output_bias=FALSE,
                                 output_A=FALSE, exact=FALSE,
                                 matched_shapes=FALSE) {
    output_bias      <-  check.bool(output_bias, "output_bias")
    exact            <-  check.bool(exact, "exact")
    output_A         <-  check.bool(output_A, "output_A")
    processed_X      <-  process.new.X(obj, X, weight = weight)
    processed_U      <-  process.new.U(U = U, U_cols = obj$info$U_cols,
                                       p = NCOL(obj$matrices$C), name = "U",
                                       allow_sparse = TRUE,
                                       allow_null = model != "ContentBased",
                                       allow_na = model %in% c("CMF", "CMF_implicit"),
                                       exact_shapes = !(model %in% c("CMF", "CMF_implicit")))
    processed_U_bin  <-  process.new.U(U = U_bin, U_cols = obj$info$U_bin_cols,
                                       p = NCOL(obj$matrices$Cb), name = "U_bin",
                                       allow_sparse=FALSE, allow_null=TRUE,
                                       allow_na=TRUE, exact_shapes=FALSE)
    
    if (NROW(processed_X$Xarr) && model == "OMF_explicit") {
        if (processed_X$n != NCOL(obj$matrices$Bm))
            stop(sprintf("'X' has %s columns than the model was fit to.",
                         ifelse(processed_X$n > NCOL(obj$matrices$Bm), "more", "less")))
    }
    
    if (matched_shapes) {
        msg_rows <- "'X' and 'U' must have the same rows."
        if (NROW(processed_X$Xarr) && NROW(processed_U$Uarr)) {
            if (processed_X$m != processed_U$m)
                stop(msg_rows)
        } else if (NROW(processed_X$Xarr) && !NROW(processed_U$Uarr)) {
            if (NROW(processed_U$Uval) || NROW(processed_U$Ucsr)) {
                if (processed_U$m < processed_X$m) {
                    processed_U$m <- processed_X$m
                } else if (processed_U$m > processed_X$m) {
                    stop(msg_rows)
                }
            }
        } else if (!NROW(processed_X$Xarr) && NROW(processed_U$Uarr)) {
            if (NROW(processed_X$Xval) || NROW(processed_X$Xcsr)) {
                if (processed_X$m < processed_U$m) {
                    processed_X$m <- processed_U$m
                } else if (processed_X$m > processed_U$m) {
                    stop(msg_rows)
                }
            }
        }
        
        if (processed_X$m != processed_U$m) {
            if (processed_X$m)
                processed_X$m <- max(c(processed_X$m, processed_U$m))
            if (processed_U$m)
                processed_U$m <- max(c(processed_X$m, processed_U$m))
        }
    }

    if (!NROW(processed_X$Xarr) && obj$info$NA_as_zero) {
        if (NROW(obj$precomputed$B_plus_bias))
            processed_X$n <- ncol(obj$precomputed$B_plus_bias)
        else
            processed_X$n <- ncol(obj$matrices$B)
    }

    if (!NROW(processed_U$Uarr) && NROW(obj$matrices$C) && obj$info$NA_as_zero_user) {
        processed_U$p <- ncol(obj$matrices$C)
    }
    
    return(list(
        processed_X = processed_X,
        processed_U = processed_U,
        processed_U_bin = processed_U_bin,
        output_bias = output_bias,
        output_A = output_A,
        exact = exact
    ))
}

#' @export
#' @rdname factors
factors.CMF <- function(model, X=NULL, U=NULL, U_bin=NULL, weight=NULL,
                        output_bias=FALSE, ...) {
    inputs <- process.data.factors(class(model)[1L], model,
                                   X = X, weight = weight,
                                   U = U, U_bin = U_bin,
                                   output_bias = output_bias)
    m_max <- max(c(inputs$processed_X$m, inputs$processed_U$m, inputs$processed_U_bin$m))
    A <- matrix(0., ncol = m_max, nrow = model$info$k_user + model$info$k + model$info$k_main)
    biasA <- numeric()
    if (NROW(model$matrices$user_bias))
        biasA <- numeric(length = m_max)
    ret_code <- .Call("call_factors_collective_explicit_multiple",
                      A, biasA, m_max,
                      inputs$processed_U$Uarr, inputs$processed_U$m, inputs$processed_U$p,
                      model$info$NA_as_zero_user, model$info$NA_as_zero,
                      model$info$nonneg,
                      inputs$processed_U$Urow, inputs$processed_U$Ucol, inputs$processed_U$Uval,
                      inputs$processed_U$Ucsr_p, inputs$processed_U$Ucsr_i, inputs$processed_U$Ucsr,
                      inputs$processed_U_bin$Uarr, inputs$processed_U_bin$m, inputs$processed_U_bin$p,
                      model$matrices$C, model$matrices$Cb,
                      model$matrices$glob_mean, model$matrices$item_bias,
                      model$matrices$U_colmeans,
                      inputs$processed_X$Xval, inputs$processed_X$Xrow, inputs$processed_X$Xcol,
                      inputs$processed_X$Xcsr_p, inputs$processed_X$Xcsr_i, inputs$processed_X$Xcsr,
                      inputs$processed_X$Xarr, inputs$processed_X$n,
                      inputs$processed_X$Warr, inputs$processed_X$Wsp,
                      model$matrices$B,
                      model$matrices$Bi, model$info$add_implicit_features,
                      model$info$k, model$info$k_user, model$info$k_item, model$info$k_main,
                      model$info$lambda, model$info$l1_lambda,
                      model$info$scale_lam, model$info$scale_lam_sideinfo,
                      model$info$scale_bias_const, model$matrices$scaling_biasA,
                      model$info$w_main, model$info$w_user, model$info$w_implicit,
                      NCOL(model$matrices$B), model$info$include_all_X,
                      model$precomputed$BtB,
                      model$precomputed$TransBtBinvBt,
                      model$precomputed$BtXbias,
                      model$precomputed$BeTBeChol,
                      model$precomputed$BiTBi,
                      model$precomputed$TransCtCinvCt,
                      model$precomputed$CtC,
                      model$precomputed$B_plus_bias,
                      model$precomputed$CtUbias,
                      model$info$nthreads)
    check.ret.code(ret_code)
    A <- t(A)
    if (output_bias) {
        return(list(factors = A, bias = biasA))
    } else {
        return(A)
    }
}

#' @export
#' @rdname factors
factors.CMF_implicit <- function(model, X=NULL, U=NULL, ...) {
    inputs <- process.data.factors(class(model)[1L], model,
                                   X = X,
                                   U = U)
    m_max <- max(c(inputs$processed_X$m, inputs$processed_U$m))
    A <- matrix(0., ncol = m_max, nrow = model$info$k_user + model$info$k + model$info$k_main)
    
    lambda <- ifelse(NROW(model$info$lambda) == 1L, model$info$lambda, model$info$lambda[3L])
    l1_lambda <- ifelse(NROW(model$info$l1_lambda) == 1L, model$info$l1_lambda, model$info$l1_lambda[3L])
    
    ret_code <- .Call("call_factors_collective_implicit_multiple",
                      A, m_max,
                      inputs$processed_U$Uarr,inputs$processed_U$m, inputs$processed_U$p,
                      model$info$NA_as_zero_user,
                      model$info$nonneg,
                      inputs$processed_U$Urow, inputs$processed_U$Ucol, inputs$processed_U$Uval,
                      inputs$processed_U$Ucsr_p, inputs$processed_U$Ucsr_i, inputs$processed_U$Ucsr,
                      inputs$processed_X$Xval, inputs$processed_X$Xrow, inputs$processed_X$Xcol,
                      inputs$processed_X$Xcsr_p, inputs$processed_X$Xcsr_i, inputs$processed_X$Xcsr,
                      model$matrices$B, NCOL(model$matrices$B),
                      model$matrices$C,
                      model$matrices$U_colmeans,
                      model$info$k, model$info$k_user, model$info$k_item, model$info$k_main,
                      lambda, l1_lambda, model$info$alpha, model$info$w_main, model$info$w_user,
                      model$info$w_main_multiplier,
                      model$info$apply_log_transf,
                      model$precomputed$BeTBe,
                      model$precomputed$BtB,
                      model$precomputed$BeTBeChol,
                      model$precomputed$CtUbias,
                      model$info$nthreads)
    
    check.ret.code(ret_code)
    A <- t(A)
    return(A)
}

#' @export
#' @rdname factors
factors.ContentBased <- function(model, U, ...) {
    inputs <- process.data.factors(class(model)[1L], model,
                                   U = U)
    m_max <- inputs$processed_U$m
    A <- matrix(0., ncol = m_max, nrow = model$info$k)
    
    ret_code <- .Call("call_factors_content_based_mutliple",
                      A, m_max, model$info$k,
                      model$matrices$C, model$matrices$C_bias,
                      inputs$processed_U$Uarr, inputs$processed_U$p,
                      inputs$processed_U$Urow, inputs$processed_U$Ucol, inputs$processed_U$Uval,
                      inputs$processed_U$Ucsr_p, inputs$processed_U$Ucsr_i, inputs$processed_U$Ucsr,
                      model$info$nthreads)
    
    check.ret.code(ret_code)
    A <- t(A)
    return(A)
}

#' @export
#' @rdname factors
factors.OMF_explicit <- function(model, X=NULL, U=NULL, weight=NULL,
                                 output_bias=FALSE,
                                 output_A = FALSE, exact=FALSE, ...) {
    if (!exact && !output_A &&
        !is.null(X) && !is.null(U)) {
        warning("'U' data is ignored in the presence of 'X' data with 'exact=FALSE'.")
        U <- NULL
    }
    inputs <- process.data.factors(class(model)[1L], model,
                                   X = X, weight = weight,
                                   U = U,
                                   output_bias = output_bias,
                                   output_A = output_A, exact = exact,
                                   matched_shapes = TRUE)
    m_max <- max(c(inputs$processed_X$m, inputs$processed_U$m))
    A <- matrix(0., ncol = m_max, nrow = model$info$k_sec + model$info$k + model$info$k_main)
    biasA <- numeric()
    if (NROW(model$matrices$user_bias))
        biasA <- numeric(length = m_max)
    Aorig <- numeric()
    if (inputs$output_A) {
        if (NROW(model$matrices$C)) {
            Aorig <- matrix(0., ncol = m_max, nrow = model$info$k + model$info$k_main)
        } else {
            warning("Option 'output_A' invalid when the model was not fit to 'U' data.")
            inputs$output_A <- FALSE
        }
    }
    
    ret_code <- .Call("call_factors_offsets_explicit_multiple",
                      A, biasA,
                      Aorig, m_max,
                      inputs$processed_U$Uarr, inputs$processed_U$p,
                      inputs$processed_U$Urow, inputs$processed_U$Ucol, inputs$processed_U$Uval,
                      inputs$processed_U$Ucsr_p, inputs$processed_U$Ucsr_i, inputs$processed_U$Ucsr,
                      inputs$processed_X$Xval, inputs$processed_X$Xrow, inputs$processed_X$Xcol,
                      inputs$processed_X$Xcsr_p, inputs$processed_X$Xcsr_i, inputs$processed_X$Xcsr,
                      inputs$processed_X$Xarr, inputs$processed_X$n,
                      inputs$processed_X$Warr, inputs$processed_X$Wsp,
                      model$matrices$Bm, model$matrices$C,
                      model$matrices$C_bias,
                      model$matrices$glob_mean, model$matrices$item_bias,
                      model$info$k, model$info$k_sec, model$info$k_main,
                      model$info$w_user,
                      model$info$lambda, inputs$exact,
                      model$precomputed$TransBtBinvBt,
                      model$precomputed$BtB,
                      model$precomputed$Bm_plus_bias,
                      model$info$nthreads)
    
    check.ret.code(ret_code)
    A <- t(A)
    if (!inputs$output_bias && !inputs$output_A)
        return(A)
    out <- list(factors = A)
    if (inputs$output_bias) {
        out$bias <- biasA
    }
    if (inputs$output_A) {
        Aorig <- t(Aorig)
        out$A <- Aorig
    }
    return(out)
}

#' @export
#' @rdname factors
factors.OMF_implicit <- function(model, X=NULL, U=NULL, output_A=FALSE, ...) {
    if (!output_A &&
        !is.null(X) && !is.null(U)) {
        warning("'U' data is ignored in the presence of 'X' data.")
        U <- NULL
    }
    inputs <- process.data.factors(class(model)[1L], model,
                                   X = X,
                                   U = U,
                                   output_A = output_A,
                                   matched_shapes = TRUE)
    m_max <- max(c(inputs$processed_X$m, inputs$processed_U$m))
    A <- matrix(0., ncol = m_max, nrow = model$info$k)
    Aorig <- numeric()
    if (inputs$output_A) {
        if (NROW(model$matrices$C)) {
            Aorig <- matrix(0., ncol = m_max, nrow = model$info$k + model$info$k_main)
        } else {
            warning("Option 'output_A' invalid when the model was not fit to 'U' data.")
            inputs$output_A <- FALSE
        }
    }
    
    ret_code <- .Call("call_factors_offsets_implicit_multiple",
                      A, m_max,
                      Aorig,
                      inputs$processed_U$Uarr, inputs$processed_U$p,
                      inputs$processed_U$Urow, inputs$processed_U$Ucol, inputs$processed_U$Uval,
                      inputs$processed_U$Ucsr_p, inputs$processed_U$Ucsr_i, inputs$processed_U$Ucsr,
                      inputs$processed_X$Xval, inputs$processed_X$Xrow, inputs$processed_X$Xcol,
                      inputs$processed_X$Xcsr_p, inputs$processed_X$Xcsr_i, inputs$processed_X$Xcsr,
                      model$matrices$Bm, model$matrices$C,
                      model$matrices$C_bias,
                      model$info$k, NCOL(model$matrices$Bm),
                      model$info$lambda, model$info$alpha,
                      model$info$apply_log_transf,
                      model$precomputed$BtB,
                      model$info$nthreads)
    
    check.ret.code(ret_code)
    A <- t(A)
    if (inputs$output_A) {
        return(list(factors = A, A = t(Aorig)))
    } else {
        return(A)
    }
}
