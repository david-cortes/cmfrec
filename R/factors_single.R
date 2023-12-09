#' @export
#' @title Calculate latent factors for a new user
#' @rdname factors_single
#' @description Determine latent factors for a new user, given either `X` data
#' (a.k.a. "warm-start"), or `U` data (a.k.a. "cold-start"), or both.
#' 
#' For example usage, see the main section \link{fit_models}.
#' @param model A collective matrix factorization model from this package - see
#' \link{fit_models} for details.
#' @param X New `X` data, either as a numeric vector (class `numeric`), or as
#' a sparse vector from package `Matrix` (class `dsparseVector`). If the `X` to
#' which the model was fit was a `data.frame`, the column/item indices will have
#' been reindexed internally, and the numeration can be found under
#' `model$info$item_mapping`. Alternatively, can instead pass the column indices
#' and values and let the model reindex them (see `X_col` and `X_val`).
#' Should pass at most one of `X` or `X_col`+`X_val`.
#' Dense `X` data is not supported for `CMF_implicit` or `OMF_implicit`.
#' @param X_col New `X` data in sparse vector format, with `X_col` denoting the
#' items/columns which are not missing. If the `X` to which the model was fit was
#' a `data.frame`, here should pass IDs matching to the second column of that `X`,
#' which will be reindexed internally. Otherwise, should have column indices with
#' numeration starting at 1 (passed as an integer vector).
#' Should pass at most one of `X` or `X_col`+`X_val`.
#' @param X_val New `X` data in sparse vector format, with `X_val` denoting the
#' associated values to each entry in `X_col`
#' (should be a numeric vector of the same length as `X_col`).
#' Should pass at most one of `X` or `X_col`+`X_val`.
#' @param weight (Only for the explicit-feedback models)
#' Associated weight to each non-missing observation in `X`. Must have the same
#' number of entries as `X` - that is, if passing a dense vector of length `n`,
#' `weight` should be a numeric vector of length `n` too, if passing a sparse
#' vector, should have a length corresponding to the number of non-missing elements.
#' or alternatively, may be a sparse matrix/vector with the same non-missing indices
#' as `X` (but this will not be checked).
#' @param U New `U` data, either as a numeric vector (class `numeric`), or as a
#' sparse vector from package `Matrix` (class `dsparseVector`). Alternatively,
#' if `U` is sparse, can instead pass the indices of the non-missing columns
#' and their values separately (see `U_col`).
#' Should pass at most one of `U` or `U_col`+`U_val`.
#' @param U_col New `U` data in sparse vector format, with `U_col` denoting the
#' attributes/columns which are not missing. Should have numeration starting at 1
#' (should be an integer vector).
#' Should pass at most one of `U` or `U_col`+`U_val`.
#' @param U_val New `U` data in sparse vector format, with `U_val` denoting the
#' associated values to each entry in `U_col`
#' (should be a numeric vector of the same length as `U_col`).
#' Should pass at most one of `U` or `U_col`+`U_val`.
#' @param U_bin Binary columns of `U` on which a sigmoid transformation will be
#' applied. Should be passed as a numeric vector. Note that `U` and `U_bin` are
#' not mutually exclusive.
#' @param output_bias Whether to also return the user bias determined by the model
#' given the data in `X`.
#' @param output_A Whether to return the raw `A` factors (the free offset).
#' @param exact (In the `OMF_explicit` model)
#' Whether to calculate `A` and `Am` with the regularization applied
#' to `A` instead of to `Am` (if using the L-BFGS method, this is how the model
#' was fit). This is usually a slower procedure.
#' Only relevant when passing `X` data.
#' @param ... Not used.
#' @return If passing `output_bias=FALSE`, `output_A=FALSE`, and in the
#' implicit-feedback models, will return a vector with the obtained latent factors.
#' If passing any of the earlier options, will return a list with the following
#' entries: \itemize{
#' \item `factors`, which will contain the obtained factors for this new user.
#' \item `bias`, which will contain the obtained bias for this new user
#' (if passing `output_bias=TRUE`) (this will be a single number).
#' \item `A` (if passing `output_A=TRUE`), which will contain the raw `A` vector
#' (which is added to the factors determined from user attributes in order to
#' obtain the factorization parameters).
#' }
#' @details Note that, regardless of whether the model was fit with the L-BFGS or
#' ALS method with CG or Cholesky solver, the new factors will be determined through the
#' Cholesky method or through the precomputed matrices (e.g. a simple matrix-vector multiply
#' for the `ContentBased` model), unless passing `U_bin` in which case they will be
#' determined through the same L-BFGS method with which the model was fit.
#' @seealso \link{factors} \link{topN_new}
factors_single <- function(model, ...) {
    UseMethod("factors_single")
}

process.data.factors.single <- function(model, obj,
                                        X = NULL, X_col = NULL, X_val = NULL, weight = NULL,
                                        U = NULL, U_col = NULL, U_val = NULL,
                                        U_bin = NULL,
                                        output_bias = FALSE, output_A = FALSE, exact = FALSE) {
    
    if (is.null(X) && is.null(X_col) && is.null(X_val) && is.null(weight) &&
        is.null(U) && is.null(U_col) && is.null(U_val) && is.null(U_bin))
        stop("No new data was passed.")
    output_bias  <-  check.bool(output_bias)
    output_A     <-  check.bool(output_A)
    exact        <-  check.bool(exact)
    if (!is.null(U_bin) && !NCOL(obj$matrices$Cb))
        stop("Model was not fit to binary side information.")
    if ((!is.null(U) || !is.null(U_col) || !is.null(U_val)) && !NCOL(obj$matrices$C))
        stop("Model was not fit to 'U' data.")
    if (is.null(X_col) != is.null(X_val))
        stop("'X_col' and 'X_val' must be passed in conjunction.")
    if (is.null(U_col) != is.null(U_val))
        stop("'U_col' and 'U_val' must be passed in conjunction.")
    if (NROW(X_col) != NROW(X_val))
        stop("'X_col' and 'X_val' must have the same number of entries.")
    if (NROW(U_col) != NROW(U_val))
        stop("'U_col' and 'U_val' must have the same number of entries.")
    if (!is.null(weight) && is.null(X) && is.null(X_col) && is.null(X_val))
        stop("Cannot pass 'weight' without 'X' data.")
    
    if (model %in% c("CMF_implicit", "OMF_implicit")) {
        if (!is.null(X) && !inherits(X, "sparseVector"))
            stop("Cannot only pass 'X' as sparse vector for implicit-feedback models.")
    }
    
    if (output_bias && !NROW(obj$matrices$user_bias))
        stop("Model was fit without biases.")
    
    processed_X   <-  process.new.X.single(X, X_col, X_val, weight,
                                           obj$info, NCOL(obj$matrices$B))
    processed_U   <-  process.new.U.single(U, U_col, U_val,
                                           obj$info$user_mapping, NCOL(obj$matrices$C),
                                           obj$info$U_cols,
                                           allow_null = model != "ContentBased",
                                           allow_na = model %in% c("CMF", "CMF_implicit"),
                                           exact_shapes = !(model %in% c("CMF", "CMF_implicit")))
    processed_Ub  <-  process.new.U.single(U_bin, NULL, NULL,
                                           obj$info$user_mapping, NCOL(obj$matrices$Cb),
                                           obj$info$U_bin_cols)
    
    if (obj$info$apply_log_transf) {
        if (NROW(processed_X$X_val)) {
            if (min(processed_X$X_val) < 1)
                stop("Cannot pass values below 1 with 'apply_log_transf=TRUE'.")
        }
    }
    
    if (!NROW(processed_X$X) && obj$info$NA_as_zero) {
        if (NROW(obj$precomputed$B_plus_bias))
            processed_X$n <- ncol(obj$precomputed$B_plus_bias)
        else
            processed_X$n <- ncol(obj$matrices$B)
    }

    if (!NROW(processed_U$U) && NROW(obj$matrices$C) && obj$info$NA_as_zero_user) {
        processed_U$p <- ncol(obj$matrices$C)
    }
    
    return(list(
        processed_X = processed_X,
        processed_U = processed_U,
        processed_Ub = processed_Ub,
        output_bias = output_bias,
        output_A = output_A,
        exact = exact
    ))
}


#' @export
#' @rdname factors_single
factors_single.CMF <- function(model, X = NULL, X_col = NULL, X_val = NULL,
                               U = NULL, U_col = NULL, U_val = NULL,
                               U_bin = NULL, weight = NULL,
                               output_bias = FALSE, ...) {
    inputs <- process.data.factors.single("CMF", model,
                                          X = X, X_col = X_col, X_val = X_val, weight = weight,
                                          U = U, U_col = U_col, U_val = U_val,
                                          U_bin = U_bin,
                                          output_bias = output_bias)
    a_vec <- numeric(model$info$k_user + model$info$k + model$info$k_main)
    a_bias <- numeric()
    if (NROW(model$matrices$user_bias))
        a_bias <- numeric(length = 1L)
    ret_code <- .Call(call_factors_collective_explicit_single,
                      a_vec, a_bias,
                      inputs$processed_U$U, inputs$processed_U$p,
                      inputs$processed_U$U_val, inputs$processed_U$U_col,
                      inputs$processed_U_bin$U, inputs$processed_U_bin$p,
                      model$info$NA_as_zero_user, model$info$NA_as_zero,
                      model$info$nonneg,
                      model$matrices$C, model$matrices$Cb,
                      model$matrices$glob_mean, model$matrices$item_bias,
                      model$matrices$U_colmeans,
                      inputs$processed_X$X_val, inputs$processed_X$X_col,
                      inputs$processed_X$X, inputs$processed_X$n,
                      inputs$processed_X$weight,
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
                      model$precomputed$CtC,
                      model$precomputed$TransCtCinvCt,
                      model$precomputed$B_plus_bias,
                      model$precomputed$CtUbias)
    check.ret.code(ret_code)
    if (inputs$output_bias) (
        return(list(factors = a_vec, bias = a_bias))
    ) else {
        return(a_vec)
    }
}

#' @export
#' @rdname factors_single
factors_single.CMF_implicit <- function(model, X = NULL, X_col = NULL, X_val = NULL,
                                        U = NULL, U_col = NULL, U_val = NULL, ...) {
    inputs <- process.data.factors.single("CMF_implicit", model,
                                          X = X, X_col = X_col, X_val = X_val,
                                          U = U, U_col = U_col, U_val = U_val)
    a_vec <- numeric(model$info$k_user + model$info$k + model$info$k_main)
    lambda <- ifelse(NROW(model$info$lambda) > 1L, model$info$lambda[3L], model$info$lambda)
    l1_lambda <- ifelse(NROW(model$info$l1_lambda) > 1L, model$info$l1_lambda[3L], model$info$l1_lambda)
    ret_code <- .Call(call_factors_collective_implicit_single,
                      a_vec,
                      inputs$processed_U$U, inputs$processed_U$p,
                      inputs$processed_U$U_val, inputs$processed_U$U_col,
                      model$info$NA_as_zero_user,
                      model$info$nonneg,
                      model$matrices$U_colmeans,
                      model$matrices$B, NCOL(model$matrices$B), model$matrices$C,
                      inputs$processed_X$X_val, inputs$processed_X$X_col,
                      model$info$k, model$info$k_user, model$info$k_item, model$info$k_main,
                      lambda, l1_lambda, model$info$alpha, model$info$w_main, model$info$w_user,
                      model$info$w_main_multiplier,
                      model$info$apply_log_transf,
                      model$precomputed$BeTBe,
                      model$precomputed$BtB,
                      model$precomputed$BeTBeChol,
                      model$precomputed$CtUbias)
    check.ret.code(ret_code)
    return(a_vec)
}

#' @export
#' @rdname factors_single
factors_single.ContentBased <- function(model,
                                        U = NULL, U_col = NULL, U_val = NULL, ...) {
    
    inputs <- process.data.factors.single("ContentBased", model,
                                          U = U, U_col = U_col, U_val = U_val)
    a_vec <- numeric(model$info$k)
    
    ret_code <- .Call(call_factors_content_based_single,
                      a_vec, model$info$k,
                      inputs$processed_U$U, inputs$processed_U$p,
                      inputs$processed_U$U_val,  inputs$processed_U$U_col,
                      model$matrices$C, model$matrices$C_bias)
    
    check.ret.code(ret_code)
    return(a_vec)
}

#' @export
#' @rdname factors_single
factors_single.OMF_explicit <- function(model, X = NULL, X_col = NULL, X_val = NULL,
                                        U = NULL, U_col = NULL, U_val = NULL, weight = NULL,
                                        output_bias = FALSE, output_A = FALSE,
                                        exact = FALSE, ...) {
    
    if (!exact && !output_A &&
        (!is.null(X) || !is.null(X_col) || !is.null(X_val)) &&
        (!is.null(U) || !is.null(U_col) || !is.null(U_val))) {
        warning("'U' data is ignored in the presence of 'X' data with 'exact=FALSE'.")
        U <- NULL
        U_col <- NULL
        U_val <- NULL
    }
    
    inputs <- process.data.factors.single("OMF_explicit", model,
                                          X = X, X_col = X_col, X_val = X_val, weight = weight,
                                          U = U, U_col = U_col, U_val = U_val,
                                          output_bias = output_bias,
                                          output_A = output_A, exact = exact)
    a_vec <- numeric(model$info$k_sec + model$info$k + model$info$k_main)
    a_bias <- numeric()
    if (NROW(model$matrices$user_bias))
        a_bias <- numeric(length = 1L)
    a_orig <- numeric()
    if (inputs$output_A) {
        if (NROW(model$matrices$C)) {
            a_orig <- numeric(model$info$k + model$info$k_main)
        } else {
            warning("Option 'output_A' invalid when the model was not fit to 'U' data.")
            inputs$output_A <- FALSE
        }
    }
    if (inputs$exact) {
        if (!model$info$nfev) {
            warning("Option 'exact' not meaningful for ALS-fitted models.")
            inputs$exact <- FALSE
        }
    }
    lambda <- ifelse(NROW(model$info$lambda) > 1L, model$info$lambda[3L], model$info$lambda)
    
    ret_code <- .Call(call_factors_offsets_explicit_single,
                      a_vec, a_bias, a_orig,
                      inputs$processed_U$U, inputs$processed_U$p,
                      inputs$processed_U$U_val,  inputs$processed_U$U_col,
                      inputs$processed_X$X_val, inputs$processed_X$X_col,
                      inputs$processed_X$X, inputs$processed_X$n,
                      inputs$processed_X$weight,
                      model$matrices$Bm, model$matrices$C,
                      model$matrices$C_bias,
                      model$matrices$glob_mean, model$matrices$item_bias,
                      model$info$k, model$info$k_sec, model$info$k_main,
                      model$info$w_user,
                      lambda,
                      inputs$exact,
                      model$precomputed$TransBtBinvBt,
                      model$precomputed$BtB,
                      model$precomputed$Bm_plus_bias)
    
    check.ret.code(ret_code)
    if (!inputs$output_bias && !inputs$output_A)
        return(a_vec)
    out <- list(factors = a_vec)
    if (inputs$output_bias) {
        out$bias <- a_bias
    }
    if (inputs$output_A) {
        out$A <- a_orig
    }
    return(out)
}

#' @export
#' @rdname factors_single
factors_single.OMF_implicit <- function(model, X = NULL, X_col = NULL, X_val = NULL,
                                        U = NULL, U_col = NULL, U_val = NULL,
                                        output_A = FALSE, ...) {
    
    if (!output_A &&
        (!is.null(X) || !is.null(X_col) || !is.null(X_val)) &&
        (!is.null(U) || !is.null(U_col) || !is.null(U_val))) {
        warning("'U' data is ignored in the presence of 'X' data.")
        U <- NULL
        U_col <- NULL
        U_val <- NULL
    }
    
    inputs <- process.data.factors.single("OMF_implicit", model,
                                          X = X, X_col = X_col, X_val = X_val,
                                          U = U, U_col = U_col, U_val = U_val,
                                          output_A = output_A)
    a_vec <- numeric(model$info$k)
    a_orig <- numeric()
    if (inputs$output_A)
        a_orig <- numeric(model$info$k)
    ret_code <- .Call(call_factors_offsets_implicit_single,
                      a_vec,
                      inputs$processed_U$U, inputs$processed_U$p,
                      inputs$processed_U$U_val, inputs$processed_U$U_col,
                      inputs$processed_X$X_val, inputs$processed_X$X_col,
                      model$matrices$Bm, model$matrices$C,
                      model$matrices$C_bias,
                      model$info$k, NCOL(model$matrices$Bm),
                      model$info$lambda, model$info$alpha,
                      model$info$apply_log_transf,
                      model$precomputed$BtB,
                      a_orig)
    
    check.ret.code(ret_code)
    if (inputs$output_A) {
        return(list(factors = a_vec, A = a_orig))
    } else {
        return(a_vec)
    }
}
