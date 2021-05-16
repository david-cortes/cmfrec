#' @export
#' @rdname predict_new
#' @title Predict entries in new `X` data
#' @description Predict entries in columns of the `X` matrix for new users/rows
#' given their new `X` and/or `U` data at the combinations [row,column] given by the entries in
#' `user` and `item` (e.g. passing `user=c(1,2,3), item=c(1,1,1)` will predict
#' X[1,1], X[2,1], X[3,1]).
#' 
#' Note: this function will not perform any internal re-indexing for the data.
#' If the `X` to which the data was fit was a `data.frame`, the numeration of the
#' items will be under `model$info$item_mapping`.
#' @param model A collective matrix factorization model from this package - see
#' \link{fit_models} for details.
#' @param items The item IDs for which to make predictions. If `X` to which the model
#' was fit was a `data.frame`, should pass IDs matching to the second column of `X`
#' (the item indices, should be a character vector),
#' otherwise should pass column numbers for `X`, with numeration
#' starting at 1 (should be an integer vector).
#' 
#' If passing `I`, will instead take these indices as row numbers for `I`
#' (only available for the \link{ContentBased} model).
#' @param rows Rows of the new `X`/`U` passed here for which to make
#' predictions, with numeration starting at 1 (should be an integer vector).
#' If not passed and there is only 1 row of data, will predict the entries in
#' `items` for that same row. If not passed and there is more than 1 row of data,
#' the number of rows in the data should match with the number of entries in `items`,
#' and will make predictions for each such combination of <entry in item, row in the data>.
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
#' Dense `X` data is not supported for `CMF_implicit` or `OMF_implicit`.
#' @param weight Associated observation weights for entries in `X`. If passed, must
#' have the same shape as `X` - that is, if `X` is a sparse matrix, should be a
#' numeric vector with length equal to the non-missing elements, if `X` is a dense
#' matrix, should also be a dense matrix with the same number of rows and columns.
#' @param U New `U` data, with rows denoting new users.
#' Can be passed in the same formats as `X`, or additionally
#' as a `data.frame`, which will be internally converted to a matrix.
#' @param U_bin New binary columns of `U`. Must be passed as a dense matrix from
#' base R or as a `data.frame`.
#' @param I (For the `ContentBased` model only) New `I` data for which to make predictions.
#' Supports the same formats as `U`.
#' @param exact (In the `OMF_explicit` model)
#' Whether to calculate `A` and `Am` with the regularization applied
#' to `A` instead of to `Am` (if using the L-BFGS method, this is how the model
#' was fit). This is usually a slower procedure.
#' @param ... Not used.
#' @return A numeric vector with the predicted values for the requested combinations
#' of users (rows in the new data) and items (columns in the old data, unless passing
#' `I` in which case will be rows of `I`). Invalid combinations will be filled with NAs.
#' @seealso \link{predict.cmfrec}
predict_new <- function(model, ...) {
    UseMethod("predict_new")
}

process.data.predict.new <- function(model, obj, X=NULL, weight=NULL,
                                     U=NULL, U_bin=NULL,
                                     items=NULL, rows=NULL,
                                     exact=FALSE) {
    if (obj$info$only_prediction_info)
        stop("Cannot use this function after dropping non-essential matrices.")
    
    exact <- check.bool(exact)
    
    if (!NROW(items))
        stop("'items' cannot be empty.")
    if (NROW(obj$info$item_mapping))
        items <- as.integer(factor(items, obj$info$item_mapping))
    if (NROW(intersect(class(items), c("numeric", "character", "matrix"))))
        items <- as.integer(items)
    if (!("integer" %in% class(items)))
        stop("Invalid 'items'.")
    items <- items - 1L
    
    
    inputs <- process.data.factors(model, obj, X=X, weight=weight,
                                   U=U, U_bin=U_bin,
                                   exact=exact,
                                   matched_shapes=FALSE)
    m_max <- max(c(inputs$processed_X$m, inputs$processed_U$m, inputs$processed_U_bin$m))
    if (m_max == 1L)
        rows <- rep(1L, NROW(items))
    
    if (!is.null(rows)) {
        if (NROW(obj$info$item_mapping))
            rows <- as.integer(factor(rows, obj$info$user_mapping))
        if (NROW(intersect(class(rows), c("numeric", "character", "matrix"))))
            rows <- as.integer(rows)
        if (!("integer" %in% class(rows)))
            stop("Invalid 'rows'.")
        if (NROW(rows) != NROW(items)) {
            if (NROW(rows) == 1L) {
                rows <- rep(rows, NROW(items))
            } else {
                stop("'rows' and 'item' must have the same number of entries.")
            }
        }
    } else {
        if (NROW(items) != m_max)
            stop("Number of entries from 'X'/'U' does not match with entries in 'item'.")
        rows <- seq(1L, NROW(items))
    }
    rows <- rows - 1L
    
    return(list(
        rows = rows,
        items = items,
        m_max = m_max,
        exact = exact,
        processed_X = inputs$processed_X,
        processed_U = inputs$processed_U,
        processed_U_bin = inputs$processed_U_bin
    ))
}

#' @export
#' @rdname predict_new
predict_new.CMF <- function(model, items, rows=NULL,
                            X=NULL, U=NULL, U_bin=NULL, weight=NULL, ...) {

    inputs <- process.data.predict.new("CMF", model, X = X, weight = weight,
                                       U = U, U_bin = U_bin,
                                       items = items, rows = rows)
    n_predict <- NROW(inputs$rows)
    scores <- numeric(length = n_predict)
    ret_code <- .Call("call_predict_X_new_collective_explicit",
                      inputs$m_max,
                      inputs$rows, inputs$items, scores,
                      model$info$nthreads,
                      as.logical(NROW(model$matrices$user_bias)),
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
                      model$precomputed$CtUbias)
    check.ret.code(ret_code)
    return(scores)
}

#' @export
#' @rdname predict_new
predict_new.CMF_implicit <- function(model, items, rows=NULL,
                                     X=NULL, U=NULL, ...) {
    inputs <- process.data.predict.new("CMF_implicit", model, X = X,
                                       U = U,
                                       items = items, rows = rows)
    n_predict <- NROW(inputs$rows)
    scores <- numeric(length = n_predict)
    lambda <- ifelse(NROW(model$info$lambda) == 6L, model$info$lambda[3L], model$info$lambda)
    l1_lambda <- ifelse(NROW(model$info$l1_lambda) == 6L, model$info$l1_lambda[3L], model$info$l1_lambda)
    
    ret_code <- .Call("call_predict_X_new_collective_implicit",
                      inputs$m_max,
                      inputs$rows, inputs$items, scores,
                      model$info$nthreads,
                      inputs$processed_U$Uarr, inputs$processed_U$m, inputs$processed_U$p,
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
                      model$precomputed$CtUbias)
    check.ret.code(ret_code)
    return(scores)
}

#' @export
#' @rdname predict_new
predict_new.OMF_explicit <- function(model, items, rows=NULL,
                                     X=NULL, U=NULL, weight=NULL,
                                     exact=FALSE, ...) {
    inputs <- process.data.predict.new("OMF_explicit", model, X = X, weight = weight,
                                       U = U,
                                       items = items, rows = rows,
                                       exact = exact)
    n_predict <- NROW(inputs$rows)
    scores <- numeric(length = n_predict)
    
    ret_code <- .Call("call_predict_X_new_offsets_explicit",
                      inputs$m_max, as.logical(NROW(model$matrices$user_bias)),
                      inputs$rows, inputs$items, scores,
                      model$info$nthreads,
                      inputs$processed_U$Uarr, inputs$processed_U$p,
                      inputs$processed_U$Urow, inputs$processed_U$Ucol, inputs$processed_U$Uval,
                      inputs$processed_U$Ucsr_p, inputs$processed_U$Ucsr_i, inputs$processed_U$Ucsr,
                      inputs$processed_X$Xval, inputs$processed_X$Xrow, inputs$processed_X$Xcol,
                      inputs$processed_X$Xcsr_p, inputs$processed_X$Xcsr_i, inputs$processed_X$Xcsr,
                      inputs$processed_X$Xarr, NCOL(model$matrices$Bm),
                      inputs$processed_X$Warr, inputs$processed_X$Wsp,
                      model$matrices$Bm, model$matrices$C,
                      model$matrices$C_bias,
                      model$matrices$glob_mean, model$matrices$item_bias,
                      model$info$k, model$info$k_sec, model$info$k_main,
                      model$info$w_user,
                      model$info$lambda, inputs$exact,
                      model$precomputed$TransBtBinvBt,
                      model$precomputed$BtB,
                      model$precomputed$Bm_plus_bias)
    check.ret.code(ret_code)
    return(scores)
}

#' @export
#' @rdname predict_new
predict_new.OMF_implicit <- function(model, items, rows=NULL,
                                     X=NULL, U=NULL, ...) {
    inputs <- process.data.predict.new("OMF_implicit", model, X = X,
                                       U = U,
                                       items = items, rows = rows)
    n_predict <- NROW(inputs$rows)
    scores <- numeric(length = n_predict)
    
    ret_code <- .Call("call_predict_X_new_offsets_implicit",
                      inputs$m_max,
                      inputs$rows, inputs$items, scores,
                      NCOL(model$matrices$Bm),
                      model$info$nthreads,
                      inputs$processed_U$Uarr, inputs$processed_U$p,
                      inputs$processed_U$Urow, inputs$processed_U$Ucol, inputs$processed_U$Uval,
                      inputs$processed_U$Ucsr_p, inputs$processed_U$Ucsr_i, inputs$processed_U$Ucsr,
                      inputs$processed_X$Xval, inputs$processed_X$Xrow, inputs$processed_X$Xcol,
                      inputs$processed_X$Xcsr_p, inputs$processed_X$Xcsr_i, inputs$processed_X$Xcsr,
                      model$matrices$Bm, model$matrices$C,
                      model$matrices$C_bias,
                      model$info$k,
                      model$info$lambda, model$info$alpha,
                      model$info$apply_log_transf,
                      model$precomputed$BtB)
    check.ret.code(ret_code)
    return(scores)
}

#' @export
#' @rdname predict_new
predict_new.ContentBased <- function(model, items=NULL, rows=NULL,
                                     U=NULL, I=NULL, ...) {
    if (!NROW(U))
        stop("'U' cannot be empty")
    if (!NROW(items) && !NROW(I))
        stop("Must pass at least one of 'items' or 'I'.")
    
    items_pass <- items
    rows_pass <- rows
    if (is.null(items))
        items_pass <- 1L
    if (is.null(rows))
        rows_pass <- seq(1L, NROW(items_pass))
    inputs <- process.data.predict.new("ContentBased", model,
                                       U = U,
                                       items = items_pass, rows = rows_pass)
    if (!is.null(items))
        items <- inputs$items
    if (!is.null(rows))
        rows <- inputs$rows
    
    
    if (!is.null(I)) {
        processed_I <- process.new.U(I, model$info$I_cols, NCOL(model$matrices$D), name="I",
                                     allow_sparse=TRUE, allow_null=FALSE,
                                     allow_na=FALSE, exact_shapes=TRUE)
        if (!processed_I$m)
            stop("'I' is empty.")
        if (is.null(items)) {
            if (is.null(rows)) {
                if (processed_I$m != inputs$m_max)
                    stop("Number of rows in 'U' and 'I' do not match.")
                n_predict <- inputs$m_max
            } else {
                items <- seq(1L, NROW(rows)) - 1L
                n_predict <- NROW(rows)
            }
        } else {
            if (NROW(rows) != NROW(items))
                stop("'items' and 'rows' must have the same number of entries.")
            n_predict <- NROW(rows)
        }
        
        if (is.null(items))
            items <- integer()
        if (is.null(rows))
            rows <- integer()
        
        scores <- numeric(length = n_predict)
        ret_code <- .Call("call_predict_X_new_content_based",
                          scores,
                          inputs$processed_U$m, processed_I$m, model$info$k,
                          rows, items,
                          inputs$processed_U$Uarr, inputs$processed_U$p,
                          inputs$processed_U$Urow, inputs$processed_U$Ucol, inputs$processed_U$Uval,
                          inputs$processed_U$Ucsr_p, inputs$processed_U$Ucsr_i, inputs$processed_U$Ucsr,
                          processed_I$Uarr, processed_I$p,
                          processed_I$Urow, processed_I$Ucol, processed_I$Uval,
                          processed_I$Ucsr_p, processed_I$Ucsr_i, processed_I$Ucsr,
                          model$matrices$C, model$matrices$C_bias,
                          model$matrices$D, model$matrices$D_bias,
                          model$matrices$glob_mean,
                          model$info$nthreads)
    } else {
        if (is.null(rows)) {
            if (inputs$processed_U$m == 1L) {
                rows <- rep(0L, NROW(items))    
            } else {
                rows <- seq(1L, inputs$processed_U$m) - 1L
            }
        }
        n_predict <- NROW(items)
        scores <- numeric(length = n_predict)
        ret_code <- .Call("call_predict_X_new_offsets_explicit",
                          inputs$processed_U$m, FALSE,
                          rows, items, scores,
                          model$info$nthreads,
                          inputs$processed_U$Uarr, inputs$processed_U$p,
                          inputs$processed_U$Urow, inputs$processed_U$Ucol, inputs$processed_U$Uval,
                          inputs$processed_U$Ucsr_p, inputs$processed_U$Ucsr_i, inputs$processed_U$Ucsr,
                          numeric(), integer(), integer(),
                          raw(), integer(), numeric(),
                          numeric(), NCOL(model$matrices$Bm),
                          numeric(), numeric(),
                          model$matrices$Bm, model$matrices$C,
                          model$matrices$C_bias,
                          model$matrices$glob_mean, model$matrices$item_bias,
                          0L, model$info$k, 0L,
                          1.,
                          model$info$lambda, FALSE,
                          numeric(),
                          numeric(),
                          numeric())
    }
    
    check.ret.code(ret_code)
    return(scores)
}
