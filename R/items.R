process.inputs.items <- function(model, obj, X=NULL, X_col=NULL, X_val=NULL, weight=NULL,
                                 I=NULL, I_col=NULL, I_val=NULL, I_bin=NULL) {
    accepted_models <- c("CMF", "CMF_implicit", "ContentBased", "OMF_explicit", "OMF_implicit")
    if (!NROW(intersect(class(obj), accepted_models)))
        stop("Method only applicable to the following models: ",
             paste(accepted_models, collapse = ", "))
    if ((NROW(I) || NROW(I_col) || NROW(I_val)) && (!NCOL(obj$matrices$D) || !NROW(obj$matrices$D)))
        stop("Model was not fit to item side info.")
    if (!is.null(I_bin) && (!NCOL(obj$matrices$Db) || !NROW(obj$matrices$Db)))
        stop("Model was not fit to binary side info ('I_bin').")
    
    if (is.null(I) && is.null(I_col) && is.null(I_val) && is.null(I_bin) &&
        is.null(X) && is.null(X_col) && is.null(X_val))
        stop("Must pass inputs 'X' or 'I' in some format.")
    if (is.null(I_col) != is.null(I_val))
        stop("'I_col' and 'I_val' must be passed in conjunction.")
    if (NROW(I_col) != NROW(I_val))
        stop("'I_col' and 'I_val' must have the same number of entries.")
    if (is.null(X_col) != is.null(X_val))
        stop("'X_col' and 'X_val' must be passed in conjunction.")
    if (NROW(X_col) != NROW(X_val))
        stop("'X_col' and 'X_val' must have the same number of entries.")
    
    if (!is.null(weight) && (is.null(X) && is.null(X_col) && is.null(X_val)))
        stop("'weight' not meaningful without 'X' data.")
    if (!is.null(weight) && NROW(intersect(class(obj), c("CMF_implicit", "OMF_implicit"))))
        stop("'weight' not supported for implicit-feedback models.")
    
    if ((!is.null(X) || !is.null(X_col) || !is.null(X_val)) &&
        ("ContentBased" %in% class(obj))) {
        warning("'X' data is ignored for 'ContentBased'.")
        X       <-  NULL
        X_col   <-  NULL
        X_val   <-  NULL
        weight  <-  NULL
    }
    
    info_pass <- list(
        user_mapping = obj$info$item_mapping,
        n_orig = max(c(NCOL(obj$matrices$A), NCOL(obj$matrices$Am))),
        include_all_X = TRUE
    )
    
    processed_X      <-  process.new.X.single(X, X_col, X_val, weight,
                                              info_pass, info_pass$n_orig)
    processed_I      <-  process.new.U.single(I, I_col, I_val, "I",
                                              obj$info$item_mapping, NCOL(obj$matrices$D),
                                              obj$info$I_cols,
                                              allow_null = TRUE,
                                              allow_na = model %in% c("CMF", "CMF_implicit"),
                                              exact_shapes = !(model %in% c("CMF", "CMF_implicit")))
    processed_I_bin  <-  process.new.U.single(I_bin, NULL, NULL, "I_bin",
                                              obj$info$item_mapping, NCOL(obj$matrices$Db),
                                              obj$info$I_bin_cols)
    
    if (NROW(processed_X$X) && NROW(intersect(class(obj), c("CMF_implicit", "OMF_implicit"))))
        stop("Cannot pass dense 'X' for implicit-feedback models.")
    
    if (!NROW(processed_X$X) && !NROW(processed_X$X_val) &&
        !NROW(processed_I$U) && !NROW(processed_I$U_val) &&
        !NROW(processed_I_bin$U))
        stop("Inputs contain no data.")

    if (!NROW(processed_X$X) && NROW(obj$precomputed$A) && obj$info$NA_as_zero) {
        processed_X$n <- ncol(obj$matrices$A)
    }

    if (!NROW(processed_I$U) && NROW(obj$matrices$D) && obj$info$NA_as_zero_item) {
        processed_I$p <- ncol(obj$matrices$D)
    }
    
    return(list(
        processed_X = processed_X,
        processed_I = processed_I,
        processed_I_bin = processed_I_bin
    ))
}


#' @export
#' @title Determine latent factors for a new item
#' @description Calculate latent factors for a new item, based on either
#' new `X` data, new `I` data, or both.
#' 
#' Be aware that the package is user/row centric, and this function is provided for
#' quick experimentation purposes only. Calculating item factors will be slower
#' than calculating user factors
#' (except for the `ContentBased` model for which both types of predictions
#' are equally fast and equally supported).
#' as it will not make usage of the precomputed
#' matrices. If item-based  predictions are required, it's recommended to use
#' instead the function \link{swap.users.and.items} and then use the resulting
#' object with \link{factors_single} or \link{factors}.
#' 
#' @param model A collective matrix factorization model from this package - see
#' \link{fit_models} for details.
#' @param X New `X` data, either as a numeric vector (class `numeric`), or as
#' a sparse vector from package `Matrix` (class `dsparseVector`). If the `X` to
#' which the model was fit was a `data.frame`, the user/row indices will have
#' been reindexed internally, and the numeration can be found under
#' `model$info$user_mapping`. Alternatively, can instead pass the column indices
#' and values and let the model reindex them (see `X_col` and `X_val`).
#' Should pass at most one of `X` or `X_col`+`X_val`.
#' 
#' Be aware that, unlikely in pretty much every other function in this package,
#' here the values are for one \bold{column} of `X`, not one \bold{row} like
#' in e.g. \link{factors_single}.
#' 
#' Dense `X` data is not supported for `CMF_implicit` or `OMF_implicit`.
#' 
#' Not supported for the `ContentBased` model.
#' @param X_col New `X` data in sparse vector format, with `X_col` denoting the
#' users/rows which are not missing. If the `X` to which the model was fit was
#' a `data.frame`, here should pass IDs matching to the first column of that `X`,
#' which will be reindexed internally. Otherwise, should have \bold{row} indices with
#' numeration starting at 1 (passed as an integer vector).
#' Should pass at most one of `X` or `X_col`+`X_val`.
#' 
#' Not supported for the `ContentBased` model.
#' @param X_val New `X` data in sparse vector format, with `X_val` denoting the
#' associated values to each entry in `X_col`
#' (should be a numeric vector of the same length as `X_col`).
#' Should pass at most one of `X` or `X_col`+`X_val`.
#' 
#' Not supported for the `ContentBased` model.
#' @param weight (Only for the explicit-feedback models)
#' Associated weight to each non-missing observation in `X`. Must have the same
#' number of entries as `X` - that is, if passing a dense vector of length `m`,
#' `weight` should be a numeric vector of length `m` too, if passing a sparse
#' vector, should have a length corresponding to the number of non-missing elements.
#' @param I New `I` data, either as a numeric vector (class `numeric`), or as a
#' sparse vector from package `Matrix` (class `dsparseVector`). Alternatively,
#' if `I` is sparse, can instead pass the indices of the non-missing columns
#' and their values separately (see `I_col`).
#' Should pass at most one of `I` or `I_col`+`I_val`.
#' @param I_col New `I` data in sparse vector format, with `I_col` denoting the
#' attributes/columns which are not missing. Should have numeration starting at 1
#' (should be an integer vector).
#' Should pass at most one of `I` or `I_col`+`I_val`.
#' @param I_val New `I` data in sparse vector format, with `I_val` denoting the
#' associated values to each entry in `I_col`
#' (should be a numeric vector of the same length as `I_col`).
#' Should pass at most one of `I` or `I_col`+`I_val`.
#' @param I_bin Binary columns of `I` on which a sigmoid transformation will be
#' applied. Should be passed as a numeric vector. Note that `I` and `I_bin` are
#' not mutually exclusive.
#' @param output_bias Whether to also return the item bias determined by the model
#' given the data in `X` (for explicit-feedback models fit with item biases).
#' @return If passing `output_bias=FALSE`, will return a vector with
#' the obtained latent factors for this item. If passing `output_bias=TRUE`, the
#' result will be a list with entry `factors` having the above vector, and entry
#' `bias` having the estimated bias.
#' @seealso \link{factors_single} \link{predict_new_items}
item_factors <- function(model, X=NULL, X_col=NULL, X_val=NULL,
                         I=NULL, I_col=NULL, I_val=NULL, I_bin=NULL,
                         weight=NULL, output_bias=FALSE) {
    if (model$info$only_prediction_info)
        stop("Cannot use this function after dropping non-essential matrices.")
    
    output_bias <- check.bool(output_bias)
    inputs <- process.inputs.items(class(model)[1L], model,
                                   X = X, X_col = X_col, X_val = X_val, weight = weight,
                                   I = I, I_col = I_col, I_val = I_val, I_bin = I_bin)
    if ("CMF" %in% class(model)) {
        lambda <- swap.lambda(model$info$lambda)
        l1_lambda <- swap.lambda(model$info$l1_lambda)
        b_vec <- numeric(model$info$k_item + model$info$k + model$info$k_main)
        b_bias <- numeric()
        if (NROW(model$matrices$item_bias))
            b_bias <- numeric(1L)

        ret_code <- .Call("call_factors_collective_explicit_single",
                          b_vec, b_bias,
                          inputs$processed_I$U, inputs$processed_I$p,
                          inputs$processed_I$U_val, inputs$processed_I$U_col,
                          inputs$processed_I_bin$U, inputs$processed_I_bin$p,
                          model$info$NA_as_zero_item, model$info$NA_as_zero,
                          model$info$nonneg,
                          model$matrices$D, model$matrices$Db,
                          model$matrices$glob_mean, model$matrices$user_bias,
                          model$matrices$I_colmeans,
                          inputs$processed_X$X_val, inputs$processed_X$X_col,
                          inputs$processed_X$X, inputs$processed_X$n,
                          inputs$processed_X$weight,
                          model$matrices$A,
                          model$matrices$Ai, model$info$add_implicit_features,
                          model$info$k, model$info$k_item, model$info$k_user, model$info$k_main,
                          lambda, l1_lambda,
                          model$info$scale_lam, model$info$scale_lam_sideinfo,
                          model$info$scale_bias_const, model$matrices$scaling_biasB,
                          model$info$w_main, model$info$w_item, model$info$w_implicit,
                          NCOL(model$matrices$A), TRUE,
                          numeric(),
                          numeric(),
                          numeric(),
                          numeric(),
                          numeric(),
                          numeric(),
                          numeric(),
                          numeric(),
                          numeric())
    } else if ("CMF_implicit" %in% class(model)) {
        b_vec <- numeric(model$info$k_item + model$info$k + model$info$k_main)
        lambda <- ifelse(NROW(model$info$lambda) == 6L, model$info$lambda[4L], model$info$lambda)
        l1_lambda <- ifelse(NROW(model$info$l1_lambda) == 6L, model$info$l1_lambda[4L], model$info$l1_lambda)
        ret_code <- .Call("call_factors_collective_implicit_single",
                          b_vec,
                          inputs$processed_I$U, inputs$processed_I$p,
                          inputs$processed_I$U_val, inputs$processed_I$U_col,
                          model$info$NA_as_zero_item,
                          model$info$nonneg,
                          model$matrices$I_colmeans,
                          model$matrices$A, NCOL(model$matrices$A),model$matrices$D,
                          inputs$processed_X$X_val, inputs$processed_X$X_col,
                          model$info$k, model$info$k_item, model$info$k_user, model$info$k_main,
                          lambda, l1_lambda, model$info$alpha, model$info$w_main, model$info$w_item,
                          model$info$w_main_multiplier,
                          model$info$apply_log_transf,
                          numeric(),
                          numeric(),
                          numeric(),
                          numeric())
    } else if ("ContentBased" %in% class(model)) {
        b_vec <- numeric(model$info$k)
        ret_code <- .Call("call_factors_content_based_single",
                          b_vec, model$info$k,
                          inputs$processed_I$U, inputs$processed_I$p,
                          inputs$processed_I$U_val, inputs$processed_I$U_col,
                          model$matrices$D, model$matrices$D_bias)
    } else if ("OMF_explicit" %in% class(model)) {
        lambda <- model$info$lambda
        lambda <- swap.lambda(model$info$lambda)
        b_vec <- numeric(model$info$k_sec + model$info$k + model$info$k_main)
        b_bias <- numeric()
        if (NROW(model$matrices$item_bias))
            b_bias <- numeric(1L)
        ret_code <- .Call("call_factors_offsets_explicit_single",
                          b_vec, b_bias, numeric(),
                          inputs$processed_I$U, inputs$processed_I$p,
                          inputs$processed_I$U_val, inputs$processed_I$U_col,
                          inputs$processed_X$X_val, inputs$processed_X$X_col,
                          inputs$processed_X$X, inputs$processed_X$n,
                          inputs$processed_X$weight,
                          model$matrices$Am, model$matrices$D,
                          model$matrices$D_bias,
                          model$matrices$glob_mean, model$matrices$user_bias,
                          model$info$k, model$info$k_sec, model$info$k_main,
                          model$info$w_item,
                          lambda,
                          FALSE,
                          numeric(),
                          numeric(),
                          numeric())
    } else if ("OMF_implicit" %in% class(model)) {
        b_vec <- numeric(model$info$k)
        ret_code <- .Call("call_factors_offsets_implicit_single",
                          b_vec,
                          inputs$processed_I$U, inputs$processed_I$p,
                          inputs$processed_I$U_val, inputs$processed_I$U_col,
                          inputs$processed_X$X_val, inputs$processed_X$X_col,
                          model$matrices$Am, model$matrices$D,
                          model$matrices$D_bias,
                          model$info$k, NCOL(model$matrices$Am),
                          model$info$lambda, model$info$alpha,
                          model$info$apply_log_transf,
                          numeric(),
                          numeric())
    } else {
        stop("Unexpected error")
    }
    
    check.ret.code(ret_code)
    if (!output_bias) {
        return(b_vec)
    } else {
        return(list(factors = b_vec, bias = b_bias))
    }
}

#' @export
#' @title Predict new columns of `X` given item attributes
#' @description Calculate the predicted values for new columns of `X` (which were
#' not present in the `X` to which the model was fit) given new `X` and/or `I` data.
#' 
#' This function can predict combinations in 3 ways:\itemize{
#' \item If passing vectors for `user` and `item`, will predict the combinations
#' of user/item given in those arrays (e.g. if `I` has 3 rows, and passing
#' `user=c(1,1,2), item=c(1,2,3)`, will predict entries X[1,1], X[1,2], X[2,3],
#' with columns of `X` (rows of `t(X)`) corresponding to the rows of `I`
#' passed here and users corresponding to the ones to which the model was fit).
#' \item If passing a vector for `user` but not for `item`, will predict the
#' value that each user would give to the corresponding row of `I`/`t(X)` (in this
#' case, the number of entries in `user` should be the same as the number of
#' rows in `I`/`t(X)`).
#' \item If passing a single value for `user`, will calculate all predictions
#' for that user for the rows of `I`/`t(X)` given in `item`, or for all rows of
#' `I`/`t(X)` if `item` is not given.
#' }
#' 
#' Be aware that the package is user/row centric, and this function is provided for
#' quick experimentation purposes only. Calculating item factors will be slower
#' than calculating user factors as it will not make usage of the precomputed
#' matrices (except for the `ContentBased` model for which both types of predictions
#' are equally fast and equally supported).
#' If item-based  predictions are required, it's recommended to use
#' instead the function \link{swap.users.and.items} and then use the resulting
#' object with \link{predict_new}.
#' 
#' @param model A collective matrix factorization model from this package - see
#' \link{fit_models} for details.
#' @param user User(s) for which the new entries will be predicted. If passing
#' a single ID, will calculate all the values in `item`, or all the values in
#' `I`/`t(X)` (see section `Description` for details).
#' 
#' If the `X` to which the model was fit was a `data.frame`, the IDs here should
#' match with the IDs of that `X` (its first column). Otherwise, should match with
#' the rows of `X` (the one to which the model was fit)
#' with numeration starting at 1 (should be an integer vector).
#' @param item Rows of `I`/`transX` (unseen columns of a new `X`) for which to make
#' predictions, with numeration starting at 1 (should be an integer vector).
#' See `Description` for details.
#' @param transX New `X` data for the items, transposed so that items denote rows
#' and columns correspond to old users (which were in the `X` to which the model was fit).
#' Note that the function will not do any reindexing - if the `X` to which the model
#' was fit was a `data.frame`, the user numeration can be found under
#' `model$info$user_mapping`.
#' 
#' Can be passed in the following formats:\itemize{
#' \item A sparse COO/triplets matrix, either from package
#' `Matrix` (class `dgTMatrix`), or from package `SparseM` (class `matrix.coo`).
#' \item A sparse matrix in CSR format, either from package
#' `Matrix` (class `dgRMatrix`), or from package `SparseM` (class `matrix.csr`).
#' Passing the input as CSR is faster than COO as it will be converted internally.
#' \item A sparse row vector from package `Matrix` (class `dsparseVector`).
#' \item A dense matrix from base R (class `matrix`), with missing entries set as NA.
#' \item A dense vector from base R (class `numeric`).
#' \item A `data.frame`.
#' }
#' @param weight Associated observation weights for entries in `transX`. If passed, must
#' have the same shape as `transX` - that is, if `transX` is a sparse matrix, should be a
#' numeric vector with length equal to the non-missing elements, if `transX` is a dense
#' matrix, should also be a dense matrix with the same number of rows and columns.
#' @param I New `I` data, with rows denoting new columns of the `X` matrix
#' (the one to which the model was fit) and/or rows of `transX`.
#' Can be passed in the same formats as `transX`, or additionally as a `data.frame`.
#' @param I_bin New binary columns of `I`. Must be passed as a dense matrix from
#' base R or as a `data.frame`.
#' @seealso \link{item_factors} \link{predict.cmfrec} \link{predict_new}
#' @return A numeric vector with the predicted value for each requested combination
#' of (user, item). Invalid combinations will be filled with NAs.
predict_new_items <- function(model, user, item=NULL,
                              transX=NULL, weight=NULL, I=NULL, I_bin=NULL) {
    accepted_models <- c("CMF", "CMF_implicit", "ContentBased", "OMF_explicit", "OMF_implicit")
    if (!NROW(intersect(class(model), accepted_models)))
        stop("Method only applicable to the following models: ",
             paste(accepted_models, collapse = ", "))
    if (model$info$only_prediction_info)
        stop("Cannot use this function after dropping non-essential matrices.")
    if ((NROW(I)) && (!NCOL(model$matrices$D) || !NROW(model$matrices$D)))
        stop("Model was not fit to item side info.")
    if (!is.null(I_bin) && (!NCOL(model$matrices$Db) || !NROW(model$matrices$Db)))
        stop("Model was not fit to binary side info ('I_bin').")
    
    if (!NROW(transX) && !NROW(I) && !NROW(I_bin))
        stop("Must pass at least one of 'transX', 'I', 'I_bin'.")
    
    if (is.null(user))
        stop("'user' cannot be empty.")
    if (!NROW(user))
        return(numeric())
    if (NROW(model$info$user_mapping))
        user <- as.integer(factor(user, model$info$user_mapping))
    if (inherits(user, c("numeric", "character", "matrix")))
        user <- as.integer(user)
    if (!inherits(user, "integer"))
        stop("'user' must be an integer vector.")
    user <- user - 1L
    
    if (!is.null(item)) {
        if (NROW(model$info$user_mapping))
            item <- as.integer(factor(item, model$info$item_mapping))
        if (inherits(item, c("numeric", "character", "matrix")))
            item <- as.integer(item)
        if (!inherits(item, "integer"))
            stop("'item' must be an integer vector.")
        item <- item - 1L
    } else {
        item <- seq(1L, NROW(user)) - 1L
    }
    
    obj_pass <- list(
        matrices = list(B = model$matrices$A, Bm = model$matrices$Am),
        info = list(user_mapping = model$info$item_mapping,
                    item_mapping = model$info$user_mapping,
                    n_orig = max(c(NCOL(model$matrices$A), NCOL(model$matrices$Am))),
                    include_all_X = TRUE)
        
    )
    
    processed_X      <-  process.new.X(obj_pass, transX, weight = weight,
                                       allow_sparse=TRUE, allow_null=TRUE,
                                       allow_reindex=FALSE)
    processed_I      <-  process.new.U(U = I, U_cols = model$info$I_cols,
                                       p = NCOL(model$matrices$D), name = "I",
                                       allow_sparse = TRUE,
                                       allow_null = as.logical(NROW(I_bin) || NROW(transX)),
                                       allow_na = class(model)[1L] %in% c("CMF", "CMF_implicit"),
                                       exact_shapes = !(class(model)[1L] %in% c("CMF", "CMF_implicit")))
    processed_I_bin  <-  process.new.U(U = I_bin, U_cols = model$info$I_bin_cols,
                                       p = NCOL(model$matrices$Db), name = "I_bin",
                                       allow_sparse=FALSE,
                                       allow_null=as.logical(NROW(I) || NROW(transX)),
                                       allow_na=TRUE, exact_shapes=FALSE)
    
    n_max <- max(c(processed_X$m, processed_I$m, processed_I_bin$m))
    if (!n_max)
        stop("Data (transX/I/I_bin) has zero rows.")
    if ("OMF_explicit" %in% class(model)) {
        if (NROW(processed_X$Xarr)) {
            if (processed_X$m < n_max)
                stop("'X' must have the same rows as 'I'.")
        }
    }
    if (NROW(intersect(class(model), c("OMF_explicit", "OMF_implicit")))) {
        if (NROW(processed_I$Uarr)) {
            if (processed_I$m < n_max)
                stop("'U' must have the same rows as 'X'.")
        }
    }
    
    scores <- numeric(NROW(user))
    
    model_class <- class(model)[1L]
    if ("CMF" %in% model_class) {
        Bbias <- numeric()
        if (NROW(model$matrices$item_bias))
            Bbias <- numeric(n_max)
        
        B <- matrix(0., ncol = n_max, nrow = model$info$k_item + model$info$k + model$info$k_main)
        ret_code <- .Call("call_factors_collective_explicit_multiple",
                          B, Bbias, n_max,
                          processed_I$Uarr, processed_I$m, processed_I$p,
                          model$info$ NA_as_zero_item, model$info$NA_as_zero,
                          model$info$nonneg,
                          processed_I$Urow, processed_I$Ucol, processed_I$Uval,
                          processed_I$Ucsr_p, processed_I$Ucsr_i, processed_I$Ucsr,
                          processed_I_bin$Uarr, processed_I_bin$m, processed_I_bin$p,
                          model$matrices$D, model$matrices$Db,
                          model$matrices$glob_mean, model$matrices$user_bias,
                          model$matrices$I_colmeans,
                          processed_X$Xval, processed_X$Xrow, processed_X$Xcol,
                          processed_X$Xcsr_p, processed_X$Xcsr_i, processed_X$Xcsr,
                          processed_X$Xarr, processed_X$n,
                          processed_X$Warr, processed_X$Wsp,
                          model$matrices$A,
                          model$matrices$Ai, model$info$add_implicit_features,
                          model$info$k, model$info$k_item, model$info$k_user, model$info$k_main,
                          swap.lambda(model$info$lambda), swap.lambda(model$info$l1_lambda),
                          model$info$scale_lam, model$info$scale_lam_sideinfo,
                          model$info$scale_bias_const, model$matrices$scaling_biasB,
                          model$info$w_main, model$info$w_item, model$info$w_implicit,
                          NCOL(model$matrices$A), model$info$include_all_X,
                          numeric(),
                          numeric(),
                          numeric(),
                          numeric(),
                          numeric(),
                          numeric(),
                          numeric(),
                          numeric(),
                          numeric(),
                          model$info$nthreads)
        check.ret.code(ret_code)
        ret_code <- .Call("call_predict_X_old_collective_explicit",
                          user, item, scores,
                          model$matrices$A, model$matrices$user_bias,
                          B, Bbias,
                          model$matrices$glob_mean,
                          model$info$k, model$info$k_user, model$info$k_item, model$info$k_main,
                          NCOL(model$matrices$A), n_max,
                          model$info$nthreads)
    } else if ("CMF_implicit" %in% model_class) {
        lambda <- ifelse(NROW(model$info$lambda) == 6L, model$info$lambda[3L], model$info$lambda)
        l1_lambda <- ifelse(NROW(model$info$l1_lambda) == 6L, model$info$l1_lambda[3L], model$info$l1_lambda)
        B <- matrix(0., ncol = n_max, nrow = model$info$k_item + model$info$k + model$info$k_main)
        ret_code <- .Call("call_factors_collective_implicit_multiple",
                          B, n_max,
                          processed_I$Uarr, processed_I$m, processed_I$p,
                          model$info$ NA_as_zero_item,
                          model$info$nonneg,
                          processed_I$Urow, processed_I$Ucol, processed_I$Uval,
                          processed_I$Ucsr_p, processed_I$Ucsr_i, processed_I$Ucsr,
                          processed_X$Xval, processed_X$Xrow, processed_X$Xcol,
                          processed_X$Xcsr_p, processed_X$Xcsr_i, processed_X$Xcsr,
                          model$matrices$A, NCOL(model$matrices$A),
                          model$matrices$D,
                          model$matrices$I_colmeans,
                          model$info$k, model$info$k_item, model$info$k_user, model$info$k_main,
                          lambda, l1_lambda, model$info$alpha, model$info$w_main, model$info$w_item,
                          model$info$w_main_multiplier,
                          model$info$apply_log_transf,
                          numeric(),
                          numeric(),
                          numeric(),
                          numeric(),
                          model$info$nthreads)
        check.ret.code(ret_code)
        ret_code <- .Call("call_predict_X_old_collective_implicit",
                          user, item, scores,
                          model$matrices$A,
                          B,
                          model$info$k, model$info$k_user, model$info$k_item, model$info$k_main,
                          NCOL(model$matrices$A), n_max,
                          model$info$nthreads)
    } else if ("ContentBased" %in% model_class) {
        B <- matrix(0., ncol = n_max, nrow = model$info$k)
        ret_code <- .Call("call_factors_content_based_mutliple",
                          B, n_max, model$info$k,
                          model$matrices$D, model$matrices$D_bias,
                          processed_I$Uarr, processed_I$p,
                          processed_I$Urow, processed_I$Ucol, processed_I$Uval,
                          processed_I$Ucsr_p, processed_I$Ucsr_i, processed_I$Ucsr,
                          model$info$nthreads)
        check.ret.code(ret_code)
        ret_code <- .Call("call_predict_X_old_collective_explicit",
                          user, item, scores,
                          model$matrices$Am, model$matrices$user_bias,
                          B, numeric(),
                          model$matrices$glob_mean,
                          model$info$k, 0L, 0L, 0L,
                          NCOL(model$matrices$Am), n_max,
                          model$info$nthreads)
    } else if ("OMF_explicit" %in% model_class) {
        Bbias <- numeric()
        if (NROW(model$matrices$item_bias))
            Bbias <- numeric(n_max)
        
        B <- matrix(0., ncol = n_max, nrow = model$info$k_sec + model$info$k + model$info$k_main)
        ret_code <- .Call("call_factors_offsets_explicit_multiple",
                          B, Bbias,
                          numeric(), n_max,
                          processed_I$Uarr, processed_I$p,
                          processed_I$Urow, processed_I$Ucol, processed_I$Uval,
                          processed_I$Ucsr_p, processed_I$Ucsr_i, processed_I$Ucsr,
                          processed_X$Xval, processed_X$Xrow, processed_X$Xcol,
                          processed_X$Xcsr_p, processed_X$Xcsr_i, processed_X$Xcsr,
                          processed_X$Xarr, processed_X$n,
                          processed_X$Warr, processed_X$Wsp,
                          model$matrices$Am, model$matrices$D,
                          model$matrices$D_bias,
                          model$matrices$glob_mean, model$matrices$user_bias,
                          model$info$k, model$info$k_sec, model$info$k_main,
                          model$info$w_item,
                          swap.lambda(model$info$lambda), FALSE,
                          numeric(),
                          numeric(),
                          numeric(),
                          model$info$nthreads)
        check.ret.code(ret_code)
        ret_code <- .Call("call_predict_X_old_offsets_explicit",
                          user, item, scores,
                          model$matrices$Am, model$matrices$user_bias,
                          B, Bbias,
                          model$matrices$glob_mean,
                          model$info$k, model$info$k_sec, model$info$k_main,
                          NCOL(model$matrices$Am), n_max,
                          model$info$nthreads)
    } else if ("OMF_implicit" %in% model_class) {
        B <- matrix(0., ncol = n_max, nrow = model$info$k)
        ret_code <- .Call("call_factors_offsets_implicit_multiple",
                          B, n_max,
                          numeric(),
                          processed_I$Uarr, processed_I$p,
                          processed_I$Urow, processed_I$Ucol, processed_I$Uval,
                          processed_I$Ucsr_p, processed_I$Ucsr_i, processed_I$Ucsr,
                          processed_X$Xval, processed_X$Xrow, processed_X$Xcol,
                          processed_X$Xcsr_p, processed_X$Xcsr_i, processed_X$Xcsr,
                          model$matrices$Am, model$matrices$D,
                          model$matrices$D_bias,
                          model$info$k, NCOL(model$matrices$Am),
                          model$info$lambda, model$info$alpha,
                          model$info$apply_log_transf,
                          numeric(),
                          model$info$nthreads)
        check.ret.code(ret_code)
        ret_code <- .Call("call_predict_X_old_offsets_implicit",
                          user, item, scores,
                          model$matrices$Am,
                          B,
                          model$info$k,
                          NCOL(model$matrices$Am), n_max,
                          model$info$nthreads)
    } else {
        stop("Unexpected error.")
    }
    check.ret.code(ret_code)
    return(scores)
}
