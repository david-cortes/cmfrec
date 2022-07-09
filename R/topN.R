
process.inputs.topN <- function(model, obj, user=NULL, a_vec=NULL, a_bias=NULL,
                                n=10L, include=NULL, exclude=NULL, output_score=FALSE) {
    if (!is.null(include) && !is.null(exclude))
        stop("Can only pass one of 'include' or 'exclude'.")
    if (inherits(include, "sparseVector")) {
        if (NROW(obj$info$item_mapping))
            stop("Cannot pass sparse vectors when fitting a model to a 'data.frame'.")
        include <- include@i
    }
    if (inherits(exclude, "sparseVector")) {
        if (NROW(obj$info$item_mapping))
            stop("Cannot pass sparse vectors when fitting a model to a 'data.frame'.")
        exclude <- exclude@i
    }
    
    output_score <- check.bool(output_score)
    n <- check.pos.int(n, TRUE)
    if (NROW(obj$info$item_mapping)) {
        if (!is.null(include))
            include <- as.integer(factor(include, obj$info$item_mapping))
        if (!is.null(exclude))
            exclude <- as.integer(factor(exclude, obj$info$item_mapping))
        if (!is.null(user))
            user <- as.integer(factor(user, obj$info$user_mapping))
    }
    
    if (is.null(a_bias))
        a_bias <- 0.
    
    if (!is.null(user)) {
        if (inherits(user, c("numeric", "character")))
            user <- as.integer(user)
        user <- check.pos.int(user, TRUE)
        if (model != "MostPopular") {
            m_max <- ifelse(model %in% c("OMF_explicit", "OMF_explicit", "ContentBased"),
                            NCOL(obj$matrices$Am), NCOL(obj$matrices$A))
            if (user > m_max)
                stop("'user' is outside the range of data passed to 'fit'.")
        } else {
            if (NROW(obj$matrices$user_bias)) {
                if (user > NROW(obj$matrices$user_bias))
                    stop("'user' is outside the range of data passed to 'fit'.")
            }
        }
        
        if (!(model %in% c("ContentBased", "MostPopular", "OMF_explicit", "OMF_implicit"))) {
            a_vec <- obj$matrices$A[, user, drop = TRUE]
        } else if (model != "MostPopular") {
            a_vec <- obj$matrices$Am[, user, drop = TRUE]
        }
        
        if (NROW(obj$matrices$user_bias))
            a_bias <- obj$matrices$user_bias[user]
    }
    
    if (!is.null(include)) {
        if (inherits(include, c("numeric", "character", "matrix")))
            include <- as.integer(include)
        if (!inherits(include, "integer"))
            stop("Invalid data type for 'include'.")
        
        if (model != "MostPopular") {
            if (max(include) > NCOL(obj$matrices$B))
                stop("'include' contains element that were not passed to 'fit'.")
        } else {
            if (max(include) > NROW(obj$matrices$item_bias))
                stop("'include' contains element that were not passed to 'fit'.")
        }
        
        include <- include - 1L
        if (any(include < 0L) || anyNA(include))
            stop("'include' contains invalid entries.")
        
        if (NROW(include) < n)
            stop("'n' is greater than the number of entries in 'include'.")
    }
    
    if (!is.null(exclude)) {
        if (inherits(exclude, c("numeric", "character", "matrix")))
            exclude <- as.integer(exclude)
        if (!inherits(exclude, "integer"))
            stop("Invalid data type for 'exclude'.")
        
        if (model != "MostPopular") {
            if (max(exclude) > NCOL(obj$matrices$B))
                stop("'exclude' contains element that were not passed to 'fit'.")
        } else {
            if (max(exclude) > NROW(obj$matrices$item_bias))
                stop("'exclude' contains element that were not passed to 'fit'.")
        }
        
        exclude <- exclude - 1L
        if (any(exclude < 0L) || anyNA(exclude))
            stop("'exclude' contains invalid entries.")
    }
    
    return(list(
        n = n,
        include = include,
        exclude = exclude,
        output_score = output_score,
        a_vec = a_vec,
        a_bias = a_bias
    ))
}

.topN <- function(model, obj, a_vec, a_bias=0.,
                  n=10L, include=NULL, exclude=NULL,
                  output_score=FALSE, reindex=TRUE) {
    outp_ix <- integer(length = n)
    outp_score <- numeric(length = ifelse(output_score, n, 0L))
    if (is.null(a_bias)) a_bias <- 0.
    
    if (model == "CMF") {
        ret_code <- .Call(call_topN_old_collective_explicit,
                          a_vec, a_bias,
                          obj$matrices$B,
                          obj$matrices$item_bias,
                          obj$matrices$glob_mean,
                          obj$info$k, obj$info$k_user, obj$info$k_item, obj$info$k_main,
                          include,
                          exclude,
                          outp_ix, outp_score,
                          obj$info$n_orig, NCOL(obj$matrices$B),
                          obj$info$include_all_X, obj$info$nthreads)
    } else if (model == "CMF_implicit") {
        ret_code <- .Call(call_topN_old_collective_implicit,
                          a_vec,
                          obj$matrices$B,
                          obj$info$k, obj$info$k_user, obj$info$k_item, obj$info$k_main,
                          include,
                          exclude,
                          outp_ix, outp_score,
                          NCOL(obj$matrices$B), obj$info$nthreads)
    } else if (model == "MostPopular") {
        ret_code <- .Call(call_topN_old_most_popular,
                          as.logical(NROW(obj$matrices$user_bias)),
                          a_bias,
                          obj$matrices$item_bias,
                          obj$matrices$glob_mean,
                          include,
                          exclude,
                          outp_ix, outp_score,
                          NROW(obj$matrices$item_bias))
    } else if (model == "ContentBased") {
        ret_code <- .Call(call_topN_old_content_based,
                          a_vec, a_bias,
                          obj$matrices$Bm,
                          obj$matrices$item_bias,
                          obj$matrices$glob_mean,
                          obj$info$k,
                          include,
                          exclude,
                          outp_ix, outp_score,
                          NCOL(obj$matrices$Bm), obj$info$nthreads)
    } else if (model == "OMF_explicit") {
        ret_code <- .Call(call_topN_old_offsets_explicit,
                          a_vec, a_bias,
                          obj$matrices$Bm,
                          obj$matrices$item_bias,
                          obj$matrices$glob_mean,
                          obj$info$k, obj$info$k_sec, obj$info$k_main,
                          include,
                          exclude,
                          outp_ix, outp_score,
                          NCOL(obj$matrices$Bm), obj$info$nthreads)
    } else if (model == "OMF_implicit") {
        ret_code <- .Call(call_topN_old_offsets_implicit,
                          a_vec,
                          obj$matrices$Bm,
                          obj$info$k,
                          include,
                          exclude,
                          outp_ix, outp_score,
                          NCOL(obj$matrices$Bm), obj$info$nthreads)
    } else {
        stop("Unexpected error.")
    }
    
    outp_ix <- outp_ix + 1L
    if (reindex && NROW(obj$info$item_mapping))
        outp_ix <- obj$info$item_mapping[outp_ix]
    
    if (output_score) {
        return(list(item=outp_ix, score=outp_score))
    } else {
        return(outp_ix)
    }
}

#' @export
#' @title Calulate top-N predictions for a new or existing user
#' @rdname topN
#' @description Determine top-ranked items for a user according to their predicted
#' values, among the items to which the model was fit.
#' 
#' Can produce rankings for existing users (which where in the `X` data to which
#' the model was fit) through function `topN`, or for new users (which were not
#' in the `X` data to which the model was fit, but for which there is now new
#' data) through function `topN_new`, assuming there is either `X` data, `U` data,
#' or both (i.e. can do cold-start and warm-start rankings).
#' 
#' For the \link{CMF} model, depending on parameter `include_all_X`, might recommend
#' items which had only side information if their predictions are high enough.
#' 
#' For the \link{ContentBased} model, might be used to rank new items (not present
#' in the `X` or `I` data to which the model was fit) given their
#' `I` data, for new users given their `U` data. For the other models, will only
#' rank existing items (columns of the `X` to which the model was fit) - see
#' \link{predict_new_items} for an alternative for the other models.
#' 
#' \bold{Important:} the model does not keep any copies of the original data, and
#' as such, it might recommend items that were already seen/rated/consumed by the
#' user. In order to avoid this, must manually pass the seen/rated/consumed entries
#' to the argument `exclude` (see details below).
#' 
#' This method produces an exact ranking by computing all item predictions
#' for a given user. As the number of items grows, this can become a rather
#' slow operation - for model serving purposes, it's usually a better idea
#' to obtain an an approximate top-N ranking through software such as
#' "hnsw" or "Milvus" from the calculated user factors and item factors.
#' @details Be aware that this function is multi-threaded. As such, if a large batch
#' of top-N predictions is to be calculated in parallel for different users
#' (through e.g. `mclapply` or similar), it's recommended to decrease the number
#' of threads in the model to 1 (e.g. `model$info$nthreads <- 1L`) and to set the
#' number of BLAS threads to 1 (through e.g. `RhpcBLASctl` or environment variables).
#' 
#' For better cold-start recommendations with \link{CMF_implicit}, one can also add
#' item biases by using the `CMF` model with parameters that would mimic `CMF_implicit`
#' plus the biases.
#' @param model A collective matrix factorization model from this package - see
#' \link{fit_models} for details.
#' @param user User (row of `X`) for which to rank items. If `X` to which the model
#' was fit was a `data.frame`, should pass an ID matching to the first column of `X`
#' (the user indices), otherwise should pass a row number for `X`, with numeration
#' starting at 1.
#' 
#' This is optional for the \link{MostPopular} model, but must be passed for all others.
#' 
#' For making recommendations about new users (that were not present in the `X` to
#' which the model was fit), should use `topN_new` and pass either `X` or `U` data.
#' 
#' For example usage, see the main section \link{fit_models}.
#' @param n Number of top-predicted items to output.
#' @param include If passing this, will only make a ranking among the item IDs
#' provided here. See the documentation for `user` for how the IDs should be passed.
#' This should be an integer or character vector, or alternatively, as a sparse vector
#' from the `Matrix` package (inheriting from class `sparseVector`),
#' from which the non-missing entries will be taken as those to include.
#' 
#' Cannot be used together with `exclude`.
#' @param exclude If passing this, will exclude from the ranking all the item IDs
#' provided here. See the documentation for `user` for how the IDs should be passed.
#' This should be an integer or character vector, or alternatively, as a sparse vector
#' from the `Matrix` package (inheriting from class `sparseVector`),
#' from which the non-missing entries will be taken as those to exclude.
#' 
#' Cannot be used together with `include`.
#' @param output_score Whether to also output the predicted values, in addition
#' to the indices of the top-predicted items.
#' @param X `X` data for a new user for which to make recommendations,
#' either as a numeric vector (class `numeric`), or as
#' a sparse vector from package `Matrix` (class `dsparseVector`). If the `X` to
#' which the model was fit was a `data.frame`, the column/item indices will have
#' been reindexed internally, and the numeration can be found under
#' `model$info$item_mapping`. Alternatively, can instead pass the column indices
#' and values and let the model reindex them (see `X_col` and `X_val`).
#' Should pass at most one of `X` or `X_col`+`X_val`.
#' 
#' Dense `X` data is not supported for `CMF_implicit` or `OMF_implicit`.
#' @param X_col `X` data for a new user for which to make recommendations,
#' in sparse vector format, with `X_col` denoting the
#' items/columns which are not missing. If the `X` to which the model was fit was
#' a `data.frame`, here should pass IDs matching to the second column of that `X`,
#' which will be reindexed internally. Otherwise, should have column indices with
#' numeration starting at 1 (passed as an integer vector).
#' Should pass at most one of `X` or `X_col`+`X_val`.
#' @param X_val `X` data for a new user for which to make recommendations,
#' in sparse vector format, with `X_val` denoting the
#' associated values to each entry in `X_col`
#' (should be a numeric vector of the same length as `X_col`).
#' Should pass at most one of `X` or `X_col`+`X_val`.
#' @param weight (Only for the explicit-feedback models)
#' Associated weight to each non-missing observation in `X`. Must have the same
#' number of entries as `X` - that is, if passing a dense vector of length `n`,
#' `weight` should be a numeric vector of length `n` too, if passing a sparse
#' vector, should have a lenght corresponding to the number of non-missing elements.
#' @param U `U` data for a new user for which to make recommendations,
#' either as a numeric vector (class `numeric`), or as a
#' sparse vector from package `Matrix` (class `dsparseVector`). Alternatively,
#' if `U` is sparse, can instead pass the indices of the non-missing columns
#' and their values separately (see `U_col`).
#' Should pass at most one of `U` or `U_col`+`U_val`.
#' @param U_col `U` data for a new user for which to make recommendations,
#' in sparse vector format, with `U_col` denoting the
#' attributes/columns which are not missing. Should have numeration starting at 1
#' (should be an integer vector).
#' Should pass at most one of `U` or `U_col`+`U_val`.
#' @param U_val `U` data for a new user for which to make recommendations,
#' in sparse vector format, with `U_val` denoting the
#' associated values to each entry in `U_col`
#' (should be a numeric vector of the same length as `U_col`).
#' Should pass at most one of `U` or `U_col`+`U_val`.
#' @param U_bin Binary columns of `U` for a new user for which to make recommendations,
#' on which a sigmoid transformation will be
#' applied. Should be passed as a numeric vector. Note that `U` and `U_bin` are
#' not mutually exclusive.
#' @param I (Only for the `ContentBased` model)
#' New `I` data to rank for the given user, with rows denoting new columns of the `X` matrix.
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
#' When passing `I`, the item indices in `include`, `exclude`, and in the resulting
#' output refer to rows of `I`, and the ranking will be made only among the
#' rows of `I` (that is, they will not be compared against the old `X` data).
#' @param exact (In the `OMF_explicit` model)
#' Whether to calculate `A` and `Am` with the regularization applied
#' to `A` instead of to `Am` (if using the L-BFGS method, this is how the model
#' was fit). This is usually a slower procedure.
#' @param ... Not used.
#' @return If passing `output_score=FALSE` (the default), will output the
#' indices of the top-predicted elements. If passing `output_score=TRUE`,
#' will pass a list with two elements:\itemize{
#' \item `item`: The indices of the top-predicted elements.
#' \item `score`: The predicted value for each corresponding element in `item`.
#' }
#' If the `X` to which the model was fit was a `data.frame` (and unless passing `I`),
#' the item indices will be taken from the same IDs in `X` (its second column) - but
#' be aware that in this case they will usually be returned as `character`.
#' Otherwise, will return the indices of the top-predicted columns of `X`
#' (or rows of `I` if passing it) with numeration starting at 1.
#' @seealso \link{factors_single} \link{predict.cmfrec} \link{predict_new}
topN <- function(model, user=NULL, n=10L, include=NULL, exclude=NULL, output_score=FALSE) {
    supported_models <- c("CMF", "CMF_implicit",
                          "MostPopular", "ContentBased",
                          "OMF_implicit", "OMF_explicit")
    if (!NROW(intersect(class(model), supported_models)))
        stop("Invalid model object - supported classes: ", paste(supported_models, collapse=", "))
    if (is.null(user) && !("MostPopular" %in% class(model)))
        stop("'user' cannot be empty for this model.")
    if (model$info$only_prediction_info)
        stop("Cannot use this function after dropping non-essential matrices.")
    inputs <- process.inputs.topN(class(model)[1L], model,
                                  user = user, n = n,
                                  include = include, exclude = exclude,
                                  output_score = output_score)
    return(.topN(class(model)[1L], model,
                 a_vec = inputs$a_vec, a_bias = inputs$a_bias,
                 n = inputs$n, include = inputs$include, exclude = inputs$exclude,
                 output_score = inputs$output_score))
}

#' @export
#' @rdname topN
topN_new <- function(model, ...) {
    UseMethod("topN_new")
}

#' @export
#' @rdname topN
topN_new.CMF <- function(model, X=NULL, X_col=NULL, X_val=NULL,
                         U=NULL, U_col=NULL, U_val=NULL, U_bin=NULL, weight=NULL,
                         n=10L, include=NULL, exclude=NULL,
                         output_score=FALSE, ...) {
    inputs <- process.inputs.topN(class(model)[1L], model,
                                  n = n,
                                  include = include, exclude = exclude,
                                  output_score = output_score)
    factors <- factors_single.CMF(model = model, X = X, X_col = X_col, X_val = X_val,
                                  weight = weight,
                                  U = U, U_col = U_col, U_val = U_val, U_bin = U_bin,
                                  output_bias = as.logical(NROW(model$matrices$user_bias)))
    if (is.list(factors)) {
        a_vec <- factors$factors
        a_bias <- factors$bias
    } else {
        a_vec <- factors
        a_bias <- NULL
    }
    return(.topN(class(model)[1L], model,
                 a_vec = a_vec, a_bias = a_bias,
                 n = inputs$n, include = inputs$include, exclude = inputs$exclude,
                 output_score = inputs$output_score))
}

#' @export
#' @rdname topN
topN_new.CMF_implicit <- function(model, X=NULL, X_col=NULL, X_val=NULL,
                                  U=NULL, U_col=NULL, U_val=NULL,
                                  n=10L, include=NULL, exclude=NULL,
                                  output_score=FALSE, ...) {
    inputs <- process.inputs.topN(class(model)[1L], model,
                                  n = n,
                                  include = include, exclude = exclude,
                                  output_score = output_score)
    factors <- factors_single.CMF_implicit(model = model, X = X, X_col = X_col, X_val = X_val,
                                           U = U, U_col = U_col, U_val = U_val)
    return(.topN(class(model)[1L], model,
                 a_vec = factors,
                 n = inputs$n, include = inputs$include, exclude = inputs$exclude,
                 output_score = inputs$output_score))
    
}

#' @export
#' @rdname topN
topN_new.ContentBased <- function(model, U=NULL, U_col=NULL, U_val=NULL, I=NULL,
                                  n=10L, include=NULL, exclude=NULL,
                                  output_score=FALSE, ...) {
    if (!is.null(I) && (!is.null(include) || !is.null(exclude)))
        stop("Cannot pass 'include' or 'exclude' when passing 'I' data.")
    
    if (is.null(I)) {
        inputs <- process.inputs.topN(class(model)[1L], model,
                                      n = n,
                                      include = include, exclude = exclude,
                                      output_score = output_score)
        factors <- factors_single.ContentBased(model = model, U = U, U_col = U_col, U_val = U_val)
        return(.topN(class(model)[1L], model,
                     a_vec = factors,
                     n = inputs$n, include = inputs$include, exclude = inputs$exclude,
                     output_score = inputs$output_score))
        
    } else {
        processed_U <- process.new.U.single(U, U_col, U_val,
                                            model$info$user_mapping, NCOL(model$matrices$C),
                                            model$info$U_cols)
        processed_I <- process.new.U(I, model$info$I_cols, NCOL(model$matrices$D),
                                     allow_sparse=TRUE, allow_null=FALSE,
                                     allow_na=FALSE, exact_shapes=TRUE)
        
        outp_ix <- integer(length = n)
        outp_score <- numeric(length = ifelse(output_score, n, 0L))
        
        ret_code <- .Call(call_topN_new_content_based,
                          model$info$k, processed_I$m,
                          processed_U$U, processed_U$p,
                          processed_U$U_val, processed_U$U_col,
                          processed_I$Uarr, processed_I$p,
                          processed_I$Urow, processed_I$Ucol, processed_I$Uval,
                          processed_I$Ucsr_p, processed_I$Ucsr_i, processed_I$Ucsr,
                          model$matrices$C, model$matrices$C_bias,
                          model$matrices$D, model$matrices$D_bias,
                          model$matrices$glob_mean,
                          outp_ix, outp_score,
                          model$info$nthreads)
        check.ret.code(ret_code)
        outp_ix <- outp_ix + 1L
        if (!is.null(row.names(I))) {
            outp_ix <- row.names(I)[outp_ix]
        }
        if (output_score) {
            return(list(item=outp_ix, score=outp_score))
        } else {
            return(outp_ix)
        }
    }
}

#' @export
#' @rdname topN
topN_new.OMF_explicit <- function(model, X=NULL, X_col=NULL, X_val=NULL,
                                  U=NULL, U_col=NULL, U_val=NULL,  weight=NULL, exact=FALSE,
                                  n=10L, include=NULL, exclude=NULL,
                                  output_score=FALSE, ...) {
    inputs <- process.inputs.topN(class(model)[1L], model,
                                  n = n,
                                  include = include, exclude = exclude,
                                  output_score = output_score)
    factors <- factors_single.OMF_explicit(model = model, X = X, X_col = X_col, X_val = X_val,
                                           weight = weight,
                                           U = U, U_col = U_col, U_val = U_val,
                                           output_bias = as.logical(NROW(model$matrices$user_bias)),
                                           output_A = FALSE,
                                           exact = exact)
    if (is.list(factors)) {
        a_vec <- factors$factors
        a_bias <- factors$bias
    } else {
        a_vec <- factors
        a_bias <- NULL
    }
    return(.topN(class(model)[1L], model,
                 a_vec = a_vec, a_bias = a_bias,
                 n = inputs$n, include = inputs$include, exclude = inputs$exclude,
                 output_score = inputs$output_score))
}

#' @export
#' @rdname topN
topN_new.OMF_implicit <- function(model, X=NULL, X_col=NULL, X_val=NULL,
                                  U=NULL, U_col=NULL, U_val=NULL,
                                  n=10L, include=NULL, exclude=NULL,
                                  output_score=FALSE, ...) {
    inputs <- process.inputs.topN(class(model)[1L], model,
                                  n = n,
                                  include = include, exclude = exclude,
                                  output_score = output_score)
    a_vec <- factors_single.CMF_implicit(model = model, X = X, X_col = X_col, X_val = X_val,
                                         U = U, U_col = U_col, U_val = U_val)
    return(.topN(class(model)[1L], model,
                 a_vec = a_vec,
                 n = inputs$n, include = inputs$include, exclude = inputs$exclude,
                 output_score = inputs$output_score))
}
