#' @export
#' @title Predict entries in the factorized `X` matrix
#' @description Predict entries in the `X` matrix according to the model
#' at the combinations [row,column] given by the entries in
#' `user` and `item` (e.g. passing `user=c(1,2,3), item=c(1,1,1)` will predict
#' X[1,1], X[2,1], X[3,1]).
#' 
#' Alternatively, might pass a sparse matrix, in which case it will make
#' predictions for all of its non-missing entries.
#' 
#' Invalid combinations (e.g. rows and columns outside of the range of `X` to
#' which the model was fit) will be filled with global mean plus biases if applicable
#' for `CMF_explicit`, and with NAs for the other models.
#' 
#' For example usage, see the main section \link{fit_models}.
#' @param object A collective matrix factorization model from this package - see
#' \link{fit_models} for details.
#' @param user The user IDs for which to make predictions. If `X` to which the model
#' was fit was a `data.frame`, should pass IDs matching to the first column of `X`
#' (the user indices, should be a character vector),
#' otherwise should pass row numbers for `X`, with numeration
#' starting at 1 (should be an integer vector).
#' 
#' If passing a single entry for `user` and `item` has more entries, will
#' predict all the entries in `item` for that single `user.`
#' 
#' Alternatively, might instead pass a sparse matrix in COO/triplets formats,
#' for which the \bold{non-missing} entries will be predicted, in which case it
#' its not necessary to pass `item`.
#' 
#' If passing a sparse matrix, can be from package `Matrix` (class `dgTMatrix` or `ngTMatrix`)
#' or from package `SparseM` (class `matrix.coo`). If using the package `softImpute`,
#' its objects of class `incomplete` might be convertable to `Matrix` objects through
#' e.g. `as(as(X, "TsparseMatrix"), "ngTMatrix")`.
#' @param item The item IDs for which to make predictions - see the documentation
#' about `user` for details about the indexing.
#' 
#' If passing a single entry for `item` and `user` has more entries, will
#' predict all the entries in `user` for that single `item`.
#' 
#' If passing a sparse matrix as `user`, `item` will be ignored.
#' @param ... Not used.
#' @return A numeric vector with the predicted values at the requested combinations.
#' If the `user` passed was a sparse matrix, and it was not of class `ngTMatrix`,
#' will instead return a sparse matrix of the same format, with the non-missing entries
#' set to the predicted values.
#' @seealso \link{predict_new} \link{topN}
predict.cmfrec <- function(object, user, item=NULL, ...) {
    if (object$info$only_prediction_info)
        stop("Cannot use this function after dropping non-essential matrices.")
    return_mat <- FALSE
    if (NROW(intersect(class(user), c("dgTMatrix", "matrix.coo", "ngTMatrix")))) {
        mat_out <- user
        return_mat <- TRUE
        if (("dgTMatrix" %in% class(mat_out)) || ("ngTMatrix" %in% class(mat_out))) {
            user <- mat_out@i + 1L
            item <- mat_out@j + 1L
        } else if ("matrix.coo" %in% class(mat_out)) {
            user <- mat_out@ia
            item <- mat_out@ja
        } else {
            stop("Unexpected error.")
        }
        if ("ngTMatrix" %in% class(mat_out))
            return_mat <- FALSE
    }
    
    if (NROW(user) == 0L && NROW(item) == 0L) {
        if (!return_mat)
            return(numeric())
        else
            return(mat_out)
    }
    
    if (("MostPopular" %in% class(object)) && !NROW(object$matrices$user_bias)) {
        user <- 1L
    }
    
    if (!NROW(user) || !NROW(item))
        stop("Must pass 'user' and 'item' together.")
    if (NROW(user) != NROW(item)) {
        if (NROW(user) == 1L && NROW(item) > 1L) {
            user <- rep(user, NROW(item))
        } else if (NROW(item) == 1L && NROW(user) > 1L) {
            item <- rep(item, NROW(user))
        } else {
            stop("'user' and 'item' must have the same number of entries.")
        }
    }
    
    if (NROW(object$info$user_mapping)) {
        user <- as.integer(factor(user, object$info$user_mapping))
        item <- as.integer(factor(item, object$info$item_mapping))
    } else {
        if (NROW(intersect(class(user), c("numeric", "character", "matrix"))))
            user <- as.integer(user)
        if (NROW(intersect(class(item), c("numeric", "character", "matrix"))))
            item <- as.integer(item)
        if (!("integer" %in% class(user)))
            stop("'user' must be an integer vector.")
        if (!("integer" %in% class(item)))
            stop("'item' must be an integer vector.")
    }
    user <- user - 1L
    item <- item - 1L
    if (NROW(user) != NROW(item))
        stop("'user' and 'item' must have the same number of entries.")
    
    scores <- numeric(length = NROW(user))
    
    if ("CMF" %in% class(object)) {
        ret_code <- .Call("call_predict_X_old_collective_explicit",
                          user, item, scores,
                          object$matrices$A, object$matrices$user_bias,
                          object$matrices$B, object$matrices$item_bias,
                          object$matrices$glob_mean,
                          object$info$k, object$info$k_user, object$info$k_item, object$info$k_main,
                          NCOL(object$matrices$A), NCOL(object$matrices$B),
                          object$info$nthreads)
    } else if ("CMF_implicit" %in% class(object)) {
        ret_code <- .Call("call_predict_X_old_collective_implicit",
                          user, item, scores,
                          object$matrices$A,
                          object$matrices$B,
                          object$info$k, object$info$k_user, object$info$k_item, object$info$k_main,
                          NCOL(object$matrices$A), NCOL(object$matrices$B),
                          object$info$nthreads)
    } else if ("MostPopular" %in% class(object)) {
        ret_code <- .Call("call_predict_X_old_most_popular",
                          user, item, scores,
                          object$matrices$user_bias, object$matrices$item_bias,
                          object$matrices$glob_mean,
                          NROW(object$matrices$user_bias),
                          NROW(object$matrices$item_bias))
    } else if ("ContentBased" %in% class(object)) {
        ret_code <- .Call("call_predict_X_old_collective_explicit",
                          user, item, scores,
                          object$matrices$Am, object$matrices$user_bias,
                          object$matrices$Bm, object$matrices$item_bias,
                          object$matrices$glob_mean,
                          object$info$k, 0L, 0L, 0L,
                          NCOL(object$matrices$Am), NCOL(object$matrices$Bm),
                          object$info$nthreads)
    } else if ("OMF_explicit" %in% class(object)) {
        ret_code <- .Call("call_predict_X_old_offsets_explicit",
                          user, item, scores,
                          object$matrices$Am, object$matrices$user_bias,
                          object$matrices$Bm, object$matrices$item_bias,
                          object$matrices$glob_mean,
                          object$info$k, object$info$k_sec, object$info$k_main,
                          NCOL(object$matrices$Am), NCOL(object$matrices$Bm),
                          object$info$nthreads)
    } else if ("OMF_implicit" %in% class(object)) {
        ret_code <- .Call("call_predict_X_old_offsets_implicit",
                          user, item, scores,
                          object$matrices$Am,
                          object$matrices$Bm,
                          object$info$k,
                          NCOL(object$matrices$Am), NCOL(object$matrices$Bm),
                          object$info$nthreads)
    } else {
        stop("Unsupported model type.")
    }
    
    check.ret.code(ret_code)
    if (!return_mat) {
        return(scores)
    } else {
        if ("dgTMatrix" %in% class(mat_out)) {
            mat_out@x   <- scores
        } else if ("matrix.coo" %in% class(mat_out)) {
            mat_out@ra  <- scores
        } else {
            stop("Unsupported iput type.")
        }
        return(mat_out)
    }
}
