#' @export
#' @title Swap users and items in the model
#' @description This function will swap the users and items in a given matrix
#' factorization model. Since the package functionality is user-centric, it
#' is generally not possible or not efficient to make predictions about items
#' (e.g. recommend users for a given item or calculate new item factors).
#' 
#' This function allows using the same API while avoiding model refitting or deep
#' copies of data by simply swapping the matrices, IDs, and hyperparameters
#' as necessary.
#' 
#' The resulting object can be used with the same functions as the original,
#' such as \link{topN} or \link{factors}, but any mention of "user" in the
#' functions will now mean "items".
#' @param model A collective matrix factorization model from this package - see
#' \link{fit_models} for details.
#' @param precompute Whether to calculate the precomputed matrices for speeding up
#' predictions in new data.
#' @return The same model object as before, but with the internal data
#' swapped in order to make predictions about items. If passing `precompute=TRUE`,
#' it will also generate precomputed matrices which can be used to speed up predictions.
#' @examples
#' library(cmfrec)
#' 
#' ### Generate a small random matrix
#' n_users <- 10L
#' n_items <- 8L
#' k <- 3L
#' set.seed(1)
#' X <- matrix(rnorm(n_users*n_items), nrow=n_users)
#' 
#' ### Factorize it
#' model <- CMF(X, k=k, verbose=FALSE, nthreads=1L)
#' 
#' ### Now swap the users and items
#' model.swapped <- swap.users.and.items(model)
#' 
#' ### These will now throw the same result
#' ### (up to numerical precision)
#' item_factors(model, X[, 1])
#' factors_single(model.swapped, X[, 1])
#' 
#' ### Swapping it again restores the original
#' model.restored <- swap.users.and.items(model.swapped)
#' 
#' ### These will throw the same result
#' topN(model, user=2, n=3)
#' topN(model.restored, user=2, n=3)
swap.users.and.items <- function(model, precompute = TRUE) {
    if (!("cmfrec" %in% class(model)))
        stop("Method is only applicable to objects from this package.")
    
    if ("MostPopular" %in% class(model)) {
        if (model$info$implicit)
            stop("Cannot swap users and items for MostPopular-implicit.")
        if (!NROW(model$matrices$user_bias))
            stop("Swapping users/items not meaningful for MostPopular with 'user_bias=FALSE'")
    }
    
    if (model$info$only_prediction_info)
        stop("Cannot use this function after dropping non-essential matrices.")
    
    new_model <- list(
        info = list(
            w_main_multiplier = model$info$w_main_multiplier,
            w_main = model$info$w_main,
            w_user = model$info$w_item,
            w_item = model$info$w_user,
            w_implicit = model$info$w_implicit,
            n_orig = max(c(NCOL(model$matrices$A), NCOL(model$matrices$Am))),
            k = model$info$k,
            k_user = model$info$k_item,
            k_item = model$info$k_user,
            k_main = model$info$k_main,
            k_sec = model$info$k_sec,
            lambda = swap.lambda(model$info$lambda),
            l1_lambda = swap.lambda(model$info$l1_lambda),
            alpha = model$info$alpha,
            nfev = model$info$nfev,
            nupd = model$info$nupd,
            user_mapping = model$info$item_mapping,
            item_mapping = model$info$user_mapping,
            U_cols = model$info$I_cols,
            I_cols = model$info$U_cols,
            U_bin_cols = model$info$I_bin_cols,
            I_bin_cols = model$info$U_bin_cols,
            implicit = model$info$implicit,
            apply_log_transf = model$info$apply_log_transf,
            NA_as_zero = model$info$NA_as_zero,
            NA_as_zero_user = model$info$NA_as_zero_item,
            NA_as_zero_item = model$info$NA_as_zero_user,
            nonneg = model$info$nonneg,
            add_implicit_features = model$info$add_implicit_features,
            include_all_X = model$info$include_all_X,
            center = model$info$center,
            scale_lam = model$info$scale_lam,
            scale_lam_sideinfo = model$info$scale_lam_sideinfo,
            only_prediction_info = model$info$only_prediction_info,
            nthreads = model$info$nthreads
        ),
        matrices = list(
            user_bias = model$matrices$item_bias,
            item_bias = model$matrices$user_bias,
            A = model$matrices$B,
            B = model$matrices$A,
            C = model$matrices$D,
            D = model$matrices$C,
            Cb = model$matrices$Db,
            Db = model$matrices$Cb,
            C_bias = model$matrices$D_bias,
            D_bias = model$matrices$C_bias,
            Ai = model$matrices$Bi,
            Bi = model$matrices$Ai,
            Am = model$matrices$Bm,
            Bm = model$matrices$Am,
            glob_mean = model$matrices$glob_mean,
            U_colmeans = model$matrices$I_colmeans,
            I_colmeans = model$matrices$U_colmeans
        ),
        precomputed = get.empty.precomputed()
    )
    class(new_model) <- class(model)
    
    if (precompute) {
        if (NROW(intersect(class(model), c("CMF", "CMF_implicit")))) {
            new_model <- precompute.for.predictions(new_model)
        } else if ("OMF_explicit" %in% class(model)) {
            user_bias  <-  as.logical(NROW(new_model$matrices$user_bias))
            n_max      <-  NCOL(new_model$matrices$Bm)
            k          <-  new_model$info$k
            k_sec      <-  new_model$info$k_sec
            k_main     <-  new_model$info$k_main
            if (user_bias) {
                new_model$precomputed$Bm_plus_bias <- matrix(0., ncol=n_max, nrow=k_sec+k+k_main+1L)
            }
            new_model$precomputed$BtB <- matrix(0., nrow=k_sec+k+k_main+user_bias, ncol=k_sec+k+k_main+user_bias)
            new_model$precomputed$TransBtBinvBt <- matrix(0., ncol=n_max, nrow=k_sec+k+k_main+user_bias)
            ret_code <- .Call("call_precompute_collective_explicit",
                              new_model$matrices$Bm, n_max, n_max, TRUE,
                              numeric(), 0L,
                              numeric(), FALSE,
                              numeric(), numeric(1L), FALSE,
                              new_model$info$k_sec + new_model$info$k + new_model$info$k_main,
                              0L, 0L, 0L,
                              as.logical(NROW(new_model$matries$user_bias)),
                              FALSE,
                              new_model$info$lambda,
                              FALSE, FALSE,
                              1., 1., 1.,
                              new_model$precomputed$Bm_plus_bias,
                              new_model$precomputed$BtB,
                              new_model$precomputed$TransBtBinvBt,
                              numeric(),
                              numeric(),
                              numeric(),
                              numeric(),
                              numeric())
            check.ret.code(ret_code)
        } else if ("OMF_implicit" %in% class(model)) {
            new_model$precomputed$BtB <- matrix(0., nrow=new_model$info$k, ncol=new_model$info$k)
            ret_code <- .Call("call_precompute_collective_implicit",
                              new_model$matrices$Bm, NCOL(new_model$matrices$Bm),
                              numeric(), 0L,
                              new_model$info$k, 0L, 0L, 0L,
                              new_model$info$lambda, 1., 1., 1.,
                              FALSE,
                              TRUE,
                              new_model$precomputed$BtB,
                              numeric(),
                              numeric())
            check.ret.code(ret_code)
        }
    }
    
    return(new_model)
}

#' @export
#' @title Drop matrices that are not used for prediction
#' @description Drops all the matrices in the model object which are not
#' used for calculating new user factors (either warm or cold), such as the
#' user biases or the item factors.
#' 
#' This is intended at decreasing memory usage in production systems which
#' use this software for calculation of user factors or top-N recommendations.
#' 
#' Can additionally drop some of the precomputed matrices which are only
#' taken in special circumstances such as when passing dense data with
#' no missing values - however, predictions that would have otherwise used
#' these matrices will become slower afterwards.
#' 
#' After dropping these non-essential matrices, it will not be possible
#' anymore to call certain methods such as `predict` or `swap.users.and.items`.
#' The methods which are intended to continue working afterwards are:\itemize{
#' \item \link{factors_single}
#' \item \link{factors}
#' \item \link{topN_new}
#' }
#' 
#' This method is only available for `CMF` and `CMF_implicit` model objects.
#' @details After calling this function and reassigning the output to the
#' original model object, one might need to call the garbage collector (by
#' running `gc()`) before any of the freed memory is shown as available.
#' @param model A model object as returned by \link{CMF} or \link{CMF_implicit}.
#' @param drop_precomputed Whether to drop the less commonly used prediction
#' matrices (see documentation above for more details).
#' @return The model object with the non-essential matrices dropped.
drop.nonessential.matrices <- function(model, drop_precomputed=TRUE) {
    if (!inherits(model, c("CMF", "CMF_implicit")))
        stop("Method is only applicable to 'CMF' and 'CMF_implicit'.")
    
    drop_precomputed <- check.bool(drop_precomputed, "drop_precomputed")
    
    model$info$user_mapping  <-  character()
    model$info$I_cols        <-  character()
    model$info$I_bin_cols    <-  character()
    
    model$matrices$A   <-  matrix(numeric(), nrow=0L, ncol=0L)
    model$matrices$Ai  <-  matrix(numeric(), nrow=0L, ncol=0L)
    model$matrices$D   <-  matrix(numeric(), nrow=0L, ncol=0L)
    model$matrices$Db  <-  matrix(numeric(), nrow=0L, ncol=0L)
    model$matrices$Am  <-  matrix(numeric(), nrow=0L, ncol=0L)
    model$matrices$I_colmeans  <-  numeric()
    model$matrices$user_bias   <-  numeric()
    model$matrices$D_bias      <-  numeric()
    
    if (NROW(model$precomputed$B_plus_bias))
        model$matrices$B       <-  matrix(numeric(), nrow=0L, ncol=0L)
    
    if (drop_precomputed) {
        model$precomputed$TransBtBinvBt  <-  matrix(numeric(), nrow=0L, ncol=0L)
        model$precomputed$TransCtCinvCt  <-  matrix(numeric(), nrow=0L, ncol=0L)
        model$precomputed$BeTBeChol      <-  matrix(numeric(), nrow=0L, ncol=0L)
        model$precomputed$BeTBe          <-  matrix(numeric(), nrow=0L, ncol=0L)
    }
    
    model$info$only_prediction_info <- TRUE
    return(model)
}
