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
            seed = model$info$seed,
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
            ret_code <- .Call(call_precompute_collective_explicit,
                              new_model$matrices$Bm, n_max, n_max, TRUE,
                              numeric(), 0L,
                              numeric(), FALSE,
                              numeric(), numeric(1L), FALSE,
                              numeric(), FALSE,
                              new_model$info$k_sec + new_model$info$k + new_model$info$k_main,
                              0L, 0L, 0L,
                              as.logical(NROW(new_model$matries$user_bias)),
                              FALSE,
                              new_model$info$lambda,
                              FALSE, FALSE, FALSE, 0.,
                              1., 1., 1.,
                              new_model$precomputed$Bm_plus_bias,
                              new_model$precomputed$BtB,
                              new_model$precomputed$TransBtBinvBt,
                              numeric(),
                              numeric(),
                              numeric(),
                              numeric(),
                              numeric(),
                              numeric())
            check.ret.code(ret_code)
        } else if ("OMF_implicit" %in% class(model)) {
            new_model$precomputed$BtB <- matrix(0., nrow=new_model$info$k, ncol=new_model$info$k)
            ret_code <- .Call(call_precompute_collective_implicit,
                              new_model$matrices$Bm, NCOL(new_model$matrices$Bm),
                              numeric(), 0L,
                              numeric(), FALSE,
                              new_model$info$k, 0L, 0L, 0L,
                              new_model$info$lambda, 1., 1., 1.,
                              FALSE,
                              TRUE,
                              new_model$precomputed$BtB,
                              numeric(),
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

#' @export
#' @title Create a CMF model object from fitted matrices
#' @description Creates a `CMF` or `CMF_implicit` model object based on fitted
#' latent factor matrices, which might have been obtained from a different software.
#' For example, the package `recosystem` has functionality for obtaining these matrices,
#' but not for producing recommendations or latent factors for new users, for which
#' this function can come in handy as it will turn such model into a `CMF` model which
#' provides all such functionality.
#' 
#' This is only available for models without side information, and does not support
#' user/item mappings.
#' @param A The obtained user factors (numeric matrix). Dimension is [k, n_users].
#' @param B The obtained item factors (numeric matrix). Dimension is [k, n_items].
#' @param glob_mean The obtained global mean, if the model is for explicit feedback
#' and underwent centering. If passing zero, will assume that the values are not to
#' be centered.
#' @param implicit Whether this is an implicit-feedback model.
#' @param precompute Whether to generate pre-computed matrices which can help to speed
#' up computations on new data (see \link{fit_models} for more details).
#' @param user_bias The obtained user biases (numeric vector).
#' If passing `NULL`, will assume that the model did not include user biases.
#' Dimension is [n_users].
#' @param item_bias The obtained item biases (numeric vector).
#' If passing `NULL`, will assume that the model did not include item biases.
#' Dimension is [n_item].
#' @param lambda Regularization parameter for the L2 norm of the model matrices
#' (see \link{fit_models} for more details). Can pass different parameters for each.
#' @param scale_lam In the explicit-feedback models, whether to scale the regularization
#' parameter according to the number of entries. This should always be assumed `TRUE`
#' for models that are fit through stochastic procedures.
#' @param l1_lambda Regularization parameter for the L1 norm of the model matrices.
#' Same format as for `lambda`.
#' @param nonneg Whether the model matrices should be constrained to be non-negative.
#' @param NA_as_zero When passing sparse matrices, whether to take missing entries as
#' zero (counting them towards the optimization objective), or to ignore them.
#' @param scaling_biasA If passing it, will assume that the model uses the option
#' `scale_bias_const=TRUE`, and will use this number as scaling
#' for the regularization of the user biases.
#' @param scaling_biasB If passing it, will assume that the model uses the option
#' `scale_bias_const=TRUE`, and will use this number as scaling
#' for the regularization of the item biases.
#' @param apply_log_transf If passing `implicit=TRUE`, whether to apply a logarithm
#' transformation on the values of `X`.
#' @param alpha If passing `implicit=TRUE`, multiplier to apply to the confidence scores
#' given by `X`.
#' @param nthreads Number of parallel threads to use for further computations.
#' @return A `CMF` (if passing `implicit=FALSE`) or `CMF_implicit` (if passing
#' `implicit=TRUE`) model object without side information, for which the usual
#' prediction functions such as \link{topN} and \link{topN_new} can be used as if
#' it had been fitted through this software.
#' @examples 
#' ### Example 'adopting' a model from 'recosystem'
#' library(cmfrec)
#' library(recosystem)
#' library(recommenderlab)
#' library(MatrixExtra)
#' 
#' ### Fitting a model with 'recosystem'
#' data("MovieLense")
#' X <- as.coo.matrix(MovieLense@data)
#' r <- Reco()
#' r$train(data_memory(X@i, X@j, X@x, index1=FALSE),
#'         out_model = NULL,
#'         opts = list(dim=10, costp_l2=0.1, costq_l2=0.1,
#'                     verbose=FALSE, nthread=1))
#' matrices <- r$output(out_memory(), out_memory())
#' glob_mean <- as.numeric(r$model$matrices$b)
#' 
#' ### Now converting it to CMF
#' model <- CMF.from.model.matrices(
#'     A=t(matrices$P), B=t(matrices$Q),
#'     glob_mean=glob_mean,
#'     lambda=0.1, scale_lam=TRUE,
#'     implicit=FALSE, nonneg=FALSE,
#'     nthreads=1
#' )
#' 
#' ### Make predictions about new users
#' factors_single(model, X[10,,drop=TRUE])
#' topN_new(model,
#'          X=X[10,,drop=TRUE],
#'          exclude=X[10,,drop=TRUE])
CMF.from.model.matrices <- function(A, B, glob_mean=0, implicit=FALSE,
                                    precompute=TRUE,
                                    user_bias=NULL, item_bias=NULL,
                                    lambda=10., scale_lam=FALSE, l1_lambda=0., nonneg=FALSE,
                                    NA_as_zero=FALSE, scaling_biasA=NULL, scaling_biasB=NULL,
                                    apply_log_transf=FALSE, alpha=1,
                                    nthreads=parallel::detectCores()) {
    ### Check the input data formats
    if (!is.matrix(A))
        stop("'A' must be a numeric matrix.")
    if (!is.matrix(B))
        stop("'B' must be a numeric matrix.")
    k <- nrow(A)
    if (nrow(B) != k)
        stop("Dimensions of 'A' and 'B' do not match.")
    if (!ncol(A) || !ncol(B) || !k)
        stop("Empty model matrices not supported.")
    
    if (typeof(glob_mean) != "double")
        glob_mean <- as.numeric(glob_mean)
    if (NROW(glob_mean) != 1L)
        stop("'glob_mean' must be a single scalar.")
    if (is.na(glob_mean))
        stop("'glob_mean' is NA.")
    
    implicit <- check.bool(implicit, "implicit")
    nthreads <- check.pos.int(nthreads, "nthreads", TRUE)
    NA_as_zero <- check.bool(NA_as_zero, "NA_as_zero")
    apply_log_transf <- check.bool(apply_log_transf, "apply_log_transf")
    scale_lam <- check.bool(scale_lam, "scale_lam")
    nonneg <- check.bool(nonneg, "nonneg")
    precompute <- check.bool(precompute, "precompute")
    alpha <- check.pos.real(alpha, "alpha")
    lambda <- check.lambda(lambda, TRUE)
    l1_lambda <-check.lambda(l1_lambda, TRUE)
    
    
    if (!is.null(user_bias)) {
        if (implicit)
            stop("Biases not supported for implicit-feedback models.")
        if (!is.vector(user_bias))
            stop("'user_bias' must be a vector.")
        if (length(user_bias) != ncol(A))
            stop("'user_bias' dimension does not match with 'A'.")
        user_bias <- as.numeric(user_bias)
    }
    if (!is.null(item_bias)) {
        if (implicit)
            stop("Biases not supported for implicit-feedback models.")
        if (!is.vector(item_bias))
            stop("'item_bias' must be a vector.")
        if (length(item_bias) != ncol(B))
            stop("'item_bias' dimension does not match with 'B'.")
        item_bias <- as.numeric(item_bias)
    }
    
    if (!is.null(scaling_biasA)) {
        if (implicit)
            stop("'scaling_biasA' not compatible with implicit-feedback model.")
        if (is.null(user_bias))
            stop("Cannot pass 'scaling_biasA' when not using user biases.")
        if (!scale_lam)
            stop("Cannot pass 'scaling_biasA' with 'scale_lam=FALSE'.")
        scaling_biasA <- check.pos.real(alpha, "scaling_biasA")
    }
    if (!is.null(scaling_biasB)) {
        if (implicit)
            stop("'scaling_biasB' not compatible with implicit-feedback model.")
        if (is.null(item_bias))
            stop("Cannot pass 'scaling_biasB' when not using user biases.")
        if (!scale_lam)
            stop("Cannot pass 'scaling_biasB' with 'scale_lam=FALSE'.")
        scaling_biasB <- check.pos.real(alpha, "scaling_biasB")
    }
    
    if (
        (!is.null(user_bias) && !is.null(item_bias)) &&
        (is.null(scaling_biasA) != is.null(scaling_biasB))
    ) {
        stop("Must pass both 'scaling_biasA' and 'scaling_biasB'.")
    }
    
    if (typeof(A) != "double")
        mode(A) <- "double"
    if (typeof(B) != "double")
        mode(B) <- "double"
    
    this <- list(
        info = get.empty.info(),
        matrices = get.empty.matrices(),
        precomputed = get.empty.precomputed()
    )
    
    this$matrices$A <- A
    this$matrices$B <- B
    this$matrices$glob_mean <- glob_mean
    if (!is.null(user_bias)) {
        this$matrices$user_bias <- user_bias
        if (!is.null(scaling_biasA))
            this$matrices$scaling_biasA <- scaling_biasA
    }
    if (!is.null(item_bias)) {
        this$matrices$item_bias <- item_bias
        if (!is.null(scaling_biasB))
            this$matrices$scaling_biasB <- scaling_biasB
    }
    
    this$info$k <- k
    this$info$lambda <- lambda
    this$info$l1_lambda <- l1_lambda
    this$info$alpha <- alpha
    this$info$implicit <- implicit
    this$info$apply_log_transf <- apply_log_transf
    this$info$NA_as_zero <- NA_as_zero
    this$info$nonneg <- nonneg
    this$info$center <- glob_mean != 0
    this$info$seed   <- 0L
    this$info$nthreads <- nthreads
    
    if (!implicit)
        class(this) <- c("CMF", "cmfrec")
    else
        class(this) <- c("CMF_implicit", "cmfrec")
    
    if (precompute)
        this <- precompute.for.predictions(this)
    return(this)
}

.onAttach <- function(libname, pkgname) {
    if (!requireNamespace("RhpcBLASctl", quietly=TRUE))
        packageStartupMessage("cmfrec: package 'RhpcBLASctl' not installed, model fitting might be slower.")
}
