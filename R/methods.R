#' @title Get information about factorization model
#' @description Print basic properties of a `cmfrec` object (a base class
#' encompassing all the models in this package).
#' @param x An object of class `cmfrec` as returned by functions \link{CMF},
#' \link{CMF_implicit}, \link{MostPopular}, \link{ContentBased},
#' \link{OMF_explicit}, \link{OMF_implicit}.
#' @param ... Extra arguments (not used).
#' @return No return value (information is printed).
#' @export
print.cmfrec <- function(x, ...) {
    cat(sprintf("'%s' Model Object\n\n", class(x)[1]))
    if (!("MostPopular" %in% class(x))) {
        cat(sprintf("Dimensions: %d x %d\n", NCOL(x$matrices$A), NCOL(x$matrices$B)))
        cat(sprintf("Latent factors (shared): %d\n\n", x$info$k))
    } else {
        cat(sprintf("Number of items: %d\n", NROW(x$matrices$item_bias)))
    }
    
    if (x$info$k_main || x$info$k_user || x$info$k_item || x$info$k_sec) {
        cat("Non-shared factors:")
        if (x$info$k_user)
            cat(sprintf(" k_user: %d", x$info$k_user))
        if (x$info$k_item)
            cat(sprintf(" k_item: %d", x$info$k_item))
        if (x$info$k_main)
            cat(sprintf(" k_main: %d", x$info$k_main))
        if (x$info$k_sec)
            cat(sprintf(" k_sec: %d", x$info$k_sec))
        cat("\n")
    }
    if (NROW(x$info$lambda) == 1) {
        cat(sprintf("Regularization: %.2g\n", x$info$lambda))
    } else {
        cat("Using different regularization parameters\n")
    }
    
    if (NROW(x$matrices$user_bias) && NROW(x$matrices$item_bias)) {
        cat("Using user and item biases\n")
    } else if (NROW(x$matrices$user_bias)) {
        cat("Using user biases\n")
    } else if (NROW(x$matrices$item_bias)) {
        cat("Using item biases\n")
    }
    
    if (!("MostPopular" %in% class(x)) && !("ContentBased" %in% class(x))) {
        if (NROW(x$matrices$C) && NROW(x$matrices$D)) {
            cat("Fit to both user and item side info\n")
        } else if (NROW(x$matrices$C)) {
            cat("Fit to user side info\n")
        } else if (NROW(x$matrices$D)) {
            cat("Fit to item side info\n")
        } else {
            cat("Fit without side info\n")
        }
    }
    
    if (x$info$nfev) {
        cat(sprintf("Number of L-BFGS updates: %d / Function evaluations: %d\n",
                    x$info$nupd, x$info$nfev))
    }
    
    if (NROW(x$precomputed$BtB))
        cat("(Model has precomputed matrices for predictions)\n")
}

#' @title Get information about factorization model
#' @description Print basic properties of a `cmfrec` object (a base class
#' encompassing all the models in this package). Same as the `print.cmfrec` function.
#' @param object An object of class `cmfrec` as returned by functions \link{CMF},
#' \link{CMF_implicit}, \link{MostPopular}, \link{ContentBased},
#' \link{OMF_explicit}, \link{OMF_implicit}.
#' @param ... Extra arguments (not used).
#' @return No return value (information is printed).
#' @seealso \link{print.cmfrec}
#' @export
summary.cmfrec <- function(object, ...) {
    print.cmfrec(object)
}
