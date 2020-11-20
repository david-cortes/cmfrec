check.lambda <- function(lambda, allow_multiple=TRUE) {
    accepted_nrow <- c(1L)
    if (allow_multiple)
        accepted_nrow <- c(accepted_nrow, 6L)
    if (!(NROW(lambda) %in% accepted_nrow))
        stop("Invalid 'lambda'.")
    lambda <- as.numeric(lambda)
    if (anyNA(lambda))
        stop("'lambda' cannot have missing values.")
    if (any(lambda < 0.))
        stop("'lambda' must be non-negative")
    return(lambda)
}

check.pos.int <- function(k, var="k", pos=FALSE) {
    if (NROW(k) != 1L)
        stop(sprintf("'%s' must be a positive integer", var))
    k <- as.integer(k)
    if (is.na(k))
        stop(sprintf("Invalid '%s'", k))
    if (pos) {
        if (k <= 0L)
            stop(sprintf("'%s' must be a positive integer", var))
    } else if (k < 0L) {
        stop(sprintf("'%s' must be a non-negative integer", var))
    }
    return(k)
}

check.bool <- function(x, var="x") {
    if (NROW(x) != 1L)
        stop(sprintf("'%s' must be a single boolean/logical.", var))
    x <- as.logical(x)
    if (is.na(x))
        stop(sprintf("'%s' cannot be missing.", var))
    return(x)
}

check.pos.real <- function(x, var="x") {
    if (NROW(x) != 1L)
        stop(sprintf("'%s' must be a single boolean/logical.", var))
    x <- as.numeric(x)
    if (is.na(x))
        stop(sprintf("'%s' cannot be missing.", var))
    if (x < 0.)
        stop(sprintf("'%s' must be non-negative.", var))
    return(x)
}

check.str.option <- function(x, var="x", allowed=c()) {
    if (NROW(x) != 1L)
        stop(sprintf("'%s' must be a single string/character.", var))
    x <- as.character(x)
    if (is.na(x))
        stop(sprintf("'%s' cannot be missing.", var))
    if (!(x %in% allowed))
        stop(sprintf("'%s' must be one of: %s", var, paste(allowed, collapse=", ")))
    return(x)
}

check.is.df <- function(df) {
    return(
        as.logical(NROW(intersect(class(df), c("data.frame", "tibble", "data.table"))))
    )
}

check.is.df.or.mat <- function(df) {
    return(("matrix" %in% class(df)) || check.is.df(df))
}

cast.data.frame <- function(df, nm="X") {
    if ("tibble" %in% class(df) || ("data.table" %in% class(df)))
        df <- as.data.frame(df)
    return(df)
}

cast.df.to.matrix <- function(df) {
    if (check.is.df(df)) {
        ## Make sure that all the columns are numeric
        coltypes <- sapply(df, class)
        if (NROW(setdiff(coltypes, c("numeric", "integer"))))
            stop("Only numeric types are supported as side info.")
        if ("integer" %in% coltypes) {
            prev_names <- row.names(df)
            df <- as.data.frame(lapply(df, as.numeric))
            row.names(df) <- prev_names
        }
        df <- as.matrix(df, rownames.force=TRUE)
    }
    return(df)
}

reindex.data <- function(X, U=NULL, I=NULL, U_bin=NULL, I_bin=NULL) {
    out <- list(
        user_mapping = character(),
        item_mapping = character(),
        X = NULL,
        U = NULL,
        I = NULL,
        U_bin = NULL,
        I_bin = NULL
    )
    
    ### Easy case: only 'X' is present
    if (is.null(U) && is.null(I) && is.null(U_bin) && is.null(I_bin)) {
        X[[1L]] <- factor(X[[1L]])
        X[[2L]] <- factor(X[[2L]])
        out$X  <- X
        out$user_mapping <- levels(X[[1L]])
        out$item_mapping <- levels(X[[2L]])
        return(out)
    }
    
    ### Entries in U
    users_X <- unique(X[[1L]])
    users_U <- NULL
    if (!is.null(U))
        users_U <- c(users_U, row.names(U))
    if (!is.null(U_bin))
        users_U <- c(users_U, row.names(U_bin))
    users_U <- unique(users_U)
        
    
    in_X_not_in_U <- setdiff(users_X, users_U)
    in_X_and_U    <- intersect(users_X, users_U)
    all_U         <- unique(union(users_X, users_U))
    if ((!is.null(U) || !is.null(U_bin)) && !NROW(in_X_and_U))
        stop("'X' and 'U'/'U_bin' have no IDs in common.")
    
    
    ### Entries in I
    items_X <- unique(X[[2L]])
    items_I <- NULL
    if (!is.null(I))
        items_I <- c(items_I, row.names(I))
    if (!is.null(I_bin))
        items_I <- c(items_I, row.names(I_bin))
    items_I <- unique(items_I)
    
    in_X_not_in_I <- setdiff(items_X, items_I)
    in_X_and_I    <- intersect(items_X, items_I)
    all_I         <- unique(union(items_X, items_I))
    if ((!is.null(I) || !is.null(I_bin)) && !NROW(in_X_and_I))
        stop("'X' and 'I'/'I_bin' have no IDs in common.")
    
    
    ### Now map and fill
    X[[1L]] <- factor(X[[1L]], levels=all_U)
    X[[2L]] <- factor(X[[2L]], levels=all_I)
    out$X  <- X
    out$user_mapping <- levels(X[[1L]])
    out$item_mapping <- levels(X[[2L]])
    
    
    if (!is.null(U)) {
        ix_U <- as.integer(factor(row.names(U), levels=levels(X[[1L]])))
        if (NROW(U) == NROW(levels(X[[1L]]))) {
            U <- U[order(ix_U), , drop=FALSE]
        } else {
            U_all <- matrix(NA_real_, nrow=NROW(U) + NROW(in_X_not_in_U), ncol=NCOL(U))
            U_all[ix_U, ] <- U
            U <- U_all
        }
        out$U <- U
    }
    if (!is.null(U_bin)) {
        ix_U <- as.integer(factor(row.names(U_bin), levels=levels(X[[1L]])))
        if (NROW(U_bin) == NROW(levels(X[[1L]]))) {
            U_bin <- U_bin[order(ix_U), , drop=FALSE]
        } else {
            U_all <- matrix(NA_real_, nrow=NROW(U_bin) + NROW(in_X_not_in_U), ncol=NCOL(U_bin))
            U_all[ix_U, ] <- U_bin
            U_bin <- U_all
        }
        out$U_bin <- U_bin
    }
    if (!is.null(I)) {
        ix_I <- as.integer(factor(row.names(I), levels=levels(X[[2L]])))
        if (NROW(I) == NROW(levels(X[[2L]]))) {
            I <- I[order(ix_I), , drop=FALSE]
        } else {
            I_all <- matrix(NA_real_, nrow=NROW(I) + NROW(in_X_not_in_I), ncol=NCOL(I))
            I_all[ix_I, ] <- I
            I <- I_all
        }
        out$I <- I
    }
    if (!is.null(I_bin)) {
        ix_I <- as.integer(factor(row.names(I_bin), levels=levels(X[[2L]])))
        if (NROW(I_bin) == NROW(levels(X[[2L]]))) {
            I_bin <- I_bin[order(ix_I), , drop=FALSE]
        } else {
            I_all <- matrix(NA_real_, nrow=NROW(I_bin) + NROW(in_X_not_in_I), ncol=NCOL(I_bin))
            I_all[ix_I, ] <- I_bin
            I_bin <- I_all
        }
        out$I_bin <- I_bin
    }
    
    return(out)
}

process.X <- function(X, weight=NULL) {
    out <- list(
        Xarr = numeric(),
        Xrow = integer(),
        Xcol = integer(),
        Xval = numeric(),
        Warr = numeric(),
        Wsp  = numeric(),
        m    = 0L,
        n    = 0L
    )
    
    if (!min(c(NROW(X), NCOL(X))))
        stop("'X' cannot be empty.")
    
    if ("data.frame" %in% class(X)) {
        if (!is.null(weight))
            stop("'weight' should be passed as 4th column of 'X' when 'X' is a 'data.frame'.")
        out$Xrow <- as.integer(X[[1L]]) - 1L
        out$Xcol <- as.integer(X[[2L]]) - 1L
        out$Xval <- .Call("deep_copy", as.numeric(X[[3L]]))
        if (NCOL(X) > 3L) {
            out$Wsp <- as.numeric(X[[4L]])
        }
        out$m <- max(out$Xrow) + 1L
        out$n <- max(out$Xcol) + 1L
    } else if ("matrix.coo" %in% class(X)) {
        out$Xrow <- X@ia - 1L
        out$Xcol <- X@ja - 1L
        out$Xval <- .Call("deep_copy", X@ra)
        out$m    <- X@dimension[1L]
        out$n    <- X@dimension[2L]
    } else if ("dgTMatrix" %in% class(X)) {
        out$Xrow <- X@i
        out$Xcol <- X@j
        out$Xval <- .Call("deep_copy", X@x)
        out$m    <- X@Dim[1L]
        out$n    <- X@Dim[2L]
    } else if ("matrix" %in% class(X)) {
        out$Xarr <- as.numeric(t(X))
        out$m    <- NROW(X)
        out$n    <- NCOL(X)
    } else {
        stop("Invalid 'X'.")
    }
    
    if (NROW(out$Xval)) {
        if (anyNA(out$Xval))
            stop("'X' cannot have NAN values if passed as sparse.")
    }
    
    if (!is.null(weight)) {
        if ("matrix" %in% class(X)) {
            if ((NROW(weight) != NROW(X)) || (NCOL(weight) != NCOL(X)))
                stop("'weight' must have the same shape as 'X'.")
            out$Warr <- as.numeric(t(weight))
        } else {
            out$Wsp  <- as.numeric(weight)
            if (NROW(out$Wsp) != NROW(out$Xval))
                stop("'weight' must have the same shape as 'X'.")
        }
    }
    
    if (NROW(out$Wsp)) {
        if (anyNA(out$Wsp))
            stop("'weight' cannot have NAN values.")
    }
    if (NROW(out$Warr)) {
        if (anyNA(out$Warr))
            stop("'weight' cannot have NAN values.")
    }
    
    return(out)
}

process.side.info <- function(U, name="U", allow_missing=TRUE) {
    out <- list(
        Uarr = numeric(),
        Urow = integer(),
        Ucol = integer(),
        Uval = numeric(),
        m    = 0L,
        p    = 0L
    )
    if (is.null(U))
        return(out)
    
    if (("matrix" %in% class(U)) || ("data.frame" %in% class(U))) {
        out$Uarr <- as.numeric(t(U))
        out$m    <- NROW(U)
        out$p    <- NCOL(U)
    } else if ("matrix.coo" %in% class(U)) {
        out$Urow <- U@ia - 1L
        out$Ucol <- U@ja - 1L
        out$Uval <- .Call("deep_copy", U@ra)
        out$m    <- U@dimension[1L]
        out$p    <- U@dimension[2L]
    } else if ("dgTMatrix" %in% class(U)) {
        out$Urow <- U@i
        out$Ucol <- U@j
        out$Uval <- .Call("deep_copy", U@x)
        out$m    <- U@Dim[1L]
        out$p    <- U@Dim[2L]
    } else {
        stop(sprintf("Invalid %s.", name))
    }
    
    if (!allow_missing) {
        if (NROW(out$Uarr)) {
            if (anyNA(out$Uarr))
                stop(sprintf("'%s' cannot have missing values.", name))
        }
    }
    if (NROW(out$Uval)) {
        if (anyNA(out$Uval))
            stop(sprintf("'%s' cannot have NAN values if passed as sparse.", name))
    }
    
    return(out)
}

get.empty.precomputed <- function() {
    return(list(
        TransBtBinvBt = matrix(numeric(), nrow=0L, ncol=0L),
        BtB = matrix(numeric(), nrow=0L, ncol=0L),
        TransCtCinvCt = matrix(numeric(), nrow=0L, ncol=0L),
        CtC = matrix(numeric(), nrow=0L, ncol=0L),
        BeTBe = matrix(numeric(), nrow=0L, ncol=0L),
        BeTBeChol = matrix(numeric(), nrow=0L, ncol=0L),
        BiTBi = matrix(numeric(), nrow=0L, ncol=0L),
        B_plus_bias = matrix(numeric(), nrow=0L, ncol=0L),
        Bm_plus_bias = matrix(numeric(), nrow=0L, ncol=0L)
    ))
}

get.empty.matrices <- function() {
    return(list(
        user_bias = numeric(),
        item_bias = numeric(),
        A = matrix(numeric(), nrow=0L, ncol=0L),
        B = matrix(numeric(), nrow=0L, ncol=0L),
        C = matrix(numeric(), nrow=0L, ncol=0L),
        D = matrix(numeric(), nrow=0L, ncol=0L),
        Cb = matrix(numeric(), nrow=0L, ncol=0L),
        Db = matrix(numeric(), nrow=0L, ncol=0L),
        C_bias = numeric(),
        D_bias = numeric(),
        Ai = matrix(numeric(), nrow=0L, ncol=0L),
        Bi = matrix(numeric(), nrow=0L, ncol=0L),
        Am = matrix(numeric(), nrow=0L, ncol=0L),
        Bm = matrix(numeric(), nrow=0L, ncol=0L),
        glob_mean = 0.,
        U_colmeans = numeric(),
        I_colmeans = numeric()
    ))
}

get.empty.info <- function() {
    return(list(
        w_main_multiplier = 1.,
        w_main = 1.,
        w_user = 1.,
        w_item = 1.,
        w_implicit = 1.,
        n_orig = 0L,
        k = 0L,
        k_user = 0L,
        k_item = 0L,
        k_main = 0L,
        k_sec = 0L,
        lambda = 1.,
        alpha = 1.,
        nfev = 0L,
        nupd = 0L,
        user_mapping = character(),
        item_mapping = character(),
        U_cols = character(),
        I_cols = character(),
        U_bin_cols = character(),
        I_bin_cols = character(),
        implicit = FALSE,
        apply_log_transf = FALSE,
        NA_as_zero = FALSE,
        NA_as_zero_user = FALSE,
        NA_as_zero_item = FALSE,
        nonneg = FALSE,
        add_implicit_features = FALSE,
        include_all_X = TRUE,
        nthreads = 1L
    ))
}

check.ret.code <- function(ret_code) {
    if (ret_code == 1L)
        stop("Could not allocate sufficient memory.")
    if (ret_code == 2L)
        stop("Invalid parameter combination.")
}

process.new.X.single <- function(X, X_col, X_val, weight, info, n_max) {
    out <- list(
        X = numeric(),
        X_col = integer(),
        X_val = numeric(),
        weight = numeric(),
        n = 0L
    )
    
    n_use <- ifelse(info$include_all_X, n_max, info$n_orig)
    
    allowed_X <- c("numeric", "integer", "matrix", "dsparseVector")
    if (!is.null(X)) {
        if (!NROW(intersect(class(X), allowed_X)))
            stop("Invalid 'X' - allowed types: ", paste(allowed_X, collapse=", "))
        if ("matrix" %in% class(X)) {
            if (NROW(X) > 1L)
                stop("'X' has more than one row.")
            X <- as.numeric(X)
        }
        if (NROW(X) > n_use)
            stop("'X' has more columns than the model was fit to.")
        if ("integer" %in% class(X))
            X <- as.numeric(X)
        if ("numeric" %in% class(X)) {
            out$X <- .Call("deep_copy", X)
            out$n <- NROW(X)
        } else {
            out$X_col <- X@i - 1L
            out$X_val <- .Call("deep_copy", X@x)
            out$n     <- n_use
        }
    }
    
    if (!is.null(X_col)) {
        if (NROW(info$item_mapping))
            X_col <- as.integer(factor(X_col, info$item_mapping))
        if (NROW(intersect(class(X_col), c("numeric", "character", "matrix"))))
            X_col <- as.integer(X_col)
        X_col <- X_col - 1L
        if (anyNA(X_col))
            stop("'X_col' cannot have missing values or new columns.")
        if (any(X_col > n_use))
            stop("'X_col' cannot contain new columns.")
        if (any(X_col < 0L))
            stop("'X_col' cannot contain negative indices.")
        if (anyNA(X_val))
            stop("'X_val' cannot have NAN values.")
        if (("integer" %in% class(X_val)) || ("matrix" %in% class(X_val)))
            X_val <- as.numeric(X_val)
        if (!("numeric" %in% class(X_val)))
            stop("'X_val' must be a numeric vector.")
        
        out$X_col <- X_col
        out$X_val <- .Call("deep_copy", X_val)
        out$n     <- n_use
    }
    
    if (!is.null(weight)) {
        if (("integer" %in% class(weight)) || ("matrix" %in% class(weight)))
            weight <- as.numeric(weight)
        if (!("numeric" %in% class(weight)))
            stop("'weight' must be a numeric vector.")
        
        if (!is.null(X)) {
            if (NROW(X) != NROW(weight))
                stop("'weight' must have the same number of entries as 'X'.")
        } else if (!is.null(X_col)) {
            if (NROW(X_col) != NROW(weight))
                stop("'weight' must have the same number of non-missing entries as 'X'.")
        }
        out$weight <- weight
    }
    
    return(out)
}

process.new.U.single <- function(U, U_col, U_val, name, mapping, p, colnames,
                                 allow_null=TRUE, allow_na=TRUE, exact_shapes=FALSE) {
    out <- list(
        U = numeric(),
        U_col = integer(),
        U_val = numeric(),
        p = 0L
    )
    
    if (!max(c(NROW(U), NCOL(U), NROW(U_col), NROW(U_val)))) {
        if (allow_null) {
            out$p <- p
            return(out)
        } else {
            stop(sprintf("'%s' cannot be empty.", name))
        }
    }
    
    if (!is.null(U)) {
        U <- cast.data.frame(U)
        if ("data.frame" %in% class(U)) {
            if (NROW(colnames))
                U <- U[, colnames, drop = TRUE]
            
            coltypes <- sapply(U, class)
            if ("integer" %in% coltypes)
                U <- as.data.frame(lapply(U, as.numeric))
            invalid <- setdiff(coltypes, c("numeric", "integer"))
            if (NROW(invalid))
                stop(sprintf("Invalid column type(s) in '%s': ", name), paste(invalid, collapse=", "))
            U <- as.numeric(U)
        }
        
        if ("matrix" %in% class(U)) {
            if (NROW(U) > 1L)
                stop(sprintf("'%s' has more than one row.", name))
            U <- as.numeric(U)
        }
    }
    
    allowed_U <- c("numeric", "integer", "dsparseVector")
    if (!is.null(U)) {
        if (!NROW(intersect(class(U), allowed_U)))
            stop(sprintf("Invalid '%s' - allowed types: %s", name, paste(allowed_U, collapse=", ")))
        if (NROW(U) > p)
            stop(sprintf("'%s' has more columns than the model was fit to.", name))
        if ("integer" %in% class(U))
            U <- as.numeric(U)
        if ("numeric" %in% class(U)) {
            if (exact_shapes && NROW(U) != p)
                stop(sprintf("'%s' has different number of columns than model was fit to.", name))
            if (!allow_na && anyNA(U))
                stop(sprintf("'%s' cannot have NAN values.", name))
            out$U <- .Call("deep_copy", U)
            out$p <- NROW(U)
        } else {
            if (U@length > p)
                stop(sprintf("'%s' has more columns than the model was fit to.", name))
            out$U_col <- U@i - 1L
            out$U_val <- .Call("deep_copy", U@x)
            out$p     <- p
        }
    }
    
    if (!is.null(U_col)) {
        if (NROW(mapping))
            U_col <- as.integer(factor(U_col, mapping))
        if (NROW(intersect(class(U_col), c("numeric", "character", "matrix"))))
            U_col <- as.integer(U_col)
        U_col <- U_col - 1L
        if (anyNA(U_col))
            stop(sprintf("'%s_col' cannot have missing values or new columns.", name))
        if (any(U_col >= p))
            stop(sprintf("'%s_col' cannot contain new columns.", name))
        if (any(U_col < 0L))
            stop(sprintf("%s_col' cannot contain negative indices.", name))
        if (anyNA(U_val))
            stop(sprintf("'%s_val' cannot have NAN values.", name))
        if (("integer" %in% class(U_val)) || ("matrix" %in% class(U_val)))
            U_val <- as.numeric(U_val)
        if (!("numeric" %in% class(U_val)))
            stop(sprintf("'%s_val' must be a numeric vector.", name))
        
        out$U_col <- U_col
        out$U_val <- .Call("deep_copy", U_val)
        out$p     <- p
    }
    
    return(out)
}

process.new.X <- function(obj, X, weight=NULL,
                          allow_sparse=TRUE, allow_null=TRUE,
                          allow_reindex=FALSE) {
    out <- list(
        Xarr = numeric(),
        Xrow = integer(),
        Xcol = integer(),
        Xval = integer(),
        Xcsr_p = raw(),
        Xcsr_i = integer(),
        Xcsr = numeric(),
        Warr = numeric(),
        Wsp = numeric(),
        m = 0L,
        n = 0L
    )
    
    if (is.null(X) && !allow_null)
        stop("'X' cannot be NULL.")
    if (!allow_null && !min(c(NROW(X), NCOL(X))))
        stop("'X' cannot be empty.")
    if (is.null(X) && !is.null(weight))
        stop("'weight' not meaningfull without 'X'.")
    
    if ("integer" %in% class(X))
        X <- as.numeric(X)
    if ("numeric" %in% class(X))
        X <- matrix(X, nrow = 1L)
    X <- cast.data.frame(X)
    
    allowed_X <- c("matrix")
    if (allow_sparse)
        allowed_X <- c(allowed_X, c("data.frame", "dsparseVector",
                                    "dgTMatrix", "matrix.coo",
                                    "dgRMatrix", "matrix.csr"))
    if (allow_null)
        allowed_X <- c(allowed_X, "NULL")
    if (!NROW(intersect(class(X), allowed_X)))
        stop("Invalid 'X' - supported types: ", paste(allowed_X, collapse=", "))
    
    if ("data.frame" %in% class(X)) {
        if (!is.null(weight))
            stop("'weight' should be the 4th column of 'X' when 'X' is a 'data.frame'.")
        if (ncol(X) < 3L)
            stop("'X' must have at least 3 columns (user, item, value).")
        if (NROW(obj$info$user_mapping) && allow_reindex) {
            X[[1L]] <- as.integer(factor(X[[1L]], obj$info$user_mapping))
            X[[2L]] <- as.integer(factor(X[[2L]], obj$info$item_mapping))
        } else {
            X[[1L]] <- as.integer(X[[1L]])
            X[[2L]] <- as.integer(X[[2L]])
        }
        X[[1L]] <- X[[1L]] - 1L
        X[[2L]] <- X[[2L]] - 1L
        if (anyNA(X[[1L]]) || anyNA(X[[2L]]) || any(X[[1L]] < 0L) || any(X[[2L]] < 0L))
            stop("'X' contains invalid indices.")
        out$m <- max(X[[1L]]) + 1L
        out$n <- max(X[[2L]]) + 1L
        
        out$Xrow <- X[[1L]]
        out$Xcol <- X[[2L]]
        out$Xval <- .Call("deep_copy", X[[3L]])
        
        if (ncol(X) >= 4L) {
            out$Wsp <- X[[4L]]
        }
        
    } else if ("dgTMatrix" %in% class(X)) {
        out$Xrow <- X@i
        out$Xcol <- X@j
        out$Xval <- .Call("deep_copy", X@x)
        out$m    <- X@Dim[1L]
        out$n    <- X@Dim[2L]
    } else if ("matrix.coo" %in% class(X)) {
        out$Xrow <- X@ia - 1L
        out$Xcol <- X@ja - 1L
        out$Xval <- .Call("deep_copy", X@ra)
        out$m    <- X@dimension[1L]
        out$n    <- X@dimension[2L]
    } else if ("dgRMatrix" %in% class(X)) {
        out$Xcsr_p <- .Call("as_size_t", X@p)
        out$Xcsr_i <- X@j
        out$Xcsr   <- .Call("deep_copy", X@x)
        out$m      <- X@Dim[1L]
        out$n      <- X@Dim[2L]
    } else if ("matrix.csr" %in% class(X)) {
        out$Xcsr_p <- .Call("as_size_t", X@ia - 1L)
        out$Xcsr_i <- X@ja - 1L
        out$Xcsr   <- .Call("deep_copy", X@ra)
        out$m      <- X@dimension[1L]
        out$n      <- X@dimension[2L]
    } else if ("dsparseVector" %in% class(X)) {
        out$Xcsr_p <- .Call("as_size_t", c(0L, NROW(X@i)))
        out$Xcsr_i <- X@i - 1L
        out$Xcsr   <- .Call("deep_copy", X@x)
        out$m      <- 1L
        out$n      <- X@length
    } else if ("matrix" %in% class(X)) {
        out$Xarr <- as.numeric(t(X))
        out$m    <- nrow(X)
        out$n    <- ncol(X)
    }
    
    n_max <- max(c(NCOL(obj$matrices$B), NCOL(obj$matrices$Bm)))
    if (!obj$info$include_all_X)
        n_max <- obj$info$n_orig
    if (out$n > n_max)
        stop("'X' contains columns that were not passed to 'fit'.")
    
    if (NROW(out$Xval)) {
        if (anyNA(out$Xval))
            stop("Values of sparse 'X' cannot be NAN.")
    }
    if (NROW(out$Xcsr)) {
        if (anyNA(out$Xcsr))
            stop("Values of sparse 'X' cannot be NAN.")
    }
    
    if (!is.null(weight)) {
        if ("integer" %in% class(weight))
            weight <- as.numeric(weight)
        weight <- cast.data.frame(weight, "weight")
        if ("data.frame" %in% class(weight))
            weight <- as.matrix(weight)
        
        allowed_weight <- ifelse(NROW(out$Xarr), "matrix", "numeric")
        if (!NROW(intersect(class(weight), allowed_weight)))
            stop(sprintf("'weight' must be of class '%s'.", allowed_weight))
        
        if ("matrix" %in% class(weight)) {
            if ((NROW(weight) != NROW(X)) || (NCOL(weight) != NCOL(X)))
                stop("'X' and 'weight' must have the same dimensions.")
            out$Wfull <- as.numeric(t(weight))
        } else {
            if (NROW(weight) != max(c(NROW(out$Xval), NROW(out$Xcsr))))
                stop("'weight' must have the same number of entries as 'X'.")
            out$Wsp <- weight
        }
    }
    
    if (NROW(out$Wfull)) {
        if (anyNA(out$Wfull))
            stop("weights cannot be NAN.")
    }
    if (NROW(out$Wsp)) {
        if (anyNA(out$Wsp))
            stop("weights cannot be NAN.")
    }
    
    if (!NROW(out$Xarr)) {
        if (obj$info$include_all_X) {
            out$n <- obj$info$n_orig
        } else {
            out$n <- max(c(NCOL(obj$matrices$B), NCOL(obj$matrices$Bm)))
        }
    }
    
    return(out)
}

process.new.U <- function(U, U_cols, p, name="U",
                          allow_sparse=TRUE, allow_null=TRUE,
                          allow_na=TRUE, exact_shapes=FALSE) {
    out <- list(
        Uarr = numeric(),
        Urow = integer(),
        Ucol = integer(),
        Uval = integer(),
        Ucsr_p = raw(),
        Ucsr_i = integer(),
        Ucsr = numeric(),
        m = 0L,
        p = 0L
    )
    
    if (is.null(U) || !max(c(NROW(U), NCOL(U)))) {
        if (allow_null) {
            return(out)
        } else {
            stop(sprintf("'%s' cannot be empty.", name))
        }
    }
    
    if ("integer" %in% class(U))
        U <- as.numeric(U)
    if ("numeric" %in% class(U))
        U <- matrix(U, nrow = 1L)
    
    U <- cast.data.frame(U)
    if ("data.frame" %in% class(U)) {
        if (NROW(U_cols))
            U <- U[, U_cols, drop=FALSE]
        U <- cast.df.to.matrix(U)
    }
    
    allowed_U <- c("matrix")
    if (allow_sparse)
        allowed_U <- c(allowed_U, c("dgTMatrix", "matrix.coo",
                                    "dgRMatrix", "matrix.csr",
                                    "dsparseVector"))
    if (!NROW(intersect(class(U), allowed_U)))
        stop(sprintf("Invalid '%s' - allowed types: ", name), paste(allowed_U, collapse=", "))
    
    msg_new_cols <- sprintf("'%s' cannot contain new columns.", name)
    
    if ("matrix" %in% class(U)) {
        if (ncol(U) > p)
            stop(msg_new_cols)
        if (exact_shapes && NCOL(U) != p)
            stop(sprintf("'%s' cannot have a different number of columns than the '%s' passed to fit",
                         name, name))
        out$Uarr <- as.numeric(t(U))
        out$m    <- nrow(U)
        out$p    <- ncol(U)
    } else if ("dgTMatrix" %in% class(U)) {
        if (U@Dim[2L] > p)
            stop(msg_new_cols)
        out$Urow <- U@i
        out$Ucol <- U@j
        out$Uval <- .Call("deep_copy", U@x)
        out$m    <- U@Dim[1L]
    } else if ("matrix.coo" %in% class(U)) {
        if (U@dimension[2L] > p)
            stop(msg_new_cols)
        out$Urow <- U@ia - 1L
        out$Ucol <- U@ja - 1L
        out$Uval <- .Call("deep_copy", U@ra)
        out$m    <- U@dimension[1L]
    } else if ("dgRMatrix" %in% class(U)) {
        if (U@Dim[2L] > p)
            stop(msg_new_cols)
        out$Ucsr_p <- .Call("as_size_t", U@p)
        out$Ucsr_i <- U@j
        out$Ucsr   <- .Call("deep_copy", U@x)
        out$m      <- U@Dim[1L]
    } else if ("matrix.csr" %in% class(U)) {
        if (U@dimension[2L] > p)
            stop(msg_new_cols)
        out$Ucsr_p <- .Call("as_size_t", U@ia - 1L)
        out$Ucsr_i <- U@ja - 1L
        out$Ucsr   <- .Call("deep_copy", U@ra)
        out$m      <- U@dimension[1L]
    } else if ("dsparseVector" %in% class(U)) {
        if (U@length > p)
            stop(msg_new_cols)
        out$Ucsr_p <- .Call("as_size_t", c(0L, NROW(U@i)))
        out$Ucsr_i <- U@i - 1L
        out$Ucsr   <- .Call("deep_copy", U@x)
        out$m      <- 1L
    } else {
        stop("Unexpected error.")
    }
    
    if (!NROW(out$Uarr))
        out$p <- p
    if (!allow_na) {
        if (NROW(out$Uarr) && anyNA(out$Uarr))
            stop(sprintf("'%s' cannot have NAN values.", name))
    }
    if (NROW(out$Uval) && anyNA(out$Uval))
        stop("Sparse inputs cannot have NAN values.")
    if (NROW(out$Ucsr) && anyNA(out$Ucsr))
        stop("Sparse inputs cannot have NAN values.")
    
    return(out)
}

swap.lambda <- function(lambda) {
    new_lambda <- lambda
    if (NROW(lambda) == 6L) {
        new_lambda[1L] = lambda[2L]
        new_lambda[2L] = lambda[1L]
        new_lambda[3L] = lambda[4L]
        new_lambda[4L] = lambda[3L]
        new_lambda[5L] = lambda[6L]
        new_lambda[6L] = lambda[5L]
    }
    return(new_lambda)
}
