#' @export
predict.cmfrec <- function(object, user, item=NULL) {
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
		user <- factor(user, object$info$user_mapping)
		item <- factor(item, object$info$item_mapping)
	}
	user <- as.integer(user) - 1L
	item <- as.integer(item) - 1L
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
