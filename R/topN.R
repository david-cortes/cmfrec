
process.inputs.topN <- function(model, obj, user=NULL, a_vec=NULL, a_bias=NULL,
								n=10L, include=NULL, exclude=NULL, output_score=FALSE) {
	if (!is.null(include) && !is.null(exclude))
		stop("Can only pass one of 'include' or 'exclude'.")
	output_score <- check.bool(output_score, "output_score")
	n <- check.pos.int(n, "n", TRUE)
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
		if (("numeric" %in% class(user)) || ("character" %in% class(user)))
			user <- as.integer(user)
		user <- check.pos.int(user, "user", TRUE)
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
		if (("numeric" %in% class(include)) || ("character" %in% class(include)))
			include <- as.integer(include)
		if (!("integer" %in% class(include)))
			stop("Invalid data type for 'include'.")
		
		if (model != "MostPopular") {
			if (max(include) > NCOL(obj$matrices$B))
				stop("'include' contains element that were not passed to 'fit'.")
		} else {
			if (max(include) > NROW(obj$matrices$item_bias))
				stop("'include' contains element that were not passed to 'fit'.")
		}
		
		include <- include - 1L
		if (any(include < 0L) || any(is.na(include)))
			stop("'include' contains invalid entries.")
		
		if (NROW(include) < n)
			stop("'n' is greater than the number of entries in 'include'.")
	}
	
	if (!is.null(exclude)) {
		if (("numeric" %in% class(exclude)) || ("character" %in% class(exclude)))
			exclude <- as.integer(exclude)
		if (!("integer" %in% class(exclude)))
			stop("Invalid data type for 'exclude'.")
		
		if (model != "MostPopular") {
			if (max(exclude) > NCOL(obj$matrices$B))
				stop("'exclude' contains element that were not passed to 'fit'.")
		} else {
			if (max(exclude) > NROW(obj$matrices$item_bias))
				stop("'exclude' contains element that were not passed to 'fit'.")
		}
		
		exclude <- exclude - 1L
		if (any(exclude < 0L) || any(is.na(exclude)))
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
	
	if (model == "CMF") {
		ret_code <- .Call("call_topN_old_collective_explicit",
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
		ret_code <- .Call("call_topN_old_collective_implicit",
						  a_vec,
						  obj$matrices$B,
						  obj$info$k, obj$info$k_user, obj$info$k_item, obj$info$k_main,
						  include,
						  exclude,
						  outp_ix, outp_score,
						  NCOL(obj$matrices$B), obj$info$nthreads)
	} else if (model == "MostPopular") {
		ret_code <- .Call("call_topN_old_most_popular",
						  as.logical(NROW(obj$matrices$user_bias)),
						  a_bias,
						  obj$matrices$item_bias,
						  obj$matrices$glob_mean,
						  include,
						  exclude,
						  outp_ix, outp_score,
						  NROW(obj$matrices$item_bias))
	} else if (model == "ContentBased") {
		ret_code <- .Call("call_topN_old_content_based",
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
		ret_code <- .Call("call_topN_old_offsets_explicit",
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
		ret_code <- .Call("call_topN_old_offsets_implicit",
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
topN <- function(model, user=NULL, n=10L, include=NULL, exclude=NULL, output_score=FALSE) {
	supported_models <- c("CMF", "CMF_implicit",
						  "MostPopular", "ContentBased",
						  "OMF_implicit", "OMF_explicit")
	if (!NROW(intersect(class(model), supported_models)))
		stop("Invalid model object - supported classes: ", paste(supported_models, collapse=", "))
	if (is.null(user) && !("MostPopular" %in% class(model)))
		stop("'user' cannot be empty for this model.")
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
topN_new.CMF <- function(model, X=NULL, X_col=NULL, X_val=NULL, weight=NULL,
						 U=NULL, U_col=NULL, U_val=NULL, U_bin=NULL,
						 n=10L, include=NULL, exclude=NULL, output_score=FALSE) {
	inputs <- process.inputs.topN(class(model)[1L], model,
								  n = n,
								  include = include, exclude = exclude,
								  output_score = output_score)
	factors <- factors_single.CMF(model = model, X = X, X_col = X_col, X_val = X_val,
								  weight = weight,
								  U = U, U_col = U_col, U_val = U_val, U_bin = U_bin,
								  output_bias = as.logical(NROW(model$matrices$user_bias)))
	if (class(factors) == "list") {
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
topN_new.CMF_implicit <- function(model, X=NULL, X_col=NULL, X_val=NULL,
								  U=NULL, U_col=NULL, U_val=NULL,
								  n=10L, include=NULL, exclude=NULL, output_score=FALSE) {
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
topN_new.ContentBased <- function(model, U=NULL, U_col=NULL, U_val=NULL, I=NULL,
								  n=10L, include=NULL, exclude=NULL, output_score=FALSE) {
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
		processed_U <- process.new.U.single(U, U_col, U_val, "U",
											model$info$user_mapping, NCOL(model$matrices$C),
											model$info$U_cols)
		processed_I <- process.new.U(I, model$info$I_cols, NCOL(model$matrices$D), "I",
									 allow_sparse=TRUE, allow_null=FALSE,
									 allow_na=FALSE, exact_shapes=TRUE)
		
		outp_ix <- integer(length = n)
		outp_score <- numeric(length = ifelse(output_score, n, 0L))
		
		ret_code <- .Call("call_topN_new_content_based",
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
topN_new.OMF_explicit <- function(model, X=NULL, X_col=NULL, X_val=NULL, weight=NULL,
								  U=NULL, U_col=NULL, U_val=NULL, exact=FALSE,
								  n=10L, include=NULL, exclude=NULL, output_score=FALSE) {
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
	if (class(factors) == "list") {
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
topN_new.OMF_implicit <- function(model, X=NULL, X_col=NULL, X_val=NULL,
								  U=NULL, U_col=NULL, U_val=NULL,
								  n=10L, include=NULL, exclude=NULL, output_score=FALSE) {
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
