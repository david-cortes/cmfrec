process.data.factors.single <- function(model, obj,
										X = NULL, X_col = NULL, X_val = NULL, weight = NULL,
										U = NULL, U_col = NULL, U_val = NULL,
										U_bin = NULL,
										output_bias = FALSE, output_A = FALSE, exact = FALSE) {
	
	if (is.null(X) && is.null(X_col) && is.null(X_val) && is.null(weight) &&
		is.null(U) && is.null(U_col) && is.null(U_val) && is.null(U_bin))
		stop("No new data was passed.")
	output_bias  <-  check.bool(output_bias, "output_bias")
	output_A     <-  check.bool(output_A, "output_A")
	exact        <-  check.bool(exact, "exact")
	if (!is.null(U_bin) && !NCOL(obj$matrices$Cb))
		stop("Model was not fit to binary side information.")
	if ((!is.null(U) || !is.null(U_col) || !is.null(U_val)) && !NCOL(obj$C))
		stop("Model was not fit to 'U' data.")
	if (is.null(X_col) != is.null(X_val))
		stop("'X_col' and 'X_val' must be passed in conjunction.")
	if (is.null(U_col) != is.null(U_val))
		stop("'U_col' and 'U_val' must be passed in conjunction.")
	if (NROW(X_col) != NROW(X_val))
		stop("'X_col' and 'X_val' must have the same number of entries.")
	if (NROW(U_col) != NROW(U_val))
		stop("'U_col' and 'U_val' must have the same number of entries.")
	if (!is.null(weight) && is.null(X) && is.null(X_col) && is.null(X_val))
		stop("Cannot pass 'weight' without 'X' data.")
	
	if (model %in% c("CMF_implicit", "OMF_implicit")) {
		if (!is.null(X) && !("dsparseVector" %in% class(X)))
			stop("Cannot pass dense 'X' for implicit-feedback models.")
	}
	
	if (output_bias && !NROW(obj$matrices$user_bias))
		stop("Model was fit without biases.")
	
	processed_X   <-  process.new.X.single(X, X_col, X_val, weight,
										   obj$info, NCOL(obj$matrices$B))
	processed_U   <-  process.new.U.single(U, U_col, U_val, "U",
										   obj$info$user_mapping, NCOL(obj$matrices$C),
										   obj$info$U_cols,
										   allow_null = model != "ContentBased",
										   allow_na = model %in% c("CMF", "CMF_implicit"),
										   exact_shapes = !(model %in% c("CMF", "CMF_implicit")))
	processed_Ub  <-  process.new.U.single(U_bin, NULL, NULL, "U_bin",
										   obj$info$user_mapping, NCOL(obj$matrices$Cb),
										   obj$info$I_cols)
	
	
	return(list(
		processed_X = processed_X,
		processed_U = processed_U,
		processed_Ub = processed_Ub,
		output_bias = output_bias,
		output_A = output_A,
		exact = exact
	))
}


#' @export
factors_single.CMF <- function(model, X = NULL, X_col = NULL, X_val = NULL, weight = NULL,
							   U = NULL, U_col = NULL, U_val = NULL,
							   U_bin = NULL,
							   output_bias = FALSE) {
	inputs <- process.data.factors.single("CMF", model,
										  X = X, X_col = X_col, X_val = X_val, weight = weight,
										  U = U, U_col = U_col, U_val = U_val,
										  U_bin = U_bin,
										  output_bias = output_bias)
	a_vec <- numeric(model$info$k_user + model$info$k + model$info$k_main)
	a_bias <- numeric()
	if (NROW(model$matrices$user_bias))
		a_bias <- numeric(length = 1L)
	ret_code <- .Call("call_factors_collective_explicit_single",
					  a_vec, a_bias,
					  inputs$processed_U$U, inputs$processed_U$p,
					  inputs$processed_U$U_val, inputs$processed_U$U_col,
					  inputs$processed_U_bin$U, inputs$processed_U_bin$p,
					  model$info$NA_as_zero_user, model$info$NA_as_zero,
					  model$matrices$C, model$matrices$Cb,
					  model$matrices$glob_mean, model$matrices$item_bias,
					  model$matrices$U_colmeans,
					  inputs$processed_X$X_val, inputs$processed_X$X_col,
					  inputs$processed_X$X, inputs$processed_X$n,
					  inputs$processed_X$weight,
					  model$matrices$B,
					  model$matrices$Bi, model$info$add_implicit_features,
					  model$info$k, model$info$k_user, model$info$k_item, model$info$k_main,
					  model$info$lambda,
					  model$info$w_main, model$info$w_user, model$info$w_implicit,
					  NCOL(model$matrices$B), model$info$include_all_X,
					  model$precomputed$TransBtBinvBt,
					  model$precomputed$BtB,
					  model$precomputed$BeTBeChol,
					  model$precomputed$BiTBi,
					  model$precomputed$CtC,
					  model$precomputed$TransCtCinvCt,
					  model$precomputed$B_plus_bias)
	check.ret.code(ret_code)
	if (inputs$output_bias) (
		return(list(factors = a_vec, bias = a_bias))
	) else {
		return(a_vec)
	}
}

#' @export
factors_single.CMF_implicit <- function(model, X = NULL, X_col = NULL, X_val = NULL,
										U = NULL, U_col = NULL, U_val = NULL) {
	inputs <- process.data.factors.single("CMF_implicit", model,
										  X = X, X_col = X_col, X_val = X_val,
										  U = U, U_col = U_col, U_val = U_val)
	a_vec <- numeric(model$info$k_user + model$info$k + model$info$k_main)
	lambda <- ifelse(NROW(model$info$lambda) > 1L, model$info$lambda[3L], model$info$lambda)
	ret_code <- .Call("call_factors_collective_implicit_single",
					  a_vec,
					  inputs$processed_U$U, inputs$processed_U$p,
					  inputs$processed_U$U_val, inputs$processed_U$U_col,
					  model$info$NA_as_zero_user,
					  model$matrices$U_colmeans,
					  model$matrices$B, NCOL(model$matrices$B), model$matrices$C,
					  inputs$processed_X$X_val, inputs$processed_X$X_col,
					  model$info$k, model$info$k_user, model$info$k_item, model$info$k_main,
					  lambda, model$info$alpha, model$info$w_main, model$info$w_user,
					  model$info$w_main_multiplier,
					  model$precomputed$BeTBe,
					  model$precomputed$BtB,
					  model$precomputed$BeTBeChol)
	check.ret.code(ret_code)
	return(a_vec)
}

#' @export
factors_single.ContentBased <- function(model,
										U = NULL, U_col = NULL, U_val = NULL) {
	
	inputs <- process.data.factors.single("ContentBased", model,
										  U = U, U_col = U_col, U_val = U_val)
	a_vec <- numeric(model$info$k)
	
	ret_code <- .Call("call_factors_content_based_single",
					  a_vec, model$info$k,
					  inputs$processed_U$U, inputs$processed_U$p,
					  inputs$processed_U$U_val,  inputs$processed_U$U_col,
					  model$matrices$C, model$matrices$C_bias)
	
	check.ret.code(ret_code)
	return(a_vec)
}

#' @export
factors_single.OMF_explicit <- function(model, X = NULL, X_col = NULL, X_val = NULL, weight = NULL,
										U = NULL, U_col = NULL, U_val = NULL,
										output_bias = FALSE, output_A = FALSE, exact = FALSE) {
	
	if (!exact && !output_A &&
		(!is.null(X) || !is.null(X_col) || !is.null(X_val)) &&
		(!is.null(U) || !is.null(U_col) || !is.null(U_val))) {
		warning("'U' data is ignored in the presence of 'X' data with 'exact=FALSE'.")
		U <- NULL
		U_col <- NULL
		U_val <- NULL
	}
	
	inputs <- process.data.factors.single("OMF_explicit", model,
										  X = X, X_col = X_col, X_val = X_val, weight = weight,
										  U = U, U_col = U_col, U_val = U_val,
										  output_bias = output_bias,
										  output_A = output_A, exact = exact)
	a_vec <- numeric(model$info$k_sec + model$info$k + model$info$k_main)
	a_bias <- numeric()
	if (NROW(model$matrices$user_bias))
		a_bias <- numeric(length = 1L)
	a_orig <- numeric()
	if (inputs$output_A) {
		if (NROW(model$matrices$C)) {
			a_orig <- numeric(model$info$k + model$info$k_main)
		} else {
			warning("Option 'output_A' invalid when the model was not fit to 'U' data.")
			inputs$output_A <- FALSE
		}
	}
	if (inputs$exact) {
		if (!model$info$nfev) {
			warning("Option 'exact' not meaningful for ALS-fitted models.")
			inputs$exact <- FALSE
		}
	}
	lambda <- ifelse(NROW(model$info$lambda) > 1L, model$info$lambda[3L], model$info$lambda)
	
	ret_code <- .Call("call_factors_offsets_explicit_single",
					  a_vec, a_bias, a_orig,
					  inputs$processed_U$U, inputs$processed_U$p,
					  inputs$processed_U$U_val,  inputs$processed_U$U_col,
					  inputs$processed_X$X_val, inputs$processed_X$X_col,
					  inputs$processed_X$X, inputs$processed_X$n,
					  inputs$processed_X$weight,
					  model$matrices$Bm, model$matrices$C,
					  model$matrices$C_bias,
					  model$matrices$glob_mean, model$matrices$item_bias,
					  model$info$k, model$info$k_sec, model$info$k_main,
					  model$info$w_user,
					  lambda,
					  inputs$exact,
					  model$precomputed$TransBtBinvBt,
					  model$precomputed$BtB,
					  model$precomputed$Bm_plus_bias)
	
	check.ret.code(ret_code)
	if (!inputs$output_bias && !inputs$output_A)
		return(a_vec)
	out <- list(factors = a_vec)
	if (inputs$output_bias) {
		out$bias <- a_bias
	}
	if (inputs$output_A) {
		out$A <- a_orig
	}
	return(out)
}

#' @export
factors_single.OMF_implicit <- function(model, X = NULL, X_col = NULL, X_val = NULL,
										U = NULL, U_col = NULL, U_val = NULL,
										output_A = FALSE) {
	
	if (!output_A &&
		(!is.null(X) || !is.null(X_col) || !is.null(X_val)) &&
		(!is.null(U) || !is.null(U_col) || !is.null(U_val))) {
		warning("'U' data is ignored in the presence of 'X' data.")
		U <- NULL
		U_col <- NULL
		U_val <- NULL
	}
	
	inputs <- process.data.factors.single("OMF_implicit", model,
										  X = X, X_col = X_col, X_val = X_val,
										  U = U, U_col = U_col, U_val = U_val,
										  output_A = output_A)
	a_vec <- numeric(model$info$k)
	a_orig <- numeric()
	if (inputs$output_A)
		a_orig <- numeric(model$info$k)
	ret_code <- .Call("call_factors_offsets_implicit_single",
					  a_vec,
					  inputs$processed_U$U, inputs$processed_U$p,
					  inputs$processed_U$U_val, inputs$processed_U$U_col,
					  inputs$processed_X$X_val, inputs$processed_X$X_col,
					  model$matrices$Bm, model$matrices$C,
					  model$matrices$C_bias,
					  model$info$k, NCOL(model$matrices$Bm),
					  model$info$lambda, model$info$alpha,
					  model$precomputed$BtB,
					  a_orig)
	
	check.ret.code(ret_code)
	if (inputs$output_A) {
		return(list(factors = a_vec, A = a_orig))
	} else {
		return(a_vec)
	}
}
