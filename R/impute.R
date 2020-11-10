
#' @export
imputeX <- function(model, X, weight = NULL, U = NULL, U_bin = NULL) {
	if (!("CMF" %in% class(model)))
		stop("Method is only applicable to 'CMF' model.")
	if (!("matrix" %in% class(X)))
		stop("'X' must be a matrix with NAN values.")
	if (!any(is.na(X)))
		return(X)
	
	inputs <- process.data.factors("CMF", model,
								   X = X, weight = weight,
								   U = U, U_bin = U_bin,
								   matched_shapes=TRUE)
	ret_code <- .Call("call_impute_X_collective_explicit",
					  inputs$processed_X$m, as.logical(NROW(model$matrices$user_bias)),
					  inputs$processed_U$Uarr, inputs$processed_U$m, inputs$processed_U$p,
					  model$info$NA_as_zero_user,
					  inputs$processed_U$Urow, inputs$processed_U$Ucol, inputs$processed_U$Uval,
					  inputs$processed_U$Ucsr_p, inputs$processed_U$Ucsr_i, inputs$processed_U$Ucsr,
					  inputs$processed_U_bin$Uarr, inputs$processed_U_bin$m, inputs$processed_U_bin$p,
					  model$matrices$C, model$matrices$Cb,
					  model$matrices$glob_mean, model$matrices$item_bias,
					  model$matrices$U_colmeans,
					  inputs$processed_X$Xarr, inputs$processed_X$n,
					  inputs$processed_X$Warr,
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
					  model$precomputed$TransCtCinvCt,
					  model$precomputed$CtC,
					  model$precomputed$B_plus_bias,
					  model$info$nthreads)
	check.ret.code(ret_code)
	inputs$processed_X$Xarr <- matrix(inputs$processed_X$Xarr, ncol = nrow(X), nrow = ncol(X))
	if (!is.null(rownames(X)))
		colnames(inputs$processed_X$Xarr) <- rownames(X)
	if (!is.null(colnames(X)))
		rownames(inputs$processed_X$Xarr) <- colnames(X)
	X <- t(inputs$processed_X$Xarr)
	return(X)
}
