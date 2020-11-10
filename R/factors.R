
process.data.factors <- function(model, obj, X=NULL, weight=NULL,
								 U=NULL, U_bin=NULL,
								 output_bias=FALSE,
								 output_A=FALSE, exact=FALSE,
								 matched_shapes=FALSE) {
	output_bias      <-  check.bool(output_bias, "output_bias")
	exact            <-  check.bool(exact, "exact")
	output_A         <-  check.bool(output_A, "output_A")
	processed_X      <-  process.new.X(obj, X, weight = weight)
	processed_U      <-  process.new.U(U = U, U_cols = obj$info$U_cols,
									   p = NCOL(obj$matrices$C), name = "U",
									   allow_sparse = TRUE,
									   allow_null = model != "ContentBased",
									   allow_na = model %in% c("CMF", "CMF_implicit"),
									   exact_shapes = !(model %in% c("CMF", "CMF_implicit")))
	processed_U_bin  <-  process.new.U(U = U_bin, U_cols = obj$info$U_bin_cols,
									   p = NCOL(obj$matrices$Cb), name = "U_bin",
									   allow_sparse=FALSE, allow_null=TRUE,
									   allow_na=TRUE, exact_shapes=FALSE)
	
	if (NROW(processed_X$Xarr) && model == "OMF_explicit") {
		if (processed_X$n != NCOL(obj$matrices$Bm))
			stop(sprintf("'X' has %s columns than the model was fit to.",
						 ifelse(processed_X$n > NCOL(obj$matrices$Bm), "more", "less")))
	}
	
	if (matched_shapes) {
		msg_rows <- "'X' and 'U' must have the same rows."
		if (NROW(processed_X$Xarr) && NROW(processed_U$Uarr)) {
			if (processed_X$m != processed_U$m)
				stop(msg_rows)
		} else if (NROW(processed_X$Xarr) && !NROW(processed_U$Uarr)) {
			if (NROW(processed_U$Uval) || NROW(processed_U$Ucsr)) {
				if (processed_U$m < processed_X$m) {
					processed_U$m <- processed_X$m
				} else if (processed_U$m > processed_X$m) {
					stop(msg_rows)
				}
			}
		} else if (!NROW(processed_X$Xarr) && NROW(processed_U$Uarr)) {
			if (NROW(processed_X$Xval) || NROW(processed_X$Xcsr)) {
				if (processed_X$m < processed_U$m) {
					processed_X$m <- processed_U$m
				} else if (processed_X$m > processed_U$m) {
					stop(msg_rows)
				}
			}
		}
		
		if (processed_X$m != processed_U$m) {
			if (processed_X$m)
				processed_X$m <- max(c(processed_X$m, processed_U$m))
			if (processed_U$m)
				processed_U$m <- max(c(processed_X$m, processed_U$m))
		}
	}
	
	return(list(
		processed_X = processed_X,
		processed_U = processed_U,
		processed_U_bin = processed_U_bin,
		output_bias = output_bias,
		output_A = output_A,
		exact = exact
	))
}

#' @export
factors.CMF <- function(model, X=NULL, weight=NULL, U=NULL, U_bin=NULL, output_bias=FALSE) {
	inputs <- process.data.factors(class(model)[1L], model,
								   X = X, weight = weight,
								   U = U, U_bin = U_bin,
								   output_bias = output_bias)
	m_max <- max(c(inputs$processed_X$m, inputs$processed_U$m, inputs$processed_U_bin$m))
	A <- matrix(0., ncol = m_max, nrow = model$info$k_user + model$info$k + model$info$k_main)
	biasA <- numeric()
	if (NROW(inputs$user_bias))
		biasA <- numeric(length = m_max)
	ret_code <- .Call("call_factors_collective_explicit_multiple",
					  A, biasA, m_max,
					  inputs$processed_U$Uarr, inputs$processed_U$m, inputs$processed_U$p,
					  model$info$NA_as_zero_user, model$info$NA_as_zero,
					  inputs$processed_U$Urow, inputs$processed_U$Ucol, inputs$processed_U$Uval,
					  inputs$processed_U$Ucsr_p, inputs$processed_U$Ucsr_i, inputs$processed_U$Ucsr,
					  inputs$processed_U_bin$Uarr, inputs$processed_U_bin$m, inputs$processed_U_bin$p,
					  model$matrices$C, model$matrices$Cb,
					  model$matrices$glob_mean, model$matrices$item_bias,
					  model$matrices$U_colmeans,
					  inputs$processed_X$Xval, inputs$processed_X$Xrow, inputs$processed_X$Xcol,
					  inputs$processed_X$Xcsr_p, inputs$processed_X$Xcsr_i, inputs$processed_X$Xcsr,
					  inputs$processed_X$Xarr, inputs$processed_X$n,
					  inputs$processed_X$Warr, inputs$processed_X$Wsp,
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
	A <- t(A)
	if (output_bias) {
		return(list(factors = A, bias = biasA))
	} else {
		return(A)
	}
}

#' @export
factors.CMF_implicit <- function(model, X=NULL, U=NULL, output_bias=FALSE) {
	inputs <- process.data.factors(class(model)[1L], model,
								   X = X,
								   U = U)
	m_max <- max(c(inputs$processed_X$m, inputs$processed_U$m))
	A <- matrix(0., ncol = m_max, nrow = model$info$k_user + model$info$k + model$info$k_main)
	
	lambda <- ifelse(NROW(model$info$lambda) == 1L, model$info$lambda, model$info$lambda[3L])
	
	ret_code <- .Call("call_factors_collective_implicit_multiple",
					  A, m_max,
					  inputs$processed_U$Uarr,inputs$processed_U$m, inputs$processed_U$p,
					  model$info$NA_as_zero_user,
					  inputs$processed_U$Urow, inputs$processed_U$Ucol, inputs$processed_U$Uval,
					  inputs$processed_U$Ucsr_p, inputs$processed_U$Ucsr_i, inputs$processed_U$Ucsr,
					  inputs$processed_X$Xval, inputs$processed_X$Xrow, inputs$processed_X$Xcol,
					  inputs$processed_X$Xcsr_p, inputs$processed_X$Xcsr_i, inputs$processed_X$Xcsr,
					  model$matrices$B, NCOL(model$matrices$B),
					  model$matrices$C,
					  model$matrices$U_colmeans,
					  model$info$k, model$info$k_user, model$info$k_item, model$info$k_main,
					  lambda, model$info$alpha, model$info$w_main, model$info$w_user,
					  model$info$w_main_multiplier,
					  model$precomputed$BeTBe,
					  model$precomputed$BtB,
					  model$precomputed$BeTBeChol,
					  model$info$nthreads)
	
	check.ret.code(ret_code)
	A <- t(A)
	return(A)
}

#' @export
factors.ContentBased <- function(model, U=NULL) {
	inputs <- process.data.factors(class(model)[1L], model,
								   U = U)
	m_max <- inputs$processed_U$m
	A <- matrix(0., ncol = m_max, nrow = model$info$k)
	
	ret_code <- .Call("call_factors_content_based_mutliple",
					  A, m_max, model$info$k,
					  model$matrices$C, model$matrices$C_bias,
					  inputs$processed_U$Uarr, inputs$processed_U$p,
					  inputs$processed_U$Urow, inputs$processed_U$Ucol, inputs$processed_U$Uval,
					  inputs$processed_U$Ucsr_p, inputs$processed_U$Ucsr_i, inputs$processed_U$Ucsr,
					  model$info$nthreads)
	
	check.ret.code(ret_code)
	A <- t(A)
	return(A)
}

#' @export
factors.OMF_explicit <- function(model, X=NULL, weight=NULL, U=NULL, output_bias=FALSE,
								 output_A = FALSE, exact=FALSE) {
	if (!exact && !output_A &&
		!is.null(X) && !is.null(U)) {
		warning("'U' data is ignored in the presence of 'X' data with 'exact=FALSE'.")
		U <- NULL
	}
	inputs <- process.data.factors(class(model)[1L], model,
								   X = X, weight = weight,
								   U = U,
								   output_bias = output_bias,
								   output_A = output_A, exact = exact,
								   matched_shapes = TRUE)
	m_max <- max(c(inputs$processed_X$m, inputs$processed_U$m))
	A <- matrix(0., ncol = m_max, nrow = model$info$k_sec + model$info$k + model$info$k_main)
	biasA <- numeric()
	if (NROW(inputs$user_bias))
		biasA <- numeric(length = m_max)
	Aorig <- numeric()
	if (inputs$output_A) {
		if (NROW(model$matrices$C)) {
			Aorig <- matrix(0., ncol = m_max, nrow = model$info$k + model$info$k_main)
		} else {
			warning("Option 'output_A' invalid when the model was not fit to 'U' data.")
			inputs$output_A <- FALSE
		}
	}
	
	ret_code <- .Call("call_factors_offsets_explicit_multiple",
					  A, biasA,
					  Aorig, m_max,
					  inputs$processed_U$Uarr, inputs$processed_U$p,
					  inputs$processed_U$Urow, inputs$processed_U$Ucol, inputs$processed_U$Uval,
					  inputs$processed_U$Ucsr_p, inputs$processed_U$Ucsr_i, inputs$processed_U$Ucsr,
					  inputs$processed_X$Xval, inputs$processed_X$Xrow, inputs$processed_X$Xcol,
					  inputs$processed_X$Xcsr_p, inputs$processed_X$Xcsr_i, inputs$processed_X$Xcsr,
					  inputs$processed_X$Xarr, inputs$processed_X$n,
					  inputs$processed_X$Warr, inputs$processed_X$Wsp,
					  model$matrices$Bm, model$matrices$C,
					  model$matrices$C_bias,
					  model$matrices$glob_mean, model$matrices$item_bias,
					  model$info$k, model$info$k_sec, model$info$k_main,
					  model$info$w_user,
					  model$info$lambda, inputs$exact,
					  model$precomputed$TransBtBinvBt,
					  model$precomputed$BtB,
					  model$precomputed$Bm_plus_bias,
					  model$info$nthreads)
	
	check.ret.code(ret_code)
	A <- t(A)
	if (!inputs$output_bias && !inputs$output_A)
		return(A)
	out <- list(factors = A)
	if (inputs$output_bias) {
		out$bias <- biasA
	}
	if (inputs$output_A) {
		Aorig <- t(Aorig)
		out$A <- Aorig
	}
	return(out)
}

#' @export
factors.OMF_implicit <- function(model, X=NULL, U=NULL, output_bias=FALSE, output_A=FALSE) {
	if (!output_A &&
		!is.null(X) && !is.null(U)) {
		warning("'U' data is ignored in the presence of 'X' data.")
		U <- NULL
	}
	inputs <- process.data.factors(class(model)[1L], model,
								   X = X,
								   U = U,
								   output_A = output_A,
								   matched_shapes = TRUE)
	m_max <- max(c(inputs$processed_X$m, inputs$processed_U$m))
	A <- matrix(0., ncol = m_max, nrow = model$info$k)
	Aorig <- numeric()
	if (inputs$output_A) {
		if (NROW(model$matrices$C)) {
			Aorig <- matrix(0., ncol = m_max, nrow = model$info$k + model$info$k_main)
		} else {
			warning("Option 'output_A' invalid when the model was not fit to 'U' data.")
			inputs$output_A <- FALSE
		}
	}
	
	ret_code <- .Call("call_factors_offsets_implicit_multiple",
					  A, m_max,
					  Aorig,
					  inputs$processed_U$Uarr, inputs$processed_U$p,
					  inputs$processed_U$Urow, inputs$processed_U$Ucol, inputs$processed_U$Uval,
					  inputs$processed_U$Ucsr_p, inputs$processed_U$Ucsr_i, inputs$processed_U$Ucsr,
					  inputs$processed_X$Xval, inputs$processed_X$Xrow, inputs$processed_X$Xcol,
					  inputs$processed_X$Xcsr_p, inputs$processed_X$Xcsr_i, inputs$processed_X$Xcsr,
					  model$matrices$Bm, model$matrices$C,
					  model$matrices$C_bias,
					  model$info$k, NCOL(model$matrices$Bm),
					  model$info$lambda, model$info$alpha,
					  model$precomputed$BtB,
					  model$info$nthreads)
	
	check.ret.code(ret_code)
	A <- t(A)
	if (inputs$output_A) {
		return(list(factors = A, A = t(Aorig)))
	} else {
		return(A)
	}
}
