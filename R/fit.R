#' @importFrom parallel detectCores

validate.inputs <- function(model, implicit=FALSE,
							X=NULL, U=NULL, I=NULL, U_bin=NULL, I_bin=NULL, weight=NULL,
							k=40L, lambda=10., method="als", use_cg=TRUE,
							user_bias=TRUE, item_bias=TRUE,
							k_user=0L, k_item=0L, k_main=0L, k_sec=0L,
							w_main=1., w_user=1., w_item=1., w_implicit=0.5,
							alpha=1., downweight=FALSE,
							add_implicit_features=FALSE,
							add_intercepts=TRUE,
							start_with_ALS=TRUE,
							maxiter=800L, niter=10L, parallelize="separate", corr_pairs=4L,
							max_cg_steps=3L, finalize_chol=TRUE,
							NA_as_zero=FALSE, NA_as_zero_user=FALSE, NA_as_zero_item=FALSE,
							precompute_for_predictions=TRUE, include_all_X=TRUE,
							verbose=TRUE, print_every=10L,
							handle_interrupt=TRUE,
							nthreads=parallel::detectCores()) {
	
	k        <-  check.pos.int(k, "k", model != "OMF_explicit")
	k_user   <-  check.pos.int(k_user, "k_user", FALSE)
	k_item   <-  check.pos.int(k_item, "k_item", FALSE)
	k_main   <-  check.pos.int(k_main, "k_main", FALSE)
	k_sec    <-  check.pos.int(k_sec, "k_sec", FALSE)
	maxiter  <-  check.pos.int(maxiter, "maxiter", FALSE)
	niter    <-  check.pos.int(niter, "niter", TRUE)
	corr_pairs    <-  check.pos.int(corr_pairs, "corr_pairs", TRUE)
	max_cg_steps  <-  check.pos.int(max_cg_steps, "max_cg_steps", TRUE)
	print_every   <-  check.pos.int(print_every, "print_every", TRUE)
	nthreads      <-  check.pos.int(nthreads, "nthreads", TRUE)
	
	use_cg     <-  check.bool(use_cg, "use_cg")
	user_bias  <-  check.bool(user_bias, "user_bias")
	item_bias  <-  check.bool(item_bias, "item_bias")
	finalize_chol    <-  check.bool(finalize_chol, "finalize_chol")
	NA_as_zero       <-  check.bool(NA_as_zero, "NA_as_zero")
	NA_as_zero_user  <-  check.bool(NA_as_zero_user, "NA_as_zero_user")
	NA_as_zero_item  <-  check.bool(NA_as_zero_item, "NA_as_zero_item")
	include_all_X    <-  check.bool(include_all_X, "include_all_X")
	verbose          <-  check.bool(verbose, "verbose")
	handle_interrupt <-  check.bool(handle_interrupt, "handle_interrupt")
	downweight       <-  check.bool(downweight, "downweight")
	implicit         <-  check.bool(implicit, "implicit")
	add_intercepts   <-  check.bool(add_intercepts, "add_intercepts")
	start_with_ALS   <-  check.bool(start_with_ALS, "start_with_ALS")
	precompute_for_predictions  <-  check.bool(precompute_for_predictions, "precompute_for_predictions")
	add_implicit_features       <-  check.bool(add_implicit_features, "add_implicit_features")
	
	w_main      <-  check.pos.real(w_main, "w_main")
	w_user      <-  check.pos.real(w_user, "w_user")
	w_item      <-  check.pos.real(w_item, "w_item")
	w_implicit  <-  check.pos.real(w_implicit, "w_implicit")
	alpha       <-  check.pos.real(alpha, "alpha")
	
	method       <-  check.str.option(method, "method", c("als", "lbfgs"))
	parallelize  <-  check.str.option(parallelize, "parallelize", c("separate", "single"))
	
	allow_different_lambda <- TRUE
	if (model == "OMF_implicit")
		allow_different_lambda <- FALSE
	if (model == "OMF_explicit" && method == "als")
		allow_different_lambda <- FALSE
	lambda <- check.lambda(lambda, allow_different_lambda)
	
	if (k_user > 0 && is.null(U) && is.null(U_bin))
		stop("Cannot pass 'k_user' with no 'U' data.")
	if (k_item > 0 && is.null(I) && is.null(I_bin))
		stop("Cannot pass 'k_item' with no 'I' data.")
	if (method == "als" && (!is.null(U_bin) || !is.null(I_bin)))
		stop("Cannot use 'method=als' when there is 'U_bin' or 'I_bin'.")
	if (NA_as_zero && (user_bias || item_bias))
		stop("user/item biases not supported with 'NA_as_zero'.")
	if ((k_user+k+k_main+1)^2 > .Machine$integer.max)
		stop("Number of factors is too large.")
	if ((method == "lbfgs") && (NA_as_zero || NA_as_zero_user || NA_as_zero_item))
		stop("Option 'NA_as_zero' not supported with 'method='lbfgs'.")
	if ((method == "lbfgs") && add_implicit_features)
		stop("Option 'add_implicit_features' not supported with 'method='lbfgs'.")
	if (nthreads < 1L)
		nthreads <- parallel::detectCores()
	
	if (!NROW(X))
		stop("'X' cannot be empty.")
	
	if (model == "ContentBased") {
		if (!NROW(U) || !NROW(I))
			stop("'ContentBased' cannot be fit without side info.")
	}
	if (model == "OMF_explicit") {
		if (!k_sec && !k && !k_main)
			stop("Must have at least one of 'k', 'k_sec', 'k_main'.")
	}
	
	reindex <- FALSE
	if (check.is.df(X)) {
		reindex <- TRUE
		if (NCOL(X) < 3L)
			stop("'X', if passed as 'data.frame', must have 3 or 4 columns.")
		if (!is.null(weight))
			stop("'weight' should be passed as fourth column of 'X'.")
		
		err_msg      <-  "If 'X' is a 'data.frame', '%s' must also be a 'data.frame'."
		msg_dup      <-  "'%s' has duplicated row names."
		msg_nonames  <-  "'%s' must have row names matching to 'X'."
		if (!is.null(U)) {
			if (!check.is.df.or.mat(U))
				stop(sprintf(err_msg, "U"))
			if (is.null(row.names(U)))
				stop(sprintf(msg_nonames, "U"))
			if (any(duplicated(row.names(U))))
				stop(sprintf(msg_dup, "U"))
		}
		if (!is.null(I)) {
			if (!check.is.df.or.mat(I))
				stop(sprintf(err_msg, "I"))
			if (is.null(row.names(I)))
				stop(sprintf(msg_nonames, "I"))
			if (any(duplicated(row.names(I))))
				stop(sprintf(msg_dup, "I"))
		}
		if (!is.null(U_bin)) {
			if (!check.is.df.or.mat(U_bin))
				stop(sprintf(err_msg, "U_bin"))
			if (is.null(row.names(U_bin)))
				stop(sprintf(msg_nonames, "U_bin"))
			if (any(duplicated(row.names(U_bin))))
				stop(sprintf(msg_dup, "U_bin"))
		}
		if (!is.null(I_bin)) {
			if (!check.is.df.or.mat(I_bin))
				stop(sprintf(err_msg, "I_bin"))
			if (is.null(row.names(I_bin)))
				stop(sprintf(msg_nonames, "I_bin"))
			if (any(duplicated(row.names(I))))
				stop(sprintf(msg_dup, "I_bin"))
		}
		
		X <- cast.data.frame(X)
	}
	
	allowed_X   <- c("data.frame", "dgTMatrix", "matrix.coo", "matrix")
	allowed_U   <- c("data.frame", "matrix")
	allowed_bin <- c("data.frame", "matrix")
	allowed_W   <- c("data.frame", "matrix", "numeric")
	msg_err     <- "Invalid '%s' - allowed types: %s"
	msg_empty   <- "'%s' is empty. If non-present, should pass it as NULL."
	if (!(model %in% c("OMF_explicit", "OMF_implicit") && method == "als"))
		allowed_U <- c(allowed_U, c("dgTMatrix", "matrix.coo"))
	
	if (!NROW(intersect(class(X), allowed_X)))
		stop(sprintf(msg_err, "X", paste(allowed_X, collapse=", ")))
	if (!is.null(U)) {
		if (!NROW(intersect(class(U), allowed_U)))
			stop(sprintf(msg_err, "U", paste(allowed_U, collapse=", ")))
		if (!max(c(NROW(U), NCOL(U))))
			stop(sprintf(msg_empty, "U"))
	}
	if (!is.null(I)) {
		if (!NROW(intersect(class(I), allowed_U)))
			stop(sprintf(msg_err, "I", paste(allowed_U, collapse=", ")))
		if (!max(c(NROW(I), NCOL(I))))
			stop(sprintf(msg_empty, "I"))
	}
	if (!is.null(U_bin)) {
		if (!NROW(intersect(class(U_bin), allowed_bin)))
			stop(sprintf(msg_err, "U_bin", paste(allowed_bin, collapse=", ")))
		if (!max(c(NROW(U_bin), NCOL(U_bin))))
			stop(sprintf(msg_empty, "U_bin"))
	}
	if (!is.null(I_bin)) {
		if (!NROW(intersect(class(I_bin), allowed_bin)))
			stop(sprintf(msg_err, "I_bin", paste(allowed_bin, collapse=", ")))
		if (!max(c(NROW(I_bin), NCOL(I_bin))))
			stop(sprintf(msg_empty, "I_bin"))
	}
	if (!is.null(weight)) {
		if (!NROW(intersect(class(weight), allowed_W)))
			stop(sprintf(msg_err, "weight", paste(allowed_W, collapse=", ")))
	}
	
	U_cols      <-  character()
	I_cols      <-  character()
	U_bin_cols  <-  character()
	I_bin_cols  <-  character()
	if ("data.frame" %in% class(U))
		U_cols <- names(U)
	if ("data.frame" %in% class(I))
		I_cols <- names(I)
	if ("data.frame" %in% class(U_bin))
		U_bin_cols <- names(U_bin)
	if ("data.frame" %in% class(I_bin))
		I_bin_cols <- names(I_bin)
	
	U      <-  cast.df.to.matrix(U)
	I      <-  cast.df.to.matrix(I)
	U_bin  <-  cast.df.to.matrix(U_bin)
	I_bin  <-  cast.df.to.matrix(I_bin)
	weight <-  cast.df.to.matrix(weight)
	
	user_mapping <- character()
	item_mapping <- character()
	if (reindex) {
		temp <- reindex.data(X, U, I, U_bin, I_bin)
		X <- temp$X
		U <- temp$U
		I <- temp$I
		U_bin <- temp$U_bin
		I_bin <- temp$I_bin
		user_mapping <- temp$user_mapping
		item_mapping <- temp$item_mapping
	}
	
	processed_X     <-  process.X(X, weight)
	processed_U     <-  process.side.info(U, "U", TRUE)
	processed_I     <-  process.side.info(I, "I", TRUE)
	processed_U_bin <-  process.side.info(U_bin, "U_bin", TRUE)
	processed_I_bin <-  process.side.info(I_bin, "I_bin", TRUE)
	
	if (model %in% c("OMF_explicit", "OMF_implicit", "ContentBased")) {
		if (NROW(processed_X$Xarr)) {
			if (NROW(processed_U$Uarr) && (processed_X$m != processed_U$m))
				stop("Rows of 'X' and 'U' must match")
			if (NROW(processed_I$Uarr) && (processed_X$n != processed_I$m))
				stop("Columns of 'X' and rows of 'I' must match")
		}
	}
	
	if (model == "ContentBased") {
		if (!processed_U$m || !processed_U$p) {
			stop("'U' cannot be empty.")
		}
		if (!processed_I$m || !processed_I$p) {
			stop("'I' cannot be empty.")
		}
	}
	
	msg_na <- "'NA_as_zero' not meaningful when passing dense data, will be ignored."
	if (NA_as_zero) {
		if (NROW(processed_X$Xarr)) {
			warning(msg_na)
			NA_as_zero <- FALSE
		} else {
			processed_X$m <- max(c(processed_X$m, processed_U$m, processed_U_bin$m))
			processed_X$n <- max(c(processed_X$n, processed_I$m, processed_I_bin$m))
		}
	}
	if (NA_as_zero_user) {
		if (NROW(processed_U$Uarr)) {
			warning(msg_na)
			NA_as_zero_user <- FALSE
		} else if (processed_U$m) {
			processed_U$m <- max(c(processed_U$m, processed_X$m, processed_U_bin$m))
		} else {
			NA_as_zero_user <- FALSE
		}
	}
	if (NA_as_zero_item) {
		if (NROW(processed_I$Uarr)) {
			warning(msg_na)
			NA_as_zero_item <- FALSE
		} else if (processed_I$m) {
			processed_I$m <- max(c(processed_I$m, processed_X$n, processed_I_bin$m))
		} else {
			NA_as_zero_item <- FALSE
		}
	}
	
	return(list(
		processed_X = processed_X,
		processed_U = processed_U,
		processed_I = processed_I,
		processed_U_bin = processed_U_bin,
		processed_I_bin = processed_I_bin,
		user_mapping = user_mapping,
		item_mapping = item_mapping,
		U_cols = U_cols, I_cols = I_cols,
		U_bin_cols = U_bin_cols, I_bin_cols = I_bin_cols,
		k = k, lambda = lambda, method = method, use_cg = use_cg,
		user_bias = user_bias, item_bias = item_bias,
		k_user = k_user, k_item = k_item, k_main = k_main, k_sec = k_sec,
		w_main = w_main, w_user = w_user, w_item = w_item, w_implicit = w_implicit,
		alpha = alpha, downweight = downweight,
		implicit = implicit,
		add_implicit_features = add_implicit_features,
		add_intercepts = add_intercepts,
		start_with_ALS = start_with_ALS,
		maxiter = maxiter, niter = niter, parallelize = parallelize, corr_pairs = corr_pairs,
		max_cg_steps = max_cg_steps, finalize_chol = finalize_chol,
		NA_as_zero = NA_as_zero, NA_as_zero_user = NA_as_zero_user, NA_as_zero_item = NA_as_zero_item,
		precompute_for_predictions = precompute_for_predictions, include_all_X = include_all_X,
		verbose = verbose, print_every = print_every,
		handle_interrupt = handle_interrupt,
		nthreads = nthreads
	))
}

#' @export
CMF <- function(X, U=NULL, I=NULL, U_bin=NULL, I_bin=NULL, weight=NULL,
				k=40L, lambda=10., method="als", use_cg=TRUE,
				user_bias=TRUE, item_bias=TRUE, add_implicit_features=FALSE,
				k_user=0L, k_item=0L, k_main=0L,
				w_main=1., w_user=1., w_item=1., w_implicit=0.5,
				maxiter=800L, niter=10L, parallelize="separate", corr_pairs=4L,
				max_cg_steps=3L, finalize_chol=TRUE,
				NA_as_zero=FALSE, NA_as_zero_user=FALSE, NA_as_zero_item=FALSE,
				precompute_for_predictions=TRUE, include_all_X=TRUE,
				verbose=TRUE, print_every=10L,
				handle_interrupt=TRUE,
				nthreads=parallel::detectCores()) {
	
	inputs <- validate.inputs(model = "CMF",
							  X = X, U = U, I = I, U_bin = U_bin, I_bin = I_bin, weight = weight,
							  k = k, lambda = lambda, method = method, use_cg = use_cg,
							  user_bias = user_bias, item_bias = item_bias,
							  add_implicit_features = add_implicit_features,
							  k_user = k_user, k_item = k_item, k_main = k_main,
							  w_main = w_main, w_user = w_user, w_item = w_item,
							  w_implicit = w_implicit,
							  maxiter = maxiter, niter = niter,
							  parallelize = parallelize, corr_pairs = corr_pairs,
							  max_cg_steps = max_cg_steps, finalize_chol = finalize_chol,
							  NA_as_zero = NA_as_zero,
							  NA_as_zero_user = NA_as_zero_user,
							  NA_as_zero_item = NA_as_zero_item,
							  precompute_for_predictions = precompute_for_predictions,
							  include_all_X = include_all_X,
							  verbose = verbose, print_every = print_every,
							  handle_interrupt = handle_interrupt,
							  nthreads = nthreads)
	return(.CMF(inputs$processed_X, inputs$processed_U, inputs$processed_I,
				inputs$processed_U_bin, inputs$processed_I_bin,
				user_mapping = inputs$user_mapping, item_mapping = inputs$item_mapping,
				inputs$U_cols, inputs$I_cols,
				inputs$U_bin_cols, inputs$I_bin_cols,
				k = inputs$k, lambda = inputs$lambda, method = inputs$method, use_cg = inputs$use_cg,
				user_bias = inputs$user_bias, item_bias = inputs$item_bias,
				add_implicit_features = inputs$add_implicit_features,
				k_user = inputs$k_user, k_item = inputs$k_item, k_main = inputs$k_main,
				w_main = inputs$w_main, w_user = inputs$w_user, w_item = inputs$w_item,
				w_implicit = inputs$w_implicit,
				maxiter = inputs$maxiter, niter = inputs$niter,
				parallelize = inputs$parallelize, corr_pairs = inputs$corr_pairs,
				max_cg_steps = inputs$max_cg_steps, finalize_chol = inputs$finalize_chol,
				NA_as_zero = inputs$NA_as_zero,
				NA_as_zero_user = inputs$NA_as_zero_user,
				NA_as_zero_item = inputs$NA_as_zero_item,
				precompute_for_predictions = inputs$precompute_for_predictions,
				include_all_X = inputs$include_all_X,
				verbose = inputs$verbose, print_every = inputs$print_every,
				handle_interrupt = inputs$handle_interrupt,
				nthreads = inputs$nthreads))
	
}

#' @export
CMF_implicit <- function(X, U=NULL, I=NULL,
						 k=40L, lambda=1., alpha=1., use_cg=TRUE,
						 k_user=0L, k_item=0L, k_main=0L,
						 w_main=1., w_user=1., w_item=1.,
						 niter=10L, downweight=FALSE,
						 max_cg_steps=3L, finalize_chol=TRUE,
						 NA_as_zero_user=FALSE, NA_as_zero_item=FALSE,
						 precompute_for_predictions=TRUE,
						 verbose=TRUE,
						 handle_interrupt=TRUE,
						 nthreads=parallel::detectCores()) {
	
	inputs <- validate.inputs(model = "CMF_implicit",
							  X = X, U = U, I = I,
							  k = k, lambda = lambda, use_cg = use_cg,
							  alpha = alpha, downweight = downweight,
							  k_user = k_user, k_item = k_item, k_main = k_main,
							  w_main = w_main, w_user = w_user, w_item = w_item,
							  niter = niter,
							  max_cg_steps = max_cg_steps, finalize_chol = finalize_chol,
							  NA_as_zero_user = NA_as_zero_user,
							  NA_as_zero_item = NA_as_zero_item,
							  precompute_for_predictions = precompute_for_predictions,
							  verbose = verbose,
							  handle_interrupt = handle_interrupt,
							  nthreads = nthreads)
	return(.CMF_implicit(inputs$processed_X, inputs$processed_U, inputs$processed_I,
						 inputs$user_mapping, inputs$item_mapping,
						 inputs$U_cols, inputs$I_cols,
						 k = inputs$k, lambda = inputs$lambda, use_cg = inputs$use_cg,
						 alpha = inputs$alpha, downweight = inputs$downweight,
						 k_user = inputs$k_user, k_item = inputs$k_item, k_main = inputs$k_main,
						 w_main = inputs$w_main, w_user = inputs$w_user, w_item = inputs$w_item,
						 niter = inputs$niter,
						 max_cg_steps = inputs$max_cg_steps, finalize_chol = inputs$finalize_chol,
						 NA_as_zero_user = inputs$NA_as_zero_user,
						 NA_as_zero_item = inputs$NA_as_zero_item,
						 precompute_for_predictions = inputs$precompute_for_predictions,
						 verbose = inputs$verbose,
						 handle_interrupt = inputs$handle_interrupt,
						 nthreads = inputs$nthreads))
	
}


.CMF <- function(processed_X, processed_U, processed_I, processed_U_bin, processed_I_bin,
				 user_mapping, item_mapping,
				 U_cols, I_cols,
				 U_bin_cols, I_bin_cols,
				 k=40L, lambda=10., method="als", use_cg=TRUE,
				 user_bias=TRUE, item_bias=TRUE,
				 add_implicit_features=FALSE,
				 k_user=0L, k_item=0L, k_main=0L,
				 w_main=1., w_user=1., w_item=1., w_implicit=0.5,
				 maxiter=400L, niter=10L, parallelize="separate", corr_pairs=4L,
				 max_cg_steps=3L, finalize_chol=TRUE,
				 NA_as_zero=FALSE, NA_as_zero_user=FALSE, NA_as_zero_item=FALSE,
				 precompute_for_predictions=TRUE, include_all_X=TRUE,
				 verbose=TRUE, print_every=10L,
				 handle_interrupt=TRUE,
				 nthreads=parallel::detectCores()) {
	
	
	this <- list(
		info = get.empty.info(),
		matrices = get.empty.matrices(),
		precomputed = get.empty.precomputed()
	)
	
	### Fill in info
	this$info$w_main      <-  w_main
	this$info$w_user      <-  w_user
	this$info$w_item      <-  w_item
	this$info$w_implicit  <-  w_implicit
	this$info$n_orig      <-  processed_X$n
	this$info$k           <-  k
	this$info$k_user      <-  k_user
	this$info$k_item      <-  k_item
	this$info$k_main      <-  k_main
	this$info$lambda      <-  lambda
	this$info$user_mapping     <-  user_mapping
	this$info$item_mapping     <-  item_mapping
	this$info$U_cols           <-  U_cols
	this$info$I_cols           <-  I_cols
	this$info$U_bin_cols       <-  U_bin_cols
	this$info$I_bin_cols       <-  I_bin_cols
	this$info$NA_as_zero       <-  NA_as_zero
	this$info$NA_as_zero_user  <-  NA_as_zero_user
	this$info$NA_as_zero_item  <-  NA_as_zero_item
	this$info$include_all_X    <-  include_all_X
	this$info$nthreads         <-  nthreads
	this$info$add_implicit_features  <-  add_implicit_features
	
	### Allocate matrices
	m_max <- max(c(processed_X$m, processed_U$m, processed_U_bin$m))
	n_max <- max(c(processed_X$n, processed_I$m, processed_I_bin$m))
	this$matrices$A <- matrix(0., ncol=m_max, nrow=k_user+k+k_main)
	this$matrices$B <- matrix(0., ncol=n_max, nrow=k_item+k+k_main)
	
	if (user_bias) {
		this$matrices$user_bias <- numeric(m_max)
	}
	if (item_bias) {
		this$matrices$item_bias <- numeric(n_max)
	}
	if (add_implicit_features) {
		this$matrices$Ai <- matrix(0., ncol=m_max, nrow=k+k_main)
		this$matrices$Bi <- matrix(0., ncol=n_max, nrow=k+k_main)
	}
	if (processed_U$p) {
		this$matrices$C <- matrix(0., ncol=processed_U$p, nrow=k_user+k)
		this$matrices$U_colmeans <- numeric(processed_U$p)
	}
	if (processed_I$p) {
		this$matrices$D <- matrix(0., ncol=processed_I$p, nrow=k_item+k)
		this$matrices$I_colmeans <- numeric(processed_I$p)
	}
	if (processed_U_bin$p) {
		this$matrices$Cb <- matrix(0., ncol=processed_U_bin$p, nrow=k_user+k)
	}
	if (processed_I_bin$p) {
		this$matrices$Db <- matrix(0., ncol=processed_I_bin$p, nrow=k_item+k)
	}
	
	### Allocate precomputed
	if (precompute_for_predictions) {
		if (user_bias) {
			this$precomputed$B_plus_bias <- matrix(0., ncol=n_max, nrow=k_item+k+k_main+1L)
		}
		this$precomputed$BtB <- matrix(0., nrow=k+k_main+user_bias, ncol=k+k_main+user_bias)
		this$precomputed$TransBtBinvBt <- matrix(0., ncol=n_max, nrow=k+k_main+user_bias)
		if (add_implicit_features)
			this$precompted$BiTBi <- matrix(0., ncol=k+k_main, nrow=k+k_main)
		if (processed_U$p) {
			this$precomputed$CtC <- matrix(0., ncol=k_user+k, nrow=k_user+k)
			this$precomputed$TransCtCinvCt <- matrix(0., ncol=processed_U$p, nrow=k_user+k)
		}
		if (add_implicit_features || processed_U$p) {
			this$precomputed$BeTBeChol <- matrix(0., nrow=k_user+k+k_main+user_bias,
												 ncol=k_user+k+k_main+user_bias)
		}
	}
	
	### Note: for some reason, R keeps pointers of old objects when calling 'get.empty.info',
	### this avoids potentially modifying other objects in the environment
	glob_mean <- numeric(1L)
	nupd <- integer(1L)
	nfev <- integer(1L)
	
	if (method == "als") {
		ret_code <- .Call("call_fit_collective_explicit_als",
						  this$matrices$user_bias, this$matrices$item_bias,
						  this$matrices$A, this$matrices$B,
						  this$matrices$C, this$matrices$D,
						  this$matrices$Ai, this$matrices$Bi,
						  glob_mean,
						  this$matrices$U_colmeans, this$matrices$I_colmeans,
						  processed_X$m, processed_X$n, k,
						  processed_X$Xrow, processed_X$Xcol, processed_X$Xval,
						  processed_X$Xarr,
						  processed_X$Warr, processed_X$Wsp,
						  user_bias, item_bias,
						  lambda,
						  processed_U$Uarr, processed_U$m, processed_U$p,
						  processed_I$Uarr, processed_I$m, processed_I$p,
						  processed_U$Urow, processed_U$Ucol, processed_U$Uval,
						  processed_I$Urow, processed_I$Ucol, processed_I$Uval,
						  NA_as_zero, NA_as_zero_user, NA_as_zero_item,
						  k_main, k_user, k_item,
						  w_main, w_user, w_item, w_implicit,
						  niter, nthreads, verbose, handle_interrupt,
						  use_cg, max_cg_steps, finalize_chol,
						  precompute_for_predictions,
						  add_implicit_features,
						  include_all_X,
						  this$precomputed$B_plus_bias,
						  this$precomputed$BtB,
						  this$precomputed$TransBtBinvBt,
						  this$precomputed$BeTBeChol,
						  this$precomputed$BiTBi,
						  this$precomputed$TransCtCinvCt,
						  this$precomputed$CtC)
	} else {
		ret_code <- .Call("call_fit_collective_explicit_lbfgs",
						  this$matrices$user_bias, this$matrices$item_bias,
						  this$matrices$A, this$matrices$B,
						  this$matrices$C, this$matrices$Cb,
						  this$matrices$D, this$matrices$Db,
						  glob_mean,
						  this$matrices$U_colmeans, this$matrices$I_colmeans,
						  processed_X$m, processed_X$n, k,
						  processed_X$Xrow, processed_X$Xcol, processed_X$Xval,
						  processed_X$Xarr,
						  processed_X$Warr, processed_X$Wsp,
						  user_bias, item_bias,
						  lambda,
						  processed_U$Uarr, processed_U$m, processed_U$p,
						  processed_I$Uarr, processed_I$m, processed_I$p,
						  processed_U_bin$Uarr, processed_U_bin$m, processed_U_bin$p,
						  processed_I_bin$Uarr, processed_I_bin$m, processed_I_bin$p,
						  processed_U$Urow, processed_U$Ucol, processed_U$Uval,
						  processed_I$Urow, processed_I$Ucol, processed_I$Uval,
						  k_main, k_user, k_item,
						  w_main, w_user, w_item,
						  corr_pairs, maxiter, print_every,
						  nupd, nfev,
						  parallelize == "single",
						  nthreads, verbose, handle_interrupt,
						  precompute_for_predictions,
						  include_all_X,
						  this$precomputed$B_plus_bias,
						  this$precomputed$BtB,
						  this$precomputed$TransBtBinvBt,
						  this$precomputed$BeTBeChol,
						  this$precomputed$TransCtCinvCt,
						  this$precomputed$CtC)
	}
	
	this$info$nupd  <-  nupd
	this$info$nfev  <-  nfev
	this$matrices$glob_mean <- glob_mean
	
	check.ret.code(ret_code)
	class(this) <- c("CMF", "cmfrec")
	return(this)
}

.CMF_implicit <- function(processed_X, processed_U, processed_I,
						  user_mapping, item_mapping,
						  U_cols, I_cols,
						  k=40L, lambda=1., use_cg=TRUE,
						  alpha=1., downweight=FALSE,
						  k_user=0L, k_item=0L, k_main=0L,
						  w_main=1., w_user=1., w_item=1.,
						  niter=10L,
						  max_cg_steps=3L, finalize_chol=TRUE,
						  NA_as_zero_user=FALSE, NA_as_zero_item=FALSE,
						  precompute_for_predictions=TRUE,
						  verbose=TRUE,
						  handle_interrupt=TRUE,
						  nthreads=parallel::detectCores()) {
	
	
	this <- list(
		info = get.empty.info(),
		matrices = get.empty.matrices(),
		precomputed = get.empty.precomputed()
	)
	
	### Fill in info
	this$info$w_main  <-  w_main
	this$info$w_user  <-  w_user
	this$info$w_item  <-  w_item
	this$info$n_orig  <-  processed_X$n
	this$info$k       <-  k
	this$info$k_user  <-  k_user
	this$info$k_item  <-  k_item
	this$info$k_main  <-  k_main
	this$info$lambda  <-  lambda
	this$info$alpha   <-  alpha
	this$info$user_mapping     <-  user_mapping
	this$info$item_mapping     <-  item_mapping
	this$info$U_cols           <-  U_cols
	this$info$I_cols           <-  I_cols
	this$info$NA_as_zero_user  <-  NA_as_zero_user
	this$info$NA_as_zero_item  <-  NA_as_zero_item
	this$info$nthreads         <-  nthreads
	
	### Allocate matrices
	m_max <- max(c(processed_X$m, processed_U$m))
	n_max <- max(c(processed_X$n, processed_I$m))
	this$matrices$A <- matrix(0., ncol=m_max, nrow=k_user+k+k_main)
	this$matrices$B <- matrix(0., ncol=n_max, nrow=k_item+k+k_main)
	
	if (processed_U$p) {
		this$matrices$C <- matrix(0., ncol=processed_U$p, nrow=k_user+k)
		this$matrices$U_colmeans <- numeric(processed_U$p)
	}
	if (processed_I$p) {
		this$matrices$D <- matrix(0., ncol=processed_I$p, nrow=k_item+k)
		this$matrices$I_colmeans <- numeric(processed_I$p)
	}

	### Allocate precomputed
	if (precompute_for_predictions) {
		this$precomputed$BtB <- matrix(0., nrow=k+k_main, ncol=k+k_main)
		if (processed_U$p) {
			this$precomputed$BeTBe <- matrix(0., nrow=k_user+k+k_main,
												ncol=k_user+k+k_main)
			this$precomputed$BeTBeChol <- matrix(0., nrow=k_user+k+k_main,
												    ncol=k_user+k+k_main)
		}
	}
	
	### Note: for some reason, R keeps pointers of old objects when calling 'get.empty.info',
	### this avoids potentially modifying other objects in the environment
	w_main_multiplier <- numeric(1L)
	
	ret_code <- .Call("call_fit_collective_implicit_als",
					  this$matrices$A, this$matrices$B,
					  this$matrices$C, this$matrices$D,
					  w_main_multiplier,
					  this$matrices$U_colmeans, this$matrices$I_colmeans,
					  processed_X$m, processed_X$n, k,
					  processed_X$Xrow, processed_X$Xcol, processed_X$Xval,
					  lambda,
					  processed_U$Uarr, processed_U$m, processed_U$p,
					  processed_I$Uarr, processed_I$m, processed_I$p,
					  processed_U$Urow, processed_U$Ucol, processed_U$Uval,
					  processed_I$Urow, processed_I$Ucol, processed_I$Uval,
					  NA_as_zero_user, NA_as_zero_item,
					  k_main, k_user, k_item,
					  w_main, w_user, w_item,
					  niter, nthreads, verbose, handle_interrupt,
					  use_cg, max_cg_steps, finalize_chol,
					  this$info$alpha, downweight,
					  precompute_for_predictions,
					  this$precomputed$BtB,
					  this$precomputed$BeTBe,
					  this$precomputed$BeTBeChol)
	
	this$info$w_main_multiplier <- w_main_multiplier
	
	check.ret.code(ret_code)
	class(this) <- c("CMF_implicit", "cmfrec")
	return(this)
}

#' @export
MostPopular <- function(X, weight=NULL, implicit=FALSE, user_bias=FALSE,
						lambda=10., alpha=1.,
						downweight=FALSE) {
	inputs <- validate.inputs(model = "MostPopular", implicit = implicit,
							  X = X, weight = weight,
							  user_bias = user_bias, lambda = lambda,
							  alpha = alpha, downweight = downweight)
	if (inputs$implicit && inputs$user_bias)
		stop("Cannot fit user biases with 'implicit=TRUE'.")
	return(.MostPopular(processed_X = inputs$processed_X,
						user_mapping = inputs$user_mapping, item_mapping = inputs$item_mapping,
						implicit = inputs$implicit,
						user_bias = inputs$user_bias,
						lambda = inputs$lambda, alpha = inputs$alpha,
						downweight = inputs$downweight))
}

.MostPopular <- function(processed_X,
						 user_mapping, item_mapping,
						 implicit=FALSE, user_bias=FALSE, lambda=10., alpha=1.,
						 downweight=FALSE) {
	
	this <- list(
		info = get.empty.info(),
		matrices = get.empty.matrices(),
		precomputed = get.empty.precomputed()
	)
	
	### Fill in info
	this$info$lambda  <-  lambda
	this$info$alpha   <-  alpha
	this$info$user_mapping  <-  user_mapping
	this$info$item_mapping  <-  item_mapping
	this$info$implicit <- implicit
	
	### Allocate matrices
	this$matrices$item_bias <- numeric(processed_X$n)
	if (user_bias)
		this$matrices$user_bias <- numeric(processed_X$m)
	
	### Note: for some reason, R keeps pointers of old objects when calling 'get.empty.info',
	### this avoids potentially modifying other objects in the environment
	glob_mean <- numeric(1L)
	w_main_multiplier <- numeric(1L)
	
	ret_code <- .Call("call_fit_most_popular",
					  this$matrices$user_bias, this$matrices$item_bias,
					  glob_mean,
					  lambda,
					  alpha,
					  processed_X$m, processed_X$n,
					  processed_X$Xrow, processed_X$Xcol, processed_X$Xval,
					  processed_X$Xarr,
					  processed_X$Warr, processed_X$Wsp,
					  implicit, downweight,
					  w_main_multiplier,
					  1L)
	
	this$matrices$glob_mean <- glob_mean
	this$info$w_main_multiplier <- w_main_multiplier
	
	check.ret.code(ret_code)
	class(this) <- c("MostPopular", "cmfrec")
	return(this)
}

#' @export
ContentBased <- function(X, U, I, weight=NULL,
						 k=20L, lambda=100., user_bias=FALSE, item_bias=FALSE,
						 add_intercepts=TRUE, maxiter=15000L, corr_pairs=3L,
						 parallelize="separate", verbose=TRUE, print_every=100L,
						 handle_interrupt=TRUE, start_with_ALS=TRUE,
						 nthreads=parallel::detectCores()) {
	inputs <- validate.inputs(model = "ContentBased",
							  X = X, U = U, I = I, weight = weight,
							  k = k, lambda = lambda, user_bias = user_bias, item_bias = item_bias,
							  add_intercepts = add_intercepts,
							  maxiter = maxiter, corr_pairs = corr_pairs,
							  parallelize = parallelize, verbose = verbose, print_every = print_every,
							  handle_interrupt = handle_interrupt, start_with_ALS = start_with_ALS,
							  nthreads = nthreads)
	return(.ContentBased(inputs$processed_X, inputs$processed_U, inputs$processed_I,
						 inputs$user_mapping, inputs$item_mapping,
						 inputs$U_cols, inputs$I_cols,
						 k = inputs$k, lambda = inputs$lambda,
						 user_bias = inputs$user_bias, item_bias = inputs$item_bias,
						 add_intercepts = inputs$add_intercepts,
						 maxiter = inputs$maxiter, corr_pairs = inputs$corr_pairs,
						 parallelize = inputs$parallelize,
						 verbose = inputs$verbose, print_every = inputs$print_every,
						 handle_interrupt = inputs$handle_interrupt,
						 start_with_ALS = inputs$start_with_ALS,
						 nthreads = inputs$start_with_ALS))
}

.ContentBased <- function(processed_X, processed_U, processed_I,
						  user_mapping, item_mapping,
						  U_cols, I_cols,
						  k=20L, lambda=100., user_bias=FALSE, item_bias=FALSE,
						  add_intercepts=TRUE, maxiter=15000L, corr_pairs=3L,
						  parallelize="separate", verbose=TRUE, print_every=100L,
						  handle_interrupt=TRUE, start_with_ALS=TRUE,
						  nthreads=parallel::detectCores()) {
	
	this <- list(
		info = get.empty.info(),
		matrices = get.empty.matrices(),
		precomputed = get.empty.precomputed()
	)
	
	### Fill in info
	this$info$k       <-  k
	this$info$lambda  <-  lambda
	this$info$user_mapping  <-  user_mapping
	this$info$item_mapping  <-  item_mapping
	this$info$U_cols    <-  U_cols
	this$info$I_cols    <-  I_cols
	this$info$nthreads  <-  nthreads
	
	### Allocate matrices
	m_max <- max(c(processed_X$m, processed_U$m))
	n_max <- max(c(processed_X$n, processed_I$m))
	this$matrices$Am <- matrix(0., ncol=m_max, nrow=k)
	this$matrices$Bm <- matrix(0., ncol=n_max, nrow=k)
	this$matrices$C  <- matrix(0., ncol=processed_U$p, nrow=k)
	this$matrices$D  <- matrix(0., ncol=processed_I$p, nrow=k)
	
	if (user_bias) {
		this$matrices$user_bias <- numeric(m_max)
	}
	if (item_bias) {
		this$matrices$item_bias <- numeric(n_max)
	}
	if (add_intercepts) {
		this$matrices$C_bias <- numeric(k)
		this$matrices$D_bias <- numeric(k)
	}
	
	### Note: for some reason, R keeps pointers of old objects when calling 'get.empty.info',
	### this avoids potentially modifying other objects in the environment
	glob_mean <- numeric(1L)
	nupd <- integer(1L)
	nfev <- integer(1L)
	
	
	ret_code <- .Call("call_fit_content_based_lbfgs",
					  this$matrices$user_bias, this$matrices$item_bias,
					  this$matrices$C, this$matrices$C_bias,
					  this$matrices$D, this$matrices$D_bias,
					  start_with_ALS,
					  glob_mean,
					  m_max, n_max, k,
					  processed_X$Xrow, processed_X$Xcol, processed_X$Xval,
					  processed_X$Xarr,
					  processed_X$Warr, processed_X$Wsp,
					  user_bias, item_bias,
					  add_intercepts,
					  lambda,
					  processed_U$Uarr, processed_U$p,
					  processed_I$Uarr, processed_I$p,
					  processed_U$Urow, processed_U$Ucol, processed_U$Uval,
					  processed_I$Urow, processed_I$Ucol, processed_I$Uval,
					  corr_pairs, maxiter,
					  nthreads, parallelize != "separate",
					  verbose, print_every, handle_interrupt,
					  nupd, nfev,
					  this$matrices$Am, this$matrices$Bm)
	
	this$info$nupd  <-  nupd
	this$info$nfev  <-  nfev
	this$matrices$glob_mean <- glob_mean
	
	
	check.ret.code(ret_code)
	class(this) <- c("ContentBased", "cmfrec")
	return(this)
}

#' @export
OMF_explicit <- function(X, U=NULL, I=NULL, weight=NULL,
						 k=50L, lambda=1e1, method="lbfgs", use_cg=TRUE,
						 user_bias=TRUE, item_bias=TRUE, k_sec=0L, k_main=0L,
						 add_intercepts=TRUE, w_user=1., w_item=1.,
						 maxiter=10000L, niter=10L, parallelize="separate", corr_pairs=7L,
						 max_cg_steps=3L, finalize_chol=TRUE,
						 NA_as_zero=FALSE,
						 verbose=TRUE, print_every=100L,
						 handle_interrupt=TRUE,
						 nthreads=parallel::detectCores()) {
	
	inputs <- validate.inputs(model = "OMF_explicit",
							  X = X, U = U, I = I, weight = weight,
							  k = k, lambda = lambda, method = method, use_cg = use_cg,
							  user_bias = user_bias, item_bias = item_bias,
							  k_sec = k_sec, k_main = k_main,
							  add_intercepts = add_intercepts, w_user = w_user, w_item = w_item,
							  maxiter = maxiter, niter = niter,
							  parallelize = parallelize, corr_pairs = corr_pairs,
							  max_cg_steps = max_cg_steps, finalize_chol = finalize_chol,
							  NA_as_zero = NA_as_zero,
							  verbose = verbose, print_every = print_every,
							  handle_interrupt = handle_interrupt,
							  nthreads = nthreads)
	return(.OMF_explicit(inputs$processed_X, inputs$processed_U, inputs$processed_I,
						 inputs$user_mapping, inputs$item_mapping,
						 inputs$U_cols, inputs$I_cols,
						 k = inputs$k, lambda = inputs$lambda,
						 method = inputs$method, use_cg = inputs$use_cg,
						 user_bias = inputs$user_bias, item_bias = inputs$item_bias,
						 k_sec = inputs$k_sec, k_main = inputs$k_main,
						 add_intercepts = inputs$add_intercepts,
						 w_user = inputs$w_user, w_item = inputs$w_item,
						 maxiter = inputs$maxiter, niter = inputs$niter,
						 parallelize = inputs$parallelize, corr_pairs = inputs$corr_pairs,
						 max_cg_steps = inputs$max_cg_steps, finalize_chol = inputs$finalize_chol,
						 NA_as_zero = inputs$NA_as_zero,
						 verbose = inputs$verbose, print_every = inputs$print_every,
						 handle_interrupt = inputs$handle_interrupt,
						 nthreads = nthreads))
}

.OMF_explicit <- function(processed_X, processed_U, processed_I,
						  user_mapping, item_mapping,
						  U_cols, I_cols,
						  k=50L, lambda=1e1, method="lbfgs", use_cg=TRUE,
						  user_bias=TRUE, item_bias=TRUE, k_sec=0L, k_main=0L,
						  add_intercepts=TRUE, w_user=1., w_item=1.,
						  maxiter=10000L, niter=10L, parallelize="separate", corr_pairs=7L,
						  max_cg_steps=3L, finalize_chol=TRUE,
						  NA_as_zero=FALSE,
						  verbose=TRUE, print_every=100L,
						  handle_interrupt=TRUE,
						  nthreads=parallel::detectCores()) {
	
	this <- list(
		info = get.empty.info(),
		matrices = get.empty.matrices(),
		precomputed = get.empty.precomputed()
	)
	
	### Fill in info
	this$info$k       <-  k
	this$info$k_sec   <-  k_sec
	this$info$k_main  <-  k_main
	this$info$lambda  <-  lambda
	this$info$user_mapping  <-  user_mapping
	this$info$item_mapping  <-  item_mapping
	this$info$U_cols    <-  U_cols
	this$info$I_cols    <-  I_cols
	this$info$nthreads  <-  nthreads
	
	### Allocate matrices
	m_max <- max(c(processed_X$m, processed_U$m))
	n_max <- max(c(processed_X$n, processed_I$m))
	this$matrices$A  <- matrix(0., ncol=m_max, nrow=ifelse(processed_U$m, k+k_main, k_sec+k+k_main))
	this$matrices$B  <- matrix(0., ncol=n_max, nrow=ifelse(processed_I$m, k+k_main, k_sec+k+k_main))
	this$matrices$Am <- matrix(0., ncol=m_max, nrow=k_sec+k+k_main)
	this$matrices$Bm <- matrix(0., ncol=n_max, nrow=k_sec+k+k_main)
	this$matrices$C  <- matrix(0., ncol=processed_U$p, nrow=k+k_sec)
	this$matrices$D  <- matrix(0., ncol=processed_I$p, nrow=k+k_sec)
	
	if (user_bias) {
		this$matrices$user_bias <- numeric(m_max)
	}
	if (item_bias) {
		this$matrices$item_bias <- numeric(n_max)
	}
	if (add_intercepts) {
		this$matrices$C_bias <- numeric(k+k_sec)
		this$matrices$D_bias <- numeric(k+k_sec)
	}
	
	### Allocate precomputed
	if (TRUE) {
		if (user_bias) {
			this$precomputed$Bm_plus_bias <- matrix(0., ncol=n_max, nrow=k_sec+k+k_main+1L)
		}
		this$precomputed$BtB <- matrix(0., nrow=k_sec+k+k_main+user_bias, ncol=k_sec+k+k_main+user_bias)
		this$precomputed$TransBtBinvBt <- matrix(0., ncol=n_max, nrow=k_sec+k+k_main+user_bias)
	}
	
	### Note: for some reason, R keeps pointers of old objects when calling 'get.empty.info',
	### this avoids potentially modifying other objects in the environment
	glob_mean <- numeric(1L)
	nupd <- integer(1L)
	nfev <- integer(1L)
	
	if (method == "lbfgs") {
		ret_code <- .Call("call_fit_offsets_explicit_lbfgs",
						  this$matrices$user_bias, this$matrices$item_bias,
						  this$matrices$A, this$matrices$B,
						  this$matrices$C, this$matrices$C_bias,
						  this$matrices$D, this$matrices$D_bias,
						  glob_mean,
						  m_max, n_max, k,
						  processed_X$Xrow, processed_X$Xcol, processed_X$Xval,
						  processed_X$Xarr,
						  processed_X$Warr, processed_X$Wsp,
						  user_bias, item_bias, add_intercepts,
						  lambda,
						  processed_U$Uarr, processed_U$p,
						  processed_I$Uarr, processed_I$p,
						  processed_U$Urow, processed_U$Ucol, processed_U$Uval,
						  processed_I$Urow, processed_I$Ucol, processed_I$Uval,
						  k_main, k_sec,
						  w_user, w_item,
						  corr_pairs, maxiter,
						  nthreads, parallelize == "single",
						  verbose, print_every, handle_interrupt,
						  nupd, nfev,
						  TRUE,
						  this$matrices$Am, this$matrices$Bm,
						  this$precomputed$Bm_plus_bias,
						  this$precomputed$BtB,
						  this$precomputed$TransBtBinvBt)
	} else {
		ret_code <- .Call("call_fit_offsets_explicit_als",
						  this$matrices$user_bias, this$matrices$item_bias,
						  this$matrices$A, this$matrices$B,
						  this$matrices$C, this$matrices$C_bias,
						  this$matrices$D, this$matrices$D_bias,
						  glob_mean,
						  m_max, n_max, k,
						  processed_X$Xrow, processed_X$Xcol, processed_X$Xval,
						  processed_X$Xarr,
						  processed_X$Warr, processed_X$Wsp,
						  user_bias, item_bias, add_intercepts,
						  lambda,
						  processed_U$Uarr, processed_U$p,
						  processed_I$Uarr, processed_I$p,
						  NA_as_zero,
						  niter,
						  nthreads, use_cg,
						  max_cg_steps, finalize_chol,
						  verbose, handle_interrupt,
						  TRUE,
						  this$matrices$Am, this$matrices$Bm,
						  this$precomputed$Bm_plus_bias,
						  this$precomputed$BtB,
						  this$precomputed$TransBtBinvBt)
	}
	
	this$info$nupd  <-  nupd
	this$info$nfev  <-  nfev
	this$matrices$glob_mean <- glob_mean
	
	
	check.ret.code(ret_code)
	class(this) <- c("OMF_explicit", "cmfrec")
	return(this)
}

#' @export
OMF_implicit <- function(X, U=NULL, I=NULL,
						 k=50L, lambda=1e0, alpha=1., use_cg=TRUE,
						 add_intercepts=TRUE, niter=10L,
						 max_cg_steps=3L, finalize_chol=TRUE,
						 verbose=FALSE,
						 handle_interrupt=TRUE,
						 nthreads=parallel::detectCores()) {
	
	inputs <- validate.inputs(model = "OMF_implicit",
							  X = X, U = U, I = I,
							  k = k, lambda = lambda, alpha = alpha, use_cg = use_cg,
							  add_intercepts = add_intercepts, niter = niter,
							  max_cg_steps = max_cg_steps, finalize_chol = finalize_chol,
							  verbose = verbose,
							  handle_interrupt = handle_interrupt,
							  nthreads = nthreads)
	return(.OMF_implicit(inputs$processed_X, inputs$processed_U, inputs$processed_I,
						 inputs$user_mapping, inputs$item_mapping,
						 inputs$U_cols, inputs$I_cols,
						 k = inputs$k, lambda = inputs$lambda, alpha = inputs$alpha,
						 use_cg = inputs$use_cg,
						 add_intercepts = inputs$add_intercepts, niter = inputs$niter,
						 max_cg_steps = inputs$max_cg_steps, finalize_chol = inputs$finalize_chol,
						 verbose = inputs$verbose,
						 handle_interrupt = inputs$handle_interrupt,
						 nthreads = nthreads))
	
}

.OMF_implicit <- function(processed_X, processed_U, processed_I,
						  user_mapping, item_mapping,
						  U_cols, I_cols,
						  k=50L, lambda=1e0, alpha=1., use_cg=TRUE,
						  add_intercepts=TRUE, niter=10L,
						  max_cg_steps=3L, finalize_chol=TRUE,
						  verbose=FALSE,
						  handle_interrupt=TRUE,
						  nthreads=parallel::detectCores()) {
	
	this <- list(
		info = get.empty.info(),
		matrices = get.empty.matrices(),
		precomputed = get.empty.precomputed()
	)
	
	### Fill in info
	this$info$k       <-  k
	this$info$lambda  <-  lambda
	this$info$alpha   <-  alpha
	this$info$user_mapping  <-  user_mapping
	this$info$item_mapping  <-  item_mapping
	this$info$U_cols    <-  U_cols
	this$info$I_cols    <-  I_cols
	this$info$nthreads  <-  nthreads
	
	### Allocate matrices
	m_max <- max(c(processed_X$m, processed_U$m))
	n_max <- max(c(processed_X$n, processed_I$m))
	this$matrices$A  <- matrix(0., ncol=m_max, nrow=k)
	this$matrices$B  <- matrix(0., ncol=n_max, nrow=k)
	this$matrices$Am <- matrix(0., ncol=m_max, nrow=k)
	this$matrices$Bm <- matrix(0., ncol=n_max, nrow=k)
	this$matrices$C  <- matrix(0., ncol=processed_U$p, nrow=k)
	this$matrices$D  <- matrix(0., ncol=processed_I$p, nrow=k)
	
	if (add_intercepts) {
		this$matrices$C_bias <- numeric(k)
		this$matrices$D_bias <- numeric(k)
	}
	
	### Allocate precomputed
	if (TRUE) {
		this$precomputed$BtB <- matrix(0., nrow=k, ncol=k)
	}
	
	ret_code <- .Call("call_fit_offsets_implicit_als",
					  this$matrices$A, this$matrices$B,
					  this$matrices$C, this$matrices$C_bias,
					  this$matrices$D, this$matrices$D_bias,
					  processed_X$m, processed_X$n, this$info$k,
					  processed_X$Xrow, processed_X$Xcol, processed_X$Xval,
					  add_intercepts,
					  this$info$lambda,
					  processed_U$Uarr, processed_U$p,
					  processed_I$Uarr, processed_I$p,
					  this$info$alpha,
					  niter,
					  this$info$nthreads, use_cg,
					  max_cg_steps, finalize_chol,
					  verbose, handle_interrupt,
					  TRUE,
					  this$matrices$Am, this$matrices$Bm,
					  this$precomputed$BtB)
	
	
	check.ret.code(ret_code)
	class(this) <- c("OMF_implicit", "cmfrec")
	return(this)
}

#' @export
precompute.for.predictions <- function(model) {
	supported_models <- c("CMF", "CMF_implicit")
	if (!NROW(intersect(class(model), supported_models)))
		stop(sprintf("Method is only applicable to ", paste(supported_models, collapse=", ")))
	
	n_use <- NCOL(model$matrices$B)
	n_max <- n_use
	if (!model$info$include_all_X)
		n_use <- model$info$n_orig
	
	user_bias  <-  as.logical(NROW(model$matrices$user_bias))
	k          <-  model$info$k
	k_user     <-  model$info$k_user
	k_item     <-  model$info$k_item
	k_main     <-  model$info$k_main
	
	p <- NCOL(model$matrices$C)
	has_U <- as.logical(p)
	add_implicit_features <- as.logical(NROW(model$matrices$Bi))
	
	if ("CMF" %in% class(model)) {
		if (user_bias) {
			model$precomputed$B_plus_bias <- matrix(0., ncol=n_max, nrow=k_item+k+k_main+1L)
		}
		model$precomputed$BtB <- matrix(0., nrow=k+k_main+user_bias, ncol=k+k_main+user_bias)
		model$precomputed$TransBtBinvBt <- matrix(0., ncol=n_use, nrow=k+k_main+user_bias)
		if (add_implicit_features)
			model$precompted$BiTBi <- matrix(0., ncol=k+k_main, nrow=k+k_main)
		if (has_U) {
			model$precomputed$CtC <- matrix(0., ncol=k_user+k, nrow=k_user+k)
			model$precomputed$TransCtCinvCt <- matrix(0., ncol=p, nrow=k_user+k)
		}
		if (add_implicit_features || has_U) {
			model$precomputed$BeTBeChol <- matrix(0., nrow=k_user+k+k_main+user_bias,
												  ncol=k_user+k+k_main+user_bias)
		}
		
		ret_code <- .Call("call_precompute_collective_explicit",
						  model$matrices$B, n_use, n_max, model$info$include_all_X,
						  model$matrices$C, p,
						  model$matrices$Bi, add_implicit_features,
						  model$info$k, model$info$k_user, model$info$k_item, model$info$k_main,
						  user_bias,
						  model$info$lambda,
						  model$info$w_main, model$info$w_user, model$info$w_implicit,
						  model$precomputed$B_plus_bias,
						  model$precomputed$BtB,
						  model$precomputed$TransBtBinvBt,
						  model$precomputed$BeTBeChol,
						  model$precompted$BiTBi,
						  model$precomputed$TransCtCinvCt,
						  model$precomputed$CtC)
	} else if ("CMF_implicit" %in% class(model)) {
		
		model$precomputed$BtB <- matrix(0., nrow=k+k_main, ncol=k+k_main)
		if (has_U) {
			model$precomputed$BeTBe <- matrix(0., nrow=k_user+k+k_main,
											  ncol=k_user+k+k_main)
			model$precomputed$BeTBeChol <- matrix(0., nrow=k_user+k+k_main,
												  ncol=k_user+k+k_main)
		}
		
		ret_code <- .Call("call_precompute_collective_implicit",
						  model$matrices$B, n_max,
						  model$matrices$C, p,
						  model$info$k, model$info$k_user, model$info$k_item, model$info$k_main,
						  model$info$lambda, model$info$w_main, model$info$w_user,
						  model$info$w_main_multiplier,
						  TRUE,
						  model$precomputed$BtB,
						  model$precomputed$BeTBe,
						  model$precomputed$BeTBeChol)
	} else {
		stop("Unexpected error.")
	}
	
	check.ret.code(ret_code)
	return(model)
}
