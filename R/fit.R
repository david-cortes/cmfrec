#' @importFrom parallel detectCores
#' @useDynLib cmfrec, .registration=TRUE
NULL

#' @rdname fit
#' @name fit_models
#' @title Matrix Factorization Models
#' @description Models for collective matrix factorization (also known as multi-view or
#' multi-way). These models try to approximate a matrix `X` as the product of two
#' lower-rank matrices `A` and `B` (that is:
#' \eqn{\mathbf{X} \approx \mathbf{A} \mathbf{B}^T}{X ~ A*t(B)})
#' by finding the values of `A` and `B` that minimize the squared error w.r.t. `X`,
#' optionally aided with side information matrices `U` and `I` about
#' rows and columns of `X`.
#' 
#' The package documentation is built with recommendation systems in mind,
#' for which it assumes that `X` is a sparse matrix in which users represent rows,
#' items represent columns, and the non-missing values denote interactions such as
#' movie ratings from users to items. The idea behind it is to recommend the missing
#' entries in `X` that have the highest predicted value according to the approximation.
#' For other domains, take any mention of users as rows and any mention of items as
#' columns (e.g. when used for topic modeling, the "users" are documents
#' and the "items" are word occurrences).
#' 
#' In the `CMF` model (main functionality of the package and most flexible model type),
#' the `A` and `B` matrices are also used to jointly factorize the side information
#' matrices - that is:
#' \eqn{\mathbf{U} \approx \mathbf{A}\mathbf{C}^T, \:\:\:
#' \mathbf{I} \approx \mathbf{B}\mathbf{D}^T}{U ~ A*t(C), I ~ B*t(D)},
#' sharing the same components or latent factors for two factorizations.
#' Informally, this means that the obtained factors now need to explain both the
#' interactions data and the attributes data, making them generalize better to
#' the non-present entries of `X` and to new data.
#' 
#' In `CMF` and the other non-implicit models, the `X` data is always centered
#' beforehand by subtracting its mean, and might optionally add user and item
#' biases (which are model parameters, not pre-estimated).
#' 
#' The model might optionally generate so-called implicit features from the same `X`
#' data, by factorizing binary matrices which tell which entries in `X` are present,
#' i.e.: \eqn{\mathbf{I}_x \approx \mathbf{A}\mathbf{B}_i^T , \:\:\:
#' \mathbf{I}_x^T \approx \mathbf{B}\mathbf{A}_i^T}{Ix ~ A*t(Bi), t(Ix) ~ B*Ai},
#' where \eqn{\mathbf{I}_x}{Ix} is an indicator matrix which is treated as
#' full (no unknown values).
#' 
#' The `CMF_implicit` model extends the collective factorization idea to the implicit-feedback
#' case, based on reference [3]. While in `CMF` the values of `X` are taken at face value
#' and the objective is to minimize squared error over the non-missing entries, in the
#' implicit-feedback variants the matrix `X` is assumed to be binary (all entries are zero
#' or one, with no unknown values), with the positive entries (those which are not
#' missing in the data) having a weight determined by `X`.
#' 
#' `CMF` is intended for explicit feedback data (e.g. movie ratings, which contain both
#' likes and dislikes), whereas `CMF_implicit` is intended for implicit feedback data
#' (e.g. number of times each user watched each movie/series, which do not contain
#' dislikes and the values are treated as confidence scores).
#' 
#' The `MostPopular` model is a simpler heuristic implemented for comparison purposes
#' which is equivalent to either `CMF` or `CMF_implicit` with `k=1` (or alternatively,
#' `k=0` plus user/item biases). If a personalized model is not able to beat this
#' heuristic under the evaluation metrics of interest, chances are that such personalized
#' model needs better tuning.
#' 
#' The `ContentBased` model offers a different alternative in which the latent factors
#' are determined directly from the user/item attributes (which are no longer optional) -
#' that is: \eqn{\mathbf{A} = \mathbf{U} \mathbf{C}, \:\:\:
#' \mathbf{B} = \mathbf{I} \mathbf{D}}{A = U*C, B = I*D}, optionally adding per-column
#' intercepts, and is aimed at cold-start predictions (such a model is extremely
#' unlikely to perform better for new users in the presence of interactions data).
#' For this model, the package provides functionality for making predictions about
#' potential new entries in `X` which involve both new rows and new columns at the
#' same time.
#' Unlike the others, it does not offer an implicit-feedback variant.
#' 
#' The `OMF_explicit` model extends the `ContentBased` by adding a free offset determined
#' for each user and item according to `X` data alone - that is:
#' \eqn{\mathbf{A}_m = \mathbf{A} + \mathbf{U} \mathbf{C}, \:\:\:
#' \mathbf{B}_m = \mathbf{B} + \mathbf{I}\mathbf{D}, \:\:\:
#' \mathbf{X} \approx \mathbf{A}_m \mathbf{B}_m^T}{Am = A + U*C, Bm = B + I*D, X ~ Am*t(Bm)},
#' and `OMF_implicit` extends the idea to the implicit-feedback case.
#' 
#' Note that `ContentBased` is equivalent to `OMF_explicit` with `k=0`, `k_main=0` and `k_sec>0`
#' (see documentation for details about these parameters).
#' For a different formulation in which user factors are determined directly for item attributes
#' (and same for items with user attributes), it's also possible to use `OMF_explicit` with
#' `k=0` while passing `k_sec` and `k_main`.
#' 
#' (`OMF_explicit` and `OMF_implicit` were only implemented for research purposes for
#' cold-start recommendations in cases in which there is side info about users but not about
#' items or vice-versa - it is not recommended to rely on them.)
#' 
#' Some extra considerations about the parameters here: \itemize{
#' \item By default, the terms in the optimization objective are not scaled by the
#' number of entries (see parameter `scale_lam`), thus hyperparameters such as `lambda`
#' will require more tuning than in other software and will require trying a wider range of values.
#' \item The regularization applied to the matrices is the same for all users and for all items.
#' \item The default hyperparameters are not geared towards speed - for faster fitting times,
#' use `method='als'`, `use_cg=TRUE`, `finalize_chol=FALSE`, `precompute_for_predictions=FALSE`,
#' `verbose=FALSE`, and pass `X` as a matrix (either sparse or dense).
#' \item The default hyperparameters are also very different than in other software - for example,
#' for `CMF_implicit`, in order to match the Python package's `implicit` hyperparameters,
#' one would have to use `k=100`, `lambda=0.01`, `niter=15`, `use_cg=TRUE`, `finalize_chol=FALSE`,
#' and use single-precision floating point numbers (not supported in the R version of this
#' package).
#' }
#' 
#' @param X The main matrix with interactions data to factorize (e.g. movie ratings by users,
#' bag-of-words representations of texts, etc.). The package is built with
#' recommender systems in mind, and will assume that `X` is a matrix in which users
#' are rows, items are columns, and values denote interactions between a given user and item.
#' Can be passed in the following formats: \itemize{
#' \item A `data.frame` representing triplets, in which there should be one row for each
#' present or non-missing interaction, with the first column denoting the user/row ID,
#' the second column the item/column ID, and the third column the value (e.g. movie rating).
#' If passed in this format, the user and item IDs will be reindexed internally, and the
#' side information matrices should have row names matching to those IDs. If there are
#' observation weights, these should be the fourth column.
#' \item A sparse matrix in COO/triplets format, either from package
#' `Matrix` (class `dgTMatrix`) or from package `SparseM` (class `matrix.coo`).
#' \item A dense matrix from base R (class `matrix`), with missing values set as `NA`/`NaN`.
#' }
#' 
#' If using the package `softImpute`, objects of type `incomplete` from that package
#' can be converted to `Matrix` objects through e.g. `as(X, "TsparseMatrix")`.
#' Sparse matrices can be created through e.g.
#' `Matrix::sparseMatrix(..., giveCsparse=FALSE)`.
#' 
#' It is recommended for faster fitting times to pass the `X` data as a matrix
#' (either sparse or dense) as then it will avoid internal reindexes.
#' 
#' Note that, generally, it's possible to pass partially disjoints sets of users/items between
#' the different matrices (e.g. it's possible for both the `X` and `U`
#' matrices to have rows that the other doesn't have).
#' If any of the inputs has less rows/columns than the other(s) (e.g.
#' `U` has more rows than `X`, or `I` has more rows than there are columns
#' in `X`), will assume that the rest of the rows/columns have only
#' missing values.
#' However, when having partially disjoint inputs, the order of
#' the rows/columns matters for speed for the `CMF` and `CMF_implicit` models
#' under the ALS method,
#' as it might run faster when the `U`/`I`
#' inputs that do not have matching rows/columns in `X` have those unmatched
#' rows/columns at the end (last rows/columns) and the `X` input is shorter.
#' See also the parameter `include_all_X` for info about predicting with
#' mismatched `X`.
#' 
#' If passed as sparse/triplets, the non-missing values should not contain any `NA`/`NaN`s.
#' @param U User attributes information. Can be passed in the following formats: \itemize{
#' \item A `matrix`, with rows corresponding to rows of `X` and columns to user
#' attributes. For the `CMF` and `CMF_implicit` models, missing values are supported
#' and should be set to `NA`/`NaN`.
#' \item A `data.frame` with the same format as above.
#' \item A sparse matrix in COO/triplets format, either from package
#' `Matrix` (class `dgTMatrix`) or from package `SparseM` (class `matrix.coo`).
#' Same as above, rows correspond to rows of `X` and columns to user
#' attributes. If passed as sparse, the non-missing values cannot contain `NA`/`NaN` -
#' see parameter `NA_as_zero_user` for how to interpret non-missing values.
#' Sparse side info is not supported for `OMF_implicit`, nor for `OMF_explicit`
#' with `method=als`.
#' }
#' If `X` is a `data.frame`, should be
#' either a `data.frame` or `matrix`, containing row names matching to the first
#' column of `X` (which
#' denotes the user/row IDs of the non-zero entries). If `U` is sparse,
#' `X` should be passed as sparse or dense matrix (not a `data.frame`).
#' 
#' Note that, if `U` is a `matrix` or `data.frame`, it should have the same
#' number of rows as `X` in the `ContentBased`, `OMF_explicit`,
#' and `OMF_implicit` models.
#' 
#' Be aware that `CMF` and `CMF_implicit` tend to perform better with dense
#' and not-too-wide user/item attributes.
#' @param I Item attributes information. Can be passed in the following formats: \itemize{
#' \item A `matrix`, with rows corresponding to columns of `X` and columns to item
#' attributes. For the `CMF` and `CMF_implicit` models, missing values are supported
#' and should be set to `NA`/`NaN`.
#' \item A `data.frame` with the same format as above.
#' \item A sparse matrix in COO/triplets format, either from package
#' `Matrix` (class `dgTMatrix`) or from package `SparseM` (class `matrix.coo`).
#' Same as above, rows correspond to columns of `X` and columns to item
#' attributes. If passed as sparse, the non-missing values cannot contain `NA`/`NaN` -
#' see parameter `NA_as_zero_item` for how to interpret non-missing values.
#' Sparse side info is not supported for `OMF_implicit`, nor for `OMF_explicit`
#' with `method=als`.
#' }
#' If `X` is a `data.frame`, should be
#' either a `data.frame` or `matrix`, containing row names matching to the second
#' column of `X` (which
#' denotes the item/column IDs of the non-zero entries). If `I` is sparse,
#' `X` should be passed as sparse or dense matrix (not a `data.frame`).
#' 
#' Note that, if `I` is a `matrix` or `data.frame`, it should have the same
#' number of rows as there are columns in `X` in the `ContentBased`, `OMF_explicit`,
#' and `OMF_implicit` models.
#' 
#' Be aware that `CMF` and `CMF_implicit` tend to perform better with dense
#' and not-too-wide user/item attributes.
#' @param U_bin User binary columns/attributes (all values should be zero, one,
#' or missing), for which a sigmoid transformation will be applied on the
#' predicted values. If `X` is a `data.frame`, should also
#' be a `data.frame`, with row names matching to the first column of `X` (which
#' denotes the user/row IDs of the non-zero entries). Cannot be passed
#' as a sparse matrix.
#' Note that `U` and `U_bin` are not mutually exclusive.
#' Only supported with ``method='lbfgs'``.
#' @param I_bin Item binary columns/attributes (all values should be zero, one,
#' or missing), for which a sigmoid transformation will be applied on the
#' predicted values. If `X` is a `data.frame`, should also
#' be a `data.frame`, with row names matching to the second column of `X` (which
#' denotes the item/column IDs of the non-zero entries). Cannot be passed
#' as a sparse matrix.
#' Note that `I` and `I_bin` are not mutually exclusive.
#' Only supported with ``method='lbfgs'``.
#' @param weight (Optional and not recommended) Observation weights for entries in `X`.
#' Must have the same shape as `X` - that is,
#' if `X` is a sparse matrix, must be a vector with the same
#' number of non-zero entries as `X`, if `X` is a dense matrix,
#' `weight` must also be a dense matrix.
#' If `X` is a `data.frame`, should be passed instead as its fourth column.
#' Cannot have missing values.
#' This is only supported for the explicit-feedback models, as the implicit-feedback
#' ones determine the weights through `X`.
#' @param k Number of latent factors to use (dimensionality of the low-rank
#' factorization) - these will be shared between the factorization of the
#' `X` matrix and the side info matrices in the `CMF` and `CMF_implicit` models,
#' and will be determined jointly by interactions and side info
#' in the `OMF_explicit` and `OMF_implicit` models.
#' Additional non-shared components
#' can also be specified through `k_user`, `k_item`, and `k_main`
#' (also `k_sec` for `OMF_explicit`).
#' Typical values are 30 to 100.
#' @param lambda Regularization parameter to apply on the squared L2 norms of the matrices.
#' Some models (`CMF`, `CMF_implicit`, `ContentBased`, and `OMF_explicit` with the L-BFGS method)
#' can use different regularization for each
#' matrix, in which case it should be an array with 6 entries (regardless of the model),
#' corresponding,
#' in this order, to: `user_bias`, `item_bias`, `A`, `B`, `C`, `D`. Note that the default
#' value for `lambda` here is much higher than in other software, and that
#' the loss/objective function is not divided by the number of entries anywhere,
#' so this parameter needs good tuning.
#' For example, a good value for the MovieLens10M would be `lambda=35`
#' (or `lambda=0.05` with `scale_lam=TRUE`), whereas for the
#' LastFM-360K, a good value would be `lambda=5`.
#' Typical values are \eqn{10^{-2}}{0.01} to \eqn{10^2}{100}, with the
#' implicit-feedback models requiring less regularization.
#' @param scale_lam Whether to scale (increase) the regularization parameter
#' for each row of the model matrices (A, B, C, D) according
#' to the number of non-missing entries in the data for that
#' particular row, as proposed in reference [7]. For the
#' A and B matrices, the regularization will only be scaled
#' according to the number of non-missing entries in `X`
#' (see also the `scale_lam_sideinfo` parameter). Note that,
#' when using the options `NA_as_zero_*`, all entries are
#' considered to be non-missing. If passing `TRUE` here, the
#' optimal value for `lambda` will be much smaller
#' (and likely below 0.1).
#' This option tends to give better results, but
#' requires more hyperparameter tuning.
#' Only supported for the ALS method. For the `MostPopular` model,
#' this is not supported when passing `implicit=TRUE`.
#' @param scale_lam_sideinfo Whether to scale (increase) the regularization
#' parameter for each row of the "A" and "B"
#' matrices according to the number of non-missing
#' entries in both `X` and the side info matrices
#' `U` and `I`. If passing `TRUE` here, `scale_lam`
#' will also be assumed to be `TRUE`.
#' @param l1_lambda Regularization parameter to apply to the L1 norm of the model matrices.
#' Can also pass different values for each matrix (see `lambda` for
#' details). Note that, when adding L1 regularization, the model will be
#' fit through a coordinate descent procedure, which is significantly
#' slower than the Cholesky method with L2 regularization.
#' Only supported with the ALS method.
#' Not recommended.
#' @param method Optimization method used to fit the model. If passing `lbfgs`, will
#' fit it through a gradient-based approach using an L-BFGS optimizer, and if
#' passing `als`, will fit it through the ALS (alternating least-squares) method.
#' L-BFGS is typically a much slower and a much less memory efficient method
#' compared to `als`, but tends to reach better local optima and allows
#' some variations of the problem which ALS doesn't, such as applying sigmoid
#' transformations for binary side information.
#' 
#' Note that not all models allow choosing the optimizer: \itemize{
#' \item `CMF_implicit` and `OMF_implicit` can only be fitted through the ALS method.
#' \item `ContentBased` can only be fitted through the L-BFGS method.
#' \item `MostPopular` can only use an ALS-like procedure, but which will ignore
#' parameters such as `niter`.
#' \item Models with non-negativity constraints can only be fitted through the ALS method,
#' and the matrices to which the constraints apply can only be determined through a
#' coordinate descent procedure (which will ignore what is passed to
#' `use_cg` and `finalize_chol`).
#' \item Models with L1 regularization can only be fitted through the ALS method,
#' and the sub-problems are solved through a coordinate-descent procedure.
#' }
#' @param use_cg In the ALS method, whether to use a conjugate gradient method to solve
#' the closed-form least squares problems. This is a faster and more
#' memory-efficient alternative than the default Cholesky solver, but less
#' exact, less numerically stable, and will require slightly more ALS
#' iterations (`niter`) to reach a good optimum.
#' In general, better results are achieved with `use_cg=FALSE` for the
#' explicit-feedback models.
#' Note that, if using this method, calculations after fitting which involve
#' new data such as \link{factors},  might produce slightly different
#' results from the factors obtained inside the fitted model with the same data,
#' due to differences in numerical precision. A workaround for this issue
#' (factors on new data that might differ slightly) is to use
#' `finalize_chol=TRUE`.
#' Even if passing `TRUE` here, will use the Cholesky method in cases in which
#' it is faster (e.g. dense matrices with no missing values),
#' and will not use the conjugate gradient method on new data.
#' This option is not available when using L1 regularization and/or
#' non-negativity constraints.
#' Ignored when using the L-BFGS method.
#' @param add_implicit_features Whether to automatically add so-called implicit features from the data,
#' as in reference [5] and similar. If using this for recommender systems
#' with small amounts of data, it's recommended to pass `TRUE` here.
#' @param user_bias Whether to add user/row biases (intercepts) to the model.
#' If using it for purposes other than recommender systems, this is is
#' usually \bold{not} suggested to include.
#' @param item_bias Whether to add item/column biases (intercepts) to the model. Be aware that using
#' item biases with low regularization for them will tend to favor items
#' with high average ratings regardless of the number of ratings the item
#' has received.
#' @param center Whether to center the "X" data by subtracting the mean value. For recommender
#' systems, it's highly recommended to pass `TRUE` here, the more so if the
#' model has user and/or item biases.
#' @param k_user Number of factors in the factorizing `A` and `C` matrices which will be used
#' only for the `U` and `U_bin` matrices, while being ignored for the `X` matrix.
#' These will be the first factors of the matrices once the model is fit.
#' Will be counted in addition to those already set by `k`.
#' @param k_item Number of factors in the factorizing `B` and `D` matrices which will be used
#' only for the `I` and `I_bin` matrices, while being ignored for the `X` matrix.
#' These will be the first factors of the matrices once the model is fit.
#' Will be counted in addition to those already set by `k`.
#' @param k_main For the `CMF` and `CMF_implicit` models, this denotes the
#' number of factors in the factorizing `A` and `B` matrices which will be used
#' only for the `X` matrix, while being ignored for the `U`, `U_bin`, `I`,
#' and `I_bin` matrices. For the `OMF_explicit` model, this denotes the number of
#' factors which are determined without the user/item side information.
#' These will be the last factors of the matrices once the model is fit.
#' Will be counted in addition to those already set by `k`.
#' @param k_sec (Only for `OMF_explicit`)
#' Number of factors in the factorizing matrices which are determined
#' exclusively from user/item attributes. These will be at the beginning
#' of the `C` and `D` matrices once the model is fit. If there are no attributes
#' for a given matrix (user/item), then that matrix will have an extra
#' `k_sec` factors (e.g. if passing user side info but not item side info,
#' then the `B` matrix will have an extra `k_sec` factors). Will be counted
#' in addition to those already set by `k`. Not supported when
#' using `method='als'`.
#' 
#' For a different model having only `k_sec` with `k=0` and `k_main=0`,
#' see the `ContentBased` model
#' @param w_main Weight in the optimization objective for the errors in the factorization
#' of the `X` matrix.
#' @param w_user For the `CMF` and `CMF_implicit` models, this denotes the
#' weight in the optimization objective for the errors in the factorization
#' of the `U` and `U_bin` matrices. For the `OMF_explicit` model, this denotes
#' the multiplier for the effect of the user attributes in the final factor matrices.
#' Ignored when passing neither `U` nor `U_bin`.
#' @param w_item For the `CMF` and `CMF_implicit` models, this denotes the
#' weight in the optimization objective for the errors in the factorization
#' of the `I` and `I_bin` matrices. For the `OMF_explicit` model, this denotes
#' the multiplier for the effect of the item attributes in the final factor matrices.
#' Ignored when passing neither `I` nor `I_bin`.
#' @param w_implicit Weight in the optimization objective for the errors in the factorizations
#' of the implicit `X` matrices. Note that, depending on the sparsity of the
#' data, the sum of errors from these factorizations might be much larger than
#' for the original `X` and a smaller value will perform better.
#' It is recommended to tune this parameter carefully.
#' Ignored when passing `add_implicit_features=FALSE`.
#' @param niter Number of alternating least-squares iterations to perform. Note that
#' one iteration denotes an update round for all the matrices rather than
#' an update of a single matrix. In general, the more iterations, the better
#' the end result. Ignored when using the L-BFGS method.
#' Typical values are 6 to 30.
#' @param maxiter Maximum L-BFGS iterations to perform. The procedure will halt if it
#' has not converged after this number of updates. Note that the `CMF` model is likely to
#' require fewer iterations to converge compared to other models,
#' whereas the `ContentBased` model, which optimizes a highly non-linear function,
#' will require more iterations and benefits from using
#' more correction pairs. Using higher regularization values might also decrease the number
#' of required iterations. Pass zero for no L-BFGS iterations limit.
#' If the procedure is spending hundreds of iterations
#' without any significant decrease in the loss function or gradient norm,
#' it's highly likely that the regularization is too low.
#' Ignored when using the ALS method.
#' @param finalize_chol When passing `use_cg=TRUE` and using the ALS method, whether to perform the last iteration with
#' the Cholesky solver. This will make it slower, but will avoid the issue
#' of potential mismatches between the resulting factors inside the model object and calls to
#' \link{factors} or similar with the same data.
#' @param max_cg_steps Maximum number of conjugate gradient iterations to perform in an ALS round.
#' Ignored when passing `use_cg=FALSE` or using the L-BFGS method.
#' @param alpha Weighting parameter for the non-zero entries in the implicit-feedback
#' model. See [3] for details. Note that, while the author's suggestion for
#' this value is 40, other software such as the Python package `implicit` use a value of 1,
#' whereas Spark uses a value of 0.01 by default,
#' and values higher than 10 are unlikely to improve results. If the data
#' has very high values, might even be beneficial to put a very low value
#' here - for example, for the LastFM-360K, values below 1 might give better results.
#' @param implicit (Only selectable for the `MostPopular` model)
#' Whether to use the implicit-feedback model, in which the `X` matrix is
#' assumed to have only binary entries and each of them having a weight
#' in the loss function given by the observer user-item interactions and
#' other parameters.
#' @param add_intercepts (Only for `ContentBased`, `OMF_explicit`, `OMF_implicit`)
#' Whether to add intercepts/biases to the user/item attribute matrices.
#' @param start_with_ALS (Only for `ContentBased`)
#' Whether to determine the initial coefficients through an ALS procedure.
#' This might help to speed up the procedure by starting closer to an
#' optimum. This option is not available when the side information is passed
#' as sparse matrices.
#' @param apply_log_transf Whether to apply a logarithm transformation on the values of `X`
#' (i.e. `X := log(X)`)
#' @param NA_as_zero Whether to take missing entries in the `X` matrix as zeros (only
#' when the `X` matrix is passed as a sparse matrix or as a `data.frame`)
#' instead of ignoring them. This is a different model from the
#' implicit-feedback version with weighted entries, and it's a much faster
#' model to fit.
#' Note that passing `TRUE` will affect the results of the functions
#' \link{factors} and \link{factors_single} (as it will assume zeros instead of missing).
#' It is possible to obtain equivalent results to the implicit-feedback
#' model if passing `TRUE` here, and then passing an `X` with
#' all values set to one and weights corresponding to the actual values
#' of `X` multiplied by `alpha`, plus 1 (`W := 1 + alpha*X` to imitate the
#' implicit-feedback model).
#' If passing this option, be aware that the defaults are also to
#' perform mean centering and add user/item biases, which might
#' be undesirable to have together with this option.
#' For the OMF_explicit model, this option will only affect the data
#' to which the model is fit, while being always assumed `FALSE`
#' for new data (e.g. when calling `factors`).
#' @param NA_as_zero_user Whether to take missing entries in the `U` matrix as zeros (only
#' when the `U` matrix is passed as a sparse matrix) instead of ignoring them.
#' Note that passing `TRUE` will affect the results of the functions
#' \link{factors} and \link{factors_single} if no data is passed there (as it will assume zeros instead of
#' missing). This option is always assumed `TRUE` for the `ContentBased`, `OMF_explicit`,
#' and `OMF_implicit` models.
#' @param NA_as_zero_item Whether to take missing entries in the `I` matrix as zeros (only
#' when the `I` matrix is passed as a sparse matrix) instead of ignoring them.
#' This option is always assumed `TRUE` for the `ContentBased`, `OMF_explicit`,
#' and `OMF_implicit` models.
#' @param nonneg Whether to constrain the `A` and `B` matrices to be non-negative.
#' In order for this to work correctly, the `X` input data must also be
#' non-negative. This constraint will also be applied to the `Ai`
#' and `Bi` matrices if passing `add_implicit_features=TRUE`.
#' 
#' \bold{Important:} be aware that the default options are to perform mean
#' centering and to add user and item biases, which might be undesirable and
#' hinder performance when having non-negativity constraints (especially mean centering).
#' 
#' This option is not available when using the L-BFGS method.
#' Note that, when determining non-negative factors, it will always
#' use a coordinate descent method, regardless of the value passed
#' for `use_cg` and `finalize_chol`.
#' When used for recommender systems, one usually wants to pass `FALSE` here.
#' For better results, do not use centering alongside this option,
#' and use a higher regularization coupled with more iterations..
#' @param nonneg_C Whether to constrain the `C` matrix to be non-negative.
#' In order for this to work correctly, the `U` input data must also be
#' non-negative.
#' @param nonneg_D Whether to constrain the `D` matrix to be non-negative.
#' In order for this to work correctly, the `I` input data must also be
#' non-negative.
#' @param max_cd_steps Maximum number of coordinate descent updates to perform per iteration.
#' Pass zero for no limit.
#' The procedure will only use coordinate descent updates when having
#' L1 regularization and/or non-negativity constraints.
#' This number should usually be larger than `k`.
#' @param precompute_for_predictions Whether to precompute some of the matrices that are used when making
#' predictions from the model. If `FALSE`, it will take longer to generate
#' predictions or top-N lists, but will use less memory and will be faster
#' to fit the model. If passing `FALSE`, can be recomputed later on-demand
#' through function \link{precompute.for.predictions}.
#' 
#' Note that for `ContentBased`, `OMF_explicit`, and `OMF_implicit`, this parameter
#' will always be assumed to be `TRUE`, due to requiring the original matrices
#' for the pre-computations.
#' @param include_all_X When passing an input `X` which has less columns than rows in
#' `I`, whether to still make calculations about the items which are in `I`
#' but not in `X`. This has three effects: (a) the \link{topN} functionality may
#' recommend such items, (b) the precomptued matrices will be less usable as
#' they will include all such items, (c) it will be possible to pass `X` data
#' to the new factors or topN functions that include such columns (rows of `I`).
#' This option is ignored when using `NA_as_zero`, and is only relevant for the `CMF`
#' model as all the other models will have the equivalent of `TRUE` here.
#' @param verbose Whether to print informational messages about the optimization
#' routine used to fit the model.
#' Be aware that, if passing `FALSE` and
#' using the L-BFGS method, the optimization routine will not respond to
#' interrupt signals.
#' @param print_every Print L-BFGS convergence messages every n-iterations. Ignored
#' when not using the L-BFGS method.
#' @param parallelize How to parallelize gradient calculations when using more than one
#' thread with `method='lbfgs'`. Passing `separate` will iterate
#' over the data twice - first
#' by rows and then by columns, letting each thread calculate results
#' for each row and column, whereas passing `single` will iterate over
#' the data only once, and then sum the obtained results from each thread.
#' Passing `separate` is much more memory-efficient and less prone to
#' irreproducibility of random seeds, but might be slower for typical
#' use-cases. Ignored when passing `nthreads=1`, or when using the ALS method,
#' or when compiling without OpenMP support.
#' @param corr_pairs Number of correction pairs to use for the L-BFGS optimization routine.
#' Recommended values are between 3 and 7. Note that higher values
#' translate into higher memory requirements. Ignored when 
#' using the ALS method.
#' @param handle_interrupt Whether to respond to interrupt signals in the optimization procedure.
#' If passing `TRUE`, whenever it receives an interrupt signal during the
#' optimzation procedure, it will termnate earlier, taking the current values
#' of the variables without finishing, instead of raising an error.
#' If passing `FALSE`, will raise an error when it is interrupted, which
#' will only be catched after the procedure is finished, and the obtained
#' object will not be usable. Note that, for models fitted with the L-BFGS method, will
#' only respond to interrupt signals when using `verbose=TRUE`.
#' @param nthreads Number of parallel threads to use. Note that, the more threads that
#' are used, the higher the memory consumption.
#' @return Returns a model object (class named just like the function that produced it,
#' plus general class `cmfrec`) on which methods such as \link{topN} and \link{factors} can be called.
#' The returned object will have the following fields:\itemize{
#' \item `info`: will contain the hyperparameters, problem dimensions, and other information
#' such as the number of threads, as passed to the function that produced the model.
#' The number of threads (`nthreads`) might be modified after-the-fact.
#' If `X` is a `data.frame`, will also contain the re-indexing of users and items under
#' `user_mapping` and `item_mapping`, respectively. For the L-BFGS method, will also
#' contain the number of function evaluations (`nfev`) and number of updates (`nupd`)
#' that were performed.
#' \item `matrices`: will contain the fitted model matrices (see section `Description`
#' for the naming and for details on what they represent),
#' but note that they will be transposed (due to R's column-major
#' representation of matrices) and it is recommended to use the package's prediction
#' functionality instead of taking the matrices directly.
#' \item `precomputed`: will contain some pre-computed calculations based on the model
#' matrices which might help speed up predictions on new data.
#' }
#' @references \itemize{
#' \item Cortes, David. "Cold-start recommendations in Collective Matrix Factorization." arXiv preprint arXiv:1809.00366 (2018).
#' \item Singh, Ajit P., and Geoffrey J. Gordon. "Relational learning via collective matrix factorization." Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. 2008.
#' \item Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative filtering for implicit feedback datasets." 2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008.
#' \item Takacs, Gabor, Istvan Pilaszy, and Domonkos Tikk. "Applications of the conjugate gradient method for implicit feedback collaborative filtering." Proceedings of the fifth ACM conference on Recommender systems. 2011.
#' \item Rendle, Steffen, Li Zhang, and Yehuda Koren. "On the difficulty of evaluating baselines: A study on recommender systems." arXiv preprint arXiv:1905.01395 (2019).
#' \item Franc, Vojtech, Vaclav Hlavac, and Mirko Navara. "Sequential coordinate-wise algorithm for the non-negative least squares problem." International Conference on Computer Analysis of Images and Patterns. Springer, Berlin, Heidelberg, 2005.
#' \item Zhou, Yunhong, et al. "Large-scale parallel collaborative filtering for the netflix prize." International conference on algorithmic applications in management. Springer, Berlin, Heidelberg, 2008.
#' }
#' @details In more details, the models predict the values of `X` as follows:\itemize{
#' \item `CMF`: \eqn{
#' \mathbf{X} \approx \mathbf{A} \mathbf{B}^T + \mu + \mathbf{b}_u + \mathbf{b}_i }{
#' X ~ A * t(B) + \mu + bias_u + bias_i}, where \eqn{\mu} is the global mean for the non-missing
#' entries in `X`, and \eqn{\mathbf{b}_u , \mathbf{b}_i}{bias_u, bias_i} are the user and
#' item biases (column and row vector, respectively). In addition, the other matrices are
#' predicted as \eqn{\mathbf{U} \approx \mathbf{A} \mathbf{C}^T + \mu_U}{U ~ A*t(C) + \mu_U}
#' and \eqn{\mathbf{I} \approx \mathbf{B} \mathbf{D}^T + \mu_I}{I ~ B*t(D) + \mu_I}, where
#' \eqn{\mu_U , \mu_I}{\mu_U, \mu_I} are the column means from the side info matrices,
#' which are determined as a simple average with no regularization (these are row
#' vectors), and if having binary variables, also \eqn{
#' \mathbf{U}_{bin} \approx \sigma(\mathbf{A} \mathbf{C}_{bin}^T)}{
#' U_bin ~ sigm(A*t(C_bin))} and \eqn{
#' \mathbf{I}_{bin} \approx \sigma(\mathbf{B} \mathbf{D}_{bin}^T)}{
#' I_bin ~ sigm(B*t(D_bin))}, where \eqn{\sigma}{'sigm'} is a sigmoid function (\eqn{
#' \sigma(x) = \frac{1}{1 + e^{-x}}}{
#' sigm(x) = 1/(1+exp(-x))}). Under the options `NA_as_zero_*`,
#' the mean(s) for that matrix are not added into the model for simplicity.
#' For the implicit features option, the other matrices are predicted simply as
#' \eqn{\mathbf{I}_x \approx \mathbf{A} \mathbf{B}_i , \:\:
#' \mathbf{I}_x^T \approx \mathbf{B} \mathbf{A}_i}{Ix ~ A*t(Bi), t(Ix) ~ B*t(Ai)}.
#' 
#' If using `k_user`, `k_item`, `k_main`, then for `X`, only columns `1` through
#' `k+k_user` are used in the approximation of `U`, and only columns `k_user+1` through
#' `k_user+k+k_main` are used for the approximation of `X` (similar thing for `B` with
#' `k_item`). The implicit factors matrices (\eqn{\mathbf{A}_i, \mathbf{B}_i}{Ai, Bi})
#' always use the same components/factors as `X`.
#' 
#' Be aware that the functions for determining new factors will by default omit
#' the bias term in the output.
#' \item `CMF_implicit`: \eqn{\mathbf{X} \approx \mathbf{A} \mathbf{B}^T}{X ~ A * t(B)},
#' while `U` and `I` remain the same as for `CMF`, and the ordering of the non-shared
#' factors is the same. Note that there is no mean centering or user/item biases in the
#' implicit-feedback model, but if desired, the `CMF` model can be made to mimic
#' `CMF_implicit` while still accommodating for mean centering and biases.
#' \item `MostPopular`: \eqn{\mathbf{X} \approx \mu + \mathbf{b}_u + \mathbf{b}_i}{
#' X ~ \mu + bias_u + bias_i} (when using `implicit=FALSE`) or \eqn{
#' \mathbf{X} \approx \mathbf{b}_i}{X ~ bias_i} (when using `implicit=TRUE`).
#' \item `ContentBased`: \eqn{\mathbf{X} \approx \mathbf{A}_m \mathbf{B}_m^T}{
#' X ~ Am * t(Bm)}, where \eqn{\mathbf{A}_m = \mathbf{U} \mathbf{C} + \mathbf{b}_C}{
#' Am = U*C + C_bias} and \eqn{\mathbf{B}_m = \mathbf{I} \mathbf{D} + \mathbf{b}_D}{
#' Bm = I * D + D_bias} - the \eqn{\mathbf{b}_C, \mathbf{b}_D}{C_bias, D_bias} are
#' per-column/factor intercepts (these are row vectors).
#' \item `OMF_explicit`: \eqn{
#' \approx \mathbf{A}_m \mathbf{B}_m^T + \mu + \mathbf{b}_u + \mathbf{b}_i }{
#' X ~ Am * t(Bm) + \mu + bias_u + bias_i}, where
#' \eqn{\mathbf{A}_m = w_u (\mathbf{U} \mathbf{C} + \mathbf{b}_C) + \mathbf{A}}{
#' Am = w_user * U * C + C_bias + A} and \eqn{
#' \mathbf{B}_m = w_i (\mathbf{I} \mathbf{D} + \mathbf{b}_D) + \mathbf{B}}{
#' Bm = w_item * (I * D + D_bias) + B}. If passing `k_sec` and/or `k_main`, then columns
#' `1` through `k_sec` of \eqn{\mathbf{A}_m, \mathbf{B}_m}{Am, Bm} are determined
#' as those same columns from \eqn{\mathbf{A}, \mathbf{B}}{A, B}, while
#' \eqn{\mathbf{U} \mathbf{C} + \mathbf{b}_C, \mathbf{I} \mathbf{D} + \mathbf{b}_D}{
#' U*C + C_bias, I*D + D_bias} will be shorter by `k_sec`
#' columns (alternatively, can be though of as having those columns artificially set to
#' zeros), and columns `k_sec+k+1` through `k_sec+k+k_main` of
#' \eqn{\mathbf{A}_m, \mathbf{B}_m}{Am, Bm}
#' are determined as those last `k_main` columns of
#' \eqn{\mathbf{U} \mathbf{C} + \mathbf{b}_C, \mathbf{I} \mathbf{D} + \mathbf{b}_D}{
#' U*C + C_bias, I*D + D_bias},
#' while \eqn{\mathbf{A}, \mathbf{B}}{A, B} will be shorter by `k_main` columns
#' (alternatively, can be though of as having those columns artificially set to
#' zeros). If one of \eqn{\mathbf{U}}{U} or \eqn{\mathbf{I}}{I} is missing,
#' then the corresponding \eqn{\mathbf{A}}{A} or \eqn{\mathbf{B}}{B} matrix will
#' be extended by `k_sec` columns (which will not be zeros) and the corresponding
#' prediction matrix (\eqn{\mathbf{A}_m, \mathbf{B}_m}{Am, Bm}) will be equivalent
#' to that matrix (which was the free offset in the presence of side information).
#' \item `OMF_implicit`: \eqn{\mathbf{X} \approx \mathbf{A}_m \mathbf{B}_m^T}{X ~ Am * t(Bm)},
#' with \eqn{\mathbf{A}_m, \mathbf{B}_m}{Am, Bm} remaining the same as for `OMF_explicit`.
#' }
#' 
#' When calling the prediction functions, new data is always transposed or deep copied
#' before passing them to the underlying C functions - as such, for the `ContentBased` model,
#' it might be faster to use the matrices directly instead (all these matrices will
#' be under `model$matrices`, but will be transposed).
#' 
#' The precomputed matrices, when they are square, will only contain the lower triangle
#' only, as they are symmetric. For `CMF` and `CMF_implicit`, one might also see
#' variations of a new matrix called `Be` (extended `B` matrix), which is from
#' reference [1] and defined as \eqn{
#' \mathbf{B}_e = [[\mathbf{0}, \mathbf{B}_s, \mathbf{B}_m], [\mathbf{C}_a, \mathbf{C}_s, \mathbf{0}]]
#' }{
#' Be = [[0, Bs, Bm], [Ca, Cs, 0]]
#' }, where \eqn{\mathbf{B}_s}{Bs} are columns `k_item+1` through `k_item+k` from `B`,
#' \eqn{\mathbf{B}_m}{Bm} are columns `k_item+k+1` through `k_item+k+k_main` from `B`,
#' \eqn{\mathbf{C}_a}{Ca} are columns `1` through `k_user` from `C`, and
#' \eqn{\mathbf{C}_s}{Cs} are columns `k_user+1` through `k_user+k` from `C`.
#' This matrix is used for the closed-form solution of a given vector of `A` in
#' the functions for predicting on new data (see reference [1] for details or if
#' you would like to use your own solver with the fitted matrices from this package),
#' as long as there are no binary columns to which to apply a transformation, in
#' which case it will always solve them with the L-BFGS method.
#' 
#' When using user biases, the precomputed matrices will have an extra column, which is
#' derived by adding an extra column to `B` (at the end) consisting of all ones
#' (this is how the user biases are calculated).
#' 
#' For the implicit-feedback models, the weights of the positive entries (defined
#' as the non-missing entries in `X`) will be given by
#' \eqn{W = 1 + \alpha \mathbf{X}}{W = 1 + \alpha * X}.
#' 
#' For the `OMF` models, the `ALS` method will first find a solution for the
#' equivalent `CMF` problem with no side information, and will then try to predict
#' the resulting matrices given the user/item attributes, assigning the residuals
#' as the free offsets. While this might sound reasonable, in practice it tends to
#' give rather different results than when fit through the L-BFGS method. Strictly
#' speaking, the regularization parameter in this case is applied to the
#' \eqn{\mathbf{A}_m, \mathbf{B}_m}{Am, Bm} matrices, and the prediction functions
#' for new data will offer an option `exact` for determining whether to apply the
#' regularization to the \eqn{\mathbf{A}, \mathbf{B}}{A, B} matrices instead.
#' 
#' Be aware that the optimization procedures rely heavily on BLAS and LAPACK function
#' calls, and as such benefit from using optimized libraries for them such as
#' MKL or OpenBLAS.
#' 
#' For reproducibility, the initializations of the model matrices (always initialized
#' as `~ Normal(0, 1)`) can be controlled
#' through `set.seed`, but if using parallelizations, there are potential sources
#' of irreproducibility of random seeds due to parallelized aggregations and/or
#' BLAS function calls, which is especially problematic for the L-BFGS method
#' with `parallelize='single'`.
#' 
#' In order to further avoid potential decimal differences in the factors obtained
#' when fitting the model and when calling the prediction functions on
#' new data, when the data is sparse, it's necessary to sort it beforehand
#' by columns/items and also pass the data data with item indices sorted beforehand
#' to the prediction functions. The package does not perform any indices sorting or
#' de-duplication of entries of sparse matrices.
#' @examples
#' library(cmfrec)
#' if (require("recommenderlab") && require("Matrix") && require("rsparse")) {
#'     ### Load the ML100K dataset (movie ratings)
#'     ### (users are rows, items are columns)
#'     data("MovieLense")
#'     X <- as(MovieLense@data, "dgTMatrix")
#'     
#'     ### Will additionally use the item genres as side info
#'     I <- MovieLenseMeta
#'     I$title <- NULL
#'     I$year  <- NULL
#'     I$url   <- NULL
#'     I <- as(as.matrix(I), "TsparseMatrix")
#'     
#'     ### Fit a factorization model
#'     ### (it's recommended to change the hyperparameters
#'     ###  and use multiple threads)
#'     set.seed(1)
#'     model <- CMF(X=X, I=I, k=10L, niter=5L,
#'                  NA_as_zero_item=TRUE,
#'                  verbose=FALSE, nthreads=1L)
#'     
#'     ### Predict rating for entries X[1,3], X[2,5], X[10,9]
#'     ### (first ID is the user, second is the movie)
#'     predict(model, user=c(1,2,10), item=c(3,5,9))
#'     
#'     ### Recommend top-5 for user ID = 10
#'     ### (Note that "Matrix" objects start their numeration at 0)
#'     seen_by_user <- MovieLense@data[10, , drop=FALSE]
#'     seen_by_user <- seen_by_user@i + 1L
#'     rec <- topN(model, user=10, n=5, exclude=seen_by_user)
#'     rec
#'     
#'     ### Print them in a more understandable format
#'     movie_names <- colnames(X)
#'     n_ratings <- colSums(as(MovieLense@data[, rec, drop=FALSE], "ngCMatrix"))
#'     avg_ratings <- colSums(MovieLense@data[, rec, drop=FALSE]) / n_ratings
#'     cat("Recommended for user_id=10:\n",
#'         paste(paste(1:length(rec), ". ", sep=""),
#'               movie_names[rec],
#'               " - Avg rating:", round(avg_ratings, 2),
#'               ", #ratings: ", n_ratings,
#'               collapse="\n", sep=""),
#'          "\n", sep="")
#'     
#'     
#'     ### Recommend assuming it is a new user,
#'     ### based on its data (ratings + side info)
#'     x_user <- as(X[10, , drop=FALSE], "sparseVector")
#'     rec_new <- topN_new(model, n=5, X=x_user, exclude=seen_by_user)
#'     cat("lists are identical: ", identical(rec_new, rec), "\n")
#'     
#'     ### (If there were user side info, could also recommend
#'     ###  based on that side information alone)
#'     
#'     ### Obtain factors for the user
#'     factors_user <- model$matrices$A[, 10, drop=TRUE]
#'     
#'     ### Re-calculate them based on the data
#'     factors_new <- factors_single(model, X=x_user)
#'     
#'     ### Should be very close, but due to numerical precision,
#'     ### might not be exactly equal (see section 'Details')
#'     cat("diff: ", factors_user - factors_new, "\n")
#'     
#'     ### Can also calculate them in batch
#'     ### (slicing is provided by package "rsparse")
#'     Xslice <- as(X, "RsparseMatrix")[1:10, , drop=FALSE]
#'     factors_multiple <- factors(model, X=Xslice)
#'     cat("diff: ", factors_multiple[10, , drop=TRUE] - factors_new, "\n")
#'     
#'     ### Can make cold-start predictions, e.g.
#'     ### predict how would users [1,2,3] rate a new item,
#'     ### given it's side information (here it's item ID = 5)
#'     predict_new_items(model, user=c(1,2,3), item=c(1,1,1), I=I[5, ])
#' }
NULL

validate.inputs <- function(model, implicit=FALSE,
                            X=NULL, U=NULL, I=NULL, U_bin=NULL, I_bin=NULL, weight=NULL,
                            k=40L, lambda=10., method="als", use_cg=TRUE,
                            user_bias=TRUE, item_bias=TRUE, center=FALSE,
                            k_user=0L, k_item=0L, k_main=0L, k_sec=0L,
                            w_main=1., w_user=1., w_item=1., w_implicit=0.5,
                            l1_lambda=0.,
                            alpha=1., downweight=FALSE,
                            add_implicit_features=FALSE,
                            scale_lam=FALSE, scale_lam_sideinfo=FALSE,
                            add_intercepts=TRUE,
                            start_with_ALS=FALSE,
                            apply_log_transf=FALSE,
                            nonneg=FALSE, nonneg_C=FALSE, nonneg_D=FALSE,
                            max_cd_steps=100L,
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
    max_cd_steps  <-  check.pos.int(max_cd_steps, "max_cd_steps", FALSE)
    print_every   <-  check.pos.int(print_every, "print_every", TRUE)
    nthreads      <-  check.pos.int(nthreads, "nthreads", TRUE)
    
    use_cg     <-  check.bool(use_cg, "use_cg")
    user_bias  <-  check.bool(user_bias, "user_bias")
    item_bias  <-  check.bool(item_bias, "item_bias")
    center     <-  check.bool(center, "center")
    finalize_chol    <-  check.bool(finalize_chol, "finalize_chol")
    NA_as_zero       <-  check.bool(NA_as_zero, "NA_as_zero")
    NA_as_zero_user  <-  check.bool(NA_as_zero_user, "NA_as_zero_user")
    NA_as_zero_item  <-  check.bool(NA_as_zero_item, "NA_as_zero_item")
    nonneg           <-  check.bool(nonneg, "nonneg")
    nonneg_C         <-  check.bool(nonneg_C, "nonneg_C")
    nonneg_D         <-  check.bool(nonneg_D, "nonneg_D")
    include_all_X    <-  check.bool(include_all_X, "include_all_X")
    verbose          <-  check.bool(verbose, "verbose")
    handle_interrupt <-  check.bool(handle_interrupt, "handle_interrupt")
    downweight       <-  check.bool(downweight, "downweight")
    implicit         <-  check.bool(implicit, "implicit")
    add_intercepts   <-  check.bool(add_intercepts, "add_intercepts")
    start_with_ALS   <-  check.bool(start_with_ALS, "start_with_ALS")
    apply_log_transf <-  check.bool(apply_log_transf, "apply_log_transf")
    precompute_for_predictions  <-  check.bool(precompute_for_predictions, "precompute_for_predictions")
    add_implicit_features       <-  check.bool(add_implicit_features, "add_implicit_features")
    scale_lam                   <-  check.bool(scale_lam, "scale_lam")
    scale_lam_sideinfo          <-  check.bool(scale_lam_sideinfo, "scale_lam_sideinfo")
    
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
    l1_lambda <- check.lambda(l1_lambda, allow_different_lambda)
    scale_lam <- scale_lam || scale_lam_sideinfo 
    
    if (k_user > 0 && is.null(U) && is.null(U_bin))
        stop("Cannot pass 'k_user' with no 'U' data.")
    if (k_item > 0 && is.null(I) && is.null(I_bin))
        stop("Cannot pass 'k_item' with no 'I' data.")
    if (method == "als" && (!is.null(U_bin) || !is.null(I_bin)))
        stop("Cannot use 'method=als' when there is 'U_bin' or 'I_bin'.")
    if (implicit && user_bias)
        stop("Cannot fit user biases with 'implicit=TRUE'.")
    if (implicit && scale_lam)
        stop("'scale_lam' not supported for implicit-feedback.")
    if ((k_user+k+k_main+1)^2 > .Machine$integer.max)
        stop("Number of factors is too large.")
    if ((method == "lbfgs") && (NA_as_zero || NA_as_zero_user || NA_as_zero_item))
        stop("Option 'NA_as_zero' not supported with 'method=\"lbfgs\"'.")
    if ((method == "lbfgs") && add_implicit_features)
        stop("Option 'add_implicit_features' not supported with 'method=\"lbfgs\"'.")
    if ((method == "lbfgs") && (scale_lam || scale_lam_sideinfo))
        stop("Option 'scale_lam' not supported with 'method=\"lbfgs\"'.")
    if ((method == "lbfgs") && (nonneg || nonneg_C || nonneg_D))
        stop("Non-negativity constraints not supported with 'method=\"lbfgs\".")
    if ((method == "lbfgs") && (l1_lambda > 0.))
        stop("L1 regularization not supported with 'method=\"lbfgs\".")
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
        if (start_with_ALS) {
            if (!NROW(processed_U$Uarr) || !NROW(processed_I$Uarr)) {
                warning("Option 'start_with_ALS' not available for sparse data.")
                start_with_ALS <- FALSE
            }
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
    
    if (apply_log_transf) {
        if (model == "MostPopular" && !implicit)
            stop("Option 'apply_log_transf' only available with 'implicit=TRUE'.")
        msg_small <- "Cannot pass values below 1 with 'apply_log_transf=TRUE'."
        if (NROW(processed_X$Xarr)) {
            if (min(processed_X$Xarr, na.rm=TRUE) < 1)
                stop(msg_small)
        } else if (NROW(processed_X$Xval)) {
            if (min(processed_X$Xval) < 1)
                stop(msg_small)
        }
    }
    
    if (nonneg && center)
        warning("Warning: fitting a model with centering and non-negativity constraints.")
    if (NA_as_zero && add_implicit_features)
        warning("Warning: will add implicit features while having 'NA_as_zero'.")
    
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
        l1_lambda = l1_lambda,
        alpha = alpha, downweight = downweight,
        implicit = implicit,
        add_implicit_features = add_implicit_features,
        scale_lam = scale_lam, scale_lam_sideinfo = scale_lam_sideinfo,
        add_intercepts = add_intercepts,
        start_with_ALS = start_with_ALS,
        apply_log_transf = apply_log_transf,
        maxiter = maxiter, niter = niter, parallelize = parallelize, corr_pairs = corr_pairs,
        max_cg_steps = max_cg_steps, finalize_chol = finalize_chol,
        NA_as_zero = NA_as_zero, NA_as_zero_user = NA_as_zero_user, NA_as_zero_item = NA_as_zero_item,
        nonneg = nonneg, nonneg_C = nonneg_C, nonneg_D = nonneg_D,
        max_cd_steps = max_cd_steps,
        precompute_for_predictions = precompute_for_predictions,
        include_all_X = include_all_X, center = center,
        verbose = verbose, print_every = print_every,
        handle_interrupt = handle_interrupt,
        nthreads = nthreads
    ))
}

#' @export
#' @rdname fit
CMF <- function(X, U=NULL, I=NULL, U_bin=NULL, I_bin=NULL, weight=NULL,
                k=40L, lambda=10., method="als", use_cg=TRUE,
                user_bias=TRUE, item_bias=TRUE, center=TRUE, add_implicit_features=FALSE,
                scale_lam=FALSE, scale_lam_sideinfo=FALSE,
                k_user=0L, k_item=0L, k_main=0L,
                w_main=1., w_user=1., w_item=1., w_implicit=0.5,
                l1_lambda=0.,
                maxiter=800L, niter=10L, parallelize="separate", corr_pairs=4L,
                max_cg_steps=3L, finalize_chol=TRUE,
                NA_as_zero=FALSE, NA_as_zero_user=FALSE, NA_as_zero_item=FALSE,
                nonneg=FALSE, nonneg_C=FALSE, nonneg_D=FALSE, max_cd_steps=100L,
                precompute_for_predictions=TRUE, include_all_X=TRUE,
                verbose=TRUE, print_every=10L,
                handle_interrupt=TRUE,
                nthreads=parallel::detectCores()) {
    
    inputs <- validate.inputs(model = "CMF",
                              X = X, U = U, I = I, U_bin = U_bin, I_bin = I_bin, weight = weight,
                              k = k, lambda = lambda, method = method, use_cg = use_cg,
                              user_bias = user_bias, item_bias = item_bias, center = center,
                              add_implicit_features = add_implicit_features,
                              scale_lam = scale_lam, scale_lam_sideinfo = scale_lam_sideinfo,
                              k_user = k_user, k_item = k_item, k_main = k_main,
                              w_main = w_main, w_user = w_user, w_item = w_item,
                              w_implicit = w_implicit,
                              l1_lambda = l1_lambda,
                              maxiter = maxiter, niter = niter,
                              parallelize = parallelize, corr_pairs = corr_pairs,
                              max_cg_steps = max_cg_steps, finalize_chol = finalize_chol,
                              NA_as_zero = NA_as_zero,
                              NA_as_zero_user = NA_as_zero_user,
                              NA_as_zero_item = NA_as_zero_item,
                              nonneg = nonneg, nonneg_C = nonneg_C, nonneg_D = nonneg_D,
                              max_cd_steps = max_cd_steps,
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
                user_bias = inputs$user_bias, item_bias = inputs$item_bias, center = inputs$center,
                add_implicit_features = inputs$add_implicit_features,
                scale_lam = inputs$scale_lam, scale_lam_sideinfo = inputs$scale_lam_sideinfo,
                k_user = inputs$k_user, k_item = inputs$k_item, k_main = inputs$k_main,
                w_main = inputs$w_main, w_user = inputs$w_user, w_item = inputs$w_item,
                w_implicit = inputs$w_implicit,
                l1_lambda = inputs$l1_lambda,
                maxiter = inputs$maxiter, niter = inputs$niter,
                parallelize = inputs$parallelize, corr_pairs = inputs$corr_pairs,
                max_cg_steps = inputs$max_cg_steps, finalize_chol = inputs$finalize_chol,
                NA_as_zero = inputs$NA_as_zero,
                NA_as_zero_user = inputs$NA_as_zero_user,
                NA_as_zero_item = inputs$NA_as_zero_item,
                nonneg = inputs$nonneg, nonneg_C = inputs$nonneg_C, nonneg_D = inputs$nonneg_D,
                max_cd_steps = inputs$max_cd_steps,
                precompute_for_predictions = inputs$precompute_for_predictions,
                include_all_X = inputs$include_all_X,
                verbose = inputs$verbose, print_every = inputs$print_every,
                handle_interrupt = inputs$handle_interrupt,
                nthreads = inputs$nthreads))
    
}

#' @export
#' @rdname fit
CMF_implicit <- function(X, U=NULL, I=NULL,
                         k=40L, lambda=1., alpha=1., use_cg=TRUE,
                         k_user=0L, k_item=0L, k_main=0L,
                         w_main=1., w_user=1., w_item=1.,
                         l1_lambda=0.,
                         niter=10L,
                         max_cg_steps=3L, finalize_chol=FALSE,
                         NA_as_zero_user=FALSE, NA_as_zero_item=FALSE,
                         nonneg=FALSE, nonneg_C=FALSE, nonneg_D=FALSE, max_cd_steps=100L,
                         apply_log_transf=FALSE,
                         precompute_for_predictions=TRUE,
                         verbose=TRUE,
                         handle_interrupt=TRUE,
                         nthreads=parallel::detectCores()) {
    
    inputs <- validate.inputs(model = "CMF_implicit",
                              X = X, U = U, I = I,
                              k = k, lambda = lambda, use_cg = use_cg,
                              alpha = alpha, downweight = FALSE,
                              k_user = k_user, k_item = k_item, k_main = k_main,
                              w_main = w_main, w_user = w_user, w_item = w_item,
                              l1_lambda = l1_lambda,
                              niter = niter,
                              max_cg_steps = max_cg_steps, finalize_chol = finalize_chol,
                              NA_as_zero_user = NA_as_zero_user,
                              NA_as_zero_item = NA_as_zero_item,
                              nonneg = nonneg, nonneg_C = nonneg_C, nonneg_D = nonneg_D,
                              max_cd_steps = max_cd_steps,
                              apply_log_transf = apply_log_transf,
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
                         l1_lambda = inputs$l1_lambda,
                         niter = inputs$niter,
                         max_cg_steps = inputs$max_cg_steps, finalize_chol = inputs$finalize_chol,
                         NA_as_zero_user = inputs$NA_as_zero_user,
                         NA_as_zero_item = inputs$NA_as_zero_item,
                         nonneg = inputs$nonneg, nonneg_C = inputs$nonneg_C, nonneg_D = inputs$nonneg_D,
                         max_cd_steps = inputs$max_cd_steps,
                         apply_log_transf = inputs$apply_log_transf,
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
                 user_bias=TRUE, item_bias=TRUE, center=TRUE,
                 add_implicit_features=FALSE,
                 scale_lam=FALSE, scale_lam_sideinfo=FALSE,
                 k_user=0L, k_item=0L, k_main=0L,
                 w_main=1., w_user=1., w_item=1., w_implicit=0.5,
                 l1_lambda=0.,
                 maxiter=800L, niter=10L, parallelize="separate", corr_pairs=4L,
                 max_cg_steps=3L, finalize_chol=TRUE,
                 NA_as_zero=FALSE, NA_as_zero_user=FALSE, NA_as_zero_item=FALSE,
                 nonneg=FALSE, nonneg_C=FALSE, nonneg_D=FALSE, max_cd_steps=100L,
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
    this$info$l1_lambda   <-  l1_lambda
    this$info$user_mapping     <-  user_mapping
    this$info$item_mapping     <-  item_mapping
    this$info$U_cols           <-  U_cols
    this$info$I_cols           <-  I_cols
    this$info$U_bin_cols       <-  U_bin_cols
    this$info$I_bin_cols       <-  I_bin_cols
    this$info$NA_as_zero       <-  NA_as_zero
    this$info$NA_as_zero_user  <-  NA_as_zero_user
    this$info$NA_as_zero_item  <-  NA_as_zero_item
    this$info$nonneg           <-  nonneg
    this$info$include_all_X    <-  include_all_X
    this$info$center           <-  center
    this$info$nthreads         <-  nthreads
    this$info$add_implicit_features  <-  add_implicit_features
    this$info$scale_lam              <-  scale_lam
    this$info$scale_lam_sideinfo     <-  scale_lam_sideinfo
    
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
        if (!add_implicit_features && !nonneg)
            this$precomputed$TransBtBinvBt <- matrix(0., ncol=n_max, nrow=k+k_main+user_bias)
        if (add_implicit_features)
            this$precompted$BiTBi <- matrix(0., ncol=k+k_main, nrow=k+k_main)
        if (processed_U$p) {
            this$precomputed$CtC <- matrix(0., ncol=k_user+k, nrow=k_user+k)
            if (!add_implicit_features && !nonneg)
                this$precomputed$TransCtCinvCt <- matrix(0., ncol=processed_U$p, nrow=k_user+k)
        }
        if ((add_implicit_features || processed_U$p) && !nonneg) {
            this$precomputed$BeTBeChol <- matrix(0., nrow=k_user+k+k_main+user_bias,
                                                 ncol=k_user+k+k_main+user_bias)
        }
        if (NA_as_zero && (center || item_bias))
            this$precomputed$BtXbias <- numeric(k+k_main+user_bias)
    }
    
    ### Note: for some reason, R keeps pointers of old objects when calling 'get.empty.info',
    ### this avoids potentially modifying other objects in the environment
    glob_mean <- numeric(1L)
    nupd <- integer(1L)
    nfev <- integer(1L)
    
    if (method == "als") {
        ### Note: R's '.Call' has a limit of 65 arguments - this one exceeds it
        ### so some parameters have to be merged.
        nonneg_CD <- as.logical(c(nonneg_C, nonneg_D))
        k_main_k_user_k_item <- as.integer(c(k_main, k_user, k_item))
        w_main_w_user_w_item_w_implicit <- as.numeric(c(w_main, w_user, w_item, w_implicit))
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
                          user_bias, item_bias, center,
                          lambda, l1_lambda,
                          scale_lam, scale_lam_sideinfo,
                          processed_U$Uarr, processed_U$m, processed_U$p,
                          processed_I$Uarr, processed_I$m, processed_I$p,
                          processed_U$Urow, processed_U$Ucol, processed_U$Uval,
                          processed_I$Urow, processed_I$Ucol, processed_I$Uval,
                          NA_as_zero, NA_as_zero_user, NA_as_zero_item,
                          k_main_k_user_k_item,
                          w_main_w_user_w_item_w_implicit,
                          niter, nthreads, verbose, handle_interrupt,
                          use_cg, max_cg_steps, finalize_chol,
                          nonneg, max_cd_steps, nonneg_CD,
                          precompute_for_predictions,
                          add_implicit_features,
                          include_all_X,
                          this$precomputed$B_plus_bias,
                          this$precomputed$BtB,
                          this$precomputed$TransBtBinvBt,
                          this$precomputed$BtXbias,
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
                          user_bias, item_bias, center,
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
                          l1_lambda=0.,
                          niter=10L,
                          max_cg_steps=3L, finalize_chol=TRUE,
                          NA_as_zero_user=FALSE, NA_as_zero_item=FALSE,
                          nonneg=FALSE, nonneg_C=FALSE, nonneg_D=FALSE, max_cd_steps=100L,
                          apply_log_transf=FALSE,
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
    this$info$w_main     <-  w_main
    this$info$w_user     <-  w_user
    this$info$w_item     <-  w_item
    this$info$n_orig     <-  processed_X$n
    this$info$k          <-  k
    this$info$k_user     <-  k_user
    this$info$k_item     <-  k_item
    this$info$k_main     <-  k_main
    this$info$lambda     <-  lambda
    this$info$l1_lambda  <-  l1_lambda
    this$info$alpha      <-  alpha
    this$info$user_mapping     <-  user_mapping
    this$info$item_mapping     <-  item_mapping
    this$info$U_cols           <-  U_cols
    this$info$I_cols           <-  I_cols
    this$info$NA_as_zero_user  <-  NA_as_zero_user
    this$info$NA_as_zero_item  <-  NA_as_zero_item
    this$info$nonneg           <-  nonneg
    this$info$implicit         <-  TRUE
    this$info$apply_log_transf <-  apply_log_transf
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
            if (!nonneg)
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
                      lambda, l1_lambda,
                      processed_U$Uarr, processed_U$m, processed_U$p,
                      processed_I$Uarr, processed_I$m, processed_I$p,
                      processed_U$Urow, processed_U$Ucol, processed_U$Uval,
                      processed_I$Urow, processed_I$Ucol, processed_I$Uval,
                      NA_as_zero_user, NA_as_zero_item,
                      k_main, k_user, k_item,
                      w_main, w_user, w_item,
                      niter, nthreads, verbose, handle_interrupt,
                      use_cg, max_cg_steps, finalize_chol,
                      nonneg, max_cd_steps, nonneg_C, nonneg_D,
                      this$info$alpha, downweight, this$info$apply_log_transf,
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
#' @rdname fit
MostPopular <- function(X, weight=NULL, implicit=FALSE, apply_log_transf=FALSE, nonneg=FALSE,
                        user_bias=ifelse(implicit, FALSE, TRUE), lambda=10., alpha=1., scale_lam=FALSE) {
    inputs <- validate.inputs(model = "MostPopular", implicit = implicit,
                              X = X, weight = weight,
                              apply_log_transf = apply_log_transf,
                              nonneg = nonneg, scale_lam = scale_lam,
                              user_bias = user_bias, lambda = lambda,
                              alpha = alpha, downweight = FALSE)
    if (inputs$downweight && !inputs$implicit)
        stop("'downweight' option only meaningful with 'implicit=TRUE'.")
    return(.MostPopular(processed_X = inputs$processed_X,
                        user_mapping = inputs$user_mapping, item_mapping = inputs$item_mapping,
                        implicit = inputs$implicit,
                        user_bias = inputs$user_bias,
                        lambda = inputs$lambda, alpha = inputs$alpha,
                        scale_lam = inputs$scale_lam,
                        downweight = inputs$downweight,
                        apply_log_transf = inputs$apply_log_transf,
                        nonneg = inputs$nonneg))
}

.MostPopular <- function(processed_X,
                         user_mapping, item_mapping,
                         implicit=FALSE, user_bias=FALSE,
                         lambda=10., scale_lam=FALSE, alpha=1.,
                         downweight=FALSE, apply_log_transf=FALSE, nonneg=FALSE) {
    
    this <- list(
        info = get.empty.info(),
        matrices = get.empty.matrices(),
        precomputed = get.empty.precomputed()
    )
    
    ### Fill in info
    this$info$lambda  <-  lambda
    this$info$alpha   <-  alpha
    this$info$center  <-  !implicit
    this$info$user_mapping  <-  user_mapping
    this$info$item_mapping  <-  item_mapping
    this$info$implicit          <-  implicit
    this$info$scale_lam         <-  scale_lam
    this$info$nonneg            <-  nonneg
    this$info$apply_log_transf  <-  apply_log_transf
    
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
                      scale_lam,
                      alpha,
                      processed_X$m, processed_X$n,
                      processed_X$Xrow, processed_X$Xcol, processed_X$Xval,
                      processed_X$Xarr,
                      processed_X$Warr, processed_X$Wsp,
                      implicit, downweight, apply_log_transf,
                      nonneg,
                      w_main_multiplier,
                      1L)
    
    this$matrices$glob_mean <- glob_mean
    this$info$w_main_multiplier <- w_main_multiplier
    
    check.ret.code(ret_code)
    class(this) <- c("MostPopular", "cmfrec")
    return(this)
}

#' @export
#' @rdname fit
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
    this$info$center  <-  TRUE
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
#' @rdname fit
OMF_explicit <- function(X, U=NULL, I=NULL, weight=NULL,
                         k=50L, lambda=1e1, method="lbfgs", use_cg=TRUE,
                         user_bias=TRUE, item_bias=TRUE, center=TRUE, k_sec=0L, k_main=0L,
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
                              user_bias = user_bias, item_bias = item_bias, center = center,
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
                         center = inputs$center,
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
                          user_bias=TRUE, item_bias=TRUE, center=TRUE,
                          k_sec=0L, k_main=0L,
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
    this$info$center  <-  center
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
                          user_bias, item_bias, center, add_intercepts,
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
                          user_bias, item_bias, center, add_intercepts,
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
#' @rdname fit
OMF_implicit <- function(X, U=NULL, I=NULL,
                         k=50L, lambda=1e0, alpha=1., use_cg=TRUE,
                         add_intercepts=TRUE, niter=10L,
                         apply_log_transf=FALSE,
                         max_cg_steps=3L, finalize_chol=FALSE,
                         verbose=FALSE,
                         handle_interrupt=TRUE,
                         nthreads=parallel::detectCores()) {
    
    inputs <- validate.inputs(model = "OMF_implicit",
                              X = X, U = U, I = I,
                              k = k, lambda = lambda, alpha = alpha, use_cg = use_cg,
                              add_intercepts = add_intercepts, niter = niter,
                              max_cg_steps = max_cg_steps, finalize_chol = finalize_chol,
                              apply_log_transf = apply_log_transf,
                              verbose = verbose,
                              handle_interrupt = handle_interrupt,
                              nthreads = nthreads)
    return(.OMF_implicit(inputs$processed_X, inputs$processed_U, inputs$processed_I,
                         inputs$user_mapping, inputs$item_mapping,
                         inputs$U_cols, inputs$I_cols,
                         k = inputs$k, lambda = inputs$lambda, alpha = inputs$alpha,
                         use_cg = inputs$use_cg,
                         add_intercepts = inputs$add_intercepts,
                         niter = inputs$niter,
                         apply_log_transf = inputs$apply_log_transf,
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
                          apply_log_transf=FALSE,
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
    this$info$implicit          <-  TRUE
    this$info$apply_log_transf  <-  apply_log_transf
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
                      this$info$alpha, this$info$apply_log_transf,
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
#' @title Precompute matrices to use for predictions
#' @description Pre-computes internal matrices which might be used to speed up
#' computations on new data in the \link{CMF} and \link{CMF_implicit} models.
#' This function does not need to be called when passing
#' `precompute_for_predictions=TRUE`.
#' @return The same model object, with the pre-calculated matrices inside it.
#' @param model A collective matrix factorization model object, for which the
#' pre-computed matrices will be calculated.
precompute.for.predictions <- function(model) {
    supported_models <- c("CMF", "CMF_implicit")
    if (!NROW(intersect(class(model), supported_models)))
        stop("Method is only applicable to: ", paste(supported_models, collapse=", "))
    
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
        if (!model$info$nonneg && !add_implicit_features)
            model$precomputed$TransBtBinvBt <- matrix(0., ncol=n_use, nrow=k+k_main+user_bias)
        if (add_implicit_features)
            model$precompted$BiTBi <- matrix(0., ncol=k+k_main, nrow=k+k_main)
        if (has_U) {
            model$precomputed$CtC <- matrix(0., ncol=k_user+k, nrow=k_user+k)
            if (!model$info$nonneg && !add_implicit_features)
                model$precomputed$TransCtCinvCt <- matrix(0., ncol=p, nrow=k_user+k)
        }
        if ((add_implicit_features || has_U) && !model$info$nonneg) {
            model$precomputed$BeTBeChol <- matrix(0., nrow=k_user+k+k_main+user_bias,
                                                  ncol=k_user+k+k_main+user_bias)
        }
        if (model$info$NA_as_zero && (model$matrices$glob_mean || NROW(model$matrices$item_bias)))
            model$precomputed$BtXbias <- numeric(k+k_main+user_bias)
        
        ret_code <- .Call("call_precompute_collective_explicit",
                          model$matrices$B, n_use, n_max, model$info$include_all_X,
                          model$matrices$C, p,
                          model$matrices$Bi, add_implicit_features,
                          model$matrices$item_bias, model$matrices$glob_mean, model$info$NA_as_zero,
                          model$info$k, model$info$k_user, model$info$k_item, model$info$k_main,
                          user_bias,
                          model$info$nonneg,
                          model$info$lambda,
                          model$info$scale_lam, model$info$scale_lam_sideinfo,
                          model$info$w_main, model$info$w_user, model$info$w_implicit,
                          model$precomputed$B_plus_bias,
                          model$precomputed$BtB,
                          model$precomputed$TransBtBinvBt,
                          model$precomputed$BtXbias,
                          model$precomputed$BeTBeChol,
                          model$precompted$BiTBi,
                          model$precomputed$TransCtCinvCt,
                          model$precomputed$CtC)
    } else if ("CMF_implicit" %in% class(model)) {
        
        model$precomputed$BtB <- matrix(0., nrow=k+k_main, ncol=k+k_main)
        if (has_U) {
            model$precomputed$BeTBe <- matrix(0., nrow=k_user+k+k_main,
                                              ncol=k_user+k+k_main)
            if (!model$info$nonneg)
                model$precomputed$BeTBeChol <- matrix(0., nrow=k_user+k+k_main,
                                                      ncol=k_user+k+k_main)
        }
        
        ret_code <- .Call("call_precompute_collective_implicit",
                          model$matrices$B, n_max,
                          model$matrices$C, p,
                          model$info$k, model$info$k_user, model$info$k_item, model$info$k_main,
                          model$info$lambda, model$info$w_main, model$info$w_user,
                          model$info$w_main_multiplier,
                          model$info$nonneg,
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
