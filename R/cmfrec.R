#' @name cmfrec
#' @title cmfrec package
#' @description This is a library for approximate low-rank matrix factorizations, mainly oriented
#' towards recommender systems but being also usable for other domains such as dimensionality reduction
#' or imputation of missing data.
#' 
#' In short, the main goal behind the models in this library is to produce an approximate factorization
#' of a matrix \eqn{\mathbf{X}} (which might potentially be sparse or very large) as the product of two
#' lower-dimensional matrices:
#' \cr
#' \eqn{ \mathbf{X} \approx \mathbf{A} \mathbf{B}^T  }
#' \cr
#' For recommender systems, it is assumed that \eqn{\mathbf{X}} is a matrix encoding user-item
#' interactions, with rows representing users, columns representing items, and values representing
#' some interaction or rating (e.g. numer of times each user listened to different songs), where higher
#' numbers mean better affinity for a given item. Under this setting, the items for which a user might have
#' higher affinity should have larger values in the approximate factorization, and thus items with larger
#' values for a given user can be considered as better candidates to recommend - the idea being to recommend
#' new items to users that have not yet been seen/consumed/bought/etc.
#' 
#' This matrix factorization might optionally be enhanced by incorporating side information about users/items
#' in the 'X' matrix being factorized, which is assumed to come in the form of additional matrices
#' \eqn{\mathbf{U}} (for users) and/or \eqn{\mathbf{I}} (for items), which might also get factorized along
#' the way sharing some of the same low-dimensional components.
#' 
#' The main function in the library is \link{CMF}, which can fit many variations of the model described
#' above under different settings (e.g. whether entries missing in a sparse matrix are to be taken as zeros
#' or ignored in the objective function, whether to apply centering, to including row/column intercepts,
#' which kind of regularization to apply, etc.).
#' 
#' A specialized function for implicit-feedback models is also available under \link{CMF_implicit}, which
#' provides more appropriate defaults for such data.
#' @section Nomenclature used throughout the library:
#' The documentation and function namings use the following naming conventions:
#' \itemize{
#' \item About data: \itemize{
#' \item 'X' -> data about interactions between users/rows and items/columns (e.g. ratings given by users to items).
#' \item 'U' -> data about user/row attributes (e.g. user's age).
#' \item 'I' -> data about item/column attributes (e.g. a movie's genre).
#' }
#' \item About functionalities: \itemize{
#' \item 'warm' -> predictions based on new, unseen 'X' data, and potentially including new 'U' data along.
#' \item 'cold' -> predictions based on new user attributes data 'U', without 'X'.
#' \item 'new' -> predictions about new items based on attributes data 'I'.
#' }
#' \item About function descriptions: \itemize{
#' \item 'existing' -> the user/item was present in the training data to which the model was fit.
#' \item 'new' -> the user/items was not present in the training data that was passed to 'fit'.
#' }
#' }
#' 
#' Be aware that the package's functions are user-centric (e.g. it will recommend items for users, but not users for items).
#' If predictions about new items are desired, it's recommended to use the method \link{swap.users.and.items}, as the item-based
#' functions which are provided for convenience might run a lot slower than their user equivalents.
#' @section Implicit and explicit feedback:
#' In recommender systems, data might come in the form of explicit user judgements about items (e.g. movie ratings) or
#' in the form of logged user activity (e.g. number of times that a user listened to each song in a catalogue). The former
#' is typically referred to as "explicit feedback", while the latter is referred to as "implicit feedback".
#' 
#' Historically, driven by the Netflix competition, formulations of this problem have geared towards predicting the rating
#' that users would give to items under explicit-feedback datasets, determining the components in the low-rank factorization
#' in a way that minimizes the deviation between predicted and observed numbers on the \bold{observed} data only (i.e. predictions
#' about items that a user did not rate do not play any role in the optimization problem for determining the low-rank factorization
#' as they are simply ignored), but this approach has turned out to oftentimes result in very low-quality recommendations,
#' particularly for users with few data, and is usually not suitable for implicit feedback as the data in such case does not contain
#' any examples of dislikes and might even come in just binary (yes/no) form.
#' 
#' As such, research has mostly shifted towards the implicit-feedback setting, in which items that are not consumed by users
#' do play a role in the optimization objective for determining the low-rank components - that is, the goal is more to predict
#' which items would users have consumed than to predict the exact rating that they'd give to them - and the evaluation of recommendation
#' quality has shifted towards looking at how items that were consumed by users would be ranked compared to unconsumed items
#' (evaluation metrics for implicit-feedback for this library can be calculated through the package
#' \href{https://cran.r-project.org/package=recometrics}{recometrics}).
#' @section Other problem domains:
#' The documentation and naming conventions in this library are all oriented towards recommender systems, with the assumption
#' that users are rows in a matrix, items are columns, and values denote interactions between them, with the idea that values
#' under different columns are comparable (e.g. the rating scale is the same for all items).
#' 
#' The concept of approximate low-rank matrix factorizations is however still useful for other problem domains, such as general
#' dimensionality reduction for large sparse data (e.g. TF-IDF matrices) or imputation of high-dimensional tabular data, in which
#' assumptions like values being comparable between different columns would not hold.
#' 
#' Be aware that functions like \link{CMF} come with some defaults that might not be reasonable in other applications, but which
#' can be changed by passing non-default arguments to functions - for example:
#' \itemize{
#' \item Global centering - the "explicit-feedback" models here will by default calculate a global mean for all entries in 'X' and
#' center the matrix by substracting this value from all entries. This is a reasonable thing to do when dealing with movie ratings
#' as all ratings follow the same scale, but if columns of the 'X' matrix represent different things that might have different ranges
#' or different distributions, global mean centering is probably not going to be desirable or useful.
#' \item User/row biases: models might also have one bias/intercept parameter per row, which in the approximation, would get added
#' to every column for that user/row. This is again a reasonable thing to do for movie ratings, but if the columns of 'X' contain
#' different types of information, it might not be a sensible thing to add.
#' \item Regularization for item/column biases: since the models perform global mean centering beforehand, the item/column-specific
#' bias/intercept parameters will get a regularization penalty ("shrinkage") applied to them, which might not be desirable if
#' global mean centering is removed.
#' }
#' @section Improving performance:
#' This library will run faster when compiled from source with non-default compiler arguments,
#' particularly `-march=native` (replace with `-mcpu=native` for ARM/PPC); and when using an
#' optimized BLAS library for R. See this guide for details:
#' \href{https://github.com/david-cortes/installing-optimized-libraries}{installing optimized libraries}.
#' @seealso \link{CMF} \link{CMF_implicit}
#' @keywords package
#' @docType package
"_PACKAGE"
