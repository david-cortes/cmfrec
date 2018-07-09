import pandas as pd, numpy as np, tensorflow as tf, os, warnings
pd.options.mode.chained_assignment = None

class CMF:
    """
    Collective matrix factorization model for recommenders systems with explicit data and side info.
    
    Fits a collective matrix factorization model to ratings data along with item and/or user side information,
    by factorizing all matrices with common (shared) factors, e.g.:
    X~=AB' and I~=BC'

    By default, the function to minimize is as follows:
    L = w_main*norm(X- AB' - Ubias_{u} - Ibias_{i})^2/|X| + w_item*norm(I-BC')^2/|I| + w_user*norm(U-AD')^2/|U|

    with added regularization as follows:
    L +=  reg_param*(norm(A)^2+norm(B)^2+norm(C)^2+norm(D)^2)

    And user/item biases
    
    Where:

        X is the ratings matrix (considering only non-missing entries).
        I is the item-attribute matrix (only supports dense inputs).
        U is the user-attribute matrix (only supports dense inputs).
        A, B, C, D are lower-dimensional matrices (the model parameters).
        |X|, |I|, |U| are the number non-missing of entries in each matrix.
        The matrix products might not use all the rows/columns of these matrices at each factorization
        (this is controlled with k_main, k_item and k_user).
        Ubias_{u} and Ibias_{i} are user and item biases, defined for each user and item ID.

    ** Be aware that, due to the regularization part, this formula as it is implies that larger datasets require
    lower regularization. You can use 'standardize_err=False' and set w1/w2/w3 to avoid this.**

    Alternatively, it can also fit an additive model with "offsets", similar in spirit to [2] (see references):

    L = norm(X - (A+UC)(B+ID)' - Ubias_{u} - Ibias_{i})^2 + reg_param*(norm(A)^2+norm(B)^2+norm(C)^2+norm(D)^2)

    This second model requires more iterations to fit (takes more time), and doesn't support missing value imputation,
    but it oftentimes provides better results, especially for cold-start recommendations and if the side information
    (users and items) is mostly binary columns.
    
    In both cases, the model is fit with full L-BFGS updates (not stochastic gradient descent or stochastic Newton),
    i.e. it calculates errors for the whole data and updates all model parameters at once during each iteration.
    This usually leads to better local optima and less need for parameter tuning, but at the cost of less
    scalability. The L-BFGS implementation used is from SciPy, which is not entirely multi-threaded, so  if your CPU
    has many cores or you have many GPUs, you can expect an speed up from parallelization on the calculations of the
    objective function and the gradient, but not so much on the calculations that happen inside L-BFGS.

    By default, the number of iterations is set at 1000, but for smaller datasets and for the offsets model, this might
    not reach convergence when using high regularization.
    
    If passing reindex=True, it will internally reindex all user and item IDs. Your data will not require
    reindexing if the IDs for users and items in the input data frame passed to .fit meet the following criteria:

    1) Are all integers.
    2) Start at zero.
    3) Don't have any enumeration gaps, i.e. if there is a user '4', user '3' must also be there.

    Adding side information about entries for which there are no ratings or vice-versa can still help to improve
    factorizations, but the more elements they have in common, the better.

    If passing reindex=False, then the matrices with side information (user and item attributes) must have exactly
    the same number of rows as the number of users/items present in the ratings data, in which case there cannot
    be entries (users or items) missing from the ratings and present in the side information or vice-versa.

    For missing entries in the user and item attributes, use numpy.nan. These will not be taken into account in
    the optimization procedure. If there is no information for use user or item, leave that row out altogether
    instead of filling with NAs. In the offsets model, missing entries will be automatically filled with zeros,
    so it's recommended to perform imputation beforehand for it.

    Can also produce non-negative matrix factorization (including the user and item attributes), but if using
    user and/or item biases, these will not be constrained to be non-negative.

    For the regular model, if there are binary columns in the data, it can apply a sigmoid transformation to the
    approximations from the factorization, but these will still be taken as a squared loss with respect to the
    original 0/1 values, so as to make the loss comparable to that of the other columns. Be sure to pass the names
    of the binary columns to '.fit', or the indexes or the columns when using 'reindex=False'.

    Note
    ----
    **Be aware that the data passed to '.fit' will be modified inplace (e.g. reindexed). Make a copy of your data beforehand if
    you require it.** If you plan to do any hyper-parameter tuning through cross-validation, you should reindex your data
    beforehand and call the CMF constructur with 'index=False'.

    Note
    ----
    The API contains parameters for both an item-attribute matrix and a user-attribute matrix,
    but you can fit the model to data with only one or none of them.
    Parameters corresponding to the factorization of a matrix for which no data is passed will be ignored.

    Note
    ----
    The model allows to make recommendations for users and items for which there is data about their
    attributes but no ratings. The quality of these predictions might however not be good, especially
    if you set k_main > 0. It is highly recommended to center your ratings if you plan on making
    predictions for user/items that were not in the training set.

    Parameters
    ----------
    k : int
        Number of common (shared) latent factors to use.
    k_main : int
        Number of additional (non-shared) latent factors to use for the ratings factorization.
        Ignored for the offsets model.
    k_user: int
        Number of additional (non-shared) latent factors to use for the user attributes factorization.
        Ignored for the offsets model.
    k_item : int
        Number of additional (non-shared) latent factors to use for the item attributes factorization.
        Ignored for the offsets model.
    w_main : float
        Weight to assign to the (mean) root squared error in factorization of the ratings matrix.
        Ignored for the offsets model.
    w_user : float
        Weight to assign to the (mean) root squared error in factorization of the user attributes matrix.
        Ignored for the offsets model.
    w_item : float
        Weight to assign to the (mean) root squared error in factorization of the item attributes matrix.
        Ignored for the offsets model.
    reg_param : float or tuple of floats
        Regularization parameter for each matrix, in this order:
        1) User-Factor, 2) Item-Factor, 3) User bias, 4) Item-bias, 5) UserAttribute-Factor, 6) ItemAttribute-Factor.
    offsets_model : bool
        Whether to fit the alternative model formulation with offsets (see description).
    nonnegative : bool
        Whether the resulting low-rank matrices (A and B) should have all non-negative entries.
        Forced to False when passing 'center_ratings=True'.
    maxiter : int
        Maximum number of iterations for which to run the optimization procedure. Recommended to use a higher number
        for the offsets model.
    standardize_err : bool
        Whether to divide the sum of squared errors from each factorized matrix by the number of non-missing entries.
        Setting this to False requires far larger regularization parameters. Note that most research papers
        don't standardize the errors, so if you try to reproduce some paper with specific parameters, you might
        want to set this to True.
        Forced to False when passing 'reweight=True'.
    reweight : bool
        Whether to automatically reweight the errors of each matrix factorization so that they get similar influence,
        accounting for the number of elements in each and the magnitudes of the entries
        (appplies in addition to weights passed as w_main, w_item and w_user).
        This is done by calculating the initial sum of squared errors with randomly initialized factor matrices,
        but it's not guaranteed to be a good criterion.
        It might be better to scale the entries of either the ratings or the attributes matrix so that they are in a similar scale
        (e.g. if the ratings are in [1,5], the attributes should ideally be in the same range and not [-10^3,10^3]).
        Ignored for the offsets model.
    reindex : bool
        Whether to reindex data internally (assign own IDs) - see description above.
    center_ratings : bool
        Whether to substract the mean rating from the ratings before fitting the model. Will be force to True if
        passing 'add_user_bias=True' or 'add_item_bias=True'.
    user_bias : bool
        Whether to use user biases (one per user) as additional model parameters.
    item_bias : bool
        Whether to use item biases (one per item) as additional model parameters.
    center_user_info : bool
        Whether to center the user attributes by substracting the mean from each column.
    center_item_info : bool
        Whether to center the item attributes by substracting the mean from each column.
    user_info_nonneg : bool
        Whether the user_attribute-factor matrix (C) should have all non-negative entries.
        Forced to false when passing 'center_user_info=True'.
    item_info_nonneg : bool
        Whether the item_attribute-factor matrix (D) should have all non-negative entries.
        Forced to false when passing 'center_item_info=True'
    keep_data : bool
        Whether to keep information about which user was associated with each item
        in the training set, so as to exclude those items later when making Top-N
        recommendations.
    save_folder : str or None
        Folder where to save all model parameters as csv files.
    produce_dicts : bool
        Whether to produce Python dictionaries for users and items, which
        are used by the prediction API of this package. You can still predict without
        them, but it might take some additional miliseconds (or more depending on the
        number of users and items).
    random_seed : int or None
        Random seed to use when starting the parameters.
    verbose : bool
        Whether to display convergence messages frol L-BFGS. If running it from
        an IPython notebook, these will be printed in the console in which the
        notebook is running, but not on the output of the cell within the notebook.

    Attributes
    ----------
    A : array (nitems, k_main + k + k_user)
        Matrix with the user-factor attributes, containing columns from both factorizations.
        If you wish to extract only the factors used for predictons, slice it like this: A[:,:k_main+k]
    B : array (nusers, k_main + k + k_item)
        Matrix with the item-factor attributes, containing columns from both factorizations.
        If you wish to extract only the factors used for predictons, slice it like this: B[:,:k_main+k]
    C : array (k + k_item, item_dim)
        Matrix of factor-item_attribute. Will have the columns that correspond to binary features in
        a separate attribute.
    D : array (k_user + k, user_dim)
        Matrix of factor-user_attribute. Will have the columns that correspond to binary features in
        a separate attribute.
    C_bin : array (k + k_item, item_dim_bin):
        Part of the C matrix that corresponds to binary columns in the item data.
        Non-negativity constraints will not apply to this matrix.
    D_bin : array (k_user + k, user_dim_bin)
        Part of the D matrix that corresponds to binary columns in the user data.
        Non-negativity constraints will not apply to this matrix.
    add_user_bias : array (nusers, )
        User biases determined by the model
    add_item_bias : array (nitems, )
        Item biases determined by the model
    user_mapping_ : array (nusers,)
        ID of the user (as passed to .fit) represented by each row of A.
    item_mapping_ : array (nitems,)
        ID of the item (as passed to .fit) represented by each row of B.
    user_dict_ : dict (nusers)
        Dictionary with the mapping between user IDs (as passed to .fit) and rows of A.
    item_dict_ : dict (nitems)
        Dictionary with the mapping between item IDs (as passed to .fit) and rows of B.
    is_fitted : bool
        Whether the model has been fit to some data.
    global_mean : float
        Global mean of the ratings.
    user_arr_means : array (user_dim, )
        Column means of the user side information matrix.
    item_arr_means : array (item_dim, )
        Column means of the item side information matrix.

    References
    ----------
    [1] Relational learning via collective matrix factorization (A. Singh, 2008)
    [2] Collaborative topic modeling for recommending scientific articles (C. Wang, D. Blei, 2011)
    """
    def __init__(self, k=30, k_main=0, k_user=0, k_item=0,
                 w_main=1.0, w_user=1.0, w_item=1.0, reg_param=1e-4,
                 offsets_model=False, nonnegative=False, maxiter=1000,
                 standardize_err=True, reweight=False, reindex=True,
                 center_ratings=True, add_user_bias=True, add_item_bias=True,
                 center_user_info=False, center_item_info=False,
                 user_info_nonneg=False, item_info_nonneg=False,
                 keep_data=True, save_folder=None, produce_dicts=True,
                 random_seed=None, verbose=True):
        ## checking input parameters
        assert isinstance(k, int)
        if k_main is None:
            k_main = 0
        if k_item is None:
            k_item = 0
        if k_user is None:
            k_user = 0
        assert isinstance(k_main, int)
        assert isinstance(k_item, int)
        assert isinstance(k_user, int)

        if isinstance(w_main, int):
            w_main = float(w_main)
        if isinstance(w_item, int):
            w_item = float(w_item)
        if isinstance(w_user, int):
            w_user = float(w_user)

        if w_user is None:
            w_user = 0
        if w_item is None:
            w_item = 0

        assert isinstance(w_main, float)
        assert isinstance(w_user, float)
        assert isinstance(w_item, float)

        if random_seed is not None:
            assert isinstance(random_seed, int)

        if maxiter is not None:
            assert maxiter>0
            assert isinstance(maxiter, int)

        if save_folder is not None:
            save_folder = os.path.expanduser(save_folder)
            assert os.path.exists(save_folder)

        ## storing these parameters
        self.k = k
        self.k_main = k_main
        self.k_item = k_item
        self.k_user = k_user
        self.w_main = w_main
        self.w_item = w_item
        self.w_user = w_user
        if (type(reg_param) == float) or (type(reg_param) == int):
            self.reg_param = [reg_param]*6
        elif isinstance(reg_param, np.ndarray):
            self.reg_param = list(reg_param.reshape(-1))
        elif reg_param.__class__.__name__ == 'Series':
            if len(reg_param) != 6:
                raise ValueError('reg_param must be a number or tuple with 6 entries.')
            self.reg_param = list(reg_param.values)
        elif (type(reg_param) == tuple) or (type(reg_param) == list):
            if len(reg_param) != 6:
                raise ValueError('reg_param must be a number or tuple with 6 entries.')
            self.reg_param = reg_param
        else:
            raise ValueError('reg_param must be a number or tuple with 6 entries.')
        self.standardize_err = bool(standardize_err)
        self.reweight = bool(reweight)
        self.produce_dicts = bool(produce_dicts)
        self.keep_data = bool(keep_data)
        self.verbose = bool(verbose)
        self.reindex = bool(reindex)
        self.center_ratings = bool(center_ratings)
        self.add_user_bias = bool(add_user_bias)
        self.add_item_bias = bool(add_item_bias)
        self.center_user_info = bool(center_user_info)
        self.center_item_info = bool(center_item_info)
        self.nonnegative = bool(nonnegative)
        self.user_info_nonneg = bool(user_info_nonneg)
        self.item_info_nonneg = bool(item_info_nonneg)
        self.offsets_model = bool(offsets_model)
        if self.offsets_model:
            self.k_main = 0
            self.k_user = 0
            self.k_item = 0
            self.reweight = False
            self.w_main = None
            self.w_user = None
            self.w_item = None
        if self.reweight:
            self.standardize_err = False


        if self.add_user_bias or self.add_item_bias:
            self.center_ratings = True

        if self.center_ratings:
            self.nonnegative = False
        if self.center_user_info:
            self.user_info_nonneg = False
        if self.center_item_info:
            self.item_info_nonneg = False

        self.maxiter = maxiter
        self.random_seed = random_seed
        self.save_folder = save_folder

        ## initializing other attributes
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.C_bin = None
        self.D_bin = None
        self.user_mapping_ = None
        self.item_mapping_ = None
        self.user_dict_ = None
        self.item_dict_ = None
        self.is_fitted = False
        self.global_mean = None
        self.user_bias = None
        self.item_bias = None
        self.user_arr_means = None
        self.item_arr_means = None
        
    def fit(self, ratings, user_info=None, item_info=None, cols_bin_user=None, cols_bin_item=None):
        """
        Fit the model to ratings data and item/user side info, using L-BFGS

        Note
        ----
        **Be aware that the data passed to '.fit' will be modified inplace (e.g. reindexed). Make a copy of your data beforehand if
        you require it (e.g. using the "deepcopy" function from the "copy" module).**
        
        Parameters
        ----------
        ratings : pandas data frame or array (nobs, 3)
            Ratings data to which to fit the model.
            If a pandas data frame, must contain the columns 'UserId','ItemId' and 'Rating'. Optionally, it might also contain a column 'Weight'.
            If a numpy array, will take the first 4 columns in that order
            If a list of tuples, must be in the format (UserId,ItemId,Rating,[Weight]) (will be coerced to data frame)
        user_info : pandas data frame or numpy array (nusers, nfeatures_user)
            Side information about the users (i.e. their attributes, as a table).
            Must contain a column called 'UserId'.
            If called with 'reindex=False', must be a numpy array,
            with rows corresponding to user ID numbers and columns to user attributes.
        item_info : pandas data frame or numpy array (nitems, nfeatures_item)
            Side information about the items (i.e. their attributes, as a table).
            Must contain a column named ItemId.
            If called with 'reindex=False', must be a numpy array,
            with rows corresponding to item ID numbers and columns to item attributes.
        cols_bin_user : array or list
            Columns of user_info that are binary (only take values 0 or 1).
            Will apply a sigmoid function to the factorized approximations of these columns.
            Ignored when called with 'offsets_model=True'.
        cols_bin_item : array or list
            Columns of item_info that are binary (only take values 0 or 1).
            Will apply a sigmoid function to the factorized approximations of these columns.
            Ignored when called with 'offsets_model=True'.

        Returns
        -------
        self : obj
            Copy of this object
        """
        
        # readjusting parameters in case they are redundant
        if item_info is None:
            self.k_user = 0
        if user_info is None:
            self.k_item = 0
        
        self._process_data(ratings, item_info, user_info, cols_bin_user, cols_bin_item)
        self._set_weights(self.random_seed)
        self._fit(self.w1, self.w2, self.w3, self.reg_param,
                  self.k, self.k_main, self.k_item, self.k_user,
                  self.random_seed, self.maxiter)

        self.is_fitted = True
        self._clear_internal_objs()
        
        return self

    def _clear_internal_objs(self):
        ## after terminating optimization
        if (self.center_user_info) and self._user_arr is not None:
            self.user_arr_means = self.user_arr_means.reshape(-1)
        if (self.center_item_info) and self._item_arr is not None:
            self.item_arr_means = self.item_arr_means.reshape(-1)

        del self._ratings
        del self._ix_u
        del self._ix_i
        del self._item_arr_notmissing
        del self._user_arr_notmissing
        del self._weights
        if self._item_arr_mismatch:
            del self._slc_item
        del self._item_arr_mismatch
        if self._user_arr_mismatch:
            del self._slc_user
        del self._user_arr_mismatch
        if self.add_user_bias:
            del self.init_u_bias
        if self.add_item_bias:
            del self.init_i_bias
        if self._user_arr is not None:
            if (self.cols_bin_user is not None) and (not self.offsets_model):
                del self._user_arr_bin
                del self._user_arr_notmissing_bin
        if self._item_arr is not None:
            if (self.cols_bin_item is not None) and (not self.offsets_model):
                del self._item_arr_bin
                del self._item_arr_notmissing_bin

        del self._item_arr
        del self._user_arr

        if self.produce_dicts and self.reindex:
            self.user_dict_ = {self.user_mapping_[i]:i for i in range(self.user_mapping_.shape[0])}
            self.item_dict_ = {self.item_mapping_[i]:i for i in range(self.item_mapping_.shape[0])}

    
    def _fit(self, w1=1.0, w2=1.0, w3=1.0, reg_param=[1e-3]*6,
             k=50, k_main=0, k_item=0, k_user=0,
             random_seed=None, maxiter=1000):
        # faster method, can be reused with the same data

        if random_seed is not None:
            tf.set_random_seed(random_seed)

        if self.add_user_bias:
            user_bias = tf.Variable(self.init_u_bias)
        if self.add_item_bias:
            item_bias = tf.Variable(self.init_i_bias)
        
        R = tf.placeholder(tf.float32, shape=(None,))
        U = tf.placeholder(tf.float32)
        I = tf.placeholder(tf.float32)
        U_bin = tf.placeholder(tf.float32)
        I_bin = tf.placeholder(tf.float32)
        W = tf.placeholder(tf.float32)
        
        I_nonmissing = tf.placeholder(tf.float32)
        U_nonmissing = tf.placeholder(tf.float32)
        I_nonmissing_bin = tf.placeholder(tf.float32)
        U_nonmissing_bin = tf.placeholder(tf.float32) 

        ### original formulation
        if not self.offsets_model:

            A = tf.Variable(tf.random_normal([self.nusers, k_main + k + k_user]))
            B = tf.Variable(tf.random_normal([self.nitems, k_main + k + k_item]))

            if self._user_arr is not None:
                C = tf.Variable(tf.random_normal([k + k_user, self.user_dim]))
                if self.cols_bin_user is not None:
                    C_bin = tf.Variable(tf.random_normal([k + k_user, self.cols_bin_user.shape[0]]))

            if self._item_arr is not None:
                D = tf.Variable(tf.random_normal([k + k_item, self.item_dim]))
                if self.cols_bin_item is not None:
                    D_bin = tf.Variable(tf.random_normal([k + k_item, self.cols_bin_item.shape[0]]))

            Ab = A[:, :k_main + k]
            Ac = A[:, k_main:]
            Ba = B[:, :k_main + k]
            Bd = B[:, k_main:]

            pred_ratings = tf.reduce_sum( tf.multiply(
                tf.gather(Ab, self._ix_u, axis=0),
                tf.gather(Ba, self._ix_i, axis=0))
                                           , axis=1)

            if self.add_user_bias:
                pred_ratings += tf.gather(user_bias, self._ix_u)
            if self.add_item_bias:
                pred_ratings += tf.gather(item_bias, self._ix_i)

            if self.reweight or (not self.standardize_err):
                if self._weights is None:
                    loss = w1 * tf.nn.l2_loss(pred_ratings - R)
                else:
                    loss = w1 * tf.nn.l2_loss( (pred_ratings - R) * W)
            else:
                if self._weights is None:
                    loss = w1 * tf.losses.mean_squared_error(pred_ratings, R)
                else:
                    loss = w1 * tf.losses.mean_squared_error(pred_ratings, R, weights=W)
            loss += reg_param[0] * tf.nn.l2_loss(A) + reg_param[1] * tf.nn.l2_loss(B)

            if self.add_user_bias:
                loss += reg_param[2] * tf.nn.l2_loss(user_bias)
            if self.add_item_bias:
                loss += reg_param[3] * tf.nn.l2_loss(item_bias)
                
            
            if self._user_arr is not None:
                if self._user_arr_mismatch:
                    pred_user = tf.matmul(tf.gather(Ac, self._slc_user), C)
                else:
                    pred_user = tf.matmul(Ac, C)
                if self._user_arr_notmissing is not None:
                    pred_user *= U_nonmissing

                if self.reweight or (not self.standardize_err):
                    loss += w2 * tf.nn.l2_loss(pred_user - U)
                else:
                    loss += w2 * tf.losses.mean_squared_error(pred_user, U)
                
                if self.cols_bin_user is not None:
                    loss += reg_param[4] * tf.nn.l2_loss(C_bin)
                    if self._user_arr_mismatch:
                        pred_user_bin = tf.sigmoid(tf.matmul(tf.gather(Ac, self._slc_user), C_bin))
                    else:
                        pred_user_bin = tf.sigmoid(tf.matmul(Ac, C_bin))
                    if self._user_arr_notmissing_bin is not None:
                        pred_user_bin *= U_nonmissing_bin

                    if self.standardize_err:
                        loss += w2 * tf.losses.mean_squared_error(pred_user_bin, U_bin)
                    else:
                        loss += w2 * tf.nn.l2_loss(pred_user_bin - U_bin)
                loss += reg_param[4] * tf.nn.l2_loss(C)

            if self._item_arr is not None:
                if self._item_arr_mismatch:
                    pred_item = tf.matmul(tf.gather(Bd, self._slc_item), D)
                else:
                    pred_item = tf.matmul(Bd, D)
                if self._item_arr_notmissing is not None:
                    pred_item *= I_nonmissing

                if self.reweight or (not self.standardize_err):
                    loss += w3 * tf.nn.l2_loss(pred_item - I)
                else:
                    loss += w3 * tf.losses.mean_squared_error(pred_item, I)
                
                if self.cols_bin_item is not None:
                    loss += reg_param[5] * tf.nn.l2_loss(D_bin)
                    if self._item_arr_mismatch:
                        pred_item_bin = tf.sigmoid(tf.matmul(tf.gather(Bd, self._slc_item), D_bin))
                    else:
                        pred_item_bin = tf.sigmoid(tf.matmul(Bd, D_bin))
                    if self._item_arr_notmissing_bin is not None:
                        pred_item_bin *= I_nonmissing_bin

                    if self.standardize_err:
                        loss += w3 * tf.losses.mean_squared_error(pred_item_bin, I_bin)
                    else:
                        loss += w3 * tf.nn.l2_loss(pred_item_bin - I_bin)
                loss += reg_param[5] * tf.nn.l2_loss(D)

        ### model with offsets instead
        else:

            A = tf.Variable(tf.random_normal([self.nusers, k]))
            B = tf.Variable(tf.random_normal([self.nitems, k]))

            if self._user_arr is not None:
                C = tf.Variable(tf.random_normal([self.user_dim, k]))
                if self._user_arr_mismatch:
                    ## if tensorflow bug gets fixed, enable this
                    ## https://github.com/tensorflow/tensorflow/issues/19717
                    ## currently will fill all user_arr and item_arr with rows of NAs
                    Acomp = tf.scatter_add(A, self._slc_user, tf.matmul(U, C))
                else:
                    Acomp = A + tf.matmul(U, C)
            else:
                Acomp = A

            if self._item_arr is not None:
                D = tf.Variable(tf.random_normal([self.item_dim, k]))
                if self._item_arr_mismatch:
                    ## if tensorflow bug gets fixed, enable this
                    ## currently will fill all user_arr and item_arr with rows of NAs
                    Bcomp = tf.scatter_add(B, self._slc_item.reshape(-1), tf.matmul(I, D))
                else:
                    Bcomp = B + tf.matmul(I, D)
            else:
                Bcomp = B

            pred_ratings = tf.reduce_sum( tf.multiply(
                    tf.gather(Acomp, self._ix_u, axis=0),
                    tf.gather(Bcomp, self._ix_i, axis=0))
                                           , axis=1)

            if self.add_user_bias:
                pred_ratings += tf.gather(user_bias, self._ix_u)
            if self.add_item_bias:
                pred_ratings += tf.gather(item_bias, self._ix_i)

            
            if self.standardize_err:
                if self._weights is None:
                    loss = tf.losses.mean_squared_error(pred_ratings, R)
                else:
                    loss = tf.losses.mean_squared_error(pred_ratings, R, weights=W)
            else:
                if self._weights is None:
                    loss = tf.nn.l2_loss(pred_ratings - R)
                else:
                    loss = tf.nn.l2_loss( (pred_ratings - R) * W)
            loss += reg_param[0] * tf.nn.l2_loss(A) + reg_param[1] * tf.nn.l2_loss(B)

            if self.add_user_bias:
                loss += reg_param[2] * tf.nn.l2_loss(user_bias)
            if self.add_item_bias:
                loss += reg_param[3] * tf.nn.l2_loss(item_bias)

            if self._user_arr is not None:
                loss += reg_param[4] * tf.nn.l2_loss(C)
            if self._item_arr is not None:
                loss += reg_param[5] * tf.nn.l2_loss(D)

        


        opts_lbfgs = {'maxiter':maxiter}
        if self.verbose:
            opts_lbfgs['disp'] = 1

        dct_bounds = dict()
        if self.nonnegative:
            dct_bounds[A] = (0, np.inf)
            dct_bounds[B] = (0, np.inf)
        if self.user_info_nonneg and self._user_arr is not None:
            dct_bounds[C] = (0, np.inf)
        if self.item_info_nonneg and self._item_arr is not None:
            dct_bounds[D] = (0, np.inf)

        if len(dct_bounds) == 0:
            dct_bounds = None

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B', options=opts_lbfgs, var_to_bounds=dct_bounds)
        model = tf.global_variables_initializer()
        sess=tf.Session()
        sess.run(model)
        with sess:
            if self.verbose:
                tf.logging.set_verbosity(tf.logging.INFO)
            else:
                tf.logging.set_verbosity(tf.logging.WARN)
            optimizer.minimize(sess, feed_dict={R:self._ratings, U:self._user_arr, I:self._item_arr, W:self._weights,
                                U_bin:self._user_arr_bin, I_bin:self._item_arr_bin,
                                I_nonmissing:self._item_arr_notmissing, U_nonmissing:self._user_arr_notmissing,
                                I_nonmissing_bin:self._item_arr_notmissing_bin,
                                U_nonmissing_bin:self._user_arr_notmissing_bin})
            self.A = A.eval(session=sess)
            self.B = B.eval(session=sess)
            if self.add_user_bias:
                self.user_bias = user_bias.eval(session=sess)
            if self.add_item_bias:
                self.item_bias = item_bias.eval(session=sess)
            if self._user_arr is not None:
                self.C = C.eval(session=sess)
                if (self.cols_bin_user is not None) and (not self.offsets_model):
                    self.C_bin = C_bin.eval(session=sess)
            if self._item_arr is not None:
                self.D = D.eval(session=sess)
                if (self.cols_bin_item is not None) and (not self.offsets_model):
                    self.D_bin = D_bin.eval(session=sess)

        
        if self.save_folder is not None:
            np.savetxt(os.path.join(self.save_folder, "A.csv"), self.A, fmt="%.10f", delimiter=',')
            np.savetxt(os.path.join(self.save_folder, "B.csv"), self.B, fmt="%.10f", delimiter=',')
            if self._user_arr is not None:
                np.savetxt(os.path.join(self.save_folder, "C.csv"), self.C, fmt="%.10f", delimiter=',')
                if (self.cols_bin_user is not None) and (not self.offsets_model):
                    np.savetxt(os.path.join(self.save_folder, "C_bin.csv"), self.C_bin, fmt="%.10f", delimiter=',')
                    pd.Series(self._cols_nonbin_user).to_csv(os.path.join(self.save_folder, "cols_user_num.csv"), index=False)
                    pd.Series(self.cols_bin_user).to_csv(os.path.join(self.save_folder, "cols_user_bin.csv"), index=False)
            if self._item_arr is not None:
                np.savetxt(os.path.join(self.save_folder, "D.csv"), self.D, fmt="%.10f", delimiter=',')
                if (self.cols_bin_item is not None) and (not self.offsets_model):
                    np.savetxt(os.path.join(self.save_folder, "D_bin.csv"), self.D_bin, fmt="%.10f", delimiter=',')
                    pd.Series(self._cols_nonbin_item).to_csv(os.path.join(self.save_folder, "cols_item_num.csv"), index=False)
                    pd.Series(self.cols_bin_item).to_csv(os.path.join(self.save_folder, "cols_item_bin.csv"), index=False)
            if self.add_user_bias:
                np.savetxt(os.path.join(self.save_folder, "user_bias.csv"), self.user_bias, fmt="%.10f", delimiter=',')
            if self.add_item_bias:
                np.savetxt(os.path.join(self.save_folder, "item_bias.csv"), self.item_bias, fmt="%.10f", delimiter=',')

        if self.offsets_model:
            self._Ab = self.A
            self._Ba = self.B

            if self._user_arr is not None:
                if self._user_arr_mismatch:
                    self._Ab[self._slc_user, :] += self._user_arr.dot(self.C)
                else:
                    self._Ab += self._user_arr.dot(self.C)
            if self._item_arr is not None:
                if self._item_arr_mismatch:
                    self._Ba[self._slc_item, :] += self._item_arr.dot(self.D)
                else:
                    self._Ba += self._item_arr.dot(self.D)
        
        else:
            self._Ab = self.A[:, :k_main + k]
            self._Ba = self.B[:, :k_main + k]
    
    def _set_null_iteminfo(self):
        self._item_arr = None
        self._item_arr_bin = None
        self._item_arr_notmissing = None
        self._item_arr_mismatch = False
        self.delete_iteminfo = False
        self.item_dim = 0
        self.cols_bin_item = None

    def _set_null_userinfo(self):
        self._user_arr = None
        self._user_arr_bin = None
        self._user_arr_notmissing = None
        self._user_arr_mismatch = False
        self.delete_userinfo = False
        self.user_dim = 0
        self.cols_bin_user = None


    def _process_data(self, ratings, item_info, user_info, cols_bin_user, cols_bin_item):
        # pasing an already properly index DF will speed up the process

        ## TODO: refactor this function and make it more modular

        ## Convert inputs to pandas data frames
        if isinstance(ratings, np.ndarray):
            assert len(ratings.shape) > 1
            assert ratings.shape[1] >= 3
            ratings = ratings.values[:,:4]
            ratings.columns = ['UserId', 'ItemId', "Rating", "Weight"][:ratings.shape[1]]

        if type(ratings) == list:
            if len(ratings[0]) == 3:
                ratings = pd.DataFrame(ratings, columns=['UserId','ItemId','Rating'])
            elif len(ratings[0]) > 3:
                ratings = pd.DataFrame(ratings, columns=['UserId','ItemId','Rating','Weight'])
            else:
                raise ValueError("If passing a list to 'ratings', it must contain tuples with (UserId, ItemId, Weight).")
            
        if ratings.__class__.__name__ == 'DataFrame':
            assert ratings.shape[0] > 0
            assert 'UserId' in ratings.columns.values
            assert 'ItemId' in ratings.columns.values
            assert 'Rating' in ratings.columns.values
            cols_take = ['UserId', 'ItemId', 'Rating']
            if 'Weight' in ratings.columns.values:
                cols_take += ['Weight']
            self._ratings = ratings[cols_take]
        else:
            raise ValueError("'ratings' must be a pandas data frame or a numpy array")

        if not self.reindex:

            if item_info is not None:

                if item_info.__class__.__name__ == 'DataFrame':
                    item_info = item_info.values
                if not isinstance(item_info, np.ndarray):
                    raise ValueError("When passing 'reindex=False', 'item_info' must be a numpy array")

                assert (self._ratings.ItemId.max() + 1) == item_info.shape[0]
                self.nitems = item_info.shape[0]
                self.item_dim = item_info.shape[1]
                self._item_arr_mismatch = False
                self._item_arr = item_info
                self.delete_iteminfo = False
                if isinstance(cols_bin_item, list):
                    cols_bin_item = np.array(cols_bin_item)
                self.cols_bin_item = cols_bin_item
                if self.cols_bin_item is not None:
                    self.cols_bin_item = self.cols_bin_item.reshape(-1).astype('int32')
                    assert (self.cols_bin_item.max() + 1 ) < self._item_arr.shape[1]
                    self._cols_nonbin_item = np.setdiff1d(np.arange(self.cols_bin_item.shape[0]), self.cols_bin_item)
                    if not self.offsets_model:
                        self._item_arr_bin = self._item_arr[:, self.cols_bin_item].astype('float32')
                        self._item_arr = self._item_arr[:, self._cols_nonbin_item]
                    else:
                        self._item_arr_bin = None
                else:
                    self._item_arr_bin = None

            else:
                self._set_null_iteminfo()

            if user_info is not None:

                if user_info.__class__.__name__ == 'DataFrame':
                    user_info = user_info.values
                if not isinstance(user_info, np.ndarray):
                    raise ValueError("When passing 'reindex=False', 'user_info' must be a numpy array")

                assert (self._ratings.UserId.max() + 1) == user_info.shape[0]
                self.nusers = user_info.shape[0]
                self.user_dim = user_info.shape[1]
                self._user_arr_mismatch = False
                self._user_arr = user_info
                self.delete_userinfo = False
                if isinstance(cols_bin_user, list):
                    cols_bin_user = np.array(cols_bin_user)
                self.cols_bin_user = cols_bin_user
                if self.cols_bin_user is not None:
                    self.cols_bin_user = self.cols_bin_user.reshape(-1).astype('int32')
                    assert (self.cols_bin_user.max() + 1 ) < self._user_arr.shape[1]
                    self._cols_nonbin_user = np.setdiff1d(np.arange(self.cols_bin_user.shape[0]), self.cols_bin_user)
                    if not self.offsets_model:
                        self._user_arr_bin = self._user_arr[:, self.cols_bin_user].astype('float32')
                        self._user_arr = self._user_arr[:, self._cols_nonbin_user]
                    else:
                        self._user_arr_bin = None
                else:
                    self._user_arr_bin = None
            else:
                self._set_null_userinfo()

            self.nusers = self._ratings.UserId.max() + 1
            self.nitems = self._ratings.ItemId.max() + 1
            self._ratings['UserId'] =  self._ratings.UserId.values.astype('int32')
            self._ratings['ItemId'] =  self._ratings.ItemId.values.astype('int32')

        ### when data needs reindexing
        else:
            self._ratings['UserId'], self.user_mapping_ = pd.factorize(self._ratings.UserId)
            self._ratings['ItemId'], self.item_mapping_ = pd.factorize(self._ratings.ItemId)
            self.user_mapping_ = np.array(self.user_mapping_).reshape(-1)
            self.item_mapping_ = np.array(self.item_mapping_).reshape(-1)
            self.nusers = self.user_mapping_.shape[0]
            self.nitems = self.item_mapping_.shape[0]
            self._ratings['UserId'] =  self._ratings.UserId.values.astype('int32')
            self._ratings['ItemId'] =  self._ratings.ItemId.values.astype('int32')
            if 'Weight' in self._ratings.columns.values:
                self._ratings['Weight'] = self._ratings.Weight.astype('float32')

            if item_info is not None:

                if isinstance(item_info, np.ndarray):
                    item_info = pd.DataFrame(item_info)
                    item_info.columns[0] = 'ItemId'
                if item_info.__class__.__name__ != 'DataFrame':
                    raise ValueError("'item_info' must be a pandas data frame with a column named 'ItemId'.")
                if 'ItemId' not in item_info.columns.values:
                    raise ValueError("'item_info' must be a pandas data frame with a column named 'ItemId'.")

                self.delete_iteminfo = False
                item_info_id_orig = item_info.ItemId.values.copy()
                cat_ids = pd.Categorical(item_info.ItemId, self.item_mapping_)

                ## there can be new IDs in this sample
                ids_new = cat_ids.isnull()
                if ids_new.sum() > 0:
                    new_ids = np.unique(item_info.ItemId.values[ids_new]).reshape(-1)
                    self.item_mapping_ = np.r_[self.item_mapping_, new_ids]
                    item_info['ItemId'] = pd.Categorical(item_info.ItemId, self.item_mapping_).codes
                    if self.item_mapping_.shape[0] == (self.nitems + item_info.shape[0]):
                        warnings.warn("'item_info' contains no users in common with 'ratings'.")
                        self.delete_iteminfo = True
                        self.item_mapping_ = self.item_mapping_[:self.nitems]
                    else:
                        self.nitems = self.item_mapping_.shape[0]
                else:
                    item_info['ItemId'] = cat_ids.codes

                item_info = item_info.loc[~item_info.ItemId.duplicated(keep='last')]
                if (item_info.shape[0] == self.nitems) and ((item_info.ItemId.max()+1) == item_info.shape[0]):
                    item_info = item_info.sort_values('ItemId')
                    self._item_arr_mismatch = False
                else:
                    if self.offsets_model:
                        ## as of tensorflow 1.8.0, it throws a bug if I try scatter_add updates,
                        ## so here I'll fill all the missing rows
                        filled_ids = np.setdiff1d(np.arange(self.nitems), item_info.ItemId)
                        cols_fill = np.setdiff1d(item_info.columns.values, np.array(['ItemId']))
                        filled_df = np.repeat(np.nan, filled_ids.shape[0]*cols_fill.shape[0]).reshape((filled_ids.shape[0],cols_fill.shape[0]))
                        filled_df = pd.DataFrame(filled_df, columns=cols_fill)
                        filled_df['ItemId'] = filled_ids
                        filled_df = filled_df[item_info.columns]
                        item_info = item_info.append(filled_df, ignore_index=True, sort=False)
                        item_info.sort_values('ItemId', inplace=True)
                        self._item_arr_mismatch = False
                    else:
                        self._item_arr_mismatch = True
                        self._slc_item = item_info.ItemId.values.astype('int32')

                if (cols_bin_item is not None) and (not self.offsets_model):
                    cols_bin_item = np.intersect1d(cols_bin_item, item_info.columns.values)
                    cols_bin_item = np.setdiff1d(cols_bin_item, np.array(['ItemId']))
                    if cols_bin_item.shape[0] == 0:
                        warnings.warn("'cols_bin_item' doesn't contain any valid column from 'item_info'.")
                        self.cols_bin_item = None
                    else:
                        self.cols_bin_item = cols_bin_item
                        self._cols_nonbin_item = np.setdiff1d(item_info.columns.values, np.r_[np.array(cols_bin_item), np.array(['ItemId'])])
                        self._item_arr_bin = item_info[cols_bin_item].values.astype('float32')
                        self._item_cols_orig = np.setdiff1d(item_info.columns.values, np.array(['ItemId']))
                        item_info = item_info[np.r_[self._cols_nonbin_item, np.array(['ItemId'])]]
                else:

                    self.cols_bin_item = None
                    self._item_arr_bin = None

                del item_info['ItemId']
                self._item_arr = item_info.values
                self.item_dim = item_info.shape[1]
            
            else:
                self._set_null_iteminfo()

            if user_info is not None:
                if isinstance(user_info, np.ndarray):
                    user_info = pd.DataFrame(user_info)
                    user_info.columns[0] = 'ItemId'
                if user_info.__class__.__name__ != 'DataFrame':
                    raise ValueError("'user_info' must be a pandas data frame with a column named 'UserId'.")
                if 'UserId' not in user_info.columns.values:
                    raise ValueError("'user_info' must be a pandas data frame with a column named 'UserId'.")

                self.delete_userinfo = False
                user_info_id_orig = user_info.UserId.values.copy()
                ## there can be new IDs in this sample
                cat_ids = pd.Categorical(user_info.UserId, self.user_mapping_)
                ids_new = cat_ids.isnull()
                if ids_new.sum() > 0:
                    new_ids = np.unique(user_info.UserId.values[ids_new]).reshape(-1)
                    self.user_mapping_ = np.r_[self.user_mapping_, new_ids]
                    user_info['UserId'] = pd.Categorical(user_info.UserId, self.user_mapping_).codes
                    if self.user_mapping_.shape[0] == (self.nusers + user_info.shape[0]):
                        warnings.warn("'user_info' contains no users in common with 'ratings'.")
                        self.delete_userinfo = True
                        self.user_mapping_ = self.user_mapping_[:self.nusers]
                    else:
                        self.nusers = self.user_mapping_.shape[0]
                else:
                    user_info['UserId'] = cat_ids.codes

                user_info = user_info.loc[~user_info.UserId.duplicated(keep='last')]
                if (user_info.shape[0] == self.nusers) and ((user_info.UserId.max()+1) == user_info.shape[0]):
                    user_info = user_info.sort_values('UserId')
                    self._user_arr_mismatch = False
                else:
                    if self.offsets_model:
                        ## as of tensorflow 1.8.0, it throws a bug if I try scatter_add updates,
                        ## so here I'll fill all the missing rows
                        filled_ids = np.setdiff1d(np.arange(self.nusers), user_info.UserId)
                        cols_fill = np.setdiff1d(user_info.columns.values, np.array(['UserId']))
                        filled_df = np.repeat(np.nan, filled_ids.shape[0]*cols_fill.shape[0]).reshape((filled_ids.shape[0],cols_fill.shape[0]))
                        filled_df = pd.DataFrame(filled_df, columns=cols_fill)
                        filled_df['UserId'] = filled_ids
                        filled_df = filled_df[user_info.columns]
                        user_info = user_info.append(filled_df, ignore_index=True, sort=False)
                        user_info.sort_values('UserId', inplace=True)
                        self._user_arr_mismatch = False
                    else:
                        self._user_arr_mismatch = True
                        self._slc_user = user_info.UserId.values.astype('int32')

                if (cols_bin_user is not None) and (not self.offsets_model):
                    cols_bin_user = np.intersect1d(cols_bin_user, user_info.columns.values)
                    cols_bin_user = np.setdiff1d(cols_bin_user, np.array(['UserId']))
                    if cols_bin_user.shape[0] == 0:
                        warnings.warn("'cols_bin_user' doesn't contain any valid column from 'user_info'.")
                        self.cols_bin_user = None
                    else:
                        self.cols_bin_user = cols_bin_user
                        self._cols_nonbin_user = np.setdiff1d(user_info.columns, np.r_[np.array(cols_bin_user), np.array(['UserId'])])
                        self._user_arr_bin = user_info[cols_bin_user].values.astype('float32')
                        self._user_cols_orig = np.setdiff1d(user_info.columns.values, np.array(['UserId']))
                        user_info = user_info[np.r_[self._cols_nonbin_user, np.array(['UserId'])]]
                else:
                    self.cols_bin_user = None
                    self._user_arr_bin = None

                del user_info['UserId']
                self._user_arr = user_info.values
                self.user_dim = user_info.shape[1]
            
            else:
                self._set_null_userinfo()


            if (self.save_folder is not None) and self.reindex:
                pd.Series(self.user_mapping_).to_csv(os.path.join(self.save_folder, 'users.csv'), index=False)
                pd.Series(self.item_mapping_).to_csv(os.path.join(self.save_folder, 'items.csv'), index=False)

        
        if self.center_ratings:
            self.global_mean = self._ratings.Rating.mean()
            self._ratings['Rating'] -= self.global_mean

        if self.add_user_bias:
            self.init_u_bias = np.zeros(self.nusers, dtype='float32')
            user_avg = self._ratings.groupby('UserId', sort=True)['Rating'].mean()
            self.init_u_bias[user_avg.index] = user_avg.values
            del user_avg

        if self.add_item_bias:
            self.init_i_bias = np.zeros(self.nitems, dtype='float32')
            item_avg = self._ratings.groupby('ItemId', sort=True)['Rating'].mean()
            self.init_i_bias[item_avg.index] = item_avg.values
            del item_avg

        self._ix_u = self._ratings.UserId.values
        self._ix_i = self._ratings.ItemId.values
        if 'Weight' in self._ratings.columns.values:
            self._weights = self._ratings.Weight.values
        else:
            self._weights = None
        if self.keep_data:
            self._store_metadata()
        self._ratings = self._ratings.Rating.values


        if self.save_folder is not None:
            with open(os.path.join(self.save_folder, "hyperparameters.txt"), "w") as pf:
                pf.write("k: %d\n" % self.k)
                pf.write("k_main: %d\n" % self.k_main)
                pf.write("k_item: %d\n" % self.k_item)
                pf.write("k_user: %d\n" % self.k_user)
                pf.write("regA: %.3f\n" % self.reg_param[0])
                pf.write("regB: %.3f\n" % self.reg_param[1])
                pf.write("reg_user_bias: %.3f\n" % self.reg_param[2])
                pf.write("reg_item_bias: %.3f\n" % self.reg_param[3])
                pf.write("regC: %.3f\n" % self.reg_param[4])
                pf.write("regD: %.3f\n" % self.reg_param[5])
                if self.random_seed is not None:
                    pf.write("random seed: %d\n" % self.random_seed)
                else:
                    pf.write("random seed: None\n")

        
        if self._item_arr is not None:
            if self.delete_iteminfo:
                self._set_null_iteminfo()
        if self._user_arr is not None:
            if self.delete_userinfo:
                self._set_null_userminfo()

        
        if self._item_arr is not None:
            self._item_arr = self._item_arr.astype('float32')
            item_arr_missing = np.isnan(self._item_arr)
            if np.sum(item_arr_missing) == 0:
                self._item_arr_notmissing = None
            else:
                self._item_arr[item_arr_missing] = 0
                self._item_arr_notmissing = (~item_arr_missing).astype('uint8')
                del item_arr_missing

            if self.center_item_info:
                self.item_arr_means = self._item_arr.mean(axis=0, keepdims=True)
                self._item_arr -= self.item_arr_means

            if (self.cols_bin_item is not None) and (not self.offsets_model):
                item_arr_missing_bin = np.isnan(self._item_arr_bin)
                if np.sum(item_arr_missing_bin) == 0:
                    self._item_arr_notmissing_bin = None
                else:
                    self._item_arr_bin[item_arr_missing_bin] = 0
                    self._item_arr_notmissing_bin = (~item_arr_missing_bin).astype('uint8')
                    del item_arr_missing_bin
            else:
                self._item_arr_notmissing_bin = None
        else:
            self._item_arr_notmissing = None
            self._item_arr_notmissing_bin = None

        if self._user_arr is not None:
            self._user_arr = self._user_arr.astype('float32')
            user_arr_missing = np.isnan(self._user_arr)
            if np.sum(user_arr_missing) == 0:
                self._user_arr_notmissing = None
            else:
                self._user_arr[user_arr_missing] = 0
                self._user_arr_notmissing = (~user_arr_missing).astype('uint8')
                del user_arr_missing

            if self.center_user_info:
                self.user_arr_means = self._user_arr.mean(axis=0, keepdims=True)
                self._user_arr -= self.user_arr_means

            if (self.cols_bin_user is not None) and (not self.offsets_model):
                user_arr_missing_bin = np.isnan(self._user_arr_bin)
                if np.sum(user_arr_missing_bin) == 0:
                    self._user_arr_notmissing_bin = None
                else:
                    self._user_arr_bin[user_arr_missing_bin] = 0
                    self._user_arr_notmissing_bin = (~user_arr_missing_bin).astype('uint8')
                    del user_arr_missing_bin
            else:
                self._user_arr_notmissing_bin = None
        else:
            self._user_arr_notmissing = None
            self._user_arr_notmissing_bin = None

    def _store_metadata(self):
        self.seen = self._ratings[['UserId', 'ItemId']].copy()
        self.seen.sort_values(['UserId', 'ItemId'], inplace=True)
        self.seen.reset_index(drop = True, inplace = True)
        self._n_seen_by_user = self.seen.groupby('UserId')['ItemId'].agg(lambda x: len(x)).values
        self._st_ix_user = np.cumsum(self._n_seen_by_user)
        self._st_ix_user = np.r_[[0], self._st_ix_user[:self._st_ix_user.shape[0]-1]]
        self.seen = self.seen.ItemId.values
        return None
            
    def _set_weights(self, random_seed):

        if self.reweight and (not self.offsets_model):
            # this initializes all parameters at random, calculates the error, and scales weights by them
            
            if random_seed is not None:
                np.random.seed(random_seed)
            
            A = np.random.normal(size = (self.nusers, self.k_main + self.k))
            B = np.random.normal(size = (self.k_main + self.k, self.nitems))
            pred = (A[self._ix_u], B[self._ix_i]).sum(axis=1)
            err = pred - self._ratings
            err_main = 1 / np.mean(err ** 2)
            err_tot = err_main
            del A, B, pred, err
            
            if self._user_arr is not None:
                A = np.random.normal(size = (self._user_arr.shape[0], self.k + self.k_user))
                C = np.random.normal(size = (self.k + self.k_user, self._user_arr.shape[1]))
                pred = A.dot(C)
                if self._user_arr_notmissing is not None:
                    pred *= self._user_arr_notmissing
                err = self._user_arr - pred
                err_user = 1 / np.mean(err ** 2)
                err_tot += err_user
                del A, C, pred, err

            if self._item_arr is not None:
                B = np.random.normal(size = (self._item_arr.shape[0],  self.k + self.k_item))
                D = np.random.normal(size = (self.k + self.k_item, self._item_arr.shape[1]))
                pred = B.dot(D)
                if self._item_arr_notmissing is not None:
                    pred *= self._item_arr_notmissing
                err = self._item_arr - pred
                err_item = 1 / np.mean(err ** 2)
                err_tot += err_item
                del B, D, pred, err
                
            ## Now with these errors given by random matrices, the weights are calculated to sum to 1
            ## and multiplied by w_main/user/item if given
            self.w1 = self.w_main * err_main / err_tot
            if self._user_arr is not None:
                self.w2 = self.w_user * err_user / err_tot
            else:
                self.w2 = 0.0
            if self._item_arr is not None:
                self.w3 = self.w_item * err_item / err_tot
            else:
                self.w3 = 0.0
                        
        else:
            # if no option is passed, weights are taken at face value
            self.w1 = self.w_main
            self.w2 = self.w_user
            self.w3 = self.w_item

            # these weights are amplified to account for missing entries, as it will later
            # take the mean of the losses, which will be zero for missing ones.
            if not self.offsets_model:
                if self._user_arr_notmissing is not None:
                    N = self._user_arr.shape[0] * self._user_arr.shape[1]
                    self.w2 *= N / (N - self._user_arr_notmissing.sum().sum())
                if self._item_arr_notmissing is not None:
                    N = self._item_arr.shape[0]*self._item_arr.shape[1]
                    self.w3 *= N / (N - self._item_arr_notmissing.sum().sum())
        
    def predict(self, user, item):
        """
        Predict ratings for combinations of users and items
        
        Note
        ----
        You can either pass an individual user and item, or arrays representing
        tuples (UserId, ItemId) with the combinatinons of users and items for which
        to predict (one row per prediction).

        Note
        ----
        If you pass any user/item which was not in the training set, the prediction
        for it will be NaN.

        Parameters
        ----------
        user : array-like (npred,) or obj
            User(s) for which to predict each item.
        item: array-like (npred,) or obj
            Item(s) for which to predict for each user.
        """
        assert self.is_fitted
        if isinstance(user, list) or isinstance(user, tuple):
            user = np.array(user)
        if isinstance(item, list) or isinstance(item, tuple):
            item = np.array(item)
        if user.__class__.__name__=='Series':
            user = user.values
        if item.__class__.__name__=='Series':
            item = item.values
            
        if isinstance(user, np.ndarray):
            if len(user.shape) > 1:
                user = user.reshape(-1)
            assert user.shape[0] > 0
            if self.reindex:
                if user.shape[0] > 1:
                    user = pd.Categorical(user, self.user_mapping_).codes
                else:
                    if self.user_dict_ is not None:
                        try:
                            user = self.user_dict_[user]
                        except:
                            user = -1
                    else:
                        user = pd.Categorical(user, self.user_mapping_).codes[0]
        else:
            if self.reindex:
                if self.user_dict_ is not None:
                    try:
                        user = self.user_dict_[user]
                    except:
                        user = -1
                else:
                    user = pd.Categorical(np.array([user]), self.user_mapping_).codes[0]
            user = np.array([user])
            
        if isinstance(item, np.ndarray):
            if len(item.shape) > 1:
                item = item.reshape(-1)
            assert item.shape[0] > 0
            if self.reindex:
                if item.shape[0] > 1:
                    item = pd.Categorical(item, self.item_mapping_).codes
                else:
                    if self.item_dict_ is not None:
                        try:
                            item = self.item_dict_[item]
                        except:
                            item = -1
                    else:
                        item = pd.Categorical(item, self.item_mapping_).codes[0]
        else:
            if self.reindex:
                if self.item_dict_ is not None:
                    try:
                        item = self.item_dict_[item]
                    except:
                        item = -1
                else:
                    item = pd.Categorical(np.array([item]), self.item_mapping_).codes[0]
            item = np.array([item])

        assert user.shape[0] == item.shape[0]
        
        if user.shape[0] == 1:
            if (user[0] == -1) or (item[0] == -1):
                return np.nan
            else:
                out = self._Ab[user].dot(self._Ba[item].T).reshape(-1)[0]
                if self.center_ratings:
                    out += self.global_mean
                if self.add_user_bias:
                    out += self.user_bias[user]
                if self.add_item_bias:
                    out += self.item_bias[item]
                if isinstance(out, np.ndarray):
                    out = out[0]
                return out
        else:
            nan_entries = (user == -1) | (item == -1)
            if nan_entries.sum() == 0:
                out = (self._Ab[user] * self._Ba[item]).sum(axis=1)
                if self.center_ratings:
                    out += self.global_mean
                if self.add_user_bias:
                    out += self.user_bias[user]
                if self.add_item_bias:
                    out += self.item_bias[item]
                return out
            else:
                non_na_user = user[~nan_entries]
                non_na_item = item[~nan_entries]
                out = np.empty(user.shape[0], dtype=self._Ab.dtype)
                out[~nan_entries] = (self._Ab[non_na_user] * self._Ba[non_na_item]).sum(axis=1)
                if self.center_ratings:
                    out += self.global_mean
                if self.add_user_bias:
                    out += self.user_bias[user]
                if self.add_item_bias:
                    out += self.item_bias[item]
                out[nan_entries] = np.nan
                return out
        
    
    def topN(self, user, n=10, exclude_seen=True, items_pool=None):
        """
        Recommend Top-N items for a user

        Outputs the Top-N items according to score predicted by the model.
        Can exclude the items for the user that were associated to her in the
        training set, and can also recommend from only a subset of user-provided items.

        Parameters
        ----------
        user : obj
            User for which to recommend.
        n : int
            Number of top items to recommend.
        exclude_seen: bool
            Whether to exclude items that were associated to the user in the training set.
        items_pool: None or array
            Items to consider for recommending to the user.
        
        Returns
        -------
        rec : array (n,)
            Top-N recommended items.
        """
        if isinstance(n, float):
            n = int(n)
        assert isinstance(n ,int)
        if self.reindex:
            if self.produce_dicts:
                try:
                    user = self.user_dict_[user]
                except:
                    raise ValueError("Can only predict for users who were in the training set.")
            else:
                user = pd.Categorical(np.array([user]), self.user_mapping_).codes[0]
                if user == -1:
                    raise ValueError("Can only predict for users who were in the training set.")
        if exclude_seen and not self.keep_data:
            raise Exception("Can only exclude seen items when passing 'keep_data=True' to .fit")
            
        if items_pool is None:
            allpreds = - (self._Ab[user].dot(self._Ba.T))
            if self.add_item_bias:
                allpreds -= self.item_bias
            if exclude_seen:
                if user < self._n_seen_by_user.shape[0]:
                    n_seen_by_user = self._n_seen_by_user[user]
                    st_ix_user = self._st_ix_user[user]
                else:
                    n_seen_by_user = 0
                    st_ix_user = 0
                n_ext = np.min([n + n_seen_by_user, self._Ba.shape[0]])
                rec = np.argpartition(allpreds, n_ext-1)[:n_ext]
                seen = self.seen[st_ix_user : st_ix_user + n_seen_by_user]
                rec = np.setdiff1d(rec, seen)
                rec = rec[np.argsort(allpreds[rec])[:n]]
                if self.reindex:
                    return self.item_mapping_[rec]
                else:
                    return rec

            else:
                n = np.min([n, self._Ba.shape[0]])
                rec = np.argpartition(allpreds, n-1)[:n]
                rec = rec[np.argsort(allpreds[rec])]
                if self.reindex:
                    return self.item_mapping_[rec]
                else:
                    return rec

        else:
            if isinstance(items_pool, list) or isinstance(items_pool, tuple):
                items_pool = np.array(items_pool)
            if items_pool.__class__.__name__=='Series':
                items_pool = items_pool.values
            if isinstance(items_pool, np.ndarray):  
                if len(items_pool.shape) > 1:
                    items_pool = items_pool.reshape(-1)
                if self.reindex:
                    items_pool_reind = pd.Categorical(items_pool, self.item_mapping_).codes
                    nan_ix = (items_pool_reind == -1)
                    if nan_ix.sum() > 0:
                        items_pool_reind = items_pool_reind[~nan_ix]
                        msg = "There were " + ("%d" % int(nan_ix.sum())) + " entries from 'item_pool'"
                        msg += "that were not in the training data and will be exluded."
                        warnings.warn(msg)
                    del nan_ix
                    if items_pool_reind.shape[0] == 0:
                        raise ValueError("No items to recommend.")
                    elif items_pool_reind.shape[0] == 1:
                        raise ValueError("Only 1 item to recommend.")
                    else:
                        pass
            else:
                raise ValueError("'items_pool' must be an array.")

            if self.reindex:
                allpreds = - self._Ab[user].dot(self._Ba[items_pool_reind].T)
                if self.add_item_bias:
                    allpreds -= self.item_bias[items_pool_reind]
            else:
                allpreds = - self._Ab[user].dot(self._Ba[items_pool].T)
                if self.add_item_bias:
                    allpreds -= self.item_bias[items_pool]
            n = np.min([n, items_pool.shape[0]])
            if exclude_seen:
                if user < self._n_seen_by_user.shape[0]:
                    n_seen_by_user = self._n_seen_by_user[user]
                    st_ix_user = self._st_ix_user[user]
                else:
                    n_seen_by_user = 0
                    st_ix_user = 0
                n_ext = np.min([n + n_seen_by_user, items_pool.shape[0]])
                rec = np.argpartition(allpreds, n_ext-1)[:n_ext]
                seen = self.seen[st_ix_user : st_ix_user + n_seen_by_user]
                if self.reindex:
                    rec = np.setdiff1d(items_pool_reind[rec], seen)
                    allpreds = - self._Ab[user].dot(self._Ba[rec].T)
                    if self.add_item_bias:
                        allpreds -= self.item_bias[rec]
                    return self.item_mapping_[rec[np.argsort(allpreds)[:n]]]
                else:
                    rec = np.setdiff1d(items_pool[rec], seen)
                    allpreds = - self._Ab[user].dot(self._Ba[rec].T)
                    if self.add_item_bias:
                        allpreds -= self.item_bias[rec]
                    return rec[np.argsort(allpreds)[:n]]
            else:
                rec = np.argpartition(allpreds, n-1)[:n]
                return items_pool[rec[np.argsort(allpreds[rec])]]

    def add_user(self, new_id, attributes, reg='auto'):
        """
        Adds a new user vector according to its attributes, in order to make predictions for her

        In the regular collective factorization model without non-negativity constraints and without binary
        columns, will calculate the latent factors vector by its closed-form solution, which is fast. In the offsets
        model, the latent factors vector is obtained by a simple matrix product, so it will be even faster. However,
        if there are non-negativity constraints and/or binary columns, there is no closed form solution,
        and it will be calculated via gradient-based optimization, so it will take longer and shouldn't be
        expected to work in 'real time'.

        Note
        ----
        For better quality cold-start recommendations, center your ratings data, use high regularization,
        assign large weights to the factorization of side information, and don't use large values for number
        of latent factors that are specific for some factorization.

        Note
        ----
        If you pass and ID that is of a different type (str, int, obj, etc.) than the IDs of the data that
        was passed to .fit, the internal indexes here might break and some of the prediction functionality
        might stop working. Be sure to pass IDs of the same type. The type of the ID will be forcibly
        converted to try to avoid this, but you might still run into problems.

        Parameters
        ----------
        new_id : obj
            ID of the new user. Ignored when called with 'reindex=False', in which case it will assign it
            ID = nusers_train + 1.
        attributes : array (user_dim, )
            Attributes of this user (side information)
        reg : float or str 'auto'
            Regularization parameter for these new attributes. If set to 'auto', will use the same regularization
            parameter that was set for the user-factor matrix.

        Returns
        -------
        Success : bool
            Returns true if the operation completes successfully
        """

        ## TODO: `add_user`, `add_item`, `topN_cold` need refactoring to be more modular

        assert self.is_fitted
        if self.C is None:
            raise ValueError("Can only add users if model was fit to user side information.")
        if self.produce_dicts:
            if new_id in self.user_dict_:
                raise ValueError("User ID is already in the model.")
        else:
            if new_id in self.user_mapping_:
                raise ValueError("User ID is already in the model.")

        if attributes.__class__.__name__ == 'DataFrame':
            attributes = attributes[self._user_cols_orig]
            attributes = attributes.values
        if attributes.__class__.__name__ == 'Series':
            attributes = attributes.loc[self._user_cols_orig]
            attributes = attributes.values
        assert isinstance(attributes, np.ndarray)
        attributes = attributes.reshape(-1)
        if self.offsets_model:
            assert attributes.shape[0] == self.C.shape[0]
        elif self.cols_bin_user is None:
            assert attributes.shape[0] == self.C.shape[1]
        else:
            assert attributes.shape[0] == self.C.shape[1] + self.C_bin.shape[1]
        attributes = attributes.astype(self.A.dtype)
        if reg == 'auto':
            reg = self.reg_param[0]
        if isinstance(reg, int):
            reg = float(reg)
        assert isinstance(reg, float)

        if (self.cols_bin_user is not None) and (not self.offsets_model):
            attributes_bin = attributes[np.in1d(self._user_cols_orig, self.cols_bin_user)].copy()
            attributes = attributes[np.in1d(self._user_cols_orig, self._cols_nonbin_user)]

        if self.center_user_info:
            attributes -= self.user_arr_means

        if self.offsets_model:
            user_vec = attributes.reshape(1,-1).dot(self.C).astype(self.A.dtype)
            self.A = np.r_[self.A, user_vec.reshape(1, -1)]
            self._Ab = np.r_[self._Ab, user_vec.reshape(1, -1)]
        else:
            user_vec = np.zeros(self.k_main + self.k + self.k_user, dtype=self.A.dtype)
            if (self.cols_bin_user is None) and (not self.nonnegative):
                if self.standardize_err:
                    reg *= self.k + self.k_user
                user_vec[self.k_main:] = np.linalg.solve(self.C.dot(self.C.T) + np.diag(np.repeat(reg,self.k + self.k_user)), self.C.dot(attributes))
            else:
                Arow = tf.Variable(tf.zeros([1, self.k + self.k_user]))
                Ctf = tf.placeholder(tf.float32)
                Cbin_tf = tf.placeholder(tf.float32)
                attr_num = tf.placeholder(tf.float32)
                attr_bin = tf.placeholder(tf.float32)
                if self.standardize_err:
                    loss  = tf.losses.mean_squared_error(tf.matmul(Arow, Ctf), attr_num)
                    if self.C_bin is not None:
                        loss += tf.losses.mean_squared_error(tf.sigmoid(tf.matmul(Arow, Cbin_tf)), attr_bin)
                    else:
                        attributes_bin = None
                    loss += reg*tf.nn.l2_loss(Arow)
                else:
                    loss  = tf.nn.l2_loss(tf.matmul(Arow, Ctf) - attr_num)
                    if self.cols_user_bin is not None:
                        loss += tf.nn.l2_loss(tf.sigmoid(tf.matmul(Arow, Cbin_tf)) - attr_bin)
                    loss += reg * tf.nn.l2_loss(Arow)
                
                opts_lbfgs = {'iprint':-1, 'disp':0}
                if self.nonnegative:
                    dct_bound = {Arow:(0, np.inf)}
                else:
                    dct_bound = dict()
                optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B', options=opts_lbfgs, var_to_bounds=dct_bound)
                model = tf.global_variables_initializer()
                sess = tf.Session()
                sess.run(model)
                with sess:
                    tf.logging.set_verbosity(tf.logging.WARN)
                    optimizer.minimize(sess, feed_dict={Ctf:self.C, Cbin_tf:self.C_bin,
                                            attr_num:attributes, attr_bin:attributes_bin})
                    user_vec[self.k_main:] = Arow.eval(session=sess).reshape(-1)

            self.A = np.r_[self.A, user_vec.reshape(1, -1)]
            self._Ab = self.A[:, :self.k_main + self.k]


        if self.reindex:
            self.user_mapping_ = np.r_[self.user_mapping_, np.array([new_id])]
            if self.produce_dicts:
                self.user_dict_[new_id] = self.user_mapping_.shape[0] - 1

        # if self.keep_data:
        #     self._st_ix_user = np.r_[self._st_ix_user, -1]
        #     self._n_seen_by_user = np.r_[self._n_seen_by_user, 0]

        if self.add_user_bias:
            self.user_bias = np.r_[self.user_bias, np.zeros(1, dtype=self.user_bias.dtype)]

        return True

    def add_item(self, new_id, attributes, reg='auto'):
        """
        Adds a new item vector according to its attributes, in order to make predictions for it

        In the regular collective factorization model without non-negativity constraints and without binary
        columns, will calculate the latent factors vector by its closed-form solution, which is fast. In the offsets
        model, the latent factors vector is obtained by a simple matrix product, so it will be even faster. However,
        if there are non-negativity constraints and/or binary columns, there is no closed form solution,
        and it will be calculated via gradient-based optimization, so it will take longer and shouldn't be
        expected to work in 'real time'.

        Note
        ----
        For better quality cold-start recommendations, center your ratings data, use high regularization,
        assign large weights to the factorization of side information, and don't use large values for number
        of latent factors that are specific for some factorization.

        Note
        ----
        If you pass and ID that is of a different type (str, int, obj, etc.) than the IDs of the data that
        was passed to .fit, the internal indexes here might break and some of the prediction functionality
        might stop working. Be sure to pass IDs of the same type. The type of the ID will be forcibly
        converted to try to avoid this, but you might still run into problems.

        Parameters
        ----------
        new_id : obj
            ID of the new item. Ignored when called with 'reindex=False', in which case it will assign it
            ID = nitems_train + 1.
        attributes : array (item_dim, )
            Attributes of this item (side information)
        reg : float or str 'auto'
            Regularization parameter for these new attributes. If set to 'auto', will use the same regularization
            parameter that was set for the item-factor matrix.

        Returns
        -------
        Success : bool
            Returns true if the operation completes successfully
        """

        ## TODO: `add_user`, `add_item`, `topN_cold` need refactoring to be more modular

        assert self.is_fitted
        if self.D is None:
            raise ValueError("Can only add items if model was fit to item side information.")
        if self.produce_dicts:
            if new_id in self.item_dict_:
                raise ValueError("Item ID is already in the model.")
        else:
            if new_id in self.item_mapping_:
                raise ValueError("Item ID is already in the model.")

        if attributes.__class__.__name__ == 'DataFrame':
            attributes = attributes.values
        if attributes.__class__.__name__ == 'Series':
            attributes = attributes.loc[self._item_cols_orig]
            attributes = attributes.values
        assert isinstance(attributes, np.ndarray)
        attributes = attributes.reshape(-1)
        if self.offsets_model:
            assert attributes.shape[0] == self.D.shape[0]
        elif self.cols_bin_item is None:
            assert attributes.shape[0] == self.D.shape[1]
        else:
            assert attributes.shape[0] == self.D.shape[1] + self.D_bin.shape[1]
        attributes = attributes.astype(self.B.dtype)
        if reg == 'auto':
            reg = self.reg_param[1]
        if isinstance(reg, int):
            reg = float(reg)
        assert isinstance(reg, float)

        if (self.cols_bin_item is not None) and (not self.offsets_model):
            attributes_bin = attributes[np.in1d(self._item_cols_orig, self.cols_bin_item)].copy()
            attributes = attributes[np.in1d(self._item_cols_orig, self._cols_nonbin_item)]

        if self.center_item_info:
            attributes -= self.item_arr_means

        if self.offsets_model:
            item_vec = attributes.reshape(1,-1).dot(self.D).astype(self.B.dtype)
            self.B = np.r_[self.B, item_vec.reshape(1, -1)]
            self._Ba = np.r_[self._Ba, item_vec.reshape(1, -1)]
        else:
            item_vec = np.zeros(self.k_main + self.k + self.k_item, dtype=self.B.dtype)
            if self.cols_bin_item is None:
                if self.standardize_err:
                    reg *= self.k + self.k_item
                item_vec[self.k_main:] = np.linalg.solve(self.D.dot(self.D.T) + np.diag(np.repeat(reg, self.k + self.k_item)), self.D.dot(attributes))
            else:
                Brow = tf.Variable(tf.zeros([1, self.k + self.k_item]))
                Dtf = tf.placeholder(tf.float32)
                Dbin_tf = tf.placeholder(tf.float32)
                attr_num = tf.placeholder(tf.float32)
                attr_bin = tf.placeholder(tf.float32)
                if self.standardize_err:
                    loss  = tf.losses.mean_squared_error(tf.matmul(Brow, Dtf), attr_num)
                    if self.D_bin is not None:
                        loss += tf.losses.mean_squared_error(tf.sigmoid(tf.matmul(Brow, Dbin_tf)), attr_bin)
                    else:
                        attributes_bin = None
                    loss += reg*tf.nn.l2_loss(Brow)
                else:
                    loss  = tf.nn.l2_loss(tf.matmul(Brow, Dtf) - attr_num)
                    if self.cols_user_bin is not None:
                        loss += tf.nn.l2_loss(tf.sigmoid(tf.matmul(Brow, Dbin_tf)) - attr_bin)
                    loss += reg * tf.nn.l2_loss(Brow)
                
                opts_lbfgs = {'iprint':-1, 'disp':0}
                if self.nonnegative:
                    dct_bound = {Brow:(0, np.inf)}
                else:
                    dct_bound = dict()
                optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B', options=opts_lbfgs, var_to_bounds=dct_bound)
                model = tf.global_variables_initializer()
                sess = tf.Session()
                sess.run(model)
                with sess:
                    optimizer.minimize(sess, feed_dict={Dtf:self.D, Dbin_tf:self.D_bin,
                                            attr_num:attributes, attr_bin:attributes_bin})
                    item_vec[self.k_main:] = Brow.eval(session=sess).reshape(-1)

            self.B = np.r_[self.B, item_vec.reshape(1, -1)]
            self._Ba = self.B[:, :self.k_main + self.k]

        if self.reindex:
            self.item_mapping_ = np.r_[self.item_mapping_, np.array([new_id])]
            if self.produce_dicts:
                self.item_dict_[new_id] = self.item_mapping_.shape[0] - 1

        if self.add_item_bias:
            self.item_bias = np.r_[self.item_bias, np.zeros(1, dtype=self.item_bias.dtype)]

        return True

    def topN_cold(self, attributes, n=10, reg='auto', items_pool=None):
        """
        Recommend Top-N items for a user that was not in the training set.

        In the regular collective factorization model without non-negativity constraints and without binary
        columns, will calculate the latent factors vector by its closed-form solution, which is fast. In the offsets
        model, the latent factors vector is obtained by a simple matrix product, so it will be even faster. However,
        if there are non-negativity constraints and/or binary columns, there is no closed form solution,
        and it will be calculated via gradient-based optimization, so it will take longer and shouldn't be
        expected to work in 'real time'.

        Note
        ----
        For better quality cold-start recommendations, center your ratings data, use high regularization,
        assign large weights to the factorization of side information, and don't use large values for number
        of latent factors that are specific for some factorization.

        Parameters
        ----------
        attributes : array (user_dim, )
            Attributes of the user. Columns must be in the same order as was passed to '.fit', but without the ID column.
        n : int
            Number of top items to recommend.
        reg : float or str 'auto'
            Regularization parameter for these new attributes. If set to 'auto', will use the same regularization
            parameter that was set for the user-factor matrix.
        items_pool: None or array
            Items to consider for recommending to the user.
        
        Returns
        -------
        rec : array (n,)
            Top-N recommended items.
        """

        ## TODO: `add_user`, `add_item`, `topN_cold` need refactoring to be more modular

        assert self.is_fitted
        if self.C is None:
            raise ValueError("Can only add users if model was fit to user side information.")
        if attributes.__class__.__name__ == 'DataFrame':
            attributes = attributes.values
        assert isinstance(attributes, np.ndarray)
        attributes = attributes.reshape(-1)
        if self.offsets_model:
            assert attributes.shape[0] == self.C.shape[0]
        elif self.cols_bin_user is None:
            assert attributes.shape[0] == self.C.shape[1]
        else:
            assert attributes.shape[0] == self.C.shape[1] + self.C_bin.shape[1]
        attributes = attributes.astype(self.A.dtype)
        
        if reg == 'auto':
            reg = self.reg_param[0]

        if isinstance(n, float):
            n = int(n)
        assert isinstance(n ,int)

        if (self.cols_bin_user is not None) and (not self.offsets_model):
            attributes_bin = attributes[np.in1d(self._user_cols_orig, self.cols_bin_user)].copy()
            attributes = attributes[np.in1d(self._user_cols_orig, self._cols_nonbin_user)]

        if self.center_user_info:
            attributes -= self.user_arr_means

        if self.offsets_model:
            user_vec = attributes.reshape(1,-1).dot(self.C).astype(self.A.dtype)
        else:
            if self.cols_bin_user is None:
                if self.standardize_err:
                    reg *= self.k + self.k_user
                user_vec = np.linalg.solve(self.C.dot(self.C.T) + np.diag(np.repeat(reg, self.k + self.k_user)), self.C.dot(attributes))
            else:
                Arow = tf.Variable(tf.zeros([1, self.k + self.k_user]))
                Ctf = tf.placeholder(tf.float32)
                Cbin_tf = tf.placeholder(tf.float32)
                attr_num = tf.placeholder(tf.float32)
                attr_bin = tf.placeholder(tf.float32)
                if self.standardize_err:
                    loss  = tf.losses.mean_squared_error(tf.matmul(Arow, Ctf), attr_num)
                    if self.C_bin is not None:
                        loss += tf.losses.mean_squared_error(tf.sigmoid(tf.matmul(Arow, Cbin_tf)), attr_bin)
                    else:
                        attributes_bin = None
                    loss += reg*tf.nn.l2_loss(Arow)
                else:
                    loss  = tf.nn.l2_loss(tf.matmul(Arow, Ctf) - attr_num)
                    if self.cols_user_bin is not None:
                        loss += tf.nn.l2_loss(tf.sigmoid(tf.matmul(Arow, Cbin_tf)) - attr_bin)
                    loss += reg * tf.nn.l2_loss(Arow)
                
                opts_lbfgs = {'iprint':-1, 'disp':0}
                if self.nonnegative:
                    dct_bound = {Arow:(0, np.inf)}
                else:
                    dct_bound = dict()
                optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B', options=opts_lbfgs, var_to_bounds=dct_bound)
                model = tf.global_variables_initializer()
                sess = tf.Session()
                sess.run(model)
                with sess:
                    optimizer.minimize(sess, feed_dict={Ctf:self.C, Cbin_tf:self.C_bin,
                                            attr_num:attributes, attr_bin:attributes_bin})
                    user_vec = Arow.eval(session=sess).reshape(-1)
            
        user_vec = -np.r_[np.zeros(self.k_main, dtype=user_vec.dtype), user_vec[:self.k]]

        if items_pool is None:
            allpreds = user_vec.dot(self._Ba.T)
            if self.add_item_bias:
                allpreds -= self.item_bias
            n = np.min([n, self._Ba.shape[0]])
            rec = np.argpartition(allpreds, n-1)[:n]
            rec = rec[np.argsort(allpreds[rec])]
            if self.reindex:
                return self.item_mapping_[rec]
            else:
                return rec

        else:
            if isinstance(items_pool, list) or isinstance(items_pool, tuple):
                items_pool = np.array(items_pool)
            if items_pool.__class__.__name__ =='Series':
                items_pool = items_pool.values
            if isinstance(items_pool, np.ndarray):  
                if len(items_pool.shape) > 1:
                    items_pool = items_pool.reshape(-1)
                if self.reindex:
                    items_pool_reind = pd.Categorical(items_pool, self.item_mapping_).codes
                    nan_ix = (items_pool_reind == -1)
                    if nan_ix.sum() > 0:
                        items_pool_reind = items_pool_reind[~nan_ix]
                        msg = "There were " + ("%d" % int(nan_ix.sum())) + " entries from 'item_pool'"
                        msg += "that were not in the training data and will be exluded."
                        warnings.warn(msg)
                    del nan_ix
                    if items_pool_reind.shape[0] == 0:
                        raise ValueError("No items to recommend.")
                    elif items_pool_reind.shape[0] == 1:
                        raise ValueError("Only 1 item to recommend.")
                    else:
                        pass
            else:
                raise ValueError("'items_pool' must be an array.")

            if self.reindex:
                allpreds = user_vec.dot(self._Ba[items_pool_reind].T)
                if self.add_item_bias:
                    allpreds -= self.item_bias[items_pool_reind]
            else:
                allpreds = user_vec.dot(self._Ba[items_pool].T)
                if self.add_item_bias:
                    allpreds -= self.item_bias[items_pool]
            n = np.min([n, items_pool.shape[0]])
            
            rec = np.argpartition(allpreds, n-1)[:n]
            return items_pool[rec[np.argsort(allpreds[rec])]]
