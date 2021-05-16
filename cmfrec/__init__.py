from . import wrapper_double, wrapper_float
import numpy as np, pandas as pd
import multiprocessing
import ctypes
import warnings

__all__ = ["CMF", "CMF_implicit",
           "OMF_explicit", "OMF_implicit",
           "MostPopular", "ContentBased"]

### TODO: this module should move from doing operations in Python to
### using the new designated C functions for each type of prediction.

class _CMF:
    def __repr__(self):
        return self.__str__()

    def set_params(self, **params):
        if not params:
            return self
        valid_params = self.get_params()
        for k,v in params.items():
            if k not in valid_params.keys():
                raise ValueError("Invalid parameter %s" % k)
            else:
                setattr(self, k, v)
        return self

    def _take_params(self, implicit=False, alpha=40., downweight=False,
                     apply_log_transf=False,
                     nonneg=False, nonneg_C=False, nonneg_D=False,
                     max_cd_steps=100,
                     k=50, lambda_=1e2, method="als", add_implicit_features=False,
                     scale_lam=False, scale_lam_sideinfo=False, scale_bias_const=False,
                     use_cg=False, max_cg_steps=3, finalize_chol=False,
                     user_bias=True, item_bias=True, center=False,
                     k_user=0, k_item=0, k_main=0,
                     w_main=1., w_user=1., w_item=1., w_implicit=0.5,
                     l1_lambda=0., center_U=True, center_I=True,
                     maxiter=400, niter=10, parallelize="separate", corr_pairs=4,
                     NA_as_zero=False, NA_as_zero_user=False, NA_as_zero_item=False,
                     precompute_for_predictions=True, use_float=False,
                     random_state=1, init="normal", verbose=True,
                     print_every=10, handle_interrupt=True,
                     produce_dicts=False, copy_data=True, nthreads=-1):
        assert method in ["als", "lbfgs"]
        assert parallelize in ["separate", "single"]
        assert init in ["normal", "gamma"]

        k = int(k) if isinstance(k, float) else k
        k_user = int(k_user) if isinstance(k_user, float) else k_user
        k_item = int(k_item) if isinstance(k_item, float) else k_item
        k_main = int(k_main) if isinstance(k_main, float) else k_main
        if not isinstance(self, OMF_explicit):
            assert isinstance(k, int) and k > 0
        else:
            assert isinstance(k, int) and k >= 0
        assert isinstance(k_user, int) and k_user >= 0
        assert isinstance(k_item, int) and k_item >= 0
        assert isinstance(k_main, int) and k_main >= 0

        if ((max(k_user, k_item) + k + k_main + max(user_bias, item_bias))**2) > np.iinfo(ctypes.c_int).max:
            raise ValueError("Number of factors is too large.")

        lambda_ = float(lambda_) if isinstance(lambda_, int) else lambda_
        lambda_ = np.array(lambda_) if lambda_.__class__.__name__ in ["list", "Series", "tuple"] else lambda_
        if lambda_.__class__.__name__ == "ndarray":
            lambda_ = lambda_.reshape(-1)
            assert lambda_.shape[0] == 6
            assert np.all(lambda_ >= 0.)
        else:
            assert isinstance(lambda_, float) and lambda_ >= 0.

        l1_lambda = float(l1_lambda) if isinstance(l1_lambda, int) else l1_lambda
        l1_lambda = np.array(l1_lambda) if l1_lambda.__class__.__name__ in ["list", "Series", "tuple"] else l1_lambda
        if l1_lambda.__class__.__name__ == "ndarray":
            l1_lambda = l1_lambda.reshape(-1)
            assert l1_lambda.shape[0] == 6
            assert np.all(l1_lambda >= 0.)
        else:
            assert isinstance(l1_lambda, float) and l1_lambda >= 0.

        
        niter = int(niter) if isinstance(niter, float) else niter
        assert isinstance(niter, int) and niter >= 0

        if not implicit and method == "lbfgs":
            maxiter = int(maxiter) if isinstance(maxiter, float) else maxiter
            assert isinstance(maxiter, int) and maxiter >= 0

        if nthreads < 1:
            nthreads = multiprocessing.cpu_count()
        if nthreads is None:
            nthreads = 1
        assert isinstance(nthreads, int) and nthreads > 0

        if not implicit and method == "lbfgs":
            print_every = int(print_every) if isinstance(print_every, float) else print_every
            assert isinstance(print_every, int) and print_every >= 0

        if not implicit and method == "lbfgs":
            corr_pairs = int(corr_pairs) if isinstance(corr_pairs, float) else corr_pairs
            assert isinstance(corr_pairs, int) and corr_pairs >= 2

        if isinstance(random_state, np.random.RandomState):
            random_state = random_state.randint(np.iinfo(np.int32).max)
        elif isinstance(random_state, np.random.Generator):
            random_state = random_state.integers(np.iinfo(np.int32).max)

        if (method == "lbfgs"):
            if (NA_as_zero or NA_as_zero_user or NA_as_zero_item):
                raise ValueError("Option 'NA_as_zero' not supported with method='lbfgs'.")
            if add_implicit_features:
                raise ValueError("Option 'add_implicit_features' not supported with method='lbfgs'.")
            if (nonneg) or (nonneg_C) or (nonneg_D):
                raise ValueError("non-negativity constraints not supported with method='lbfgs'.")
            if (scale_lam) or (scale_lam_sideinfo):
                raise ValueError("'scale_lam' not supported with method='lbfgs'.")
            if l1_lambda != 0.:
                raise ValueError("L1 regularization not supported with method='lbfgs'.")

        if method == "als":
            assert max_cg_steps > 0

        if max_cd_steps is None:
            max_cd_steps = 0
        if isinstance(max_cd_steps, float):
            max_cd_steps = int(max_cd_steps)
        assert max_cd_steps >= 0
        assert isinstance(max_cd_steps, int)

        w_main = float(w_main) if isinstance(w_main, int) else w_main
        w_user = float(w_user) if isinstance(w_user, int) else w_user
        w_item = float(w_item) if isinstance(w_item, int) else w_item
        w_implicit = float(w_implicit) if isinstance(w_implicit, int) else w_implicit
        assert isinstance(w_main, float) and w_main > 0
        assert isinstance(w_user, float) and w_user > 0
        assert isinstance(w_item, float) and w_item > 0
        assert isinstance(w_implicit, float) and w_implicit > 0

        if implicit:
            alpha = float(alpha) if isinstance(alpha, int) else alpha
            assert isinstance(alpha, float) and alpha > 0.

        if (center and nonneg):
            warnings.warn("Warning: will fit a model with centering and non-negativity constraints.")
        if (center_U and nonneg_C):
            warnings.warn("Warning: will fit a model with centering in 'U' and non-negativity constraints in 'C'.")
        if (center_I and nonneg_D):
            warnings.warn("Warning: will fit a model with centering in 'I' and non-negativity constraints in 'D'.")
        if (NA_as_zero and add_implicit_features):
            warnings.warn("Warning: will add implicit features while having 'NA_as_zero'.")

        self.k = k
        self.k_user = k_user
        self.k_item = k_item
        self.k_main = k_main
        self.lambda_ = lambda_
        self.l1_lambda = l1_lambda
        self.scale_lam = bool(scale_lam)
        self.scale_lam_sideinfo = bool(scale_lam_sideinfo) or self.scale_lam
        self.scale_bias_const = bool(scale_bias_const)
        self.alpha = alpha
        self.w_main = w_main
        self.w_user = w_user
        self.w_item = w_item
        self.w_implicit = w_implicit
        self.downweight = bool(downweight)
        self.user_bias = bool(user_bias)
        self.item_bias = bool(item_bias)
        self.center = bool(center) and not bool(implicit)
        self.center_U = bool(center_U)
        self.center_I = bool(center_I)
        self.method = method
        self.add_implicit_features = bool(add_implicit_features)
        self.apply_log_transf = bool(apply_log_transf)
        self.use_cg = bool(use_cg)
        self.max_cg_steps = int(max_cg_steps)
        self.max_cd_steps = int(max_cd_steps)
        self.finalize_chol = bool(finalize_chol)
        self.maxiter = maxiter
        self.niter = niter
        self.parallelize = parallelize
        self.NA_as_zero = bool(NA_as_zero)
        self.NA_as_zero_user = bool(NA_as_zero_user)
        self.NA_as_zero_item = bool(NA_as_zero_item)
        self.nonneg = bool(nonneg)
        self.nonneg_C = bool(nonneg_C)
        self.nonneg_D = bool(nonneg_D)
        self.precompute_for_predictions = bool(precompute_for_predictions)
        self.include_all_X = True
        self.use_float = bool(use_float)
        self.init = init
        self.verbose = bool(verbose)
        self.print_every = print_every
        self.corr_pairs = corr_pairs
        self.random_state = int(random_state)
        self.produce_dicts = bool(produce_dicts)
        self.handle_interrupt = bool(handle_interrupt)
        self.copy_data = bool(copy_data)
        self.nthreads = nthreads

        self._implicit = bool(implicit)
        self.dtype_ = ctypes.c_float if use_float else ctypes.c_double

        self._k_pred = k
        self._k_main_col = self.k_main

        self._reset()

    def _reset(self):
        self.A_ = np.empty((0,0), dtype=self.dtype_)
        self.B_ = np.empty((0,0), dtype=self.dtype_)
        self.C_ = np.empty((0,0), dtype=self.dtype_)
        self.D_ = np.empty((0,0), dtype=self.dtype_)
        self.Cbin_ = np.empty((0,0), dtype=self.dtype_)
        self.Dbin_ = np.empty((0,0), dtype=self.dtype_)
        self.Ai_ = np.empty((0,0), dtype=self.dtype_)
        self.Bi_ = np.empty((0,0), dtype=self.dtype_)
        self.user_bias_ = np.empty(0, dtype=self.dtype_)
        self.item_bias_ = np.empty(0, dtype=self.dtype_)
        self.scaling_biasA_ = 0.
        self.scaling_biasB_ = 0.
        self.C_bias_ = np.empty(0, dtype=self.dtype_)
        self.D_bias_ = np.empty(0, dtype=self.dtype_)
        self.glob_mean_ = 0.

        self._TransBtBinvBt = np.empty((0,0), dtype=self.dtype_)
        ## will have lambda added for implicit but not for explicit, dim is k+k_main
        self._BtB = np.empty((0,0), dtype=self.dtype_)
        self._BtXbias = np.empty(0, dtype=self.dtype_)
        self._TransCtCinvCt = np.empty((0,0), dtype=self.dtype_)
        ## will be multiplied by w_user already
        self._CtC = np.empty((0,0), dtype=self.dtype_)
        self._BeTBe = np.empty((0,0), dtype=self.dtype_)
        self._BeTBeChol = np.empty((0,0), dtype=self.dtype_)
        self._BiTBi = np.empty((0,0), dtype=self.dtype_)
        self._CtUbias = np.empty(0, dtype=self.dtype_)

        self._A_pred = np.empty((0,0), dtype=self.dtype_)
        self._B_pred = np.empty((0,0), dtype=self.dtype_)
        self._B_plus_bias = np.empty((0,0), dtype=self.dtype_)

        self._U_cols = np.empty(0, dtype=object)
        self._I_cols = np.empty(0, dtype=object)
        self._Ub_cols = np.empty(0, dtype=object)
        self._Ib_cols = np.empty(0, dtype=object)
        self._U_colmeans = np.empty(0, dtype=self.dtype_)
        self._I_colmeans = np.empty(0, dtype=self.dtype_)
        self._w_main_multiplier = 1.

        self.is_fitted_ = False
        self._only_prediction_info = False
        self.nfev_ = None
        self.nupd_ = None
        self.user_mapping_ = np.array([], dtype=object)
        self.item_mapping_ = np.array([], dtype=object)
        self.reindex_ = False
        self.user_dict_ = dict()
        self.item_dict_ = dict()

    def _take_params_offsets(self, k_sec=0, k_main=0, add_intercepts=True):
        k_sec = int(k_sec) if isinstance(k_sec, float) else k_sec
        k_main = int(k_main) if isinstance(k_main, float) else k_main
        assert isinstance(k_sec, int) and k_sec >= 0
        assert isinstance(k_main, int) and k_main >= 0

        if ((max(k_sec, k_main) + self.k)**2 + 1) > np.iinfo(ctypes.c_int).max:
            raise ValueError("Number of factors is too large.")

        if self.method == "als":
            if self._implicit:
                msg = " not supported for implicit-feedback."
            else:
                msg = " not supported with method='als'."
            if k_sec > 0 or k_main > 0:
                raise ValueError("'k_sec' and 'k_main'" + msg)
            if isinstance(self.lambda_, np.ndarray):
                raise ValueError("Different regularization for each parameter is" + msg)
            if self.w_user != 1. or self.w_item != 1.:
                raise ValueError("'w_user' and 'w_main' are" + msg)

        self.k_sec = k_sec
        self.k_main = k_main

        self._k_pred = self.k_sec + self.k + self.k_main
        self._k_main_col = 0
        self.add_intercepts = bool(add_intercepts)


    def _append_NAs(self, U, m_u, p, append_U):
        U_new = np.repeat(np.nan, m_u*p).reshape((m_u, p))
        U_new[np.setdiff1d(np.arange(m_u), append_U), :] = U
        return U_new

    def _decompose_coo(self, X):
        row = X.row.astype(ctypes.c_int)
        col = X.col.astype(ctypes.c_int)
        val = X.data.astype(self.dtype_)
        return row, col, val

    def _process_U_arr(self, U):
        Urow = np.empty(0, dtype=ctypes.c_int)
        Ucol = np.empty(0, dtype=ctypes.c_int)
        Uval = np.empty(0, dtype=self.dtype_)
        Uarr = np.empty((0,0), dtype=self.dtype_)
        Ucols = np.empty(0, dtype=object)
        m = 0
        p = 0
        if U.__class__.__name__ == "coo_matrix":
            Urow, Ucol, Uval = self._decompose_coo(U)
            m, p = U.shape
        elif U is not None:
            if U.__class__.__name__ == "DataFrame":
                Ucols = U.columns.to_numpy()
                U = U.to_numpy()
            Uarr = np.ascontiguousarray(U).astype(self.dtype_)
            m, p = Uarr.shape
        return Urow, Ucol, Uval, Uarr, Ucols, m, p

    def _convert_ids(self, X, U, U_bin, col="UserId"):
        ### Note: if one 'UserId' column is a Pandas Categorical, then all
        ### of them in the other DataFrames have to be too.
        swapped = False
        append_U = np.empty(0, dtype=object)
        append_Ub = np.empty(0, dtype=object)
        msg = "'X' and side info have no IDs in common."
        if (U is not None) and (U_bin is not None):
            user_ids1 = np.intersect1d(U[col].to_numpy(), X[col].to_numpy())
            user_ids2 = np.intersect1d(U_bin[col].to_numpy(), X[col].to_numpy())
            user_ids3 = np.intersect1d(U_bin[col].to_numpy(), U[col].to_numpy())
            if (user_ids1.shape[0] == 0) and (user_ids2.shape[0] == 0):
                raise ValueError(msg)
            user_ids = np.intersect1d(user_ids1, user_ids2)
            u_not_x = np.setdiff1d(U[col].to_numpy(), X[col].to_numpy())
            x_not_u = np.setdiff1d(X[col].to_numpy(), U[col].to_numpy())
            b_not_x = np.setdiff1d(U_bin[col].to_numpy(), X[col].to_numpy())
            x_not_b = np.setdiff1d(X[col].to_numpy(), U_bin[col].to_numpy())
            b_not_u = np.setdiff1d(U_bin[col].to_numpy(), U[col].to_numpy())
            u_not_b = np.setdiff1d(U[col].to_numpy(), U_bin[col].to_numpy())

            ### There can be cases in which the sets are disjoint,
            ### and will need to add NAs to one of the inputs.
            if (u_not_x.shape[0] == 0 and
                x_not_u.shape[0] == 0 and
                b_not_x.shape[0] == 0 and
                x_not_b.shape[0] == 0 and
                b_not_u.shape[0] == 0 and
                u_not_b.shape[0] == 0):
                user_ids = user_ids
            else:
                if u_not_b.shape[0] >= b_not_u.shape[0]:
                    user_ids = np.r_[user_ids, user_ids1, X[col].to_numpy(), user_ids3, U[col].to_numpy(), U_bin[col].to_numpy()]
                    append_U = x_not_u
                    append_Ub = np.r_[x_not_b, u_not_b]
                else:
                    user_ids = np.r_[user_ids, user_ids2, X[col].to_numpy(), user_ids3, U_bin[col].to_numpy(), U[col].to_numpy()]
                    append_U = np.r_[x_not_u, b_not_u]
                    append_Ub = x_not_b

            _, user_mapping_ = pd.factorize(user_ids)
            X[col] = pd.Categorical(X[col], user_mapping_).codes.astype(ctypes.c_int)
            U[col] = pd.Categorical(U[col], user_mapping_).codes.astype(ctypes.c_int)
            U_bin[col] = pd.Categorical(U_bin[col], user_mapping_).codes.astype(ctypes.c_int)

            if append_U.shape[0]:
                append_U = pd.Categorical(np.unique(append_U), user_mapping_).codes.astype(ctypes.c_int)
                append_U = np.sort(append_U)

            if append_Ub.shape[0]:
                append_Ub = pd.Categorical(np.unique(append_Ub), user_mapping_).codes.astype(ctypes.c_int)
                append_Ub = np.sort(append_Ub)

        else:
            if (U is None) and (U_bin is not None):
                U, U_bin = U_bin, U
                swapped = True

            if (U is not None):
                user_ids = np.intersect1d(U[col].to_numpy(), X[col].to_numpy())
                if user_ids.shape[0] == 0:
                    raise ValueError(msg)

                u_not_x = np.setdiff1d(U[col].to_numpy(), X[col].to_numpy())
                x_not_u = np.setdiff1d(X[col].to_numpy(), U[col].to_numpy())
                if (u_not_x.shape[0]) or (x_not_u.shape[0]):
                    ### Case0: both have the same entries
                    ### This is the ideal situation
                    if (x_not_u.shape[0] == 0) and (u_not_x.shape[0] == 0):
                        user_ids = user_ids
                    ### Case1: X has IDs that U doesn't, but not the other way around
                    ### Here there's no need to do anything special afterwards
                    if (x_not_u.shape[0] > 0) and (u_not_x.shape[0] == 0):
                        user_ids = np.r_[user_ids, x_not_u]
                    ### Case2: U has IDs that X doesn't, but not the other way around
                    ### Don't need to do anything special afterwards either
                    elif (u_not_x.shape[0] > 0) and (x_not_u.shape[0] == 0):
                        user_ids = np.r_[user_ids, u_not_x]
                    ### Case3: both have IDs that the others don't
                    else:
                        user_ids = np.r_[user_ids, X[col].to_numpy(), U[col].to_numpy()]
                        append_U = x_not_u

                _, user_mapping_ = pd.factorize(user_ids)
                if user_mapping_.__class__.__name__ == "CategoricalIndex":
                    user_mapping_ = user_mapping_.to_numpy()
                X[col] = pd.Categorical(X[col], user_mapping_).codes.astype(ctypes.c_int)
                U[col] = pd.Categorical(U[col], user_mapping_).codes.astype(ctypes.c_int)
                if append_U.shape[0]:
                    append_U = pd.Categorical(append_U, user_mapping_).codes.astype(ctypes.c_int)
                    append_U = np.sort(append_U)

            else:
                X[col], user_mapping_ = pd.factorize(X[col].to_numpy())
                if user_mapping_.__class__.__name__ == "CategoricalIndex":
                    user_mapping_ = user_mapping_.to_numpy()

        if swapped:
            U, U_bin = U_bin, U
            append_U, append_Ub = append_Ub, append_U
        return X, U, U_bin, user_mapping_, append_U, append_Ub

    def _process_U_df(self, U, is_I=False, df_name="U"):
        Urow = np.empty(0, dtype=ctypes.c_int)
        Ucol = np.empty(0, dtype=ctypes.c_int)
        Uval = np.empty(0, dtype=self.dtype_)
        Uarr = np.empty((0,0), dtype=self.dtype_)
        Ucols = np.empty(0, dtype=object)
        cl_take = "ItemId" if is_I else "UserId"
        m = 0
        p = 0
        if U is not None:
            if "ColumnId" in U.columns.values:
                Urow = U[cl_take].astype(ctypes.c_int).to_numpy()
                Ucol = U.ColumnId.astype(ctypes.c_int).to_numpy()
                if "Value" not in U.columns.values:
                    msg = "If passing sparse '%s', must have column 'Value'."
                    msg = msg % df_name
                    raise ValueError(msg)
                Uval = U.Value.astype(self.dtype_).to_numpy()
                m = int(Urow.max() + 1)
                p = int(Ucol.max() + 1)
            else:
                U = U.sort_values(cl_take)
                Uarr = U[[cl for cl in U.columns.values if cl != cl_take]]
                Ucols = Uarr.columns.to_numpy()
                Uarr = Uarr.astype(self.dtype_).to_numpy()
                if np.isfortran(Uarr):
                    Uarr = np.ascontiguousarray(Uarr)
                m, p = Uarr.shape
        return Urow, Ucol, Uval, Uarr, Ucols, m, p

    def _process_new_U(self, U, U_col, U_val, U_bin, is_I=False):
        letter = "U" if not is_I else "I"
        name = "user" if not is_I else "item"
        Mat = self.C_ if not is_I else self.D_
        MatBin = self.Cbin_ if not is_I else self.Dbin_
        Cols = self._U_cols if not is_I else self._I_cols
        ColsBin = self._Ub_cols if not is_I else self._Ib_cols
        dct = self.user_dict_ if not is_I else self.item_dict_
        mapping = self.user_mapping_ if not is_I else self.item_mapping_

        if ((U_col is not None) and (U_val is None)) or ((U_col is None) and (U_val is  not None)):
            raise ValueError("Must pass '%s_col' and '%s_val' together."
                             % (letter, letter))
        if (U_col is not None) and (U is not None):
            raise ValueError("Can only pass %s info in one format."
                             % name)
        if (U is None) and (U_col is None) and (U_bin is None):
            raise ValueError("Must pass %s side information in some format."
                             % name)

        if self.copy_data:
            U = U.copy() if U is not None else U
            U_bin = U_bin.copy() if U_bin is not None else U_bin
            U_col = U_col.copy() if U_col is not None else U_col
            U_val = U_val.copy() if U_val is not None else U_val

        ###
        if U is not None:
            if Mat.shape[0] == 0:
                raise ValueError("Model was not fit to %s data." % name)
            if U.__class__.__name__ == "DataFrame" and Cols.shape[0]:
                U = U[Cols]
            U = np.array(U).reshape(-1).astype(self.dtype_)
            if U.shape[0] != Mat.shape[0]:
                raise ValueError("Dimensions of %s don't match with earlier data."
                                 % letter)
        else:
            U = np.empty(0, dtype=self.dtype_)
        ###
        if U_bin is not None:
            if MatBin.shape[0] == 0:
                raise ValueError("Model was not fit to %s binary data." % name)
            if (U_bin.__class__.__name__  == "DataFrame") and (ColsBin.shape[0]):
                U_bin = U_bin[ColsBin]
            U_bin = np.array(U_bin).reshape(-1).astype(self.dtype_)
            if U_bin.shape[0] != MatBin.shape[0]:
                raise ValueError("Dimensions of %s_bin don't match with earlier data."
                                 % letter)
        else:
            U_bin = np.empty(0, dtype=self.dtype_)
        ###
        if U_col is not None:
            if Mat.shape[0] == 0:
                raise ValueError("Model was not fit to %s data." % name)
            U_val = np.array(U_val).reshape(-1).astype(self.dtype_)
            if U_val.shape[0] == 0:
                if np.array(U_col).shape[0] > 0:
                    raise ValueError("'%s_col' and '%s_val' must have the same number of entries." % (letter, letter))
                U_col = np.empty(0, dtype=ctypes.c_int)
                U_val = np.empty(0, dtype=self.dtype_)
            else:
                if self.reindex_:
                    if len(dct):
                        try:
                            U_col = np.array([dct[u] for u in U_col])
                        except:
                            raise ValueError("Sparse inputs cannot contain missing values.")
                    else:
                        U_col = pd.Categorical(U_col, mapping).codes.astype(ctypes.c_int)
                        if np.any(U_col < 0):
                            raise ValueError("Sparse inputs cannot contain missing values.")
                    U_col = U_col.astype(ctypes.c_int)
                else:
                    U_col = np.array(U_col).reshape(-1).astype(ctypes.c_int)
                    imin, imax = U_col.min(), U_col.max()
                    if np.isnan(imin) or np.isnan(imax):
                        raise ValueError("Sparse inputs cannot contain missing values.")
                    if (imin < 0) or (imax >= Mat.shape[0]):
                        msg  = "Column indices for user info must be within the range"
                        msg += " of the data that was pased to 'fit'."
                        raise ValueError(msg)
            if U_val.shape[0] != U_col.shape[0]:
                raise ValueError("'%s_col' and '%s_val' must have the same number of entries." % (letter, letter))
        else:
            U_col = np.empty(0, dtype=ctypes.c_int)
            U_val = np.empty(0, dtype=self.dtype_)
        ###

        return U, U_col, U_val, U_bin

    def _process_new_U_2d(self, U, is_I=False, allow_csr=False):
        letter = "U" if not is_I else "I"
        col_id = "UserId" if not is_I else "ItemId"
        Cols = self._U_cols if not is_I else self._I_cols
        Mat = self.C_ if not is_I else self.D_

        if self.copy_data and U is not None:
            U = U.copy()

        Uarr = np.empty((0,0), dtype=self.dtype_)
        Urow = np.empty(0, dtype=ctypes.c_int)
        Ucol = np.empty(0, dtype=ctypes.c_int)
        Uval = np.empty(0, dtype=self.dtype_)
        Ucsr_p = np.empty(0, dtype=ctypes.c_size_t)
        Ucsr_i = np.empty(0, dtype=ctypes.c_int)
        Ucsr = np.empty(0, dtype=self.dtype_)
        m, p = U.shape if U is not None else (0,0)
        if (p != Mat.shape[0]) and (Mat.shape[0] > 0) and (p > 0):
            msg  = "'%s' must have the same columns "
            msg += "as the data passed to 'fit'."
            raise ValueError(msg % letter)

        if U.__class__.__name__ == "DataFrame":
            if col_id in U.columns.values:
                warnings.warn("'%s' not meaningful for new inputs." % col_id)
            if Cols.shape[0]:
                U = U[Cols]
            Uarr = U.to_numpy()
            Uarr = np.ascontiguousarray(Uarr).astype(self.dtype_)

        elif U.__class__.__name__ == "coo_matrix":
            Urow = U.row.astype(ctypes.c_int)
            Ucol = U.col.astype(ctypes.c_int)
            Uval = U.data.astype(self.dtype_)
        elif U.__class__.__name__ == "csr_matrix":
            if not allow_csr:
                raise ValueError("Sparse matrices only supported in COO format.")
            Ucsr_p = U.indptr.astype(ctypes.c_size_t)
            Ucsr_i = U.indices.astype(ctypes.c_int)
            Ucsr = U.data.astype(self.dtype_)
        elif U.__class__.__name__ == "ndarray":
            Uarr = np.ascontiguousarray(U).astype(self.dtype_)
        elif U is None:
            pass
        else:
            if not allow_csr:
                msg = "'%s' must be a Pandas DataFrame, SciPy sparse COO, or NumPy array."
            else:
                msg = "'%s' must be a Pandas DataFrame, SciPy sparse CSR or COO, or NumPy array."
            raise ValueError(msg % letter)

        return Uarr, Urow, Ucol, Uval, Ucsr_p, Ucsr_i, Ucsr, m, p

    def _process_new_Ub_2d(self, U_bin, is_I=False):
        letter = "U" if not is_I else "I"
        col_id = "UserId" if not is_I else "ItemId"
        Cols = self._Ub_cols if not is_I else self._Ib_cols
        Mat = self.Cbin_ if not is_I else self.Dbin_

        Ub_arr = np.empty((0,0), dtype=self.dtype_)

        m_ub, pbin = U_bin.shape if U_bin is not None else (0,0)

        if max(m_ub, pbin) and (not Mat.shape[0] or not Mat.shape[1]):
            raise ValueError("Cannot pass binary data if model was not fit to binary side info.")

        if (pbin != Mat.shape[0]) and (Mat.shape[0] > 0) and (pbin > 0):
            msg  = "'%s_bin' must have the same columns "
            msg += "as the data passed to 'fit'."
            raise ValueError(msg % letter)

        if self.copy_data and U_bin is not None:
            U_bin = U_bin.copy() if U_bin is not None else U_bin

        if U_bin.__class__.__name__ == "DataFrame":
            if col_id in U_bin.columns.values:
                warnings.warn("'%s' not meaningful for new inputs." % col_id)
            if Cols.shape[0]:
                U_bin = U_bin[Cols]
            Ub_arr = U_bin.to_numpy()
            Ub_arr = np.ascontiguousarray(Ub_arr).astype(self.dtype_)
        elif Ub_arr.__class__.__name__ == "ndarray":
            Ub_arr = np.ascontiguousarray(Ub_arr).astype(self.dtype_)
        elif Ub_arr is None:
            pass
        else:
            raise ValueError("'%s_bin' must be a Pandas DataFrame or NumPy array."
                             % letter)

        return Ub_arr, m_ub, pbin

    def _process_new_X_2d(self, X, W=None):
        if len(X.shape) != 2:
            raise ValueError("'X' must be 2-dimensional.")

        Xarr = np.empty((0,0), dtype=self.dtype_)
        Xrow = np.empty(0, dtype=ctypes.c_int)
        Xcol = np.empty(0, dtype=ctypes.c_int)
        Xval = np.empty(0, dtype=self.dtype_)
        Xcsr_p = np.empty(0, dtype=ctypes.c_size_t)
        Xcsr_i = np.empty(0, dtype=ctypes.c_int)
        Xcsr = np.empty(0, dtype=self.dtype_)
        W_dense = np.empty((0,0), dtype=self.dtype_)
        W_sp = np.empty(0, dtype=self.dtype_)
        m, n = X.shape

        if self.copy_data:
            X = X.copy() if X is not None else X
            W = W.copy() if W is not None else W

        if X.__class__.__name__ == "coo_matrix":
            Xrow = X.row.astype(ctypes.c_int)
            Xcol = X.col.astype(ctypes.c_int)
            Xval = X.data.astype(self.dtype_)
            if W is not None:
                W_sp = np.array(W).reshape(-1).astype(self.dtype_)
                if W_sp.shape[0] != Xval.shape[0]:
                    msg =  "'W' must have the same number of non-zero entries "
                    msg += "as 'X'."
                    raise ValueError(msg)
        elif X.__class__.__name__ == "csr_matrix":
            Xcsr_p = X.indptr.astype(ctypes.c_size_t)
            Xcsr_i = X.indices.astype(ctypes.c_int)
            Xcsr = X.data.astype(self.dtype_)
            if W is not None:
                W_sp = np.array(W).reshape(-1).astype(self.dtype_)
                if W_sp.shape[0] != Xcsr.shape[0]:
                    msg =  "'W' must have the same number of non-zero entries "
                    msg += "as 'X'."
                    raise ValueError(msg)
        elif X.__class__.__name__ == "ndarray":
            Xarr = np.ascontiguousarray(X).astype(self.dtype_)
            if W is not None:
                assert W.shape[0] == X.shape[0]
                assert W.shape[1] == X.shape[1]
                W_dense = np.ascontiguousarray(W).astype(self.dtype_)
        else:
            raise ValueError("'X' must be a SciPy CSR or COO matrix, or NumPy array.")

        if n > self._n_orig:
            raise ValueError("'X' has more columns than what was passed to 'fit'.")

        if self.apply_log_transf:
            if Xval.min() < 1:
                raise ValueError("Cannot pass values below 1 with 'apply_log_transf=True'.")

        return Xarr, Xrow, Xcol, Xval, Xcsr_p, Xcsr_i, Xcsr, m, n, W_dense, W_sp

    def _process_users_items(self, user, item, include, exclude, allows_no_item=True):
        if (include is not None and np.any(pd.isnull(include))) \
            or (exclude is not None and np.any(pd.isnull(exclude))):
            raise ValueError("'include' and 'exclude' should not contain missing values.")
        if include is not None and exclude is not None:
            raise ValueError("Cannot pass 'include' and 'exclude' together.")
        include = np.array(include).reshape(-1) if include is not None \
                    else np.empty(0, dtype=ctypes.c_int)
        exclude = np.array(exclude).reshape(-1) if exclude is not None \
                    else np.empty(0, dtype=ctypes.c_int)

        if isinstance(user, list) or isinstance(user, tuple):
            user = np.array(user)
        if isinstance(item, list) or isinstance(item, tuple):
            item = np.array(item)
        if user.__class__.__name__=='Series':
            user = user.to_numpy()
        if item.__class__.__name__=='Series':
            item = item.to_numpy()
            
        if user is not None:
            if isinstance(user, np.ndarray):
                if len(user.shape) > 1:
                    user = user.reshape(-1)
                assert user.shape[0] > 0
                if self.reindex_:
                    if user.shape[0] > 1:
                        user = pd.Categorical(user, self.user_mapping_).codes.astype(ctypes.c_int)
                    else:
                        if len(self.user_dict_):
                            try:
                                user = self.user_dict_[user]
                            except:
                                user = -1
                        else:
                            user = pd.Categorical(user, self.user_mapping_).codes[0]
            else:
                if self.reindex_:
                    if len(self.user_dict_):
                        try:
                            user = self.user_dict_[user]
                        except:
                            user = -1
                    else:
                        user = pd.Categorical(np.array([user]), self.user_mapping_).codes[0]
                user = np.array([user])
            
        
        if item is not None:
            if isinstance(item, np.ndarray):
                if len(item.shape) > 1:
                    item = item.reshape(-1)
                assert item.shape[0] > 0
                if self.reindex_:
                    if item.shape[0] > 1:
                        item = pd.Categorical(item, self.item_mapping_).codes.astype(ctypes.c_int)
                    else:
                        if len(self.item_dict_):
                            try:
                                item = self.item_dict_[item[0]]
                            except:
                                item = -1
                        else:
                            item = pd.Categorical(item, self.item_mapping_).codes[0]
            else:
                if self.reindex_:
                    if len(self.item_dict_):
                        try:
                            item = self.item_dict_[item]
                        except:
                            item = -1
                    else:
                        item = pd.Categorical(np.array([item]), self.item_mapping_).codes[0]
                item = np.array([item])
        else:
            if not allows_no_item:
                raise ValueError("Must pass IDs for 'item'.")

        if self.reindex_:
            msg = "'%s' should contain only items that were passed to 'fit'."
            if include.shape[0]:
                if len(self.item_dict_):
                    try:
                        include = np.array([self.item_dict_[i] for i in include])
                    except:
                        raise ValueError(msg % "include")
                else:
                    include = pd.Categorical(include, self.item_mapping_).codes.astype(ctypes.c_int)
                    if np.any(include < 0):
                        raise ValueError(msg % "include")
                include = include.astype(ctypes.c_int).reshape(-1)
            if exclude.shape[0]:
                if len(self.item_dict_):
                    try:
                        exclude = np.array([self.item_dict_[i] for i in exclude])
                    except:
                        raise ValueError(msg % "exclude")
                else:
                    exclude = pd.Categorical(exclude, self.item_mapping_).codes.astype(ctypes.c_int)
                    if np.any(exclude < 0):
                        raise ValueError(msg % "exclude")
                exclude = exclude.astype(ctypes.c_int).reshape(-1)

        else:
            msg  = "'%s' entries must be within the range of the %s (%s)"
            msg += " of the data that was passed to 'fit'."
            if include.shape[0]:
                imin, imax = include.min(), include.max()
                if (imin < 0) or (imax >= self._B_pred.shape[0]):
                    raise ValueError(msg % ("include", "items", "columns"))
            if exclude.shape[0]:
                emin, emax = exclude.min(), exclude.max()
                if (emin < 0) or (emax >= self._B_pred.shape[0]):
                    raise ValueError(msg % ("exclude", "items", "columns"))

        if user is not None:
            user = user.astype(ctypes.c_int)
        if item is not None:
            item = item.astype(ctypes.c_int)
        if include.dtype != ctypes.c_int:
            include = include.astype(ctypes.c_int)
        if exclude.dtype != ctypes.c_int:
            exclude = exclude.astype(ctypes.c_int)


        return user, item, include, exclude

    def _fit_common(self, X, U=None, I=None, U_bin=None, I_bin=None, W=None,
                    enforce_same_shape=False):
        if (U_bin is not None or I_bin is not None) and self.method != "lbfgs":
            msg  = "Binary side info is only supported when using method='lbfgs'."
            raise ValueError(msg)

        if self.copy_data:
            X = X.copy()
            U = U.copy() if U is not None else None
            I = I.copy() if I is not None else None
            U_bin = U_bin.copy() if U_bin is not None else None
            I_bin = I_bin.copy() if I_bin is not None else None
            W = W.copy() if W is not None else None

        self._reset()

        if X.__class__.__name__ == "DataFrame":
            msg = "If passing 'X' as DataFrame, '%s' must also be a DataFrame."
            if U is not None and U.__class__.__name__ != "DataFrame":
                raise ValueError(msg % "U")
            if I is not None and I.__class__.__name__ != "DataFrame":
                raise ValueError(msg % "I")
            if U_bin is not None and U_bin.__class__.__name__ != "DataFrame":
                raise ValueError(msg % "U_bin")
            if I_bin is not None and I_bin.__class__.__name__ != "DataFrame":
                raise ValueError(msg % "I_bin")
            if W is not None:
                msg  = "Passing 'W' with 'X' as DataFrame is not supported."
                msg += " Weight should be under a column in the DataFrame, "
                msg += "called 'Weight'."
                raise ValueError(msg)

            assert "UserId" in X.columns.values
            assert "ItemId" in X.columns.values
            if (self._implicit) and ("Rating" in X.columns.values) and ("Value" not in X.columns.values):
                X = X.rename(columns={"Rating":"Value"}, copy=False, inplace=False)
            if self._implicit:
                assert "Value" in X.columns.values
            else:
                assert "Rating" in X.columns.values

            if U is not None:
                assert "UserId" in U.columns.values
            if I is not None:
                assert "ItemId" in I.columns.values
            if U_bin is not None:
                assert "UserId" in U_bin.columns.values
            if I_bin is not None:
                assert "ItemId" in I_bin.columns.values

            X, U, U_bin, self.user_mapping_, append_U, append_Ub = self._convert_ids(X, U, U_bin, "UserId")
            X, I, I_bin, self.item_mapping_, append_I, append_Ib = self._convert_ids(X, I, I_bin, "ItemId")

            Xrow = X.UserId.astype(ctypes.c_int).to_numpy()
            Xcol = X.ItemId.astype(ctypes.c_int).to_numpy()
            if self._implicit:
                Xval = X.Value.astype(self.dtype_).to_numpy()
            else:
                Xval = X.Rating.astype(self.dtype_).to_numpy()
            if Xval.shape[0] == 0:
                raise ValueError("'X' contains no non-zero entries.")
            Xarr = np.empty((0,0), dtype=self.dtype_)
            W_sp = np.empty(0, dtype=self.dtype_)
            if "Weight" in X.columns.values:
                W_sp = X.Weight.astype(self.dtype_).to_numpy()
            W_dense = np.empty((0,0), dtype=self.dtype_)

            Urow, Ucol, Uval, Uarr, self._U_cols, m_u, p = self._process_U_df(U, False, "U")
            Irow, Icol, Ival, Iarr, self._I_cols, n_i, q = self._process_U_df(I, True, "I")

            Ub_arr = np.empty((0,0), dtype=self.dtype_)
            Ib_arr = np.empty((0,0), dtype=self.dtype_)
            m_ub = 0
            pbin = 0
            n_ib = 0
            qbin = 0
            msg = "Binary side info data cannot be passed in sparse format."
            if U_bin is not None:
                if "ColumnId" in U_bin.columns.values:
                    raise ValueError(msg)
                _1, _2, _3, Ub_arr, self._Ub_cols, m_ub, pbin = self._process_U_df(U_bin, False, "U_bin")
            if I_bin is not None:
                if "ColumnId" in I_bin.columns.values:
                    raise ValueError(msg)
                _1, _2, _3, Ib_arr, self._Ib_cols, n_ib, qbin = self._process_U_df(I_bin, True, "U_bin")

            m_u += append_U.shape[0]
            n_i += append_I.shape[0]
            if append_U.shape[0] and Uarr is not None:
                if enforce_same_shape:
                    raise ValueError("'X' and 'U' must have the same rows.")
                Uarr = self._append_NAs(Uarr, m_u, p, append_U)
            if append_I.shape[0] and Iarr is not None:
                if enforce_same_shape:
                    raise ValueError("Columns of 'X' must match with rows of 'I'.")
                Iarr = self._append_NAs(Iarr, n_i, q, append_I)
            if append_Ub.shape[0]:
                m_ub += append_Ub.shape[0]
                Ub_arr = self._append_NAs(Ub_arr, m_ub, pbin, append_Ub)
            if append_Ib.shape[0]:
                n_ib += append_Ib.shape[0]
                Ib_arr = self._append_NAs(Ib_arr, n_ib, qbin, append_Ib)

            self.reindex_ = True
            if self.produce_dicts:
                self.user_dict_ = {self.user_mapping_[i]:i for i in range(self.user_mapping_.shape[0])}
                self.item_dict_ = {self.item_mapping_[i]:i for i in range(self.item_mapping_.shape[0])}

        elif X.__class__.__name__ in ["coo_matrix", "ndarray"]:
            allowed_sideinfo = ["DataFrame", "ndarray", "coo_matrix"]
            allowed_bin = ["DataFrame", "ndarray"]
            msg = " must be a Pandas DataFrame, NumPy array, or SciPy sparse COO matrix."
            msg_bin = " must be a Pandas DataFrame or NumPy array."
            if U is not None and U.__class__.__name__ not in allowed_sideinfo:
                raise ValueError("'U'" + msg)
            if I is not None and I.__class__.__name__ not in allowed_sideinfo:
                raise ValueError("'I'" + msg)
            if U_bin is not None and U_bin.__class__.__name__ not in allowed_bin:
                raise ValueError("'U_bin'" + msg_bin)
            if I_bin is not None and I_bin.__class__.__name__ not in allowed_bin:
                raise ValueError("'I_bin'" + msg_bin)
            if W is not None:
                if isinstance(W, list):
                    W = np.array(W)
                if (len(W.shape) > 1) and (X.__class__.__name__ == "coo_matrix"):
                    W = W.reshape(-1)
                if W.__class__.__name__ != "ndarray" or \
                   (X.__class__.__name__ == "coo_matrix" and W.shape[0] != X.nnz) or\
                   (X.__class__.__name__ == "ndarray" and W.shape[0] != X.shape[0]):
                    raise ValueError("'W' must be a 1-d array with the same number of entries as 'X'.")

            if (self._implicit) and (X.__class__.__name__ == "ndarray") and (self.k_sec == 0):
                raise ValueError("Dense arrays for 'X' not supported with implicit-feedback.")

            Xrow, Xcol, Xval, Xarr, _1, _2, _3 = self._process_U_arr(X)
            Urow, Ucol, Uval, Uarr, self._U_cols, m_u, p = self._process_U_arr(U)
            Irow, Icol, Ival, Iarr, self._I_cols, n_i, q = self._process_U_arr(I)
            _1, _2, _3, Ub_arr, self._Ub_cols, m_ub, pbin = self._process_U_arr(U_bin)
            _1, _2, _3, Ib_arr, self._Ib_cols, n_ib, qbin = self._process_U_arr(I_bin)

            if (X.__class__.__name__ == "coo_matrix") and (Xval.shape[0] == 0):
                raise ValueError("'X' contains no non-zero entries.")

            W_sp = np.empty(0, dtype=self.dtype_)
            W_dense = np.empty((0,0), dtype=self.dtype_)
            if W is not None:
                if X.__class__.__name__ == "coo_matrix":
                    W_sp = W.astype(self.dtype_)
                else:
                    W_dense = W.astype(self.dtype_)

            self.reindex_ = False
        
        else:
            msg = "'X' must be a Pandas DataFrame, SciPy COO matrix, or NumPy array."
            raise ValueError(msg)

        if Xarr.shape[0]:
            m, n = Xarr.shape
        else:
            m = int(Xrow.max() + 1)
            n = int(Xcol.max() + 1)
            if X.__class__.__name__ == "coo_matrix":
                m = max(m, X.shape[0])
                n = max(n, X.shape[1])
            if enforce_same_shape:
                m = max(m, m_u, m_ub)
                n = max(n, n_i, n_ib)

        if enforce_same_shape:
            msg_err_rows = "'X' and 'U%s' must have the same rows."
            msg_err_cols = "Columns of 'X' must match with rows of 'I%s'."
            if Uarr.shape[0]:
                if Uarr.shape[0] != m:
                    raise ValueError(msg_err_rows % "")
            if Iarr.shape[0]:
                if Iarr.shape[0] != n:
                    raise ValueError(msg_err_cols % "")
            if Uval.shape[0]:
                if m_u != m:
                    raise ValueError(msg_err_rows % "")
            if Ival.shape[0]:
                if n_i != n:
                    raise ValueError(msg_err_cols % "")
            if Ub_arr.shape[0]:
                if m_ub != m:
                    raise ValueError(msg_err_rows % "_bin")
            if Ib_arr.shape[0]:
                if n_ib != n:
                    raise ValueError(msg_err_rows % "_bin")

        if max(m, n, m_u, n_i, p, q, m_ub, n_ib, pbin, qbin) > np.iinfo(ctypes.c_int).max:
            msg  = "Error: dimensionality of the inputs is too high. "
            msg += "Number of rows/columns cannot be more than INT_MAX."
            raise ValueError(msg)

        if (max(m_u, m_ub, p, pbin) == 0) and (self.k_user):
            self.k_user = 0
            warnings.warn("No user side info provided, will set 'k_user' to zero.")
        if (max(n_i, n_ib, q, qbin) == 0) and (self.k_item):
            self.k_item = 0
            warnings.warn("No item side info provided, will set 'k_item' to zero.")
        if (m == 0) or (n == 0):
            raise ValueError("'X' must have at least one row and column.")

        if self.apply_log_transf:
            msg_small = "Cannot pass values below 1 with 'apply_log_transf=True'."
            if Xarr.shape[0]:
                if np.nanmin(Xarr) < 1:
                    raise ValueError(msg_small)
            elif Xval.shape[0]:
                if Xval.min() < 1:
                    raise ValueError(msg_small)

        if (self.NA_as_zero) and (Xarr.shape[0]):
            warnings.warn("Warning: using 'NA_as_zero', but passed dense 'X'.")
        if (self.NA_as_zero_user) and (Uarr.shape[0]):
            warnings.warn("Warning: using 'NA_as_zero_user', but passed dense 'U'.")
        if (self.NA_as_zero_item) and (Iarr.shape[0]):
            warnings.warn("Warning: using 'NA_as_zero_item', but passed dense 'I'.")


        return self._fit(Xrow, Xcol, Xval, W_sp, Xarr, W_dense,
                         Uarr, Urow, Ucol, Uval, Ub_arr,
                         Iarr, Irow, Icol, Ival, Ib_arr,
                         m, n, m_u, n_i, p, q,
                         m_ub, n_ib, pbin, qbin)

    def predict(self, user, item):
        """
        Predict ratings/values given by existing users to existing items

        Note
        ----
        For CMF explicit, invalid combinations of users and items will be
        set to the global mean plus biases if applicable. For other models,
        invalid combinations will be set as NaN.

        Parameters
        ----------
        user : array-like(n,)
            Users for whom ratings/values are to be predicted. If 'X' passed to
            fit was a  DataFrame, must match with the entries in its 'UserId'
            column, otherwise should match with the rows of 'X'.
        item : array-like(n,)
            Items for whom ratings/values are to be predicted. If 'X' passed to
            fit was a  DataFrame, must match with the entries in its 'ItemId'
            column, otherwise should match with the columns of 'X'.
            Each entry in ``item`` will be matched with the corresponding entry
            of ``user`` at the same position in the array/list.

        Returns
        -------
        scores : array(n,)
            Predicted ratings for the requested user-item combinations.
        """
        if user is None and item is None:
            raise ValueError("Must pass valid user(s) and item(s).")
        return self._predict(user=user, a_vec=None, a_bias=0., item=item)

    def _predict(self, user=None, a_vec=None, a_bias=0., item=None):
        assert self.is_fitted_
        if self._only_prediction_info:
            raise ValueError("Cannot use this function after dropping non-essential matrices.")

        user, item, _1, _2 = self._process_users_items(user, item, None, None)

        c_funs = wrapper_float if self.use_float else wrapper_double

        if user is not None:
            assert user.shape[0] == item.shape[0]
        
            if user.shape[0] == 1:
                if (user[0] == -1) or (item[0] == -1):
                    if isinstance(self, CMF):
                        out = self.glob_mean_
                        if (user[0] >= 0) and (self.user_bias):
                            out += self.user_bias_[user]
                        if (item[0] >= 0) and (self.item_bias):
                            out += self.item_bias_[item]
                        if (self.center) or (self.user_bias and user[0] >= 0) or (self.item_bias and item[0] >= 0):
                            return out
                    return np.nan
                else:
                    out = self._A_pred[user, self.k_user:].dot(self._B_pred[item, self.k_item:].T).reshape(-1)[0]
                    out += self.glob_mean_
                    if self.user_bias:
                        out += self.user_bias_[user]
                    if self.item_bias:
                        out += self.item_bias_[item]
                    if isinstance(out, np.ndarray):
                        out = out[0]
                    return out
            else:
                n_users = max(self._A_pred.shape[0], self.user_bias_.shape[0])
                n_items = max(self._B_pred.shape[0], self.item_bias_.shape[0])
                if isinstance(self, CMF):
                    return c_funs.call_predict_X_old_collective_explicit(
                        self._A_pred,
                        self._B_pred,
                        self.user_bias_,
                        self.item_bias_,
                        self.glob_mean_,
                        np.array(user).astype(ctypes.c_int),
                        np.array(item).astype(ctypes.c_int),
                        self._k_pred, self.k_user, self.k_item, self._k_main_col,
                        self.nthreads
                    )
                else:
                    return c_funs.call_predict_multiple(
                        self._A_pred,
                        self._B_pred,
                        self.user_bias_,
                        self.item_bias_,
                        self.glob_mean_,
                        np.array(user).astype(ctypes.c_int),
                        np.array(item).astype(ctypes.c_int),
                        self._k_pred, self.k_user, self.k_item, self._k_main_col,
                        self.nthreads
                    )

        #### When passing the factors directly
        else:
            item = np.array([item]).reshape(-1)
            nan_entries = (item == -1)
            outp = self._B_pred[item, self.k_item:].reshape((item.shape[0],-1)).dot(a_vec[self.k_user:])
            outp += a_bias + self.glob_mean_
            if self.item_bias:
                outp += self.item_bias_[item]
            outp[nan_entries] = np.nan
            return outp

    def _predict_new(self, user, B):
        n = B.shape[0]
        user, _1, _2, _3 = self._process_users_items(user, None, None, None)
        nan_entries = (user < 0) | \
                      (user >= max(self._A_pred.shape[0], self.user_bias_.shape[0]))

        c_funs = wrapper_float if self.use_float else wrapper_double

        if user.shape[0] != n:
            raise ValueError("'user' must have the same number of entries as item info.")

        return c_funs.call_predict_multiple(
                    self._A_pred,
                    B,
                    self.user_bias_,
                    np.zeros(n, dtype=self.dtype_) if self.item_bias \
                        else np.empty(0, dtype=self.dtype_),
                    self.glob_mean_,
                    np.array(user).astype(ctypes.c_int),
                    np.arange(n).astype(ctypes.c_int),
                    self._k_pred, self.k_user, self.k_item, self._k_main_col,
                    self.nthreads
                )

    def _predict_user_multiple(self, A, item, bias=None):
        m = A.shape[0]
        _1, item, _2, _3 = self._process_users_items(None, item, None, None)
        nan_entries = (item < 0) | \
                      (item >= max(self._B_pred.shape[0], self.item_bias_.shape[0]))

        c_funs = wrapper_float if self.use_float else wrapper_double

        if item.shape[0] != m:
            raise ValueError("'item' must have the same number of entries as user info.")

        if bias is None:
            bias = np.zeros(m, dtype=self.dtype_) if self.user_bias \
                        else np.empty(0, dtype=self.dtype_)

        if isinstance(self, CMF):
            return c_funs.call_predict_X_old_collective_explicit(
                        A,
                        self._B_pred,
                        bias,
                        self.item_bias_,
                        self.glob_mean_,
                        np.arange(m).astype(ctypes.c_int),
                        np.array(item).astype(ctypes.c_int),
                        self._k_pred, self.k_user, self.k_item, self._k_main_col,
                        self.nthreads
                    )
        else:
            return c_funs.call_predict_multiple(
                        A,
                        self._B_pred,
                        bias,
                        self.item_bias_,
                        self.glob_mean_,
                        np.arange(m).astype(ctypes.c_int),
                        np.array(item).astype(ctypes.c_int),
                        self._k_pred, self.k_user, self.k_item, self._k_main_col,
                        self.nthreads
                    )

    def topN(self, user, n=10, include=None, exclude=None, output_score=False):
        """
        Rank top-N highest-predicted items for an existing user

        Parameters
        ----------
        user : int or obj
            User for which to rank the items. If 'X' passed to 'fit' was a
            DataFrame, must match with the entries in its 'UserId' column,
            otherwise should match with the rows of 'X'.
        n : int
            Number of top-N highest-predicted results to output.
        include : array-like
            List of items which will be ranked. If passing this, will only
            make a ranking among these items. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include or 'exclude'.
        exclude : array-like
            List of items to exclude from the ranking. If passing this, will
            rank all the items except for these. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include or 'exclude'.
        output_score : bool
            Whether to output the scores in addition to the IDs. If passing
            'False', will return a single array with the item IDs, otherwise
            will return a tuple with the item IDs and the scores.

        Returns
        -------
        items : array(n,)
            The top-N highest predicted items for this user. If the 'X' data passed to
            fit was a DataFrame, will contain the item IDs from its column
            'ItemId', otherwise will be integers matching to the columns of 'X'.
        scores : array(n,)
            The predicted scores for the top-N items. Will only be returned
            when passing ``output_score=True``, in which case the result will
            be a tuple with these two entries.
        """
        if user is None:
            raise ValueError("Must pass a valid user.")
        
        return self._topN(user=user, a_vec=None, a_bias=None, n=n,
                          include=include, exclude=exclude,
                          output_score=output_score)

    def _topN(self, user=None, a_vec=None, a_bias=0, B=None,
              n=10, include=None, exclude=None, output_score=False):
        assert self.is_fitted_
        if self._only_prediction_info:
            raise ValueError("Cannot use this function after dropping non-essential matrices.")
        
        user, _, include, exclude = self._process_users_items(user, None, include, exclude)

        c_funs = wrapper_float if self.use_float else wrapper_double

        if (include.shape[0] > 0) and (include.shape[0] < n):
            raise ValueError("'include' has fewer than 'n' entries.")
        if (exclude.shape[0] > 0) and ((self._B_pred.shape[0] - exclude.shape[0]) < n):
            msg  = "'exclude' has a number of entries which leaves behind "
            msg += "fewer than 'n' to rank."
            raise ValueError(msg)

        if user is not None:
            user = user[0]
            a_vec = self._A_pred[user].reshape(-1)
        user_bias_ = 0.
        if self.user_bias:
            if user is not None:
                user_bias_ = self.user_bias_[user]
            else:
                user_bias_ = a_bias
        outp_ix, outp_score = c_funs.call_topN(
            a_vec,
            (self._B_pred[:self._n_orig] if not self.include_all_X else self._B_pred) if B is None else B,
            self.item_bias_ if B is None else \
                (np.zeros(n, dtype=self.dtype_) if self.item_bias \
                            else np.empty(0, dtype=self.dtype_)),
            self.glob_mean_, user_bias_,
            include,
            exclude,
            n,
            self._k_pred, self.k_user, self.k_item, self._k_main_col,
            bool(output_score),
            self.nthreads
        )

        if (self.reindex_) and (B is None):
            outp_ix = self.item_mapping_[outp_ix]
        if output_score:
            return outp_ix, outp_score
        else:
            return outp_ix

    def _factors_cold(self, U=None, U_bin=None, U_col=None, U_val=None):
        assert self.is_fitted_
        if (self.C_.shape[0] == 0) and (self.Cbin_.shape[0] == 0):
            raise ValueError("Method is only available when fitting the model to user side info.")

        c_funs = wrapper_float if self.use_float else wrapper_double

        U, U_col, U_val, U_bin = self._process_new_U(U, U_col, U_val, U_bin)

        if isinstance(self.lambda_, np.ndarray):
            lambda_ = self.lambda_[2]
            lambda_bias = self.lambda_[0]
        else:
            lambda_ = self.lambda_
            lambda_bias = self.lambda_

        if isinstance(self.l1_lambda, np.ndarray):
            l1_lambda = self.l1_lambda[2]
            l1_lambda_bias = self.l1_lambda[0]
        else:
            l1_lambda = self.l1_lambda
            l1_lambda_bias = self.l1_lambda
        
        if not self._implicit:
            _, a_vec = c_funs.call_factors_collective_explicit_single(
                    np.empty(0, dtype=self.dtype_),
                    np.empty(0, dtype=self.dtype_),
                    np.empty(0, dtype=self.dtype_),
                    np.empty(0, dtype=ctypes.c_int),
                    np.empty(0, dtype=self.dtype_),
                    U,
                    U_val,
                    U_col,
                    U_bin,
                    self._U_colmeans,
                    self.item_bias_,
                    self.B_,
                    self._B_plus_bias,
                    self.C_,
                    self.Cbin_,
                    self.Bi_,
                    self._BtB,
                    self._TransBtBinvBt,
                    self._BtXbias,
                    self._BeTBeChol,
                    self._BiTBi,
                    self._CtC,
                    self._TransCtCinvCt,
                    self._CtUbias,
                    self.glob_mean_,
                    self._n_orig,
                    self.k, self.k_user, self.k_item, self.k_main,
                    lambda_, lambda_bias,
                    l1_lambda, l1_lambda_bias,
                    self.scale_lam, self.scale_lam_sideinfo,
                    self.scale_bias_const, self.scaling_biasA_,
                    self.w_user, self.w_main, self.w_implicit,
                    self.user_bias,
                    self.NA_as_zero_user, self.NA_as_zero,
                    self.nonneg,
                    self.add_implicit_features,
                    self.include_all_X
            )
        else:
            a_vec = c_funs.call_factors_collective_implicit_single(
                np.empty(0, dtype=self.dtype_),
                np.empty(0, dtype=ctypes.c_int),
                U,
                U_val,
                U_col,
                self._U_colmeans,
                self.B_,
                self.C_,
                self._BeTBe,
                self._BtB,
                self._BeTBeChol,
                self._CtUbias,
                self.k, self.k_user, self.k_item, self.k_main,
                lambda_, l1_lambda, self.alpha,
                self._w_main_multiplier,
                self.w_user, self.w_main,
                self.apply_log_transf,
                self.NA_as_zero_user,
                self.nonneg
            )
        return a_vec

    def _factors_warm_common(self, X=None, X_col=None, X_val=None, W=None,
                             U=None, U_bin=None, U_col=None, U_val=None,
                             return_bias=False, exact=False, output_a=False):
        assert self.is_fitted_

        if (return_bias) and (not self.user_bias):
            raise ValueError("Cannot return bias with model that was fit without it.")

        if ((X_col is not None) and (X_val is None)) or ((X_col is None) and (X_val is  not None)):
            raise ValueError("Must pass 'X_col' and 'X_val' together.")
        if (X_col is not None) and (X is not None):
            raise ValueError("Can only pass 'X' in one format.")
        if (X is None) and (X_col is None):
            raise ValueError("Must pass 'X' in some format.")
        if (self.C_.shape[0] == 0) and (U is not None or U_col is not None or U_val is not None):
            raise ValueError("Cannot pass user information if the model was not fit to it.")
        if (self.Cbin_.shape[0] == 0) and (U_bin is not None):
            raise ValueError("Cannot pass binary user information if the model was not fit to it.")

        if self.copy_data:
            X = X.copy() if X is not None else None
            X_col = X_col.copy() if X_col is not None else None
            X_val = X_val.copy() if X_val is not None else None
            W = W.copy() if W is not None else None

        if (U is not None) or (U_val is not None) or (U_bin is not None):
            U, U_col, U_val, U_bin = self._process_new_U(U, U_col, U_val, U_bin)
        else:
            U = np.empty(0, dtype=self.dtype_)
            U_bin = np.empty(0, dtype=self.dtype_)
            U_val = np.empty(0, dtype=self.dtype_)
            U_col = np.empty(0, dtype=ctypes.c_int)

        if X is not None:
            X_col = np.empty(0, dtype=ctypes.c_int)
            X_val = np.empty(0, dtype=self.dtype_)
            W_sp = np.empty(0, dtype=self.dtype_)
            if len(X.shape) > 1:
                warnings.warn("Passed a 2-d array for 'X' - method expects a single row.")
            X = np.array(X).reshape(-1).astype(self.dtype_)
            if X.shape[0] != self._n_orig:
                raise ValueError("'X' must have the same columns as when passed to 'fit'.")
            if W is not None:
                W_dense = np.array(W).reshape(-1).astype(self.dtype_)
                if W_dense.shape[0] != X.shape[0]:
                    raise ValueError("'W' must have the same number of entries as X.")
            else:
                W_dense = np.empty(0, dtype=self.dtype_)
        else:
            X = np.empty(0, dtype=self.dtype_)
            W_dense = np.empty(0, dtype=self.dtype_)
            X_val = np.array(X_val).reshape(-1).astype(self.dtype_)

            if X_val.shape[0] == 0:
                X_col = np.array(X_col).reshape(-1).astype(ctypes.c_int)
                if X_col.shape[0] > 0:
                    raise ValueError("'X_col' and 'X_val' must have the same number of entries.")
            else:
                if self.reindex_:
                    X_col = np.array(X_col).reshape(-1)
                    X_col = pd.Categorical(X_col, self.item_mapping_).codes.astype(ctypes.c_int)
                    if np.any(X_col < 0):
                        raise ValueError("'X_col' must have the same item/column entries as passed to 'fit'.")
                else:
                    X_col = np.array(X_col).reshape(-1).astype(ctypes.c_int)
                    imin, imax = np.min(X_col), np.max(X_col)
                    if (imin < 0) or (imax >= self._n_orig) or np.isnan(imin) or np.isnan(imax):
                        msg  = "Column indices ('X_col') must be within the range"
                        msg += " of the data that was pased to 'fit'."
                        raise ValueError(msg)
                if X_col.max() >= self._n_orig:
                    raise ValueError("'X' cannot contain new columns.")

            if X_val.shape[0] != X_col.shape[0]:
                raise ValueError("'X_col' and 'X_val' must have the same number of entries.")
            if X_val.shape[0] == 0:
                raise ValueError("'X' is empty.")

            if W is not None:
                W_sp = np.array(W).reshape(-1).astype(self.dtype_)
                if W_sp.shape[0] != X_col.shape[0]:
                    raise ValueError("'W' must have the same number of entries as 'X_val'.")
            else:
                W_sp = np.empty(0, dtype=self.dtype_)

        if self.apply_log_transf:
            if Xval.min() < 1:
                raise ValueError("Cannot pass values below 1 with 'apply_log_transf=True'.")

        if not isinstance(self, OMF_explicit) and not isinstance(self, OMF_implicit):
            return self._factors_warm(X, W_dense, X_val, X_col, W_sp,
                                      U, U_val, U_col, U_bin, return_bias)
        elif isinstance(self, OMF_implicit):
            return self._factors_warm(X, W_dense, X_val, X_col, W_sp,
                                      U, U_val, U_col, U_bin, bool(output_a))
        else:
            return self._factors_warm(X, W_dense, X_val, X_col, W_sp,
                                      U, U_val, U_col, U_bin, return_bias,
                                      bool(exact), bool(output_a))

    def _process_transform_inputs(self, X, U, U_bin, W, replace_existing):
        if (X is None) and (U is None) and (U_bin is None):
            if (self.Cbin_.shape[0]) or (self.Dbin_.shape[0]):
                raise ValueError("Must pass at least one of 'X', 'U', 'U_bin'.")
            else:
                raise ValueError("Must pass at least one of 'X', 'U'.")
        if (not replace_existing):
            if (X is None):
                raise ValueError("Must pass 'X' if not passing 'replace_existing'.")
            if X.__class__.__name__ == "ndarray":
                mask_take = ~pd.isnull(X)
            elif X.__class__.__name__ == "coo_matrix":
                mask_take = np.repeat(False, X.shape[0]*X.shape[1]).reshape((X.shape[0], X.shape[1]))
                mask_take[X.row, X.col] = True
            else:
                raise ValueError("'X' must be a SciPy COO matrix or NumPy array.")

            Xorig = X.copy()
        else:
            mask_take = None
            Xorig = None

        Xarr, Xrow, Xcol, Xval, Xcsr_p, Xcsr_i, Xcsr, m_x, n, W_dense, W_sp = \
            self._process_new_X_2d(X=X, W=W)
        Uarr, Urow, Ucol, Uval, Ucsr_p, Ucsr_i, Ucsr, m_u, p = \
            self._process_new_U_2d(U=U, is_I=False, allow_csr=True)
        Ub_arr, m_ub, pbin = self._process_new_Ub_2d(U_bin=U_bin, is_I=False)

        msg  = "'X' and '%s' must have the same rows. "
        msg += "Non present values should be passed as np.nan for dense, "
        msg += "or missing with matching shapes for sparse."
        if (m_x > 0) and (m_u > 0) and (m_x != m_u):
            if (min(m_x, m_u) == m_x) and (Xcsr_p.shape[0] or Xval.shape[0]):
                if Xcsr_p.shape[0]:
                    diff = m_u - m_x
                    fill = Xcsr_p[-1]
                    Xcsr_p = np.r_[Xcsr_p, np.repeat(fill, diff)]
                else:
                    m_x = m_u
            elif (min(m_x, m_u) == m_u) and (Uval.shape[0] or Ucsr_p.shape[0]):
                if Ucsr_p.shape[0]:
                    diff = m_x - m_u
                    fill = Ucsr_p[-1]
                    Ucsr_p = np.r_[Ucsr_p, np.repeat(fill, diff)]
                else:
                    m_u = m_x
            else:
                raise ValueError(msg % "U")
        if (m_x > 0) and (m_ub > 0) and (m_x != m_ub):
            if (min(m_x, m_ub) == m_x) and (Xcsr_p.shape[0]):
                diff = m_ub - m_x
                fill = Xcsr_p[-1]
                Xcsr_p = np.r_[Xcsr_p, np.repeat(fill, diff)]
            else:
                raise ValueError(msg % "U_bin")

        if isinstance(self.lambda_, np.ndarray):
            lambda_ = self.lambda_[2]
            lambda_bias = self.lambda_[0]
        else:
            lambda_ = self.lambda_
            lambda_bias = self.lambda_

        if isinstance(self.l1_lambda, np.ndarray):
            l1_lambda = self.l1_lambda[2]
            l1_lambda_bias = self.l1_lambda[0]
        else:
            l1_lambda = self.l1_lambda
            l1_lambda_bias = self.l1_lambda

        return Xrow, Xcol, Xval, W_sp, Xarr, \
               Xcsr_p, Xcsr_i, Xcsr, \
               W_dense, Xorig, mask_take, \
               Uarr, Urow, Ucol, Uval, Ub_arr, \
               Ucsr_p, Ucsr_i, Ucsr, \
               n, m_u, m_x, p, pbin, \
               lambda_, lambda_bias, \
               l1_lambda, l1_lambda_bias

    def _transform_step(self, A, A_bias, mask_take, Xorig):
        outp = A[:, self.k_user:].dot(self._B_pred[:, self.k_item:].T) \
                + self.glob_mean_
        if self.user_bias:
            outp += A_bias.reshape((-1,1))
        if self.item_bias:
            outp += self.item_bias_.reshape((1,-1))

        if mask_take is not None:
            if Xorig.__class__.__name__ == "ndarray":
                outp[mask_take] = Xorig[mask_take]
            elif Xorig.__class__.__name__ == "coo_matrix":
                outp[mask_take] = Xorig.data
            else:
                raise ValueError("'X' must be a SciPy COO matrix or NumPy array.")

        return outp

    def _process_multiple_common(self, X, U, U_bin, W):
        if (X is None) and (U is None) and (U_bin is None):
            if (self.Cbin_.shape[0]) or (self.Dbin_.shape[0]):
                raise ValueError("Must pass at least one of 'X', 'U', 'U_bin'.")
            else:
                raise ValueError("Must pass at least one of 'X', 'U'.")

        Xarr, Xrow, Xcol, Xval, Xcsr_p, Xcsr_i, Xcsr, m_x, n, W_dense, W_sp = \
            self._process_new_X_2d(X=X, W=W)
        Uarr, Urow, Ucol, Uval, Ucsr_p, Ucsr_i, Ucsr, m_u, p = \
            self._process_new_U_2d(U=U, is_I=False, allow_csr=True)
        Ub_arr, m_ub, pbin = self._process_new_Ub_2d(U_bin=U_bin, is_I=False)

        if (self.NA_as_zero) and (Xcsr_p.shape[0]) and (m_x < max(m_u, m_ub)):
            diff = max(m_u, m_ub) - m_x
            fill = Xcsr_p[-1]
            Xcsr_p = np.r_[Xcsr_p, np.repeat(fill, diff)]
            m_x = max(m_x, m_u, m_ub)

        if (self.NA_as_zero_user) and (Xcsr_p.shape[0]) and (m_u < max(m_x, m_ub)):
            diff = max(m_x, m_ub) - m_u
            fill = Ucsr_p[-1]
            Ucsr_p = np.r_[Ucsr_p, np.repeat(fill, diff)]
            m_u = max(m_x, m_u, m_ub)

        if isinstance(self.lambda_, np.ndarray):
            lambda_ = self.lambda_[2]
            lambda_bias = self.lambda_[0]
        else:
            lambda_ = self.lambda_
            lambda_bias = self.lambda_

        if isinstance(self.l1_lambda, np.ndarray):
            l1_lambda = self.l1_lambda[2]
            l1_lambda_bias = self.l1_lambda[0]
        else:
            l1_lambda = self.l1_lambda
            l1_lambda_bias = self.l1_lambda

        if self.apply_log_transf:
            msg_small ="Cannot pass values below 1 with 'apply_log_transf=True'."
            if Xval.shape[0]:
                if Xval.min() < 1:
                    raise ValueError(msg_small)
            if Xcsr.shape[0]:
                if Xcsr.min() < 1:
                    raise ValueError(msg_small)

        return Xrow, Xcol, Xval, W_sp, \
               Xcsr_p, Xcsr_i, Xcsr, \
               Xarr, W_dense, \
               Uarr, Urow, Ucol, Uval, Ub_arr, \
               Ucsr_p, Ucsr_i, Ucsr, \
               n, m_u, m_x, p, pbin, \
               lambda_, lambda_bias, \
               l1_lambda, l1_lambda_bias

    def _factors_multiple_common(self, X, U, U_bin, W):
        Xrow, Xcol, Xval, W_sp, \
        Xcsr_p, Xcsr_i, Xcsr, \
        Xarr, W_dense, \
        Uarr, Urow, Ucol, Uval, Ub_arr, \
        Ucsr_p, Ucsr_i, Ucsr, \
        n, m_u, m_x, p, pbin, \
        lambda_, lambda_bias, \
        l1_lambda, l1_lambda_bias = self._process_multiple_common(X, U, U_bin, W)
        A, A_bias = self._factors_multiple(
            Xrow, Xcol, Xval, W_sp,
            Xcsr_p, Xcsr_i, Xcsr,
            Xarr, W_dense,
            Uarr, Urow, Ucol, Uval, Ub_arr,
            Ucsr_p, Ucsr_i, Ucsr,
            n, m_u, m_x, p, pbin,
            lambda_, lambda_bias,
            l1_lambda, l1_lambda_bias
        )
        return A, A_bias

    def _factors_multiple(self,
                          Xrow, Xcol, Xval, W_sp,
                          Xcsr_p, Xcsr_i, Xcsr,
                          Xarr, W_dense,
                          Uarr, Urow, Ucol, Uval, Ub_arr,
                          Ucsr_p, Ucsr_i, Ucsr,
                          n, m_u, m_x, p, pbin,
                          lambda_, lambda_bias,
                          l1_lambda, l1_lambda_bias):
        c_funs = wrapper_float if self.use_float else wrapper_double
        
        if (not self._implicit):
            A, A_bias = c_funs.call_factors_collective_explicit_multiple(
                Xrow,
                Xcol,
                Xval,
                Xcsr_p, Xcsr_i, Xcsr,
                W_sp,
                Xarr,
                W_dense,
                Uarr,
                Urow,
                Ucol,
                Uval,
                Ucsr_p, Ucsr_i, Ucsr,
                Ub_arr,
                self._U_colmeans,
                self.item_bias_,
                self._B_pred,
                self._B_plus_bias,
                self.Bi_,
                self.C_,
                self.Cbin_,
                self._BtB,
                self._TransBtBinvBt,
                self._BtXbias,
                self._BeTBeChol,
                self._BiTBi,
                self._TransCtCinvCt,
                self._CtC,
                self._CtUbias,
                m_u, m_x,
                self.glob_mean_,
                self._n_orig,
                self._k_pred, self.k_user, self.k_item, self._k_main_col,
                lambda_, lambda_bias,
                l1_lambda, l1_lambda_bias,
                self.scale_lam, self.scale_lam_sideinfo,
                self.scale_bias_const, self.scaling_biasA_,
                self.w_user, self.w_main, self.w_implicit,
                self.user_bias,
                self.NA_as_zero_user, self.NA_as_zero,
                self.nonneg,
                self.add_implicit_features,
                self.include_all_X,
                self.nthreads
            )

        else:
            A_bias = np.zeros(0, dtype=self.dtype_)
            A = c_funs.call_factors_collective_implicit_multiple(
                Xrow,
                Xcol,
                Xval,
                Xcsr_p, Xcsr_i, Xcsr,
                Uarr,
                Urow,
                Ucol,
                Uval,
                Ucsr_p, Ucsr_i, Ucsr,
                self._U_colmeans,
                self.B_,
                self.C_,
                self._BeTBe,
                self._BtB,
                self._BeTBeChol,
                self._CtUbias,
                n, m_u, m_x,
                self.k, self.k_user, self.k_item, self.k_main,
                lambda_, l1_lambda, self.alpha,
                self._w_main_multiplier,
                self.w_user, self.w_main,
                self.apply_log_transf,
                self.NA_as_zero_user,
                self.nonneg,
                self.nthreads
            )

        return A, A_bias

    def _item_factors_cold(self, I=None, I_bin=None, I_col=None, I_val=None):
        assert self.is_fitted_
        if self._only_prediction_info:
            raise ValueError("Cannot use this function after dropping non-essential matrices.")
        if (self.D_.shape[0] == 0) and (self.Dbin_.shape[0] == 0):
            msg  = "Can only use this method when "
            msg += "fitting the model to item side info."
            raise ValueError(msg)

        if isinstance(self.lambda_, np.ndarray):
            lambda_ = self.lambda_[3]
            lambda_bias = self.lambda_[1]
        else:
            lambda_ = self.lambda_
            lambda_bias = self.lambda_

        if isinstance(self.l1_lambda, np.ndarray):
            l1_lambda = self.l1_lambda[3]
            lambda_bias = self.l1_lambda[1]
        else:
            l1_lambda = self.l1_lambda
            lambda_bias = self.l1_lambda

        I, I_col, I_val, I_bin = self._process_new_U(U=I, U_col=I_col, U_val=I_val, U_bin=I_bin, is_I=True)

        c_funs = wrapper_float if self.use_float else wrapper_double
        
        if (not self._implicit):
            _, b_vec = c_funs.call_factors_collective_explicit_single(
                np.empty(0, dtype=self.dtype_),
                np.empty(0, dtype=self.dtype_),
                np.empty(0, dtype=self.dtype_),
                np.empty(0, dtype=ctypes.c_int),
                np.empty(0, dtype=self.dtype_),
                I,
                I_val,
                I_col,
                I_bin,
                self._I_colmeans,
                self.user_bias_,
                self.A_,
                np.empty((0,0), dtype=self.dtype_),
                self.D_,
                self.D_bin_,
                self.Ai_,
                np.empty((0,0), dtype=self.dtype_),
                np.empty((0,0), dtype=self.dtype_),
                np.empty(0, dtype=self.dtype_),
                np.empty((0,0), dtype=self.dtype_),
                np.empty((0,0), dtype=self.dtype_),
                np.empty((0,0), dtype=self.dtype_),
                np.empty((0,0), dtype=self.dtype_),
                np.empty(0, dtype=self.dtype_),
                self.glob_mean_,
                self.A_.shape[0],
                self.k, self.k_item, self.k_user, self.k_main,
                lambda_, lambda_bias,
                l1_lambda, l1_lambda_bias,
                self.scale_lam, self.scale_lam_sideinfo,
                self.scale_bias_const, self.scaling_biasB_,
                self.w_item, self.w_main, self.w_implicit,
                self.item_bias,
                self.NA_as_zero_item, self.NA_as_zero,
                self.nonneg,
                self.add_implicit_features,
                False
            )
        else:
            b_vec = c_funs.call_factors_collective_implicit_single(
                np.empty(0, dtype=self.dtype_),
                np.empty(0, dtype=ctypes.c_int),
                I,
                I_val,
                I_col,
                self._I_colmeans,
                self.A_,
                self.D_,
                np.empty((0,0), dtype=self.dtype_),
                np.empty((0,0), dtype=self.dtype_),
                np.empty((0,0), dtype=self.dtype_),
                np.empty(0, dtype=self.dtype_),
                self.k, self.k_item, self.k_user, self.k_main,
                lambda_, l1_lambda, self.alpha,
                self._w_main_multiplier,
                self.w_item, self.w_main,
                self.apply_log_transf,
                self.NA_as_zero_item,
                self.nonneg
            )
        return b_vec

    def _factors_cold_multiple(self, U=None, U_bin=None, is_I=False):
        assert self.is_fitted_

        letter = "U" if not is_I else "I"
        infoname = "user" if not is_I else "item"
        Mat = self.C_ if not is_I else self.D_
        MatBin = self.Cbin_ if not is_I else self.Dbin_

        if (U is None) and (U_bin is None):
            raise ValueError("Must pass at least one of '%s' or '%s_bin'." %
                             (letter, letter))
        if (Mat.shape[0] == 0) and (MatBin.shape[0] == 0):
            msg  = "Can only use this method when "
            msg += "fitting the model to %s side info."
            raise ValueError(msg % infoname)

        msg  = "Can only use %s side info when the model was fit to it."
        if (Mat.shape[0] == 0) and (U is not None):
            raise ValueError(msg % infoname)
        if (MatBin.shape[0] == 0) and (U_bin is not None):
            raise ValueError(msg % (infoname + " binary"))
        if (U is not None) and (len(U.shape) != 2):
            raise ValueError("'%s' must be 2-dimensional." % letter)
        if (U_bin is not None) and (len(U_bin.shape) != 2):
            raise ValueError("'%s_bin' must be 2-dimensional." % letter)

        if isinstance(self.lambda_, np.ndarray):
            if not is_I:
                lambda_ = self.lambda_[2]
                lambda_bias = self.lambda_[0]
            else:
                lambda_ = self.lambda_[3]
                lambda_bias = self.lambda_[1]
        else:
            lambda_ = self.lambda_
            lambda_bias = self.lambda_

        if isinstance(self.l1_lambda, np.ndarray):
            if not is_I:
                l1_lambda = self.l1_lambda[2]
                l1_lambda_bias = self.l1_lambda[0]
            else:
                l1_lambda = self.l1_lambda[3]
                l1_lambda_bias = self.l1_lambda[1]
        else:
            l1_lambda = self.l1_lambda
            l1_lambda_bias = self.l1_lambda

        Uarr, Urow, Ucol, Uval, Ucsr_p, Ucsr_i, Ucsr, m_u, p = \
            self._process_new_U_2d(U=U, is_I=is_I, allow_csr=True)
        Ub_arr, m_ub, pbin = self._process_new_Ub_2d(U_bin=U_bin, is_I=is_I)

        empty_arr = np.empty((0,0), dtype=self.dtype_)

        c_funs = wrapper_float if self.use_float else wrapper_double

        if (not self._implicit):
            A, _ = c_funs.call_factors_collective_explicit_multiple(
                np.empty(0, dtype=ctypes.c_int),
                np.empty(0, dtype=ctypes.c_int),
                np.empty(0, dtype=self.dtype_),
                np.empty(0, dtype=ctypes.c_size_t),
                np.empty(0, dtype=ctypes.c_int),
                np.empty(0, dtype=self.dtype_),
                np.empty(0, dtype=self.dtype_),
                np.empty((0,0), dtype=self.dtype_),
                np.empty((0,0), dtype=self.dtype_),
                Uarr,
                Urow,
                Ucol,
                Uval,
                Ucsr_p, Ucsr_i, Ucsr,
                Ub_arr,
                self._U_colmeans if not is_I else self._I_colmeans,
                self.item_bias_,
                self.B_ if not is_I else self.A_,
                self._B_plus_bias if not is_I else empty_arr,
                self.Bi_ if not is_I else self.Ai_,
                Mat,
                MatBin,
                self._BtB if not is_I else empty_arr,
                self._TransBtBinvBt if not is_I else empty_arr,
                self._BtXbias if not is_I else np.empty(0, dtype=self.dtype_),
                self._BeTBeChol if not is_I else empty_arr,
                self._BiTBi if not is_I else empty_arr,
                self._TransCtCinvCt if not is_I else empty_arr,
                self._CtC if not is_I else empty_arr,
                self._CtUbias if not is_I else np.empty(0, dtype=self.dtype_),
                m_u, 0,
                self.glob_mean_,
                self._n_orig if not is_I else self.A_.shape[0],
                self.k,
                self.k_user if not is_I else self.k_item,
                self.k_item if not is_I else self.k_user,
                self.k_main,
                lambda_, lambda_bias,
                l1_lambda, l1_lambda_bias,
                self.scale_lam, self.scale_lam_sideinfo,
                self.scale_bias_const, self.scaling_biasA_ if not is_I else self.scaling_biasB_,
                self.w_user if not is_I else self.w_item,
                self.w_main, self.w_implicit,
                self.user_bias if not is_I else self.item_bias,
                self.NA_as_zero_user if not is_I else self.NA_as_zero_item,
                self.NA_as_zero,
                self.nonneg,
                self.add_implicit_features,
                self.include_all_X if not is_I else True,
                self.nthreads
            )
        else:
            A = c_funs.call_factors_collective_implicit_multiple(
                    np.empty(0, dtype=ctypes.c_int),
                    np.empty(0, dtype=ctypes.c_int),
                    np.empty(0, dtype=self.dtype_),
                    np.empty(0, dtype=ctypes.c_size_t),
                    np.empty(0, dtype=ctypes.c_int),
                    np.empty(0, dtype=self.dtype_),
                    Uarr,
                    Urow,
                    Ucol,
                    Uval,
                    Ucsr_p, Ucsr_i, Ucsr,
                    self._U_colmeans if not is_I else self._I_colmeans,
                    self.B_ if not is_I else self.A_,
                    Mat,
                    self._BeTBe if not is_I else empty_arr,
                    self._BtB if not is_I else empty_arr,
                    self._BeTBeChol if not is_I else empty_arr,
                    self._CtUbias if not is_I else np.empty(0, dtype=self.dtype_),
                    n, m_u, 0,
                    self.k,
                    self.k_user if not is_I else self.k_item,
                    self.k_item if not is_I else self.k_user,
                    self.k_main,
                    lambda_, l1_lambda, self.alpha,
                    self._w_main_multiplier,
                    self.w_user if not is_I else self.w_item, self.w_main,
                    self.apply_log_transf,
                    self.NA_as_zero_user if not is_I else self.NA_as_zero_item,
                    self.nonneg,
                    self.nthreads
                )
        return A

    def swap_users_and_items(self, precompute = True):
        """
        Swap the users and items in a factorization model

        This method will generate a new object that will have the users
        and items of this object swapped, and such result can be used under
        the same methods such as ``topN``, in which any mention of users will
        now mean items and vice-versa.

        Note
        ----
        The resulting object will not generate any deep copies of the
        original model's objects.

        Parameters
        ----------
        precompute : bool
            Whether to produce the precomputed matrices which might help
            to speed up predictions on new data.

        Returns
        -------
        model : obj
            An object of the same class as this one, but with the user
            and items swapped.
        """
        assert self.is_fitted_
        if self._only_prediction_info:
            raise ValueError("Cannot use this function after dropping non-essential matrices.")

        new_lambda = self.lambda_
        if isinstance(new_lambda, np.ndarray) and (new_lambda.shape[0] == 6):
            new_lambda = self.lambda_.copy()
            new_lambda[0], new_lambda[1] = new_lambda[1], new_lambda[0]
            new_lambda[2], new_lambda[3] = new_lambda[3], new_lambda[2]
            new_lambda[4], new_lambda[5] = new_lambda[5], new_lambda[4]

        new_l1_lambda = self.l1_lambda
        if isinstance(new_l1_lambda, np.ndarray) and (new_l1_lambda.shape[0] == 6):
            new_l1_lambda = self.l1_lambda.copy()
            new_l1_lambda[0], new_l1_lambda[1] = new_l1_lambda[1], new_l1_lambda[0]
            new_l1_lambda[2], new_l1_lambda[3] = new_l1_lambda[3], new_l1_lambda[2]
            new_l1_lambda[4], new_l1_lambda[5] = new_l1_lambda[5], new_l1_lambda[4]

        if isinstance(self, CMF):
            new_model = CMF(
                k=self.k, lambda_=new_lambda, method=self.method, use_cg=self.use_cg,
                user_bias=self.item_bias, item_bias=self.user_bias, add_implicit_features=self.add_implicit_features,
                k_user=self.k_item, k_item=self.k_user, k_main=self.k_main,
                w_main=self.w_main, w_user=self.w_item, w_item=self.w_user, w_implicit=self.w_implicit,
                l1_lambda=new_l1_lambda,
                scale_lam=self.scale_lam, scale_lam_sideinfo=self.scale_lam_sideinfo,
                maxiter=self.maxiter, niter=self.niter, parallelize=self.parallelize, corr_pairs=self.corr_pairs,
                max_cg_steps=self.max_cg_steps, finalize_chol=self.finalize_chol,
                NA_as_zero=self.NA_as_zero, NA_as_zero_user=self.NA_as_zero_item, NA_as_zero_item=self.NA_as_zero_user,
                nonneg=self.nonneg,
                precompute_for_predictions=precompute, include_all_X=True,
                use_float=self.use_float,
                random_state=self.random_state, verbose=self.verbose, print_every=self.print_every,
                handle_interrupt=self.handle_interrupt, produce_dicts=self.produce_dicts,
                copy_data=self.copy_data, nthreads=self.nthreads)
        elif isinstance(self, CMF_implicit):
            new_model = CMF_implicit(
                k=self.k, lambda_=new_lambda, alpha=self.alpha, use_cg=self.use_cg,
                k_user=self.k_item, k_item=self.k_user, k_main=self.k_main,
                w_main=self.w_main, w_user=self.w_item, w_item=self.w_user,
                l1_lambda=new_l1_lambda,
                scale_lam=self.scale_lam, scale_lam_sideinfo=self.scale_lam_sideinfo,
                niter=self.niter, NA_as_zero_user=self.NA_as_zero_item, NA_as_zero_item=self.NA_as_zero_user,
                nonneg=self.nonneg,
                apply_log_transf=self.apply_log_transf,
                precompute_for_predictions=self.precompute_for_predictions, use_float=self.use_float,
                max_cg_steps=self.max_cg_steps, finalize_chol=self.finalize_chol,
                random_state=self.random_state, init=self.init, verbose=self.verbose,
                produce_dicts=self.produce_dicts, handle_interrupt=self.handle_interrupt,
                copy_data=self.copy_data, nthreads=self.nthreads)
        elif isinstance(self, MostPopular):
            if self.implicit:
                raise ValueError("Cannot swap users and items for MostPopular-implicit.")
            if not self.user_bias:
                raise  ValueError("Swapping users/items not meaningful for MostPopular with 'user_bias=False'")
            new_model = MostPopular(
                implicit=self.implicit, user_bias=True, lambda_=new_lambda, alpha=self.alpha,
                apply_log_transf=self.apply_log_transf,
                use_float=self.use_float, produce_dicts=self.produce_dicts,
                copy_data=self.copy_data, nthreads=self.nthreads)
        elif isinstance(self, ContentBased):
            new_model = ContentBased(
                k=self.k, lambda_=new_lambda, user_bias=self.user_bias, item_bias=self.item_bias,
                add_intercepts=self.add_intercepts, maxiter=self.maxiter, corr_pairs=self.corr_pairs,
                parallelize=self.parallelize, verbose=self.verbose, print_every=self.print_every,
                random_state=self.random_state, use_float=self.use_float,
                produce_dicts=self.produce_dicts, handle_interrupt=self.handle_interrupt, start_with_ALS=self.start_with_ALS,
                copy_data=self.copy_data, nthreads=self.nthreads)
        elif isinstance(self, OMF_explicit):
            new_model = OMF_explicit(
                k=self.k, lambda_=new_lambda, method=self.method, use_cg=self.use_cg,
                user_bias=self.item_bias, item_bias=self.user_bias, k_sec=self.k_sec, k_main=self.k_main,
                add_intercepts=self.add_intercepts, w_user=self.w_item, w_item=self.w_user,
                maxiter=self.maxiter, niter=self.niter, parallelize=self.parallelize, corr_pairs=self.corr_pairs,
                max_cg_steps=self.max_cg_steps, finalize_chol=self.finalize_chol,
                NA_as_zero=self.NA_as_zero, use_float=self.use_float,
                random_state=self.random_state, verbose=self.verbose, print_every=self.print_every,
                produce_dicts=self.produce_dicts, handle_interrupt=self.handle_interrupt,
                copy_data=self.copy_data, nthreads=self.nthreads)
        elif isinstance(self, OMF_implicit):
            new_model = OMF_implicit(
                k=self.k, lambda_=new_lambda, alpha=self.alpha, use_cg=self.use_cg, downweight=self.downweight,
                add_intercepts=self.add_intercepts, niter=self.niter,
                apply_log_transf=self.apply_log_transf,
                use_float=self.use_float,
                max_cg_steps=self.max_cg_steps, finalize_chol=self.finalize_chol,
                random_state=self.random_state, verbose=self.verbose,
                produce_dicts=self.produce_dicts, handle_interrupt=self.handle_interrupt,
                copy_data=self.copy_data, nthreads=self.nthreads)
        else:
            raise ValueError("Unexpected error.")


        new_model.A_ = self.B_
        new_model.B_ = self.A_
        new_model.C_ = self.D_
        new_model.D_ = self.C_
        new_model.Cbin_ = self.Dbin_
        new_model.Dbin_ = self.Cbin_
        new_model.Ai_ = self.Bi_
        new_model.Bi_ = self.Ai_
        new_model.user_bias_ = self.item_bias_
        new_model.item_bias_ = self.user_bias_
        new_model.C_bias_ = self.D_bias_
        new_model.D_bias_ = self.C_bias_
        new_model.glob_mean_ = self.glob_mean_

        new_model._U_cols = self._I_cols
        new_model._I_cols = self._U_cols
        new_model._Ub_cols = self._Ib_cols
        new_model._Ib_cols = self._Ub_cols
        new_model._U_colmeans = self._I_colmeans
        new_model._I_colmeans = self._U_colmeans
        new_model._w_main_multiplier = self._w_main_multiplier

        new_model.is_fitted_ = True
        new_model.nfev_ = self.nfev_
        new_model.nupd_ = self.nupd_
        new_model.user_mapping_ = self.item_mapping_
        new_model.item_mapping_ = self.user_mapping_
        new_model.reindex_ = self.reindex_
        new_model.user_dict_ = self.item_dict_
        new_model.item_dict_ = self.user_dict_

        new_model._A_pred = self._B_pred
        new_model._B_pred = self._A_pred
        new_model._n_orig = self._A_pred.shape[0]

        if precompute:
            if isinstance(self, CMF) or isinstance(self, CMF_implicit):
                self.force_precompute_for_predictions()
            elif isinstance(self, OMF_explicit):
                if new_lambda.shape[0] == 6:
                    lambda_ = new_lambda[2]
                    lam_bias = new_lambda[0]
                else:
                    lambda_ = new_lambda
                    lam_bias = new_lambda
                c_funs = wrapper_float if self.use_float else wrapper_double
                new_model._B_plus_bias, new_model._BtB, new_model._TransBtBinvBt, \
                _1, _2, _3, _4, _5, _6 = \
                    c_funs.precompute_matrices_collective_explicit(
                        new_model.B_,
                        new_model.C_,
                        new_model.Bi_,
                        new_model.item_bias_,
                        new_model._U_colmeans,
                        new_model.user_bias, False,
                        new_model.n_orig,
                        new_model.k_sec + new_model.k + new_model.k_main,
                        0, 0, 0,
                        lambda_, lam_bias,
                        1., 1., 1.,
                        glob_mean = 0.,
                        scale_lam = 0, scale_lam_sideinfo = 0,
                        scale_bias_const = 0, scaling_biasA = 0.,
                        NA_as_zero_X = 0,
                        NA_as_zero_U = 0,
                        nonneg = self.nonneg,
                        include_all_X = True
                    )
            elif isinstance(self, OMF_implicit):
                c_funs = wrapper_float if self.use_float else wrapper_double
                new_model._BtB, _1, _2, _3 = \
                    c_funs.precompute_matrices_collective_implicit(
                        new_model.B_,
                        new_model.C_,
                        new_model._U_colmeans,
                        new_model.k, 0, 0, 0,
                        new_lambda, 1., 1.,
                        1., False, False
                    )

        return new_model

    def drop_nonessential_matrices(self, drop_precomputed=True):
        """
        Drop matrices that are not used for prediction

        Drops all the matrices in the model object which are not
        used for calculating new user factors (either warm or cold), such as the
        user biases or the item factors.

        This is intended at decreasing memory usage in production systems which
        use this software for calculation of user factors or top-N recommendations.

        Can additionally drop some of the precomputed matrices which are only
        taken in special circumstances such as when passing dense data with
        no missing values - however, predictions that would have otherwise used
        these matrices will become slower afterwards.

        After dropping these non-essential matrices, it will not be possible
        anymore to call certain methods such as ``predict`` or ``swap_users_and_items``.
        The methods which are intended to continue working afterwards are:
            
            - ``factors_warm``

            - ``factors_cold``

            - ``factors_multiple``

            - ``topN_warm``

            - ``topN_cold``
            

        Parameters
        ----------
        drop_precomputed : bool
            Whether to drop the less commonly used prediction
            matrices (see documentation above for more details).

        Returns
        -------
        self : obj
            This object with the non-essential matrices dropped.
        """
        assert self.is_fitted_
        if (not isinstance(self, CMF)) and (not isinstance(self, CMF_implicit)):
            raise ValueError("Method is only applicable to 'CMF' and 'CMF_implicit'.")

        self._only_prediction_info = True

        self.user_mapping_ = np.array([], dtype=object)
        self.user_dict_ = dict()
        self.item_dict_ = dict()
        self._I_cols = np.empty(0, dtype=object)
        self._Ib_cols = np.empty(0, dtype=object)

        self.A_ = np.empty((0,0), dtype=self.dtype_)
        self.Ai_ = np.empty((0,0), dtype=self.dtype_)
        self.D_ = np.empty((0,0), dtype=self.dtype_)
        self.Dbin_ = np.empty((0,0), dtype=self.dtype_)
        self._A_pred = np.empty((0,0), dtype=self.dtype_)
        self._I_colmeans = np.empty(0, dtype=self.dtype_)
        self.user_bias_ = np.empty(0, dtype=self.dtype_)
        self.D_bias_ = np.empty(0, dtype=self.dtype_)

        if self._B_plus_bias.shape[0]:
            self.B_ = np.empty((0,0), dtype=self.dtype_)
            self._B_pred = np.empty((0,0), dtype=self.dtype_)

        if drop_precomputed:
            self._TransBtBinvBt = np.empty((0,0), dtype=self.dtype_)
            self._TransCtCinvCt = np.empty((0,0), dtype=self.dtype_)
            self._BeTBeChol = np.empty((0,0), dtype=self.dtype_)
            self._BeTBe = np.empty((0,0), dtype=self.dtype_)

        return self



class CMF(_CMF):
    """
    Collective or multi-view matrix factorization

    Tries to approximate the 'X' interactions matrix  by a formula as follows:
        X ~ A * t(B)
    
    While at the same time also approximating the user/row side information
    matrix 'U' and the item/column side information matrix 'I' as follows:
        U ~ A * t(C), 
        
        I ~ B * t(D)
    
    The matrices ("A", "B", "C", "D") are obtained by minimizing the error
    with respect to the non-missing entries in the input data ("X", "U", "I").
    Might apply sigmoid transformations to binary columns in U and I too.

    This is the most flexible of the models available in this package, and
    can also mimic the implicit-feedback version through the option 'NA_as_zero'
    plus an array of weights.

    Note
    ----
    The default arguments are not geared towards speed.
    For faster fitting, use ``method="als"``, ``use_cg=True``,
    ``finalize_chol=False``, ``use_float=True``,
    ``precompute_for_predictions=False``, ``produce_dicts=False``,
    and pass COO matrices or NumPy arrays instead of DataFrames to ``fit``.

    Note
    ----
    By default, the model optimization objective will not scale any of its
    terms according to number of entries (see parameter ``scale_lam``),
    so hyperparameters such as ``lambda_`` will require more tuning than
    in other software and trying out values over a wider range.

    Parameters
    ----------
    k : int
        Number of latent factors to use (dimensionality of the low-rank
        factorization), which will be shared between the factorization of the
        'X' matrix and the side info matrices. Additional non-shared components
        can also be specified through ``k_user``, ``k_item``, and ``k_main``.
        Typical values are 30 to 100.
    lambda_ : float or array(6,)
        Regularization parameter. Can also use different regularization for each
        matrix, in which case it should be an array with 6 entries, corresponding,
        in this order, to: user_bias, item_bias, A, B, C, D. Note that the default
        value for ``lambda_`` here is much higher than in other software, and that
        the loss/objective function is not divided by the number of entries anywhere,
        so this parameter needs good tuning.
        For example, a good value for the MovieLens10M would be ``lambda_=35.``
        (or ``lambda=0.05`` with ``scale_lam=True``).
        Typical values are 10^-2 to 10^2.
    method : str, one of "lbfgs" or "als"
        Optimization method used to fit the model. If passing ``'lbfgs'``, will
        fit it through a gradient-based approach using an L-BFGS optimizer.
        L-BFGS is typically a much slower and a much less memory efficient method
        compared to ``'als'``, but tends to reach better local optima and allows
        some variations of the problem which ALS doesn't, such as applying sigmoid
        transformations for binary side information.
    use_cg : bool
        In the ALS method, whether to use a conjugate gradient method to solve
        the closed-form least squares problems. This is a faster and more
        memory-efficient alternative than the default Cholesky solver, but less
        exact, less numerically stable, and will require slightly more ALS
        iterations (``niter``) to reach a good optimum.
        In general, better results are achieved with ``use_cg=False``.
        Note that, if using this method, calculations after fitting which involve
        new data such as ``factors_warm``,  might produce slightly different
        results from the factors obtained from calling ``fit`` with the same data,
        due to differences in numerical precision. A workaround for this issue
        (factors on new data that might differ slightly) is to use
        ``finalize_chol=True``.
        Even if passing "True" here, will use the Cholesky method in cases in which
        it is faster (e.g. dense matrices with no missing values),
        and will not use the conjugate gradient method on new data.
        This option is not available when using L1 regularization and/or
        non-negativity constraints.
        Ignored when passing ``method="lbfgs"``.
    user_bias : bool
        Whether to add user/row biases (intercepts) to the model.
        If using it for purposes other than recommender systems, this is is
        usually **not** suggested to include.
    item_bias : bool
        Whether to add item/column biases (intercepts) to the model. Be aware that using
        item biases with low regularization for them will tend to favor items
        with high average ratings regardless of the number of ratings the item
        has received.
    center : bool
        Whether to center the "X" data by subtracting the mean value. For recommender
        systems, it's highly recommended to pass "True" here, the more so if the
        model has user and/or item biases.
    add_implicit_features : bool
        Whether to automatically add so-called implicit features from the data,
        as in reference [5] and similar. If using this for recommender systems
        with small amounts of data, it's recommended to pass 'True' here.
    scale_lam : bool
        Whether to scale (increase) the regularization parameter
        for each row of the model matrices (A, B, C, D) according
        to the number of non-missing entries in the data for that
        particular row, as proposed in reference [7]. For the
        A and B matrices, the regularization will only be scaled
        according to the number of non-missing entries in "X"
        (see also the ``scale_lam_sideinfo`` parameter). Note that,
        when using the options ``NA_as_zero_*``, all entries are
        considered to be non-missing. If passing "True" here, the
        optimal value for ``lambda_`` will be much smaller
        (and likely below 0.1).
        This option tends to give better results, but
        requires more hyperparameter tuning.
        Only supported for ``method="als"``.

        When generating factors based on side information alone,
        if passing ``scale_lam_sideinfo``, will regularize assuming
        there was one observation present. Be aware that using this option
        **without** ``scale_lam_sideinfo=True`` can lead to bad cold-start
        recommendations as it will set a very small regularization for
        users who have no 'X' data.

        Warning: in smaller datasets, using this option can result in top-N
        recommendations having mostly items with very few interactions (see
        parameter ``scale_bias_const``).
    scale_lam_sideinfo : bool
        Whether to scale (increase) the regularization
        parameter for each row of the "A" and "B"
        matrices according to the number of non-missing
        entries in both "X" and the side info matrices
        "U" and "I". If passing "True" here, ``scale_lam``
        will also be assumed to be "True".
    scale_bias_const : bool
        When passing ``scale_lam=True`` and ``user_bias=True`` or ``item_bias=True``,
        whether to apply the same scaling to the regularization **of the biases** to all
        users and items, according to the average number of non-missing entries rather
        than to the number of entries for each specific user/item.

        While this tends to result in worse RMSE, it tends to make the top-N
        recommendations less likely to select items with only a few interactions
        from only a few users.

        Ignored when passing ``scale_lam=False`` or not using user/item biases.
    k_user : int
        Number of factors in the factorizing A and C matrices which will be used
        only for the 'U' and 'U_bin' matrices, while being ignored for the 'X' matrix.
        These will be the first factors of the matrices once the model is fit.
        Will be counted in addition to those already set by ``k``.
    k_item : int
        Number of factors in the factorizing B and D matrices which will be used
        only for the 'I' and 'I_bin' matrices, while being ignored for the 'X' matrix.
        These will be the first factors of the matrices once the model is fit.
        Will be counted in addition to those already set by ``k``.
    k_main : int
        Number of factors in the factorizing A and B matrices which will be used
        only for the 'X' matrix, while being ignored for the 'U', 'U_bin', 'I',
        and 'I_bin' matrices.
        These will be the last factors of the matrices once the model is fit.
        Will be counted in addition to those already set by ``k``.
    w_main : float
        Weight in the optimization objective for the errors in the factorization
        of the 'X' matrix.
    w_user : float
        Weight in the optimization objective for the errors in the factorization
        of the 'U' and 'U_bin' matrices. Ignored when passing neither 'U' nor
        'U_bin' to 'fit'.
    w_item : float
        Weight in the optimization objective for the errors in the factorization
        of the 'I' and 'I_bin' matrices. Ignored when passing neither 'I' nor
        'I_bin' to 'fit'.
    w_implicit : float
        Weight in the optimization objective for the errors in the factorizations
        of the implicit 'X' matrices. Note that, depending on the sparsity of the
        data, the sum of errors from these factorizations might be much larger than
        for the original 'X' and a smaller value will perform better.
        It is recommended to tune this parameter carefully.
        Ignored when passing ``add_implicit_features=False``.
    l1_lambda : float or array(6,)
        Regularization parameter to apply to the L1 norm of the model matrices.
        Can also pass different values for each matrix (see ``lambda_`` for
        details). Note that, when adding L1 regularization, the model will be
        fit through a coordinate descent procedure, which is significantly
        slower than the Cholesky method with L2 regularization.
        Only supported with ``method="als"``.
        Not recommended.
    center_U : bool
        Whether to center the 'U' matrix column-by-column. Be aware that this
        is a simple mean centering without regularization. One might want to
        turn this option off when using ``NA_as_zero_user=True``.
    center_I : bool
        Whether to center the 'I' matrix column-by-column. Be aware that this
        is a simple mean centering without regularization. One might want to
        turn this option off when using ``NA_as_zero_item=True``.
    maxiter : int
        Maximum L-BFGS iterations to perform. The procedure will halt if it
        has not converged after this number of updates. Note that, compared to
        the ohter models, fewer iterations will be required for converge
        here. Using higher regularization values might also decrease the number
        of required iterations. Pass zero for no L-BFGS iterations limit.
        If the procedure is spending hundreds of iterations
        without any significant decrease in the loss function or gradient norm,
        it's highly likely that the regularization is too low.
        Ignored when passing ``method='als'``.
    niter : int
        Number of alternating least-squares iterations to perform. Note that
        one iteration denotes an update round for all the matrices rather than
        an update of a single matrix. In general, the more iterations, the better
        the end result. Ignored when passing ``method='lbfgs'``.
        Typical values are 6 to 30.
    parallelize : str, "separate" or "single"
        How to parallelize gradient calculations when using more than one
        thread with ``method='lbfgs'``. Passing ``'separate'`` will iterate
        over the data twice - first
        by rows and then by columns, letting each thread calculate results
        for each row and column, whereas passing ``'single'`` will iterate over
        the data only once, and then sum the obtained results from each thread.
        Passing ``'separate'`` is much more memory-efficient and less prone to
        irreproducibility of random seeds, but might be slower for typical
        use-cases. Ignored when passing ``nthreads=1``, or ``method='als'``,
        or when compiling without OpenMP support.
    corr_pairs : int
        Number of correction pairs to use for the L-BFGS optimization routine.
        Recommended values are between 3 and 7. Note that higher values
        translate into higher memory requirements. Ignored when passing
        ``method='als'``.
    max_cg_steps : int
        Maximum number of conjugate gradient iterations to perform in an ALS round.
        Ignored when passing ``use_cg=False`` or ``method="lbfgs"``.
    finalize_chol : bool
        When passing ``use_cg=True`` and ``method="als"``, whether to perform the last iteration with
        the Cholesky solver. This will make it slower, but will avoid the issue
        of potential mismatches between the result from ``fit`` and calls to
        ``factors_warm`` or similar with the same data.
    NA_as_zero : bool
        Whether to take missing entries in the 'X' matrix as zeros (only
        when the 'X' matrix is passed as sparse COO matrix or DataFrame)
        instead of ignoring them. Note that this is a different model from the
        implicit-feedback version with weighted entries, and it's a much faster
        model to fit.
        Note that passing "True" will affect the results of the functions named
        "cold" (as it will assume zeros instead of missing).
        It is possible to obtain equivalent results to the implicit-feedback
        model if passing "True" here, and then passing an "X" to fit with
        all values set to one and weights corresponding to the actual values
        of "X" multiplied by alpha, plus 1 (W := 1 + alpha*X to imitate the
        implicit-feedback model).
        If passing this option, be aware that the defaults are also to
        perform mean centering and add user/item biases, which might
        be undesirable to have together with this option.
    NA_as_zero_user : bool
        Whether to take missing entries in the 'U' matrix as zeros (only
        when the 'U' matrix is passed as sparse COO matrix) instead of ignoring them.
        Note that passing "True" will affect the results of the functions named
        "warm" if no data is passed there (as it will assume zeros instead of
        missing).
    NA_as_zero_item : bool
        Whether to take missing entries in the 'I' matrix as zeros (only
        when the 'I' matrix is passed as sparse COO matrix) instead of ignoring them.
    nonneg : bool
        Whether to constrain the 'A' and 'B' matrices to be non-negative.
        In order for this to work correctly, the 'X' input data must also be
        non-negative. This constraint will also be applied to the 'Ai'
        and 'Bi' matrices if passing ``add_implicit_features=True``.

        **Important:** be aware that the default options are to perform mean
        centering and to add user and item biases, which might be undesirable and
        hinder performance when having non-negativity constraints
        (especially mean centering).

        This option is not available when using the L-BFGS method.
        Note that, when determining non-negative factors, it will always
        use a coordinate descent method, regardless of the value passed
        for ``use_cg`` and ``finalize_chol``.
        When used for recommender systems, one usually wants to pass 'False' here.
        For better results, do not use centering alongside this option,
        and use a higher regularization coupled with more iterations.
    nonneg_C: bool
        Whether to constrain the 'C' matrix to be non-negative.
        In order for this to work correctly, the 'U' input data must also be
        non-negative.

        Note: by default, the 'U' data will be centered by columns, which
        doesn't play well with non-negativity constraints. One will likely
        want to pass ``center_U=False`` along with this.
    nonneg_D: bool
        Whether to constrain the 'D' matrix to be non-negative.
        In order for this to work correctly, the 'I' input data must also be
        non-negative.

        Note: by default, the 'I' data will be centered by columns, which
        doesn't play well with non-negativity constraints. One will likely
        want to pass ``center_I=False`` along with this.
    max_cd_steps : int
        Maximum number of coordinate descent updates to perform per iteration.
        Pass zero for no limit.
        The procedure will only use coordinate descent updates when having
        L1 regularization and/or non-negativity constraints.
        This number should usually be larger than ``k``.
    precompute_for_predictions : bool
        Whether to precompute some of the matrices that are used when making
        predictions from the model. If 'False', it will take longer to generate
        predictions or top-N lists, but will use less memory and will be faster
        to fit the model. If passing 'False', can be recomputed later on-demand
        through method 'force_precompute_for_predictions'.
    include_all_X : bool
        When passing an input "X" to ``fit`` which has less columns than rows in
        "I", whether to still make calculations about the items which are in "I"
        but not in "X". This has three effects: (a) the ``topN`` functionality may
        recommend such items, (b) the precomptued matrices will be less usable as
        they will include all such items, (c) it will be possible to pass "X" data
        to the new factors or topN functions that include such columns (rows of "I").
        This option is ignored when using ``NA_as_zero``.
    use_float : bool
        Whether to use C float type for the model parameters (typically this is
        ``np.float32``). If passing ``False``, will use C double (typically this
        is ``np.float64``). Using float types will speed up computations and
        use less memory, at the expense of reduced numerical precision.
    random_state : int, RandomState, or Generator
        Seed used to initialize parameters at random. If passing a NumPy
        RandomState or Generator, will use it to draw a random integer. Note
        however that, if using more than one thread, results might not be
        100% reproducible with ``method='lbfgs'`` due to round-off errors
        in parallelized aggregations.
    verbose : bool
        Whether to print informational messages about the optimization
        routine used to fit the model. Note that, if running this from a
        Jupyter notebook, these messages will be printed in the console,
        not in the notebook itself. Be aware that, if passing 'False' and
        ``method='lbfgs'``, the optimization routine will not respond to
        interrupt signals.
    print_every : int
        Print L-BFGS convergence messages every n-iterations. Ignored
        when passing ``verbose=False`` or ``method='als'``.
    produce_dicts : bool
        Whether to produce Python dicts from the mappings between user/item
        IDs passed to 'fit' and the internal IDs used by the class. Having
        these dicts might speed up some computations such as 'predict',
        but it will add some extra overhead at the time of fitting the model
        and extra memory usage. Ignored when passing the data as matrices
        and arrays instead of data frames.
    handle_interrupt : bool
        When receiving an interrupt signal, whether the model should stop
        early and leave a usable object with the parameters obtained up
        to the point when it was interrupted (when passing 'True'), or
        raise an interrupt exception without producing a fitted model object
        (when passing 'False').
    copy_data : bool
        Whether to make copies of the input data that is passed to this
        object's methods (``fit``, ``predict``, etc.), in order to avoid
        modifying such data in-place. Passing ``False`` will save some
        computation time and memory usage.
    nthreads : int
        Number of parallel threads to use. If passing -1, will take the
        maximum available number of threads in the system.

    Attributes
    ----------
    is_fitted_ : bool
        Whether the model has been fitted to data.
    reindex_ : bool
        Whether the IDs passed to 'fit' were reindexed internally
        (this will only happen when passing data frames to 'fit').
    user_mapping_ : array(m,) or array(0,)
        Correspondence of internal user (row) IDs to IDs in the data
        passed to 'fit'. Will only be non-empty when passing a data frame
        as input to 'X'.
    item_mapping_ : array(n,) or array(0,)
        Correspondence of internal item (column) IDs to IDs in the data
        passed to 'fit'. Will only be non-empty when passing a data frame
        as input to 'X'.
    user_dict_ : dict
        Python dict version of ``user_mapping_``. Only filled-in when
        passing ``produce_dicts=True`` and when passing data frames to 'fit'.
    item_dict_ : dict
        Python dict version of ``item_mapping_``. Only filled-in when
        passing ``produce_dicts=True`` and when passing data frames to 'fit'.
    glob_mean_ : float
        The global mean of the non-missing entries in 'X' passed to 'fit'.
    user_bias_ : array(m,), or array(0,)
        The obtained biases for each user (row in the 'X' matrix).
        If passing ``user_bias=False``, this array
        will be empty.
    item_bias_ : array(n,)
        The obtained biases for each item (column in the 'X' matrix).
        If passing ``item_bias=False``, this array
        will be empty.
    A_ : array(m, k_user+k+k_main)
        The obtained user factors.
    B_ : array(n, k_item+k+k_main)
        The obtained item factors.
    C_ : array(p, k_user+k)
        The obtained user-attributes factors.
    D_ : array(q, k_item+k)
        The obtained item attributes factors.
    Ai_ : array(m, k+k_main) or array(0, 0)
        The obtain implicit user factors.
    Bi_ : array(n, k+k_main) or array(0, 0)
        The obtained implicit item factors.
    nfev_ : int
        Number of function and gradient evaluations performed during the
        L-BFGS optimization procedure.
    nupd_ : int
        Number of L-BFGS updates performed during the optimization procedure.

    References
    ----------
    .. [1] Cortes, David.
           "Cold-start recommendations in Collective Matrix Factorization."
           arXiv preprint arXiv:1809.00366 (2018).
    .. [2] Singh, Ajit P., and Geoffrey J. Gordon.
           "Relational learning via collective matrix factorization."
           Proceedings of the 14th ACM SIGKDD international conference on
           Knowledge discovery and data mining. 2008.
    .. [4] Takacs, Gabor, Istvan Pilaszy, and Domonkos Tikk.
           "Applications of the conjugate gradient method for implicit feedback collaborative filtering."
           Proceedings of the fifth ACM conference on Recommender systems. 2011.
    .. [5] Rendle, Steffen, Li Zhang, and Yehuda Koren.
           "On the difficulty of evaluating baselines: A study on recommender systems."
           arXiv preprint arXiv:1905.01395 (2019).
    .. [6] Franc, Vojtěch, Václav Hlaváč, and Mirko Navara.
           "Sequential coordinate-wise algorithm for the
           non-negative least squares problem."
           International Conference on Computer Analysis of Images and Patterns.
           Springer, Berlin, Heidelberg, 2005.
    .. [7] Zhou, Yunhong, et al.
           "Large-scale parallel collaborative filtering for the netflix prize."
           International conference on algorithmic applications in management.
           Springer, Berlin, Heidelberg, 2008.
    """
    def __init__(self, k=40, lambda_=1e+1, method="als", use_cg=True,
                 user_bias=True, item_bias=True, center=True,
                 add_implicit_features=False,
                 scale_lam=False, scale_lam_sideinfo=False, scale_bias_const=False,
                 k_user=0, k_item=0, k_main=0,
                 w_main=1., w_user=1., w_item=1., w_implicit=0.5,
                 l1_lambda=0., center_U=True, center_I=True,
                 maxiter=800, niter=10, parallelize="separate", corr_pairs=4,
                 max_cg_steps=3, finalize_chol=True,
                 NA_as_zero=False, NA_as_zero_user=False, NA_as_zero_item=False,
                 nonneg=False, nonneg_C=False, nonneg_D=False, max_cd_steps=100,
                 precompute_for_predictions=True, include_all_X=True,
                 use_float=False,
                 random_state=1, verbose=True, print_every=10,
                 handle_interrupt=True, produce_dicts=False,
                 copy_data=True, nthreads=-1):
        self._take_params(implicit=False, alpha=0., downweight=False,
                          k=k, lambda_=lambda_, method=method,
                          add_implicit_features=add_implicit_features,
                          scale_lam=scale_lam, scale_lam_sideinfo=scale_lam_sideinfo,
                          scale_bias_const=scale_bias_const,
                          nonneg=nonneg, nonneg_C=nonneg_C, nonneg_D=nonneg_D,
                          use_cg=use_cg, max_cg_steps=max_cg_steps,
                          max_cd_steps=max_cd_steps,
                          finalize_chol=finalize_chol,
                          user_bias=user_bias, item_bias=item_bias,
                          center=center,
                          k_user=k_user, k_item=k_item, k_main=k_main,
                          w_main=w_main, w_user=w_user, w_item=w_item,
                          w_implicit=w_implicit,
                          l1_lambda=l1_lambda, center_U=center_U, center_I=center_I,
                          maxiter=maxiter, niter=niter, parallelize=parallelize,
                          corr_pairs=corr_pairs,
                          NA_as_zero=NA_as_zero, NA_as_zero_user=NA_as_zero_user,
                          NA_as_zero_item=NA_as_zero_item,
                          precompute_for_predictions=precompute_for_predictions,
                          use_float=use_float,
                          random_state=random_state, init="normal",
                          verbose=verbose, print_every=print_every,
                          handle_interrupt=handle_interrupt,
                          produce_dicts=produce_dicts, copy_data=copy_data,
                          nthreads=nthreads)
        self.include_all_X = bool(include_all_X)
        if (self.NA_as_zero) and (not self.include_all_X):
            warnings.warn("Warning: 'include_all_X' is forced to 'True' when using 'NA_as_zero'.")
            self.include_all_X = True

    def __str__(self):
        msg  = "Collective matrix factorization model\n"
        msg += "(explicit-feedback variant)\n\n"
        if not self.is_fitted_:
            msg += "Model has not been fitted to data.\n"
        return msg

    def get_params(self, deep=None):
        return {
            "k" : self.k, "lambda_" : self.lambda_, "method" : self.method,
            "user_bias" : self.user_bias, "item_bias" : self.item_bias,
            "k_user" : self.k_user, "k_item" : self.k_item, "k_main" : self.k_main,
            "w_main" : self.w_main, "w_user" : self.w_user, "w_item" : self.w_item,
            "maxiter" : self.maxiter, "niter" : self.niter,
            "parallelize" : self.parallelize, "corr_pairs" : self.corr_pairs,
            "NA_as_zero" : self.NA_as_zero,
            "NA_as_zero_user" : self.NA_as_zero_user,
            "NA_as_zero_item" : self.NA_as_zero_item,
            "random_state" : self.random_state,
            "use_cg" : self.use_cg, "use_float" : self.use_float,
            "nthreads" : self.nthreads
        }

    def fit(self, X, U=None, I=None, U_bin=None, I_bin=None, W=None):
        """
        Fit model to explicit-feedback data and user/item attributes

        Note
        ----
        It's possible to pass partially disjoints sets of users/items between
        the different matrices (e.g. it's possible for both the 'X' and 'U'
        matrices to have rows that the other doesn't have).
        The procedure supports missing values for all inputs (except for "W").
        If any of the inputs has less rows/columns than the other(s) (e.g.
        "U" has more rows than "X", or "I" has more rows than there are columns
        in "X"), will assume that the rest of the rows/columns have only
        missing values.
        Note however that when having partially disjoint inputs, the order of
        the rows/columns matters for speed, as it might run faster when the "U"/"I"
        inputs that do not have matching rows/columns in "X" have those unmatched
        rows/columns at the end (last rows/columns) and the "X" input is shorter.
        See also the parameter ``include_all_X`` for info about predicting with
        mismatched "X".

        Note
        ----
        When passing NumPy arrays, missing (unobserved) entries should 
        have value ``np.nan``. When passing sparse inputs, the zero-valued entries
        will be considered as missing (unless using "NA_as_zero"), and it should
        not contain "NaN" values among the non-zero entries.

        Note
        ----
        In order to avoid potential decimal differences in the factors obtained
        when fitting the model and when calling the prediction functions on
        new data, when the data is sparse, it's necessary to sort it beforehand
        by columns and also pass the data data with indices sorted (by column)
        to the prediction functions.

        Parameters
        ----------
        X : DataFrame(nnz, 3), DataFrame(nnz, 4), array(m, n), or sparse COO(m, n)
            Matrix to factorize (e.g. ratings). Can be passed as a SciPy
            sparse COO matrix (recommended), as a dense NumPy array, or
            as a Pandas DataFrame, in which case it should contain the
            following columns: 'UserId', 'ItemId', and 'Rating'.
            Might additionally have a column 'Weight'. If passing a DataFrame,
            the IDs will be internally remapped.
            If passing sparse 'U' or sparse 'I', 'X' cannot be passed as
            a DataFrame.
        U : array(m, p), COO(m, p), DataFrame(m, p+1), or None
            User attributes information. If 'X' is a DataFrame, should also
            be a DataFrame, containing column 'UserId'. If 'U' is sparse,
            'X' should be passed as a sparse COO matrix or as a dense NumPy array.
        U_bin : array(m, p_bin), DataFrame(m, p_bin+1), or None
            User binary attributes information (all values should be zero, one,
            or missing). If 'X' is a DataFrame, should also
            be a DataFrame, containing column 'UserId'. Cannot be passed
            as a sparse matrix.
            Note that 'U' and 'U_bin' are not mutually exclusive.
            Only supported with ``method='lbfgs'``.
        I : array(n, q), COO(n, q), DataFrame(n, q+1), or None
            Item attributes information. If 'X' is a DataFrame, should also
            be a DataFrame, containing column 'ItemId'. If 'I' is sparse,
            'X' should be passed as a sparse COO matrix or as a dense NumPy array.
        I_bin : array(n, q_bin), DataFrame(n, q_bin+1), or None
            Item binary attributes information (all values should be zero, one,
            or missing). If 'X' is a DataFrame, should also
            be a DataFrame, containing column 'ItemId'. Cannot be passed
            as a sparse matrix.
            Note that 'I' and 'I_bin' are not mutually exclusive.
            Only supported with ``method='lbfgs'``.
        W : None, array(nnz,), or array(m, n)
            Observation weights. Must have the same shape as 'X' - that is,
            if 'X' is a sparse COO matrix, must be a 1-d array with the same
            number of non-zero entries as 'X.data', if 'X' is a 2-d array,
            'W' must also be a 2-d array.
            Cannot have missing values.

        Returns
        -------
        self
        
        """
        return self._fit_common(X, U=U, I=I, U_bin=U_bin, I_bin=I_bin, W=W)

    def _fit(self, Xrow, Xcol, Xval, W_sp, Xarr, W_dense,
             Uarr, Urow, Ucol, Uval, Ub_arr,
             Iarr, Irow, Icol, Ival, Ib_arr,
             m, n, m_u, n_i, p, q,
             m_ub, n_ib, pbin, qbin):
        c_funs = wrapper_float if self.use_float else wrapper_double
        if self.method == "lbfgs":
            self.glob_mean_,  self._U_colmeans, self._I_colmeans, values, self.nupd_, self.nfev_, self._B_plus_bias = \
                c_funs.call_fit_collective_explicit_lbfgs(
                    Xrow,
                    Xcol,
                    Xval,
                    W_sp,
                    Xarr,
                    W_dense,
                    Uarr,
                    Urow,
                    Ucol,
                    Uval,
                    Ub_arr,
                    Iarr,
                    Irow,
                    Icol,
                    Ival,
                    Ib_arr,
                    m, n, m_u, n_i, p, q,
                    self.k, self.k_user, self.k_item, self.k_main,
                    self.w_main, self.w_user, self.w_item,
                    self.user_bias, self.item_bias, self.center,
                    self.lambda_ if isinstance(self.lambda_, float) else 0.,
                    self.lambda_ if isinstance(self.lambda_, np.ndarray) else np.empty(0, dtype=self.dtype_),
                    self.verbose, self.print_every,
                    self.corr_pairs, self.maxiter,
                    self.nthreads, self.parallelize != "separate",
                    self.random_state,
                    self.handle_interrupt
                )
            self.user_bias_, self.item_bias_, self.A_, self.B_, self.C_, self.Cbin_, self.D_, self.Dbin_ = \
                c_funs.unpack_values_lbfgs_collective(
                    values,
                    self.user_bias, self.item_bias,
                    self.k, self.k_user, self.k_item, self.k_main,
                    m, n, p, q,
                    pbin, qbin,
                    m_u, n_i, m_ub, n_ib
                )
            self._n_orig = self.B_.shape[0] if self.include_all_X else n
            self.is_fitted_ = True
            if self.precompute_for_predictions:
                self.force_precompute_for_predictions()
        else:
            self.user_bias_, self.item_bias_, \
            self.A_, self.B_, self.C_, self.D_, self.Ai_, self.Bi_, \
            self.glob_mean_,  self._U_colmeans, self._I_colmeans, \
            self._B_plus_bias, self._BtB, self._TransBtBinvBt, self._BtXbias, \
            self._BeTBeChol, self._BiTBi, self._TransCtCinvCt, self._CtC, \
            self._CtUbias, self.scaling_biasA_, self.scaling_biasB_ = \
                c_funs.call_fit_collective_explicit_als(
                    Xrow,
                    Xcol,
                    Xval,
                    W_sp,
                    Xarr,
                    W_dense,
                    Uarr,
                    Urow,
                    Ucol,
                    Uval,
                    Iarr,
                    Irow,
                    Icol,
                    Ival,
                    self.NA_as_zero, self.NA_as_zero_user, self.NA_as_zero_item,
                    m, n, m_u, n_i, p, q,
                    self.k, self.k_user, self.k_item, self.k_main,
                    self.w_main, self.w_user, self.w_item, self.w_implicit,
                    self.user_bias, self.item_bias, self.center,
                    self.lambda_ if isinstance(self.lambda_, float) else 0.,
                    self.lambda_ if isinstance(self.lambda_, np.ndarray) else np.empty(0, dtype=self.dtype_),
                    self.l1_lambda if isinstance(self.l1_lambda, float) else 0.,
                    self.l1_lambda if isinstance(self.l1_lambda, np.ndarray) else np.empty(0, dtype=self.dtype_),
                    self.center_U, self.center_I,
                    self.scale_lam, self.scale_lam_sideinfo, self.scale_bias_const,
                    self.verbose, self.nthreads,
                    self.use_cg, self.max_cg_steps,
                    self.finalize_chol,
                    self.nonneg, self.nonneg_C, self.nonneg_D, self.max_cd_steps,
                    self.random_state, self.niter,
                    self.handle_interrupt,
                    precompute_for_predictions=self.precompute_for_predictions,
                    add_implicit_features=self.add_implicit_features,
                    include_all_X=self.include_all_X
                )
            self._n_orig = self.B_.shape[0] if (self.include_all_X or self.NA_as_zero) else n

        self._A_pred = self.A_
        self._B_pred = self.B_
        self.is_fitted_ = True
        return self

    def predict_cold(self, items, U=None, U_bin=None, U_col=None, U_val=None):
        """
        Predict rating given by a new user to existing items, given U

        Note
        ----
        If using ``NA_as_zero``, this function will assume that all
        the 'X' values are zeros rather than being missing.

        Parameters
        ----------
        items : array-like(n,)
            Items whose ratings are to be predicted. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'.
        U : array(p,), or None
            User attributes in the new data (1-row only).
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_bin : array(p_bin,)
            User binary attributes in the new data (1-row only).
            Missing entries should have value ``np.nan``.
            Only supported with ``method='lbfgs'``.
        U_col : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_val : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.

        Returns
        -------
        scores : array(n,)
            Predicted ratings for the requested items, for this user.
        """
        a_vec = self.factors_cold(U, U_bin, U_col, U_val)
        return self._predict(user=None, a_vec=a_vec, a_bias=0., item=items)

    def predict_cold_multiple(self, item, U=None, U_bin=None):
        """
        Predict rating given by new users to existing items, given U

        Note
        ----
        If using ``NA_as_zero``, this function will assume that all
        the 'X' values are zeros rather than being missing.

        Parameters
        ----------
        item : array-like(m,)
            Items for which ratings/values are to be predicted. If 'X' passed to
            fit was a  DataFrame, must match with the entries in its 'ItemId'
            column, otherwise should match with the columns of 'X'.
        U : array(m, p), CSR matrix(m, q), COO matrix(m, q), or None
            Attributes for the users for which to predict ratings/values.
            Data frames with 'UserId' column are not supported.
            Must have one row per entry
            in ``item``.
        U_bin : array(m, p_bin), or None
            Binary attributes for the users to predict ratings/values.
            Data frames with 'UserId'
            column are not supported. Must have one row per entry
            in ``user``.
            Only supported with ``method='lbfgs'``.

        Returns
        -------
        scores : array(m,)
            Predicted ratings for the requested user-item combinations.
        """
        A = self._factors_cold_multiple(U=U, U_bin=U_bin, is_I=False)
        return self._predict_user_multiple(A, item)

    def topN_cold(self, n=10, U=None, U_bin=None, U_col=None, U_val=None,
                  include=None, exclude=None, output_score=False):
        """
        Compute top-N highest-predicted items for a new user, given 'U'

        Note
        ----
        If using ``NA_as_zero``, this function will assume that all
        the 'X' values are zeros rather than being missing.

        Parameters
        ----------
        n : int
            Number of top-N highest-predicted results to output.
        U : array(p,), or None
            User attributes in the new data (1-row only).
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_bin : array(p_bin,)
            User binary attributes in the new data (1-row only).
            Missing entries should have value ``np.nan``.
            Only supported with ``method='lbfgs'``.
        U_col : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_val : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.
        include : array-like
            List of items which will be ranked. If passing this, will only
            make a ranking among these items. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include or 'exclude'.
        exclude : array-like
            List of items to exclude from the ranking. If passing this, will
            rank all the items except for these. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include or 'exclude'.
        output_score : bool
            Whether to output the scores in addition to the IDs. If passing
            'False', will return a single array with the item IDs, otherwise
            will return a tuple with the item IDs and the scores.

        Returns
        -------
        items : array(n,)
            The top-N highest predicted items for this user. If the 'X' data passed to
            fit was a DataFrame, will contain the item IDs from its column
            'ItemId', otherwise will be integers matching to the columns of 'X'.
        scores : array(n,)
            The predicted scores for the top-N items. Will only be returned
            when passing ``output_score=True``, in which case the result will
            be a tuple with these two entries.
        """
        a_vec = self.factors_cold(U, U_bin, U_col, U_val)
        return self._topN(user=None, a_vec=a_vec, n=n,
                          include=include, exclude=exclude,
                          output_score=output_score)

    def factors_cold(self, U=None, U_bin=None, U_col=None, U_val=None):
        """
        Determine user-factors from new data, given U

        Note
        ----
        If using ``NA_as_zero``, this function will assume that all
        the 'X' values are zeros rather than being missing.

        Parameters
        ----------
        U : array(p,), or None
            User attributes in the new data (1-row only).
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_bin : array(p_bin,)
            User binary attributes in the new data (1-row only).
            Missing entries should have value ``np.nan``.
            Only supported with ``method='lbfgs'``.
        U_col : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_val : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.

        Returns
        -------
        factors : array(k_user+k+k_main,)
            The user-factors as determined by the model.
        """
        return self._factors_cold(U=U, U_bin=U_bin, U_col=U_col, U_val=U_val)

    def item_factors_cold(self, I=None, I_bin=None, I_col=None, I_val=None):
        """
        Determine item-factors from new data, given I

        Note
        ----
        Calculating item factors might be a lot slower than user factors,
        as the model does not keep precomputed matrices that might speed
        up these factor calculations. If this function is goint to be used
        frequently, it's advised to build the model swapping the users
        and items instead.

        Parameters
        ----------
        I : array(q,), or None
            Attributes for the new item, in dense format.
            Should only pass one of 'I' or 'I_col'+'I_val'.
        I_bin : array(q_bin,), or None
            Binary attributes for the new item, in dense format.
            Only supported with ``method='lbfgs'``.
        I_col : None or array(nnz)
            Attributes for the new item, in sparse format.
            'I_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'I' or 'I_col'+'I_val'.
        I_val : None or array(nnz)
            Attributes for the new item, in sparse format.
            'I_val' should contain the values in the columns
            given by 'I_col'.
            Should only pass one of 'I' or 'I_col'+'I_val'.

        Returns
        -------
        factors : array(k_item+k+k_main,)
            The item-factors as determined by the model.
        """
        return self._item_factors_cold(I=I, I_bin=I_bin, I_col=I_col, I_val=I_val)

    def predict_new(self, user, I=None, I_bin=None):
        """
        Predict rating given by existing users to new items, given I

        Note
        ----
        Calculating item factors might be a lot slower than user factors,
        as the model does not keep precomputed matrices that might speed
        up these factor calculations. If this function is goint to be used
        frequently, it's advised to build the model swapping the users
        and items instead.

        Parameters
        ----------
        user : array-like(n,)
            Users for whom ratings/values are to be predicted. If 'X' passed to
            fit was a  DataFrame, must match with the entries in its 'UserId'
            column, otherwise should match with the rows of 'X'.
        I : array(n, q), CSR matrix(n, q), COO matrix(n, q), or None
            Attributes for the items for which to predict ratings/values. Data frames with 'ItemId' column are not supported. Must have one row per entry
            in ``user``. Might contain missing values.
        I_bin : array(n, q_bin), or None
            Binary attributes for the items to predict ratings/values.
            Data frames with 'ItemId'
            column are not supported. Must have one row per entry
            in ``user``. Might contain missing values.
            Only supported with ``method='lbfgs'``.

        Returns
        -------
        scores : array(n,)
            Predicted ratings for the requested user-item combinations.
        """
        if self._only_prediction_info:
            raise ValueError("Cannot use this function after dropping non-essential matrices.")
        B = self._factors_cold_multiple(U=I, U_bin=I_bin, is_I=True)
        return self._predict_new(user, B)

    def topN_new(self, user, I=None, I_bin=None, n=10, output_score=False):
        """
        Rank top-N highest-predicted items for an existing user, given 'I'

        Note
        ----
        If the model was fit to both 'I' and 'I_bin', can pass a partially-
        disjoint set to both - that is, both can have rows that the other doesn't.
        In such case, the rows that they have in common should come first, and
        then one of them appended missing values so that one of the matrices
        ends up containing all the rows of the other.

        Parameters
        ----------
        user : int or obj
            User for which to rank the items. If 'X' passed to 'fit' was a
            data frame, must match with entries in its 'UserId' column,
            otherwise should match with the rows on 'X'.
        I : array(m, q), CSR matrix(m, q), COO matrix(m, q), or None
            Attributes for the items to rank. Data frames with 'ItemId'
            column are not supported.
        I_bin : array(m, q_bin), or None
            Binary attributes for the items to rank. Data frames with 'ItemId'
            column are not supported.
            Only supported with ``method='lbfgs'``.
        n : int
            Number of top-N highest-predicted results to output. Must be
            less or equal than the number of rows in 'I'/'I_bin'.
        output_score : bool
            Whether to output the scores in addition to the IDs. If passing
            'False', will return a single array with the item IDs, otherwise
            will return a tuple with the item IDs and the scores.

        Returns
        -------
        items : array(n,)
            The top-N highest predicted items for this user, as integers
            matching to the rows of 'I'/'I_bin'.
        scores : array(n,)
            The predicted scores for the top-N items. Will only be returned
            when passing ``output_score=True``, in which case the result will
            be a tuple with these two entries.
        """
        if self._only_prediction_info:
            raise ValueError("Cannot use this function after dropping non-essential matrices.")
        B = self._factors_cold_multiple(U=I, U_bin=I_bin, is_I=True)
        return self._topN(user=user, B=B, n=n, output_score=output_score)

    def factors_warm(self, X=None, X_col=None, X_val=None, W=None,
                     U=None, U_bin=None, U_col=None, U_val=None,
                     return_bias=False):
        """
        Determine user latent factors based on new ratings data

        Parameters
        ----------
        X : array(n,) or None
            Observed 'X' data for the new user, in dense format.
            Non-observed entries should have value ``np.nan``.
            Should only pass one of 'X' or 'X_col'+'X_val'.
        X_col : array(nnz,) or None
            Observed 'X' data for the new user, in sparse format.
            'X_col' should contain the column indices (items) of the observed 
            entries. If 'X' passed to 'fit' was a data frame, should have
            entries from 'ItemId' column, otherwise should have column numbers
            (starting at zero).
            Should only pass one of 'X' or 'X_col'+'X_val'.
        X_val : array(nnz,) or None
            Observed 'X' data for the new user, in sparse format.
            'X_val' should contain the values in the columns/items given by
            'X_col'.
            Should only pass one of 'X' or 'X_col'+'X_val'.
        W : array(nnz,), array(n,), or None
            Weights for the observed entries in 'X'. If passed, should
            have the same shape as 'X' - that is, if 'X' is passed as
            a dense array, should have 'n' entries, otherwise should
            have 'nnz' entries.
        U : array(p,), or None
            User attributes in the new data (1-row only).
            Should only pass one of 'U' or 'U_col'+'U_val'.
            User side information is not strictly required, and can
            skip both.
        U_bin : array(p_bin,)
            User binary attributes in the new data (1-row only).
            Missing entries should have value ``np.nan``.
            Only supported with ``method='lbfgs'``.
            User side info is not strictly required and can be skipped.
        U_col : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
            User side information is not strictly required, and can
            skip both.
        U_val : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.
            User side information is not strictly required, and can
            skip both.
        return_bias : bool
            Whether to return also the user bias determined by the model
            given the data in 'X'. If passing 'False', will return an array
            with the factors. If passing 'True', will return a tuple in which
            the first entry will be an array with the factors, and the second
            entry will be the estimated bias.
        return_raw_A : bool
            Whether to return the raw A factors (the free offset), or the
            factors used in the factorization, to which the attributes
            component has been added.

        Returns
        -------
        factors : array(k_user+k+k_main,) or array(k+k_main,)
            User factors as determined from the data in 'X'.
        bias : float
            User bias as determined from the data in 'X'. Only returned if
            passing ``return_bias=True``.
        """
        return self._factors_warm_common(X=X, X_col=X_col, X_val=X_val, W=W,
                                         U=U, U_bin=U_bin, U_col=U_col, U_val=U_val,
                                         return_bias=return_bias)

    def _factors_warm(self, X, W_dense, X_val, X_col, W_sp,
                      U, U_val, U_col, U_bin, return_bias):
        if isinstance(self.lambda_, np.ndarray):
            lambda_ = self.lambda_[2]
            lambda_bias = self.lambda_[0]
        else:
            lambda_ = self.lambda_
            lambda_bias = self.lambda_

        if isinstance(self.l1_lambda, np.ndarray):
            l1_lambda = self.l1_lambda[2]
            l1_lambda_bias = self.l1_lambda[0]
        else:
            l1_lambda = self.l1_lambda
            l1_lambda_bias = self.l1_lambda

        c_funs = wrapper_float if self.use_float else wrapper_double
        a_bias, a_vec = c_funs.call_factors_collective_explicit_single(
            X,
            W_dense,
            X_val,
            X_col,
            W_sp,
            U,
            U_val,
            U_col,
            U_bin,
            self._U_colmeans,
            self.item_bias_,
            self.B_,
            self._B_plus_bias,
            self.C_,
            self.Cbin_,
            self.Bi_,
            self._BtB,
            self._TransBtBinvBt,
            self._BtXbias,
            self._BeTBeChol,
            self._BiTBi,
            self._CtC,
            self._TransCtCinvCt,
            self._CtUbias,
            self.glob_mean_,
            self._n_orig,
            self.k, self.k_user, self.k_item, self.k_main,
            lambda_, lambda_bias,
            l1_lambda, l1_lambda_bias,
            self.scale_lam, self.scale_lam_sideinfo,
            self.scale_bias_const, self.scaling_biasA_,
            self.w_user, self.w_main, self.w_implicit,
            self.user_bias,
            self.NA_as_zero_user, self.NA_as_zero,
            self.nonneg,
            self.add_implicit_features,
            self.include_all_X
        )

        if return_bias:
            return a_vec, a_bias
        else:
            return a_vec

    def factors_multiple(self, X=None, U=None, U_bin=None, W=None,
                         return_bias=False):
        """
        Determine user latent factors based on new data (warm and cold)

        Determines latent factors for multiple rows/users at once given new
        data for them.

        Note
        ----
        See the documentation of "fit" for details about handling of missing values.

        Note
        ----
        If fitting the model to DataFrame inputs (instead of NumPy arrays and/or
        SciPy sparse matrices), the IDs are reindexed internally,
        and the inputs provided here should match with the numeration that was
        produced by the model. The mappings in such case are available under
        attributes ``self.user_mapping_`` and ``self.item_mapping_``.
        
        Parameters
        ----------
        X : array(m_x, n), CSR matrix(m_x, n), COO matrix(m_x, n), or None
            New 'X' data.
        U : array(m_u, p), CSR matrix(m_u, p), COO matrix(m_u, p), or None
            User attributes information for rows in 'X'.
        U_bin : array(m_ub, p_bin) or None
            User binary attributes for each row in 'X'.
            Only supported with ``method='lbfgs'``.
        W : array(m_x, n), array(nnz,), or None
            Observation weights. Must have the same shape as 'X' - that is,
            if 'X' is a sparse COO matrix, must be a 1-d array with the same
            number of non-zero entries as 'X.data', if 'X' is a 2-d array,
            'W' must also be a 2-d array.
        return_bias : bool
            Whether to return also the user bias determined by the model
            given the data in 'X'. If passing 'False', will return an array
            with the factors. If passing 'True', will return a tuple in which
            the first entry will be an array with the factors, and the second
            entry will be the estimated bias.

        Returns
        -------
        A : array(max(m_x,m_u,m_ub), k_user+k+k_main)
            The new factors determined for all the rows given the new data.
        bias : array(max(m_x,m_u,m_ub)) or None
            The user bias given the new 'X' data. Only returned if passing
            ``return_bias=True``.
        """
        if (X is None) and (U is None) and (U_bin is None):
            raise ValueError("Must pass at least one of 'X', 'U', 'U_bin'.")
        if (W is not None) and (X is None):
            raise ValueError("Cannot pass 'W' without 'X'.")
        
        A, A_bias = self._factors_multiple_common(X, U, U_bin, W)
        if return_bias:
            return A, A_bias
        else:
            return A


    def predict_warm(self, items, X=None, X_col=None, X_val=None, W=None,
                     U=None, U_bin=None, U_col=None, U_val=None):
        """
        Predict ratings for existing items, for a new user, given 'X'

        Parameters
        ----------
        items : array-like(n,)
            Items whose ratings are to be predicted. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'.
        X : array(n,) or None
            Observed 'X' data for the new user, in dense format.
            Non-observed entries should have value ``np.nan``.
            Should only pass one of 'X' or 'X_col'+'X_val'.
        X_col : array(nnz,) or None
            Observed 'X' data for the new user, in sparse format.
            'X_col' should contain the column indices (items) of the observed 
            entries. If 'X' passed to 'fit' was a data frame, should have
            entries from 'ItemId' column, otherwise should have column numbers
            (starting at zero).
            Should only pass one of 'X' or 'X_col'+'X_val'.
        X_val : array(nnz,) or None
            Observed 'X' data for the new user, in sparse format.
            'X_val' should contain the values in the columns/items given by
            'X_col'.
            Should only pass one of 'X' or 'X_col'+'X_val'.
        W : array(nnz,), array(n,), or None
            Weights for the observed entries in 'X'. If passed, should
            have the same shape as 'X' - that is, if 'X' is passed as
            a dense array, should have 'n' entries, otherwise should
            have 'nnz' entries.
        U : array(p,), or None
            User attributes in the new data (1-row only).
            Should only pass one of 'U' or 'U_col'+'U_val'.
            User side information is not strictly required, and can
            skip both.
        U_bin : array(p_bin,)
            User binary attributes in the new data (1-row only).
            Missing entries should have value ``np.nan``.
            Only supported with ``method='lbfgs'``.
            User side info is not strictly required and can be skipped.
        U_col : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
            User side information is not strictly required, and can
            skip both.
        U_val : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.
            User side information is not strictly required, and can
            skip both.
        
        Returns
        -------
        scores : array(n,)
            Predicted values for the requested items for a user defined by
            the given values of 'X' in 'X_col' and 'X_val'.
        """
        a_vec, a_bias = self.factors_warm(X=X, X_col=X_col, X_val=X_val, W=W,
                                          U=U, U_bin=U_bin, U_col=U_col, U_val=U_val,
                                          return_bias=True)
        return self._predict(user=None, a_vec=a_vec, a_bias=a_bias, item=items)

    def predict_warm_multiple(self, X, item, U=None, U_bin=None, W=None):
        """
        Predict ratings for existing items, for new users, given 'X'

        Note
        ----
        See the documentation of "fit" for details about handling of missing values.

        Parameters
        ----------
        X : array(m, n), CSR matrix(m, n) , or COO matrix(m, n)
            New 'X' data with potentially missing entries.
            Must have one row per entry of ``item``.
        item : array-like(m,)
            Items for whom ratings/values are to be predicted. If 'X' passed to
            fit was a  DataFrame, must match with the entries in its 'ItemId'
            column, otherwise should match with the columns of 'X'.
            Each entry in ``item`` will be matched with the corresponding row
            of ``X``.
        U : array(m, p), CSR matrix(m, p), COO matrix(m, p), or None
            User attributes information for each row in 'X'.
        U_bin : array(m, p_bin)
            User binary attributes for each row in 'X'.
            Only supported with ``method='lbfgs'``.
        W : array(m, n), array(nnz,), or None
            Observation weights. Must have the same shape as 'X' - that is,
            if 'X' is a sparse COO matrix, must be a 1-d array with the same
            number of non-zero entries as 'X.data', if 'X' is a 2-d array,
            'W' must also be a 2-d array.

        Returns
        -------
        scores : array(m,)
            Predicted ratings for the requested user-item combinations.
        """
        c_funs = wrapper_float if self.use_float else wrapper_double

        Xrow, Xcol, Xval, W_sp, Xarr, \
        Xcsr_p, Xcsr_i, Xcsr, \
        W_dense, Xorig, mask_take, \
        Uarr, Urow, Ucol, Uval, Ub_arr, \
        Ucsr_p, Ucsr_i, Ucsr, \
        n, m_u, m_x, p, pbin, \
        lambda_, lambda_bias, \
        l1_lambda, l1_lambda_bias = \
            self._process_transform_inputs(X=X, U=U, U_bin=U_bin, W=W,
                                           replace_existing=True)
        A, A_bias = c_funs.call_factors_collective_explicit_multiple(
                Xrow,
                Xcol,
                Xval,
                Xcsr_p, Xcsr_i, Xcsr,
                W_sp,
                Xarr,
                W_dense,
                Uarr,
                Urow,
                Ucol,
                Uval,
                Ucsr_p, Ucsr_i, Ucsr,
                Ub_arr,
                self._U_colmeans,
                self.item_bias_,
                self._B_pred,
                self._B_plus_bias,
                self.Bi_,
                self.C_,
                self.Cbin_,
                self._BtB,
                self._TransBtBinvBt,
                self._BtXbias,
                self._BeTBeChol,
                self._BiTBi,
                self._TransCtCinvCt,
                self._CtC,
                self._CtUbias,
                m_u, m_x,
                self.glob_mean_,
                self._n_orig,
                self._k_pred, self.k_user, self.k_item, self._k_main_col,
                lambda_, lambda_bias,
                l1_lambda, l1_lambda_bias,
                self.scale_lam, self.scale_lam_sideinfo,
                self.scale_bias_const, self.scaling_biasA_,
                self.w_user, self.w_main, self.w_implicit,
                self.user_bias,
                self.NA_as_zero_user, self.NA_as_zero,
                self.nonneg,
                self.add_implicit_features,
                self.include_all_X,
                self.nthreads
            )
        return self._predict_user_multiple(A, item, bias=A_bias)

    def topN_warm(self, n=10, X=None, X_col=None, X_val=None, W=None,
                  U=None, U_bin=None, U_col=None, U_val=None,
                  include=None, exclude=None, output_score=False):
        """
        Compute top-N highest-predicted items for a new user, given 'X'

        Parameters
        ----------
        n : int
            Number of top-N highest-predicted results to output.
        X : array(n,) or None
            Observed 'X' data for the new user, in dense format.
            Non-observed entries should have value ``np.nan``.
            Should only pass one of 'X' or 'X_col'+'X_val'.
        X_col : array(nnz,) or None
            Observed 'X' data for the new user, in sparse format.
            'X_col' should contain the column indices (items) of the observed 
            entries. If 'X' passed to 'fit' was a data frame, should have
            entries from 'ItemId' column, otherwise should have column numbers
            (starting at zero).
            Should only pass one of 'X' or 'X_col'+'X_val'.
        X_val : array(nnz,) or None
            Observed 'X' data for the new user, in sparse format.
            'X_val' should contain the values in the columns/items given by
            'X_col'.
            Should only pass one of 'X' or 'X_col'+'X_val'.
        W : array(nnz,), array(n,), or None
            Weights for the observed entries in 'X'. If passed, should
            have the same shape as 'X' - that is, if 'X' is passed as
            a dense array, should have 'n' entries, otherwise should
            have 'nnz' entries.
        U : array(p,), or None
            User attributes in the new data (1-row only).
            Should only pass one of 'U' or 'U_col'+'U_val'.
            User side information is not strictly required, and can
            skip both.
        U_bin : array(p_bin,)
            User binary attributes in the new data (1-row only).
            Missing entries should have value ``np.nan``.
            Only supported with ``method='lbfgs'``.
            User side info is not strictly required and can be skipped.
        U_col : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
            User side information is not strictly required, and can
            skip both.
        U_val : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.
            User side information is not strictly required, and can
            skip both.
        include : array-like
            List of items which will be ranked. If passing this, will only
            make a ranking among these items. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include or 'exclude'.
        exclude : array-like
            List of items to exclude from the ranking. If passing this, will
            rank all the items except for these. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include or 'exclude'.
        output_score : bool
            Whether to output the scores in addition to the IDs. If passing
            'False', will return a single array with the item IDs, otherwise
            will return a tuple with the item IDs and the scores.

        Returns
        -------
        items : array(n,)
            The top-N highest predicted items for this user. If the 'X' data passed to
            fit was a DataFrame, will contain the item IDs from its column
            'ItemId', otherwise will be integers matching to the columns of 'X'.
        scores : array(n,)
            The predicted scores for the top-N items. Will only be returned
            when passing ``output_score=True``, in which case the result will
            be a tuple with these two entries.
        """
        a_vec, a_bias = self.factors_warm(X=X, X_col=X_col, X_val=X_val, W=W,
                                          U=U, U_bin=U_bin, U_col=U_col, U_val=U_val,
                                          return_bias=True)
        return self._topN(user=None, a_vec=a_vec, a_bias=a_bias, n=n,
                          include=include, exclude=exclude,
                          output_score=output_score)

    def transform(self, X=None, y=None, U=None, U_bin=None, W=None,
                  replace_existing=False):
        """
        Reconstruct missing entries of the 'X' matrix

        Will reconstruct/impute all the missing entries in the 'X' matrix as
        determined by the model. This method is intended to be used for imputing
        tabular data, and can be used as part of SciKit-Learn pipelines.

        Note
        ----
        It's possible to use this method with 'X' alone, with 'U'/'U_bin'
        alone, or with both 'X' and 'U'/'U_bin' together, in which case
        both matrices must have the same rows.

        Note
        ----
        If fitting the model to DataFrame inputs (instead of NumPy arrays and/or
        SciPy sparse matrices), the IDs are reindexed internally,
        and the inputs provided here should match with the numeration that was
        produced by the model. The mappings in such case are available under
        attributes ``self.user_mapping_`` and ``self.item_mapping_``.

        Parameters
        ----------
        X : array(m, n), or None
            New 'X' data with potentially missing entries which are to be imputed.
            Missing entries should have value ``np.nan`` when passing a dense
            array.
        y : None
            Not used. Kept as a placeholder for compatibility with SciKit-Learn
            pipelines.
        U : array(m, p), CSR matrix(m, p), COO matrix(m, p), or None
            User attributes information for each row in 'X'.
        U_bin : array(m, p_bin) or None
            User binary attributes for each row in 'X'.
            Only supported with ``method='lbfgs'``.
        W : array(m, n), array(nnz,), or None
            Observation weights. Must have the same shape as 'X' - that is,
            if 'X' is a sparse COO matrix, must be a 1-d array with the same
            number of non-zero entries as 'X.data', if 'X' is a 2-d array,
            'W' must also be a 2-d array.

        Returns
        -------
        X : array(m, n)
            The 'X' matrix as a dense array with all missing entries imputed
            according to the model.
        """
        if self._only_prediction_info:
            raise ValueError("Cannot use this function after dropping non-essential matrices.")
        if (X is not None) and (X.__class__.__name__ != "ndarray"):
            raise ValueError("'X' must be a NumPy array.")

        Xrow, Xcol, Xval, W_sp, Xarr, \
        Xcsr_p, Xcsr_i, Xcsr, \
        W_dense, Xorig, mask_take, \
        Uarr, Urow, Ucol, Uval, Ub_arr, \
        Ucsr_p, Ucsr_i, Ucsr, \
        n, m_u, m_x, p, pbin, \
        lambda_, lambda_bias, \
        l1_lambda, l1_lambda_bias = \
            self._process_transform_inputs(X=X, U=U, U_bin=U_bin, W=W,
                                           replace_existing=replace_existing)

        if Xarr.shape[0] == 0:
            Xarr = np.repeat(np.nan, self._n_orig*m_x).reshape((m_x, self._n_orig))

        c_funs = wrapper_float if self.use_float else wrapper_double
        return c_funs.call_impute_X_collective_explicit(
            Xarr,
            W_dense,
            Uarr,
            Urow,
            Ucol,
            Uval,
            Ucsr_p, Ucsr_i, Ucsr,
            Ub_arr,
            self._U_colmeans,
            self.item_bias_,
            self.B_,
            self._B_plus_bias,
            self.Bi_,
            self.C_,
            self.Cbin_,
            self._BtB,
            self._TransBtBinvBt,
            self._BeTBeChol,
            self._BiTBi,
            self._TransCtCinvCt,
            self._CtC,
            self._CtUbias,
            m_u,
            self.glob_mean_,
            self._n_orig,
            self.k, self.k_user, self.k_item, self.k_main,
            lambda_, lambda_bias,
            l1_lambda, l1_lambda_bias,
            self.scale_lam, self.scale_lam_sideinfo,
            self.scale_bias_const, self.scaling_biasA_,
            self.w_user, self.w_main, self.w_implicit,
            self.user_bias,
            self.NA_as_zero_user,
            self.nonneg,
            self.add_implicit_features,
            self.include_all_X,
            self.nthreads
        )

    def force_precompute_for_predictions(self):
        """
        Precompute internal matrices that are used for predictions

        Note
        ----
        It's not necessary to call this method if passing
        ``precompute_for_predictions=True``.

        Returns
        -------
        self
        
        """
        ### TODO: should have an option to precompute also for item factors
        assert self.is_fitted_
        if isinstance(self.lambda_, np.ndarray):
            lambda_ = self.lambda_[2]
            lambda_bias = self.lambda_[0]
        else:
            lambda_ = self.lambda_
            lambda_bias = self.lambda_
        c_funs = wrapper_float if self.use_float else wrapper_double
        
        self._B_plus_bias, self._BtB, self._TransBtBinvBt, self._BtXbias, \
        self._BeTBeChol, self._BiTBi, self._TransCtCinvCt, self._CtC, \
        self._CtUbias = \
            c_funs.precompute_matrices_collective_explicit(
                self.B_,
                self.C_,
                self.Bi_,
                self.item_bias_,
                self._U_colmeans,
                self.user_bias, self.add_implicit_features,
                self._n_orig,
                self.k, self.k_user, self.k_item, self.k_main,
                lambda_, lambda_bias,
                self.w_main, self.w_user, self.w_implicit,
                self.glob_mean_,
                self.scale_lam, self.scale_lam_sideinfo,
                self.scale_bias_const, self.scaling_biasA_,
                self.NA_as_zero,
                self.NA_as_zero_user,
                self.nonneg,
                self.include_all_X
            )
        return self

    @staticmethod
    def from_model_matrices(A, B, glob_mean=0., precompute=True,
                            user_bias=None, item_bias=None,
                            lambda_=1e+1, scale_lam=False, l1_lambda=0., nonneg=False,
                            NA_as_zero=False, scaling_biasA=None, scaling_biasB=None,
                            use_float=False, nthreads=-1):
        """
        Create a CMF model object from fitted matrices

        Creates a `CMF` model object based on fitted
        latent factor matrices, which might have been obtained from a different software.
        For example, the package ``python-libmf`` has functionality for obtaining these matrices,
        but not for producing recommendations or latent factors for new users, for which
        this function can come in handy as it will turn such model into a `CMF` model which
        provides all such functionality.

        This is only available for models without side information, and does not support
        user/item mappings.

        Note
        ----
        This is a static class method, should be called like this:
            ``CMF.from_model_matrices(...)``
        (i.e. no parentheses after 'CMF')

        Parameters
        ----------
        A : array(n_users, k)
            The obtained user factors.
        B : array(n_items, k)
            The obtained item factors.
        glob_mean : float
            The obtained global mean, if the model
            underwent centering. If passing zero, will assume that the values are not to
            be centered.
        precompute : bool
            Whether to generate pre-computed matrices which can help to speed
            up computations on new data.
        user_bias : None or array(n_users,)
            The obtained user biases.
            If passing ``None``, will assume that the model did not include user biases.
        item_bias : None or array(n_items,)
            The obtained item biases.
            If passing ``None``, will assume that the model did not include item biases.
        lambda_ : float or array(6,)
            Regularization parameter.
            See the documentation for ``__init__`` for details.
        scale_lam : bool
            Whether to scale (increase) the regularization parameter
            for each row of the model matrices according
            to the number of non-missing entries in the data for that
            particular row.
        l1_lambda : float or array(6,)
            Regularization parameter to apply to the L1 norm of the model matrices.
            See the documentation for ``__init__`` for details.
        nonneg : bool
            Whether to constrain the 'A' and 'B' matrices to be non-negative.
        NA_as_zero : bool
            Whether to take missing entries in the 'X' matrix as zeros (only
            when the 'X' matrix is passed as sparse COO matrix)
            instead of ignoring them.
            See the documentation for ``__init__`` for details.
        scaling_biasA : None or float
            If passing it, will assume that the model uses the option
            ``scale_bias_const=True``, and will use this number as scaling
            for the regularization of the user biases.
        scaling_biasB : None or float
            If passing it, will assume that the model uses the option
            ``scale_bias_const=True``, and will use this number as scaling
            for the regularization of the item biases.
        use_float : bool
            Whether to use C float type for the model parameters (typically this is
            ``np.float32``). If passing ``False``, will use C double (typically this
            is ``np.float64``). Using float types will speed up computations and
            use less memory, at the expense of reduced numerical precision.
        nthreads : int
            Number of parallel threads to use. If passing -1, will take the
            maximum available number of threads in the system.

        Returns
        -------
        model : CMF
            A ``CMF`` model object without side information, for which the usual
            prediction methods such as ``topN`` and ``topN_warm`` can be used as if
            it had been fitted through this software.
        """
        if scaling_biasA is not None:
            if user_bias is None:
                raise ValueError("Cannot pass 'scaling_biasA' when not using user biases.")
            if not scale_lam:
                raise ValueError("Cannot pass 'scaling_biasA' with 'scale_lam=False'.")
            scaling_biasA = float(scaling_biasA)
        if scaling_biasB is not None:
            if item_bias is None:
                raise ValueError("Cannot pass 'scaling_biasB' when not using item biases.")
            if not scale_lam:
                raise ValueError("Cannot pass 'scaling_biasB' with 'scale_lam=False'.")
            scaling_biasB = float(scaling_biasB)

        if (
                ((user_bias is not None) and (item_bias is not None)) and
                ((scaling_biasA is None) != (scaling_biasB is None))
        ):
            raise ValueError("Must pass both 'scaling_biasA' and 'scaling_biasB'.")

        if (not isinstance(A, np.ndarray)) or (not A.flags["C_CONTIGUOUS"]):
            A = np.ascontiguousarray(A)
        if (not isinstance(B, np.ndarray)) or (not B.flags["C_CONTIGUOUS"]):
            B = np.ascontiguousarray(B)
        if (len(A.shape) != 2) or (len(B.shape) != 2):
            raise ValueError("Model matrices must be 2-dimensional.")

        k = A.shape[1]
        if (B.shape[1] != k):
            raise ValueError("Dimensions of 'A' and 'B' do not match.")
        if (not A.shape[0]) or (not B.shape[0]) or (not k):
            raise ValueError("Empty model matrices not supported.")

        glob_mean = float(glob_mean)
        if pd.isnull(glob_mean):
            raise ValueError("'glob_mean' is NA.")
        center = glob_mean != 0.

        new_model = CMF(k = k,
                        user_bias = user_bias is not None,
                        item_bias = item_bias is not None,
                        center = center,
                        lambda_ = lambda_,
                        l1_lambda = l1_lambda,
                        scale_lam = scale_lam,
                        nonneg = nonneg,
                        NA_as_zero = NA_as_zero,
                        use_float = use_float,
                        nthreads = nthreads)

        dtype = ctypes.c_double if not use_float else ctypes.c_float

        if user_bias is not None:
            if not isinstance(user_bias, np.ndarray):
                user_bias = np.array(user_bias).reshape(-1)
            if user_bias.shape[0] != A.shape[0]:
                raise ValueError("'user_bias' dimension does not match with 'A'.")
            if not user_bias.flags["C_CONTIGUOUS"]:
                user_bias = np.ascontiguousarray(user_bias)
            if user_bias.dtype != dtype:
                user_bias = user_bias.astype(dtype)
        if item_bias is not None:
            if not isinstance(item_bias, np.ndarray):
                item_bias = np.array(item_bias).reshape(-1)
            if item_bias.shape[0] != B.shape[0]:
                raise ValueError("'item_bias' dimension does not match with 'B'.")
            if not item_bias.flags["C_CONTIGUOUS"]:
                item_bias = np.ascontiguousarray(item_bias)
            if item_bias.dtype != dtype:
                item_bias = item_bias.astype(dtype)

        if (A.dtype != dtype):
            A = A.astype(dtype)
        if (B.dtype != dtype):
            B = B.astype(dtype)

        new_model.A_ = A
        new_model.B_ = B
        new_model.glob_mean_ = glob_mean
        if user_bias is not None:
            new_model.user_bias_ = user_bias
            if scaling_biasA is not None:
                new_model.scaling_biasA_ = scaling_biasA
        if item_bias is not None:
            new_model.item_bias_ = item_bias
            if scaling_biasB is not None:
                new_model.scaling_biasB_ = scaling_biasB

        new_model._A_pred = A
        new_model._B_pred = B
        new_model._n_orig = B.shape[0]
        new_model.reindex_ = False

        new_model.is_fitted_ = True
        if precompute:
            new_model.force_precompute_for_predictions()
        return new_model
        

class CMF_implicit(_CMF):
    """
    Collective model for implicit-feedback data

    Tries to approximate the 'X' interactions matrix  by a formula as follows:
        X ~ A * t(B)
    While at the same time also approximating the user side information
    matrix 'U' and the item side information matrix 'I' as follows:
        U ~ A * t(C), 
        I ~ B * t(D)

    Note
    ----
    The default hyperparameters in this software are very different from others.
    For example, to match those of the package ``implicit``, the corresponding
    hyperparameters here would be ``use_cg=True``, ``finalize_chol=False``,
    ``k=100``, ``lambda_=0.01``, ``niter=15``, ``use_float=True``, `alpha=1.``,
    (see the individual documentation of each hyperarameter
    for details).

    Note
    ----
    The default arguments are not geared towards speed.
    For faster fitting, use ``use_cg=True``,
    ``finalize_chol=False``, ``use_float=True``,
    ``precompute_for_predictions=False``, ``produce_dicts=False``,
    and pass COO matrices or NumPy arrays instead of DataFrames to ``fit``.

    Note
    ----
    The model optimization objective will not scale any of its terms according
    to number of entries, so hyperparameters such as ``lambda_`` will require
    more tuning than in other software and trying out values over a wider range.

    Note
    ----
    This model is fit through the alternating least-squares method only,
    it does not offer a gradient-based approach like the explicit-feedback
    version.

    Note
    ----
    This model will not perform mean centering and will not fit uer/item
    biases. If desired, an equivalent problem formulation can be made through
    ``CMF`` which can accommodate mean centering and biases.

    Parameters
    ----------
    k : int
        Number of latent factors to use (dimensionality of the low-rank
        factorization), which will be shared between the factorization of the
        'X' matrix and the side info matrices. Additional non-shared components
        can also be specified through ``k_user``, ``k_item``, and ``k_main``.
        Typical values are 30 to 100.
    lambda_ : float or array(6,)
        Regularization parameter. Can also use different regularization for each
        matrix, in which case it should be an array with 6 entries, corresponding,
        in this order, to: <ignored>, <ignored>, A, B, C, D. Note that the default
        value for ``lambda_`` here is much higher than in other software, and that
        the loss/objective function is not divided by the number of entries.
        For example, a good number for the LastFM-360K could be ``lambda_=5``.
        Typical values are 10^-2 to 10^2.
    alpha : float
        Weighting parameter for the non-zero entries in the implicit-feedback
        model. See [3] for details. Note that, while the author's suggestion for
        this value is 40, other software such as ``implicit`` use a value of 1,
        whereas Spark uses a value of 0.01 by default,
        and values higher than 10 are unlikely to improve results. If the data
        has very high values, might even be beneficial to put a very low value
        here - for example, for the LastFM-360K, values below 1 might
        give better results.
    use_cg : bool
        In the ALS method, whether to use a conjugate gradient method to solve
        the closed-form least squares problems. This is a faster and more
        memory-efficient alternative than the default Cholesky solver, but less
        exact, less numerically stable, and will require slightly more ALS
        iterations (``niter``) to reach a good optimum.
        In general, better results are achieved with ``use_cg=False``.
        Note that, if using this method, calculations after fitting which involve
        new data such as ``factors_warm``,  might produce slightly different
        results from the factors obtained from calling ``fit`` with the same data,
        due to differences in numerical precision. A workaround for this issue
        (factors on new data that might differ slightly) is to use
        ``finalize_chol=True``.
        Even if passing "True" here, will use the Cholesky method in cases in which
        it is faster (e.g. dense matrices with no missing values),
        and will not use the conjugate gradient method on new data.
        This option is not available when using L1 regularization and/or
        non-negativity constraints.
    k_user : int
        Number of factors in the factorizing A and C matrices which will be used
        only for the 'U' matrix, while being ignored for the 'X' matrix.
        These will be the first factors of the matrices once the model is fit.
        Will be counted in addition to those already set by ``k``.
    k_item : int
        Number of factors in the factorizing B and D matrices which will be used
        only for the 'I' matrix, while being ignored for the 'X' matrix.
        These will be the first factors of the matrices once the model is fit.
        Will be counted in addition to those already set by ``k``.
    k_main : int
        Number of factors in the factorizing A and B matrices which will be used
        only for the 'X' matrix, while being ignored for the 'U' and 'I' matrices.
        These will be the last factors of the matrices once the model is fit.
        Will be counted in addition to those already set by ``k``.
    w_main : float
        Weight in the optimization objective for the errors in the factorization
        of the 'X' matrix.
        Note that, since the "X" matrix is considered to be full with mostly zero
        values, the overall sum of errors for "X" will be much larger than for the
        side info matrices (especially if using large ``alpha``), thus it's
        recommended to give higher weights to the side info matrices than to
        the main matrix.
    w_user : float
        Weight in the optimization objective for the errors in the factorization
        of the 'U' matrix. Ignored when not passing 'U' to 'fit'.
        Note that, since the "X" matrix is considered to be full with mostly zero
        values, the overall sum of errors for "X" will be much larger than for the
        side info matrices (especially if using large ``alpha``), thus it's
        recommended to give higher weights to the side info matrices than to
        the main matrix.
    w_item : float
        Weight in the optimization objective for the errors in the factorization
        of the 'I' matrix. Ignored when not passing 'I' to 'fit'.
        Note that, since the "X" matrix is considered to be full with mostly zero
        values, the overall sum of errors for "X" will be much larger than for the
        side info matrices (especially if using large ``alpha``), thus it's
        recommended to give higher weights to the side info matrices than to
        the main matrix.
    l1_lambda : float or array(6,)
        Regularization parameter to apply to the L1 norm of the model matrices.
        Can also pass different values for each matrix (see ``lambda_`` for
        details). Note that, when adding L1 regularization, the model will be
        git through a coordinate descent procedure, which is significantly
        slower than the Cholesky method with L2 regularization.
        Not recommended.
    center_U : bool
        Whether to center the 'U' matrix column-by-column. Be aware that this
        is a simple mean centering without regularization. One might want to
        turn this option off when using ``NA_as_zero_user=True``.
    center_I : bool
        Whether to center the 'I' matrix column-by-column. Be aware that this
        is a simple mean centering without regularization. One might want to
        turn this option off when using ``NA_as_zero_item=True``.
    niter : int
        Number of alternating least-squares iterations to perform. Note that
        one iteration denotes an update round for all the matrices rather than
        an update of a single matrix. In general, the more iterations, the better
        the end result.
        Typical values are 6 to 30.
    NA_as_zero_user : bool
        Whether to take missing entries in the 'U' matrix as zeros (only
        when the 'U' matrix is passed as sparse COO matrix) instead of ignoring them.
        Note that passing "True" will affect the results of the functions named
        "warm" if no data is passed there (as it will assume zeros instead of
        missing).
    NA_as_zero_item : bool
        Whether to take missing entries in the 'I' matrix as zeros (only
        when the 'I' matrix is passed as sparse COO matrix) instead of ignoring them.
    nonneg : bool
        Whether to constrain the 'A' and 'B' matrices to be non-negative.
        In order for this to work correctly, the 'X' input data must also be
        non-negative. This constraint will also be applied to the 'Ai'
        and 'Bi' matrices if passing ``add_implicit_features=True``.
        This option is not available when using the L-BFGS method.
        Note that, when determining non-negative factors, it will always
        use a coordinate descent method, regardless of the value passed
        for ``use_cg`` and ``finalize_chol``.
        When used for recommender systems, one usually wants to pass 'False' here.
        For better results, use a higher regularization and more iterations.
    nonneg_C: bool
        Whether to constrain the 'C' matrix to be non-negative.
        In order for this to work correctly, the 'U' input data must also be
        non-negative.
    nonneg_D: bool
        Whether to constrain the 'D' matrix to be non-negative.
        In order for this to work correctly, the 'I' input data must also be
        non-negative.
    max_cd_steps : int
        Maximum number of coordinate descent updates to perform per iteration.
        Pass zero for no limit.
        The procedure will only use coordinate descent updates when having
        L1 regularization and/or non-negativity constraints.
        This number should usually be larger than ``k``.
    apply_log_transf : bool
        Whether to apply a logarithm transformation on the values of 'X'
        (i.e. 'X := log(X)')
    precompute_for_predictions : bool
        Whether to precompute some of the matrices that are used when making
        predictions from the model. If 'False', it will take longer to generate
        predictions or top-N lists, but will use less memory and will be faster
        to fit the model. If passing 'False', can be recomputed later on-demand
        through method 'force_precompute_for_predictions'.
    use_float : bool
        Whether to use C float type for the model parameters (typically this is
        ``np.float32``). If passing ``False``, will use C double (typically this
        is ``np.float64``). Using float types will speed up computations and
        use less memory, at the expense of reduced numerical precision.
    max_cg_steps : int
        Maximum number of conjugate gradient iterations to perform in an ALS round.
        Ignored when passing ``use_cg=False``.
    finalize_chol : bool
        When passing ``use_cg=True``, whether to perform the last iteration with
        the Cholesky solver. This will make it slower, but will avoid the issue
        of potential mismatches between the result from ``fit`` and calls to
        ``factors_warm`` or similar with the same data.
    random_state : int, RandomState, or Generator
        Seed used to initialize parameters at random. If passing a NumPy
        RandomState or Generator, will use it to draw a random integer.
    init : str, "normal" or "gamma"
        Distribution used to initialize the model parameters. Both
        distributions are likely to reach similar end results, but the
        distribution of the factors themselves will be different.
    verbose : bool
        Whether to print informational messages about the optimization
        routine used to fit the model. Note that, if running this from a
        Jupyter notebook, these messages will be printed in the console,
        not in the notebook itself.
    produce_dicts : bool
        Whether to produce Python dicts from the mappings between user/item
        IDs passed to 'fit' and the internal IDs used by the class. Having
        these dicts might speed up some computations such as 'predict',
        but it will add some extra overhead at the time of fitting the model
        and extra memory usage. Ignored when passing the data as matrices
        and arrays instead of data frames.
    handle_interrupt : bool
        When receiving an interrupt signal, whether the model should stop
        early and leave a usable object with the parameters obtained up
        to the point when it was interrupted (when passing 'True'), or
        raise an interrupt exception without producing a fitted model object
        (when passing 'False').
    copy_data : bool
        Whether to make copies of the input data that is passed to this
        object's methods (``fit``, ``predict``, etc.), in order to avoid
        modifying such data in-place. Passing ``False`` will save some
        computation time and memory usage.
    nthreads : int
        Number of parallel threads to use. If passing -1, will take the
        maximum available number of threads in the system.

    Attributes
    ----------
    is_fitted_ : bool
        Whether the model has been fitted to data.
    reindex_ : bool
        Whether the IDs passed to 'fit' were reindexed internally
        (this will only happen when passing data frames to 'fit').
    user_mapping_ : array(m,) or array(0,)
        Correspondence of internal user (row) IDs to IDs in the data
        passed to 'fit'. Will only be non-empty when passing a data frame
        as input to 'X'.
    item_mapping_ : array(n,) or array(0,)
        Correspondence of internal item (column) IDs to IDs in the data
        passed to 'fit'. Will only be non-empty when passing a data frame
        as input to 'X'.
    user_dict_ : dict
        Python dict version of ``user_mapping_``. Only filled-in when
        passing ``produce_dicts=True`` and when passing data frames to 'fit'.
    item_dict_ : dict
        Python dict version of ``item_mapping_``. Only filled-in when
        passing ``produce_dicts=True`` and when passing data frames to 'fit'.
    A_ : array(m, k_user+k+k_main)
        The obtained user factors.
    B_ : array(n, k_item+k+k_main)
        The obtained item factors.
    C_ : array(p, k_user+k)
        The obtained user-attributes factors.
    D_ : array(q, k_item+k)
        The obtained item attributes factors.

    References
    ----------
    .. [1] Cortes, David.
           "Cold-start recommendations in Collective Matrix Factorization."
           arXiv preprint arXiv:1809.00366 (2018).
    .. [2] Singh, Ajit P., and Geoffrey J. Gordon.
           "Relational learning via collective matrix factorization."
           Proceedings of the 14th ACM SIGKDD international conference on
           Knowledge discovery and data mining. 2008.
    .. [3] Hu, Yifan, Yehuda Koren, and Chris Volinsky.
           "Collaborative filtering for implicit feedback datasets."
           2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008.
    .. [4] Takacs, Gabor, Istvan Pilaszy, and Domonkos Tikk.
           "Applications of the conjugate gradient method for implicit feedback collaborative filtering."
           Proceedings of the fifth ACM conference on Recommender systems. 2011.
    .. [5] Franc, Vojtěch, Václav Hlaváč, and Mirko Navara.
           "Sequential coordinate-wise algorithm for the
           non-negative least squares problem."
           International Conference on Computer Analysis of Images and Patterns.
           Springer, Berlin, Heidelberg, 2005.
    """
    def __init__(self, k=50, lambda_=1e0, alpha=1., use_cg=True,
                 k_user=0, k_item=0, k_main=0,
                 w_main=1., w_user=10., w_item=10.,
                 l1_lambda=0., center_U=True, center_I=True,
                 niter=10, NA_as_zero_user=False, NA_as_zero_item=False,
                 nonneg=False, nonneg_C=False, nonneg_D=False, max_cd_steps=100,
                 apply_log_transf=False,
                 precompute_for_predictions=True, use_float=False,
                 max_cg_steps=3, finalize_chol=False,
                 random_state=1, init="normal", verbose=False,
                 produce_dicts=False, handle_interrupt=True,
                 copy_data=True, nthreads=-1):
        self._take_params(implicit=True, alpha=alpha, downweight=False,
                          k=k, lambda_=lambda_, method="als",
                          use_cg=use_cg, max_cg_steps=max_cg_steps,
                          finalize_chol=finalize_chol,
                          apply_log_transf=apply_log_transf,
                          nonneg=nonneg, nonneg_C=nonneg_C, nonneg_D=nonneg_D,
                          max_cd_steps=max_cd_steps,
                          user_bias=False, item_bias=False,
                          k_user=k_user, k_item=k_item, k_main=k_main,
                          w_main=w_main, w_user=w_user, w_item=w_item,
                          l1_lambda=l1_lambda, center_U=center_U, center_I=center_I,
                          maxiter=0, niter=niter, parallelize="separate",
                          corr_pairs=0,
                          NA_as_zero=False, NA_as_zero_user=NA_as_zero_user,
                          NA_as_zero_item=NA_as_zero_item,
                          precompute_for_predictions=precompute_for_predictions,
                          use_float=use_float,
                          random_state=random_state, init=init,
                          verbose=verbose, print_every=0,
                          handle_interrupt=handle_interrupt,
                          produce_dicts=produce_dicts, copy_data=copy_data,
                          nthreads=nthreads)

    def __str__(self):
        msg  = "Collective matrix factorization model\n"
        msg += "(implicit-feedback variant)\n\n"
        if not self.is_fitted_:
            msg += "Model has not been fitted to data.\n"
        return msg

    def get_params(self, deep=None):
        return {
            "k" : self.k, "lambda_" : self.lambda_, "alpha" : self.alpha,
            "k_user" : self.k_user, "k_item" : self.k_item, "k_main" : self.k_main,
            "w_main" : self.w_main, "w_user" : self.w_user, "w_item" : self.w_item,
            "downweight" : self.downweight,
            "niter" : self.niter, "NA_as_zero_user" : self.NA_as_zero_user,
            "NA_as_zero_item" : self.NA_as_zero_item,
            "use_float" : self.use_float, "use_cg" : self.use_cg,
            "random_state" : self.random_state, "init" : self.init,
            "nthreads" : self.nthreads
        }

    def fit(self, X, U=None, I=None):
        """
        Fit model to implicit-feedback data and user/item attributes

        Note
        ----
        It's possible to pass partially disjoints sets of users/items between
        the different matrices (e.g. it's possible for both the 'X' and 'U'
        matrices to have rows that the other doesn't have), but note that
        missing values in 'X' are treated as zeros.
        The procedure supports missing values for "U" and "I".
        If any of the inputs has less rows/columns than the other(s) (e.g.
        "U" has more rows than "X", or "I" has more rows than there are columns
        in "X"), will assume that the rest of the rows/columns have only
        missing values (zero values for "X").
        Note however that when having partially disjoint inputs, the order of
        the rows/columns matters for speed, as it might run faster when the "U"/"I"
        inputs that do not have matching rows/columns in "X" have those unmatched
        rows/columns at the end (last rows/columns) and the "X" input is shorter.

        Note
        ----
        When passing NumPy arrays, missing (unobserved) entries should 
        have value ``np.nan``. When passing sparse inputs, the zero-valued entries
        will be considered as missing (unless using "NA_as_zero", and except for
        "X" for which missing will always be treated as zero), and it should
        not contain "NaN" values among the non-zero entries.

        Note
        ----
        In order to avoid potential decimal differences in the factors obtained
        when fitting the model and when calling the prediction functions on
        new data, when the data is sparse, it's necessary to sort it beforehand
        by columns and also pass the data data with indices sorted (by column)
        to the prediction functions.

        Parameters
        ----------
        X : DataFrame(nnz, 3), or sparse COO(m, n)
            Matrix to factorize. Can be passed as a SciPy
            sparse COO matrix (recommended), or
            as a Pandas DataFrame, in which case it should contain the
            following columns: 'UserId', 'ItemId', and 'Value'.
            If passing a DataFrame,
            the IDs will be internally remapped.
        U : array(m, p), COO(m, p), DataFrame(m, p+1), or None
            User attributes information. If 'X' is a DataFrame, should also
            be a DataFrame, containing column 'UserId'. If 'U' is sparse,
            'X' should be passed as a sparse COO matrix too.
        I : array(n, q), COO(n, q), DataFrame(n, q+1), or None
            Item attributes information. If 'X' is a DataFrame, should also
            be a DataFrame, containing column 'ItemId'. If 'I' is sparse,
            'X' should be passed as a sparse COO matrix too.

        Returns
        -------
        self

        """
        if X.__class__.__name__ not in ("coo_matrix", "DataFrame"):
            raise ValueError("'X' must be a Pandas DataFrame or SciPy sparse COO matrix.")
        return self._fit_common(X, U=U, I=I, U_bin=None, I_bin=None, W=None)

    def _fit(self,
             Xrow, Xcol, Xval, W_sp, Xarr, W_dense,
             Uarr, Urow, Ucol, Uval, Ub_arr,
             Iarr, Irow, Icol, Ival, Ib_arr,
             m, n, m_u, n_i, p, q,
             m_ub, n_ib, pbin, qbin):
        c_funs = wrapper_float if self.use_float else wrapper_double

        self.A_, self.B_, self.C_, self.D_, \
        self._U_colmeans, self._I_colmeans, self._w_main_multiplier, \
        self._BtB, self._BeTBe, self._BeTBeChol, self._CtUbias = \
            c_funs.call_fit_collective_implicit_als(
                Xrow,
                Xcol,
                Xval,
                Uarr,
                Urow,
                Ucol,
                Uval,
                Iarr,
                Irow,
                Icol,
                Ival,
                self.NA_as_zero_user, self.NA_as_zero_item,
                m, n, m_u, n_i, p, q,
                self.k, self.k_user, self.k_item, self.k_main,
                self.w_main, self.w_user, self.w_item,
                self.lambda_ if isinstance(self.lambda_, float) else 0.,
                self.alpha, self.downweight, self.apply_log_transf,
                self.lambda_ if isinstance(self.lambda_, np.ndarray) else np.empty(0, dtype=self.dtype_),
                self.l1_lambda if isinstance(self.l1_lambda, float) else 0.,
                self.l1_lambda if isinstance(self.l1_lambda, np.ndarray) else np.empty(0, dtype=self.dtype_),
                self.center_U, self.center_I,
                self.verbose, self.niter,
                self.nthreads, self.use_cg,
                self.max_cg_steps, self.finalize_chol,
                self.nonneg, self.nonneg_C, self.nonneg_D, self.max_cd_steps,
                self.random_state, init=self.init,
                handle_interrupt=self.handle_interrupt,
                precompute_for_predictions=self.precompute_for_predictions
            )

        self._A_pred = self.A_
        self._B_pred = self.B_
        self._n_orig = self.B_.shape[0]
        self.is_fitted_ = True
        return self

    def force_precompute_for_predictions(self):
        """
        Precompute internal matrices that are used for predictions

        Note
        ----
        It's not necessary to call this method if passing
        ``precompute_for_predictions=True``.

        Returns
        -------
        self

        """
        ### TODO: should have an option to precompute also for item factors
        assert self.is_fitted_
        if isinstance(self.lambda_, np.ndarray):
            lambda_ = self.lambda_[2]
        else:
            lambda_ = self.lambda_
        c_funs = wrapper_float if self.use_float else wrapper_double
        self._BtB, self._BeTBe, self._BeTBeChol = \
            c_funs.precompute_matrices_collective_implicit(
                self.B_, self.C_, self._U_colmeans,
                self.k, self.k_main, self.k_user, self.k_item,
                lambda_,
                self.w_main, self.w_user,
                self._w_main_multiplier,
                self.nonneg,
                self.NA_as_zero_U
            )
        return self

    def factors_cold(self, U=None, U_col=None, U_val=None):
        """
        Determine user-factors from new data, given U

        Parameters
        ----------
        U : array(p,), or None
            User attributes in the new data (1-row only).
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_col : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_val : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.

        Returns
        -------
        factors : array(k_user+k+k_main,)
            The user-factors as determined by the model.
        """
        return self._factors_cold(U=U, U_bin=None, U_col=U_col, U_val=U_val)

    def topN_cold(self, n=10, U=None, U_col=None, U_val=None,
                  include=None, exclude=None, output_score=False):
        """
        Compute top-N highest-predicted items for a new user, given 'U'

        Parameters
        ----------
        n : int
            Number of top-N highest-predicted results to output.
        U : array(p,), or None
            Attributes for the new user, in dense format.
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_col : None or array(nnz)
            Attributes for the new user, in sparse format.
            'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_val : None or array(nnz)
            Attributes for the new user, in sparse format.
            'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.
        include : array-like
            List of items which will be ranked. If passing this, will only
            make a ranking among these items. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include or 'exclude'.
        exclude : array-like
            List of items to exclude from the ranking. If passing this, will
            rank all the items except for these. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include or 'exclude'.
        output_score : bool
            Whether to output the scores in addition to the IDs. If passing
            'False', will return a single array with the item IDs, otherwise
            will return a tuple with the item IDs and the scores.

        Returns
        -------
        items : array(n,)
            The top-N highest predicted items for this user. If the 'X' data passed to
            fit was a DataFrame, will contain the item IDs from its column
            'ItemId', otherwise will be integers matching to the columns of 'X'.
        scores : array(n,)
            The predicted scores for the top-N items. Will only be returned
            when passing ``output_score=True``, in which case the result will
            be a tuple with these two entries.
        """
        a_vec = self.factors_cold(U, U_col, U_val)
        return self._topN(user=None, a_vec=a_vec, a_bias=0., n=n,
                          include=include, exclude=exclude,
                          output_score=output_score)

    def predict_cold(self, items, U=None, U_col=None, U_val=None):
        """
        Predict value/confidence given by a new user to existing items, given U

        Parameters
        ----------
        items : array-like(n,)
            Items whose ratings are to be predicted. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'.
        U : array(p,), or None
            Attributes for the new user, in dense format.
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_col : None or array(nnz)
            Attributes for the new user, in sparse format.
            'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_val : None or array(nnz)
            Attributes for the new user, in sparse format.
            'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.

        Returns
        -------
        scores : array(n,)
            Predicted ratings for the requested items, for this user.
        """
        a_vec = self.factors_cold(U, U_col, U_val)
        return self._predict(user=None, a_vec=a_vec, a_bias=0., item=items)

    def predict_cold_multiple(self, item, U):
        """
        Predict value/confidence given by new users to existing items, given U

        Note
        ----
        See the documentation of "fit" for details about handling of missing values.

        Parameters
        ----------
        item : array-like(m,)
            Items for which ratings/values are to be predicted. If 'X' passed to
            fit was a  DataFrame, must match with the entries in its 'ItemId'
            column, otherwise should match with the columns of 'X'.
        U : array(m, p), CSR matrix(m, q), or COO matrix(m, q)
            Attributes for the users for which to predict ratings/values.
            Data frames with 'UserId' column are not supported.
            Must have one row per entry
            in ``item``.

        Returns
        -------
        scores : array(m,)
            Predicted ratings for the requested user-item combinations.
        """
        A = self._factors_cold_multiple(U=U, U_bin=None, is_I=False)
        return self._predict_user_multiple(A, item)

    def item_factors_cold(self, I=None, I_col=None, I_val=None):
        """
        Determine item-factors from new data, given I

        Note
        ----
        Calculating item factors might be a lot slower than user factors,
        as the model does not keep precomputed matrices that might speed
        up these factor calculations. If this function is goint to be used
        frequently, it's advised to build the model swapping the users
        and items instead.

        Parameters
        ----------
        I : array(q,), or None
            Attributes for the new item, in dense format.
            Should only pass one of 'I' or 'I_col'+'I_val'.
        I_col : None or array(nnz)
            Attributes for the new item, in sparse format.
            'I_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'I' or 'I_col'+'I_val'.
        I_val : None or array(nnz)
            Attributes for the new item, in sparse format.
            'I_val' should contain the values in the columns
            given by 'I_col'.
            Should only pass one of 'I' or 'I_col'+'I_val'.

        Returns
        -------
        factors : array(k_item+k+k_main,)
            The item-factors as determined by the model.
        """
        return self._item_factors_cold(I=I, I_bin=None, I_col=I_col, I_val=I_val)

    def predict_new(self, user, I):
        """
        Predict rating given by existing users to new items, given I

        Note
        ----
        Calculating item factors might be a lot slower than user factors,
        as the model does not keep precomputed matrices that might speed
        up these factor calculations. If this function is goint to be used
        frequently, it's advised to build the model swapping the users
        and items instead.

        Parameters
        ----------
        user : array-like(n,)
            Users for whom ratings/values are to be predicted. If 'X' passed to
            fit was a  DataFrame, must match with the entries in its 'UserId'
            column, otherwise should match with the rows of 'X'.
        I : array(n, q), CSR matrix(n, q), or COO matrix(n, q)
            Attributes for the items for which to predict ratings/values. Data frames with 'ItemId' column are not supported. Must have one row per entry
            in ``user``.

        Returns
        -------
        scores : array(n,)
            Predicted ratings for the requested user-item combinations.
        """
        if self._only_prediction_info:
            raise ValueError("Cannot use this function after dropping non-essential matrices.")
        B = self._factors_cold_multiple(U=I, U_bin=None, is_I=True)
        return self._predict_new(user, B)

    def topN_new(self, user, I=None, n=10, output_score=False):
        """
        Rank top-N highest-predicted items for an existing user, given 'I'

        Parameters
        ----------
        user : int or obj
            User for which to rank the items. If 'X' passed to 'fit' was a
            data frame, must match with entries in its 'UserId' column,
            otherwise should match with the rows on 'X'.
        I : array(m, q), CSR matrix(m, q), or COO matrix(m, q)
            Attributes for the items to rank. Data frames with 'ItemId'
            column are not supported.
        n : int
            Number of top-N highest-predicted results to output. Must be
            less or equal than the number of rows in I.
        output_score : bool
            Whether to output the scores in addition to the IDs. If passing
            'False', will return a single array with the item IDs, otherwise
            will return a tuple with the item IDs and the scores.

        Returns
        -------
        items : array(n,)
            The top-N highest predicted items for this user, as integers
            matching to the rows of 'I'.
        scores : array(n,)
            The predicted scores for the top-N items. Will only be returned
            when passing ``output_score=True``, in which case the result will
            be a tuple with these two entries.
        """
        if self._only_prediction_info:
            raise ValueError("Cannot use this function after dropping non-essential matrices.")
        B = self._factors_cold_multiple(U=I, U_bin=None, is_I=True)
        return self._topN(user=user, B=B, n=n, output_score=output_score)

    def factors_warm(self, X_col, X_val,
                     U=None, U_col=None, U_val=None):
        """
        Determine user latent factors based on new interactions data

        Parameters
        ----------
        X_col : array(nnz,)
            Observed new 'X' data for a given user, in sparse format.
            'X_col' should contain the column indices (items) of the observed 
            entries. If 'X' passed to 'fit' was a data frame, should have
            entries from 'ItemId' column, otherwise should have column numbers
            (starting at zero).
        X_val : array(nnz,)
            Observed new 'X' data for a given user, in sparse format.
            'X_val' should contain the values in the columns/items given by
            'X_col'.
        U : array(p,), or None
            User attributes in the new data (1-row only).
            Should only pass one of 'U' or 'U_col'+'U_val'.
            User side information is not strictly required, and can
            skip both.
        U_col : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
            User side information is not strictly required, and can
            skip both.
        U_val : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.
            User side information is not strictly required, and can
            skip both.

        Returns
        -------
        factors : array(k_user+k+k_main,)
            User factors as determined from the data in 'X_col' and 'X_val'.
        """
        return self._factors_warm_common(X=None, X_col=X_col, X_val=X_val, W=None,
                                         U=U, U_bin=None, U_col=U_col, U_val=U_val,
                                         return_bias=False)

    def _factors_warm(self, X, W_dense, X_val, X_col, W_sp,
                      U, U_val, U_col, U_bin, return_bias):
        if isinstance(self.lambda_, np.ndarray):
            lambda_ = self.lambda_[2]
        else:
            lambda_ = self.lambda_

        c_funs = wrapper_float if self.use_float else wrapper_double
        a_vec = c_funs.call_factors_collective_implicit_single(
            X_val,
            X_col,
            U,
            U_val,
            U_col,
            self._U_colmeans,
            self.B_,
            self.C_,
            self._BeTBe,
            self._BtB,
            self._BeTBeChol,
            self._CtUbias,
            self.k, self.k_user, self.k_item, self.k_main,
            lambda_, l1_lambda, self.alpha,
            self._w_main_multiplier,
            self.w_user, self.w_main,
            self.apply_log_transf,
            self.NA_as_zero_user,
            self.nonneg
        )

        return a_vec

    def factors_multiple(self, X=None, U=None):
        """
        Determine user latent factors based on new data (warm and cold)

        Determines latent factors for multiple rows/users at once given new
        data for them.

        Note
        ----
        See the documentation of "fit" for details about handling of missing values.

        Note
        ----
        If fitting the model to DataFrame inputs (instead of NumPy arrays and/or
        SciPy sparse matrices), the IDs are reindexed internally,
        and the inputs provided here should match with the numeration that was
        produced by the model. The mappings in such case are available under
        attributes ``self.user_mapping_`` and ``self.item_mapping_``.
        
        Parameters
        ----------
        X : CSR matrix(m_x, n), COO matrix(m_x, n), or None
            New 'X' data.
        U : array(m_u, p), CSR matrix(m_u, p), COO matrix(m_u, p), or None
            User attributes information for rows in 'X'.

        Returns
        -------
        A : array(max(m_x,m_u), k_user+k+k_main)
            The new factors determined for all the rows given the new data.
        """
        if (X is None) and (U is None):
            raise ValueError("Must pass at least one of 'X', 'U'.")
        
        A, _ = self._factors_multiple_common(X, U, None, None)
        return A


    def topN_warm(self, n=10, X_col=None, X_val=None,
                  U=None, U_col=None, U_val=None,
                  include=None, exclude=None, output_score=False):
        """
        Compute top-N highest-predicted items for a new user, given 'X'

        Parameters
        ----------
        n : int
            Number of top-N highest-predicted results to output.
        X_col : array(nnz,)
            Observed new 'X' data for a given user, in sparse format.
            'X_col' should contain the column indices (items) of the observed 
            entries. If 'X' passed to 'fit' was a data frame, should have
            entries from 'ItemId' column, otherwise should have column numbers
            (starting at zero).
        X_val : array(nnz,)
            Observed new 'X' data for a given user, in sparse format.
            'X_val' should contain the values in the columns/items given by
            'X_col'.
        U : array(p,), or None
            User attributes in the new data (1-row only).
            Should only pass one of 'U' or 'U_col'+'U_val'.
            User side information is not strictly required, and can
            skip both.
        U_col : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
            User side information is not strictly required, and can
            skip both.
        U_val : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.
            User side information is not strictly required, and can
            skip both.
        include : array-like
            List of items which will be ranked. If passing this, will only
            make a ranking among these items. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include or 'exclude'.
        exclude : array-like
            List of items to exclude from the ranking. If passing this, will
            rank all the items except for these. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include or 'exclude'.
        output_score : bool
            Whether to output the scores in addition to the IDs. If passing
            'False', will return a single array with the item IDs, otherwise
            will return a tuple with the item IDs and the scores.
        
        Returns
        -------
        items : array(n,)
            The top-N highest predicted items for this user. If the 'X' data passed to
            fit was a DataFrame, will contain the item IDs from its column
            'ItemId', otherwise will be integers matching to the columns of 'X'.
        scores : array(n,)
            The predicted scores for the top-N items. Will only be returned
            when passing ``output_score=True``, in which case the result will
            be a tuple with these two entries.
        """
        a_vec = self.factors_warm(X_col=X_col, X_val=X_val,
                                  U=U, U_col=U_col, U_val=U_val)
        return self._topN(user=None, a_vec=a_vec, a_bias=0., n=n,
                          include=include, exclude=exclude,
                          output_score=output_score)

    def predict_warm(self, items, X_col, X_val,
                     U=None, U_col=None, U_val=None):
        """
        Predict scores for existing items, for a new user, given 'X'

        Parameters
        ----------
        items : array-like(n,)
            Items whose ratings are to be predicted. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'.
        X_col : array(nnz,)
            Observed new 'X' data for a given user, in sparse format.
            'X_col' should contain the column indices (items) of the observed 
            entries. If 'X' passed to 'fit' was a data frame, should have
            entries from 'ItemId' column, otherwise should have column numbers
            (starting at zero).
        X_val : array(nnz,)
            Observed new 'X' data for a given user, in sparse format.
            'X_val' should contain the values in the columns/items given by
            'X_col'.
        U : array(p,), or None
            User attributes in the new data (1-row only).
            Should only pass one of 'U' or 'U_col'+'U_val'.
            User side information is not strictly required, and can
            skip both.
        U_col : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
            User side information is not strictly required, and can
            skip both.
        U_val : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.
            User side information is not strictly required, and can
            skip both.
        
        Returns
        -------
        scores : array(n,)
            Predicted values for the requested items for a user defined by
            the given values of 'X' in 'X_col' and 'X_val', plus 'U' if passed.
        """
        a_vec = self.factors_warm(X_col=X_col, X_val=X_val,
                                  U=U, U_col=U_col, U_val=U_val)
        return self._predict(user=None, a_vec=a_vec, a_bias=0., item=items)

    def predict_warm_multiple(self, X, item, U=None):
        """
        Predict scores for existing items, for new users, given 'X'

        Note
        ----
        See the documentation of "fit" for details about handling of missing values.

        Note
        ----
        If fitting the model to DataFrame inputs (instead of NumPy arrays and/or
        SciPy sparse matrices), the IDs are reindexed internally,
        and the inputs provided here should match with the numeration that was
        produced by the model. The mappings in such case are available under
        attributes ``self.user_mapping_`` and ``self.item_mapping_``.

        Parameters
        ----------
        X : CSR matrix(m, n) , or COO matrix(m, n)
            New 'X' data with potentially missing entries.
            Must have one row per entry of ``item``.
        item : array-like(m,)
            Items for whom ratings/values are to be predicted. If 'X' passed to
            fit was a  DataFrame, must match with the entries in its 'ItemId'
            column, otherwise should match with the columns of 'X'.
            Each entry in ``item`` will be matched with the corresponding row
            of ``X``.
        U : array(m, p), CSR matrix(m, p), COO matrix(m, p), or None
            User attributes information for each row in 'X'.

        Returns
        -------
        scores : array(m,)
            Predicted ratings for the requested user-item combinations.
        """
        Xrow, Xcol, Xval, W_sp, Xarr, \
        Xcsr_p, Xcsr_i, Xcsr, \
        W_dense, Xorig, mask_take, \
        Uarr, Urow, Ucol, Uval, Ub_arr, \
        Ucsr_p, Ucsr_i, Ucsr, \
        n, m_u, m_x, p, pbin, \
        lambda_, lambda_bias, \
        l1_lambda, l1_lambda_bias = \
            self._process_transform_inputs(X=X, U=U, U_bin=None, W=None,
                                           replace_existing=True)

        c_funs = wrapper_float if self.use_float else wrapper_double
        A = c_funs.call_factors_collective_implicit_multiple(
            Xrow,
            Xcol,
            Xval,
            Xcsr_p, Xcsr_i, Xcsr,
            Uarr,
            Urow,
            Ucol,
            Uval,
            Ucsr_p, Ucsr_i, Ucsr,
            self._U_colmeans,
            self._B_pred,
            self.C_,
            self._BeTBe,
            self._BtB,
            self._BeTBeChol,
            self._CtUbias,
            n, m_u, m_x,
            self.k, self.k_user, self.k_item, self.k_main,
            lambda_, l1_lambda, self.alpha,
            self._w_main_multiplier,
            self.w_user, self.w_main,
            self.apply_log_transf,
            self.NA_as_zero_user,
            self.nonneg,
            self.nthreads
        )

        return self._predict_user_multiple(A, item, bias=None)

    @staticmethod
    def from_model_matrices(A, B, precompute=True,
                            lambda_=1e0, l1_lambda=0., nonneg=False,
                            apply_log_transf=False, alpha=1.,
                            use_float=False, nthreads=-1):
        """
        Create a CMF_implicit model object from fitted matrices

        Creates a `CMF_implicit` model object based on fitted
        latent factor matrices, which might have been obtained from a different software.
        For example, the package ``python-libmf`` has functionality for obtaining these matrices,
        but not for producing recommendations or latent factors for new users, for which
        this function can come in handy as it will turn such model into a `CMF_implicit` model which
        provides all such functionality.

        This is only available for models without side information, and does not support
        user/item mappings.

        Note
        ----
        This is a static class method, should be called like this:
            ``CMF_implicit.from_model_matrices(...)``
        (i.e. no parentheses after 'CMF_implicit')

        Parameters
        ----------
        A : array(n_users, k)
            The obtained user factors.
        B : array(n_items, k)
            The obtained item factors.
        precompute : bool
            Whether to generate pre-computed matrices which can help to speed
            up computations on new data.
        lambda_ : float or array(6,)
            Regularization parameter.
            See the documentation for ``__init__`` for details.
        l1_lambda : float or array(6,)
            Regularization parameter to apply to the L1 norm of the model matrices.
            See the documentation for ``__init__`` for details.
        nonneg : bool
            Whether to constrain the 'A' and 'B' matrices to be non-negative.
        apply_log_transf : bool
            Whether to apply a logarithm transformation on the values of 'X.
        alpha : float
            Multiplier to apply to the confidence scores given by 'X'.
        use_float : bool
            Whether to use C float type for the model parameters (typically this is
            ``np.float32``). If passing ``False``, will use C double (typically this
            is ``np.float64``). Using float types will speed up computations and
            use less memory, at the expense of reduced numerical precision.
        nthreads : int
            Number of parallel threads to use. If passing -1, will take the
            maximum available number of threads in the system.

        Returns
        -------
        model : CMF_implicit
            A ``CMF_implicit`` model object without side information, for which the usual
            prediction methods such as ``topN`` and ``topN_warm`` can be used as if
            it had been fitted through this software.
        """
        if (not isinstance(A, np.ndarray)) or (not A.flags["C_CONTIGUOUS"]):
            A = np.ascontiguousarray(A)
        if (not isinstance(B, np.ndarray)) or (not B.flags["C_CONTIGUOUS"]):
            B = np.ascontiguousarray(B)
        if (len(A.shape) != 2) or (len(B.shape) != 2):
            raise ValueError("Model matrices must be 2-dimensional.")

        k = A.shape[1]
        if (B.shape[1] != k):
            raise ValueError("Dimensions of 'A' and 'B' do not match.")
        if (not A.shape[0]) or (not B.shape[0]) or (not k):
            raise ValueError("Empty model matrices not supported.")


        dtype = ctypes.c_double if not use_float else ctypes.c_float

        new_model = CMF_implicit(k = k,
                                 lambda_ = lambda_,
                                 l1_lambda = l1_lambda,
                                 nonneg = nonneg,
                                 apply_log_transf = apply_log_transf,
                                 alpha = alpha,
                                 use_float = use_float,
                                 nthreads = nthreads)

        if (A.dtype != dtype):
            A = A.astype(dtype)
        if (B.dtype != dtype):
            B = B.astype(dtype)

        new_model.A_ = A
        new_model.B_ = B

        new_model._A_pred = A
        new_model._B_pred = B
        new_model._n_orig = B.shape[0]
        new_model.reindex_ = False

        new_model.is_fitted_ = True
        if precompute:
            new_model.force_precompute_for_predictions()
        return new_model


class _OMF_Base(_CMF):
    def factors_cold(self, U=None, U_col=None, U_val=None):
        """
        Determine user-factors from new data, given U
        
        Note
        ----
        For large-scale usage, these factors can be obtained by a
        matrix multiplication of the attributes matrix and the
        attribute (model parameter) ``C_``, plus the intercept if
        present (``C_bias_``).

        Note
        ----
        The argument 'NA_as_zero' (if available)
        is ignored here - thus, it assumes all the 'X' values are missing.

        Parameters
        ----------
        U : array(p,), or None
            User attributes in the new data (1-row only).
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_col : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_val : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.

        Returns
        -------
        factors : array(k_sec+k+k_main,)
            The user-factors as determined by the model.
        """
        assert self.is_fitted_
        if self.C_.shape[0] == 0:
            raise ValueError("Method is only available when fitting the model to user side info.")
        U, U_col, U_val, _ = self._process_new_U(U, U_col, U_val, None)

        c_funs = wrapper_float if self.use_float else wrapper_double
        a_vec = c_funs.call_factors_offsets_cold(
            U,
            U_val,
            U_col,
            self.C_,
            self.C_bias_,
            self.k,
            self.k_sec, self.k_main,
            self.w_user
        )
        return a_vec

    def predict_cold(self, items, U=None, U_col=None, U_val=None):
        """
        Predict rating/confidence given by a new user to existing items, given U

        Note
        ----
        The argument 'NA_as_zero' (if available)
        is ignored here - thus, it assumes all the 'X' values are missing.

        Parameters
        ----------
        items : array-like(n,)
            Items whose ratings are to be predicted. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'.
        U : array(p,), or None
            Attributes for the new user, in dense format.
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_col : None or array(nnz)
            Attributes for the new user, in sparse format.
            'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_val : None or array(nnz)
            Attributes for the new user, in sparse format.
            'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.

        Returns
        -------
        scores : array(n,)
            Predicted ratings for the requested items, for this user.
        """
        a_vec = self.factors_cold(U, U_col, U_val)
        return self._predict(user=None, a_vec=a_vec, a_bias=0., item=items)

    def topN_cold(self, n=10, U=None, U_col=None, U_val=None,
                  include=None, exclude=None, output_score=False):
        """
        Compute top-N highest-predicted items for a new user, given 'U'

        Note
        ----
        The argument 'NA_as_zero' (if available)
        is ignored here - thus, it assumes all the 'X' values are missing.

        Parameters
        ----------
        n : int
            Number of top-N highest-predicted results to output.
        U : array(p,), or None
            Attributes for the new user, in dense format.
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_col : None or array(nnz)
            Attributes for the new user, in sparse format.
            'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_val : None or array(nnz)
            Attributes for the new user, in sparse format.
            'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.
        include : array-like
            List of items which will be ranked. If passing this, will only
            make a ranking among these items. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include or 'exclude'.
        exclude : array-like
            List of items to exclude from the ranking. If passing this, will
            rank all the items except for these. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include or 'exclude'.
        output_score : bool
            Whether to output the scores in addition to the IDs. If passing
            'False', will return a single array with the item IDs, otherwise
            will return a tuple with the item IDs and the scores.

        Returns
        -------
        items : array(n,)
            The top-N highest predicted items for this user. If the 'X' data passed to
            fit was a DataFrame, will contain the item IDs from its column
            'ItemId', otherwise will be integers matching to the columns of 'X'.
        scores : array(n,)
            The predicted scores for the top-N items. Will only be returned
            when passing ``output_score=True``, in which case the result will
            be a tuple with these two entries.
        """
        a_vec = self.factors_cold(U, U_col, U_val)
        return self._topN(user=None, a_vec=a_vec, n=n,
                          include=include, exclude=exclude,
                          output_score=output_score)

class _OMF(_OMF_Base):
    def item_factors_cold(self, I=None, I_col=None, I_val=None):
        """
        Determine item-factors from new data, given I

        Parameters
        ----------
        I : array(q,), or None
            Attributes for the new item, in dense format.
            Should only pass one of 'I' or 'I_col'+'I_val'.
        I_col : None or array(nnz)
            Attributes for the new item, in sparse format.
            'I_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'I' or 'I_col'+'I_val'.
        I_val : None or array(nnz)
            Attributes for the new item, in sparse format.
            'I_val' should contain the values in the columns
            given by 'I_col'.
            Should only pass one of 'I' or 'I_col'+'I_val'.

        Returns
        -------
        factors : array(k_sec+k+k_main,)
            The item-factors as determined by the model.
        """
        assert self.is_fitted_
        if self.D_.shape[0] == 0:
            msg  = "Can only use this method when "
            msg += "fitting the model to item side info."
            raise ValueError(msg)

        if isinstance(self.lambda_, np.ndarray):
            lambda_ = self.lambda_[3]
        else:
            lambda_ = self.lambda_

        I, I_col, I_val, _ = self._process_new_U(U=I, U_col=I_col, U_val=I_val, U_bin=None, is_I=True)

        c_funs = wrapper_float if self.use_float else wrapper_double
        b_vec = c_funs.call_factors_offsets_cold(
            I,
            I_val,
            I_col,
            self.D_,
            self.D_bias_,
            self.k,
            self.k_sec, self.k_main,
            self.w_item
        )
        return b_vec

    def _factors_cold_multiple(self, U, is_I=False):
        assert self.is_fitted_

        letter = "U" if not is_I else "I"
        infoname = "user" if not is_I else "item"
        Mat = self.C_ if not is_I else self.D_
        MatBias = self.C_bias_ if not is_I else self.D_bias_

        if U is None:
            raise ValueError("Must pass '%s'." % letter)
        if Mat.shape[0] == 0:
            msg  = "Can only use this method when fitting the model to %s side info."
            raise ValueError(msg % infoname)
        if (U is not None) and (len(U.shape) != 2):
            raise ValueError("'%s' must be 2-dimensional." % letter)

        if isinstance(self.lambda_, np.ndarray):
            if not is_I:
                lambda_ = self.lambda_[2]
            else:
                lambda_ = self.lambda_[3]
        else:
            lambda_ = self.lambda_

        Uarr, Urow, Ucol, Uval, Ucsr_p, Ucsr_i, Ucsr, m_u, p = \
            self._process_new_U_2d(U=U, is_I=is_I, allow_csr=True)

        empty_arr = np.empty((0,0), dtype=self.dtype_)

        c_funs = wrapper_float if self.use_float else wrapper_double
        if not self._implicit:
            A, _1, _2 = c_funs.call_factors_offsets_explicit_multiple(
                            np.empty(0, dtype=ctypes.c_int),
                            np.empty(0, dtype=ctypes.c_int),
                            np.empty(0, dtype=self.dtype_),
                            np.empty(0, dtype=ctypes.c_size_t),
                            np.empty(0, dtype=ctypes.c_int),
                            np.empty(0, dtype=self.dtype_),
                            np.empty(0, dtype=self.dtype_),
                            np.empty((0,0), dtype=self.dtype_),
                            np.empty((0,0), dtype=self.dtype_),
                            Uarr,
                            Urow,
                            Ucol,
                            Uval,
                            Ucsr_p, Ucsr_i, Ucsr,
                            self.item_bias_ if not is_I else self.user_bias_,
                            self._B_pred if not is_I else self._A_pred,
                            self._B_plus_bias if not is_I else empty_arr,
                            Mat,
                            MatBias,
                            self._TransBtBinvBt if not is_I else empty_arr,
                            self._BtB if not is_I else empty_arr,
                            glob_mean,
                            m_u, 0,
                            self.k, self.k_sec, self.k_main,
                            lambda_, lambda_,
                            self.w_user if not is_I else self.w_item,
                            self.user_bias if not is_I else self.item_bias,
                            0, 0,
                            self.nthreads
                        )
        else:
            A, _ = c_funs.call_factors_offsets_implicit_multiple(
                        np.empty(0, dtype=ctypes.c_int),
                        np.empty(0, dtype=ctypes.c_int),
                        np.empty(0, dtype=self.dtype_),
                        np.empty(0, dtype=ctypes.c_size_t),
                        np.empty(0, dtype=ctypes.c_int),
                        np.empty(0, dtype=self.dtype_),
                        Uarr,
                        Urow,
                        Ucol,
                        Uval,
                        Ucsr_p, Ucsr_i, Ucsr,
                        self._B_pred if not is_I else self._A_pred,
                        Mat,
                        MatBias,
                        self._BtB if not is_I else empty_arr,
                        m_u, 0,
                        self.k,
                        lambda_, self.alpha,
                        self.apply_log_transf,
                        0,
                        self.nthreads
                    )
        return A

    def predict_cold_multiple(self, item, U):
        """
        Predict rating/confidence given by new users to existing items, given U

        Parameters
        ----------
        item : array-like(m,)
            Items for which ratings/values are to be predicted. If 'X' passed to
            fit was a  DataFrame, must match with the entries in its 'ItemId'
            column, otherwise should match with the columns of 'X'.
        U : array(m, p), CSR matrix(m, q), or COO matrix(m, q)
            Attributes for the users for which to predict ratings/values.
            Data frames with 'UserId' column are not supported.
            Must have one row per entry
            in ``item``.

        Returns
        -------
        scores : array(m,)
            Predicted ratings for the requested user-item combinations.
        """
        A = self._factors_cold_multiple(U=U, U_bin=None, is_I=False)
        return self._predict_user_multiple(A, item)

    def predict_new(self, user, I):
        """
        Predict rating given by existing users to new items, given I

        Parameters
        ----------
        user : array-like(n,)
            Users for whom ratings/values are to be predicted. If 'X' passed to
            fit was a  DataFrame, must match with the entries in its 'UserId'
            column, otherwise should match with the rows of 'X'.
        I : array(n, q), or COO matrix(n, q)
            Attributes for the items for which to predict ratings/values.
            Data frames with 'ItemId' column are not supported.
            Must have one row per entry in ``user``.

        Returns
        -------
        scores : array(n,)
            Predicted ratings for the requested user-item combinations.
        """
        B = self._factors_cold_multiple(U=I, is_I=True)
        return self._predict_new(user, B)

    def topN_new(self, user, I, n=10, output_score=False):
        """
        Rank top-N highest-predicted items for an existing user, given 'I'

        Parameters
        ----------
        user : int or obj
            User for which to rank the items. If 'X' passed to 'fit' was a
            data frame, must match with entries in its 'UserId' column,
            otherwise should match with the rows on 'X'.
        I : array(m, q), or COO matrix(m, q)
            Attributes for the items to rank. Data frames with 'ItemId'
            column are not supported.
        n : int
            Number of top-N highest-predicted results to output. Must be
            less or equal than the number of rows in I.
        output_score : bool
            Whether to output the scores in addition to the IDs. If passing
            'False', will return a single array with the item IDs, otherwise
            will return a tuple with the item IDs and the scores.

        Returns
        -------
        items : array(n,)
            The top-N highest predicted items for this user, as integers
            matching to the rows of 'I'.
        scores : array(n,)
            The predicted scores for the top-N items. Will only be returned
            when passing ``output_score=True``, in which case the result will
            be a tuple with these two entries.
        """
        B = self._factors_cold_multiple(I=I, is_I=True)
        return self._topN(user=user, B=B, n=n, output_score=output_score)

class OMF_explicit(_OMF):
    """
    Offsets model for explicit-feedback data

    Tries to approximate the 'X' ratings matrix using the user side information
    'U' and item side information 'I' by a formula as follows:
        X ~ (A + U*C) * t(B + I*D)

    Note
    ----
    This model is meant to be fit to ratings data with side info about either
    users or items. If there is side info about both, it's better to use
    the content-based model instead.

    Note
    ----
    This model is meant for cold-start predictions (that is, based on side
    information alone). It is extremely unlikely to bring improvements compared
    to situations in which the classical model is able to make predictions.

    Note
    ----
    The ALS method works by first fitting a model with no side info and then
    reconstructing the parameters by least squares approximations, so when
    making warm-start predictions, the results will be exactly the same as if
    not using any side information (user/item attributes). The ALS procedure
    for this model was implemented for experimentation purposes only, and it's
    recommended to use L-BFGS instead.

    Note
    ----
    It's advised to experiment with tuning the maximum number of L-BFGS iterations
    and stopping earlier. Be aware that this model requires a lot more iterations
    to reach convergence compared to the classic and the collective models.

    Note
    ----
    The model optimization objective will not scale any of its terms according
    to number of entries, so hyperparameters such as ``lambda_`` will require
    more tuning than in other software and trying out values over a wider range.

    Parameters
    ----------
    k : int
        Number of latent factors to use (dimensionality of the low-rank
        factorization), which will have a free component and an attribute-dependent
        component. Other additional separate factors can be specified through
        ``k_sec`` and ``k_main``.
        Optionally, this parameter might be set to zero while setting ``k_sec``
        and ``k_main`` for a different type of model.
        Typical values are 30 to 100.
    lambda_ : float or array(6,)
        Regularization parameter. Can also use different regularization for each
        matrix, in which case it should be an array with 6 entries, corresponding,
        in this order, to: user_bias, item_bias, A, B, C, D.
        The attribute biases will have the same regularization as the matrices
        to which they apply (C and D).
        Note that the default
        value for ``lambda_`` here is much higher than in other software, and that
        the loss/objective function is not divided by the number of entries.
        For example, a good value for the MovieLens10M would be ``lambda_=35.``.
        Typical values are 10^-2 to 10^2.
        Passing different regularization for each matrix is not supported with
        ``method='als'``.
    method : str, one of "lbfgs" or "als"
        Optimization method used to fit the model. If passing ``'lbfgs'``, will
        fit it through a gradient-based approach using an L-BFGS optimizer.
        If passing ``'als'``, will first obtain the solution ignoring the side
        information using an alternating least-squares procedure (the classical
        model described in other papers), then reconstruct the model matrices
        by a least-squares approximation. The ALS approach was implemented for
        experimentation purposes only and is not recommended.
    use_cg : bool
        In the ALS method, whether to use a conjugate gradient method to solve
        the closed-form least squares problems. This is a faster and more
        memory-efficient alternative than the default Cholesky solver, but less
        exact, less numerically stable, and will require slightly more ALS
        iterations (``niter``) to reach a good optimum.
        In general, better results are achieved with ``use_cg=False``.
        Note that, if using this method, calculations after fitting which involve
        new data such as ``factors_warm``,  might produce slightly different
        results from the factors obtained from calling ``fit`` with the same data,
        due to differences in numerical precision. A workaround for this issue
        (factors on new data that might differ slightly) is to use
        ``finalize_chol=True``.
        Even if passing "True" here, will use the Cholesky method in cases in which
        it is faster (e.g. dense matrices with no missing values),
        and will not use the conjugate gradient method on new data.
        Ignored when passing ``method="lbfgs"``.
    user_bias : bool
        Whether to add user biases (intercepts) to the model.
    item_bias : bool
        Whether to add item biases (intercepts) to the model. Be aware that using
        item biases with low regularization for them will tend to favor items
        with high average ratings regardless of the number of ratings the item
        has received.
    center : bool
        Whether to center the "X" data by subtracting the mean value. For recommender
        systems, it's highly recommended to pass "True" here, the more so if the
        model has user and/or item biases.
    k_sec : int
        Number of factors in the factorizing matrices which are determined
        exclusively from user/item attributes. These will be at the beginning
        of the C and D matrices once the model is fit. If there are no attributes
        for a given matrix (user/item), then that matrix will have an extra
        ``k_sec`` factors (e.g. if passing user side info but not item side info,
        then the B matrix will have an extra ``k_sec`` factors). Will be counted
        in addition to those already set by ``k``. Not supported when
        using ``method='als'``.
        
        For a different model having only ``k_sec`` with ``k=0`` and ``k_main=0``,
        see the ``ContentBased`` class.
    k_main : int
        Number of factors in the factorizing matrices which are determined
        without any user/item attributes. These will be at the end of the
        A and B matrices once the model is fit. Will be counted in addition to
        those already set by ``k``. Not supported when using ``method='als'``.
    add_intercepts : bool
        Whether to add intercepts/biases to the user/item attribute matrices.
    w_user : float
        Multiplier for the effect of the attributes contribution to the
        factorizing matrix A (that is, Am = A + w_user*U*C). Passing values
        larger than 1 has the effect of giving less freedoom to the free offset
        term.
    w_item : float
        Multiplier for the effect of the attributes contribution to the
        factorizing matrix B (that is, Bm = B + w_item*I*D). Passing values
        larger than 1 has the effect of giving less freedoom to the free offset
        term.
    maxiter : int
        Maximum L-BFGS iterations to perform. The procedure will halt if it
        has not converged after this number of updates. Note that, compared to
        the collective model, more iterations will be required for converge
        here. Using higher regularization values might also decrease the number
        of required iterations. Pass zero for no L-BFGS iterations limit.
        If the procedure is spending thousands of iterations
        without any significant decrease in the loss function or gradient norm,
        it's highly likely that the regularization is too low.
        Ignored when passing ``method='als'``.
    niter : int
        Number of alternating least-squares iterations to perform. Note that
        one iteration denotes an update round for all the matrices rather than
        an update of a single matrix. In general, the more iterations, the better
        the end result. Ignored when passing ``method='lbfgs'``.
        Typical values are 6 to 30.
    parallelize : str, "separate" or "single"
        How to parallelize gradient calculations when using more than one
        thread with ``method='lbfgs'``. Passing ``'separate'`` will iterate
        over the data twice - first
        by rows and then by columns, letting each thread calculate results
        for each row and column, whereas passing ``'single'`` will iterate over
        the data only once, and then sum the obtained results from each thread.
        Passing ``'separate'`` is much more memory-efficient and less prone to
        irreproducibility of random seeds, but might be slower for typical
        use-cases. Ignored when passing ``nthreads=1``, or ``method='als'``,
        or when compiling without OpenMP support.parallelize : str, "separate" or "single"
        How to parallelize gradient calculations when using more than one
        thread. Passing ``'separate'`` will iterate over the data twice - first
        by rows and then by columns, letting each thread calculate results
        for each row and column, whereas passing ``'single'`` will iterate over
        the data only once, and then sum the obtained results from each thread.
        Passing ``'separate'`` is much more memory-efficient and less prone to
        irreproducibility of random seeds, but might be slower for typical
        use-cases. Ignored when passing ``nthreads=1`` or compiling without
        OpenMP support.
    corr_pairs : int
        Number of correction pairs to use for the L-BFGS optimization routine.
        Recommended values are between 3 and 7. Note that higher values
        translate into higher memory requirements. Ignored when passing
        ``method='als'``.
    max_cg_steps : int
        Maximum number of conjugate gradient iterations to perform in an ALS round.
        Ignored when passing ``use_cg=False`` or ``method="lbfgs"``.
    finalize_chol : bool
        When passing ``use_cg=True`` and ``method="als"``, whether to perform the last iteration with
        the Cholesky solver. This will make it slower, but will avoid the issue
        of potential mismatches between the result from ``fit`` and calls to
        ``factors_warm`` or similar with the same data.
    NA_as_zero : bool
        Whether to take missing entries in the 'X' matrix as zeros (only
        when the 'X' matrix is passed as sparse COO matrix or DataFrame)
        instead of ignoring them. Note that this is a different model from the
        implicit-feedback version with weighted entries, and it's a much faster
        model to fit. Be aware that this option will be ignored later when
        predicting on new data - that is, non-present values will be treated
        as missing.
        If passing this option, be aware that the defaults are also to
        perform mean centering and add user/item biases, which might
        be undesirable to have together with this option.
    use_float : bool
        Whether to use C float type for the model parameters (typically this is
        ``np.float32``). If passing ``False``, will use C double (typically this
        is ``np.float64``). Using float types will speed up computations and
        use less memory, at the expense of reduced numerical precision.
    random_state : int, RandomState, or Generator
        Seed used to initialize parameters at random. If passing a NumPy
        RandomState or Generator, will use it to draw a random integer. Note
        however that, if using more than one thread, results might not be
        100% reproducible with ``method='lbfgs'`` due to round-off errors
        in parallelized aggregations.
    verbose : bool
        Whether to print informational messages about the optimization
        routine used to fit the model. Note that, if running this from a
        Jupyter notebook, these messages will be printed in the console,
        not in the notebook itself. Be aware that, if passing 'False' and
        ``method='lbfgs'``, the optimization routine will not respond to
        interrupt signals.
    print_every : int
        Print L-BFGS convergence messages every n-iterations. Ignored
        when passing ``verbose=False`` or ``method='als'``.
    handle_interrupt : bool
        When receiving an interrupt signal, whether the model should stop
        early and leave a usable object with the parameters obtained up
        to the point when it was interrupted (when passing 'True'), or
        raise an interrupt exception without producing a fitted model object
        (when passing 'False').
    produce_dicts : bool
        Whether to produce Python dicts from the mappings between user/item
        IDs passed to 'fit' and the internal IDs used by the class. Having
        these dicts might speed up some computations such as 'predict',
        but it will add some extra overhead at the time of fitting the model
        and extra memory usage. Ignored when passing the data as matrices
        and arrays instead of data frames.
    copy_data : bool
        Whether to make copies of the input data that is passed to this
        object's methods (``fit``, ``predict``, etc.), in order to avoid
        modifying such data in-place. Passing ``False`` will save some
        computation time and memory usage.
    nthreads : int
        Number of parallel threads to use. If passing -1, will take the
        maximum available number of threads in the system.

    Attributes
    ----------
    is_fitted_ : bool
        Whether the model has been fitted to data.
    reindex_ : bool
        Whether the IDs passed to 'fit' were reindexed internally
        (this will only happen when passing data frames to 'fit').
    user_mapping_ : array(m,) or array(0,)
        Correspondence of internal user (row) IDs to IDs in the data
        passed to 'fit'. Will only be non-empty when passing a data frame
        as input to 'X'.
    item_mapping_ : array(n,) or array(0,)
        Correspondence of internal item (column) IDs to IDs in the data
        passed to 'fit'. Will only be non-empty when passing a data frame
        as input to 'X'.
    user_dict_ : dict
        Python dict version of ``user_mapping_``. Only filled-in when
        passing ``produce_dicts=True`` and when passing data frames to 'fit'.
    item_dict_ : dict
        Python dict version of ``item_mapping_``. Only filled-in when
        passing ``produce_dicts=True`` and when passing data frames to 'fit'.
    glob_mean_ : float
        The global mean of the non-missing entries in 'X' passed to 'fit'.
    user_bias_ : array(m,), or array(0,)
        The obtained biases for each user (row in the 'X' matrix).
        If passing ``user_bias=False``, this array
        will be empty.
    item_bias_ : array(n,)
        The obtained biases for each item (column in the 'X' matrix).
        If passing ``item_bias=False``, this array
        will be empty.
    A_ : array(m, k+k_main) or array(m, k_sec+k+k_main)
        The free offset for the user-factors obtained from user attributes
        and matrix C_. If passing ``k_sec>0`` and no user side information,
        this matrix will have an extra ``k_sec`` columns at the beginning.
    B_ : array(n, k+k_main) or array(m, k_sec+k+k_main)
        The free offset for the item-factors obtained from item attributes
        and matrix D_. If passing ``k_sec>0`` and no item side information,
        this matrix will have an extra ``k_sec`` columns at the beginning.
    C_ : array(p, k_sec+k)
        The obtained coefficients for the user attributes.
    D_ : array(q, k_sec+k)
        The obtained coefficients for the item attributes.
    C_bias_ : array(k_sec+k)
        The intercepts/biases for the C matrix.
    D_bias_ : array(k_sec+k)
        The intercepts/biases for the D matrix.
    nfev_ : int
        Number of function and gradient evaluations performed during the
        L-BFGS optimization procedure.
    nupd_ : int
        Number of L-BFGS updates performed during the optimization procedure.

    References
    ----------
    .. [1] Cortes, David.
           "Cold-start recommendations in Collective Matrix Factorization."
           arXiv preprint arXiv:1809.00366 (2018).
    """
    def __init__(self, k=50, lambda_=1e1, method="lbfgs", use_cg=True,
                 user_bias=True, item_bias=True, center=True, k_sec=0, k_main=0,
                 add_intercepts=True, w_user=1., w_item=1.,
                 maxiter=10000, niter=10, parallelize="separate", corr_pairs=7,
                 max_cg_steps=3, finalize_chol=True,
                 NA_as_zero=False, use_float=False,
                 random_state=1, verbose=True, print_every=100,
                 produce_dicts=False, handle_interrupt=True,
                 copy_data=True, nthreads=-1):
        assert k>0 or k_sec>0 or k_main>0
        self._take_params(implicit=False, alpha=0., downweight=False,
                          k=1, lambda_=lambda_, method=method,
                          use_cg=use_cg, max_cg_steps=max_cg_steps,
                          finalize_chol=finalize_chol,
                          user_bias=user_bias, item_bias=item_bias,
                          center=center,
                          k_user=0, k_item=0, k_main=0,
                          w_main=1., w_user=w_user, w_item=w_item,
                          maxiter=maxiter, niter=niter, parallelize=parallelize,
                          corr_pairs=corr_pairs,
                          NA_as_zero=NA_as_zero,
                          NA_as_zero_user=False, NA_as_zero_item=False,
                          precompute_for_predictions=True,
                          use_float=use_float,
                          random_state=random_state, init="normal",
                          verbose=verbose, print_every=print_every,
                          handle_interrupt=handle_interrupt,
                          produce_dicts=produce_dicts, copy_data=copy_data,
                          nthreads=nthreads)
        self.k = int(k)
        self._take_params_offsets(k_sec=k_sec, k_main=k_main,
                                  add_intercepts=add_intercepts)
        if self.method == "als":
            msg  = "This model was implemented for experimentation purposes."
            msg += " Performance is likely to be bad. Be warned."
            warnings.warn(msg)

    def __str__(self):
        msg  = "Offsets factorization model\n"
        msg += "(explicit-feedback variant)\n\n"
        if not self.is_fitted_:
            msg += "Model has not been fitted to data.\n"
        return msg

    def get_params(self, deep=None):
        return {
            "k" : self.k, "lambda_" : self.lambda_, "method" : self.method,
            "user_bias" : self.user_bias, "item_bias" : self.item_bias,
            "k_sec" : self.k_sec, "k_main" : self.k_main,
            "add_intercepts" : self.add_intercepts,
            "w_user" : self.w_user, "w_item" : self.w_item,
            "maxiter" : self.maxiter, "niter" : self.niter,
            "parallelize" : self.parallelize, "corr_pairs" : self.corr_pairs,
            "NA_as_zero" : self.NA_as_zero, "use_float" : self.use_float,
            "use_cg" : self.use_cg, "random_state" : self.random_state,
            "nthreads" : self.nthreads
        }

    def fit(self, X, U=None, I=None, W=None):
        """
        Fit model to explicit-feedback data and user/item attributes

        Note
        ----
        None of the side info inputs should have missing values. If passing side
        information 'U' and/or 'I', all entries (users/items) must be present
        in both the main matrix and the side info matrix.

        Note
        ----
        In order to avoid potential decimal differences in the factors obtained
        when fitting the model and when calling the prediction functions on
        new data, when the data is sparse, it's necessary to sort it beforehand
        by columns and also pass the data data with indices sorted (by column)
        to the prediction functions.
        
        Parameters
        ----------
        X : DataFrame(nnz, 3), DataFrame(nnz, 4), array(m, n), or sparse COO(m, n)
            Matrix to factorize (e.g. ratings). Can be passed as a SciPy
            sparse COO matrix (recommended), as a dense NumPy array, or
            as a Pandas DataFrame, in which case it should contain the
            following columns: 'UserId', 'ItemId', and 'Rating'.
            If passing a NumPy array, missing (unobserved) entries should 
            have value ``np.nan``.
            Might additionally have a column 'Weight'. If passing a DataFrame,
            the IDs will be internally remapped.
            If passing sparse 'U' or sparse 'I', 'X' cannot be passed as
            a DataFrame.
        U : array(m, p), COO(m, p), DataFrame(m, p+1), or None
            User attributes information. If 'X' is a DataFrame, should also
            be a DataFrame, containing column 'UserId'. If 'U' is sparse,
            'X' should be passed as a sparse COO matrix or as a dense NumPy array.
            Should not contain any missing values.
        I : array(n, q), COO(n, q), DataFrame(n, q+1), or None
            Item attributes information. If 'X' is a DataFrame, should also
            be a DataFrame, containing column 'ItemId'. If 'I' is sparse,
            'X' should be passed as a sparse COO matrix or as a dense NumPy array.
            Should not contain any missing values.
        W : None, array(nnz,), or array(m, n)
            Observation weights. Must have the same shape as 'X' - that is,
            if 'X' is a sparse COO matrix, must be a 1-d array with the same
            number of non-zero entries as 'X.data', if 'X' is a 2-d array,
            'W' must also be a 2-d array.

        Returns
        -------
        self

        """
        if self.method == "als":
            if "coo_matrix" in [U.__class__.__name__, I.__class__.__name__]:
                msg  = "Cannot pass user/item side info in sparse format"
                msg += " when using method='als'."
                raise ValueError(msg)
        return self._fit_common(X, U=U, I=I, U_bin=None, I_bin=None, W=W,
                                enforce_same_shape=True)

    def _fit(self, Xrow, Xcol, Xval, W_sp, Xarr, W_dense,
             Uarr, Urow, Ucol, Uval, Ub_arr,
             Iarr, Irow, Icol, Ival, Ib_arr,
             m, n, m_u, n_i, p, q,
             m_ub, n_ib, pbin, qbin):
        self._A_pred = np.empty((0,0), dtype=self.dtype_)
        self._B_pred = np.empty((0,0), dtype=self.dtype_)

        c_funs = wrapper_float if self.use_float else wrapper_double
        if self.method == "lbfgs":
            self.glob_mean_, self._A_pred, self._B_pred, values, self.nupd_, self.nfev_, self._B_plus_bias = \
                c_funs.call_fit_offsets_explicit_lbfgs_internal(
                    Xrow,
                    Xcol,
                    Xval,
                    W_sp,
                    Xarr,
                    W_dense,
                    Uarr,
                    Urow,
                    Ucol,
                    Uval,
                    Iarr,
                    Irow,
                    Icol,
                    Ival,
                    m, n, p, q,
                    self.k, self.k_sec, self.k_main,
                    self.w_user, self.w_item,
                    self.user_bias, self.item_bias, self.center,
                    self.add_intercepts,
                    self.lambda_ if isinstance(self.lambda_, float) else 0.,
                    self.lambda_ if isinstance(self.lambda_, np.ndarray) else np.empty(0, dtype=self.dtype_),
                    self.verbose, self.print_every,
                    self.corr_pairs, self.maxiter,
                    self.nthreads, self.parallelize != "separate",
                    self.random_state,
                    self.handle_interrupt
            )
            self.user_bias_, self.item_bias_, self.A_, self.B_, self.C_, self.D_, \
            self.C_bias_, self.D_bias_ = \
                c_funs.unpack_values_lbfgs_offsets(
                    values,
                    self.user_bias, self.item_bias,
                    self.k, self.k_sec, self.k_main,
                    m, n, p, q,
                    self.add_intercepts
                )
            if (not Uarr.shape[0]) and (not Uval.shape[0]):
                self._A_pred = self.A_
            if (not Iarr.shape[0]) and (not Ival.shape[0]):
                self._B_pred = self.B_
            
            if self.precompute_for_predictions:
                if isinstance(self.lambda_, np.ndarray):
                    lambda_ = self.lambda_[2]
                    lambda_bias = self.lambda_[0]
                else:
                    lambda_ = self.lambda_
                    lambda_bias = self.lambda_

                self.is_fitted_ = True
                _1, self._BtB, self._TransBtBinvBt, _2, _3, _4, _5, _6, _7 = \
                    c_funs.precompute_matrices_collective_explicit(
                        self._B_pred,
                        np.empty((0,0), dtype=self.dtype_),
                        np.empty((0,0), dtype=self.dtype_),
                        np.empty(0, dtype=self.dtype_),
                        np.empty(0, dtype=self.dtype_),
                        self.user_bias, False,
                        self._B_pred.shape[0],
                        self.k_sec+self.k+self.k_main,
                        0, 0, 0,
                        lambda_, lambda_bias,
                        1., 1., 1.,
                        glob_mean = 0.,
                        scale_lam = 0, scale_lam_sideinfo = 0,
                        scale_bias_const = 0, scaling_biasA = 0.,
                        NA_as_zero_X = 0,
                        NA_as_zero_U = 0,
                        nonneg = 0,
                        include_all_X = 1
                    )

        else:
            self.user_bias_, self.item_bias_, self.A_, self.B_, self.C_, self.D_, \
            self._A_pred, self._B_pred, self.glob_mean_, \
            self._B_plus_bias, self._BtB, self._TransBtBinvBt = \
                c_funs.call_fit_offsets_explicit_als(
                    Xrow,
                    Xcol,
                    Xval,
                    W_sp,
                    Xarr,
                    W_dense,
                    Uarr,
                    Iarr,
                    self.NA_as_zero,
                    m, n, p, q,
                    self.k,
                    self.user_bias, self.item_bias, self.center,
                    self.add_intercepts,
                    self.lambda_,
                    self.verbose, self.nthreads,
                    self.use_cg, self.max_cg_steps,
                    self.finalize_chol,
                    self.random_state, self.niter,
                    self.handle_interrupt,
                    precompute_for_predictions=self.precompute_for_predictions
                )
        
        self._n_orig = self._B_pred.shape[0]
        self.is_fitted_ = True
        return self

    def factors_warm(self, X=None, X_col=None, X_val=None, W=None,
                     U=None, U_col=None, U_val=None,
                     return_bias=False, return_raw_A=False, exact=False):
        """
        Determine user latent factors based on new ratings data

        Note
        ----
        The argument 'NA_as_zero' is ignored here.

        Parameters
        ----------
        X : array(n,) or None
            Observed new 'X' data for a given user, in dense format.
            Non-observed entries should have value ``np.nan``.
        X_col : array(nnz,) or None
            Observed new 'X' data for a given user, in sparse format.
            'X_col' should contain the column indices (items) of the observed 
            entries. If 'X' passed to 'fit' was a data frame, should have
            entries from 'ItemId' column, otherwise should have column numbers
            (starting at zero).
        X_val : array(nnz,) or None
            Observed new 'X' data for a given user, in sparse format.
            'X_val' should contain the values in the columns/items given by
            'X_col'.
        W : array(nnz,), array(n,), or None
            Weights for the observed entries in 'X'. If passed, should
            have the same shape as 'X' - that is, if 'X' is passed as
            a dense array, should have 'n' entries, otherwise should
            have 'nnz' entries.
        U : array(p,), or None
            User attributes in the new data (1-row only).
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_col : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_val : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.
        return_bias : bool
            Whether to return also the user bias determined by the model
            given the data in 'X'. If passing 'False', will return an array
            with the factors. If passing 'True', will return a tuple in which
            the first entry will be an array with the factors, and the second
            entry will be the estimated bias.
        return_raw_A : bool
            Whether to return the raw A factors (the free offset), or the
            factors used in the factorization, to which the attributes
            component has been added.
        exact : bool
            Whether to calculate "A" and "Am" with the regularization applied
            to "A" instead of to "Am". This is usually a slower procedure.
            Only relevant when passing "X" data.

        Returns
        -------
        factors : array(k_sec+k+k_main,) or array(k+k_main,)
            User factors as determined from the data in 'X'.
        bias : float
            User bias as determined from the data in 'X'. Only returned if
            passing ``return_bias=True``.
        """
        if (U is None) and (U_col is None) and (U_val is None) \
            and (self.k_sec) and (self.C_.shape[0] > 0):
            warnings.warn("Method not reliable with k_sec>0 and no user info.")
        if (self.k == 0) and (self.k_sec == 0) \
            and (X is None) and (X_val is None) \
            and (self.C_.shape[0] > 0):
            msg  = "Method not available without user side info "
            msg += "when using k=0 and k_main=0."
            raise ValueError(msg)
        if (self.k_sec == 0) and (self.k_main == 0) \
            and (self.w_user == 0) and (self.w_item == 0):
            if (U is not None) or (U_col is not None) or (U_val is not None):
                msg  = "User side info is not used for warm-start predictions "
                msg += "with this combination of hyperparameters."
                warnings.warn(msg)
            outp = self._factors_warm_common(X=X, X_col=X_col, X_val=X_val, W=W,
                                             U=None, U_bin=None,
                                             U_col=None, U_val=None,
                                             return_bias=return_bias,
                                             output_a=return_raw_A)
        else:
            outp = self._factors_warm_common(X=X, X_col=X_col, X_val=X_val, W=W,
                                             U=U, U_bin=None, U_col=U_col, U_val=U_val,
                                             return_bias=return_bias,
                                             output_a=return_raw_A)
        a_bias = 0.
        if return_bias:
            a_vec, a_pred, a_bias = outp
        else:
            a_vec, a_pred = outp
        outp_a = a_vec if return_raw_A else a_pred
        if return_bias:
            return outp_a, a_bias
        else:
            return outp_a

    def _factors_warm(self, X, W_dense, X_val, X_col, W_sp,
                      U, U_val, U_col, U_bin, return_bias,
                      exact, output_a):
        if isinstance(self.lambda_, np.ndarray):
            lambda_ = self.lambda_[2]
            lambda_bias = self.lambda_[0]
        else:
            lambda_ = self.lambda_
            lambda_bias = self.lambda_

        c_funs = wrapper_float if self.use_float else wrapper_double
        a_bias, a_pred, a_vec = c_funs.call_factors_offsets_warm_explicit(
            X,
            W_dense,
            X_val,
            X_col,
            W_sp,
            U,
            U_val,
            U_col,
            self.item_bias_,
            self._B_pred,
            self._B_plus_bias,
            self.C_,
            self.C_bias_,
            self._TransBtBinvBt,
            self._BtB,
            self.glob_mean_,
            self.k, self.k_sec, self.k_main,
            lambda_, lambda_bias,
            self.w_user,
            self.user_bias,
            exact, output_a
        )

        if return_bias:
            return a_vec, a_pred, a_bias
        else:
            return a_vec, a_pred

    def predict_warm(self, items, X=None, X_col=None, X_val=None, W=None,
                     U=None, U_col=None, U_val=None):
        """
        Predict ratings for existing items, for a new user, given 'X'

        Note
        ----
        The argument 'NA_as_zero' is ignored here.

        Parameters
        ----------
        items : array-like(n,)
            Items whose ratings are to be predicted. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'.
        X : array(n,) or None
            Observed 'X' data for the new user, in dense format.
            Non-observed entries should have value ``np.nan``.
        X_col : array(nnz,) or None
            Observed 'X' data for the new user, in sparse format.
            'X_col' should contain the column indices (items) of the observed 
            entries. If 'X' passed to 'fit' was a data frame, should have
            entries from 'ItemId' column, otherwise should have column numbers
            (starting at zero).
        X_val : array(nnz,) or None
            Observed 'X' data for the new user, in sparse format.
            'X_val' should contain the values in the columns/items given by
            'X_col'.
        W : array(nnz,), array(n,), or None
            Weights for the observed entries in 'X'. If passed, should
            have the same shape as 'X' - that is, if 'X' is passed as
            a dense array, should have 'n' entries, otherwise should
            have 'nnz' entries.
        U : array(p,), or None
            Attributes for the new user, in dense format.
            Should only pass one of 'U' or 'U_col'+'U_val'.
            Not used when using ``k_sec=0``.
        U_col : None or array(nnz)
            Attributes for the new user, in sparse format.
            'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
            Not used when using ``k_sec=0``.
        U_val : None or array(nnz)
            Attributes for the new user, in sparse format.
            'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.
            Not used when using ``k_sec=0``.
        
        Returns
        -------
        scores : array(n,)
            Predicted values for the requested items for a user defined by
            the given values of 'X' in 'X_col' and 'X_val'.
        """
        a_vec, a_bias = self.factors_warm(X=X, X_col=X_col, X_val=X_val, W=W,
                                          U=U, U_col=U_col, U_val=U_val,
                                          return_bias=True)
        return self._predict(user=None, a_vec=a_vec, a_bias=a_bias, item=items)

    def predict_warm_multiple(self, X, item, U=None, W=None):
        """
        Predict ratings for existing items, for new users, given 'X'

        Note
        ----
        The argument 'NA_as_zero' is ignored here.

        Parameters
        ----------
        X : array(m, n), CSR matrix(m, n) , or COO matrix(m, n)
            New 'X' data with potentially missing entries.
            Missing entries should have value ``np.nan`` when passing a dense
            array.
            Must have one row per entry of ``item``.
        item : array-like(m,)
            Items for whom ratings/values are to be predicted. If 'X' passed to
            fit was a  DataFrame, must match with the entries in its 'ItemId'
            column, otherwise should match with the columns of 'X'.
            Each entry in ``item`` will be matched with the corresponding row
            of ``X``.
        U : array(m, p), CSR matrix(m, p), COO matrix(m, p), or None
            User attributes information for each row in 'X'.
            Should not contain any missing values.
        W : array(m, n), array(nnz,), or None
            Observation weights. Must have the same shape as 'X' - that is,
            if 'X' is a sparse COO matrix, must be a 1-d array with the same
            number of non-zero entries as 'X.data', if 'X' is a 2-d array,
            'W' must also be a 2-d array.

        Returns
        -------
        scores : array(m,)
            Predicted ratings for the requested user-item combinations.
        """
        if (self.k_sec == 0) and (self.k_main == 0) \
            and (self.w_user == 0) and (self.w_item == 0):
            if U is not None:
                msg  = "User side info is not used for warm-start predictions "
                msg += "with this combination of hyperparameters."
                warnings.warn(msg) 
                U = None
        
        Xrow, Xcol, Xval, W_sp, Xarr, \
        Xcsr_p, Xcsr_i, Xcsr, \
        W_dense, Xorig, mask_take, \
        Uarr, Urow, Ucol, Uval, _1, Ucsr_p, Ucsr_i, Ucsr, \
        n, m_u, m_x, p, _2, \
        lambda_, lambda_bias, \
        l1_lambda, l1_lambda_bias = \
            self._process_transform_inputs(X=X, U=U, U_bin=None, W=W,
                                           replace_existing=True)

        c_funs = wrapper_float if self.use_float else wrapper_double
        A, A_bias, _ = c_funs.call_factors_offsets_explicit_multiple(
            Xrow,
            Xcol,
            Xval,
            Xcsr_p, Xcsr_i, Xcsr,
            W_sp,
            Xarr,
            W_dense,
            Uarr,
            Urow,
            Ucol,
            Uval,
            Ucsr_p, Ucsr_i, Ucsr,
            self.item_bias_,
            self._B_pred,
            self._B_plus_bias,
            self.C_,
            self.C_bias_,
            self._TransBtBinvBt,
            self._BtB,
            self.glob_mean_,
            m_x, n,
            self.k, self.k_sec, self.k_main,
            lambda_, lambda_bias,
            self.w_user,
            self.user_bias,
            0, 0,
            self.nthreads
        )

        return self._predict_user_multiple(A, item, bias=A_bias)

    def topN_warm(self, n=10, X=None, X_col=None, X_val=None, W=None,
                  U=None, U_col=None, U_val=None,
                  include=None, exclude=None, output_score=False):
        """
        Compute top-N highest-predicted items for a new user, given 'X'

        Note
        ----
        The argument 'NA_as_zero' is ignored here.

        Parameters
        ----------
        n : int
            Number of top-N highest-predicted results to output.
        X : array(n,) or None
            Observed 'X' data for the new user, in dense format.
            Non-observed entries should have value ``np.nan``.
            Should only pass one of 'X' or 'X_col'+'X_val'.
        X_col : array(nnz,) or None
            Observed 'X' data for the new user, in sparse format.
            'X_col' should contain the column indices (items) of the observed 
            entries. If 'X' passed to 'fit' was a data frame, should have
            entries from 'ItemId' column, otherwise should have column numbers
            (starting at zero).
            Should only pass one of 'X' or 'X_col'+'X_val'.
        X_val : array(nnz,) or None
            Observed 'X' data for the new user, in sparse format.
            'X_val' should contain the values in the columns/items given by
            'X_col'.
            Should only pass one of 'X' or 'X_col'+'X_val'.
        W : array(nnz,), array(n,), or None
            Weights for the observed entries in 'X'. If passed, should
            have the same shape as 'X' - that is, if 'X' is passed as
            a dense array, should have 'n' entries, otherwise should
            have 'nnz' entries.
        U : array(p,), or None
            Attributes for the new user, in dense format.
            Should only pass one of 'U' or 'U_col'+'U_val'.
            Not used when using ``k_sec=0``.
        U_col : None or array(nnz)
            Attributes for the new user, in sparse format.
            'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
            Not used when using ``k_sec=0``.
        U_val : None or array(nnz)
            Attributes for the new user, in sparse format.
            'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.
            Not used when using ``k_sec=0``.
        include : array-like
            List of items which will be ranked. If passing this, will only
            make a ranking among these items. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include or 'exclude'.
        exclude : array-like
            List of items to exclude from the ranking. If passing this, will
            rank all the items except for these. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include or 'exclude'.
        output_score : bool
            Whether to output the scores in addition to the IDs. If passing
            'False', will return a single array with the item IDs, otherwise
            will return a tuple with the item IDs and the scores.

        Returns
        -------
        items : array(n,)
            The top-N highest predicted items for this user. If the 'X' data passed to
            fit was a DataFrame, will contain the item IDs from its column
            'ItemId', otherwise will be integers matching to the columns of 'X'.
        scores : array(n,)
            The predicted scores for the top-N items. Will only be returned
            when passing ``output_score=True``, in which case the result will
            be a tuple with these two entries.
        """
        a_vec, a_bias = self.factors_warm(X=X, X_col=X_col, X_val=X_val, W=W,
                                          U=U, U_col=U_col, U_val=U_val,
                                          return_bias=True)
        return self._topN(user=None, a_vec=a_vec, a_bias=a_bias, n=n,
                          include=include, exclude=exclude,
                          output_score=output_score)

    def transform(self, X, y=None, U=None, W=None, replace_existing=False):
        """
        Reconstruct entries of the 'X' matrix

        Will reconstruct all the entries in the 'X' matrix as determined
        by the model. This method is intended to be used for imputing tabular
        data, and can be used as part of SciKit-Learn pipelines.

        Note
        ----
        The argument 'NA_as_zero' is ignored here.

        Note
        ----
        If fitting the model to DataFrame inputs (instead of NumPy arrays and/or
        SciPy sparse matrices), the IDs are reindexed internally,
        and the inputs provided here should match with the numeration that was
        produced by the model. The mappings in such case are available under
        attributes ``self.user_mapping_`` and ``self.item_mapping_``.

        Parameters
        ----------
        X : array(m, n)
            New 'X' data with potentially missing entries which are to be imputed.
            Missing entries should have value ``np.nan``.
        y : None
            Not used. Kept as a placeholder for compatibility with SciKit-Learn
            pipelines.
        U : array(m, p), CSR matrix(m, p), COO matrix(m, p), or None
            User attributes information for each row in 'X'.
            Should not contain any missing values.
        W : array(m, n), array(nnz,), or None
            Observation weights. Must have the same shape as 'X' - that is,
            if 'X' is a sparse COO matrix, must be a 1-d array with the same
            number of non-zero entries as 'X.data', if 'X' is a 2-d array,
            'W' must also be a 2-d array.

        Returns
        -------
        X : array(m, n)
            The 'X' matrix as a dense array with all entries as determined by
            the model. Note that this will be returned as a dense NumPy array.
        """
        if (self.k_sec == 0) and (self.k_main == 0) \
            and (self.w_user == 0) and (self.w_item == 0):
            if U is not None:
                msg  = "User side info is not used for warm-start predictions "
                msg += "with this combination of hyperparameters."
                warnings.warn(msg) 
            return self._transform(X=X, U=None, U_bin=None, W=W)
        
        Xrow, Xcol, Xval, W_sp, Xarr, \
        Xcsr_p, Xcsr_i, Xcsr, \
        W_dense, Xorig, mask_take, \
        Uarr, Urow, Ucol, Uval, _1, Ucsr_p, Ucsr_i, Ucsr, \
        n, m_u, m_x, p, _2, \
        lambda_, lambda_bias, \
        l1_lambda, l1_lambda_bias = \
            self._process_transform_inputs(X=X, U=U, U_bin=None, W=W,
                                           replace_existing=False)

        c_funs = wrapper_float if self.use_float else wrapper_double
        A, A_bias, _1 = c_funs.call_factors_offsets_explicit_multiple(
                            Xrow,
                            Xcol,
                            Xval,
                            Xcsr_p, Xcsr_i, Xcsr,
                            W_sp,
                            Xarr,
                            W_dense,
                            Uarr,
                            Urow,
                            Ucol,
                            Uval,
                            Ucsr_p, Ucsr_i, Ucsr,
                            self.item_bias_,
                            self._B_pred,
                            self._B_plus_bias,
                            self.C_,
                            self.C_bias_,
                            self._TransBtBinvBt,
                            self._BtB,
                            self.glob_mean_,
                            m_x, n,
                            self.k, self.k_sec, self.k_main,
                            lambda_, lambda_bias,
                            self.w_user,
                            self.user_bias,
                            0, 0,
                            self.nthreads
                        )
        return self._transform_step(A, A_bias, mask_take, Xorig)


class OMF_implicit(_OMF):
    """
    Offsets model for implicit-feedback data

    Tries to approximate the 'X' interactions matrix using the user side information
    'U' and item side information 'I' by a formula as follows:
        X ~ (A + U*C) * t(B + I*D)

    Note
    ----
    This model was implemented for experimentation purposes only. Performance
    is likely to be bad. Be warned.

    Note
    ----
    This works by first fitting a model with no side info and then reconstructing
    the parameters by least squares approximations, so when making warm-start
    predictions, the results will be exactly the same as if not using any
    side information (user/item attributes).

    Note
    ----
    The model optimization objective will not scale any of its terms according
    to number of entries, so hyperparameters such as ``lambda_`` will require
    more tuning than in other software and trying out values over a wider range.

    Parameters
    ----------
    k : int
        Number of latent factors to use (dimensionality of the low-rank
        approximation).
        Typical values are 30 to 100.
    lambda_ : float
        Regularization parameter. Note that the default
        value for ``lambda_`` here is much higher than in other software, and that
        the loss/objective function is not divided by the number of entries.
        For example, a good number for the LastFM-360K could be ``lambda_=5``.
        Typical values are 10^-2 to 10^2.
    alpha : float
        Weighting parameter for the non-zero entries in the implicit-feedback
        model. See [2] for details. Note that, while the author's suggestion for
        this value is 40, other software such as ``implicit`` use a value of 1,
        whereas Spark uses a value of 0.01 by default
        If the data
        has very high values, might even be beneficial to put a very low value
        here - for example, for the LastFM-360K, values below 1 might
        give better results.
    use_cg : bool
        In the ALS method, whether to use a conjugate gradient method to solve
        the closed-form least squares problems. This is a faster and more
        memory-efficient alternative than the default Cholesky solver, but less
        exact, less numerically stable, and will require slightly more ALS
        iterations (``niter``) to reach a good optimum.
        In general, better results are achieved with ``use_cg=False``.
        Note that, if using this method, calculations after fitting which involve
        new data such as ``factors_warm``,  might produce slightly different
        results from the factors obtained from calling ``fit`` with the same data,
        due to differences in numerical precision. A workaround for this issue
        (factors on new data that might differ slightly) is to use
        ``finalize_chol=True``.
        Even if passing "True" here, will use the Cholesky method in cases in which
        it is faster (e.g. dense matrices with no missing values),
        and will not use the conjugate gradient method on new data.
    add_intercepts : bool
        Whether to add intercepts/biases to the user/item attribute matrices.
    niter : int
        Number of alternating least-squares iterations to perform. Note that
        one iteration denotes an update round for all the matrices rather than
        an update of a single matrix. In general, the more iterations, the better
        the end result.
        Typical values are 6 to 30.
    apply_log_transf : bool
        Whether to apply a logarithm transformation on the values of 'X'
        (i.e. 'X := log(X)')
    use_float : bool
        Whether to use C float type for the model parameters (typically this is
        ``np.float32``). If passing ``False``, will use C double (typically this
        is ``np.float64``). Using float types will speed up computations and
        use less memory, at the expense of reduced numerical precision.
    max_cg_steps : int
        Maximum number of conjugate gradient iterations to perform in an ALS round.
        Ignored when passing ``use_cg=False``.
    finalize_chol : bool
        When passing ``use_cg=True``, whether to perform the last iteration with
        the Cholesky solver. This will make it slower, but will avoid the issue
        of potential mismatches between the result from ``fit`` and calls to
        ``factors_warm`` or similar with the same data.
    random_state : int, RandomState, or Generator
        Seed used to initialize parameters at random. If passing a NumPy
        RandomState or Generator, will use it to draw a random integer.
    verbose : bool
        Whether to print informational messages about the optimization
        routine used to fit the model. Note that, if running this from a
        Jupyter notebook, these messages will be printed in the console,
        not in the notebook itself.
    handle_interrupt : bool
        When receiving an interrupt signal, whether the model should stop
        early and leave a usable object with the parameters obtained up
        to the point when it was interrupted (when passing 'True'), or
        raise an interrupt exception without producing a fitted model object
        (when passing 'False').
    produce_dicts : bool
        Whether to produce Python dicts from the mappings between user/item
        IDs passed to 'fit' and the internal IDs used by the class. Having
        these dicts might speed up some computations such as 'predict',
        but it will add some extra overhead at the time of fitting the model
        and extra memory usage. Ignored when passing the data as matrices
        and arrays instead of data frames.
    copy_data : bool
        Whether to make copies of the input data that is passed to this
        object's methods (``fit``, ``predict``, etc.), in order to avoid
        modifying such data in-place. Passing ``False`` will save some
        computation time and memory usage.
    nthreads : int
        Number of parallel threads to use. If passing -1, will take the
        maximum available number of threads in the system.

    Attributes
    ----------
    is_fitted_ : bool
        Whether the model has been fitted to data.
    reindex_ : bool
        Whether the IDs passed to 'fit' were reindexed internally
        (this will only happen when passing data frames to 'fit').
    user_mapping_ : array(m,) or array(0,)
        Correspondence of internal user (row) IDs to IDs in the data
        passed to 'fit'. Will only be non-empty when passing a data frame
        as input to 'X'.
    item_mapping_ : array(n,) or array(0,)
        Correspondence of internal item (column) IDs to IDs in the data
        passed to 'fit'. Will only be non-empty when passing a data frame
        as input to 'X'.
    user_dict_ : dict
        Python dict version of ``user_mapping_``. Only filled-in when
        passing ``produce_dicts=True`` and when passing data frames to 'fit'.
    item_dict_ : dict
        Python dict version of ``item_mapping_``. Only filled-in when
        passing ``produce_dicts=True`` and when passing data frames to 'fit'.
    A_ : array(m, k)
        The free offset for the user-factors obtained from user attributes
        and matrix C_.
    B_ : array(n, k)
        The free offset for the item-factors obtained from item attributes
        and matrix D_.
    C_ : array(p, k)
        The obtained coefficients for the user attributes.
    D_ : array(q, k)
        The obtained coefficients for the item attributes.
    C_bias_ : array(k)
        The intercepts/biases for the C matrix.
    D_bias_ : array(k)
        The intercepts/biases for the D matrix.

    References
    ----------
    .. [1] Cortes, David.
           "Cold-start recommendations in Collective Matrix Factorization."
           arXiv preprint arXiv:1809.00366 (2018).
    .. [2] Hu, Yifan, Yehuda Koren, and Chris Volinsky.
           "Collaborative filtering for implicit feedback datasets."
           2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008.
    .. [3] Takacs, Gabor, Istvan Pilaszy, and Domonkos Tikk.
           "Applications of the conjugate gradient method for implicit feedback collaborative filtering."
           Proceedings of the fifth ACM conference on Recommender systems. 2011.
    """
    def __init__(self, k=50, lambda_=1e0, alpha=1., use_cg=True,
                 add_intercepts=True, niter=10,
                 apply_log_transf=False, use_float=False,
                 max_cg_steps=3, finalize_chol=False,
                 random_state=1, verbose=False,
                 produce_dicts=False, handle_interrupt=True,
                 copy_data=True, nthreads=-1):
        self._take_params(implicit=True, alpha=alpha, downweight=False,
                          k=k, lambda_=lambda_, method="als",
                          apply_log_transf=apply_log_transf,
                          use_cg=use_cg, max_cg_steps=max_cg_steps,
                          finalize_chol=finalize_chol,
                          user_bias=False, item_bias=False,
                          k_user=0, k_item=0, k_main=0,
                          w_main=1., w_user=1., w_item=1.,
                          maxiter=0, niter=niter,
                          corr_pairs=0,
                          NA_as_zero=False,
                          NA_as_zero_user=False, NA_as_zero_item=False,
                          precompute_for_predictions=True,
                          use_float=use_float,
                          random_state=random_state, init="normal",
                          verbose=verbose, print_every=0,
                          handle_interrupt=handle_interrupt,
                          produce_dicts=produce_dicts, copy_data=copy_data,
                          nthreads=nthreads)
        self._take_params_offsets(k_sec=0, k_main=0, add_intercepts=add_intercepts)
        msg  = "This model was implemented for experimentation purposes."
        msg += " Performance is likely to be bad. Be warned."
        warnings.warn(msg)

    def __str__(self):
        msg  = "Offsets factorization model\n"
        msg += "(implicit-feedback variant)\n\n"
        if not self.is_fitted_:
            msg += "Model has not been fitted to data.\n"
        return msg

    def get_params(self, deep=None):
        return {
            "k" : self.k, "lambda_" : self.lambda_, "alpha" : self.alpha,
            "downweight" : self.downweight,
            "add_intercepts" : self.add_intercepts, "niter" : self.niter,
            "use_float" : self.use_float, "use_cg" : self.use_cg,
            "random_state" : self.random_state,
            "nthreads" : self.nthreads
        }

    def fit(self, X, U=None, I=None):
        """
        Fit model to implicit-feedback data and user/item attributes

        Note
        ----
        None of the side info inputs should have missing values. If passing side
        information 'U' and/or 'I', all entries (users/items) must be present
        in both the main matrix and the side info matrix.

        Note
        ----
        In order to avoid potential decimal differences in the factors obtained
        when fitting the model and when calling the prediction functions on
        new data, when the data is sparse, it's necessary to sort it beforehand
        by columns and also pass the data data with indices sorted (by column)
        to the prediction functions.

        Parameters
        ----------
        X : DataFrame(nnz, 3), or sparse COO(m, n)
            Matrix to factorize. Can be passed as a SciPy
            sparse COO matrix (recommended), or
            as a Pandas DataFrame, in which case it should contain the
            following columns: 'UserId', 'ItemId', and 'Value'.
            If passing a NumPy array, missing (unobserved) entries should 
            have value ``np.nan``.
            If passing a DataFrame,
            the IDs will be internally remapped.
        U : array(m, p), COO(m, p), DataFrame(m, p+1), or None
            User attributes information. If 'X' is a DataFrame, should also
            be a DataFrame, containing column 'UserId'. If 'U' is sparse,
            'X' should be passed as a sparse COO matrix too.
            Should not contain any missing values.
        I : array(n, q), COO(n, q), DataFrame(n, q+1), or None
            Item attributes information. If 'X' is a DataFrame, should also
            be a DataFrame, containing column 'ItemId'. If 'I' is sparse,
            'X' should be passed as a sparse COO matrix too.
            Should not contain any missing values.

        Returns
        -------
        self

        """
        if "coo_matrix" in [U.__class__.__name__, I.__class__.__name__]:
            msg  = "Cannot pass user/item side info in sparse format"
            msg += " for implicit-feedback model."
            raise ValueError(msg)
        return self._fit_common(X, U=U, I=I, U_bin=None, I_bin=None, W=None,
                                enforce_same_shape=True)

    def _fit(self, Xrow, Xcol, Xval, W_sp, Xarr, W_dense,
             Uarr, Urow, Ucol, Uval, Ub_arr,
             Iarr, Irow, Icol, Ival, Ib_arr,
             m, n, m_u, n_i, p, q,
             m_ub, n_ib, pbin, qbin):
        c_funs = wrapper_float if self.use_float else wrapper_double
        self._w_main_multiplier = 1.
        self.A_, self.B_, self.C_, self.D_, \
        self._A_pred, self._B_pred, self._BtB = \
            c_funs.call_fit_offsets_implicit_als(
                Xrow,
                Xcol,
                Xval,
                Uarr,
                Iarr,
                m, n, p, q,
                self.k, self.add_intercepts,
                self.lambda_, self.alpha, self.apply_log_transf,
                self.verbose, self.nthreads, self.use_cg,
                self.max_cg_steps, self.finalize_chol,
                self.downweight,
                self.apply_log_transf,
                self.random_state, self.niter,
                self.handle_interrupt
            )
        self._n_orig = self.B_.shape[0]
        self.is_fitted_ = True
        return self

    def factors_warm(self, X_col, X_val, return_raw_A=False):
        """
        Determine user latent factors based on new interactions data

        Parameters
        ----------
        X_col : array(nnz,)
            Observed new 'X' data for a given user, in sparse format.
            'X_col' should contain the column indices (items) of the observed 
            entries. If 'X' passed to 'fit' was a data frame, should have
            entries from 'ItemId' column, otherwise should have column numbers
            (starting at zero).
        X_val : array(nnz,)
            Observed new 'X' data for a given user, in sparse format.
            'X_val' should contain the values in the columns/items given by
            'X_col'.
        return_raw_A : bool
            Whether to return the raw A factors (the free offset), or the
            factors used in the factorization, to which the attributes
            component has been added.

        Returns
        -------
        factors : array(k,)
            User factors as determined from the data in 'X_col' and 'X_val'.
        """
        if (X_col is None) or (X_val is None):
            raise ValueError("Must pass 'X_col' and 'X_val'.")
        return self._factors_warm_common(X=None, X_col=X_col, X_val=X_val, W=None,
                                         U=None, U_bin=None, U_col=None, U_val=None,
                                         output_a=return_raw_A)

    def _factors_warm(self, X, W_dense, X_val, X_col, W_sp,
                      U, U_val, U_col, U_bin, output_a):
        c_funs = wrapper_float if self.use_float else wrapper_double
        a_pred, a_vec = c_funs.call_factors_offsets_implicit_single(
            X_val,
            X_col,
            U,
            np.empty(0, dtype=self.dtype_),
            np.empty(0, dtype=ctypes.c_int),
            self._B_pred,
            self.C_,
            self.C_bias_,
            self._TransBtBinvBt,
            self._BtB,
            self.k,
            self.lambda_, self.alpha,
            self.apply_log_transf,
            output_a
        )
        if output_a:
            return a_vec
        else:
            return a_pred

    def predict_warm(self, items, X_col, X_val):
        """
        Predict scores for existing items, for a new user, given 'X'

        Parameters
        ----------
        items : array-like(n,)
            Items whose ratings are to be predicted. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'.
        X_col : array(nnz,)
            Observed new 'X' data for a given user, in sparse format.
            'X_col' should contain the column indices (items) of the observed 
            entries. If 'X' passed to 'fit' was a data frame, should have
            entries from 'ItemId' column, otherwise should have column numbers
            (starting at zero).
        X_val : array(nnz,)
            Observed new 'X' data for a given user, in sparse format.
            'X_val' should contain the values in the columns/items given by
            'X_col'.
        
        Returns
        -------
        scores : array(n,)
            Predicted values for the requested items for a user defined by
            the given values of 'X' in 'X_col' and 'X_val'.
        """
        a_vec = self.factors_warm(X_col=X_col, X_val=X_val)
        return self._predict(user=None, a_vec=a_vec, a_bias=0., item=items)

    def predict_warm_multiple(self, X, item, U=None):
        """
        Predict scores for existing items, for new users, given 'X'

        Parameters
        ----------
        X : array(m, n), CSR matrix(m, n) , or COO matrix(m, n)
            New 'X' data with potentially missing entries.
            Missing entries should have value ``np.nan`` when passing a dense
            array.
            Must have one row per entry of ``item``.
        item : array-like(m,)
            Items for whom ratings/values are to be predicted. If 'X' passed to
            fit was a  DataFrame, must match with the entries in its 'ItemId'
            column, otherwise should match with the columns of 'X'.
            Each entry in ``item`` will be matched with the corresponding row
            of ``X``.
        U : array(m, p), CSR matrix(m, p), COO matrix(m, p), or None
            User attributes information for each row in 'X'.
            Should not contain any missing values.

        Returns
        -------
        scores : array(m,)
            Predicted ratings for the requested user-item combinations.
        """
        Xrow, Xcol, Xval, W_sp, Xarr, \
        Xcsr_p, Xcsr_i, Xcsr, \
        W_dense, Xorig, mask_take, \
        Uarr, Urow, Ucol, Uval, _1, Ucsr_p, Ucsr_i, Ucsr, \
        n, m_u, m_x, p, _2, \
        lambda_, lambda_bias, \
        l1_lambda, l1_lambda_bias = \
            self._process_transform_inputs(X=X, U=U, U_bin=None, W=None,
                                           replace_existing=True)

        c_funs = wrapper_float if self.use_float else wrapper_double
        A, _ = c_funs.call_factors_offsets_implicit_multiple(
                    Xrow,
                    Xcol,
                    Xval,
                    Xcsr_p, Xcsr_i, Xcsr,
                    Uarr,
                    Urow,
                    Ucol,
                    Uval,
                    Ucsr_p,
                    Ucsr_i,
                    Ucsr,
                    self._B_pred,
                    self._C,
                    self._C_bias,
                    self._BtB,
                    m_x, n,
                    self.k,
                    lambda_, self.alpha,
                    self.apply_log_transf,
                    0,
                    self.nthreads
                )
        return self._predict_user_multiple(A, item, bias=None)

    def topN_warm(self, n=10, X_col=None, X_val=None,
                  include=None, exclude=None, output_score=False):
        """
        Compute top-N highest-predicted items for a new user, given 'X'

        Parameters
        ----------
        n : int
            Number of top-N highest-predicted results to output.
        X_col : array(nnz,)
            Observed new 'X' data for a given user, in sparse format.
            'X_col' should contain the column indices (items) of the observed 
            entries. If 'X' passed to 'fit' was a data frame, should have
            entries from 'ItemId' column, otherwise should have column numbers
            (starting at zero).
        X_val : array(nnz,)
            Observed new 'X' data for a given user, in sparse format.
            'X_val' should contain the values in the columns/items given by
            'X_col'.
        include : array-like
            List of items which will be ranked. If passing this, will only
            make a ranking among these items. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include or 'exclude'.
        exclude : array-like
            List of items to exclude from the ranking. If passing this, will
            rank all the items except for these. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include or 'exclude'.
        output_score : bool
            Whether to output the scores in addition to the IDs. If passing
            'False', will return a single array with the item IDs, otherwise
            will return a tuple with the item IDs and the scores.
        
        Returns
        -------
        items : array(n,)
            The top-N highest predicted items for this user. If the 'X' data passed to
            fit was a DataFrame, will contain the item IDs from its column
            'ItemId', otherwise will be integers matching to the columns of 'X'.
        scores : array(n,)
            The predicted scores for the top-N items. Will only be returned
            when passing ``output_score=True``, in which case the result will
            be a tuple with these two entries.
        """
        if (X_col is None) or (X_val is None):
            raise ValueError("Must pass 'X_col' and 'X_val'.")
        a_vec = self.factors_warm(X_col=X_col, X_val=X_val)
        return self._topN(user=None, a_vec=a_vec, a_bias=0., n=n,
                          include=include, exclude=exclude,
                          output_score=output_score)

class ContentBased(_OMF_Base):
    """
    Content-based recommendation model

    Fits a recommendation model to explicit-feedback data based on user
    and item attributes only, making it a more ideal approach for cold-start
    recommendations and with faster prediction times. Follows the same
    factorization approach as the classical model, but with the latent-factor
    matrices being determined as linear combinations of the user and item
    attributes - this is similar to a two-layer neural network with
    separate layers for each input.

    The 'X' is approximated using the user side information
    'U' and item side information 'I' by a formula as follows:
        X ~ (U*C) * t(I*D)

    Note
    ----
    This is a highly non-linear model that will take many more L-BFGS
    iterations to converge compared to the other models. It's advised
    to experiment with tuning the maximum number of iterations.

    Note
    ----
    The input data for attributes does not undergo any transformations when
    fitting this model, which is to some extent sensible to the scales of the
    variables and their means in the same way as regularized linear regression.

    Note
    ----
    In order to obtain the final user-factors and item-factors matrices
    that are used to factorize 'X' from a fitted-model object, you'll
    need to perform a matrix multiplication between the side info
    ('U' and 'I') and the fitted parameters ('C_' and 'D_') - e.g.
    'A = U*model.C_ + model.C_bias_'.

    Parameters
    ----------
    k : int
        Number of latent factors to use (dimensionality of the low-rank
        approximation).
        Recommended values are 30 to 100.
    lambda_ : float or array(6,)
        Regularization parameter. Can also use different regularization for each
        matrix, in which case it should be an array with 6 entries, corresponding,
        in this order, to: user_bias, item_bias, [ignored], [ignored], C, D.
        Note that the default
        value for ``lambda_`` here is much higher than in other software, and that
        the loss/objective function is not divided by the number of entries.
        Recommended values are 10^-2 to 10^2.
    user_bias : bool
        Whether to add user biases (intercepts) to the model.
    item_bias : bool
        Whether to add item biases (intercepts) to the model. Be aware that using
        item biases with low regularization for them will tend to favor items
        with high average ratings regardless of the number of ratings the item
        has received.
    add_intercepts : bool
        Whether to add intercepts/biases to the user/item attribute matrices.
    maxiter : int
        Maximum L-BFGS iterations to perform. The procedure will halt if it
        has not converged after this number of updates. Note that, compared to
        the collective model, more iterations will be required for converge
        here. Using higher regularization values might also decrease the number
        of required iterations. Pass zero for no L-BFGS iterations limit.
        If the procedure is spending thousands of iterations
        without any significant decrease in the loss function or gradient norm,
        it's highly likely that the regularization is too low.
    corr_pairs : int
        Number of correction pairs to use for the L-BFGS optimization routine.
        Recommended values are between 3 and 7. Note that higher values
        translate into higher memory requirements.
    parallelize : str, "separate" or "single"
        How to parallelize gradient calculations when using more than one
        thread. Passing ``'separate'`` will iterate over the data twice - first
        by rows and then by columns, letting each thread calculate results
        for each row and column, whereas passing ``'single'`` will iterate over
        the data only once, and then sum the obtained results from each thread.
        Passing ``'separate'`` is much more memory-efficient and less prone to
        irreproducibility of random seeds, but might be slower for typical
        use-cases. Ignored when passing ``nthreads=1`` or compiling without
        OpenMP support.
    verbose : bool
        Whether to print informational messages about the optimization
        routine used to fit the model. Note that, if running this from a
        Jupyter notebook, these messages will be printed in the console,
        not in the notebook itself. Be aware that, if passing 'False', the
        optimization routine will not respond to interrupt signals.
    print_every : int
        Print L-BFGS convergence messages every n-iterations. Ignored
        when passing ``verbose=False``.
    random_state : int, RandomState, or Generator
        Seed used to initialize parameters at random. If passing a NumPy
        RandomState or Generator, will use it to draw a random integer. Note
        however that, if using more than one thread, results might not be
        100% reproducible due to round-off errors in parallelized aggregations.
    use_float : bool
        Whether to use C float type for the model parameters (typically this is
        ``np.float32``). If passing ``False``, will use C double (typically this
        is ``np.float64``). Using float types will speed up computations and
        use less memory, at the expense of reduced numerical precision.
    produce_dicts : bool
        Whether to produce Python dicts from the mappings between user/item
        IDs passed to 'fit' and the internal IDs used by the class. Having
        these dicts might speed up some computations such as 'predict',
        but it will add some extra overhead at the time of fitting the model
        and extra memory usage. Ignored when passing the data as matrices
        and arrays instead of data frames.
    handle_interrupt : bool
        When receiving an interrupt signal, whether the model should stop
        early and leave a usable object with the parameters obtained up
        to the point when it was interrupted (when passing 'True'), or
        raise an interrupt exception without producing a fitted model object
        (when passing 'False').
    start_with_ALS : bool
        Whether to determine the initial coefficients through an ALS procedure.
        This might help to speed up the procedure by starting closer to an
        optimum. This option is not available when the side information is passed
        as sparse matrices.
    copy_data : bool
        Whether to make copies of the input data that is passed to this
        object's methods (``fit``, ``predict``, etc.), in order to avoid
        modifying such data in-place. Passing ``False`` will save some
        computation time and memory usage.
    nthreads : int
        Number of parallel threads to use. If passing -1, will take the
        maximum available number of threads in the system.

    Attributes
    ----------
    is_fitted_ : bool
        Whether the model has been fitted to data.
    reindex_ : bool
        Whether the IDs passed to 'fit' were reindexed internally
        (this will only happen when passing data frames to 'fit').
    user_mapping_ : array(m,) or array(0,)
        Correspondence of internal user (row) IDs to IDs in the data
        passed to 'fit'. Will only be non-empty when passing a data frame
        as input to 'X'.
    item_mapping_ : array(n,) or array(0,)
        Correspondence of internal item (column) IDs to IDs in the data
        passed to 'fit'. Will only be non-empty when passing a data frame
        as input to 'X'.
    user_dict_ : dict
        Python dict version of ``user_mapping_``. Only filled-in when
        passing ``produce_dicts=True`` and when passing data frames to 'fit'.
    item_dict_ : dict
        Python dict version of ``item_mapping_``. Only filled-in when
        passing ``produce_dicts=True`` and when passing data frames to 'fit'.
    glob_mean_ : float
        The global mean of the non-missing entries in 'X' passed to 'fit'.
    user_bias_ : array(m,), or array(0,)
        The obtained biases for each user (row in the 'X' matrix).
        If passing ``user_bias=False`` (the default), this array
        will be empty.
    item_bias_ : array(n,)
        The obtained biases for each item (column in the 'X' matrix).
        If passing ``item_bias=False`` (the default), this array
        will be empty.
    C_ : array(p, k)
        The obtained coefficients for the user attributes.
    D_ : array(q, k)
        The obtained coefficients for the item attributes.
    C_bias_ : array(k)
        The intercepts/biases for the C matrix.
    D_bias_ : array(k)
        The intercepts/biases for the D matrix.
    nfev_ : int
        Number of function and gradient evaluations performed during the
        L-BFGS optimization procedure.
    nupd_ : int
        Number of L-BFGS updates performed during the optimization procedure.
    
    References
    ----------
    .. [1] Cortes, David.
           "Cold-start recommendations in Collective Matrix Factorization."
           arXiv preprint arXiv:1809.00366 (2018).
    """
    def __init__(self, k=20, lambda_=1e2, user_bias=False, item_bias=False,
                 add_intercepts=True, maxiter=15000, corr_pairs=3,
                 parallelize="separate", verbose=True, print_every=100,
                 random_state=1, use_float=False,
                 produce_dicts=False, handle_interrupt=True, start_with_ALS=True,
                 copy_data=True, nthreads=-1):
        self._take_params(implicit=False, alpha=40., downweight=False,
                          k=1, lambda_=lambda_, method="lbfgs", use_cg=False,
                          user_bias=user_bias, item_bias=item_bias,
                          center=True,
                          k_user=0, k_item=0, k_main=0,
                          w_main=1., w_user=1., w_item=1.,
                          maxiter=maxiter,
                          niter=0, parallelize="separate",
                          corr_pairs=corr_pairs,
                          NA_as_zero=False,
                          NA_as_zero_user=False, NA_as_zero_item=False,
                          precompute_for_predictions=True, use_float=use_float,
                          random_state=random_state,
                          init="normal", verbose=verbose, print_every=print_every,
                          handle_interrupt=handle_interrupt,
                          produce_dicts=produce_dicts, copy_data=copy_data,
                          nthreads=nthreads)
        self._take_params_offsets(k_sec=k, k_main=0,
                                  add_intercepts=add_intercepts)
        self.k = 0
        self._k_pred = self.k_sec
        self.start_with_ALS = bool(start_with_ALS)

    def __str__(self):
        msg  = "Content-based factorization model\n"
        msg += "(explicit-feedback)\n\n"
        if not self.is_fitted_:
            msg += "Model has not been fitted to data.\n"
        return msg

    def get_params(self, deep=None):
        return {
            "k" : self.k, "lambda_" : self.lambda_,
            "user_bias" : self.user_bias, "item_bias" : self.item_bias,
            "add_intercepts" : self.add_intercepts, "maxiter" : self.maxiter,
            "corr_pairs" : self.corr_pairs,
            "parallelize" : self.parallelize, "verbose" : self.verbose,
            "print_every" : self.print_every,
            "random_state" : self.random_state, "use_float" : self.use_float,
            "nthreads" : self.nthreads
        }

    def fit(self, X, U, I, W=None):
        """
        Fit model to explicit-feedback data based on user-item attributes

        Note
        ----
        None of the side info inputs should have missing values. All entries (users/items)
        must be present in both the main matrix and the side info matrix.

        Note
        ----
        In order to avoid potential decimal differences in the factors obtained
        when fitting the model and when calling the prediction functions on
        new data, when the data is sparse, it's necessary to sort it beforehand
        by columns and also pass the data data with indices sorted (by column)
        to the prediction functions.

        Parameters
        ----------
        X : DataFrame(nnz, 3), DataFrame(nnz, 4), array(m, n), or sparse COO(m, n)
            Matrix to factorize (e.g. ratings). Can be passed as a SciPy
            sparse COO matrix (recommended), as a dense NumPy array, or
            as a Pandas DataFrame, in which case it should contain the
            following columns: 'UserId', 'ItemId', and 'Rating'.
            If passing a NumPy array, missing (unobserved) entries should 
            have value ``np.nan``.
            Might additionally have a column 'Weight'. If passing a DataFrame,
            the IDs will be internally remapped.
            If passing sparse 'U' or sparse 'I', 'X' cannot be passed as
            a DataFrame.
        U : array(m, p), COO(m, p), DataFrame(m, p+1)
            User attributes information. If 'X' is a DataFrame, should also
            be a DataFrame, containing column 'UserId'. If 'U' is sparse,
            'X' should be passed as a sparse COO matrix or as a dense NumPy array.
            Should not contain any missing values.
        I : array(n, q), COO(n, q), DataFrame(n, q+1)
            Item attributes information. If 'X' is a DataFrame, should also
            be a DataFrame, containing column 'ItemId'. If 'I' is sparse,
            'X' should be passed as a sparse COO matrix or as a dense NumPy array.
            Should not contain any missing values.
        W : None, array(nnz,), or array(m, n)
            Observation weights. Must have the same shape as 'X' - that is,
            if 'X' is a sparse COO matrix, must be a 1-d array with the same
            number of non-zero entries as 'X.data', if 'X' is a 2-d array,
            'W' must also be a 2-d array.

        Returns
        -------
        self

        """
        return self._fit_common(X, U=U, I=I, U_bin=None, I_bin=None, W=W,
                                enforce_same_shape=True)

    def _fit(self, Xrow, Xcol, Xval, W_sp, Xarr, W_dense,
             Uarr, Urow, Ucol, Uval, Ub_arr,
             Iarr, Irow, Icol, Ival, Ib_arr,
             m, n, m_u, n_i, p, q,
             m_ub, n_ib, pbin, qbin):
        c_funs = wrapper_float if self.use_float else wrapper_double

        if self.start_with_ALS:
            if (not Uarr.shape[0]) or (not Iarr.shape[0]):
                warnings.warn("Option 'start_with_ALS' not available for sparse data.")
                self.start_with_ALS = False

        self.user_bias_, self.item_bias_, \
        self.C_, self.D_, self.C_bias_, self.D_bias_, \
        self._A_pred, self._B_pred, self.glob_mean_, self.nupd_, self.nfev_ = \
            c_funs.call_fit_content_based_lbfgs(
                Xrow,
                Xcol,
                Xval,
                W_sp,
                Xarr,
                W_dense,
                Uarr,
                Urow,
                Ucol,
                Uval,
                Iarr,
                Irow,
                Icol,
                Ival,
                m, n, p, q,
                self.k_sec,
                self.user_bias, self.item_bias,
                self.add_intercepts,
                self.lambda_ if isinstance(self.lambda_, float) else 0.,
                self.lambda_ if isinstance(self.lambda_, np.ndarray) else np.empty(0, dtype=self.dtype_),
                self.verbose, self.print_every,
                self.corr_pairs, self.maxiter,
                self.nthreads, self.parallelize != "separate",
                self.random_state,
                self.handle_interrupt,
                start_with_ALS=self.start_with_ALS
        )

        self._n_orig = 0
        self.is_fitted_ = True
        return self

    def factors_cold(self, U=None, U_col=None, U_val=None):
        """
        Determine user-factors from new data, given U

        Note
        ----
        For large-scale usage, these factors can be obtained by a
        matrix multiplication of the attributes matrix and the
        attribute (model parameter) ``C_``, plus the intercept if
        present (``C_bias_``).

        Parameters
        ----------
        U : array(p,), or None
            User attributes in the new data (1-row only).
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_col : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_val : None or array(nnz)
            User attributes in the new data (1-row only), in sparse
            format. 'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.
        
        Returns
        -------
        factors : array(k,)
            The user-factors as determined by the model.
        """
        assert self.is_fitted_
        U, U_col, U_val, _ = self._process_new_U(U, U_col, U_val, None)
        c_funs = wrapper_float if self.use_float else wrapper_double
        a_vec = c_funs.call_factors_content_based_single(
            U,
            U_val,
            U_col,
            self.C_,
            self.C_bias_
        )
        return a_vec

    def factors_multiple(self, U=None):
        """
        Determine user-factors from new data for multiple rows, given U

        Parameters
        ----------
        U : array-like(m, p)
            User attributes in the new data.

        Returns
        -------
        factors : array(m, k)
            The user-factors as determined by the model.
        """
        factors = U.dot(self.C_)
        if self.C_bias_.shape[0]:
            factors[:] += self.C_bias_.reshape((1,-1))
        return factors

    def topN_new(self, n=10, U=None, U_col=None, U_val=None,
                 I=None, output_score=False):
        """
        Compute top-N highest-predicted items for a given user, given U

        Parameters
        ----------
        n : int
            Number of top-N highest-predicted results to output.
        U : array(p,), or None
            User attributes for the user for whom to rank items.
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_col : None or array(nnz)
            User attributes for the user for whom to rank items, in sparse
            format. 'U_col' should contain the column indices of the
            non-zero entries (starting at zero).
            Should only pass one of 'U' or 'U_col'+'U_val'.
        U_val : None or array(nnz)
            User attributes for the user for whom to rank items, in sparse
            format. 'U_val' should contain the values in the columns
            given by 'U_col'.
            Should only pass one of 'U' or 'U_col'+'U_val'.
        I : array(n2, q), CSR(n2, q), or COO(n2, q)
            Attributes for the items to rank (each row corresponding to an
            item). Must have at least 'n' rows.
        output_score : bool
            Whether to output the scores in addition to the row numbers.
            If passing 'False', will return a single array with the item numbers, otherwise will return a tuple with the item numbers and the scores.

        Returns
        -------
        items : array(n,)
            The top-N highest predicted items among 'I'.
        scores : array(n,)
            The predicted scores for the top-N items. Will only be returned
            when passing ``output_score=True``, in which case the result will
            be a tuple with these two entries.
        """
        assert self.is_fitted_

        U, U_col, U_val, _ = self._process_new_U(U, U_col, U_val, None)
        Iarr, Irow, Icol, Ival, Icsr_p, Icsr_i, Icsr, n_i, q = \
            self._process_new_U_2d(I, is_I=True, allow_csr=True)

        if n > n_i:
            raise ValueError("There are fewer than 'n' items to rank.")

        c_funs = wrapper_float if self.use_float else wrapper_double
        rank_new, scores_new = c_funs.call_topN_new_content_based(
            U,
            U_val,
            U_col,
            Iarr,
            Irow,
            Icol,
            Ival,
            Icsr_p, Icsr_i, Icsr,
            self.C_,
            self.C_bias_,
            self.D_,
            self.D_bias_,
            n_i,
            self.glob_mean_,
            n, output_score,
            self.nthreads
        )

        if output_score:
            return rank_new, scores_new
        else:
            return rank_new

    def predict_new(self, U, I):
        """
        Predict rating given by new users to new items, given U and I

        Parameters
        ----------
        U : array(n, p), CSR(n, p), or COO(n, p)
            Attributes for the users whose ratings are to be predicted.
            Each row will be matched to the corresponding row of 'I'.
        I : array(n, q), CSR(n, q), or COO(n, q)
            Attributes for the items whose ratings are to be predicted.
            Each row will be matched to the corresponding row of 'U'.

        Returns
        -------
        scores : array(n,)
            Predicted ratings for the requested user-item combinations.
        """
        assert self.is_fitted_
        if U.shape[0] != I.shape[0]:
            raise ValueError("'U' and 'I' must have the same number of rows.")
        Uarr, Urow, Ucol, Uval, Ucsr_p, Ucsr_i, Ucsr, m_u, p = \
            self._process_new_U_2d(U, is_I=False, allow_csr=True)
        Iarr, Irow, Icol, Ival, Icsr_p, Icsr_i, Icsr, n_i, q = \
            self._process_new_U_2d(I, is_I=True, allow_csr=True)

        c_funs = wrapper_float if self.use_float else wrapper_double
        scores_new = c_funs.call_predict_X_new_content_based(
            Uarr,
            Urow,
            Ucol,
            Uval,
            Ucsr_p, Ucsr_i, Ucsr,
            Iarr,
            Irow,
            Icol,
            Ival,
            Icsr_p, Icsr_i, Icsr,
            self.C_,
            self.C_bias_,
            self.D_,
            self.D_bias_,
            m_u,
            self.glob_mean_,
            self.nthreads
        )
        return scores_new

    def predict_cold(self, U, items):
        """
        Predict rating given by new users to existing items, given U

        Parameters
        ----------
        U : array(n, p), CSR(n, p), or COO(n, p)
            Attributes for the users whose ratings are to be predicted.
            Each row will be matched to the corresponding row of 'items'.
        items : array-like(n,)
            Items whose ratings are to be predicted. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'.

        Returns
        -------
        scores : array(n,)
            Predicted ratings for the requested user-item combinations.
        """
        assert self.is_fitted_
        items = np.array(items).reshape(-1)
        assert items.shape[0] == U.shape[0]

        _1, items, _2, _3 = self._process_users_items(None, items, None, None)

        Uarr, Urow, Ucol, Uval, Ucsr_p, Ucsr_i, Ucsr, m_u, p = \
            self._process_new_U_2d(U, is_I=False, allow_csr=True)

        c_funs = wrapper_float if self.use_float else wrapper_double
        scores_new = c_funs.call_predict_X_old_content_based(
            Uarr,
            Urow,
            Ucol,
            Uval,
            Ucsr_p, Ucsr_i, Ucsr,
            items,
            self._B_pred,
            self.C_,
            self.C_bias_,
            self.item_bias_,
            m_u,
            self.glob_mean_,
            self.nthreads
        )
        return scores_new

class MostPopular(_CMF):
    """
    Non-Personalized recommender model

    Fits a model with only the intercept terms (biases), in order to provide
    non-personalized recommendations.

    This class is provided as a benchmark - if your personalized-recommendations
    model does not manage to beat this under the evaluation metrics of interest,
    chances are, that model needs to be reworked.

    It minimizes the same objective functions as the other classes and offers
    the same options (e.g. centering, scaling regulatization, etc.),
    but fitting only the biases.

    Parameters
    ----------
    implicit : bool
        Whether to use the implicit-feedback model, in which the 'X' matrix is
        assumed to have only binary entries and each of them having a weight
        in the loss function given by the observer user-item interactions and
        other parameters.
    center : bool
        Whether to center the "X" data by subtracting the mean value.
        Ignored (assumed "False") when passing ``implicit=True``.
    user_bias : bool
        Whether to add user biases to the model. Not supported for implicit
        feedback (``implicit=True``).
    lambda_ : float
        Regularization parameter. For the explicit-feedback case (default),
        lower values will tend to favor the highest-rated items regardless
        of the number of observations. Note that the default
        value for ``lambda_`` here is much higher than in other software, and that
        the loss/objective function is not divided by the number of entries.
    alpha : float
        Weighting parameter for the non-zero entries in the implicit-feedback
        model. See [2] for details. Note that, while the author's suggestion for
        this value is 40, other software such as ``implicit`` use a value of 1,
        whereas Spark uses a value of 0.01 by default
        See the documentation of ``CMF_implicit`` for more details.
    NA_as_zero : bool
        Whether to take missing entries in the 'X' matrix as zeros (only
        when the 'X' matrix is passed as sparse COO matrix or DataFrame)
        instead of ignoring them.
    scale_lam : bool
        Whether to scale (increase) the regularization parameter for each
        estimated bias according to the number of non-missing entries in
        the data. This is only available when passing ``implicit=False``.

        It is not recommended to use this option, as when passing ``True``,
        it tends to recommend items which have a single user interaction with
        the maximum possible value (e.g. 5-star movies from only 1 user).
        By default, ``scale_bias_const`` is also set to ``True``, so in order
        to have the regularization scale for each user/item, that option also
        needs to be turned off.
    scale_bias_const : bool
        When passing ``scale_lam=True``,
        whether to apply the same scaling to the regularization for all
        users and items, according to the average number of non-missing entries rather
        than to the number of entries for each specific user/item.

        While this tends to result in worse RMSE, it tends to make the top-N
        recommendations less likely to select items with only a few interactions
        from only a few users.

        Ignored when passing ``scale_lam=False``.
    apply_log_transf : bool
        Whether to apply a logarithm transformation on the values of 'X'
        (i.e. 'X := log(X)'). This is only available with ``implicit=True``.
    use_float : bool
        Whether to use C float type for the model parameters (typically this is
        ``np.float32``). If passing ``False``, will use C double (typically this
        is ``np.float64``). Using float types will speed up computations and
        use less memory, at the expense of reduced numerical precision.
    produce_dicts : bool
        Whether to produce Python dicts from the mappings between user/item
        IDs passed to 'fit' and the internal IDs used by the class. Having
        these dicts might speed up some computations such as 'predict',
        but it will add some extra overhead at the time of fitting the model
        and extra memory usage. Ignored when passing the data as matrices
        and arrays instead of data frames.
    copy_data : bool
        Whether to make copies of the input data that is passed to this
        object's methods (``fit``, ``predict``, etc.), in order to avoid
        modifying such data in-place. Passing ``False`` will save some
        computation time and memory usage.
    nthreads : int
        Number of parallel threads to use.  If passing -1, will take the
        maximum available number of threads in the system. Most of the
        work is done single-threaded however.

    Attributes
    ----------
    is_fitted_ : bool
        Whether the model has been fitted to data.
    reindex_ : bool
        Whether the IDs passed to 'fit' were reindexed internally
        (this will only happen when passing data frames to 'fit').
    user_mapping_ : array(m,) or array(0,)
        Correspondence of internal user (row) IDs to IDs in the data
        passed to 'fit'. Will only be non-empty when passing a data frame
        as input to 'X'.
    item_mapping_ : array(n,) or array(0,)
        Correspondence of internal item (column) IDs to IDs in the data
        passed to 'fit'. Will only be non-empty when passing a data frame
        as input to 'X'.
    user_dict_ : dict
        Python dict version of ``user_mapping_``. Only filled-in when
        passing ``produce_dicts=True`` and when passing data frames to 'fit'.
    item_dict_ : dict
        Python dict version of ``item_mapping_``. Only filled-in when
        passing ``produce_dicts=True`` and when passing data frames to 'fit'.
    glob_mean_ : float
        The global mean of the non-missing entries in 'X' passed to 'fit'
        (only for explicit-feedback case).
    user_bias_ : array(m,), or array(0,)
        The obtained biases for each user (row in the 'X' matrix).
        If passing ``user_bias=False`` (the default), this array
        will be empty.
    item_bias_ : array(n,)
        The obtained biases for each item (column in the 'X' matrix).
        Items are ranked according to these values.

    References
    ----------
    .. [1] Koren, Yehuda, Robert Bell, and Chris Volinsky.
           "Matrix factorization techniques for recommender systems."
           Computer 42.8 (2009): 30-37.
    .. [2] Hu, Yifan, Yehuda Koren, and Chris Volinsky.
           "Collaborative filtering for implicit feedback datasets."
           2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008.
    """
    def __init__(self, implicit=False, center=True, user_bias=False, lambda_=1e1, alpha=1.,
                 NA_as_zero=False, scale_lam=False, scale_bias_const=False, apply_log_transf=False,
                 use_float=False, produce_dicts=False,
                 copy_data=True, nthreads=-1):
        self._take_params(implicit=implicit, alpha=alpha, downweight=False,
                          k=1, lambda_=lambda_, method="als", use_cg=False,
                          apply_log_transf=apply_log_transf,
                          scale_lam=scale_lam, scale_bias_const=scale_bias_const,
                          user_bias=user_bias, item_bias=True,
                          center=center and not implicit,
                          k_user=0, k_item=0, k_main=0,
                          w_main=1., w_user=1., w_item=1.,
                          maxiter=0, niter=0, parallelize="separate",
                          corr_pairs=0,
                          NA_as_zero=NA_as_zero, NA_as_zero_user=False,
                          NA_as_zero_item=False,
                          precompute_for_predictions=False,
                          use_float=use_float,
                          random_state=1, init="normal",
                          verbose=0, print_every=0,
                          handle_interrupt=False,
                          produce_dicts=produce_dicts, copy_data=copy_data,
                          nthreads=nthreads)
        self.k = 0
        self.niter = 0
        self.implicit = bool(implicit)
        if self.implicit:
            if self.scale_lam:
                raise ValueError("'scale_lam' not supported for implicit-feedback.")
            if self.NA_as_zero:
                warnings.warn("'NA_as_zero' ignored with 'implicit=True'.")
                self.NA_as_zero = False
        if (not self.implicit) and (self.apply_log_transf):
            raise ValueError("Option 'apply_log_transf' only available for 'implicit=True'.")

    def __str__(self):
        msg  = "Most-Popular recommendation model\n"
        if self._implicit:
            msg += "(implicit-feedback variant)\n\n"
        else:
            msg += "(explicit-feedback variant)\n\n"
        if not self.is_fitted_:
            msg += "Model has not been fitted to data.\n"
        return msg

    def get_params(self, deep=None):
        return {
            "implicit" : self.implicit, "user_bias" : self.user_bias,
            "lambda_" : self.lambda_, "alpha" : self.alpha,
            "downweight" : self.downweight, "use_float" : self.use_float,
            "nthreads" : self.nthreads
        }

    def fit(self, X, W=None):
        """
        Fit intercepts-only model to data.

        Parameters
        ----------
        X : DataFrame(nnz, 3), DataFrame(nnz, 4), array(m, n), or sparse COO(m, n)
            Matrix to factorize (e.g. ratings). Can be passed as a SciPy
            sparse COO matrix (recommended), as a dense NumPy array, or
            as a Pandas DataFrame, in which case it should contain the
            following columns: 'UserId', 'ItemId', and either 'Rating'
            (explicit-feedback, default) or 'Value' (implicit feedback).
            If passing a NumPy array, missing (unobserved) entries should 
            have value ``np.nan`` under both explicit and implicit feedback.
            Might additionally have a column 'Weight' for the
            explicit-feedback case. If passing a DataFrame, the IDs will
            be internally remapped.
        W : None, array(nnz,), or array(m, n)
            Observation weights. Must have the same shape as 'X' - that is,
            if 'X' is a sparse COO matrix, must be a 1-d array with the same
            number of non-zero entries as 'X.data', if 'X' is a 2-d array,
            'W' must also be a 2-d array.

        Returns
        -------
        self

        """
        if (self.implicit) and (W is not None):
            raise ValueError("'W' not supported when using 'implicit=True'.")
        return self._fit_common(X, U=None, I=None, U_bin=None, I_bin=None, W=W)

    def _fit(self, Xrow, Xcol, Xval, W_sp, Xarr, W_dense,
             Uarr, Urow, Ucol, Uval, Ub_arr,
             Iarr, Irow, Icol, Ival, Ib_arr,
             m, n, m_u, n_i, p, q,
             m_ub, n_ib, pbin, qbin):
        if isinstance(self.lambda_, np.ndarray):
            lambda_user = self.lambda_[0]
            lambda_item = self.lambda_[1]
        else:
            lambda_user = self.lambda_
            lambda_item = self.lambda_

        if self.implicit and Xarr.shape[0]:
            raise ValueError("Cannot pass dense 'X' with 'implicit=True'.")

        c_funs = wrapper_float if self.use_float else wrapper_double
        self.glob_mean_, self.user_bias_, self.item_bias_, self._w_main_multiplier = \
            c_funs.call_fit_most_popular(
                Xrow,
                Xcol,
                Xval,
                W_sp,
                Xarr,
                W_dense,
                m, n,
                lambda_user, lambda_item,
                self.alpha,
                self.user_bias,
                self.implicit,
                False,
                self.scale_lam,
                self.scale_bias_const,
                self.apply_log_transf,
                self.nonneg,
                self.center,
                self.NA_as_zero,
                self.nthreads
            )

        self._A_pred = np.zeros((m,1), dtype=self.dtype_)
        self._B_pred = np.zeros((n,1), dtype=self.dtype_)
        self._n_orig = n
        self.is_fitted_ = True
        return self

    def topN(self, user=None, n=10, include=None, exclude=None, output_score=False):
        """
        Compute top-N highest-predicted items

        Parameters
        ----------
        user : int or obj
            User for which to rank the items. If 'X' passed to 'fit' was a
            DataFrame, must match with the entries in its 'UserId' column,
            otherwise should match with the rows of 'X'.
            Only relevant if using user biases and outputting score.
        n : int
            Number of top-N highest-predicted results to output.
        include : array-like
            List of items which will be ranked. If passing this, will only
            make a ranking among these items. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include or 'exclude'.
        exclude : array-like
            List of items to exclude from the ranking. If passing this, will
            rank all the items except for these. If 'X' passed to fit was a
            DataFrame, must match with the entries in its 'ItemId' column,
            otherwise should match with the columns of 'X'. Can only pass
            one of 'include or 'exclude'.
        output_score : bool
            Whether to output the scores in addition to the IDs. If passing
            'False', will return a single array with the item IDs, otherwise
            will return a tuple with the item IDs and the scores.

        Returns
        -------
        items : array(n,)
            The top-N highest predicted items. If the 'X' data passed to
            fit was a DataFrame, will contain the item IDs from its column
            'ItemId', otherwise will be integers matching to the columns of 'X'.
        scores : array(n,)
            The predicted scores for the top-N items. Will only be returned
            when passing ``output_score=True``, in which case the result will
            be a tuple with these two entries.
        """
        if (user is not None) and (not self.user_bias):
            warnings.warn("Passing user is not meaningful without user biases.")

        if (user is not None) and (self.user_bias):
            return self._topN(user=user, a_vec=None, a_bias=None, n=n,
                              include=include, exclude=exclude,
                              output_score=output_score)
        else:
            return self._topN(user=None,
                              a_vec=np.zeros(1, dtype=self.dtype_),
                              a_bias=0., n=n,
                              include=include, exclude=exclude,
                              output_score=output_score)

class CMF_imputer(CMF):
    """
    A wrapper for CMF allowing argument 'y' in 'fit' and
    'transform' (used as a placeholder only, not used for anything),
    which can be used as part of SciKit-Learn pipelines due to having
    this extra parameter.

    Everything else is exactly the same as for 'CMF'
    """
    def fit(self, X, y=None, U=None, I=None, U_bin=None, I_bin=None, W=None):
        return super().fit(X=X, U=U, U_bin=U_bin, I=I, I_bin=I_bin, W=W)
