from . import wrapper_double, wrapper_float
import numpy as np, pandas as pd
import multiprocessing
import ctypes
import warnings

class _CMF:
    def __init__(self):
        pass

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

    def _take_params(self, implicit=False, alpha=40., downweight=True,
                     k=50, lambda_=1e2, method="als", use_cg=False,
                     user_bias=True, item_bias=True, k_user=0, k_item=0, k_main=0,
                     w_main=1., w_user=1., w_item=1.,
                     maxiter=400, niter=10, parallelize="separate", corr_pairs=4,
                     NA_as_zero=False, NA_as_zero_user=False, NA_as_zero_item=False,
                     precompute_for_predictions=True, use_float=False,
                     random_state=1, init="normal", verbose=True, print_every=10,
                     produce_dicts=False, copy_data=True, nthreads=-1):
        assert method in ["als", "lbfgs"]
        assert parallelize in ["separate", "single"]
        assert init in ["normal", "gamma"]

        k = int(k) if isinstance(k, float) else k
        k_user = int(k_user) if isinstance(k_user, float) else k_user
        k_item = int(k_item) if isinstance(k_item, float) else k_item
        k_main = int(k_main) if isinstance(k_main, float) else k_main
        assert isinstance(k, int) and k > 0
        assert isinstance(k_user, int) and k_user >= 0
        assert isinstance(k_item, int) and k_item >= 0
        assert isinstance(k_main, int) and k_main >= 0

        lambda_ = float(lambda_) if isinstance(lambda_, int) else lambda_
        lambda_ = np.array(lambda_) if lambda_.__class__.__name__ in ["list", "Series", "tuple"] else lambda_
        if lambda_.__class__.__name__ == "ndarray":
            lambda_ = lambda_.reshape(-1)
            assert lambda_.shape[0] == 6
            assert np.all(lambda_ >= 0)
        else:
            assert isinstance(lambda_, float) and lambda_ >= 0

        
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

        if random_state.__class__.__name__ == "RandomState":
            random_state = random_state.randint(np.iinfo(np.int32).max)

        if (method == "lbfgs") and (NA_as_zero or NA_as_zero_user or NA_as_zero_item):
            raise ValueError("Option 'NA_as_zero' not supported with method='lbfgs'.")

        w_main = float(w_main) if isinstance(w_main, int) else w_main
        w_user = float(w_user) if isinstance(w_user, int) else w_user
        w_item = float(w_item) if isinstance(w_item, int) else w_item
        assert isinstance(w_main, float) and w_main > 0
        assert isinstance(w_user, float) and w_user > 0
        assert isinstance(w_item, float) and w_item > 0

        if implicit:
            alpha = float(alpha) if isinstance(alpha, int) else alpha
            assert isinstance(alpha, float) and alpha > 0.

        if NA_as_zero and (user_bias or item_bias):
            raise ValueError("Biases are not supported when using 'NA_as_zero'.")

        self.k = k
        self.k_user = k_user
        self.k_item = k_item
        self.k_main = k_main
        self.lambda_ = lambda_
        self.alpha = alpha
        self.w_main = w_main
        self.w_user = w_user
        self.w_item = w_item
        self.downweight = bool(downweight)
        self.user_bias = bool(user_bias)
        self.item_bias = bool(item_bias)
        self.method = method
        self.use_cg = bool(use_cg)
        self.maxiter = maxiter
        self.niter = niter
        self.parallelize = parallelize
        self.NA_as_zero = bool(NA_as_zero)
        self.NA_as_zero_user = bool(NA_as_zero_user)
        self.NA_as_zero_item = bool(NA_as_zero_item)
        self.precompute_for_predictions = bool(precompute_for_predictions)
        self.use_float = bool(use_float)
        self.init = init
        self.verbose = bool(verbose)
        self.print_every = print_every
        self.corr_pairs = corr_pairs
        self.random_state = int(random_state)
        self.produce_dicts = bool(produce_dicts)
        self.copy_data = bool(copy_data)
        self.nthreads = nthreads

        self._implicit = bool(implicit)
        self.dtype_ = ctypes.c_float if use_float else ctypes.c_double
        if self.use_float:
            self.c_funs = wrapper_float
        else:
            self.c_funs = wrapper_double

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
        self.user_bias_ = np.empty(0, dtype=self.dtype_)
        self.item_bias_ = np.empty(0, dtype=self.dtype_)
        self.C_bias_ = np.empty(0, dtype=self.dtype_)
        self.D_bias_ = np.empty(0, dtype=self.dtype_)
        self.glob_mean_ = 0.

        self._BtBinvBt = np.empty((0,0), dtype=self.dtype_)
        self._BtB = np.empty((0,0), dtype=self.dtype_)
        self._BtBchol = np.empty((0,0), dtype=self.dtype_)
        self._CtCinvCt = np.empty((0,0), dtype=self.dtype_)
        self._CtC = np.empty((0,0), dtype=self.dtype_)
        self._CtCchol = np.empty((0,0), dtype=self.dtype_)
        self._BeTBe = np.empty((0,0), dtype=self.dtype_)
        self._BtB_padded = np.empty((0,0), dtype=self.dtype_)
        self._BtB_shrunk = np.empty((0,0), dtype=self.dtype_)

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
        self.nfev_ = None
        self.nupd_ = None
        self.user_mapping_ = np.array([], dtype=object)
        self.item_mapping_ = np.array([], dtype=object)
        self.reindex_ = None
        self.user_dict_ = dict()
        self.item_dict_ = dict()

    def _take_params_offsets(self, k_sec=0, k_main=0, add_intercepts=True):
        k_sec = int(k_sec) if isinstance(k_sec, float) else k_sec
        k_main = int(k_main) if isinstance(k_main, float) else k_main
        assert isinstance(k_sec, int) and k_sec >= 0
        assert isinstance(k_main, int) and k_main >= 0

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
        Ucsr_p = np.empty(0, dtype=ctypes.c_long)
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
            Ucsr_p = U.indptr.astype(ctypes.c_long)
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
        Xcsr_p = np.empty(0, dtype=ctypes.c_long)
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
            Xcsr_p = X.indptr.astype(ctypes.c_long)
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
                    raise ValueError(msg % ("include", "users", "rows"))
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
                msg += "  Weight should be under a column in the DataFrame, "
                msg += "called 'Weight'."
                raise ValueError(msg)

            assert "UserId" in X.columns.values
            assert "ItemId" in X.columns.values
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

        if enforce_same_shape:
            if Uarr.shape[0]:
                if Uarr.shape[0] != m:
                    raise ValueError("'X' and 'U' must have the same rows.")
            if Iarr.shape[0]:
                if Iarr.shape[0] != n:
                    raise ValueError("Columns of 'X' must match with rows of 'I'.")

        return self._fit(Xrow, Xcol, Xval, W_sp, Xarr, W_dense,
                         Uarr, Urow, Ucol, Uval, Ub_arr,
                         Iarr, Irow, Icol, Ival, Ib_arr,
                         m, n, m_u, n_i, p, q,
                         m_ub, n_ib, pbin, qbin)

    def predict(self, user, item):
        """
        Predict ratings/values given by existing users to existing items

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

        user, item, _1, _2 = self._process_users_items(user, item, None, None)

        if user is not None:
            assert user.shape[0] == item.shape[0]
        
            if user.shape[0] == 1:
                if (user[0] == -1) or (item[0] == -1):
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
                nan_entries = (user < 0) | (item < 0)
                if ~np.any(nan_entries):
                    return self.c_funs.call_predict_multiple(
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
                    non_na_user = (user >= 0)
                    non_na_item = (item >= 0)
                    outp = self.c_funs.call_predict_multiple(
                        self._A_pred,
                        self._B_pred,
                        self.user_bias_,
                        self.item_bias_,
                        self.glob_mean_,
                        np.where(non_na_user, np.array(user).astype(ctypes.c_int), np.zeros(user.shape[0], dtype=ctypes.c_int)),
                        np.where(non_na_item, np.array(item).astype(ctypes.c_int), np.zeros(item.shape[0], dtype=ctypes.c_int)),
                        self._k_pred, self.k_user, self.k_item, self._k_main_col,
                        self.nthreads
                    )
                    outp[nan_entries] = np.nan
                    return outp
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
        nan_entries = (user < 0)

        if user.shape[0] != n:
            raise ValueError("'user' must have the same number of entries as item info.")

        if ~np.any(nan_entries):
            return self.c_funs.call_predict_multiple(
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
        else:
            non_na_user = ~nan_entries
            outp = self.c_funs.call_predict_multiple(
                        self._A_pred,
                        B,
                        self.user_bias_,
                        np.zeros(n, dtype=self.dtype_) if self.item_bias \
                            else np.empty(0, dtype=self.dtype_),
                        self.glob_mean_,
                        np.array(user).astype(ctypes.c_int),
                        np.where(non_na_user,
                                 np.arange(n).astype(ctypes.c_int),
                                 np.zeros(n, dtype=ctypes.c_int)),
                        self._k_pred, self.k_user, self.k_item, self._k_main_col,
                        self.nthreads
                    )
            outp[nan_entries] = np.nan
            return outp

    def _predict_user_multiple(self, A, item, bias=None):
        m = A.shape[0]
        _1, item, _2, _3 = self._process_users_items(None, item, None, None)
        nan_entries = (item < 0)

        if item.shape[0] != m:
            raise ValueError("'item' must have the same number of entries as user info.")

        if bias is None:
            bias = np.zeros(m, dtype=self.dtype_) if self.user_bias \
                        else np.empty(0, dtype=self.dtype_)

        if ~np.any(nan_entries):
            return self.c_funs.call_predict_multiple(
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
            non_na_item = ~nan_entries
            outp = self.c_funs.call_predict_multiple(
                        A,
                        self._B_pred,
                        bias,
                        self.item_bias_,
                        self.glob_mean_,
                        np.where(non_na_item,
                                 np.arange(m).astype(ctypes.c_int),
                                 np.zeros(m, dtype=ctypes.c_int)),
                        np.array(item).astype(ctypes.c_int),
                        self._k_pred, self.k_user, self.k_item, self._k_main_col,
                        self.nthreads
                    )
            outp[nan_entries] = np.nan
            return outp

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
        user, _, include, exclude = self._process_users_items(user, None, include, exclude)

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
        outp_ix, outp_score = self.c_funs.call_topN(
            a_vec,
            self._B_pred if B is None else B,
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

        U, U_col, U_val, U_bin = self._process_new_U(U, U_col, U_val, U_bin)

        if isinstance(self.lambda_, np.ndarray):
            lambda_ = self.lambda_[2]
        else:
            lambda_ = self.lambda_
        
        a_vec = self.c_funs.call_factors_collective_cold(
            U,
            U_val,
            U_col,
            U_bin,
            self.C_,
            self.Cbin_,
            self._CtCinvCt,
            self._CtC,
            self._CtCchol,
            self._U_colmeans,
            self.C_.shape[0], self.k,
            self.k_user, self.k_main,
            lambda_, self.w_user,
            self.NA_as_zero_user
        )
        return a_vec

    def _factors_warm_common(self, X=None, X_col=None, X_val=None, W=None,
                             U=None, U_bin=None, U_col=None, U_val=None,
                             return_bias=False):
        assert self.is_fitted_

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
            if X.shape[0] != self.B_.shape[0]:
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

            if self.reindex_:
                X_col = np.array(X_col).reshape(-1)
                X_col = pd.Categorical(X_col, self.item_mapping_).codes.astype(ctypes.c_int)
                if np.any(X_col < 0):
                    raise ValueError("'X_col' must have the same item/column entries as passed to 'fit'.")
            else:
                X_col = np.array(X_col).reshape(-1).astype(ctypes.c_int)
                imin, imax = np.min(X_col), np.max(X_col)
                if (imin < 0) or (imax >= self.B_.shape[0]) or np.isnan(imin) or np.isnan(imax):
                    msg  = "Column indices ('X_col') must be within the range"
                    msg += " of the data that was pased to 'fit'."
                    raise ValueError(msg)

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

        return self._factors_warm(X, W_dense, X_val, X_col, W_sp,
                                  U, U_val, U_col, U_bin, return_bias)

    def _process_transform_inputs(self, X, U, U_bin, W, replace_existing):
        if (X is None) and (U is None) and (U_bin):
            raise ValueError("Must pass at least one of 'X', 'U', 'U_bin'.")
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
            raise ValueError(msg % "U")
        if (m_x > 0) and (m_ub > 0) and (m_x != m_ub):
            raise ValueError(msg % "U_bin")

        if isinstance(self.lambda_, np.ndarray):
            lambda_ = self.lambda_[2]
            lambda_bias = self.lambda_[0]
        else:
            lambda_ = self.lambda_
            lambda_bias = self.lambda_

        return Xrow, Xcol, Xval, W_sp, Xarr, \
               Xcsr_p, Xcsr_i, Xcsr, \
               W_dense, Xorig, mask_take, \
               Uarr, Urow, Ucol, Uval, Ub_arr, \
               Ucsr_p, Ucsr_i, Ucsr, \
               n, m_u, m_x, p, pbin, \
               lambda_, lambda_bias

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

    def _item_factors_cold(self, I=None, I_bin=None, I_col=None, I_val=None):
        assert self.is_fitted_
        if (self.D_.shape[0] == 0) and (self.Dbin_.shape[0] == 0):
            msg  = "Can only use this method when "
            msg += "fitting the model to item side info."
            raise ValueError(msg)

        if isinstance(self.lambda_, np.ndarray):
            lambda_ = self.lambda_[3]
        else:
            lambda_ = self.lambda_

        I, I_col, I_val, I_bin = self._process_new_U(U=I, U_col=I_col, U_val=I_val, U_bin=I_bin, is_I=True)

        b_vec = self.c_funs.call_factors_collective_cold(
            I,
            I_val,
            I_col,
            I_bin,
            self.D_,
            self.Dbin_,
            np.empty((0,0), dtype=self.dtype_),
            np.empty((0,0), dtype=self.dtype_),
            np.empty((0,0), dtype=self.dtype_),
            self._I_colmeans,
            self.D_.shape[0], self.k,
            self.k_item, self.k_main,
            lambda_, self.w_item,
            self.NA_as_zero_item
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
            else:
                lambda_ = self.lambda_[3]
        else:
            lambda_ = self.lambda_

        Uarr, Urow, Ucol, Uval, Ucsr_p, Ucsr_i, Ucsr, m_u, p = \
            self._process_new_U_2d(U=U, is_I=is_I, allow_csr=True)
        Ub_arr, m_ub, pbin = self._process_new_Ub_2d(U_bin=U_bin, is_I=is_I)

        A = self.c_funs.call_collective_factors_cold_multiple(
            Uarr,
            Urow,
            Ucol,
            Uval,
            Ucsr_p, Ucsr_i, Ucsr,
            Ub_arr,
            Mat,
            MatBin,
            np.empty((0,0), dtype=self.dtype_),
            np.empty((0,0), dtype=self.dtype_),
            np.empty((0,0), dtype=self.dtype_),
            self._U_colmeans if not is_I else self._I_colmeans,
            m_u, m_ub,
            self.k, self.k_user if not is_I else self.k_item, self.k_main,
            lambda_, self.w_user if not is_I else self.w_item,
            self.NA_as_zero_user if not is_I else self.NA_as_zero_item,
            self.nthreads
        )
        return A


class CMF_explicit(_CMF):
    """
    Collective model for explicit-feedback data

    Tries to approximate the 'X' interactions matrix  by a formula as follows:
    X ~ A * t(B)
    While at the same time also approximating the user side information
    matrix 'U' and the item side information matrix 'I' as follows:
    U ~ A * t(C)
    I ~ B * t(D)
    Might apply sigmoid transformations to binary columns in U and I too.

    Parameters
    ----------
    k : int
        Number of latent factors to use (dimensionality of the low-rank
        factorization), which will be shared between the factorization of the
        'X' matrix and the side info matrices. Additional non-shared components
        can also be specified through ``k_user``, ``k_item``, and ``k_main``.
    lambda_ : float or array(6,)
        Regularization parameter. Can also use different regulatization for each
        matrix, in which case it should be an array with 6 entries, corresponding,
        in this order, to: user_bias, item_bias, A, B, C, D.
    method : str, one of "lbfgs" or "als"
        Optimization method used to fit the model. If passing ``'lbfgs'``, will
        fit it through a gradient-based approach using an L-BFGS optimizer.
        L-BFGS is typically a much slower and a much less memory efficient method
        compared to ``'als'``, but tends to reach better local optima and allows
        some variations of the problem which ALS doesn't, such as applying sigmoid
        transformations for binary side information.
    user_bias : bool
        Whether to add user biases (intercepts) to the model.
    item_bias : bool
        Whether to add item biases (intercepts) to the model. Be aware that using
        item biases with low regularization for them will tend to favor items
        with high average ratings regardless of the number of ratings the item
        has received.
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
    maxiter : int
        Maximum L-BFGS iterations to perform. The procedure will halt if it
        has not converged after this number of updates. Note that, compared to
        the ohter models, fewer iterations will be required for converge
        here. Using higher regularization values might also decrease the number
        of required iterations. Pass zero for no L-BFGS iterations limit.
        If you see that the procedure is spending hundreds of iterations
        without any significant decrease in the loss function or gradient norm,
        it's highly likely that the regularization is too low.
        Ignored when passing ``method='als'``.
    niter : int
        Number of alternating least-squares iterations to perform. Note that
        one iteration denotes an update round for all the matrices rather than
        an update of a single matrix. Ignored when passing ``method='lbfgs'``.
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
        Number of correction paris to use for the L-BFGS optimization routine.
        Recommended values are between 3 and 7. Note that higher values
        translate into higher memory requirements. Ignored when passing
        ``method='als'``.
    NA_as_zero : bool
        Whether to take missing entries in the 'X' matrix as zeros (only
        when the 'X' matrix is passed as sparse COO matrix or DataFrame)
        instead of ignoring them. Note that this is a different model from the
        implicit-feedback version with weighted entries, and it's a much faster
        model to fit.
    NA_as_zero_user : bool
        Whether to take missing entries in the 'U' matrix as zeros (only
        when the 'U' matrix is passed as sparse COO matrix) instead of ignoring them.
    NA_as_zero_item : bool
        Whether to take missing entries in the 'I' matrix as zeros (only
        when the 'I' matrix is passed as sparse COO matrix) instead of ignoring them.
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
    use_cg : bool
        Whether to use a conjugate gradient method to solve the closed-form
        least squares problems. This was implemented for experimentation
        purposes only - will not provide any advantage over the default
        Cholesky solver.
    random_state : int or RandomState
        Seed used to initialize parameters at random. If passing a NumPy
        RandomState object, will use it to draw a random integer. Note
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
    """
    def __init__(self, k=50, lambda_=1e1, method="als",
                 user_bias=True, item_bias=True, k_user=0, k_item=0, k_main=0,
                 w_main=1., w_user=1., w_item=1.,
                 maxiter=400, niter=10, parallelize="separate", corr_pairs=4,
                 NA_as_zero=False, NA_as_zero_user=False, NA_as_zero_item=False,
                 precompute_for_predictions=True, use_float=False, use_cg=False,
                 random_state=1, verbose=True, print_every=10,
                 produce_dicts=False, copy_data=True, nthreads=-1):
        self._take_params(implicit=False, alpha=0., downweight=False,
                          k=k, lambda_=lambda_, method=method, use_cg=use_cg,
                          user_bias=user_bias, item_bias=item_bias,
                          k_user=k_user, k_item=k_item, k_main=k_main,
                          w_main=w_main, w_user=w_user, w_item=w_item,
                          maxiter=maxiter, niter=niter, parallelize=parallelize,
                          corr_pairs=corr_pairs,
                          NA_as_zero=NA_as_zero, NA_as_zero_user=NA_as_zero_user,
                          NA_as_zero_item=NA_as_zero_item,
                          precompute_for_predictions=precompute_for_predictions,
                          use_float=use_float,
                          random_state=random_state, init="normal",
                          verbose=verbose, print_every=print_every,
                          produce_dicts=produce_dicts, copy_data=copy_data,
                          nthreads=nthreads)

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
        In this case, the overlapping entries should come first in both
        matrices. If there are entries in e.g. 'U' or 'I' which 'X' doesn't have,
        and at the same time, entries in 'X' which 'U' or 'I' don't have,
        one of the matrices should be appended missing entries (``np.nan`` for
        dense arrays, shapes for COO matrices). This is done internally when
        passing data frames with 'UserId' and 'ItemId'.

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
        y : None
            Not used. Kept as a place-holder for compatibility with SciKit-Learn
            pipelines.
        U : array(m, p), COO(m, p), DataFrame(m, p+1), or None
            User attributes information. If 'X' is a DataFrame, should also
            be a DataFrame, containing column 'UserId'. If 'U' is sparse,
            'X' should be passed as a sparse COO matrix too.
            Might contain missing values.
        U_bin : array(m, p_bin), DataFrame(m, p_bin+1), or None
            User binary attributes information (all values should be zero, one,
            or missing). If 'X' is a DataFrame, should also
            be a DataFrame, containing column 'UserId'. Cannot be passed
            as a sparse matrix.
            Might contain missing values.
            Note that 'U' and 'U_bin' are not mutually exclusive.
            Only supported with ``method='lbfgs'``.
        I : array(n, q), COO(n, q), DataFrame(n, q+1), or None
            Item attributes information. If 'X' is a DataFrame, should also
            be a DataFrame, containing column 'ItemId'. If 'I' is sparse,
            'X' should be passed as a sparse COO matrix too.
            Might contain missing values.
        I_bin : array(n, q_bin), DataFrame(n, q_bin+1), or None
            Item binary attributes information (all values should be zero, one,
            or missing). If 'X' is a DataFrame, should also
            be a DataFrame, containing column 'ItemId'. Cannot be passed
            as a sparse matrix.
            Might contain missing values.
            Note that 'I' and 'I_bin' are not mutually exclusive.
            Only supported with ``method='lbfgs'``.
        W : None, array(nnz,), or array(m, n)
            Observation weights. Must have the same shape as 'X' - that is,
            if 'X' is a sparse COO matrix, must be a 1-d array with the same
            number of non-zero entries as 'X.data', if 'X' is a 2-d array,
            'W' must also be a 2-d array.

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
        if self.method == "lbfgs":
            self.glob_mean_,  self._U_colmeans, self._I_colmeans, values, self.nupd_, self.nfev_, self._B_plus_bias = \
                self.c_funs.call_fit_collective_explicit_lbfgs(
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
                    self.user_bias, self.item_bias,
                    self.lambda_ if isinstance(self.lambda_, float) else 0.,
                    self.lambda_ if isinstance(self.lambda_, np.ndarray) else np.empty(0, dtype=self.dtype_),
                    self.verbose, self.print_every,
                    self.corr_pairs, self.maxiter,
                    self.nthreads, self.parallelize != "separate",
                    self.random_state
                )
            self.user_bias_, self.item_bias_, self.A_, self.B_, self.C_, self.Cbin_, self.D_, self.Dbin_ = \
                self.c_funs.unpack_values_lbfgs_collective(
                    values,
                    self.user_bias, self.item_bias,
                    self.k, self.k_user, self.k_item, self.k_main,
                    m, n, p, q,
                    pbin, qbin,
                    m_u, n_i, m_ub, n_ib
                )
        else:
            self.glob_mean_,  self._U_colmeans, self._I_colmeans, values, self._B_plus_bias = \
                self.c_funs.call_fit_collective_explicit_als(
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
                    self.w_main, self.w_user, self.w_item,
                    self.user_bias, self.item_bias,
                    self.lambda_ if isinstance(self.lambda_, float) else 0.,
                    self.lambda_ if isinstance(self.lambda_, np.ndarray) else np.empty(0, dtype=self.dtype_),
                    self.verbose, self.nthreads, self.use_cg,
                    self.random_state, self.niter
                )
            self.user_bias_, self.item_bias_, self.A_, self.B_, self.C_, self.D_ = \
                self.c_funs.unpack_values_collective_als(
                    values,
                    self.user_bias, self.item_bias,
                    self.k, self.k_user, self.k_item, self.k_main,
                    m, n, p, q,
                    m_u, n_i
                )

        self._A_pred = self.A_
        self._B_pred = self.B_
        self.is_fitted_ = True
        if self.precompute_for_predictions:
            self.force_precompute_for_predictions()
        return self

    def predict_cold(self, items, U=None, U_bin=None, U_col=None, U_val=None):
        """
        Predict rating given by a new user to existing items, given U

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
            in ``item``. Might contain missing values.
        U_bin : array(m, p_bin), or None
            Binary attributes for the users to predict ratings/values.
            Data frames with 'UserId'
            column are not supported. Must have one row per entry
            in ``user``. Might contain missing values.
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

        a_bias, a_vec = self.c_funs.call_factors_collective_warm_explicit(
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
            self._BtBinvBt,
            self._BtB,
            self._BtBchol,
            self._CtC,
            self.glob_mean_,
            self.k, self.k_user, self.k_item, self.k_main,
            lambda_, lambda_bias,
            self.w_user, self.w_main,
            self.user_bias,
            self.NA_as_zero_user, self.NA_as_zero
        )

        if return_bias:
            return a_vec, a_bias
        else:
            return a_vec


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
            Missing entries should have value ``np.nan`` when passing a dense
            array.
        U_bin : array(m, p_bin)
            User binary attributes for each row in 'X'.
            Missing entries should have value ``np.nan``.
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
        Xrow, Xcol, Xval, W_sp, Xarr, \
        Xcsr_p, Xcsr_i, Xcsr, \
        W_dense, Xorig, mask_take, \
        Uarr, Urow, Ucol, Uval, Ub_arr, \
        Ucsr_p, Ucsr_i, Ucsr, \
        n, m_u, m_x, p, pbin, \
        lambda_, lambda_bias = \
            self._process_transform_inputs(X=X, U=U, U_bin=U_bin, W=W,
                                           replace_existing=True)
        A, A_bias = self.c_funs.call_collective_factors_warm_multiple(
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
                self.C_,
                self.Cbin_,
                self._BtBinvBt,
                self._BtB,
                self._CtCinvCt,
                self._CtC,
                self._CtCchol,
                n, m_u, m_x,
                self.glob_mean_,
                self._k_pred, self.k_user, self.k_item, self._k_main_col,
                lambda_, lambda_bias,
                self.w_user, self.w_main,
                self.user_bias,
                self.NA_as_zero_user, self.NA_as_zero,
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
        Reconstruct entries of the 'X' matrix

        Will reconstruct all the entries in the 'X' matrix as determined
        by the model. This method is intended to be used for imputing tabular
        data, and can be used as part of SciKit-Learn pipelines.

        Note
        ----
        It's possible to use this method with 'X' alone, with 'U'/'U_bin'
        alone, or with both 'X' and 'U'/'U_bin' together, in which case
        both matrices must have the same rows.

        Parameters
        ----------
        X : array(m, n), CSR matrix(m, n), or COO matrix(m, n)
            New 'X' data with potentially missing entries which are to be imputed.
            Missing entries should have value ``np.nan`` when passing a dense
            array.
        y : None
            Not used. Kept as a placeholder for compatibility with SciKit-Learn
            pipelines.
        U : array(m, p), CSR matrix(m, p), COO matrix(m, p), or None
            User attributes information for each row in 'X'.
            Missing entries should have value ``np.nan`` when passing a dense
            array.
        U_bin : array(m, p_bin)
            User binary attributes for each row in 'X'.
            Missing entries should have value ``np.nan``.
            Only supported with ``method='lbfgs'``.
        W : array(m, n), array(nnz,), or None
            Observation weights. Must have the same shape as 'X' - that is,
            if 'X' is a sparse COO matrix, must be a 1-d array with the same
            number of non-zero entries as 'X.data', if 'X' is a 2-d array,
            'W' must also be a 2-d array.
        replace_existing : bool
            Whether to replace existing non-missing entries in 'X' with the
            model predictions - that is, if passing 'False', will only fill
            in the missing entries in 'X', leaving the non-missing entries
            as they were.

        Returns
        -------
        X : array(m, n)
            The 'X' matrix as a dense array with all entries as determined by
            the model. Note that this will be returned as a dense NumPy array.
        """
        Xrow, Xcol, Xval, W_sp, Xarr, \
        Xcsr_p, Xcsr_i, Xcsr, \
        W_dense, Xorig, mask_take, \
        Uarr, Urow, Ucol, Uval, Ub_arr, \
        Ucsr_p, Ucsr_i, Ucsr, \
        n, m_u, m_x, p, pbin, \
        lambda_, lambda_bias = \
            self._process_transform_inputs(X=X, U=U, U_bin=U_bin, W=W,
                                           replace_existing=replace_existing)

        if (Xarr.shape[0] == 0) and (Xval.shape[0] == 0):
            A = self.c_funs.call_collective_factors_cold_multiple(
                Uarr,
                Urow,
                Ucol,
                Uval,
                Ucsr_p, Ucsr_i, Ucsr,
                Ub_arr,
                self.C_,
                self.Cbin_,
                self._CtCinvCt,
                self._CtC,
                self._CtCchol,
                self._U_colmeans,
                m_u, m_ubin,
                self._k_pred, self.k_user, self._k_main_col,
                lambda_, self.w_user,
                self.NA_as_zero_user,
                self.nthreads
            )
            A_bias = np.empty(0, dtype=self.dtype_)
        else:
            A, A_bias = self.c_funs.call_collective_factors_warm_multiple(
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
                self.C_,
                self.Cbin_,
                self._BtBinvBt,
                self._BtB,
                self._CtCinvCt,
                self._CtC,
                self._CtCchol,
                n, m_u, m_x,
                self.glob_mean_,
                self._k_pred, self.k_user, self.k_item, self._k_main_col,
                lambda_, lambda_bias,
                self.w_user, self.w_main,
                self.user_bias,
                self.NA_as_zero_user, self.NA_as_zero,
                self.nthreads
            )

        return self._transform_step(A, A_bias, mask_take, Xorig)

    def force_precompute_for_predictions(self):
        """
        Precompute internal matrices that are used for predictions

        Note
        ----
        You don't need to call this method if passing
        ``precompute_for_predictions=True``.

        Returns
        -------
        self
        
        """
        assert self.is_fitted_
        if isinstance(self.lambda_, np.ndarray):
            lambda_ = self.lambda_[2]
            lambda_bias = self.lambda_[0]
        else:
            lambda_ = self.lambda_
            lambda_bias = self.lambda_
        self._BtBinvBt, self._BtB, self._BtBchol, self._CtCinvCt, self._CtC, self._CtCchol = \
            self.c_funs.precompute_matrices_collective_explicit(
                self.B_,
                self._B_plus_bias,
                self.k, self.k_main, self.k_user, self.k_item,
                self.C_,
                lambda_, lambda_bias, self.w_main, self.w_user,
                self.C_.shape[0]>0, self.Cbin_.shape[0]>0
            )
        return self

class CMF_implicit(_CMF):
    """
    Collective model for implicit-feedback data

    Tries to approximate the 'X' interactions matrix  by a formula as follows:
    X ~ A * t(B)
    While at the same time also approximating the user side information
    matrix 'U' and the item side information matrix 'I' as follows:
    U ~ A * t(C)
    I ~ B * t(D)

    Note
    ----
    This model is fit through the alternating least-squares method only,
    it does not offer a gradient-based approach like the explicit-feedback
    version.

    Parameters
    ----------
    k : int
        Number of latent factors to use (dimensionality of the low-rank
        factorization), which will be shared between the factorization of the
        'X' matrix and the side info matrices. Additional non-shared components
        can also be specified through ``k_user``, ``k_item``, and ``k_main``.
    lambda_ : float or array(6,)
        Regularization parameter. Can also use different regulatization for each
        matrix, in which case it should be an array with 6 entries, corresponding,
        in this order, to: user_bias, item_bias, A, B, C, D.
    alpha : float
        Weighting parameter for the non-zero entries in the implicit-feedback
        model. See [3] for details.
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
    w_user : float
        Weight in the optimization objective for the errors in the factorization
        of the 'U' matrix. Ignored when not passing 'U' to 'fit'.
    w_item : float
        Weight in the optimization objective for the errors in the factorization
        of the 'I' matrix. Ignored when not passing 'I' to 'fit'.
    downweight : bool
        Whether to decrease the weight of the 'X' matrix being factorized
        according to the number of present entries. This has the same effect
        as rescaling (increasing) the regularization parameter for the A and B
        matrices while increasing ``w_user`` and ``w_item``.
    niter : int
        Number of alternating least-squares iterations to perform. Note that
        one iteration denotes an update round for all the matrices rather than
        an update of a single matrix.
    NA_as_zero_user : bool
        Whether to take missing entries in the 'U' matrix as zeros (only
        when the 'U' matrix is passed as sparse COO matrix) instead of ignoring them.
    NA_as_zero_item : bool
        Whether to take missing entries in the 'I' matrix as zeros (only
        when the 'I' matrix is passed as sparse COO matrix) instead of ignoring them.
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
    use_cg : bool
        Whether to use a conjugate gradient method to solve the closed-form
        least squares problems. This was implemented for experimentation
        purposes only - will not provide any advantage over the default
        Cholesky solver.
    random_state : int or RandomState
        Seed used to initialize parameters at random. If passing a NumPy
        RandomState object, will use it to draw a random integer.
    init : str, "normal" or "gamma"
        Distribution used to initialize the model parameters. Both
        distributions will reach similar end results, but the distribution
        of the factors themselves will be slightly different.
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
    """
    def __init__(self, k=50, lambda_=1e3, alpha=40.,
                 k_user=0, k_item=0, k_main=0,
                 w_main=1., w_user=1., w_item=1., downweight=True,
                 niter=10, NA_as_zero_user=False, NA_as_zero_item=False,
                 precompute_for_predictions=True, use_float=False, use_cg=False,
                 random_state=1, init="normal", verbose=False,
                 produce_dicts=False, copy_data=True, nthreads=-1):
        self._take_params(implicit=True, alpha=alpha, downweight=downweight,
                          k=k, lambda_=lambda_, method="als", use_cg=use_cg,
                          user_bias=False, item_bias=False,
                          k_user=k_user, k_item=k_item, k_main=k_main,
                          w_main=w_main, w_user=w_user, w_item=w_item,
                          maxiter=0, niter=niter, parallelize="separate",
                          corr_pairs=0,
                          NA_as_zero=False, NA_as_zero_user=NA_as_zero_user,
                          NA_as_zero_item=NA_as_zero_item,
                          precompute_for_predictions=precompute_for_predictions,
                          use_float=use_float,
                          random_state=random_state, init=init,
                          verbose=verbose, print_every=0,
                          produce_dicts=produce_dicts, copy_data=copy_data,
                          nthreads=nthreads)

    def __str__(self):
        msg  = "Collective matrix factorization model\n"
        msg += "(implicit-feedback variant)\n\n"
        if not self.is_fitted_:
            msg += "Model has not been fitted to data.\n"
        return msg

    def get_parms(self, deep=None):
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
        matrices to have rows that the other doesn't have).
        In this case, the overlapping entries should come first in both
        matrices. If there are entries in 'U' or 'I' which 'X' doesn't have,
        and at the same time, entries in 'X' which 'U' or 'I' don't have,
        one of the matrices should be appended missing entries (``np.nan`` for
        dense arrays, shapes for COO matrices). This is done internally when
        passing data frames with 'UserId' and 'ItemId'.

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
            Might contain missing values.
        I : array(n, q), COO(n, q), DataFrame(n, q+1), or None
            Item attributes information. If 'X' is a DataFrame, should also
            be a DataFrame, containing column 'ItemId'. If 'I' is sparse,
            'X' should be passed as a sparse COO matrix too.
            Might contain missing values.

        Returns
        -------
        self

        """
        if X.__class__.__name__ not in ("coo_matrix", "DataFrame"):
            raise ValueError("'X' must be a Pandas DataFrame or SciPy sparse COO matrix.")
        return self._fit_common(X, U=U, I=I, U_bin=None, I_bin=None, W=None)

    def _fit(self, Xrow, Xcol, Xval, W_sp, Xarr, W_dense,
             Uarr, Urow, Ucol, Uval, Ub_arr,
             Iarr, Irow, Icol, Ival, Ib_arr,
             m, n, m_u, n_i, p, q,
             m_ub, n_ib, pbin, qbin):
        self._U_colmeans, self._I_colmeans, values, self._w_main_multiplier = \
            self.c_funs.call_fit_collective_implicit_als(
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
                self.alpha, self.downweight,
                self.lambda_ if isinstance(self.lambda_, np.ndarray) else np.empty(0, dtype=self.dtype_),
                self.verbose, self.niter,
                self.nthreads, self.use_cg, self.random_state,
                init=self.init
            )

        self.A_, self.B_, self.C_, self.D_ = \
            self.c_funs.unpack_values_collective_implicit(
                values,
                self.k, self.k_user, self.k_item, self.k_main,
                m, n, p, q,
                m_u, n_i
            )

        self._A_pred = self.A_
        self._B_pred = self.B_
        self.is_fitted_ = True
        if self.precompute_for_predictions:
            self.force_precompute_for_predictions()
        return self

    def force_precompute_for_predictions(self):
        """
        Precompute internal matrices that are used for predictions

        Note
        ----
        You don't need to call this method if passing
        ``precompute_for_predictions=True``.

        Returns
        -------
        self

        """
        assert self.is_fitted_
        if isinstance(self.lambda_, np.ndarray):
            lambda_ = self.lambda_[2]
        else:
            lambda_ = self.lambda_
        self._BeTBe, self._BtB_padded, self._BtB_shrunk, self._CtCinvCt, self._CtC, self._CtCchol = \
            self.c_funs.precompute_matrices_collective_implicit(
                self.B_,
                self.k, self.k_main, self.k_user, self.k_item,
                self.C_,
                lambda_,
                self.w_main, self.w_user,
                self._w_main_multiplier
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
            in ``item``. Might contain missing values.

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

        a_vec = self.c_funs.call_factors_collective_warm_implicit(
            X_val,
            X_col,
            U,
            U_val,
            U_col,
            self._U_colmeans,
            self.B_,
            self.C_,
            self._BeTBe,
            self._BtB_padded,
            self._BtB_shrunk,
            self.k, self.k_user, self.k_item, self.k_main,
            lambda_, self.alpha,
            self._w_main_multiplier,
            self.w_user, self.w_main,
            self.NA_as_zero_user
        )

        return a_vec


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
            Missing entries should have value ``np.nan`` when passing a dense
            array.

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
        lambda_, lambda_bias = \
            self._process_transform_inputs(X=X, U=U, U_bin=None, W=None,
                                           replace_existing=True)

        A = self.c_funs.call_collective_factors_warm_implicit_multiple(
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
            self._BtB_padded,
            self._BtB_shrunk,
            self._CtCinvCt,
            self._CtC,
            self._CtCchol,
            n, m_u, m_x,
            self.k, self.k_user, self.k_item, self.k_main,
            lambda_, self.alpha,
            self._w_main_multiplier,
            self.w_user, self.w_main,
            self.NA_as_zero_user,
            self.nthreads
        )

        return self._predict_user_multiple(A, item, bias=None)

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

        a_vec = self.c_funs.call_factors_offsets_cold(
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

        b_vec = self.c_funs.call_factors_offsets_cold(
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

        A = self.c_funs.call_offsets_factors_cold_multiple(
            Uarr,
            Urow,
            Ucol,
            Uval,
            Ucsr_p, Ucsr_i, Ucsr,
            Mat,
            self.C_bias_ if not is_I else self.D_bias_,
            m_u,
            self.k,
            self.k_sec, self.k_main,
            self.w_user if not is_I else self.w_item,
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
            Attributes for the items for which to predict ratings/values. Data frames with 'ItemId' column are not supported. Must have one row per entry
            in ``user``.

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
    for this model was implemented for experimentation purposes and it's
    recommended to use L-BFGS instead.

    Note
    ----
    You might want to experiment with tuning the maximum number of L-BFGS iterations
    and stopping earlier. Be aware that this model requires a lot more iterations
    to reach convergence compared to the classic and the collective models.

    Parameters
    ----------
    k : int
        Number of latent factors to use (dimensionality of the low-rank
        factorization), which will have a free component and an attribute-dependent
        component. Other additional separate factors can be specified through
        ``k_sec`` and ``k_main``.
    lambda_ : float or array(6,)
        Regularization parameter. Can also use different regulatization for each
        matrix, in which case it should be an array with 6 entries, corresponding,
        in this order, to: user_bias, item_bias, A, B, C, D.
        The attribute biases will have the same regularization as the matrices
        to which they apply (C and D).
        Passing different regulatization for each matrix is not supported with
        ``method='als'``.
    method : str, one of "lbfgs" or "als"
        Optimization method used to fit the model. If passing ``'lbfgs'``, will
        fit it through a gradient-based approach using an L-BFGS optimizer.
        If passing ``'als'``, will first obtain the solution ignoring the side
        information using an alternating least-squares procedure (the classical
        model described in other papers), then reconstruct the model matrices
        by a least-squares approximation. The ALS approach was implemented for
        experimentation purposes only and is not recommended.
    user_bias : bool
        Whether to add user biases (intercepts) to the model.
    item_bias : bool
        Whether to add item biases (intercepts) to the model. Be aware that using
        item biases with low regularization for them will tend to favor items
        with high average ratings regardless of the number of ratings the item
        has received.
    k_sec : int
        Number of factors in the factorizing matrices which are determined
        exclusively from user/item attributes. These will be at the beginning
        of the C and D matrices once the model is fit. If there are no attributes
        for a given matrix (user/item), then that matrix will have an extra
        ``k_sec`` factors (e.g. if passing user side info but not item side info,
        then the B matrix will have an extra ``k_sec`` factors). Will be counted
        in addition to those already set by ``k``. Not supported when
        using ``method='als'``.
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
        If you see that the procedure is spending thousands of iterations
        without any significant decrease in the loss function or gradient norm,
        it's highly likely that the regularization is too low.
        Ignored when passing ``method='als'``.
    niter : int
        Number of alternating least-squares iterations to perform. Note that
        one iteration denotes an update round for all the matrices rather than
        an update of a single matrix. Ignored when passing ``method='lbfgs'``.
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
        Number of correction paris to use for the L-BFGS optimization routine.
        Recommended values are between 3 and 7. Note that higher values
        translate into higher memory requirements. Ignored when passing
        ``method='als'``.
    NA_as_zero : bool
        Whether to take missing entries in the 'X' matrix as zeros (only
        when the 'X' matrix is passed as sparse COO matrix or DataFrame)
        instead of ignoring them. Note that this is a different model from the
        implicit-feedback version with weighted entries, and it's a much faster
        model to fit.
    use_float : bool
        Whether to use C float type for the model parameters (typically this is
        ``np.float32``). If passing ``False``, will use C double (typically this
        is ``np.float64``). Using float types will speed up computations and
        use less memory, at the expense of reduced numerical precision.
    use_cg : bool
        Whether to use a conjugate gradient method to solve the closed-form
        least squares problems. This was implemented for experimentation
        purposes only - will not provide any advantage over the default
        Cholesky solver.
    random_state : int or RandomState
        Seed used to initialize parameters at random. If passing a NumPy
        RandomState object, will use it to draw a random integer. Note
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
    def __init__(self, k=50, lambda_=1e1, method="lbfgs",
                 user_bias=True, item_bias=True, k_sec=0, k_main=0,
                 add_intercepts=True, w_user=1., w_item=1.,
                 maxiter=10000, niter=10, parallelize="separate", corr_pairs=7,
                 NA_as_zero=False, use_float=False, use_cg=False,
                 random_state=1, verbose=True, print_every=100,
                 produce_dicts=False, copy_data=True, nthreads=-1):
        assert k>0 or k_sec>0 or k_main>0
        self._take_params(implicit=False, alpha=0., downweight=False,
                          k=1, lambda_=lambda_, method=method, use_cg=use_cg,
                          user_bias=user_bias, item_bias=item_bias,
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
        y : None
            Not used. Kept as a place-holder for compatibility with SciKit-Learn
            pipelines.
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

        if self.method == "lbfgs":
            self.glob_mean_, self._A_pred, self._B_pred, values, self.nupd_, self.nfev_, self._B_plus_bias = \
                self.c_funs.call_fit_offsets_explicit_lbfgs(
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
                    self.user_bias, self.item_bias,
                    self.add_intercepts,
                    self.lambda_ if isinstance(self.lambda_, float) else 0.,
                    self.lambda_ if isinstance(self.lambda_, np.ndarray) else np.empty(0, dtype=self.dtype_),
                    self.verbose, self.print_every,
                    self.corr_pairs, self.maxiter,
                    self.nthreads, self.parallelize != "separate",
                    self.random_state
            )
            self.user_bias_, self.item_bias_, self.A_, self.B_, self.C_, self.D_, \
            self.C_bias_, self.D_bias_ = \
                self.c_funs.unpack_values_lbfgs_offsets(
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
        else:
            self.glob_mean_, self._A_pred, self._B_pred, values, self._B_plus_bias = \
                self.c_funs.call_fit_offsets_explicit_als(
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
                    self.user_bias, self.item_bias,
                    self.add_intercepts,
                    self.lambda_,
                    self.verbose, self.nthreads, self.use_cg,
                    self.random_state, self.niter
                )
            self.user_bias_, self.item_bias_, self.A_, self.B_, self.C_, self.D_, \
            self.C_bias_, self.D_bias_ = \
                self.c_funs.unpack_values_offsets_explicit_als(
                    values,
                    self.user_bias, self.item_bias,
                    self.k,
                    m, n, p, q,
                    self.add_intercepts
                )
        
        if isinstance(self.lambda_, np.ndarray):
            lambda_ = self.lambda_[2]
            lambda_bias = self.lambda_[0]
        else:
            lambda_ = self.lambda_
            lambda_bias = self.lambda_
        
        self._A_pred, self._B_pred, self._BtBinvBt, self._BtB, self._BtBchol = \
            self.c_funs.precompute_matrices_offsets_explicit(
                self.A_,
                self.B_,
                self.C_,
                self.C_bias_,
                self.D_,
                self.D_bias_,
                self._A_pred,
                self._B_pred,
                self._B_plus_bias,
                Uarr,
                Iarr,
                np.empty(0, dtype=ctypes.c_long),
                np.empty(0, dtype=ctypes.c_int),
                np.empty(0, dtype=self.dtype_),
                np.empty(0, dtype=ctypes.c_long),
                np.empty(0, dtype=ctypes.c_int),
                np.empty(0, dtype=self.dtype_),
                self.k, self.k_main, self.k_sec,
                lambda_, lambda_bias, self.w_user, self.w_item,
                self.nthreads
            )

        self.is_fitted_ = True
        return self

    def factors_warm(self, X=None, X_col=None, X_val=None, W=None,
                     U=None, U_col=None, U_val=None,
                     return_bias=False, return_raw_A=False):
        """
        Determine user latent factors based on new ratings data

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
            msg  = "Cannot use this method without side info "
            msg += "when using k_sec>0."
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
                                             return_bias=return_bias)
        else:
            outp = self._factors_warm_common(X=X, X_col=X_col, X_val=X_val, W=W,
                                             U=U, U_bin=None, U_col=U_col, U_val=U_val,
                                             return_bias=return_bias)
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
                      U, U_val, U_col, U_bin, return_bias):
        if isinstance(self.lambda_, np.ndarray):
            lambda_ = self.lambda_[2]
            lambda_bias = self.lambda_[0]
        else:
            lambda_ = self.lambda_
            lambda_bias = self.lambda_

        a_bias, a_pred, a_vec = self.c_funs.call_factors_offsets_warm_explicit(
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
            self._BtBinvBt,
            self._BtB,
            self.glob_mean_,
            self.k, self.k_sec, self.k_main,
            lambda_, lambda_bias,
            self.w_user,
            self.user_bias,
            0, 1
        )

        if return_bias:
            return a_vec, a_pred, a_bias
        else:
            return a_vec, a_pred

    def predict_warm(self, items, X=None, X_col=None, X_val=None, W=None,
                     U=None, U_col=None, U_val=None):
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
        lambda_, lambda_bias = \
            self._process_transform_inputs(X=X, U=U, U_bin=None, W=W,
                                           replace_existing=True)

        A, A_bias, _ = self.c_funs.call_offsets_factors_warm_multiple(
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
            self._BtBinvBt,
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

        Parameters
        ----------
        X : array(m, n), CSR matrix(m, n) , or COO matrix(m, n)
            New 'X' data with potentially missing entries which are to be imputed.
            Missing entries should have value ``np.nan`` when passing a dense
            array.
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
        replace_existing : bool
            Whether to replace existing non-missing entries in 'X' with the
            model predictions - that is, if passing 'False', will only fill
            in the missing entries in 'X', leaving the non-missing entries
            as they were.

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
        lambda_, lambda_bias = \
            self._process_transform_inputs(X=X, U=U, U_bin=None, W=W,
                                           replace_existing=replace_existing)

        A, A_bias, _1 = self.c_funs.call_offsets_factors_warm_multiple(
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
            self._BtBinvBt,
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

    Parameters
    ----------
    k : int
        Number of latent factors to use (dimensionality of the low-rank
        approximation).
    lambda_ : float
        Regularization parameter.
    alpha : float
        Weighting parameter for the non-zero entries in the implicit-feedback
        model. See [2] for details.
    downweight : bool
        Whether to decrease the weight of the 'X' matrix being factorized
        according to the number of present entries. This has the same effect
        as rescaling (increasing) the regularization parameter.
    add_intercepts : bool
        Whether to add intercepts/biases to the user/item attribute matrices.
    niter : int
        Number of alternating least-squares iterations to perform. Note that
        one iteration denotes an update round for all the matrices rather than
        an update of a single matrix.
    use_float : bool
        Whether to use C float type for the model parameters (typically this is
        ``np.float32``). If passing ``False``, will use C double (typically this
        is ``np.float64``). Using float types will speed up computations and
        use less memory, at the expense of reduced numerical precision.
    use_cg : bool
        Whether to use a conjugate gradient method to solve the closed-form
        least squares problems. This was implemented for experimentation
        purposes only - will not provide any advantage over the default
        Cholesky solver.
    random_state : int or RandomState
        Seed used to initialize parameters at random. If passing a NumPy
        RandomState object, will use it to draw a random integer.
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
    """
    def __init__(self, k=50, lambda_=1e3, alpha=40., downweight=True,
                 add_intercepts=True, niter=10, use_float=False, use_cg=False,
                 random_state=1, verbose=False,
                 produce_dicts=False, copy_data=True, nthreads=-1):
        self._take_params(implicit=True, alpha=alpha, downweight=downweight,
                          k=k, lambda_=lambda_, method="als", use_cg=use_cg,
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

    def get_parms(self, deep=None):
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
        self._A_pred, self._B_pred, values, self._w_main_multiplier = \
            self.c_funs.call_fit_offsets_implicit_als(
                Xrow,
                Xcol,
                Xval,
                Uarr,
                Iarr,
                m, n, p, q,
                self.k, self.add_intercepts,
                self.lambda_, self.alpha,
                self.verbose, self.nthreads, self.use_cg,
                self.downweight,
                self.random_state, self.niter
            )
        self.A_, self.B_, self.C_, self.D_, self.C_bias_, self.D_bias_ = \
            self.c_funs.unpack_values_offsets_implicit_als(
                values,
                self.k,
                m, n, p, q,
                self.add_intercepts
            )

        if isinstance(self.lambda_, np.ndarray):
            lambda_ = self.lambda_[2]
        else:
            lambda_ = self.lambda_

        self._A_pred, self._B_pred, self._BtB = \
            self.c_funs.precompute_matrices_offsets_implicit(
                self.C_,
                self.C_bias_,
                self.D_,
                self.D_bias_,
                self._A_pred,
                self._B_pred,
                Uarr,
                Iarr,
                np.empty(0, dtype=ctypes.c_long),
                np.empty(0, dtype=ctypes.c_int),
                np.empty(0, dtype=self.dtype_),
                np.empty(0, dtype=ctypes.c_long),
                np.empty(0, dtype=ctypes.c_int),
                np.empty(0, dtype=self.dtype_),
                self.k,
                lambda_,
                self.nthreads
            )

        self.is_fitted_ = True
        return self

    def factors_warm(self, X_col, X_val):
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

        Returns
        -------
        factors : array(k,)
            User factors as determined from the data in 'X_col' and 'X_val'.
        """
        if (X_col is None) or (X_val is None):
            raise ValueError("Must pass 'X_col' and 'X_val'.")
        return self._factors_warm_common(X=None, X_col=X_col, X_val=X_val, W=None,
                                         U=None, U_bin=None, U_col=None, U_val=None,
                                         return_bias=False)

    def _factors_warm(self, X, W_dense, X_val, X_col, W_sp,
                      U, U_val, U_col, U_bin, return_bias):
        a_pred, a_vec = self.c_funs.call_factors_offsets_warm_implicit(
            X_val,
            X_col,
            U,
            np.empty(0, dtype=self.dtype_),
            np.empty(0, dtype=ctypes.c_int),
            self._B_pred,
            self.C_,
            self.C_bias_,
            self._BtBinvBt,
            self._BtB,
            self.k, 0, 0,
            self.lambda_, self.alpha,
            self._w_main_multiplier,
            0,
            0
        )
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

    def predict_warm_multiple(self, X, item):
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
        lambda_, lambda_bias = \
            self._process_transform_inputs(X=X, U=None, U_bin=None, W=None,
                                           replace_existing=True)

        A, _ = self.c_funs.call_offsets_factors_warm_implicit_multiple(
            Xrow,
            Xcol,
            Xval,
            Xcsr_p, Xcsr_i, Xcsr,
            self._B_pred,
            self.C_,
            self._BtBinvBt,
            self._BtB,
            m_x, n,
            self._k_pred,
            lambda_, self.alpha,
            self._w_main_multiplier,
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
    iterations to converge compared to the other models. You might also
    want to experiment with tuning the maximum number of iterations.

    Note
    ----
    The input data for attributes does not undergo any transformations when
    fitting this model, which is to some extent sensible to the scales of the variables and their means in the same way as regulaized linear regression.

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
    lambda_ : float or array(6,)
        Regularization parameter. Can also use different regulatization for each
        matrix, in which case it should be an array with 6 entries, corresponding,
        in this order, to: user_bias, item_bias, [ignored], [ignored], C, D.
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
        If you see that the procedure is spending thousands of iterations
        without any significant decrease in the loss function or gradient norm,
        it's highly likely that the regularization is too low.
    corr_pairs : int
        Number of correction paris to use for the L-BFGS optimization routine.
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
    random_state : int or RandomState
        Seed used to initialize parameters at random. If passing a NumPy
        RandomState object, will use it to draw a random integer. Note
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
                 produce_dicts=False, copy_data=True, nthreads=-1):
        self._take_params(implicit=False, alpha=40., downweight=False,
                          k=1, lambda_=lambda_, method="lbfgs", use_cg=False,
                          user_bias=user_bias, item_bias=item_bias,
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
                          produce_dicts=produce_dicts, copy_data=copy_data,
                          nthreads=nthreads)
        self._take_params_offsets(k_sec=k, k_main=0,
                                  add_intercepts=add_intercepts)
        self.k = 0
        self._k_pred = self.k_sec

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
        U : array(m, p), COO(m, p), DataFrame(m, p+1)
            User attributes information. If 'X' is a DataFrame, should also
            be a DataFrame, containing column 'UserId'. If 'U' is sparse,
            'X' should be passed as a sparse COO matrix too.
            Should not contain any missing values.
        I : array(n, q), COO(n, q), DataFrame(n, q+1)
            Item attributes information. If 'X' is a DataFrame, should also
            be a DataFrame, containing column 'ItemId'. If 'I' is sparse,
            'X' should be passed as a sparse COO matrix too.
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
        self.glob_mean_, self._A_pred, self._B_pred, values, self.nupd_, self.nfev_, B_plus_bias = \
            self.c_funs.call_fit_offsets_explicit_lbfgs(
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
                0, self.k_sec, 0,
                1., 1.,
                self.user_bias, self.item_bias,
                self.add_intercepts,
                self.lambda_ if isinstance(self.lambda_, float) else 0.,
                self.lambda_ if isinstance(self.lambda_, np.ndarray) else np.empty(0, dtype=self.dtype_),
                self.verbose, self.print_every,
                self.corr_pairs, self.maxiter,
                self.nthreads, self.parallelize != "separate",
                self.random_state
        )

        self.user_bias_, self.item_bias_, _1, _2, self.C_, self.D_, \
        self.C_bias_, self.D_bias_ = \
            self.c_funs.unpack_values_lbfgs_offsets(
                values,
                self.user_bias, self.item_bias,
                self.k, self.k_sec, self.k_main,
                m, n, p, q,
                self.add_intercepts
            )

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
        a_vec = self.c_funs.call_factors_content_based(
            U,
            U_val,
            U_col,
            self.C_,
            self.C_bias_
        )
        return a_vec

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

        rank_new, scores_new = self.c_funs.call_rank_content_based_new(
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

        scores_new = self.c_funs.call_predict_content_based_new(
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

        scores_new = self.c_funs.call_predict_content_based_old(
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

    Parameters
    ----------
    implicit : bool
        Whether to use the implicit-feedback model, in which the 'X' matrix is
        assumed to have only binary entries and each of them having a weight
        in the loss function given by the observer user-item interactions and
        other parameters,
    user_bias : bool
        Whether to add user biases to the model. Not supported for implicit
        feedback (``implicit=True``).
    lambda_ : float
        Regularization parameter. For the explicit-feedback case (default),
        lower values will tend to favor the highest-rated items regardless
        of the number of observations.
    alpha : float
        Weighting parameter for the non-zero entries in the implicit-feedback
        model. See [2] for details.
    downweight : bool
        (Only when passing ``implicit=True``) Whether to decrease the weight
        of the 'X' matrix being factorized according to the number of
        present entries. This has the same effect as rescaling (increasing)
        the regularization parameter. Provided for better comparability
        against the personalized-recommender models in this package.
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
    def __init__(self, implicit=False, user_bias=False, lambda_=1e1, alpha=40.,
                 downweight=True, use_float=False, produce_dicts=False,
                 copy_data=True, nthreads=-1):
        self._take_params(implicit=implicit, alpha=alpha, downweight=False,
                          k=1, lambda_=lambda_, method="als", use_cg=False,
                          user_bias=user_bias, item_bias=True,
                          k_user=0, k_item=0, k_main=0,
                          w_main=1., w_user=1., w_item=1.,
                          maxiter=0, niter=0, parallelize="separate",
                          corr_pairs=0,
                          NA_as_zero=False, NA_as_zero_user=False,
                          NA_as_zero_item=False,
                          precompute_for_predictions=False,
                          use_float=use_float,
                          random_state=1, init="normal",
                          verbose=0, print_every=0,
                          produce_dicts=produce_dicts, copy_data=copy_data,
                          nthreads=nthreads)
        self.k = 0
        self.niter = 0
        self.implicit = bool(implicit)
        if self.implicit and self.user_bias:
            raise ValueError("'user_bias' not supported for implicit-feedback.")

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

        self.glob_mean_, self.user_bias_, self.item_bias_, self._w_main_multiplier = \
            self.c_funs.call_fit_most_popular(
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
                self.nthreads
            )

        self._A_pred = np.zeros((m,1), dtype=self.dtype_)
        self._B_pred = np.zeros((n,1), dtype=self.dtype_)
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

class CMF_imputer(CMF_explicit):
    """
    A wrapper for CMF_explicit allowing argument 'y' in 'fit' and
    'transform' (used as a placeholder only, not used for anything),
    which can be used as part of SciKit-Learn pipelines due to having
    this extra parameter.

    Everything else is exactly the same as for 'CMF_explicit'
    """
    def fit(self, X, y=None, U=None, I=None, U_bin=None, I_bin=None, W=None):
        return super().fit(X=X, U=U, U_bin=U_bin, I=I, I_bin=I_bin, W=W)
