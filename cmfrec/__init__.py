import pandas as pd, numpy as np, tensorflow as tf
pd.options.mode.chained_assignment = None

class CMF:
    def __init__(self, k=50, k_main=0, k_item=0, k_user=0, w_main=1.0, w_item=1.0, w_user=1.0,
                 reg_param=1e-4, standardize=True, reweight=False):
        """
        Collective matrix factorization model for recommenders systems with explicit data and side info.
        
        
        Fits a collective matrix factorization model to ratings data along with item and/or user side information,
        by factorizing all matrices with common (shared) item-factors, e.g.:
        X~=AB' and I~=BC'
        By default, the function to minimize is as follows:
        L = w_main*norm(X-AB')^2/nX + w_item*norm(I-BC')^2/nI + w_user*norm(U-AD')^2/nU + reg_param*(norm(A)^2+norm(B)^2+norm(C)^2+norm(D)^2)
        Where:
            X is the ratings matrix (considering only non-missing entries)
            I is the item-attribute matrix (only supports dense, i.e. all non-missing entries)
            U is the user-attribute matrix (only supports dense, i.e. all non-missing entries)
            A, B, C, D are lower-dimensional matrices (the model parameters)
            nX, nI, nU are the number of entries in each matrix
            The matrix products might not use all the rows/columns of these matrices at each factorization
            (this is controlled with k_main, k_item and k_user)
        
        More details can be found on the paper
        'Relational learning via collective matrix factorization (2008) from Singh, A. P., & Gordon, G. J.'
        
        The model is fit using L-BFGS, interfaced through Tensorflow
        
        Note
        ----
        All users and items are reindexed internally, so you can use non-integer IDs and don't have to worry about enumeration gaps.
        The API contains parameters for both an item-attribute matrix and a user-attribute matrix,
        but you can fit the model to data with only one or none of them.
        Parameters corresponding to the factorization of a matrix for which no data is passed will be ignored.

        Parameters
        ----------
        k : int
            Number of common (shared) latent factors to use.
        k_main : int
            Number of additional latent factors to use for the ratings factorization.
        k_item : int
            Number of additional latent factors to use for the item attributes factorization.
        k_user: int
            Number of additional latent factors to use for the user attributes factorization.
        w_main : float
            Weight to assign to the root squared error in factorization of the ratings matrix.
        w_item : float
            Weight to assign to the root squared error in factorization of the item attributes matrix.
        w_user : float
            Weight to assign to the root squared error in factorization of the user attributes matrix.
        reg_param : float or tuple of floats
            Regularization parameter for each matrix, in this order:
            1) User-Factor, 2) Item-Factor, 3) ItemAttribute-Factor, 4) UserAttribute-Factor.
        standardize : bool
            Whether to divide the sum of squared errors from each factorized matrix by the number of entries.
            Setting this to false requires far larger regularization parameters.
            Ignored when passing 'reweight=True'.
        reweight : bool
            Whether to automatically reweight the errors of each matrix factorization so that they get similar influence,
            accounting for the number of elements in each and the magnitudes of the entries
            (appplies in addition to weights passes as w_main, w_item and w_user).
            This is done by calculating the initial sum of squared errors with randomly initialized factor matrices,
            but it's not guaranteed to be a good criterion.
            Note that when using this, the weights of the factorizations are shrunk, so the regularization
            parameter should also be changed.
            
            It might be better to scale the entries of either the ratings or the attributes matrix so that they are in a similar scale
            (e.g. if the ratings are in [1,5], the attributes should ideally be in the same range and not [-10^3,10^3]).
        """
        self.k=k
        self.k_main=k_main
        self.k_item=k_item
        self.k_user=k_user
        self.w_main=w_main
        self.w_item=w_item
        self.w_user=w_user
        if (type(reg_param)==float) or (type(reg_param)==int):
            self.reg_param=(reg_param,reg_param,reg_param,reg_param)
        elif (type(reg_param)==tuple) or (type(reg_param)==list):
            if len(reg_param)!=4:
                raise ValueError('reg_param must be a number or tuple with 4 numbers')
            self.reg_param=reg_param
        else:
            raise ValueError('reg_param must be a number or tuple with 4 numbers')
        self.standardize=standardize
        self.reweight=reweight
        
    def fit(self, ratings, item_info=None, user_info=None, reindex=True, use_dense=True,
            save_entries=True, random_seed=None, maxiter=1000, save_dataset=False):
        """
        Fit the model to ratings data and item/user side info, using L-BFGS
        
        
        Note
        ----
        The model is fit with full L-BFGS updates (not stochastic gradient descent or stochastic Newton),
        i.e. it calculates errors for the whole data and updates all model parameters at once during each iteration.
        By default, the number of iterations is set at 2000, but for smaller datasets, this might not reach convergence.
        
        Ratings are not centered when fitting the model. If you require it, you'll have to center them beforehand
        (e.g. subtracting global/user/item bias from each rating).
        However, I've found centering to make the model worse so I wouldn't recommend it.
        
        The model will only be fit with users and items that are present in both the ratings and the side info data.
        If after filtering there are no matching entries, it will throw an error.
        Although the API can accommodate both item and user side information,
        you can still fit the model without them.
        
        
        Parameters
        ----------
        ratings : list of tuples or pandas data frame
            Ratings data to which to fit the model.
            If a pandas data frame, must contain the columns 'UserId','ItemId' and 'Rating'.
            If a list of tuples, must be in the format (UserId,ItemId,Rating) (will be coerced to data frame)
        item_info : pandas data frame or numpy array
            Side information about the items (i.e. their attributes, as a table).
            Must contain a column named ItemId.
            If called with 'reindex=False', must be a numpy array,
            with rows corresponding to item ID numbers and columns to item attributes.
        user_info : pandas data frame
            Side information about the users (i.e. their attributes, as a table).
            Must contain a column called 'UserId'.
            If called with 'reindex=False', must be a numpy array,
            with rows corresponding to user ID numbers and columns to user attributes.
        reindex : bool
            Whether to reindex internally all the IDs passed for users and items.
            If your data is already properly enumerated, i.e. all item and user IDs are consecutive integers starting at zero,
            each row/column of the ratings matrix corresponds to the user and item with that number,
            and same for the item and/or user attribute matrix, this option will be faster.
            If this is set to 'True', you will need to pass in ratings data as a data frame and side info as numpy arrays.
        use_dense : bool
            Whether to create a dense matrix of predictions. This will allocate a full matrix of dimensions n_users*n_items
            in RAM memory, so if the dataset is large, it will probably be too large to calculate, but for smaller datasets,
            it will speed up slicing.
        save_entries : bool
            Whether to save the information about which users rated which items.
            This can later be used to filter out already rated items when getting Top-N recommendations for a user.
            Forced to False when passing reindex=False.
        random_seed: int
            Random seed to be used to get a starting point.
        maxiter : int
            Maximum number of iterations.
        save_dataset : bool
            Whether to save the raw data that passed (after processing) in the model itself. This is helpful if you want to try
            different hypterparameters later using the '_fit' method, as it won't have to reindex the data again.
            The reindex data is stored internally in attributes '_X', '_lst_slices', '_prod_arr', and '_user_arr'.
            
        Attributes
        ----------
        A : numpy.ndarray (nitems, k_main + k + k_user)
            Matrix with the user-factor attributes, containing columns from both factorizations.
            If you wish to extract only the factors used for predictons, slice it like this: U[:,:k_main+k]
        B : numpy.ndarray (nusers, k_main + k + k_item)
            Matrix with the item-factor attributes, containing columns from both factorizations.
            If you wish to extract only the factors used for predictons, slice it like this: V[:,:k_main+k]
        user_orig_to_int : dict
            Mapping of user IDs as they appear in the rating and user_info data to the rows of U.
        item_orig_to_int : dict
            Mapping of item IDs as they appear in the rating and item_info data to the rows of V.
        """
        
        # readjusting parameters in case they are redundant
        if item_info is None:
            self.k_user=0
        if user_info is None:
            self.k_item=0
        
        self.save_entries=save_entries
        self._process_data(ratings,item_info,user_info,reindex)
        self._set_weights(random_seed)
        
        self._fit(self.w1,self.w2,self.w3,self.reg_param,
                  self.k,self.k_main,self.k_item,self.k_user,
                  use_dense,random_seed,maxiter)
        
        if not save_dataset:
            del self._X
            del self._lst_slices
            del self._prod_arr
            del self._user_arr
            del self._prod_arr_notmissing
            del self._user_arr_notmissing
        
        return self
    
    def _fit(self, w1=1.0, w2=1.0, w3=1.0, reg_param=[1e-3]*4,
             k=50, k_main=0, k_item=0, k_user=0,
             use_dense=False,random_seed=None,maxiter=1000):
        # faster method, can be reused with the same data
        
        if random_seed is not None:
            tf.set_random_seed(random_seed)
        
        A=tf.Variable(tf.random_normal([self._m1,k_main+k+k_user]))
        B=tf.Variable(tf.random_normal([self._m2,k_main+k+k_item]))
        C=tf.Variable(tf.random_normal([k+k_item,self._m3]))
        D=tf.Variable(tf.random_normal([k+k_user,self._m4]))
        
        R=tf.placeholder(tf.float32, shape=(None,))
        U=tf.placeholder(tf.float32)
        I=tf.placeholder(tf.float32)
        
        I_nonmissing=tf.placeholder(tf.float32)
        U_nonmissing=tf.placeholder(tf.float32)

        Ab=A[:,:k_main+k]
        Ad=A[:,k_main:]
        Ba=B[:,:k_main+k]
        Bc=B[:,k_main:]

        if use_dense:
            pred_ratings=tf.gather_nd(tf.matmul(Ab,Ba, transpose_b=True), self._lst_slices)
        else:
            pred_ratings=tf.reduce_sum(tf.multiply(
                tf.gather(Ab, self._lst_slices[:,0], axis=0), tf.gather(Ba, self._lst_slices[:,1], axis=0))
                                       , axis=1)
        err_ratings=tf.losses.mean_squared_error(pred_ratings,R)
        loss=w1*tf.losses.mean_squared_error(pred_ratings,R) + reg_param[0]*tf.nn.l2_loss(A) + reg_param[1]*tf.nn.l2_loss(B)
        
        if self._prod_arr is not None:
            pred_item=tf.matmul(Bc,C)
            loss+=reg_param[2]*tf.nn.l2_loss(C)
            if self._prod_arr_notmissing is not None:
                pred_item*=I_nonmissing
            loss+=w2*tf.losses.mean_squared_error(pred_item,I) + reg_param[2]*tf.nn.l2_loss(C)
            
        if self._user_arr is not None:
            pred_user=tf.matmul(Ad,D)
            if self._user_arr_notmissing is not None:
                pred_user*=U_nonmissing
            loss+=w2*tf.losses.mean_squared_error(pred_user,U) + reg_param[2]*tf.nn.l2_loss(C)

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B', options={'maxiter':maxiter})
        model = tf.global_variables_initializer()
        sess=tf.Session()
        sess.run(model)
        with sess:
            sess.run(model)
            optimizer.minimize(sess, feed_dict={R:self._X, U:self._user_arr, I:self._prod_arr,
                                I_nonmissing:self._prod_arr_notmissing, U_nonmissing:self._user_arr_notmissing})
            self.A=A.eval(session=sess)
            self.B=B.eval(session=sess)
    
    def _process_data(self,ratings,item_info,user_info,reindex):
        # pasing an already properly index DF will speed up the process
        if not reindex:
            if type(ratings)!=pd.core.frame.DataFrame:
                raise ValueError('ratings must be a pandas data frame')
            if (item_info is not None) and (type(item_info)!=numpy.ndarray):
                raise ValueError('item_info must be a numpy array')
            if (user_info is not None) and (type(user_info)!=numpy.ndarray):
                raise ValueError('user_info must be a numpy array')
                
            self._X=ratings.Rating
            self._lst_slices=np.array([list(i) for i in zip(ratings_df.UserId,ratings_df.ItemId)])
            self.user_orig_to_int={i:i for i in range(ratings.shape[0])}
            self.item_orig_to_int={i:i for i in range(ratings.shape[1])}
            self._user_int_to_orig=self.user_orig_to_int
            self._user_int_to_orig=self.item_orig_to_int
            self._m1=ratings.UserId.max()
            self._m2=ratings.ItemId.max()
            if item_info is not None:
                self._m3=item_info.shape[1]
                self._prod_arr=item_info
            else:
                self._m3=0
                self._prod_arr=None
            if user_info is not None:
                self._m4=user_info.shape[1]
                self._user_arr=user_info
            else:
                self._m4=0
                self._user_arr=None
            self.save_entries=False
            
        # reindexing all entries
        else:
            # getting ratings as a DF
            if type(ratings)==list:
                ratings_df=pd.DataFrame(ratings,columns=['UserId','ItemId','Rating'])
            elif type(ratings)==pd.core.frame.DataFrame:
                if ('UserId' not in ratings.columns.values) or ('ItemId' not in ratings.columns.values) or ('Rating' not in ratings.columns.values):
                    raise ValueError("Ratings data frame must contain the columns 'UserId','ItemId' and 'Rating'")
                ratings_df=ratings[['UserId','ItemId','Rating']].copy()
            else:
                raise ValueError("Ratings must be a list of tuples or pandas data frame")

            # getting item side info
            if item_info is not None:
                if type(item_info)!=pd.core.frame.DataFrame:
                    raise ValueError('item_info must be a pandas data frame with a column named ItemId')
                if 'ItemId' not in item_info.columns.values:
                    raise ValueError('item_info must be a pandas data frame with a column named ItemId')
                itemset=set(list(item_info.ItemId))
                ratings_df=ratings_df.loc[ratings_df.ItemId.isin(itemset)]

            # getting user side info
            if user_info is not None:
                if type(user_info)!=pd.core.frame.DataFrame:
                    raise ValueError('user_info must be a pandas data frame with a column named UserId')
                if 'UserId' not in user_info.columns.values:
                    raise ValueError('item_info must be a pandas data frame with a column named UserId')
                userset=set(list(user_info.UserId))
                ratings_df=ratings_df.loc[ratings_df.UserId.isin(userset)]

            if ratings_df.shape[0]==0:
                raise ValueError('There are no ratings for the users and items with side info')

            cnt_users=0
            cnt_items=0
            self.user_orig_to_int=dict()
            self.item_orig_to_int=dict()
            self._user_int_to_orig=dict()
            self._item_int_to_orig=dict()
            for i in ratings_df.itertuples():
                if i.UserId not in self.user_orig_to_int:
                    self.user_orig_to_int[i.UserId]=cnt_users
                    self._user_int_to_orig[cnt_users]=i.UserId
                    cnt_users+=1
                if i.ItemId not in self.item_orig_to_int:
                    self.item_orig_to_int[i.ItemId]=cnt_items
                    self._item_int_to_orig[cnt_items]=i.ItemId
                    cnt_items+=1

            if item_info is not None:
                itemset=set(list(ratings_df.ItemId))
                self._prod_arr=item_info.loc[item_info.ItemId.isin(itemset)]
                self._prod_arr['ItemId']=self._prod_arr.ItemId.map(lambda x: self.item_orig_to_int[x])
                self._prod_arr=self._prod_arr.sort_values('ItemId').set_index('ItemId').as_matrix()
                self._m3=self._prod_arr.shape[1]
                if self._m3==0:
                    raise ValueError("item_info doesn't contain items in common with ratings data")
                prod_arr_missing=np.isnan(self._prod_arr)
                if np.sum(prod_arr_missing)==0:
                    self._prod_arr_notmissing=None
                else:
                    self._prod_arr[prod_arr_missing]=0
                    self._prod_arr_notmissing=(~prod_arr_missing).astype('uint8')
                    del prod_arr_missing
            else:
                self._m3=0
                self._prod_arr=None
                self._prod_arr_notmissing=None

            if user_info is not None:
                userset=set(list(ratings_df.UserId))
                self._user_arr=user_info.loc[user_info.UserId.isin(userset)]
                self._user_arr['UserId']=self._user_arr.UserId.map(lambda x: self.user_orig_to_int[x])
                self._user_arr=self._user_arr.sort_values('UserId').set_index('UserId').as_matrix()
                self._m4=self._user_arr.shape[1]
                if self._m4==0:
                    raise ValueError("user_info doesn't contain items in common with ratings data")
                user_arr_missing=np.isnan(self._user_arr)
                if np.sum(user_arr_missing)==0:
                    self._user_arr_notmissing=None
                else:
                    self._user_arr[user_arr_missing]=0
                    self._user_arr_notmissing=(~user_arr_missing).astype('uint8')
                    del user_arr_missing
                    
            else:
                self._m4=0
                self._user_arr=None
                self._user_arr_notmissing=None

            ratings_df['UserId']=ratings_df.UserId.map(lambda x: self.user_orig_to_int[x])
            ratings_df['ItemId']=ratings_df.ItemId.map(lambda x: self.item_orig_to_int[x])
            self._m1=cnt_users
            self._m2=cnt_items
            
            self._X=ratings_df.Rating
            self._lst_slices=np.array([list(i) for i in zip(ratings_df.UserId,ratings_df.ItemId)])

            if self.save_entries:
                self._items_rated_per_user=ratings_df.groupby('UserId')['ItemId'].agg(lambda x: set(x))
            
    def _set_weights(self, random_seed):
        m1,m2,m3,m4,k,ux,zx,qx=self._m1,self._m2,self._m3,self._m4,self.k,self.k_main,self.k_item,self.k_user
        
        if self.reweight:
            # this initializes all parameters at random, calculates the error, and scales weights by them
            if random_seed is not None:
                np.random.seed(random_seed)
            
            mat1=np.random.normal(size=(m1,ux+k))
            mat2=np.random.normal(size=(ux+k,m2))
            ind1=tuple([i[0] for i in self._lst_slices])
            ind2=tuple([i[1] for i in self._lst_slices])
            pred=np.dot(mat1,mat2)[ind1,ind2]
            err=pred-self._X
            err_main=1/np.mean(err**2)
            err_tot=err_main
            
            if self._prod_arr is not None:
                mat1=np.random.normal(size=(m2,k+zx))
                mat2=np.random.normal(size=(k+zx,m3))
                pred=mat1.dot(mat2)
                err=self._prod_arr-pred
                err_item=1/np.mean(err**2)
                err_tot+=err_item
                
            if self._user_arr is not None:
                mat1=np.random.normal(size=(m1,k+qx))
                mat2=np.random.normal(size=(k+qx,m4))
                pred=mat1.dot(mat2)
                err=self._user_arr-pred
                err_user=1/np.mean(err**2)
                err_tot+=err_user
                
            del mat1
            del mat2
                
            self.w1=self.w_main*err_main/err_tot
            if self._prod_arr is not None:
                self.w2=self.w_item*err_item/err_tot
            else:
                self.w2=0.0
            if self._user_arr is not None:
                self.w3=self.w_user*err_user/err_tot
            else:
                self.w3=0.0
                
        elif not self.standardize:
            # this multiplies the loss (MSE) from each matrix by its number of entries 
            nmain=self._X.shape[0]
            self.w1=self.w_main*nmain
            
            if self._prod_arr is not None:
                nitem=self._prod_arr.shape[0]*self._prod_arr.shape[1]
                self.w2=self.w_item*nitem
            else:
                self.w2=0.0
                
            if self._user_arr is not None:
                nuser=self._user_arr.shape[0]*self._user_arr.shape[1]
                self.w3=self.w_user*nuser
            else:
                self.w3=0.0
        else:
            # if no option is passed, weights are taken at face value
            self.w1=self.w_main
            self.w2=self.w_item
            self.w3=self.w_user
        
    def predict(self, UserId, ItemId):
        """Predict the rating that a given user would give to a given item"""
        try:
            user=self.user_orig_to_int[UserId]
        except:
            raise ValueError('Invalid user')
        try:
            item=self.item_orig_to_int[ItemId]
        except:
            raise ValueError('Invalid item')
            
        return self.A[user,:self.k_main+self.k].dot(self.B[item,:self.k_main+self.k].T)
    
    def top_n(self, UserId, n=10, scores=False, filter_rated=True):
        """
        Get Top-N recommendations for a given user
        
        Parameters
        ----------
        UserId
            User for which to produce a Top-N recommended list.
        n: int
            Length of the recommended list.
        scores: bool
            Whether to return the predicted ratings along with the IDs of the recommended items.
        filter_rated : bool
            Whether to filter out items that were already rated by the user from the Top-N list.
            This requires the model to have been called with 'save_entries=True', otherwise will be ignored.
        
        Returns
        -------
        list
            Top-N recommended items for the user.
            If 'scores=True', list of tuples containing (Item,Score)
        """
        try:
            user=self.user_orig_to_int[UserId]
        except:
            raise ValueError('Invalid user')
            
        preds=-self.A[user,:self.k_main+self.k].dot(self.B[:,:self.k_main+self.k].T)
        best=np.argsort(preds)
        if self.save_entries and filter_rated:
            out=list()
            for i in range(self._m2):
                if best[i] not in self._items_rated_per_user[user]:
                    if not scores:
                        out.append(self._item_int_to_orig[best[i]])
                    else:
                        out.append((self._item_int_to_orig[best[i]],-preds[best[i]]))
                    if len(out)==n:
                        break
            return out
        else:
            if not scores:
                return [self._item_int_to_orig[i] for i in best[:n]]
            else:
                preds=sorted(preds)
                return [(self._item_int_to_orig[best[i]],-preds[i]) for i in range(n)]
        
    def get_user_factor_vector(self, UserId):
        """Get the User-LatentFactor vector for a given user"""
        try:
            return self.A[self.user_orig_to_int[UserId],:]
        except:
            raise ValueError('Invalid user')
            
    def get_item_factor_vector(self, ItemId):
        """Get the Item-LatentFactor vector for a given item"""
        try:
            return self.B[self.item_orig_to_int[ItemId],:]
        except:
            raise ValueError('Invalid user')
