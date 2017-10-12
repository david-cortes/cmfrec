import pandas as pd, numpy as np
from casadi import MX, nlpsol, dot, mtimes, sqrt
from scipy.sparse import coo_matrix
pd.options.mode.chained_assignment = None

class CMF:
    def __init__(self, k=50, k_main=0, k_item=0, k_user=0, w_main=1.0, w_item=1.0, w_user=1.0,
                 reg_param=1.0, standardize=False, reweight=False):
        """
        Collective matrix factorization model for recommenders systems with explicit data and side info.
        
        
        Fits a collective matrix factorization model to ratings data along with item and/or user side information,
        by factorizing all matrices with common (shared) item-factors, e.g.:
        X~=UV' and M~=VZ'
        By default, the function to minimize is as follows:
        L = w_main*norm(X-UV') + w_item*norm(I-VZ') + w_user*norm(Q-UP') + reg_param*(norm(U)+norm(V)+norm(Z)+norm(P))
        Where:
            X is the ratings matrix (considering only non-missing entries)
            I is the item-attribute matrix (only supports dense, i.e. all non-missing entries)
            Q is the user-attribute matrix (only supports dense, i.e. all non-missing entries)
            U, V, Z, P are lower-dimensional matrices (the model parameters)
            The matrix products might not use all the rows/columns of these matrices at each factorization
            (this is controlled with k_main, k_item and k_user)
        
        More details can be found on the paper
        'Relational learning via collective matrix factorization (2008) from Singh, A. P., & Gordon, G. J.'
        
        The model is fit using BFGS with IPOPT as the workhorse, interfaced through CasADi
        
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
            accounting for the number of elements in each and the magnitudes of the entries.
            This is done by calculating the initial sum of squared errors with randomly initialized factor matrices,
            but it's not guaranteed to be a good criterion.
            
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
        
    def fit(self, ratings, item_info=None, user_info=None, reindex=True, save_entries=True,
            random_seed=None, print_time=False, ipopt_options={'tol':1e2,'max_iter':200,
         'hessian_approximation':'limited-memory','linear_scaling_on_demand':'no',"print_level":0}):
        """
        Fit the model to ratings data and item/user side info, using BFGS
        
        
        Note
        ----
        Never recommended to change the hessian approximation to exact hessian.
        The model is fit with full BFGS updates (not stochastic gradient descent or stochastic Newton),
        i.e. it calculates errors for the whole data and updates all model parameters at once during each iteration.
        By default, the number of iterations is set at 200, but this doesn't get anywhere close to convergence.
        Nevertheless, running it for more iterations  doesn't seem to improve cross-validated recommendations.
        
        Ratings are not centered when fitting the model. If you require it, you'll have to center them beforehand
        (e.g. subtracting global/user/item bias from each rating).
        However, I've found centering to make the model worse so I wouldn't recommend it.
        
        The model will only be fit with users and items that are present in both the ratings and the side info data.
        If after filtering there are no matching entries, it will throw an error.
        Although the API can accommodate both item and user side information,
        you can still fit the model without them.
        
        
        Parameters
        ----------
        ratings : list of tuples, pandas data frame or csc sparse matrix
            Ratings data to which to fit the model.
            If a pandas data frame, must contain the columns 'UserId','ItemId' and 'Rating'.
            If a list of tuples, must be in the format (UserId,ItemId,Rating) (will be coerced to data frame)
            If called with 'reindex=False', must be a csc_sparse matrix,
            with rows corresponding to user ID numbers and columns to item ID numbers.
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
            If this is set to 'True', you will need to pass in data as matrices rather than data frames.
        save_entries : bool
            Whether to save the information about which users rated which items.
            This can later be used to filter out already rated items when getting Top-N recommendations for a user.
            Set to False when passing reindex=False.
        random_seed: int
            Random seed to be used to get a starting point.
        print_time : bool
            Whether to print the time spent at different parts of the optimization routine.
        ipopt_options : dict
            Additional options to be passed to IPOPT - you can find the details here:
            https://www.coin-or.org/Ipopt/documentation/node40.html
            
        Attributes
        ----------
        U : numpy.ndarray (nitems, k_main + k + k_user)
            Matrix with the user-factor attributes, containing columns from both factorizations.
            If you wish to extract only the factors used for predictons, slice it like this: U[:,:k_main+k]
        V : numpy.ndarray (nusers, k_main + k + k_item)
            Matrix with the item-factor attributes, containing columns from both factorizations.
            If you wish to extract only the factors used for predictons, slice it like this: V[:,:k_main+k]
        user_orig_to_int : dict
            Mapping of user IDs as they appear in the rating and user_info data to the rows of U.
        item_orig_to_int : dict
            Mapping of item IDs as they appear in the rating and item_info data to the rows of V.
        """
        self.save_entries=save_entries
        self._process_data(ratings,item_info,user_info,reindex)
        m1,m2,m3,m4,k,ux,zx,qx=self._m1,self._m2,self._m3,self._m4,self.k,self.k_main,self.k_item,self.k_user
        if random_seed is not None:
            np.random.seed(random_seed)
        x0=np.random.normal(size=m1*(ux+k+qx)+m2*(ux+k+zx)+m3*(k+zx)+m4*(k+qx))
        self._set_weights(x0)
        w1,w2,w3=self.w1,self.w2,self.w3
        
        Vars=MX.sym('B',m1*(ux+k+qx)+m2*(ux+k+zx)+m3*(k+zx)+m4*(k+qx))
        U=Vars[:m1*(ux+k+qx)]
        Uv=Vars[:m1*(ux+k)].reshape((m1,ux+k))
        Up=Vars[m1*ux:m1*(ux+k+qx)].reshape((m1,k+qx))
        V=Vars[m1*(ux+k+qx):m1*(ux+k+qx)+m2*(ux+k+zx)]
        Vu=Vars[m1*(ux+k+qx):m1*(ux+k+qx)+m2*(ux+k)].reshape((m2,ux+k))
        Vz=Vars[m1*(ux+k+qx)+m2*ux:m1*(ux+k+qx)+m2*(ux+k+zx)].reshape((m2,k+zx))
        Z=Vars[m1*(ux+k+qx)+m2*(ux+k+zx):m1*(ux+k+qx)+m2*(ux+k+zx)+m3*(k+zx)].reshape((k+zx,m3))
        P=Vars[m1*(ux+k+qx)+m2*(ux+k+zx)+m3*(k+zx):].reshape((k+qx,m4))
        
        pred_main=mtimes(Uv,Vu.T)
        err_main=-pred_main*self._W+self._X
        loss=w1*dot(err_main,err_main)+self.reg_param[0]*dot(U,U)+self.reg_param[1]*dot(V,V)+\
        self.reg_param[2]*dot(Z,Z)+self.reg_param[3]*dot(P,P)
        
        if self._prod_arr is not None:
            pred_item=mtimes(Vz,Z)
            err_item=self._prod_arr-pred_item
            loss+=w2*dot(err_item,err_item)
            
        if self._user_arr is not None:
            pred_user=mtimes(Up,P)
            err_user=self._user_arr-pred_user
            loss+=w3*dot(err_user,err_user)
        
        solver = nlpsol("solver", "ipopt", {'x':Vars,'f':loss},{'print_time':print_time,'ipopt':ipopt_options})
        res=solver(x0=x0)
        
        self.U=np.array(res['x'][:m1*(ux+k+qx)].reshape((m1,ux+k+qx)))
        self.V=np.array(res['x'][m1*(ux+k+qx):m1*(ux+k+qx)+m2*(ux+k+zx)].reshape((m2,ux+k+zx)))
        
        del self._X
        del self._W
        del self._prod_arr
        del self._user_arr
    
    def _process_data(self,ratings,item_info,user_info,reindex):
        if not reindex:
            if type(ratings)!=scipy.sparse.csc.csc_matrix:
                raise ValueError('ratings must be a sparse csc matrix')
            if (item_info is not None) and (type(item_info)!=numpy.ndarray):
                raise ValueError('item_info must be a numpy array')
            if (user_info is not None) and (type(user_info)!=numpy.ndarray):
                raise ValueError('user_info must be a numpy array')
            
            if item_info is not None:
                assert ratings.shape[1]==item_info.shape[0]
            if user_info is not None:
                assert ratings.shape[0]==user_info.shape[0]
                
            self._X=ratings
            self._W=1*(self._X>0)
            self.user_orig_to_int={i:i for i in range(ratings.shape[0])}
            self.item_orig_to_int={i:i for i in range(ratings.shape[1])}
            self._user_int_to_orig=self.user_orig_to_int
            self._user_int_to_orig=self.item_orig_to_int
            self._m1=ratings.shape[0]
            self._m2=ratings.shape[1]
            if item_info is not None:
                self._m3=item_info.shape[1]
                self._prod_arr=item_info
            else:
                self._m3=0
                self._prod_arr=none
            if user_info is not None:
                self._m4=user_info.shape[1]
                self._user_arr=user_info
            else:
                self._m4=0
                self._user_arr=None
            self.save_entries=False
        else:
            if type(ratings)==list:
                ratings_df=pd.DataFrame(ratings,columns=['UserId','ItemId','Rating'])
            elif type(ratings)==pd.core.frame.DataFrame:
                if ('UserId' not in ratings.columns.values) or ('ItemId' not in ratings.columns.values) or ('Rating' not in ratings.columns.values):
                    raise ValueError("Ratings data frame must contain the columns 'UserId','ItemId' and 'Rating'")
                ratings_df=ratings[['UserId','ItemId','Rating']].copy()
            else:
                raise ValueError("Ratings must be a list of tuples or pandas data frame")

            if item_info is not None:
                if type(item_info)!=pd.core.frame.DataFrame:
                    raise ValueError('item_info must be a pandas data frame with a column named ItemId')
                if 'ItemId' not in item_info.columns.values:
                    raise ValueError('item_info must be a pandas data frame with a column named ItemId')
                itemset=set(list(item_info.ItemId))
                ratings_df=ratings_df.loc[ratings_df.ItemId.map(lambda x: x in itemset)]

            if user_info is not None:
                if type(user_info)!=pd.core.frame.DataFrame:
                    raise ValueError('user_info must be a pandas data frame with a column named UserId')
                if 'UserId' not in user_info.columns.values:
                    raise ValueError('item_info must be a pandas data frame with a column named UserId')
                userset=set(list(user_info.UserId))
                ratings_df=ratings_df.loc[ratings_df.UserId.map(lambda x: x in userset)]

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
                self._prod_arr=item_info.loc[item_info.ItemId.map(lambda x: x in itemset)]
                self._prod_arr['ItemId']=self._prod_arr.ItemId.map(lambda x: self.item_orig_to_int[x])
                self._prod_arr=self._prod_arr.sort_values('ItemId').set_index('ItemId').as_matrix()
                self._m3=self._prod_arr.shape[1]
                if self._m3==0:
                    raise ValueError("item_info doesn't contain items in common with ratings data")
            else:
                self._m3=0
                self._prod_arr=None

            if user_info is not None:
                userset=set(list(ratings_df.UserId))
                self._user_arr=user_info.loc[user_info.UserId.map(lambda x: x in userset)]
                self._user_arr['UserId']=self._user_arr.UserId.map(lambda x: self.user_orig_to_int[x])
                self._user_arr=self._user_arr.sort_values('UserId').set_index('UserId').as_matrix()
                self._m4=self._user_arr.shape[1]
                if self._m4==0:
                    raise ValueError("user_info doesn't contain items in common with ratings data")
            else:
                self._m4=0
                self._user_arr=None

            ratings_df['UserId']=ratings_df.UserId.map(lambda x: self.user_orig_to_int[x])
            ratings_df['ItemId']=ratings_df.ItemId.map(lambda x: self.item_orig_to_int[x])
            self._m1=cnt_users
            self._m2=cnt_items

            self._X=coo_matrix((ratings_df.Rating,(ratings_df.UserId,ratings_df.ItemId))).tocsc()
            self._W=1*self._X>0

            if self.save_entries:
                self._items_rated_per_user=ratings_df.groupby('UserId')['ItemId'].agg(lambda x: set(x))
            
    def _set_weights(self,x0):
        m1,m2,m3,m4,k,ux,zx,qx=self._m1,self._m2,self._m3,self._m4,self.k,self.k_main,self.k_item,self.k_user
        
        if self.standardize:
            self._nmain=np.sum(self._W)
            if self._prod_arr is not None:
                self._nitem=self._prod_arr.shape[0]*self._prod_arr.shape[1]
            else:
                self._nitem=1
            if self._user_arr is not None:
                self._nuser=self._user_arr.shape[0]*self._user_arr.shape[1]
            else:
                self._nuser=1
        else:
            self._nmain=1
            self._nitem=1
            self._nuser=1
            
        if self.reweight:
            mat1=x0[:m1*(ux+k)].reshape(m1,ux+k)
            mat2=x0[m1*(ux+k+qx):m1*(ux+k+qx)+m2*(ux+k)].reshape(ux+k,m2)
            pred=mat1.dot(mat2)
            err=self._X-self._W.multiply(pred)
            err_main=1/np.sum(err.power(2))
            err_tot=err_main
            
            if self._prod_arr is not None:
                mat1=x0[m1*(ux+k+qx)+m2*ux:m1*(ux+k+qx)+m2*(ux+k+zx)].reshape(m2,k+zx)
                mat2=x0[m1*(ux+k+qx)+m2*(ux+k+zx):m1*(ux+k+qx)+m2*(ux+k+zx)+m3*(k+zx)].reshape(k+zx,m3)
                pred=mat1.dot(mat2)
                err=self._prod_arr-pred
                err_item=1/np.sum(err**2)
                err_tot+=err_item
                
            if self._user_arr is not None:
                mat1=x0[m1*ux:m1*(ux+k+qx)].reshape(m1,k+qx)
                mat2=x0[m1*(ux+k+qx)+m2*(ux+k+zx)+m3*(k+zx):].reshape(k+qx,m4)
                pred=mat1.dot(mat2)
                err=self._user_arr-pred
                err_user=1/np.sum(err**2)
                err_tot+=err_user
                
            self.w1=self.w_main*err_main/err_tot
            if self._prod_arr is not None:
                self.w2=self.w_item*err_item/err_tot
            if self._user_arr is not None:
                self.w3=self.w_user*err_user/err_tot
                
        elif self.standardize:
            self._nmain=np.sum(self._W)
            self.w1=self.w_main/nitem
            if self._prod_arr is not None:
                nitem=self._prod_arr.shape[0]*self._prod_arr.shape[1]
                self.w2=self.w_item/nitem
            if self._user_arr is not None:
                nuser=self._user_arr.shape[0]*self._user_arr.shape[1]
                self.w3=self.w_user/nuser
        else:
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
            
        return self.U[user,:self.k_main+self.k].dot(self.V[item,:self.k_main+self.k].T)
    
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
            
        preds=-self.U[user,:self.k_main+self.k].dot(self.V[:,:self.k_main+self.k].T)
        best=np.argsort(preds)
        if self.save_entries and filter_rated:
            out=list()
            for i in range(self._m2):
                if best[i] not in self._items_rated_per_user[user]:
                    if scores:
                        out.append(best[i])
                    else:
                        out.append((best[i],-preds[best[i]]))
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
            return self.U[self.user_orig_to_int[UserId],:]
        except:
            raise ValueError('Invalid user')
            
    def get_item_factor_vector(self, ItemId):
        """Get the Item-LatentFactor vector for a given item"""
        try:
            return self.V[self.item_orig_to_int[ItemId],:]
        except:
            raise ValueError('Invalid user')
