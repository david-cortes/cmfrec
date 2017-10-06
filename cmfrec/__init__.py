import pandas as pd, numpy as np
from casadi import MX,nlpsol,dot,mtimes
from scipy.sparse import coo_matrix
pd.options.mode.chained_assignment = None

class CMF:
    def __init__(self,k=50,k_main=0,k_sec=0,w1=1.0,w2=1.0,reg_param=(.1,.1,.1),reweight=True):
        """
        Collective matrix factorization model (for recommenders systems with explicit data)
        
        
        Fits a double matrix factorization model to ratings data along with item side information,
        by factorizing both matrices with shared item-factors, e.g.:
        X~=UV^T and M~=VZ^T
        More details can be found on the paper Relational learning via collective matrix factorization (2008) from Singh, A. P., & Gordon, G. J.
        
        The model is fit using BFGS with IPOPT as the workhorse, interfaced through CasADi
        
        Note
        ----
        All users and items are reindexed internally, so you can use non-integer IDs and don't have to worry about enumeration gaps.

        Parameters
        ----------
        k : int
            Number of common (shared) latent factors to use.
        k_main : int
            Number of additional latent factors to use for the ratings factorization.
        k_sec : int
            Number of additional latent factors to use for the item attributes factorization.
        w1 : float
            Weight to assign to the root squared error in factorization of the ratings matrix.
        w2: float
             Weight to assign to the root squared error in factorization of the item attributes matrix.
        reg_param : tuple of floats
            Regularization parameter for each matrix, in this order:
            1) User-Factor, 2) Item-Factor, 3) Attribute-Factor.
        reweight : bool
            Whether to automatically reweight the errors of factorizing each matrix so that both get similar influence,
            accounting for the number of elements in each and the magnitudes of entries.
            This is done by calculating the initial root squared errors with randomly initialized factor matrices,
            but it's not guaranteed to be a good criterion.
            
            It's still recommended to scale the entries of either the ratings or the attributes matrix so that they are in a similar scale
            (e.g. if the ratings are in [1,5], the attributes should ideally be in the same range and not [-10^3,10^3]).
        """
        self.k=k
        self.k1=k_main
        self.k2=k_sec
        self.w1=w1
        self.w1=w2
        self.reg1=reg_param[0]
        self.reg2=reg_param[1]
        self.reg3=reg_param[2]
        self.reweight=reweight
        
    def fit(self,ratings,item_info,random_seed=None,print_time=False,ipopt_options={'tol':1e2,'max_iter':200,
         'hessian_approximation':'limited-memory','linear_scaling_on_demand':'no',"print_level":0}):
        """
        Fit the model to ratings data and item side info, using BFGS
        
        
        Note
        ----
        Never recommended to change the hessian approximation to exact hessian.
        The model is fit with full BFGS updates (not stochastic gradient descent or stochastic Newton),
        i.e. it calculates errors for the whole data and updates all model parameters at each iteration.
        By default, the number of iterations is set at 200, but this doesn't get anywhere close to convergence.
        Nevertheless, running it for more iterations  doesn't seem to improve cross-validated predictions.
        
        Ratings are not centered when fitting the model. If you require it, you'll have to center them beforehand
        (e.g. subtracting global/user/item bias from each rating).
        
        The model will only be fit with items that are present in both the ratings and the item side info.
        
        
        Parameters
        ----------
        ratings: list of tuples or pandas data frame
            Ratings data to which to fit the model.
            If a pandas data frame, must contain the columns 'UserId','ItemId' and 'Rating'.
            If a list of tuples, must be in the format (UserId,ItemId,Rating) (will be coerced to data frame)
        item_info: pandas data frame
            Side information about the items (i.e. their attributes, in a table).
            Must contain a column named ItemId.
        random_seed: int
            Random seed to be used to get a starting point.
        print_time: bool
            Whether to print the time spent at different parts of the optimization routine.
        ipopt_options: dict
            Additional options to be passed to IPOPT - you can find the details here:
            https://www.coin-or.org/Ipopt/documentation/node40.html
        """
        
        self._process_data(ratings,item_info)
        
        if random_seed is not None:
            np.random.seed(random_seed)
        m1,m2,m3,k,k1,k2=self._m1,self._m2,self._m3,self.k,self.k1,self.k2
        
        x0=np.random.normal(size=m1*(k1+k)+m2*(k1+k+k2)+m3*(k+k2))
        if self.reweight:
            err1=np.sum((self._X-self._W.multiply(x0[:m1*(k1+k)].reshape(m1,k1+k).dot(x0[m1*(k1+k):m1*(k1+k)+m2*(k1+k)].reshape(k1+k,m2)))).power(2))
            err2=np.linalg.norm(x0[m1*(k1+k)+m2*k1:m1*(k1+k)+m2*(k1+k+k2)].reshape(m2,k+k2).dot(x0[m1*(k1+k)+m2*(k1+k+k2):].reshape(k+k2,m3))-self._prod_arr)
            w1=err2/(err1+err2)
            w2=err1/(err1+err2)
        else:
            w1=self.w1
            w2=self.w2
        
        Vars=MX.sym('Y',m1*(k1+k)+m2*(k1+k+k2)+m3*(k+k2))
        U=Vars[:m1*(k1+k)].reshape((m1,k1+k))
        Vu=Vars[m1*(k1+k):m1*(k1+k)+m2*(k1+k)].reshape((m2,k1+k))
        Vz=Vars[m1*(k1+k)+m2*k1:m1*(k1+k)+m2*(k1+k+k2)].reshape((m2,k+k2))
        V=Vars[m1*(k1+k):m1*(k1+k)+m2*(k1+k+k2)]
        Z=Vars[m1*(k1+k)+m2*(k1+k+k2):].reshape((k+k2,m3))
        pred=mtimes(U,Vu.T)
        pred2=mtimes(Vz,Z)
        err_main=-pred*self._W+self._X
        err_sec=self._prod_arr-pred2

        reg=1
        loss=w1*dot(err_main,err_main)+w2*dot(err_sec,err_sec)+self.reg1*dot(U,U)+self.reg2*dot(V,V)+self.reg3*dot(Z,Z)
        solver = nlpsol("solver", "ipopt", {'x':Vars,'f':loss},{'print_time':print_time,'ipopt':ipopt_options})
        res=solver(x0=x0)
        
        
        self.U=np.array(res['x'][:m1*(k1+k)].reshape((m1,k1+k)))
        self.V=np.array(res['x'][m1*(k1+k):m1*(k1+k)+m2*(k1+k+k2)].reshape((m2,k1+k+k2)))
        self.Z=np.array(res['x'][m1*(k1+k)+m2*(k1+k+k2):].reshape((m3,k+k2)))
        
        del self._X
        del self._W
        del self._prod_arr
    
    def _process_data(self,ratings,item_info):
        self._user_orig_to_int=dict()
        self._item_orig_to_int=dict()
        self._user_int_to_orig=dict()
        self._item_int_to_orig=dict()
        
        self._attributename_to_int=dict()
        self._int_to_attributename=dict()
        
        if type(ratings)==list:
            ratings_df=pd.DataFrame(ratings,columns=['UserId','ItemId','Rating'])
        elif type(ratings)==pd.core.frame.DataFrame:
            if ('UserId' not in ratings.columns.values) or ('ItemId' not in ratings.columns.values) or ('Rating' not in ratings.columns.values):
                raise ValueError("Ratings data frame must contain the columns 'UserId','ItemId' and 'Rating'")
            ratings_df=ratings[['UserId','ItemId','Rating']].copy()
        else:
            raise ValueError("Ratings must be a list of tuples or pandas data frame")
            
        if type(item_info)!=pd.core.frame.DataFrame:
            raise ValueError('item_info must be a pandas data frame with a column named ItemId')
        if 'ItemId' not in item_info.columns.values:
            raise ValueError('item_info must be a pandas data frame with a column named ItemId')
        self._prod_arr=item_info.loc[item_info.ItemId.map(lambda x: x in set(list(ratings_df.ItemId)))]
        ratings_df=ratings_df.loc[ratings_df.ItemId.map(lambda x: x in set(list(self._prod_arr.ItemId)))]
        
        cnt_users=0
        cnt_items=0
        for i in ratings_df.itertuples():
            if i.UserId not in self._user_orig_to_int:
                self._user_orig_to_int[i.UserId]=cnt_users
                self._user_int_to_orig[cnt_users]=i.UserId
                cnt_users+=1
            if i.ItemId not in self._item_orig_to_int:
                self._item_orig_to_int[i.ItemId]=cnt_items
                self._item_int_to_orig[cnt_items]=i.ItemId
                cnt_items+=1

        ratings_df['UserId']=ratings_df.UserId.map(lambda x: self._user_orig_to_int[x])
        ratings_df['ItemId']=ratings_df.ItemId.map(lambda x: self._item_orig_to_int[x])
        self._prod_arr['ItemId']=self._prod_arr.ItemId.map(lambda x: self._item_orig_to_int[x])
        
        cnt_cols=0
        for i in self._prod_arr.columns.values:
            if i not in self._attributename_to_int:
                self._attributename_to_int[i]=cnt_cols
                self._int_to_attributename[cnt_cols]=i
                cnt_cols+=1
        
        self._m1=cnt_users
        self._m2=cnt_items
        self._m3=self._prod_arr.shape[1]-1
        
        self._prod_arr=self._prod_arr.sort_values('ItemId').set_index('ItemId').as_matrix()
        self._X=coo_matrix((ratings_df.Rating,(ratings_df.UserId,ratings_df.ItemId))).tocsc()
        self._W=1*self._X>0
        
    
    def predict(self,UserId,ItemId):
        """Predict the rating that a given user would give to a given item"""
        try:
            user=self._user_orig_to_int[UserId]
        except:
            raise ValueError('Invalid user')
        try:
            item=self._item_orig_to_int[ItemId]
        except:
            raise ValueError('Invalid item')
            
        return self.U[user,:].dot(self.V[item,:self.k1+self.k].T)
    
    def top_n(self,UserId,n,scores=False):
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
        
        Returns
        -------
        list
            Top-N recommended items for the user.
            If 'scores=True', list of tuples containing (Item,Score)
        """
        try:
            user=self._user_orig_to_int[UserId]
        except:
            raise ValueError('Invalid user')
            
        preds=self.U[user,:].dot(self.V[:,:self.k1+self.k].T)
        best=np.argsort(-preds)
        if not scores:
            return [self._item_int_to_orig[i] for i in best[:n]]
        else:
            return [(self._item_int_to_orig[i],-i) for i in best[:n]]
        
    def get_user_factor_vector(self,UserId):
        """Get the User-LatentFactor vector for a given user"""
        try:
            return self.U[self._user_orig_to_int[UserId],:]
        except:
            raise ValueError('Invalid user')
            
    def get_item_factor_vector(self,ItemId):
        """Get the Item-LatentFactor vector for a given item"""
        try:
            return self.V[self._item_orig_to_int[ItemId],:]
        except:
            raise ValueError('Invalid user')
            
    def get_prodattribute_factor_vector(self,AttributeName):
        """Get the ItemAttribute-LatentFactor vector for a given item item"""
        try:
            return self.Z[self._attributename_to_int[AttributeName],:]
        except:
            raise ValueError('Invalid attribute')
