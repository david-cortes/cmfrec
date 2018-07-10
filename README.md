# Collective Matrix Factorization

Python implementation of collective matrix factorization, based on the paper _Relational learning via collective matrix factorization (A. Singh, 2008)_, with some enhancements. This is a collaborative filtering model for recommender systems that takes as input explicit item ratings and side information about users and/or items. The overall idea was extended here to also be able to do cold-start recommendations (of users and items that were not in the training data). Implementation is done in TensorFlow and optimization is with L-BFGS.

The extended version of the paper ("Relational learning via Collective Matrix Factorization") can be downloaded here:
[http://ra.adm.cs.cmu.edu/anon/usr/ftp/ml2008/CMU-ML-08-109.pdf](http://ra.adm.cs.cmu.edu/anon/usr/ftp/ml2008/CMU-ML-08-109.pdf)

## Basic model

The model consist in predicting the rating that a user would give to an item by performing a low-rank matrix factorization of explicit ratings `X~=AB'`, using side information about the items (such as movie tags) and/or users (such as their demographic info) by also factorizing the item side info matrix and/or the user side info matrix `X~=AB', U~=AC', I~=BD'`, sharing the same item/user-factor matrix used to factorize the ratings, or sharing only some of the latent factors.

This also has the side effect of allowing to recommend for users and items for which there is side information but no ratings, although these predictions might not be as high quality.

For more details on the formulas, see the IPython notebook example.

## Instalation

Package is available on PyPI, can be installed with:

```pip install cmfrec```

## Usage
```python
import pandas as pd, numpy as np
from cmfrec import CMF

## simulating some movie ratings
nusers = 10**2
nitems = 10**2
nobs = 10**3
ratings = pd.DataFrame({
	'UserId' : np.random.randint(nusers, size=nobs),
	'ItemId' : np.random.randint(nitems, size=nobs),
	'Rating' : np.random.randint(low=1, high=6, size=nobs)
	})

## random product side information
user_dim = 10
user_attributes = pd.DataFrame(np.random.normal(size=(nusers, user_dim)))
user_attributes['UserId'] = np.arange(nusers)
user_attributes = user_attributes.sample(int(nusers/2), replace=False)

item_dim = 5
item_attributes = pd.DataFrame(np.random.normal(size=(nitems, item_dim)))
item_attributes['ItemId'] = np.arange(nitems)
item_attributes = item_attributes.sample(int(nitems/2), replace=False)

# fitting a model and making some recommendations
recommender = CMF(k=20, k_main=3, k_user=2, k_item=1, reg_param=1e-4)
recommender.fit(ratings=ratings, user_info=user_attributes, item_info=item_attributes,
	cols_bin_user=None, cols_bin_item=None)
recommender.topN(user=4, n=10)
recommender.topN_cold(attributes=np.random.normal(size=user_dim), n=10)
recommender.predict(user=0, item=0)
recommender.predict(user=[0,0,1], item=[0,1,0])

# adding more users and items without refitting
recommender.add_item(new_id=10**3, attributes=np.random.normal(size=item_dim), reg='auto')
recommender.add_user(new_id=10**3, attributes=np.random.normal(size=user_dim), reg=1e-3)
recommender.topN(10**3)
recommender.predict(user=10**3, item=10**3)

## full constructor call
recommender = CMF(k=30, k_main=0, k_user=0, k_item=0,
	 w_main=1.0, w_user=1.0, w_item=1.0, reg_param=1e-4,
	 offsets_model=False, nonnegative=False, maxiter=1000,
	 standardize_err=True, reweight=False, reindex=True,
	 center_ratings=True, add_user_bias=True, add_item_bias=True,
	 center_user_info=False, center_item_info=False,
	 user_info_nonneg=False, item_info_nonneg=False,
	 keep_data=True, save_folder=None, produce_dicts=True,
	 random_seed=None, verbose=True)

```

Users and items are internally reindexed, so you can use strings or non-consecutive numbers as IDs when passing data to the `CMF` object's methods.

For more details see the online documentation.

## Documentation

Documentation is available at readthedocs: [http://cmfrec.readthedocs.io/en/latest/](http://cmfrec.readthedocs.io/en/latest/)

It is also internally documented through docstrings (e.g. you can try `help(cmfrec.CMF))`, `help(cmfrec.CMF.fit)`, etc.

For a detailed example using the MovieLens data with user demographic info and movie genres see the IPython notebook [Collaborative filtering with side information](http://nbviewer.jupyter.org/github/david-cortes/cmfrec/blob/master/example/cmfrec_movielens_sideinfo.ipynb).

## Model details

By default, the function to minimize is as follows:

```
L(A,B,C,D) = w_main*sum_sq(X-AB' - Ubias_{u} - Ibias_{i})/|X| + w_user*sum_sq(U-AC')/|U| + w_item*sum_sq(I-BD')/|I| + regl.
	regl. = reg*(norm(A)^2+norm(B)^2+norm(C)^2+norm(D)^2)
 ```

Where:
* X is the ratings matrix (considering only non-missing entries).
* I is the item-attribute matrix (only supports dense inputs).
* U is the user-attribute matrix (only supports dense inputs).
* A, B, C, D are lower-dimensional matrices (the model parameters).
* |X|, |I|, |U| are the number of non-missing entries in each matrix.
* The matrix products might not use all the rows/columns of these matrices at each factorization
 (this is controlled with k_main, k_item and k_user).
* Ubias_{u} and Ibias_{i} are user and item biases, defined for each user and item ID.

Additionally, for binary columns (having only 0/1 values), it can apply a sigmoid function to force them to be between zero and one. The loss with respect to the real value is still squared loss though (`(sigmoid(pred) - real)^2`), so as to put it in the same scale as the rest of the variables.

The matrix-products might not use all the rows/columns of these shared latent factor matrices at each factorization (this is controlled with `k_main`, `k_item` and `k_user` in the initialization). Although the package API has arguments for both user and item side information, you can fit the model with only one or none of them.

The model benefits from adding side information about users and items for which there are no ratings, but recommendations about these users and items might not be very good.

Note that, in the simplest case with all factors shared and all matrices weighted the same, the model simplifies to factorizing an extended block matrix `X_ext = [[X, U], [I', .]]`, which can be done using any other matrix factorization library (e.g. pyspark’s `ALS` module).

Alternatively, it can also fit a different model formulation in which, in broad terms, ratings are predicted from latent 'coefficients' based on their attributes, and an offset is added to these coefficients based on the rating history of the users, similar in spirit to [2]  (see references at the bottom), but with all-normal distributions:

```
 L = sum_sq(X - (A+UC)(B+ID)' - Ubias_{u} - Ibias_{i}) + regl.
 ```

This second formulation usually takes longer to fit (requires more iterations), and is more recommended if the side information about users and items is mostly binary columns, and for cold-start recommendations. To use this alternative formulation, pass `offsets_model=False` to the CMF constructor.

For a web-scale implementation (without in-memory data) of the first algorithm see the implementation in Vowpal Wabbit:
[https://github.com/JohnLangford/vowpal_wabbit/wiki/Matrix-factorization-example](https://github.com/JohnLangford/vowpal_wabbit/wiki/Matrix-factorization-example)

## Cold-start recommendations

When using side information about users and/or items, it is possible to add new latent factor vectors based on the factorization of the side information alone. In a low-rank factorization, if one matrix is fixed, the other one can be calculated with a closed-form solution. In this case, the latent factor vectors for new users and items can be calculated with the same updates used in alternating least squares, in such a way that they minimize their error on the side information factorization - as these latent factors are shared with the factorization of the ratings (user feedback), they can at the same time be used for making recommendations.

For the alternative 'offsets model' formulation, cold-start recommendations are obtained by a simple vector-matrix product of the attributes with the model coefficients.

You might to tune the model parameters differently when optimizing for cold-start and warm-start recommendations. See the documentation for some tips.

## Implementation Notes

Calculations of objective and gradient are done with TensorFlow, interfacing an external optimizer. Since it allows for easy slight reformulations of the problem, you can easily control small aspects such as whether to divide the error by the number of entries in the matrices or not, whether or not to include user/item biases, etc.

The implementation here is not entirely true to the paper, as the model is fit with full L-BFGS updates rather than stochastic gradient descent or stochastic Newton, i.e. it calculates errors for the whole data and updates all model parameters at once during each iteration. This usually leads to better local optima and less need for parameter tuning, but is not as scalable. The L-BFGS implementation used is from SciPy, which is not entirely multi-threaded, so  if your CPU has many cores or you have many GPUs, you can expect an speed up from parallelization on the calculations of the objective function and the gradient, but not so much on the calculations that happen inside  L-BFGS.

As a reference point, 1,000 iterations (usually enough to fit a model with high regularization) over the movielens-1M and 1128 movie tags per movie + user demographics takes around 15 minutes in a regular desktop computer and uses around 2GB of RAM.

The package has only been tested under Python 3.


## Some comments

This kind of model requires a lot more hyperparameter tuning that regular low-rank matrix factorization, and fitting a model with badly-tuned parameters might result in worse recommendations compared to discarding the side information. In general, using lower regularization parameters will require more iterations to converge. Also note that, by default, the sums of squared errors in the loss function are divided by the number of entries, so the optimal regularization parameters will likely be very, very small, and the larger the inputs, the smaller the optimal regularization values (you can change this behavior by passing `standardize_err=False`, but be sure to then tune the weights for each factorization).

If your dataset is larger than the MovieLens ratings, adding product side information is unlikely to add more predictive power, but good user side information might still be valuable.

## References
* Singh, Ajit P., and Geoffrey J. Gordon. "Relational learning via collective matrix factorization." Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2008.
* Wang, Chong, and David M. Blei. "Collaborative topic modeling for recommending scientific articles." Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 201.
