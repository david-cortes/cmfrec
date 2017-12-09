# cmfrec

Python implementation of the collaborative filtering algorithm for explicit feedback data with item and/or user side information described in Singh, A. P., & Gordon, G. J. (2008, August). Relational learning via collective matrix factorization. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 650-658). ACM.

The extended version of the paper (Relational learning via Collective Matrix Factorization) can be downloaded here:
[http://ra.adm.cs.cmu.edu/anon/usr/ftp/ml2008/CMU-ML-08-109.pdf](http://ra.adm.cs.cmu.edu/anon/usr/ftp/ml2008/CMU-ML-08-109.pdf)

## Basic model
The model consist in predicting the rating that a user would give to an item by a low-rank matrix factorization `X~=UV'`, trying to add side information about the items (such as movie tags) by also factorizing the item-attribute matrix, sharing the same item-factor matrix for both factorizations (or only sharing some of the latent factors), i.e. trying to minimize a weighted sum of errors from both low-rank factorizations.

By default, the function to minimize is as follows:

```L = w_main*norm(X-AB')^2/nX + w_item*norm(I-BC')^2/nI + w_user*norm(U-AD')^2/nU + reg_param*(norm(A)^2+norm(B)^2+norm(C)^2+norm(D)^2)```

Where:
* X is the ratings matrix (considering only non-missing entries)
* I is the item-attribute matrix (only supports dense, i.e. all non-missing entries)
* U is the user-attribute matrix (only supports dense, i.e. all non-missing entries)
* A, B, C, D are lower-dimensional matrices (the model parameters)
* nX, nI, nU are the number of entries in each matrix

The matrix-products might not use all the rows/columns of these shared matrices at each factorization (this is controlled with `k_main`, `k_item` and `k_user` in the initialization). Although the package API has arguments for both user and item side information, you can fit the model with only one or none of them.

Note that, in the simplest case with all factors shared and all matrices weighted the same, the model simplifies to factorizing an extended block matrix `X_ext = [[X, U], [I’, .]]`, which can be done using any other matrix factorization library (e.g. pyspark’s ALS module).

## Instalation
Package is available on PyPI, can be installed with

```pip install cmfrec```

## Usage
```
import pandas as pd, numpy as np
from cmfrec import CMF

# simulating some movie ratings
ratings=pd.DataFrame([(i,j,np.round(np.random.uniform(1,5),0)) for i in range(100) for j in range(40) if np.random.random()>.3],columns=['UserId','ItemId','Rating'])

# random product side information
prod_attributes=pd.DataFrame(np.random.normal(size=(40,60)))
prod_attributes['ItemId']=[i for i in range(prod_attributes.shape[0])]

# fitting a model and making some recommendations
recc=CMF(k=20,k_main=3,k_item=5,reg_param=1e-4)
recc.fit(ratings,prod_attributes)
recc.top_n(UserId=4, n=10)
recc.predict(UserId=0, ItemId=0)
```

For a more detailed example using the MovieLens data with user demographic info and movie genres see [this IPython notebook](http://nbviewer.jupyter.org/github/david-cortes/cmfrec/blob/master/example/cmfrec_movielens_sideinfo.ipynb).

The code is documented internally through docstrings (e.g. you can try `help(CMF)`)

## Implementation Notes
The implementation here is not entirely true to the paper, as the model is fit with full L-BFGS updates rather than stochastic Newton, thus not recommended for web-scale datasets. Most of the calculations are done in Tensorflow, interfacing an external L-BFGS solver, thus speed should be quite fast. As a reference point, 1,000 iterations (usually enough to fit a model with high regularization) over the movielens-1M and 1128 movie tags per movie + user demographics takes around 15 minutes in a regular desktop computer and uses around ~2GB RAM.

If you want to try different hyperparameters on the same data, it also contains a faster `._fit` method that can be used after `.fit` has been called at least once, as it will save the data but reindexed internally. Also, if your data doesn’t need reindexing, you can pass `reindex=False` to `.fit` to speed up the process.

The package has only been tested under Python 3.


## Some comments
This kind of model requires a lot more hyperparameter tuning that regular low-rank matrix factorization, and fitting a model with badly-tuned parameters might result in worse recommendations compared to discarding the side information. In general, using lower regularization parameters will require more iterations to converge. Also note that, by default, the sums of squared errors in the loss function are divided by the number of entries, so the optimal regularization parameters will likely be very, very small. Larger numbers of users, items, ratings, latent factors, etc. also result in smaller optimal values for the regularization parameters.

If your dataset is larger than the MovieLens ratings, adding product side information is unlikely to add more predictive power, but good user side information might still be valuable.

## References
* Singh, A. P., & Gordon, G. J. (2008, August). Relational learning via collective matrix factorization. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 650-658). ACM.
