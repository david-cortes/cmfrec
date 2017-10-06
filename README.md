# cmfrec

Python implementation of the collaborative filtering algorithm for explicit feedback data with item side information described in Singh, A. P., & Gordon, G. J. (2008, August). Relational learning via collective matrix factorization. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 650-658). ACM.

The extended version of the paper (Relational learning via Collective Matrix Factorization) can be downloaded here:
[http://ra.adm.cs.cmu.edu/anon/usr/ftp/ml2008/CMU-ML-08-109.pdf](http://ra.adm.cs.cmu.edu/anon/usr/ftp/ml2008/CMU-ML-08-109.pdf)

## Basic model
The model consist in predicting the rating that a user would give to an item by a low-rank matrix factorization X~=UV’, trying to add side information about the items (such as movie tags) by also factorizing the item-attribute matrix, sharing the same item-factor matrix for both factorizations (or only sharing some of the latent factors), i.e. trying to minimize a weighted sum of errors from both low-rank factorizations:

```w1*sum((X-UV‘)^2) + w2*sum((M-VZ‘)^2) + lambda*(norm(U)+norm(V)+norm(Z)```

(Where X ist the ratings matrix, M is the item attribute matrix, and U, V and Z are lower dimensional matrices)

## Instalation
Package is available on PyPI, can be installed with

```pip install cmfrec```

## Usage
```
import pandas as pd, numpy
from cmfrec import CMF

# simulating some movie ratings
ratings=pd.DataFrame([(i,j,np.round(np.random.uniform(1,5),0)) for i in range(100) for j in range(40) if np.random.random()>.3],columns=['UserId','ItemId','Rating'])

# random product side information
prod_attributes=pd.DataFrame(np.random.normal(size=(40,60)))
prod_attributes['ItemId']=[i for i in range(prod_attributes.shape[0])]

# fitting a model and making some recommendations
recc=CMF(k=20,k_main=3,k_sec=5,reg_param=(0,0,0))
recc.fit(ratings,prod_attributes,ipopt_options={'hessian_approximation':'limited-memory','max_iter':100,'print_level':0})recc.predict(UserId=10, ItemId=5)
recc.top_n(UserId=4, n=10)
```

The code is documented internally through docstrings (e.g. you can try `?CMF`)

## Implementation Notes
The implementation here is not entirely true to the paper, as the model is fit with full BFGS updates rather than Newton-Rhapsody, SGD or stochastic Newton, thus not recommended for large datasets (be wary of RAM memory consumption!). The code quality is research-grade and the optimization routine doesn’t exploit the whole structure of the hessian. As a point of reference, 200 iterations over the movielens-1M and 1128 movie tags per movie takes around 1 hour in a regular computer.


All the optimization routine is done with IPOPT as the workhorse, interfaced through CasADi.

The package has only been tested under Python 3.


## Some comments
Although the idea seems very sound, it requires a lot more hyperparameter tuning that regular low-rank matrix factorization, and fitting a model with badly-tuned parameters will result in worse recommendations compared to discarding the item side information.

If your dataset is larger than MovieLens ratings, adding product side information is unlikely to add more predictive power.

## References
* Singh, A. P., & Gordon, G. J. (2008, August). Relational learning via collective matrix factorization. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 650-658). ACM.
