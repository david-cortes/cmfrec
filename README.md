# Collective Matrix Factorization

Implementation of collective matrix factorization, based on _Relational learning via collective matrix factorization_ ([2]), with some enhancements and alternative models for cold-start recommendations as described in _Cold-start recommendations in Collective Matrix Factorization_ ([1]), and adding implicit-feedback variants as described in _Collaborative filtering for implicit feedback datasets_ ([3]).

This is a hybrid collaborative filtering model for recommender systems that takes as input either explicit item ratings or implicit-feedback data, and side information about users and/or items (although it can also fit pure collaborative-filtering and pure content-based models). The overall idea was extended here to also be able to do cold-start recommendations (for users and items that were not in the training data but which have side information available).

Although the package was developed with recommender systems in mind, it can also be used in other domains (e.g. topic modeling, dimensionality reduction, missing value imputation) - just take any mention of users as rows in the main matrix, any mention of items as columns, and use the "explicit" models.

For more information about the implementation here, or if you would like to cite this in your research, see ["Cold-start recommendations in Collective Matrix Factorization"](https://arxiv.org/abs/1809.00366)

For a similar package with Poisson distributions see [ctpfrec](https://github.com/david-cortes/ctpfrec).

Written in C with Python and R interfaces. An additional Ruby interface can be found [here](https://github.com/ankane/cmfrec).

## Update 2020-03-20

The package has been rewritten in C with Python wrappers. If you've used earlier versions of this package which relied on Tensorflow for the calculations (and before that, Casadi), the optimal hyperparameters will be very different now as it has changed some details of the loss function such as not dividing some terms by the number of entries.

The new version is faster, multi-threaded, and has some new functionality, but if for some reason you still need the old one, it can be found under the git branch "tensorflow".

## Update 2020-10-27

The package has now introduced a conjugate gradient method for the ALS procedures and has undergone many improvements in terms of speed, memory usage, and numerical precision. The models with ALS-CG are now competitive in speed against libraries such as `implicit` or `rsparse`. The C code now also contains the full prediction API.

## Update 2020-11-04

The package can now automatically generate so-called "implicit features" for the explicit-feedback models (see [5] and similar) and use them in addition to real side information, even if said side information is not sparse.

## Highlights

* Can fit factorization models with or without user and/or item side information.
* Can fit the usual explicit-feedback model as well as the implicit-feedback model with weighted binary entries (see [3]).
* For the explicit-feedback model, can automatically add implicit features (created from the same "X" data).
* Can be used for cold-start recommendations (when using side information).
* Supports user and item biases in the explicit-feedback models (these are not just pre-estimated beforehand as in other software).
* Can fit models with non-negativity constraints on the factors and/or with L1 regularization.
* Provides an API for top-N recommended lists and for calculating latent factors from new data.
* Can work with both sparse and dense matrices for each input (e.g. can also be used as a general missing-value imputer for 2-d data), and can work efficiently with a mix of dense and sparse inputs.
* Can produce factorizations for variations of the problem such as sparse inputs with missing-as-zero instead of missing-as-unknown (e.g. when used for dimensionality reduction).
* Can use either an alternating least-squares procedure (ALS) or a gradient-based procedure using an L-BFGS optimizer for the explicit-feedback models (the package bundles a modified version of [Okazaki's C implementation](https://github.com/chokkan/liblbfgs)).
* For the ALS option, can use either the exact Cholesky method or the faster conjugate gradient method (see [4]).
* Can produce models with constant regularization or with dynamically-adjusted regularization as in [7].
* Provides a content-based model and other models aimed at better cold-start recommendations.
* Provides an intercepts-only "most-popular" model for non-personalized recommendations, which can be used as a benchmark as it uses the same hyperparameters as the other models.
* Allows variations of the original collective factorization models such as setting some factors to be used only for one factorization, setting different weights for the errors on each matrix, or setting different regularization parameters for each matrix.
* Can use sigmoid transformations for binary-distributed columns in the side info data.
* Can work with large datasets (supports arrays/matrices larger than `INT_MAX`).
* Supports observation weights (for the explicit-feedback models).

## Basic idea

The model consist in predicting the rating (weighted confidence for implicit-feedback case) that a user would give to an item by performing a low-rank factorization of an interactions matrix (e.g. ratings)
```
X ~ A * t(B)
```
Using side information about the items (such as movie tags) and/or users (such as their demographic info) by also factorizing the item side info matrix and/or the user side info matrix
```
U ~ A * t(C),   I ~ B * t(D)
```
Sharing the same item/user-factor matrix used to factorize the ratings, or sharing only some of the latent factors.

This also has the side effect of allowing recommendations for users and items for which there is side information but no ratings, although these predictions might not be as high quality.

Alternatively, can produce factorizations in wich the factor matrices are determined from the attributes directly (e.g. `A = U * C`), with or without a free offset.

While the method was initially devised for recommender systems, can also be used as a general technique for dimensionality reduction by taking the `A` matrix as low-dimensional factors, which can be calculated for new data too.

## Instalation

* Python:

```
pip install cmfrec
```

(Note: NumPy must already be installed in the Python environment before attempting to install `cmfrec`)

As it contains C code, it requires a C compiler. On Windows, this usually means it requires a Visual Studio Build Tools installation (with MSVC140 component for conda, or MinGW + GCC), and if using Anaconda, might also require configuring it to use said Visual Studio instead of MinGW.

**Note for macOS users:** on macOS, this package will compile without multi-threading capabilities. This is due to default apple's redistribution of clang not providing OpenMP modules, and aliasing it to gcc which causes confusions in build scripts. If you have a non-apple version of clang with the OpenMP modules, or if you have gcc installed, you can compile this package with multi-threading enabled by setting up an environment variable `ENABLE_OMP=1`:

```
export ENABLE_OMP=1
pip install cmfrec
```
(Alternatively, can also pass argument enable-omp to the setup.py file: python `setup.py install enable-omp`)

Will also by default use MKL if it finds it - for OpenBLAS can set an environment variable `USE_OPENBLAS=1` or pass argument `openblas` to `setup.py`.

* R:

Latest version (recommended):
```r
remotes::install_github("david-cortes/cmfrec")
```

Older version from CRAN (has some minor bugs):
```r
install.packages("cmfrec")
```

* Ruby:

See [external repository](https://github.com/ankane/cmfrec).

* C:

Package can be built as a shared library - see the CMake build file for options:
```
git clone https://www.github.com/david-cortes/cmfrec.git
cd cmfrec
mkdir build
cd build
cmake ..
make

## For a system-wide install
sudo make install
sudo ldconfig
```

Linkage is then done with `-lcmfrec`.


**Note:** this package relies heavily on BLAS and LAPACK functions for calculations. It's recommended to use MKL (in Python, comes by default in Anaconda, in R for Windows, can be gotten through Microsoft's R distribution) or OpenBLAS as backend for them, but note that, as of OpenBLAS 0.3.9, some of the functions used here might be significantly faster in MKL depending on CPU architecture.


## Sample Usage

* Python

```python
import numpy as np, pandas as pd
from cmfrec import CMF

### Generate random data
n_users = 4
n_items = 5
n_ratings = 10
n_user_attr = 4
n_item_attr = 5
k = 3
np.random.seed(1)

### 'X' matrix (can also pass it as SciPy COO)
ratings = pd.DataFrame({
    "UserId" : np.random.randint(n_users, size=n_ratings),
    "ItemId" : np.random.randint(n_items, size=n_ratings),
    "Rating" : np.random.normal(loc=3., size=n_ratings)
})
### 'U' matrix (can also pass it as NumPy if X is a SciPy COO)
user_info = pd.DataFrame(np.random.normal(size = (n_users, n_user_attr)))
user_info["UserId"] = np.arange(n_users)
### 'I' matrix (can also pass it as NumPy if X is a SciPy COO)
item_info = pd.DataFrame(np.random.normal(size = (n_items, n_item_attr)))
item_info["ItemId"] = np.arange(n_items)

### Fit the model
model = CMF(method="als", k=k)
model.fit(X=ratings, U=user_info, I=item_info)

### Predict rating that user 3 would give to items 2 and 4
model.predict(user=[3, 3], item=[2, 4])

### Top-5 highest predicted for user 3
model.topN(user=3, n=5)

### Top-5 highest predicted for user 3, if it were a new user
model.topN_warm(X_col=ratings.ItemId.loc[ratings.UserId == 3],
                X_val=ratings.Rating.loc[ratings.UserId == 3],
                U=user_info.iloc[[3]],
                n=5)

### Top-5 highest predicted for user 3, based on side information
model.topN_cold(U=user_info.iloc[[3]], n=5)

### Calculating the latent factors
model.factors_warm(X_col=ratings.ItemId.loc[ratings.UserId == 3],
                   X_val=ratings.Rating.loc[ratings.UserId == 3],
                   U=user_info.iloc[[3]])
```

Users and items can be reindexed internally (if passing data frames, but not when pasing sparse or dense matrices), so you can use strings or non-consecutive numbers as IDs when passing data to the object's methods.

* R:

(See `?fit_models` for a better and longer example with real data)

```r
library(cmfrec)

n_users <- 4
n_items <- 5
n_ratings <- 10
n_user_attr <- 4
n_item_attr <- 5
k <- 3
set.seed(1)

### 'X' matrix (can also pass it as TsparseMatrix, matrix.coo, etc.)
ratings <- data.frame(
    UserId = sample(n_users, n_ratings, replace=TRUE),
    ItemId = sample(n_items, n_ratings, replace=TRUE),
    Rating = rnorm(n_ratings, mean=3)
)
### 'U' matrix (can also pass it as TsparseMatrix, DF, etc.)
user_info <- matrix(rnorm(n_users*n_user_attr), nrow=n_users)
rownames(user_info) <- 1:n_users ## These denote which ID is each row
### 'I' matrix (can also pass it as TsparseMatrix, DF, etc.)
item_info <- matrix(rnorm(n_items*n_item_attr), nrow=n_items)
rownames(item_info) <- 1:n_items ## These denote which ID is each row

### Fit the model
model <- CMF(X=ratings, U=user_info, I=item_info,
             method="als", k=k)

### Predict rating that user 3 would give to items 2 and 4
predict(model, user=c(3, 3), item=c(2, 4))

### Top-5 highest predicted for user 3
topN(model, user=3, n=5)

### Top-5 highest predicted for user 3, if it were a new user
topN_new(model, n=5,
         X_col=ratings$ItemId[ratings$UserId == 3],
         X_val=ratings$Rating[ratings$UserId == 3],
         U=user_info[3,])

### Top-5 highest predicted for user 3, based on side information
topN_new(model, U=user_info[3,], n=5)

### Calculating the latent factors
factors_single(model,
               X_col=ratings$ItemId[ratings$UserId == 3],
               X_val=ratings$Rating[ratings$UserId == 3],
               U=user_info[3,])
```

Users and items can be reindexed internally (if passing data frames, but not when pasing sparse or dense matrices), so you can use strings or non-consecutive numbers as IDs when passing data to the object's methods.

* Ruby:

See [external repository](https://github.com/ankane/cmfrec).

* C:

See file [example/c_example.c](https://github.com/david-cortes/cmfrec/blob/master/example/c_example.c)

For more details see the online documentation.

## Getting started

For a longer example with real data see the notebook [MovieLens Recommender with Side Information](http://nbviewer.jupyter.org/github/david-cortes/cmfrec/blob/master/example/cmfrec_movielens_sideinfo.ipynb).

## Documentation

* Python:

Documentation is available at ReadTheDocs: [http://cmfrec.readthedocs.io/en/latest/](http://cmfrec.readthedocs.io/en/latest/).

* R:

Documentation is available inside the package (e.g. `?CMF`) and at [CRAN](https://cran.r-project.org/web/packages/cmfrec/index.html).

* Ruby:

See [external repository](https://github.com/ankane/cmfrec) for syntax and [Python docs](http://cmfrec.readthedocs.io/en/latest/) for details about each parameter.

* C:

Documentation is available in the public header that is generated through the build script - see [include/cmfrec.h.in](https://github.com/david-cortes/cmfrec/blob/master/include/cmfrec.h.in).

## Some comments

This kind of model requires a lot more hyperparameter tuning that regular low-rank matrix factorization, and fitting a model with badly-tuned parameters might result in worse recommendations compared to discarding the side information.

If your dataset is larger than the MovieLens ratings, adding product side information is unlikely to add more predictive power, but good user side information might still be valuable.

## Troubleshooting

For any installation problems or errors encountered with this software, please open an issue in this GitHub page with a reproducible example, the error message that you see, and description of your setup (e.g. Python version, NumPy version, operating system).

## References

* [1] Cortes, David. "Cold-start recommendations in Collective Matrix Factorization." arXiv preprint arXiv:1809.00366 (2018).
* [2] Singh, Ajit P., and Geoffrey J. Gordon. "Relational learning via collective matrix factorization." Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2008.
* [3] Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative filtering for implicit feedback datasets." 2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008.
* [4] Takács, Gábor, István Pilászy, and Domonkos Tikk. "Applications of the conjugate gradient method for implicit feedback collaborative filtering." Proceedings of the fifth ACM conference on Recommender systems. 2011.
* [5] Rendle, Steffen, Li Zhang, and Yehuda Koren. "On the difficulty of evaluating baselines: A study on recommender systems." arXiv preprint arXiv:1905.01395 (2019).
* [6] Franc, Vojtěch, Václav Hlaváč, and Mirko Navara. "Sequential coordinate-wise algorithm for the non-negative least squares problem." International Conference on Computer Analysis of Images and Patterns. Springer, Berlin, Heidelberg, 2005.
* [7] Zhou, Yunhong, et al. "Large-scale parallel collaborative filtering for the netflix prize." International conference on algorithmic applications in management. Springer, Berlin, Heidelberg, 2008.
