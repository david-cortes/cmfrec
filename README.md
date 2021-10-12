# Collective Matrix Factorization

Implementation of collective matrix factorization, based on _Relational learning via collective matrix factorization_ ([2]), with some enhancements and alternative models for cold-start recommendations as described in _Cold-start recommendations in Collective Matrix Factorization_ ([1]), and adding implicit-feedback variants as described in _Collaborative filtering for implicit feedback datasets_ ([3]).

This is a hybrid collaborative filtering model for recommender systems that takes as input either explicit item ratings or implicit-feedback data, and side information about users and/or items (although it can also fit pure collaborative-filtering and pure content-based models). The overall idea was extended here to also be able to do cold-start recommendations (for users and items that were not in the training data but which have side information available).

Although the package was developed with recommender systems in mind, it can also be used in other domains (e.g. topic modeling, dimensionality reduction, [missing value imputation](http://nbviewer.jupyter.org/github/david-cortes/cmfrec/blob/master/example/cmfrec_imputer.ipynb)) - just take any mention of users as rows in the main matrix, any mention of items as columns, and use the "explicit" models.

For more information about the implementation here, or if you would like to cite this in your research, see ["Cold-start recommendations in Collective Matrix Factorization"](https://arxiv.org/abs/1809.00366)

For a similar package with Poisson distributions see [ctpfrec](https://github.com/david-cortes/ctpfrec).

Written in C with Python and R interfaces. An additional Ruby interface can be found [here](https://github.com/ankane/cmfrec).

*********************
For an introduction to the library and methods, see:
* [MovieLens Recommender with Side Information](http://nbviewer.jupyter.org/github/david-cortes/cmfrec/blob/master/example/cmfrec_movielens_sideinfo.ipynb) (Python).
* [R vignette](http://htmlpreview.github.io/?https://github.com/david-cortes/cmfrec/blob/master/inst/doc/cmfrec_vignette.html) (R).


## Comparison against other libraries


For the full benchmark, code, and details see [benchmarks](https://github.com/david-cortes/cmfrec/tree/master/benchmark).

Comparing the classical matrix factorization model for explicit feedback **without side information** in different software libraries (50 factors, 15 iterations, regularization of 0.05, `float64` when supported) on the MovieLens10M dataset:

| Library       | Method   | Biases | Time (s) | RMSE         | Additional |
| :---:         | :---:    | :---:  | :---:    | :---:        | :---:
| cmfrec        | ALS-CG   | Yes    | 13.64    | 0.788233     | 
| cmfrec        | ALS-Chol | Yes    | 35.35    | **0.782414** | Implicit features
| LibMF         | SGD      | No     | **1.79** | 0.785585     | float32
| Spark         | ALS-Chol | No     | 81       | 0.791316     | Manual center
| cornac        | SGD      | Yes    | 13.9     | 0.816548     |
| Surprise      | SGD      | Yes    | 178      | 1.060049     |
| spotlight     | ADAM     | No     | 12141    | 1.054698     | See details
| LensKit       | ALS-CD   | Static | 26.8     | 0.796050     | Manual thread control
| PyRecLab      | SGD      | Yes    | 90       | 0.812566     | Reads from disk
| rsparse       | ALS-Chol | Yes    | 30.13    | 0.786935     |
| softImpute    | ALS-Chol | Static | 88.93    | 0.810450     | Unscaled lambda
| softImpute    | ALS-SVD  | Static | 195.73   | 0.808293     | Unscaled lambda
| Vowpal Wabbit | SGD      | Yes    | 293      | 1.054546     | See details

_Benchmark for implicit feedback models to come in the future._

## Basic Idea


(See introductory notebook above for more details)

The model consist in predicting the rating (or weighted confidence for implicit-feedback case) that a user would give to an item by performing a low-rank factorization of an interactions matrix `X` of size `users` x `items` (e.g. ratings)
```
X ~ A * B.T
```
(where `A` and `B` are the fitted model matrices)

But does so using side information about the items (such as movie tags) and/or users (such as their demographic info) by also factorizing the item side info matrix and/or the user side info matrix
```
U ~ A * C.T,   I ~ B * D.T
```
Sharing the same item/user-factor matrix used to factorize the ratings, or sharing only some of the latent factors.

This also has the side effect of allowing recommendations for users and items for which there is side information but no ratings, although these predictions might not be as high quality.

Alternatively, can produce factorizations in wich the factor matrices are determined from the attributes directly (e.g. `A = U * C`), with or without a free offset.

While the method was initially devised for recommender systems, can also be used as a general technique for dimensionality reduction by taking the `A` matrix as low-dimensional factors, which can be calculated for new data too.

Alternatively, it might also produce good results when used as an imputer for missing values in tabular data. The Python version is scikit-learn compatible and has a separate class aimed at being used for imputation in scikit-learn pipelines. [Example here](http://nbviewer.jupyter.org/github/david-cortes/cmfrec/blob/master/example/cmfrec_imputer.ipynb).

## Update 2020-03-20


The package has been rewritten in C with Python wrappers. If you've used earlier versions of this package which relied on Tensorflow for the calculations, the optimal hyperparameters will be very different now as it has changed some details of the loss function such as not dividing some terms by the number of entries.

The new version is faster, multi-threaded, and has some new functionality, but if for some reason you still need the old one, it can be found under the git branch "tensorflow".

## Highlights


* Can fit factorization models with or without user and/or item side information.
* Can fit the usual explicit-feedback model as well as the implicit-feedback model with weighted binary entries (see [3]).
* For the explicit-feedback model, can automatically add implicit features (created from the same "X" data).
* Can be used for cold-start recommendations (when using side information).
* Can be compiled for single and double precision (`float32` and `float64`) - the Python package comes with both versions.
* Supports user and item biases (these are not just pre-estimated beforehand as in other software).
* Can fit models with non-negativity constraints on the factors and/or with L1 regularization.
* Provides an API for top-N recommended lists and for calculating latent factors from new data.
* Can work with both sparse and dense matrices for each input (e.g. can also be used as a general missing-value imputer for 2D data - [example](http://nbviewer.jupyter.org/github/david-cortes/cmfrec/blob/master/example/cmfrec_imputer.ipynb)), and can work efficiently with a mix of dense and sparse inputs.
* Can produce factorizations for variations of the problem such as sparse inputs with missing-as-zero instead of missing-as-unknown (e.g. when used for dimensionality reduction).
* Can use either an alternating least-squares procedure (ALS) or a gradient-based procedure using an L-BFGS optimizer for the explicit-feedback models (the package bundles a modified version of [Okazaki's C implementation](https://github.com/chokkan/liblbfgs)).
* For the ALS option, can use either the exact Cholesky method or the faster conjugate gradient method (see [4]). Can also use coordinate descent methods (when having non-negativity constraints or L1 regularization).
* Can produce models with constant regularization or with dynamically-adjusted regularization as in [7].
* Provides a content-based model and other models aimed at better cold-start recommendations.
* Provides an intercepts-only "most-popular" model for non-personalized recommendations, which can be used as a benchmark as it uses the same hyperparameters as the other models.
* Allows variations of the original collective factorization models such as setting some factors to be used only for one factorization, setting different weights for the errors on each matrix, or setting different regularization parameters for each matrix.
* Can use sigmoid transformations for binary-distributed columns in the side info data.
* Can work with large datasets (supports arrays/matrices larger than `INT_MAX`).
* Supports observation weights (for the explicit-feedback models).


## Installation

* Python:

```
pip install cmfrec
```
or if that fails:
```
pip install --no-use-pep517 cmfrec
```

(See performance tips below)

** *

**Note for macOS users:** on macOS, the Python version of this package might compile **without** multi-threading capabilities. In order to enable multi-threading support, first install OpenMP:
```
brew install libomp
```
And then reinstall this package: `pip install --force-reinstall cmfrec`.

** *

* R:

```r
install.packages("cmfrec")
```

(See performance tips below)

* Ruby:

See [external repository](https://github.com/ankane/cmfrec).

* C:

Package can be built as a shared library (requires BLAS and LAPACK) - see the CMake build file for options:
```
git clone https://www.github.com/david-cortes/cmfrec.git
cd cmfrec
mkdir build
cd build
cmake -DUSE_MARCH_NATIVE=1 ..
cmake --build .

## For a system-wide install
sudo make install
sudo ldconfig
```

Linkage is then done with `-lcmfrec`.

By default, it compiles for types `double` and `int`, but this can be changed in the CMake script to `float` (passing option `-DUSE_FLOAT=1`) and/or `int64_t` (passing option `-DUSE_INT64=1` - but note that if not using MKL, will require manually setting linkage to a BLAS library with int64 support - see the CMakeLists.txt file). Additionally, if the LAPACK library used does not do fully IEEE754-compliant propagation of NAs (e.g. some very old versions of OpenBLAS), can pass option `-DNO_NAN_PROPAGATION=1`.

Be aware that the snippet above includes option `-DUSE_MARCH_NATIVE=1`, which will make it use the highest-available CPU instruction set (e.g. AVX2) and will produces objects that might not run on older CPUs - to build more "portable" objects, remove this option from the cmake command.

## Performance tips

This package relies heavily on BLAS and LAPACK functions for calculations. It's recommended to use with MKL, OpenBLAS, or BLIS. Additionally, if using it from R, the package will benefit from enabling optimizations which are not CRAN-compliant (see below).

Different backends for BLAS can make a large difference in speed - for example, on an AMD Ryzen 2700, MKL2021 makes models take 4x longer to fit than MKL2020, and using OpenBLAS-pthreads takes around 1.3x longer to fit models compared to OpenBLAS-openmp.

This library calls BLAS routines from parallel OpenMP blocks, which can cause issues with some BLAS backends - for example, if using MKL and compiling this package with GCC on linux, it *could* have issues with conflicting OpenMPs which could be solved by adding an environment variable `MKL_THREADING_LAYER=GNU`. For the most part, it tries to disable BLAS multi-threading in openmp blocks, but the mechanism might not work with all BLAS libraries or if swapping BLAS libraries after having compiled `cmfrec`.


Hints:

* In Python, MKL comes by default in Anaconda installs. If using a non-windows OS, OpenBLAS can alternatively be installed in an Anaconda environment through the `nomkl` package. If not using Anaconda, `pip` installs of NumPy + SciPy are likely to bundle OpenBLAS (pthreads version).
* In R for Windows, see [this link](https://github.com/david-cortes/R-openblas-in-windows) for instructions on getting OpenBLAS. Alternatively, Microsoft's R distribution comes with MKL preinstalled.
* In R for other OSes, R typically uses the default system BLAS and LAPACK. On debian and debian-based systems such as ubuntu, these can be controlled through the [alternatives system](https://wiki.debian.org/DebianScience/LinearAlgebraLibraries) - see [this StackOverflow post](https://stackoverflow.com/a/49842944/5941695) for an example of setting MKL as the default backend. By default on debian, R will link to OpenBLAS-pthreads, but it is easy to make it use OpenBLAS-openmp (for example, by installing `libopenblas-openmp-dev` before installing R, or by using the alternatives system).
* If using MKL and compiling this package with GCC (default in most linux distributions, oftentimes also in anaconda for windows), one might want to set an environment variable `MKL_THREADING_LAYER=GNU`. In Linux and macOS, this can be done by adding `export MKL_THREADING_LAYER=GNU` in `~/.bashrc` or `~/.profile`, while in Windows it can be set through the control panel.


For optimal performance in R, it's recommended to set a custom Makevars file with extra compiler optimizations, and then install the package from source. On Linux, simply create a text file `~/.R/Makevars` containing this line: `CFLAGS += -O3 -march=native` (plus an empty line at the end). Then install `cmfrec` with `install.packages("cmfrec")`.

Alternatively, one can also install this package from source but editing the `Makevars` file under `src` by uncommenting the lines that are commented out, which will trigger better compiler optimizations which are not CRAN-compliant (GCC only). For alternative ways of doing this see the "Performance tips" section in the docs. This basically amounts to adding compilation options `-std=c99 -O3 -march=native`, which are typically not the defaults in R.

In modern CPUs, this can make some optimization routines in `cmfrec` roughly 25% faster.

Earlier Python versions of `cmfrec` used the package `findblas` to link to BLAS's CBLAS interface, while newer versions take the BLAS from SciPy and build a CBLAS wrapper around it, which can make it run slightly lower. To use `findblas`, define an environment variable `USE_FINDBLAS=1` before installing:_
```
export USE_FINDBLAS=1
pip install cmfrec
```
_(Can also define `USE_OPENBLAS=1` to forcibly use `-lopenblas`)_


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

(See also [example using it for imputing missing values](http://nbviewer.jupyter.org/github/david-cortes/cmfrec/blob/master/example/cmfrec_imputer.ipynb))

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

For a longer example with real data see the Python notebook [MovieLens Recommender with Side Information](http://nbviewer.jupyter.org/github/david-cortes/cmfrec/blob/master/example/cmfrec_movielens_sideinfo.ipynb) and the [R vignette](http://htmlpreview.github.io/?https://github.com/david-cortes/cmfrec/blob/master/inst/doc/cmfrec_vignette.html).

## Documentation

* Python:

Documentation is available at ReadTheDocs: [http://cmfrec.readthedocs.io/en/latest/](http://cmfrec.readthedocs.io/en/latest/).

* R:

Documentation is available inside the package (e.g. `?CMF`) and at [CRAN](https://cran.r-project.org/web/packages/cmfrec/index.html).

* Ruby:

See [external repository](https://github.com/ankane/cmfrec) for syntax and [Python docs](http://cmfrec.readthedocs.io/en/latest/) for details about each parameter.

* C:

Documentation is available in the public header that is generated through the build script - see [include/cmfrec.h.in](https://github.com/david-cortes/cmfrec/blob/master/include/cmfrec.h.in).

## Evaluating model quality

Metrics for implicit-feedback model quality for this package can be calculated using the [recometrics](https://github.com/david-cortes/recometrics) library.

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
