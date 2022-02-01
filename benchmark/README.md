# Matrix Factorization benchmarks

This is a speed and quality benchmark comparing the `cmfrec` library against other libraries for matrix factorization. The computations were ran on a Ryzen 2 2700 CPU (3.2GHz, 8c/16t), linking against OpenBLAS 0.3.15 (with some exceptions linking against MKL when not possible to use OpenBLAS).

While `cmfrec` focuses on models with side information, the benchmarks here are about collaborative filtering models based on user/item interactions alone, using medium-sized public datasets, as it makes it easier to compare to other libraries.

As can be seen from the benchmarks, while not exactly its intended purpose, `cmfrec` also performs very well at matrix factorization without side information, and is flexible enough to allow implementing many variations of the classical model with ideas proposed in different papers.

## Explicit-feedback models

The comparisons here use the [MovieLens10M](https://grouplens.org/datasets/movielens/10m/) dataset, fitting the classical matrix factorization model described in _Large-scale parallel collaborative filtering for the netflix prize_ (see main package page for the full references), which:

* Minimizes squared error with respect to the _known_ entries in the data.
* Adds a regularization penalty on the L2 norm of the factorizing matrices which scales (increases) along with the number of present entries for each user and item.
* Performs mean centering and adds user and item biases, but does not scale the variance of the rows and columns of the matrix to factorize.

Models are fit using the same hyperparameters for all libraries: 50 latent factors, regularization of 0.05, 15 ALS iterations, computing in double precision (a.k.a. `float64`) when possible; with some exceptions when necessary (not all libraries allow scaled regularization or biases, and some libraries compared against might not use ALS).

The time measurements are done by fitting the model to the full data, while the RMSE comparisons use a random train-test split with all the users and items in the test set being also in the train set.

#### Results

| Library            | Lang  | Method   | Biases | Time (s) | RMSE         | Additional |
| :---:              | :---: | :---:    | :---:  | :---:    | :---:        | :---:
| cmfrec             | Py    | ALS-CG   | Yes    | 13.64    | 0.788233     |
| cmfrec             | Py    | ALS-CG   | No     | 12.57    | 0.791481     |
| cmfrec             | Py    | ALS-Chol | Yes    | 30.91    | 0.786923     |
| cmfrec             | Py    | ALS-CG   | Yes    | 22.09    | 0.785427     | Implicit features
| cmfrec             | Py    | ALS-Chol | Yes    | 35.35    | **0.782414** | Implicit features
| spark              | Py    | ALS-Chol | No     | 81       | 0.791316     | Manual center
| cornac             | Py    | SGD      | Yes    | 13.9     | 0.816548     |
| spotlight          | Py    | ADAM     | No     | 12141    | 1.054698     | See details
| Surprise           | Py    | SGD      | Yes    | 178      | 1.060049     |
| Surprise           | Py    | SGD      | Yes    | timeout  | timeout      | Implicit features
| LensKit            | Py    | ALS-CD   | Static | 26.8     | 0.796050     | Manual thread control
| LensKit            | Py    | ALS-Chol | Static | 37.6     | 0.796044     | Manual thread control
| LensKit            | Py    | SVD      | Static | 8.88     | 0.838194     |
| PyRecLab           | Py    | SGD      | Yes    | 90       | 0.812566     | Reads from disk
| cmfrec             | R     | ALS-CG   | Yes    | 12.85    | 0.788356     |
| cmfrec             | R     | ALS-CG   | No     | 11.61    | 0.791409     |
| cmfrec             | R     | ALS-Chol | Yes    | 29.78    | 0.786971     |
| cmfrec             | R     | ALS-CG   | Yes    | 21.23    | 0.785484     | Implicit features
| cmfrec             | R     | ALS-Chol | Yes    | 34.67    | 0.782465     | Implicit features
| rsparse            | R     | ALS-CG   | Yes    | 30.26    | 0.788547     |
| rsparse            | R     | ALS-CG   | No     | 21.54    | 0.791412     |
| rsparse            | R     | ALS-Chol | Yes    | 30.02    | 0.786935     |
| recosystem (libmf) | R     | SGD      | No     | **1.79** | 0.785585     | Single precision
| softImpute         | R     | ALS-Chol | Static | 88.96    | 0.810450     | Unscaled lambda
| softImpute         | R     | ALS-SVD  | Static | 195.34   | 0.808293     | Unscaled lambda
| Vowpal Wabbit      | CLI   | SGD      | Yes    | 293      | 1.054546     | See details


Unsuccessful attempts:

* The Bayesian PMF model from `cornac`  produced results too far from optimal due to the hyperparameters.
* I did not manage to get the `QRec` library running.

Clarifications:

* Getting `cmfrec` to run at optimal speeds with OpenBLAS in Python required some workarounds (see https://github.com/xianyi/OpenBLAS/issues/3237), ~including setting the number of openblas threads to 1 and~ (now done automatically) hard-coding a replacement for one of the BLAS functions (included as of v3.0.3). The R version somehow managed to run fast without any of that. Getting it running fast with MKL did not require any additional tuning.
* `cmfrec` offers the option of co-factorizing a binarized version of the interactions matrix, sharing the same latent components as with the regular factorization - these are marked as "Implicit features".
* `spark` does not perform any mean centering, so this had to be done manually in a separate step beforehand. Without centering, the RMSE would be much worse.
* Some libraries will only pre-estimate the biases instead of updating them during each iteration. These are marked as "Static".
* `libmf` (and by extension `recosystem`) only supports single-precision mode (a.k.a. `float32`).
* `libmf` does not link to BLAS or LAPACK, providing instead manual SIMD code. The version tested here was compiled to use AVX instructions (CPU supports AVX2 but `libmf` doesn't). It uses a different optimization procedure than the others so there's less need for such functions.
* The R libraries all had an overwritten `Makeconf` file, overwriting the default compilation arguments to use `-O3` and `-march=native` in the packages tested here (default is `-O2` and `-msse2`). `cmfrec` would otherwise run a bit slower in R. These compilation arguments did not make any difference for the other libraries.
* `lenskit` was running into issues with nested parallelism, and thus the number of BLAS and LAPACK threads had to be manually limited from outside using `threadpoolctl` (see [this issue](https://github.com/lenskit/lkpy/issues/257)). Without this manual step, the running times are much worse.
* `spotlight` does not perform any mean centering, so it was done manually just like for spark. It uses single-precison only, and calculations are done through PyTorch, which at the time of writing did not work with OpenBLAS and was instead using MKL as linear algebra backend.
* Vowpal Wabbit does not work with in-memory data, but rather reads it from a file as it goes and stores the matrices it is estimating in files, thus having an additional IO time barrier. It is not multi-threaded in the computations that it makes, and does the computations in single precision (a.k.a. `float32`). It will additionally take hashes of the IDs which can mix up some users and items, thus not being exactly the same model type as the others. At the time of writing this, it offered two modes for matrix factorization (`--rank` and ``--new_mf``) - the one used here was `--rank`, with the learning rate copied from `libmf` (= 0.1), and while performing better than with the defaults, this might not be optimal. The ``--new_mf`` mode did not finish after running for half an hour.
* The spark runtime did not manage to find the system-installed BLAS and LAPACK libraries (this was on Debian testing, with packages installed from the `main` repository, but using the `openmp` variant of `libopenblas`), so it ended up using an unoptimized version from NetLib which is slower.

## Implicit-feedback models

The comparisons here use the [LastFM-360K dataset](http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html), and fitting the _implicit_ variant of the classical matrix factorization model (also known as "iALS" or "WRMF") as proposed in _"Collaborative filtering for implicit feedback datasets"_, which:

* Sets all the entries to a value of either zero (if a user/item combination was not observed in the data) or one (if a user-item combination was observed in the data), with a weight given by the values for each user-item combination (in this case, those values are the counts of each time a user played a given song).
* Minimizes squared error with respect to **all** the entries in the user/items matrix.
* Does not add centering or biases to the model.

In the original paper, the model proposed setting weights according to a function such as `W = 1 + alpha*X`, and recommended a value of `alpha=40`, but in practice, values less than zero tend to give better results, and many libraries instead use a simpler formula `W = X`.

Models are fit using the same hyperparameters for all libraries: 50 latent factors, regularization of 5, an alpha multiplier of 1 when it is added as an option, 15 ALS iterations, computing in double precision (a.k.a. `float64`) when possible; with some exceptions when necessary.

The time measurements are done by fitting the model to the full data, while the metrics are computed by leaving aside a test sample of 10,000 random users for which 30% of their item interactions were left for testing and 70% for training. The test users were included in the training data (but without the test items withheld for them), which is not the most appropriate way of evaluating models, but many of the libraries compared do not offer the option to compute factors for new users so it was not possible to do the comparison otherwise.

#### Results

| Library  | Lang  | Method   | Weight  | Time (s) | P@10        |   MAP        | Additional |
| :---:    | :---: | :---:    | :---:   | :---:    | :---:       |  :---:       | :---:
| cmfrec   | Py    | ALS-CG   | W=1+a*X | 31.9     | 0.16969     | 0.12135      |
| cmfrec   | Py    | ALS-Chol | W=1+a*X | 53.1     | 0.1701      | 0.121761     |
| implicit | Py    | ALS-CG   | W=X     | **29.0** | 0.17007     | 0.120986     |
| implicit | Py    | ALS-Chol | W=X     | 98       | 0.17031     | 0.121167     |
| LensKit  | Py    | ALS-CG   | W=1+a*X | 68       | **0.17069** | 0.121846     |
| LensKit  | Py    | ALS-Chol | W=1+a*X | 84       | 0.16941     | **0.122121** |
| cornac   | Py    | ADAM     | W=X     | 13338    | 0.00889     | 0.006288     | float32
| Spark    | Py    | ALS-Chol | W=1+a*X | oom      | oom         | oom          | See details
| cmfrec   | R     | ALS-CG   | W=1+a*X | 29.52    | 0.16969     | 0.12135      |
| cmfrec   | R     | ALS-Chol | W=1+a*X | 51.28    | 0.1701      | 0.121761     |
| rsparse  | R     | ALS-CG   | W=X     | 39.18    | 0.16998     | 0.121242     |
| rsparse  | R     | ALS-Chol | W=X     | 69.75    | 0.16941     | 0.121353     |
| LibMF    | R     | ALS-CD   | W=X     | 143.67   | 0.14307     | 0.093755     | float32
| qmf      | CLI   | ALS-Chol | W=1+a*X | 102      | 0.17019     | 0.122017     |

Clarifications:
* PySpark somehow started dumping dozens of gigabytes to disk until running out of space, despite using more than 4x more RAM than all the other libraries. This was despite:
    * Using `intermediateStorageLevel='MEMORY_ONLY'` and `finalStorageLevel='MEMORY_ONLY'` (both were ignored by the software).
    * Setting `spark.driver.memory` to the maximum available.
    * Setting the java options to pre-allocate more memory than was needed and setting its heap memory limit to the maximum in the system.
    * Persisting the data in spark before using it.
    * Enabling arrow execution in spark (and without it, it would not use more than 1 thread).
    * Playing with different block sizes and other configurations.

As such, it was not possible to compare against the matrix factorization implementation from spark. Nevertheless, from the benchmarks done by others, it may be extrapolated that it should be around 10x slower than `implicit` in a setting like this, which would put it at around 300 seconds.
