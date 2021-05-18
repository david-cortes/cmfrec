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

The time measurements are done by fitting the model to the full data, while the RMSE comparisons use the suggested train-test split that is provided within the MovieLens10M dataset.

#### Results

| Library            | Lang  | Method   | Biases | Time (s) | RMSE         | Additional |
| :---:              | :---: | :---:    | :---:  | :---:    | :---:        | :---:
| cmfrec             | Py    | ALS-CG   | Yes    | 13.71    | 0.788233     |
| cmfrec             | Py    | ALS-CG   | No     | 11.83    | 0.791481     |
| cmfrec             | Py    | ALS-Chol | Yes    | 31.24    | 0.786923     |
| cmfrec             | Py    | ALS-CG   | Yes    | 23.04    | 0.785427     | Implicit features
| cmfrec             | Py    | ALS-Chol | Yes    | 38.13    | **0.782414** | Implicit features
| spark              | Py    | ALS-Chol | No     | 81       | 0.791316     | Manual center
| cornac             | Py    | SGD      | Yes    | 13.9     | 0.816548     |
| spotlight          | Py    | ADAM     | No     | 12141    | 1.054698     | See details
| Surprise           | Py    | SGD      | Yes    | 178      | 1.060049     |
| Surprise           | Py    | SGD      | Yes    | timeout  | timeout      | Implicit features
| LensKit            | Py    | ALS-CD   | Static | 26.8     | 0.796050     | Manual thread control
| LensKit            | Py    | ALS-Chol | Static | 37.6     | 0.796044     | Manual thread control
| LensKit            | Py    | SVD      | Static | 8.88     | 0.838194     |
| PyRecLab           | Py    | SGD      | Yes    | 90       | 0.812566     | Reads from disk
| cmfrec             | R     | ALS-CG   | Yes    | 12.71    | 0.788356     |
| cmfrec             | R     | ALS-CG   | No     | 10.95    | 0.791409     |
| cmfrec             | R     | ALS-Chol | Yes    | 53.93    | 0.786971     |
| cmfrec             | R     | ALS-CG   | Yes    | 21.17    | 0.785484     | Implicit features
| cmfrec             | R     | ALS-Chol | Yes    | 58.57    | 0.782465     | Implicit features
| rsparse            | R     | ALS-CG   | Yes    | 31.10    | 0.788547     |
| rsparse            | R     | ALS-CG   | No     | 21.62    | 0.791412     |
| rsparse            | R     | ALS-Chol | Yes    | 30.13    | 0.786935     |
| recosystem (libmf) | R     | SGD      | No     | **1.80** | 0.785478     | Single precision
| softImpute         | R     | ALS-Chol | Static | 88.93    | 0.810450     | Unscaled lambda
| softImpute         | R     | ALS-SVD  | Static | 195.73   | 0.808293     | Unscaled lambda
| Vowpal Wabbit      | CLI   | SGD      | Yes    | 293      | 1.054546     | See details


Unsuccessful attempts:

* The Bayesian PMF model from `cornac`  produced results too far from optimal due to the hyperparameters.
* I did not manage to get the `QRec` library running.

Clarifications:

* Getting `cmfrec` to run at optimal speeds with OpenBLAS in Python required some workarounds (see https://github.com/xianyi/OpenBLAS/issues/3237), including setting the number of openblas threads to 1 and hard-coding a replacement for one of the BLAS functions (included as of v3.0.3). The R version somehow managed to run fast without any of that. Getting it running fast with MKL did not require any additional tuning.
* `cmfrec` offers the option of co-factorizing a binarized version of the interactions matrix, sharing the same latent components as with the regular factorization - these are marked as "Implicit features".
* `spark` does not perform any mean centering, so this had to be done manually in a separate step beforehand. Without centering, the RMSE would be much worse.
* Some libraries will only pre-estimate the biases instead of updating them during each iteration. These are marked as "Static".
* `libmf` (and by extension `recosystem`) only supports single-precision mode (a.k.a. `float32`).
* `libmf` does not link to BLAS or LAPACK, providing instead manual SIMD code. The version tested here was compiled to use AVX instructions (CPU supports AVX2 but `libmf` doesn't). It uses a different optimization procedure than the others so there's less need for such functions.
* The libraries all use their default compilation arguments, save for `recosystem` - this means e.g. that R libraries might not make use of all available SIMD instructions for the CPU, for example.
* `lenskit` was running into issues with nested parallelism, and thus the number of BLAS and LAPACK threads had to be manually limited from outside using `threadpoolctl` (see [this issue](https://github.com/lenskit/lkpy/issues/257)). Without this manual step, the running times are much worse.
* `spotlight` does not perform any mean centering, so it was done manually just like for spark. It uses single-precison only, and calculations are done through PyTorch, which at the time of writing did not work with OpenBLAS and was instead using MKL as linear algebra backend.
* Vowpal Wabbit does not work with in-memory data, but rather reads it from a file as it goes and stores the matrices it is estimating in files, thus having an additional IO time barrier. It is not multi-threaded in the computations that it makes, and does the computations in single precision (a.k.a. `float32`). It will additionally take hashes of the IDs which can mix up some users and items, thus not being exactly the same model type as the others. At the time of writing this, it offered two modes for matrix factorization (`--rank` and ``--new_mf``) - the one used here was `--rank`, with the learning rate copied from `libmf` (= 0.1), and while performing better than with the defaults, this might not be optimal. The ``--new_mf`` mode did not finish after running for half an hour.
* The spark runtime did not manage to find the system-installed BLAS and LAPACK libraries (this was on Debian testing, with packages installed from the `main` repository, but using the `openmp` variant of `libopenblas`), so it ended up using an unoptimized version from NetLib which is slower.

## Implicit-feedback models

Benchmark with the LastFM-360k dataset to come in the future.
