# How to use

These are mathematical tests aiming at detecting possible errors in the correctness of the code. It tests it with different combinations of input types and variables against a naive Python implementation with gradients obtained through numeric differentiation and optimal values obtained through the SciPy L-BFGS-B optimizer.

In order to use them:
* Run `python test_<some_file>.py`
* Look up for some error (e.g. Ctrl+F `ERROR`)
* Check what do the lines with errors have in common
* Narrow down the criteria by altering the test file and testing only the combinations that print the error, until all or most of the lines are errors
* At that point, it should be easier to see where in the source code does the error come from.

They are **not** intended as unit tests - some might take very long to run, and do not test aspects such as allocating the correct sizes of arrays or the like. Some possible combinations of inputs are left untested. It's suggested to also use them with the address sanitizer and openmp enabled once the lines show no errors.

# TODO

* Need to add tests for cases with non-negativity constraints.
* Need to add tests for the case of missing as zero + centering + bias (should look very different from the current tests).
* Need to add tests for static biases.
* Need to add tests for dynamic regularization and L1 regularization.
