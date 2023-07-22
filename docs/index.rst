.. cmfrec documentation master file, created by
   sphinx-quickstart on Tue Oct 27 18:05:49 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Collective Matrix Factorization
===============================

This is the documentation page for the Python package *cmfrec*. For more details, see the project's GitHub page:

`<https://www.github.com/david-cortes/cmfrec/>`_

For the R version, see the CRAN page:

`<https://cran.r-project.org/web/packages/cmfrec/index.html>`_

Installation
=================================
Package is available on PyPI, can be installed with
::

    pip install cmfrec
    

This library will run faster when compiled from source with non-default compiler arguments,
particularly ``-march=native`` (replace with ``-mcpu=native`` for ARM/PPC), which gets added automatically
when installing from ``pip``, and when using an optimized BLAS library for SciPy.

See this guide for details:

`<https://github.com/david-cortes/installing-optimized-libraries>`_

Sample Usage
============

For an introduction to the package and methods, see the example IPython notebook building recommendation models on the MovieLens10M dataset with and without side information:

`<http://nbviewer.jupyter.org/github/david-cortes/cmfrec/blob/master/example/cmfrec_movielens_sideinfo.ipynb>`_

Model evaluation
================

Metrics for implicit-feedback models or recommendation quality can be calculated using the recometrics library:

`<https://www.github.com/david-cortes/recometrics/>`_


Naming conventions
==================

This package uses the following general naming conventions:

About data:
 - 'X' -> data about interactions between users/rows and items/columns (e.g. ratings given by users to items).
 - 'U' -> data about user/row attributes (e.g. user's age).
 - 'I' -> data about item/column attributes (e.g. a movie's genre).

About function naming:
 - 'warm' -> predictions based on new, unseen 'X' data, and potentially including new 'U' data along.
 - 'cold' -> predictions based on new user attributes data 'U', without 'X'.
 - 'new' -> predictions about new items based on attributes data 'I'.

About function descriptions:
 - 'existing' -> the user/item was present in the training data that was passed to 'fit'.
 - 'new' -> the user/items was not present in the training data that was passed to 'fit'.


Be aware that the package's functions are user-centric (e.g. it will recommend items for users, but not users for items). If predictions about new items are desired, it's recommended to use the method 'swap_users_and_items', as the item-based functions which are provided for convenience might run a lot slower than their user equivalents.

Implicit and explicit feedback
==============================

In recommender systems, data might come in the form of explicit user judgements about items (e.g. movie ratings) or
in the form of logged user activity (e.g. number of times that a user listened to each song in a catalogue). The former
is typically referred to as "explicit feedback", while the latter is referred to as "implicit feedback".

Historically, driven by the Netflix competition, formulations of the recommendation problem have geared towards predicting the rating
that users would give to items under explicit-feedback datasets, determining the components in the low-rank factorization
in a way that minimizes the deviation between predicted and observed numbers on the **observed** data only (i.e. predictions
about items that a user did not rate do not play any role in the optimization problem for determining the low-rank factorization
as they are simply ignored), but this approach has turned out to oftentimes result in very low-quality recommendations,
particularly for users with few data, and is usually not suitable for implicit feedback as the data in such case does not contain
any examples of dislikes and might even come in just binary (yes/no) form.

As such, research has mostly shifted towards the implicit-feedback setting, in which items that are not consumed by users
do play a role in the optimization objective for determining the low-rank components - that is, the goal is more to predict
which items would users have consumed than to predict the exact rating that they'd give to them - and the evaluation of recommendation
quality has shifted towards looking at how items that were consumed by users would be ranked compared to unconsumed items.

Other problem domains
=====================

The documentation and naming conventions in this library are all oriented towards recommender systems, with the assumption
that users are rows in a matrix, items are columns, and values denote interactions between them, with the idea that values
under different columns are comparable (e.g. the rating scale is the same for all items).

The concept of approximate low-rank matrix factorizations is however still useful for other problem domains, such as general
dimensionality reduction for large sparse data (e.g. TF-IDF matrices) or imputation of high-dimensional tabular data, in which
assumptions like values being comparable between different columns would not hold.

Be aware that classes like ``CMF`` come with some defaults that might not be reasonable in other applications, but which
can be changed by passing non-default arguments - for example:

- Global centering - the "explicit-feedback" models here will by default calculate a global mean for all entries in 'X' and
center the matrix by substracting this value from all entries. This is a reasonable thing to do when dealing with movie ratings
as all ratings follow the same scale, but if columns of the 'X' matrix represent different things that might have different ranges
or different distributions, global mean centering is probably not going to be desirable or useful.
- User/row biases: models might also have one bias/intercept parameter per row, which in the approximation, would get added
to every column for that user/row. This is again a reasonable thing to do for movie ratings, but if the columns of 'X' contain
different types of information, it might not be a sensible thing to add.
- Regularization for item/column biases: since the models perform global mean centering beforehand, the item/column-specific
bias/intercept parameters will get a regularization penalty ("shrinkage") applied to them, which might not be desirable if
global mean centering is removed.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   index


Models
======

CMF
------------

.. autoclass:: cmfrec.CMF
    :members:
    :undoc-members:
    :inherited-members:

CMF_implicit
------------

.. autoclass:: cmfrec.CMF_implicit
    :members:
    :undoc-members:
    :inherited-members:

OMF_explicit
------------

.. autoclass:: cmfrec.OMF_explicit
    :members:
    :undoc-members:
    :inherited-members:

OMF_implicit
------------

.. autoclass:: cmfrec.OMF_implicit
    :members:
    :undoc-members:
    :inherited-members:

ContentBased
------------

.. autoclass:: cmfrec.ContentBased
    :members:
    :undoc-members:
    :inherited-members:

MostPopular
-----------

.. autoclass:: cmfrec.MostPopular
    :members:
    :undoc-members:
    :inherited-members:

CMF_imputer
-----------

.. autoclass:: cmfrec.CMF_imputer
    :members:
    :undoc-members:
    :inherited-members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
