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
