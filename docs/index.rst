.. cmfrec documentation master file, created by
   sphinx-quickstart on Tue Jun  5 21:41:58 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Collective Matrix Factorization
===============================

This is the documentation page for the python package *cmfrec*. For 
more details, see the project's GitHub page:

`<https://www.github.com/david-cortes/cmfrec/>`_

Installation
=================================
Package is available on PyPI, can be installed with
::

    pip install cmfrec
    
Naming conventions
==================

This package uses the following general naming conventions on functions:
* 'warm' -> predictions based on new, unseen 'X' data, and potentially including new 'U' data along.
* 'cold' -> predictions based on new user attributes data 'U', without 'X'.
* 'new' -> predictions about new items based on attributes data 'I'.

And the following conventions in docs:
* 'existing' -> the user/item was present in the training data that was passed to 'fit'.
* 'new' -> the user/items was not present in the training data that was passed to 'fit'.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: cmfrec
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
