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
 - 'warm' -> predictions based on new, unseen 'X' data, and potentially including new 'U' data along.
 - 'cold' -> predictions based on new user attributes data 'U', without 'X'.
 - 'new' -> predictions about new items based on attributes data 'I'.

And the following conventions in docs:
 - 'existing' -> the user/item was present in the training data that was passed to 'fit'.
 - 'new' -> the user/items was not present in the training data that was passed to 'fit'.

 Also be aware that many of the model attributes in fitted-model objects, such as the A and B matrices,
 have a name ending in underscore (e.g. *CMF_explicit.A_*), but the docs you are reading here might
 render them without that underscore in the generated HTML. See the docstrings in the source code
 to be sure.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   index


Models
======

CMF_explicit
------------

.. autoclass:: cmfrec.CMF_explicit
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
