from distutils.core import setup
setup(
  name = 'cmfrec',
  packages = ['cmfrec'],
  install_requires=[
   'pandas',
   'numpy',
   'scipy',
   'dask>=0.16.0',
   'tensorflow>=1.0.0'
],
  version = '0.4.2',
  description = 'Collaborative filtering with user and item side information based on collective matrix factorization',
  author = 'David Cortes',
  author_email = 'david.cortes.rivera@gmail.com',
  url = 'https://github.com/david-cortes/cmfrec',
  download_url = 'https://github.com/david-cortes/cmfrec/archive/0.4.2.tar.gz',
  keywords = ['collaborative filtering', 'collective matrix factorization', 'relational learning'],
  classifiers = [],
)