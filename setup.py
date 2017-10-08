from distutils.core import setup
setup(
  name = 'cmfrec',
  packages = ['cmfrec'],
  install_requires=[
   'pandas',
   'numpy',
   'scipy',
   'casadi'
],
  version = '0.3',
  description = 'Collaborative filtering with user and item side information based on collective matrix factorization',
  author = 'David Cortes',
  author_email = 'david.cortes.rivera@gmail.com',
  url = 'https://github.com/david-cortes/cmfrec',
  download_url = 'https://github.com/david-cortes/cmfrec/archive/0.3.tar.gz',
  keywords = ['collaborative filtering', 'collective matrix factorization', 'relational learning'],
  classifiers = [],
)