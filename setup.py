from distutils.core import setup
setup(
  name = 'cmfrec',
  packages = ['cmfrec'],
  install_requires=[
   'pandas>=0.24',
   'numpy',
   'scipy',
   'tensorflow>=1.0.0,<2.0.0'
],
  version = '0.5.3',
  description = 'Collaborative filtering with user and item side information based on collective matrix factorization',
  author = 'David Cortes',
  author_email = 'david.cortes.rivera@gmail.com',
  url = 'https://github.com/david-cortes/cmfrec',
  keywords = ['collaborative filtering', 'collective matrix factorization', 'relational learning'],
  classifiers = [],
)