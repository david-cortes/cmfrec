from distutils.core import setup
setup(
  name = 'cmfrec',
  packages = ['cmfrec'],
  install_requires=[
   'pandas>=0.18.0',
   'numpy',
   'scipy',
   'casadi==3.1.1'
],
  version = '0.1',
  description = 'Collaborartive filtering with item side information based on double matrix factorization',
  author = 'David Cortes',
  author_email = 'david.cortes.rivera@gmail.com',
  url = 'https://github.com/david-cortes/cmfrec',
  download_url = 'https://github.com/david-cortes/cmfrec/archive/0.1.tar.gz',
  keywords = ['collaborative filtering', 'collective matrix factorization'],
  classifiers = [],
)