__version__ = '0.0.0'

from setuptools import setup
from setuptools import find_packages

setup(name=               'vebm',
      version=            __version__,
      description=        'VEBM code',
      url=                'https://github.com/pawij/vebm',
      classifiers=			  ['Intended Audience :: Science/Research',
                   			  'Programming Language :: Python',
                   			  'Topic :: Scientific/Engineering',
                   			  'Programming Language :: Python :: 3.10'],
      maintainer=         'Peter Wijeratne',
      maintainer_email=   'p.wijeratne@pm.me',
      license=		  'MIT',
      packages=           find_packages('src'),
      package_dir=        {"": "src"},
      python_requires=    '>=3.10',
      install_requires=   ['numpy>=1.26,<2',
                           'scipy>=1.11',
                           'scikit-learn>=1.3',
                           'torch>=2.2',
                           'matplotlib>=3.7'],
      entry_points=	  {},
      zip_safe=		  False)
