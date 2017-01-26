import os
from setuptools import setup

parentdir = os.path.abspath(os.path.dirname(__file__))
initpy = os.path.join(parentdir, 'psmlearn', '__init__.py')
assert os.path.exists(initpy), "couldn't construct init.py, not found: %s" % initpy
with open(initpy, mode='r') as f:
    version=f.read().split('__version__=')[1].split()[0].strip()

setup(name='psmlearn',
      version=version,
      description='machine learning support for LCLS/SLAC with psana',
      url='http://github.com/slaclab/psmlearn',
      author='David Schneider',
      author_email='davidsch@slac.stanford.edu',
      license='Stanford',
      packages=['psmlearn'],
      install_requires=['h5batchreader>=0.1.0',
                        'tensorflow',
                        'numpy',
                        'h5py',
                        'yaml',
                        'matplotlib',
                        'scipy',
                        'jinja2',
                        'mpi4py',
                        'sklearn'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
