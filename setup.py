from setuptools import setup

setup(name='psana-mlearn',
      version='0.0.0',
      description='machine learning support for LCLS/SLAC with psana',
      url='http://github.com/davidslac/psana-mlearn',
      author='David Schneider',
      author_email='davidsch@slac.stanford.edu',
      license='Stanford',
      packages=['psmlearn'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
