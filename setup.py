import io
from setuptools import find_packages, setup
import os

REQUIRED = [ 'numpy', 'scipy', 'pandas', 'numba', 'matplotlib']

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.rst' is present in your MANIFEST.in file!
with io.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

setup(name='PyDHAMed',
      version='0.1',
      description='Dynamic Histogram Analysis To Determine Free Energies and Rates from Biased Simulations',
      url='https://github.com/bio-phys/PyDHAMed',
      author='Lukas Stelzl, Gerhard Hummer',
      author_email='flyingcircus@example.com',
      install_requires=REQUIRED,
      license='BSD-3-Clause',
      packages=['pydhamed'],
      zip_safe=False)
