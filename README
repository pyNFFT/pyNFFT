PyNFFT - Pythonic bindings around the NFFT library
==================================================

"The NFFT is a C subroutine library for computing the nonequispaced discrete
Fourier transform (NDFT) in one or more dimensions, of arbitrary input size,
and of complex data."

The `NFFT library <http://www-user.tu-chemnitz.de/~potts/nfft/index.php>`_ is
licensed under GPLv2.

This wrapper provides a somewhat Pythonic access to some of the core NFFT
library functionalities and is largely inspired from the pyFFTW project
developped by Henry Gomersall (http://hgomersall.github.io/pyFFTW/).

The documentation is hosted on `pythonhosted
<http://pythonhosted.org/pyNFFT/>`_, the source code is available on `github
<https://github.com/ghisvail/pyNFFT>`_ and the Python package index page is
`here <https://pypi.python.org/pypi/pyNFFT>`_.

Important
---------

There has been a major API change from version 1.2.x to 1.3 in prevision of 
version 2.0. Please check the documentation for more information.

The only officially supported platform for version 1.3 is Linux. Support for 
more platforms is scheduled for the 2.x development tree.

Usage
-----

See the `tutorial <http://pythonhosted.org/pyNFFT/tutorial.html>`_ 
section of the documentation.

Installation
------------

Support for pip/easy_install has been added via the `Python Package Index
<http://pypi.python.org/pypi/>`_. The pyNFFT package can be installed with::
        
    pip install pynfft

Installation will fail if the NFFT library is not installed in a system-aware
location. A workaround is to first call pip with::

    pip install --no-install pynfft

cd to where pip downloaded the package, then build with `setup.py`::

    python setup.py build_ext -I <path_to_include> -L <path_to_lib>
    -R <path_to_lib>

and do a final call to pip::

    pip install --no-download pynfft

Building
--------

The pyNFFT package can be built from the cloned git repository by calling::

    python setup.py build

and then installed with::

    python setup.py install

The build process requires Cython in order to generate the cythonized 
c-files::

    python setup.py build_ext --inplace

Build info
----------

The NFFT library has to be compiled with the --enable-openmp flag to allow for
the generation of the threaded version of the library.  Without it, any attempt
to building the project will fail.

Requirements
------------

- Python 2.7 or greater
- Numpy 1.6 or greater
- NFFT library 3.2 or greater, compiled with openMP support
- (optional) Cython 0.12 or greater

Contributing
------------

See the CONTRIBUTING file.

License
-------

The pyNFFT project is licensed under the GPLv3.  See the bundled COPYING file
for more details.
