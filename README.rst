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

By default, pyNFFT expects that the NFFT library has been compiled with the 
--enable-openmp flag to allow for the generation of the threaded version of 
the library. If the threaded version of the NFFT library is not available, 
pyNFFT can be build and linked against the unthreaded version of the library
with::

    NFFT_THREADS=no pip install pynfft
    
or::

    NFFT_THREADS=no python setup.py install

pyNFFT further assumes that the underlying FFTW library is available in a 
multi-threaded version. On Unix platforms, the default is to link against
both libfftw3_threads.so and libfftw3.so, while on Windows platforms, a 
combined fftw3 library containing both multi- and single-threaded library
functions are assumed. If necessary, the default behaviour can be overwritten
with setting FFTW_THREADS_COMBINED to 'yes' or 'No' as required. 

Similarly, the use of a threaded version of the FFTW library can be entirely
disabled with FFTW_THREADS=no.

If pyNFFT is build with python setup.py, equivalent settings can also be made
in the setup.cfg file. For details, see the description in setup.cfg.template.

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
