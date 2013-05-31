pyNFFT - Cython wrapper around the NFFT library
===============================================

"The NFFT is a C subroutine library for computing the nonequispaced discrete
Fourier transform (NDFT) in one or more dimensions, of arbitrary input size,
and of complex data."

The NFFT library is licensed under GPLv2 and available at:
http://www-user.tu-chemnitz.de/~potts/nfft/index.php

This wrapper provides a somewhat Pythonic access to some of the core NFFT 
library functionalities and is largely inspired from the pyFFTW project 
developped by Henry Gomersall (http://hgomersall.github.io/pyFFTW/).

This project is still work in progress and is still considered beta quality.
In particular, the API is not yet frozen and is likely to change as the 
development continues. Please consult the documentation and changelog for 
more information.

Requirements
------------
- Python 2.7 or greater
- Numpy 1.6 or greater
- NFFT 3.2 or greater
- Cython 0.15 or greater (optional)

Installation
------------

Support for pip/easy_install via the 
`Python Package Index <http://pypi.python.org/pypi/>`_ is coming. Right now, 
the preferred way is to build and install pyNFFT manually.

Building
--------

To build the package inplace using the Cython .pyx files::

    $python cython_setup build_ext --inplace

If your NFFT library is installed in non system-aware location, you can 
specify the location of the library and include files. For instance, if you 
installed the NFFT library in $HOME/local::

    $python cython_setup build_ext --inplace --library-dirs=$HOME/local/lib
    --include-dirs=$HOME/local/include

This command will refresh the C-extension files, which can then be installed 
with a call to the regular setup.py script::

    $python setup.py build
    $python setup.py install

Build info
----------

NFFT has to be compiled with the --enable-openmp flag to allow for the
generation of the threaded version of the library. Without it, any attempt to
building the project will fail.


