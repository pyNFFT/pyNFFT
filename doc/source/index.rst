.. pyNFFT documentation master file, created by
   sphinx-quickstart on Thu May 30 19:37:18 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyNFFT's documentation!
==================================

Introduction
------------

**pyNFFT** is a set of Pythonic wrapper classes around the `NFFT library 
<http://www-user.tu-chemnitz.de/~potts/nfft/>`_. The aim is to provide access 
to the core functionalities of the library using a more straightforward 
instantiation through Python classes, while keeping similar naming 
conventions to the C-library structures and routines.

Right now, only the NFFT and solver components of the library are supported.
Support for other components `may` come in the future, but is conditionned 
by the author's need for them. If you're interested in getting these components
wrapped as well, please feel free to contribute.

The design of the pyNFFT package assumes that the NFFT has been **compiled with
OpenMP support**.

The core interface of the NFFT is provided by the unified class, 
:class:`pynfft.nfft.NFFT`. The solver interface is in 
:class:`pynfft.solver.Solver`.

A comprehensive unittest suite is included with the source on the repository.
The suite will be updated as more functionalities get introducted.

Content
-------

.. toctree::
   :maxdepth: 2

   tutorial.rst
   api.rst

About this documentation
------------------------

This documentation is generated using the `Sphinx
<http://sphinx.pocoo.org/>`_ documentation generator.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

