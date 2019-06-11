Using the NFFT
==============

In this tutorial, we assume that you are already familiar with the `non-uniform discrete Fourier transform <http://en.wikipedia.org/wiki/Non-uniform_discrete_Fourier_transform>`_ and the basics of the `NFFT library <http://www-user.tu-chemnitz.de/~potts/nfft/>`_ used for fast computation of NDFTs. 

Like the `FFTW library <http://www.fftw.org/>`_, the NFFT library relies on a specific data structure, called a plan, which stores all the data required for efficient computation and re-use of the NDFT.
Each plan is tailored for a specific transform, depending on the geometry, level of precomputation and design parameters.
The `NFFT manual <http://www-user.tu-chemnitz.de/~potts/nfft/guide3/html/index.html>`_ contains a comprehensive explanation on the NFFT implementation.

The pyNFFT package provides a set of Pythonic wrappers around some of the main data structures and routines of the NFFT library.
Currently, only the core NFFT component has been wrapped.
(The NFFT C library supports several other transform types, e.g., spherical transform (NFSFT) or transform on the rotation group (NFSOFT).)

This tutorial is split into three main sections.
In the first one, the general workflow for using the core of the :class:`pynfft.NFFT` class will be explained.
Then, the :class:`pynfft.NFFT` class API will be shown in detail and illustrated with examples for the univariate and multivariate cases.
Finally, an example for least-squares approximate inversion using `SciPy's sparse solvers <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_ solvers will be shown.

.. _workflow:
 
Workflow
--------

For users already familiar with the NFFT C library, the workflow should look familiar.
It consists in the following three steps:

    #. instantiation

    #. precomputation

    #. execution

In step 1, information such as the geometry of the transform or the desired level of precomputation is provided to the constructor, which takes care of allocating the internal arrays of the plan.

Precomputation (step 2) can be started once the locations of the non-uniform nodes have been provided to the plan.
Depending on the size of the transform and level of precomputation, this step may take some time.

Finally (step 3), the forward or adjoint NFFT is computed by first setting the input data in either ``f_hat`` (forward) or ``f`` (adjoint), calling the corresponding transform method, and reading the output from ``f`` (forward) or ``f_hat`` (adjoint).

.. _using_nfft:

Using the NFFT
--------------

The core of this library is encapsulated in the :class:`pynfft.NFFT` class.

**Instantiation**

The bare minimum to instantiate a new :class:`pynfft.NFFT` plan is to specify the geometry to the transform, i.e. the shape `N` of the coefficient array `f_hat`, and the number `M` of non-uniform nodes:

    >>> from pynfft.nfft import NFFT
    >>> plan = NFFT([16, 16], 92)
    >>> plan.M
    96
    >>> plan.N
    (16, 16)

More control over floating point precision, storage and speed of the NFFT can be gained by overriding the default design parameters `m`, `n`, `prec` (not present in C NFFT) and `flags`.
For more information, please consult the `NFFT manual <http://www-user.tu-chemnitz.de/~potts/nfft/guide3/html/index.html>`_ or the :class:`pyfftw.NFFT` class documentation.

**Precomputation**

Precomputation *must* be performed before calling any of the transforms, otherwise an error is raised.
The non-uniform spatial nodes :attr:`pynfft.nfft.NFFT.x` of the :class:`pynfft.NFFT` instance should be set before calling the :meth:`pynfft.nfft.NFFT.precompute` method for precomputation:

    >>> plan.x.shape  # (M, ndim)
    (92, 2)
    >>> x = ...  # compute some spatial nodes
    >>> plan.x[:] = x
    >>> plan.precompute()  

**Execution**

The actual forward and adjoint NFFT are performed by first filling the :attr:`pynfft.nfft.NFFT.f_hat` (forward) or :attr:`pynfft.nfft.NFFT.f` (adjoint) with values, and then calling the desired method :meth:`pynfft.nfft.NFFT.trafo` or :meth:`pynfft.nfft.NFFT.adjoint`:

    >>> # Forward transform
    >>> f_hat = ...  # get coefficients
    >>> plan.f_hat[:] = f_hat
    >>> plan.trafo()
    >>> f = plan.f.copy()
    >>> # Adjoint transform
    >>> f = ...  # get values at non-uniform nodes
    >>> plan.f[:] = f
    >>> plan.adjoint()
    >>> f_hat = plan.f_hat.copy()

.. note::
   It is important to realize that all publicliy visible arrays owned by the plan (:attr:`pynfft.nfft.NFFT.x`, :attr:`pynfft.nfft.NFFT.f_hat` and :attr:`pynfft.nfft.NFFT.f`) are direct references to the plan-internal arrays and should normally be copied before being using further in subsequent mutating computations.
   Failing to do so can result in surprising behavior, e.g.,

       >>> f = plan.f  # reference to `plan.f`, no copy
       >>> plan.f_hat[:] = ...  # new values
       >>> plan.trafo()  # `f` has now been mutated!

.. _using_scipy_solvers:

Using SciPy Least-squares solvers
---------------------------------

Non-uniform FFTs are generally not invertible.
Only a least-squares solution to the inversion problem can always be computed, using iterative solvers.
Instead of wrapping the NFFT solver, ``pynfft`` chooses the more flexible and extensible approach of wrapping a plan as a SciPy :class:`scipy.sparse.linalg.interface.LinearOperator`.
A :class:`scipy.sparse.linalg.interface.LinearOperator` is an interface class that can be used as a drop-in replacement for a dense matrix in `many iterative solvers <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html#solving-linear-problems>`_, due to the fact that one typically only needs the forward and adjoint *actions* of a matrix on an input vector.

The :mod:`pyfftw.linop` module provides a helper function that constructs such a :class:`scipy.sparse.linalg.interface.LinearOperator` from an existing (and precomputed!) :class:`pynfft.NFFT` plan.
It can be used in a least-squares solver, e.g., `LSQR <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html#scipy.sparse.linalg.lsqr>`_, as follows:

    >>> plan = ...
    >>> plan.precompute()
    >>> A = pynfft.linop.as_linop(plan)
    >>> b = ...  # result of an earlier forward transform
    >>> res = scipy.sparse.linalg.lsqr(A, b)
    >>> f_hat_ls = res[0]  # least-squares solution

Usually it's a good idea to also provide a nonzero ``damp`` parameter to ``lsqr`` for regularization.
