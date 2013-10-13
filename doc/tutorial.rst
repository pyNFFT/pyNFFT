Using the NFFT
==============

In this tutorial, we assume that you are already familiar with the 
`non-uniform discrete Fourier transform 
<http://en.wikipedia.org/wiki/Non-uniform_discrete_Fourier_transform>` and the 
`NFFT library <http://www-user.tu-chemnitz.de/~potts/nfft/>` used for fast 
computation of NDFTs. 

Like the `FFTW library <http://www.fftw.org/>`, the NFFT library relies on a 
specific data structure, called a plan, which stores all the data required for 
efficient computation and re-use of the NDFT. Each plan is tailored for a 
specific transform, depending on the geometry, level of precomputation and 
design parameters. The `NFFT manual 
<http://www-user.tu-chemnitz.de/~potts/nfft/guide3/html/index.html>` contains 
exhaustive explanation on the NFFT implementation.

The pyNFFT package provides a set of Pythonic wrappers around the main data 
structures of the NFFT library. Use of Python wrappers allows to simplify the 
manipulation of the library, whilst benefiting from the significant speedup 
provided by its C-implementation. Although the NFFT library supports many more 
applications, only the NFFT and iterative solver components have been wrapped 
so far. 

This tutorial is split into three main sections. In the first one, the 
general :ref:`workflow <workflow>` for using the core of the 
:class:`pynfft.NFFT` class will be explained. Then, the :class:`pynfft.NFFT` 
class API will be detailed and illustrated with examples for the univariate 
and multivariate cases. Finally, the :class:`pynfft.Solver` iterative solver 
class will be briefly presented. 

Important
---------

Starting from pynfft v1.1, the NFFT API has been changed significantly to be 
more pythonic and loosen up the workflow. Compatibility with the old v1.0 API 
could not be ensured but the updated tutorial should give you enough 
indications to update your code.

.. _workflow:
 
Workflow
--------

For users already familiar with the NFFT C-library, the workflow is basically 
the same. It consists in the following three steps:

    #. instantiation

    #. precomputation

    #. execution

In step 1, information such as the geometry of the transform or the desired 
level of precomputation is provided to the constructor, which takes care of 
allocating the internal arrays of the plan.

Precomputation (step 2) can be started once the location of the non-uniform 
nodes have been set to the plan. Depending on the size of the transform and 
level of precomputation, this step may take some time.

Finally (step 3), the forward or adjoint NFFT is computed by first setting 
the input data in either `f_hat` (forward) or `f` (adjoint), calling the 
corresponding function, and reading the output in `f` (forward) or `f_hat` 
(adjoint).

.. _using_nfft

Using the NFFT
--------------

The core of this library is provided through the :class:`pyfftw.NFFT class`. 
All nfft components provided by the NFFT library are fully encapsulated within 
this class.

In its simplest form, a pyfftw.NFFT object is created with a pair of 
complementary numpy arrays: the `f` and `f_hat` arrays, which can both act as 
input or output depending on the chosen transform, forward or adjoint.

These arrays must be C-contiguous and of numpy.complex128 type. Support for 
other floating precision could be coming later, when the NFFT library starts 
supporting multiple floating precision builds, like the FFTW does.

**instantiation**

The bare minimum to instantiate a new :class:`pynfft.NFFT` object is to 
specify the data arrays `f` and `f_hat`. The constructor will try to guess the 
geometry parameters from these arrays: `M`, the number of non-uniform nodes, 
and `N`, the shape of the uniform data.

    >>> from pynfft import NFFT
    >>> f = np.empty(96, dtype=np.complex128)
    >>> f_hat = np.empty([16, 16], dtype=np.complex128)
    >>> this_nfft = NFFT(f=f, f_hat=f_hat)

If a flatten view of the `f_hat` array is provided, the uniform grid shape 
should be provided separately with the optional `N` parameter.

    >>> this_nfft = NFFT(f=f, f_hat=f_hat.ravel(), N=N)  

The constructor still supports the old v1.0 API, where a new NFFT object can 
be instantiated with just the geometry parameters `M` and `N`:

    >>> this_nfft = NFFT(M=M, N=N)

In this case, the `f` and `f_hat` data arrays are internal to the object but 
can still be manipulated with the properties :attr:`pynfft.NFFT.f` and 
:attr:`pynfft.NFFT.f_hat`.

More control over the precision, storage and speed of the NFFT can be gained 
by overriding the default design parameters `m`, `n` and `flags`. For more 
information, please consult the `NFFT manual 
<http://www-user.tu-chemnitz.de/~potts/nfft/guide3/html/index.html>`.

**precomputation**

Precomputation *must* be performed before calling any of the transforms. It 
can be done at construct time by passing the optional parameters `precompute` 
and `x` the vector of nodes.

    >>> this_nfft = NFFT(f=f, f_hat=f_hat, x=x, precompute=True)

Otherwise, the user has to set the nodes manually using the 
:attr:`pynfft.NFFT.x` property 

    >>> this_nfft = NFFT(f=f, f_hat=f_hat)
    >>> this_nfft.x[...] = some_x[...]
    >>> this_nfft.precompute()   

**execution**

The NFFT now supports two different conventions for computing the forward and 
adjoint NFFT. The "executer" API, and is articulated around low-level routines 
which update the internal arrays the NFFT object was constructed with. For the 
forward transform:

    >>> this_nfft.f_hat[...] = some_f_hat[...]
    >>> this_nfft.execute_trafo()
    >>> # results are stored in this_nfft.f

and the adjoint transform:

    >>> this_nfft.f[:] = some_f[:]
    >>> this_nfft.execute_adjoint()
    >>> # results are stored in this_nfft.f_hat

The new "caller" API provides more pythonic helpers around the "executer" API. 
The forward transform can be simply computed with:

    >>> ret = this_nfft.forward(f_hat=some_f_hat) 

and the adjoint with:

    >>> ret = this_nfft.adjoint(f=some_f)  

The :meth:`pynfft.NFFT.forward` and :meth:`pynfft.NFFT.forward` callers both 
accept `f` and `f_hat` arguments to replace the internal data arrays with. 
These new arrays must be compatible with the geometry provided at construct 
time, otherwise an exception will be raised.

.. _using_solver

Using the iterative solver
--------------------------

**instantiation**

The instantiation of a :class:`pynfft.Solver` object requires an instance of 
:class:`pynfft.NFFT`. The following code shows you a simple example:

    >>> from pynfft import NFFT, Solver
    >>> this_nfft = NFFT(f=some_f, f_hat=some_F, x=some_x)
    >>> this_solver = Solver(this_nfft)

It is strongly recommended to use an already *precomputed* 
:class:`pynfft.NFFT` object to instantiate a :class:`pynfft.Solver` object, or 
at the very least, make sure to call its precompute method before carrying on 
with the solver.

Since the solver will typically run several iterations before converging to a 
stable solution, it is also strongly encourage to use the maximum level of 
precomputation to speed-up each call to the NFFT. Please check the paragraph 
regarding the choice of precomputation flags for the :class:`pynfft.NFFT`. 

By default, the :class:`pynfft.Solver` class uses the Conjugate Gradient of 
the first kind method (CGNR flag). This may be overriden in the constructor:

    >>> this_solver = Solver(this_nfft, flags='CGNE')

Convergence to a stable solution can be significantly speed-up using the 
right pre-conditioning weights. These can be specified by the flags 
'PRECOMPUTE_WEIGHT' and 'PRECOMPUTE_DAMP' and accessed by the 
:attr:`pynfft.Solver.w` and :attr:`pynfft.Solver.w_hat` attributes. By 
default, these weights are set to 1.

    >>> this_solver = Solver(this_nfft, flags=('PRECOMPUTE_WEIGHT'))
    >>> this_solver.w = some_w

**using the solver**

Before iterating, the solver has to be intialized. As a reminder, make sure 
the :class:`pynfft.NFFT` object used to instantiate the solver has been 
*precomputed*. Otherwise, the solver will be in an undefined state and will 
not behave properly.

Initialization of the solver is performed first setting the 
non-uniform samples :attr:`pynfft.Solver.y` and initial guess of the solution 
:attr:`pynfft.Solver.f_hat_iter` and then calling the 
:meth:`pynfft.Solver.before_loop` method.

    >>> this_solver.y = some_y
    >>> this_solver.f_hat_iter = some_f_hat_iter
    >>> this_solver.before_loop()

By default, the initial guess of the solution is set to 0, which makes the
first iteration of the solver essentially behave like a standard call to the 
adjoint NFFT.

After initialization of the solver, a single iteration can be
performed by calling the :meth:`pynfft.Solver.loop_one_step` method. With each 
iteration the current solution is written in the 
:attr:`pynfft.Solver.f_hat_iter` attribute.

    >>> this_solver.loop_one_step()
    >>> print this_solver.f_hat_iter
    >>> this_solver.loop_one_step()
    >>> print this_solver.f_hat_iter

The :class:`pynfft.Solver` class only supports one iteration at a time. It is 
at the discretion to implement the desired stopping condition, based for 
instance on a maximum iteration count or a threshold value on the residuals. 
The residuals can be read through the :attr:`pynfft.Solver.r_iter` attribute. 
Below are two simple examples:

    - with a maximum number of iterations:

    >>> niter = 10  # set number of iterations to 10
    >>> for iiter in range(niter):
    >>>	    this_solv.loop_one_step()

    - with a threshold value:

    >>> threshold = 1e-3
    >>> try:
    >>>	    while True:
    >>>		this_solver.loop_one_step()
    >>>		if(np.all(this_solver.r_iter < threshold)):
    >>>		    raise StopCondition
    >>> except StopCondition:
    >>>	    # rest of the algorithm