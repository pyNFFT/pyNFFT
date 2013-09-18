# -*- coding: utf-8 -*-
#
# Copyright (C) 2013  Ghislain Vaillant
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc cimport limits
from cnfft3 cimport *

cdef extern from *:
    int Py_AtExit(void (*callback)()) 

# Initialize module
# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

# initialize FFTW threads
fftw_init_threads()

# register cleanup callbacks
cdef void _cleanup():
    fftw_cleanup()
    fftw_cleanup_threads()

Py_AtExit(_cleanup)


########
# NFFT #
########

cdef object nfft_supported_flags_tuple
nfft_supported_flags_tuple = (
    'PRE_PHI_HUT',
    'FG_PSI',
    'PRE_LIN_PSI',
    'PRE_FG_PSI',
    'PRE_PSI',
    'PRE_FULL_PSI',
    )
nfft_supported_flags = nfft_supported_flags_tuple

cdef object nfft_flags_dict
nfft_flags_dict = {
    'PRE_PHI_HUT':PRE_PHI_HUT,
    'FG_PSI':FG_PSI,
    'PRE_LIN_PSI':PRE_LIN_PSI,
    'PRE_FG_PSI':PRE_FG_PSI,
    'PRE_PSI':PRE_PSI,
    'PRE_FULL_PSI':PRE_FULL_PSI,
    'MALLOC_X':MALLOC_X,
    'MALLOC_F_HAT':MALLOC_F_HAT,
    'MALLOC_F':MALLOC_F,
    'FFT_OUT_OF_PLACE':FFT_OUT_OF_PLACE,
    'FFTW_INIT':FFTW_INIT,
    'NFFT_SORT_NODES':NFFT_SORT_NODES,
    'NFFT_OMP_BLOCKWISE_ADJOINT':NFFT_OMP_BLOCKWISE_ADJOINT,
    'PRE_ONE_PSI':PRE_ONE_PSI,
    }
nfft_flags = nfft_flags_dict.copy()

cdef object fftw_flags_dict
fftw_flags_dict = {
    'FFTW_ESTIMATE':FFTW_ESTIMATE,
    'FFTW_DESTROY_INPUT':FFTW_DESTROY_INPUT,
    }
fftw_flags = fftw_flags_dict.copy()


cdef class NFFT:
    '''
    NFFT is a class for computing the multivariate Non-uniform Discrete
    Fourier (NDFT) transform using the NFFT library. The interface is
    designed to be somewhat pythonic, while retaining the features and
    naming of the C code internals. The computation of the NFFT is achieved
    in 3 steps: instantiation, precomputation and execution.

    On instantiation, sanity checks on the size parameters and computation
    flags are performed prior to initialization of the internal plan.
    External data arrays may be provided, otherwise internal Numpy arrays
    will be used. Any incompatibilities detected in the parameters will raise
    a ``ValueError`` exception.

    The nodes must be initialized prior to precomputing the operator with the
    :meth:`pynfft.NFFT.precompute` method.

    The forward and adjoint NFFT operation may be performed by calling the
    :meth:`pynfft.NFFT.trafo` or :meth:`pynfft.NFFT.adjoint`
    methods. The NDFT may also be computed by calling the
    :meth:`pynfft.NFFT.trafo_direct` or
    :meth:`pynfft.NFFT.adjoint_direct`.
    '''
    cdef nfft_plan _plan
    cdef int _d
    cdef int _m
    cdef int _M_total
    cdef int _N_total
    cdef object _f
    cdef object _f_hat
    cdef object _x
    cdef object _N
    cdef object _n
    cdef object _dtype
    cdef object _flags

    # where the C-related content of the class is being initialized
    def __cinit__(self, N, M, n=None, m=12, x=None, f=None, f_hat=None,
                  flags=None, *args, **kwargs):

        # support only double / double complex NFFT
        # TODO: if support for multiple floating precision lands in the
        # NFFT library, adapt this section to dynamically figure the
        # real and complex dtypes
        dtype_real = np.dtype('float64')
        dtype_complex = np.dtype('complex128')

        # NOTE: use of reshape([-1, 1]) to avoid working with 0-d arrays which
        # cannot be indexed explictly
        N = np.asarray(N).reshape([-1, 1])
        M_total = np.asarray(M).reshape([-1, 1])
        n = np.asarray(n).reshape([-1, 1]) if n is not None else 2 * N
        m = np.asarray(m).reshape([-1, 1])
        N_total = np.asarray(np.prod(N)).reshape([-1, 1])
        d = N.size

        # make sure N and n lengths are compatible
        if n.size != d:
            raise ValueError("N and n must be of same size")

        # make sure all size parameters fit with int32 storage dtype of
        # nfft_plan, otherwise high risks of malloc errors
        cdef int t
        for t in range(0, d):
            if not N[t, 0] > 0:
                raise ValueError('N must be strictly positive')
            if N[t, 0] >= <Py_ssize_t>limits.INT_MAX:
                raise ValueError('N must be less than ', str(limits.INT_MAX))
            if not n[t, 0] > 0:
                raise ValueError('n must be strictly positive')
            if n[t, 0] >= <Py_ssize_t>limits.INT_MAX:
                raise ValueError('n must be less than ', str(limits.INT_MAX))
        if not M_total[0, 0] > 0:
            raise ValueError("M must be a strictly positive scalar")
        if M_total[0, 0] >= <Py_ssize_t>limits.INT_MAX:
            raise ValueError('M must be less than ', str(limits.INT_MAX))
        if not m[0, 0] > 0:
            raise ValueError("m must be a strictly positive scalar")
        if m[0, 0] >= <Py_ssize_t>limits.INT_MAX:
            raise ValueError('m must be less than ', str(limits.INT_MAX))
        if not N_total[0, 0] > 0:
            raise ValueError("M must be a strictly positive scalar")
        if N_total[0, 0] >= <Py_ssize_t>limits.INT_MAX:
            raise ValueError('M must be less than ', str(limits.INT_MAX))

        # if external arrays are provided, checks whether they are compatible
        if x is not None:
            if not x.flags.c_contiguous:
                raise ValueError('x array must be contiguous')
            if x.dtype != dtype_real:
                raise ValueError('x must be of type float64')
            if x.size != M_total * d:
                raise ValueError('x must be of size %d'%(M_total * d))
            self._x = x
        else:
            self._x = np.empty(M_total*d, dtype=dtype_real)

        if f is not None:
            if not f.flags.c_contiguous:
                raise ValueError('f array must be contiguous')
            if f.dtype != dtype_complex:
                raise ValueError('f must be of type float64')
            if f.size != M_total:
                raise ValueError('f must be of size %d'%(M_total))
            self._f = f
        else:
            self._f = np.empty(M_total, dtype=dtype_complex)

        if f_hat is not None:
            if not f_hat.flags.c_contiguous:
                raise ValueError('f_hat array must be contiguous')
            if f_hat.dtype != dtype_complex:
                raise ValueError('f_hat must be of type float64')
            if f_hat.size != N_total:
                raise ValueError('f_hat must be of size %d'%(N_total))
            self._f_hat = f_hat
        else:
            self._f_hat = np.empty(N_total, dtype=dtype_complex)

        # convert tuple of litteral precomputation flags to its expected
        # C-compatible value. Each flag is a power of 2, which allows to compute
        # this value using BITOR operations.
        cdef unsigned int _nfft_flags = 0
        cdef unsigned int _fftw_flags = 0
        flags_used = ()

        # sanity checks on user specified flags if any,
        # else use default ones:
        if flags is not None:
            try:
                flags = tuple(flags)
            except:
                flags = (flags,)
            finally:
                for each_flag in flags:
                    if each_flag not in nfft_supported_flags_tuple:
                        raise ValueError('Unsupported flag: %s'%(each_flag))
                flags_used += flags
        else:
            flags_used += ('PRE_PHI_HUT', 'PRE_PSI',)

        # set specific flags, for which we don't want the user to have a say
        # on:
        # FFTW specific flags
        flags_used += ('FFTW_INIT', 'FFT_OUT_OF_PLACE', 'FFTW_ESTIMATE',
                'FFTW_DESTROY_INPUT',)

        # Parallel computation flag
        flags_used += ('NFFT_SORT_NODES',)

        # Parallel computation flag, set only if multivariate transform
        if d > 1:
            flags_used += ('NFFT_OMP_BLOCKWISE_ADJOINT',)

        # Calculate the flag code for the guru interface used for
        # initialization
        for each_flag in flags_used:
            try:
                _nfft_flags |= nfft_flags_dict[each_flag]
            except KeyError:
                try:
                    _fftw_flags |= fftw_flags_dict[each_flag]
                except KeyError:
                    raise ValueError('Invalid flag: ' + '\'' +
                        each_flag + '\' is not a valid flag.')

        # initialize plan
        cdef int _d = d
        cdef int _m = m[0, 0]
        cdef int _M_total = M_total[0, 0]
        cdef int _N_total = N_total[0, 0]

        cdef int *_N = <int *>malloc(sizeof(int) * _d)
        if _N == NULL:
            raise MemoryError
        for t in range(0, d):
            _N[t] = N[t, 0]

        cdef int *_n = <int *>malloc(sizeof(int) * _d)
        if _n == NULL:
            raise MemoryError
        for t in range(0, d):
            _n[t] = n[t, 0]

        try:
            nfft_init_guru(&self._plan, _d, _N, _M_total, _n, _m,
                    _nfft_flags, _fftw_flags)
        except:
            raise MemoryError
        finally:
            free(_N)
            free(_n)

        self._plan.x = (
            <double *>np.PyArray_DATA(self._x))
        self._plan.f = (
            <fftw_complex *>np.PyArray_DATA(self._f))
        self._plan.f_hat = (
            <fftw_complex *>np.PyArray_DATA(self._f_hat))
        self._d = d
        self._m = m[0, 0]
        self._M_total = M_total
        self._N_total = N_total
        self._N = tuple([N[t, 0] for t in range(d)])
        self._n = tuple([n[t, 0] for t in range(d)])
        self._dtype = dtype_real
        self._flags = flags_used

    # here, just holds the documentation of the class constructor
    def __init__(self, N, M, n=None, m=12, x=None, f=None, f_hat=None,
                 flags=None, *args, **kwargs):
        '''
        :param N: multi-bandwith size.
        :type N: int, tuple of int
        :param M: number of non-uniform samples.
        :type M: int
        :param n: oversampled multi-bandwith, default to 2 * N.
        :type n: int, tuple of int
        :param m: Cut-off parameter of the window function.
        :type m: int
        :param x: external array holding the nodes.
        :type x: ndarray
        :param f: external array holding the non-uniform samples.
        :type f: ndarray
        :param f_hat: external array holding the Fourier coefficients.
        :type f_hat: ndarray
        :param flags: list of precomputation flags, see note below.
        :type flags: tuple

        **Precomputation flags**

        This table lists the supported precomputation flags for the NFFT.

        +----------------------------+--------------------------------------------------+
        | Flag                       | Description                                      |
        +============================+==================================================+
        | PRE_PHI_HUT                | Precompute the roll-off correction coefficients. |
        +----------------------------+--------------------------------------------------+
        | FG_PSI                     | Convolution uses Fast Gaussian properties.       |
        +----------------------------+--------------------------------------------------+
        | PRE_LIN_PSI                | Convolution uses a precomputed look-up table.    |
        +----------------------------+--------------------------------------------------+
        | PRE_FG_PSI                 | Precompute Fast Gaussian.                        |
        +----------------------------+--------------------------------------------------+
        | PRE_PSI                    | Standard precomputation, uses M*(2m+2)*d values. |
        +----------------------------+--------------------------------------------------+
        | PRE_FULL_PSI               | Full precomputation, uses M*(2m+2)^d values.     |
        +----------------------------+--------------------------------------------------+

        Default value is ``flags = ('PRE_PHI_HUT', 'PRE_PSI')``.
        '''
        pass

    # where the C-related content of the class needs to be cleaned
    def __dealloc__(self):
        nfft_finalize(&self._plan)

    cpdef precompute(self):
        '''
        Precomputes the NFFT plan internals.

        .. warning::
            The nodes :attr:`pynfft.NFFT.x` must be initialized before
            precomputing.
        '''
        with nogil:
            nfft_precompute_one_psi(&self._plan)

    cpdef trafo(self):
        '''
        Performs the forward NFFT.

        Reads :attr:`pynfft.NFFT.f_hat` and stores the result in
        :attr:`pynfft.NFFT.f`.
        '''
        with nogil:
            nfft_trafo(&self._plan)

    cpdef trafo_direct(self):
        '''
        Performs the forward NDFT.

        Reads :attr:`pynfft.NFFT.f_hat` and stores the result in
        :attr:`pynfft.NFFT.f`.
        '''
        with nogil:
             nfft_trafo_direct(&self._plan)

    cpdef adjoint(self):
        '''
        Performs the adjoint NFFT.

        Reads :attr:`pynfft.NFFT.f` and stores the result in
        :attr:`pynfft.NFFT.f_hat`.
        '''
        with nogil:
            nfft_adjoint(&self._plan)

    cpdef adjoint_direct(self):
        '''
        Performs the adjoint NDFT.

        Reads :attr:`pynfft.NFFT.f` and stores the result in
        :attr:`pynfft.NFFT.f_hat`.
        '''
        with nogil:
             nfft_adjoint_direct(&self._plan)

    def __get_f(self):
        '''
        The vector of non-uniform samples.
        '''
        return self._f

    def __set_f(self, new_f):
        self._f[:] = new_f.ravel()[:]

    f = property(__get_f, __set_f)

    def __get_f_hat(self):
        '''
        The vector of Fourier coefficients.
        '''
        return self._f_hat

    def __set_f_hat(self, new_f_hat):
        self._f_hat[:] = new_f_hat.ravel()[:]

    f_hat = property(__get_f_hat, __set_f_hat)

    def __get_x(self):
        '''
        The nodes in time/spatial domain.
        '''
        return self._x

    def __set_x(self, new_x):
        self._x[:] = new_x.ravel()[:]

    x = property(__get_x, __set_x)

    def __get_d(self):
        '''
        The dimensionality of the NFFT.
        '''
        return self._d

    d = property(__get_d)

    def __get_m(self):
        '''
        The cut-off parameter of the window function.
        '''
        return self._m

    m = property(__get_m)

    def __get_M_total(self):
        '''
        The total number of samples.
        '''
        return self._M_total

    M_total = property(__get_M_total)

    def __get_N_total(self):
        '''
        The total number of Fourier coefficients.
        '''
        return self._N_total

    N_total = property(__get_N_total)

    def __get_N(self):
        '''
        The multi-bandwith size.
        '''
        return self._N

    N = property(__get_N)

    def __get_n(self):
        '''
        The oversampled multi-bandwith size.
        '''
        return self._n

    n = property(__get_n)

    def __get_dtype(self):
        '''
        The floating precision.
        '''
        return self._dtype

    dtype = property(__get_dtype)

    def __get_flags(self):
        '''
        The precomputation flags.
        '''
        return self._flags

    flags = property(__get_flags)


##########
# Solver #
##########

cdef object solver_flags_dict
solver_flags_dict = {
    'LANDWEBER':LANDWEBER,
    'STEEPEST_DESCENT':STEEPEST_DESCENT,
    'CGNR':CGNR,
    'CGNE':CGNE,
    'NORMS_FOR_LANDWEBER':NORMS_FOR_LANDWEBER,
    'PRECOMPUTE_WEIGHT':PRECOMPUTE_WEIGHT,
    'PRECOMPUTE_DAMP':PRECOMPUTE_DAMP,
    }
solver_flags = solver_flags_dict.copy()

cdef class Solver:
    '''
    Solver is a class for computing the adjoint NFFT iteratively. Using the
    solver should theoretically lead to more accurate results, even with just
    one iteration, than using :meth:`pynfft.NFFT.adjoint` or
    :meth:`pynfft.NFFT.adjoint_direct`.

    The instantiation requires a NFFT object used internally for the multiple
    forward and adjoint NFFT performed. The class uses conjugate-gradient as
    the default solver but alternative solvers can be specified.

    Because the stopping conidition of the iterative computation may change
    from one application to another, the implementation only let you carry
    one iteration at a time with a call to
    :meth:`pynfft.Solver.loop_one_step`. Initialization of the solver
    is done by calling the :meth:`pynfft.Solver.before_loop` method.

    The class exposes the internals of the solver through call to their
    respective properties. They should be treated as read-only values.
    '''
    cdef solver_plan_complex _plan
    cdef NFFT _nfft_plan
    cdef object _w
    cdef object _w_hat
    cdef object _y
    cdef object _f_hat_iter
    cdef object _r_iter
    cdef object _dtype
    cdef object _flags

    def __cinit__(self, NFFT nfft_plan, flags=None):

        # support only double / double complex NFFT
        # TODO: if support for multiple floating precision lands in the
        # NFFT library, adapt this section to dynamically figure the
        # real and complex dtypes
        dtype_real = np.dtype('float64')
        dtype_complex = np.dtype('complex128')

        # convert tuple of litteral precomputation flags to its expected
        # C-compatible value. Each flag is a power of 2, which allows to compute
        # this value using BITOR operations.
        cdef unsigned int _flags = 0
        flags_used = ()

        # sanity checks on user specified flags if any,
        # else use default ones:
        if flags is not None:
            try:
                flags = tuple(flags)
            except:
                flags = (flags,)
            finally:
                flags_used += flags
        else:
            flags_used += ('CGNR',)

        for each_flag in flags_used:
            try:
                _flags |= solver_flags_dict[each_flag]
            except KeyError:
                raise ValueError('Invalid flag: ' + '\'' +
                        each_flag + '\' is not a valid flag.')

        # initialize plan
        try:
            solver_init_advanced_complex(&self._plan,
                <nfft_mv_plan_complex*>&(nfft_plan._plan), _flags)
        except:
            raise MemoryError

        self._nfft_plan = nfft_plan

        cdef np.npy_intp shape[1]
        cdef int M_total = nfft_plan._M_total
        cdef int N_total = nfft_plan._N_total

        if 'PRECOMPUTE_WEIGHT' in flags_used:
            shape[0] = M_total
            self._w = np.PyArray_SimpleNewFromData(1, shape,
                np.NPY_FLOAT64, <void *>(self._plan.w))
            self._w[:] = 1  # make sure weights are initialized
        else:
            self._w = None

        if 'PRECOMPUTE_DAMP' in flags_used:
            shape[0] = N_total
            self._w_hat = np.PyArray_SimpleNewFromData(1, shape,
                np.NPY_FLOAT64, <void *>(self._plan.w_hat))
            self._w_hat[:] = 1  # make sure weights are initialized
        else:
            self._w_hat = None

        shape[0] = M_total
        self._y = np.PyArray_SimpleNewFromData(1, shape,
            np.NPY_COMPLEX128, <void *>(self._plan.y))

        shape[0] = N_total
        self._f_hat_iter = np.PyArray_SimpleNewFromData(1, shape,
            np.NPY_COMPLEX128, <void *>(self._plan.f_hat_iter))
        self._f_hat_iter[:] = 0  # default initial guess

        shape[0] = M_total
        self._r_iter = np.PyArray_SimpleNewFromData(1, shape,
            np.NPY_COMPLEX128, <void *>(self._plan.r_iter))

        self._dtype = dtype_real
        self._flags = flags_used


    def __init__(self, nfft_plan, flags=None):
        '''
        :param plan: instance of NFFT.
        :type plan: :class:`pynfft.NFFT`
        :param flags: list of instantiation flags, see below.
        :type flags: tuple

        **Instantiation flags**

        +---------------------+-----------------------------------------------------------------------------+
        | Flag                | Description                                                                 |
        +=====================+=============================================================================+
        | LANDWEBER           | Use Landweber (Richardson) iteration.                                       |
        +---------------------+-----------------------------------------------------------------------------+
        | STEEPEST_DESCENT    | Use steepest descent iteration.                                             |
        +---------------------+-----------------------------------------------------------------------------+
        | CGNR                | Use conjugate gradient (normal equation of the 1st kind).                   |
        +---------------------+-----------------------------------------------------------------------------+
        | CGNE                | Use conjugate gradient (normal equation of the 2nd kind).                   |
        +---------------------+-----------------------------------------------------------------------------+
        | NORMS_FOR_LANDWEBER | Use Landweber iteration to compute the residual norm.                       |
        +---------------------+-----------------------------------------------------------------------------+
        | PRECOMPUTE_WEIGHT   | Weight the samples, e.g. to cope with varying sampling density.             |
        +---------------------+-----------------------------------------------------------------------------+
        | PRECOMPUTE_DAMP     | Weight the Fourier coefficients, e.g. to favour fast decaying coefficients. |
        +---------------------+-----------------------------------------------------------------------------+

        Default value is ``flags = ('CGNR',)``.
        '''
        pass

    def __dealloc__(self):
        solver_finalize_complex(&self._plan)

    cpdef before_loop(self):
        '''
        Initialize solver internals.
        '''
        with nogil:
            solver_before_loop_complex(&self._plan)

    cpdef loop_one_step(self):
        '''
        Perform one iteration.
        '''
        with nogil:
            solver_loop_one_step_complex(&self._plan)

    def __get_w(self):
        '''
        Weighting factors.
        '''
        return self._w

    def __set_w(self, new_w):
        if self._w is not None:
            self._w.ravel()[:] = new_w.ravel()[:]

    w = property(__get_w, __set_w)

    def __get_w_hat(self):
        '''
        Damping factors.
        '''
        return self._w_hat

    def __set_w_hat(self, new_w_hat):
        if self._w_hat is not None:
            self._w_hat.ravel()[:] = new_w_hat.ravel()[:]

    w_hat = property(__get_w_hat, __set_w_hat)

    def __get_y(self):
        '''
        Right hand side, samples.
        '''
        return self._y

    def __set_y(self, new_y):
        if self._y is not None:
            self._y.ravel()[:] = new_y.ravel()[:]

    y = property(__get_y, __set_y)

    def __get_f_hat_iter(self):
        '''
        Iterative solution.
        '''
        return self._f_hat_iter

    def __set_f_hat_iter(self, new_f_hat_iter):
        if self._f_hat_iter is not None:
            self._f_hat_iter.ravel()[:] = new_f_hat_iter.ravel()[:]

    f_hat_iter = property(__get_f_hat_iter, __set_f_hat_iter)

    def __get_r_iter(self):
        '''
        Residual vector.
        '''
        return self._r_iter

    r_iter = property(__get_r_iter)

    def __get_dtype(self):
        '''
        The floating precision.
        '''
        return self._dtype

    dtype = property(__get_dtype)

    def __get_flags(self):
        '''
        The precomputation flags.
        '''
        return self._flags

    flags = property(__get_flags)
