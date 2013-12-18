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


cdef class NFFT(object):
    '''
    NFFT is a class for computing the multivariate Non-uniform Discrete
    Fourier (NDFT) transform using the NFFT library. The interface is
    designed to be somewhat pythonic, whilst preserving the workflow of the 
    original C-library. Computation of the NFFT is achieved in 3 steps : 
    instantiation, precomputation and execution.
    
    On instantiation, the geometry of the transform is guessed from the shape 
    of the input arrays :attr:`f` and :attr:`f_hat`. The node array :attr:`x` 
    can be optionally provided, otherwise it will be created internally.
    
    Precomputation initializes the internals of the transform prior to 
    execution, and is called with the :meth:`precompute` method.
    
    The forward and adjoint NFFT can be called with :meth:`forward` and 
    :meth:`adjoint` respectively. Each of these methods support internal array 
    update and coercion to the right internal dtype.
    '''
    cdef nfft_plan _plan
    cdef int _d
    cdef int _M
    cdef int _m
    cdef object _N
    cdef object _n
    cdef object _dtype
    cdef object _flags
    cdef object __f   
    cdef object __f_hat
    cdef object __x

    # where the C-related content of the class is being initialized
    def __cinit__(self, f, f_hat, x=None, n=None, m=12, flags=None,
                  precompute=False, *args, **kwargs):

        # support only double / double complex NFFT
        # TODO: if support for multiple floating precision lands in the
        # NFFT library, adapt this section to dynamically figure the
        # real and complex dtypes
        dtype_real = np.dtype('float64')
        dtype_complex = np.dtype('complex128')

        # precompute at construct time possible if x is provided
        precompute =  precompute and (x is not None)

        # sanity checks on input arrays
        if not isinstance(f, np.ndarray):
            raise ValueError('f must be an instance of numpy.ndarray')

        if not f.flags.c_contiguous:
            raise ValueError('f must be C-contiguous')        

        if f.dtype != dtype_complex:
            raise ValueError('f must be of type %s'%(dtype_complex))                     

        if not isinstance(f_hat, np.ndarray):
            raise ValueError('f_hat: must be an instance of numpy.ndarray')                    

        if not f_hat.flags.c_contiguous:
            raise ValueError('f_hat must be C-contiguous')        

        if f_hat.dtype != dtype_complex:
            raise ValueError('f_hat must be of type %s'%(dtype_complex))

        # guess geometry from input array if missing from optional inputs
        M = f.size
        N = f_hat.shape
        d = f_hat.ndim
        n = n if n is not None else [2 * Nt for Nt in N]
        if len(n) != d:
            raise ValueError('n should be of same length as N')       
        N_total = np.prod(N)
        n_total = np.prod(n)

        # check geometry is compatible with C-class internals
        int_max = <Py_ssize_t>limits.INT_MAX
        if not all([Nt > 0 for Nt in N]):
            raise ValueError('N must be strictly positive')
        if not all([Nt < int_max for Nt in N]):
            raise ValueError('N exceeds integer limit value')
        if not N_total < int_max:
            raise ValueError('product of N exceeds integer limit value')
        if not all([nt > 0 for nt in n]):
            raise ValueError('n must be strictly positive')
        if not all([nt < int_max for nt in n]):
            raise ValueError('n exceeds integer limit value')        
        if not n_total < int_max:
            raise ValueError('product of n exceeds integer limit value')
        if not M > 0:
            raise ValueError('M must be strictly positive')
        if not M < int_max:
            raise ValueError('M exceeds integer limit value')
        if not m > 0:
            raise ValueError('m must be strictly positive')
        if not m < int_max:
            raise ValueError('m exceeds integer limit value')

        # sanity check on optional x array
        if x is not None:
            if not isinstance(x, np.ndarray):
                raise ValueError('x must be an instance of numpy.ndarray')
    
            if not x.flags.c_contiguous:
                raise ValueError('x must be C-contiguous')        
    
            if x.dtype != dtype_real:
                raise ValueError('x must be of type %s'%(dtype_real))
            
            try:
                x = x.reshape([M, d])
            except ValueError:
                raise ValueError('x is incompatible with geometry')
        else:
            x = np.empty([M, d], dtype=dtype_real)

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
        cdef int *p_N = <int *>malloc(sizeof(int) * d)
        if p_N == NULL:
            raise MemoryError
        for t in range(d):
            p_N[t] = N[t]

        cdef int *p_n = <int *>malloc(sizeof(int) * d)
        if p_n == NULL:
            raise MemoryError
        for t in range(d):
            p_n[t] = n[t]

        try:
            nfft_init_guru(&self._plan, d, p_N, M, p_n, m,
                    _nfft_flags, _fftw_flags)
        except:
            raise MemoryError
        finally:
            free(p_N)
            free(p_n)

        self.__x = x
        self.__f = f
        self.__f_hat = f_hat
        self._d = d
        self._M = M
        self._m = m
        self._N = N
        self._n = n
        self._dtype = dtype_complex
        self._flags = flags_used

        # connect Python arrays to plan internals
        self._update_f()
        self._update_f_hat()
        self._update_x()

        # optional precomputation
        if precompute:
            self.execute_precomputation()

    # here, just holds the documentation of the class constructor
    def __init__(self, f, f_hat, x=None, n=None, m=12, flags=None,
                 *args, **kwargs):
        '''
        :param f: external array holding the non-uniform samples.
        :type f: ndarray
        :param f_hat: external array holding the Fourier coefficients.
        :type f_hat: ndarray
        :param x: optional array holding the nodes.
        :type x: ndarray
        :param n: oversampled multi-bandwith, default to 2 * N.
        :type n: tuple of int
        :param m: Cut-off parameter of the window function.
        :type m: int
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

    def __dealloc__(self):
        nfft_finalize(&self._plan)

    def forward(self, use_dft=False):
        '''Performs the forward NFFT.

        :param use_dft: whether to use the DFT instead of the fast algorithm.
        :type use_dft: boolean
        :returns: the updated f array.
        :rtype: ndarray

        '''
        if use_dft:
            self.execute_trafo_direct()
        else:
            self.execute_trafo()
        return self.f
    
    def adjoint(self, use_dft=False):
        '''Performs the adjoint NFFT.

        :param use_dft: whether to use the DFT instead of the fast algorithm.
        :type use_dft: boolean
        :returns: the updated f_hat array.
        :rtype: ndarray

        '''
        if use_dft:
            self.execute_adjoint_direct()
        else:
            self.execute_adjoint()
        return self.f_hat

    def precompute(self):
        '''Precomputes the NFFT plan internals.'''
        self.execute_precomputation()

    cdef execute_precomputation(self):
        with nogil:
            nfft_precompute_one_psi(&self._plan)

    cdef execute_trafo(self):
        with nogil:
            nfft_trafo(&self._plan)

    cdef execute_trafo_direct(self):
        with nogil:
            nfft_trafo_direct(&self._plan)

    cpdef execute_adjoint(self):
        with nogil:
            nfft_adjoint(&self._plan)

    cpdef execute_adjoint_direct(self):
        with nogil:
            nfft_adjoint_direct(&self._plan)

    @property
    def f(self):
        '''The vector of non-uniform samples.'''
        return self.__f

    @f.setter
    def f(self, new_f):
        if not isinstance(new_f, np.ndarray):
            raise ValueError('array is not an instance of numpy.ndarray')     
        if not new_f.flags.c_contiguous:
            raise ValueError('array is not C-contiguous')
        if new_f.dtype != self.__f.dtype:
            raise ValueError('array dtype is not compatible')
        try:
            new_f = new_f.reshape(self.M)
        except ValueError:
            raise ValueError('array shape is not compatible')
        self.__f = new_f
        self._update_f()

    cdef _update_f(self):
        '''C-level array updater.''' 
        self._plan.f = <fftw_complex *>np.PyArray_DATA(self.__f)        

    @property
    def f_hat(self):
        '''The vector of Fourier coefficients.'''
        return self.__f_hat

    @f_hat.setter
    def f_hat(self, new_f_hat):
        if not isinstance(new_f_hat, np.ndarray):
            raise ValueError('array is not an instance of numpy.ndarray')  
        if not new_f_hat.flags.c_contiguous:
            raise ValueError('array is not C-contiguous')
        if new_f_hat.dtype != self.__f_hat.dtype:
            raise ValueError('array dtype is not compatible')
        try:
            new_f_hat = new_f_hat.reshape(self.N)
        except ValueError:
            raise ValueError('array shape is not compatible')
        self.__f_hat = new_f_hat
        self._update_f_hat()

    cdef _update_f_hat(self):
        '''C-level array updater.'''
        self._plan.f_hat = <fftw_complex *>np.PyArray_DATA(self.__f_hat)       

    @property
    def x(self):
        '''The nodes in time/spatial domain.'''
        return self.__x

    @x.setter
    def x(self, new_x):
        if not isinstance(new_x, np.ndarray):
            raise ValueError('array is not an instance of numpy.ndarray')                    
        if not new_x.flags.c_contiguous:
            raise ValueError('array is not C-contiguous')
        if new_x.dtype != self.__x.dtype:
            raise ValueError('array dtype is not compatible')
        try:
            new_x = new_x.reshape([self.M, self.d])
        except ValueError:
            raise ValueError('array shape is not compatible')
        self.__x = new_x
        self._update_x()
        
    cdef _update_x(self):
        '''C-level array updater.'''        
        self._plan.x = <double *>np.PyArray_DATA(self.__x)

    @property
    def d(self):
        '''The dimensionality of the NFFT.'''
        return self._d

    @property
    def m(self):
        '''The cut-off parameter of the window function.'''
        return self._m

    @property
    def M(self):
        '''The total number of samples.'''
        return self._M

    @property
    def N_total(self):
        '''The total number of Fourier coefficients.'''
        return np.prod(self.N)

    @property
    def N(self):
        '''The multi-bandwith size.'''
        return self._N

    @property
    def n(self):
        '''The oversampled multi-bandwith size.'''
        return self._n

    @property
    def dtype(self):
        '''The dtype of the NFFT.'''
        return self._dtype

    @property
    def flags(self):
        '''The precomputation flags.'''
        return self._flags


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
    Solver is a class for computing the adjoint NFFT iteratively..

    The instantiation requires a NFFT object used internally for the multiple
    forward and adjoint NFFT performed. The class uses conjugate-gradient as
    the default solver but alternative solvers can be specified.

    Because the stopping conidition of the iterative computation may change
    from one application to another, the implementation only let you carry
    one iteration at a time with a call to :meth:`loop_one_step`. 
    Initialization of the solver is done by calling the :meth:`before_loop` 
    method.

    The class exposes the internals of the solver through call to their
    respective properties.
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
        d = nfft_plan.d
        M = nfft_plan.M
        N = nfft_plan.N

        cdef np.npy_intp shape_M[1]
        shape_M[0] = M

        self._r_iter = np.PyArray_SimpleNewFromData(1, shape_M,
            np.NPY_COMPLEX128, <void *>(self._plan.r_iter))

        self._y = np.PyArray_SimpleNewFromData(1, shape_M,
            np.NPY_COMPLEX128, <void *>(self._plan.y))

        if 'PRECOMPUTE_WEIGHT' in flags_used:
            self._w = np.PyArray_SimpleNewFromData(1, shape_M,
                np.NPY_FLOAT64, <void *>(self._plan.w))
            self._w[:] = 1  # make sure weights are initialized
        else:
            self._w = None

        cdef np.npy_intp *shape_N
        try:
            shape_N = <np.npy_intp*>malloc(d*sizeof(np.npy_intp))
        except:
            raise MemoryError
        for dt in range(d):
            shape_N[dt] = N[dt]

        self._f_hat_iter = np.PyArray_SimpleNewFromData(d, shape_N,
            np.NPY_COMPLEX128, <void *>(self._plan.f_hat_iter))
        self._f_hat_iter[:] = 0  # default initial guess

        if 'PRECOMPUTE_DAMP' in flags_used:
            self._w_hat = np.PyArray_SimpleNewFromData(d, shape_N,
                np.NPY_FLOAT64, <void *>(self._plan.w_hat))
            self._w_hat[:] = 1  # make sure weights are initialized
        else:
            self._w_hat = None

        free(shape_N)

        self._dtype = dtype_complex
        self._flags = flags_used


    def __init__(self, nfft_plan, flags=None):
        '''
        :param plan: instance of NFFT.
        :type plan: :class:`NFFT`
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
        '''Initialize solver internals.'''
        with nogil:
            solver_before_loop_complex(&self._plan)

    cpdef loop_one_step(self):
        '''Perform one iteration.'''
        with nogil:
            solver_loop_one_step_complex(&self._plan)

    @property
    def w(self):
        '''Weighting factors.'''
        return self._w

    @property
    def w_hat(self):
        '''Damping factors.'''
        return self._w_hat

    @property
    def y(self):
        '''Right hand side, samples.'''
        return self._y

    @property
    def f_hat_iter(self):
        '''Iterative solution.'''
        return self._f_hat_iter

    @property
    def r_iter(self):
        '''Residual vector.'''
        return self._r_iter

    @property
    def dtype(self):
        '''The dtype of the solver.'''
        return self._dtype

    @property
    def flags(self):
        '''The precomputation flags.'''
        return self._flags
