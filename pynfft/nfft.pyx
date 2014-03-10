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

import copy
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

nfft_flags = copy.copy(nfft_flags_dict)

fftw_flags_dict = {
    'FFTW_ESTIMATE':FFTW_ESTIMATE,
    'FFTW_DESTROY_INPUT':FFTW_DESTROY_INPUT,
    }

fftw_flags = copy.copy(fftw_flags_dict)

nfft_supported_flags_tuple = (
    'PRE_PHI_HUT',
    'FG_PSI',
    'PRE_LIN_PSI',
    'PRE_FG_PSI',
    'PRE_PSI',
    'PRE_FULL_PSI',
    )

nfft_supported_flags = copy.copy(nfft_supported_flags_tuple)


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

        self._x = x
        self._f = f
        self._f_hat = f_hat
        self._d = d
        self._M = M
        self._m = m
        self._N = N
        self._n = n
        self._dtype = dtype_complex
        self._flags = flags_used

        # connect Python arrays to plan internals
        self._plan.f = <fftw_complex *>np.PyArray_DATA(self._f)
        self._plan.f_hat = <fftw_complex *>np.PyArray_DATA(self._f_hat)
        self._plan.x = <double *>np.PyArray_DATA(self._x)


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
            self.nfft_trafo_direct()
        else:
            self.nfft_trafo()
        return self.f
    
    def adjoint(self, use_dft=False):
        '''Performs the adjoint NFFT.

        :param use_dft: whether to use the DFT instead of the fast algorithm.
        :type use_dft: boolean
        :returns: the updated f_hat array.
        :rtype: ndarray

        '''
        if use_dft:
            self.nfft_adjoint_direct()
        else:
            self.nfft_adjoint()
        return self.f_hat

    def precompute(self):
        '''Precomputes the NFFT plan internals.'''
        self.nfft_precompute()

    cdef void nfft_precompute(self):
        with nogil:
            nfft_precompute_one_psi(&self._plan)

    cdef void nfft_trafo(self):
        with nogil:
            nfft_trafo(&self._plan)

    cdef void nfft_trafo_direct(self):
        with nogil:
            nfft_trafo_direct(&self._plan)

    cdef void nfft_adjoint(self):
        with nogil:
            nfft_adjoint(&self._plan)

    cdef void nfft_adjoint_direct(self):
        with nogil:
            nfft_adjoint_direct(&self._plan)

    property f:
        
        '''The vector of non-uniform samples.'''

        def __get__(self):
            return self._f

        def __set__(self, array):
            self._f.ravel()[:] = array.ravel()

    property f_hat:

        '''The vector of Fourier coefficients.'''
        
        def __get__(self):
            return self._f_hat

        def __set__(self, array):
            self._f_hat.ravel()[:] = array.ravel()

    property x:
    
        '''The nodes in time/spatial domain.'''
        
        def __get__(self):
            return self._x

        def __set__(self, array):
            self._x.ravel()[:] = array.ravel()
   
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

