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
fftwf_init_threads()
fftwl_init_threads()

# register cleanup callbacks
cdef void _cleanup():
    fftw_cleanup()
    fftw_cleanup_threads()
    fftwf_cleanup()
    fftwf_cleanup_threads()
    fftwl_cleanup()
    fftwl_cleanup_threads()

Py_AtExit(_cleanup)


nfft_flags_dict = {
    'PRE_PHI_HUT': PRE_PHI_HUT,
    'FG_PSI': FG_PSI,
    'PRE_LIN_PSI': PRE_LIN_PSI,
    'PRE_FG_PSI': PRE_FG_PSI,
    'PRE_PSI': PRE_PSI,
    'PRE_FULL_PSI': PRE_FULL_PSI,
    'MALLOC_X': MALLOC_X,
    'MALLOC_F_HAT': MALLOC_F_HAT,
    'MALLOC_F': MALLOC_F,
    'FFT_OUT_OF_PLACE': FFT_OUT_OF_PLACE,
    'FFTW_INIT': FFTW_INIT,
    'NFFT_SORT_NODES': NFFT_SORT_NODES,
    'NFFT_OMP_BLOCKWISE_ADJOINT': NFFT_OMP_BLOCKWISE_ADJOINT,
    'PRE_ONE_PSI': PRE_ONE_PSI,
}

nfft_flags = copy.copy(nfft_flags_dict)

fftw_flags_dict = {
    'FFTW_ESTIMATE': FFTW_ESTIMATE,
    'FFTW_DESTROY_INPUT': FFTW_DESTROY_INPUT,
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

    On instantiation, the geometry of the transform is provided. Optional
    computation parameters may also be defined.

    Precomputation initializes the internals of the transform prior to
    execution. First the non-uniform locations must be given to the plan
    via its :attr:`x` attribute. Computation can then be called with the
    :meth:`precompute` method.

    The forward and adjoint NFFT can be eventually performed by calling the
    :meth:`forward` and :meth:`adjoint` methods respectively. The input/output
    of the transform can be read/written by access to the :attr:`f` and
    :attr:`f_hat` attributes.
    '''

    # All allocation and initialization of C structures happens here
    def __cinit__(self, N, M, n=None, m=12, flags=None, prec='double',
                  *args, **kwargs):

        # raise AssertionError

        # 'long double' -> 'longdouble' (latter understood by np.dtype)
        dtype_real = np.dtype(str(prec).replace(' ', ''))
        dtype_complex = np.result_type(1j, dtype_real)
        if dtype_complex not in (np.complex64, np.complex128, np.complex256):
            raise ValueError('`prec` {!r} not recognized'.format(prec))

        # Sanity checks on geometry parameters
        try:
            N = tuple(N)
        except TypeError:
            N = (N,)

        if n is None:
            n = [2 * Nt for Nt in N]

        try:
            n = tuple(n)
        except TypeError:
            n = (n,)

        d = len(N)
        N_total = np.prod(N)
        n_total = np.prod(n)

        if len(n) != d:
            raise ValueError('n should be of same length as N')

        # Check geometry is compatible with C-class internals
        int_max = <Py_ssize_t>limits.INT_MAX
        if not all(Nt > 0 for Nt in N):
            raise ValueError('N must be strictly positive')
        if not all(Nt < int_max for Nt in N):
            raise ValueError('N exceeds integer limit value')
        if not N_total < int_max:
            raise ValueError('product of N exceeds integer limit value')
        if not all(nt > 0 for nt in n):
            raise ValueError('n must be strictly positive')
        if not all(nt < int_max for nt in n):
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

        # Safeguard against oversampled grid size too small for kernel size
        if not all(nt > m for nt in n):
            raise ValueError('n must be larger than m')

        # Convert tuple of literal precomputation flags to its expected
        # C-compatible value. Each flag is a power of 2, which allows to compute
        # this value using BITOR operations.
        cdef unsigned int _nfft_flags = 0
        cdef unsigned int _fftw_flags = 0
        flags_used = ()

        # Sanity checks on user specified flags if any,
        # else use default ones:
        if flags is not None:
            try:
                flags = tuple(flags)
            except:
                flags = (flags,)
            finally:
                for flag in flags:
                    if flag not in nfft_supported_flags_tuple:
                        raise ValueError('Unsupported flag: {}'.format(flag))
                flags_used += flags
        else:
            flags_used += ('PRE_PHI_HUT', 'PRE_PSI')

        # Set specific flags, for which we don't want the user to have a say on

        # FFTW specific flags
        flags_used += ('FFTW_INIT', 'FFT_OUT_OF_PLACE', 'FFTW_ESTIMATE',
                       'FFTW_DESTROY_INPUT',)

        # Memory allocation flags
        flags_used += ('MALLOC_F', 'MALLOC_F_HAT', 'MALLOC_X')

        # Parallel computation flag
        flags_used += ('NFFT_SORT_NODES',)

        # Parallel computation flag, set only if multivariate transform
        if d > 1:
            flags_used += ('NFFT_OMP_BLOCKWISE_ADJOINT',)

        # Calculate the flag code for the guru interface used for
        # initialization
        for flag in flags_used:
            try:
                _nfft_flags |= nfft_flags_dict[flag]
            except KeyError:
                try:
                    _fftw_flags |= fftw_flags_dict[flag]
                except KeyError:
                    raise ValueError("Invalid flag: '{}'".format(flag))

        # Initialize plan
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

        if dtype_complex == np.complex64:
            nfftf_init_guru(
                &self._plan_flt, d, p_N, M, p_n, m, _nfft_flags, _fftw_flags
            )
        elif dtype_complex == np.complex128:
            nfft_init_guru(
                &self._plan_dbl, d, p_N, M, p_n, m, _nfft_flags, _fftw_flags
            )
        elif dtype_complex == np.complex256:
            # segfaults
            nfftl_init_guru(
                &self._plan_ldbl, d, p_N, M, p_n, m, _nfft_flags, _fftw_flags
            )
        else:
            raise RuntimeError

        free(p_N)
        free(p_n)

        # Create array views
        cdef np.npy_intp *shape_f_hat
        shape_f_hat = <np.npy_intp *> malloc(d * sizeof(np.npy_intp))
        if shape_f_hat == NULL:
            raise MemoryError
        for dt in range(d):
            shape_f_hat[dt] = N[dt]

        if dtype_complex == np.complex64:
            self._f_hat = np.PyArray_SimpleNewFromData(
                d, shape_f_hat, np.NPY_COMPLEX64,
                <void *>(self._plan_flt.f_hat)
            )
        if dtype_complex == np.complex128:
            self._f_hat = np.PyArray_SimpleNewFromData(
                d, shape_f_hat, np.NPY_COMPLEX128,
                <void *>(self._plan_dbl.f_hat)
            )
        if dtype_complex == np.complex256:
            self._f_hat = np.PyArray_SimpleNewFromData(
                d, shape_f_hat, np.NPY_COMPLEX256,
                <void *>(self._plan_ldbl.f_hat)
            )
        else:
            raise RuntimeError

        free(shape_f_hat)

        cdef np.npy_intp shape_f[1]
        shape_f[0] = M

        if dtype_complex == np.complex64:
            self._f = np.PyArray_SimpleNewFromData(
                1, shape_f, np.NPY_COMPLEX64, <void *>(self._plan_flt.f)
            )
        elif dtype_complex == np.complex128:
            self._f = np.PyArray_SimpleNewFromData(
                1, shape_f, np.NPY_COMPLEX128, <void *>(self._plan_dbl.f)
            )
        elif dtype_complex == np.complex256:
            self._f = np.PyArray_SimpleNewFromData(
                1, shape_f, np.NPY_COMPLEX256, <void *>(self._plan_ldbl.f)
            )
        else:
            raise RuntimeError

        cdef np.npy_intp shape_x[2]
        shape_x[0] = M
        shape_x[1] = d

        if dtype_complex == np.complex64:
            self._x = np.PyArray_SimpleNewFromData(
                2, shape_x, np.NPY_FLOAT32, <void *>(self._plan_flt.x)
            )
        elif dtype_complex == np.complex128:
            self._x = np.PyArray_SimpleNewFromData(
                2, shape_x, np.NPY_FLOAT64, <void *>(self._plan_dbl.x)
            )
        elif dtype_complex == np.complex256:
            self._x = np.PyArray_SimpleNewFromData(
                2, shape_x, np.NPY_FLOAT128, <void *>(self._plan_ldbl.x)
            )
        else:
            raise RuntimeError

        # Initialize class members
        self._d = d
        self._M = M
        self._m = m
        self._N = N
        self._n = n
        self._dtype = dtype_complex
        self._flags = flags_used


    # Just holds the documentation of the class constructor
    def __init__(self, N, M, n=None, m=12, flags=None, prec='double',
                 *args, **kwargs):
        '''
        :param N: multi-bandwith.
        :type N: tuple of int
        :param M: total number of samples.
        :type n: int
        :param n: oversampled multi-bandwith, defaults to ``2 * N``.
        :type n: tuple of int
        :param m: Cut-off parameter of the window function.
        :type m: int
        :param flags: list of precomputation flags, see note below.
        :type flags: tuple
        :param prec: Floating point precision, can be ``'double'`` (default), ``'single'`` or ``'long double'``
        :type prec: string

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
        if self._dtype == np.complex64:
            nfftf_finalize(&self._plan_flt)
        elif self._dtype == np.complex128:
            nfft_finalize(&self._plan_dbl)
        elif self._dtype == np.complex256:
            nfftl_finalize(&self._plan_ldbl)
        else:
            raise RuntimeError

    def precompute(self):
        '''Precomputes the NFFT plan internals.'''
        self._precompute()

    def trafo(self, use_dft=False):
        '''Performs the forward NFFT.

        :param use_dft: whether to use the DFT instead of the fast algorithm.
        :type use_dft: boolean
        :returns: the updated f array.
        :rtype: ndarray

        '''
        if use_dft:
            self._trafo_direct()
        else:
            self._trafo()
        return self.f

    def adjoint(self, use_dft=False):
        '''Performs the adjoint NFFT.

        :param use_dft: whether to use the DFT instead of the fast algorithm.
        :type use_dft: boolean
        :returns: the updated f_hat array.
        :rtype: ndarray

        '''
        if use_dft:
            self._adjoint_direct()
        else:
            self._adjoint()
        return self.f_hat

    cdef void _precompute(self):
        if self._dtype == np.complex64:
            with nogil:
                nfftf_precompute_one_psi(&self._plan_flt)
        elif self._dtype == np.complex128:
            with nogil:
                nfft_precompute_one_psi(&self._plan_dbl)
        elif self._dtype == np.complex256:
            with nogil:
                nfftl_precompute_one_psi(&self._plan_ldbl)

    cdef void _trafo(self):
        if self._dtype == np.complex64:
            with nogil:
                nfftf_trafo(&self._plan_flt)
        elif self._dtype == np.complex128:
            with nogil:
                nfft_trafo(&self._plan_dbl)
        elif self._dtype == np.complex256:
            with nogil:
                nfftl_trafo(&self._plan_ldbl)

    cdef void _trafo_direct(self):
        if self._dtype == np.complex64:
            with nogil:
                nfftf_trafo_direct(&self._plan_flt)
        elif self._dtype == np.complex128:
            with nogil:
                nfft_trafo_direct(&self._plan_dbl)
        elif self._dtype == np.complex256:
            with nogil:
                nfftl_trafo_direct(&self._plan_ldbl)

    cdef void _adjoint(self):
        if self._dtype == np.complex64:
            with nogil:
                nfftf_adjoint(&self._plan_flt)
        elif self._dtype == np.complex128:
            with nogil:
                nfft_adjoint(&self._plan_dbl)
        elif self._dtype == np.complex256:
            with nogil:
                nfftl_adjoint(&self._plan_ldbl)

    cdef void _adjoint_direct(self):
        if self._dtype == np.complex64:
            with nogil:
                nfftf_adjoint_direct(&self._plan_flt)
        if self._dtype == np.complex128:
            with nogil:
                nfft_adjoint_direct(&self._plan_dbl)
        if self._dtype == np.complex256:
            with nogil:
                nfftl_adjoint_direct(&self._plan_ldbl)

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

