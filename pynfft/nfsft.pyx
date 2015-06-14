# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Taco Cohen
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
import nfft

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

nfsft_flags_dict = {
    'NFSFT_NORMALIZED': NFSFT_NORMALIZED,
    'NFSFT_USE_NDFT': NFSFT_USE_NDFT,
    'NFSFT_USE_DPT': NFSFT_USE_DPT,
    'NFSFT_MALLOC_X': NFSFT_MALLOC_X,                        # Used in nfsft_init_guru; automatically set by pyNFFT
    'NFSFT_MALLOC_F_HAT': NFSFT_MALLOC_F_HAT,                # Used in nfsft_init_guru; automatically set by pyNFFT
    'NFSFT_MALLOC_F': NFSFT_MALLOC_F,                        # Used in nfsft_init_guru; automatically set by pyNFFT
    'NFSFT_PRESERVE_F_HAT': NFSFT_PRESERVE_F_HAT,
    'NFSFT_PRESERVE_X': NFSFT_PRESERVE_X,
    'NFSFT_PRESERVE_F': NFSFT_PRESERVE_F,
    'NFSFT_DESTROY_F_HAT': NFSFT_DESTROY_F_HAT,
    'NFSFT_DESTROY_X': NFSFT_DESTROY_X,
    'NFSFT_DESTROY_F': NFSFT_DESTROY_F,
    'NFSFT_NO_DIRECT_ALGORITHM': NFSFT_NO_DIRECT_ALGORITHM,  # Used in nfsft_precompute
    'NFSFT_NO_FAST_ALGORITHM': NFSFT_NO_FAST_ALGORITHM,      # Used in nfsft_precompute
    'NFSFT_ZERO_F_HAT': NFSFT_ZERO_F_HAT,
    }

nfsft_flags = copy.copy(nfsft_flags_dict)

nfsft_supported_flags_tuple = (
     # Flags NFSFT_MALLOC_X, NFSFT_MALLOC_F, NFSFT_MALLOC_F_HAT
     # are not supported, because we they are used internally in this python interface
     # (we always ask the NFFT library to allocate these)
     # Flags NFSFT_DESTROY_F_HAT, NFSFT_DESTROY_X, NFSFT_DESTROY_F are not used anywhere in nfsft.c
    'NFSFT_NORMALIZED',
    'NFSFT_USE_NDFT',
    'NFSFT_USE_DPT',
    'NFSFT_PRESERVE_F_HAT',
    'NFSFT_PRESERVE_X',
    'NFSFT_PRESERVE_F'
    'NFSFT_NO_FAST_ALGORITHM',
    'NFSFT_NO_DIRECT_ALGORITHM',
    'NFSFT_ZERO_F_HAT'
    )


cdef int _precomputed_N = -1

cdef class NFSFT(object):
    """
    NFSFT is a class for computing the Non-equispaced Fast Spherical
    Fourier Transform (NFSFT) using the NFFT library. The interface is
    designed to be more pythonic than the C-library.

    The NFSFT class follows a RAII (Resource Allocation Is Initialization)
    pattern. On instantiation, the geometry of the transform (i.e. the sampling
    grid) is provided. Optional computation parameters may also be defined.
    Global precomputation is then performed in the constructor.

    The forward and adjoint NFSFT can be performed by calling the
    :meth:`forward` and :meth:`adjoint` methods respectively. The input/output 
    of the transform can be read/written by access to the :attr:`f` and 
    :attr:`f_hat` attributes.
    """

    # where the C-related content of the class is being initialized
    def __cinit__(self, N, x, m=12, kappa=1000., nfsft_flags=None, nfft_flags=None, *args, **kwargs):
        """

        :param N: the number of Fourier coefficients
        :type int
        :param x: the sampling set. These are the points in the spatial domain at which we wish to evaluate
        our functions. Shape (M, 2) - first column is theta in [0, pi], second column is phi in [0, 2 pi].
        :param m: the cutoff for the nfft that is used internally.
        :param kappa: stabilization parameter
          (see Keiner, J., Potts, D., Fast evaluation of quadrature formulae on the sphere. Math. Comput. 77, 2008)
        :param nfsft_flags:
        :param nfft_flags:
        :param args:
        :param kwargs:
        :return:
        """
        # support only double / double complex NFFT
        # TODO: if support for multiple floating precision lands in the NFFT library,
        # adapt this section to dynamically figure the real and complex dtypes
        dtype_real = np.dtype('float64')
        dtype_complex = np.dtype('complex128')

        # check geometry is compatible with C-class internals
        int_max = <Py_ssize_t>limits.INT_MAX
        if not N > 0:
            raise ValueError('N must be strictly positive')
        if not N < int_max:
            raise ValueError('N exceeds integer limit value')
        if not m > 0:
            raise ValueError('m must be strictly positive')
        if not m < int_max:
            raise ValueError('m exceeds integer limit value')

        # initialize private class members
        self._d = 2
        self._M = x.shape[0]  # M
        self._m = m
        self._N = N
        self._dtype = dtype_complex

        # sanity checks on user specified flags if any,
        # else use default ones:
        if nfsft_flags is not None:
            try:
                nfsft_flags = tuple(nfsft_flags)
            except:
                nfsft_flags = (nfsft_flags,)
            finally:
                for each_flag in nfsft_flags:
                    if each_flag not in nfsft_supported_flags_tuple:
                        raise ValueError('Unsupported flag: %s' % each_flag)
        else:
            nfsft_flags = tuple()
        self._nfsft_flags = nfsft_flags

        # memory allocation flags; we want NFFT to allocate the memory for us.
        # No need to bother the python user with this.
        nfsft_flags_internal = ('NFSFT_MALLOC_F', 'NFSFT_MALLOC_F_HAT', 'NFSFT_MALLOC_X')

        # convert tuple of literal precomputation flags to its expected
        # C-compatible value. Each flag is a power of 2, which allows to compute
        # this value using BITOR operations.
        cdef unsigned int _nfsft_flags = 0
        for each_flag in nfsft_flags + nfsft_flags_internal:
            try:
                _nfsft_flags |= nfsft_flags_dict[each_flag]
            except KeyError:
                raise ValueError('Invalid flag: ' + '\'' + each_flag + '\' is not a valid flag.')


        # sanity checks on user specified flags if any,
        # else use default ones:
        if nfft_flags is not None:
            try:
                nfft_flags = tuple(nfft_flags)
            except:
                nfft_flags = (nfft_flags,)
            finally:
                for each_flag in nfft_flags:
                    if each_flag not in nfft.nfft_supported_flags:
                        raise ValueError('Unsupported flag: %s' % each_flag)
        else:
            nfft_flags = ('PRE_PHI_HUT', 'PRE_PSI')
        self._nfft_flags = nfft_flags

        # Parallel computation flag
        nfft_flags_internal = ('NFFT_SORT_NODES', 'NFFT_OMP_BLOCKWISE_ADJOINT')

        # nfft precomputation flags; defaults from nfsft_init_advanced() from nfsft.c
        #nfft_flags_used += ('PRE_PHI_HUT', 'PRE_PSI', 'FFTW_INIT', 'FFT_OUT_OF_PLACE')
        nfft_flags_internal += ('FFTW_INIT', 'FFT_OUT_OF_PLACE')

        # convert tuple of literal precomputation flags to its expected
        # C-compatible value. Each flag is a power of 2, which allows to compute
        # this value using BITOR operations.
        cdef unsigned int _nfft_flags = 0
        for each_flag in nfft_flags + nfft_flags_internal:
            try:
                _nfft_flags |= nfft.nfft_flags[each_flag]
            except KeyError:
                raise ValueError('Invalid flag: ' + '\'' + each_flag + '\' is not a valid flag.')

        # First, do global precomputation
        # This requires a maximum order N
        # If another instance of this class was created before, the precomputation was done already,
        # but possibly with a different N.
        # The following appears to work: if _precomputed_N is larger or equal to our current N, we don't need to
        # recompute. If the previous _precomputed_N is smaller, we need to do the recomputation again and
        # if a precomputation was already done at some point, we need to forget it first.
        global _precomputed_N
        if _precomputed_N < N:
            if _precomputed_N != -1:
                nfsft_forget()
            nfsft_precompute(N, kappa, 0, 0)  #  _nfsft_flags=0, _fpt_flags=0)
            _precomputed_N = N

        # Initialize the plan:
        nfsft_init_guru(&self._plan,
                        N,
                        self._M,
                        _nfsft_flags,
                        _nfft_flags,
                        m)

        # create array views
        cdef np.npy_intp *shape_f_hat
        shape_f_hat = <np.npy_intp *> malloc(1 * sizeof(np.npy_intp))
        if shape_f_hat == NULL:
            raise MemoryError

        # Note: N_total() uses self._N, so this member has to be initialized before this call.
        shape_f_hat[0] = self.N_total
        self._f_hat = np.PyArray_SimpleNewFromData(1, shape_f_hat,
            np.NPY_COMPLEX128, <void *>(self._plan.f_hat))
        free(shape_f_hat)

        cdef np.npy_intp shape_f[1]
        shape_f[0] = self._M
        self._f = np.PyArray_SimpleNewFromData(1, shape_f,
            np.NPY_COMPLEX128, <void *>(self._plan.f))

        cdef np.npy_intp shape_x[2]
        shape_x[0] = self._M
        shape_x[1] = 2
        self._x = np.PyArray_SimpleNewFromData(2, shape_x,
            np.NPY_FLOAT64, <void *>(self._plan.x))

        # Set the sampling set and do sampling-set-dependent pre-computations:
        self._set_x(x)


    # here, just holds the documentation of the class constructor
    def __init__(self, N, x, m=12, alpha=1000., nfsft_flags=None, nfft_flags=None, *args, **kwargs):
        '''
        :param N: multi-bandwidth.
        :type N: tuple of int
        :param x: sampling set
        :param n: oversampled multi-bandwidth, default to 2 * N.
        :type n: tuple of int
        :param m: Cut-off parameter of the window function.
        :type m: int
        :param flags: list of pre-computation flags, see note below.
        :type flags: tuple
        
        **Precomputation flags**

        This table lists the supported precomputation flags for the NFFT.

        +----------------------------+--------------------------------------------------+
        | Flag                       | Description                                      |
        +============================+==================================================+
        |
        +----------------------------+--------------------------------------------------+

        Default value is ``flags =
        '''
        pass

    def __dealloc__(self):
        nfsft_finalize(&self._plan)

    def trafo(self, use_dft=False, return_copy=True):
        """
        Performs the forward NFSFT, from spectral to spatial.

        :param use_dft: whether to use the DFT instead of the fast algorithm.
        :type use_dft: boolean
        :returns: the updated f array.
        :rtype: ndarray
        """

        if use_dft:
            self._trafo_direct()
        else:
            self._trafo()

        if return_copy:
            return self.f.copy()
        else:
            return self.f
    
    def adjoint(self, use_dft=False, return_copy=True, return_flat=True):
        """
        Performs the adjoint NFSFT.

        :param use_dft: whether to use the DFT instead of the fast algorithm.
        :type use_dft: boolean
        :returns: the updated f_hat array.
        :rtype: ndarray
        """
        if use_dft:
            self._adjoint_direct()
        else:
            self._adjoint()

        if return_flat:
            if not return_copy:
                raise ValueError('Cannot return f_hat in flat format without copying data.')
            retval = self.get_f_hat_flat()
        else:
            retval = self.f_hat

        if return_copy:
            return retval.copy()
        else:
            return retval

    def spectral_index_flat(self, l, m):
        """
        Compute the flat index in self.f_hat of harmonic (l, m)

        :param l: the degree of the spherical harmonic (0 <= l <= :attr:'N')
        :param m: the order of the spherical harmonic (-l <= m <= l)
        :return: the index in the flattened coefficient vector f_hat as return by get_f_hat_flat()
        """
        # sum_i=0^{l-1} (2 i + 1) = l**2 is the index of (l=l, m=-l)
        # so l ** 2 + l is the index of (l, 0),
        # so l ** 2 + l + m is the index of (l, m)
        return l ** 2 + l + m

    def get_f_hat_flat(self):
        """
	Return a copy of the internal array f_hat in a user-friendly, flat layout.
	The (l, m) spectral coefficients are stored in this order:
	(0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, -1), ...
	"""
	cdef int i, l, m
        out = np.zeros((self._N + 1) ** 2, dtype='complex')
        i = 0
        for l in range(0, self._N + 1):
            for m in range(-l, l + 1):
                out[i] = self._f_hat[self._spectral_index(l, m)]
                i += 1
        return out

    def set_f_hat_flat(self, f_hat):
        """
        Set the spectral coefficients f_hat from a flat array.
        The elements of the array are assumed to correspond to spectral indices:
        (l, m) = (0, 0), (1, -1), (1, 0), (1, 1), (2, -2), ..., (N, N)
	(in that order)

        The length of this array is N_total = sum_l=0^N (2l + 1) = (N + 1)^2
        :param f_hat: the flat array of spectral coefficients
        """
        if f_hat.ndim != 1:
            raise ValueError('set_f_hat_flat requires a flat f_hat as input')
        if f_hat.size != (self._N + 1) ** 2:
            raise ValueError('Invalid size for f_hat (expected: (N+1)^2 = ' + str((self._N + 1) ** 2) +
                             ', found: ' + str(f_hat.size))
        cdef int l
        cdef int m
        cdef int i = 0
        for l in range(self._N + 1):
            for m in range(-l, l + 1):
                self._f_hat[self._spectral_index(l, m)] = f_hat[i]
                i += 1

    # Private methods
    cdef void _trafo(self):
        with nogil:
            nfsft_trafo(&self._plan)

    cdef void _trafo_direct(self):
        with nogil:
            nfsft_trafo_direct(&self._plan)

    cdef void _adjoint(self):
        with nogil:
            nfsft_adjoint(&self._plan)

    cdef void _adjoint_direct(self):
        with nogil:
            nfsft_adjoint_direct(&self._plan)

    cdef int _spectral_index(self, int l, int m):
        """
        Index of element (l, m) in the internal array self._f_hat
        """
        # Got this formula from nfft3.h, macro NFSFT_INDEX(l, m)
        # Apparently, this internal array is much larger than is needed to store N_total number of coefficients.
        return (2 * self._N + 2) * (self._N - m + 1) + self._N + l + 1

    def _set_x(self, x):
        """
        Set the sampling set self._x and perform precomputations.
        This function performs the translation to NFFT's storage scheme for x,
        from a more pythonic one used in the public interface of this class.

        :param x:
        :return:
        """
        if x.ndim != 2:
            raise ValueError('x.ndim must equal 2')
        if x.shape[0] != self._M:
            raise ValueError('x.shape[0] must equal self.M')
        if x.shape[1] != 2:
            raise ValueError('x.shape[1] must equal 2')

        theta = x[:, 0]
        phi = x[:, 1]

        if np.any(theta < 0) or np.any(theta > np.pi):
            raise ValueError('All theta angles must be in [0, pi]')
        if np.any(phi < 0) or np.any(phi >= 2 * np.pi):
            raise ValueError('All phi angles must be in [0, 2 pi)')

        nu_tilde = theta / (2 * np.pi)
        phi_tilde = (phi / (2 * np.pi)) * (phi < np.pi) + (phi / (2 * np.pi) - 1) * (phi >= np.pi)
        even_inds = np.arange(0, 2 * self._M, 2)
        odd_inds = np.arange(1, 2 * self._M, 2)
        self._x.ravel()[even_inds] = phi_tilde
        self._x.ravel()[odd_inds] = nu_tilde

        # Whenever we change the sampling set, we need to redo precomputations:
        nfsft_precompute_x(&self._plan)

    property f:

        """The vector of non-uniform samples, as a flat array."""

        def __get__(self):
            return self._f

        def __set__(self, array):
            self._f.ravel()[:] = array.ravel()

    property f_hat:

        """
        The vector of Fourier coefficients, shape (N + 1, 2 * N + 1).
        Element (l, m) can be accessed as f_hat[l, N - m].
        f_hat is a view on the memory used internally, so no memory is copied.
        """
        
        def __get__(self):

	    # For some reason, the NFFT library doesn't use all coefficients of f_hat.
            # Internally, it uses an array of length (2N + 1)^2, even though only
            # sum_l=0^N (2l + 1) = (N + 1)^2 coefficients are needed.

            # Index this array as f_hat[l, N - m]
            return self._f_hat.reshape(2 * self._N + 2, 2 * self._N + 2)[1:, self._N + 1:].T

        def __set__(self, f_hat):
            #self._f_hat.ravel()[:] = f_hat.ravel()

            """
            cdef int l
            cdef int m
            cdef int i = 0
            for l in range(self._N + 1):
                for m in range(-l, l + 1):
                    self._f_hat[self._spectral_index(l, m)] = f_hat[i]
                    i += 1
            """
            self._f_hat.reshape(2 * self._N + 2, 2 * self._N + 2)[1:, self._N + 1:] = np.array(f_hat).T


    property x:

        """The nodes (nu_i, phi_i) in the spatial domain."""
        
        def __get__(self):

            even_inds = np.arange(0, 2 * self._M, 2)
            odd_inds = np.arange(1, 2 * self._M, 2)
            phi_tilde = self._x.ravel()[even_inds]
            nu_tilde = self._x.ravel()[odd_inds]
            phi = (phi_tilde * 2 * np.pi) * (phi_tilde >= 0.) + ((phi_tilde + 1) * 2 * np.pi) * (phi_tilde < 0.)
            nu = nu_tilde * 2 * np.pi
            return np.c_[nu[:, None], phi[:, None]]


    @property
    def d(self):
        """The dimensionality of the NFSFT. This is always 2, since we are working with the 2-sphere."""
        return self._d

    @property
    def m(self):
        """The cut-off parameter of the window function"""
        return self._m

    @property
    def M(self):
        """The total number of samples."""
        return self._M

    @property
    def M_total(self):
        """The total number of samples."""
        return self._M

    @property
    def N(self):
        """The bandwith."""
        return self._N

    @property
    def N_total(self):
        """The total number of Fourier coefficients."""
        return (2 * self._N + 2) * (2 * self._N + 2)

    @property
    def dtype(self):
        """The dtype of the NFFT."""
        return self._dtype

    @property
    def nfft_flags(self):
        """The NFFT flags"""
        return self._nfft_flags

    @property
    def nfsft_flags(self):
        """The NFSFT flags"""
        return self._nfsft_flags