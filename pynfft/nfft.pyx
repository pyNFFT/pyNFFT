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
#
# Ghislain Vaillant
# ghislain.vallant@kcl.ac.uk

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc cimport limits
from cnfft3 cimport *


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
_nfft_flags_dict = nfft_flags_dict.copy()

cdef object fftw_flags_dict
fftw_flags_dict = {
    'FFTW_ESTIMATE':FFTW_ESTIMATE,
    'FFTW_DESTROY_INPUT':FFTW_DESTROY_INPUT,
}
_fftw_flags_dict = fftw_flags_dict.copy()


# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


cdef class NFFT:

    # where the C-related content of the class is being initialized
    def __cinit__(self, N, M, n=None, m=12, x=None, f=None, f_hat=None,
                  dtype=None, flags=None, *args, **kwargs):
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
            if x.flags['C_CONITGUOUS'] is False:
                raise ValueError('x array must be contiguous')
            if x.dtype != np.float64:
                raise ValueError('x must be of type float64')
            if x.size != M_total * d
                raise ValueError('x must be of size %d'%(M_total * d))

        if f is not None:
            if f.flags['C_CONITGUOUS'] is False:
                raise ValueError('f array must be contiguous')
            if f.dtype != np.complex128:
                raise ValueError('f must be of type float64')
            if f.size != M_total
                raise ValueError('f must be of size %d'%(M_total))

        if f_hat is not None:
            if f_hat.flags['C_CONITGUOUS'] is False:
                raise ValueError('f_hat array must be contiguous')
            if f_hat.dtype != np.complex128:
                raise ValueError('f_hat must be of type float64')
            if f_hat.size != M_total
                raise ValueError('f_hat must be of size %d'%(N_total))

        # convert tuple of litteral precomputation flags to its expected
        # C-compatible value. Each flag is a power of 2, which allows to compute
        # this value using BITOR operations.
        flags_used = []
        cdef unsigned int _nfft_flags = 0
        cdef unsigned int _fftw_flags = 0

        nfft_flags = flags
        if nfft_flags is None:
            # default nfft flags, adapted from nfft.c
            if d > 1:
                nfft_flags = ('PRE_PHI_HUT',
                              'PRE_PSI',
                              'FFTW_INIT',
                              'FFT_OUT_OF_PLACE',
                              'NFFT_SORT_NODES',
                              'NFFT_OMP_BLOCKWISE_ADJOINT',
                              'FFTW_ESTIMATE',
                              'FFTW_DESTROY_INPUT',)
            else:
                nfft_flags = ('PRE_PHI_HUT',
                              'PRE_PSI',
                              'FFTW_INIT',
                              'FFT_OUT_OF_PLACE',
                              'FFTW_ESTIMATE',
                              'FFTW_DESTROY_INPUT',)

        if x is None:
            nfft_flags = nfft_flags + ('MALLOC_X',)

        if f is None:
            nfft_flags = nfft_flags + ('MALLOC_F',)

        if f_hat is None:
            nfft_flags = nfft_flags + ('MALLOC_F_HAT',)

        for each_flag in nfft_flags:
            try:
                _nfft_flags |= nfft_flags_dict[each_flag]
                flags_used.append(each_flag)
            except KeyError:
                try:
                    _fftw_flags |= fftw_flags_dict[each_flag]
                    flags_used.append(each_flag)
                except KeyError:
                    raise ValueError('Invalid flag: ' + '\'' +
                        each_flag + '\' is not a valid flag.')

        # intialize plan

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

        cdef nfft_plan *_plan = &self.__plan
        try:
            nfft_init_guru(_plan, _d, _N, _M_total, _n, _m,
                           _nfft_flags, _fftw_flags)
        except:
            raise MemoryError
        finally:
            free(_N)
            free(_n)

        self._d = self.__plan.d
        self._m = self.__plan.m
        self._M_total = self.__plan.M_total
        self._N_total = self.__plan.N_total
        self._N = self.__plan.N
        self._dtype = np.dtype(dtype) if dtype is not None else np.float64
        self._flags = tuple(flags_used)

        cdef np.npy_intp shape[1]
        shape[0] = self._d * self._M_total
        self._x = np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_FLOAT64, <void *>self.__plan.x)
        shape[0] = self._M_total
        self._f = np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_COMPLEX128, <void *>self.__plan.f)
        shape[0] = self._N_total
        self._f_hat = np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_COMPLEX128, <void *>self.__plan.f_hat)

    # here, just holds the documentation of the class constructor
    def __init__(self, N, M, n=None, m=12, dtype=None, flags=None,
                 *args, **kwargs):
        pass

    # where the C-related content of the class needs to be cleaned
    def __dealloc__(self):
        nfft_finalize(&self.__plan)

    cpdef precompute(self):
        nfft_precompute_one_psi(&self.__plan)

    cpdef trafo(self):
        nfft_trafo(&self.__plan)

    cpdef trafo_direct(self):
        nfft_trafo_direct(&self.__plan)

    cpdef adjoint(self):
        nfft_adjoint(&self.__plan)

    cpdef adjoint_direct(self):
        nfft_adjoint_direct(&self.__plan)

    def __get_f(self):
        return self._f.copy()

    def __set_f(self, new_f):
        if new_f is not None and new_f is not self._f:
            if (<object>new_f).size != self._f.size:
                raise ValueError("Incompatible input")
            self._f[:] = new_f.ravel()[:]

    f = property(__get_f, __set_f)

    def __get_f_hat(self):
        return self._f_hat.copy()

    def __set_f_hat(self, new_f_hat):
        if new_f_hat is not None and new_f_hat is not self._f_hat:
            if (<object>new_f_hat).size != self._f_hat.size:
                raise ValueError("Incompatible input")
            self._f_hat[:] = new_f_hat.ravel()[:]

    f_hat = property(__get_f_hat, __set_f_hat)

    def __get_x(self):
        return self._x.copy()

    def __set_x(self, new_x):
        if new_x is not None and new_x is not self._x:
            if (<object>new_x).size != self._x.size:
                raise ValueError("Incompatible input")
            self._x[:] = new_x.ravel()[:]

    x = property(__get_x, __set_x)

    def __get_d(self):
        return self._d

    d = property(__get_d)

    def __get_m(self):
        return self._m

    m = property(__get_m)

    def __get_M_total(self):
        return self._M_total

    M_total = property(__get_M_total)

    def __get_N_total(self):
        return self._N_total

    N_total = property(__get_N_total)

    def __get_N(self):
        N = []
        for d in range(self._d):
            N.append(self._N[d])
        return tuple(N)

    N = property(__get_N)

    def __get_dtype(self):
        return self._dtype

    dtype = property(__get_dtype)

    def __get_flags(self):
        return self._flags

    flags = property(__get_flags)
