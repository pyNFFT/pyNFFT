# -*- coding: utf-8 -*-
#
# Copyright (c) 2013, 2014 Ghislain Antony Vaillant
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

__all__ = ('nfft_plan_proxy', 'nfft_plan_flags', 'fftw_plan_flags')

# For forward declarations of nfft plan and function prototypes
from cnfft3 cimport (nfft_plan, fftw_complex)
from cnfft3 cimport (nfft_malloc, nfft_finalize, nfft_free, nfft_check,
                     nfft_init_1d, nfft_init_2d, nfft_init_3d, nfft_init,
                     nfft_init_guru, nfft_trafo_direct, nfft_adjoint_direct,
                     nfft_trafo, nfft_adjoint, nfft_precompute_one_psi)
from cnfft3 cimport (PRE_PHI_HUT, FG_PSI, PRE_LIN_PSI, PRE_FG_PSI, PRE_PSI,
                     PRE_FULL_PSI, MALLOC_X, MALLOC_F_HAT, MALLOC_F,
                     FFT_OUT_OF_PLACE, FFTW_INIT, NFFT_SORT_NODES,
                     NFFT_OMP_BLOCKWISE_ADJOINT, PRE_ONE_PSI)
from cnfft3 cimport FFTW_ESTIMATE, FFTW_DESTROY_INPUT
from cnfft3 cimport fftw_init_threads, fftw_cleanup, fftw_cleanup_threads

# Import numpy C-API
from numpy cimport ndarray, npy_intp
from numpy cimport NPY_FLOAT64, NPY_COMPLEX128
from numpy cimport PyArray_New, PyArray_DATA, PyArray_CopyInto
from numpy cimport NPY_CARRAY, NPY_OWNDATA

# Copy
from copy import copy

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
from numpy cimport import_array
import_array()

# Necessary for usage of threaded FFTW
fftw_init_threads()

# Register cleanup callback to the Python module
cdef extern from *:
    int Py_AtExit(void (*callback)())

cdef void _cleanup():
    fftw_cleanup()
    fftw_cleanup_threads()

Py_AtExit(_cleanup)


# Dictionary mapping the NFFT plan flag names to their mask value
cdef dict _nfft_plan_flags = {
    'PRE_PHI_HUT'                   : PRE_PHI_HUT,
    'FG_PSI'                        : FG_PSI,
    'PRE_LIN_PSI'                   : PRE_LIN_PSI,
    'PRE_FG_PSI'                    : PRE_FG_PSI,
    'PRE_PSI'                       : PRE_PSI,
    'PRE_FULL_PSI'                  : PRE_FULL_PSI,
    'MALLOC_X'                      : MALLOC_X,
    'MALLOC_F_HAT'                  : MALLOC_F_HAT,
    'MALLOC_F'                      : MALLOC_F,
    'FFT_OUT_OF_PLACE'              : FFT_OUT_OF_PLACE,
    'FFTW_INIT'                     : FFTW_INIT,
    'NFFT_SORT_NODES'               : NFFT_SORT_NODES,
    'NFFT_OMP_BLOCKWISE_ADJOINT'    : NFFT_OMP_BLOCKWISE_ADJOINT,
    'PRE_ONE_PSI'                   : PRE_ONE_PSI,
    }
nfft_plan_flags = copy(_nfft_plan_flags)


# Dictionary mapping the FFTW plan flag names to their mask value
cdef dict _fftw_plan_flags = {
    'FFTW_ESTIMATE'                 : FFTW_ESTIMATE,
    'FFTW_DESTROY_INPUT'            : FFTW_DESTROY_INPUT,
    }
fftw_plan_flags = copy(_fftw_plan_flags)


# Proxy class wrapping the nfft plan data structure and associated functions
cdef class nfft_plan_proxy:
    cdef nfft_plan *plan
    cdef object _f_hat
    cdef object _f
    cdef object _x

    def __cinit__(self):
        self.plan = <nfft_plan*> nfft_malloc(sizeof(nfft_plan))

    def __dealloc__(self):
        if self._is_initialized():
            nfft_finalize(self.plan)
        nfft_free(self.plan)

    @classmethod
    def init_1d(cls, int N, int M):
        cdef nfft_plan_proxy self = cls()
        nfft_init_1d(self.plan, N, M)
        self._initialize_arrays()
        return self

    @classmethod
    def init_2d(cls, int N1, int N2, int M):
        cdef nfft_plan_proxy self = cls()
        nfft_init_2d(self.plan, N1, N2, M)
        self._initialize_arrays()
        return self

    @classmethod
    def init_3d(cls, int N1, int N2, int N3, int M):
        cdef nfft_plan_proxy self = cls()
        nfft_init_3d(self.plan, N1, N2, N3, M)
        self._initialize_arrays()
        return self

    @classmethod
    def init(cls, int d, object N not None, int M):
        cdef nfft_plan_proxy self = cls()
        cdef int *N_ptr = NULL
        if len(N) != d:
            return None
        N_ptr = <int*> nfft_malloc(d*sizeof(int))
        for t in range(d):
            N_ptr[t] = N[t]
        nfft_init(self.plan, d, N_ptr, M)
        nfft_free(N_ptr)
        self._initialize_arrays()
        return self

    @classmethod
    def init_guru(cls, int d, object N not None, int M, object n not None,
                  int m, int nfft_flags, int fftw_flags):
        cdef nfft_plan_proxy self = cls()
        cdef int *N_ptr = NULL
        cdef int *n_ptr = NULL
        cdef unsigned int nfft_flags_uint=0, fftw_flags_uint=0
        if len(N) != d:
            return None
        if len(n) != d:
            return None
        N_ptr = <int*> nfft_malloc(d*sizeof(int))
        n_ptr = <int*> nfft_malloc(d*sizeof(int))
        for t in range(d):
            N_ptr[t] = N[t]
            n_ptr[t] = n[t]
        nfft_init_guru(self.plan, d, N_ptr, M, n_ptr, m, nfft_flags,
                       fftw_flags)
        nfft_free(N_ptr)
        nfft_free(n_ptr)
        self._initialize_arrays()
        return self

    def trafo_direct(self):
        if not self._is_initialized():
            raise RuntimeError("plan is not initialized")
        self._trafo_direct()

    def adjoint_direct(self):
        if not self._is_initialized():
            raise RuntimeError("plan is not initialized")
        self._adjoint_direct()

    def trafo(self):
        if not self._is_initialized():
            raise RuntimeError("plan is not initialized")
        self._trafo()

    def adjoint(self):
        if not self._is_initialized():
            raise RuntimeError("plan is not initialized")
        self._adjoint()

    def precompute_one_psi(self):
        if not self._is_initialized():
            raise RuntimeError("plan is not initialized")
        self._precompute_one_psi()

    def check(self):
        if not self._is_initialized():
            raise RuntimeError("plan is not initialized")
        errmsg = self._check()
        if errmsg is not None:
            raise RuntimeError(errmsg)

    cdef void _trafo_direct(self):
        with nogil:
            nfft_trafo_direct(self.plan)

    cdef void _adjoint_direct(self):
        with nogil:
            nfft_adjoint_direct(self.plan)

    cdef void _trafo(self):
        with nogil:
            nfft_trafo(self.plan)

    cdef void _adjoint(self):
        with nogil:
            nfft_adjoint(self.plan)

    cdef void _precompute_one_psi(self):
        with nogil:
            nfft_precompute_one_psi(self.plan)

    cdef bytes _check(self):
        cdef const char *c_errmsg
        cdef bytes py_errmsg
        c_errmsg = nfft_check(self.plan)
        if c_errmsg != NULL:
            py_errmsg = <bytes> c_errmsg
            return py_errmsg
        else:
            return None

    cdef bint _has_malloc_f_hat(self):
        return (self.plan.nfft_flags and MALLOC_F_HAT)

    cdef bint _has_malloc_f(self):
        return (self.plan.nfft_flags and MALLOC_F)

    cdef bint _has_malloc_x(self):
        return (self.plan.nfft_flags and MALLOC_X)

    cdef bint _is_initialized(self):
        return (self.plan != NULL       and
                self._f_hat is not None and
                self._f is not None     and
                self._x is not None)

    cdef void _initialize_arrays(self):
        cdef npy_intp shape[1]
        cdef int flags
        if self._has_malloc_f_hat():
            shape[0] = self.N_total
            self._f_hat = PyArray_New(
                ndarray, 1, shape, NPY_COMPLEX128, NULL,
                <void *>self.plan.f_hat, sizeof(fftw_complex),
                NPY_CARRAY, None)
        else:
            self._f_hat = None
        if self._has_malloc_f():
            shape[0] = self.M_total
            self._f = PyArray_New(
                ndarray, 1, shape, NPY_COMPLEX128, NULL,
                <void *>self.plan.f, sizeof(fftw_complex),
                NPY_CARRAY, None)
        else:
            self._f = None
        if self._has_malloc_x():
            shape[0] = self.M_total * self.d
            self._x = PyArray_New(
                ndarray, 1, shape, NPY_FLOAT64, NULL,
                <void *>self.plan.x, sizeof(double),
                NPY_CARRAY, None)
        else:
            self._x = None

    @property
    def N_total(self):
        return self.plan.N_total

    @property
    def M_total(self):
        return self.plan.M_total

    property f_hat:
        def __get__(self):
            return self._f_hat
        def __set__(self, value):
            if self._has_malloc_f_hat():
                PyArray_CopyInto(self._f_hat, value)
            else:
                self._f_hat = value
                self.plan.f_hat = <fftw_complex *>PyArray_DATA(self._f_hat)

    property f:
        def __get__(self):
            return self._f
        def __set__(self, value):
            if self._has_malloc_f():
                PyArray_CopyInto(self._f, value)
            else:
                self._f = value
                self.plan.f = <fftw_complex *>PyArray_DATA(self._f)

    @property
    def d(self):
        return self.plan.d

    @property
    def N(self):
        return [self.plan.N[i] for i in range(self.d)]

    @property
    def n(self):
        return [self.plan.n[i] for i in range(self.d)]

    @property
    def m(self):
        return self.plan.m

    @property
    def nfft_flags(self):
        return self.plan.nfft_flags

    @property
    def fftw_flags(self):
        return self.plan.fftw_flags

    property x:
        def __get__(self):
            return self._x
        def __set__(self, value):
            if self._has_malloc_x():
                PyArray_CopyInto(self._x, value)
            else:
                self._x = value
                self.plan.x = <double *>PyArray_DATA(self._x)
