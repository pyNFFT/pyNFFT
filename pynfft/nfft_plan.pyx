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

# public API
__all__ = ('nfft_plan_proxy', 'nfft_plan_flags', 'fftw_plan_flags')

from nfft_plan cimport *

# For forward declarations of nfft plan and function prototypes
from cnfft3 cimport (nfft_plan, fftw_complex)
from cnfft3 cimport (nfft_malloc, nfft_finalize, nfft_free, nfft_check,
                     nfft_init_guru, nfft_trafo_direct, nfft_adjoint_direct,
                     nfft_trafo, nfft_adjoint, nfft_precompute_one_psi)
from cnfft3 cimport fftw_init_threads, fftw_cleanup, fftw_cleanup_threads
from cnfft3 cimport (PRE_PHI_HUT, FG_PSI, PRE_LIN_PSI, PRE_FG_PSI, PRE_PSI,
                     PRE_FULL_PSI, MALLOC_X, MALLOC_F_HAT, MALLOC_F,
                     FFT_OUT_OF_PLACE, FFTW_INIT, NFFT_SORT_NODES,
                     NFFT_OMP_BLOCKWISE_ADJOINT, PRE_ONE_PSI, FFTW_ESTIMATE,
                     FFTW_DESTROY_INPUT)

# Import numpy C-API
from numpy cimport ndarray, npy_intp
from numpy cimport NPY_FLOAT64, NPY_COMPLEX128
from numpy cimport PyArray_New, PyArray_DATA, PyArray_CopyInto
from numpy cimport NPY_CARRAY, NPY_OWNDATA

# Expose flags to Python API
from copy import copy

# Dictionary mapping the NFFT plan flag names to their mask value
cdef dict nfft_plan_flags_dict = {
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
nfft_plan_flags = copy(nfft_plan_flags_dict)

# Dictionary mapping the FFTW plan flag names to their mask value
cdef dict fftw_plan_flags_dict = {
    'FFTW_ESTIMATE'                 : FFTW_ESTIMATE,
    'FFTW_DESTROY_INPUT'            : FFTW_DESTROY_INPUT,
    }
fftw_plan_flags = copy(fftw_plan_flags_dict)

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


cdef void *_nfft_plan_malloc():
    return nfft_malloc(sizeof(nfft_plan))

cdef void _nfft_plan_destroy(void *plan):
    nfft_finalize(<nfft_plan*> plan)
    nfft_free(plan)

cdef void _nfft_init_guru(void *plan, int d, int *N, int M, int *n, int m,
                          unsigned int nfft_flags, unsigned int fftw_flags):
    nfft_init_guru(<nfft_plan*> plan, d, N, M, n, m, nfft_flags, fftw_flags)

cdef void _nfft_trafo_direct(void *plan) nogil:
    nfft_trafo_direct(<nfft_plan*> plan)

cdef void _nfft_adjoint_direct(void *plan) nogil:
    nfft_adjoint_direct(<nfft_plan*> plan)

cdef void _nfft_precompute_one_psi(void *plan) nogil:
    nfft_precompute_one_psi(<nfft_plan*> plan)

cdef const char *_nfft_check(void *plan):
    return nfft_check(<nfft_plan*> plan)

#cdef void _nfft_plan_set_f_hat(void *plan, void *data):
#    cdef nfft_plan *this_plan = <nfft_plan*> plan
#    this_plan.f_hat = <fftw_complex*> data
#
#cdef void _nfft_plan_set_f(void *plan, void *data):
#    cdef nfft_plan *this_plan = <nfft_plan*> plan
#    this_plan.f = <fftw_complex*> data
#
#cdef void _nfft_plan_set_x(void *plan, void *data):
#    cdef nfft_plan *this_plan = <nfft_plan*> plan
#    this_plan.x = <double *> data

cdef class nfft_plan_proxy(mv_plan_proxy):
    
    def __cinit__(self, dtype='cdouble', *args, **kwargs):
        self._x                     = None
        self._plan_malloc           = &_nfft_plan_malloc
        self._plan_destroy          = &_nfft_plan_destroy
        self._plan_init_guru        = &_nfft_init_guru
        self._plan_trafo_direct     = &_nfft_trafo_direct
        self._plan_adjoint_direct   = &_nfft_adjoint_direct
        self._plan_precompute       = &_nfft_precompute_one_psi
        self._plan_check            = &_nfft_check

    cpdef init_1d(self, int N, int M):
        self.init_guru((N,), M, (2*N,), 6, 0, 0)
 
    cpdef init_2d(self, int N1, int N2, int M):
        self.init_guru((N1, N2), M, (2*N1, 2*N2), 6, 0, 0)

    cpdef init_3d(self, int N1, int N2, int N3, int M):
        self.init_guru((N1, N2), M, (2*N1, 2*N2), 6, 0, 0)
   
    cpdef init(self, object N, int M):
        self.init_guru(N, M, [2 * Nt for Nt in N], 6, 0, 0)
    
    cpdef init_guru(self, object N, int M, object n, int m,
                    unsigned int nfft_flags, unsigned int fftw_flags):
        if not self._is_initialized:
            self._is_initialized = True
        else:
            raise RuntimeError("plan is already initialized")

    cpdef trafo_direct(self):
        if self._is_initialized:    
            with nogil:
                self._plan_trafo_direct(self._plan)
        else:
            raise RuntimeError("plan is not initialized")
    
    cpdef adjoint_direct(self):
        if self._is_initialized:      
            with nogil:
                self._plan_adjoint_direct(self._plan)
        else:
            raise RuntimeError("plan is not initialized")
    
    cpdef precompute(self):
        if self._is_initialized:        
            with nogil:
                self._plan_precompute(self._plan)
        else:
            raise RuntimeError("plan is not initialized")

    cpdef check(self):
        cdef const char *c_errmsg
        cdef bytes py_errmsg
        if self._is_initialized:
            c_errmsg = self._plan_check(self._plan)
            if c_errmsg != NULL:
                py_errmsg = <bytes> c_errmsg
                raise RuntimeError(py_errmsg)
        else:
            raise RuntimeError("plan is not initialized")

    @property
    def d(self):
        return self._d

    @property
    def N(self):
        return self._N

    @property
    def n(self):
        return self._n

    @property
    def m(self):
        return self._m

    @property
    def nfft_flags(self):
        return self._nfft_flags

    @property
    def fftw_flags(self):
        return self._fftw_flags

    @property 
    def x(self):
        def __get__(self):
            return self._x
        def __set__(self, object value):
            if self._is_initialized:
                PyArray_CopyInto(self._x, value)