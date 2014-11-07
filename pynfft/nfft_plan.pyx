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

from nfft_plan cimport *

# For forward declarations of nfft plan and function prototypes
from cnfft3 cimport (nfft_plan, fftw_complex)
from cnfft3 cimport (nfft_malloc, nfft_finalize, nfft_free, nfft_check,
                     nfft_init_1d, nfft_init_2d, nfft_init_3d, nfft_init,
                     nfft_init_guru, nfft_trafo_direct, nfft_adjoint_direct,
                     nfft_trafo, nfft_adjoint, nfft_precompute_one_psi)
from cfftw3 cimport fftw_init_threads, fftw_cleanup, fftw_cleanup_threads

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

cdef void _nfft_plan_free(void *plan):
    nfft_free(<nfft_plan*> plan)

cdef void _nfft_init_1d(void *plan, int N, int M):
    nfft_init_1d(<nfft_plan*> plan, N, M)
    
cdef void _nfft_init_2d(void *plan, int N1, int N2, int M):
    nfft_init_2d(<nfft_plan*> plan, N1, N2, M)    

cdef void _nfft_init_3d(void *plan, int N1, int N2, int N3, int M):
    nfft_init_3d(<nfft_plan*> plan, N1, N2, N3, M)

cdef void _nfft_init(void *plan, int d, int *N, int M):
    nfft_init(<nfft_plan*> plan, d, N, M)

cdef void _nfft_init_guru(void *plan, int d, int *N, int M, int *n, int m,
                           unsigned int nfft_flags, unsigned int fftw_flags):
    nfft_init_guru(<nfft_plan*> plan, d, N, M, n, m, nfft_flags, fftw_flags)

cdef void _nfft_trafo_direct(void *plan) nogil:
    nfft_trafo_direct(<nfft_plan*> plan)

cdef void _nfft_adjoint_direct(void *plan) nogil:
    nfft_adjoint_direct(<nfft_plan*> plan)
    
cdef void _nfft_trafo(void *plan) nogil:
    nfft_trafo(<nfft_plan*> plan)

cdef void _nfft_adjoint(void *plan) nogil:
    nfft_adjoint(<nfft_plan*> plan)    

cdef void _nfft_precompute_one_psi(void *plan) nogil:
    nfft_precompute_one_psi(<nfft_plan*> plan)

cdef const char *_nfft_check(void *plan):
    return nfft_check(<nfft_plan*> plan)

cdef void _nfft_finalize(void *plan):
    nfft_finalize(<nfft_plan*> plan)

cdef int _nfft_plan_get_N_total(void *plan):
    return (<nfft_plan*> plan).N_total

cdef int _nfft_plan_get_M_total(void *plan):
    return (<nfft_plan*> plan).M_total

cdef int _nfft_plan_get_d(void *plan):
    return (<nfft_plan*> plan).d

cdef int *_nfft_plan_get_N(void *plan):
    return (<nfft_plan*> plan).N

cdef int *_nfft_plan_get_n(void *plan):
    return (<nfft_plan*> plan).n

cdef int _nfft_plan_get_m(void *plan):
    return (<nfft_plan*> plan).m

cdef unsigned int _nfft_plan_get_nfft_flags(void *plan):
    return (<nfft_plan*> plan).nfft_flags

cdef unsigned int _nfft_plan_get_fftw_flags(void *plan):
    return (<nfft_plan*> plan).fftw_flags

cdef void *_nfft_plan_get_f_hat(void *plan):
    cdef nfft_plan *this_plan = <nfft_plan*> plan
    return <void *>this_plan.f_hat

cdef void _nfft_plan_set_f_hat(void *plan, void *data):
    cdef nfft_plan *this_plan = <nfft_plan*> plan
    this_plan.f_hat = <fftw_complex*> data

cdef void *_nfft_plan_get_f(void *plan):
    cdef nfft_plan *this_plan = <nfft_plan*> plan
    return <void *>this_plan.f

cdef void _nfft_plan_set_f(void *plan, void *data):
    cdef nfft_plan *this_plan = <nfft_plan*> plan
    this_plan.f = <fftw_complex*> data

cdef void *_nfft_plan_get_x(void *plan):
    cdef nfft_plan *this_plan = <nfft_plan*> plan
    return <void *>this_plan.x

cdef void _nfft_plan_set_x(void *plan, void *data):
    cdef nfft_plan *this_plan = <nfft_plan*> plan
    this_plan.x = <double *> data

cdef class nfft_plan_proxy(mv_plan_proxy):
    
    def __cinit__(self):
        self._plan                  = NULL
        self._is_initialized        = False
        self._f_hat                 = None
        self._f                     = None
        self._x                     = None
        self._plan_get_N_total      = &_nfft_plan_get_N_total
        self._plan_get_M_total      = &_nfft_plan_get_M_total
        self._plan_get_f_hat        = &_nfft_plan_get_f_hat
        self._plan_set_f_hat        = &_nfft_plan_set_f_hat
        self._plan_get_f            = &_nfft_plan_get_f_hat
        self._plan_set_f            = &_nfft_plan_set_f_hat
        self._plan_get_d            = &_nfft_plan_get_d
        self._plan_get_N            = &_nfft_plan_get_N
        self._plan_get_n            = &_nfft_plan_get_n
        self._plan_get_m            = &_nfft_plan_get_m
        self._plan_get_nfft_flags   = &_nfft_plan_get_nfft_flags
        self._plan_get_fftw_flags   = &_nfft_plan_get_fftw_flags     
        self._plan_get_x            = &_nfft_plan_get_x
        self._plan_set_x            = &_nfft_plan_set_x   
        self._plan_malloc           = &_nfft_plan_malloc
        self._plan_free             = &_nfft_plan_free
        self._plan_trafo            = &_nfft_trafo
        self._plan_adjoint          = &_nfft_adjoint
        self._plan_finalize         = &_nfft_finalize
        self._plan_init_1d          = &_nfft_init_1d
        self._plan_init_2d          = &_nfft_init_2d
        self._plan_init_3d          = &_nfft_init_3d
        self._plan_init             = &_nfft_init
        self._plan_init_guru        = &_nfft_init_guru
        self._plan_trafo_direct     = &_nfft_trafo_direct
        self._plan_adjoint_direct   = &_nfft_adjoint_direct
        self._plan_precompute       = &_nfft_precompute_one_psi
        self._plan_check            = &_nfft_check
    
    @classmethod
    def init_1d(cls, int N, int M):
        cdef nfft_plan_proxy self = cls()
        self._plan = self._plan_malloc()
        self._plan_init_1d(self._plan, N, M)
        self._is_initialized = True
        return self
 
    @classmethod
    def init_2d(cls, int N1, int N2, int M):
        cdef nfft_plan_proxy self = cls()
        self._plan = self._plan_malloc()
        self._plan_init_2d(self._plan, N1, N2, M)
        self._is_initialized = True
        return self

    @classmethod
    def init_3d(cls, int N1, int N2, int N3, int M):
        cdef nfft_plan_proxy self = cls()
        self._plan = self._plan_malloc()
        self._plan_init_3d(self._plan, N1, N2, N3, M)
        self._is_initialized = True
        return self
   
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
    
    cpdef precompute_one_psi(self):
        self.precompute()

    cpdef check(self):
        cdef const char *c_errmsg
        cdef bytes py_errmsg
        if self.is_initialized:
            c_errmsg = self._plan_check(self._plan)
            if c_errmsg != NULL:
                py_errmsg = <bytes> c_errmsg
                raise RuntimeError(py_errmsg)
        else:
            raise RuntimeError("plan is not initialized")

    @property 
    def f_hat(self):
        if self._is_initialized:
            return self._f_hat

    @property 
    def f(self):
        if self._is_initialized:
            return self._f

    @property
    def d(self):
        if self._is_initialized:
            return self._plan_get_d(self._plan)

    @property
    def N(self):
        cdef int *N = NULL
        if self._is_initialized:
            N = self._plan_get_N(self._plan)
            return [N[i] for i in range(self.d)]

    @property
    def n(self):
        cdef int *n = NULL
        if self._is_initialized:
            n = self._plan_get_n(self._plan)
            return [n[i] for i in range(self.d)]

    @property
    def m(self):
        if self._is_initialized:
            return self._plan_get_m(self._plan)

    @property
    def nfft_flags(self):
        if self._is_initialized:
            return self._plan_get_nfft_flags(self._plan)

    @property
    def fftw_flags(self):
        if self._is_initialized:
            return self._plan_get_fftw_flags(self._plan)

    @property 
    def x(self):
        if self._is_initialized:
            return self._x

#
#
## Proxy class wrapping the nfft plan data structure and associated functions
#cdef class nfft_plan_proxy:
#    cdef nfft_plan *plan
#    cdef object _f_hat
#    cdef object _f
#    cdef object _x
#
#    cdef _nfft_plan_malloc_func         _plan_malloc
#    cdef _nfft_init_1d_func             _plan_init_1d
#    cdef _nfft_init_2d_func             _plan_init_2d
#    cdef _nfft_init_3d_func             _plan_init_3d
#    cdef _nfft_init_func                _plan_init
#    cdef _nfft_init_guru_func           _plan_init_guru
#    cdef _nfft_trafo_direct_func        _plan_trafo_direct
#    cdef _nfft_adjoint_direct_func      _plan_adjoint_direct
#    cdef _nfft_trafo_func               _plan_trafo
#    cdef _nfft_adjoint_func             _plan_adjoint
#    cdef _nfft_precompute_one_psi_func  _plan_precompute_one_psi
#    cdef _nfft_check_func               _plan_check
#    cdef _nfft_finalize_func            _plan_finalize
#    cdef _nfft_plan_free_func           _plan_free
#
#    def __cinit__(self):
#        self._plan = NULL
#        self._f_hat = None
#        self._f = None
#        self._plan_malloc = _nfft_plan_malloc
#        self._plan_init_1d = _nfft_init_1d
#        self._plan_init_2d = _nfft_init_2d
#        self._plan_init_3d = _nfft_init_3d
#        self._plan_init = _nfft_init
#        self._plan_init_guru = _nfft_init_guru
#        self._plan_trafo_direct = _nfft_trafo_direct
#        self._plan_adjoint_direct = _nfft_adjoint_direct
#        self._plan_trafo = _nfft_trafo
#        self._plan_adjoint = _nfft_adjoint
#        self._plan_precompute_one_psi = _nfft_precompute_one_psi
#        self._plan_check = _nfft_check
#        self._plan_finalize = _nfft_finalize 
#        self._plan_free = _nfft_plan_free
#        self._plan_set_f_hat = _nfft_plan_set_f_hat
#        self._plan_set_f = _nfft_plan_set_f
#        self._plan_set_x = _nfft_plan_set_x
#
#    def __dealloc__(self):
#        if self.is_initialized():
#            self._plan_finalize(self.plan)
#            self._plan_free(self.plan)
#
#    @classmethod
#    def init_1d(cls, int N, int M):
#        cdef nfft_plan_proxy self = cls()
#        self._plan_init_1d(self.plan, N, M)
#        self._initialize_arrays()
#        return self
#
#    @classmethod
#    def init_2d(cls, int N1, int N2, int M):
#        cdef nfft_plan_proxy self = cls()
#        self._plan_init_2d(self.plan, N1, N2, M)
#        self._initialize_arrays()
#        return self
#
#    @classmethod
#    def init_3d(cls, int N1, int N2, int N3, int M):
#        cdef nfft_plan_proxy self = cls()
#        self._plan_init_3d(self.plan, N1, N2, N3, M)
#        self._initialize_arrays()
#        return self
#
#    @classmethod
#    def init(cls, int d, object N not None, int M):
#        cdef nfft_plan_proxy self = cls()
#        cdef int *N_ptr = NULL
#        if len(N) != d:
#            return None
#        N_ptr = <int*> nfft_malloc(d*sizeof(int))
#        for t in range(d):
#            N_ptr[t] = N[t]
#        self._plan_init(self.plan, d, N_ptr, M)
#        nfft_free(N_ptr)
#        self._initialize_arrays()
#        return self
#
#    @classmethod
#    def init_guru(cls, int d, object N not None, int M, object n not None,
#                  int m, int nfft_flags, int fftw_flags):
#        cdef nfft_plan_proxy self = cls()
#        cdef int *N_ptr = NULL
#        cdef int *n_ptr = NULL
#        cdef unsigned int nfft_flags_uint=0, fftw_flags_uint=0
#        if len(N) != d:
#            return None
#        if len(n) != d:
#            return None
#        N_ptr = <int*> nfft_malloc(d*sizeof(int))
#        n_ptr = <int*> nfft_malloc(d*sizeof(int))
#        for t in range(d):
#            N_ptr[t] = N[t]
#            n_ptr[t] = n[t]
#        self._plan_init_guru(self.plan, d, N_ptr, M, n_ptr, m, nfft_flags,
#                             fftw_flags)
#        nfft_free(N_ptr)
#        nfft_free(n_ptr)
#        self._initialize_arrays()
#        return self
#
#    cpdef trafo_direct(self):
#        if self.is_initialized:
#            with nogil:
#                self._plan_trafo_direct(self._plan)
#        else:
#            raise RuntimeError("plan is not initialized")
#
#    cpdef adjoint_direct(self):
#        if self.is_initialized:
#            with nogil:
#                self._plan_adjoint_direct(self._plan)
#        else:
#            raise RuntimeError("plan is not initialized")
#
#    cpdef trafo(self):
#        if self.is_initialized:
#            with nogil:
#                self._plan_trafo(self._plan)
#        else:
#            raise RuntimeError("plan is not initialized"))
#
#    cpdef adjoint(self):
#        if self.is_initialized:
#            with nogil:
#                self._plan_adjoint(self._plan)
#        else:
#            raise RuntimeError("plan is not initialized")
#
#    cpdef precompute_one_psi(self):
#        if self.is_initialized:
#            with nogil:
#                self._plan_precompute_one_psi(self._plan)
#        else:
#            raise RuntimeError("plan is not initialized")
#
#    cpdef check(self):
#        cdef const char *c_errmsg
#        cdef bytes py_errmsg
#        if self.is_initialized:
#            c_errmsg = self._plan_check(self._plan)
#            if c_errmsg != NULL:
#                py_errmsg = <bytes> c_errmsg
#                raise RuntimeError(py_errmsg)
#        else:
#            raise RuntimeError("plan is not initialized")
#
#    cdef bint _is_initialized(self):
#        return (self._plan != NULL)
#
#    cdef void _initialize_arrays(self):
#        pass
#
#    cdef void _update_arrays(self):
#        pass
#    
#    cdef void _connect_arrays(self):
#        self._plan_set_f_hat(self._plan, self._f_hat)
#        self._plan_set_f(self._plan, self._f)
#        self._plan_set_x(self._plan, self._x)        
#
#    @property
#    def is_initialized(self):
#        return self._is_initialized()
#
#    @property
#    def N_total(self):
#        return self._plan_get_N_total(self._plan)
#
#    @property
#    def M_total(self):
#        return self._plan_get_M_total(self._plan)
#
#    @property 
#    def f_hat(self):
#        return self._f_hat
#
#    @property 
#    def f(self):
#        return self._f
#
#    @property
#    def d(self):
#        return self._plan_get_d(self._plan)
#
#    @property
#    def N(self):
#        cdef int *N = self._plan_get_N(self._plan)
#        return [N[i] for i in range(self.d)]
#
#    @property
#    def n(self):
#        cdef int *n = self._plan_get_n(self._plan)
#        return [self.plan.n[i] for i in range(self.d)]
#
#    @property
#    def m(self):
#        return self._plan_get_d(self._plan)
#
#    @property
#    def nfft_flags(self):
#        return self._plan_get_nfft_flags(self._plan)
#
#    @property
#    def fftw_flags(self):
#        return self._plan_get_fftw_flags(self._plan)
#
#    @property 
#    def x(self):
#        return self._x
