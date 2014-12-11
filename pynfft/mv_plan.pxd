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

from cnfft3 cimport (nfft_mv_plan_double, nfft_mv_plan_complex)
from numpy cimport NPY_FLOAT64, NPY_COMPLEX128

# More consistent aliases
ctypedef nfft_mv_plan_double mv_plan_double
ctypedef nfft_mv_plan_complex mv_plan_cdouble

# Function pointers defined for all plans
ctypedef void (*_mv_plan_trafo_func) (void *) nogil
ctypedef void (*_mv_plan_adjoint_func) (void *) nogil

# - trafo
cdef inline void _mv_plan_double_trafo(void *plan) nogil:
    cdef mv_plan_double *this_plan = <mv_plan_double*>(plan)
    this_plan.mv_trafo(plan)

cdef inline void _mv_plan_complex_trafo(void *plan) nogil:
    cdef mv_plan_cdouble *this_plan = <mv_plan_cdouble*>(plan)
    this_plan.mv_trafo(plan)

cdef inline void _build_plan_trafo_func_list(_mv_plan_trafo_func func_list[2]):
    func_list[0] = <_mv_plan_trafo_func>(&_mv_plan_double_trafo)
    func_list[1] = <_mv_plan_trafo_func>(&_mv_plan_complex_trafo)

# - adjoint
cdef inline void _mv_plan_double_adjoint(void *plan) nogil:
    cdef mv_plan_double *this_plan = <mv_plan_double*>(plan)
    this_plan.mv_adjoint(plan)

cdef inline void _mv_plan_complex_adjoint(void *plan) nogil:
    cdef mv_plan_cdouble *this_plan = <mv_plan_cdouble*>(plan)
    this_plan.mv_adjoint(plan)

cdef inline void _build_plan_adjoint_func_list(_mv_plan_adjoint_func func_list[2]):
    func_list[0] = <_mv_plan_adjoint_func>(&_mv_plan_double_adjoint)
    func_list[1] = <_mv_plan_adjoint_func>(&_mv_plan_complex_adjoint)

# Base plan class
cdef class mv_plan_proxy:
    cdef void *_plan
    cdef bint _is_initialized
    cdef object _dtype
    cdef int _N_total
    cdef int _M_total
    cdef object _f_hat
    cdef object _f
    cdef _mv_plan_trafo_func _plan_trafo
    cdef _mv_plan_adjoint_func _plan_adjoint
    cpdef initialize_arrays(self)
    cpdef update_arrays(self, object f_hat, object f)
    cpdef connect_arrays(self)
    cpdef check_if_initialized(self)
    cpdef trafo(self)
    cpdef adjoint(self)