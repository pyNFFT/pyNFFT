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

# Explicit aliases
ctypedef nfft_mv_plan_double mv_plan_real_double
ctypedef nfft_mv_plan_complex mv_plan_complex_double

# Generic function pointers defined for all plans
ctypedef void (*mv_plan_generic_trafo) (void *) nogil
ctypedef void (*mv_plan_generic_adjoint) (void *) nogil


# Real plan
# - trafo
# -- double precision


# Complex plan
# -



# Double precision
# - Real plans
# - trafo
cdef inline void mv_plan_real_double_trafo(void *plan) nogil:
    cdef mv_plan_real_double *this_plan = <mv_plan_real_double*>(plan)
    this_plan.mv_trafo(plan)
# - adjoint
cdef inline void mv_plan_real_double_adjoint(void *plan) nogil:
    cdef mv_plan_real_double *this_plan = <mv_plan_real_double*>(plan)
    this_plan.mv_trafo(plan)
# - Complex plans
# - trafo
cdef inline void mv_plan_complex_double_trafo(void *plan) nogil:
    cdef mv_plan_real_double *this_plan = <mv_plan_real_double*>(plan)
    this_plan.mv_adjoint(plan)
# - adjoint
cdef inline void mv_plan_complex_double_adjoint(void *plan) nogil:
    cdef mv_plan_complex_double *this_plan = <mv_plan_complex_double*>(plan)
    this_plan.mv_adjoint(plan)

# 
cdef inline void _build_plan_trafo_func_list(_mv_plan_trafo_func func_list[2]):
    func_list[0] = <_mv_plan_trafo_func>(&_mv_plan_double_trafo)
    func_list[1] = <_mv_plan_trafo_func>(&_mv_plan_complex_trafo)

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
    cdef mv_plan_generic_trafo      _plan_trafo
    cdef mv_plan_generic_adjoint    _plan_adjoint
    cpdef check(self)
    cpdef trafo(self)
    cpdef adjoint(self)