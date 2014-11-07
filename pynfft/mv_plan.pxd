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

# Function pointers to plan management functions common to all plans
ctypedef void *(*_plan_malloc_func)         ()
ctypedef void  (*_plan_free_func)           (void *)
ctypedef void  (*_plan_trafo_func)          (void *) nogil
ctypedef void  (*_plan_adjoint_func)        (void *) nogil
ctypedef void  (*_plan_finalize_func)       (void *)
ctypedef int   (*_plan_get_N_total_func)    (void *)
ctypedef int   (*_plan_get_M_total_func)    (void *)
ctypedef void *(*_plan_get_f_hat_func)      (void *)
ctypedef void  (*_plan_set_f_hat_func)      (void *, void *)
ctypedef void *(*_plan_get_f_func)          (void *)
ctypedef void  (*_plan_set_f_func)          (void *, void *)

# Base plan class
cdef class mv_plan_proxy:
    cdef void   *_plan
    cdef bint   _is_initialized
    cdef object _f_hat
    cdef object _f

    cdef _plan_get_N_total_func _plan_get_N_total
    cdef _plan_get_M_total_func _plan_get_M_total
    cdef _plan_get_f_hat_func   _plan_get_f_hat
    cdef _plan_set_f_hat_func   _plan_set_f_hat
    cdef _plan_get_f_func       _plan_get_f
    cdef _plan_set_f_func       _plan_set_f

    cdef _plan_malloc_func      _plan_malloc 
    cdef _plan_free_func        _plan_free
    cdef _plan_trafo_func       _plan_trafo
    cdef _plan_adjoint_func     _plan_adjoint
    cdef _plan_finalize_func    _plan_finalize

    cpdef trafo(self)
    cpdef adjoint(self)