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

from cnfft3 cimport *
from mv_plan cimport *
import numpy
from numpy cimport PyArray_CopyInto
from numpy cimport NPY_FLOAT64, NPY_COMPLEX128

 
cdef void _mv_plan_double_trafo(void *plan) nogil:
    cdef nfft_mv_plan_double *this_plan = <nfft_mv_plan_double*>(plan)
    this_plan.mv_trafo(plan)

cdef void _mv_plan_double_adjoint(void *plan) nogil:
    cdef nfft_mv_plan_double *this_plan = <nfft_mv_plan_double*>(plan)
    this_plan.mv_adjoint(plan)

cdef void _mv_plan_complex_trafo(void *plan) nogil:
    cdef nfft_mv_plan_complex *this_plan = <nfft_mv_plan_complex*>(plan)
    this_plan.mv_trafo(plan)

cdef void _mv_plan_complex_adjoint(void *plan) nogil:
    cdef nfft_mv_plan_complex *this_plan = <nfft_mv_plan_complex*>(plan)
    this_plan.mv_adjoint(plan)


cdef dict _mv_plan_typenum_to_index = {
    NPY_FLOAT64: 0,
    NPY_COMPLEX128: 1,
}

cdef _plan_trafo_func _plan_trafo_func_list[2]
cdef void _build_plan_trafo_func_list():
    _plan_trafo_func_list[0] = <_plan_trafo_func>(&_mv_plan_double_trafo)
    _plan_trafo_func_list[1] = <_plan_trafo_func>(&_mv_plan_complex_trafo)

cdef _plan_adjoint_func _plan_adjoint_func_list[2]
cdef void _build_plan_adjoint_func_list():
    _plan_adjoint_func_list[0] = <_plan_adjoint_func>(&_mv_plan_double_adjoint)
    _plan_adjoint_func_list[1] = <_plan_adjoint_func>(&_mv_plan_complex_adjoint)


### Module initialization
# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
from numpy cimport import_array
import_array()

# Populate lists of function pointers
_build_plan_trafo_func_list()
_build_plan_adjoint_func_list()
###

cdef class mv_plan_proxy:
    """
    Base plan class.
    
    Implements the minimal interface shared by all plans of the NFFT library. 
    It holds the main direct and adjoint computation methods (trafo and 
    adjoint methods), the internal arrays used for computation (f_hat and f), 
    and their data type.

    This class is only meant to be used as a base class for deriving more 
    specific plans.
    """    

    def __cinit__(self, dtype, *args, **kwargs):
        # Define the appropriate real and complex data types for the interface 
        # arrays. Will be also used to select the right function pointers for 
        # the individual methods of the derived for transparent support of 
        # multi-precision.
        # For now, only double is supported.
        self._dtype = numpy.dtype(dtype)       
        idx = _mv_plan_typenum_to_index[self._dtype.num]
        # To be malloc'd / assigned by the derived plan
        self._plan = NULL
        self._is_initialized = False
        self._N_total = 0
        self._M_total = 0
        self._f_hat = None
        self._f = None
        self._plan_trafo = _plan_trafo_func_list[idx]
        self._plan_adjoint = _plan_adjoint_func_list[idx]

    def __init__(self, dtype, *args, **kwargs):
        """Instantiate a base plan.
        
        The base plan is only responsible for initializing the relevant 
        pointers to NULL and initialize the internal arrays common to all 
        plans (f_hat and f) to valid Numpy arrays with the right size and 
        dtype. The rest should be handled by the derived classes.         
        """
        pass              

    def __dealloc__(self):
        """Clear a base plan.
        
        Responsiblity for properly destroying the internal C-plan should be 
        handled by the derived class.
        """
        pass

    cpdef trafo(self):
        """Compute the forward NFFT on current plan."""
        if self._is_initialized:
            with nogil:
                self._plan_trafo(self._plan)
        else:
            raise RuntimeError("plan is not initialized")
        
    cpdef adjoint(self):
        """Compute the adjoint NFFT on current plan."""
        if self._is_initialized:
            with nogil:
                self._plan_adjoint(self._plan)
        else:
            raise RuntimeError("plan is not initialized")

    @property
    def dtype(self):
        return self._dtype

    @property
    def N_total(self):
        return self._N_total

    @property
    def M_total(self):
        return self._M_total
        
    property f_hat:
        def __get__(self):
            return self._f_hat
        def __set__(self, object value):
            if self._is_initialized:
                PyArray_CopyInto(self._f_hat, value)
        
    property f:
        def __get__(self):
            return self._f
        def __set__(self, object value):
            if self._is_initialized:
                PyArray_CopyInto(self._f, value)