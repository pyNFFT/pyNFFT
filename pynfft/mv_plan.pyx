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

from mv_plan cimport *
from cnfft3 cimport *
import numpy
from numpy cimport PyArray_CopyInto

### Module initialization
# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
from numpy cimport import_array
import_array()

# Expose lists of function pointers
# Typenum to index conversion table
cdef dict _mv_plan_typenum_to_index = {
    NPY_FLOAT64: 0,
    NPY_COMPLEX128: 1,
}
# - trafo
cdef _mv_plan_trafo_func _mv_plan_trafo_func_list[2]
_build_plan_trafo_func_list(_mv_plan_trafo_func_list)
# - adjoint
cdef _mv_plan_adjoint_func _mv_plan_adjoint_func_list[2]
_build_plan_adjoint_func_list(_mv_plan_adjoint_func_list)
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
        handled by the derived class. The base class only clears the memory
        allocated for the internal pointer.
        """
        if self._is_initialized:
            nfft_free(self._plan)

    cpdef check(self):
        if not self._is_initialized:        
            raise RuntimeError("plan is not initialized")

    cpdef trafo(self):
        """Compute the forward NFFT on current plan."""
        self.check()
        with nogil:
            self._plan_trafo(self._plan)
        
    cpdef adjoint(self):
        """Compute the adjoint NFFT on current plan."""
        self.check()
        with nogil:
            self._plan_adjoint(self._plan)

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
            if value is not None:
                PyArray_CopyInto(self._f_hat, value)

    property f:
        def __get__(self):
            return self._f
        def __set__(self, object value):
            if value is not None:
                PyArray_CopyInto(self._f, value)