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
import numpy
cimport numpy


cdef class mv_plan:
    """
    Base plan class.
    
    Implements the minimal interface shared by all plans of the NFFT library. 
    It holds the main direct and ajoint computation methods (trafo and 
    adjoint methods), the internal arrays used for computation (f_hat and f), 
    and the real and complex data types compatible with the C-structs. In 
    addition, a check method is provided which is run before any call to the 
    computation methods.

    This class is only meant to be used as a base class for deriving more 
    specific plans. It is however *not* an abstract class. 
    """    

    def __cinit__(self, N_total, M_total, *args, **kwargs):
        # To be malloc'd by derived class to the wrapped plan
        self._plan = NULL
        # To be assigned to the relevant methods of the wrapped plan
        self._plan_malloc = NULL
        self._plan_finalize = NULL
        self._plan_check = NULL
        self._plan_trafo = NULL
        self._plan_adjoint = NULL
        # Define the appropriate real and complex data types for the interface 
        # arrays. Will be also used to select the right function pointers for 
        # the individual methods for transparent support of multi-precision.        
        # For now, only double is supported.
        real_dtype = numpy.dtype('double')
        cplx_dtype = numpy.result_type(real_dtype, 1j)
        self._real_dtype = real_dtype
        self._cplx_dtype = cplx_dtype
        # Store the dimensions of the interface arrays
        self._N_total = N_total
        self._M_total = M_total

    def __init__(self, N_total, M_total, *args, **kwargs):
        """Instantiate a base plan.
        
        The base plan is only responsible for initializing the relevant 
        pointers to NULL and initialize the internal arrays common to all 
        plans (f_hat and f) to valid Numpy arrays with the right size and 
        dtype. The rest should be handled by the derived classes.         
        """
        self._f_hat = numpy.empty(self.N_total, dtype=self.complex_dtype)
        self._f = numpy.empty(self.M_total, dtype=self.complex_dtype)                

    def __dealloc__(self):
        """Clear a base plan.
        
        Responsiblity of properly destroying the internal C-plan is handled to 
        the derived class.
        """
        pass

    cpdef check(self):
        """Check the state of a plan.

        Sanity checks for the internal plan structure. The bare minimum is to 
        to check whether the structure has been initialized.
        
        Additional checks for the wrapped plan structure may be performed by 
        assigning the internal plan_check function pointer to the relevant 
        c-function.
        """
        cdef bytes errmsg
        if self._plan == NULL:
            raise RuntimeError("plan is not initialized")
        else:
            if self._plan_check != NULL:
                errmsg = self._plan_check(self._plan)
                if errmsg is not None:
                    raise RuntimeError(errmsg)

    cpdef trafo(self):
        """Compute the forward NFFT on current plan.""" 
        self.check()
        with nogil:
            if self._plan_trafo != NULL:
                self._plan_trafo(self._plan)
        
    cpdef adjoint(self):
        """Compute the adjoint NFFT on current plan."""
        self.check()
        with nogil:
            if self._plan_adjoint != NULL:
                self._plan_adjoint(self._plan)

    @property
    def real_dtype(self):
        return self._real_dtype
        
    @property
    def complex_dtype(self):
        return self._cplx_dtype

    @property
    def M_total(self):
        return self._f_hat.size

    @property
    def N_total(self):
        return self._f.size
        
    property f_hat:
        def __get__(self): return self._f_hat
        def __set__(self, value):
            value = numpy.ascontigousarray(value, dtype=self.complex_dtype).reshape(self.N_total)
            self._f_hat = value
        
    property f:
        def __get__(self): return self._f
        def __set__(self, value):
            value = numpy.ascontigousarray(value, dtype=self.complex_dtype).reshape(self.M_total)
            self._f = value