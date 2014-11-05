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

from cnfft3 cimport (LANDWEBER, STEEPEST_DESCENT, CGNR, CGNE,
                     NORMS_FOR_LANDWEBER, PRECOMPUTE_WEIGHT,
                     PRECOMPUTE_DAMP)
from cnfft3 cimport solver_plan_complex, nfft_mv_plan_complex
from cnfft3 cimport (solver_init_advanced_complex, solver_finalize_complex,
                     solver_before_loop_complex, solver_loop_one_step_complex)
from cnfft3 cimport nfft_malloc, nfft_free
from copy import copy


# Python API
__all__ = ('solver_plan_proxy', 'solver_plan_flags')


# Dictionary mapping the solver plan flag names to their mask value
cdef dict _solver_plan_flags = {
    'LANDWEBER'             : LANDWEBER,
    'STEEPEST_DESCENT'      : STEEPEST_DESCENT,
    'CGNR'                  : CGNR,
    'CGNE'                  : CGNE,
    'NORMS_FOR_LANDWEBER'   : NORMS_FOR_LANDWEBER,
    'PRECOMPUTE_WEIGHT'     : PRECOMPUTE_WEIGHT,
    'PRECOMPUTE_DAMP'       : PRECOMPUTE_DAMP,
}
solver_plan_flags = copy(_solver_plan_flags)


cdef class solver_plan_proxy:
    cdef solver_plan_complex *plan
    cdef bint _is_initialized
    cdef object _w
    cdef object _w_hat
    cdef object _y
    cdef object _f_hat_iter

    def __cinit__(self):
        self.plan = NULL
        self._is_initialized = False

    def __dealloc__(self):
        if self._is_initialized:
            solver_finalize_complex(self.plan)
            nfft_free(self.plan)

    @staticmethod
    def init_advanced(cls, nfft_plan, unsigned int flags):
        cdef solver_plan_proxy self = cls()
        self.plan = <solver_plan_complex*> nfft_malloc(sizeof(solver_plan_complex))
        cdef nfft_mv_plan_complex *mv_plan = <nfft_mv_plan_complex*>nfft_plan.plan
        solver_init_advanced_complex(self.plan, mv_plan, flags)
        self._is_initialized = True

    def before_loop(self):
        if not self.is_initialized():
            raise RuntimeError("plan is not initialized")
        self._before_loop()

    def loop_one_step(self):
        if not self.is_initialized():
            raise RuntimeError("plan is not initialized")
        self._loop_one_step()

    cpdef bint is_initialized(self):
        return self._is_initialized

    cdef _before_loop(self):
        with nogil:
            solver_before_loop_complex(self.plan)

    cdef _loop_one_step(self):
        with nogil:
            solver_loop_one_step_complex(self.plan)