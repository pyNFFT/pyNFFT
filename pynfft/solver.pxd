# -*- coding: utf-8 -*-
#
# Copyright (C) 2013  Ghislain Vaillant
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
from numpy cimport npy_intp

ctypedef void *(*solver_generic_init)(void *nfft_plan, unsigned flags)
ctypedef void (*solver_generic_finalize)(void *plan)
ctypedef void (*solver_generic_before_loop)(void *plan)
ctypedef void (*solver_generic_loop_one_step)(void *plan)
ctypedef object (*solver_generic_get_w)(void *plan, npy_intp shape[])
ctypedef object (*solver_generic_get_w_hat)(void *plan, npy_intp shape[])
ctypedef object (*solver_generic_get_y)(void *plan, npy_intp shape[])
ctypedef object (*solver_generic_get_f_hat_iter)(void *plan, npy_intp shape[])
ctypedef object (*solver_generic_get_r_iter)(void *plan, npy_intp shape[])

cdef class Solver:
    cdef void *__plan
    cdef void *__nfft_plan
    cdef object _w
    cdef object _w_hat
    cdef object _y
    cdef object _f_hat_iter
    cdef object _r_iter
    cdef object _dtype
    cdef object _flags
    cpdef before_loop(self)
    cpdef loop_one_step(self)
    cdef solver_generic_init __solver_init
    cdef solver_generic_finalize __solver_finalize
    cdef solver_generic_before_loop __solver_before_loop
    cdef solver_generic_loop_one_step __solver_loop_one_step
    cdef solver_generic_get_w __solver_get_w
    cdef solver_generic_get_w_hat __solver_get_w_hat
    cdef solver_generic_get_y __solver_get_y
    cdef solver_generic_get_f_hat_iter __solver_get_f_hat_iter
    cdef solver_generic_get_r_iter __solver_get_r_iter

cdef object solver_flags_dict
