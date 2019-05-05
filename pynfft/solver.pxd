# -*- coding: utf-8 -*-
#
# Copyright (C) 2013-2014  Ghislain Vaillant
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

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from cnfft3 cimport solver_plan_complex, solverf_plan_complex, solverl_plan_complex
from .nfft cimport NFFT

cdef object solver_flags_dict

cdef class Solver(object):
    cdef:
        solverf_plan_complex _plan_flt
        solver_plan_complex _plan_dbl
        solverl_plan_complex _plan_ldbl
        object _w
        object _w_hat
        object _y
        object _f_hat_iter
        object _r_iter
        object _z_hat_iter
        object _p_hat_iter
        object _v_iter
        object _dtype
        object _flags
        void _before_loop(self)
        void _loop_one_step(self)

