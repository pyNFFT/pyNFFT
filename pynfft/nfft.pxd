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

from cnfft3 cimport nfft_plan

cdef class NFFT:
    cdef nfft_plan __plan
    cdef object _f
    cdef object _f_hat
    cdef object _x
    cdef int _d
    cdef int _m
    cdef int _M_total
    cdef int _N_total
    cdef int *_N
    cdef object _dtype
    cdef object _flags
    cpdef precompute(self)
    cpdef trafo(self)
    cpdef trafo_direct(self)
    cpdef adjoint(self)
    cpdef adjoint_direct(self)
