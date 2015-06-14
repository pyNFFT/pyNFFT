# -*- coding: utf-8 -*-
#
# Copyright (C) 2014-2015  Taco Cohen
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

from cnfft3 cimport nfsft_plan

cdef object nfsft_flags_dict
cdef object nfsft_supported_flags_tuple

cdef class NFSFT(object):
    cdef nfsft_plan _plan
    cdef int _d
    cdef int _M
    cdef int _m
    cdef object _N
    cdef object _dtype
    cdef object _nfsft_flags
    cdef object _nfft_flags
    cdef object _f   
    cdef object _f_hat
    cdef object _x
    cdef void _trafo(self)
    cdef void _trafo_direct(self)    
    cdef void _adjoint(self)
    cdef void _adjoint_direct(self)
    cdef int _spectral_index(self, int l, int m)
