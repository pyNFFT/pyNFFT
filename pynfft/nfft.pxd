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

from cnfft3 cimport nfft_plan

cdef object nfft_flags_dict
cdef object fftw_flags_dict
cdef object nfft_supported_flags_tuple

cdef class NFFT(object):
    cdef nfft_plan _plan
    cdef int _d
    cdef int _M
    cdef int _m
    cdef object _N
    cdef object _n
    cdef object _dtype
    cdef object _flags
    cdef object _f   
    cdef object _f_hat
    cdef object _x
    cdef void _precompute(self)
    cdef void _trafo(self)
    cdef void _trafo_direct(self)    
    cdef void _adjoint(self)
    cdef void _adjoint_direct(self)

