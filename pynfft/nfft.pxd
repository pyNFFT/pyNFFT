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

from cnfft3 cimport nfft_plan, nfftf_plan, nfftl_plan


cdef:
    object nfft_flags_dict
    object fftw_flags_dict
    object nfft_supported_flags_tuple


ctypedef union _nfft_plan:
    nfft_plan dbl
    nfftf_plan flt
    nfftl_plan ldbl


cdef class NFFT(object):
    cdef:
        _nfft_plan _plan
        int _dbl
        int _flt
        int _ldbl
        int _d
        int _M
        int _m
        object _N
        object _n
        object _dtype
        object _flags
        object _f
        object _f_hat
        object _x
        void _precompute(self)
        void _trafo(self)
        void _trafo_direct(self)
        void _adjoint(self)
        void _adjoint_direct(self)
