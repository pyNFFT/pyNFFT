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

from cnfft3 cimport *

cdef object nfft_flags_dict

cdef object fftw_flags_dict
fftw_flags_dict = {
    'FFTW_ESTIMATE':FFTW_ESTIMATE,
    'FFTW_DESTROY_INPUT':FFTW_DESTROY_INPUT,
    }

cdef object nfft_supported_flags_tuple
nfft_supported_flags_tuple = (
    'PRE_PHI_HUT',
    'FG_PSI',
    'PRE_LIN_PSI',
    'PRE_FG_PSI',
    'PRE_PSI',
    'PRE_FULL_PSI',
    )


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
    cdef void execute_adjoint(self)
    cdef void execute_adjoint_direct(self)
    cdef void execute_precomputation(self)
    cdef void execute_trafo(self)
    cdef void execute_trafo_direct(self)
