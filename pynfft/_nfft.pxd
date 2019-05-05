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

# cython: nonecheck=True

from cnfft3 cimport nfft_plan, nfftf_plan, nfftl_plan


cdef class _NFFT(object):
    cdef:
        nfftf_plan _planf
        nfft_plan _plan
        nfftl_plan _planl
        public object f
        public object f_hat
        public object x
        int _is_float
        int _is_double
        int _is_longdouble
        cpdef void _precompute(self) except *
        cpdef void _trafo(self) except *
        cpdef void _trafo_direct(self) except *
        cpdef void _adjoint(self) except *
        cpdef void _adjoint_direct(self) except *

