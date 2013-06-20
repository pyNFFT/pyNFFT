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

# function pointers for data type abstraction
ctypedef void *(*nfft_generic_init)(int d, int *N, int M, int *n, int m,
                                    unsigned nfft_flags, unsigned fftw_flags)
ctypedef void (*nfft_generic_finalize)(void *_plan)
ctypedef void (*nfft_generic_precompute)(void *_plan) nogil
ctypedef void (*nfft_generic_trafo)(void *_plan) nogil
ctypedef void (*nfft_generic_trafo_direct)(void *_plan) nogil
ctypedef void (*nfft_generic_adjoint)(void *_plan) nogil
ctypedef void (*nfft_generic_adjoint_direct)(void *_plan) nogil
ctypedef void (*nfft_generic_set_x)(void *_plan, object x)
ctypedef void (*nfft_generic_set_f)(void *_plan, object f)
ctypedef void (*nfft_generic_set_f_hat)(void *_plan, object f_hat)

cdef class NFFT:
    cdef void *__plan
    cdef nfft_generic_init __nfft_init
    cdef nfft_generic_finalize __nfft_finalize
    cdef nfft_generic_precompute __nfft_precompute
    cdef nfft_generic_trafo __nfft_trafo
    cdef nfft_generic_trafo_direct __nfft_trafo_direct
    cdef nfft_generic_adjoint __nfft_adjoint
    cdef nfft_generic_adjoint_direct __nfft_adjoint_direct
    cdef nfft_generic_set_x __nfft_set_x
    cdef nfft_generic_set_f __nfft_set_f
    cdef nfft_generic_set_f_hat __nfft_set_f_hat
    cdef object _f
    cdef object _f_hat
    cdef object _x
    cdef int _d
    cdef int _m
    cdef int _M_total
    cdef int _N_total
    cdef object _N
    cdef object _dtype
    cdef object _flags
    cpdef precompute(self)
    cpdef trafo(self)
    cpdef trafo_direct(self)
    cpdef adjoint(self)
    cpdef adjoint_direct(self)

cdef object nfft_supported_flags_tuple
cdef object nfft_flags_dict
cdef object fftw_flags_dict
