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

# cython: nonecheck=True

print("1")

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from cnfft3 cimport *

print("2")

# Initialize numpy module
# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
cnp.import_array()

print("3")

cdef extern from 'Python.h':
    int Py_AtExit(void (*callback)())

print("4")

# Initialize FFTW threads
fftw_init_threads()
fftwf_init_threads()
fftwl_init_threads()

print("5")

# Register cleanup callbacks
cdef void _cleanup():
    fftw_cleanup()
    fftw_cleanup_threads()
    fftwf_cleanup()
    fftwf_cleanup_threads()
    fftwl_cleanup()
    fftwl_cleanup_threads()

Py_AtExit(_cleanup)

print("6")

# Mappings from strings to binary flags from nfft3.h and fftw3.h
NFFT_FLAGS = {
    'PRE_PHI_HUT': PRE_PHI_HUT,
    'FG_PSI': FG_PSI,
    'PRE_LIN_PSI': PRE_LIN_PSI,
    'PRE_FG_PSI': PRE_FG_PSI,
    'PRE_PSI': PRE_PSI,
    'PRE_FULL_PSI': PRE_FULL_PSI,
    'MALLOC_X': MALLOC_X,
    'MALLOC_F_HAT': MALLOC_F_HAT,
    'MALLOC_F': MALLOC_F,
    'FFT_OUT_OF_PLACE': FFT_OUT_OF_PLACE,
    'FFTW_INIT': FFTW_INIT,
    'NFFT_SORT_NODES': NFFT_SORT_NODES,
    'NFFT_OMP_BLOCKWISE_ADJOINT': NFFT_OMP_BLOCKWISE_ADJOINT,
    'PRE_ONE_PSI': PRE_ONE_PSI,
}
FFTW_FLAGS = {
    'FFTW_ESTIMATE': FFTW_ESTIMATE,
    'FFTW_DESTROY_INPUT': FFTW_DESTROY_INPUT,
}

print("7")


cdef class _NFFT:

    """Wrapper class for C NNFT plan structs."""

    def __cinit__(
        self,
        dtype,
        int d,
        int[:] N not None,
        int[:] M not None,
        int[:] n not None,
        int[:] m not None,
        flags,
    ):

        print("c1")

        # --- Prepare input --- #

        is_float = (dtype == np.complex64)
        is_double = (dtype == np.complex128)
        is_longdouble = (dtype == np.complex256)
        assert is_float or is_double or is_longdouble

        # Convert flags to binary representation for the C interface
        cdef:
            unsigned int nfft_flags
            unsigned int fftw_flags

        nfft_flags = 0
        fftw_flags = 0
        for flag in flags:
            if flag in NFFT_FLAGS:
                nfft_flags |= NFFT_FLAGS[flag]
            elif flag in FFTW_FLAGS:
                fftw_flags |= FFTW_FLAGS[flag]
            else:
                raise RuntimeError

        # --- Initialize plan --- #

        if is_float:
            nfftf_init_guru(
                &self._planf, d, &N[0], M[0], &n[0], m[0], nfft_flags, fftw_flags
            )
        elif is_double:
            nfft_init_guru(
                &self._plan, d, &N[0], M[0], &n[0], m[0], nfft_flags, fftw_flags
            )
        elif is_longdouble:
            nfftl_init_guru(
                &self._planl, d, &N[0], M[0], &n[0], m[0], nfft_flags, fftw_flags
            )
        else:
            raise RuntimeError

        # --- Create array views --- #

        # Convert int arrays to npy_intp
        cdef cnp.npy_intp *N_npy = <cnp.npy_intp *> malloc(d * sizeof(np.npy_intp))
        if N_npy == NULL:
            raise MemoryError
        for i in range(d):
            N_npy[i] = N[i]
        cdef cnp.npy_intp *n_npy = <cnp.npy_intp *> malloc(d * sizeof(np.npy_intp))
        if n_npy == NULL:
            raise MemoryError
        for i in range(d):
            n_npy[i] = n[i]
        cdef cnp.npy_intp M_npy[1]
        M_npy[0] = M[0]
        cdef cnp.npy_intp x_shp[2]
        x_shp[0] = M[0]
        x_shp[1] = d

        # View of f_hat
        if is_float:
            self.f_hat = cnp.PyArray_SimpleNewFromData(
                d, N_npy, cnp.NPY_COMPLEX64, <void *>(self._planf.f_hat)
            )
        elif is_double:
            self.f_hat = cnp.PyArray_SimpleNewFromData(
                d, N_npy, cnp.NPY_COMPLEX128, <void *>(self._plan.f_hat)
            )
        elif is_longdouble:
            self.f_hat = cnp.PyArray_SimpleNewFromData(
                d, N_npy, cnp.NPY_COMPLEX256, <void *>(self._planl.f_hat)
            )
        else:
            raise RuntimeError

        # View of f
        if is_float:
            self.f = cnp.PyArray_SimpleNewFromData(
                1, M_npy, cnp.NPY_COMPLEX64, <void *>(self._planf.f)
            )
        elif is_double:
            self.f = cnp.PyArray_SimpleNewFromData(
                1, M_npy, cnp.NPY_COMPLEX128, <void *>(self._plan.f)
            )
        elif is_longdouble:
            self.f = cnp.PyArray_SimpleNewFromData(
                1, M_npy, cnp.NPY_COMPLEX256, <void *>(self._planl.f)
            )
        else:
            raise RuntimeError

        # View of x
        if is_float:
            self.x = cnp.PyArray_SimpleNewFromData(
                2, x_shp, cnp.NPY_FLOAT32, <void *>(self._plan_flt.x)
            )
        elif is_double:
            self.x = cnp.PyArray_SimpleNewFromData(
                2, x_shp, cnp.NPY_FLOAT64, <void *>(self._plan_dbl.x)
            )
        elif is_longdouble:
            self.x = cnp.PyArray_SimpleNewFromData(
                2, x_shp, cnp.NPY_FLOAT128, <void *>(self._plan_ldbl.x)
            )
        else:
            raise RuntimeError

        free(N_npy)
        free(M_npy)

        # Store precision flags
        self._is_float = is_float
        self._is_double = is_double
        self._is_longdouble = is_longdouble

    def __init__(self):
        print("init")

    def __dealloc__(self):
        if self._is_float:
            nfftf_finalize(&self._planf)
        elif self._is_double:
            nfft_finalize(&self._plan)
        elif self._is_longdouble:
            nfftl_finalize(&self._planl)
        else:
            raise RuntimeError

    cpdef void _precompute(self) except *:
        if self._is_float:
            with nogil:
                nfftf_precompute_one_psi(&self._planf)
        elif self._is_double:
            with nogil:
                nfft_precompute_one_psi(&self._plan)
        elif self._is_longdouble:
            with nogil:
                nfftl_precompute_one_psi(&self._planl)
        else:
            raise RuntimeError

    cpdef void _trafo(self) except *:
        if self._is_float:
            with nogil:
                nfftf_trafo(&self._planf)
        elif self._is_double:
            with nogil:
                nfft_trafo(&self._plan)
        elif self._is_longdouble:
            with nogil:
                nfftl_trafo(&self._planl)
        else:
            raise RuntimeError

    cpdef void _trafo_direct(self) except *:
        if self._is_float:
            with nogil:
                nfftf_trafo_direct(&self._planf)
        elif self._is_double:
            with nogil:
                nfft_trafo_direct(&self._plan)
        elif self._is_longdouble:
            with nogil:
                nfftl_trafo_direct(&self._planl)
        else:
            raise RuntimeError

    cpdef void _adjoint(self) except *:
        if self._is_float:
            with nogil:
                nfftf_adjoint(&self._planf)
        elif self._is_double:
            with nogil:
                nfft_adjoint(&self._plan)
        elif self._is_longdouble:
            with nogil:
                nfftl_adjoint(&self._planl)
        else:
            raise RuntimeError

    cpdef void _adjoint_direct(self) except *:
        if self._is_float:
            with nogil:
                nfftf_adjoint_direct(&self._planf)
        elif self._is_double:
            with nogil:
                nfft_adjoint_direct(&self._plan)
        elif self._is_longdouble:
            with nogil:
                nfftl_adjoint_direct(&self._planl)
        else:
            raise RuntimeError
