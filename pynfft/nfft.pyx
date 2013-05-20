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
#
# Ghislain Vaillant
# ghislain.vallant@kcl.ac.uk

import numpy as np
cimport numpy as np
from libc cimport limits
from cnfft3 cimport *


cdef object nfft_flags_dict
nfft_flags_dict = {
    'PRE_PHI_HUT':PRE_PHI_HUT,
    'FG_PSI':FG_PSI,
    'PRE_LIN_PSI':PRE_LIN_PSI,
    'PRE_FG_PSI':PRE_FG_PSI,
    'PRE_PSI':PRE_PSI,
    'PRE_FULL_PSI':PRE_FULL_PSI,
    'FFT_OUT_OF_PLACE':FFT_OUT_OF_PLACE,
    'FFTW_INIT':FFTW_INIT,
    'NFFT_SORT_NODES':NFFT_SORT_NODES,
    'NFFT_OMP_BLOCKWISE_ADJOINT':NFFT_OMP_BLOCKWISE_ADJOINT,
    'PRE_ONE_PSI':PRE_ONE_PSI,
    }
_nfft_flags_dict = nfft_flags_dict.copy()

cdef object fftw_flags_dict
fftw_flags_dict = {
    'FFTW_ESTIMATE':FFTW_ESTIMATE,
    'FFTW_DESTROY_INPUT':FFTW_DESTROY_INPUT,
}
_fftw_flags_dict = fftw_flags_dict.copy()


cdef class NFFT:

    # where the C-related content of the class is being initialized
    def __cinit__(self):
        pass

    # here, just holds the documentation of the class constructor
    def __init__(self):
        pass

    # where the C-related content of the class needs to be cleaned
    def __dealloc__(self):
        pass

    cpdef precompute(self):
        pass

    cpdef trafo(self):
        pass

    cpdef trafo_direct(self):
        pass

    cpdef adjoint(self):
        pass

    cpdef adjoint_direct(self):
        pass

    def __get_f(self):
        pass

    def __set_f(self, new_f):
        pass

    f = property(__get_f, __set_f)

    def __get_f_hat(self):
        pass

    def __set_f_hat(self, new_f_hat):
        pass

    f_hat = property(__get_f_hat, __set_f_hat)

    def __get_x(self):
        pass

    def __set_x(self, new_x):
        pass

    x = property(__get_x, __set_x)

    def __get_d(self):
        pass

    d = property(__get_d)

    def __get_m(self):
        pass

    m = property(__get_m)

    def __get_M_total(self):
        pass

    M_total = property(__get_M_total)

    def __get_N_total(self):
        pass

    N_total = property(__get_N_total)

    def __get_N(self):
        pass

    N = property(__get_N)

    def __get_dtype(self):
        pass

    dtype = property(__get_dtype)

    def __get_flags(self):
        pass

    flags = property(__get_flags)
