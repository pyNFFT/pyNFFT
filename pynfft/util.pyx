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

import numpy as np
cimport numpy as np
from cnfft3util cimport *


# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

def vrand_unit_complex(np.ndarray x not None):
    '''
    Utilitary function for initializing a vector of knots to random
    values within the range [-0.5, 0.5).

    Used for testing :attr:`pynfft.NFFT.f` and
    :attr:`pynfft.NFFT.f_hat`.

    :param x: pre-allocated array
    :type x: ndarray <complex64, complex128 or complex256>
    '''
    cdef np.uint8_t[:] buf
    buf = x.ravel().view(np.uint8)

    if x.dtype == np.complex64:
        nfftf_vrand_unit_complex(<fftwf_complex *>&buf[0], x.size)
    elif x.dtype == np.complex128:
        nfft_vrand_unit_complex(<fftw_complex *>&buf[0], x.size)
    elif x.dtype == np.complex256:
        nfftl_vrand_unit_complex(<fftwl_complex *>&buf[0], x.size)
    else:
        raise ValueError("bad dtype {}".format(x.dtype))

def vrand_shifted_unit_double (np.ndarray x not None):
    '''
    Utilitary function for initializing a vector of data to random
    complex values within the range [0, 1).

    Used for testing :attr:`pynfft.NFFT.x`.

    :param x: pre-allocated array
    :type x: ndarray <float64>
    '''
    cdef np.uint8_t[:] buf
    buf = x.ravel().view(np.uint8)

    if x.dtype == np.float32:
        nfftf_vrand_shifted_unit_double(<float *>&buf[0], x.size)
    elif x.dtype == np.float64:
        nfft_vrand_shifted_unit_double(<double *>&buf[0], x.size)
    elif x.dtype == np.float128:
        nfftl_vrand_shifted_unit_double(<long double *>&buf[0], x.size)
    else:
        raise ValueError("bad dtype {}".format(x.dtype))
