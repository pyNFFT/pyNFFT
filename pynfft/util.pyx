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

def vrand_unit_complex (object[np.complex128_t, mode='c'] x not None):
    '''
    Utilitary function for initializing a vector of knots to random
    values within the range [-0.5, 0.5).

    Used for testing :attr:`pynfft.NFFT.x`.

    :param x: pre-allocated array
    :type x: ndarray <complex128>
    '''
    nfft_vrand_unit_complex(<fftw_complex *>&x[0], x.size)

def vrand_shifted_unit_double (object[np.float64_t, mode='c'] x not None):
    '''
    Utilitary function for initializing a vector of data to random
    complex values within the range [0, 1).

    Used for testing :attr:`pynfft.NFFT.f` and
    :attr:`pynfft.NFFT.f_hat`.

    :param x: pre-allocated array
    :type x: ndarray <float64>
    '''
    nfft_vrand_shifted_unit_double(<double *>&x[0], x.size)
