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

def voronoi_weights_1d (object[np.float64_t, mode='c'] w not None,
                        object[np.float64_t, mode='c'] x not None):
    '''
    Utilitary function for computing density compensation weights from 1D knots.

    :param w: pre-allocated array
    :type w: ndarray <float64>
    :param x: ordered 1D knots
    :type x: ndarray <float64>
    '''
    if x.size != w.size:
        raise ValueError('Incompatible size between weights and nodes \
                         (%d, %d)'%(w.size, x.size))
    nfft_voronoi_weights_1d(<double *>&w[0], <double *>&x[0], w.size)

def voronoi_weights_S2 (object[np.float64_t, mode='c'] w not None,
                        object[np.float64_t, mode='c'] xi not None):
    '''
    Utilitary function for computing density compensation weights from knots
    located on the surface of a sphere.

    Useful for reconstruction of 3D radial data.

    :param w: pre-allocated array
    :type w: ndarray <float64>
    :param xi: angular locations (2D) on the unit sphere
    :type xi: ndarray <float64>
    '''
    if xi.size != 2 * w.size:
        raise ValueError('Incompatible size between weights and nodes \
                         (%d, %d)'%(w.size, xi.size))
    nfft_voronoi_weights_S2(<double *>&w[0], <double *>&xi[0], w.size)
