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

from cnfft3 cimport fftw_complex, fftwf_complex, fftwl_complex

cdef extern from "nfft3.h":

    void nfft_vrand_unit_complex(fftw_complex *x, int n)
 	    # Inits a vector of random complex numbers in \
        # $[0,1]\times[0,1]{\rm i}$ .

    void nfft_vrand_shifted_unit_double(double *x, int n)
        # Inits a vector of random double numbers in $[-1/2,1/2]$ .

    void nfftf_vrand_unit_complex(fftwf_complex *x, int n)
    void nfftf_vrand_shifted_unit_double(float *x, int n)

    void nfftl_vrand_unit_complex (fftwl_complex *x, int n)
    void nfftl_vrand_shifted_unit_double(long double *x, int n)
