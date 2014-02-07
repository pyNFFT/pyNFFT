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

from __future__ import division
import numpy
import unittest
from numpy import pi
from numpy.testing import assert_allclose
from pynfft import NFFT
from pynfft.nfft import fftw_flags, nfft_flags, nfft_supported_flags
from pynfft.util import vrand_unit_complex, vrand_shifted_unit_double


class Test_NFFT_init(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Test_NFFT_init, self).__init__(*args, **kwargs)
        M, N = 32, (12, 12)        
        self.x = numpy.ones(len(N) * M, dtype=numpy.float64)
        self.f = numpy.empty(M, dtype=numpy.complex128)
        self.f_hat = numpy.empty(N, dtype=numpy.complex128) 
        self.m = 6
        self.flags = ('PRE_PHI_HUT', 'FG_PSI', 'PRE_FG_PSI')

    def test_default_args(self):
        Nfft = NFFT(self.f, self.f_hat, self.x)
        default_m = 12
        default_flags = ('PRE_PHI_HUT', 'PRE_PSI')        
        self.assertEqual(Nfft.m, default_m)        
        for each_flag in default_flags:
            self.assertIn(each_flag, Nfft.flags)

    def test_user_specified_args(self):
        Nfft = NFFT(self.f, self.f_hat, self.x, m=self.m, flags=self.flags)
        self.assertEqual(Nfft.d, self.f_hat.ndim)
        self.assertEqual(Nfft.N, self.f_hat.shape)
        self.assertEqual(Nfft.N_total, self.f_hat.size)
        self.assertEqual(Nfft.M, self.f.size)
        self.assertEqual(Nfft.m, self.m)
        for each_flag in self.flags:
            self.assertIn(each_flag, Nfft.flags)
    

class Test_NFFT_runtime(unittest.TestCase):

    N = (32, 32)
    M = 1280

    def fdft(self, f_hat):
        N = self.N
        k = numpy.mgrid[slice(N[0]), slice(N[1])]
        k = k.reshape([2, -1]) - numpy.asarray(N).reshape([2, 1]) / 2
        x = self.x.reshape([-1, 2])
        F = numpy.exp(-2j * pi * numpy.dot(x, k))
        f_dft = numpy.dot(F, f_hat.ravel())
        return f_dft
        
    def idft(self, f):
        N = self.N
        k = numpy.mgrid[slice(N[0]), slice(N[1])]
        k = k.reshape([2, -1]) - numpy.asarray(N).reshape([2, 1]) / 2
        x = self.x.reshape([-1, 2])
        F = numpy.exp(-2j * pi * numpy.dot(x, k))
        f_hat_dft = numpy.dot(numpy.conjugate(F).T, f)
        return f_hat_dft        

    def generate_new_arrays(self, init_f=False, init_f_hat=False):
        f = numpy.empty(self.M, dtype=numpy.complex128)
        f_hat = numpy.empty(self.N, dtype=numpy.complex128)
        if init_f:
            vrand_unit_complex(f.ravel())
        if init_f_hat:
            vrand_unit_complex(f_hat.ravel())
        return f, f_hat

    def generate_nfft_plan(self):
        Nfft = NFFT(f=self.f, f_hat=self.f_hat, x=self.x, precompute=True)
        return Nfft

    def __init__(self, *args, **kwargs):
        super(Test_NFFT_runtime, self).__init__(*args, **kwargs)
        self.x = numpy.empty(self.M*len(self.N), dtype=numpy.float64)
        vrand_shifted_unit_double(self.x.ravel())
        self.f, self.f_hat = self.generate_new_arrays()

    def test_trafo(self):
        Nfft = self.generate_nfft_plan()
        vrand_unit_complex(self.f_hat.ravel())
        f = Nfft.forward(use_dft=False)
        assert_allclose(f, self.fdft(self.f_hat), rtol=1e-3)

    def test_trafo_direct(self):
        Nfft = self.generate_nfft_plan()
        vrand_unit_complex(self.f_hat.ravel())
        f = Nfft.forward(use_dft=True)
        assert_allclose(f, self.fdft(self.f_hat), rtol=1e-3)

    def test_adjoint(self):
        Nfft = self.generate_nfft_plan()
        vrand_unit_complex(self.f)
        f_hat = Nfft.adjoint(use_dft=False)
        assert_allclose(f_hat.ravel(), self.idft(self.f), rtol=1e-3)

    def test_adjoint_direct(self):
        Nfft = self.generate_nfft_plan()
        vrand_unit_complex(self.f)
        f_hat = Nfft.adjoint(use_dft=True)
        assert_allclose(f_hat.ravel(), self.idft(self.f), rtol=1e-3)


class Test_NFFT_errors(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Test_NFFT_errors, self).__init__(*args, **kwargs)

    def test_for_invalid_m(self):
        M, N = 20, 32
        f = numpy.empty(M, dtype=numpy.complex128)
        f_hat = numpy.empty(N, dtype=numpy.complex128)
        
        failing_m = (-1, 1+numpy.iinfo(numpy.int32).max)
        for some_m in failing_m:
            self.assertRaises(ValueError,
                              lambda: NFFT(f=f, f_hat=f_hat, m=some_m))

    def test_for_invalid_n(self):
        M, N = 20, 32
        f = numpy.empty(M, dtype=numpy.complex128)
        f_hat = numpy.empty(N, dtype=numpy.complex128)
        
        failing_n = (-1, 1+numpy.iinfo(numpy.int32).max)
        for some_n in failing_n:
            self.assertRaises(ValueError,
                              lambda: NFFT(f=f, f_hat=f_hat, n=(some_n,)))

    def test_for_invalid_x(self):
        M, N = 20, 32
        x = numpy.empty(M, dtype=numpy.float64)
        f = numpy.empty(M, dtype=numpy.complex128)
        f_hat = numpy.empty(N, dtype=numpy.complex128)
        # array must be contigous
        self.assertRaises(ValueError,
                          lambda: NFFT(f=f, f_hat=f_hat, x=x[::2]))
        # array must be of the right size
        self.assertRaises(ValueError,
                          lambda: NFFT(f=f, f_hat=f_hat, x=x[:-2]))
        # array must be of the right type
        self.assertRaises(ValueError,
                          lambda: NFFT(f=f, f_hat=f_hat,
                                       x=x.astype(numpy.float32)))

    def test_for_invalid_f(self):
        M, N = 20, 32
        x = numpy.empty(M, dtype=numpy.float64)
        f = numpy.empty(M, dtype=numpy.complex128)
        f_hat = numpy.empty(N, dtype=numpy.complex128)
        # array must be contigous
        self.assertRaises(ValueError,
                          lambda: NFFT(f=f[::2], f_hat=f_hat))
        # array must be of the right type
        self.assertRaises(ValueError,
                          lambda: NFFT(f=f.astype(numpy.complex64), f_hat=f_hat))

    def test_for_invalid_f_hat(self):
        M, N = 20, 32
        x = numpy.empty(M, dtype=numpy.float64)
        f = numpy.empty(M, dtype=numpy.complex128)
        f_hat = numpy.empty(N, dtype=numpy.complex128)
        # array must be contigous
        self.assertRaises(ValueError,
                          lambda: NFFT(f=f, f_hat=f_hat[::2]))
        # array must be of the right type
        self.assertRaises(ValueError,
                          lambda: NFFT(f=f, f_hat=f_hat.astype(numpy.complex64)))

    def test_for_invalid_flags(self):
        M, N = 20, 32
        x = numpy.empty(M, dtype=numpy.float64)
        f = numpy.empty(M, dtype=numpy.complex128)
        f_hat = numpy.empty(N, dtype=numpy.complex128)
        # non existing flags
        invalid_flags = ('PRE_PHI_HOT', 'PRE_FOOL_PSI', 'FG_RADIO_PSI')
        for flag in invalid_flags:
            self.assertRaises(ValueError,
                              lambda: NFFT(f=f, f_hat=f_hat, flags=(flag,)))
        # managed flags
        managed_flags = []
        managed_flags.append([flag for flag in nfft_flags.keys()
                              if flag not in nfft_supported_flags])
        managed_flags.append([flag for flag in fftw_flags.keys()
                              if flag not in nfft_supported_flags])
        for flag in managed_flags:
            self.assertRaises(ValueError,
                              lambda: NFFT(f=f, f_hat=f_hat, flags=(flag,)))
