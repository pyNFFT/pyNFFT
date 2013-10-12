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
        Nfft = NFFT(self.x, self.f, self.f_hat)
        default_m = 12
        default_flags = ('PRE_PHI_HUT', 'PRE_PSI')        
        self.assertEqual(Nfft.m, default_m)        
        for each_flag in default_flags:
            self.assertIn(each_flag, Nfft.flags)

    def test_user_specified_args(self):
        Nfft = NFFT(self.x, self.f, self.f_hat, m=self.m, flags=self.flags)
        self.assertEqual(Nfft.d, self.f_hat.ndim)
        self.assertEqual(Nfft.N, self.f_hat.shape)
        self.assertEqual(Nfft.N_total, self.f_hat.size)
        self.assertEqual(Nfft.M, self.f.size)
        self.assertEqual(Nfft.m, self.m)
        for each_flag in self.flags:
            self.assertIn(each_flag, Nfft.flags)
    
    def test_precomputation_flag(self):
        Nfft = NFFT(self.x, self.f, self.f_hat, m=self.m, flags=self.flags,
                    precompute=False)
        self.assertFalse(Nfft.precomputed)
        Nfft.precompute()
        self.assertTrue(Nfft.precomputed)


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

    def __init__(self, *args, **kwargs):
        super(Test_NFFT_runtime, self).__init__(*args, **kwargs)
        self.x = numpy.empty(self.M*len(self.N), dtype=numpy.float64)
        vrand_shifted_unit_double(self.x.ravel())
        self.f, self.f_hat = self.generate_new_arrays()

    def test_trafo(self):
        Nfft = NFFT(self.x, self.f, self.f_hat, precompute=True)
        vrand_unit_complex(self.f_hat.ravel())
        Nfft.execute_trafo()
        assert_allclose(self.f, self.fdft(self.f_hat), rtol=1e-3)

    def test_trafo_newapi(self):
        Nfft = NFFT(self.x, self.f, self.f_hat, precompute=True)
        new_f, new_f_hat = self.generate_new_arrays(init_f_hat=True)
        Nfft.forward(f_hat=new_f_hat, use_dft=False)        
        assert_allclose(self.f, self.fdft(new_f_hat), rtol=1e-3)
        Nfft.forward(f=new_f, f_hat=new_f_hat, use_dft=False)     
        assert_allclose(new_f, self.fdft(new_f_hat), rtol=1e-3)

    def test_trafo_direct(self):
        Nfft = NFFT(self.x, self.f, self.f_hat, precompute=True)
        vrand_unit_complex(self.f_hat.ravel())
        Nfft.execute_trafo_direct()
        assert_allclose(self.f, self.fdft(self.f_hat), rtol=1e-3)

    def test_trafo_direct_newapi(self):
        Nfft = NFFT(self.x, self.f, self.f_hat, precompute=True)
        new_f, new_f_hat = self.generate_new_arrays(init_f_hat=True)
        Nfft.forward(f_hat=new_f_hat, use_dft=True)        
        assert_allclose(self.f, self.fdft(new_f_hat), rtol=1e-3)
        Nfft.forward(f=new_f, f_hat=new_f_hat, use_dft=True)     
        assert_allclose(new_f, self.fdft(new_f_hat), rtol=1e-3)

    def test_adjoint(self):
        Nfft = NFFT(self.x, self.f, self.f_hat, precompute=True)
        vrand_unit_complex(self.f.ravel())
        Nfft.execute_adjoint()
        assert_allclose(self.f_hat.ravel(), self.idft(self.f), rtol=1e-3)

    def test_adjoint_newapi(self):
        Nfft = NFFT(self.x, self.f, self.f_hat, precompute=True)
        new_f, new_f_hat = self.generate_new_arrays(init_f=True)
        Nfft.adjoint(f=new_f, use_dft=False)
        assert_allclose(self.f_hat.ravel(), self.idft(new_f), rtol=1e-3)
        Nfft.adjoint(f=new_f, f_hat=new_f_hat, use_dft=False)
        assert_allclose(new_f_hat.ravel(), self.idft(new_f), rtol=1e-3)

    def test_adjoint_direct(self):
        Nfft = NFFT(self.x, self.f, self.f_hat, precompute=True)
        vrand_unit_complex(self.f)
        Nfft.execute_adjoint_direct()
        assert_allclose(self.f_hat.ravel(), self.idft(self.f), rtol=1e-3)

    def test_adjoint_direct_newapi(self):
        Nfft = NFFT(self.x, self.f, self.f_hat, precompute=True)
        new_f, new_f_hat = self.generate_new_arrays(init_f=True)
        Nfft.adjoint(f=new_f, use_dft=True)
        assert_allclose(self.f_hat.ravel(), self.idft(new_f), rtol=1e-3)
        Nfft.adjoint(f=new_f, f_hat=new_f_hat, use_dft=True)
        assert_allclose(new_f_hat.ravel(), self.idft(new_f), rtol=1e-3)


class Test_NFFT_errors(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Test_NFFT_errors, self).__init__(*args, **kwargs)

    def test_for_invalid_N(self):
        M, N = 20, 32
        x = numpy.empty(M, dtype=numpy.float64)
        f = numpy.empty(M, dtype=numpy.complex128)
        f_hat = numpy.empty(N, dtype=numpy.complex128)
        
        failingN = (-1, 1+numpy.iinfo(numpy.int32).max,
                    (4, numpy.iinfo(numpy.int32).max / 2))
        for someN in failingN:
            self.assertRaises(ValueError,
                              lambda: NFFT(x, f, f_hat, N=(someN,)))

    def test_for_invalid_M(self):
        M, N = 20, 32
        x = numpy.empty(M, dtype=numpy.float64)
        f = numpy.empty(M, dtype=numpy.complex128)
        f_hat = numpy.empty(N, dtype=numpy.complex128)
        
        failingM = (-1, 1+numpy.iinfo(numpy.int32).max)
        for someM in failingM:
            self.assertRaises(ValueError,
                              lambda: NFFT(x, f, f_hat, M=someM))

    def test_for_invalid_n(self):
        M, N = 20, 32
        x = numpy.empty(M, dtype=numpy.float64)
        f = numpy.empty(M, dtype=numpy.complex128)
        f_hat = numpy.empty(N, dtype=numpy.complex128)
        
        failingn = (-1, 1+numpy.iinfo(numpy.int32).max,
                    (4, numpy.iinfo(numpy.int32).max / 2))
        for somen in failingn:
            self.assertRaises(ValueError,
                              lambda: NFFT(x, f, f_hat, n=(somen,)))

    def test_for_invalid_x(self):
        M, N = 20, 32
        x = numpy.empty(M, dtype=numpy.float64)
        f = numpy.empty(M, dtype=numpy.complex128)
        f_hat = numpy.empty(N, dtype=numpy.complex128)
        # array must be contigous
        self.assertRaises(ValueError, lambda: NFFT(x[::2], f, f_hat, M, (N,)))
        # array must be of the right size
        self.assertRaises(ValueError, lambda: NFFT(x[:-2], f, f_hat, M, (N,)))
        # array must be of the right type
        self.assertRaises(ValueError,
                lambda: NFFT(x.astype(numpy.float32), f, f_hat, M, (N,)))

    def test_for_invalid_f(self):
        M, N = 20, 32
        x = numpy.empty(M, dtype=numpy.float64)
        f = numpy.empty(M, dtype=numpy.complex128)
        f_hat = numpy.empty(N, dtype=numpy.complex128)
        # array must be contigous
        self.assertRaises(ValueError, lambda: NFFT(x, f[::2], f_hat, M, (N,)))  
        # array must be of the right size
        self.assertRaises(ValueError, lambda: NFFT(x, f[:-2], f_hat, M, (N,)))
        # array must be of the right type
        self.assertRaises(ValueError,
                lambda: NFFT(x, f.astype(numpy.complex64), f_hat, M, (N,)))

    def test_for_invalid_f_hat(self):
        M, N = 20, 32
        x = numpy.empty(M, dtype=numpy.float64)
        f = numpy.empty(M, dtype=numpy.complex128)
        f_hat = numpy.empty(N, dtype=numpy.complex128)
        # array must be contigous
        self.assertRaises(ValueError, lambda: NFFT(x, f, f_hat[::2], M, (N,)))
        # array must be of the right size
        self.assertRaises(ValueError, lambda: NFFT(x, f, f_hat[:-2], M, (N,)))
        # array must be of the right type
        self.assertRaises(ValueError,
                lambda: NFFT(x, f, f_hat.astype(numpy.complex64), M, (N,)))

    def test_for_invalid_flags(self):
        M, N = 20, 32
        x = numpy.empty(M, dtype=numpy.float64)
        f = numpy.empty(M, dtype=numpy.complex128)
        f_hat = numpy.empty(N, dtype=numpy.complex128)
        # non existing flags
        invalid_flags = ('PRE_PHI_HOT', 'PRE_FOOL_PSI', 'FG_RADIO_PSI')
        for flag in invalid_flags:
            self.assertRaises(ValueError,
                              lambda: NFFT(x, f, f_hat, flags=(flag,)))
        # managed flags
        managed_flags = []
        managed_flags.append([flag for flag in nfft_flags.keys()
                              if flag not in nfft_supported_flags])
        managed_flags.append([flag for flag in fftw_flags.keys()
                              if flag not in nfft_supported_flags])
        for flag in managed_flags:
            self.assertRaises(ValueError,
                              lambda: NFFT(x, f, f_hat, flags=(flag,)))

    def test_for_precomputation_safeguard(self):
        M, N = 20, 32
        x = numpy.empty(M, dtype=numpy.float64)
        f = numpy.empty(M, dtype=numpy.complex128)
        f_hat = numpy.empty(N, dtype=numpy.complex128)
        Nfft = NFFT(x, f, f_hat)
        self.assertRaises(RuntimeError, lambda: Nfft.execute_trafo())       
        self.assertRaises(RuntimeError, lambda: Nfft.execute_trafo_direct()) 
        self.assertRaises(RuntimeError, lambda: Nfft.execute_adjoint()) 
        self.assertRaises(RuntimeError, lambda: Nfft.execute_adjoint_direct())
