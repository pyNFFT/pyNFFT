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
        self.N = (16, 16)
        self.M = 96
        self.m = 6
        self.flags = ('PRE_PHI_HUT', 'FG_PSI', 'PRE_FG_PSI')

    def test_default_args(self):
        Nfft = NFFT(N=self.N, M=self.M)

        default_m = 12
        self.assertEqual(Nfft.m, default_m)

        default_flags = ('PRE_PHI_HUT', 'PRE_PSI')
        for each_flag in default_flags:
            self.assertIn(each_flag, Nfft.flags)

    def test_user_specified_args(self):
        Nfft = NFFT(N=self.N, M=self.M, m=self.m,
                    flags=self.flags)

        self.assertEqual(Nfft.d, len(self.N))

        for t, Nt in enumerate(self.N):
            self.assertEqual(Nfft.N[t], Nt)

        self.assertEqual(Nfft.N_total, numpy.prod(self.N))

        self.assertEqual(Nfft.M_total, self.M)

        self.assertEqual(Nfft.m, self.m)

        for each_flag in self.flags:
            self.assertIn(each_flag, Nfft.flags)
    
    def test_precomputation_flag(self):
        Nfft = NFFT(N=self.N, M=self.M, m=self.m,
                    flags=self.flags)
        self.assertFalse(Nfft.precomputed)
        Nfft.x = numpy.ones(Nfft.M_total * Nfft.d)
        Nfft.precompute()
        self.assertTrue(Nfft.precomputed)


class Test_NFFT_runtime(unittest.TestCase):

    N = (32, 32)
    M = 1280

    @staticmethod
    def compare_with_fdft(Nfft):
        N = Nfft.N
        f = Nfft.f
        f_hat = Nfft.f_hat
        k = numpy.mgrid[slice(N[0]), slice(N[1])]
        k = k.reshape([2, -1]) - numpy.asarray(N).reshape([2, 1]) / 2
        x = Nfft.x.reshape([-1, 2])
        F = numpy.exp(-2j * pi * numpy.dot(x, k))
        f_dft = numpy.dot(F, f_hat)
        assert_allclose(f, f_dft, rtol=1e-3)

    @staticmethod
    def compare_with_idft(Nfft):
        N = Nfft.N
        f = Nfft.f
        f_hat = Nfft.f_hat
        k = numpy.mgrid[slice(N[0]), slice(N[1])]
        k = k.reshape([2, -1]) - numpy.asarray(N).reshape([2, 1]) / 2
        x = Nfft.x.reshape([-1, 2])
        F = numpy.exp(-2j * pi * numpy.dot(x, k))
        f_hat_dft = numpy.dot(numpy.conjugate(F).T, f)
        assert_allclose(f_hat, f_hat_dft, rtol=1e-3)

    def __init__(self, *args, **kwargs):
        super(Test_NFFT_runtime, self).__init__(*args, **kwargs)
        self.Nfft = NFFT(N=self.N, M=self.M)
        vrand_shifted_unit_double(self.Nfft.x)
        self.Nfft.precompute()

    def test_trafo(self):
        vrand_unit_complex(self.Nfft.f_hat)
        self.Nfft.trafo()
        self.compare_with_fdft(self.Nfft)

    def test_trafo_direct(self):
        vrand_unit_complex(self.Nfft.f_hat)
        self.Nfft.trafo_direct()
        self.compare_with_fdft(self.Nfft)

    def test_adjoint(self):
        vrand_unit_complex(self.Nfft.f)
        self.Nfft.adjoint()
        self.compare_with_idft(self.Nfft)

    def test_adjoint_direct(self):
        vrand_unit_complex(self.Nfft.f)
        self.Nfft.adjoint_direct()
        self.compare_with_idft(self.Nfft)


class Test_NFFT_errors(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Test_NFFT_errors, self).__init__(*args, **kwargs)

    def test_for_invalid_N(self):
        # N must be between 0 and INT_MAX
        N = -1
        M = 32
        self.assertRaises(ValueError, lambda: NFFT(N=N, M=M))
        N = numpy.iinfo(numpy.int32).max + 1
        self.assertRaises(ValueError, lambda: NFFT(N=N, M=M))
        # N_total should not be more than INT_MAX
        N = (4, numpy.iinfo(numpy.int32).max / 2)
        self.assertRaises(ValueError, lambda: NFFT(N=N, M=M))

    def test_for_invalid_M(self):
        # M_total must be between 0 and INT_MAX
        N = 32
        M = -1
        self.assertRaises(ValueError, lambda: NFFT(N=N, M=M))
        M = numpy.iinfo(numpy.int32).max + 1
        self.assertRaises(ValueError, lambda: NFFT(N=N, M=M))

    def test_for_invalid_n(self):
        # n must be between 0 and INT_MAX
        N = 32
        M = 32
        n = -1
        self.assertRaises(ValueError, lambda: NFFT(N=N, M=M, n=n))
        n = numpy.iinfo(numpy.int32).max + 1
        self.assertRaises(ValueError, lambda: NFFT(N=N, M=M, n=n))

    def test_for_invalid_x(self):
        N = 32
        M = 32
        x = numpy.linspace(-0.5, 0.5, M, endpoint=False)
        x = x.astype(numpy.float64)
        # array must be contigous
        self.assertRaises(ValueError,
                          lambda: NFFT(N=N, M=M/2, x=x[::2]))
        # array must be of the right size
        self.assertRaises(ValueError,
                          lambda: NFFT(N=N, M=M, x=x[:M/2]))
        # array must be of the right type
        x = x.astype(numpy.float32)
        self.assertRaises(ValueError,
                          lambda: NFFT(N=N, M=M, x=x))

    def test_for_invalid_f(self):
        N = 32
        M = 32
        f = numpy.arange(M)
        f = f.astype(numpy.complex128)
        # array must be contigous
        self.assertRaises(ValueError,
                          lambda: NFFT(N=N, M=M/2, f=f[::2]))
        # array must be of the right size
        self.assertRaises(ValueError,
                          lambda: NFFT(N=N, M=M, f=f[:M/2]))
        # array must be of the right type
        f = f.astype(numpy.complex64)
        self.assertRaises(ValueError,
                          lambda: NFFT(N=N, M=M, f=f))

    def test_for_invalid_f_hat(self):
        N = 32
        M = 32
        f_hat = numpy.arange(N)
        f_hat = f_hat.astype(numpy.complex128)
        # array must be contigous
        self.assertRaises(ValueError,
                          lambda: NFFT(N=N/2, M=M, f_hat=f_hat[::2]))
        # array must be of the right size
        self.assertRaises(ValueError,
                          lambda: NFFT(N=N, M=M, f_hat=f_hat[:N/2]))
        # array must be of the right type
        f_hat = f_hat.astype(numpy.complex64)
        self.assertRaises(ValueError,
                          lambda: NFFT(N=N, M=M, f_hat=f_hat))

    def test_for_invalid_flags(self):
        N = 32
        M = 32
        # non existing flags
        invalid_flags = ('PRE_PHI_HOT', 'PRE_FOOL_PSI', 'FG_RADIO_PSI')
        for flag in invalid_flags:
            self.assertRaises(ValueError,
                              lambda: NFFT(N=N, M=M, flags=(flag,)))
        # managed flags
        managed_flags = []
        managed_flags.append([flag for flag in nfft_flags.keys()
                              if flag not in nfft_supported_flags])
        managed_flags.append([flag for flag in fftw_flags.keys()
                              if flag not in nfft_supported_flags])
        for flag in managed_flags:
            self.assertRaises(ValueError,
                              lambda: NFFT(N=N, M=M, flags=(flag,)))

    def test_for_precomputation_safeguard(self):
        N, M = 32, 32
        Nfft = NFFT(N=N, M=M)
        self.assertRaises(RuntimeError, lambda: Nfft.trafo())       
        self.assertRaises(RuntimeError, lambda: Nfft.trafo_direct()) 
        self.assertRaises(RuntimeError, lambda: Nfft.adjoint()) 
        self.assertRaises(RuntimeError, lambda: Nfft.adjoint_direct())


def suite():
    suite = unittest.TestSuite()
    suite.addTest(Test_NFFT_init("test_default_args"))
    suite.addTest(Test_NFFT_init("test_user_specified_args"))
    suite.addTest(Test_NFFT_init("test_precomputation_flag"))
    suite.addTest(Test_NFFT_runtime("test_trafo"))
    suite.addTest(Test_NFFT_runtime("test_trafo_direct"))
    suite.addTest(Test_NFFT_runtime("test_adjoint"))
    suite.addTest(Test_NFFT_runtime("test_adjoint_direct"))
    suite.addTest(Test_NFFT_errors('test_for_invalid_N'))
    suite.addTest(Test_NFFT_errors('test_for_invalid_M'))
    suite.addTest(Test_NFFT_errors('test_for_invalid_n'))
    suite.addTest(Test_NFFT_errors('test_for_invalid_x'))
    suite.addTest(Test_NFFT_errors('test_for_invalid_f'))
    suite.addTest(Test_NFFT_errors('test_for_invalid_f_hat'))
    suite.addTest(Test_NFFT_errors('test_for_invalid_flags'))
    suite.addTest(Test_NFFT_errors('test_for_precomputation_safeguard'))    
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
