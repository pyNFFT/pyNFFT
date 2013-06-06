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

import numpy
import unittest
from pynfft.nfft import (NFFT, fftw_flags, nfft_flags,
                         nfft_supported_flags)


class TestNFFTError(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestNFFTError, self).__init__(*args, **kwargs)


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
        self.assertRaises(ValueError, lambda: NFFT(N=N, M=M/2, x=x[::2]))
        # array must be of the right size
        self.assertRaises(ValueError, lambda: NFFT(N=N, M=M, x=x[:M/2]))
        # array must be of the right type
        x = x.astype(numpy.float32)
        self.assertRaises(ValueError, lambda: NFFT(N=N, M=M, x=x))


    def test_for_invalid_f(self):
        N = 32
        M = 32
        f = numpy.arange(M)
        f = f.astype(numpy.complex128)
        # array must be contigous
        self.assertRaises(ValueError, lambda: NFFT(N=N, M=M/2, f=f[::2]))
        # array must be of the right size
        self.assertRaises(ValueError, lambda: NFFT(N=N, M=M, f=f[:M/2]))
        # array must be of the right type
        f = f.astype(numpy.complex64)
        self.assertRaises(ValueError, lambda: NFFT(N=N, M=M, f=f))


    def test_for_invalid_f_hat(self):
        N = 32
        M = 32
        f_hat = numpy.arange(N)
        f_hat = f_hat.astype(numpy.complex128)
        # array must be contigous
        self.assertRaises(ValueError, lambda: NFFT(N=N/2, M=M, f_hat=f_hat[::2]))
        # array must be of the right size
        self.assertRaises(ValueError, lambda: NFFT(N=N, M=M, f_hat=f_hat[:N/2]))
        # array must be of the right type
        f_hat = f_hat.astype(numpy.complex64)
        self.assertRaises(ValueError, lambda: NFFT(N=N, M=M, f_hat=f_hat))


    def test_for_invalid_flags(self):
        N = 32
        M = 32
        # non existing flags
        invalid_flags = ('PRE_PHI_HOT', 'PRE_FOOL_PSI', 'FG_RADIO_PSI')
        for flag in invalid_flags:
            self.assertRaises(ValueError, lambda: NFFT(N=N, M=M, flags=(flag,)))
        # managed flags
        managed_flags = []
        managed_flags.append([flag for flag in nfft_flags.keys()
                              if flag not in nfft_supported_flags])
        managed_flags.append([flag for flag in fftw_flags.keys()
                              if flag not in nfft_supported_flags])
        for flag in managed_flags:
            self.assertRaises(ValueError,
                              lambda: NFFT(N=N, M=M, flags=(flag,)))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestNFFTError('test_for_invalid_N'))
    suite.addTest(TestNFFTError('test_for_invalid_M'))
    suite.addTest(TestNFFTError('test_for_invalid_n'))
    suite.addTest(TestNFFTError('test_for_invalid_x'))
    suite.addTest(TestNFFTError('test_for_invalid_f'))
    suite.addTest(TestNFFTError('test_for_invalid_f_hat'))
    suite.addTest(TestNFFTError('test_for_invalid_flags'))
    return suite

if __name__ == '__main__':
    suite = suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
