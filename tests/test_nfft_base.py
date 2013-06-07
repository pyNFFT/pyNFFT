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
import numpy as np
import unittest
from numpy import pi
from numpy.testing import assert_allclose
from pynfft.nfft import NFFT
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

        self.assertEqual(Nfft.N_total, np.prod(self.N))

        self.assertEqual(Nfft.M_total, self.M)

        self.assertEqual(Nfft.m, self.m)

        for each_flag in self.flags:
            self.assertIn(each_flag, Nfft.flags)


class Test_NFFT_runtime(unittest.TestCase):

    N = (32, 32)
    M = 1280

    @staticmethod
    def compare_with_fdft(Nfft):
        N = Nfft.N
        f = Nfft.f
        f_hat = Nfft.f_hat
        k = np.mgrid[slice(N[0]), slice(N[1])]
        k = k.reshape([2, -1]) - np.asarray(N).reshape([2, 1]) / 2
        x = Nfft.x.reshape([-1, 2])
        F = np.exp(-2j * pi * np.dot(x, k))
        f_dft = np.dot(F, f_hat)
        assert_allclose(f, f_dft, rtol=1e-3)

    @staticmethod
    def compare_with_idft(Nfft):
        N = Nfft.N
        f = Nfft.f
        f_hat = Nfft.f_hat
        k = np.mgrid[slice(N[0]), slice(N[1])]
        k = k.reshape([2, -1]) - np.asarray(N).reshape([2, 1]) / 2
        x = Nfft.x.reshape([-1, 2])
        F = np.exp(-2j * pi * np.dot(x, k))
        f_hat_dft = np.dot(np.conjugate(F).T, f)
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


def suite():
    suite = unittest.TestSuite()
    suite.addTest(Test_NFFT_init("test_default_args"))
    suite.addTest(Test_NFFT_init("test_user_specified_args"))
    suite.addTest(Test_NFFT_runtime("test_trafo"))
    suite.addTest(Test_NFFT_runtime("test_trafo_direct"))
    suite.addTest(Test_NFFT_runtime("test_adjoint"))
    suite.addTest(Test_NFFT_runtime("test_adjoint_direct"))
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
