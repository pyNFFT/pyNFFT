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

import unittest


class Test_NFFT_init(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Test_NFFT_init, self).__init__(*args, **kwargs)

    def test_default_args(self):
        pass

    def test_user_specified_args(self):
        pass


class Test_NFFT_runtime(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Test_NFFT_runtime, self).__init__(*args, **kwargs)

    def test_trafo(self):
        pass

    def test_trafo_direct(self):
        pass

    def test_adjoint(self):
        pass

    def test_adjoint_direct(self):
        pass

    def reference_fdft(x, f_hat, f):
        pass

    def reference_idft(x, f, f_hat):
        pass


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
