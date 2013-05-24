"""
Unit tests for NFFTPY (cython wrapper for NFFT libraries)

This also serves as a simple example of using NFFTPY from Python.

In part, this is a python translation of NFFT's examples/nfft/simple_test.c.

However, for reproducibility, instead of using NFFT's pseudo-random data
generator here, we use the input and output arrays which were previously
generated and saved to files by simple_test_class.pyx, and were spot-checked
by hand against the data printed by NFFT's original simple_test.c.

These data files are different when generated on different systems, presumably
because of different pseudo-random number generation, but they should work on
any system. Here, for each of the 1d and 2d tests, we use two sets of files,
which were generated on 32 and 64-bit Ubuntu VMs.

FIXME: Need unit tests examining non-random data sampled from simple
functions with known transforms.

"""
import os
import numpy as np
from numpy.testing import (assert_equal, assert_array_almost_equal)
import unittest
import pynfft.nfft as nfft


def read_sample_data(filename, pw):
    """
    Read a set of sample data which was written by simple_test_class.pyx. It
    consists of 4 concatenated arrays, each preceded by its number of elements.
    Validate the length of each array against the expected length, given the
    plan wrapper pw.

    Returns a list of 4 numpy arrays:
        x_data : dtype=float
        f_hat_data, f_data, adjoint_f_hat_data : dtype=complex
    """
    num_x = pw.d * pw.M_total
    data_filename = os.path.join(os.path.dirname(__file__), filename)
    data = np.loadtxt(data_filename, dtype=np.complex128)
    data_divided = []
    i = 0
    corruption_msg = ('test data in %s is apparently corrupted at row %%i' %
                      data_filename)
    for expected_len in (num_x, pw.N_total, pw.M_total, pw.N_total):
        n_elem = int(round(data[i].real))
        if n_elem != expected_len:
            raise IOError(corruption_msg % i)
        i += 1
        next_elem = i + n_elem
        data_divided.append(data[i:next_elem])
        i += n_elem
    if next_elem != len(data):
        raise IOError(corruption_msg % i)
    data_divided[0] = data_divided[0].real.astype(np.float64)
    return data_divided


def check_a_plan(pw, x_data, f_hat_data, f_data, adjoint_f_hat_data):
    """
    After a plan is initialized, feed it data, compute transforms,
    and check the results.
    """
    # init pseudo random nodes and check that their values took:
    pw.x = x_data
    _x = pw.x
    assert_array_almost_equal(_x, x_data)

    # precompute psi, the entries of the matrix B
    pw.precompute()

    # init pseudo random Fourier coefficients and check their values took:
    pw.f_hat = f_hat_data
    _f_hat = pw.f_hat
    assert_array_almost_equal(_f_hat, f_hat_data)

    # direct trafo and test the result
    pw.trafo_direct()
    _f = pw.f
    assert_array_almost_equal(_f, f_data)

    # approx. trafo and check the result
    # first clear the result array to be sure that it is actually touched.
    pw.f = np.zeros_like(f_data)
    pw.trafo()
    _f2 = pw.f
    assert_array_almost_equal(_f2, f_data)

    # direct adjoint and check the result
    pw.adjoint_direct()
    _f_hat2 = pw.f_hat
    assert_array_almost_equal(_f_hat2, adjoint_f_hat_data)

    # approx. adjoint and check the result.
    # first clear the result array to be sure that it is actually touched.
    pw.f_hat = np.zeros_like(f_hat_data)
    pw.adjoint()
    _f_hat3 = pw.f_hat
    assert_array_almost_equal(_f_hat3, adjoint_f_hat_data)


def read_and_check(pw, data_filename):
    sample_data_arrays = read_sample_data(data_filename, pw)
    check_a_plan(pw, *sample_data_arrays)


def test_nfft_1d():
    """
    Reproduce and check the 1d case from examples/nfft/simple_test.c.
    """
    N = 14
    M = 19
    for data_file in ('simple_test_nfft_1d_32.txt',
                      'simple_test_nfft_1d_64.txt'):
        # init a one dimensional plan
        pw = nfft.NFFT(N, M)
        assert_equal(pw.M_total, M)
        assert_equal(pw.N_total, N)
        assert_equal(pw.d, 1)
        read_and_check(pw, data_file)


def test_nfft_2d():
    """
    Reproduce and check the 2d case from examples/nfft/simple_test.c.
    """
    N = np.array([32, 14], dtype=np.int32)
    n = np.array([64, 32], dtype=np.int32)
    M = N.prod()
    for data_file in ('simple_test_nfft_2d_32.txt',
                      'simple_test_nfft_2d_64.txt'):

        # init a two dimensional plan
        flags = ('PRE_PHI_HUT', 'PRE_FULL_PSI', 'MALLOC_F_HAT', 'MALLOC_X',
                 'MALLOC_F', 'FFTW_INIT', 'FFT_OUT_OF_PLACE')
        pw = nfft.NFFT(N, M, n, m=7, flags=flags)

        assert_equal(pw.M_total, M)
        assert_equal(pw.d, 2)
        read_and_check(pw, data_file)


test_nfft_1d_case = unittest.FunctionTestCase(test_nfft_1d)
test_nfft_2d_case = unittest.FunctionTestCase(test_nfft_2d)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests([test_nfft_1d_case, test_nfft_2d_case])
    unittest.TextTestRunner(verbosity=2).run(suite)
