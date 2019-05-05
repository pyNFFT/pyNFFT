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

from functools import reduce

import numpy as np

from ._nfft import _NFFT, NFFT_FLAGS


class NFFT(object):

    """Non-uniform Discrete Fourier transform class.

    ``NFFT`` is a class for computing the multivariate Non-uniform Discrete
    Fourier Transform (NDFT) using the `NFFT
    <https://www-user.tu-chemnitz.de/~potts/nfft/>`_ library. The interface
    is designed to be somewhat Pythonic, whilst preserving the workflow of
    the original C library. Computation of the NFFT is achieved in 3 steps:
    instantiation, precomputation and execution.

    On instantiation, the geometry of the transform is provided. Optional
    computation parameters may also be defined.

    Precomputation initializes the internals of the transform prior to
    execution. First the non-uniform locations must be given to the plan
    via its :attr:`x` attribute. Necessary pre-computations can then be
    carried out with the :meth:`precompute` method.

    The forward and adjoint NFFT can eventually be performed by calling the
    :meth:`forward` and :meth:`adjoint` methods, respectively. The
    inputs/outputs of the transform can be read/written by access to the
    :attr:`f` and :attr:`f_hat` attributes.

    :param N: multi-bandwith.
    :type N: tuple of int
    :param M: total number of samples.
    :type n: int
    :param n: oversampled multi-bandwith, defaults to ``2 * N``.
    :type n: tuple of int
    :param m: Cut-off parameter of the window function.
    :type m: int
    :param flags: list of precomputation flags, see note below.
    :type flags: tuple
    :param prec: Floating point precision, can be ``'double'`` (default), ``'single'`` or ``'long double'``
    :type prec: string

    **Precomputation flags**

    This table lists the supported precomputation flags for the NFFT.

    +----------------------------+--------------------------------------------------+
    | Flag                       | Description                                      |
    +============================+==================================================+
    | PRE_PHI_HUT                | Precompute the roll-off correction coefficients. |
    +----------------------------+--------------------------------------------------+
    | FG_PSI                     | Convolution uses Fast Gaussian properties.       |
    +----------------------------+--------------------------------------------------+
    | PRE_LIN_PSI                | Convolution uses a precomputed look-up table.    |
    +----------------------------+--------------------------------------------------+
    | PRE_FG_PSI                 | Precompute Fast Gaussian.                        |
    +----------------------------+--------------------------------------------------+
    | PRE_PSI                    | Standard precomputation, uses M*(2m+2)*d values. |
    +----------------------------+--------------------------------------------------+
    | PRE_FULL_PSI               | Full precomputation, uses M*(2m+2)^d values.     |
    +----------------------------+--------------------------------------------------+

    Default set of flags is ``('PRE_PHI_HUT', 'PRE_PSI')``.
    """

    def __init__(self, N, M, n=None, m=12, flags=None, prec='double'):
        # Convert and check input parameters
        try:
            N = tuple(int(Ni) for Ni in N)
        except TypeError:
            N = (int(N),)
        if not all(Ni > 0 for Ni in N):
            raise ValueError('`N` must be positive')
        d = len(N)
        # Use arbitrary-size int vs. np.prod which uses fixed-size integers
        N_total = reduce(lambda i, j: i * j, N, 1)

        M = int(M)
        if M <= 0:
            raise ValueError('`M` must be positive')

        if n is None:
            n = tuple(2 * Ni for Ni in N)
        try:
            n = tuple(int(ni) for ni in n)
        except TypeError:
            n = (int(n),)
        if len(n) != d:
            raise ValueError(
                '`n` must have the same length as `N`, but '
                '{} = len(n) != len(N) = {}'.format(len(n), d)
            )
        if not all(ni > 0 for ni in n):
            raise ValueError('`n` must be positive')
        n_total = reduce(lambda i, j: i * j, n, 1)

        m = int(m)
        if m <= 0:
            raise ValueError('`m` must be positive')

        # Safeguard against oversampled grid size being too small
        # for kernel size
        if not all(ni > m for ni in n):
            raise ValueError('`n` must be larger than `m`')

        if flags is None:
            flags = ('PRE_PHI_HUT', 'PRE_PSI')
        else:
            try:
                flags = tuple(flags)
            except:
                flags = (flags,)
            finally:
                # Only allow NFFT flags
                for flag in flags:
                    if flag not in NFFT_FLAGS:
                        raise ValueError('Unsupported flag: {}'.format(flag))

        # Set specific flags unconditionally
        # TODO: allow user control for some?

        # FFTW flags
        flags += (
            'FFTW_INIT', 'FFT_OUT_OF_PLACE', 'FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'
        )

        # Memory allocation flags
        flags += ('MALLOC_F', 'MALLOC_F_HAT', 'MALLOC_X')

        # Parallel computation flag
        flags += ('NFFT_SORT_NODES',)

        # Parallel computation flag, set only for multivariate transforms
        if d > 1:
            flags += ('NFFT_OMP_BLOCKWISE_ADJOINT',)

        # 'long double' -> 'longdouble' (latter understood by np.dtype)
        dtype_real = np.dtype(str(prec).replace(' ', ''))
        dtype_complex = np.result_type(1j, dtype_real)
        if dtype_complex not in ('complex64', 'complex128', 'complex256'):
            raise ValueError('`prec` {!r} not recognized'.format(prec))

        # Create arrays for the C interface and check for int overflow
        N_arr = np.array(N, dtype=np.int_)
        assert np.prod(N_arr) == N_total, str((np.prod(N_arr), N_total))
        M_arr = np.array([M], dtype=np.int_)
        assert np.prod(M_arr) == M, str((np.prod(M_arr), M))
        n_arr = np.array(n, dtype=np.int_)
        assert np.prod(n_arr) == n_total, str((np.prod(n_arr), n_total))
        m_arr = np.array(m, dtype=np.int_)
        assert np.prod(m_arr) == m, str((np.prod(m_arr), m))
        x_shp_arr = np.array((M, d), dtype=np.int_)
        assert np.prod(x_shp_arr) == M * d, str((np.prod(x_shp_arr), M * d))

        # Create wrapper plan
        self._plan = _NFFT(
            dtype_complex, d, N_arr, M_arr, n_arr, m_arr, flags
        )

        # Set misc member attributes
        self._d = d
        self._M = M
        self._m = m
        self._N = N
        self._n = n
        self._dtype = dtype_complex
        self._flags = flags

    def precompute(self):
        """Precompute the NFFT plan internals."""
        self._plan._precompute()

    def trafo(self, use_dft=False):
        """Perform the forward NFFT.

        :param use_dft: whether to use the DFT instead of the fast algorithm.
        :type use_dft: boolean
        :returns: the updated :attr:`f` array.
        :rtype: ndarray
        """
        if use_dft:
            self._plan._trafo_direct()
        else:
            self._plan._trafo()
        return self.f

    def adjoint(self, use_dft=False):
        """Perform the adjoint NFFT.

        :param use_dft: whether to use the DFT instead of the fast algorithm.
        :type use_dft: boolean
        :returns: the updated :attr:`f_hat` array.
        :rtype: ndarray
        """
        if use_dft:
            self._plan._adjoint_direct()
        else:
            self._plan._adjoint()
        return self.f_hat

    # --- Pass-through from C plan --- #

    @property
    def f(self):
        """1D array of non-uniform samples of length :attr:`M`."""
        return self._plan.f

    @f.setter
    def f(self, new_f):
        self.f.ravel()[:] = new_f.ravel()

    @property
    def f_hat(self):
        """Array of Fourier coefficients with shape :attr:`N`."""
        return self._plan.f_hat

    @f_hat.setter
    def f_hat(self, new_f_hat):
        self.f_hat.ravel()[:] = new_f_hat.ravel()

    @property
    def x(self):
        """Nodes of the non-uniform FFT, shape ``(M, d)``."""
        return self._plan.x

    @x.setter
    def x(self, new_x):
        self.x.ravel()[:] = new_x.ravel()

    # --- Plan properties --- #

    @property
    def d(self):
        """Dimensionality of the NFFT."""
        return self._d

    @property
    def N(self):
        """Multi-bandwidth size."""
        return self._plan._N

    @property
    def N_total(self):
        """Total number of Fourier coefficients."""
        return int(np.prod(self.N))

    @property
    def M(self):
        """Total number of samples."""
        return self._M

    @property
    def n(self):
        """Oversampled multi-bandwidth size."""
        return self._n

    @property
    def m(self):
        """Cut-off parameter of the window function."""
        return self._m

    @property
    def dtype(self):
        """Data type of the NFFT."""
        return self._dtype

    @property
    def flags(self):
        """NFFT and FFTW transform flags."""
        return self._flags
