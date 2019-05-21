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

from ._nfft import _NFFTDouble, _NFFTFloat, _NFFTLongDouble

NFFT_FLAGS = {
    'PRE_PHI_HUT': 1 << 0,
    'FG_PSI': 1 << 1,
    'PRE_LIN_PSI': 1 << 2,
    'PRE_FG_PSI': 1 << 3,
    'PRE_PSI': 1 << 4,
    'PRE_FULL_PSI': 1 << 5,
    'MALLOC_X': 1 << 6,
    'MALLOC_F_HAT': 1 << 7,
    'MALLOC_F': 1 << 8,
    'FFT_OUT_OF_PLACE': 1 << 9,
    'FFTW_INIT': 1 << 10,
    'NFFT_SORT_NODES': 1 << 11,
    'NFFT_OMP_BLOCKWISE_ADJOINT': 1 <<12,
}
NFFT_FLAGS['PRE_ONE_PSI'] = (
    NFFT_FLAGS['PRE_LIN_PSI']
    | NFFT_FLAGS['PRE_FG_PSI']
    | NFFT_FLAGS['PRE_PSI']
    | NFFT_FLAGS['PRE_FULL_PSI']
)
FFTW_FLAGS = {
    'FFTW_DESTROY_INPUT': 1 << 0,
    'FFTW_ESTIMATE': 1 << 6,
}


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

        M = int(M)
        if M <= 0:
            raise ValueError('`M` must be positive')

        m = int(m)
        if m <= 0:
            raise ValueError('`m` must be positive')

        if n is None:
            n = tuple(max(2 * Ni, m + 1) for Ni in N)
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

        nfft_flags = 0
        fftw_flags = 0
        for flag in flags:
            if flag in NFFT_FLAGS:
                nfft_flags |= NFFT_FLAGS[flag]
            elif flag in FFTW_FLAGS:
                fftw_flags |= FFTW_FLAGS[flag]
            else:
                raise ValueError('Unsupported flag: {}'.format(flag))

        # 'long double' -> 'longdouble' (latter understood by np.dtype)
        dtype_real = np.dtype(str(prec).replace(' ', ''))
        dtype_complex = np.result_type(1j, dtype_real)
        if dtype_complex not in ('complex64', 'complex128', 'complex256'):
            raise ValueError('`prec` {!r} not recognized'.format(prec))

        # Create wrapper plan
        if dtype_complex == 'complex64':
            self._plan = _NFFTFloat(N, M, n, m, nfft_flags, fftw_flags)
        elif dtype_complex == 'complex128':
            self._plan = _NFFTDouble(N, M, n, m, nfft_flags, fftw_flags)
        elif dtype_complex == 'complex256':
            self._plan = _NFFTLongDouble(N, M, n, m, nfft_flags, fftw_flags)

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
        self._plan.trafo(use_dft)
        return self.f

    def adjoint(self, use_dft=False):
        """Perform the adjoint NFFT.

        :param use_dft: whether to use the DFT instead of the fast algorithm.
        :type use_dft: boolean
        :returns: the updated :attr:`f_hat` array.
        :rtype: ndarray
        """
        self._plan.adjoint(use_dft)
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
