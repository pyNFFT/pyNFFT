# -*- coding: utf-8 -*-
#
# Copyright PyNFFT developers and contributors
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


def random_unit_complex(shape, dtype):
    """Return an array of random complex numbers in [0, 1) + [0, 1) * i.

    Used for testing :attr:`pynfft.NFFT.f` and
    :attr:`pynfft.NFFT.f_hat`.

    :param shape: number of samples per axis
    :type shape: int or tuple of int
    :param dtype: complex data type
    :type dtype: data-type
    """
    try:
        shape = (int(shape),)
    except TypeError:
        shape = tuple(int(n) for n in shape)

    dtype, dtype_in = np.dtype(dtype), dtype
    if dtype.kind != "c":
        raise ValueError(
            "`dtype` must be a complex data type, got {!r}" "".format(dtype_in)
        )

    return (
        np.random.uniform(size=shape) + 1j * np.random.uniform(size=shape)
    ).astype(dtype)


def random_unit_shifted(shape, dtype):
    """Return a vector of random real numbers in [-0.5, 0.5).

    Used for testing :attr:`pynfft.NFFT.x`.

    :param shape: number of samples per axis
    :type shape: int or tuple of int
    :param dtype: real floating-point data type
    :type dtype: data-type
    """
    try:
        shape = (int(shape),)
    except TypeError:
        shape = tuple(int(n) for n in shape)

    dtype, dtype_in = np.dtype(dtype), dtype
    if dtype.kind != "f":
        raise ValueError(
            "`dtype` must be a real floating-point data type, got {!r}"
            "".format(dtype_in)
        )
    return np.random.uniform(-0.5, 0.5, size=shape).astype(dtype)
