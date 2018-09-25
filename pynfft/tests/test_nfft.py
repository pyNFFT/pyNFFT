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
import pytest
from numpy import pi
from pynfft.nfft import NFFT
from pynfft.util import vrand_unit_complex, vrand_shifted_unit_double


# --- Test fixtures --- #


@pytest.fixture(
    scope='module',
    params=[6, 12],
    ids=[' m={} '.format(p) for p in [6, 12]],
)
def m(request):
    return request.param


@pytest.fixture(
    scope='module',
    params=[True, False],
    ids=[' use_dft={} '.format(p) for p in [True, False]],
)
def use_dft(request):
    return request.param


@pytest.fixture(
    scope='module',
    params=['single', 'double', 'long double'],
    ids=[' prec={!r} '.format(p) for p in ['single', 'double', 'long double']],
)
def prec(request):
    return request.param


shapes = (
    (8, 8),
    (16, 16),
    (24, 24),
    (32, 32),
    (64, 64),
    ((8, 8), 8 * 8),
    ((16, 16), 16 * 16),
    ((24, 24), 24 * 24),
    ((32, 32), 32 * 32),
    ((64, 64), 64 * 64),
    ((8, 8, 8), 8 * 8 * 8),
    ((16, 16, 8), 8 * 8 * 8),
    ((16, 16, 16), 16 * 16 * 16),
)

@pytest.fixture(
    scope='module',
    params=shapes,
    ids=[' shapes={} '.format(arg) for arg in shapes],
)
def plan(request, m, prec):
    N, M = request.param

    if prec == 'single' and m > 6:
        # Unstable
        pytest.skip('likely to produce NaN')

    pl = NFFT(N, M, prec=prec, m=m)
    vrand_shifted_unit_double(pl.x.ravel())
    pl.precompute()
    return pl


# --- Helpers --- #


def fdft(x, f_hat):
    N = f_hat.shape
    d = x.shape[-1]
    k = np.mgrid[[slice(-Nt/2, Nt/2) for Nt in N]]
    k = k.reshape([d, -1])
    x = x.reshape([-1, d])
    F = np.exp(-2j * pi * np.dot(x, k))
    f_dft = np.dot(F, f_hat.ravel())
    return f_dft

def rdft(x, f, N):
    d = x.shape[-1]
    k = np.mgrid[[slice(-Nt/2, Nt/2) for Nt in N]]
    k = k.reshape([d, -1])
    x = x.reshape([-1, d])
    F = np.exp(-2j * pi * np.dot(x, k))
    f_hat_dft = np.dot(np.conjugate(F).T, f)
    f_hat = f_hat_dft.reshape(N)
    return f_hat


# --- Tests --- #


def test_forward(plan, use_dft):
    rtol = 1e-3 if plan.dtype == 'complex64' else 1e-7
    vrand_unit_complex(plan.f_hat.ravel())
    assert np.allclose(
        plan.trafo(use_dft=use_dft),
        fdft(plan.x, plan.f_hat),
        rtol=rtol,
    )


def test_adjoint(plan, use_dft):
    rtol = 1e-3 if plan.dtype == 'complex64' else 1e-7
    vrand_unit_complex(plan.f.ravel())
    assert np.allclose(
        plan.adjoint(use_dft=use_dft),
        rdft(plan.x, plan.f, plan.N),
        rtol=rtol,
    )
