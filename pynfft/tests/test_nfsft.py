# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Taco Cohen
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
from numpy import pi
from numpy.testing import assert_allclose
from pynfft.nfsft import NFSFT
from scipy.special import sph_harm
from pynfft.util import vrand_unit_complex, vrand_shifted_unit_double

def nfsft_trafo_naive(x, f_hat):
    
    N_total = f_hat.size
    N = int(np.sqrt(N_total) - 1)
    M = x.shape[0]

    theta = x[:, 0]
    phi = x[:, 1]

    B = np.empty((M, N_total), dtype='complex')
    i = 0
    for l in range(N + 1):
        for m in range(-l, l + 1):
            # The scipy spherical harmonics use a different normalization and phase convention,
            # so we fix that here:
            B[:, i] = sph_harm(m, l, phi, theta) / np.sqrt((2 * l + 1) / (4 * np.pi))
            B[:, i] *= ((-1) ** (m * (m > 0)))
            i += 1

    return B.dot(f_hat)


def nfsft_adjoint_naive(x, f, N):

    N_total = (N + 1) ** 2
    M = x.shape[0]

    theta = x[:, 0]
    phi = x[:, 1]
    
    B = np.empty((M, N_total), dtype='complex')
    i = 0
    for l in range(N + 1):
        for m in range(-l, l + 1):
            # The scipy spherical harmonics use a different normalization and phase convention,
            # so we fix that here:
            B[:, i] = sph_harm(m, l, phi, theta) / np.sqrt((2 * l + 1) / (4 * np.pi))
            B[:, i] *= ((-1) ** (m * (m > 0)))
            i += 1

    return B.conj().T.dot(f)


def check_forward_nfsft(plan):
    f_hat = np.random.randn((plan.N + 1) ** 2) + 1j * np.random.randn((plan.N + 1) ** 2)
    plan.set_f_hat_flat(f_hat)
    assert_allclose(plan.trafo(False), nfsft_trafo_naive(plan.x, f_hat))

def check_forward_ndsft(plan):
    f_hat = np.random.randn((plan.N + 1) ** 2) + 1j * np.random.randn((plan.N + 1) ** 2)
    plan.set_f_hat_flat(f_hat)
    assert_allclose(plan.trafo(True), nfsft_trafo_naive(plan.x, f_hat))

def check_adjoint_nfsft(plan):
    plan.f = np.random.randn(plan.f.size)
    assert_allclose(plan.adjoint(False), nfsft_adjoint_naive(plan.x, plan.f, plan.N))

def check_adjoint_ndsft(plan):
    plan.f = np.random.randn(plan.f.size)
    assert_allclose(plan.adjoint(True), nfsft_adjoint_naive(plan.x, plan.f, plan.N))

tested_nfsft_args = (
    (8, 8, dict(m=6)),
    (16, 16, dict()),
    (24, 24, dict()),
    (32, 32, dict()),
    (64, 64, dict()),
)

def test_forward_nfsft():
    for N, M, nfsft_kwargs in tested_nfsft_args:
        x = np.random.randn(M, 2)
        x[:, 0] = x[:, 0] % np.pi
        x[:, 1] = x[:, 1] % (2 * np.pi)
        plan = NFSFT(N=N, x=x, **nfsft_kwargs)
        yield check_forward_nfsft, plan

def test_forward_ndsft():
    for N, M, nfsft_kwargs in tested_nfsft_args:
        x = np.random.randn(M, 2)
        x[:, 0] = x[:, 0] % np.pi
        x[:, 1] = x[:, 1] % (2 * np.pi)
        plan = NFSFT(N=N, x=x, **nfsft_kwargs)
        yield check_forward_ndsft, plan

def test_adjoint_nfsft():
    for N, M, nfsft_kwargs in tested_nfsft_args:
        x = np.random.randn(M, 2)
        x[:, 0] = x[:, 0] % np.pi
        x[:, 1] = x[:, 1] % (2 * np.pi)
        plan = NFSFT(N=N, x=x, **nfsft_kwargs)
        yield check_adjoint_nfsft, plan

def test_adjoint_ndsft():
    for N, M, nfsft_kwargs in tested_nfsft_args:
        x = np.random.randn(M, 2)
        x[:, 0] = x[:, 0] % np.pi
        x[:, 1] = x[:, 1] % (2 * np.pi)
        plan = NFSFT(N=N, x=x, **nfsft_kwargs)
        yield check_adjoint_ndsft, plan
