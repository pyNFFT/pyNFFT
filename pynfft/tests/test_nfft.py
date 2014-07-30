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
from numpy import pi
from numpy.testing import assert_allclose
from pynfft.nfft import NFFT
from pynfft.util import vrand_unit_complex, vrand_shifted_unit_double


def fdft(x, f_hat):
    N = f_hat.shape
    d = x.shape[-1]
    k = numpy.mgrid[[slice(-Nt/2, Nt/2) for Nt in N]]
    k = k.reshape([d, -1])
    x = x.reshape([-1, d])
    F = numpy.exp(-2j * pi * numpy.dot(x, k))
    f_dft = numpy.dot(F, f_hat.ravel())
    return f_dft

def rdft(x, f, N):
    d = x.shape[-1]
    k = numpy.mgrid[[slice(-Nt/2, Nt/2) for Nt in N]]
    k = k.reshape([d, -1])
    x = x.reshape([-1, d])
    F = numpy.exp(-2j * pi * numpy.dot(x, k))
    f_hat_dft = numpy.dot(numpy.conjugate(F).T, f)
    f_hat = f_hat_dft.reshape(N)        
    return f_hat

def check_forward_nfft(plan):
    vrand_unit_complex(plan.f_hat.ravel())
    assert_allclose(plan.trafo(), fdft(plan.x, plan.f_hat))

def check_forward_ndft(plan):
    vrand_unit_complex(plan.f_hat.ravel())
    assert_allclose(plan.trafo(use_dft=True), fdft(plan.x, plan.f_hat))

def check_adjoint_nfft(plan):
    vrand_unit_complex(plan.f.ravel())
    assert_allclose(plan.adjoint(), rdft(plan.x, plan.f, plan.N))

def check_adjoint_ndft(plan):
    vrand_unit_complex(plan.f.ravel())
    assert_allclose(plan.adjoint(use_dft=True), rdft(plan.x, plan.f, plan.N))

tested_nfft_args = (
    (8, 8, dict(m=6)),
    (16, 16, dict()),
    (24, 24, dict()),
    (32, 32, dict()),
    (64, 64, dict()),
    ((8, 8), 8*8, dict(m=6)),
    ((16, 16), 16*16, dict()),
    ((24, 24), 24*24, dict()),
    ((32, 32), 32*32, dict()),
    ((64, 64), 64*64, dict()),
    ((8, 8, 8), 8*8*8, dict(m=6)),
    ((16, 16, 8), 8*8*8, dict(m=6)),
    ((16, 16, 16), 16*16*16, dict()),
)

def test_forward_nfft():
    for N, M, nfft_kwargs in tested_nfft_args:
        plan = NFFT(N, M, **nfft_kwargs)
        vrand_shifted_unit_double(plan.x.ravel())
        plan.precompute()
        yield check_forward_nfft, plan

def test_forward_ndft():
    for N, M, nfft_kwargs in tested_nfft_args:
        plan = NFFT(N, M, **nfft_kwargs)
        vrand_shifted_unit_double(plan.x.ravel())
        plan.precompute()
        yield check_forward_ndft, plan

def test_adjoint_nfft():
    for N, M, nfft_kwargs in tested_nfft_args:
        plan = NFFT(N, M, **nfft_kwargs)
        vrand_shifted_unit_double(plan.x.ravel())
        plan.precompute()
        yield check_adjoint_nfft, plan

def test_adjoint_ndft():
    for N, M, nfft_kwargs in tested_nfft_args:
        plan = NFFT(N, M, **nfft_kwargs)
        vrand_shifted_unit_double(plan.x.ravel())
        plan.precompute()
        yield check_adjoint_ndft, plan

