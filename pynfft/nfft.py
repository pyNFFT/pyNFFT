# -*- coding: utf-8 -*-
#
# Copyright (c) 2013, 2014 Ghislain Antony Vaillant
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

__all__ = ('NFFT',)

import numpy
from pynfft.nfft_plan import (nfft_plan_proxy, nfft_plan_flags,
                              fftw_plan_flags)


class NFFT(object):

    _allowed_flags = ('PRE_PHI_HUT',
                      'FG_PSI',
                      'PRE_LIN_PSI',
                      'PRE_FG_PSI',
                      'PRE_PSI',
                      'PRE_FULL_PSI',
                      'FFT_OUT_OF_PLACE',
                      'FFTW_INIT',
                      'NFFT_SORT_NODES',
                      'NFFT_OMP_BLOCKWISE_ADJOINT',
                      'PRE_ONE_PSI')

    def __init__(self, N, M, n=None, m=6, flags=None, f_hat=None, f=None,
                 x=None, precompute=False, *args, **kwargs):
        # extract dimensionality of the transform
        d = len(N)
        # store shape, dtype and floating precision of operator
        self._shape = tuple([N, M])
        self._dtype = numpy.dtype('complex128')
        self._precision = numpy.dtype('float64')
        # (optional) calculate the FFTW length
        nextpow2 = lambda x: 2 ** numpy.ceil(numpy.log2(x))
        n = n if n is not None else [nextpow2(Nt) for Nt in N]
        # control default and valid flags
        if flags is not None:
            flags = [flag for flag in flags if flag in self._allowed_flags]
        else:
            # same logic as in nfft_init
            flags = ['PRE_PHI_HUT', 'PRE_PSI', 'FFTW_INIT',
                     'FFT_OUT_OF_PLACE', 'NFFT_SORT_NODES',]
            if d > 1:
                flags += ('NFFT_OMP_BLOCKWISE_ADJOINT',)
        if f_hat is None:
            flags += ('MALLOC_F_HAT',)
        if f is None:
            flags += ('MALLOC_F',)
        if x is None:
            flags += ('MALLOC_X',)
        # same logic as in nfft_init
        flags += ('FFTW_ESTIMATE', 'FFTW_DESTROY')
        # store flags
        self._flags = tuple(flags)
        # convert list of flags to integer parameters
        nfft_flags = 0
        fftw_flags = 0
        for flag in flags:
            if flag in nfft_plan_flags.keys():
                nfft_flags |= nfft_plan_flags[flag]
            if flag in fftw_plan_flags.keys():
                fftw_flags |= fftw_plan_flags[flag]
        # instantiate internal plan object
        self._plan = nfft_plan_proxy.init_guru(d, N, M, n, m, nfft_flags,
                                               fftw_flags)
        # set internal arrays with optional array arguments
        self._update_arrays(f_hat=f_hat, f=f, x=x)
        # (optional) precompute the plan right after creation
        if precompute:
            self.precompute()

    @classmethod
    def from_arrays(cls, f_hat, f, **kwargs):
        return cls(N=f_hat.shape, M=f.size, f_hat=f_hat, f=f, **kwargs)

    def trafo(self, f_hat=None, f=None, use_dft=False):
        self._update_arrays(f_hat=f_hat, f=f)
        self._plan.check()
        if use_dft:
            self._plan.trafo_direct()
        else:
            self._plan.trafo()
        return self.f

    def adjoint(self, f=None, f_hat=None, use_dft=False):
        self._update_arrays(f_hat=f_hat, f=f)
        self._plan.check()
        if use_dft:
            self._plan.adjoint_direct()
        else:
            self._plan.adjoint()
        return self.f_hat

    def precompute(self, x=None):
        self._update_arrays(x=x)
        self.plan.precompute()

    def _update_arrays(self, f_hat=None, f=None, x=None):
        if f_hat is not None:
            self._plan.f_hat = f_hat
        if f is not None:
            self._plan.f = f
        if x is not None:
            self._plan.x = x

    @property
    def plan(self):
        return self._plan

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def precision(self):
        return self._precision

    @property
    def flags(self):
        return self._flags

    @property
    def N_total(self):
        return self.plan.N_total

    @property
    def M_total(self):
        return self.plan.M_total

    @property
    def f_hat(self):
        return self._plan.f_hat.reshape(self.shape[0])

    @f_hat.setter
    def f_hat(self, value):
        self._update_arrays(f_hat=value)

    @property
    def f(self):
        return self._plan.f.reshape(self.shape[1])

    @f.setter
    def f(self, value):
        self._update_arrays(f=value)

    @property
    def d(self):
        return self.plan.d

    @property
    def x(self):
        return self._plan.x.reshape([self.shape[1], self.d])

    @x.setter
    def x(self, value):
        self._update_arrays(x=value)
