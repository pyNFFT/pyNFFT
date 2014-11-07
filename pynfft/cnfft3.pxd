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

from cfftw3 cimport fftw_complex


cdef extern from "nfft3.h":

    # malloc routines
    void *nfft_malloc(size_t)
    void nfft_free(void*)

    # base plan structure common to any nfft-like plan
    ctypedef struct nfft_mv_plan_complex:
        int N_total
        int M_total
        fftw_complex *f_hat
        fftw_complex *f
        void (*mv_trafo)(void*) nogil
        void (*mv_adjoint)(void*) nogil

    # precomputation flags for the NFFT plan
    ctypedef enum:
        PRE_PHI_HUT
        FG_PSI
        PRE_LIN_PSI
        PRE_FG_PSI
        PRE_PSI
        PRE_FULL_PSI
        MALLOC_X
        MALLOC_F_HAT
        MALLOC_F
        FFT_OUT_OF_PLACE
        FFTW_INIT
        NFFT_SORT_NODES
        NFFT_OMP_BLOCKWISE_ADJOINT
        PRE_ONE_PSI

    # double precision NFFT plan
    ctypedef struct nfft_plan:
        int N_total
            # Total number of Fourier coefficients.
        int M_total
            # Total number of samples.
        fftw_complex *f_hat
            # Vector of Fourier coefficients, size is N_total.
        fftw_complex *f
            # Vector of samples, size is M_total.
        int d
            # Dimension, rank.
        int *N
            # Multi bandwidth.
        int *n
            # FFTW length, equal to sigma*N.
        int m
            # Cut-off parameter of the window function.
        unsigned int nfft_flags
            # Flags for precomputation.
        unsigned int fftw_flags
            # Flags for the FFTW.
        double *x
            # Nodes in time/spatial domain, size is M_total.

    void nfft_trafo_direct(nfft_plan *) nogil
        # Computes a NDFT.

    void nfft_adjoint_direct(nfft_plan *) nogil
        # Computes an adjoint NDFT.

    void nfft_trafo(nfft_plan *) nogil
        # Computes a NFFT, see the definition.

    void nfft_adjoint(nfft_plan *) nogil
        # Computes an adjoint NFFT, see the definition.

    void nfft_init_1d(nfft_plan *, int, int)
        # Initialisation of a transform plan, wrapper d=1.

    void nfft_init_2d(nfft_plan *, int, int, int)
        # Initialisation of a transform plan, wrapper d=2.

    void nfft_init_3d(nfft_plan *, int, int, int, int)
        # Initialisation of a transform plan, wrapper d=3.

    void nfft_init(nfft_plan *, int, int *, int)
        # Initialisation of a transform plan, simple.

    void nfft_init_guru(nfft_plan *, int, int *, int, int *, int,
                        unsigned int, unsigned int)
        # Initialisation of a transform plan, guru.

    void nfft_precompute_one_psi(nfft_plan *) nogil
        # Precomputation for a transform plan.

    const char* nfft_check(nfft_plan *)
        # Checks a transform plan for frequently used bad parameter.

    void nfft_finalize(nfft_plan *)
        # Destroys a transform plan.

    # precomputation flags for solver plan
    ctypedef enum:
        LANDWEBER
        STEEPEST_DESCENT
        CGNR
        CGNE
        NORMS_FOR_LANDWEBER
        PRECOMPUTE_WEIGHT
        PRECOMPUTE_DAMP

    # complex solver plan
    ctypedef struct solver_plan_complex:
        nfft_mv_plan_complex *mv
            # matrix vector multiplication.
        unsigned int flags
            # iteration type
        double *w
            # weighting factors
        double *w_hat
            # damping factors
        fftw_complex *y
            # right hand side, samples
        fftw_complex *f_hat_iter
            # iterative solution
        fftw_complex *r_iter
            # iterated residual vector
        fftw_complex *z_hat_iter
            # residual of normal equation of the first kind
        fftw_complex *p_hat_iter
            # search direction
        fftw_complex *v_iter
            # residual vector update
        double alpha_iter
            #  step size for search direction
        double beta_iter
            #  step size for search correction
        double dot_r_iter
            #  weighted dotproduct of r_iter
        double dot_r_iter_old
            #  previous dot_r_iter
        double dot_z_hat_iter
            #  weighted dotproduct of z_hat_iter
        double dot_z_hat_iter_old
            #  previous dot_z_hat_iter
        double dot_p_hat_iter
            #  weighted dotproduct of p_hat_iter
        double dot_v_iter
            # weighted dotproduct of v_iter

    void solver_init_advanced_complex(solver_plan_complex *,
                                      nfft_mv_plan_complex *,
                                      unsigned int)
        # Advanced initialisation.

    void solver_before_loop_complex(solver_plan_complex *) nogil
        # Setting up residuals before the actual iteration.

    void solver_loop_one_step_complex(solver_plan_complex *) nogil
        # Doing one step in the iteration.

    void solver_finalize_complex(solver_plan_complex *)
        # Destroys the plan for the inverse transform.