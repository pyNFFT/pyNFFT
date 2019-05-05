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


cdef extern from "nfft3.h":

    # --- Precomputation flags --- #

    # for the NFFT component
    ctypedef enum:
        PRE_PHI_HUT      #(1U<< 0)
        FG_PSI           #(1U<< 1)
        PRE_LIN_PSI      #(1U<< 2)
        PRE_FG_PSI       #(1U<< 3)
        PRE_PSI          #(1U<< 4)
        PRE_FULL_PSI     #(1U<< 5)
        MALLOC_X         #(1U<< 6)
        MALLOC_F_HAT     #(1U<< 7)
        MALLOC_F         #(1U<< 8)
        FFT_OUT_OF_PLACE #(1U<< 9)
        FFTW_INIT        #(1U<< 10)
        NFFT_SORT_NODES  #(1U<< 11)
        NFFT_OMP_BLOCKWISE_ADJOINT #(1U<<12)
        PRE_ONE_PSI  # (PRE_LIN_PSI | PRE_FG_PSI | PRE_PSI | PRE_FULL_PSI)

    # for the FFTW component
    ctypedef enum:
        FFTW_ESTIMATE
        FFTW_DESTROY_INPUT

    # --- Complex data types --- #

    ctypedef double fftw_complex[2]
    ctypedef float fftwf_complex[2]
    ctypedef long double fftwl_complex[2]

    # --- NFFT plans --- #

    # double precision
    ctypedef struct nfft_plan:
        fftw_complex *f_hat
            # Vector of Fourier coefficients, size is N_total
        fftw_complex *f
            # Vector of samples, size is M_total
        double *x
            # Nodes in time/spatial domain, size is d*M

    # single precision
    ctypedef struct nfftf_plan:
        fftwf_complex *f_hat
        fftwf_complex *f
        float *x

    # extended precision
    ctypedef struct nfftl_plan:
        fftwl_complex *f_hat
        fftwl_complex *f
        long double *x

    # --- Transform functions --- #

    # double precision

    void nfft_trafo_direct(nfft_plan *ths) nogil
        # Computes an NDFT

    void nfft_adjoint_direct(nfft_plan *ths) nogil
        # Computes an adjoint NDFT

    void nfft_trafo(nfft_plan *ths) nogil
        # Computes an NFFT, see the definition

    void nfft_adjoint(nfft_plan *ths) nogil
        # Computes an adjoint NFFT, see the definition

    # single precision

    void nfftf_trafo_direct(nfftf_plan *ths) nogil
    void nfftf_adjoint_direct(nfftf_plan *ths) nogil
    void nfftf_trafo(nfftf_plan *ths) nogil
    void nfftf_adjoint(nfftf_plan *ths) nogil

    # extended precision

    void nfftl_trafo_direct(nfftl_plan *ths) nogil
    void nfftl_adjoint_direct(nfftl_plan *ths) nogil
    void nfftl_trafo(nfftl_plan *ths) nogil
    void nfftl_adjoint(nfftl_plan *ths) nogil

    # --- Advanced interface --- #

    # double precision

    void nfft_init_guru(nfft_plan *ths, int d, int *N, int M, int *n, int m,
                        unsigned int nfft_flags, unsigned int fftw_flags)
        # Initialisation of a transform plan, guru

    void nfft_precompute_one_psi(nfft_plan *ths) nogil
        # Precomputation for a transform plan

    # single precision

    void nfftf_init_guru(nfftf_plan *ths, int d, int *N, int M, int *n, int m,
                         unsigned int nfft_flags, unsigned int fftw_flags)
    void nfftf_precompute_one_psi (nfftf_plan *ths) nogil

    # extended precision
    void nfftl_init_guru(nfftl_plan *ths, int d, int *N, int M, int *n, int m,
                         unsigned int nfft_flags, unsigned int fftw_flags)
    void nfftl_precompute_one_psi(nfftl_plan *ths) nogil

    # --- Cleanup --- #

    void nfft_finalize(nfft_plan *ths)
        # Destroys a transform plan
    void nfftf_finalize(nfftf_plan *ths)
    void nfftl_finalize(nfftl_plan *ths)

    # --- Solver precomputation flags --- #

    ctypedef enum:
        LANDWEBER             #(1U<< 0)
        STEEPEST_DESCENT      #(1U<< 1)
        CGNR                  #(1U<< 2)
        CGNE                  #(1U<< 3)
        NORMS_FOR_LANDWEBER   #(1U<< 4)
        PRECOMPUTE_WEIGHT     #(1U<< 5)
        PRECOMPUTE_DAMP       #(1U<< 6)

    # --- Solver plans --- #

    # stripped down aliases of NFFT plans used by solver
    ctypedef struct nfft_mv_plan_complex:
        pass
    ctypedef struct nfftf_mv_plan_complex:
        pass
    ctypedef struct nfftl_mv_plan_complex:
        pass

    # complex solver plan, double precision
    ctypedef struct solver_plan_complex:
        nfft_mv_plan_complex *mv
            # matrix vector multiplication.
        unsigned flags
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
            # residual of normal equation of \ first kind
        fftw_complex *p_hat_iter
            # search direction.
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
            #  weighted dotproduct of \ z_hat_iter
        double dot_z_hat_iter_old
            #  previous dot_z_hat_iter
        double dot_p_hat_iter
            #  weighted dotproduct of \ p_hat_iter
        double dot_v_iter
            # weighted dotproduct of v_iter

    # complex solver plan, single precision
    ctypedef struct solverf_plan_complex:
        nfftf_mv_plan_complex *mv
        unsigned flags
        float *w
        float *w_hat
        fftwf_complex *y
        fftwf_complex *f_hat_iter
        fftwf_complex *r_iter
        fftwf_complex *z_hat_iter
        fftwf_complex *p_hat_iter
        fftwf_complex *v_iter
        float alpha_iter
        float beta_iter
        float dot_r_iter
        float dot_r_iter_old
        float dot_z_hat_iter
        float dot_z_hat_iter_old
        float dot_p_hat_iter
        float dot_v_iter

    # complex solver plan, extended precision
    ctypedef struct solverl_plan_complex:
        nfftl_mv_plan_complex *mv
        unsigned flags
        long double *w
        long double *w_hat
        fftwl_complex *y
        fftwl_complex *f_hat_iter
        fftwl_complex *r_iter
        fftwl_complex *z_hat_iter
        fftwl_complex *p_hat_iter
        fftwl_complex *v_iter
        long double alpha_iter
        long double beta_iter
        long double dot_r_iter
        long double dot_r_iter_old
        long double dot_z_hat_iter
        long double dot_z_hat_iter_old
        long double dot_p_hat_iter
        long double dot_v_iter

    # --- Advanced interface --- #

    # double precision

    void solver_init_advanced_complex(solver_plan_complex *ths,
                                      nfft_mv_plan_complex *mv,
                                      unsigned flags)
        # Advanced initialisation.

    void solver_before_loop_complex(solver_plan_complex *ths) nogil
        # Setting up residuals before the actual iteration.

    void solver_loop_one_step_complex(solver_plan_complex *ths) nogil
        # Doing one step in the iteration.

    # single precision

    void solverf_init_advanced_complex(solverf_plan_complex *ths,
                                       nfftf_mv_plan_complex *mv,
                                       unsigned flags)
    void solverf_before_loop_complex(solverf_plan_complex *ths) nogil
    void solverf_loop_one_step_complex(solverf_plan_complex *ths) nogil

    # extended precision

    void solverl_init_advanced_complex(solverl_plan_complex *ths,
                                       nfftl_mv_plan_complex *mv,
                                       unsigned flags)
    void solverl_before_loop_complex(solverl_plan_complex *ths) nogil
    void solverl_loop_one_step_complex(solverl_plan_complex *ths) nogil

    # --- Cleanup --- #

    void solver_finalize_complex(solver_plan_complex *ths)
        # Destroys the plan for the inverse transform.
    void solverf_finalize_complex(solverf_plan_complex *ths)
    void solverl_finalize_complex(solverl_plan_complex *ths)


cdef extern from "fftw3.h":

    void fftw_cleanup()
        # cleanup routines

    void fftw_init_threads()
        # threading routines

    void fftw_cleanup_threads()
        # cleanup routines

    void fftwf_cleanup()
    void fftwf_init_threads()
    void fftwf_cleanup_threads()

    void fftwl_cleanup()
    void fftwl_init_threads()
    void fftwl_cleanup_threads()
