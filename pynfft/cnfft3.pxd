# -*- coding: utf-8 -*-
#
# Copyright (C) 2013  Ghislain Vaillant
# Copyright (C) 2014  Taco Cohen
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

    # FFTW-related declarations
    # precomputation flags for the FFTW component
    ctypedef enum:
        FFTW_ESTIMATE
        FFTW_DESTROY_INPUT

    # double precision complex type
    ctypedef double fftw_complex[2]

    ########
    # nfft #
    ########

    # precomputation flags for the NFFT component
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
        PRE_ONE_PSI #(PRE_LIN_PSI| PRE_FG_PSI| PRE_PSI| PRE_FULL_PSI)

    # double precision NFFT plan
    ctypedef struct nfft_plan:
        fftw_complex *f_hat
            # Vector of Fourier coefficients, size is N_total float_types.
        fftw_complex *f
            # Vector of samples, size is M_total float types.
        double *x
            # Nodes in time/spatial domain, size is $dM$ doubles.

    # stripped down alias of a NFFT plan used by solver
    ctypedef struct nfft_mv_plan_complex:
        pass

    void nfft_trafo_direct (nfft_plan *ths) nogil
        # Computes a NDFT.

    void nfft_adjoint_direct (nfft_plan *ths) nogil
        # Computes an adjoint NDFT.

    void nfft_trafo (nfft_plan *ths) nogil
        # Computes a NFFT, see the definition.

    void nfft_adjoint (nfft_plan *ths) nogil
        # Computes an adjoint NFFT, see the definition.

    void nfft_init_guru (nfft_plan *ths, int d, int *N, int M, int *n, int m, unsigned nfft_flags, unsigned fftw_flags)
        # Initialisation of a transform plan, guru.

    void nfft_precompute_one_psi (nfft_plan *ths) nogil
        # Precomputation for a transform plan.

    void nfft_finalize (nfft_plan *ths)
        # Destroys a transform plan.

    #########
    # nfsft #
    #########

    ctypedef struct nfsft_plan:
        fftw_complex *f_hat
            # Vector of Fourier coefficients, size is N_total float_types.
        fftw_complex *f
            # Vector of samples, size is M_total float types.
        double *x
            # Nodes in time/spatial domain, size is $dM$ doubles.

    void nfsft_init_guru(nfsft_plan *plan, int N, int M,
                         unsigned int nfsft_flags, unsigned int nfft_flags, int nfft_cutoff)

    void nfsft_trafo_direct(nfsft_plan* plan) nogil

    void nfsft_adjoint_direct(nfsft_plan* plan) nogil

    void nfsft_trafo(nfsft_plan* plan) nogil

    void nfsft_adjoint(nfsft_plan* plan) nogil

    void nfsft_finalize(nfsft_plan *plan)

    void nfsft_forget() nogil

    void nfsft_precompute_x(nfsft_plan *plan) nogil

    void nfsft_precompute(int N, double kappa, unsigned int nfsft_flags, unsigned int fpt_flags) nogil

    # init flags
    ctypedef enum:
        NFSFT_NORMALIZED      # (1U << 0)
        NFSFT_USE_NDFT        # (1U << 1)
        NFSFT_USE_DPT         # (1U << 2)
        NFSFT_MALLOC_X        # (1U << 3)
        NFSFT_MALLOC_F_HAT    # (1U << 5)
        NFSFT_MALLOC_F        # (1U << 6)
        NFSFT_PRESERVE_F_HAT  # (1U << 7)
        NFSFT_PRESERVE_X      # (1U << 8)
        NFSFT_PRESERVE_F      # (1U << 9)
        NFSFT_DESTROY_F_HAT   # (1U << 10)
        NFSFT_DESTROY_X       # (1U << 11)
        NFSFT_DESTROY_F       # (1U << 12)
    # precompute flags
        NFSFT_NO_DIRECT_ALGORITHM  # (1U << 13)
        NFSFT_NO_FAST_ALGORITHM    # (1U << 14)
        NFSFT_ZERO_F_HAT           # (1U << 16)


    ##########
    # Solver #
    ##########

    ctypedef enum:
        # precomputation flags for solver
        LANDWEBER             #(1U<< 0)
        STEEPEST_DESCENT      #(1U<< 1)
        CGNR                  #(1U<< 2)
        CGNE                  #(1U<< 3)
        NORMS_FOR_LANDWEBER   #(1U<< 4)
        PRECOMPUTE_WEIGHT     #(1U<< 5)
        PRECOMPUTE_DAMP       #(1U<< 6)

    # complex solver plan
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

    void solver_init_advanced_complex(solver_plan_complex *ths,
                                      nfft_mv_plan_complex *mv,
                                      unsigned flags)
        # Advanced initialisation.

    void solver_before_loop_complex(solver_plan_complex *ths) nogil
        # Setting up residuals before the actual iteration.

    void solver_loop_one_step_complex(solver_plan_complex *ths) nogil
        # Doing one step in the iteration.

    void solver_finalize_complex(solver_plan_complex *ths)
        # Destroys the plan for the inverse transform.




cdef extern from "fftw3.h":
    
    void fftw_cleanup()
        # cleanup routines
        
    void fftw_init_threads()
        # threading routines

    void fftw_cleanup_threads()
        # cleanup routines
