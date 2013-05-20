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
#
# Ghislain Vaillant
# ghislain.vallant@kcl.ac.uk

cdef extern from "nfft3.h":

    # enums
    # =====

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

    # precomputation flags for the FFTW component
    ctypedef enum:
        FFTW_ESTIMATE
        FFTW_DESTROY_INPUT


    # structs and types
    # =================

    # double precision complex type
    ctypedef double fftw_complex[2]

    # double precision NFFT plan
    ctypedef struct nfft_plan:
        int N_total
            # Total number of Fourier coefficients.
        int M_total
            # Total number of samples.
        fftw_complex *f_hat
            # Vector of Fourier coefficients, size is N_total float_types.
        fftw_complex *f
            # Vector of samples, size is M_total float types.
        int d
            # Dimension, rank.
        int *N
            # Multi bandwidth.
        int m
            # Cut-off parameter of the window function, default value is 6 \
            # (KAISER_BESSEL), 9 (SINC_POWER), 11 (B_SPLINE), 12 (GAUSSIAN).
        double *x
            # Nodes in time/spatial domain, size is $dM$ doubles.


    # functions
    # =========

    void nfft_trafo_direct (nfft_plan *ths) nogil
        # Computes a NDFT.

    void nfft_adjoint_direct (nfft_plan *ths) nogil
        # Computes an adjoint NDFT.

    void nfft_trafo (nfft_plan *ths) nogil
        # Computes a NFFT, see the definition.

    void nfft_adjoint (nfft_plan *ths) nogil
        # Computes an adjoint NFFT, see the definition.

    void nfft_init_guru (nfft_plan *ths, int d, int *N, int M, int *n, int m,
                         unsigned nfft_flags, unsigned fftw_flags)
        # Initialisation of a transform plan, guru.

    void nfft_precompute_one_psi (nfft_plan *ths) nogil
        # Precomputation for a transform plan.

    void nfft_finalize (nfft_plan *ths)
        # Destroys a transform plan.
