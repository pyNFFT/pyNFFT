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
    ctypedef enum:
        # Precomputation flags for the NFFT component
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

    ctypedef enum:
        # Precomputation flags for the FFTW component
        FFTW_ESTIMATE
        FFTW_DESTROY_INPUT
