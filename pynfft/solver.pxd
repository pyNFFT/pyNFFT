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

from cnfft3 cimport solver_plan_complex

cdef class Solver:
    cdef solver_plan_complex __plan
    cdef object _w
    cdef object _w_hat
    cdef object _y
    cdef object _f_hat_iter
    cdef object _r_iter
    cdef object _p_hat_iter
    cdef object _dtype
    cdef object _flags
    cpdef before_loop(self)
    cpdef loop_one_step(self)
