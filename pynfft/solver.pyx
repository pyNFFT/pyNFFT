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

import numpy as np
cimport numpy as np
from cnfft3 cimport *
from nfft cimport NFFT

cdef object solver_flags_dict
solver_flags_dict = {
    'LANDWEBER':LANDWEBER,
    'STEEPEST_DESCENT':STEEPEST_DESCENT,
    'CGNR':CGNR,
    'CGNE':CGNE,
    'NORMS_FOR_LANDWEBER':NORMS_FOR_LANDWEBER,
    'PRECOMPUTE_WEIGHT':PRECOMPUTE_WEIGHT,
    'PRECOMPUTE_DAMP':PRECOMPUTE_DAMP,
    }
_solver_flags_dict = solver_flags_dict.copy()


cdef class Solver:

    def __cinit__(self, nfft_plan, flags=None):

        # flags management
        flags_used = []
        cdef unsigned int _solver_flags = 0

        solver_flags = flags
        if solver_flags is None:
            solver_flags = (
                    'CGNR',
                    'PRECOMPUTE_WEIGHT',
                    'PRECOMPUTE_DAMP',
                    )

        for each_flag in solver_flags:
            try:
                _solver_flags |= solver_flags_dict[each_flag]
                flags_used.append(each_flag)
            except KeyError:
                raise ValueError('Invalid flag: ' + '\'' +
                        each_flag + '\' is not a valid flag.')

        # initialize plan
        cdef NFFT _nfft_plan = nfft_plan
        cdef int _M_total = _nfft_plan._M_total
        cdef int _N_total = _nfft_plan._M_total

        solver_init_advanced_complex(
            &self.__plan, <nfft_mv_plan_complex *>&(_nfft_plan.__plan),
            _solver_flags)

        self._dtype = _nfft_plan.dtype
        self._flags = tuple(flags_used)

        cdef np.npy_intp shape[1]
        shape[0] = _M_total
        self._w = np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_FLOAT64, <void *>self.__plan.w)
        shape[0] = _N_total
        self._w_hat = np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_FLOAT64, <void *>self.__plan.w_hat)
        shape[0] = _M_total
        self._y = np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_COMPLEX128, <void *>self.__plan.y)
        shape[0] = _N_total
        self._f_hat_iter = np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_COMPLEX128, <void *>self.__plan.f_hat_iter)
        shape[0] = _M_total
        self._r_iter = np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_COMPLEX128, <void *>self.__plan.r_iter)
        shape[0] = _N_total
        self._p_hat_iter = np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_COMPLEX128, <void *>self.__plan.p_hat_iter)

    def __init__(self, nfft_plan, flags=None):
        pass

    def __dealloc__(self):
        solver_finalize_complex(&self.__plan)

    cpdef before_loop(self):
        solver_before_loop_complex(&self.__plan)

    cpdef loop_one_step(self):
        solver_loop_one_step_complex(&self.__plan)

    def __get_w(self):
        return self._w.copy()

    def __set_w(self, new_w):
        if new_w is not None and new_w is not self._w:
            if (<object>new_w).size != self._w.size:
                raise ValueError("Incompatible input")
            self._w[:] = new_w.ravel()[:]

    w = property(__get_w, __set_w)

    def __get_w_hat(self):
        return self._w_hat.copy()

    def __set_w_hat(self, new_w_hat):
        if new_w_hat is not None and new_w_hat is not self._w_hat:
            if (<object>new_w_hat).size != self._w_hat.size:
                raise ValueError("Incompatible input")
            self._w_hat[:] = new_w_hat.ravel()[:]

    w_hat = property(__get_w_hat, __set_w_hat)

    def __get_f_hat_iter(self):
        return self._f_hat_iter.copy()

    def __set_f_hat_iter(self, new_f_hat_iter):
        if new_f_hat_iter is not None and new_f_hat_iter is not self._f_hat_iter:
            if (<object>new_f_hat_iter).size != self._f_hat_iter.size:
                raise ValueError("Incompatible input")
            self._f_hat_iter[:] = new_f_hat_iter.ravel()[:]

    f_hat_iter = property(__get_f_hat_iter, __set_f_hat_iter)

    def __get_r_iter(self):
        return self._r_iter.copy()

    def __set_r_iter(self, new_r_iter):
        if new_r_iter is not None and new_r_iter is not self._r_iter:
            if (<object>new_r_iter).size != self._r_iter.size:
                raise ValueError("Incompatible input")
            self._r_iter[:] = new_r_iter.ravel()[:]

    r_iter = property(__get_r_iter, __set_r_iter)

    def __get_p_hat_iter(self):
        return self._p_hat_iter.copy()

    def __set_p_hat_iter(self, new_p_hat_iter):
        if new_p_hat_iter is not None and new_p_hat_iter is not self._p_hat_iter:
            if (<object>new_p_hat_iter).size != self._p_hat_iter.size:
                raise ValueError("Incompatible input")
            self._p_hat_iter[:] = new_p_hat_iter.ravel()[:]

    p_hat_iter = property(__get_p_hat_iter, __set_p_hat_iter)
