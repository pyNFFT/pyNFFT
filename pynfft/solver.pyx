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


# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


cdef class Solver:

    def __cinit__(self, NFFT nfft_plan, flags=None):

        # flags management
        flags_used = []
        cdef unsigned int _solver_flags = 0

        solver_flags = flags
        if solver_flags is None:
            solver_flags = ('CGNR',)

        for each_flag in solver_flags:
            try:
                _solver_flags |= solver_flags_dict[each_flag]
                flags_used.append(each_flag)
            except KeyError:
                raise ValueError('Invalid flag: ' + '\'' +
                        each_flag + '\' is not a valid flag.')

        # initialize plan
        try:
            solver_init_advanced_complex(
                <solver_plan_complex *>&self.__plan,
                <nfft_mv_plan_complex *>&(nfft_plan.__plan),
                _solver_flags)
        except:
            raise MemoryError

        self._dtype = nfft_plan._dtype
        self._flags = tuple(flags_used)

        cdef np.npy_intp shape[1]
        cdef int _M_total = nfft_plan._M_total
        cdef int _N_total = nfft_plan._N_total

        if 'PRECOMPUTE_WEIGHT' in solver_flags:
            shape[0] = _M_total
            self._w = np.PyArray_SimpleNewFromData(
                1, shape, np.NPY_FLOAT64, <void *>self.__plan.w)
            self._w[:] = 1  # make sure weights are initialized
        else:
            self._w = None

        if 'PRECOMPUTE_DAMP' in solver_flags:
            shape[0] = _N_total
            self._w_hat = np.PyArray_SimpleNewFromData(
                1, shape, np.NPY_FLOAT64, <void *>self.__plan.w_hat)
            self._w_hat[:] = 1  # make sure weights are initialized
        else:
            self._w_hat = None

        shape[0] = _M_total
        self._y = np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_COMPLEX128, <void *>self.__plan.y)

        shape[0] = _M_total
        self._r_iter = np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_COMPLEX128, <void *>self.__plan.r_iter)

        shape[0] = _N_total
        self._f_hat_iter = np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_COMPLEX128, <void *>self.__plan.f_hat_iter)
        self._f_hat_iter[:] = 0  # initial guess

        shape[0] = _N_total
        self._z_hat_iter = np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_COMPLEX128, <void *>self.__plan.z_hat_iter)

        shape[0] = _N_total
        self._p_hat_iter = np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_COMPLEX128, <void *>self.__plan.p_hat_iter)

        shape[0] = _M_total
        self._v_iter = np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_COMPLEX128, <void *>self.__plan.v_iter)


    def __init__(self, nfft_plan, flags=None):
        pass


    def __dealloc__(self):
        solver_finalize_complex(&self.__plan)


    cpdef before_loop(self):
        solver_before_loop_complex(&self.__plan)


    cpdef loop_one_step(self):
        solver_loop_one_step_complex(&self.__plan)


    def __get_w(self):
        return self._w

    def __set_w(self, new_w):
        if self._w is not None:
            self._w.ravel()[:] = new_w.ravel()[:]

    w = property(__get_w, __set_w)


    def __get_w_hat(self):
        return self._w_hat

    def __set_w_hat(self, new_w_hat):
        if self._w_hat is not None:
            self._w_hat.ravel()[:] = new_w_hat.ravel()[:]

    w_hat = property(__get_w_hat, __set_w_hat)


    def __get_y(self):
        return self._y

    def __set_y(self, new_y):
        if self._y is not None:
            self._y.ravel()[:] = new_y.ravel()[:]

    y = property(__get_y, __set_y)


    def __get_f_hat_iter(self):
        return self._f_hat_iter

    def __set_f_hat_iter(self, new_f_hat_iter):
        if self._f_hat_iter is not None:
            self._f_hat_iter.ravel()[:] = new_f_hat_iter.ravel()[:]

    f_hat_iter = property(__get_f_hat_iter, __set_f_hat_iter)


    def __get_r_iter(self):
        return self._r_iter

    r_iter = property(__get_r_iter)


    def __get_z_hat_iter(self):
        return self._z_hat_iter

    z_hat_iter = property(__get_z_hat_iter)


    def __get_p_hat_iter(self):
        return self._p_hat_iter

    p_hat_iter = property(__get_p_hat_iter)


    def __get_v_iter(self):
        return self._v_iter

    v_iter = property(__get_v_iter)


    def __get_alpha_iter(self):
        return self.__plan.alpha_iter

    alpha_iter = property(__get_alpha_iter)


    def __get_beta_iter(self):
        return self.__plan.beta_iter

    beta_iter = property(__get_beta_iter)


    def __get_dot_r_iter(self):
        return self.__plan.dot_r_iter

    dot_r_iter = property(__get_dot_r_iter)


    def __get_dot_r_iter_old(self):
        return self.__plan.dot_r_iter_old

    dot_r_iter_old = property(__get_dot_r_iter_old)


    def __get_dot_z_hat_iter(self):
        return self.__plan.dot_z_hat_iter

    dot_z_hat_iter = property(__get_dot_z_hat_iter)


    def __get_dot_z_hat_iter_old(self):
        return self.__plan.dot_z_hat_iter_old

    dot_z_hat_iter_old = property(__get_dot_z_hat_iter_old)


    def __get_dot_p_hat_iter(self):
        return self.__plan.dot_p_hat_iter

    dot_p_hat_iter = property(__get_dot_p_hat_iter)


    def __get_dot_v_iter(self):
        return self.__plan.dot_v_iter

    dot_v_iter = property(__get_dot_v_iter)


    def __get_dtype(self):
        return self._dtype

    dtype = property(__get_dtype)


    def __get_flags(self):
        return self._flags

    flags = property(__get_flags)
