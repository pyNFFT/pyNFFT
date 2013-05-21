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

cdef class Solver:

    def __cinit__(self):
        pass

    def __init__(self):
        pass

    def __dealloc__(self):
        pass

    cpdef before_loop(self):
        pass

    cpdef loop_one_step(self):
        pass

    def __get_w(self):
        pass

    def __set_w(self, new_w):
        pass

    w = property(__get_w, __set_w)

    def __get_w_hat(self):
        pass

    def __set_w_hat(self, new_w_hat):
        pass

    w_hat = property(__get_w_hat, __set_w_hat)

    def __get_f_hat_iter(self):
        pass

    def __set_f_hat_iter(self, new_f_hat_iter):
        pass

    f_hat_iter = property(__get_f_hat_iter, __set_f_hat_iter)

    def __get_r_iter(self):
        pass

    def __set_r_iter(self, new_r_iter):
        pass

    r_iter = property(__get_r_iter, __set_r_iter)

    def __get_p_hat_iter(self):
        pass

    def __set_p_hat_iter(self, new_p_hat_iter):
        pass

    p_hat_iter = property(__get_p_hat_iter, __set_p_hat_iter)
