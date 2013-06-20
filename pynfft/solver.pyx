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

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from cnfft3 cimport (solver_init_advanced_complex, solver_before_loop_complex,
                     solver_loop_one_step_complex, solver_finalize_complex,
                     nfft_mv_plan_complex, solver_plan_complex)
from cnfft3 cimport (LANDWEBER, STEEPEST_DESCENT, CGNR, CGNE,
                     NORMS_FOR_LANDWEBER, PRECOMPUTE_WEIGHT,
                     PRECOMPUTE_DAMP,)
from nfft cimport NFFT


# exposes flag management internals for testing
solver_flags_dict = {
    'LANDWEBER':LANDWEBER,
    'STEEPEST_DESCENT':STEEPEST_DESCENT,
    'CGNR':CGNR,
    'CGNE':CGNE,
    'NORMS_FOR_LANDWEBER':NORMS_FOR_LANDWEBER,
    'PRECOMPUTE_WEIGHT':PRECOMPUTE_WEIGHT,
    'PRECOMPUTE_DAMP':PRECOMPUTE_DAMP,
    }
solver_flags = solver_flags_dict.copy()


cdef void *solver_init_double(void *nfft_plan, unsigned flags):
    cdef solver_plan_complex *ths = (
            <solver_plan_complex *> malloc(sizeof(solver_plan_complex)))
    if ths != NULL:
        solver_init_advanced_complex(
                ths, <nfft_mv_plan_complex*> nfft_plan, flags)
    return ths

cdef void solver_finalize_double(void *plan):
    cdef solver_plan_complex *ths = <solver_plan_complex *> plan
    solver_finalize_complex(ths)

cdef void solver_before_loop_double(void *plan):
    cdef solver_plan_complex *ths = <solver_plan_complex *> plan
    solver_before_loop_complex(ths)

cdef void solver_loop_one_step_double(void *plan):
    cdef solver_plan_complex *ths = <solver_plan_complex *> plan
    solver_loop_one_step_complex(ths)

cdef object solver_get_w_double(void *plan, np.npy_intp shape[]):
    cdef solver_plan_complex *ths = <solver_plan_complex *> plan
    return np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64,
            <void *>(ths.w))

cdef object solver_get_w_hat_double(void *plan, np.npy_intp shape[]):
    cdef solver_plan_complex *ths = <solver_plan_complex *> plan
    return np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64,
            <void *>(ths.w_hat))

cdef object solver_get_y_double(void *plan, np.npy_intp shape[]):
    cdef solver_plan_complex *ths = <solver_plan_complex *> plan
    return np.PyArray_SimpleNewFromData(1, shape, np.NPY_COMPLEX128,
            <void *>(ths.y))

cdef object solver_get_f_hat_iter_double(void *plan, np.npy_intp shape[]):
    cdef solver_plan_complex *ths = <solver_plan_complex *> plan
    return np.PyArray_SimpleNewFromData(1, shape, np.NPY_COMPLEX128,
            <void *>(ths.f_hat_iter))

cdef object solver_get_r_iter_double(void *plan, np.npy_intp shape[]):
    cdef solver_plan_complex *ths = <solver_plan_complex *> plan
    return np.PyArray_SimpleNewFromData(1, shape, np.NPY_COMPLEX128,
            <void *>(ths.r_iter))

cdef solver_generic_init solver_init_per_dtype[1]

cdef solver_generic_init* _build_solver_init_list():
    solver_init_per_dtype[0] = <solver_generic_init>&solver_init_double
    #solver_init_per_dtype[1] = <solver_generic_init>&solver_init_single
    #solver_init_per_dtype[2] = <solver_generic_init>&solver_init_ldouble

cdef solver_generic_finalize solver_finalize_per_dtype[1]

cdef solver_generic_finalize* _build_solver_finalize_list():
    solver_finalize_per_dtype[0] = <solver_generic_finalize>&solver_finalize_double
    #solver_finalize_per_dtype[1] = <solver_generic_finalize>&solver_finalize_single
    #solver_finalize_per_dtype[2] = <solver_generic_finalize>&solver_finalize_ldouble

cdef solver_generic_before_loop solver_before_loop_per_dtype[1]

cdef solver_generic_before_loop* _build_solver_before_loop_list():
    solver_before_loop_per_dtype[0] = <solver_generic_before_loop>&solver_before_loop_double
    #solver_before_loop_per_dtype[1] = <solver_generic_before_loop>&solver_before_loop_single
    #solver_before_loop_per_dtype[2] = <solver_generic_before_loop>&solver_before_loop_ldouble

cdef solver_generic_loop_one_step solver_loop_one_step_per_dtype[1]

cdef solver_generic_loop_one_step* _build_solver_loop_one_step_list():
    solver_loop_one_step_per_dtype[0] = <solver_generic_loop_one_step>&solver_loop_one_step_double
    #solver_loop_one_step_per_dtype[1] = <solver_generic_loop_one_step>&solver_loop_one_step_single
    #solver_loop_one_step_per_dtype[2] = <solver_generic_loop_one_step>&solver_loop_one_step_ldouble

cdef solver_generic_get_w solver_get_w_per_dtype[1]

cdef solver_generic_get_w* _build_solver_get_w_list():
    solver_get_w_per_dtype[0] = <solver_generic_get_w>&solver_get_w_double
    #solver_get_w_per_dtype[1] = <solver_generic_get_w>&solver_get_w_single
    #solver_get_w_per_dtype[2] = <solver_generic_get_w>&solver_get_w_ldouble

cdef solver_generic_get_w_hat solver_get_w_hat_per_dtype[1]

cdef solver_generic_get_w_hat* _build_solver_get_w_hat_list():
    solver_get_w_hat_per_dtype[0] = <solver_generic_get_w_hat>&solver_get_w_hat_double
    #solver_get_w_hat_per_dtype[1] = <solver_generic_get_w_hat>&solver_get_w_hat_single
    #solver_get_w_hat_per_dtype[2] = <solver_generic_get_w_hat>&solver_get_w_hat_ldouble

cdef solver_generic_get_y solver_get_y_per_dtype[1]

cdef solver_generic_get_y* _build_solver_get_y_list():
    solver_get_y_per_dtype[0] = <solver_generic_get_y>&solver_get_y_double
    #solver_get_y_per_dtype[1] = <solver_generic_get_y>&solver_get_y_single
    #solver_get_y_per_dtype[2] = <solver_generic_get_y>&solver_get_y_ldouble

cdef solver_generic_get_f_hat_iter solver_get_f_hat_iter_per_dtype[1]

cdef solver_generic_get_f_hat_iter* _build_solver_get_f_hat_iter_list():
    solver_get_f_hat_iter_per_dtype[0] = <solver_generic_get_f_hat_iter>&solver_get_f_hat_iter_double
    #solver_get_f_hat_iter_per_dtype[1] = <solver_generic_get_f_hat_iter>&solver_get_f_hat_iter_single
    #solver_get_f_hat_iter_per_dtype[2] = <solver_generic_get_f_hat_iter>&solver_get_f_hat_iter_ldouble

cdef solver_generic_get_r_iter solver_get_r_iter_per_dtype[1]

cdef solver_generic_get_r_iter* _build_solver_get_r_iter_list():
    solver_get_r_iter_per_dtype[0] = <solver_generic_get_r_iter>&solver_get_r_iter_double
    #solver_get_r_iter_per_dtype[1] = <solver_generic_get_r_iter>&solver_get_r_iter_single
    #solver_get_r_iter_per_dtype[2] = <solver_generic_get_r_iter>&solver_get_r_iter_ldouble


cdef object solver_complex_dtypes
solver_complex_dtypes = {
        np.dtype('float64'): np.dtype('complex128')
        #np.dtype('float32'): np.dtype('complex64')
        #np.dtype('float128'): np.dtype('complex256')
        }

cdef object solver_dtype_to_index
solver_dtype_to_index = {
        np.dtype('float64'): 0
        #np.dtype('float32'): 1
        #np.dtype('float128'): 2
        }


# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

# initialize module
_build_solver_init_list()
_build_solver_finalize_list()
_build_solver_before_loop_list()
_build_solver_loop_one_step_list()
_build_solver_get_w_list()
_build_solver_get_w_hat_list()
_build_solver_get_y_list()
_build_solver_get_f_hat_iter_list()
_build_solver_get_r_iter_list()


cdef class Solver:
    '''
    Solver is a class for computing the adjoint NFFT iteratively. Using the
    solver should theoretically lead to more accurate results, even with just
    one iteration, than using :meth:`~pynfft.nfft.NFFT.adjoint` or
    :meth:`~pynfft.nfft.NFFT.adjoint_direct`.

    The instantiation requires a NFFT object used internally for the multiple
    forward and adjoint NFFT performed. The class uses conjugate-gradient as
    the default solver but alternative solvers can be specified.

    Because the stopping conidition of the iterative computation may change
    from one application to another, the implementation only let you carry
    one iteration at a time with a call to
    :meth:`~pynfft.solver.Solver.loop_one_step`. Initialization of the solver
    is done by calling the :meth:`~pynfft.solver.Solver.before_loop` method.

    The class exposes the internals of the solver through call to their
    respective properties. They should be treated as read-only values.
    '''
    def __cinit__(self, NFFT nfft_plan, flags=None):

        # check dtype and assign function pointers accordingly
        dtype = nfft_plan.dtype
        try:
            dtype_complex = solver_complex_dtypes[dtype]
            func_idx = solver_dtype_to_index[dtype]
            self.__solver_init = solver_init_per_dtype[func_idx]
            self.__solver_finalize = solver_finalize_per_dtype[func_idx]
            self.__solver_before_loop = solver_before_loop_per_dtype[func_idx]
            self.__solver_loop_one_step = solver_loop_one_step_per_dtype[func_idx]
            self.__solver_get_w = solver_get_w_per_dtype[func_idx]
            self.__solver_get_w_hat = solver_get_w_hat_per_dtype[func_idx]
            self.__solver_get_y = solver_get_y_per_dtype[func_idx]
            self.__solver_get_f_hat_iter = solver_get_f_hat_iter_per_dtype[func_idx]
            self.__solver_get_r_iter = solver_get_r_iter_per_dtype[func_idx]
        except KeyError:
            raise ValueError('dtype %s is not supported' % dtype)

        # convert tuple of litteral precomputation flags to its expected
        # C-compatible value. Each flag is a power of 2, which allows to compute
        # this value using BITOR operations.
        cdef unsigned int _flags = 0
        flags_used = ()

        # sanity checks on user specified flags if any,
        # else use default ones:
        if flags is not None:
            try:
                flags = tuple(flags)
            except:
                flags = (flags,)
            finally:
                flags_used += flags
        else:
            flags_used += ('CGNR',)

        for each_flag in flags_used:
            try:
                _flags |= solver_flags_dict[each_flag]
            except KeyError:
                raise ValueError('Invalid flag: ' + '\'' +
                        each_flag + '\' is not a valid flag.')

        # initialize plan
        self.__plan = self.__solver_init(nfft_plan.__plan, _flags)
        if self.__plan == NULL:
            raise MemoryError

        self._dtype = dtype
        self._flags = flags_used

        cdef np.npy_intp shape[1]
        cdef int M_total = nfft_plan._M_total
        cdef int N_total = nfft_plan._N_total

        if 'PRECOMPUTE_WEIGHT' in flags_used:
            shape[0] = M_total
            self._w = self.__solver_get_w(self.__plan, shape)
            self._w[:] = 1  # make sure weights are initialized
        else:
            self._w = None

        if 'PRECOMPUTE_DAMP' in flags_used:
            shape[0] = N_total
            self._w_hat = self.__solver_get_w_hat(self.__plan, shape)
        else:
            self._w_hat = None

        shape[0] = M_total
        self._y = self.__solver_get_y(self.__plan, shape)

        shape[0] = N_total
        self._f_hat_iter = self.__solver_get_f_hat_iter(self.__plan, shape)
        self._f_hat_iter[:] = 0  # default initial guess

        shape[0] = M_total
        self._r_iter = self.__solver_get_r_iter(self.__plan, shape)


    def __init__(self, nfft_plan, flags=None):
        '''
        :param plan: instance of NFFT.
        :type plan: :class:`~pynfft.nfft.NFFT`
        :param flags: list of instantiation flags, see below.
        :type flags: tuple

        .. _instantiation_flags::

        **Instantiation flags**

        +---------------------+-----------------------------------------------------------------------------+
        | Flag                | Description                                                                 |
        +=====================+=============================================================================+
        | LANDWEBER           | Use Landweber (Richardson) iteration.                                       |
        +---------------------+-----------------------------------------------------------------------------+
        | STEEPEST_DESCENT    | Use steepest descent iteration.                                             |
        +---------------------+-----------------------------------------------------------------------------+
        | CGNR                | Use conjugate gradient (normal equation of the 1st kind).                   |
        +---------------------+-----------------------------------------------------------------------------+
        | CGNE                | Use conjugate gradient (normal equation of the 2nd kind).                   |
        +---------------------+-----------------------------------------------------------------------------+
        | NORMS_FOR_LANDWEBER | Use Landweber iteration to compute the residual norm.                       |
        +---------------------+-----------------------------------------------------------------------------+
        | PRECOMPUTE_WEIGHT   | Weight the samples, e.g. to cope with varying sampling density.             |
        +---------------------+-----------------------------------------------------------------------------+
        | PRECOMPUTE_DAMP     | Weight the Fourier coefficients, e.g. to favour fast decaying coefficients. |
        +---------------------+-----------------------------------------------------------------------------+

        Default value is ``flags = ('CGNR',)``.
        '''
        pass

    def __dealloc__(self):
        if self.__plan != NULL:
            self.__solver_finalize(self.__plan)
            free(self.__plan)

    cpdef before_loop(self):
        '''
        Initialize solver internals.
        '''
        if self.__plan != NULL:
            self.__solver_before_loop(self.__plan)

    cpdef loop_one_step(self):
        '''
        Perform one iteration.
        '''
        if self.__plan != NULL:
            self.__solver_loop_one_step(self.__plan)

    def __get_w(self):
        '''
        Weighting factors.
        '''
        return self._w

    def __set_w(self, new_w):
        if self._w is not None:
            self._w.ravel()[:] = new_w.ravel()[:]

    w = property(__get_w, __set_w)

    def __get_w_hat(self):
        '''
        Damping factors.
        '''
        return self._w_hat

    def __set_w_hat(self, new_w_hat):
        if self._w_hat is not None:
            self._w_hat.ravel()[:] = new_w_hat.ravel()[:]

    w_hat = property(__get_w_hat, __set_w_hat)

    def __get_y(self):
        '''
        Right hand side, samples.
        '''
        return self._y

    def __set_y(self, new_y):
        if self._y is not None:
            self._y.ravel()[:] = new_y.ravel()[:]

    y = property(__get_y, __set_y)

    def __get_f_hat_iter(self):
        '''
        Iterative solution.
        '''
        return self._f_hat_iter

    def __set_f_hat_iter(self, new_f_hat_iter):
        if self._f_hat_iter is not None:
            self._f_hat_iter.ravel()[:] = new_f_hat_iter.ravel()[:]

    f_hat_iter = property(__get_f_hat_iter, __set_f_hat_iter)

    def __get_r_iter(self):
        '''
        Residual vector.
        '''
        return self._r_iter

    r_iter = property(__get_r_iter)

    def __get_dtype(self):
        '''
        The floating precision.
        '''
        return self._dtype

    dtype = property(__get_dtype)

    def __get_flags(self):
        '''
        The precomputation flags.
        '''
        return self._flags

    flags = property(__get_flags)
