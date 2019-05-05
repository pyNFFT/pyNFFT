# -*- coding: utf-8 -*-
#
# Copyright (C) 2013-2014  Ghislain Vaillant
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

import copy
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from cnfft3 cimport *
from .nfft cimport NFFT
from .nfft import NFFT

# Initialize module
# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

solver_flags_dict = {
    'LANDWEBER': LANDWEBER,
    'STEEPEST_DESCENT': STEEPEST_DESCENT,
    'CGNR': CGNR,
    'CGNE': CGNE,
    'NORMS_FOR_LANDWEBER': NORMS_FOR_LANDWEBER,
    'PRECOMPUTE_WEIGHT': PRECOMPUTE_WEIGHT,
    'PRECOMPUTE_DAMP': PRECOMPUTE_DAMP,
}

solver_flags = copy.copy(solver_flags_dict)


cdef class Solver(object):
    '''
    Solver is a class for computing the inverse NFFT iteratively.

    The solver's instantiation requires an initialized NFFT object used
    internally for the multiple forward and adjoint NFFT performed. The class
    uses conjugate-gradient as the default solver but alternative solvers may
    be specified at construct-time.

    The solver must be first initialized by calling the :meth:`before_loop`
    method.

    The solver's implementation lets you carry one iteration at a time with a
    call to :meth:`loop_one_step`. It is left to the user to chose whichever
    stopping condition to apply.

    The class exposes the internals of the solver through its respective
    properties. For instance, the residuals for the current iteration can be
    accessed via the :attr:r_iter attribute.
    '''

    def __cinit__(self, NFFT nfft_plan, flags=None):

        dtype_complex = nfft_plan._dtype
        dtype_real = np.dtype(dtype_complex.char.lower())

        # Convert tuple of literal precomputation flags to its expected
        # C-compatible value. Each flag is a power of 2, which allows to compute
        # this value using BITOR operations.
        cdef unsigned int _flags = 0
        flags_used = ('PRECOMPUTE_WEIGHT', 'PRECOMPUTE_DAMP')

        # Sanity checks on user specified flags if any,
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

        for flag in flags_used:
            try:
                _flags |= solver_flags_dict[flag]
            except KeyError:
                raise ValueError("Invalid flag: '{}'".format(flag))

        # Initialize plan
        try:
            if dtype_complex == np.complex64:
                solverf_init_advanced_complex(
                    &self._plan_flt, <nfftf_mv_plan_complex*>&(nfft_plan._plan_flt), _flags
                )
            elif dtype_complex == np.complex128:
                solver_init_advanced_complex(
                    &self._plan_dbl, <nfft_mv_plan_complex*>&(nfft_plan._plan_dbl), _flags
                )
            elif dtype_complex == np.complex256:
                solverl_init_advanced_complex(
                    &self._plan_ldbl, <nfftl_mv_plan_complex*>&(nfft_plan._plan_ldbl), _flags
                )
            else:
                raise RuntimeError
        except:
            raise MemoryError

        self._nfft_plan = nfft_plan
        d = nfft_plan.d
        M = nfft_plan.M
        N = nfft_plan.N

        cdef np.npy_intp shape_M[1]
        shape_M[0] = M

        cdef np.npy_intp *shape_N
        try:
            shape_N = <np.npy_intp*>malloc(d*sizeof(np.npy_intp))
        except:
            raise MemoryError
        for dt in range(d):
            shape_N[dt] = N[dt]

        if dtype_complex == np.complex64:
            self._w = np.PyArray_SimpleNewFromData(
                1, shape_M, np.NPY_FLOAT32, <void *>(self._plan_flt.w)
            )
            self._w.ravel()[:] = 1  # make sure weights are initialized
            self._w_hat = np.PyArray_SimpleNewFromData(
                d, shape_N, np.NPY_FLOAT32, <void *>(self._plan_flt.w_hat)
            )
            self._w_hat.ravel()[:] = 1  # make sure weights are initialized
            self._y = np.PyArray_SimpleNewFromData(
                1, shape_M, np.NPY_COMPLEX64, <void *>(self._plan_flt.y)
            )
            self._f_hat_iter = np.PyArray_SimpleNewFromData(
                d, shape_N, np.NPY_COMPLEX64, <void *>(self._plan_flt.f_hat_iter)
            )
            self._f_hat_iter.ravel()[:] = 0  # default initial guess
            self._r_iter = np.PyArray_SimpleNewFromData(
                1, shape_M, np.NPY_COMPLEX64, <void *>(self._plan_flt.r_iter)
            )
            self._z_hat_iter = np.PyArray_SimpleNewFromData(
                d, shape_N, np.NPY_COMPLEX64, <void *>(self._plan_flt.z_hat_iter)
            )
            self._p_hat_iter = np.PyArray_SimpleNewFromData(
                d, shape_N, np.NPY_COMPLEX64, <void *>(self._plan_flt.p_hat_iter)
            )
            self._v_iter = np.PyArray_SimpleNewFromData(
                1, shape_M, np.NPY_COMPLEX64, <void *>(self._plan_flt.v_iter)
            )

        elif dtype_complex == np.complex128:
            self._w = np.PyArray_SimpleNewFromData(
                1, shape_M, np.NPY_FLOAT64, <void *>(self._plan_dbl.w)
            )
            self._w.ravel()[:] = 1  # make sure weights are initialized
            self._w_hat = np.PyArray_SimpleNewFromData(
                d, shape_N, np.NPY_FLOAT64, <void *>(self._plan_dbl.w_hat)
            )
            self._w_hat.ravel()[:] = 1  # make sure weights are initialized
            self._y = np.PyArray_SimpleNewFromData(
                1, shape_M, np.NPY_COMPLEX128, <void *>(self._plan_dbl.y)
            )
            self._f_hat_iter = np.PyArray_SimpleNewFromData(
                d, shape_N, np.NPY_COMPLEX128, <void *>(self._plan_dbl.f_hat_iter)
            )
            self._f_hat_iter.ravel()[:] = 0  # default initial guess
            self._r_iter = np.PyArray_SimpleNewFromData(
                1, shape_M, np.NPY_COMPLEX128, <void *>(self._plan_dbl.r_iter)
            )
            self._z_hat_iter = np.PyArray_SimpleNewFromData(
                d, shape_N, np.NPY_COMPLEX128, <void *>(self._plan_dbl.z_hat_iter)
            )
            self._p_hat_iter = np.PyArray_SimpleNewFromData(
                d, shape_N, np.NPY_COMPLEX128, <void *>(self._plan_dbl.p_hat_iter)
            )
            self._v_iter = np.PyArray_SimpleNewFromData(
                1, shape_M, np.NPY_COMPLEX128, <void *>(self._plan_dbl.v_iter)
            )

        elif dtype_complex == np.complex256:
            self._w = np.PyArray_SimpleNewFromData(
                1, shape_M, np.NPY_FLOAT128, <void *>(self._plan_ldbl.w)
            )
            self._w.ravel()[:] = 1  # make sure weights are initialized
            self._w_hat = np.PyArray_SimpleNewFromData(
                d, shape_N, np.NPY_FLOAT128, <void *>(self._plan_ldbl.w_hat)
            )
            self._w_hat.ravel()[:] = 1  # make sure weights are initialized
            self._y = np.PyArray_SimpleNewFromData(
                1, shape_M, np.NPY_COMPLEX256, <void *>(self._plan_ldbl.y)
            )
            self._f_hat_iter = np.PyArray_SimpleNewFromData(
                d, shape_N, np.NPY_COMPLEX256, <void *>(self._plan_ldbl.f_hat_iter)
            )
            self._f_hat_iter.ravel()[:] = 0  # default initial guess
            self._r_iter = np.PyArray_SimpleNewFromData(
                1, shape_M, np.NPY_COMPLEX256, <void *>(self._plan_ldbl.r_iter)
            )
            self._z_hat_iter = np.PyArray_SimpleNewFromData(
                d, shape_N, np.NPY_COMPLEX256, <void *>(self._plan_ldbl.z_hat_iter)
            )
            self._p_hat_iter = np.PyArray_SimpleNewFromData(
                d, shape_N, np.NPY_COMPLEX256, <void *>(self._plan_ldbl.p_hat_iter)
            )
            self._v_iter = np.PyArray_SimpleNewFromData(
                1, shape_M, np.NPY_COMPLEX256, <void *>(self._plan_ldbl.v_iter)
            )

        else:
            raise RuntimeError

        free(shape_N)

        self._dtype = dtype_complex
        self._flags = flags_used

    def __init__(self, nfft_plan, flags=None):
        '''
        :param plan: instance of NFFT.
        :type plan: :class:`NFFT`
        :param flags: list of instantiation flags, see below.
        :type flags: tuple

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
        if self._dtype == np.complex64:
            solverf_finalize_complex(&self._plan_flt)
        elif self._dtype == np.complex128:
            solver_finalize_complex(&self._plan_dbl)
        elif self._dtype == np.complex256:
            solverl_finalize_complex(&self._plan_ldbl)

    def before_loop(self):
        '''Initialize the solver internals.'''
        self._before_loop()

    def loop_one_step(self):
        '''Perform one iteration of the solver.'''
        self._loop_one_step()

    cdef void _before_loop(self):
        if self._dtype == np.complex64:
            with nogil:
                solverf_before_loop_complex(&self._plan_flt)
        elif self._dtype == np.complex128:
            with nogil:
                solver_before_loop_complex(&self._plan_dbl)
        elif self._dtype == np.complex256:
            with nogil:
                solverl_before_loop_complex(&self._plan_ldbl)
        else:
            raise RuntimeError

    cdef void _loop_one_step(self):
        if self._dtype == np.complex64:
            with nogil:
                solverf_loop_one_step_complex(&self._plan_flt)
        elif self._dtype == np.complex128:
            with nogil:
                solver_loop_one_step_complex(&self._plan_dbl)
        elif self._dtype == np.complex256:
            with nogil:
                solverl_loop_one_step_complex(&self._plan_ldbl)
        else:
            raise RuntimeError

    property w:

        '''Weighting factors.'''

        def __get__(self):
            return self._w

        def __set__(self, array):
            self._w.ravel()[:] = array.ravel()

    property w_hat:

        '''Damping factors.'''

        def __get__(self):
            return self._w_hat

        def __set__(self, array):
            self._w_hat.ravel()[:] = array.ravel()

    property y:

        '''Right hand side, samples.'''

        def __get__(self):
            return self._y

        def __set__(self, array):
            self._y.ravel()[:] = array.ravel()

    property f_hat_iter:

        '''Iterative solution.'''

        def __get__(self):
            return self._f_hat_iter

        def __set__(self, array):
            self._f_hat_iter.ravel()[:] = array.ravel()

    @property
    def r_iter(self):
        '''Residual vector.'''
        return self._r_iter

    @property
    def z_hat_iter(self):
        '''Residual of normal equation of the first kind.'''
        return self._z_hat_iter

    @property
    def p_hat_iter(self):
        '''Search direction.'''
        return self._p_hat_iter

    @property
    def v_iter(self):
        '''Residual vector update.'''
        return self._v_iter

    @property
    def alpha_iter(self):
        '''Step size for search direction.'''
        if self._dtype == np.complex64:
            return self._plan_flt.alpha_iter
        elif self._dtype == np.complex128:
            return self._plan_dbl.alpha_iter
        elif self._dtype == np.complex256:
            return self._plan_ldbl.alpha_iter
        else:
            raise RuntimeError

    @property
    def beta_iter(self):
        '''Step size for search direction.'''
        if self._dtype == np.complex64:
            return self._plan_flt.beta_iter
        elif self._dtype == np.complex128:
            return self._plan_dbl.beta_iter
        elif self._dtype == np.complex256:
            return self._plan_ldbl.beta_iter
        else:
            raise RuntimeError

    @property
    def dot_r_iter(self):
        '''Weighted dotproduct of r_iter.'''
        if self._dtype == np.complex64:
            return self._plan_flt.dot_r_iter
        elif self._dtype == np.complex128:
            return self._plan_dbl.dot_r_iter
        elif self._dtype == np.complex256:
            return self._plan_ldbl.dot_r_iter
        else:
            raise RuntimeError

    @property
    def dot_r_iter_old(self):
        '''Previous dot_r_iter.'''
        if self._dtype == np.complex64:
            return self._plan_flt.dot_r_iter_old
        elif self._dtype == np.complex128:
            return self._plan_dbl.dot_r_iter_old
        elif self._dtype == np.complex256:
            return self._plan_ldbl.dot_r_iter_old
        else:
            raise RuntimeError

    @property
    def dot_z_hat_iter(self):
        '''Weighted dotproduct of z_hat_iter.'''
        if self._dtype == np.complex64:
            return self._plan_flt.dot_z_hat_iter
        elif self._dtype == np.complex128:
            return self._plan_dbl.dot_z_hat_iter
        elif self._dtype == np.complex256:
            return self._plan_ldbl.dot_z_hat_iter
        else:
            raise RuntimeError

    @property
    def dot_z_hat_iter_old(self):
        '''Previous dot_z_hat_iter.'''
        if self._dtype == np.complex64:
            return self._plan_flt.dot_z_hat_iter_old
        elif self._dtype == np.complex128:
            return self._plan_dbl.dot_z_hat_iter_old
        elif self._dtype == np.complex256:
            return self._plan_ldbl.dot_z_hat_iter_old
        else:
            raise RuntimeError

    @property
    def dot_p_hat_iter(self):
        '''Weighted dotproduct of p_hat_iter.'''
        if self._dtype == np.complex64:
            return self._plan_flt.dot_p_hat_iter
        elif self._dtype == np.complex128:
            return self._plan_dbl.dot_p_hat_iter
        elif self._dtype == np.complex256:
            return self._plan_ldbl.dot_p_hat_iter
        else:
            raise RuntimeError

    @property
    def dot_v_iter(self):
        '''Weighted dotproduct of v_iter.'''
        if self._dtype == np.complex64:
            return self._plan_flt.dot_v_iter
        elif self._dtype == np.complex128:
            return self._plan_dbl.dot_v_iter
        elif self._dtype == np.complex256:
            return self._plan_ldbl.dot_v_iter
        else:
            raise RuntimeError

    @property
    def dtype(self):
        '''The dtype of the solver.'''
        return self._dtype

    @property
    def flags(self):
        '''The precomputation flags.'''
        return self._flags
