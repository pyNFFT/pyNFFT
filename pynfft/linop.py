# -*- coding: utf-8 -*-
#
# Copyright PyNFFT developers and contributors
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

from .nfft import NFFT


def as_linop(plan):
    """Construct a SciPy LinearOperator from an NFFT plan.

    :param plan: The NFFT plan to be wrapped. Its ``precompute()`` method
    must have been run, and its ``x`` array is assumed to be initialized.
    :type plan: ``nfft.NFFT``

    :returns: A linear operator backed by the given plan.
    :rtype: ``scipy.sparse.linalg.LinearOperator``
    """
    # Expensive import, do lazily
    from scipy.sparse.linalg import LinearOperator

    if not isinstance(plan, NFFT):
        raise TypeError(
            "`plan` must be an `NFFT` instance, got {!r}".format(plan)
        )

    mat_shape = (plan.M, plan.N_total)

    def matvec(v):
        plan.f_hat[:] = v.reshape(plan.N)
        plan.trafo()
        return plan.f.ravel()

    def rmatvec(u):
        plan.f[:] = u.reshape((plan.M,))
        plan.adjoint()
        return plan.f_hat.ravel()

    return LinearOperator(
        shape=mat_shape,
        dtype=plan.dtype,
        matvec=matvec,
        rmatvec=rmatvec,
    )
