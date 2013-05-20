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

import numpy as np
cimport numpy as np
from libc cimport limits
from cnfft3 cimport *


cdef class NFFT:

    # where the C-related content of the class is being initialized
    def __cinit__(self):
        pass

    # here, just holds the documentation of the class constructor
    def __init__(self):
        pass

    # where the C-related content of the class needs to be cleaned
    def __dealloc__(self):
        pass

    cpdef precompute(self):
        pass

    cpdef trafo(self):
        pass

    cpdef trafo_direct(self):
        pass

    cpdef adjoint(self):
        pass

    cpdef adjoint_direct(self):
        pass
