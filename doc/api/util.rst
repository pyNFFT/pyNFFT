``pynfft.util`` - Utility functions
===================================

.. automodule:: pynfft.util

Functions used for initialization of :class:`pynfft.NFFT` attributes in test
scripts. For instance::

   >>> from pynfft import NFFT
   >>> from pynfft.util import vrand_unit_complex, vrand_shifted_unit_double
   >>> Nfft.NFFT(N=(16, 16), M=92)
   >>> vrand_shifted_unit_double(Nfft.x)
   >>> Nfft.precompute()
   >>> vrand_unit_complex(Nfft.f_hat)
   >>> Nfft.trafo()

.. autofunction:: pynfft.util.vrand_unit_complex(x)

.. autofunction:: pynfft.util.vrand_shifted_unit_double(x)

Functions used for computing the density compensation weights necessary for the
iterative solver and adjoint NFFT.

.. autofunction:: pynfft.util.voronoi_weights_1d(w, x)

.. autofunction:: pynfft.util.voronoi_weights_S2(w, xi)
