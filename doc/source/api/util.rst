``pynfft.util`` - Utility functions
===================================

.. automodule:: pynfft.util

Functions used for initialization of :class:`pynfft.NFFT` attributes in test
scripts. For instance::

   >>> from pynfft.util import vrand_unit_complex, vrand_shifted_unit_double
   >>> x = np.empty(20, dtype=np.float64)
   >>> vrand_shifted_unit_double(x)
   >>> f_hat = np.empty(16, dtype=np.complex128)
   >>> vrand_unit_complex(f_hat)

.. autofunction:: pynfft.util.vrand_unit_complex(x)

.. autofunction:: pynfft.util.vrand_shifted_unit_double(x)

Functions used for computing the density compensation weights necessary for the
iterative solver and adjoint NFFT.

.. autofunction:: pynfft.util.voronoi_weights_1d(w, x)

.. autofunction:: pynfft.util.voronoi_weights_S2(w, xi)
