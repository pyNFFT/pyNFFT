``pynfft.util`` - Utility Functions
===================================

.. automodule:: pynfft.util

Functions used to generate random vectors for :class:`pynfft.NFFT` attributes in test scripts.
For instance::

   >>> from pynfft.util import random_unit_complex, random_unit_shifted
   >>> x = random_unit_shifted(20, dtype='float64')
   >>> f_hat = random_unit_complex(16, dtype='complex128')

.. autofunction:: pynfft.util.random_unit_complex(size, dtype)

.. autofunction:: pynfft.util.random_shifted_unit(size, dtype)
