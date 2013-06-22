``pynfft.nfft`` - The core NFFT functionalities
===============================================

.. automodule:: pynfft.nfft

NFFT Class
----------

.. autoclass:: pynfft.nfft.NFFT(N, M, n=None, m=12, x=None, f=None, f_hat=None, dtype=None, flags=None)

   .. autoattribute:: pynfft.nfft.NFFT.d

   .. autoattribute:: pynfft.nfft.NFFT.N

   .. autoattribute:: pynfft.nfft.NFFT.N_total
   
   .. autoattribute:: pynfft.nfft.NFFT.M_total
   
   .. autoattribute:: pynfft.nfft.NFFT.m

   .. autoattribute:: pynfft.nfft.NFFT.x

   .. autoattribute:: pynfft.nfft.NFFT.f

   .. autoattribute:: pynfft.nfft.NFFT.f_hat

   .. autoattribute:: pynfft.nfft.NFFT.dtype

   .. autoattribute:: pynfft.nfft.NFFT.flags

   .. automethod:: pynfft.nfft.NFFT.precompute

   .. automethod:: pynfft.nfft.NFFT.trafo

   .. automethod:: pynfft.nfft.NFFT.trafo_direct

   .. automethod:: pynfft.nfft.NFFT.adjoint

   .. automethod:: pynfft.nfft.NFFT.adjoint_direct
