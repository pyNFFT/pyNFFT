``pynfft`` - The core NFFT functionalities
==========================================

.. automodule:: pynfft

NFFT Class
----------

.. autoclass:: pynfft.NFFT(N, M, n=None, m=12, x=None, f=None, f_hat=None, flags=None)

   .. autoattribute:: pynfft.NFFT.d

   .. autoattribute:: pynfft.NFFT.N

   .. autoattribute:: pynfft.NFFT.N_total
   
   .. autoattribute:: pynfft.NFFT.M_total
   
   .. autoattribute:: pynfft.NFFT.m

   .. autoattribute:: pynfft.NFFT.n

   .. autoattribute:: pynfft.NFFT.x

   .. autoattribute:: pynfft.NFFT.f

   .. autoattribute:: pynfft.NFFT.f_hat

   .. autoattribute:: pynfft.NFFT.dtype

   .. autoattribute:: pynfft.NFFT.flags

   .. automethod:: pynfft.NFFT.precompute

   .. automethod:: pynfft.NFFT.trafo

   .. automethod:: pynfft.NFFT.trafo_direct

   .. automethod:: pynfft.NFFT.adjoint

   .. automethod:: pynfft.NFFT.adjoint_direct

Solver Class
------------

.. autoclass:: pynfft.Solver(nfft_plan, flags=None)

   .. autoattribute:: pynfft.Solver.w
   
   .. autoattribute:: pynfft.Solver.w_hat
   
   .. autoattribute:: pynfft.Solver.y

   .. autoattribute:: pynfft.Solver.f_hat_iter

   .. autoattribute:: pynfft.Solver.r_iter

   .. autoattribute:: pynfft.Solver.dtype

   .. autoattribute:: pynfft.Solver.flags

   .. automethod:: pynfft.Solver.before_loop

   .. automethod:: pynfft.Solver.loop_one_step
