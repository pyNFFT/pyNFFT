``pynfft`` - The core NFFT functionalities
==========================================

.. automodule:: pynfft

NFFT Class
----------

.. autoclass:: pynfft.NFFT(x, f, f_hat, M, N, n=None, m=12, flags=None, precompute=False)

   .. autoattribute:: pynfft.NFFT.d

   .. autoattribute:: pynfft.NFFT.N

   .. autoattribute:: pynfft.NFFT.N_total
   
   .. autoattribute:: pynfft.NFFT.M
   
   .. autoattribute:: pynfft.NFFT.m

   .. autoattribute:: pynfft.NFFT.n

   .. autoattribute:: pynfft.NFFT.x

   .. autoattribute:: pynfft.NFFT.f

   .. autoattribute:: pynfft.NFFT.f_hat

   .. autoattribute:: pynfft.NFFT.dtype

   .. autoattribute:: pynfft.NFFT.flags

   .. automethod:: pynfft.NFFT.precompute

   .. automethod:: pynfft.NFFT.forward

   .. automethod:: pynfft.NFFT.adjoint

   .. automethod:: pynfft.NFFT.execute_trafo

   .. automethod:: pynfft.NFFT.execute_trafo_direct

   .. automethod:: pynfft.NFFT.execute_adjoint

   .. automethod:: pynfft.NFFT.execute_adjoint_direct

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
