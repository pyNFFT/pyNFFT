``pynfft.solver`` - iterative solver for the adjoint NFFT
=========================================================

.. automodule:: pynfft.solver

Solver Class
------------

.. autoclass:: pynfft.solver.Solver(nfft_plan, flags=None)

   .. autoattribute:: pynfft.solver.Solver.w
   
   .. autoattribute:: pynfft.solver.Solver.w_hat
   
   .. autoattribute:: pynfft.solver.Solver.y

   .. autoattribute:: pynfft.solver.Solver.f_hat_iter

   .. autoattribute:: pynfft.solver.Solver.r_iter

   .. autoattribute:: pynfft.solver.Solver.dtype

   .. autoattribute:: pynfft.solver.Solver.flags

   .. automethod:: pynfft.solver.Solver.before_loop

   .. automethod:: pynfft.solver.Solver.loop_one_step
