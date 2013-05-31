Changelog
=========


Changes in version 0.2.4
------------------------

    - Add sphinx documentation


Changes in version 0.2.3
------------------------

    - Level-up Solver class with the improvement made on NFFT

    - Update test suite for pynfft.nfft


Changes in version 0.2.2
------------------------

    - Completed switch to non-malloc'd arrays for x, f and f_hat. These are now managed by internal or external numpy arrays

    - Remove management of obsolete MALLOC flags

    - Fix broken test for non-contiguousness for external arrays


Changes in version 0.2.1
------------------------

    - Added experimental support for external python arrays, which can replace the internal malloc'd x, f and f_hat


Changes in version 0.2
------------------------

    - Added support for the solver component of the NFFT library


Changes in version 0.1.1
------------------------

    - Added non-complete test coverage for pynfft.nfft


Version 0.1
-----------

    - Initial release. Experimental support for the nfft component of the NFFT library
