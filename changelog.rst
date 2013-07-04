Changelog
=========

Changes in version 0.6
----------------------

    - pynfft.nfft: enable openmp support

Changes in version 0.5
----------------------

    - pynfft.nfft: rewrite NFFT class internals to support multiple floating 
      point precision, coming in a future version of libnfft3

    - Documenation: first draft of the tutorial section

Changes in version 0.4.1
------------------------

    - New simplified test suite.

Changes in version 0.4
----------------------

    - Improved flag management: NFFT now only accepts the list of supported 
      flags listed in its documentaton.

    - pynfft.util: utility functions listed in nfft3util.h. Only, the random
      initializers and Voronoi weights computation functions have been wrapped.

    - Changelog is no longer part of the sphinx tree.

Changes in version 0.3.1
------------------------

    - Fixed issue #1: crash in test_nfft due to use of MALLOC flags


Changes in version 0.3
----------------------

    - Improve precomputation flag management in NFFT and Solver classes

    - Various code improvements

    - Update documentation for all modules


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
