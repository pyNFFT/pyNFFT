Installing / Upgrading
======================
.. highlight:: bash

Right now, only installation from source is supported. Support for
pip/easy_install via the `Python Package Index <http://pypi.python.org/pypi/>`_ is
coming.

Installing from source
----------------------
To install pyNFFT directly from source, you will need to install the NFFT
library manually. If the NFFT is installed in a system-wide location, then
installing pyNFFT requires the following commands:: 

        $ git clone https://ghisvail@bitbucket.org/ghisvail/pynfft
        $ cd pynfft/
        $ python setup.py install

If the NFFT library is not installed in a standard location, for example in
$HOME/local, then you will need to break the installation process into the following steps::

        $ python setup.py build_ext --library-dirs=$HOME/local/lib --include-dirs=$HOME/local/include
        $ python setup.py install

