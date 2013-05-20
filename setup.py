
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Ghislain Vaillant
# ghislain.vallant@kcl.ac.uk

from distutils.core import setup
from distutils.extension import Extension

import os
import numpy
import sys

include_dirs = [numpy.get_include()]
library_dirs = []
package_data = {}

libraries = ['nfft3', 'nfft3_threads', 'm']

ext_modules = [
    Extension('pynfft.nfft',
        sources=[os.path.join('pynfft', 'nfft.c')],
        libraries=libraries,
        library_dirs=library_dirs,
        include_dirs=include_dirs)]

long_description = ''''''

version = '0.1.0'

setup_args = {
    'name': 'pyNFFT',
    'version': version,
    'author': 'Ghislain Vaillant',
    'author_email': 'ghislain.vaillant@kcl.ac.uk',
    'description': 'A pythonic wrapper around NFFT',
    'url': 'https://bitbucket.org/ghisvail/pynfft',
    'long_description': long_description,
    'classifiers': [
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: POSIX :: Linux',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
    ],
    'packages':['pyfftw', 'pyfftw.builders', 'pyfftw.interfaces'],
    'ext_modules': ext_modules,
    'include_dirs': include_dirs,
    'package_data': package_data,
}

if __name__ == '__main__':
    setup(**setup_args)
