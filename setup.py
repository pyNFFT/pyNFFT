
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

include_dirs = [numpy.get_include()]
library_dirs = []
package_data = {}

libraries = ['nfft3', 'nfft3_threads', 'm']

ext_modules = [
    Extension(
        name='pynfft.nfft',
        sources=[os.path.join('pynfft', 'nfft.c')],
        libraries=libraries,
        library_dirs=library_dirs,
        include_dirs=include_dirs,
        extra_compile_args='-O3 -fomit-frame-pointer -malign-double '
                           '-fstrict-aliasing -ffast-math'.split(),
    ),
    Extension(
        name='pynfft.solver',
        sources=[os.path.join('pynfft', 'solver.c')],
        libraries=libraries,
        library_dirs=library_dirs,
        include_dirs=include_dirs,
        extra_compile_args='-O3 -fomit-frame-pointer -malign-double '
                           '-fstrict-aliasing -ffast-math'.split(),
    ),
]

version = '0.2.3'

setup_args = {
    'name': 'pyNFFT',
    'version': version,
    'author': 'Ghislain Vaillant',
    'author_email': 'ghislain.vaillant@kcl.ac.uk',
    'description': 'A pythonic wrapper around NFFT',
    'url': 'https://bitbucket.org/ghisvail/pynfft',
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
    'packages': ['pynfft'],
    'ext_modules': ext_modules,
    'include_dirs': include_dirs,
    'package_data': package_data,
}

if __name__ == '__main__':
    setup(**setup_args)
