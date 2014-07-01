#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013  Ghislain Vaillant
#
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

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

import os
import os.path
import numpy
import shutil

setup_dir = dir = os.path.dirname(os.path.abspath(__file__))
package_name = 'pynfft'
package_dir = os.path.join(setup_dir, package_name)

include_dirs = [numpy.get_include()]
library_dirs = []
package_data = {}
libraries = ['nfft3_threads', 'nfft3', 'fftw3_threads', 'fftw3', 'm']

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [
        Extension(
            name=package_name+'.nfft',
            sources=[os.path.join(package_dir, 'nfft.pyx')],
            libraries=libraries,
            library_dirs=library_dirs,
            include_dirs=include_dirs,
        ),
        Extension(
            name=package_name+'.solver',
            sources=[os.path.join(package_dir, 'solver.pyx')],
            libraries=libraries,
            library_dirs=library_dirs,
            include_dirs=include_dirs,
        ),
        Extension(
            name=package_name+'.util',
            sources=[os.path.join(package_dir, 'util.pyx')],
            libraries=libraries,
            library_dirs=library_dirs,
            include_dirs=include_dirs,
        ),
    ]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [
        Extension(
            name=package_name+'.nfft',
            sources=[os.path.join(package_dir, 'nfft.c')],
            libraries=libraries,
            library_dirs=library_dirs,
            include_dirs=include_dirs,
        ),
        Extension(
            name=package_name+'.solver',
            sources=[os.path.join(package_dir, 'solver.c')],
            libraries=libraries,
            library_dirs=library_dirs,
            include_dirs=include_dirs,
        ),
        Extension(
            name=package_name+'.util',
            sources=[os.path.join(package_dir, 'util.c')],
            libraries=libraries,
            library_dirs=library_dirs,
            include_dirs=include_dirs,
        ),
    ]


version = '1.3.1'
release = True
if not release:
    version += '-dev'

long_description = '''"The NFFT is a C subroutine library for computing the
nonequispaced discrete Fourier transform (NDFT) in one or more dimensions, of
arbitrary input size, and of complex data."

The NFFT library is licensed under GPLv2 and available at:
    http://www-user.tu-chemnitz.de/~potts/nfft/index.php

This wrapper provides a somewhat Pythonic access to some of the core NFFT
library functionalities and is largely inspired from the pyFFTW project
developped by Henry Gomersall (http://hgomersall.github.io/pyFFTW/).

This project is still work in progress and is still considered beta quality. In
particular, the API is not yet frozen and is likely to change as the
development continues. Please consult the documentation and changelog for more
information.'''

classifiers = [
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    'Operating System :: POSIX :: Linux',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Multimedia :: Sound/Audio :: Analysis',
]

setup_args = {
    'name': 'pyNFFT',
    'version': version,
    'author': 'Ghislain Vaillant',
    'author_email': 'ghisvail@gmail.com',
    'description': 'A pythonic wrapper around NFFT',
    'long_description': long_description,
    'url': 'https://github.com/ghisvail/pyNFFT.git',
    'classifiers': classifiers,
    'packages': [package_name],
    'ext_modules': ext_modules,
    'include_dirs': include_dirs,
    'package_data': package_data,
    'cmdclass': cmdclass,
    'install_requires': ['numpy'],
}

if __name__ == '__main__':
    setup(**setup_args)
