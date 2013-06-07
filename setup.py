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
    Extension(
        name='pynfft.util',
        sources=[os.path.join('pynfft', 'util.c')],
        libraries=libraries,
        library_dirs=library_dirs,
        include_dirs=include_dirs,
        extra_compile_args='-O3 -fomit-frame-pointer -malign-double '
                           '-fstrict-aliasing -ffast-math'.split(),
    ),
]


def get_version():
    basedir = os.path.dirname(__file__)
    with open(os.path.join(basedir, 'pynfft/_version.py')) as f:
        variables = {}
        exec(f.read(), variables)
        return variables.get('VERSION')
    raise RuntimeError('No version info found.')

version = get_version()

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
    'packages': ['pynfft'],
    'ext_modules': ext_modules,
    'include_dirs': include_dirs,
    'package_data': package_data,
}

if __name__ == '__main__':
    setup(**setup_args)
