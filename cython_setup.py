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
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setup import setup_args, libraries, library_dirs, include_dirs
import os

include_dirs.append('pynfft')

ext_modules = [
    Extension(
        name='pynfft.nfft',
        sources=[os.path.join('pynfft', 'nfft.pyx')],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_compile_args='-O3 -fomit-frame-pointer -malign-double '
                           '-fstrict-aliasing -ffast-math'.split(),
    ),
    Extension(
        name='pynfft.util',
        sources=[os.path.join('pynfft', 'util.pyx')],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_compile_args='-O3 -fomit-frame-pointer -malign-double '
                           '-fstrict-aliasing -ffast-math'.split(),
    ),
]

setup_args['cmdclass'] = {'build_ext': build_ext}
setup_args['ext_modules'] = cythonize(ext_modules)

setup(**setup_args)
