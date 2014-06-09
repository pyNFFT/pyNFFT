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
    from setuptools import setup, Command, Extension
except ImportError:
    from distutils.core import setup, Command, Extension

import os
import os.path
import sys
import numpy
import shutil

if sys.version_info[0] < 3:
    import ConfigParser as configparser
else:
    import configparser

setup_dir = dir = os.path.dirname(os.path.abspath(__file__))
package_name = 'pynfft'
package_dir = os.path.join(setup_dir, package_name)

if sys.platform.startswith('win'):
    fftw_threads = True    # use multi-threaded fftw libraries
    fftw_combined = True   # Windows usually has combined fftw3_threads and fftw3 libraries
    nfft_threads = True    # use multi-threaded nfft libraries
else:
    fftw_threads = True    # use multi-threaded fftw libraries
    fftw_combined = False  # Unices usually have seperate fftw3_threads and fftw3 libraries
    nfft_threads = True    # use multi-threaded nfft libraries

setup_cfg = 'setup.cfg'
ncconfig = None
if os.path.exists(setup_cfg):
    config = configparser.SafeConfigParser()
    config.read(setup_cfg)
    try: fftw_threads = config.getboolean("options", "fftw-threads")
    except: pass
    try: fftw_combined = config.getboolean("options", "fftw-threads-combined")
    except: pass
    try: nfft_threads = config.getboolean("options", "nfft-threads")
    except: pass

fftw_threads_env = os.environ.get('FFTW_THREADS')
if fftw_threads_env is not None:
    fftw_threads = fftw_threads_env.lower() in ('yes', 'y', 'true', 't', '1')

fftw_combined_env = os.environ.get('FFTW_THREADS_COMBINED')
if fftw_combined_env is not None:
    fftw_combined = fftw_combined_env.lower() in ('yes', 'y', 'true', 't', '1')

nfft_threads_env = os.environ.get('NFFT_THREADS')
if nfft_threads_env is not None:
    nfft_threads = nfft_threads_env.lower() in ('yes', 'y', 'true', 't', '1')

args = []
for arg in list(sys.argv):
    if arg == "--with-fftw-threads" or arg == "--enable-fftw-threads":
        fftw_threads = True ; args.append(arg)
    elif arg == "--without-fftw-threads" or arg == "--disable-fftw-threads":
        fftw_threads = False ; args.append(arg)
    elif arg == "--with-combined-fftw-threads" or arg == "--enable-combined-fftw-threads":
        fftw_combined = True ; args.append(arg)
    elif arg == "--without-combined-fftw-threads" or arg == "--disable-combined-fftw-threads":
        fftw_combined = False ; args.append(arg)
    elif arg == "--with-nfft-threads" or arg == "--enable-nfft-threads":
        nfft_threads = True ; args.append(arg)
    elif arg == "--without-nfft-threads" or arg == "--disable-nfft-threads":
        nfft_threads = False ; args.append(arg)
for arg in args:
    sys.argv.remove(arg)

with open(os.path.join("pynfft", "config.h"), "w") as config_h:
    config_h.write("/* pynfft/config.h. Generated from setup.py. */\n\n")
    config_h.write("/* Define to enable multi-threaded FFTW. */\n")
    if fftw_threads:
        config_h.write("#define FFTW_THREADS\n\n")
    else:
        config_h.write("/* #undef FFTW_THREADS */\n\n")
    config_h.write("/* Define to enable multi-threaded NFFT (requires OpemMP build of NFFT). */\n")
    if nfft_threads:
        config_h.write("#define NFFT_THREADS\n")
    else:
        config_h.write("/* #undef NFFT_THREADS */\n")

include_dirs = [numpy.get_include()]
library_dirs = []
package_data = {}

if fftw_threads and not fftw_combined:
    libraries = ['fftw3_threads', 'fftw3', 'm']
else:
    libraries = ['fftw3', 'm']

if nfft_threads:
    libraries = ['nfft3_threads', 'nfft3'] + libraries
else:
    libraries = ['nfft3'] + libraries

try:
    from Cython.Distutils import build_ext as build_ext
    ext_modules = [
        Extension(
            name=package_name+'.nfft',
            sources=[os.path.join(package_dir, 'nfft.pyx'), os.path.join(package_dir, "threading.c")],
            libraries=libraries,
            library_dirs=library_dirs,
            include_dirs=include_dirs,
            extra_compile_args='-O3 -fomit-frame-pointer -malign-double '
            '-fstrict-aliasing -ffast-math'.split(),
        ),
        Extension(
            name=package_name+'.solver',
            sources=[os.path.join(package_dir, 'solver.pyx'), os.path.join(package_dir, "threading.c")],
            libraries=libraries,
            library_dirs=library_dirs,
            include_dirs=include_dirs,
            extra_compile_args='-O3 -fomit-frame-pointer -malign-double '
            '-fstrict-aliasing -ffast-math'.split(),
        ),
        Extension(
            name=package_name+'.util',
            sources=[os.path.join(package_dir, 'util.pyx'), os.path.join(package_dir, "threading.c")],
            libraries=libraries,
            library_dirs=library_dirs,
            include_dirs=include_dirs,
            extra_compile_args='-O3 -fomit-frame-pointer -malign-double '
            '-fstrict-aliasing -ffast-math'.split(),
        ),
    ]

except ImportError as e:
    ext_modules = [
        Extension(
            name=package_name+'.nfft',
            sources=[os.path.join(package_dir, 'nfft.c'), os.path.join(package_dir, "threading.c")],
            libraries=libraries,
            library_dirs=library_dirs,
            include_dirs=include_dirs,
            extra_compile_args='-O3 -fomit-frame-pointer -malign-double '
            '-fstrict-aliasing -ffast-math'.split(),
        ),
        Extension(
            name=package_name+'.solver',
            sources=[os.path.join(package_dir, 'solver.c'), os.path.join(package_dir, "threading.c")],
            libraries=libraries,
            library_dirs=library_dirs,
            include_dirs=include_dirs,
            extra_compile_args='-O3 -fomit-frame-pointer -malign-double '
            '-fstrict-aliasing -ffast-math'.split(),
        ),
        Extension(
            name=package_name+'.util',
            sources=[os.path.join(package_dir, 'util.c'), os.path.join(package_dir, "threading.c")],
            libraries=libraries,
            library_dirs=library_dirs,
            include_dirs=include_dirs,
            extra_compile_args='-O3 -fomit-frame-pointer -malign-double '
            '-fstrict-aliasing -ffast-math'.split(),
        ),
    ]


class CleanCommand(Command):

    description = "Force clean of build files and directories."
    user_options = []

    def initialize_options(self):
        self.all = None

    def finalize_options(self):
        pass

    def run(self):
        for _dir in [os.path.join(setup_dir, d)
                for d in ('build', 'dist', 'doc/build', 'pyNFFT.egg-info')]:
            if os.path.exists(_dir):
                shutil.rmtree(_dir)
        for root, _, files in os.walk(package_dir):
            for _file in files:
                if not _file.endswith(('.py', '.pyx', '.pxd', '.pxi', 'threading.c')):
                    os.remove(os.path.join(package_dir, _file))


version = '1.3.0'
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
    'cmdclass': {'build_ext': build_ext,
                 'clean': CleanCommand,},
    'install_requires': ['numpy'],
}

if __name__ == '__main__':
    setup(**setup_args)
