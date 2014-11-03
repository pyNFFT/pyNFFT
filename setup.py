#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import shutil
from distutils.core import Command

# Define global path variables
setup_dir = dir = os.path.dirname(os.path.abspath(__file__))
package_name = 'pynfft'
package_dir = os.path.join(setup_dir, package_name)


# Define utility functions to build the extensions
def get_common_extension_args():
    import numpy
    common_extension_args = dict(
        libraries=['nfft3_threads', 'nfft3', 'fftw3_threads', 'fftw3'],
        library_dirs=[],
        include_dirs=[numpy.get_include()],
        extra_compile_args='-O3 -fomit-frame-pointer -malign-double '
        '-fstrict-aliasing '.split(),
        define_macros=[("NPY_NO_DEPRECATED_API", None)],
        )
    return common_extension_args

def get_extensions():
    from distutils.extension import Extension
    try:
        from Cython.Build import cythonize
        has_cython = True
    except ImportError:
        has_cython = False
    ext_modules = []
    common_extension_args = get_common_extension_args()
    module_base_names = ('nfft_plan', 'solver_plan', 'util')
    module_file_ext = '.pyx' if has_cython else '.c'
    for base_name in module_base_names:
        ext_modules.append(
            Extension(
                name = package_name + '.' + base_name,
                sources = [os.path.join(package_dir, base_name + module_file_ext)],
                **common_extension_args
                )
            )
    if has_cython:
        return cythonize(ext_modules)
    else:
        return ext_modules


# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')


# Define custom clean command
class CleanCommand(Command):
    """Custom distutils command to clean the .so and .pyc files."""

    user_options = [("all", "a", "")]

    def initialize_options(self):
        self.all = True
        self._clean_me = []
        self._clean_trees = []
        self._clean_exclude = []
        # clean Cython generated files and cache
        for root, dirs, files in os.walk(package_dir):
            for f in files:
                if f in self._clean_exclude:
                    continue
                if os.path.splitext(f)[-1] in ('.pyc', '.so', '.o',
                                               '.pyo',
                                               '.pyd', '.c', '.orig'):
                    self._clean_me.append(os.path.join(root, f))
            for d in dirs:
                if d == '__pycache__':
                    self._clean_trees.append(os.path.join(root, d))
        # clean build and sdist directories in root
        for d in ('build', 'dist'):
            if os.path.exists(d):
                self._clean_trees.append(d)

    def finalize_options(self):
        pass

    def run(self):
        for clean_me in self._clean_me:
            try:
                os.unlink(clean_me)
            except Exception:
                pass
        for clean_tree in self._clean_trees:
            try:
                shutil.rmtree(clean_tree)
            except Exception:
                pass


LONG_DESCRIPTION = '''"The NFFT is a C subroutine library for computing the
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

CLASSIFIERS = [
    'Programming Language :: Cython',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.2',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: BSD License',
    'Operating System :: POSIX :: Linux',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Multimedia :: Sound/Audio :: Analysis',
]

MAJOR = 2
MINOR = 0
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

# borrowed from scipy
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

# borrowed from scipy
def get_version_info():
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('pynfft/version.py'):
        # must be a source distribution, use existing version file
        # load it as a separate module in order not to load __init__.py
        import imp
        version = imp.load_source('pynfft.version', 'pynfft/version.py')
        GIT_REVISION = version.git_revision
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev-' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION

# borrowed from scipy
def write_version_py(filename='pynfft/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    f = open(filename, 'w')
    try:
        f.write(cnt % {'version': VERSION,
                       'full_version' : FULLVERSION,
                       'git_revision' : GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        f.close()


def setup_package():
    # Use setuptools if available
    try:
        from setuptools import setup
    except ImportError:
        from distutils.core import setup

    # Get current version
    FULLVERSION, GIT_REVISION = get_version_info()

    # Refresh version file
    write_version_py()

    # Common setup args
    setup_args = dict(
        name = 'pyNFFT',
        version = FULLVERSION,
        author = 'Ghislain Antony Vaillant',
        author_email = 'ghisvail@gmail.com',
        description = 'A pythonic wrapper around NFFT',
        long_description = LONG_DESCRIPTION,
        url = 'https://github.com/ghisvail/pyNFFT.git',
        cmdclass = {'clean': CleanCommand},
        classifiers = CLASSIFIERS,
        platforms=['Linux', 'Unix'],
        )

    if len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
            sys.argv[1] in ('--help-commands', 'egg_info', '--version',
                            'clean')):
        # For these actions, NumPy is not required.
        pass
    else:
        setup_args['packages'] = ['pynfft', 'pynfft.tests']
        setup_args['ext_modules'] = get_extensions()
        setup_args['test_suite'] = 'nose.collector'
        setup_args['setup_requires'] = ['numpy>=1.7',]
        setup_args['install_requires'] = ['numpy>=1.7',]

    setup(**setup_args)


if __name__ == '__main__':
    setup_package()
