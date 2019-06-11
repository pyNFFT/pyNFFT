# -*- coding: utf-8 -*-
#
# Copyright 2013-2019 PyNFFT developers and contributors
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

import os
import shutil
import subprocess
import sys
from itertools import product
from os import path

import setuptools
from setuptools import Command, Extension, setup
from setuptools.command.build_ext import build_ext

setup_dir = path.dirname(path.abspath(__file__))
package_dir = path.join(setup_dir, "pynfft")


# --- Version info --- #


MAJOR = 1
MINOR = 4
MICRO = 0
ISRELEASED = True
VERSION = "{}.{}.{}".format(MAJOR, MINOR, MICRO)


# Borrowed from SciPy
def git_version():
    def _minimal_ext_cmd(cmd):
        # Construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env
        ).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        GIT_REVISION = out.strip().decode("ascii")
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


# Borrowed from SciPy
def get_version_info():
    FULLVERSION = VERSION
    if path.exists(".git"):
        GIT_REVISION = git_version()
    elif path.exists("pynfft/version.py"):
        # must be a source distribution, use existing version file
        # load it as a separate module in order not to load __init__.py
        import imp

        version = imp.load_source("pynfft.version", "pynfft/version.py")
        GIT_REVISION = version.git_revision
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += ".dev-" + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


# Borrowed from SciPy
def write_version_py(filename="pynfft/version.py"):
    version_fmt = """
# THIS FILE IS GENERATED FROM SETUP.PY
short_version = '{VERSION}'
version = '{VERSION}'
full_version = '{FULLVERSION}'
git_revision = '{GIT_REVISION}'
release = {ISRELEASED}

if not release:
    version = full_version
""".lstrip()
    FULLVERSION, GIT_REVISION = get_version_info()

    with open(filename, "w") as fp:
        fp.write(version_fmt.format(**globals()))


# --- Cleanup command --- #


class CleanCommand(Command):

    """Custom distutils command to clean build files."""

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
                if path.splitext(f)[-1] in (
                    ".pyc",
                    ".so",
                    ".o",
                    ".pyo",
                    ".pyd",
                ):
                    self._clean_me.append(path.join(root, f))
            for d in dirs:
                if d == "__pycache__":
                    self._clean_trees.append(path.join(root, d))
        # clean build and sdist directories in root
        for d in ("build", "dist"):
            if path.exists(d):
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


# --- pybind11 extensions --- #


def numpy_include():
    import numpy

    return numpy.get_include()


def py_include():
    import pybind11

    return pybind11.get_include()


def nfft_include():
    return path.abspath(path.join(py_include(), os.pardir))


class GetInclude(object):
    def __init__(self, getter, *args, **kwargs):
        self.getter = getter
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.getter(*self.args, **self.kwargs)


def common_extension_args():
    fft_libs = [
        pre + suf + thrd_ext
        for pre, suf, thrd_ext in product(
            ["nfft3", "fftw3"], ["", "f", "l"], ["", "_threads"]
        )
    ]
    common_ext_args = dict(
        include_dirs=[
            GetInclude(f) for f in (py_include, numpy_include, nfft_include)
        ],
        libraries=fft_libs + ["m"],
        library_dirs=[],
        extra_compile_args=(
            "-O3 -fomit-frame-pointer -fstrict-aliasing -ffast-math".split()
        ),
    )
    return common_ext_args


ext_modules = [
    Extension(
        "pynfft._nfft",
        [path.join(package_dir, "_nfft.cpp")],
        language="c++",
        **common_extension_args()  # no trailing comma
    )
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 flag is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, "-std=c++14"):
        return "-std=c++14"
    elif has_flag(compiler, "-std=c++11"):
        return "-std=c++11"
    else:
        raise RuntimeError(
            "Unsupported compiler -- at least C++11 support is needed!"
        )


class BuildExt(build_ext):

    """A custom build extension for adding compiler-specific options."""

    c_opts = {"msvc": ["/EHsc"], "unix": []}

    if sys.platform == "darwin":
        c_opts["unix"].extend(["-stdlib=libc++", "-mmacosx-version-min=10.7"])

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == "unix":
            opts.append(
                '-DVERSION_INFO="{}"'.format(self.distribution.get_version())
            )
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append(
                '/DVERSION_INFO=\\"{}\\"'.format(
                    self.distribution.get_version()
                )
            )
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


# --- setup --- #

# Get current version & refresh version file
FULLVERSION, GIT_REVISION = get_version_info()
write_version_py()

setup(
    version=FULLVERSION,
    cmdclass={"build_ext": BuildExt, "clean": CleanCommand},
    ext_modules=ext_modules,
    tests_require=["pytest"],
)
