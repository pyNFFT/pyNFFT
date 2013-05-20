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
from Cython.Distutils import build_ext
from setup import setup_args, libraries, library_dirs, include_dirs
import os

include_dirs.append('pynfft')

ext_modules = [
    Extension(
        'pynfft.nfft',
        sources=[os.path.join('pynfft', 'nfft.pyx')],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs)]

setup_args['cmdclass'] = {'build_ext': build_ext}
setup_args['ext_modules'] = ext_modules

setup(**setup_args)
