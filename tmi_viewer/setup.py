import Cython.Compiler.Main

import numpy
from numpy.distutils.command import build_src
from numpy.distutils.misc_util import Configuration

from distutils.dir_util import mkpath

import os

PACKAGE_NAME = "tmi_viewer"

def configuration(parent_package = "", top_path = None):
	from numpy.distutils.misc_util import Configuration
	CONFIG = Configuration(PACKAGE_NAME, 
		parent_name = parent_package, 
		top_path = top_path)

	CONFIG.make_config_py()
	return CONFIG

if __name__ == '__main__':
	from numpy.distutils.core import setup
	setup(**configuration(top_path='').todict())
