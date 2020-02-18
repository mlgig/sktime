#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = "Thach Le Nguyen"

import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("classifiers", parent_package, top_path)

    config.add_extension(
        name="mrrandom",
        sources=["mrrandom.pyx"],
        # sources=["mrseql_wrapper.cpp"],
        extra_compile_args=['-std=c++11'],
        language="c++",
        #include_dirs=[numpy.get_include()]
    )
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())