# -*- coding: utf-8 -*-
"""Package to read self-drawn y=f(x) functions from an image."""

from setuptools import setup, find_packages
import pathlib
from codecs import open as openc

def get_requirements():
    """Read requirements.txt and return a list of requirements."""
    here = pathlib.Path(__file__).absolute().parent
    requirements = []
    filename = here.joinpath('requirements.txt')
    with openc(filename, encoding='utf-8') as fileh:
        for lines in fileh:
            requirements.append(lines.strip())
    return requirements

setup(
    name='pic2func',
    version='0.1',
    description='Package to read y=f(x) functions from an image.',
    author='Wouter Vervust',
    license='MIT',
    packages=find_packages(),
    install_requires=get_requirements(),
    python_requires='>=3.10',
)
