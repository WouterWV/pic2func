# -*- coding: utf-8 -*-
"""Package to read y=f(x) functions from an image.

Package structure
-----------------

Modules
~~~~~~~
__init__.py
    Import core functions from the modules.

detect.py
    Detect axes, function, tickmarks and tickvalues from an image array.

function.py
    Methods to transform ij pixel coordinates to desired xy coordinates.

imgfuncs.py
    Methods to extract color channels and other data from an image array.

main.py
    Main scripts to read y=f(x) functions from an image.

predict.py
    CNN model to predict the digits given at tickvalues.

DATA
~~~~
cnn
    Folder with the CNN tensorflow model weights.

"""
from .main import function_from_picture, fourier_function_from_picture
