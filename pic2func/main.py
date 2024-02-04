# -*- coding: utf-8 -*-
"""Main module.

Functions to get an (x,y) function from a picture.

Methods defined here
---------------------
function_from_picture(picpath, verbose=False)
    Get an (x,y) function from a picture.

fourier_function_from_picture(picpath, n=10, verbose=False)
    Get an (x,y) function and its Fourier series from a picture.

"""
import os
import imageio as im
from .predict import define_model, predict_tickvalues
from .imgfuncs import im2xyframe, rgb2bw, rgb2g, rgb2r
from .detect import (get_ijcurve, detect_axes, get_ticks, remove_ticks,
                     get_tickvals)
from .function import get_xyfunc, nogibbsdftsample

def function_from_picture(picpath, verbose=False):
    """Get an (x,y) function from a picture.

    Parameters
    ----------
    picpath : str
        The path to the picture (jpg or png).
    verbose : bool
        If True, print the steps of the process.

    Returns
    -------
    np.array
        Nx2 array: (x,y) points.

    """
    weightspath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "cnn/weights.index")
    digitmodel = define_model()
    digitmodel.load_weights(weightspath[:-6])
    picpng = im.imread(picpath)
    picbw = im2xyframe(rgb2bw(picpng))
    picticks = im2xyframe(rgb2g(picpng))
    piccurve = im2xyframe(rgb2r(picpng))
    curveij = get_ijcurve(piccurve)
    axes = detect_axes(picbw, verbose=verbose)
    iticks, jticks = get_ticks(picticks, axes)
    picnr, tick_picids = remove_ticks(picticks, axes, iticks, jticks)
    tickvals = get_tickvals(picnr, tick_picids, proximity=0.01, verbose=False)
    ixticks, jyticks = predict_tickvalues(iticks, jticks, tickvals, digitmodel)
    f = get_xyfunc(curveij, ixticks, jyticks, axes)  # Nx2 array: (x,y) points
    return f

def fourier_function_from_picture(picpath, n=10, verbose=False):
    """Get an (x,y) function and its Fourier series from a picture.

    Parameters
    ----------
    picpath : str
        The path to the picture (jpg or png).
    n : int
        The number of terms in the Fourier series.
    verbose : bool
        If True, print the steps of the process.

    Returns
    -------
    np.array
        Nx2 array: (x,y) points.
    np.array
        2n+1x2 array: (a,b) coefficients.

    """
    weightspath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "cnn/weights.index")
    digitmodel = define_model()
    digitmodel.load_weights(weightspath[:-6])
    picpng = im.imread(picpath)
    picbw = im2xyframe(rgb2bw(picpng))
    picticks = im2xyframe(rgb2g(picpng))
    piccurve = im2xyframe(rgb2r(picpng))
    curveij = get_ijcurve(piccurve)
    axes = detect_axes(picbw, verbose=verbose)
    iticks, jticks = get_ticks(picticks, axes)
    picnr, tick_picids = remove_ticks(picticks, axes, iticks, jticks)
    tickvals = get_tickvals(picnr, tick_picids, proximity=0.01)
    ixticks, jyticks = predict_tickvalues(iticks, jticks, tickvals, digitmodel)
    f = get_xyfunc(curveij,ixticks,jyticks,axes)  # Nx2 array: (x,y) points
    xsample, fsample = nogibbsdftsample(f[:,1], n, f[:,0])
    return f, xsample, fsample
