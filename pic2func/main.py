# -*- coding: utf-8 -*-
from .predict import define_model, predict_tickvalues
from .imgfuncs import im2xyframe, rgb2bw, rgb2g, rgb2r
from .detect import (get_IJcurve, detect_axes, get_ticks, remove_ticks,
                     get_tickvals)
from .function import get_XYfunc, dft
import imageio as im

"""Main module. 

Functions to get an (x,y) function from a picture.

Methods defined here
---------------------
function_from_picture(picpath, verbose=False)
    Get an (x,y) function from a picture.

fourier_function_from_picture(picpath, n=10, verbose=False)
    Get an (x,y) function and its Fourier series from a picture.

"""

def function_from_picture(picpath, verbose=False):
    digitmodel = define_model()
    digitmodel.load_weights("weights")
    picpng = im.imread(picpath)
    picbw = im2xyframe(rgb2bw(picpng))
    picticks = im2xyframe(rgb2g(picpng))
    piccurve = im2xyframe(rgb2r(picpng))
    curveIJ = get_IJcurve(piccurve)
    axes = detect_axes(picbw, verbose=verbose)
    Iticks, Jticks = get_ticks(picticks, axes)
    picnr, tick_picids = remove_ticks(picticks, axes, Iticks, Jticks)
    tickvals = get_tickvals(picnr, tick_picids, proximity=0.01)
    IXticks, JYticks = predict_tickvalues(Iticks, Jticks, tickvals, digitmodel)
    f = get_XYfunc(curveIJ, IXticks, JYticks, axes)  # Nx2 array: (x,y) points
    return f

def fourier_function_from_picture(picpath, n=10, verbose=False):
    digitmodel = define_model()
    digitmodel.load_weights("weights")
    picpng = im.imread(picpath)
    picbw = im2xyframe(rgb2bw(picpng))
    picticks = im2xyframe(rgb2g(picpng))
    piccurve = im2xyframe(rgb2r(picpng))
    curveIJ = get_IJcurve(piccurve)
    axes = detect_axes(picbw, verbose=verbose)
    Iticks, Jticks = get_ticks(picticks, axes)
    picnr, tick_picids = remove_ticks(picticks, axes, Iticks, Jticks)
    tickvals = get_tickvals(picnr, tick_picids, proximity=0.01)
    IXticks, JYticks = predict_tickvalues(Iticks, Jticks, tickvals, digitmodel)
    f = get_XYfunc(curveIJ,IXticks,JYticks,axes)  # Nx2 array: (x,y) points
    dft_f = dft(f[:,1],n) 
    return f, dft_f
