
# -*- coding: utf-8 -*-
"""Functions to convert ij data to xy data.

Methods defined here
---------------------
get_xyfunc(curveij, ixticks, jyticks, axes)
    Converts the ij curve to an xy function.

ijcurve_to_ijfunction(curve)
    Convert an ij curve into an ij function.

get_scaleandshift_ijtoxy(ixticks, jyticks, axes)
    Using the ticks, the ij function can be converted to an xy function.

dft(x, nc)
    Computes the first 2*nc + 1 Fourier coeffs of the DFT associated with x,
    and recreates x using only these Fourier coefficients. If you want to return
    x recreated at different points, you will need to create a different def.

dftsample(x, nc, nis, nos=200)
    Computes the first 2*nc + 1 Fourier coeffs of the DFT associated with x,
    and recreates x using only these Fourier coefficients. If you want to return
    x recreated at different points, you will need to create a different def.

nogibbsdftsample(x, nc, nis, nos=200)
    gibbs phenomenom will occur when x[0] != x[-1]. So we just extend the
    profile left and right, linearly, to reach to eachother. Then we discard
    those points.

"""
import numpy as np

def get_xyfunc(curveij,ixticks,jyticks,axes):
    """
    Converts the ij curve to an xy function.
    Step1: Convert the ij curve to an ij function.
    Step2: Use the ticks to convert the ij function to an xy function.

    Parameters
    ----------
    curveij : nx2 array
        (I,J) points of the curve. (For one I, many J's...)
    ixticks : list
        A list of tick values of the I axis.
    jyticks : list
        A list of tick values of the J axis.
    axes : list
        The axes of the plot.

    Returns
    -------
    np.array
        n_unique_ivals x 2 array with (x,y) values.

    """
    fij = ijcurve_to_ijfunction(curveij) #
    scale_ij_to_xy, shift_ij = get_scaleandshift_ijtoxy(ixticks, jyticks, axes)
    fxy = np.matmul((fij+shift_ij),scale_ij_to_xy)
    return fxy

def ijcurve_to_ijfunction(curve):
    """Convert an ij curve into an ij function.

    The ij curve is the drawing by the user. For each x value, there are
    multiple y values. This function will average the y values for each x value.

    Parameters
    ----------
    curve : nx2 array
        (I,J) points of the curve.

    Returns
    -------
    np.array
        n_unique_ivals x 2 array with (I,J) points of the function.

    """
    ivals = curve[:,0]
    jvals = curve[:,1]
    sortids = ivals.argsort()
    isort = ivals[sortids]
    jsort = jvals[sortids]
    ijfunc = []
    ivals=[]
    spin = isort[0]
    val = [jsort[0]]
    for I,J in zip(isort[1:],jsort[1:]):
        if I == spin:
            val = val + [J]
        else:
            ijfunc.append(np.mean(np.array(val)))
            ivals.append(spin)
            spin = I
            val = [J]

    return (np.array([ivals, ijfunc])).T

def get_scaleandshift_ijtoxy(ixticks, jyticks, axes):
    """Using the ticks, the ij function can be converted to an xy function.

    At least one tick per axis is required. If only one tick is given, it
    cannot be 0.

    Parameters
    ----------
    ixticks : list
        A list of tick values of the I axis.
    jyticks : list
        A list of tick values of the J axis.
    axes : list
        The axes of the plot.

    Returns
    -------
    np.array
        2x2 array with the scale.
    np.array
        2x1 array with the shift (negative of origin in ij coordinates).

    """
    # We need at least one tick in each direction.
    assert len(ixticks) > 0, "No ticks found in I axis."
    assert len(jyticks) > 0, "No ticks found in J axis."
    # Define the crosspoint of I, J axes in ij coordinates
    crossij = np.array([axes[1],axes[0]])  # crosspoint of the axes

    # Define the first tick of I, J axes in ij coordinates
    i1 = np.array([ixticks[0][0], 0])  # first tick of I
    j1 = np.array([0, jyticks[0][0]])  # first tick of J
    # And their corresponding ticks in xy coordinates
    x1 = np.array([ixticks[0][1], 0])  # first tick of X
    y1 = np.array([0, jyticks[0][1]])  # first tick of Y
    # If a second tick is given, define it as well
    if len(ixticks)==2:
        i2 = np.array([ixticks[1][0], 0])  # second tick of I
        x2 = np.array([ixticks[1][1], 0])  # second tick of X
    elif len(ixticks)>2:
        msg = "More than 2 tickmarks along one axis is not supported yet."
        raise NotImplementedError(msg)
    else:  # if only one tick is given, assume the second tick is at the origin
        assert i1[0] != 0, "If you use only one x-tick, it cannot be 0."
        i2 = np.array([crossij[0], 0])  # second tick of I
        x2 = np.array([0, 0])  # second tick of X
    if len(jyticks)==2:
        j2 = np.array([0, jyticks[1][0]])
        y2 = np.array([0, jyticks[1][1]])
    elif len(jyticks)>2:
        msg = "More than 2 tickmarks along one axis is not supported yet."
        raise NotImplementedError(msg)
    else:
        assert j1[1] != 0, "If you use only one y-tick, it cannot be 0."
        j2 = np.array([0, crossij[1]])  # second tick of J
        y2 = np.array([0, 0])  # second tick of Y

    xscale = (x2[0]-x1[0])/(i2[0]-i1[0])  # x/I
    yscale = (y2[1]-y1[1])/(j2[1]-j1[1])  # y/J
    scale = np.array([[xscale,0],[0,yscale]])  # 2x2

    jorig = -y2[1]/yscale + j2[1]
    iorig = -x2[0]/xscale + i2[0]
    shift = -np.array([iorig,jorig])  # 2x1

    # Print all variables defined in this function
    # for name, value in locals().items():
    #    print(name, value)

    return scale, shift  # xy = scale.(ij + shift)

def dft(x,nc):
    """
    Computes the first 2*nc + 1 Fourier coefficients of the DFT associated with x,
    and recreates x using only these Fourier coefficients. If you want to return
    x recreated at different points, you will need to create a different def.
    Aka, we get the coefficients -B,-B+1, ..., -1, 0, 1, 2, ..., B, B.
    Or, since the DFT spectrum is periodic, we get 0, 1, ..., 2*B.
    """
    p = len(x)
    n = np.array(list(range(p)))
    k = np.array(list(range(-nc, nc, 1)))
    f = np.array([[np.exp(-1j*2*np.pi/p*i*j) for i in n] for j in k])
    X = np.matmul(f,x)

    return 1/p*np.matmul(X,np.conj(f))


def dftsample(x,nc,nis,nos=200):
    """
    Computes the first 2*nc + 1 Fourier coefficients of the DFT associated with x,
    and recreates x using only these Fourier coefficients. If you want to return
    x recreated at different points, you will need to create a different def.
    Aka, we get the coefficients -B,-B+1, ..., -1, 0, 1, 2, ..., B, B.
    Or, since the DFT spectrum is periodic, we get 0, 1, ..., 2*B.
    """
    p = len(x)

    n = np.array(list(range(p)))
    k = np.array(list(range(-nc, nc, 1)))
    f = np.array([[np.exp(-1j*2*np.pi/p*i*j) for i in n] for j in k])
    X = np.matmul(f,x)

    nsample=np.linspace(0,len(x),nos)
    fsample = np.array([[np.exp(-1j*2*np.pi/p*i*j) for i in nsample] for j in k])
    noutputpoints = np.linspace(nis[0],nis[-1],nos)
    return noutputpoints, 1/p*np.matmul(X,np.conj(fsample))

def nogibbsdftsample(x,nc,nis,nos=200):
    """
    gibbs phenomenom will occur when x[0] != x[-1]. So we just extend the profile
    left and right, linearly, to reach to eachother. Then we discard those points.
    """
    p = len(x)
    extra = int(p*0.2)
    mid = (x[-1] + x[0])/2
    xleftextra = np.linspace(mid,x[0],extra)
    xrightextra = np.linspace(x[-1],mid,extra)
    x_extended = np.array(xleftextra.tolist()+x.tolist()+xrightextra.tolist())
    p_extended = len(x_extended)

    n = np.array(list(range(p_extended)))
    k = np.array(list(range(-nc, nc, 1)))
    f = np.array([[np.exp(-1j*2*np.pi/p_extended*i*j) for i in n] for j in k])

    X = np.matmul(f,x_extended)

    extra_sample = int(nos*0.2)
    nsample_extended=np.linspace(0,len(x_extended),nos+2*extra_sample)
    fsample_extended = np.array([[np.exp(-1j*2*np.pi/p_extended*i*j)\
                                  for i in nsample_extended] for j in k])
    x_extended_subsampled = 1/p_extended*np.matmul(X,np.conj(fsample_extended))

    noutputpoints =  np.linspace(nis[0],nis[-1],nos)
    x_subsampled = x_extended_subsampled[extra_sample:-extra_sample]
    return noutputpoints, x_subsampled
