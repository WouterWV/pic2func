# -*- coding: utf-8 -*-
import numpy as np

"""Functions to convert IJ data to XY data.

Methods defined here
---------------------
get_XYfunc(curveij, ixticks, jyticks, axes)
    Converts the IJ curve to an XY function.

IJcurve_to_IJfunction(curve)
    Convert an IJ curve into an IJ function.

get_scaleandshift_IJtoXY(ixticks, jyticks, axes)
    Using the ticks, the IJ function can be converted to an XY function.

dft(x, Nc)
    Computes the first 2*Nc + 1 Fourier coeffs of the DFT associated with x,
    and recreates x using only these Fourier coefficients. If you want to return
    x recreated at different points, you will need to create a different def.

dftsample(x, Nc, Nis, Nos=200)
    Computes the first 2*Nc + 1 Fourier coeffs of the DFT associated with x,
    and recreates x using only these Fourier coefficients. If you want to return
    x recreated at different points, you will need to create a different def.

nogibbsdftsample(x, Nc, Nis, Nos=200)
    gibbs phenomenom will occur when x[0] != x[-1]. So we just extend the
    profile left and right, linearly, to reach to eachother. Then we discard
    those points.

"""

def get_XYfunc(curveij,ixticks,jyticks,axes):
    """
    Converts the IJ curve to an XY function.
    Step1: Convert the IJ curve to an IJ function.
    Step2: Use the ticks to convert the IJ function to an XY function.

    Parameters
    ----------
    curveij : Nx2 array
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
        N_unique_Ivals x 2 array with (x,y) values.

    """
    fij = IJcurve_to_IJfunction(curveij) #
    scale_ij_to_xy, shift_ij = get_scaleandshift_IJtoXY(ixticks, jyticks, axes)
    fxy = np.matmul((fij+shift_ij),scale_ij_to_xy)
    return fxy

def IJcurve_to_IJfunction(curve):
    """Convert an IJ curve into an IJ function.

    The IJ curve is the drawing by the user. For each x value, there are 
    multiple y values. This function will average the y values for each x value.
    
    Parameters
    ----------
    curve : Nx2 array
        (I,J) points of the curve.
        
    Returns
    -------
    np.array
        N_unique_Ivals x 2 array with (I,J) points of the function.

    """
    Ivals = curve[:,0]
    Jvals = curve[:,1]
    sortids = Ivals.argsort()
    Isort = Ivals[sortids]
    Jsort = Jvals[sortids]
    ijfunc = []
    ivals=[]
    spin = Isort[0]
    val = [Jsort[0]]
    for I,J in zip(Isort[1:],Jsort[1:]):
        if I == spin:
            val = val + [J]
        else:
            ijfunc.append(np.mean(np.array(val)))
            ivals.append(spin)
            spin = I
            val = [J]

    return (np.array([ivals, ijfunc])).T

def get_scaleandshift_IJtoXYold(ixticks, jyticks, axes):
    """Using the ticks, the IJ function can be converted to an XY function.
    
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
        2x1 array with the shift.

    """
    # First determine if origin is given
    origin_in_ticks = False
    for ixtick,jytick in zip(ixticks,jyticks):
        if (ixtick[-1] == 0) or (jytick[-1] == 0):
            origin_in_ticks == True
    if origin_in_ticks:
        raise NotImplementedError #TODO
    else:
        # check if multiple ticks are given in X or Y direction:
        if len(ixticks)>1 or len(jyticks)>1:
            raise NotImplementedError #TODO, check consistency of ticks...
        else:
            ixtick = np.array(ixticks[0],dtype=float)
            jytick = np.array(jyticks[0],dtype=float)
            originIJ = np.array([axes[1],axes[0]])
            originXY = np.array([0,0])
            xscale = ixtick[-1]/(ixtick[0]-originIJ[0])  # x/I
            yscale = jytick[-1]/(jytick[0]-originIJ[1])  # y/J
    scale = np.array([[xscale,0],[0,yscale]]) #2x2
    IJshift = -originIJ  # 2x1
    print("input:\n")
    print("ixticks: ",ixticks)
    print("jyticks: ",jyticks)
    print("axes: ",axes)
    print("intermediate:\n")
    print("originIJ:", originIJ)
    print("output:\n ")
    print("scale: ",scale)
    print("IJshift: ",IJshift)
    return scale, IJshift  # xy = scale.(IJ + IJshift)

def get_scaleandshift_IJtoXY(ixticks, jyticks, axes):
    """Using the ticks, the IJ function can be converted to an XY function.

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
        2x1 array with the shift (negative of origin in IJ coordinates).

    """
    # We need at least one tick in each direction.
    assert len(ixticks) > 0, "No ticks found in I axis."
    assert len(jyticks) > 0, "No ticks found in J axis."
    # Define the crosspoint of I, J axes in IJ coordinates
    crossIJ = np.array([axes[1],axes[0]])  # crosspoint of the axes
    # We don't know whether crossIJ corresponds to the origin, as the user may
    # use a 0 tick.
    originIJ = None
    # Define the first tick of I, J axes in IJ coordinates
    I1 = np.array([ixticks[0][0], 0])  # first tick of I
    J1 = np.array([0, jyticks[0][0]])  # first tick of J
    # And their corresponding ticks in XY coordinates
    X1 = np.array([ixticks[0][1], 0])  # first tick of X
    Y1 = np.array([0, jyticks[0][1]])  # first tick of Y
    # If a second tick is given, define it as well
    if len(ixticks)==2:
        I2 = np.array([ixticks[1][0], 0])  # second tick of I
        X2 = np.array([ixticks[1][1], 0])  # second tick of X
    elif len(ixticks)>2: 
        msg = "More than 2 tickmarks along one axis is not supported yet."
        raise NotImplementedError(msg)
    else:  # if only one tick is given, assume the second tick is at the origin
        assert I1[0] != 0, "If you use only one x-tick, it cannot be 0."
        I2 = np.array([crossIJ[0], 0])  # second tick of I
        X2 = np.array([0, 0])  # second tick of X
    if len(jyticks)==2:
        J2 = np.array([0, jyticks[1][0]])
        Y2 = np.array([0, jyticks[1][1]])
    elif len(jyticks)>2:
        msg = "More than 2 tickmarks along one axis is not supported yet."
        raise NotImplementedError(msg)
    else:
        assert J1[1] != 0, "If you use only one y-tick, it cannot be 0."
        J2 = np.array([0, crossIJ[1]])  # second tick of J
        Y2 = np.array([0, 0])  # second tick of Y

    xscale = (X2[0]-X1[0])/(I2[0]-I1[0])  # x/I
    yscale = (Y2[1]-Y1[1])/(J2[1]-J1[1])  # y/J
    scale = np.array([[xscale,0],[0,yscale]])  # 2x2

    Jorig = -Y2[1]/yscale + J2[1]
    Iorig = -X2[0]/xscale + I2[0]
    shift = -np.array([Iorig,Jorig])  # 2x1

    # Print all variables defined in this function
    # for name, value in locals().items():
    #    print(name, value)

    return scale, shift  # xy = scale.(IJ + shift)

def dft(x,Nc):
    """
    Computes the first 2*Nc + 1 Fourier coefficients of the DFT associated with x,
    and recreates x using only these Fourier coefficients. If you want to return
    x recreated at different points, you will need to create a different def.
    Aka, we get the coefficients -B,-B+1, ..., -1, 0, 1, 2, ..., B, B. 
    Or, since the DFT spectrum is periodic, we get 0, 1, ..., 2*B. 
    """
    P = len(x)
    N = np.array([i for i in range(P)])
    K = np.array([i for i in range(-Nc,Nc,1)])   
    F = np.array([[np.exp(-1j*2*np.pi/P*i*j) for i in N] for j in K])
    X = np.matmul(F,x)

    return 1/P*np.matmul(X,np.conj(F))


def dftsample(x,Nc,Nis,Nos=200):
    """
    Computes the first 2*Nc + 1 Fourier coefficients of the DFT associated with x,
    and recreates x using only these Fourier coefficients. If you want to return
    x recreated at different points, you will need to create a different def.
    Aka, we get the coefficients -B,-B+1, ..., -1, 0, 1, 2, ..., B, B. 
    Or, since the DFT spectrum is periodic, we get 0, 1, ..., 2*B. 
    """
    P = len(x)
    
    N = np.array([i for i in range(P)])
    K = np.array([i for i in range(-Nc,Nc,1)])   
    F = np.array([[np.exp(-1j*2*np.pi/P*i*j) for i in N] for j in K])
    X = np.matmul(F,x)
    
    Nsample=np.linspace(0,len(x),Nos)
    Fsample = np.array([[np.exp(-1j*2*np.pi/P*i*j) for i in Nsample] for j in K])
    Noutputpoints = np.linspace(Nis[0],Nis[-1],Nos)
    return Noutputpoints, 1/P*np.matmul(X,np.conj(Fsample))

def nogibbsdftsample(x,Nc,Nis,Nos=200):
    """
    gibbs phenomenom will occur when x[0] != x[-1]. So we just extend the profile
    left and right, linearly, to reach to eachother. Then we discard those points.
    """
    tempP = len(x)
    P = len(x)
    extra = int(P*0.2)
    mid = (x[-1] + x[0])/2
    xleftextra = np.linspace(mid,x[0],extra)
    xrightextra = np.linspace(x[-1],mid,extra)
    x_extended = np.array(xleftextra.tolist()+x.tolist()+xrightextra.tolist())
    P_extended = len(x_extended)
    
    N = np.array([i for i in range(P_extended)])
    K = np.array([i for i in range(-Nc,Nc,1)])   
    F = np.array([[np.exp(-1j*2*np.pi/P_extended*i*j) for i in N] for j in K])
    
    X = np.matmul(F,x_extended)
    
    extra_sample = int(Nos*0.2)
    Nsample_extended=np.linspace(0,len(x_extended),Nos+2*extra_sample)
    Fsample_extended = np.array([[np.exp(-1j*2*np.pi/P_extended*i*j) for i in Nsample_extended] for j in K])
    Noutputpoints_extended = np.linspace(Nis[0],Nis[-1],Nos+2*extra_sample)
    x_extended_subsampled = 1/P_extended*np.matmul(X,np.conj(Fsample_extended))
    
    Noutputpoints =  np.linspace(Nis[0],Nis[-1],Nos)
    x_subsampled = x_extended_subsampled[extra_sample:-extra_sample]
    return Noutputpoints, x_subsampled
