# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

"""Array image manipulation.

The following functions are used to extract the black, white, red, green or
blue channel from an rgb(a) picture. By extraction it is meant that only
the 'true' color content is extracted (e.g. while a white pixel carries a
255 value in the red channel, we do not wish to extract it as red).

Methods defined here
---------------------
rgb2bw(pic,threshold=255)
    Convert rgb(a) to bw. (NOT grayscale).

rgb2r(pic,threshold=200)
    Extract the red channel from an rgb(a) picture.

rgb2g(pic,threshold=200)
    Extract the green channel from an rgb(a) picture.

rgb2b(pic,threshold=200)
    Extract the blue channel from an rgb(a) picture.

get_xydim(pic)
    Get the x and y dimensions of a picture.

im2xyframe(pic)
    Transpose then flip along the second axis.

xy2imframe(pic)
    Flip along the second axis then transpose.

plotpic(pic, colbar=False)
    Plot a picture. If colbar is True, a colorbar is added.

"""

def rgb2bw(pic,threshold=255):
    """Convert rgb(a) to bw. (NOT grayscale).

    Parameters
    ----------
    pic : np.array
        (x,y,3) or (x,y,4), from.jpg or .png, respectively.
    threshold : int
        The threshold for the conversion. Default is 255, which is pure white.

    Returns
    -------
    np.array
        (x,y) array with 1's and 0's. 1 is black, 0 is white.

    """
    return (np.mean(pic[:,:,:3],axis=-1) < threshold).astype(int)

def rgb2r(pic,threshold=200):
    """Extract the red channel from an rgb(a) picture. It ignores white pixels
    by checking whether the average of the rgb channels is less than 250.

    Parameters
    ----------
    pic : np.array
        (x,y,3) or (x,y,4), from.jpg or .png, respectively.
    threshold : int
        The threshold for red extraction. Default is 200.

    Returns
    -------
    np.array
        (x,y) array with 1's and 0's. 1 is red, 0 is not red.

    """
    return (((pic[:,:,0] > threshold).astype(int) +\
             ((np.mean(pic[:,:,:3], axis=-1) < 250).astype(int)))\
                == 2).astype(int)

def rgb2g(pic,threshold=200):
    """Extract the green channel from an rgb(a) picture. It ignores white pixels
    by checking whether the average of the rgb channels is less than 250.

    Parameters
    ----------
    pic : np.array
        (x,y,3) or (x,y,4), from.jpg or .png, respectively.
    threshold : int
        The threshold for green extraction. Default is 200.

    Returns
    -------
    np.array
        (x,y) array with 1's and 0's. 1 is green, 0 is not green.

    """
    return (((pic[:,:,1] > threshold).astype(int) +\
             ((np.mean(pic[:,:,:3], axis=-1) < 250).astype(int)))\
                == 2).astype(int)

def rgb2b(pic,threshold=200):
    """Extract the blue channel from an rgb(a) picture. It ignores white pixels
    by checking whether the average of the rgb channels is less than 250.

    Parameters
    ----------
    pic : np.array
        (x,y,3) or (x,y,4), from.jpg or .png, respectively.
    threshold : int
        The threshold for blue extraction. Default is 200.

    Returns
    -------
    np.array
        (x,y) array with 1's and 0's. 1 is blue, 0 is not blue.
    """
    return (((pic[:,:,2] > threshold).astype(int) +\
             ((np.mean(pic[:,:,:3], axis=-1) < 250).astype(int)))\
                == 2).astype(int)

def get_xydim(pic):
    """Get the x and y dimensions of a picture."""
    return (pic.shape)[0], (pic.shape)[1]

def im2xyframe(pic):
    """Transpose then flip along the second axis."""
    return (np.flip(pic.T,axis=1))

def xy2imframe(pic):
    """Flip along the second axis then transpose."""
    return (np.flip(pic,axis=1)).T

def plotpic(pic, colbar=False):
    """Plot a picture."""
    fig,ax=plt.subplots()
    ax.imshow(pic)
    if colbar:
        plt.colorbar(f)
    fig.show()
