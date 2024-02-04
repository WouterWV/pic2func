# -*- coding: utf-8 -*-
import numpy as np
import imageio as im
import subprocess
from .imgfuncs import get_xydim, xy2imframe

"""Functions that seek axes, tickmarks and tickvalues in pictures.

Methods defined here
---------------------
detect_axes(pic, minlen_axis=0.1, verbose=False)
    Seek for the x and y axes in a picture.

detect_line(v, L)
    Detects whether there is a line of 1's in a vector v, longer than L.

get_ticks(pic_g, axes)
    Collects the tickvalues from a picture.

get_tickmeans(v)
    Get the indices of the tickmarks in a vector v.

remove_ticks(pic, axes, Iticks, Jticks)
    Removes the tickmarks from the green-channel picture.

remove_tick(pic, J, I)
    Removes a tickmark from the green-channel picture using a floodfill.

get_tickvals(pic, tick_pickids, proximity=0.05)
    Returns a list of 28x28 images of the numbers (tickvals).

scale_numbers(numbers, a=28)
    Scales the list of number arrays to an axa black and white array.

get_numbers_from_groups(groups)
    Returns the numbers from the groups of digits.

reshape_numbers(numbers)
    Reshapes the list of number arrays.

make_square(digit)
    Turns a number array into a square one.

sort_digits_in_numbers(numbers)
    Sorts the digits in the numbers. (digits 1 2 --> 12 or 21?)

group_ticks(ticks,digits)
    Groups the ticks and digits together, such that each tick has its
    corresponding digits.

get_IJcurve(piccurve)
    Returns the indices of the 1's in the piccurve.

obj_edge_dist(A,B)
    Returns the closest distance between x_i and y_j elements.
"""


def detect_axes(pic, minlen_axis=0.1, verbose=False):
    """Seek for the x and y axes in a picture.

    If no axes are found, the bottom left corner is chosen as the origin.
    Warns when multiple axes are detected along one direction, which
    are separated by more than minlen_axis_1 * minlen_axis_2 * N_axis, but
    the longest of the possibilities is chosen by default.

    Parameters
    ----------
    pic : np.array
        (x,y) array with 1's and 0's. It is assumed to be the output of 
        imgfuncs.rgb2bw(original_picture).
    minlen_axis : float
        The minimum length of the axes, in fractions of the picture size. 
        Default is 0.1.
    verbose : bool
        If True, prints the intermediate results. Default is False.

    Returns
    -------
    tuple
        (Iaxis,Jaxis), the indices of the axes in the picture.

    """
    NI,NJ = get_xydim(pic)
    minI = int(minlen_axis*NI)
    minJ = int(minlen_axis*NJ)

    # Initiate axes through origin (bottom left)
    Iaxis = NI
    Jaxis = 0

    if verbose:
        print("NI:",NI)
        print("NJ:",NJ)
        print("minI:",minI)
        print("minJ:",minJ)

    # Detection
    possJaxes, possIaxes = [], []
    possJlens, possIlens = [], []

    for i in range(NI):
        detect, maxL = detect_line(pic[i,:],minJ)
        if detect:
            possJaxes.append(i)
            possJlens.append(maxL)
    for j in range(NJ):
        detect, maxL = detect_line(pic[:,j],minI)
        if detect:
            possIaxes.append(j)
            possIlens.append(maxL)

    if verbose:
        print("possIaxes:",possIaxes)
        print("possJaxes:",possJaxes)
    if possIaxes:  # If possitibilites for J axis are found
        longest_Iaxis_index = np.argmax(possIlens)
        Iaxis = possIaxes[longest_Iaxis_index]
        # Iaxis = max(set(possIaxes), key=possIaxes.count)
    if possJaxes:  # If possitibilites for J axis are found
        longest_Jaxis_index = np.argmax(possJlens)
        Jaxis = possJaxes[longest_Jaxis_index]
        # Jaxis = max(set(possJaxes), key=possJaxes.count)
    if verbose:
        print("I-axis goes through J=",Iaxis)
        print("J-axis goes through I=",Jaxis)
    
    return Iaxis,Jaxis

def detect_line(v, L):
    """
    Detects whether there is a line of 1's in a vector v, longer than L.

    Parameters
    ----------
    v : np.array
        1D array with 1's and 0's.
    L : int
        The minimum length of the line.

    Returns
    -------
    tuple
        (bool, int), where the bool is True if there's a line longer than L,
        and the int is the length of the longest line (0 if no line is found).

    """    
    assert L <= len(v), "Can't detect line larger than the vector itself"

    N = len(v)
    # Add zeros at the extrema's, for the case v starts/ends 
    # with 1s. Then we get a correct start/stop condition at
    # the endpoints. 
    v0 = np.array([0] + v.tolist()+[0]) # add zero at extrema
    diff = (v0[1:]-v0[:-1]) # -1(stop), 0 or 1(start)
    startids = np.where(diff==1)[0]
    stopids = np.where(diff==-1)[0]
    if len(startids) == 0: # No 1s in v ...
        return False, 0
    elif len(stopids) == 0: # One single 1, so len is N - start_idx
        maxL = N - startids[0] # This scenario is impossible due to addboed 0s
        return maxL >= L, maxL
    else: # len(stopids) == len(startids) due to added zeros to v
        maxL = np.max(stopids-startids)
        return  maxL >= L, maxL
    
def get_ticks(pic_g, axes):
    """Collects the tickvalues from a picture. 
    
    Tickmarks are added by the user in green, on the black axes. We can 
    easilty find them by looking for green pixels on the already detected axes.
    
    Parameters
    ----------
    pic_g : np.array
        (x,y,3) array with 1's and 0's. It is assumed to be the output of
        imgfuncs.rgb2g(original_picture).
    axes : tuple
        (Iaxis,Jaxis), the indices of the axes in the picture.

    Returns
    -------
    tuple
        (Iticks, Jticks), the indices of the ticks in the picture.
    """
    NI, NJ = get_xydim(pic_g)
    Iax_Jval = axes[0]
    Jax_Ival = axes[1]
    Iaxis = np.array(([0] + (pic_g[:,Iax_Jval]).tolist() + [0]))
    Jaxis = np.array(([0] + (pic_g[Jax_Ival,:]).tolist() + [0]))
    Iticks = get_tickmeans(Iaxis)
    Jticks = get_tickmeans(Jaxis)
    return Iticks, Jticks

def get_tickmeans(v):
    """Get the indices of the tickmarks in a vector v.

    If no tickmarks are found, we just return 0.

    Parameters
    ----------
    v : np.array
        1D array with 1's and 0's. (An axis line, known to include tickmarks)

    Returns
    -------
    list
        The indices of the tickmarks in the vector.
    """
    diff = (v[1:]-v[:-1])  # -1(stop), 0 or 1(start)
    startids = np.where(diff==1)[0]
    stopids = np.where(diff==-1)[0]
    if len(startids) == 0:  # No 1s in v ...
        return [0],[0]
    elif len(stopids) == 0:  # One single 1, so len is N - start_idx
        raise ValueError("Tickmark not closed. Add b/w pixels on the right.")
    else:  # len(stopids) == len(startids) due to added zeros to v
        assert len(stopids) == len(startids)              
        ticlens = (stopids-startids)
        ticks = []
        for startid,ticklen in zip(startids,ticlens):
            tick = startid + int(ticklen/2)
            ticks.append(tick)
        return ticks

def remove_ticks(pic, axes, Iticks, Jticks):
    """Removes the tickmarks from the green-channel picture.
    As the actual tickvalues are also green colored, I chose to remove the 
    tickmarks (such that they are not interpreted as numbers).

    Parameters
    ----------
    pic : np.array
        (x,y,3) array with 1's and 0's. It is assumed to be the output of
        imgfuncs.rgb2g(original_picture).
    axes : tuple
        (Iaxis,Jaxis), the indices of the axes in the picture.
    Iticks : list
        The indices of the tickmarks in the I-axis.
    Jticks : list
        The indices of the tickmarks in the J-axis.

    Returns
    -------
    np.array
        (x,y) array with 1's and 0's. 1 is green, 0 is not green.
        This is pic with the tickmarks removed.
    list
        The indices of the removed tickmarks.
    """
    Iax_Jval = axes[0]
    Jax_Ival = axes[1]
    tick_ids = []
    for Itick in Iticks:
        pic,tick_id = remove_tick(pic, Iax_Jval, Itick)
        tick_ids.append(tick_id)
    for Jtick in Jticks:
        pic,tick_id = remove_tick(pic, Jtick, Jax_Ival)
        tick_ids.append(tick_id)
    return pic,tick_ids

def remove_tick(pic, J, I):
    """Removes a tickmark from the green-channel picture using a floodfill.
    
    Parameters
    ----------
    pic : np.array
        (x,y,3) array with 1's and 0's. It is assumed to be the output of
        imgfuncs.rgb2g(original_picture).
    J : int
        The index of the tickmark in the J-axis.
    I : int
        The index of the tickmark in the I-axis.
        
    Returns
    -------
    np.array
        (x,y) array with 1's and 0's. 1 is green, 0 is not green.
        This is pic with the tickmark removed.
    tuple
        (I,J), the indices of the removed tickmark.
    """
    ones = [[I,J]]
    done = False
    tick_ids = []
    iteration = 1
    while not done:
        for one in ones:
            pic[one[0], one[1]] = 0
            tick_ids.append(one)
        new_ones = []
        for one in ones:
            i = one[0]
            j = one[1]
            testids = [[i+1,j+1], [i+1,j], [i+1,j-1], [i,j+1], [i,j-1],
                       [i-1,j+1], [i-1,j], [i-1,j-1]]
            for tid in testids:
                if pic[tid[0],tid[1]] == 1:
                    new_ones.append(tid)
        ones = [list(x) for x in set(tuple(x) for x in new_ones)]
        if not ones:  # if ones is empty:
            done = True
        iteration += 1
    return pic,tick_ids

def get_tickvals(pic, tick_pickids, proximity=0.05):
    """
    Returns a list of 28x28 images of the numbers (tickvals). Each listelement 
    is a list, containing the group of ordered digits of that tickvalue. 
    The first element of this listelement list is the tickvalue itself,
    followed by the actual digits.

    Parameters
    ----------
    pic : np.array
        (x,y,3) array with 1's and 0's. It is assumed to be the output of
        imgfuncs.rgb2g(original_picture) WHERE the tickmarks are removed.
    tick_pickids : list
        The indices of the removed tickmarks.

    Returns
    -------

    """
    numbers_picids = []
    ones_left = np.where(pic==1)
    while(len(ones_left[0]) > 0):
        startone = [ones_left[0][0], ones_left[1][0]]
        pic, number = grow_and_remove_number(pic, startone)
        ones_left = np.where(pic == 1)
        numbers_picids.append(number)
    # first el of group is tick, then numbers
    groups = group_ticks(tick_pickids,numbers_picids)
    numbers = get_numbers_from_groups(groups)
    squared_numbers = reshape_numbers(numbers)
    scaled_nrs = scale_numbers(squared_numbers,28)
    return scaled_nrs

def scale_numbers(numbers, a=28):
    """Scales the list of number arrays to an axa black and white array, which 
    is saved to a jpg file. The jpg file serves as input for the classifier.
    
    Parameters
    ----------
    numbers : list
        The list of number arrays. A number array is literally an array that 
        contains the pixels that form a number.
    a : int
        The length of the side of the square. Default is 28.

    Returns
    -------
    list
        The list of scaled number arrays.

    """
    scaled_numbers = []
    for n,number in enumerate(numbers):
        scaled_digits = []
        for d,digit in enumerate(number):
            fn = ""+str(n)+"_"+str(d)+"_digit.tiff"
            on = "scaled_"+str(n)+"_"+str(d)+"_digit.jpg"
            im.imwrite(fn, xy2imframe(digit))
            cmd = ['convert',fn, '-resize', str(a),'-quality', '100', on]
            exe = subprocess.Popen(cmd, stdin = subprocess.PIPE,shell=False)
            exe.communicate()
            scaled_digit = im.imread(on)/255
            scaled_digits.append(scaled_digit)
        scaled_numbers.append(scaled_digits)

    return scaled_numbers
    
def get_numbers_from_groups(groups):
    """Returns the numbers from the groups of digits."""
    numbers = []
    for group in groups:
        digits = []
        for digit in group[1:]:
            digitpixels = []
            for pixel in digit:
                digitpixels.append(pixel)
            digits.append(np.array(digitpixels))
        numbers.append(digits)
    sorted_nrs = sort_digits_in_numbers(numbers)

    return sorted_nrs

def reshape_numbers(numbers):
    """Reshapes the list of number arrays, by turning each number array into 
    a square one (which will later be resized to 28x28 pixels).
    
    """
    # first make square
    shaped_numbers = []
    for number in numbers:
        shaped_digits = []
        for digit in number:
            sq_nr = make_square(digit)
            shaped_digits.append(sq_nr)
        shaped_numbers.append(shaped_digits)

    return shaped_numbers
        
def make_square(digit):
    """Turns a number array into a square one."""
    digit = np.array(digit)
    I_min = np.min(digit[:,0])
    J_min = np.min(digit[:,1])
    digit[:,0] = digit[:,0]-I_min
    digit[:,1] = digit[:,1]-J_min
    I_max = np.max(digit[:,0])
    J_max = np.max(digit[:,1])
    N_max = np.max([I_max+1,J_max+1])
    shift = int(N_max/8)
    delta = int((J_max - I_max)/2)
    digit_array = np.zeros([N_max+shift*2,N_max+shift*2])
    for i,j in zip(digit[:,0],digit[:,1]):
        if delta > 0:
            digit_array[i+shift+delta,j+shift] = 1
        elif delta < 0:
            digit_array[i+shift,j+shift+delta] = 1
        else:
            digit_array[i+shift,j+shift] = 1

    return digit_array
        

def sort_digits_in_numbers(numbers):
    """Sorts the digits in the numbers, such that the first digit is the one
    with the lowest I-value.
    
    """
    #numbers: Ngroups, each list element has Ndigits elements
    sorted_numbers = []
    for number in numbers:
        least_I_vals = []
        for digit in number:
            least_I_vals.append(np.min(digit[:,0]))
        least_I_vals, number = zip(*sorted(zip(least_I_vals, number)))
        least_I_vals, number = (list(t) for t in zip(*sorted(zip(least_I_vals,
                                                                 number))))
        sorted_numbers.append(number)

    return sorted_numbers

def grow_and_remove_number(pic, startone):
    """Grows a number (using a floodfill) from a starting point, and then 
    removes it from the picture.

    Parameters
    ----------
    pic : np.array
        (x,y) array with 1's and 0's. 1 is green, 0 is not green.
        This is pic_g with the tickmarks (and some numbers) removed!
    startone : list
        The starting point of the number to be grown and removed.

    Returns
    -------
    np.array
        (x,y) array with 1's and 0's. 1 is green, 0 is not green.
        This is pic with the number removed.
    list
        The indices of the removed number.

    """
    ones = [[startone[0],startone[1]]]
    done = False
    tick_ids = []
    iteration = 1
    while not done:
        for one in ones:
            pic[one[0],one[1]] = 0
            tick_ids.append(one)
        new_ones = []
        for one in ones:
            i = one[0]
            j = one[1]
            testids = [[i+1,j+1], [i+1,j], [i+1,j-1], [i,j+1], [i,j-1],
                       [i-1,j+1], [i-1,j], [i-1,j-1]]
            for tid in testids:
                if pic[tid[0],tid[1]] == 1:
                    new_ones.append(tid)
        ones = [list(x) for x in set(tuple(x) for x in new_ones)]
        if not ones:  # if ones is empty:
            done = True
        iteration += 1
    return pic, tick_ids

def group_ticks(ticks,digits):
    """Groups the ticks and digits together, such that each tick has its
    corresponding digits. The first element of each group is the tick, then
    the digits.

    Parameters
    ----------
    ticks : list
        The indices of the removed tickmarks.
    digits : list
        The indices of the removed numbers.

    Returns
    -------
    list
        The groups of ticks and digits.

    """
    Ndigits = len(digits)
    # We need Ndigits assignments, and Nticks groups
    assignments = 0
    # First element of each group is the tick, then digits
    groups = [[el] for el in ticks]
    # First we assign one digit to each tickmark, i.e. the closest one
    for i,tick in enumerate(ticks):
        dists = []
        for j,digit in enumerate(digits):
            dists.append(obj_edge_dist(tick, digit))
        min_id = np.argmin(dists)
        groups[i].append(digits.pop(min_id))
        assignments += 1
    # And now we search the closest group for each digit. 
    # Let me assume we don't get weird cases here...
    while assignments < Ndigits:
        for i,digit in enumerate(digits):
            dists = []
            for j,group in enumerate(groups):
                mindist_group_to_el = None
                for groupel in group:
                    if mindist_group_to_el == None:
                        mindist_group_to_el = obj_edge_dist(groupel, digit)
                    else:
                        mintest = obj_edge_dist(groupel,digit)
                        if mintest<mindist_group_to_el:
                            mindist_group_to_el = mintest
                dists.append(mindist_group_to_el)
            min_id = np.argmin(dists)
            groups[min_id].append(digits.pop(0))
            assignments += 1
    return groups

def get_IJcurve(piccurve):
    """Returns the indices of the 1's in the piccurve.
    
    Parameters
    ----------
    piccurve : np.array
        (x,y) array with 1's and 0's. 1 is red, 0 is not red.
        This is the output of imgfuncs.rgb2r (red-channel).

    Returns
    -------
    np.array
        The indices of the 1's in the piccurve.    
    """
    NI,NJ = get_xydim(piccurve)
    IJpoints = []
    for I in range(NI):
        for J in range(NJ):
            if piccurve[I,J] == 1:
                IJpoints.append(np.array([I,J]))
    return np.array(IJpoints)
        
def obj_edge_dist(A,B):
    """Returns the closest distance between A_i and B_j elements."""
    return np.min(np.array([[(a[0] - b[0])**2 + (a[1] - b[1])**2 for a in A]\
                            for b in B]))
