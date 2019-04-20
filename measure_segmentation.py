# Measure segmentation of sheet music

# Imports
import os.path as path
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
import warnings

# Score-retrieval inputs
try:
    import score_retrieval.data as data
    from benchmarks import call_benchmark
except:
    warnings.warn("Warning: Install the score-retrieval repository")


########################
# Measure Segmentation #
########################


def binarize_score(score):
    '''
    params: 
        score: a gray scale image of score
    returns:
        a binarized image of the score
    '''
    gray = cv.bitwise_not(score)
    # Binarize image using adaptive threshold
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    return bw

def find_vertical_lines(score, filter_height = 30):
    '''
    params:
        score: a gray scale image of the score
        filter_height: magic number that adjusts the filter height
    returns:
        a numpy array with only vertical arrays using erosion and dilation
    '''
    # get vertical lines only image
    verticals = np.copy(binarize_score(score))
    # Create erosion and dilation filter
    rows, _ = verticals.shape
    vertical_size = rows // filter_height
    vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
    # Perform erosion and dilation
    verticals = cv.erode(verticals, vertical_structure)
    verticals = cv.dilate(verticals, vertical_structure)

    return verticals

def find_staves(score, split_type = 'average'):
    '''
    params:
          score: a gray scale image of the score
          
          split_type: 'average' or 'strict'. 'average' takes the average
                        between where one staff is predicted to end and the
                        next is predicted to start. It classifies everything up
                        to that line as the first staff and everything after
                        that line as the second (so on and so forth for the
                        rest of the staves). 'strict' splits the staves by
                        where the each staff is predicted to start and end.
                        Another way to think of it is that 'average' guarantees
                        that if you stacked all of the split staves together,
                        you'd get the original image, whereas 'split' does not.
    returns: 
        a list of tuples containing stave start and end positions: [(stave_start, stave_end)]
    '''
    
    # get vertical lines
    verticals = find_vertical_lines(score)

    # Normalize the verticals
    if verticals.max() != 0:
        verts_norm = verticals // verticals.max()
    else:
        verts_norm = verticals

    # Get a horizontal sum of the verticals
    horiz_sum_verts = verts_norm.sum(axis=1)

    # find the mode of the distribution
    horiz_sum_hist = np.bincount(horiz_sum_verts.astype(int))
    mode = np.argmax(horiz_sum_hist)

    # tuples of (start,end) denoting where to split the image at
    staff_split_indices = None
    if split_type == 'average':
        staff_split_indices = list(split_indices_average(horiz_sum_verts, lambda x: x <= mode))
    elif split_type == 'strict':
        staff_split_indices = list(split_indices(horiz_sum_verts, lambda x: x <= mode))
    else:
        raise Exception('Invalid split_type given')
    
    # Stave split indices
    staves_start_end = staff_split_indices
    if len(staff_split_indices) == 0:
        staves_start_end = [(0, score.shape[1])]
    
    return staves_start_end
        

def find_bars(score):
    '''
    params:
        score: A gray scale version of score
    returns: 
        A list contaning a 3-tuple of (stave-index, bar start and bar end)
    '''
    #######
    # Hyperparameters:
    thresholder = True  # Use thresholding for minMax thresholding
    switch_magic_number = 0.01  # Threshold for deciding whether to add all bars or no
    clean_up = True     # Use the bar cleanup algorithm to remove small bars
    width_magic_number = 10     # Minimum % width threshold for bar cleanup algorithm
    #######

    staves_start_end = find_staves(score)
    bars_start_end = []
    noisy_verticals = find_vertical_lines(score, filter_height=40) 

    for start, end in staves_start_end:
        # for each staff, find maxima
        one_staff = list(cut_array(noisy_verticals, [(start, end)]))[0]
        sum_array = one_staff.sum(axis=0)
        maxima = find_peaks(sum_array)
        maxima_list = [(sum_array[i], i) for i in maxima[0]]
        maxima_list = sorted(maxima_list)

        if maxima_list != []:
            minimum = maxima_list[0][0]
            maximum = maxima_list[-1][0]
            # Perform min_max threshold
            if thresholder:
                if abs(maximum - minimum) / noisy_verticals.shape[1] > switch_magic_number:
                    threshold = (maxima_list[0][0] + maxima_list[-1][0]) / 2  
                    filtered = [x[1] for x in maxima_list if x[0] > threshold ]
                else:
                    filtered = [x[1] for x in maxima_list]
            else:
                filtered = [x[1] for x in maxima_list]

            # Sort out the bars by width
            filtered = sorted(filtered)
            bars_in_this_stave = []
            for i in filtered:
                bars_in_this_stave += [(i, start, end)]
            
            # Perform the cleanup algorithm
            if clean_up:
                cleaned_up_bars = cleanup_bars(bars_in_this_stave, score.shape[0] / width_magic_number )
                if cleaned_up_bars is not None:
                    bars_start_end += cleaned_up_bars
            else:
                bars_start_end += bars_in_this_stave
        else:
            bars_start_end += [(0, start, end)]
            bars_start_end += [(score.shape[0], start, end)]
             
    return bars_start_end


##########################
# Retrieval and Printing #
##########################

def create_bar_waveforms(score):
    '''
    params:
        score: a gray scale input image
    returns:
        a benchmark call to the pytorch cnn
    Note: this function utilizes score-retrieval
    '''
    #################
    # Hyperparameters
    bar_height = 128
    bar_width = 128
    #################

    bars_start_end = find_bars(score)
    im_list = []
    if len(bars_start_end) <= 1: # if there is one bar, split staff into 2 parts
        im_list.append(score[bars_start_end[0][1]:bars_start_end[0][2], 0:bars_start_end[0][0]])
        im_list.append(score[bars_start_end[0][1]:bars_start_end[0][2], bars_start_end[0][0]:score.shape[1]])
    # Cycle through all bars and create crops
    for i in range(len(bars_start_end) - 1):
        cropped_bar = score[bars_start_end[i][1]:bars_start_end[i][2], bars_start_end[i][0]:bars_start_end[i+1][0]]
        if cropped_bar.size != 0:
            im_list.append(cropped_bar)
    
    # Downsample all images
    images = [downsample_image(cv.cvtColor(bar,cv.COLOR_GRAY2RGB), height=bar_height, width=bar_width)
                for bar in im_list ]
    if images ==[]:
        return None
    # Perform the benchmark call
    return call_benchmark(images=images)


##################
# Helper Methods #
##################

def split_indices(array, comparator=(lambda x: x == 0)):
    '''Input: 1-D array of indicies of zeros of horizontal summation
    Output: Generator of indicies to split images by discontinuities in zeros'''
    indices = np.where(comparator(array))[0]
    # we dont want to add 1 to last element
    for i in range(indices.size - 1):
        if indices[i+1] - indices[i] != 1:
            yield (indices[i], indices[i+1])

def split_indices_average(array, comparator=(lambda x: x == 0)):
    '''Input: 1-D array of indicies of zeros of horizontal summation
    Output: Iterator of indicies to split image at by average of zeros'''
    line_pair = list(split_indices(array, comparator))
    line_pair = [(0, 0)] + line_pair + [(array.size-1, array.size-1)]
    for i in range(len(line_pair) - 2):
        a = line_pair[i][1]
        b = line_pair[i+1][0]
        a1 = line_pair[i+1][1]
        b1 = line_pair[i+2][0]
        yield ( a + ((b-a)//2) , a1 + ((b1-a1)//2))

def cut_array(array, positions, direction="H"):
    '''Input: array: image array, positions: array of start end tuples
       Output: array of image arrays cut by positions'''
    for start , end in positions:
        if (direction == "H"):
            yield array[start:end, :]
        else:
            yield array[:, start:end]

def cleanup_bars(bars, width):
    '''
    Cleans up a set of bars in staves globally using a recursive approach
    params: 
        bars: bars in a stave to clean up
        width: width threshold above which a bar is considered a bar
    returns: 
        cleaned up bars
    '''
    # Atleast have 2 bars
    if len(bars) > 1:
        l_diffs = []
        # calculate the distances between bars
        for i in range(len(bars) - 1):
            l_diffs.append(abs(bars[i][0] - bars[i+1][0]))
        # Check if base case is triggered
        if min(l_diffs) < width:
            # Select the appropriate bar to omit
            lowest_index = l_diffs.index(min(l_diffs))
            if lowest_index == 0:
                new_bars = [bars[0]] + bars[2:]
            elif lowest_index == len(l_diffs) - 1:
                new_bars = bars[0:-2] + [bars[-1]]
            else:
                if l_diffs[lowest_index - 1] < l_diffs[lowest_index+1]:
                    new_bars = bars[0:lowest_index] + bars[lowest_index+1:]
                else:
                    new_bars = bars[0:lowest_index+1] + bars[lowest_index+2:]

            # recursively cleanup remaining bars
            return cleanup_bars(new_bars, width)
        else:
            # base case
            return bars
    else:
        return bars

def downsample_image(image, rate=None, width=None, height=None):
    '''
    Downsamples 'image' by a ratio 'rate' or by a mentioned size ('width' and 'height')
    '''
    if rate is not None:
        new_shape = (int(image.shape[0] * rate), int(image.shape[1] * rate))
    if width is not None and height is not None:
        new_shape = (width, height)
    return cv.resize(image, new_shape)