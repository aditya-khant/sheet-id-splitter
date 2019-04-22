# Based in part off of: https://docs.opencv.org/4.0.0/dd/dd7/tutorial_morph_lines_detection.html

import os.path as path
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from scipy.signal import find_peaks

from deprecated_measure_segmentation.score_splitter_cleanup import start_end_voice_lines
from deprecated_measure_segmentation.score_splitter_cleanup import start_end_voice_lines_by_staff
from deprecated_measure_segmentation.score_splitter_cleanup import find_horizontal_lines
import deprecated_measure_segmentation.tsai_bars as tb

# requires score_retrieval
import score_retrieval.data as data
# requires cnnpytorch
from benchmarks import call_benchmark


# TODO: add bar splitting, add ability to iterate through directory, unify docstring format.
class Score:
    def __init__(self, score, name):
        '''
        params:
          score - a grayscale image of the score
          name  - identifier for the score
        '''
        # binary conversion
        print("score.shape =", score.shape)
        gray = cv.bitwise_not(score)
        bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
        self._score_gray = gray
        self._score     = score
        self._score_bw  = bw
        self._name      = name
        self._verticals = None
        self._noisy_verticals = None #useful for bar line detection (Also note this is normalized)
        self._staves    = None
        self._staves_verticals = None
        self._staves_start_end = []
        self._bars = None
        self._bars_start_end = []
        self._bar_waveform = None
        self._voice_lines_by_staff = None
        self._voice_lines_by_page = None
        self._horizontals = None
        # TODO: eventually structure as 3-dimensional array of images
        # dimension 0: staves
        # dimension 1: bars
        # dimension 2: voices

    def _find_vertical_lines(self):
        '''
        generates an image of the same shape as 'self._score_bw' where only the vertical lines remain.
        '''
        self._horizontals = find_horizontal_lines(self._score_bw)
        self._verticals = np.copy(self._score_bw)
        self._noisy_verticals = np.copy(self._score_bw)
        # Specify size on vertical axis
        rows, _ = self._verticals.shape
        # TODO: Smaller the magic number bigger the filtered out lines are
        # TODO: figure out how to find the optimal value
        vertical_size = rows // 30
        # Create structure element for extracting vertical lines through morphology operations
        vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size)) # the one should be changed
        # Apply morphology operations
        self._verticals = cv.erode(self._verticals, vertical_structure)
        self._verticals = cv.dilate(self._verticals, vertical_structure)

        vertical_size = rows // 40
        # Create structure element for extracting vertical lines through morphology operations
        vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
        self._noisy_verticals = cv.erode(self._noisy_verticals, vertical_structure)
        self._noisy_verticals = cv.dilate(self._noisy_verticals, vertical_structure)
        #normalized
        self._noisy_verticals = self._noisy_verticals // self._noisy_verticals.max()

    def _find_staves(self, split_type = 'average', plot_split_lines = False, imwrite=False):
        '''
        params:
          split_type -- 'average' or 'strict'. 'average' takes the average
                        between where one staff is predicted to end and the
                        next is predicted to start. It classifies everything up
                        to that line as the first staff and everything after
                        that line as the second (so on and so forth for the
                        rest of the staves). 'strict' splits the staves by
                        where the each staff is predicted to start and end.
                        Another way to think of it is that 'average' guarantees
                        that if you stacked all of the split staves together,
                        you'd get the original image, whereas 'split' does not.
          plot_split_lines -- plot and save an image depicting where to split
                              to get the staves
          imwrite -- use openCv to draw lines and save
        returns:
          an array of the separated staves of the score
        '''
        if self._verticals is None:
            self._find_vertical_lines()

        # normalize and find the horizontal sum of the vertical lines
        if self._verticals.max() != 0:
            verts_norm = self._verticals // self._verticals.max()
        else:
            verts_norm = self._verticals
        horiz_sum_verts = verts_norm.sum(axis=1)

        horiz_sum_hist = np.bincount(horiz_sum_verts.astype(int))
        avg_min = np.argmax(horiz_sum_hist)

        # tuples of (start,end) denoting where to split the image at
        staff_split_indices = None
        if split_type == 'average':
            staff_split_indices = list(split_indices_average(horiz_sum_verts, lambda x: x <= avg_min))
        elif split_type == 'strict':
            staff_split_indices = list(split_indices(horiz_sum_verts, lambda x: x <= avg_min))
        else:
            raise Exception('Invalid split_type given')




        # if told to, plot the source
        if plot_split_lines:
            plt.figure(figsize=(10,13))
            plt.imshow(self._score, aspect="auto", cmap = "gray")

        if imwrite:
            img_color = cv.cvtColor(self._score ,cv.COLOR_GRAY2RGB)

        # split the score and verticals
        self._staves = []
        self._staves_verticals = []
        self._staves_start_end = staff_split_indices
        for (start, end) in staff_split_indices:
            self._staves.append(self._score[start:end])
            self._staves_verticals.append(self._verticals[start:end])
            # if told to, plot the split lines
            if plot_split_lines:
                plt.axhline(y=start, color='r')
                plt.axhline(y=end, color='r')
            if imwrite:
                cv.line(img_color, (0, start), (self._score.shape[1], start), (255,0,0), 5 )
                cv.line(img_color, (0, end), (self._score.shape[1], end), (255,0,0), 5 )

        if len(staff_split_indices) == 0:
            self._staves_start_end = [(0, self._score.shape[1])]
            self._staves.append(self._score)
            self._staves_verticals.append(self._verticals)

        # if told to, save the image
        if plot_split_lines:
            plt.savefig('{}.png'.format(self._name))
            plt.clf()

        if imwrite:
            print('{}.png'.format(self._name))
            cv.imwrite('{}.png'.format(self._name), img_color)

    def _find_bars(self):
        '''
        Finds the bars in the image
        '''
        if self._staves is None:
            self._find_staves()
        self._bars = []
        self._bars_start_end = []
        for i in range(len(self._staves_verticals)):
            staff = self._staves[i]
            staff_vert = self._staves_verticals[i]
            verts_norm = staff_vert // staff_vert.max()
            vert_sum_verts = verts_norm.sum(axis=0)
            threshold = np.sort(vert_sum_verts)[-1*(vert_sum_verts.size // 10)]
            bar_split_indices = list(split_indices(vert_sum_verts, lambda x: x >= threshold))
            self._bars_start_end.append(bar_split_indices)
            for start, end in bar_split_indices:
                self._bars.append(staff[start:end])


    def _create_vertical_input(self):
        """Returns vertical images"""
        if self._verticals is None:
            self._find_vertical_lines()
        image = downsample_image(cv.cvtColor(self._verticals,cv.COLOR_GRAY2RGB), by_rate=False, by_size=True)
        if image ==[]:
            return None
        return call_benchmark(images=[image])

    def _create_bar_waveforms(self):
        '''
        Returns list of vertical sum waveforms
        '''
        if self._bars is None:
            self._find_bars()
        self._bar_waveform = []
        for bar in self._bars:
            self._bar_waveform.append(bar.sum(axis=0))

    def _create_staff_waveforms(self):
        '''
        Returns list of vertical sum waveforms
        '''
        if self._staves is None:
            self._find_staves()
        self._staff_waveform = []
        for staff in self._staves:
            self._staff_waveform.append(staff.sum(axis=0))

    def _create_cnn_staff_waveforms(self):
        if self._staves is None:
            self._find_staves()
        if self._staves == []:
            return None
        # downsample then convert to RGB
        shape_min_width, shape_min_height = min(staff.shape for staff in self._staves)
        images = [downsample_image(cv.cvtColor(staff,cv.COLOR_GRAY2RGB), by_rate=False, by_size=True)
                  for staff in self._staves]
        if images ==[]:
            return None
        return call_benchmark(images=images)

    def _create_cnn_bars_waveforms(self):
        if self._bars_start_end == []:
            self._find_bars_using_peaks()
        # downsample then convert to RGB
        im_list = []
        if len(self._bars_start_end) <= 1: # if there is one bar, split staff into 2 parts
            im_list.append(self._score[self._bars_start_end[0][1]:self._bars_start_end[0][2], 0:self._bars_start_end[0][0]])
            im_list.append(self._score[self._bars_start_end[0][1]:self._bars_start_end[0][2], self._bars_start_end[0][0]:self._score.shape[1]])
        for i in range(len(self._bars_start_end) - 1):
            cropped_bar = self._score[self._bars_start_end[i][1]:self._bars_start_end[i][2], self._bars_start_end[i][0]:self._bars_start_end[i+1][0]]
            if cropped_bar.size != 0:
                # print("cropped_bar.shape =", cropped_bar.shape)
                im_list.append(cropped_bar)
        bar_height = 128
        bar_width = 128
        images = [downsample_image(cv.cvtColor(bar,cv.COLOR_GRAY2RGB), by_rate=False, by_size=True, height=bar_height, width=bar_width)
                  for bar in im_list ]
        if images ==[]:
            return None
        return call_benchmark(images=images)

    def _find_voice_lines_page(self):
        """Finds voice lines on a page"""
        if self._voice_lines_by_page is None:
            self._voice_lines_by_page = []
        sum_array = np.sum(self._score_gray, axis=1)
        minima = argrelextrema(sum_array, np.less)
        minima_list = [(sum_array[i], i) for i in minima[0]]
        minima_list = sorted(minima_list)
        if minima_list != []:
            threshold = (minima_list[0][0] + minima_list[-1][0]) / 2  #minMax Threshold
            filtered_minima = [x[1] for x in minima_list if x[0] < threshold ]
            filtered_minima = sorted(filtered_minima)
            self._voice_lines_by_page += filtered_minima
        else:
            self._voice_lines_by_page += 0

    def _find_voice_lines(self):
        """Find voice lines from staves"""
        if self._staves is None:
            self._find_staves()
        if self._staves == []:
            return None
        if self._voice_lines_by_staff is None:
            self._voice_lines_by_staff = []
        for staff in self._staves:
            sum_array = np.sum(staff, axis=1)
            minima = argrelextrema(sum_array, np.less)
            minima_list = [(sum_array[i], i) for i in minima[0]]
            minima_list = sorted(minima_list)
            if minima_list == []:
                self._voice_lines_by_staff.append([])
                continue
            threshold = (minima_list[0][0] + minima_list[-1][0]) / 2  #minMax Threshold
            filtered_minima = [x[1] for x in minima_list if x[0] < threshold ]
            filtered_minima = sorted(filtered_minima)
            self._voice_lines_by_staff.append(filtered_minima)


    def _generate_pretty_image(self, bars=True, staves = True, voice=False, voice_by_page = False):
        '''
        Generates bars and staves on an image
        '''
        if self._bars is None:
            self._find_bars()
        if self._voice_lines_by_staff is None:
            self._find_voice_lines()

        img_color = cv.cvtColor(self._score ,cv.COLOR_GRAY2RGB)
        for (staff_start, staff_end), bar_lines, voice_lines in zip(self._staves_start_end, self._bars_start_end, self._voice_lines_by_staff):
            if (staves):
                cv.line(img_color, (0, staff_start), (self._score.shape[1], staff_start), (255,0,0), 5 )
                cv.line(img_color, (0, staff_end), (self._score.shape[1], staff_end), (255,0,0), 5 )
            if (bars):
                if len(bar_lines[0]) == 2:
                    for (bar_start, bar_end) in bar_lines:
                        cv.line(img_color, (bar_start, staff_start), (bar_start, staff_end), (0,0,255), 5 )
                        cv.line(img_color, (bar_end, staff_start), (bar_end, staff_end), (0,0,255), 5 )
            if voice:
                for line_val in voice_lines:
                    cv.line(img_color, (0, staff_start + line_val), (self._score.shape[1], staff_start + line_val), (0,255,0), 5 )

        cv.imwrite('{}.png'.format(self._name), img_color)

    def _print_with_bars(self, toggle="staves", stuff = "12", name = None, vert = False):
        """Prints bars and staves for new tuples of bars"""
        if name is None:
            name = self._name

        if toggle == "staves":
            self._find_bars_using_staves()
        elif toggle == "peaks":
            self._find_bars_using_peaks()
        elif toggle == "intersect":
            self._find_bars_by_intersection()
        elif toggle == "hybrid" or toggle == "tb":
            self._find_bars_using_tb()
        else:
            raise Exception("Check Toggle")
        if (vert):
            img_color = cv.cvtColor(cv.bitwise_not(self._verticals) ,cv.COLOR_GRAY2RGB)
        else:
            img_color = cv.cvtColor(self._score ,cv.COLOR_GRAY2RGB)
        print("Staves Length: {}".format(len(self._staves_start_end)))
        print("Bars Length: {}".format(len(self._bars_start_end)))
        if "1" in stuff:
            for (staff_start, staff_end) in self._staves_start_end:
                cv.line(img_color, (0, staff_start), (self._score.shape[1], staff_start), (255,0,0), 2 )
                cv.line(img_color, (0, staff_end), (self._score.shape[1], staff_end), (255,0,0), 2 )
        if "2" in stuff:
            for i, start, end in self._bars_start_end:
                cv.line(img_color, (i, start), (i, end), (0, 0,255), 2)
        cv.imwrite('{}.png'.format(name), img_color)

    def _find_bars_using_staves(self):
        """Finds bars using top 5 and bottom 5 pixels"""
        if self._staves is None:
            self._find_staves()
        self._bars_start_end = []

        magic_number = 5
        cole_voice_lines = start_end_voice_lines_by_staff(self._staves_start_end, self._verticals, self._horizontals)
        for lines in cole_voice_lines:
            if lines != []:
                start = lines[0][0]
                end = lines[-1][-1]
                for i in range(self._verticals.shape[1]):
                    if self._verticals[start + magic_number][i]:
                        if self._verticals[end - magic_number][i]:
                            self._bars_start_end += [(i, start, end)]


    def _find_bars_using_peaks(self, clean_up = True, thresholder = True):
        """Uses peaks and min maxing to find bars"""
        if self._staves is None:
            self._find_staves()
        self._bars_start_end = []
        self._bars = []

        for start, end in self._staves_start_end:
            one_staff = list(cut_array(self._noisy_verticals, [(start, end)]))[0]
            sum_array = one_staff.sum(axis=0)
            maxima = find_peaks(sum_array)
            maxima_list = [(sum_array[i], i) for i in maxima[0]]
            maxima_list = sorted(maxima_list)

            switch_magic_number = 0.01
            thresh_magic_number = 2
            bar_list = []
            if maxima_list != []:
                minimum = maxima_list[0][0]
                maximum = maxima_list[-1][0]
                if thresholder:
                    if abs(maximum - minimum) / self._noisy_verticals.shape[1] > switch_magic_number:
                        threshold = (maxima_list[0][0] + maxima_list[-1][0]) / thresh_magic_number   #minMax Threshold
                        filtered = [x[1] for x in maxima_list if x[0] > threshold ]
                    else:
                        filtered = [x[1] for x in maxima_list]
                else:
                    filtered = [x[1] for x in maxima_list]
                filtered = sorted(filtered)
                bars_in_this_stave = []
                for i in filtered:
                    bars_in_this_stave += [(i, start, end)]
                    bar_list.append(i)

                if clean_up:
                    width_magic_number = 10
                    cleaned_up_bars = cleanup_bars(bars_in_this_stave, self._score.shape[0] / width_magic_number )
                    if cleaned_up_bars is not None:
                        self._bars_start_end += cleaned_up_bars
                else:
                    self._bars_start_end += bars_in_this_stave
            else:
                self._bars_start_end += [(0, start, end)]
                self._bars_start_end += [(self._score.shape[0], start, end)]
                bar_list.append(0)
                bar_list.append(self._score.shape[0])
            self._bars.append(bar_list)

    def _find_bars_using_tb(self, clean_up = False, path = None):
        if self._staves is None:
            self._find_staves()
        self._bars_start_end = []
        self._bars = []
        # measures = tb.extractMeasuresHybrid(self._score)
        # if measures is not None:
        #     while (len(measures) > 0):
        #         meas = measures[0]
        #         for start, end in self._staves_start_end:
        #             if (meas[0] > start and meas[0] < end) or (meas[2] > start and meas[2] < end):
        #                 self._bars_start_end += [(meas[1], start, end)]
        #                 self._bars_start_end += [(meas[3], start, end)]
        #                 print("triggered")
        #                 break
        #         measures = measures[1:]


        for start, end in self._staves_start_end:
            one_staff = list(cut_array(self._score, [(start, end)]))[0]
            bar_lines = tb.extractMeasures(one_staff, visualize=True, path=path)
            bar_list = []
            bars_in_this_stave = []
            # print(bar_lines)
            if bar_lines is not None:
                for i, j, k, l in bar_lines:
                    bars_in_this_stave += [(j, start, end)]
                    bar_list.append(i)
                    bars_in_this_stave += [(l, start, end)]
                    bar_list.append(j)
                if clean_up:
                    width_magic_number = 10
                    cleaned_up_bars = cleanup_bars(bars_in_this_stave, self._score.shape[0] / width_magic_number )
                    if cleaned_up_bars is not None:
                        self._bars_start_end += cleaned_up_bars
                else:
                    self._bars_start_end += bars_in_this_stave
            else:
                self._bars_start_end += [(0, start, end)]
                self._bars_start_end += [(self._score.shape[0], start, end)]
                bar_list.append(0)
                bar_list.append(self._score.shape[0])

            self._bars.append(bar_list)

    def _find_bars_by_intersection(self):
        if self._staves is None:
            self._find_staves()
        self._bars_start_end = []
        intersections = np.logical_and(self._horizontals, self._verticals)
        for start , end in self._staves_start_end:
            staff = intersections[start:end, :]
            sum_staff = staff.sum(axis=0)
            bar_lines = find_peaks(sum_staff)
            bars_for_staff = [(i,start, end) for i in bar_lines[0]]
            self._bars_start_end += bars_for_staff



def downsample_image(image, by_rate= True, rate=0.3, by_size=False, width=1024, height=1024):
    '''
    Downsamples 'image' by a ratio 'rate' or by a mentioned size ('width' and 'height')
    '''
    if by_rate:
        new_shape = (int(image.shape[0] * rate), int(image.shape[1] * rate))
    if by_size:
        new_shape = (width, height)
    return cv.resize(image, new_shape)

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

def test_staves(dataset='mini_dataset', output_dir='./test_staves/'):
    '''
    Test the staff splitting by rendering where the score would be split for
    each file.
    '''
    for i, (label, image_file) in enumerate(data.index_images(dataset=dataset)):
        image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
        name = path.split(label)[-1]
        print('processing image {0} with name {1}'.format(i, name))
        # add 'i' to disambiguate pieces
        s = Score(image, output_dir + name + str(i))
        s._find_staves(imwrite= True)
        create_waveforms(image)

def create_waveforms(image, name=""):
    '''
    Input: Image
    Output: Array of cnn staff waveforms
    '''
    s = Score(image, name)
    # return s._create_vertical_input()
    return s._create_cnn_staff_waveforms()
    # s._create_bar_waveforms()
    # return s._bar_waveform

def create_bar_waveforms(image, name=""):
    '''
    Input: Image
    Output: Array of cnn staff waveforms
    '''
    s = Score(image, name)
    return s._create_cnn_bars_waveforms()


def test_bar_waveforms(dataset='mini_dataset', output_dir='./test_staves/'):
    '''
    Test the staff splitting by rendering where the score would be split for
    each file.
    '''
    ret_sum = 0
    ret_counter = 0
    temp_toggle = False
    for i, (label, image_file) in enumerate(data.index_images(dataset=dataset)):
        image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
        name = path.split(label)[-1]
        print('processing image {0} with name {1}'.format(i, name))
        # add 'i' to disambiguate pieces
        s = Score(image, output_dir + name + str(i))
        s._create_bar_waveforms()
        if not temp_toggle:
            print(s._bar_waveform[0])
            plt.scatter(np.arange(s._bar_waveform[0].size), s._bar_waveform[0])
            plt.show()
            temp_toggle = True
        LC = [ len(x)  for x in s._bar_waveform]
        ret_sum += sum(LC)
        ret_counter += len(LC)

    print(ret_sum/ret_counter)

def test_pretty_print(dataset='mini_dataset', output_dir='/home/ckurashige/voice_lines/'):
    '''
    Test the staff splitting by rendering where the score would be split for
    each file.
    '''
    for i, (label, image_file) in enumerate(data.index_images(dataset=dataset)):
        image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
        name = path.split(label)[-1]
        print('processing image {0} with name {1}'.format(i, name))
        # add 'i' to disambiguate pieces
        s = Score(image, output_dir + name + str(i))
        s._generate_pretty_image()

def test_bar_print(dataset='mini_dataset', output_dir='/home/ckurashige/bars_using_staves/', toggle="staves", stuff="12"):
    '''
    Test the staff splitting by rendering where the score would be split for
    each file.
    '''
    for i, (label, image_file) in enumerate(zip(data.database_labels, data.database_paths)):
        image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
        name = path.split(label)[-1]
        print('processing image {0} with name {1}'.format(i, name))
        # add 'i' to disambiguate pieces
        s = Score(image, output_dir + name + str(i))
        s._print_with_bars(toggle=toggle, stuff=stuff)



def cleanup_bars(bars, width):
    """Cleans up a set of bars in staves globally"""
    if len(bars) > 1:
        l_diffs = []
        for i in range(len(bars) - 1):
            l_diffs.append(abs(bars[i][0] - bars[i+1][0]))
        if min(l_diffs) < width:
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

            return cleanup_bars(new_bars, width)
        else:
            return bars
    else:
        return bars



def linear_cleanup_bars(bars, width):
    """Cleans up a set of bars in staves after overdetection linearly"""
    if len(bars) <= 1:
        return bars
    elif len(bars) < 4:
        l_diffs = []
        for i in range(len(bars) - 1):
            l_diffs.append(abs(bars[i][0] - bars[i+1][0]))
        if l_diffs[0] < width:
            return linear_cleanup_bars(bars[1:], width)
        else:
            return [bars[0]] + linear_cleanup_bars(bars[1:], width)
    else:
        l_diffs = []
        for i in range(3):
            l_diffs.append(abs(bars[i][0] - bars[i+1][0]))

        if l_diffs[0] < width:
            new_bars = [bars[0]] + bars[2:]
            return linear_cleanup_bars(new_bars, width)
        elif l_diffs[1] < width:
            if l_diffs[0] < l_diffs[2]:
                new_bars = [bars[0]] + bars[2:]
            else:
                new_bars = bars[0:2] + bars[3:]
            return linear_cleanup_bars(new_bars, width)
        else:
            return [bars[0]] + linear_cleanup_bars(bars[1:], width)

def cnn_bar_img(dataset='mini_dataset', output_dir='/home/ckurashige/bars_for_cnn/', length = 30):
    '''
    Generates bar images images for the cnn
    '''
    for i, (label, image_file) in enumerate(data.index_images(dataset=dataset)):
        if i > 100:
            return
        image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
        name = path.split(label)[-1]
        print('processing image {0} with name {1}'.format(i, name))
        # add 'i' to disambiguate pieces
        s = Score(image, name)
        s._find_bars_using_peaks(clean_up=False, thresholder=False)
        img_color = cv.cvtColor(s._score ,cv.COLOR_GRAY2RGB)
        print("Staves Length: {}".format(len(s._staves_start_end)))
        print("Bars Length: {}".format(len(s._bars_start_end)))
        # for i, start, end in s._bars_start_end:
        #     cv.line(img_color, (i, start), (i, end), (0,0,255), 2)
        for ind, (bar_index, bar_start, bar_end) in enumerate(s._bars_start_end):
            location = output_dir+'image_{0}_{1}_bar_{2}.png'.format(i, s._name, ind)

            cropped_bar = s._score[bar_start:bar_end, bar_index-length:bar_index+length]
            if cropped_bar.size == 0:
                print("Empty bar generated for {0} at bar {1}".format(s._name, ind))
            else:
                print("Writing image to: {}".format(location))
                cv.imwrite(location, cropped_bar)

def cnn_txt_staves(dataset='mini_dataset', output_dir='/home/ckurashige/bar_label_data/'):
    """CNN pretraining thing"""
    for i, (label, image_file) in enumerate(data.index_images(dataset=dataset)):
        image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
        name = path.split(label)[-1]
        print('processing image {0} with name {1}'.format(i, name))
        # add 'i' to disambiguate pieces
        s = Score(image, name)
        s._find_bars_using_peaks(clean_up=False, thresholder=False)
        for ind, (stave, bars) in enumerate(zip(s._staves,s._bars)):
            cv.imwrite(output_dir+"image_{0}_{1}_stave_{2}.png".format(i, name, ind),stave)
            with open(output_dir+"image_{0}_{1}_stave_{2}.txt".format(i, name, ind), 'w') as f:
                for bar in bars:
                    f.write("{}\n".format(bar))

def get_ten_thousand_bars(dataset="mini_dataset",output_dir='/home/ckurashige/ten_thousand_bars/'):
    for i, (label, image_file) in enumerate(data.index_images(dataset=dataset)):
        image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
        name = path.split(label)[-1]
        print('processing image {0} with name {1}'.format(i, name))
        # add 'i' to disambiguate pieces
        s = Score(image, name)
        s._find_bars_using_peaks()

        for x in range(len(s._bars_start_end) - 1):
            cropped_bar = s._score[s._bars_start_end[x][1]:s._bars_start_end[x][2], s._bars_start_end[x][0]:s._bars_start_end[x+1][0]]
            if cropped_bar.size != 0:
                cv.imwrite(output_dir+"image_{0}_{1}_bar_{2}.png".format(i, s._name, x) ,cropped_bar)


def cnn_bar_size_printout(dataset="piano_dataset",output_dir='/home/ckurashige/yadayada/'):
    for i, (label, image_file) in enumerate(data.index_images(dataset=dataset)):
        if i < 100:
            image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
            name = path.split(label)[-1]
            print('processing image {0} with name {1}'.format(i, name))
            # add 'i' to disambiguate pieces
            s = Score(image, output_dir + name + str(i))
            s._create_cnn_bars_waveforms()
        else:
            break

def tsai_bar_printout(output_dir='/home/ckurashige/tsai_bars/'):
    for i, (label, image_file) in enumerate(zip(data.database_labels, data.database_paths)):
        image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
        name = path.split(label)[-1]
        print('processing image {0} with name {1}'.format(i, name))
        s = Score(image, label)
        s._find_staves()
        for j, (start, end) in enumerate(s._staves_start_end):
            one_staff = list(cut_array(s._score, [(start, end)]))[0]
            tb.extractMeasures(one_staff, path=output_dir+"image_{0}_{1}_{2}.png".format(i, name, j), visualize=True)


def paper_bar_printout():
    for i, (label, image_file) in enumerate(zip(data.database_labels, data.database_paths)):
        if i > 30:
            break
        else:
            output_dir = '/home/ckurashige/paper_bars/'
            image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
            name = path.split(label)[-1]
            print('processing image {0} with name {1}'.format(i, name))
            # add 'i' to disambiguate pieces
            s = Score(image, output_dir + name + str(i))
            s._print_with_bars(toggle='peaks', stuff="0", name= s._name+"_piece")
            s._print_with_bars(toggle='peaks', stuff="1", name=s._name+"_staves")
            s._print_with_bars(toggle='peaks', stuff="12", name=s._name+"_bars")
            s._print_with_bars(toggle='peaks', stuff="0", name= s._name+"_piece_vert", vert=True)
            s._print_with_bars(toggle='peaks', stuff="1", name=s._name+"_staves_vert", vert=True)
            s._print_with_bars(toggle='peaks', stuff="12", name=s._name+"_bars_vert",  vert=True)


if __name__ == '__main__':

    # get_ten_thousand_bars()
    # cnn_bar_img(length=50)
    # cnn_txt_staves()
    # test_bar_print(dataset="piano_dataset",output_dir='/home/ckurashige/bars_using_avg_min/', toggle='peaks')
    # test_bar_print(output_dir='/home/ckurashige/bars_using_intersections/', toggle='intersect')
    # cnn_bar_size_printout()
    # tsai_bar_printout()
    # test_bar_print(toggle='tb', output_dir='/home/ckurashige/tsai_bars_hybrid/')
    paper_bar_printout()
