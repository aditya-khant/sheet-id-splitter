# Based in part off of: https://docs.opencv.org/4.0.0/dd/dd7/tutorial_morph_lines_detection.html

import os.path as path
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

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
        print(score.shape)
        gray = cv.bitwise_not(score)
        bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
        self._score_gray = gray
        self._score     = score
        self._score_bw  = bw
        self._name      = name
        self._verticals = None
        self._staves    = None
        self._staves_verticals = None
        self._staves_start_end = []
        self._bars = None
        self._bars_start_end = []
        self._bar_waveform = None
        self._voice_lines_by_staff = None
        self._voice_lines_by_page = None
        # TODO: eventually structure as 3-dimensional array of images
        # dimension 0: staves
        # dimension 1: bars
        # dimension 2: voices

    def _find_vertical_lines(self):
        '''
        generates an image of the same shape as 'self._score_bw' where only the vertical lines remain.
        '''
        self._verticals = np.copy(self._score_bw)
        # Specify size on vertical axis
        rows, _ = self._verticals.shape
        # TODO: Smaller the magic number bigger the filtered out lines are
        # TODO: figure out how to find the optimal value
        vertical_size = rows // 30
        # Create structure element for extracting vertical lines through morphology operations
        vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
        # Apply morphology operations
        self._verticals = cv.erode(self._verticals, vertical_structure)
        self._verticals = cv.dilate(self._verticals, vertical_structure)

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
        verts_norm = self._verticals // self._verticals.max()
        horiz_sum_verts = verts_norm.sum(axis=1)
        # tuples of (start,end) denoting where to split the image at
        staff_split_indices = None
        if split_type == 'average':
            staff_split_indices = list(split_indices_average(horiz_sum_verts))
        elif split_type == 'strict':
            staff_split_indices = list(split_indices(horiz_sum_verts))
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
        min_width = 300
        min_height = 390
        image = downsample_image(cv.cvtColor(self._verticals,cv.COLOR_GRAY2RGB), by_rate=False, by_size=True, width=min_width, height=min_height)
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
        min_width = 500
        min_height = 200
        images = [downsample_image(cv.cvtColor(staff,cv.COLOR_GRAY2RGB), by_rate=False, by_size=True, width=min_width, height=min_height)
                  for staff in self._staves]
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


    def _generate_pretty_image(self, bars=True, staves = True, voice=True, voice_by_page = False):
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
                for (bar_start, bar_end) in bar_lines:
                    cv.line(img_color, (bar_start, staff_start), (bar_start, staff_end), (0,0,255), 5 )
                    cv.line(img_color, (bar_end, staff_start), (bar_end, staff_end), (0,0,255), 5 )
            if voice:
                for line_val in voice_lines:
                    cv.line(img_color, (0, staff_start + line_val), (self._score.shape[1], staff_start + line_val), (0,255,0), 5 )
            
        cv.imwrite('{}.png'.format(self._name), img_color)



def downsample_image(image, by_rate= True, rate=0.3, by_size=False, width = 500, height = 300 ):
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

def create_waveforms(image, name="", down_sample_rate=0.5):
    '''
    Input: Image
    Output: Array of cnn staff waveforms
    '''
    s = Score(image, name)
    return s._create_vertical_input()
    # return s._create_cnn_staff_waveforms()
    # s._create_bar_waveforms()
    # return s._bar_waveform

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

if __name__ == '__main__':
    # test_staves()
    # test_bar_waveforms()
    test_pretty_print()


#### Deprecated Code ####
# TODO: Integrate the code into existing code
def fitStaffLines(scores, height):
    N = len(scores)
    idx_RH, idx_LH, best_score = 0, 0, -1
    idxs_sorted = np.argsort(scores)[::-1]
    min_separation = int(height * 1.66)
    idx1 = idxs_sorted[0]
    for j in range(1, N):
        idx2 = idxs_sorted[j]
        curScore = scores[idx1] + scores[idx2]
        sep = np.abs(idx1 - idx2)
        if sep > min_separation and curScore > best_score:
            best_score = curScore
            idx_RH = min(idx1, idx2)
            idx_LH = max(idx1, idx2)
            break
    return best_score, idx_RH, idx_LH

def locateStaffLines(s, min_height = 60, max_height = 120, plot = True):
    rsums = np.sum(s, axis=1)
    bestScore = 0
    lineLocs = np.zeros(10)
    for h in range(min_height,max_height+1):
        idxs = h * np.arange(5) / 4.0
        idxs = idxs.round().astype('int')
        filt = np.zeros(h+1)
        filt[idxs] = 1 # create comb filter
        scores = np.convolve(rsums, filt, 'valid')
        curScore, idx_RH, idx_LH = fitStaffLines(scores, h)
        if curScore > bestScore:
            bestScore = curScore
            lineLocs[0:5] = idxs + idx_RH
            lineLocs[5:] = idxs + idx_LH
    
    if plot:
        plt.plot(rsums)
        for i in range(len(lineLocs)):
            plt.axvline(x=lineLocs[i], color='r', linewidth=1)
        plt.show()
        
    return lineLocs
