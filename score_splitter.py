# Based in part off of: https://docs.opencv.org/4.0.0/dd/dd7/tutorial_morph_lines_detection.html

import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2 as cv
import os.path as path

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
        # TODO: why is 30 here?
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
            # non_zeros = vert_sum_verts[np.where(vert_sum_verts > 0)]
            # top 10 values and looking for x less than
            # threshold = sorted(non_zeros)[-10:][0]
            bar_split_indices = list(split_indices(vert_sum_verts, lambda x: x > 0))
            self._bars_start_end.append(bar_split_indices)
            for start, end in bar_split_indices:
                self._bars.append(staff[start:end])

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
        return call_benchmark(images=[cv.cvtColor(staff,cv.COLOR_GRAY2RGB) for staff in self._staves])

    def _generate_pretty_image(self):
        '''
        Generates bars and staves on an image
        '''
        if self._bars is None:
            self._find_bars()
        img_color = cv.cvtColor(self._score ,cv.COLOR_GRAY2RGB)
        for (staff_start, staff_end), bar_lines in zip(self._staves_start_end, self._bars_start_end):
            cv.line(img_color, (0, staff_start), (self._score.shape[1], staff_start), (255,0,0), 5 )
            cv.line(img_color, (0, staff_end), (self._score.shape[1], staff_end), (255,0,0), 5 )
            for (bar_start, bar_end) in bar_lines:
                cv.line(img_color, (bar_start, staff_start), (bar_start, staff_end), (0,0,255), 5 )
                cv.line(img_color, (bar_end, staff_start), (bar_end, staff_end), (0,0,255), 5 )
        cv.imwrite('{}.png'.format(self._name), img_color)



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

def create_waveforms(image, name=""):
    '''
    Input: Image
    Output: Array of cnn staff waveforms
    '''
    s = Score(image, name)
    return s._create_cnn_staff_waveforms()
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

def test_pretty_print(dataset='mini_dataset', output_dir='/home/ckurashige/pretty_output/'):
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
    test_staves()
    # test_bar_waveforms()
    # test_pretty_print()


# TODO: clean up below

# # In[2]:


# img_path = "./04352.png" # Scanned
# # img_path = "./37048.png" # Perfect


# # In[3]:


# src = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
# print(src.shape)
# # Show source image
# plt.figure(figsize=(10,13))
# plt.imshow(src, aspect="auto", cmap = "gray")


# # In[4]:


# # Convert image to binary
# gray = cv.bitwise_not(src)
# bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,                             cv.THRESH_BINARY, 15, -2)
# plt.figure(figsize=(10,13))
# plt.imshow(bw, aspect="auto", cmap = "gray")


# # In[5]:


# # ----VERTICAL LINE DETECTION----
# vertical = np.copy(bw)
# # Specify size on vertical axis
# rows = vertical.shape[0]
# verticalsize = rows // 30
# # Create structure element for extracting vertical lines through morphology operations
# verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
# # Apply morphology operations
# vertical = cv.erode(vertical, verticalStructure)
# vertical = cv.dilate(vertical, verticalStructure)
# # Show extracted vertical lines
# plt.figure(figsize=(10,13))
# plt.imshow(vertical, aspect="auto")


# # In[6]:


# # ---- HORIZONTAL LINE DETECTION ----
# horizontal = np.copy(bw)

# # [init]
# # [horiz]
# # Specify size on horizontal axis
# cols = horizontal.shape[1]
# horizontal_size = cols // 30
# # Create structure element for extracting horizontal lines through morphology operations
# horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
# # Apply morphology operations
# horizontal = cv.erode(horizontal, horizontalStructure)
# horizontal = cv.dilate(horizontal, horizontalStructure)
# plt.figure(figsize=(10,13))
# plt.imshow(horizontal, aspect="auto")


# # In[7]:


# cv.imwrite("vert.png", vertical)
# cv.imwrite("horiz.png", horizontal)


# # # Line Splitting

# # In[8]:


# vert_norm = vertical // vertical.max()
# horiz_sum_vert = vert_norm.sum(axis=1)
# plt.scatter(np.arange(horiz_sum_vert.size), horiz_sum_vert)


# # In[9]:


# def get_split_indices(array, comparator=(lambda x: x == 0)):
#     '''Input: 1-D array of indicies of zeros of horizontal summation
#     Output: Generator of indicies to split images by discontinuities in zeros'''
#     indices = np.where(comparator(array))[0]
#     # we dont want to add 1 to last element
#     for i in range(indices.size - 1):
#         if indices[i+1] - indices[i] != 1:
#             yield (indices[i], indices[i+1])


# # In[10]:


# line_pairs = list(get_split_indices(horiz_sum_vert))
# line_pairs


# # In[11]:


# def get_split_indices_average(array):
#     '''Input: 1-D array of indicies of zeros of horizontal summation
#     Output: Iterator of indicies to split image at by average of zeros'''
#     line_pair = list(get_split_indices(array))
#     line_pair = [(0, 0)] + line_pair + [(array.size, array.size)]
#     for i in range(len(line_pair) - 2):
#         a = line_pair[i][1]
#         b = line_pair[i+1][0]
#         a1 = line_pair[i+1][1]
#         b1 = line_pair[i+2][0]
#         yield ( a + ((b-a)//2) , a1 + ((b1-a1)//2))


# # In[12]:


# line_cuts = list(get_split_indices_average(horiz_sum_vert))
# line_cuts


# # In[13]:


# plt.figure(figsize=(10,13))
# plt.imshow(src, aspect="auto", cmap = "gray")
# for start, end in line_pairs:
#     plt.axhline(y=start, color = 'r')
#     plt.axhline(y = end, color = 'b')
# for cut in line_cuts:
#     plt.axhline(y=cut[0], color = 'g')
# # Note: This does not show the last green cut.


# # In[14]:


# def cut_array(array, positions, direction="H"):
#     '''Input: array: image array, positions: array of start end tuples
#        Output: array of image arrays cut by positions'''
#     for start , end in positions:
#         if (direction == "H"):
#             yield array[start:end, :]
#         else:
#             yield array[:, start:end]


# # In[15]:


# array_of_cuts = list(cut_array(src, line_cuts))
# for i, cut in enumerate(cut_array(src, line_cuts)):
#     plt.figure(figsize = (10,2))
#     plt.imshow(cut, cmap="gray")
#     cv.imwrite("./cuts/"+img_path +"_cut_"+str(i)+".png", cut)


# # # Vertical Split

# # In[16]:


# plt.figure(figsize = (20,4))
# one_cut = array_of_cuts[0]
# plt.imshow(one_cut, cmap="gray")


# # In[17]:


# array_of_vert = list(cut_array(vert_norm, line_cuts))
# one_vert = array_of_vert[0]
# vertical_sum_vert = one_vert.sum(axis=0)
# plt.scatter(np.arange(vertical_sum_vert.size), vertical_sum_vert)


# # In[18]:


# bar_lines = list(get_split_indices(vertical_sum_vert, lambda x: x > 0))
# bar_lines


# # In[19]:


# plt.figure(figsize = (20,4))
# plt.imshow(array_of_cuts[0], cmap="gray")
# for start, end in bar_lines:
#     plt.axvline(x = start, color = 'r')
#     plt.axvline(x = end, color = 'b')


# # In[20]:


# array_of_bars = list(cut_array(one_cut, bar_lines, "V") )
# for i, cut in enumerate(array_of_bars):
#     plt.figure(figsize = (4,2))
#     plt.imshow(cut, cmap="gray")
#     cv.imwrite("./cuts/"+img_path +"_bar_cut_"+str(i)+".png", cut)


# # In[21]:


# for bar in array_of_bars:
#     bar_sum = bar.sum(axis=0)
#     plt.figure()
#     plt.scatter(np.arange(bar_sum.size), bar_sum)


# # In[22]:


# # MIGHT BE IMPORTANT?
# # Extract edges and smooth image according to the logic
# # 1. extract edges
# # 2. dilate(edges)
# # 3. src.copyTo(smooth)
# # 4. blur smooth img
# # 5. smooth.copyTo(src, edges)
# # '''
# # # Step 1
# # edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
# #                             cv.THRESH_BINARY, 3, -2)
# # show_wait_destroy("edges", edges)
# # # Step 2
# # kernel = np.ones((2, 2), np.uint8)
# # edges = cv.dilate(edges, kernel)
# # show_wait_destroy("dilate", edges)
# # # Step 3
# # smooth = np.copy(vertical)
# # # Step 4
# # smooth = cv.blur(smooth, (2, 2))
# # # Step 5
# # (rows, cols) = np.where(edges != 0)
# # vertical[rows, cols] = smooth[rows, cols]
# # # Show final result
# # show_wait_destroy("smooth - final", vertical)
# # # [smooth]

