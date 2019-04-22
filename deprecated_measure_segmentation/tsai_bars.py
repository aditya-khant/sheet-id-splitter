import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import glob
import random
from shutil import copyfile
import os.path
import os
import subprocess
from pdf2image import convert_from_path
from skimage import filters, measure
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import matplotlib.patches as mpatches
from scipy.signal import convolve2d, medfilt

from benchmarks import call_benchmark


def getNormImage(img):
    X = 1 - np.array(img) / 255.0 # scale and invert
    return X

def downsample_image(image, by_rate= True, rate=0.3, by_size=False, width=1024, height=1024):
    '''
    Downsamples 'image' by a ratio 'rate' or by a mentioned size ('width' and 'height')
    '''
    if by_rate:
        new_shape = (int(image.shape[0] * rate), int(image.shape[1] * rate))
    if by_size:
        new_shape = (width, height)
    return cv2.resize(image, new_shape)

def morphFilterLinesVert(arr, kernel_length = 51, kernel_width = 5):
    lines = cv2.dilate(arr, np.ones((3,1)), iterations = 1) # fill in small gaps
    lines = cv2.erode(lines, np.ones((3,1)), iterations = 1) # undo expansion
    vkernel = np.ones((kernel_length, 1), np.uint8)
    hkernel = np.ones((1, kernel_width), np.uint8)
    lines = cv2.erode(lines, vkernel, iterations = 1)
    lines = cv2.dilate(lines, vkernel, iterations = 1)
    lines = cv2.dilate(lines, hkernel, iterations = 1)
    return lines

def binarize_std(img, numStds = 2):
    # set default to 2 stdevs so that we leave a lot of false positives
    mean = np.mean(img.ravel())
    std = np.std(img.ravel())
    thresh = mean + numStds * std
    binarized = img > thresh
    return binarized, thresh

def createCombFilters(stavelen_ll, stavelen_ul):
    # create comb filters of different lengths
    # e.g. if length is 45, then spikes at indices 0, 11, 22, 33, 44
    # e.g. if length is 44, then spikes at 0 [1.0], 10 [.25], 11 [.75], 21 [.5], 22 [.5], 32 [.75], 33 [.25], 43 [1.0]
    stavelens = np.arange(stavelen_ll, stavelen_ul)
    combfilts = np.zeros((len(stavelens), stavelens[-1], 1))
    for i, stavelen in enumerate(stavelens):
        for j in range(5):
            idx = j * (stavelen-1) / 4.0
            idx_below = int(idx)
            idx_above = idx_below + 1
            remainder = idx - idx_below
            combfilts[i, idx_below, 0] = 1 - remainder
            if idx_above < combfilts.shape[1]:
                combfilts[i, idx_above, 0] = remainder
    return combfilts, stavelens

def computeStaveFeatureMap(img, combfilts):

    # cut hlines into left and right halves, compute median of row pixel values
    lhcols = img.shape[1] // 2
    img_left = img[:,0:lhcols] # separate into two halfs to handle lines being slightly off horizontal
    img_right = img[:,lhcols:]
    rmeds_left = np.median(img_left, axis=1, keepdims = True)
    rmeds_right = np.median(img_right, axis=1, keepdims = True)
    rmeds = np.hstack((rmeds_left, rmeds_right))

    # apply comb filters
    featmap = []
    for i in range(combfilts.shape[0]):
        m = convolve2d(rmeds, np.fliplr(np.flipud(combfilts[i])), mode = 'valid')
        featmap.append(m)
    featmap = np.array(featmap)

    return featmap, rmeds

def estimateStaveHeight(featmap, stavelens, topN = 20):
    scores = []
    for i in range(featmap.shape[0]):
        featValsSorted = sorted(featmap[i].ravel(), reverse = True)
        score = np.sum(featValsSorted[0:topN])
        scores.append(score)
    argmax = np.argmax(scores)
    staveH = stavelens[argmax]
    return staveH, argmax, scores

def getEstStaffLineLocs(feat, barlines, staveHeight, img, delta = 1):
    preds = []
    for bbox in barlines:
        rtop = bbox[0]
        rbot = bbox[2]
        c = int(np.mean((bbox[1], bbox[3])))

        pair = []
        for r in [rtop, rbot]:
            if r == rtop:
                rupper = min(r + delta * staveHeight + 1, img.shape[0])
                rlower = max(r - delta * staveHeight, 0)
            else:
                rupper = min(r + 1, img.shape[0])
                rlower = max(r - 2 * delta * staveHeight, 0)
            featColIdx = 0 if c < img.shape[1] // 2 else 1
            reg = feat[rlower:rupper, featColIdx]
            roffset = reg.argmax()
            rstart = rlower + roffset
            rend = rstart + staveHeight - 1
            pair.append((rstart, rend, c, r))
        preds.append(pair)

    return preds

def filterCandidates(candidates, estStaffLineLocs, tol = 5, mindist = 3):
    result = []
    for i, bbox in enumerate(candidates):
        topStave, botStave = estStaffLineLocs[i] # pair of (rstart, rend , c, r)
        isValid = isBarline(bbox[0], bbox[2], topStave, botStave, tol, mindist)
        if isValid:
            result.append(bbox)
    return result

def calcVertOverlap(bbox1, bbox2):
    # each bbox is a tuple (row_min, col_min, row_max, col_max)
    rstart = max(bbox1[0], bbox2[0])
    rend = min(bbox1[2], bbox2[2])
    overlap = np.clip(rend - rstart, 0, None)
    return overlap

def isBarline(bar_rtop, bar_rbot, top_stave, bot_stave, tol, mindist):
    topClose = np.abs(bar_rtop - top_stave[0]) < tol
    bottomClose = np.abs(bar_rbot - bot_stave[1]) < tol
    noOverlap = top_stave[1] < bot_stave[0]
    staffH = top_stave[1] - top_stave[0]
    farEnoughApart = (bar_rbot - bar_rtop) > mindist * staffH
    isBarline = topClose and bottomClose and noOverlap and farEnoughApart
    return isBarline

def clusterBarlines(barlines):
    clusters = -1*np.ones(len(barlines), dtype=np.int8)
    clusterIndex = 0
    for i, bbox in enumerate(barlines):
        if clusters[i] == -1: # has not been assigned a cluster yet
            anchor = bbox
            for j in range(i,len(barlines)):
                overlap = calcVertOverlap(anchor, barlines[j])
                if overlap > 0:
                    clusters[j] = clusterIndex
            clusterIndex += 1

    # return barlines by cluster & sorted left to right
    result = []
    for i in range(clusterIndex):
        curCluster = []
        for j, bbox in enumerate(barlines):
            if clusters[j] == i:
                curCluster.append(bbox)
        curCluster.sort(key = lambda x: x[1]) # sort by col, increasing
        result.append(curCluster)

    return result

def getMeasureBB(clusters):
    measures = [] # bbox for each measure
    for cl in clusters:
        for i, tup in enumerate(cl): # each tup is (row_min, col_min, row_max, col_max)
            if i < len(cl) - 1:

                # coord left side
                row_topL = tup[0]
                row_botL = tup[2]
                col_min = int(np.mean((tup[1], tup[3])))

                # coord right side
                next_tup = cl[i+1]
                row_topR = next_tup[0]
                row_botR = next_tup[2]
                col_max = int(np.mean((next_tup[1], next_tup[3]))) + 1

                # measure bbox
                row_min = min(row_topL, row_topR)
                row_max = max(row_botL, row_botR)
                measures.append((row_min, col_min, row_max, col_max))
    return measures

def visualizeMeasures(measures, img, path):
    plt.figure(figsize = (20,20))
    plt.imshow(img, cmap='gray')
    ax = plt.gca()
    for bbox in measures:
        row_min, col_min, row_max, col_max = bbox
        rect = mpatches.Rectangle((col_min, row_min), col_max - col_min, row_max - row_min, linewidth=2 , edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.savefig(path)

def buffer_measures(measures, buffer_pct):
    new_measures=[]
    for bbox in measures:
        row_min, col_min, row_max, col_max = bbox
        buffer = int((row_max-row_min)*buffer_pct)
        row_min = row_min - buffer
        row_max = row_max + buffer
        new_measures += [(row_min, col_min, row_max, col_max)]
    return new_measures

def extractMeasures(img, path = None, visualize = False):
    '''
    Input: gray png image of score
    Output: The bar waveforms after they have been processed by the benchmark
            CNNs.
    '''
    ####### parameters #######
    resizeW = 1000
    resizeH = 200
    morphFilterLength = 51
    morphFilterWidth = 5
    binarizeThreshStd = 2
    staveHeightMin = 15
    staveHeightMax = 30
    staveHeightTopN = 20
    estStaffLineDelta = 1
    barlineTol = 1
    minBarlineLen = 3
    buffer_pct = 0.4
    #for reampling for the CNN
    bar_height = 128
    bar_width = 128
    ##########################

    # prep image
    img = cv2.resize(img, (resizeW, resizeH))
    X = getNormImage(img)

    # get barline candidates
    vlines = morphFilterLinesVert(X, morphFilterLength, morphFilterWidth)
    vlines_bin, binarize_thresh = binarize_std(vlines, binarizeThreshStd) # threshold in units of std above mean
    vlabels = measure.label(vlines_bin)
    candidates = [reg.bbox for reg in regionprops(vlabels)]

    # staff line detection
    combfilts, stavelens = createCombFilters(staveHeightMin, staveHeightMax)
    featmap, rmeds = computeStaveFeatureMap(X, combfilts)
    staveHeight, staveLenIdx, staveLenScores = estimateStaveHeight(featmap, stavelens, staveHeightTopN)
    staveFeats = featmap[staveLenIdx]
    estStaffLineLocs = getEstStaffLineLocs(staveFeats, candidates, staveHeight, X, estStaffLineDelta)

    # filter & cluster candidates
    tol = barlineTol * staveHeight // 4 + 1 # 1 barlineTol = distance between two adjacent staff lines
    barlines = filterCandidates(candidates, estStaffLineLocs, tol, minBarlineLen) # minBarLineLen in units of staveHeight
    bar_clusters = clusterBarlines(barlines)
    measures = getMeasureBB(bar_clusters)
    measures = buffer_measures(measures, buffer_pct)
    # visualize result
    if visualize:
        if path is not None:
            return visualizeMeasures(measures, img, path)

    #split the actual image into the bars

    img_list = []

    for bbox in measures:
        bar = np.array(img)[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if bar.size != 0:
            img_list.append(bar)

    images = [downsample_image(cv2.cvtColor(bar,cv2.COLOR_GRAY2RGB), by_rate=False, by_size=True, height=bar_height, width=bar_width)
                  for bar in img_list]

    if not images:
        return images
    else:
        return call_benchmark(images=images)


# Modified version of Prof Tsai's hybrid splitter.
def extractMeasuresHybrid(img):
    '''
    Input: gray png image of score
    Output: The bar waveforms after they have been processed by the benchmark
            CNNs.
    '''
    ####### parameters #######
    resizeW = 1000
    resizeH = 1000
    morphFilterLength = 51
    morphFilterWidth = 5
    binarizeThreshStd = 2
    staveHeightMin = 15
    staveHeightMax = 30
    staveHeightTopN = 20
    estStaffLineDelta = 1
    barlineTol = 1
    minBarlineLen = 3
    buffer_pct = 0.4
    #for reampling for the CNN
    bar_height = 128
    bar_width = 128
    ##########################

    # prep image
    img = cv2.resize(img, (resizeW, resizeH))
    X = getNormImage(img)

    # get barline candidates
    vlines = morphFilterLinesVert(X, morphFilterLength, morphFilterWidth)
    vlines_bin, binarize_thresh = binarize_std(vlines, binarizeThreshStd) # threshold in units of std above mean
    vlabels = measure.label(vlines_bin)
    candidates = [reg.bbox for reg in regionprops(vlabels)]

    # staff line detection
    combfilts, stavelens = createCombFilters(staveHeightMin, staveHeightMax)
    featmap, rmeds = computeStaveFeatureMap(X, combfilts)
    staveHeight, staveLenIdx, staveLenScores = estimateStaveHeight(featmap, stavelens, staveHeightTopN)
    staveFeats = featmap[staveLenIdx]
    estStaffLineLocs = getEstStaffLineLocs(staveFeats, candidates, staveHeight, X, estStaffLineDelta)

    # filter & cluster candidates
    tol = barlineTol * staveHeight // 4 + 1 # 1 barlineTol = distance between two adjacent staff lines
    barlines = filterCandidates(candidates, estStaffLineLocs, tol, minBarlineLen) # minBarLineLen in units of staveHeight
    bar_clusters = clusterBarlines(barlines)
    measures = getMeasureBB(bar_clusters)
    # measures = buffer_measures(measures, buffer_pct)

    return measures

