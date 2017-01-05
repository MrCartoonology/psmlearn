from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import math
import numpy as np

__TRACE_CACHE=set()
__DEBUG_CACHE=set()
__CACHE_SIZE=10000

def _new_to_cache(msg, CACHE):
    global __CACHE_SIZE
    if msg in CACHE:
        return False
    if len(CACHE)>=__CACHE_SIZE:
        CACHE.pop()
    CACHE.add(msg)
    return True

def logTrace(hdr, msg, checkcache=True, flag=True):
    if not flag: return
    msg = "TRACE %s: %s" % (hdr, msg)
    if (not checkcache) or  _new_to_cache(msg, __TRACE_CACHE):
        print(msg)
        sys.stdout.flush()

def logDebug(hdr, msg, checkcache=True, flag=True):
    if not flag: return
    msg = "DBG %s: %s" % (hdr, msg)
    if (not checkcache) or _new_to_cache(msg, __DEBUG_CACHE):
        print(msg)
        sys.stdout.flush()

def convert_to_one_hot(labels, numLabels):
    '''converts a 1D integer vector to one hot labeleling.
    '''
    if isinstance(labels, list):
        labels = np.array(labels)
    labelsOneHot = np.zeros((len(labels), numLabels), dtype=np.int32)
    for label in range(numLabels):
        rowsToSet = np.where(labels==label)[0]
        labelsOneHot[rowsToSet,label] = 1
    assert np.sum(labelsOneHot) == len(labels), "labels must have entries not in [0,%d)" % numLabels
    return labelsOneHot

def get_confusion_matrix_one_hot(model_results, truth):
    '''model_results and truth should be for one-hot format, i.e, have >= 2 columns,
    where truth is 0/1, and max along each row of model_results is model result
    '''
    assert model_results.shape == truth.shape
    assert np.sum(truth)==truth.shape[0]
    num_outputs = truth.shape[1]
    confusion_matrix = np.zeros((num_outputs, num_outputs), dtype=np.int32)
    predictions = np.argmax(model_results,axis=1)
    assert len(predictions)==truth.shape[0]

    for actual_class in range(num_outputs):
        idx_examples_this_class = truth[:,actual_class]==1
        prediction_for_this_class = predictions[idx_examples_this_class]
        for predicted_class in range(num_outputs):
            count = np.sum(prediction_for_this_class==predicted_class)
            confusion_matrix[actual_class, predicted_class] = count
    assert np.sum(confusion_matrix)==len(truth), "np.sum(confusion_matrix)=%d !=len(truth)=%d, cmat=%r truth=%r" % \
        (np.sum(confusion_matrix), len(truth), confusion_matrix, truth)
    return confusion_matrix

def cmat2str(confusion_matrix, fmtLen=None):
    accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    if fmtLen is None:
        fmtLen = int(math.floor(math.log(np.max(confusion_matrix),10)))+1
    fmtstr = '%' + str(fmtLen) + 'd'
    cmat_rows = []
    for row in range(confusion_matrix.shape[0]):
        cmat_rows.append(' '.join(map(lambda x: fmtstr % x, confusion_matrix[row,:])))
    return accuracy, cmat_rows

def get_best_correct_one_hot(scores, truth, label):
    '''looks at scores that are correct for label.
    Returns best score, and row for that score.
    Returns None, None if scores never predict label correctly.
    '''
    predict = np.argmax(scores, axis=1)
    truth = np.argmax(truth, axis=1)
    correct = predict == truth
    correct_and_label = np.logical_and(correct, truth == label)
    orig_rows = np.arange(scores.shape[0])
    orig_rows = orig_rows[correct_and_label]
    if len(orig_rows)==0:
        return None, None
    scores = scores[correct_and_label,label]
    best_score = np.max(scores)
    row = orig_rows[np.argmax(scores)]
    return row, best_score


########### more analytical utility functions ##########

def inplace_log_thresh(img,thresh):
    replace = img >= thresh
    newval = np.log(1.0 + img[replace] - thresh)
    img[replace]=thresh + newval

def log_thresh(img,thresh):
    img = img.astype(np.float32, copy=True)
    inplace_log_thresh(img, thresh)
    return img

def replicate(img, num_channels, dtype):
    assert len(img.shape)==2
    repShape = list(img.shape)+[num_channels]
    rep = np.empty(repShape, dtype)
    for ch in range(num_channels):
        rep[:,:,ch] = img[:]
    return rep

def topn(arr, n):
    '''return topn positions in array (as flat indicies)
    and topn values
    '''
    inds = arr.argsort()
    pos = list(inds[-n:])
    vals = list(arr[pos])
    pos.reverse()
    vals.reverse()
    return pos, vals

def start_signal_window(img, direc, window_len):
    assert direc in ['hproj','vproj']
    if direc == 'vproj':
        proj=np.mean(img, axis=0)
    else:
        proj = np.mean(img, axis=1)
    cum_proj = np.cumsum(proj)
    NN=len(cum_proj)
    if window_len >= NN:
        return 0
    signal_window = cum_proj.copy()
    signal_window[window_len:] -= cum_proj[0:NN-window_len]
    end_max_signal = max(window_len,np.argmax(signal_window))
    start_max_signal = end_max_signal-window_len
    return start_max_signal

def start_signal_window_batch(img_batch, window_len, direc='vproj', channel_ordering='tf'):
    assert len(img_batch.shape) == 4
    assert channel_ordering in ['tf','th']
    batch_size = img_batch.shape[0]
    starts = np.zeros(batch_size, dtype=np.int64)
    for imgIdx in range(batch_size):
        img = img_batch[imgIdx]
        if channel_ordering == 'tf':
            img = np.mean(img, axis=2)
        elif channel_ordering == 'th':
            img = np.mean(img, axis=0)
        starts[imgIdx] = start_signal_window(img, direc, window_len)
    return starts

def extract_signal_window(start, window_len, img, direc='vproj', channel_ordering='tf'):
    assert direc in ['vproj','hproj']
    assert channel_ordering in ['tf','th']
    if channel_ordering == 'tf' and direc == 'vproj':
        return img[:,start:(start+window_len),:]
    if channel_ordering == 'tf' and direc == 'hproj':
        return img[start:(start+window_len),:,:]
    if channel_ordering == 'th' and direc == 'vproj':
        return img[:,:,start:(start+window_len)]
    if channel_ordering == 'th' and direc == 'hproj':
        return img[:,start:(start+window_len),:]
    

def extract_signal_window_batch(starts, img_batch, window_len, direc='vproj', channel_ordering='tf'):
    assert len(img_batch.shape) == 4
    assert direc in ['vproj','hproj']
    assert channel_ordering in ['tf','th']

    if channel_ordering == 'tf':
        num_channels = img_batch.shape[3]
        HH,WW = img_batch.shape[1], img_batch.shape[2]
    else:
        num_channels = img_batch.shape[1]
        HH,WW = img_batch.shape[2], img_batch.shape[3]
    batch_size = img_batch.shape[0]

    if direc == 'vproj':
        new_batch_shape = [batch_size, HH, window_len]
    elif direc == 'hproj':
        new_batch_shape = [batch_size, window_len, WW]

    if channel_ordering == 'tf':
        new_batch_shape.append(num_channels)
    else:
        new_batch_shape.insert(1,num_channels)

    new_batch = np.empty(new_batch_shape, dtype=img_batch.dtype)

    for imgIdx, start in zip(range(batch_size), starts):
        new_batch[imgIdx] = extract_signal_window(start, window_len, img_batch[imgIdx],
                                                  direc, channel_ordering)
    return new_batch
