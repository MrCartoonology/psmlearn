from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import h5py

def plotRowsLabelSort(dataLoc, labelsLoc, plt=None):
    if plt is None:
        import matplotlib.pyplot as plt
    plt.ion()
    plt.figure()
    datah5 = h5py.File(dataLoc['file'], 'r')
    data = datah5[dataLoc['dset']][:]
    labelh5 = h5py.File(labelsLoc['file'],'r')
    labels = labelh5[labelsLoc['dset']][:]

def _imshowImg(plt, img):
    if len(img.shape)==3:
        mn = np.min(img)
        mx = np.max(img)
        plt_img = img - mn
        plt_img /= (mx-mn)
        assert np.min(plt_img)>=0.0
        assert np.max(plt_img)<=1.0
    else:
        plt_img = img
    plt.imshow(plt_img, interpolation='none')
    
def compareImages(plt, figH, title_imgA, title_imgB):
    plt.figure(figH)
    plt.subplot(1,2,1)
    titleA, imgA = title_imgA
    titleB, imgB = title_imgB
    _imshowImg(plt, imgA)
    plt.title(titleA)
    plt.subplot(1,2,2)
    _imshowImg(plt, imgB)
    plt.title(titleB)
    plt.pause(.1)
