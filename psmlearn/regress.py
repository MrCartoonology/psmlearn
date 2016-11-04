from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import h5py
from sklearn.feature_selection import VarianceThreshold
from sklearn import linear_model
from . import h5util

def do_regression(X, Y, imgs=None):
    N = len(X)
    print("do_regression N=%d" % N)
    assert N==len(Y)
    assert len(Y.shape)==2
    assert len(X.shape)==2
    predictY = np.zeros(Y.shape, dtype=np.float)
    for row in range(N):
        trainX = np.zeros((N-1, X.shape[1]), dtype=np.float)
        trainY = np.zeros((N-1, Y.shape[1]), dtype=np.float)
        testX = np.zeros((1,X.shape[1]), dtype=np.float)
        testY = np.zeros((1,Y.shape[1]), dtype=np.float)
        regr = linear_model.LinearRegression()
        trainX[0:row,:] = X[0:row,:].copy()
        trainY[0:row,:] = Y[0:row,:].copy()
        trainX[row:,:] = X[(row+1):,:].copy()
        trainY[row:,:] = Y[(row+1):,:].copy()
        testX[0,:] = X[row,:]
        regr.fit(trainX, trainY)
        testY[:] = regr.predict(testX)[:]
        ansY = Y[row,:]
        predictY[row,:] = testY[0,:]
    return predictY

def regress(inputh5, outputh5, problems, variance_feature_select=-1, force=False):
    assert not os.path.exists(outputh5) or force, "output file: %s exists, use --force to overwrite" % outputh5
    h5in = h5py.File(inputh5,'r')
    h5out = h5py.File(outputh5,'w')
    for nm, nmDict in problems.iteritems():
        print("-- %s --" % nm)
        X=h5util.hcat_load(h5in, dsets=nmDict['X_ds'], include=nmDict['include_ds'])
        Y=h5util.hcat_load(h5in, dsets=nmDict['Y_ds'], include=nmDict['include_ds'])
        assert X.shape[0]>0, "There is not data to do regression on"
        
        if variance_feature_select > 0.0:
            X = VarianceThreshold(variance_feature_select).fit_transform(X)
            print("nm=%s after variance thresh=%.3f, %d feats" % (nm, variance_feature_select, X.shape[1]))

        predictY = do_regression(X=X, Y=Y)

        incIdx=h5in[nmDict['include_ds']][:]==1
        outDs = '%s_predict' % nm
        h5out[outDs] = np.zeros((len(incIdx), Y.shape[1]), dtype=Y.dtype)
        origRows = np.where(True==incIdx)[0]
        assert len(origRows)==predictY.shape[0], "number of entries in include not equal to number of rows in predicted - nm=%s" % nm
        for idx, row in enumerate(origRows):
            h5out[outDs][row,:] = predictY[idx,:]
    h5out.close()
    h5in.close()
