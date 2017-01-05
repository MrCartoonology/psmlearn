from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import random
import tensorflow as tf
from scipy.misc import imresize
from psmlearn.pipeline import Pipeline
import psmlearn.h5util as h5util
import psmlearn.util as util
from psmlearn import tsne
import psmlearn.plot as psplot
import psmlearn
import numpy as np
import h5py
import time

################ main code
class AccBeam(object):
    def __init__(self):
        pass

    def init(self, config, pipeline):
        '''pipeline will have set the random seeds for python and numpy, 
        '''
        self.hdr='AccBeam'
        self.dset = psmlearn.get_dataset('accbeam', X=['yag','vcc'], subbkg=True, dev=config.dev)

    def view_orig(self, plot, pipeline, plotFigH, config, step2h5list):
        plt = pipeline.plt
        basicIter = self.dset.train_iter(batchsize=1, epochs=1)
        for X,Y,meta,batchinfo in basicIter:
            plt.clf(plotFigH)
            plt.subplot(1,2,1)
            plt.imshow(X[0][0])
            plt.imshow(X[1][0])

### pipeline ###########
if __name__ == '__main__':
    stepImpl = AccBeam()
    outputdir = psmlearn.dataloc.getDefalutOutputDir(project='accbeam')
    pipeline = Pipeline(stepImpl=stepImpl, outputdir=outputdir)
    pipeline.add_step_method_plot('view_orig')
    pipeline.run()
