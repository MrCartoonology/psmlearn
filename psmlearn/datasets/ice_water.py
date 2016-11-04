from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import random

from h5batchreader import DataSetGroup as H5DataSetGroup

from  . import dataloc
from .. import util
from . h5batchdataset import H5BatchDataset
from . dataset import Dataset

ICEWATER_DESCR='''
sorting ice from water in diffraction, ideally locate ice clusters
'''
class IceWaterDataset(H5BatchDataset):
    def __init__(self, 
                 project='ice_water',
                 subproject='cxi25410',
                 verbose=True,
                 testmode=False):
        h5files = self.getH5filesFromOneGlobPattern(project=project, subproject=subproject, 
                                                    testmode=testmode, globmatch='ice_data.h5')
        H5BatchDataset.__init__(self, 
                                project=project, 
                                subproject=subproject, 
                                verbose=verbose, 
                                descr=ICEWATER_DESCR,
                                name='IceWater(subproject=%s)' % subproject,
                                h5files = h5files,
                                X=['data'],
                                Y_to_onehot=['label'],
                                Y_onehot_num_outputs=[2],
                                meta_dset_names=[],
                                )


