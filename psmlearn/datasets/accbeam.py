from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import dataloc
from . dataset import Dataset

ACCBEAM_DESCR='''
locating beam on VCC and YAG screens for laser diode calibration,
accelerator beam shaping. From Siqi, matlab files.
'''
class AccBeamVccYagDataset(Dataset):
    def __init__(self, project, subproject, verbose, **kwargs):
        Dataset.__init__(self, project=project, subproject=subproject, 
                         verbose=verbose, descr=ACCBEAM_DESCR)
        raise Exception("AccBeamVccYagDataset not implemented")

