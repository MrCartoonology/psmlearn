from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . h5batchdataset import H5BatchDataset

CHUCKVIRUS_DESCR='''
diffraction patterns from two different viruses
'''
class ChuckVirusDataset(H5BatchDataset):
    def __init__(self, 
                 project='diffraction',
                 subproject='yoon82_amo86615',
                 verbose=True,
                 testmode=False):
        h5files = self.getH5filesFromOneGlobPattern(project=project, subproject=subproject, 
                                                    testmode=testmode, globmatch='amo86615_chuck_virus.h5')
        H5BatchDataset.__init__(self, 
                                project=project, 
                                subproject=subproject, 
                                verbose=verbose, 
                                descr=CHUCKVIRUS_DESCR,
                                name='ChuckVirus(subproject=%s)' % subproject,
                                h5files = h5files,
                                X=['adu'],
                                Y_to_onehot=['labels'],
                                Y_onehot_num_outputs=[2],
                                meta_dset_names=[],
                                )
