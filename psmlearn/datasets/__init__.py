from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from . import dataloc
from .. import util
from .xtcav import ImgMLearnDataset
from .ice_water import IceWaterDataset
from .diffraction import ChuckVirusDataset
#from .accbeam import AccBeamVccYagDataset

###### All datasets #######
DATASETS={('xtcav','amo86815_small'):ImgMLearnDataset,
          ('xtcav','amo86815_full'):ImgMLearnDataset,
          ('ice_water','cxi25410'):IceWaterDataset,
          ('diffraction','yoon82_amo86615'):ChuckVirusDataset,
      }
#          ('accbeam','siqili'):AccBeamVccYagDataset}

DATASET_GROUPS=list(set([ky[0] for ky in DATASETS]))
DATASET_GROUPS.sort()
DATASET_GROUPS=','.join(DATASET_GROUPS)

def get_dataset(project, **kwargs):
    '''get a psmlearn dataset object corresponding to data in the
    psmlearn data area. Data is organized by
    project/subproject under this directory.

    call this function with no arguments for help.
    '''
    
    projectDir = dataloc.getProjectDir(project)
    verbose = kwargs.pop('verbose',False)
    subproject = kwargs.pop('subproject','default_subproject')
    subProjectDir = dataloc.getSubProjectDir(project, subproject)
    if subproject == 'default_subproject':
        subproject = os.path.basename(os.path.realpath(subProjectDir))
        subProjectDir = dataloc.getSubProjectDir(project, subproject)
        util.logTrace(hdr='datasets',
                      msg="project=%s, default_subproject detected, subproject set to: %s" % 
                      (project,subproject),
                      flag=verbose)

    dataset_id = (project, subproject)
    
    assert dataset_id in DATASETS, \
        "%s not in known datasets. known datsets are:\n%r" % \
        (dataset_id, DATASETS.keys())
    cl=DATASETS[dataset_id]
    return cl(project=project,
              subproject=subproject,
              verbose=verbose,
              **kwargs)

