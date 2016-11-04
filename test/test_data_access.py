from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import psmlearn

fvecnames=['bld.ebeam.ebeamCharge',
           'bld.ebeam.ebeamDumpCharge',
           'bld.ebeam.ebeamEnergyBC1',
           'bld.ebeam.ebeamEnergyBC2',
           'bld.ebeam.ebeamL3Energy',
           'bld.ebeam.ebeamLTU250',
           'bld.ebeam.ebeamLTU450',
           'bld.ebeam.ebeamLTUAngX',
           'bld.ebeam.ebeamLTUAngY',
           'bld.ebeam.ebeamLTUPosX',
           'bld.ebeam.ebeamLTUPosY',
           'bld.ebeam.ebeamPhotonEnergy',
           'bld.ebeam.ebeamPkCurrBC1',
           'bld.ebeam.ebeamPkCurrBC2',
           'bld.ebeam.ebeamUndAngX',
           'bld.ebeam.ebeamUndAngY',
           'bld.ebeam.ebeamUndPosX',
           'bld.ebeam.ebeamUndPosY',
           'bld.ebeam.ebeamXTCAVAmpl',
           'bld.ebeam.ebeamXTCAVPhase']


def test_data_access():
    Fvec = datasets.fvec_from_H5_datasets(fvecnames)
    dset = Dataset('ImgMLearnSmall',
                   labels=['acq.enPeaksLabel', 'acq.e1.pos', 'acq.e2.pos'],
                   features=['xtcavimg', Fvec])
    iter = dgen(randomize=True, seed=932, args=None, num=10)
    for X,Y,meta in iter:
        img = X[0]
        fvec = X[1]
        pk, e1, e2 = Y        
        print("img.shape=%s pk=%d e1=%.2f e2=%.2f meta=%s" % (img.shape, pk, e1, e2, meta))
        
if __name__ == '__main__':
    test_data_access()
    
