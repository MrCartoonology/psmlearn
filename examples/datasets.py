from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import psmlearn

def demo_datasets():
    dsets = {'train':[],'predict':[]}
    dsets['train'].append(psmlearn.dataset(project='xtcav', 
                                           X='img',
                                           Y='enPeak',
                                           testmode=True,
                                           verbose=True))
    dsets['train'].append(psmlearn.dataset(project='xtcav',                                   
                                           testmode=True,
                                           X='img_bldall',
                                           Y='timePeak',
                                           verbose=True))
    dsets['train'].append(psmlearn.dataset(project='xtcav',                                   
                                           testmode=True,
                                           X='img_bldsel',
                                           Y='enAll',
                                           verbose=True))
    dsets['train'].append(psmlearn.dataset(project='xtcav',                                   
                                           testmode=True,
                                           X='img_bldsel',
                                           Y='lasing',
                                           verbose=True))
    dsets['train'].append(psmlearn.dataset(project='ice_water',                                   
                                           testmode=True,
                                           verbose=True))

    dsets['train'].append(psmlearn.dataset(project='diffraction',                                   
                                           testmode=True,
                                           verbose=True))

    dsets['predict'].append(psmlearn.dataset(project='xtcav',                                   
                                             X='img',
                                             testmode=True,
                                             predict=True,
                                             verbose=True))

    for dset in dsets['train']:
        print(dset)
        dset.split(train=80, validation=10, test=10)
        for name in ['train','validation','test']:
            if name == 'train': 
                h5iter = dset.train_iter(batchsize=1, epochs=1, num_batches=3)
            if name == 'validation': 
                h5iter = dset.validation_iter(batchsize=1, epochs=1, num_batches=3)
            if name == 'test': 
                h5iter = dset.test_iter(batchsize=1, epochs=1, num_batches=3)
            for X,Y,meta,batch_info in h5iter:
                batch=batch_info['batch']
                epoch=batch_info['epoch']
                print("%s: train_iter: batch=%2d epoch=%d len(X)=%d len(Y)=%d len(meta)=%d" %
                      (dset, batch, epoch, len(X), len(Y), len(meta)))

    for dset in dsets['predict']:
        h5iter = dset.train_iter(batchsize=1, epochs=1, num_batches=3)
        for X,Y,meta,batch_info in h5iter:
            batch=batch_info['batch']
            epoch=batch_info['epoch']
            print("%s: train_iter: batch=%2d epoch=%d len(X)=%d len(Y)=%d len(meta)=%d" %
                  (dset, batch, epoch, len(X), len(Y), len(meta)))

if __name__ == '__main__':
    demo_datasets()
