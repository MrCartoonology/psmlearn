from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import copy
import random
import numpy as np

import h5batchreader
assert h5batchreader.__version__=='v1.0.0', "got version: %s" % h5batchreader.__version__
from h5batchreader import H5BatchReader

from . dataset import Dataset
from . import dataloc
from .. import util
from .. import h5util

class H5BatchDataset(Dataset):
    '''Creates a psmlearn.Dataset interface around a set of hdf5 files in the
    psmlearn data tree. The dataset is for machine learning.

    Each hdf5 file has a collection of aligned datasets off the root group.
    The rows of these datasets samples.  Arguments:

    These arguments give the location of the files in the psmlearn tree, and describe
    them

    project
    subproject
    verbose
    descr
    name

    h5files - a list of the hdf5 files for the dataset
    X - a list of the datasets for the features.
    X_dset_groups - a list of DataSetGropus
                 Y=[], Y_to_onehot=[],
                 Y_onehot_num_outputs=[],
                 meta_dset_names=[],
                 dev=False,
                 add_batch_info_to_meta=True,
                 include_if_one_mask_datasets=[],
                 exclude_if_negone_mask_datasets=[]):
    '''
    def __init__(self, project, subproject, verbose, descr, name,
                 h5files, X=[], X_dset_groups=[],
                 Y=[], Y_to_onehot=[],
                 Y_onehot_num_outputs=[],
                 meta_dset_names=[],
                 dev=False,
                 add_batch_info_to_meta=True,
                 include_if_one_mask_datasets=[],
                 exclude_if_negone_mask_datasets=[]):

        Dataset.__init__(self, 
                         project=project, 
                         subproject=subproject, 
                         verbose=verbose, 
                         descr=descr,
                         name=name,
                         dev=dev)
        assert len(h5files)>0        
        self.h5files = h5files
        self.X=X
        self.X_dset_groups=X_dset_groups
        self.Y=Y
        self.Y_to_onehot=Y_to_onehot
        self.Y_onehot_num_outputs=Y_onehot_num_outputs
        self.meta_dset_names=meta_dset_names
        self.meta_and_batch_dset_names = meta_dset_names + ['file','row']
        self.add_batch_info_to_meta = add_batch_info_to_meta
        self.include_if_one_mask_datasets = include_if_one_mask_datasets
        self.exclude_if_negone_mask_datasets = exclude_if_negone_mask_datasets

        dsets=self.X + self.Y_to_onehot + self.Y + self.meta_dset_names
        dset_groups = self.X_dset_groups
        
        self.h5br = H5BatchReader(h5files=h5files,
                                  dsets=dsets,
                                  dset_groups=dset_groups,
                                  include_if_one_mask_datasets=include_if_one_mask_datasets,
                                  exclude_if_negone_mask_datasets=exclude_if_negone_mask_datasets,
                                  verbose=verbose)
        
    def split(self, **kwargs):
        seed = kwargs.pop('seed', None)
        if seed is not None:
            pystate = random.getstate()
            npstate = np.random.get_state()
            random.seed(seed)
            np.random.seed(seed)
            
        self.h5br.split(**kwargs)

        if seed is not None:
            random.setstate(pystate)
            np.random.set_state(npstate)

    def iter_from(self, h5files, X, Y, **kwargs):
        '''return a new batch iterator from a collection of h5files.
        '''
        assert isinstance(X,list)
        assert isinstance(Y,list)
        assert len(X)
        assert len(Y)
        assert isinstance(X[0],str)
        assert isinstance(Y[0],str)
        h5br = H5BatchReader(h5files=h5files,
                             dsets=X+Y+['meta'])
        h5br.split(train=100,validation=0,test=0)
        new_iter = h5br.train_iter(**kwargs)
        return IterFromWrapper(new_iter, X, Y)

    def num_samples_train(self):
        return self.h5br.num_samples_train()
    
    def num_samples_validation(self):
        return self.h5br.num_samples_validation()

    def num_samples_test(self):
        return self.h5br.num_samples_test()

    def train_iter(self, **kwargs):
        seed = kwargs.pop('seed', None)
        if seed is not None:
            pystate = random.getstate()
            npstate = np.random.get_state()
            random.seed(seed)
            np.random.seed(seed)
            
        h5br_iter = self.h5br.train_iter(**kwargs)

        if seed is not None:
            random.setstate(pystate)
            np.random.set_state(npstate)
            
        return H5BatchDatasetIterWrapper(h5br_iter, self)

    def validation_iter(self, **kwargs):
        h5br_iter = self.h5br.validation_iter(**kwargs)
        return H5BatchDatasetIterWrapper(h5br_iter, self)

    def test_iter(self, **kwargs):
        h5br_iter = self.h5br.test_iter(**kwargs)
        return H5BatchDatasetIterWrapper(h5br_iter, self)

    def getH5filesFromOneGlobPattern(self, project, subproject, testmode, globmatch):
        subprojectDir = dataloc.getSubProjectDir(project=project, subproject=subproject)
        hdf5 = os.path.join(subprojectDir, 'hdf5')
        assert os.path.exists(hdf5), "dir %s doesn't exist" % hdf5
        globpath=os.path.join(hdf5, globmatch)
        h5files = glob.glob(globpath)
        assert len(h5files)>0, "didn't get any files from %s" % globpath
        if testmode:
            random.shuffle(h5files)
            h5files = h5files[0:3]
        return h5files

############### Iter Wrappers


def unpack_batchinfo(batch):
    '''takes batch as returned by h5batchreader.H5BatchReader, which includes 
    data, meta, as well as batch info, and just pulls out batch info items.
    '''
    batchinfo = {'epoch':batch['epoch'],
                 'batch':batch['batch'],
                 'step':batch['step'],
                 'filesRows':batch['filesRows'],
                 'readtime':batch['readtime'],
                 'size':batch['size']}
    return batchinfo

def make_list_from_batch(batch, dset_or_dset_groups_name_list):
    arr_list = []
    for nm in dset_or_dset_groups_name_list:
        if nm in batch['dsets']:
            arr_list.append(batch['dsets'][nm])
        else:
            assert nm in batch['dset_groups'], "nm=%s is in neither the batch 'dsets' or 'dset_groups'" % nm
            arr_list.append(batch['dset_groups'][nm])
    return arr_list

def unpack_XYmeta(batch, X, 
                  Y_to_onehot, Y_onehot_num_outputs, Y,
                  meta_dset_names):
    '''takes batch as returned by h5batchreader.H5BatchReader, which includes 
    dset, dset_groups, meta, as well as batch info, and returns X,Y,meta.

    X = list of what is in X, followed by X_dset_groups
    Y = list of Y_to_one_hot (transformed using Y_onehot_num_outputs list)
        followed by Y
    meta = everything in meta_dset_names
    '''
    Xarrs = make_list_from_batch(batch, X) 

    Yarrs = []
    for nm, numOutputs in zip(Y_to_onehot,
                              Y_onehot_num_outputs):
        labels = batch['dsets'][nm]
        Yarrs.append(util.convert_to_one_hot(labels, numOutputs))
    for nm in Y:
        Yarrs.append(batch['dsets'][nm])

    meta = [batch['dsets'][nm] for nm in meta_dset_names]
    return Xarrs,Yarrs,meta

class IterFromWrapper(object):
    def __init__(self, data_iter, X, Y):
        self.data_iter=data_iter
        self.X=X
        self.Y=Y

    def __iter__(self):
        return self

    def shuffle(self):
        self.data_iter.samples.shuffle()
        
    def next(self):
        batch = self.data_iter.next()
        batchinfo = unpack_batchinfo(batch)
        X=make_list_from_batch(batch, self.X)
        Y=make_list_from_batch(batch, self.Y)
        meta = batch['dsets']['meta']
        return X,Y,meta,batchinfo
                                        
class H5BatchDatasetIterWrapper(object):
    def __init__(self, h5batch_iter, h5batchDataset):
        self.h5batch_iter = h5batch_iter
        self.h5batchDataset = h5batchDataset

    def get_h5files(self):
        return self.h5batch_iter.get_h5files()

    def __len__(self):
        return len(self.h5batch_iter)

    def samplesPerEpoch(self):
        return self.h5batch_iter.samplesPerEpoch()
    
    def unpack_XYmeta(self, batch):
        return unpack_XYmeta(batch=batch,
                             X=self.h5batchDataset.X + self.h5batchDataset.X_dset_groups,
                             Y_to_onehot=self.h5batchDataset.Y_to_onehot,
                             Y_onehot_num_outputs=self.h5batchDataset.Y_onehot_num_outputs,
                             Y=self.h5batchDataset.Y,
                             meta_dset_names=self.h5batchDataset.meta_dset_names)

    def next(self):
        batch = self.h5batch_iter.next()
        batchinfo = unpack_batchinfo(batch)
        X, Y, meta = self.unpack_XYmeta(batch)
        if self.h5batchDataset.add_batch_info_to_meta:
           meta.append(batchinfo['filesRows']['file'])
           meta.append(batchinfo['filesRows']['row'])            
           meta_compound = h5util.pack_in_compound(meta, self.h5batchDataset.meta_and_batch_dset_names )
        else:
           meta_compound = h5util.pack_in_compound(meta, self.h5batchDataset.meta_dset_names)
        
        return X, Y, meta_compound, batchinfo

    def __iter__(self):
        return self
        
