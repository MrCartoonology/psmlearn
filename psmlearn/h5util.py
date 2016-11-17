from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import h5py
import sys

def is_integral(dtype):
    return dtype in  [np.int, 
                      np.int8, np.uint8, 
                      np.int16, np.uint16, 
                      np.int32, np.uint32,
                      np.int64, np.uint64]

def hcat_load(h5, dsets, include):
    if not isinstance(h5, h5py.File):
        h5 = h5py.File(h5,'r')
    inc = h5[include][:]
    assert is_integral(inc.dtype), "include dataset %s is not integral" % include
    assert 0 == len(set(inc).difference(set([0,1]))), "include dataset %s is not 0 or 1, it is %r" % (include, set(inc))
    selIdx = inc==1

    for dset in dsets:
        assert len(h5[dset].shape)==2, "dset %s does not have shape 2" % dset

    X = np.hstack([h5[dset][:] for dset in dsets])
    return X[selIdx,:]                

def write_to_h5(output_file, datadict):
    '''writes a dict, and a list of dicts into a 'meta' group
    '''
    h5 = h5py.File(output_file,'w')
    for ky, val in datadict.iteritems():
        h5[ky]=val
    h5.close()

def read_from_h5(h5in):
    if not isinstance(h5in, h5py.File):
        h5in = h5py.File(h5in, 'r')
    res = {}    
    for ky in h5in.keys():
        if not isinstance(h5in[ky], h5py.Dataset): continue
        try:
            res[ky]=h5in[ky][:]
        except:
            res[ky]=h5in[ky].value
    h5in.close()
    return res

def pack_in_compound(arrList, nameOrder):
    assert len(arrList)>0
    assert len(arrList)==len(nameOrder)
    assert isinstance(arrList, list)
    N = len(arrList[0])
    for k in range(1,len(arrList)):
        assert len(arrList[k])==N
    dtypes = [arr.dtype for arr in arrList]
    list_for_dtype = [(nm,dt) for nm,dt in zip(nameOrder, dtypes)]
    dtype = np.dtype(list_for_dtype)
    compound = np.empty(N, dtype=dtype)
    for arr,nm in zip(arrList,nameOrder):
        compound[nm][:] = arr[:]
    return compound

        
    
def write_config(h5, config):
    gr=h5.create_group('config')
    for nm, val in config.__dict__.iteritems():
        gr[nm]=val
        
