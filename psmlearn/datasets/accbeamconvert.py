from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

########## IMPORTS ###############
## standard imports
import os
import sys
import glob

# external package imports
import numpy as np
import scipy.io as sio
import h5py

# this package imports
from . import dataloc

######################################
__doc__='''
converts original matlab files from siqili for accelerator beam finding to hdf5
'''

def translate_numpy_obj_array(fname, nm, matarr, box, OVERRIDE=None):
    '''matarr should look like a [1,N] numpy array of object, and each object 
    is a numpy array.

    The box=True means the datatset will be N x 4, but some of the entries may be
    zero.

    OVERRIDE is for box, if we want to pass in new labels.
    '''
    assert len(matarr.shape) == 2
    assert matarr.shape[0] == 1
    assert matarr.shape[1] > 0

    if OVERRIDE:
        if nm not in OVERRIDE:
            OVERRIDE=None
        else:
            OVERRIDE=OVERRIDE[nm]
    if box:
        finalshape = (matarr.shape[1], 4)
        dtype = np.int16
    else:
        arr=matarr[0,0]
        finalshape=tuple([matarr.shape[1]] + list(arr.shape))
        dtype=arr.dtype
    ans = np.zeros(finalshape, dtype)
    for row in range(matarr.shape[1]):
        if OVERRIDE and box and row in OVERRIDE:
            ans[row]=OVERRIDE[row]
            continue
        if box and matarr[0,row].shape == (0,0): continue
        if box and matarr[0,row].shape == (1,4) or (not box):
            ans[row]=matarr[0,row]
        else:
            print("skipping row=%d key=%s of file=%s" % (row, nm, fname))
            print("  it should be a box, but shape is not (0,0) or (1,4), val=%s" % matarr[0,row])
    return ans

def translate_labelimg(fin, fout, SCHEMA, OVERRIDE):
    '''fin is a .mat filename,
    fout a .h5 filename
    schema - what to do with each dataset
    '''
    mat = sio.loadmat(fin)
    h5 = h5py.File(fout,'w')
    print("fin=%s fout=%s" % (fin,fout))
    for ky in mat.keys():
        if ky not in SCHEMA:
            print("WARNING: %s not in SCHEMA. val=\n%r" % (ky, mat[ky]))
            continue
        what_to_do = SCHEMA[ky]
        if what_to_do == 'skip': continue
        if what_to_do == 'str':
            h5[ky]=mat[ky]
        if what_to_do == 'arr':
            h5[ky]=mat[ky]
        if what_to_do == 'boxarr':
            h5[ky]=translate_numpy_obj_array(fin, ky, mat[ky], box=True, OVERRIDE=OVERRIDE)
        if what_to_do == 'objectarr':
            h5[ky]=translate_numpy_obj_array(fin, ky, mat[ky], box=False)

SCHEMA={
    'yagImg' :            'objectarr',
    'yagbox' :            'boxarr',
    'vccbox' :            'boxarr',
    'vccImg' :            'objectarr',
    'vccbeam':            'arr',
    'yagbeam':            'arr',
    'log'    :            'skip',
    'vccbkg' :            'arr',
    'yagbkg' :            'arr',
    '__header__'  :       'str',
    '__globals__' :       'skip',
    '__version__' :       'str',
}

OVERRIDE = {4:{'vccbox':{69:[232,350,258,377],
                        },
              },
           } 

def run():
    subdir = dataloc.getSubProjectDir('accbeam','siqili')
    hdf5 = os.path.join(subdir, 'hdf5')
    if not os.path.exists(hdf5):
        os.mkdir(hdf5)
    matlab = os.path.join(subdir, 'matlab')
    assert os.path.exists(matlab)
    BASE='labeledimg'
    GLOB=BASE + '*.mat'
    DIG=BASE + '%d.mat'
    DIG_H5=BASE + '%d.h5'
    all_labelimg_mats = glob.glob(os.path.join(matlab, GLOB))
    for fileno in [1,2,4]:
        fname = os.path.join(matlab, DIG % fileno)
        assert fname in all_labelimg_mats, "file not present: %s, glob=%s" % (fname, '\n'.join(all_labelimg_mats))
        all_labelimg_mats.remove(fname)
    assert len(all_labelimg_mats)==0, "unexpected files: %s" % all_labelimg_mats

    global SCHEMA
    
    for fileno in [1,2,4]:
        input_fname = os.path.join(matlab, DIG % fileno)
        output_fname = os.path.join(hdf5, DIG_H5 % fileno)
        translate_labelimg(input_fname, output_fname, SCHEMA, OVERRIDE.get(fileno,None))
