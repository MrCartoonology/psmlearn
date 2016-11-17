from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest
import numpy as np
import h5py
import tempfile
import psmlearn.h5util as h5util

class TestH5util( unittest.TestCase):
    def setUp(self):
        h, self.fname = tempfile.mkstemp(suffix=".h5", prefix="tmp_test_psmlearn_h5util")
        os.close(h)

    def tearDown(self):
        os.unlink(self.fname)

    def test_write_read(self):
        write = {'A':np.random.randint(low=0, high=10, size=(10,3)),
                 'B':np.random.randint(low=-3, high=20, size=(5,2))}
        h5util.write_to_h5(self.fname, write)
        read = h5util.read_from_h5(self.fname)

        
        self.assertEqual(set(write.keys()),set(read.keys()))
        for ky in write.keys():
            self.assertTrue(np.all(write[ky]==read[ky]))

    def test_meta(self):
        names = ['seconds','nano']
        meta = {'seconds':np.array([ 1, 1, 1, 1, 2, 2, 2], dtype=np.int32),
                'nano':   np.array([23,25,31,45,56,69,92], dtype=np.int32)}
        compound = h5util.pack_in_compound(meta, names)
        for nm in names:
            self.assertTrue(np.all(meta[nm]==compound[nm]))
        h5 = h5py.File(self.fname,'w')
        h5['meta']=compound
        h5.close()
        
        match = compound.copy()
        match['seconds'][0]=3
        match['seconds'][2]=5
        match['nano'][4]=3

        # should match rows 1,3,5,6
        # matched = h5util.match_meta(match, self.fname)
        
if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], '-v'])
