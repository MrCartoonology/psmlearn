from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest
import numpy as np
import psmlearn.util as util

class PSMLearnUtil(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_convert_to_one_hot(self):
        labels = np.array([0,0,1,2,0,2,1], dtype=np.int32)
        numLabels = 3
        oneHot = util.convert_to_one_hot(labels, numLabels)
        ans = np.array([[1,0,0],
                        [1,0,0],
                        [0,1,0],
                        [0,0,1],
                        [1,0,0],
                        [0,0,1],
                        [0,1,0]], dtype=np.int32)
        self.assertTrue(np.all(ans==oneHot), msg="answer=\n%r\noneHot=\n%r\n"% (ans, oneHot))

    def test_get_confusion_matrix_one_hot(self):
        truth = np.array([[1,0,0],
                          [1,0,0],
                          [0,1,0],
                          [0,0,1],
                          [1,0,0],
                          [0,0,1],
                          [0,1,0]], dtype=np.int32)
        model_results = np.array([[1,.9,0],
                                  [1,12,0],
                                  [0,1,0],
                                  [0,0,1],
                                  [1,0,2],
                                  [0,3,1],
                                  [0,1,0]], dtype=np.float32)
        ans=np.array([[1,1,1],
                      [0,2,0],
                      [0,1,1]], dtype=np.int)
        cmat = util.get_confusion_matrix_one_hot(model_results, truth)
        self.assertTrue(np.all(cmat==ans), msg="cmat=\n%r\nans=\n%r\n" % (cmat, ans))
    def test_cmat2str(self):
        truth = np.array([[1,0,0],
                          [1,0,0],
                          [0,1,0],
                          [0,0,1],
                          [1,0,0],
                          [0,0,1],
                          [0,1,0]], dtype=np.int32)
        model_results = np.array([[1,.9,0],
                                  [1,12,0],
                                  [0,1,0],
                                  [0,0,1],
                                  [1,0,2],
                                  [0,3,1],
                                  [0,1,0]], dtype=np.float32)
        cmat = util.get_confusion_matrix_one_hot(model_results, truth)
        acc, cmat_rows = util.cmat2str(cmat)
        print(acc)
        print(cmat_rows)
        self.assertEqual(cmat_rows[0], '1 1 1')
        self.assertEqual(cmat_rows[1], '0 2 0')
        self.assertEqual(cmat_rows[2], '0 1 1')
        self.assertAlmostEqual(acc, 4.0/7.0)

    def test_get_best_correct_one_hot(self):
        scores = np.array([[1,2,0],
                           [9,99,20],  # best score when correct for label==1
                           [102,2,1],
                           [10,200,1],
                           [3,0,1]], dtype=np.float32)
        truth = np.array([[0,1,0],
                          [0,1,0],
                          [1,0,0],
                          [0,0,1],
                          [0,1,0]], dtype=np.int32)
        label = 1
        row, best_score = util.get_best_correct_one_hot(scores, truth, label)
        self.assertEqual(row,1)
        self.assertAlmostEqual(best_score,99.0)
        row, best_score = util.get_best_correct_one_hot(scores, truth, 2)
        self.assertTrue(row is None)
        self.assertTrue(best_score is None)
    
if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], '-v'])
