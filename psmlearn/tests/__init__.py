from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from unittest import TestCase

import numpy as np
import psmlearn

class TestUtil(TestCase):
    def test_convert_to_one_hot(self):
        one_hot = psmlearn.util.convert_to_one_hot(labels=[1,2,2,1], numLabels=4)
        one_hot_ans = np.array([[0,1,0,0],
                                [0,0,1,0],
                                [0,0,1,0],
                                [0,1,0,0]], dtype=np.int32)
        self.assertTrue(np.array_equal(one_hot, one_hot_ans), msg='one_hot=%r not equal to expected=%r' %
                        (one_hot, one_hot_ans))

    def test_get_confusion_matrix_one_hot(self):
        model_results = np.array([[0.1, 0.3],
                                  [0.9, 0.6],
                                  [0.1,0.8],
                                  [0.5, 0.3]])
        truth = np.array([[0,1],
                          [1,0],
                          [1,0],
                          [1,0]], np.int32)
        ans = np.array([[2,1],
                        [0,1]], np.int32)
                    
        cmat = psmlearn.util.get_confusion_matrix_one_hot(model_results=model_results, truth=truth)
        self.assertTrue(np.array_equal(ans, cmat), msg='get_confusion_matrix_one_hot(model_results=%r,truth=%r) != %r instead it is %r' %
                        (model_results, truth, cmat, ans))

    def test_cmat2str(self):
        cmat = np.array([[2,1],
                         [1,3]], dtype=np.int32)
        fmtLen=3
        acc,cmat_rows = psmlearn.util.cmat2str(cmat,fmtLen)
        self.assertAlmostEqual(acc,5.0/7.0)
        self.assertEqual(len(cmat_rows),2)
        self.assertEqual('  2   1', cmat_rows[0])
        self.assertEqual('  1   3', cmat_rows[1])

    def test_get_best_correct_one_hot(self):
        scores = np.array([[0.4, 0.6, 0.1],
                           [0.8, 0.2, 0.3],
                           [0.9, 0.94,0.3],
                           [0.9, 0.95,0.2]])
        truth = np.array([[0,1,0],
                          [0,0,1],
                          [0,1,0],
                          [1,0,0]])
        label = 1
        row, score = psmlearn.util.get_best_correct_one_hot(scores, truth, label)
        self.assertEqual(row, 2, msg="row=%d != 2" % (row,))
        self.assertAlmostEqual(score, 0.94, msg='score=%.2f != 0.94' % score)
