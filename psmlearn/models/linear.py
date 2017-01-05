from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import h5py
from .base import tf, dense_layer_ops, get_reg_term, cross_entropy_loss_ops


class LinearClassifier(object):
    def __init__(self, num_features, num_outputs, config,
                 features_dtype=tf.float32, labels_dtype=tf.float32):
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.config = config
        self.vars_to_init = []
        
        with tf.name_scope(name='logits'):
            self.X_pl = tf.placeholder(dtype=features_dtype, shape=(None, num_features), name='X')
            self.Y_pl = tf.placeholder(dtype=labels_dtype, shape=(None, num_outputs), name='Y')
            self.W, self.B, self.logits = dense_layer_ops(X=self.X_pl,
                                                          num_X=num_features,
                                                          num_Y=num_outputs,
                                                          config=config)
        
        with tf.name_scope(name='loss'):
            self.loss, self.opt_loss = cross_entropy_loss_ops(logits=self.logits,
                                                              labels=self.Y_pl,
                                                              config=config,
                                                              vars_to_reg=[self.W])
        self.vars_to_init.extend([self.W, self.B])
        
    def get_training_feed_dict(self,X,Y):
        feed_dict = {self.X_pl:X, self.Y_pl:Y}
        return feed_dict

    def get_validation_feed_dict(self,X,Y):
        feed_dict = {self.X_pl:X, self.Y_pl:Y}
        return feed_dict

    def train_ops(self):
        return []

    def predict_op(self):
        return self.logits
    
    def get_W_B(self, sess):
        return sess.run([self.W, self.B])

    def get_logits(self, sess):
        return sess.run([self.logits])[0]

    def save(self, h5group, sess):
        h5group['W'], h5group['B'] =self.get_W_B(sess)
            
    def restore(self, h5group, sess):
        W = h5group['W'][:]
        B = h5group['B'][:]
        sess.run(self.W.assign(W))
        sess.run(self.B.assign(B))

    def restore_from_file(self, fname, sess):
        h5 = h5py.File(fname)
        h5group = h5['model']
        self.restore(h5group, sess)
