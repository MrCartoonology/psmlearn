from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import h5py
if os.environ.get('MOCK_TENSORFLOW',False):
    import psmlearn.mock_tensorflow as tf
else:
    import tensorflow as tf

class Model(object):
    def __init__(self, **kwargs):
        self.X_placeholder = kwargs.pop('X_placeholder')
        self.Y_placeholder = kwargs.pop('Y_placeholder')
        self.trainflag_placeholder = kwargs.pop('trainflag_placeholder')
        self.X_processed = kwargs.pop('X_processed')
        self.nnet        = kwargs.pop('nnet')
        self.train_ops   = kwargs.pop('train_ops')
        self.predict_op  = kwargs.pop('predict_op')
        self.sess        = kwargs.pop('sess')
        self.saver       = kwargs.pop('saver')


def dense_layer_ops(X, num_X, num_Y, config):
    W = tf.Variable(tf.truncated_normal([num_X, num_Y],
                                        mean=0.0,
                                        stddev=np.float32(config.var_init_stddev)),
                    name='W')
    B = tf.Variable(tf.constant(value=np.float32(config.bias_init),
                                dtype=tf.float32,
                                shape=[num_Y]),
                    name='B')
    Y = tf.add(tf.matmul(X, W), B, name='Y')

    return W,B,Y

def get_reg_term(config, vars_to_reg):
    reg_term = tf.constant(value=0.0, dtype=tf.float32)
    l2reg = np.float32(config.l2reg)
    l1reg = np.float32(config.l1reg)
    if (l2reg>0.0 or l1reg>0.0) and len(vars_to_reg)>0:
        for x_var in vars_to_reg:
            if l2reg>0.0:
                x_squared = x_var * x_var
                l2norm = tf.reduce_sum(x_squared)
                reg_term += l2reg * l2norm
            if l1reg>0.0:
                x_abs = tf.abs(x_var)
                l1norm = tf.reduce_sum(x_abs)
                reg_term += l1reg * l1norm
    return reg_term

def cross_entropy_loss_ops(logits, labels, config, vars_to_reg):
    cross_entropy_loss_all = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    loss = tf.reduce_mean(cross_entropy_loss_all)
    reg_term = get_reg_term(config, vars_to_reg)
    opt_loss = loss + reg_term
    return loss, opt_loss

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
        
