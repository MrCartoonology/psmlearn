from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import math
import numpy as np
import h5py
import tensorflow as tf
from . import util
from . import h5util

class ClassificationTrainer(object):
    def __init__(self,
                 train_iter,
                 validation_iter,
                 config,
                 sess,
                 save_fname,
                 force,
                 model):
        self.train_iter=train_iter
        self.validation_iter=validation_iter
        self.config=config
        self.sess=sess
        self.model=model
        self.vars_to_init = []
        self.batches_per_validation_epoch = None
        self.best_accuracy=0.0
        self.last_best_accuracy_step = 0
        self.hdr = "ClassificationTrainer"
        self.save_fname = save_fname
        self.force = force
        assert not os.path.exists(save_fname) or force, "output file: %s exists, and force is not True" % save_fname
        self.saved = False
        
        current_variables = set(tf.all_variables())

        self.add_optimizer()

        new_variables = set(tf.all_variables())-current_variables
        vars_to_init = list(new_variables) + self.model.vars_to_init
        print("about to initialize:\n %s" % '\n  '.join([xx.name for xx in vars_to_init]))
        sess.run(tf.initialize_variables(vars_to_init))

    def add_optimizer(self):
        self.global_step = tf.Variable(0, trainable=False)
        if self.config.decay_learning_rate:
            self.learning_rate = tf.train.exponential_decay(learning_rate=self.config.learning_rate,
                                                            global_step=self.global_step,
                                                            decay_steps=self.config.learning_rate_decay_steps,
                                                            decay_rate=self.config.learning_rate_decay_rate,
                                                            staircase=self.config.learning_rate_decay_staircase)
        else:
            self.learning_rate = config.learning_rate

        optimizer=None
        if self.config.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                   momentum=self.config.optimizer_param_momentum)
        assert optimizer is not None, "don't know config.optimizer=%s" % self.config.optimizer
        
        train_op = optimizer.minimize(self.model.opt_loss, global_step=self.global_step)
        self.train_ops = [train_op] + self.model.train_ops()


    def trace(self, msg):
        util.logTrace(hdr=self.hdr, msg=msg)

    def save(self):
        h5 = h5py.File(self.save_fname, 'w')
        h5util.write_config(h5, self.config)
        trainGroup = h5.create_group('train')
        trainGroup['valdidation_accuracy']=self.best_accuracy
        trainGroup['step']=self.last_best_accuracy_step
        trainGroup['validation_confusion_matrix']=self.valid_cmat
        modelGroup = h5.create_group('model')
        self.model.save(modelGroup, self.sess)
        h5.close()        
        
    def stop_training(self, step):
        if step < self.config.min_train_steps + self.config.plateau_steps:
            return False

        if step - self.last_best_accuracy_step > self.config.plateau_steps:
            self.trace("stop_training - plateau detected")
            return True

        return False

    def is_eval_step(self, step):
        if step % self.config.eval_steps == 0 and step > 0:
            return True
        return False

    def is_train_report_step(self, step):
        if step % self.config.train_report_steps == 0 and step > 0:
            return True
        return False
    
    def run(self):
        print("Starting training.")
        sys.stdout.flush()
        for Xlist,Ylist,meta,batchinfo in self.train_iter:
            X=Xlist[0]
            Y=Ylist[0]
            t0=time.time()
            step = batchinfo['step']
            if self.stop_training(step):
                break
            train_feed_dict = self.model.get_training_feed_dict(X=X,Y=Y)
            self.sess.run(self.train_ops, feed_dict=train_feed_dict)
            train_info = None
            eval_info = None
            if self.is_eval_step(step) or self.is_train_report_step(step):
                train_info = self.train_report(feed_dict=train_feed_dict, Y=Y)
            if self.is_eval_step(step):
                eval_info = self.do_eval(step)
            if train_info is None and eval_info is None:
                continue
            
            eval_seconds = time.time()-t0
            self.train_message(step=step,
                               eval_seconds=eval_seconds,
                               train_info=train_info,
                               eval_info=eval_info)
            sys.stdout.flush()
        if not self.saved:
            self.save()

    def train_report(self, feed_dict, Y):
        logits,loss,opt_loss,lr = self.sess.run([self.model.logits, self.model.loss, self.model.opt_loss, self.learning_rate], feed_dict=feed_dict)
        train_cmat = util.get_confusion_matrix_one_hot(logits, Y)
        train_acc, train_cmat_rows = util.cmat2str(train_cmat)
        return {'cmat':train_cmat_rows,
                'acc':train_acc,
                'loss':loss,
                'learning_rate':lr,
                'opt_loss':opt_loss}

    
    def do_eval(self, step):
        t0 = time.time()
        logits, Y = self.run_through_validation_epoch()
        valid_cmat = util.get_confusion_matrix_one_hot(logits, Y)
        valid_acc, valid_cmat_rows = util.cmat2str(valid_cmat)
        
        better = False
        if valid_acc > self.best_accuracy:
            self.best_accuracy = valid_acc
            self.last_best_accuracy_step = step
            self.valid_cmat = valid_cmat
            self.save()
            better  = True
        return {'cmat':valid_cmat_rows,
                'acc':valid_acc,
                'better':better}
    

    def train_message(self, step, eval_seconds, train_info, eval_info):
        msg = 'step=%5d ev.tm=%.1f' % (step, eval_seconds)
        msg += ' loss=%.3e' % train_info['loss']
        msg += ' opt_loss=%.3e' % train_info['opt_loss']
        msg += ' tr.acc=%.2f' % train_info['acc']
        msg += ' lr=%.3e' % train_info['learning_rate']
        train_cmat = train_info['cmat']
        if eval_info is not None:
            msg += ' ev.acc=%.2f' % eval_info['acc']
            valid_cmat = eval_info['cmat']
        else:
            msg += ' ev.acc=----'
            
        N = len(msg)
        msg += ' train.cmat=%s' % train_cmat.pop(0)
        if eval_info:
            msg += '  valid.cmat=%s' % valid_cmat.pop(0)
            if eval_info['better']:
                msg += ' * better'
                
        for tr_row in train_cmat:
            msg += '\n'
            msg += ' '*N
            msg += '            %s' % tr_row
            if eval_info:
                msg += '             %s' % valid_cmat.pop(0)
        print(msg)
    
    def run_through_validation_epoch(self):
        starting_epoch = None
        batches = 0
        logits_all = []
        Y_all = []
        for Xlist,Ylist,meta,batchinfo in self.validation_iter:
            if starting_epoch is None:
                starting_epoch = batchinfo['epoch']
            if batchinfo['epoch'] > starting_epoch:
                break
            batches += 1
            X=Xlist[0]
            Y=Ylist[0]
            assert np.sum(Y)==Y.shape[0], "Y=%r" % Y
            feed_dict=self.model.get_validation_feed_dict(X=X,Y=Y)
            ops = [self.model.logits]
            logits, = self.sess.run(ops, feed_dict=feed_dict)
            assert logits.shape[0]==Y.shape[0]
            logits_all.append(logits)
            Y_all.append(Y)
        logits_all = np.concatenate(logits_all)
        Y_all = np.concatenate(Y_all)
        assert logits_all.shape[0]==Y_all.shape[0]
            
        if self.batches_per_validation_epoch is None:
            self.batches_per_validation_epoch = batches
        elif batches != self.batches_per_validation_epoch:
            self.trace("used to be %d batches in a validation epoch, now there are %d" %
                       (self.batches_per_validation_epoch, batches))
            self.batches_per_validation_epoch = batches
        return logits_all, Y_all
    
        
    def get_accuracy(self):
        pass
    
