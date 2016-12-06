from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import numpy as np
import h5py
import random
import tensorflow as tf
import psmlearn
from psmlearn.pipeline import Pipeline

__doc__ = '''
We expand on tut01, add saliency maps.
'''

import tut01

        
class NewSteps(tut01.MySteps):
    def __init__(self):
        tut01.MySteps.__init__(self)

    def add_commandline_arguments(self, parser):
        tut01.MySteps.add_commandline_arguments(self, parser)
        
    def init(self, config, pipeline):
        tut01.MySteps.init(self, config, pipeline)
        self.saliency_op = tf.gradients(ys=self.logits, xs=self.imgs_pl, grad_ys=self.logits_pl)[0]
        self.gbprop_op = psmlearn.saliencymaps.gbprop_op(self.relus,
                                                         self.imgs_pl,
                                                         self.logits,
                                                         self.logits_pl)

    def saliency(self, config, pipeline, step2h5list, output_files):
        h5py.File(output_files[0],'w')

    def view_saliency(self, plot, pipeline, plotFigH, config, step2h5list):
        plt = pipeline.plt
        train_fname = step2h5list['train'][0]
        saved_model = train_fname.rsplit('.h5',1)[0] + '.tfmodel'
        assert os.path.exists(saved_model), "fname=%s doesn't exist" % saved_model

        trainh5 = h5py.File(train_fname,'r')
        cmat = trainh5['validation_confusion_matrix'][:]
        acc = np.trace(cmat)/float(np.sum(cmat))

        print("model accuracy is %.2f" % acc)

        sess = pipeline.session
        
        self.saver.restore(sess, saved_model)

        logthresh = config.logthresh
        batchsize=1
        basic_iter = self.dset.test_iter(batchsize=batchsize, epochs=1)
        correct=0
        wrong=0
        plt.figure(plotFigH)
        for X,Y,meta,batchinfo in basic_iter:
            img_batch = X[0]
            onehot_batch_label=Y[0]
            prep_batch = tut01.prep_imgs(img_batch, logthresh, expanddims=3)
            logits = sess.run(self.logits, {self.imgs_pl:prep_batch})
            predicted = np.argmax(logits[0,:])
            if predicted != np.argmax(onehot_batch_label[0,:]):
                wrong += 1
                continue
            correct += 1
            scores = np.exp(logits[0])/np.sum(np.exp(logits[0]))
            scores_str='%.2f %.2f %.2f %.2f' % (scores[0], scores[1], scores[2], scores[3])
            dimg = sess.run(self.saliency_op, feed_dict={self.imgs_pl:prep_batch,
                                                         self.logits_pl:onehot_batch_label.astype(np.float32)})
            gbprop = sess.run(self.gbprop_op, feed_dict={self.imgs_pl:prep_batch,
                                                         self.logits_pl:onehot_batch_label.astype(np.float32)})
            rprop = sess.run(self.relprop_op, feed_dict={self.imgs_pl:prep_batch,
                                                         self.R_pl:onehot_batch_label[0].astype(np.float32)})
            print(rprop.shape)
            print(rprop.flatten()[0:3])
            
            plt.clf()
            plt.subplot(1,4,1)
            plt.title("dimgs")
            plt.imshow(dimg[0,:,:,0], interpolation='none', origin='lower')

            plt.subplot(1,4,2)
            plt.title("label=%d scores %s" % (predicted, scores_str))
            plt.imshow(prep_batch[0,:,:,0], interpolation='none', origin='lower')

            plt.subplot(1,4,3)
            plt.title("gbprop")
            plt.imshow(gbprop[0,:,:,0], interpolation='none', origin='lower')

            plt.subplot(1,4,4)
            plt.title("relprop")
            plt.imshow(rprop[:,:,0], interpolation='none', origin='lower')
            if pipeline.stop_plots(): break
            
if __name__ == '__main__':
    newsteps = NewSteps()
    pipeline = Pipeline(stepImpl=newsteps,
                        outputdir='.')        # can be overriden with command line arguments

    # you can add your own command line arguments however you like by modifying
    # pipeline.parser (you can't override pipeline args, accidentally using an existing argument
    # generates a Python exception)
    newsteps.add_commandline_arguments(pipeline.parser)
    pipeline.add_step_method_plot(name='view')
    pipeline.add_step_method(name='train')
    pipeline.add_step_method(name='saliency')
    pipeline.add_step_method_plot(name='view_saliency')
    
    # init will parse command line arguments, read config, and call mysteps.init()
    pipeline.run()
