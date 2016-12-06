from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import h5py
import random
import tensorflow as tf
import psmlearn
from psmlearn.pipeline import Pipeline

__doc__ = '''
A typical analysis may involve several steps, such as

* preprocess the images
* train a model
* analyze the model

There may be a number of options one wants to try for these
steps.

Below we isolate functions to do these steps. 

Then we integrate them in a machine learning pipeline using the
psmlearn.Pipeline class.
'''

def prep_imgs(imgs, logthresh=200.0, expanddims=3):
    '''Expects a batch of images, a 3D array of 2D images, 
    like a (10, 726, 568) block of images.

    returns a batch, with an extra dimension for channels based on
    the expanddims, if expanddims is 3, will return (10, 726, 568, 1)

    if logthresh is 200, spots where imgs > logthresh 
    are set to log(imgs-199) + 200
    '''
    imgs = psmlearn.util.log_thresh(imgs, logthresh)
    if expanddims is not None:
        imgs = np.expand_dims(imgs, axis=expanddims)
    return imgs

def make_logits(X, num_outputs):
    relus = []
    
    ## layer 1 
    kern01 = tf.Variable(tf.truncated_normal([9,9,1,1], mean=0.0, stddev=0.03))
    conv01 = tf.nn.conv2d(X, kern01, strides=(1,1,1,1), padding="SAME")
    bias01 = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[1]))
    addBias01 = tf.nn.bias_add(conv01, bias01)
    nonlinear01 =tf.nn.relu(addBias01)
    relus.append(nonlinear01)
    pool01 = tf.nn.max_pool(value=nonlinear01, ksize=(1,8,8,1), 
                            strides=(1,7,7,1), padding="SAME")

    ## layer 2
    kern02 = tf.Variable(tf.truncated_normal([6,6,1,8], mean=0.0, stddev=0.03))
    conv02 = tf.nn.conv2d(pool01, kern02, strides=(1,1,1,1), padding="SAME")
    bias02 = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[8]))
    addBias02 = tf.nn.bias_add(conv02, bias02)
    nonlinear02 =tf.nn.relu(addBias02)
    relus.append(nonlinear02)
    pool02 = tf.nn.max_pool(value=nonlinear02, ksize=(1,6,6,1), 
                            strides=(1,5,5,1), padding="SAME")
    
    num_inputs_to_layer03 = 1
    for dim in pool02.get_shape()[1:].as_list():
        num_inputs_to_layer03 *= dim
    input_to_layer03 = tf.reshape(pool02, [-1, num_inputs_to_layer03])
    print("outputs from conv layers: %d" % num_inputs_to_layer03)
    ## layer 3
    weights03 = tf.Variable(tf.truncated_normal([num_inputs_to_layer03, 16], mean=0.0, stddev=0.03))
    bias03 = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[16]))
    xw_plus_b = tf.nn.xw_plus_b(input_to_layer03, weights03, bias03)
    nonlinear03 = tf.nn.relu(xw_plus_b)
    relus.append(nonlinear03)

    ## layer 4
    weights04 = tf.Variable(tf.truncated_normal([16, num_outputs], mean=0.0, stddev=0.1))
    bias04 = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[num_outputs]))
    logits =  tf.nn.xw_plus_b(nonlinear03, weights04, bias04)

    return logits, relus


def make_ops(X,Y, learning_rate=0.01, learning_decay_rate=.99, momentum=0.85):
    '''returns tensorflow ops for a CNN model F, such that F(X)=Y.
    F will be have 2-3 convolutaional layers and two dense layers.
    ops returned will be:
    
      logits, relus, loss, train_op

    train_op is used to train, using a momentum optimizer.
    logits can be used to evalute the model.
    '''
    num_outputs = int(Y.get_shape()[1])
    logits, relus = make_logits(X, num_outputs)

    ## loss 
    cross_entropy_loss_all = tf.nn.softmax_cross_entropy_with_logits(logits, Y)
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss_all)

    ## training
    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                               global_step=global_step,
                                               decay_steps=100,
                                               decay_rate=learning_decay_rate,
                                               staircase=True)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
    train_op = optimizer.minimize(cross_entropy_loss, global_step=global_step)

    return logits, relus, cross_entropy_loss, train_op

def get_validation_confusion_matrix(valid_iter, logthresh, sess, logits, imgs_pl):
    cmat = None
    for X,Y,meta,batchinfo in valid_iter:
        imgs = prep_imgs(imgs=X[0], logthresh=logthresh, expanddims=3)
        labels = Y[0]
        logits_arr = sess.run(logits, feed_dict={imgs_pl:imgs})
        batch_cmat = psmlearn.util.get_confusion_matrix_one_hot(logits_arr, labels)
        if cmat is None:
            cmat = batch_cmat
        else:
            cmat += batch_cmat
    return cmat            
        
class MySteps(object):
    def __init__(self):
        pass

    def add_commandline_arguments(self, parser):
        '''takes the pipeline parser as input and adds command line
        arguments. 
        '''
        parser.add_argument('--logthresh', type=float, help='where to apply a log threshold to the images', default=200)
        parser.add_argument('--batchsize', type=int, help='batchsize for training', default=64)
        parser.add_argument('--lr', type=float, help='initial learning rate during training', default=0.002)
        parser.add_argument('--lrdecay', type=float, help='learning rate decay, every 100 steps, decay by this much', default=0.999)
        parser.add_argument('--momentum', type=float, help='amount of momentum for momentum optimizer', default=0.85)
        parser.add_argument('--maxsteps', type=int, help='maximum number of training steps to take', default=500)
        parser.add_argument('--validation', type=int, help='steps between validations', default=100)
        
    def init(self, config, pipeline):
        '''This is always called before running whichever steps will be run.
        A good place to create tensorflow ops that will be re-used.
        '''
        self.dset = psmlearn.get_dataset(project='xtcav',     # psmlearn knows about a few LCLS machine learning datasets
                                         subproject='amo86815_full',
                                         X='img',     # each dataset may have one or more features to use, ie, images and/or BLD
                                         Y='enPeak',  # each dataset may have one or more values to learn
                                         verbose=True,
                                         dev=config.dev)  # pipeline will manage the --dev option, for fast development

        # 'xtcav' is the only dataset that is fully implemented right now. Options for X and Y are:
        # X= img - just return the xtcavimg
        #    img_bldall: return two things, xtcav_img and vec of all Bld
        #    img_bldsel: return two things, xtcavimg and subset of bld
        # Y=
        # enPeak:   1 item: acq.enPeaksLabel (as one hot - 4 values)
        # timePeak: 1 item: acq.peaksLabel (as one hot - 4 values)
        # lasing:   1 item: lasing (as one hot - 2 values)
        # enAll:    5 items: enPeak(one hot), acq.e1.pos, acq.e2.pos, acq.e1.ampl, acq.e2.ampl
        # timeAll:  5 items: timePeak(one hot), acq.t1.pos, acq.t2.pos, acq.t1.ampl, acq.t2.ampl

        self.dset.split(train=97, validation=1, test=2, seed=config.split_seed)

        train_config = pipeline.get_config('train')
        learning_rate = train_config.lr
        momentum = train_config.momentum
        learning_decay_rate = train_config.lrdecay
        batchsize = int(train_config.batchsize)

        self.imgs_pl = tf.placeholder(tf.float32, (None, 726, 568, 1))
        self.labels_pl = tf.placeholder(tf.int32, (None, 4))

        self.logits, self.relus, self.loss, self.train_op = make_ops(self.imgs_pl, self.labels_pl,
                                                                     learning_rate=learning_rate,
                                                                     learning_decay_rate=learning_decay_rate,
                                                                     momentum=momentum)

        init = tf.initialize_all_variables()
        self.saver = tf.train.Saver()
        sess = pipeline.session
        sess.run(init)
        
    def view(self, plot, pipeline, plotFigH, config, step2h5list):
        # any arguments on the command line are named variables in the config passed to each step
        logthresh = config.logthresh

        batchsize=1
        basic_iter = self.dset.train_iter(batchsize=batchsize, epochs=1)

        h5files = basic_iter.get_h5files()  # all the h5files in this dataset
        
        print("first few h5files for dataset: %s" % '\n'.join(basic_iter.get_h5files()[0:4]))

        for X,Y,meta,batchinfo in basic_iter:
            assert isinstance(X, list), "X is a list of the features specified when the dataset was constructed"
            assert isinstance(Y, list), "Y is a list of outputs specified when the dataset was constructed"
            img_batch = X[0]
            assert len(img_batch.shape)==3 and img_batch.shape[0]==batchsize, \
                "iter returns batches, first dimension is batchsize"

            onehot_batch_label=Y[0]
            assert onehot_batch_label.shape==(batchsize,4), "for this dataset, 'enPeak' means return labels as one-hot"
            label = np.argmax(onehot_batch_label[0,:])

            # meta depends on the dataset. For 'xtcav', it is a numpy array with a dtype that has named fields
            # based on the event from the LCLS data that it came from (but not the dataset, that is amo86815),
            # as well as where in the h5 files for this datathe h5 files for this dataset.
            
            # batchinfo contains some info about the batch, it also has file/row information.
            # files are integers that are relative to the h5files that the iter returns above.
            
            h5file = h5files[meta['file'][0]]
            print("label=%d step=%d epoch=%d readtime=%4f fiducials=%d seconds=%d nano=%d\n  run=%d run.index=%d h5file=%s row=%s\n" %
                  (label,
                   batchinfo['step'],
                   batchinfo['epoch'],
                                batchinfo['readtime'],
                                          meta['evt.fiducials'][0],    # meta is a numpy array, index 0 for first of batch
                                               meta['evt.seconds'][0],
                                                      meta['evt.nanoseconds'][0],
                                                             meta['run'][0],
                                                                            meta['run.index'][0],
                                                                                                    h5file, meta['row'][0]))
            prep_batch = psmlearn.util.log_thresh(img_batch, logthresh)
            psmlearn.plot.compareImages(pipeline.plt, plotFigH,
                                        ("orig label=%d" % label, img_batch[0]),
                                        ("log thresh=%.1f" % logthresh, prep_batch[0]), colorbars=True)
            if pipeline.stop_plots():
                break
    
    def train(self, config, pipeline, step2h5list, output_files):
        '''train a model on the data. 
        Takes the typical arguments for a psmlearn.pipeline step:

        config - this is a namedtuple of configuration. 
                 pipeline configuration is read from a optional
                 yaml file and the command line. Command line takes
                 precedence.

        pipeline - reference to the pipeline object.

        step2h5list - a dict, keys are the previous step names and values are list of output files.
                      This is how a step reads output from previous layers.

        output_files - this is a list of the output files this step produces.
                       pipeline will check to make sure each output file has been created
                       by the step, if not an error is produced. 
                      
        The features of pipeline are management of 
        * output files
        * configuration
        for different runs, the user just specifies the prefix for the run.
        '''
        # train takes the typical arguments for a pipeline 
        tic_all = time.time()
        saved_model = output_files[0].rsplit('.h5',1)[0] + '.tfmodel'

        train_iter = self.dset.train_iter(batchsize=config.batchsize)

        read_time = 0.0
        prep_time = 0.0
        train_time = 0.0
        validation_time = 0.0
        sess = pipeline.session
        best_acc = 0.0
        saved_models = {}
        
        for X,Y,meta,batchinfo in train_iter:
            if batchinfo['step'] > config.maxsteps:
                print("training has reached %d steps, stopping" % batchinfo['step'])
                break

            read_time += batchinfo['readtime']

            tic = time.time()
            imgs = prep_imgs(imgs=X[0], logthresh=config.logthresh, expanddims=3)
            prep_time += time.time()-tic

            tic = time.time()
            labels = Y[0]
            sess.run(self.train_op, feed_dict={self.imgs_pl:imgs, self.labels_pl:labels})
            train_time += time.time()-tic

            tic = time.time()
            if (batchinfo['step'] % config.validation == 0) and (batchinfo['step']>0):
                arr_logits, loss_val = sess.run([self.logits, self.loss],
                                                feed_dict={self.imgs_pl:imgs, self.labels_pl:labels})
                train_cmat = psmlearn.util.get_confusion_matrix_one_hot(arr_logits, labels)
                valid_iter = self.dset.validation_iter(batchsize=config.batchsize, epochs=1)
                valid_cmat = get_validation_confusion_matrix(valid_iter=valid_iter,
                                                             logthresh=config.logthresh,
                                                             sess=sess,
                                                             logits=self.logits,
                                                             imgs_pl=self.imgs_pl)
                train_acc, train_cmat_rows = psmlearn.util.cmat2str(train_cmat)
                valid_acc, valid_cmat_rows = psmlearn.util.cmat2str(valid_cmat)

                saved = False
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    saved = True
                    saved_models[self.saver.save(sess, saved_model)] = valid_acc
                msg = "step=%4d train.acc=%5.2f valid.acc=%5.2f loss=%5.2e train.cmat="
                msg %= (batchinfo['step'], train_acc, valid_acc, loss_val)
                N = len(msg)
                msg += '%s valid.cmat=%s' % (train_cmat_rows.pop(0), valid_cmat_rows.pop(0))
                if saved:
                    msg += ' ** saved'
                print(msg)
                while len(train_cmat_rows):
                    print((' '*N) + "%s            %s" % (train_cmat_rows.pop(0), valid_cmat_rows.pop(0)))
            validation_time += time.time()-tic
            
        output_fname = output_files[0]
        h5=h5py.File(output_fname,'w')
        models = h5.create_group('models')
        psmlearn.h5util.write_to_h5(h5, saved_models)
        time_all = time.time()-tic_all
        psmlearn.h5util.write_config(h5, config)
        psmlearn.h5util.write_to_h5(h5, {'read_time':read_time,
                                         'prep_time':prep_time,
                                         'train_time':train_time,
                                         'validation_time':validation_time,
                                         'all_time':time_all,
                                         })
        try:
            h5['validation_confusion_matrix']=valid_cmat
        except NameError:
            pass
                                       
        h5.close()
        print("-- training done. times:")
        print("-- read_time: %.1f sec" % read_time)
        print("-- prep_time: %.1f sec" % prep_time)
        print("-- train_time: %.1f sec" % train_time)
        print("-- validation_time: %.1f sec" % validation_time)
        print("-- all_time: %.1f sec" % time_all)
        
if __name__ == '__main__':
    mysteps = MySteps()
    pipeline = Pipeline(stepImpl=mysteps,
                        outputdir='.')        # can be overriden with command line arguments

    # you can add your own command line arguments however you like by modifying
    # pipeline.parser (you can't override pipeline args, accidentally using an existing argument
    # generates a Python exception)
    mysteps.add_commandline_arguments(pipeline.parser)
    pipeline.add_step_method_plot(name='view')
    pipeline.add_step_method(name='train')

    # run will do the following
    #  parse command line arguments
    #  read config file (if given)
    #  setup tensorflow gpu device placement (if specified)
    #  call mysteps.init()
    #  execute all steps that need to be done
    pipeline.run()
