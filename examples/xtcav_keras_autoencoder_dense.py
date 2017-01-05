from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# system imports
import os
import sys
import time
import glob
import random

# external 
import numpy as np
import h5py
from scipy.misc import imresize

#from keras import regularizers
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.models import Model
import keras.backend as K

# internal
from h5batchreader import H5BatchReader, DataSetGroup
import psmlearn
from psmlearn.pipeline import Pipeline

__doc__='''
we want to get better labels of the xtcav.

We'll work with the amo86815 data, but now we'll work with runs 69, 70,71, and 72,73, the
molecular runs. We'll do a 0/1 classifier, but use the gasdet to identify lasing, and we'll
only train on a subset of the data, threshold on gasdet. Maybe this high gasdet threshold will
be enough to sort down the data as well?

'''

def sampling(args):
    z_mean, z_log_var, batchsize, latent_dim, epsilon_std = args
    epsilon = K.random_normal(shape=(batchsize, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2.0) * epsilon

class Keras_Autoencoder_Dense(object):
    def __init__(self, flattened_num_img_pixels):

        self.encoding_dim = 32

        self.input_img = Input(shape=(flattened_num_img_pixels,))

        # weird, kears example uses 10e-5, but I just get a big blob. There are 5000 pixels in our images,
        # and only 784 in MNIST? 6.3 times more? Or do I just need to train more?
        # took out regulizer, got a better loss, 0.5964, but still a big mean like blob
        self.encoded = Dense(self.encoding_dim, activation='relu')(self.input_img)
#                             activity_regularizer=regularizers.activity_l1(10e-7))(self.input_img)
        self.decoded = Dense(flattened_num_img_pixels, activation='sigmoid')(self.encoded)

        self.autoencoder = Model(input=self.input_img, output=self.decoded)

        self.encoder = Model(input=self.input_img, output=self.decoded)

        # create a placeholder for an encoded (32-dimensional) input
        self.encoded_input = Input(shape=(self.encoding_dim,))

        # retrieve the last layer of the autoencoder model
        self.decoder_layer = self.autoencoder.layers[-1]

        # create the decoder model
        self.decoder = Model(input=self.encoded_input, 
                             output=self.decoder_layer(self.encoded_input))

    
class Keras_VAE(object):
    def __init__(self, batch_shape):
        xx = Input(batch_shape=(batch_shape))
        kernel_size=9
        kernel_ch=4
        self.latent_dim = 2
        conv = Convolution2D(kernel_size, kernel_size, kernel_ch, border_mode='same', activation='relu')(xx)
        hidden = MaxPooling2D(pool_size=(kernel_size, kernel_size))(conv)
        conv = Convolution2D(kernel_size, kernel_size, kernel_ch, border_mode='same', activation='relu')(hidden)
        hidden = Flatten(MaxPooling2D(pool_size=(kernel_size, kernel_size))(conv))
        z_mean = Dense(self.latent_dim)(hidden)
        z_log_var = Dense(self.latent_dim)(hidden)
    
        self.batchsize = batch_shape[0]
        self.epsilon_std = 1.0
        self.z_mean = z_mean
        self.z_log_var = z_log_var

        self.z = Lambda(sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var, self.batchsize, self.latent_dim, self.epsilon_std])

    
#################
class MySteps():
    def __init__(self):
        pass
    
    def add_commandline_arguments(self, parser):
        parser.add_argument('--las', type=float,
                            help='label shot as lasing if gasdet is above this value, '
                                  'default is 0.07, where we see a lot of double lasing',
                            default=0.07)
        parser.add_argument('--nolas', type=float,
                            help='label shot as no lasing if gasdet is below this value, '
                                 'default 0.017, we dont see much lasing at that level',
                            default=0.0017)
        parser.add_argument('--logthresh', type=float,
                            help='log threshold default=220.0, ',
                            default=220.0)
        parser.add_argument('--sigwin', type=int,
                            help='width of signal window, vproj def 232, ',
                            default=232)
        parser.add_argument('--reduced_w', type=int,
                            help='reduced width, default=50',
                            default=50)
        parser.add_argument('--reduced_h', type=int,
                            help='reduced height, default=100',
                            default=100)
        parser.add_argument('--trainsteps', type=int,
                            help='number of training steps, default=2500',
                            default=2500)
        parser.add_argument('--datadir', type=str,
                            help='where the hdf5 files for training/testing are',
                            default='/scratch/davidsch/psmlearn/xtcav/default_subproject/hdf5')

    def init(self, config, pipeline):
        assert os.path.exists(config.datadir), "path %s doesn't exist" % config.datadir
        h5files = glob.glob(os.path.join(config.datadir, "amo86815*.h5"))
        if config.dev:
            random.shuffle(h5files)
            h5files = h5files[0:10]
        gasdet = DataSetGroup(name='gasdet', dsets=['bld.gasdet.f_11_ENRC',
                                                    'bld.gasdet.f_12_ENRC',
                                                    'bld.gasdet.f_21_ENRC',
                                                    'bld.gasdet.f_22_ENRC'])        
        self.hbr = H5BatchReader(h5files,
                                 dset_groups=[gasdet],
                                 dsets=['xtcavimg', 'lasing', 'run', ])

        self.hbr.split(train=90, validation=5, test=5)


    def view_nolas(self, plot, pipeline, plotFigH, config, step2h5list):
        # any arguments on the command line are named variables in the config passed to each step
        logthresh = config.logthresh

        batchsize=128
        basic_iter = self.hbr.train_iter(batchsize=batchsize, epochs=1)

        plt = pipeline.plt
        plt.figure(plotFigH)
        plt.ion()
        
        for batch in basic_iter:
            img_batch = batch['dsets']['xtcavimg'].astype(np.float32)
            gasdet = batch['dset_groups']['gasdet']
            run = batch['dsets']['run']
            agg_gasdet = np.mean(gasdet,axis=1)
            nolas = np.logical_or(run==69, agg_gasdet < config.nolas)
            las = np.logical_and(run>69, agg_gasdet > config.las)            
            include = np.logical_or(nolas, las)
            
            scratch = agg_gasdet.copy()
            scratch[np.logical_not(nolas)]=0.0
            max_of_nolas_row = np.argmax(scratch)
            max_of_nolas =  np.max(scratch)

            scratch = agg_gasdet.copy()
            scratch[np.logical_not(las)]=9999.0
            min_of_las_row = np.argmin(scratch)
            min_of_las =  np.min(scratch)

            prep_nolas = psmlearn.util.log_thresh(img_batch[max_of_nolas_row], config.logthresh)
            prep_las = psmlearn.util.log_thresh(img_batch[min_of_las_row], config.logthresh)

            psmlearn.plot.compareImages(plt, plotFigH,
                                       ("max of nolas bld=%.5f" % max_of_nolas, prep_nolas),
                                       ("min of las bld=%.5f" % min_of_las, prep_las),
                                       colorbars=True)
            if pipeline.stop_plots():
                break

    def label_stats(self, config, pipeline, step2h5list, output_files):
        '''This step went through 314k samples, the filtering kept 133k,
        so we are throwing out 2/3 of the data. The ratios are 
        lasing = 111k and nolasing = 23k.
        '''
        logthresh = config.logthresh
        batchsize=1024
        basic_iter = self.hbr.train_iter(batchsize=batchsize, epochs=1)

        totals={'las':0,'nolas':0,'all':0,'include':0}
        
        for batch in basic_iter:
#            img_batch = batch['dsets']['xtcavimg'].astype(np.float32)
            gasdet = batch['dset_groups']['gasdet']
            run = batch['dsets']['run']
            agg_gasdet = np.mean(gasdet,axis=1)
            nolas = np.logical_or(run==69, agg_gasdet < config.nolas)
            las = np.logical_and(run>69, agg_gasdet > config.las)            
            include = np.logical_or(nolas, las)
            totals['las'] += np.sum(las)
            totals['nolas'] += np.sum(nolas)
            totals['include'] += np.sum(include)
            totals['all'] += batch['size']
            pipeline.trace("batch %d: totals=%r" % (batch['step'], totals))
        h5 = h5py.File(output_files[0],'w')
        psmlearn.h5util.write_to_h5(h5, totals)
        psmlearn.h5util.write_config(h5, config)

    def prep(self, config, pipeline, step2h5list, output_files):
        '''goes through all the data
        stores the reduced images in memory
        writes it all out to the h5 files
        '''
        assert len(output_files)==3
        orig_H, orig_W = (726, 568)
        signal_window_width = config.sigwin
        reduced_W = config.reduced_w
        reduced_H = config.reduced_h
        mx = config.logthresh + 12.0
        batchsize=32
        MXROWS = 150123
        for partition, output_file, iterfn in zip(['train','validation','test'],
                                                  output_files,
                                                  [self.hbr.train_iter, 
                                                   self.hbr.validation_iter, 
                                                   self.hbr.test_iter]):
            print("===========\n%s" % output_file)
            dataiter = iterfn(batchsize=batchsize, epochs=1)

            arrays_all = {'imgs':None,
                          'gasdet':None,
                          'labels_onehot':None,
                          'labels':None,
                          'starts':None}
            row = 0
            for batch in dataiter:
                t0 = time.time()
                prep_img, gasdet, labels_onehot, labels = filter_and_label_batch(batch, config)
                starts = psmlearn.util.start_signal_window_batch(prep_img, signal_window_width, 'vproj','tf')
                prep_img = psmlearn.util.extract_signal_window_batch(starts, prep_img, signal_window_width, 'vproj', 'tf')
                ## prep_img is NN x 726 x 232 x 1
                prep_img = reduce_and_flatten(prep_img, (reduced_H, reduced_W))
                prep_img /= mx
                arrays_batch = {'imgs':prep_img, 'gasdet':gasdet, 
                                'labels_onehot':labels_onehot, 
                                'labels':labels, 'starts':starts}
                if arrays_all['imgs'] is None:
                    for ky in arrays_all:
                        arrays_all[ky] = arrays_batch[ky].copy()
                else:
                    for ky in arrays_all:
                        old_shape = list(arrays_all[ky].shape)
                        start = old_shape[0]
                        new_shape = [xx for xx in old_shape]
                        new_shape[0] += arrays_batch[ky].shape[0]
                        arrays_all[ky].resize(new_shape, refcheck=False)
                        arrays_all[ky][start:] = arrays_batch[ky]

                batchtime = batch['readtime'] + (time.time()-t0)
                t0 = time.time()
                print("step=%5d batchtime=%.2f" % (batch['step'], batchtime))

            h5 = h5py.File(output_file,'w')
            psmlearn.h5util.write_config(h5, config)
            for ky in arrays_all:
                h5[ky] = arrays_all[ky]
            h5.close()

    def train(self, config, pipeline, step2h5list, output_files):
        h5train, h5validation, h5test = step2h5list['prep']

        h5 = h5py.File(h5train,'r')
        reduced_W = h5['config']['reduced_w'].value
        reduced_H = h5['config']['reduced_h'].value
        h5.close()

        data_augment_offset = 0 # 2
        augment_W = reduced_W - data_augment_offset
        augment_H = reduced_H

        ML = Keras_Autoencoder_Dense(flattened_num_img_pixels = augment_H * augment_W)
        ML.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        
        trainbr = H5BatchReader([h5train], 
                            dsets=['imgs','gasdet','labels','labels_onehot','starts'])
        validbr = H5BatchReader([h5validation], 
                            dsets=['imgs','gasdet','labels','labels_onehot','starts'])
        trainbr.split(train=100, validation=0, test=0)
        validbr.split(train=0, validation=100, test=0)

        batchsize = 512
        train_iter = trainbr.train_iter(batchsize=batchsize)

        num_batches = config.trainsteps
        steps_per_batch = 3
        best = 99999
        lastsave = 0
        for batch in train_iter:
            t0 = time.time()
            imgs = batch['dsets']['imgs']
            gasdet = batch['dsets']['gasdet']
            labels = batch['dsets']['labels']
            starts = batch['dsets']['starts']
            labels_onehot = batch['dsets']['labels_onehot']
            batchtime = batch['readtime'] + (time.time()-t0)
            t0 = time.time()
            for step in range(steps_per_batch):
                loss = ML.autoencoder.train_on_batch(imgs, imgs)
            traintime = time.time()-t0
            xtra = ''
            if loss < best and batch['step']-lastsave > 25:
                best = loss
                lastsvae =  batch['step']
                ML.autoencoder.save(output_files[0])
                xtra = ' *saved*'
            print("step=%5d loss=%8.3f batchtime=%.2f traintime=%.2f %s" % 
                  (batch['step'], loss, batchtime, traintime, xtra))
            
            if batch['step'] > num_batches: break

    ## not a step, called by other plot steps
    def view_trained_model(self, trained_model_file, plot, pipeline, plotFigH, config, step2h5list):
        # get the data that we trained on, as well as new validation data
        h5train, h5validation, h5test = step2h5list['prep']

        # read in the data to look at it
        h5 = h5py.File(h5validation,'r')
        imgs = h5['imgs'][:]
        gasdet = h5['gasdet'][:]
        labels = h5['labels'][:]

        reduced_W = h5['config']['reduced_w'].value
        reduced_H = h5['config']['reduced_h'].value
        h5.close()

        # read in the model and load the trained weights
        ML = Keras_Autoencoder_Dense(flattened_num_img_pixels = reduced_H * reduced_W)
        ML.autoencoder.load_weights(trained_model_file)

        encoded_imgs = ML.autoencoder.predict(imgs)
        plt = pipeline.plt
        
        idxs= range(len(imgs))
        random.shuffle(idxs)
        
        plt.figure()
        for idx in idxs:
            orig_img = np.reshape(imgs[idx],(reduced_H, reduced_W))
            encoded_img = np.reshape(encoded_imgs[idx],(reduced_H, reduced_W))
            diff = orig_img - encoded_img
            vmin = min(np.min(orig_img), np.min(encoded_img))
            vmax = max(np.max(orig_img), np.max(encoded_img))

            plt.clf()
            plt.subplot(1,3,1)
            plt.cla()
            plt.imshow(orig_img, vmin=vmin, vmax=vmax, interpolation='none')
            plt.title("orig lab=%d gd=%.2f" % (labels[idx], np.mean(gasdet[idx])))

            plt.subplot(1,3,2)
            plt.cla()
            plt.imshow(encoded_img, vmin=vmin, vmax=vmax, interpolation='none')
            plt.title("encoded")

            plt.subplot(1,3,3)
            plt.cla()
            plt.imshow(diff, interpolation='none')
            plt.colorbar()

            if pipeline.stop_plots(): 
                break


    def view_train(self, plot, pipeline, plotFigH, config, step2h5list):
        trained_model_file = step2h5list['train'][0]
        self.view_trained_model(trained_model_file, plot, pipeline, plotFigH, config, step2h5list)

    def view_fit(self, plot, pipeline, plotFigH, config, step2h5list):
        trained_model_file = step2h5list['fit'][0]
        self.view_trained_model(trained_model_file, plot, pipeline, plotFigH, config, step2h5list)

    def fit(self, config, pipeline, step2h5list, output_files):
        h5train, h5validation, h5test = step2h5list['prep']

        h5 = h5py.File(h5train,'r')
        train_imgs = h5['imgs'][:]
        reduced_W = h5['config']['reduced_w'].value
        reduced_H = h5['config']['reduced_h'].value
        h5.close()

        h5 = h5py.File(h5validation,'r')
        validation_imgs = h5['imgs'][:]

        ML = Keras_Autoencoder_Dense(flattened_num_img_pixels = reduced_H * reduced_W)
        ML.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        ML.autoencoder.fit(train_imgs, train_imgs, nb_epoch=50,
                           batch_size=512,
                           shuffle=True,
                           validation_data=(validation_imgs, validation_imgs))
        ML.autoencoder.save(output_files[0])

        
################
def reduce_and_flatten(imgs, newshape):
    assert len(newshape)==2
    assert len(imgs.shape)==4
    assert imgs.shape[3]==1
    LL =newshape[0]*newshape[1]
    NN = imgs.shape[0]
    new_imgs = np.empty((NN,LL), dtype=imgs.dtype)
    for idx in range(NN):
        new_imgs[idx,:] = imresize(imgs[idx,:,:,0], newshape).flatten()[:]
    return new_imgs

def filter_and_label_batch(batch, config):
    '''takes batch of data with images and gasdet, filters, labels and pre-processes images.

    ARGS:
      batch - dict with
        batch['dsets']['xtcavimg'] = the images
        batch['dset_groups']['gasdet']  gasdet values
        batch['run']  run - run 69 is always nolasing

      config - namedtuple with
        las
        nolas   how to label the images in runs > 69, anything inbetween nolas and las
                is thrown out
      doprep - True to preprocess
    RETURNS: 
  
      prep_img     logthresh applied, and channel added
      gasdet        gasdet values for the filtered images
      labels_onehot
      labels
    '''
    img_batch = batch['dsets']['xtcavimg'].astype(np.float32)
    gasdet = batch['dset_groups']['gasdet']
    run = batch['dsets']['run']

    labels_onehot = np.zeros((batch['size'],2), np.int32)
    labels = np.zeros(batch['size'], np.int32)

    agg_gasdet = np.mean(gasdet, axis=1)
    nolas = np.logical_or(run==69, agg_gasdet < config.nolas)
    las = np.logical_and(run>69, agg_gasdet > config.las)
    labels_onehot[nolas,0]=1
    labels_onehot[las,1]=1
    labels[las]=1
    include = np.logical_or(las,nolas)

    img_batch = img_batch[include]
    labels = labels[include]
    labels_onehot=labels_onehot[include]
    gasdet = gasdet[include]
    
    prep_batch = psmlearn.util.log_thresh(img_batch, thresh=config.logthresh)
    shape = prep_batch.shape
    ch_shape = tuple(list(shape)+[1])
    prep_img = np.resize(prep_batch, ch_shape)

    return prep_img, gasdet, labels_onehot, labels

def data_augment(prep_img, gasdet, labels_onehot, labels, batchsize, offset_max=5):
    '''creates a batch of batchsize.
    offsets each image by random amount.

    sets 1/2 of needed to random from nolasing, and 1/2 from lasing.
    '''
    N = len(prep_img)
    for nn in [len(gasdet), len(labels_onehot), len(labels)]:
        assert N == nn
    assert batchsize >= N
    toadd = batchsize - N
    img_offsets = np.random.randint(0,offset_max,batchsize)
    cols = prep_img.shape[2] - offset_max

    augmented_batch = {}
    augmentzip = zip(['prep_img', 'gasdet', 'labels_onehot', 'labels'],
                     [ prep_img,   gasdet,   labels_onehot,   labels])

    for name, orig in augmentzip:
        shape = list(orig.shape)
        shape[0]=batchsize
        if name == 'prep_img':
            shape[2] = cols
        augmented = np.empty(tuple(shape), dtype=orig.dtype)
        if name == 'prep_img':            
            for ii in range(N):
                offset = img_offsets[ii]
                augmented[ii] = orig[ii,:,offset:(offset+cols),:]
        else:
            augmented[0:N]=orig
        augmented_batch[name]=augmented

    lasing_rows = np.where(labels == 1)[0]
    nolasing_rows = np.where(labels == 0)[0]

    if len(nolasing_rows)>0:
        toadd_nolas = toadd//2
        nolas_idx = np.random.randint(0, len(nolasing_rows), toadd_nolas)
        orig_rows = nolasing_rows[nolas_idx]
        for name, orig in augmentzip:
            if name == 'prep_img':
                for ii, row in zip(range(N, N+toadd_nolas), orig_rows):
                    offset = img_offsets[ii]
                    augmented_batch[name][ii] = orig[row,:,offset:(offset+cols),:]
            else:
                augmented_batch[name][N:(N+toadd_nolas)] = orig[orig_rows]
        N += toadd_nolas

    if N < batchsize:
        toadd_las = batchsize - N
        las_idx = np.random.randint(0, len(lasing_rows), toadd_las)
        orig_rows = lasing_rows[las_idx]
        for name, orig in augmentzip:
            if name == 'prep_img':
                for ii, row in zip(range(N, N+toadd_las), orig_rows):
                    offset = img_offsets[ii]
                    augmented_batch[name][ii] = orig[row,:,offset:(offset+cols),:]
            else:
                augmented_batch[name][N:(N+toadd_las)] = orig[orig_rows]
        
    return augmented_batch['prep_img'], augmented_batch['gasdet'], augmented_batch['labels_onehot'], augmented_batch['labels']

    
######################################
if __name__ == '__main__':
    mysteps = MySteps()
    pipeline = Pipeline(stepImpl = mysteps,
                        session = K.get_session(),
                        defprefix='xtcav_autoencoder_dense',
                        outputdir='/scratch/davidsch/dataprep')        # can be overriden with command line arguments
    mysteps.add_commandline_arguments(pipeline.parser)
    pipeline.add_step_method_plot(name='view_nolas')
#    pipeline.add_step_method(name='label_stats')
    pipeline.add_step_method(name='prep', output_files=['_train','_validation', '_test'])
    pipeline.add_step_method(name='train', output_files=['_model'])
    pipeline.add_step_method_plot(name='view_train')
    pipeline.add_step_method(name='fit')
    pipeline.add_step_method_plot(name='view_fit')
    
    pipeline.run()
