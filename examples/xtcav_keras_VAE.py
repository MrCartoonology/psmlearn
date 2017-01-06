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
from scipy.stats import norm

#from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Convolution2D
from keras.layers import BatchNormalization, MaxPooling2D, UpSampling2D
from keras import objectives
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

##############
def find_images_to_interpolate(starts, labels):
    start_median = np.median(starts)
    similar_starts = np.logical_and(starts > start_median - 0.5, starts < start_median + 0.5)
    assert np.sum(similar_starts)>0

    nolasing = np.where(np.logical_and(labels == 0, similar_starts))[0]
    assert len(nolasing)>0

    lasing = np.where(np.logical_and(labels == 1, similar_starts))[0]
    assert len(lasing)>0

    lasing_idx = lasing[np.random.randint(len(lasing))] 
    nolasing_idx = nolasing[np.random.randint(len(nolasing))] 

    return nolasing_idx, lasing_idx

def sampling(args, **kwargs):
    z_mean, z_log_var = args
    batch_size = kwargs.pop('batch_size')
    latent_dim = kwargs.pop('latent_dim')
    epsilon_std = kwargs.pop('epsilon_std')
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2.0) * epsilon

###################
class Keras_VAE_Dense(object):
    def __init__(self, imgH, imgW):
        self.batch_size = 100
        self.original_dim = imgH*imgW
        self.latent_dim = 2
        self.intermediate_dim = 256
        self.nb_epoch = 50
        self.epsilon_std = 1.0
        self.encoding_dim = 32

        self.input_img = Input(batch_shape=(self.batch_size, self.original_dim,))
        self.hidden = Dense(self.intermediate_dim, activation='relu')(self.input_img)
        self.z_mean = Dense(self.latent_dim)(self.hidden)
        self.z_log_var = Dense(self.latent_dim)(self.hidden)

        self.z = Lambda(sampling, 
                        arguments={'batch_size': self.batch_size, 
                                   'latent_dim': self.latent_dim,
                                   'epsilon_std': self.epsilon_std}) \
            ([self.z_mean,self.z_log_var])
        
        self.decoder_h = Dense(self.intermediate_dim, activation='relu')
        self.decoder_mean = Dense(self.original_dim, activation='sigmoid')
        self.h_decoded = self.decoder_h(self.z)
        self.img_decoded_mean = self.decoder_mean(self.h_decoded)

        self.VAE = Model(self.input_img, self.img_decoded_mean)

    def vae_loss(self, x, y):
        xent_loss = self.original_dim * objectives.binary_crossentropy(x,y)
        k1_loss = -0.5 * K.sum( 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return xent_loss + k1_loss

    
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
        pass


    def prep(self, config, pipeline, step2h5list, output_files):
        '''links to dense output
        '''
        assert len(output_files)==3
        outputdir=pipeline.outputdir
        old_prefix='xtcav_autoencoder_dense'
        new_prefix = config.prefix
        old_files = pipeline._step_output_files(outputdir=outputdir,
                                                prefix=old_prefix,
                                                stepname='prep',
                                                output_files=['_train','_validation', '_test'])
        for oldfile, newfile in zip(old_files, output_files):
            assert os.path.exists(oldfile), "oldfile %s doesn't exist" % oldfile
            if not os.path.exists(newfile):
                os.symlink(oldfile, newfile)

    def fit(self, config, pipeline, step2h5list, output_files):
        h5train, h5validation, h5test = step2h5list['prep']

        h5 = h5py.File(h5train,'r')
        train_imgs = h5['imgs'][:]
        img_W = h5['config']['reduced_w'].value
        img_H = h5['config']['reduced_h'].value
        h5.close()

        h5 = h5py.File(h5validation,'r')
        validation_imgs = h5['imgs'][:]

        ML = Keras_VAE_Dense(img_H, img_W)
        ML.VAE.summary()
        ML.VAE.compile(optimizer='rmsprop', loss=ML.vae_loss)

        # looks like fit will send through the last batch as a partial batches, and we have
        # built VAE to use batches of 100, so cut samples

        trainN = len(train_imgs) - (len(train_imgs) % ML.batch_size)
        validN = len(validation_imgs) - (len(validation_imgs) % ML.batch_size)
        ML.VAE.fit(train_imgs[0:trainN], train_imgs[0:trainN], 
                   nb_epoch=500,
                   batch_size=ML.batch_size,
                   shuffle=True,
                   validation_data=(validation_imgs[0:validN], validation_imgs[0:validN]))
        ML.VAE.save(output_files[0])

    ## not a step, called by other plot steps
    def view_trained_model(self, trained_model_file, plot, pipeline, plotFigH, config, step2h5list):
        # get the data that we trained on, as well as new validation data
        h5train, h5validation, h5test = step2h5list['prep']

        h5 = h5py.File(h5validation,'r')
        img_W = h5['config']['reduced_w'].value
        img_H = h5['config']['reduced_h'].value

        # read in the model and load the trained weights
        ML = Keras_VAE_Dense(img_H, img_W)
        ML.VAE.load_weights(trained_model_file)
        
        imgs = h5['imgs'][0:ML.batch_size]
        gasdet = h5['gasdet'][0:ML.batch_size]
        labels = h5['labels'][0:ML.batch_size]

        img_W = h5['config']['reduced_w'].value
        h5.close()

        encoded_imgs = ML.VAE.predict(imgs, batch_size=ML.batch_size)

        plt = pipeline.plt
        assert plt

        idxs= range(len(imgs))
        random.shuffle(idxs)
        
        plt.figure()
        for idx in idxs:
            orig_img = np.reshape(imgs[idx],(img_H, img_W))
            encoded_img = np.reshape(encoded_imgs[idx],(img_H, img_W))
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


    def view_fit(self, plot, pipeline, plotFigH, config, step2h5list):
        trained_model_file = step2h5list['fit'][0]
        self.view_trained_model(trained_model_file, plot, pipeline, plotFigH, config, step2h5list)

    
######################################
if __name__ == '__main__':
    mysteps = MySteps()
    pipeline = Pipeline(stepImpl = mysteps,
                        session = K.get_session(),
                        defprefix='xtcav_VAE',
                        outputdir='/scratch/davidsch/dataprep')        # can be overriden with command line arguments
    mysteps.add_commandline_arguments(pipeline.parser)
    pipeline.add_step_method(name='prep', output_files=['_train','_validation', '_test'])
    pipeline.add_step_method(name='fit')
    pipeline.add_step_method_plot(name='view_fit')
    
    pipeline.run()
