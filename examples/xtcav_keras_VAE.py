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
from keras.layers import Input, Dense, Lambda, Convolution2D, Flatten, Reshape
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
    def dataprep(self, imgs):
        return imgs
    def __init__(self, imgH, imgW):
        self.batch_size = 100
        self.original_dim = imgH*imgW
        self.latent_dim = 2
        self.intermediate_dim = 256
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

        self.encoder = Model(self.input_img, self.z_mean)
        
        self.decoder_input = Input(shape=(self.latent_dim,))
        _h_decoded = self.decoder_h(self.decoder_input)
        _img_decoded_mean = self.decoder_mean(_h_decoded)
        self.generator = Model(self.decoder_input, _img_decoded_mean)

    def vae_loss(self, x, y):
        xent_loss = self.original_dim * objectives.binary_crossentropy(x,y)
        k1_loss = -0.5 * K.sum( 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return xent_loss + k1_loss

class Keras_VAE_CNN(object):
    def dataprep(self, imgs):
        NN = len(imgs)
        assert imgs.shape==(NN,5000)
        X = np.zeros((NN,108,54,1), dtype=imgs.dtype)
        for idx in range(NN):
            X[idx,4:104,2:52,0]=imgs[idx].reshape((100,50))
        return X
    
    def __init__(self, img_H, img_W):
        assert img_H == 100
        assert img_W == 50
        self.original_dim = 5000
        img_H += 8
        img_W += 4

        self.batch_size = 100
        self.latent_dim = 2
        self.epsilon_std = 1.0

        self.input_img = Input(batch_shape=(self.batch_size, img_H, img_W, 1))

        x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(self.input_img)
        # (108, 54, 16)
        
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        # (54, 27, 16)
        
        x = Convolution2D(8, 5, 5, activation='relu', border_mode='same')(x)
        # (54, 27, 8)
        
        x = MaxPooling2D((3, 3), border_mode='same')(x)
        # (18, 9, 8)
        
        x = Convolution2D(4, 7, 5, activation='relu', border_mode='same')(x)
        # (18, 9, 4 )
        
        x = MaxPooling2D((3, 3), border_mode='same')(x)
        # (6, 3, 4)

        self.hidden = Flatten()(x)
        
        self.z_mean = Dense(self.latent_dim)(self.hidden)
        self.z_log_var = Dense(self.latent_dim)(self.hidden)

        
        self.z = Lambda(sampling, 
                        arguments={'batch_size': self.batch_size, 
                                   'latent_dim': self.latent_dim,
                                   'epsilon_std': self.epsilon_std}) \
                                   ([self.z_mean,self.z_log_var])

        self.encoder_layers = []
        self.encoder_layers.append(Dense(6*3*4, activation='relu'))
        self.encoder_layers.append(Reshape((6,3,4)))
        # (6, 3, 4)

        self.encoder_layers.append(Convolution2D(4, 5, 5, activation='relu', border_mode='same'))
        # (6, 3, 4)

        self.encoder_layers.append(UpSampling2D((3,3)))
        # (18, 9, 4)

        self.encoder_layers.append(Convolution2D(8,5,5, activation='relu', border_mode='same'))
        # (18, 9, 8)

        self.encoder_layers.append(UpSampling2D((3,3)))
        # (54, 27, 8)

        self.encoder_layers.append(Convolution2D(16,3,3, activation='relu', border_mode='same'))
        # (54, 27, 16)

        self.encoder_layers.append(UpSampling2D((2,2)))
        # (108,94,16)

        self.encoder_layers.append(Convolution2D(1,3,3, activation='sigmoid', border_mode='same'))

        x = self.z
        for layer in self.encoder_layers:
            x = layer(x)
        self.img_decoded_mean = x
        self.VAE = Model(self.input_img, self.img_decoded_mean)

        self.encoder = Model(self.input_img, self.z_mean)

        self.decoder_input = Input(shape=(self.latent_dim,))
        x = self.decoder_input
        for layer in self.encoder_layers:
            x = layer(x)
        self.generator = Model(self.decoder_input, x)
        
    def vae_loss(self, x, y):
        xent_loss = self.original_dim * objectives.binary_crossentropy(Flatten()(x),Flatten()(y))
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
        parser.add_argument('--cnn', action='store_true',
                            help='use the cnn VAE')
        parser.add_argument('--logthresh', type=float,
                            help='log threshold default=220.0, ',
                            default=220.0)
        parser.add_argument('--nb', type=int,
                            help='number of epochs',
                            default=50),
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

        if config.cnn:
            ML = Keras_VAE_CNN(img_H, img_W)
        else:
            ML = Keras_VAE_Dense(img_H, img_W)
        ML.VAE.summary()
        ML.VAE.compile(optimizer='Adam', loss=ML.vae_loss)

        # looks like fit will send through the last batch as a partial batches, and we have
        # built VAE to use batches of 100, so cut samples

        trainN = len(train_imgs) - (len(train_imgs) % ML.batch_size)
        validN = len(validation_imgs) - (len(validation_imgs) % ML.batch_size)
        train_imgs = ML.dataprep(train_imgs[0:trainN])
        validation_imgs = ML.dataprep(validation_imgs[0:validN])

#        import IPython
#        IPython.embed()
        ML.VAE.fit(train_imgs, train_imgs,
                   nb_epoch=config.nb,
                   batch_size=ML.batch_size,
                   shuffle=True,
                   validation_data=(validation_imgs, validation_imgs))
        ML.VAE.save(output_files[0])

    ## not a step, called by other plot steps
    def view_trained_model(self, trained_model_file, plot, pipeline, plotFigH, config, step2h5list):
        # get the data that we trained on, as well as new validation data
        h5train, h5validation, h5test = step2h5list['prep']

        h5 = h5py.File(h5validation,'r')
        img_W = h5['config']['reduced_w'].value
        img_H = h5['config']['reduced_h'].value

        # read in the model and load the trained weights
        if config.cnn:
            ML = Keras_VAE_CNN(img_H, img_W)
        else:
            ML = Keras_VAE_Dense(img_H, img_W)
        ML.VAE.load_weights(trained_model_file)

        # read in a multiple of batch_size imgs, labels, etc
        NN = len(h5['imgs'])
        NN = NN - (NN % ML.batch_size)
        imgs = ML.dataprep(h5['imgs'][0:NN])
        gasdet = h5['gasdet'][0:NN]
        labels = h5['labels'][0:NN]
        h5.close()

        plt = pipeline.plt
        assert plt

        # latent space plot
        imgs_encoded = ML.encoder.predict(imgs, batch_size=ML.batch_size)
        colors = labels.astype(np.float)
        plt.figure(plotFigH, figsize=(18,10))
        plt.scatter(x=imgs_encoded[:,0], y=imgs_encoded[:,1], s=80, c=colors, alpha=0.5)
        plt.title("%d encoded test images, %d nolasing (blue)" % (NN, np.sum(labels==0)))
        plt.pause(.1)

        # generated images
        nrows=7
        ncols=24
        shape = (img_H, img_W)
        ZLIM=6
        grid_x = np.linspace(-ZLIM,ZLIM,ncols)
        grid_y = np.linspace(-ZLIM,ZLIM,nrows)
        canvas = np.zeros((nrows*shape[0],ncols*shape[1])) 
        
        for col, z0 in enumerate(grid_x):
            for row, z1 in enumerate(grid_y):
                z_sample = np.array([[z1,z0]]) * ML.epsilon_std
                gen_sample = ML.generator.predict(z_sample)
                if config.cnn:
                    img_sample = gen_sample[0,4:104,2:52,0]
                else:
                    img_sample = gen_sample[0].reshape(img_H, img_W)
                row_pix = row * shape[0]
                col_pix = col * shape[1]
                canvas[row_pix:(row_pix + shape[0]), col_pix:(col_pix + shape[1])] = img_sample
        plt.figure(plotFigH+1)
        plt.imshow(canvas, interpolation='none')
        plt.title('generated images')
        plt.xlabel('z1 in [-6,6]')
        plt.ylabel('z0 in [-6,6]')
        plt.pause(.1)

        # decoded images, compare to originals
        imgs_decoded = ML.VAE.predict(imgs, batch_size=ML.batch_size)
        idxs= range(len(imgs))
        random.shuffle(idxs)
        plt.figure(plotFigH+2)
        for idx in idxs:
            if config.cnn:
                orig_img = imgs[idx,4:104,2:52,0]
                decoded_img = imgs_decoded[idx,4:104,2:52,0]
            else:
                orig_img = np.reshape(imgs[idx],(img_H, img_W))
                decoded_img = np.reshape(imgs_decoded[idx],(img_H, img_W))
            diff = orig_img - decoded_img
            vmin = min(np.min(orig_img), np.min(decoded_img))
            vmax = max(np.max(orig_img), np.max(decoded_img))

            plt.clf()
            plt.subplot(1,3,1)
            plt.cla()
            plt.imshow(orig_img, vmin=vmin, vmax=vmax, interpolation='none')
            plt.title("orig lab=%d gd=%.2f" % (labels[idx], np.mean(gasdet[idx])))

            plt.subplot(1,3,2)
            plt.cla()
            plt.imshow(decoded_img, vmin=vmin, vmax=vmax, interpolation='none')
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
