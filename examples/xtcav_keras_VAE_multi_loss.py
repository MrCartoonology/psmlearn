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

# from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Convolution2D, Flatten, Reshape
from keras.layers import BatchNormalization, MaxPooling2D, UpSampling2D
from keras import objectives
import keras.backend as K
import tensorflow as tf

# internal
import psmlearn
from psmlearn.pipeline import Pipeline

######## helper
def read_data(h5file, ML, dev):
    h5 = h5py.File(h5file,'r')
    imgs = h5['imgs'][:]
    N = len(imgs)
    if dev:
        N = min(2000, N)
    N -= N % ML.batch_size
    data = {}
    data['imgs'] = np.zeros((N,108,54,1), dtype=np.float32)
    data['vert_edges'] = np.zeros((N,108,54,1), dtype=np.float32)
    data['horiz_edges'] = np.zeros((N,108,54,1), dtype=np.float32)
    for idx in range(N):                
        data['imgs'][idx,4:104,2:52,0]=imgs[idx].reshape((100,50))
    print("read images")
    for idx in range(0,N,ML.batch_size):
        data['vert_edges'][idx:(idx+ML.batch_size)] = K.get_session().run(ML.vert_edges_fn(data['imgs'][idx:idx+ML.batch_size]))
        data['horiz_edges'][idx:(idx+ML.batch_size)] = K.get_session().run(ML.horiz_edges_fn(data['imgs'][idx:idx+ML.batch_size]))
        print(idx)
    data['gasdet'] = np.mean(h5['gasdet'][0:N],1)
    data['labels'] = h5['labels_onehot'][0:N]
    data['labels_int'] = h5['labels'][0:N]
    h5.close()
    return data

def sampling(args, **kwargs):
    z_mean, z_log_var = args
    batch_size = kwargs.pop('batch_size')
    latent_dim = kwargs.pop('latent_dim')
    epsilon_std = kwargs.pop('epsilon_std')
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2.0) * epsilon

def tf_edge(args, **kwargs):
    X = args
    W = kwargs.pop('W')
    assert K.backend()=='tensorflow'

    Y = tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')
    Y /= 2.0
    Y += 0.5       # --> back to [0,1]

    return Y
    
class Keras_VAE_CNN(object):
    def __init__(self, batch_size, latent_dim):
        self.set_config(batch_size=batch_size, latent_dim=latent_dim)
        self.set_inputs()
        self.make_vae()
        self.make_horiz_ver_edges()
        self.make_gasdet_label()
        self.define_models()
        
    def set_config(self, batch_size, latent_dim):
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.epsilon_std = 1.0
        self.img_H = 108
        self.img_W = 54
        self.loss_weights = {'GEN':5000.0,
                             'K1':1.0,
                             'EDGE':1000.0,
                             'LABEL':10.0,
                             'GASDET':10.0}
    def set_inputs(self):
        self.input_img = Input(batch_shape=(self.batch_size, self.img_H, self.img_W, 1))
        self.input_labels = Input(batch_shape=(self.batch_size,2))
        self.input_gasdet = Input(batch_shape=(self.batch_size,1))
        
    def make_vae(self):
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

        self.encoder_layers.append(Convolution2D(1,3,3, activation='sigmoid', border_mode='same', name='img_decoded_mean'))

        x = self.z
        for layer in self.encoder_layers:
            x = layer(x)
        self.img_decoded_mean = x

    def make_horiz_ver_edges(self):
        assert K.backend()=='tensorflow'
        # working around keras issue by dropping to tensorflow
        W_horiz_edges = np.ones((2,1,1,1), dtype = np.float32)
        W_horiz_edges[1,0,0,0] = -1.0

        W_vert_edges = np.ones((1,2,1,1), dtype = np.float32)
        W_vert_edges[0,1,0,0] = -1.0

        self.W_horiz_edges = tf.Variable(initial_value=W_horiz_edges, trainable=False)
        self.W_vert_edges = tf.Variable(initial_value=W_vert_edges, trainable=False)

        self.horiz_edges_fn = Lambda(tf_edge, arguments={'W':self.W_horiz_edges}, name='horiz_edges')
        self.horiz_edges_output = self.horiz_edges_fn([self.input_img])

        self.vert_edges_fn = Lambda(tf_edge, arguments={'W':self.W_vert_edges}, name='vert_edges')
        self.vert_edges_output = self.vert_edges_fn([self.input_img])

    def make_gasdet_label(self):
        x = Dense(8, activation='relu')(self.z)
        self.gasdet = Dense(1, name='gasdet')(x)
        self.label_logits = Dense(2, name='label_logits')(self.z)
        
    def define_models(self):
        self.VAE = Model(input=[self.input_img],
                         output=[self.img_decoded_mean,
                                 self.horiz_edges_output,
                                 self.vert_edges_output,
                                 self.gasdet,
                                 self.label_logits])

        self.encoder = Model(self.input_img, self.z_mean)

        self.decoder_input = Input(shape=(self.latent_dim,))
        x = self.decoder_input
        for layer in self.encoder_layers:
            x = layer(x)
        self.generator = Model(self.decoder_input, x)

    def img_decoded_mean_loss_fn(self, x, y):
        x = Flatten()(x)
        y = Flatten()(y)
        self.vae_loss = self.loss_weights['GEN'] * objectives.binary_crossentropy(x,y)
        
        self.k1_loss = -0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        self.k1_loss *= self.loss_weights['K1']

        return self.vae_loss + self.k1_loss

    def edges_loss_fn(self, x, y):
        loss = objectives.binary_crossentropy(Flatten()(x), Flatten()(y))
        loss *= self.loss_weights['EDGE']
        return loss

    def horiz_edges_loss_fn(self, x, y):
        self.horiz_edge_loss = self.edges_loss_fn(x,y)
        return self.horiz_edge_loss

    def vert_edges_loss_fn(self, x, y):
        self.vert_edge_loss = self.edges_loss_fn(x,y)
        return self.vert_edge_loss

    def label_loss_fn(self, x, y):
        self.label_loss = objectives.binary_crossentropy(x,y)
        self.label_loss *= self.loss_weights['LABEL']
        return self.label_loss

    def gasdet_loss_fn(self, x, y):
        self.gasdet_loss = objectives.mean_squared_error(x,y)
        self.gasdet_loss *= self.loss_weights['GASDET']
        return self.gasdet_loss

        
################
class MySteps():
    def __init__(self):
        pass
    
    def add_commandline_arguments(self, parser):
        parser.add_argument('--nb', type=int,
                            help='number of epochs',
                            default=50),
        
        parser.add_argument('--batch_size', type=int,
                            help='batch size',
                            default=128),
        
        parser.add_argument('--latent_dim', type=int,
                            help='latent_dim',
                            default=2),
        
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

        ML = Keras_VAE_CNN(batch_size=config.batch_size, latent_dim=config.latent_dim)
        ML.VAE.summary()

        ML.VAE.compile(optimizer='Adam',
                       loss={'img_decoded_mean': ML.img_decoded_mean_loss_fn,
                             'horiz_edges':ML.horiz_edges_loss_fn,
                             'vert_edges':ML.vert_edges_loss_fn,
                             'gasdet':ML.gasdet_loss_fn,
                             'label_logits':ML.label_loss_fn})

        train_data = read_data(h5train, ML, config.dev)
        validation_data = read_data(h5validation, ML, config.dev)

        ML.VAE.fit(x=train_data['imgs'],
                   y={'img_decoded_mean': train_data['imgs'],
                      'horiz_edges':train_data['horiz_edges'],
                      'vert_edges':train_data['vert_edges'],
                      'gasdet':train_data['gasdet'],                        
                      'label_logits':train_data['labels']},
                   nb_epoch=config.nb,
                   batch_size=ML.batch_size,
                   shuffle=True,
                   validation_data=(validation_data['imgs'],
                                    {'img_decoded_mean': validation_data['imgs'],
                                    'horiz_edges':validation_data['horiz_edges'],
                                    'vert_edges':validation_data['vert_edges'],
                                    'gasdet':validation_data['gasdet'],                        
                                    'label_logits':validation_data['labels']})
                   )
        # save doesn't work?
        ML.VAE.save_weights(output_files[0])

    ## not a step, called by other plot steps
    def view_trained_model(self, trained_model_file, plot, pipeline, plotFigH, config, step2h5list):
        # read in the model and load the trained weights
        ML = Keras_VAE_CNN(batch_size=config.batch_size, latent_dim=config.latent_dim)
        ML.VAE.load_weights(trained_model_file)

        h5train, h5validation, h5test = step2h5list['prep']
        data = read_data(h5validation, ML, config.dev)

        plt = pipeline.plt
        assert plt

        # latent space plot
        imgs_encoded = ML.encoder.predict(data['imgs'], batch_size=ML.batch_size)
        labels = np.argmax(data['labels'], axis=1)
        assert len(labels)==len(data['labels'])
        colors = labels.astype(np.float)
        plt.figure(plotFigH, figsize=(18,10))
        plt.scatter(x=imgs_encoded[:,0], y=imgs_encoded[:,1], s=80, c=colors, alpha=0.5)
        plt.title("%d encoded test images, %d nolasing (blue)" % (len(labels), np.sum(labels==0)))
        plt.pause(.1)

        # generated images
        nrows=7
        ncols=24
        shape = (100,50)
        ZLIM=6
        grid_x = np.linspace(-ZLIM,ZLIM,ncols)
        grid_y = np.linspace(-ZLIM,ZLIM,nrows)
        canvas = np.zeros((nrows*shape[0],ncols*shape[1])) 
        
        for col, z0 in enumerate(grid_x):
            for row, z1 in enumerate(grid_y):
                z_sample = np.array([[z1,z0]]) * ML.epsilon_std
                gen_sample = ML.generator.predict(z_sample)
                img_sample = gen_sample[0,4:104,2:52,0]
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
        imgs_decoded, imgs_horiz, imgs_vert, gasdet_predict, label_predict  = ML.VAE.predict(data['imgs'], batch_size=ML.batch_size)
        idxs= range(len(data['imgs']))
        random.shuffle(idxs)
        plt.figure(plotFigH+2)
        for idx in idxs:
            orig_img = data['imgs'][idx,4:104,2:52,0]
            decoded_img = imgs_decoded[idx,4:104,2:52,0]
            diff = orig_img - decoded_img
            vmin = min(np.min(orig_img), np.min(decoded_img))
            vmax = max(np.max(orig_img), np.max(decoded_img))

            plt.clf()
            plt.subplot(1,3,1)
            plt.cla()
            predicted_label = 1
            if label_predict[idx,0]>label_predict[idx,1]:
                predicted_label = 0
            plt.imshow(orig_img, vmin=vmin, vmax=vmax, interpolation='none')
            plt.title("orig lab=%d/%d gd=%.2f/%.2f" % (data['labels_int'][idx],
                                                       predicted_label,
                                                       np.mean(data['gasdet'][idx]),
                                                       gasdet_predict[idx,0]))

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
                        defprefix='xtcav_VAE_ml',
                        outputdir='/scratch/davidsch/dataprep')        # can be overriden with command line arguments
    mysteps.add_commandline_arguments(pipeline.parser)
    pipeline.add_step_method(name='prep', output_files=['_train','_validation', '_test'])
    pipeline.add_step_method(name='fit')
    pipeline.add_step_method_plot(name='view_fit')
    
    pipeline.run()
