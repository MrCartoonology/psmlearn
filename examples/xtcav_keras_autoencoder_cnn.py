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

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import keras.backend as K

# internal
from h5batchreader import H5BatchReader, DataSetGroup
import psmlearn
from psmlearn.pipeline import Pipeline

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

###############
class Keras_Autoencoder_CNN(object):
    def dataprep(self, imgs):
        NN = len(imgs)
        assert imgs.shape==(NN,5000)
        X = np.zeros((NN,108,54,1), dtype=imgs.dtype)
        for idx in range(NN):
            X[idx,4:104,2:52,0]=imgs[idx]
        return X
    
    def __init__(self, img_H, img_W):
        assert img_H==108
        assert img_W==54
        # 108 54

        self.input_img = Input(shape=(img_H, img_W, 1))

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
        
        self.encoded = MaxPooling2D((6, 3), border_mode='same')(x)
        # (3, 3, 4)

        # at this point the representation is (3, 3, 4) i.e. 36-dimensional

        x = Convolution2D(4, 7, 5, activation='relu', border_mode='same')(self.encoded)
        # (3,3,4)
        
        x = UpSampling2D((6, 3))(x)
        # (18,9,4)
        
        x = Convolution2D(8,5,5, activation='relu', border_mode='same')(x)
        # (18, 9, 8)
        
        x = UpSampling2D((3, 3))(x)
        # (54, 27, 8)
        
        x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
        # (54, 27, 16)
        
        x = UpSampling2D((2, 2))(x)
        # (108, 54, 16)
        
        self.decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

        self.autoencoder = Model(self.input_img, self.decoded)

        
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


    ## not a step, called by other plot steps
    def view_trained_model(self, trained_model_file, plot, pipeline, plotFigH, config, step2h5list):
        # get the data that we trained on, as well as new validation data
        h5train, h5validation, h5test = step2h5list['prep']

        # read in the data to look at it
        h5 = h5py.File(h5validation,'r')
        gasdet = h5['gasdet'][:]
        labels = h5['labels'][:]

        reduced_W = h5['config']['reduced_w'].value
        reduced_H = h5['config']['reduced_h'].value
        NN = len(h5['imgs'])

        imgs = np.zeros((NN,108,54,1), dtype=np.float32)
        imgs[:,4:104,2:52,0] = np.resize(h5['imgs'],(NN, 100, 50))
        h5.close()

        # read in the model and load the trained weights
        ML = Keras_Autoencoder_CNN(108,54)
        ML.autoencoder.load_weights(trained_model_file)

        encoded_imgs = ML.autoencoder.predict(imgs)
        plt = pipeline.plt
        
        idxs= range(len(imgs))
        random.shuffle(idxs)
        
        plt.figure()
        for idx in idxs:
            orig_img = imgs[idx,:,:,0]
            encoded_img = encoded_imgs[idx,:,:,0]
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
            plt.title("encoded valid idx=%d" % idx)

            plt.subplot(1,3,3)
            plt.cla()
            plt.imshow(diff, interpolation='none')
            plt.colorbar()

            if pipeline.stop_plots(): 
                break

    def view_fit(self, plot, pipeline, plotFigH, config, step2h5list):
        trained_model_file = step2h5list['fit'][0]
        self.view_trained_model(trained_model_file, plot, pipeline, plotFigH, config, step2h5list)

    def fit(self, config, pipeline, step2h5list, output_files):
        h5train, h5validation, h5test = step2h5list['prep']

        h5 = h5py.File(h5train,'r')
        reduced_W = h5['config']['reduced_w'].value
        reduced_H = h5['config']['reduced_h'].value
        num_imgs = len(h5['imgs'])
        
        train_imgs = np.zeros((num_imgs, 108, 54, 1), dtype=h5['imgs'].dtype)
        train_imgs[:,4:104,2:52:,0] = np.reshape(h5['imgs'][:],(num_imgs, 100, 50))
        h5.close()

        h5 = h5py.File(h5validation,'r')
        NN = len(h5['imgs'])
        validation_imgs = np.zeros((NN, 108, 54, 1), dtype=h5['imgs'].dtype)
        validation_imgs[:,4:104,2:52,0] = np.reshape(h5['imgs'][:],(NN,100,50))
        h5.close()
        
        ML = Keras_Autoencoder_CNN(108,54)
        ML.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        ML.autoencoder.summary()
        ML.autoencoder.fit(train_imgs, train_imgs, nb_epoch=50,
                           batch_size=512,
                           shuffle=True,
                           validation_data=(validation_imgs, validation_imgs))
        ML.autoencoder.save(output_files[0])

            
    def interpolate(self, cmd, **kwargs):
        assert cmd in ['interactive', 'files']
        step2h5list=kwargs.pop('step2h5list')
        plotFigH=kwargs.pop('plotFigH')
        config=kwargs.pop('config')
        pipeline=kwargs.pop('pipeline')
        if cmd=='interactive':
            plot=kwargs.pop('plot')
        elif cmd == 'files':
            imgprefix=kwargs.pop('imgprefix')

        # get the data that we trained on, as well as new validation data
        h5train, h5validation, h5test = step2h5list['prep']

        # read in the data to look at it
        h5 = h5py.File(h5validation,'r')
        gasdet = h5['gasdet'][:]
        labels = h5['labels'][:]
        starts = h5['starts'][:]
        NN = len(h5['imgs'])

        imgs = np.zeros((NN,108,54,1), dtype=np.float32)
        imgs[:,4:104,2:52,0] = np.resize(h5['imgs'],(NN, 100, 50))
        h5.close()

        # read in the model and load the trained weights
        ML = Keras_Autoencoder_CNN(108,54)
        trained_model_file = step2h5list['fit'][0]
        ML.autoencoder.load_weights(trained_model_file)

        encoded_imgs = ML.autoencoder.predict(imgs)
        plt = pipeline.plt
        if not plt:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

        nolasing_idx, lasing_idx = find_images_to_interpolate(starts, labels)

        img_batch = np.zeros((2,108,54,1), dtype=np.float32)
        img_batch[0] = imgs[nolasing_idx]
        img_batch[1] = imgs[lasing_idx]

        encoded = K.get_session().run(ML.encoded, feed_dict={ML.input_img:img_batch})
        encoded_shape = list(encoded.shape)
        num_interp = 25
        encoded_shape[0]=num_interp
        encoded_batch = np.zeros(encoded_shape, dtype=np.float32)
        encoded_batch[0]=encoded[0]
        encoded_batch[num_interp-1]=encoded[1]
        
        direc = encoded[1]-encoded[0]
        for idx in range(1,num_interp-1):
            delta = (idx/float(num_interp)) * direc
            encoded_batch[idx] = encoded[0] + delta

        encoded_imgs = K.get_session().run(ML.decoded, feed_dict={ML.encoded:encoded_batch})

        plt.figure(plotFigH, figsize=(11,8))
        plt.subplot(1,5,1)
        plt.imshow(img_batch[0,:,:,0], interpolation='none')
        plt.title("original no lasing")

        plt.subplot(1,5,2)
        plt.imshow(encoded_imgs[0,:,:,0], interpolation='none')
        plt.title("orig nolas encoded")

        plt.subplot(1,5,4)
        plt.imshow(encoded_imgs[num_interp-1,:,:,0], interpolation='none')
        plt.title("orig las encoded")

        plt.subplot(1,5,5)
        plt.imshow(img_batch[1,:,:,0], interpolation='none')
        plt.title("original lasing")

        vmin = np.min(encoded_imgs)
        vmax = np.max(encoded_imgs)

        img_files = []
        plt.subplot(1,5,3)
        for idx in range(1,num_interp-1):
            encoded_img = encoded_imgs[idx,:,:,0]
            plt.imshow(encoded_img, vmin=vmin, vmax=vmax, interpolation='none')
            plt.title("interp %d" % idx)
            if cmd == 'interactive':
                if pipeline.stop_plots(): 
                    break
            else:
                plt.pause(.1)
                img_file = imgprefix + '_%03d.png' % idx
                plt.savefig(img_file)
                img_files.append(img_file)
        return img_files

    def view_interpolate(self, plot, pipeline, plotFigH, config, step2h5list):
        self.interpolate('interactive', plot=plot, pipeline=pipeline, plotFigH=plotFigH, config=config, step2h5list=step2h5list)

    def interpolate_video(self, config, pipeline, step2h5list, output_files):
        imgprefix = output_files[0] + '_imgprefix'
        video_file = output_files[0].replace('.h5','.mp4')
        img_files = self.interpolate('files', step2h5list=step2h5list, config=config, imgprefix=imgprefix, plotFigH=23, pipeline=pipeline) 
        # below, my safari mac broswer can read it, but not the linux browsers.
        cmd = 'ffmpeg -framerate 2 -i ' + imgprefix + '_%03d.png -pix_fmt yuv420p ' + video_file
        print(cmd)
        os.system(cmd)
        h5py.File(output_files[0],'w')

######################################
if __name__ == '__main__':
    mysteps = MySteps()
    pipeline = Pipeline(stepImpl = mysteps,
                        session = K.get_session(),                        
                        defprefix='xtcav_autoencoder_cnn',
                        outputdir='/scratch/davidsch/dataprep')        # can be overriden with command line arguments
    mysteps.add_commandline_arguments(pipeline.parser)
    pipeline.add_step_method(name='prep',
                             output_files=['_train','_validation', '_test'])
    pipeline.add_step_method(name='fit')
    pipeline.add_step_method_plot(name='view_fit')
    pipeline.add_step_method_plot(name='view_interpolate')
    pipeline.add_step_method(name='interpolate_video')
    
    pipeline.run()
