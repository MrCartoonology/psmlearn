from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import h5py
if os.environ.get('MOCK_TENSORFLOW',False):
    import psmlearn.mock_tensorflow as tf
else:
    import tensorflow as tf
import logging

import psmlearn.util as util
import psmlearn.h5util as h5util
from  psmlearn.pipeline import Pipeline
import psmlearn

def data_gen_with_imgprep(randomize=False, seed=109, args=None, num=10,
                          step2h5list={}, pipeline=None):
    raw_data_gen = pipeline.default_data_gen
    prepare_for_vgg16 = pipeline.prepare_for_vgg16
    imgprep_config = pipeline.get_config('imgprep')
    channel_mean = h5py.File(step2h5list['data_stats'][0],'r')['channel_mean'].value

    raw_iter = raw_data_gen(randomize=randomize, seed=seed, args=args, num=num)
    
    for X,Y,meta in raw_iter:
        img = X[0]
        prep_img = prepare_for_vgg16(img, imgprep_config, channel_mean=channel_mean)
        X[0]=prep_img
        yield X,Y,meta

def data_stats(data_iter, config, pipeline, step2h5list, output_files):
    '''create one output file with the channel mean.
    '''
    prepare_for_vgg16 = pipeline.prepare_for_vgg16
    all_means = []
    num = 0
    for X,Y,meta in data_iter:
        img = X[0]
        prep_img = prepare_for_vgg16(img,
                                     pipeline.get_config('imgprep'),
                                     channel_mean=None)
        all_means.append(np.mean(prep_img[:,:,0]))
        num += 1
    h5util.write_to_h5(output_files[0],{'datalen':num,
                                        'channel_mean': np.mean(all_means)})

def plot_imgprep(data_iter, config, pipeline, plot, plotFigH, step2h5list):
    plt = pipeline.plt
    plt.figure(plotFigH, figsize=(10,10))
    plt.clf()
    h5 = h5py.File(step2h5list['data_stats'][0],'r')
    channel_mean = h5['channel_mean'].value
    prepare_for_vgg16 = pipeline.prepare_for_vgg16
    for X,Y,meta in data_iter:
        img = X[0]
        prep_img = prepare_for_vgg16(img, config, channel_mean=channel_mean)
        mn = np.min(prep_img)
        mx = np.max(prep_img)
        plt_img = prep_img - mn
        plt_img /= (mx-mn)
        plt.subplot(1,2,1)
        plt.imshow(img, interpolation='none')
        plt.title('orig')
        plt.subplot(1,2,2)
        plt.imshow(plt_img, interpolation='none')
        plt.title('imgprep: scaled mn=%.1f mx=%.1f' % (mn, mx))
        if pipeline.stop_plots():
            break

def model_layers(data_iter, config, pipeline, step2h5list, output_files):
    data_stats = h5util.read_from_h5(step2h5list['data_stats'][0])
    channel_mean = data_stats['channel_mean']
    datalen = data_stats['datalen']
    model = pipeline.vgg16()
    prepare_for_vgg16 = pipeline.prepare_for_vgg16
    imgprep_config = pipeline.get_config('imgprep')
    vgg16 = pipeline.vgg16()
    h5out = h5py.File(output_files[0],'w')
    idx = 0
    layer_names = ['fc1','fc2']
    name2dset = {}
    all_meta = []
    for X,Y,meta in data_iter:
        all_meta.append(meta)
        img = X[0]
        prep_img = prepare_for_vgg16(img, imgprep_config, channel_mean=channel_mean)
        prep_img = np.reshape(prep_img, [1]+list(prep_img.shape))
        layers = vgg16.get_model_layers(sess=pipeline.session,
                                        imgs=prep_img,
                                        layer_names=layer_names)
        if idx == 0:
            for layer, name in zip(layers, layer_names):
                dset_shape = tuple([datalen] + list(layer.shape[1:]))
                name2dset[name] = h5out.create_dataset(name, shape=dset_shape, dtype=layer.dtype)
        for layer, name in zip(layers, layer_names):
            name2dset[name][idx,:] = layer[0,:]
        idx += 1
    h5util.write_meta(h5out, all_meta)
    
class Vgg16Pipeline(Pipeline):
    '''base class for pipeline.
    subclasses should
    override init and call base class
    implement 'data_stats', 'model_layers' methods that take lists of input and output files
    '''
    def __init__(self, outputdir='.',
                 default_data_gen=None,
                 default_data_gen_params={},
                 prepare_for_vgg16=None,
                 layers=('fc1','fc2'),
                 description='', epilog='', session=None, plt=None, comm=None):        
        super(Vgg16Pipeline, self).__init__(outputdir=outputdir,
                                            default_data_gen=default_data_gen,
                                            default_data_gen_params=default_data_gen_params,
                                            description=description, epilog=epilog,
                                            session=session, plt=plt, comm=comm)
        assert prepare_for_vgg16
        self.prepare_for_vgg16 = prepare_for_vgg16
        self.add_step_fn(name='data_stats', fn=data_stats) 
        self.add_step_fn_plot(name='plot_imgprep', fn=plot_imgprep)
        self.add_step_fn(name='model_layers', fn=model_layers)
        self._vgg16 = None
        
    def vgg16(self):
        if self._vgg16 is None:
            self._vgg16 = psmlearn.vgg16.create(session=self.session,
                                                weights='/reg/d/ana01/temp/davidsch/mlearn/vgg16/vgg16_weights.npz')
        return self._vgg16
    
    def validateConfig(self, config):
        expected_keys = ['imgprep']
        assert set(config.keys())==set(expected_keys), "config keys=%s but expected=%s" % (config.keys(), expected_keys)
        
    def add_step_fn_with_imgprep_iter(self, name, fn, data_gen_params={}, help=''):
        data_gen = data_gen_with_imgprep
        self._add_step(name=name, inst=None, fn_or_method=fn,
                       plot=False, help=help, what_data_gen='STEP_DATA_GEN',
                       data_gen=data_gen_with_imgprep,
                       data_gen_params=data_gen_params)

    def add_step_fn_with_imgprep_iter(self, name, fn, data_gen_params={}, help=''):
        data_gen = data_gen_with_imgprep
        self._add_step(name=name, inst=None, fn_or_method=fn,
                       plot=False, help=help, what_data_gen='STEP_DATA_GEN',
                       data_gen=data_gen_with_imgprep,
                       data_gen_params=data_gen_params)

