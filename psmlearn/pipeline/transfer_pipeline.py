from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

self.step_names = ['view_orig_data',
                           'data_stats',
                           'view_data_stats',
                           'model_layers',
                           'view_model_layers']


import os
import sys
import argparse
import numpy as np
import h5py
import os
if os.environ.get('MOCK_TENSORFLOW',None):
    import psmlearn.mock_tensorflow as tf
else:
    import tensorflow as tf
import logging

import psmlearn.util as util
import psmlearn.h5util as h5util

parserEpilog = '''
generic pipeline for transfer learning:

* data_stats - the model may need some overall stats of the data, i.e, channel means
* plot_data_stats - take a look at raw and preprocessed
* data_layers - run the data through the model and save some output layers
* plot_layers - plot output layers in some fasion
* feat_sel - select some subset of the output layers (eliminate dead neurons, etc)
* tsne - generate tsne embedding of layers
* plot_tsne - look at tsne plot
* 
7. test classifier
8. identify high scoring neurons for each label
9. guided back prop
'''

def _redo_flag_name(name):
    return name
#    return 'do_%s' % name


def _getDefaultPipelineParser(output_dir, 
                              description='', 
                              extraEpilog=None,
                              step_names=[]):
    global parserEpilog
    epilog = parserEpilog
    if extraEpilog:
        epilog += '\n'
        epilog += extraEpilog
    parser = argparse.ArgumentParser(description=description,
                                     epilog=epilog,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('prefix', type=str, help='prefix for filenames')
    parser.add_argument('--redoall', action='store_true', help='redo all steps', default=False)
    parser.add_argument('--outputdir', type=str, help='output directory default=%s' % output_dir, default=output_dir)
    parser.add_argument('--seed', type=int, help='seed for random number generators', default=39819)
    parser.add_argument('--view', action='store_true', help='view step summary plots, if avail')
    parser.add_argument('--viewall', action='store_true', help='view step detailed plots, if avail')
    parser.add_argument('--loglvl', type=str, help='one of DEBUG,INFO,WARN,ERROR,CRITICAL.', default='DEBUG')
    parser.add_argument('--force', action='store_true', help='overwrite existing filenames')
    parser.add_argument('--clean', action='store_true', help='delete all output for this prefix')
    parser.add_argument('--config', type=str, help='config file for steps. a .yml file', default=None)
    
    for name in step_names:
        redo = _redo_flag_name(name)
        parser.add_argument('--%s' % redo, action='store_true', help='do or redo step %s' % name, default=False)

    return parser

    
class Pipeline(object):
    '''base class for pipeline.
    subclasses should
    override init and call base class
    implement 'data_stats', 'model_layers' methods that take lists of input and output files
    '''
    def __init__(self, output_dir='.', sess=None):
        self.initialized = False
        self.outupt_dir = output_dir
        self.name2stepinfo = {}
        for name in self.step_names:
            self.name2stepinfo[name] = {'name':name,
                                        'data_gen_randomizerandomize_data_iter':False,
                                        'present':False, 
                                        'data_gen':None,
                                        'data_gen_preprocess':None}
        if sess is None:
            sess = tf.Session()
        self.sess = sess
        
    def get_parser(self, **kwargs):
        parser = _getDefaultPipelineParser(output_dir=kwargs.pop('output_dir'), 
                                           description=kwargs.pop('description',''), 
                                           extraEpilog=kwargs.pop('extraEpilog',None),
                                           step_names=self.step_names)
        return parser
    
    def init(self, args, data_gen_preprocess, stats_data_generator, layers_data_generator):
        self.initialized = True
        self.nm2stepinfo['data_stats']['data_gen'] = stats_data_generator
        self.nm2stepinfo['data_stats']['data_gen_preprocess'] = data_gen_preprocess
        self.nm2stepinfo['model_layers']['data_gen'] = layers_data_generator
        self.nm2stepinfo['model_layers']['data_gen_preprocess'] = data_gen_preprocess
        
        self.args = args
        self.outputdir = self.args.outputdir
        self.prefix = self.args.prefix

        self.steps = self._createStepsFromInfo()

        self.hdr = 'Pipeline'
        nm2lvl = {'INFO':logging.INFO,
                  'DEBUG':logging.DEBUG}
#                  'ERROR':logging.ERROR,
#                  'CRITICAL':logging.CRITICAL,
#                  'ERROR':logging.ERROR}
        assert args.loglvl in nm2lvl, "loglvl must be one of %s" % str(nm2lvl.keys())
        self.doTrace=nm2lvl[args.loglvl] <= logging.INFO
        self.doDebug=nm2lvl[args.loglvl] <= logging.DEBUG
        if self.args.clean:
            self.trace('clean received - call run() to delete all files for prefix')

    def _createStepsFromInfo(self):
        basefname = os.path.join(self.outputdir, self.prefix)
        steps = []
        previous_step_output = []
        for step_name in self.step_names:
            step_info = self.name2stepinfo[step_name]
            output_file = basefname + '_' + step_name + '.h5'
            next_step = _Step(info=step_info,
                              instance=self,
                              input_files=previous_step_output,
                              output_files=[output_file])
            previous_step_output = [output_file]
            steps.append(next_step)
        return steps
    

    def trace(self, msg):
        util.logTrace(self.hdr, msg, self.doTrace)

    def debug(self, msg):
        util.logDebug(self.hdr, msg, self.doDebug)

    def manage_step(self, step):
        assert self.initialized, "Pipeline not initialized, call init()"
        for fname in step.input_files:
            assert os.path.exists(fname), "Can't execute step=%s, input file=%s doesn't exist" % (step.name, fname)
        all_output_exists = all([os.path.exists(fname) for fname in step.output_files])
        any_output_exists = any([os.path.exists(fname) for fname in step.output_files])
        if all_output_exists and not (self.args.redoall or step.redo(self.args)):
            self.trace("step=%s already done, all output exists" % step.name)
        else:
            if any_output_exists and not self.args.force:
                raise Exception("Some of the output files: %s already exist, use --force to overwrite" % step)
            self.trace("running step=%s" % step)
            step.run(self.data_generator)

    def run(self):
        assert self.initialized, "Pipeline object not initialized, call init()"
        if self.args.clean:
            for step in self.steps:
                for fname in step.output_files:
                    if os.path.exists(fname):
                        os.unlink(fname)
                        self.trace("Deleted file: %s" % fname)
        else:
            for step in self.steps:
                self.manage_step(step)

class Vgg16(Pipeline):
    def __init__(self):
        super(Vgg16, self).__init__( **kwargs)

    def get_parser(self, **kwargs):
        parser = super(Vgg16, self).get_parser(**kwargs)
        parser.add_argument('--vgg16weights', type=str, help='weights file for vgg16', default='data/vgg16_weights.npz')
  
    def init(self,args, data_generator, img2vgg16, img2vgg16params=None):
        super(Vgg16Pipeline, self).init(args=args, data_generator=data_generator)
        self.img2vgg16 = img2vgg16
        self.img2vgg16params = img2vgg16params
        self.hdr='Vgg16Pipeline'
        self._vgg = None

    def vgg(self):
        if self._vgg is None:
            self.trace('creating vgg16 from weights=%s' % self.args.vgg16weights)
            self._vgg = psmlearn.vgg16.create(session=self.session, 
                                              weights=self.args.vgg16weights)
        return self._vgg

    def getParser(self, outputdir):
        parser.add_argument('--vgg16weights', type=str, help='weights file for vgg16', default='data/vgg16_weights.npz')

    def data_stats(self, data_iter, input_files, output_files):
        means = []
        for imgNum,img in enumerate(data_iter):
            vgg16img = self.img2vgg16(img=img, channel_mean=None, **self.img2vgg16params)
            means.append(np.mean(vgg16img))
            self.debug('data_stats: img %5d mean=%.2f' % (imgNum, means[-1]))
        channel_mean = np.mean(np.array(means))
        h5util.dict2h5(output_files[0], {'channel_mean':channel_mean,
                                         'number_images':(imgNum+1)})
        self.trace('data_stats: finished - final channel mean=%.2f' % channel_mean)

    def model_layers(self, data_iter, input_files, output_files):
        vgg = self.vgg()
        h5in = h5py.File(input_files[0],'r')
        channel_mean = h5in['channel_mean'].value
        num_images = h5in['number_images'].value
        self.trace('write_model_layers: starting - loaded channel mean=%.2f, num_images=%d' % (channel_mean, num_images))
        h5 = h5py.File(output_files[0],'w')
        first = True
        for img in data_ter:            
            preprocessed_img = self.preprocess_img(img=img, 
                                                   channel_mean=channel_mean, 
                                                   **self.preprocess_params)
#            fc1, fc2 = vgg.get_output_layers(vgg16i
                
            first = False
