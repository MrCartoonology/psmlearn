from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time
from mpi4py import MPI
import os

import sys
import numpy as np
import random
if os.environ.get('MOCK_TENSORFLOW',None):
    import psmlearn.mock_tensorflow as tf
else:
    import tensorflow as tf
from scipy.misc import imresize
from psmlearn.pipeline import Pipeline
import psmlearn.h5util as h5util
import psmlearn.util as util
from psmlearn import tsne
import psmlearn.plot as psplot
import psmlearn
import numpy as np
import h5py
import time

comm=MPI.COMM_WORLD
rank=comm.rank
worldsize=comm.size
from jinja2 import Environment, FileSystemLoader


def get_random_val_from(X):
    idx = random.randint(0, len(X)-1)
    return X[idx]

def dowork(jobs):
    template = Environment(loader=FileSystemLoader('.')).get_template('config.yaml.jinja2')
    for job in jobs:
        config = template.render(job)
        prefix = 'vgg16_xtcav_xvalidate_job_%3.3d' % job['job']
        config_file = os.path.join('config',prefix + '.yaml')
        fout=file(config_file,'w')
        fout.write('%s\n' % config)
        fout.close()
        time.sleep(.01)

        stepImpl = XtcavVgg16()
        outputdir = psmlearn.dataloc.getDefalutOutputDir(project='xtcav')
        pipeline = Pipeline(stepImpl=stepImpl, outputdir=outputdir)
        stepImpl.add_arguments(pipeline.parser)
        pipeline.add_step_method(name='roi')
        pipeline.add_step_method_plot(name='plot_roi')
        pipeline.add_step_method(name='compute_channel_mean')
        pipeline.add_step_method(name='compute_vgg16_codewords',
                                 output_files=['_train','_validation','_test'])
        pipeline.add_step_method(name='train_on_codewords')
        pipeline.init(command_line=['--log=DEBUG', '--dev','--force', '--redoall', '--config', config_file, prefix])
        pipeline.run()
        
        
class XValidate(object):
    def __init__(self, config_template, seed, num_jobs):
        self.config_template = config_template
        self.seed = seed
        self.num_jobs = num_jobs
        self.params = {}

    def config_param(self, name, values=None, limits=None, logscale=None):
        assert name not in self.params, "name=%s already in params" % name
        assert values or limits, "set one of values or limits for param %s" % name
        self.params[name]={'values':values,
                           'limits':limits,
                           'logscale':logscale}

    def run(self):
        print("rank=%d of %d" % (rank, worldsize))
        random.seed(23209101)
        optimizer_param_momentum_all = [0.0, 0.3, 0.6, 0.8, 0.9, .95, .98]
        learning_rate_decay_rate_all = [0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 0.9999]
        all_jobs = []
        NFACTOR=1
        for job in range(NFACTOR * worldsize):
            thresh = float(random.randint(1,200))
            learning_rate = 10.0 ** (-1.0 * (random.random()*4))
            l2reg = max(0.0, 10.0**(-1.0*(random.random()*4))-1e-3)
            l1reg = max(0.0, 10.0**(-1.0*(random.random()*4))-1e-3)
            optimizer_param_momentum = get_random_val_from(optimizer_param_momentum_all)
            learning_rate_decay_rate = get_random_val_from(learning_rate_decay_rate_all)
            all_jobs.append({'job':job,
                             'thresh':thresh,
                             'learning_rate':learning_rate,
                             'l2reg':l2reg,
                             'l1reg':l1reg,
                             'optimizer_param_momentum':optimizer_param_momentum,
                             'prefix':'vgg16-xtcav-xvalidate-job_%3.3d' % job,
                             'learning_rate_decay_rate':learning_rate_decay_rate})
        jobs_this_rank = [job for job in all_jobs if job['job'] % worldsize == rank]    
        dowork(jobs_this_rank)
