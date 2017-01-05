from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import copy
import time
import argparse
import numpy as np
import h5py
if os.environ.get('MOCK_TENSORFLOW',False):
    import psmlearn.mock_tensorflow as tf
else:
    import tensorflow as tf
import logging
import yaml
from mpi4py import MPI
import psmlearn.util as util
import psmlearn.h5util as h5util

from .step import Step, WHAT_DATA_GEN

def _redo_flag_name(name):
    return name
#    return 'do_%s' % name


def _addPipelineArgs(parser, outputdir, defprefix=None):
    parser.add_argument('--prefix', type=str, help='prefix for filenames', default=defprefix)
    parser.add_argument('--redoall', action='store_true', help='redo all steps', default=False)
    parser.add_argument('--outputdir', type=str, help='output directory default=%s' % outputdir, default=outputdir)
    parser.add_argument('--dev', action='store_true',help='develop mode, for shortening datasets, load times, etc.')
    parser.add_argument('--split_seed', type=int, help='seed for dataset random number generators - keep fixed over hyper parameter search', default=39819)
    parser.add_argument('--job_seed', type=int, help='seed for per job random values', default=19123)
    parser.add_argument('--plot', type=int, help='set to 1 or greater to have plot step functions called, they will get this arg', default=0)
    parser.add_argument('--log', type=str, help='one of DEBUG,INFO,WARN,ERROR,CRITICAL.', default='INFO')
    parser.add_argument('--force', action='store_true', help='overwrite existing filenames')
    parser.add_argument('--clean', action='store_true', help='delete all output for this prefix')
    parser.add_argument('--gpu', type=int, help='limit to one gpu on a multi-device system', default=0)
    parser.add_argument('--cores', type=int, help='cores for tensorflow inter/intra ops', default=0)
    parser.add_argument('-c', '--config', type=str, help='config file for steps. a .yml file', default=None)

    
class Pipeline(object):
    '''base class for pipeline. Provides managements of steps, for re-running, and providing
    previous step output to later steps, and global pipeline reources to the steps
    '''
    def __init__(self,
                 outputdir='.',
                 stepImpl=None,
                 description='',
                 epilog='',
                 session=None,
                 plt=None,
                 defprefix=None,
                 comm=None):
        self.initialized = False
        self.outputdir = outputdir
        self.stepImpl = stepImpl
        self.description=description
        self.epilog=epilog
        self.session = session
        self.plt = plt
        self.defprefix = defprefix
        if comm is None: comm = MPI.COMM_WORLD
        self.comm = comm
        self.rank = comm.Get_rank()
        self.world_size = comm.Get_size()
        
        self.args = None
        self.all_step_config = None
        self.steps = []
        self.name2step = {}
        self._steps_fixed = False

#        tf.reset_default_graph()
        
        self.doTrace=False
        self.doDebug=False
        self.hdr = 'Pipeline'
        if self.world_size > 1:
            self.hdr += ':rnk=%2.2d' % self.rank
        self.parser = argparse.ArgumentParser(add_help=False)
        _addPipelineArgs(parser=self.parser, outputdir=self.outputdir, defprefix=self.defprefix)

    def __del__(self):
        pass
#        if self.session is not None:
#            self.session.close()
            
        
    def _add_step(self,
                  name,
                  stepImpl,
                  fn_or_method,
                  plot, help,
                  output_files,
                  what_data_gen,
                  data_gen,
                  data_gen_params):
        assert name != 'init', "step name 'init' reserved for initializing stepImpl instances"
        assert what_data_gen in WHAT_DATA_GEN, "what_data_gen must be one of %s" % WHAT_DATA_GEN
        if what_data_gen == 'NO_DATA_GEN':
            assert not data_gen
            assert not data_gen_params
        assert not self.initialized, "pipeline has already been initialized, steps are fixed. can't add step %s" % name

        if what_data_gen in ['RAW_DATA_GEN', 'STEP_DATA_GEN']:
            if not data_gen:
                data_gen = self.default_data_gen
            if not data_gen_params:
                data_gen_params = self.default_data_gen_params
        if len(output_files)==0:
            output_suffixes= [name + '.h5']
        else:
            output_suffixes= [name + extra + '.h5' for extra in output_files]
        for step in self.steps:
            for suffix in step.output_suffixes:
                assert suffix not in output_suffixes, "step %s has output suffix=%s that collides with suffix for step %s" % (name, suffix, step.name)
        # separate step figH by 10 so they can make up to 10 plots without colliding
        plotFigH = len(self.steps)*10
        step = Step(name=name,
                    stepImpl=stepImpl, fn_or_method=fn_or_method,
                    what_data_gen=what_data_gen,
                    data_gen=data_gen,
                    data_gen_params=data_gen_params,
                    plot=plot,
                    plotFigH=plotFigH,
                    pipeline=self,
                    output_suffixes=output_suffixes)
        self.steps.append(step)
        self.name2step[name]=step
        self.parser.add_argument('--%s' % name, action='store_true', help='just execute step %s' % name, default=False)
        
    def add_step_fn_plot(self, name, fn, help='', what_data_gen='RAW_DATA_GEN', data_gen=None, data_gen_params={}):
        self._add_step(name=name, stepImpl=None, fn_or_method=fn, plot=True, output_files=[], help=help, what_data_gen=what_data_gen, data_gen=data_gen, data_gen_params=data_gen_params)
        
    def add_step_fn(self, name, fn, help='', what_data_gen='RAW_DATA_GEN', data_gen=None, data_gen_params={}):
        self._add_step(name=name, stepImpl=None, fn_or_method=fn, plot=False, output_files=[],  help=help, what_data_gen=what_data_gen, data_gen=data_gen, data_gen_params=data_gen_params)

    def add_step_method_plot(self, name, help=''):
        stepImpl=self.stepImpl
        method = getattr(stepImpl, name)
        self._add_step(name=name,
                       stepImpl=stepImpl,
                       fn_or_method=method,
                       plot=True,
                       help=help,
                       output_files=[], 
                       what_data_gen='NO_DATA_GEN',
                       data_gen=None,
                       data_gen_params=None)

    def add_step_method(self, name, output_files=[], help=''):
        '''adds step to pipeline from method of stepImpl.

        Must construct pipeline with stepImpl argument.

        ARGS:
          name: (str) name of method of the stepImpl argument. Must have the signature:
                name(self, config, pipeline, step2h5list, output_files)

          output_files: (optional, list of strings) if step produces multiple output
            files, provide a list of names here. These will be pieces of the complete
            name.

        With the default output_files, the output file will be 

        prefix_name.h5

        if output_files is given, say it is ['train','test'] the output files will be

        prefix_name_train.h5
        prefix_name_test.h5
        '''
        stepImpl=self.stepImpl
        method = getattr(stepImpl, name)
        self._add_step(name=name,
                       stepImpl=stepImpl,
                       fn_or_method=method,
                       plot=False,
                       help=help,
                       output_files=output_files,
                       what_data_gen='NO_DATA_GEN',
                       data_gen=None,
                       data_gen_params=None)
    
    def add_step_fn_no_iter(self, name, fn, help=''):
        self._add_step(name=name, stepImpl=None, fn_or_method=fn,
                       plot=False, help=help, what_data_gen='NO_DATA_GEN',
                       data_gen=None,
                       data_gen_params=None)

    def add_step_fn_plot_no_iter(self, name, fn, help=''):
        self._add_step(name=name, stepImpl=None, fn_or_method=fn,
                       plot=True, help=help, what_data_gen='NO_DATA_GEN',
                       data_gen=None,
                       data_gen_params=None)

    def trace(self, msg, checkcache=True):        
        util.logTrace(hdr=self.hdr, msg=msg, checkcache=checkcache, flag=self.doTrace)

    def warning(self, msg):
        sys.stderr.write("WARNING: %s: %s\n" % (self.hdr, msg))
        sys.stderr.flush()
        
    def debug(self, msg, checkcache=True):
        util.logDebug(hdr=self.hdr, msg=msg, checkcache=checkcache, flag=self.doDebug)

    def get_step_output_files(self, step):
        assert self.args is not None, "arguments not parsed"
        if step.plot:
            return []
        outputdir = self.args.outputdir
        prefix = self.args.prefix
        output_prefix = os.path.join(outputdir, prefix)
        outputs = []
        for suffix in step.output_suffixes:
            outputs.append('_'.join([output_prefix,  suffix]))
        return outputs

    def set_plt(self):
        import matplotlib.pyplot as plt
        self.plt = plt

    def do_plot_step(self, ran_last_step, step):
        do_step = step.redo(self.args)
        if do_step and not self.plt:
            self.set_plt()
        if do_step or (ran_last_step and self.plt):
            return True
        return False

    def doClean(self):
        self.trace("Cleaning output files")
        for step in self.steps:
            output_files = self.get_step_output_files(step)
            for fname in output_files:
                if os.path.exists(fname):
                    os.unlink(fname)
                    self.trace("step=%s Deleted file: %s" % (step.name, fname))
                else:
                    self.trace("step=%s output file: %s doesn't exist" % (step.name, fname))


    def validateConfig(self, config):
        if config is None:
            self.trace("validate config: no config")
            return
        for step in self.steps:
            name = step.name
            if name not in config:
                pass
#                self.warning("validate config: step %s does not have an entry in config" % name)
        for entry in config:
            if entry == 'init':
                continue
            if entry not in self.steps:
                pass
#                self.warning("validate config: entry for %s does not correspond to step" % entry)

    def _read_configfile(self, command_line):
        self._set_args_and_plt(command_line=command_line)
        msg = "init"
        all_step_config = None
        if self.args.config:
            all_step_config = yaml.load(file(self.args.config, 'r'))
            self.validateConfig(all_step_config)
            msg += " loaded config from %s" % self.args.config
        else:
            msg += " no config yml file given on command line."
        self.trace(msg)
        return all_step_config
    
    def run(self, command_line=None):
        if not self.initialized:
            self.all_step_config = self._read_configfile(command_line)
            # set initialized to true after stepsImpl.init is called
            
        init_config = self.get_config(name='init')

        if init_config.clean:
            self.doClean()

        if init_config.gpu >= 0:
            with tf.device('/gpu:%d' % init_config.gpu):
                self.trace("set gpu device to %d" % init_config.gpu)
                self._run_with_device(init_config)
        else:
            self._run_with_device(init_config)

    def _run_with_device(self, init_config):
        if self.session:
            self._run_with_session(init_config)
        else:
            tfConfig = tf.ConfigProto()
            if init_config.gpu >=0:
                tfConfig.gpu_options.allow_growth = True
                tfConfig.log_device_placement = False
                tfConfig.allow_soft_placement=True
            if init_config.cores > 0:
                tfConfig.inter_op_parallelism_threads = config.cores
                tfConfig.intra_op_parallelism_threads = config.cores
            # config.use_per_session_threads=True
            self.session = tf.InteractiveSession(config=tfConfig)
            self.trace("created tf session")
            self._run_with_session(init_config)

    def _run_with_session(self, init_config):
        if not self.initialized:
            if self.stepImpl is not None:
                self.stepImpl.init(config=init_config, pipeline=self)
            if init_config.job_seed > 0:
                np.random.seed(init_config.job_seed)
                random.seed(init_config.job_seed)
                tf.set_random_seed(init_config.job_seed)
            self.initialized=True
        
        self.trace("running pipeline steps")

        step2h5list = {}
        ran_last_step=True
        for step in self.steps:
            msg = str(step)
            if step.plot:
                if self.do_plot_step(ran_last_step, step):
                    msg += " -- running"
                    self.trace(msg)
                    step.run(step2h5list=step2h5list, output_files=None, plot=self.args.plot)
                else:
                    self.trace(msg + " -- skipping plot step")
            else:
                output_files = self.get_step_output_files(step)
                self.manage_step(step, step2h5list, output_files)
                step2h5list[step.name]= output_files

    def manage_step(self, step, step2h5list, output_files):
        all_output_exists = all([os.path.exists(fname) for fname in output_files])
        any_output_exists = any([os.path.exists(fname) for fname in output_files])
        if all_output_exists and not (self.args.redoall or step.redo(self.args)):
            self.trace("step=%s already done, all output exists" % step.name)
        else:
            if any_output_exists and not self.args.force:
                raise Exception("Some of the output files: %s already exist, use --force to overwrite" % step)
            self.trace("running step=%s" % step.name)
            t0 = time.time()
            step.run(step2h5list, output_files)
            step_time = time.time()-t0
            for fname in output_files:
                assert os.path.exists(fname), "step=%s did not create output file: %s" % (step.name, fname)
            self.trace("step=%s output[0]=%s time=%.2f" % (step.name, os.path.basename(output_files[0]), step_time))

    def stop_plots(self):
        self.plt.pause(.1)
        if raw_input('hit enter or q to quit plots (for this step)').lower().strip()=='q':
            return True
        return False
    
    def get_config(self, name):
        '''returns dict of options, command line args override config object, apply to all names
        '''
        class Config(object):
            def __init__(self):
                pass
            
        args = self.args
        config = self.all_step_config
        nameConfig = Config()
        
        if not config:
            self.trace("get_config(%s). No yaml config. All config from args." % name)
        if config and not name in config:
            self.trace("get_config(%s). yaml config present, but no config for step. All config from args." % name)
        if config and name in config:
            config = copy.deepcopy(config[name])
            for ky,val in config.iteritems():
                setattr(nameConfig,ky,val)
        for ky,val in vars(args).iteritems():
            if hasattr(nameConfig, ky):
                self.trace("  get_config(%s). overwrite yaml config key %s with value from args" % \
                           (name, ky))
            setattr(nameConfig,ky,val)
        return nameConfig

    def _set_args_and_plt(self, command_line=None):
        descr = "pipeline for managing sequence of analysis steps. The steps are:\n"
        for step in self.steps:
            descr += '  %s\n' % step.name
        descr += 'The pipeline only re-runs steps if their output files are not present, or if\n'
        descr += 'switches below are used.  A individual step can be re-run with a switch'
        if self.description:
            descr += '\n----------\n%s' % self.description
                                            

        final_parser = argparse.ArgumentParser(parents=[self.parser],
                                               description=descr,
                                               epilog=self.epilog,
                                               formatter_class=argparse.RawDescriptionHelpFormatter)
        if command_line:
            args = final_parser.parse_args(command_line)
        else:
            args = final_parser.parse_args()

        nm2lvl = {'INFO':logging.INFO,
                  'DEBUG':logging.DEBUG}
#                  'ERROR':logging.ERROR,
#                  'CRITICAL':logging.CRITICAL,
#                  'ERROR':logging.ERROR}
        assert args.log in nm2lvl, "log must be one of %s" % str(nm2lvl.keys())
        self.doTrace=nm2lvl[args.log] <= logging.INFO
        self.doDebug=nm2lvl[args.log] <= logging.DEBUG
        if args.plot:
            self.set_plt()
        self.args=args    
