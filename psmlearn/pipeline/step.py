from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import copy
import inspect
import traceback

# define different kinds of data generators.
WHAT_DATA_GEN=['NO_DATA_GEN',  # there won't be one
               'RAW_DATA_GEN', # go through raw data, gets passed 4 args, randomize, seed, args, num
               'STEP_DATA_GEN', # gets passed 5 args, gets steps2h5list as well
]

class Step(object):
    def __init__(self, name, stepImpl, fn_or_method,
                 what_data_gen = 'NO_DATA_GEN',
                 data_gen=None, data_gen_params={},
                 plot=False, plotFigH=0,
                 pipeline=None, output_suffixes=None):
        assert what_data_gen in WHAT_DATA_GEN, "The what_data_gen must be one of %s" % WHAT_DATA_GEN
        self.name = name
        self.stepImpl=stepImpl
        self.fn_or_method = fn_or_method
        self.what_data_gen = what_data_gen
        self.data_gen = data_gen
        self.data_gen_params = data_gen_params
        self.plot = plot
        self.plotFigH = plotFigH
        self.pipeline = pipeline
        self.output_suffixes = output_suffixes
        if self.what_data_gen != 'NO_DATA_GEN':
            assert self.data_gen, "what_data_gen specifies data_generator, but data_gen arg not set"
            
    def isMethod(self):
        return self.stepImpl != None

    def redo(self, args):
        return getattr(args, self.name)
    
    def __str__(self):
        msg='step=%s plt=%s method=%s figH=%d datagen=%s' % \
            (self.name, self.plot, self.isMethod(), self.plotFigH, self.what_data_gen)
        if self.output_suffixes:
            msg += ' output=%s' % ','.join(self.output_suffixes)
        return msg

    def run(self, step2h5list, output_files, plot=0):
        '''a step is called with a signature depending on how it was added. For instance:
        '''
        kwargs = {}
        data_iter = None
        if self.what_data_gen == 'NO_DATA_GEN':
            pass
        else:
            if self.what_data_gen in ['RAW_DATA_GEN', 'STEP_DATA_GEN']:
                data_gen_params = copy.deepcopy(self.data_gen_params)
                if not ('num' in data_gen_params) and (num>0): 
                    data_gen_params['num']=num
            if self.what_data_gen == 'STEP_DATA_GEN':
                data_gen_params['step2h5list'] = step2h5list
                data_gen_params['pipeline'] = self.pipeline
            data_iter = self.data_gen(**data_gen_params)

        if data_iter:
            kwargs['data_iter'] = data_iter

        config = self.pipeline.get_config(self.name)
        kwargs['config'] = config

        kwargs['pipeline'] = self.pipeline

        if self.plot:
            kwargs['plot'] = plot
            kwargs['plotFigH'] = self.plotFigH
        else:
            kwargs['output_files'] = output_files

        kwargs['step2h5list'] = step2h5list

        argspec = inspect.getargspec(self.fn_or_method)
        args = copy.deepcopy(argspec[0])
        for nm in kwargs.keys():
            assert nm in args, "kwargs constructed for step=%s are: %s, but %s is not in argspec: %s" % (self, kwargs.keys(), nm, argspec)
            args.remove(nm)
        if self.isMethod():
            assert 'self' in args, "step %s is a method, but argspec=%s doesn't have 'self'" % (self, argspec)
        try:
            self.fn_or_method(**kwargs)
        except Exception, exp:
            print(">>>>>>>>>>>>>>>>>>>>")
            traceback.print_exc()
            print("<<<<<<<<<<<<<<<<<<<<")
            for fname in output_files:
                if os.path.exists(fname):
                    sys.stderr.write("WARNING: step: %s failed, deleting output file: %s\n" % (self, fname))
                    os.unlink(fname)
            raise exp
