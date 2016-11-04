from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import h5py
import random
import tensorflow as tf
from psmlearn.pipeline import Pipeline
from psmlearn import h5util

##########
def data_gen(randomize=False, seed=109, args=None, num=10,
             num_labels=3, feats=['imgs','fvec'], labels_2_return=['A','B']):
    '''
    a data generator returns an iterator that goes once through a datasource.
    The parameters that pipeline will use are
    
    randomize - shuffle the data before iteratoring (for batch training)
    seed      - randome number generator seed
    args      - command line arguments as parsed by pipeline

    The iterator must return 3 items:
      1) list of featurs, i.e, [image, bld_vector]
      2) list of labels, i.e, [categoryA, regressValueB], or None or empty list if no labels
      3) meta - anything user wants, file/row to keep track of data, usually a dictionary, 
                should be able to write the dict to hdf5 - appending each ky/value to growing
                datasets.
                meta can be none as well.
    '''
    np.random.seed(seed)
    random.seed(seed)
    data_meta = range(num)
    data = {'imgs':[np.random.rand(100,100) for k in range(num)],
            'fvec': [np.random.rand(10) for k in range(num)]
            }
    labels = {'A':[mm % num_labels for mm in data_meta],
            'B':[np.random.rand() for mm in data_meta]
    }
    if randomize:
        random.shuffle(data_meta)

    for meta in data_meta:
        sample_features = [data[ky][meta] for ky in feats]
        sample_labels = [labels[ky][meta] for ky in labels_2_return]
        yield sample_features, sample_labels, meta

#########
def plot_orig_data(data_iter, config, plot, plotFigH, pipeline, step2h5list):
    plt = pipeline.plt    
    plt.figure(plotFigH)
    plt.clf()
    aggregate = None
    for X,Y,meta in data_iter:
        img=X[0]
        if plot >= 2:
            plt.imshow(img, interpolation='none')
            plt.title("original data: meta=%s" % str(meta))
            plt.pause(.1)
            if 'q' == raw_input('hit enter, or q to quit').strip().lower():
                break
        if aggregate is None:
            aggregate = img.copy()
        else:
            aggregate += img
    plt.clf()
    plt.imshow(aggregate, interpolation='none')
    plt.title('agg')
    plt.pause(.1)
    raw_input('hit enter')
    
def stepA(data_iter, config, pipeline, step2h5list, output_files):
    h5util.write_to_h5(output_files[0],{'X':[1,2,3], 'Y':[3,1,3]})

def plot_stepA(data_iter, config, pipeline, plot, plotFigH, step2h5list):
    plt = pipeline.plt
    plt.figure(plotFigH)
    stepAh5 = step2h5list['stepA'][0]
    X,Y = stepAh5['X'][:], stepAh5['Y'][:]
    plt.plot(X,Y)
    plt.title("step A")
    plt.pause(.1)
    raw_input('hit enter')
                   
pipeline = Pipeline(outputdir='.',
                    default_data_gen=data_gen,
                    description='basic pipeline',
                    epilog='do this.')

pipeline.add_step_fn_plot(name='plot_orig_data', fn=plot_orig_data)
pipeline.add_step_fn(name='stepA', fn=stepA)
pipeline.add_step_fn_plot(name='plot_stepA', fn=plot_stepA)
pipeline.run()


