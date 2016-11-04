import os
import numpy as np
import h5py

def get_examples_file(fname):
    testdir=os.path.split(os.path.abspath(__file__))[0]
    pkgdir = os.path.split(testdir)[0]
    exampledir = os.path.join(pkgdir, 'examples')
    assert os.path.exists(exampledir), "directory %s doesn't exist" % exampledir
    script = os.path.join(exampledir, fname)
    assert os.path.exists(script), "script file %s doesn't exist" % script
    return script

def basic_pipeline_commands_test(script, name):
    prefix = 'test_pipeline_%s' % name
    ret_hdr_cmds = [(0, '--%s CLEANING ANY OLD--' % name, 'python %s --clean %s' % (script, prefix)),
                    (0, '--%s CREATING FROM SCRATCH--' % name, 'python %s %s' % (script, prefix)),
                    (-1, '--%s FAIL TO REDO--' % name, 'python %s --redoall %s' % (script, prefix)),
                    (0, '--%s FORCE REDO--' % name, 'python %s --force --redoall %s' % (script, prefix)),
    ]
    for ret_hdr_cmd in ret_hdr_cmds:
        ret,hdr,cmd = ret_hdr_cmd
        print(hdr)
        if ret == 0:
            assert 0 == os.system(cmd), "command failure: %s" % cmd
        else:
            os.system(cmd)
    return prefix

def h5checks_test(h5checks, prefix):
    for h5check in h5checks:
        fname = prefix + '_' + h5check['filename'] + '.h5'
        assert os.path.exists(fname), "h5checks_test, fname=%s doesn't exist" % fname
        h5 = h5py.File(fname,'r')
        for dset in h5check['keys']:
            assert dset in h5, "h5checks_test, fname=%s doesn't have key %s" % (fname, dset)                      

def test_pipeline_basic():
    script = get_examples_file('pipeline-basic.py')
    prefix = basic_pipeline_commands_test(script, 'basic')
    h5checks_test([{'filename':'stepA', 'keys':['X','Y']}], prefix)
    fname = '%s_stepA.h5' % prefix
    h5=h5py.File(fname, 'r')
    X,Y=h5['X'][:], h5['Y'][:]
    ansX = np.array([1,2,3])
    ansY = np.array([3,1,3])
    assert np.all(np.equal(ansX,X)), "basic pipline test - fname=%s dset X=%s != expected=%s" % (fname, X, ansX)
    assert np.all(np.equal(ansY,Y)), "basic pipline test - fname=%s dset Y=%s != expected=%s" % (fname, Y, ansY)
    
def test_pipeline_vgg16():
    script = get_examples_file('pipeline-vgg16.py')
    prefix=basic_pipeline_commands_test(script, 'vgg16')
    h5checks = [{'filename':'data_stats','keys':['channel_mean','datalen']},
                {'filename':'model_layers','keys':['fc1','fc2','meta']},
                {'filename':'tsne_imgs','keys':['labels','meta','tsne_imgs']},
                {'filename':'tsne_cws','keys':['labels','meta','tsne_cws']},
    ]
    h5checks_test(h5checks, prefix)

    
if __name__ == '__main__':
    test_pipeline_basic()
    test_pipeline_vgg16()
