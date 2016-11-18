from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
if os.environ.get('MOCK_TENSORFLOW',False):
    import psmlearn.mock_tensorflow as tf
    from psmlearn.mock_tensorflow import ops
    from psmlearn.mock_tensorflow import gen_nn_ops
else:
    import tensorflow as tf
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import gen_nn_ops

# http://stackoverflow.com/questions/38340791/guided-back-propagation-in-tensor-flow
def guided_backprop_op(fn, relus, X):
    assert len(relus)>0, "no relus"
    oplist = [X] + [op for op in relus]
    next_relu = oplist.pop()
    Dafter = tf.gradients(fn, next_relu)[0]
    Dafter_thresh= tf.to_float(Dafter < 0.0)*Dafter
    print(next_relu)
    while len(oplist):
        last_relu = next_relu
        next_relu = oplist.pop()
        print(next_relu)
        try:
            Dafter = tf.gradients(last_relu, next_relu, grad_ys=Dafter_thresh)[0]
        except:
            print("tf.gradients failed for:\n  last_relu=%s\n  next_relu=%s" % (last_relu, next_relu))
            import IPython
            IPython.embed()
        Dafter_thresh = tf.to_float(Dafter < 0.0)*Dafter
        if Dafter_thresh.get_shape()[0] == 1:
            Dafter_thresh = tf.squeeze(Dafter_thresh,[0])
    return Dafter

### this doesn't work, still get the same unguided map
# https://gist.github.com/falcondai/561d5eec7fed9ebf48751d124a77b087
#@ops.RegisterGradient("GuidedRelu")
#def _GuidedReluGrad(op, grad):
#    return tf.zeros(grad.get_shape())
#    return tf.select(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))

class SaliencyMap(object):
    def __init__(self, model):
        self.model = model
        self.sess = model.sess
        self.img_processed = model.X_processed
        self.img_raw = model.X_placeholder
        self.logit2bprop = {}
        self.logit2gbprop = {}
        g = tf.get_default_graph()
        X = self.img_processed
        for logit in range(model.nnet.logits.get_shape()[1]):
            fn = model.nnet.logits[:,logit]
            self.logit2bprop[logit] = tf.gradients(fn, self.img_processed)[0]
            self.logit2gbprop[logit] = guided_backprop_op(fn, model.nnet.after_relus, X)

    def calc(self, raw_img, logit, fn='gbprop'):
        assert len(raw_img.shape)==4
        assert raw_img.shape[0]==1, "only do batch size of 1"
        assert fn in ['bprop', 'gbprop'], "fn must be one of bprop or gpprop, but it is %s" % fn
        ops = [self.img_processed]
        feed_dict = {self.img_raw:raw_img}
        if fn == 'bprop':
            ops.append(self.logit2bprop[logit])
            img_processed, dimg = self.sess.run(ops, feed_dict=feed_dict)
        elif fn == 'gbprop':
            ops.append(self.logit2gbprop[logit])
            img_processed, dimg = self.sess.run(ops, feed_dict=feed_dict)

        assert len(img_processed.shape)==4
        assert img_processed.shape[0]==1
        img_processed = img_processed[0,:,:,:]
        
        assert len(dimg.shape)==4
        assert dimg.shape[0]==1
        dimg = dimg[0,:,:,:]

        return img_processed, dimg
    
