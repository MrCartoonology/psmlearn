import numpy as np
import tensorflow as tf

def make_dict_from_param_list(paramlist):
    res = {}
    for param in paramlist:
        res[param.name]=param
    return res

class Rule(object):
    def __init__(self):
        pass
    def isNaive(self):
        return False
    def isEps(self):
        return False
    def isBeta(self):
        return False
    
class Naive(Rule):
    def __init__(self):
        pass
    def isNaive(self):
        return True
    
class EpsVariant(Rule):
    def __init__(self, eps=1e-3):
        assert eps > 0
        self.eps=eps
    def isEps(self):
        return True
    
class BetaVariant(Rule):
    def __init__(self, beta=-1e-3):
        assert beta < 0
        self.beta = beta
        self.alpha = 1.0 - beta
        assert np.abs(self.alpha + self.beta - 1.0) < 1e-7
        
    def isBeta(self):
        return True
    
def relprop_from_scalar_fn(X, W, fn, rule, debug=False, name=''):
    '''compute relevance propagation from a scalar function.
    ARSG:
      X  - 1D tensor
      W  - 1D array, same dim as X
     fn  - scalar function value
    '''
    assert isinstance(W, np.ndarray)
    assert len(W.shape)==1
    N = len(W)
    assert X.get_shape()==(N,)

    Z = tf.mul(X, W)
    assert Z.get_shape()==(N,)
    
    if rule.isNaive():
        Z_norm = tf.reduce_sum(Z)
        assert Z_norm.get_shape()==()
        
        R = Z / Z_norm
        assert R.get_shape()==(N,)
        
        R *= fn
        assert R.get_shape()==(N,)

        if debug:
            return R, Z_norm, Z
        else:
            return R

    elif rule.isEps():
        Z_norm = tf.reduce_sum(Z)
        assert Z_norm.get_shape()==()

        Z_norm += rule.eps * tf.sign(Z_norm)
        assert Z_norm.get_shape()==()

        R = (Z / Z_norm ) * fn
        assert R.get_shape()==(N,)

        if debug:
            return R, Z_norm, Z
        else:
            print("nm=%s relprop_from_scalar_fn(X=%s W=%s)=%s" % (name, X.get_shape(), W.shape, R.get_shape()))
            return R
        
    elif rule.isBeta():
        Zpos = tf.nn.relu(Z)
        assert Zpos.get_shape()==(N,)

        Zneg = -1.0 * tf.nn.relu(-1.0 * Z)
        assert Zneg.get_shape()==(N,)

        Zpos_sum = tf.reduce_sum(Zpos)
        assert Zpos_sum.get_shape()==()

        Zneg_sum = tf.reduce_sum(Zneg)
        assert Zneg_sum.get_shape()==()

        R = tf.zeros_like(Z)
        assert R.get_shape()==(N,)

        def zeros(): return tf.zeros_like(Z)
        def alpha_term(): return Zpos * (rule.alpha / Zpos_sum)
        def beta_term(): return Zneg * (rule.beta / Zneg_sum) 
        R += tf.cond(Zpos_sum > 0,
                     alpha_term,
                     zeros)
        R += tf.cond(Zneg_sum < 0,
                     beta_term,
                     zeros)
        R *= fn
        assert R.get_shape()==(N,)

        if debug:
            return R, Zneg_sum, Zpos_sum, Zneg, Zpos, Z
        else:
            return R
    raise Exception("unknonwn rule")

def relprop_from_fully_connected(X, W, R, rule, debug=False, name=''):
    '''compute relevance propagation from a scalar function.
    ARSG:
      X  - 1 x N tensor
      W  - N x M tensor/matrix
      R  - M x 1 tensor
    '''
    assert len(X.get_shape())==1, "X=%s" % X
    assert len(R.get_shape())==1, "R=%s" % R
    N    = int(X.get_shape()[0])
    M    = int(R.get_shape()[0])
    assert W.get_shape()==(N,M)
    
    X    = tf.reshape(X, (N,1))
    Z    = tf.mul(X, W)    
    assert Z.get_shape()==(N,M)
    
    if rule.isNaive():
        Z_norm = tf.reduce_sum(Z,0)
        assert Z_norm.get_shape()==(M,)

        R_norm = R / Z_norm
        assert R_norm.get_shape()==(M,)

        Rnew = tf.matmul(Z, tf.reshape(R_norm,(M,1)))
        Rnew = tf.reshape(Rnew, (N,))
                         
        if debug:
            return Rnew, R_norm, Z
        else:
            return Rnew

    elif rule.isEps():
        Z_norm = tf.reduce_sum(Z,0)
        assert Z_norm.get_shape()==(M,)

        Z_norm += rule.eps * tf.sign(Z_norm)
        assert Z_norm.get_shape()==(M,)

        R_norm = tf.div(R, Z_norm)
        assert R_norm.get_shape()==(M,)

        Rnew = tf.reshape(tf.matmul(Z, tf.reshape(R_norm,[M,1])),(N,))
        assert Rnew.get_shape()==(N,)

        if debug:
            return Rnew, R_norm, Z
        else:
            print("nm=%s relprop_from_fully_connected(X=%s,W=%s,R=%s)=%s" % (name, X.get_shape(), W.get_shape(), R.get_shape(), Rnew.get_shape()))
            return Rnew

    elif rule.isBeta():
        Zpos = tf.nn.relu(Z)
        assert Zpos.get_shape()==(N,M)
        
        Zneg = -1.0 * tf.nn.relu(-1.0 * Z)
        assert Zneg.get_shape()==(N,M)

        Zpos_sum = tf.reduce_sum(Zpos,0)
        Zneg_sum = tf.reduce_sum(Zneg,0)
        assert Zpos_sum.get_shape()==(M,)
        assert Zneg_sum.get_shape()==(M,)

        pos_factor = tf.maximum(Zpos_sum, 1e-10)
        assert pos_factor.get_shape()==(M,)
        Zpos_norm = rule.alpha * ( Zpos / tf.reshape(pos_factor,(1,M)) )
        assert Zpos_norm.get_shape()==(N,M)

        neg_factor = tf.minimum(Zneg_sum, -1e-10)
        assert neg_factor.get_shape()==(M,)
        Zneg_norm = rule.beta * ( Zneg / tf.reshape(neg_factor,(1,M)) )
        assert Zneg_norm.get_shape()==(N,M)

        Znorm = Zpos_norm + Zneg_norm
        assert Znorm.get_shape()==(N,M)

        Rnew = tf.reshape(tf.matmul(Znorm, tf.reshape(R, (M,1))), (N,))
        assert Rnew.get_shape() == (N,)
        if debug:
            return Rnew, Znorm, Z
        else:
            return Rnew
    raise Exception("unknonwn rule")

def relprop_from_max(max_input, max_output, R, name=''):
    assert max_input.get_shape()[0].value == None, "expect batch ops"
    assert max_output.get_shape()[0].value == None, "expect batch ops"
#    assert max_output.get_shape()[1:]==R.get_shape(), "max_output.shape=%r != R.shape=%r" % (max_output.get_shape()[1:], R.get_shape())
    assert R.get_shape()[0]==1
    res = tf.gradients(max_output, max_input, R)[0][0]
    print("nm=%s relprop_from_max(%s,%s,%s)=%s" % (name, max_output.get_shape(), max_input.get_shape(), R.get_shape(), res.get_shape()))
    return res

def relprop_from_conv(X,Y,R,rule,K,strides,padding='SAME',data_format="NHWC", name=''):
    X2 = tf.mul(X,X)
    X2 *= 0.5
    X2conv = tf.nn.conv2d(X2, K, strides, padding, data_format=data_format)
    if Y is None:
        Y = tf.nn.conv2d(X, K, strides, padding, data_format=data_format)
    if rule.isBeta():
        raise Exception("relprop_from_conv not implemented for beta rule")
    if rule.isNaive():
        grad_ys = R / Y
    elif rule.isEps():
        Ynorm = Y + rule.eps * tf.sign(Y)
        grad_ys = R / Ynorm
    res = tf.gradients(X2conv, X, grad_ys)[0][0]
    print("nm=%s relprop_from_conv(X=%s,R=%s)=%s" % (name, X.get_shape(), R.get_shape(), res.get_shape()))
    return res

class LRPVgg16(object):
    '''compute relevance back propagation for vgg16
    for each of the logits. 
    ARGS:
    vgg16 - model from psmlearn
    W_fc2_logits - numpy array, matrix of weights with 4096 rows, and N columns,
    one column for each logit
    '''
    def __init__(self, vgg16, W_fc2_logits, rule):

        assert isinstance(W_fc2_logits, np.ndarray)
        assert len(W_fc2_logits.shape)==2
        assert W_fc2_logits.shape[0]==4096
        
        self.W_fc2_logits_arr=W_fc2_logits
        self.rule = rule
        self.imgs_pl = vgg16.imgs

        self.vggParams = make_dict_from_param_list(vgg16.parameters)
        
        num_logits = W_fc2_logits.shape[1]
        self.logit2ops = {}
        for logit in [3]: #range(num_logits):
            ww_fc2_logit = W_fc2_logits.T[logit,:].copy()
            assert ww_fc2_logit.shape == (4096,)
            R_fc2 = relprop_from_scalar_fn(X=vgg16.fc2[0],
                                           W=ww_fc2_logit,
                                           fn=100.0, rule=self.rule, name='R_fc2')
            assert R_fc2.get_shape() == (4096,), "R_fc2=%s" % R_fc2
            
            ww_fc1_fc2 = self.vggParams['fc2/weights:0']
            assert ww_fc1_fc2.get_shape()==(4096, 4096) 
            R_fc1 = relprop_from_fully_connected(X=vgg16.fc1[0],
                                                 W=ww_fc1_fc2,
                                                 R=R_fc2,
                                                 rule=self.rule, name='R_fc1')
            assert R_fc1.get_shape() == (4096,), "R_fc1=%s" % R_fc1
            ww_pool5_fc2 = self.vggParams['fc1/weights:0']
            assert ww_pool5_fc2.get_shape()==(25088, 4096)
            pool5_flat = tf.reshape(vgg16.pool5,[-1,25088])
            
            R_pool5flat = relprop_from_fully_connected(X=pool5_flat[0],
                                                       W=ww_pool5_fc2,
                                                       R=R_fc1,
                                                       rule=self.rule, name='R_pool5flat')
            assert R_pool5flat.get_shape()==(25088,), "R_pool5flat=%s" % R_pool5flat

            ### pool5 conv5
            R_pool5 = tf.reshape(R_pool5flat, (1,7,7,512))

            R_conv5_3 = relprop_from_max(max_input=vgg16.conv5_3,
                                         max_output=vgg16.pool5,
                                         R=R_pool5, name='R_conv5_3')

            K= self.vggParams['conv5_3/weights:0']
            assert K.get_shape()==(3,3,512,512)
            R_conv5_2 = relprop_from_conv(X = vgg16.conv5_2,
                                          Y = vgg16.conv_input_to_conv5_3,
                                          R = R_conv5_3, rule=self.rule, K=K, strides=[1,1,1,1],
                                          padding='SAME', name='R_conv5_2')

            K= self.vggParams['conv5_2/weights:0']
            assert K.get_shape()==(3,3,512,512)
            R_conv5_1 = relprop_from_conv(X = vgg16.conv5_1,
                                          Y = vgg16.conv_input_to_conv5_2,
                                          R = R_conv5_2, rule=self.rule, K=K, strides=[1,1,1,1],
                                          padding='SAME', name='R_conv5_1')

            ### pool4 conv4
            K = self.vggParams['conv5_1/weights:0']
            assert K.get_shape()==(3,3,512,512)
            R_pool4 = relprop_from_conv(X = vgg16.pool4,
                                        Y = vgg16.conv_input_to_conv5_1,
                                        R = R_conv5_1, rule = self.rule, K=K, strides=[1,1,1,1],
                                        padding='SAME', name='R_pool4')
            R_pool4 = tf.expand_dims(R_pool4, 0)
            R_conv4_3 = relprop_from_max(max_input=vgg16.conv4_3,
                                         max_output=vgg16.pool4,
                                         R=R_pool4, name='R_conv4_3')

            K = self.vggParams['conv4_3/weights:0']
            assert K.get_shape()==(3,3,512,512)
            R_conv4_2 = relprop_from_conv(X=vgg16.conv4_2,
                                          Y=vgg16.conv_input_to_conv4_3,
                                          R=R_conv4_3, rule=self.rule, K=K, strides=[1,1,1,1],
                                          padding='SAME', name='R_conv4_2')

            K = self.vggParams['conv4_2/weights:0']
            assert K.get_shape()==(3,3,512,512)
            R_conv4_1 = relprop_from_conv(X=vgg16.conv4_1,
                                          Y=vgg16.conv_input_to_conv4_2,
                                          R=R_conv4_2, rule=self.rule, K=K, strides=[1,1,1,1],
                                          padding='SAME', name='R_conv4_1')

            ### pool3 conv3
            K = self.vggParams['conv4_1/weights:0']
            assert K.get_shape()==(3,3,256,512)
            R_pool3 = relprop_from_conv(X = vgg16.pool3,
                                        Y = vgg16.conv_input_to_conv4_1,
                                        R = R_conv4_1, rule = self.rule, K=K, strides=[1,1,1,1],
                                        padding='SAME', name='R_pool3')
            R_pool3 = tf.expand_dims(R_pool3, 0)
            R_conv3_3 = relprop_from_max(max_input=vgg16.conv3_3,
                                         max_output=vgg16.pool3,
                                         R=R_pool3, name='R_conv3_3')

            K = self.vggParams['conv3_3/weights:0']
            assert K.get_shape()==(3,3,256,256)
            R_conv3_2 = relprop_from_conv(X=vgg16.conv3_2,
                                          Y=vgg16.conv_input_to_conv3_3,
                                          R=R_conv3_3, rule=self.rule, K=K, strides=[1,1,1,1],
                                          padding='SAME', name='R_conv3_2')

            K = self.vggParams['conv3_2/weights:0']
            assert K.get_shape()==(3,3,256,256)
            R_conv3_1 = relprop_from_conv(X=vgg16.conv3_1,
                                          Y=vgg16.conv_input_to_conv3_2,
                                          R=R_conv3_2, rule=self.rule, K=K, strides=[1,1,1,1],
                                          padding='SAME', name='R_conv3_1')

                                          
            ### pool2 conv2
            K = self.vggParams['conv3_1/weights:0']
            assert K.get_shape()==(3,3,128,256)
            R_pool2 = relprop_from_conv(X = vgg16.pool2,
                                        Y = vgg16.conv_input_to_conv3_1,
                                        R = R_conv3_1, rule = self.rule, K=K, strides=[1,1,1,1],
                                        padding='SAME', name='R_pool2')
            R_pool2 = tf.expand_dims(R_pool2, 0)
            R_conv2_2 = relprop_from_max(max_input=vgg16.conv2_2,
                                         max_output=vgg16.pool2,
                                         R=R_pool2, name='R_conv2_2')

            K = self.vggParams['conv2_2/weights:0']
            assert K.get_shape()==(3,3,128,128)
            R_conv2_1 = relprop_from_conv(X=vgg16.conv2_1,
                                          Y=vgg16.conv_input_to_conv2_2,
                                          R=R_conv2_2, rule=self.rule, K=K, strides=[1,1,1,1],
                                          padding='SAME', name='R_conv2_1')

            ### pool1 conv1
            K = self.vggParams['conv2_1/weights:0']
            assert K.get_shape()==(3,3,64,128)
            R_pool1 = relprop_from_conv(X = vgg16.pool1,
                                        Y = vgg16.conv_input_to_conv2_1,
                                        R = R_conv2_1, rule = self.rule, K=K, strides=[1,1,1,1],
                                        padding='SAME', name='R_pool1')
            R_pool1 = tf.expand_dims(R_pool1, 0)
            R_conv1_2 = relprop_from_max(max_input=vgg16.conv1_2,
                                         max_output=vgg16.pool1,
                                         R=R_pool1, name='R_conv1_2')

            K = self.vggParams['conv1_2/weights:0']
            assert K.get_shape()==(3,3,64,64)
            R_conv1_1 = relprop_from_conv(X=vgg16.conv1_1,
                                          Y=vgg16.conv_input_to_conv1_2,
                                          R=R_conv1_2, rule=self.rule, K=K, strides=[1,1,1,1],
                                          padding='SAME', name='R_conv1_1')


            #### imgs
            K = self.vggParams['conv1_1/weights:0']
            assert K.get_shape()==(3,3,3,64)
            R_img = relprop_from_conv(X=vgg16.imgs,
                                      Y=vgg16.conv_input_to_conv1_1,
                                      R=R_conv1_1, rule=self.rule, K=K, strides=[1,1,1,1],
                                      padding='SAME', name='R_imgs')
                                          
            self.logit2ops[logit]=[('R_fc2',R_fc2),
                                   ('R_fc1',R_fc1),
                                   ('R_pool5flat',R_pool5flat),
                                   ('R_pool5',R_pool5),
                                   ('R_conv5_3',R_conv5_3),
                                   ('R_conv5_2',R_conv5_2),
                                   ('R_conv5_1',R_conv5_1),
                                   ('R_pool4',R_pool4),
                                   ('R_conv4_3',R_conv4_3),
                                   ('R_conv4_2',R_conv4_2),
                                   ('R_conv4_1',R_conv4_1),
                                   ('R_pool3',R_pool3),
                                   ('R_conv3_2',R_conv3_2),
                                   ('R_conv3_1',R_conv3_1),
                                   ('R_pool2',R_pool2),
                                   ('R_conv2_2',R_conv2_2),
                                   ('R_conv2_1',R_conv2_1),
                                   ('R_img',R_img),
            ]
                                   
                                   
    def compute(self, sess, logit, batch_img):
        name_op_list = self.logit2ops[logit]
        names = [name_op[0] for name_op in name_op_list]
        ops = [name_op[1] for name_op in name_op_list]
        assert batch_img.shape==(1,224,224,3)
        arrs = sess.run(ops, feed_dict={self.imgs_pl:batch_img})
        return zip(names, arrs)
    
                        
