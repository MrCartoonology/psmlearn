from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class MockObject(object):
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self,type, value, traceback):
        pass

    def get_shape(self):
        return [0,100,100]
    
    def __getattr__(self,name):
        print("######## %s ############" % name)
        if name == '__trunc__':
            return 0
        self.name=MockObject()
        return self.name

    def conv2d(*args, **kwargs):
        return MockObject()

    def __call__(*args, **kwargs):
        return MockObject()

    def __getitem__(*args, **kwargs):
        return MockObject()
    
def ConfigProto():
    return MockObject()

def constant(*args, **kwargs):
    return None

def matmul(*args, **kwargs):
    return None

def Session(*args, **kwargs):
    return MockObject()

def device(*args, **kwargs):
    return MockObject()

def reset_default_graph():
    pass

def InteractiveSession(*args, **kwarsg):
    return MockObject()

def float32():
    return MockObject()

def ops():
    return MockObject()

def gen_nn_ops():
    return MockObject()

def placeholder(*args, **kwargs):
    return MockObject()

def name_scope(*args, **kwargs):
    return MockObject()

def Variable(*args, **kwargs):
    return MockObject()

def truncated_normal(*args, **kwargs):
    return MockObject()

def reshape(*args, **kwargs):
    return MockObject()

def initialize_variables(*args, **kwargs):
    return MockObject()

def _nn(*args, **kwargs):
    return MockObject()

def set_random_seed(*args, **kwargs):
    return MockObject()

nn = _nn()
