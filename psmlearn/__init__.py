from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__='v0.1.0'

'''
Machine Learning package for datasets at LCLS/SLAC - photon science (thus the ps in psmlearn)

dataloc - low level access to data locations, for example

dataloc.getProjectDir('xtcav') # gives that project directory, current project directories are
   accbeam  diffraction  ice_water  vgg16  xtcav

get_dataset(project='xtcav')  # high level access, returns a dataset object you can do things with.
 
'''

from . datasets import dataloc
from . datasets import get_dataset
from . import boxutil
from . import visualize
from . import vgg16
from . import regress
from . import h5util
from . import util
from . tensorflow_train import ClassificationTrainer
from . models import Model
from . models import LinearClassifier
from . saliencymaps import SaliencyMap
from . plot import plotRowsLabelSort
from . pipeline import Pipeline
from . tsne import tsne
from . xvalidate import XValidate
