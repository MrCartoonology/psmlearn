from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

############ FILE LOCATION #######################
_MLEARNDIR = '/reg/d/ana01/temp/davidsch/psmlearn'
PSMLEARN_ENV = 'PSMLEARN_BASEDIR'

def getDefalutOutputDir(project, subproject='default_subproject'):
    '''get the default output directory for a subproject.
    Subprojects have a scratch subdirectory. In there a directory with the 
    username is made. This is the default output dir. Ie, 
    psmlearn/xtcav/subproject/scratch/davidsch
    '''
    projectpath = getProjectDir(project)
    subprojectpath = os.path.join(projectpath, subproject)
    assert os.path.exists(subprojectpath), "subproject dir doesn't exist: %s" % subprojectpath
    subprojectScratch = os.path.join(subprojectpath, 'scratch')
    assert os.path.exists(subprojectScratch), "scratch dir doesn't exist: %s" % subprojectScratch
    user = os.environ.get('USER',None)
    assert user is not None, "Could not get 'USER' from environment, can't construct default scratch dir"
    userScratch = os.path.join(subprojectScratch, user)
    if not os.path.exists(userScratch):
        os.mkdir(userScratch)
        print("getDefaultOutputDir: created dir: %s" % userScratch)
    return userScratch

def getBaseDir():
    global _MLEARNDIR
    global PSMLEARN_ENV
    basedir = os.environ.get(PSMLEARN_ENV, _MLEARNDIR)
    assert os.path.exists(basedir), "base dir for mlearning project doesn't exist. Looked for: %s you can set the environment variable %s to specify a different basedir" % (basedir, PSMLEARN_ENV)
    return basedir

def getProjectDir(project):
    '''get project directory, for example, if project=xtcav
    psmlearn/xtcav
    '''
    basedir = getBaseDir()
    projectpath = os.path.join(basedir, project)
    assert os.path.exists(projectpath), "project dir doesn't exist: %s" % projectpath
    return projectpath

def getSubProjectDir(project, subproject):
    '''get subproject directory
    '''
    projectpath = getProjectDir(project)
    subprojectpath = os.path.join(projectpath, subproject)
    assert os.path.exists(subprojectpath), "subproject dir doesn't exist: %s" % subprojectpath
    return subprojectpath

def getProjectFile(project, fname):
    '''return path to a file or directory under a project.
    '''
    projectDir = getProjectDir(projectdir)
    fullfname = os.path.join(projectDir, fname)
    assert os.path.exists(fullfname), "File: %s not found" % fullfname
    return fullfname

def getProjectCalibFile(project, fname):
    projectDir = getProjectDir(project)
    assert os.path.exists(projectDir), "project dir %s doesn't exist" % projectDir
    calibDir = os.path.join(projectDir, 'calib')
    assert os.path.exists(calibDir), "project calib dir %s doesn't exist" % calibDir
    fullfname = os.path.join(calibDir, fname)
    assert os.path.exists(fullfname), "File: %s not found" % fullfname
    return fullfname

def getSubProjectFile(project, subproject, fname):
    subprojectDir = getSubProjectDir(project, subproject)
    fullfname = os.path.join(subprojectDir, fname)
    assert os.path.exists(fullfname), "File: %s not found" % fullfname
    return fullfname
