from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

########## IMPORTS ###############
## standard imports
import os
import sys
import glob

# external package imports
import scipy.io as sio

# this package imports
from . import dataloc


class BeamData(object):
    def __init__(self, preprocess, datadir='data', subbkg=False, filenums=[1,2], prefix=None, force=False, nn=0):
        self.mat_filenums = filenums
        self.subbkg = subbkg

        self.rawdata = None
        self.labels = {'yag':None, 'vcc':None}
        self.rawROIs = {'yag':{}, 'vcc':{}}

        self.processed = {'yag':None, 'vcc':None}
        self.processedBoxes = {'yag':{}, 'vcc':{}}
        self.processedROIs = {'yag':{}, 'vcc':{}}
        self.meta = {'filenums':None}

        self.nn = nn
        self.datadir = datadir

    def get_number_of_samples(self):
        assert self.rawdata is not None, "data not loaded"
        return len(self.processed['yag'])

    def get_image(self, nm, row):
        assert self.processed[nm] is not None, "data not loaded"
        return self.processed[nm][row]
        
    def names(self):
        return ['yag','vcc']

    def subBackgroundIfRequired(self, yag, vcc, matdata):
        def sub(img,bkg):
            img2 = img.astype(np.float32)
            img2 -= bkg.astype(np.float32)
            img2 = np.maximum(0.0, img2)
            return img2.astype(img.dtype)
        
        if not self.subbkg: return yag, vcc
        if 'yagbkg' in matdata.keys():
            yag = sub(yag, matdata['yagbkg'])
        if 'vccbkg' in matdata.keys():
            vcc = sub(vcc, matdata['vccbkg'])
        return yag, vcc

    def check_processed_means(self):
        for nm in self.names():
            mean = np.mean(self.processed[nm][:])
            if np.abs(mean) >= .1:
                sys.stderr.write("WARNING: check_processed_means: mean=%.1f for %s\n" % (mean,nm))
 
    def loadall(self, reload=False, plotextra=False):
        ######## helpers #######
        def getLabel(bx):
            if len(bx)==0:
                return 0
            if len(bx[0,:])==4:
                return 1
            return -1
            
        #########################
        if self.rawdata is not None and not reload:
            return

        data = makeEmptyBeamLocData()
        assert os.path.exists(self.datadir), "datadir %s doesn't exist" % self.datadir
        
        sampleIdx = -1
        for filenum in self.mat_filenums:
            mat_fname = os.path.join(self.datadir, 'labeledimg%d.mat' % filenum)
            mat = sio.loadmat(mat_fname)
            matdata = parseBeamMat(mat)
            print("file %d has %d entries" % (filenum, len(matdata['yagImg'])))
        self.rawdata = data

        self.check_rawdata()
        for nm in ['yag','vcc']:
            self.labels[nm] = np.array(self.rawdata[nm]['label'])
        self.meta['filenums'] = np.array(self.rawdata['yag']['file'])
        self.processed['yag'] = self.preprocess.preProcessAll(nm='yag', 
                                                              imgs=self.rawdata['yag']['img'])
        self.processed['vcc'] = self.preprocess.preProcessAll(nm='vcc', 
                                                              imgs=self.rawdata['vcc']['img'])
        self.calcProcessedBoxes()
        self.extractProcessesROIs()

    def viewbkg(self):
        plt.figure(30, figsize=(12,12))
        plt.clf()
        assert self.rawdata is not None, "data not loaded"
        fnum = 4
        for ky,log,subplt in zip(['yagbkg', 'vccbkg', 'yagbeam', 'vccbeam'],
                                 [True, False, True, True],
                                 [1,2,3,4]):
            kyData = self.rawdata.get(ky,None)
            if kyData is None:
                print("beamdata: viewbkg - no %s in data"  % ky)
                continue
            if kyData.keys() != [fnum]:
                print("Warning: code is hard coded for file %d, but data has %s for files %s" % 
                      (fnum, ky, kyData.keys()))
            plt.subplot(2,2,subplt)
            if log:
                plt.imshow(np.log(1.0+np.maximum(0.0,kyData[fnum])), interpolation='none')
                plt.title("%s (log scale) file %d" % (ky, fnum))
            else:
                plt.imshow(kyData[fnum], interpolation='none')
                plt.title("%s file %d" % (ky, fnum))
            plt.colorbar()
        plt.pause(.1)
        raw_input("all done with viewbkg")

    def calcProcessedBoxes(self):
        raw_shape = {'yag':self.rawdata['yag']['img'][0].shape,
                     'vcc':self.rawdata['vcc']['img'][0].shape}
        processed_shape = {'yag':self.processed['yag'][0].shape,
                           'vcc':self.processed['vcc'][0].shape}
        scale_factors = {}
        for nm in ['yag','vcc']:
            scale_factors[nm]=tuple([processed_shape[nm][ii]/float(raw_shape[nm][ii]) for ii in range(2)])

        for nm in ['yag','vcc']:
            rawboxs = self.rawdata[nm]['box']
            yscale, xscale = scale_factors[nm]
            for sampleIdx in range(len(self.labels[nm])):
                if 0==self.labels[nm][sampleIdx]: continue
                ymin,ymax,xmin,xmax = rawboxs[sampleIdx][0,:]  # y y x x
                processedbx = np.array([ymin*yscale, ymax*yscale, xmin*xscale, xmax*xscale], dtype=np.float32)
                self.processedBoxes[nm][sampleIdx]=processedbx

    def extractProcessesROIs(self):
        for nm in ['yag','vcc']:
            for sampleIdx in range(len(self.labels[nm])):
                if 0==self.labels[nm][sampleIdx]: continue
                img = self.processed[nm][sampleIdx]
                bx = self.processedBoxes[nm][sampleIdx]
                ymin, ymax, xmin, xmax = map(int,bx[:])
                self.processedROIs[nm][sampleIdx] = img[ymin:ymax,xmin:xmax].copy()
                
    def check_rawdata(self):
        filenum = np.array(self.rawdata['yag']['file']).astype(np.int32)
        assert np.all(filenum==np.array(self.rawdata['vcc']['file']).astype(np.int32))

    def copy_to_h5(self, h5out):
        NN = len(self.labels['yag'])
        h5out['filenums']=self.meta['filenums']
        for nm in ['yag','vcc']:
            h5out['label_%s'%nm] = self.labels[nm]
            h5out['%sbox' % nm] = np.zeros((NN,4), dtype=np.float32)
            for row, bx in self.processedBoxes[nm].iteritems():
                h5out['%sbox' % nm][row,:] = bx[:]
            h5out['%s_preprocessed_imgs'%nm] = self.processed[nm]

    def get_img_box_roi(self, nm, stage, sampleIdx):
        hasBox = self.labels[nm][sampleIdx]==1
        box = None
        roi = None
        if stage == 'raw':
            img = self.rawdata[nm]['img'][sampleIdx]
            if hasBox:
                box = self.rawdata[nm]['box'][sampleIdx][0,:]
                roi = self.rawROIs[nm][sampleIdx]
        elif stage == 'processed':
            img = self.processed[nm][sampleIdx]
            if hasBox:
                box = self.processedBoxes[nm][sampleIdx]
                roi = self.processedROIs[nm][sampleIdx]
        return img, box, roi

    def plotROIs(self, figH=1, save=0, view=True):
        if (save<=0) and (not view):
            sys.stderr.write("WARNING: plotROIs called with save<=0 and view=False, returning early\n")
            return
        for nm in ['yag','vcc']:
            keys = self.rawROIs[nm].keys()
            keys.sort()
            rawOnGrid = psmlearn.make_grid([self.rawROIs[nm][k] for k in keys])
            processedOnGrid = psmlearn.make_grid([self.processedROIs[nm][k] for k in keys])
            fig = plt.figure(figH, figsize=(18,12))
            plt.clf()
            plt.subplot(1,2,1)
            plt.imshow(rawOnGrid, interpolation='none')
            plt.colorbar()
            plt.title("-- raw %s --" % nm)

            plt.subplot(1,2,2)
            plt.imshow(processedOnGrid, interpolation='none')
            plt.colorbar()
            plt.title("-- processed %s --" % nm)

            plt.figtext(0.05, 0.95,"pre-processing Alg: %s" % self.preprocess.alg, fontsize='large', fontweight='bold')
            plt.pause(.1)
            if save>0:
                fname = '%s-%s-rois.png' % (self.prefix, nm)
                assert (not os.path.exists(fname)) or self.force, "Filename: %s exists, use --force to overwrite" % fname
                fig.savefig(fname)
            if view:
                raw_input('hit enter')

    def plots(self, figH=2, do_sigprocess=False, save=False, view=True):
        if (save<=0) and (not view):
            sys.stderr.write("WARNING: plots called with save<=0 and view=False, returning early\n")
            return
        NN = len(self.labels['yag'])
        fig = plt.figure(figH, figsize=(18, 12))
        plot_order = range(NN)
        random.shuffle(plot_order)
        user_quit = False
        sp_hits={'yag_raw':0, 'yag_processed':0, 'vcc_raw':0, 'vcc_processed':0}
        sp_misses={'yag_raw':0, 'yag_processed':0, 'vcc_raw':0, 'vcc_processed':0}
        for plotNumber, sampleIdx in enumerate(plot_order):
            if user_quit: break
            plt.clf()
            fileNum = self.meta['filenums'][sampleIdx]
            subplt = 0
            msg = 'sample=%d file=%d white cross (sigpross sol) pre-alg=%s' % (sampleIdx, fileNum, self.preprocess.alg)
            for nm in ['yag','vcc']:
                for stage in ['raw', 'processed']:
                    subplt += 1
                    img, box, roi = self.get_img_box_roi(nm, stage, sampleIdx)
                    plt.subplot(2,2,subplt)
                    plt.imshow(img, interpolation='none')
                    spsol='no bx'
                    do_pause = False
                    if 1==self.labels[nm][sampleIdx]:
                        if do_sigprocess:
                            sig_row, sig_col = signal_processing.signal_processing_solution(img)
                            plt.plot(sig_col, sig_row, 'w+', mew=4, ms=8)
                            sp_hit = boxutil.in_box(box, y=sig_row, x=sig_col)
                            if sp_hit:
                                sp_hits['%s_%s' % (nm,stage)] += 1
                                spsol='sp=hit!'
                            else:
                                sp_misses['%s_%s' % (nm,stage)] += 1
                                spsol = 'sp=miss:('
                                if stage == 'raw':
                                    do_pause = True
                        boxutil.plot_box(box, plt)
                        plt.xlim([0,img.shape[1]])
                        plt.ylim([0,img.shape[0]])
                    plt.title('%s %s %s' % (nm, stage, spsol))
                    plt.colorbar()
                    if do_pause:
                        plt.pause(.1)
                        raw_input("hit enter miss row=%.1f col=%.1f" % (sig_row, sig_col))
            plt.figtext(0.05, 0.95, msg, fontsize='large', fontweight='bold')
            plt.pause(.1)
            if save>0 and plotNumber <=save:
                fname = '%s-%s-row-%s-filenum-%d.png' % (self.prefix, nm, sampleIdx, fileNum)
                assert (not os.path.exists(fname)) or self.force, "Filename: %s exists, use --force to overwrite" % fname
                fig.savefig(fname)
            msg += ' hit enter (q if done)'
#            if view:
#                res =raw_input(msg)
#                if res.strip().lower()=='q':
#                    user_quit = True
#                    break
#            else:
#                print(msg)
        for nm in ['yag','vcc']:
            for stage in ['raw','processed']:
                sp_accuracy = sp_hits['%s_%s' % (nm,stage)]/float(sp_hits['%s_%s' % (nm,stage)] + sp_misses['%s_%s' % (nm,stage)])
                print("-- signal processing accuracy name=%s stage=%s: %.2f" % (nm,stage,sp_accuracy))

    def max_stats(self):
        '''what is max value in yag/vcc inside the boxes? vs outside?
        In [56]: accdata.max_stats()
        stats: mx_vcc_box=255.0 mx_yag_box=37.0 mx_vcc=255.0 mx_yag=255.0 min_vcc=0.0 min_yag=0.0
        '''
        stats = {'raw':{'yag':{'mxall':0, 'mxbox':0},
                        'vcc':{'mxall':0, 'mxbox':0}},
                 'processed':{'yag':{'mxall':0, 'mxbox':0},
                              'vcc':{'mxall':0, 'mxbox':0}}}

        for row in range(len(self.labels['yag'])):
            for stage, stagedict in stats.iteritems():
                for nm, nmdict in stagedict.iteritems():
                    if 0==self.labels[nm][row]: continue
                    img, box, roi = self.get_img_box_roi(nm, stage, row)
                    nmdict['mxall'] = max(nmdict['mxall'], np.max(img))
                    nmdict['mxbox'] = max(nmdict['mxbox'], np.max(roi))
                    
        for stage, stagedict in stats.iteritems():
            for nm, nmdict in stagedict.iteritems():
                print("nm=%s stage=%s mxall=%.1f mxbox=%.1f" % (nm, stage, nmdict['mxall'], nmdict['mxbox']))



class AccBeamVccYagDataset(Dataset):
    def __init__(self, project, subproject, verbose, **kwargs):
        Dataset.__init__(self, project=project,
                         subproject=subproject, 
                         verbose=verbose,
                         descr=ACCBEAM_DESCR,
                         name=kwargs.pop('name','accbeam'),
                         dev=kwargs.pop('dev',False))
        X=kwargs.pop('X',None)
        if X is None:
            self.X=['yag','vcc']
        else:
            assert isinstance(X,list)
            self.X=list(X)
        self.subbkg = kwargs.pop('subbkg',True)
        assert len(kwargs)==0, "unknown kwargs: %s" % kwargs.keys()

        self.labels = {'yag':None, 'vcc':None}
        self.rawROIs = {'yag':{}, 'vcc':{}}
        self.mat_files=[1,2,4]
        self.loaded=False
        self.data=None
        self.loadData()
        
    def loadData(self):        
        def getLabel(bx):
            if len(bx)==0:
                return 0
            if len(bx[0,:])==4:
                return 1
            return -1
            
        if self.loaded: return
        sdir=dataloc.getSubProjectFile(self.project, self.subproject, 'matlab')
        data=makeEmptyBeamLocData()
        for num in self.mat_files:
            mat_file = os.path.join(sdir, 'labeledimg%d.mat' % num)
            assert os.path.exists(mat_file), "file %s doesn't exist" % mat_file
            mat = sio.loadmat(mat_fname)
            matdata = parseBeamMat(mat)
            print("file %d has %d entries" % (filenum, len(matdata['yagImg'])))
            yagImgs, vccImgs, yagBoxs, vccBoxs = \
                    matdata['yagImg'], matdata['vccImg'], matdata['yagbox'], matdata['vccbox']
            for entry, yag, vcc, yagbx, vccbx in zip(range(len(yagImgs)), yagImgs, vccImgs, yagBoxs, vccBoxs):
                sampleIdx += 1
                if self.nn > 0 and sampleIdx > self.nn: break
                yag, vcc = self.subBackgroundIfRequired(yag, vcc, matdata)
                updateBeamLocData(data, yag, vcc, yagbx, vccbx, filenum)
                yagLabel = getLabel(yagbx)
                vccLabel = getLabel(vccbx)
                if yagLabel == -1:
                    yagLabel = 0
                    sys.stderr.write("WARNING: entry=%d yag for filenum=%s assigning label=0 yagbx=%r\n" % 
                                     (entry, filenum, yagbx))
                if vccLabel == -1:
                    vccLabel = 0
                    sys.stderr.write("WARNING: entry=%d vcc for filenum=%s assigning label=0 vccbx=%r\n" % 
                                     (entry, filenum, vccbx))
                if 1==yagLabel:
                    self.rawROIs['yag'][sampleIdx]=boxutil.extract_box(yag, yagbx[0,:])
                if 1==vccLabel:
                    self.rawROIs['vcc'][sampleIdx]=boxutil.extract_box(vcc, vccbx[0,:])
                data['yag']['label'].append(yagLabel)
                data['vcc']['label'].append(vccLabel)
            for ky in ['yagbkg', 'vccbkg', 'yagbeam', 'vccbeam']:
                if ky in matdata:
                    data[ky][filenum] = matdata[ky]
        self.rawdata = data

        self.check_rawdata()
        for nm in ['yag','vcc']:
            self.labels[nm] = np.array(self.rawdata[nm]['label'])
        self.meta['filenums'] = np.array(self.rawdata['yag']['file'])
        self.loaded=True
        
    def get_number_of_samples(self):
        self.loadData()
        return len(self.data['yag'])

    def train_iter(batchsize, epochs):
        self.loadData()
        
    
