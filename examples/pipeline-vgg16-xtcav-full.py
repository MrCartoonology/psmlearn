from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import random
import tensorflow as tf
from scipy.misc import imresize
from psmlearn.pipeline import Pipeline
import psmlearn.h5util as h5util
import psmlearn.util as util
from psmlearn import tsne
import psmlearn.plot as psplot
import psmlearn
import numpy as np
import h5py
import time

############ helper functions
def topn(arr, n):
    inds = arr.argsort()
    pos = list(inds[-n:])
    vals = list(arr[pos])
    pos.reverse()
    vals.reverse()
    return pos, vals


################ main code
class XtcavVgg16Full(object):
    def __init__(self):
        '''Step implementation for pipeline, doing xtcav analysis using transfer learning with 
        vgg16. Don't do any initialization in __init__, wait until command line args are parsed
        and entire config can be created - use init
        '''
        self.hdr='XtcavVgg16'
        self.doTrace = False
        self._vgg16 = None
        self._model = None
        self.dev = False
        self._datastats = None
        self._datastats_codewords = None

    def get_datastat(self, name, step2h5list):
        if name in ['codeword_num_features']:
            if None is self._datastats_codewords:
                h5 = h5py.File(step2h5list['compute_vgg16_codewords'][0],'r')
                self._datastats_codewords = {}
                self._datastats_codewords['codeword_num_features'] = h5['codewords'].shape[1]
            return self._datastats_codewords['codeword_num_features']
        
        if None is self._datastats:
            self._datastats = h5util.read_from_h5(step2h5list['compute_channel_mean'][0])
        assert name in self._datastats, "name=%s not in datasats, keys are: %r" % (name, self._datastats.keys())
        return self._datastats[name]
    
    def add_arguments(self, parser):
        pass

    def init(self, config, pipeline):
        '''pipeline will have set the random seeds for python and numpy, 
        '''
        self.pipeline=pipeline
        self.doTrace = pipeline.doTrace
        dset = psmlearn.get_dataset('xtcav', subproject='amo86815_full', X='img', Y='enPeak', verbose=True, dev=config.dev)
        dset.split(train=90, validation=5, test=5, seed=config.seed)
        self.dset = dset
        self.dev = config.dev
        
    def vgg16(self):
        if self._vgg16 is None:
            vgg16_weights_file = psmlearn.dataloc.getProjectCalibFile(project='vgg16', fname='vgg16_weights.npz')
            self._vgg16 = psmlearn.vgg16.create(session=pipeline.session,
                                                weights=vgg16_weights_file,
                                                dev=self.dev)
        return self._vgg16

    def model(self, pipeline, step2h5list, restore=False):
        if self._model is None:
            config = pipeline.get_config('train_on_codewords')
            num_features = self.get_datastat('codeword_num_features', step2h5list)
            num_outputs = self.get_datastat('num_outputs', step2h5list)
            self._model = psmlearn.models.LinearClassifier(num_features=num_features,
                                                           num_outputs=num_outputs,
                                                           config=config)
        if restore:
            self._model.restore_from_file(step2h5list['train_on_codewords'][0], pipeline.session)

        return self._model

    def prepare_for_vgg16(self, img, config, channel_mean=None):
        prep_img = img.astype(np.float32)
        thresh = config.thresh
        util.logTrace('prepare_for_vgg16', 'thresh=%.1f' % thresh)
        util.replaceWithLogIfAbove(prep_img, thresh)
        prep_resized = imresize(prep_img,(224,224), interp='lanczos', mode='F')
        
        if channel_mean: prep_resized -= channel_mean
        rgb = util.replicate(prep_resized, numChannels=3, dtype=np.float32)
        return rgb

    ########### STEPS #######
    def crop(self, config, pipeline, step2h5list, output_files):
        pass
    
    def compute_channel_mean(self, config, pipeline, step2h5list, output_files):
        prep_config = pipeline.get_config(name='prepare_for_vgg16')
        basic_iter = self.dset.train_iter(batchsize=1, epochs=1, num_batches=config.num_to_sample)
        means = []
        num_outputs = 0
        for X,Y,meta,batchinfo in basic_iter:
            if num_outputs==0:
                num_outputs = Y[0].shape[1]
            img_batch = X[0]
            img = img_batch[0,:,:]
            if config.dev:
                means.append(img[np.random.randint(0,img.shape[0]), np.random.randint(0,img.shape[1])])
            else:
                prep_img = self.prepare_for_vgg16(img, config=prep_config, channel_mean=0)
                means.append(np.mean(prep_img[:,:,0]))
            if batchinfo['step'] % 10 == 0:
                pipeline.debug("compute_channel_mean: %d" % len(means), checkcache=False)
        h5=h5py.File(output_files[0],'w')
        h5['channel_mean'] = np.mean(np.array(means))
        h5['number_samples_train'] = self.dset.num_samples_train()
        h5['number_samples_validation'] = self.dset.num_samples_train()
        h5['number_samples_test'] = self.dset.num_samples_test()
        h5['num_outputs'] = num_outputs
        h5['files'] = basic_iter.get_h5files()

        # get number of validation and test samples
        basic_iter = self.dset.validation_iter(batchsize=1, epochs=1)
        validation_num=0
        for batch in basic_iter: validation_num+=1

        basic_iter = self.dset.test_iter(batchsize=1, epochs=1)
        test_num=0
        for batch in basic_iter: test_num += 1
        
        pipeline.trace("compute_channel_mean: finished", checkcache=False)

    def plot_vgg16_img_prep(self, plot, pipeline, plotFigH, config, step2h5list):
        prep_config = pipeline.get_config(name='prepare_for_vgg16')
        channel_mean = self.get_datastat('channel_mean', step2h5list)
        util.logTrace("plot_vgg16_img_prep", "channel_mean is %.2f" % channel_mean)
        basic_iter = self.dset.train_iter(batchsize=1, epochs=1)
        plt = pipeline.plt
        plt.figure(plotFigH)
        plt.clf()
        for X,Y,meta,batchinfo in basic_iter:
            img_batch = X[0]
            img = img_batch[0,:,:]
            prep_img = self.prepare_for_vgg16(img, config=prep_config, channel_mean=channel_mean)
            psplot.compareImages(plt, plotFigH, ("orig",img), ("vgg16 prep",prep_img))
            if pipeline.stop_plots(): break
            
    def tsne_on_img_prep(self, config, pipeline, step2h5list, output_files):
        h5=h5py.File(output_files[0],'w')

    def compute_vgg16_codewords(self, config, pipeline, step2h5list, output_files):
        channel_mean = self.get_datastat('channel_mean',step2h5list)
        imgprep_config = pipeline.get_config('prepare_for_vgg16')

        assert output_files[0].endswith('_train.h5')
        assert output_files[1].endswith('_validation.h5')
        assert output_files[2].endswith('_test.h5')

        iterList = [('train',
                     output_files[0],
                     self.dset.train_iter(batchsize=1,epochs=1)),
                    ('validation',
                     output_files[1],
                     self.dset.validation_iter(batchsize=1,epochs=1)),
                    ('test',
                     output_files[2],
                     self.dset.test_iter(batchsize=1,epochs=1)),
        ]

        for splitFnameIter in iterList:
            split, fname, batch_iter = splitFnameIter
            datalen = len(batch_iter)
            # datalen may be < samplesPerEpoch is dev is
            assert datalen > 0 and datalen <= batch_iter.samplesPerEpoch()
            h5out = h5py.File(fname, 'w')
            idx = 0
            all_meta = None
            all_Y = None
            for Xlist,Ylist,meta,batchinfo in batch_iter:
                vgg_t0=time.time()
                img_batch = Xlist[0]
                img = img_batch[0]
                Y=Ylist[0]
                assert np.sum(Y)==1
                if all_meta is None:
                    all_meta = np.zeros(datalen, dtype=meta.dtype)
                if all_Y is None:
                    shape = [datalen]
                    if len(Y.shape)>1:
                        shape += list(Y.shape[1:])
                    all_Y = np.zeros(shape, dtype=Y.dtype)
                all_meta[idx]=meta
                all_Y[idx,:]=Y
                prep_img = self.prepare_for_vgg16(img, imgprep_config, channel_mean=channel_mean)
                prep_img = np.reshape(prep_img, [1]+list(prep_img.shape))
                fc2 = self.vgg16().get_model_layers(sess=pipeline.session,
                                                    imgs=prep_img,
                                                    layer_names=['fc2'])[0]
                vgg_time=time.time()-vgg_t0
                h5_t0=time.time()
                if idx == 0:
                    shape = (datalen, fc2.shape[1])
                    h5out.create_dataset('codewords', shape=shape, dtype=fc2.dtype)
                h5out['codewords'][idx,:] = fc2[:]
                h5_time=time.time()-h5_t0
                idx += 1
                pipeline.debug("compute_codewords: split=%s idx=%d batch_read_time=%.3f vgg_time=%.3f h5write_time=%.3f" %
                               (split,idx, batchinfo['readtime'], vgg_time, h5_time), checkcache=False)
            h5out['meta']=all_meta
            h5out['Y'] = all_Y
            h5out['files']=batch_iter.get_h5files()
        pipeline.trace("compute_codewords: done with all", checkcache=False)
        
    def tsne_on_vgg16_codewords(self, config, pipeline, step2h5list, output_files):
        h5=h5py.File(output_files[0],'w')

    def train_on_codewords(self, config, pipeline, step2h5list, output_files):
        model = self.model(pipeline, step2h5list)
        h5files_train = [step2h5list['compute_vgg16_codewords'][0]]
        h5files_validation = [step2h5list['compute_vgg16_codewords'][1]]

        epochs=0
        if config.dev:
            epochs=4
            config.train_batchsize=32
            config.eval_steps=4
            config.train_report_steps=2
        train_iter = self.dset.iter_from(h5files=h5files_train,
                                         X=['codewords'], Y=['Y'],
                                         epochs=epochs,
                                         batchsize=config.train_batchsize)

        validation_iter = self.dset.iter_from(h5files=h5files_validation,                                              
                                              X=['codewords'], Y=['Y'],
                                              batchsize=config.train_batchsize)

        trainer = psmlearn.ClassificationTrainer(train_iter=train_iter,
                                                 validation_iter=validation_iter,
                                                 config=config,
                                                 sess=pipeline.session,
                                                 save_fname=output_files[0],
                                                 force=config.force,
                                                 model=model)
        trainer.run()


    def get_image(self, meta, step2h5list, do_vgg_prep):
        h5files = self.get_datastat('files', step2h5list)
        img = self.dset.get_image(meta, h5files)
        if do_vgg_prep:
            channel_mean = self.get_datastat('channel_mean', step2h5list)
            config = self.pipeline.get_config('prepare_for_vgg16')
            img = self.prepare_for_vgg16(img, config=config, channel_mean=channel_mean)
        return img
    
    def get_images(self, meta, step2h5list, do_vgg_prep):
        images = None
        for ii in range(len(meta)):
            img = self.get_image(meta[ii:(ii+1)], step2h5list, do_vgg_prep)
            if images is None:
                shape = [len(meta)] + list(img.shape)
                images = np.empty(shape=tuple(shape), dtype=img.dtype)
            images[ii,:] = img[:]
        return images
    
    def vgg16_output(self, config, pipeline, step2h5list, output_files):
        model = self.model(pipeline, step2h5list, restore=True)
        imgprep_config = pipeline.get_config('prepare_for_vgg16')
        channel_mean = self.get_datastat('channel_mean',step2h5list)
        W,B = model.get_W_B(pipeline.session)
        h5files = step2h5list['compute_vgg16_codewords'][0:2]
        dataiter = self.dset.iter_from(h5files=h5files,
                                       X=['codewords'], Y=['Y'],
                                       batchsize=1,
                                       epochs=1)
        dataiter.shuffle()
        
        samples_meta = []
        samples_label = []
        activation_dict = {}
        layer_names = ['fc2','fc1','pool5','pool4','pool3','pool2','pool1']
        layer_vgg16_output = {}
        for name in layer_names:
            layer_vgg16_output[name]=[]

        num_saved = 0
        for Xlist, Ylist, meta, batchinfo in dataiter:
            if num_saved >= config.num_samples:
                break
            t0 = time.time()
            sample = batchinfo['batch']
            X=Xlist[0]
            Y=Ylist[0]
            logits = np.matmul(X,W)+B
            predicted = np.argmax(logits)
            label = np.argmax(Y)
            if predicted != label:
                pipeline.debug("vgg16_output: skipping sample %d, misclassified" % sample, checkcache=False)
                continue
            num_saved += 1
            samples_meta.append(meta)
            samples_label.append(label)
            prep_img = self.get_image(meta, step2h5list, do_vgg_prep=True)
            prep_img = np.reshape(prep_img, [1]+list(prep_img.shape))
            layers = self.vgg16().get_model_layers(sess=pipeline.session,
                                                   imgs=prep_img,
                                                   layer_names=layer_names)
            for nm, vgg16_output in zip(layer_names, layers):
                layer_vgg16_output[nm].append(vgg16_output)
            vgg_time=time.time()-t0
            pipeline.debug(msg='activation %5d label=%d vgg.time=%.2f' % (num_saved, label, vgg_time), checkcache=False)

        h5 = h5py.File(output_files[0],'w')
        h5['label']=samples_label
        h5['meta']=np.concatenate(samples_meta)
        for nm, arrList in layer_vgg16_output.iteritems():
            one_shape = arrList[0].shape
            expected_shape = tuple([len(arrList)*one_shape[0]] + list(one_shape[1:]))
            arr = np.concatenate(arrList)
            assert arr.shape == expected_shape, "nm=%s one_shape=%r expected=%r but concatenate shape=%r" % (nm, one_shape, expected_shape, arr.shape)
            h5[nm]=arr
            

    def get_W(self, step2h5list):
        h5 = h5py.File(step2h5list['train_on_codewords'][0],'r')
        return h5['model']['W'][:]

    def get_vgg16_output(self, step2h5list):
        res = h5util.read_from_h5(step2h5list['vgg16_output'][0])
        meta = res['meta']
        label = res['label']
        del res['meta']
        del res['label']
        return res, meta, label
        
    def neurons(self, config, pipeline, step2h5list, output_files):
        '''identify the neurons with the highest contributions'''        
        W = self.get_W(step2h5list)
        layer_output, meta, labels = self.get_vgg16_output(step2h5list)
        fc2_W, fc2_B = self.vgg16().get_W_B('fc2')
        fc1_W, fc1_B = self.vgg16().get_W_B('fc1')

        h5=h5py.File(output_files[0],'w')

        unique_labels = list(set(labels))

        for lbl in unique_labels:
            lblGroup = h5.create_group('label_%d' % lbl)
            fc2 = layer_output['fc2'][labels==lbl]
            fc2_act = np.mean(fc2 * W[:,lbl], axis=0)
            lblGroup['fc2_act_hist']=fc2_act
            pos, vals = topn(fc2_act, config.topn)
            lblGroup['fc2_topn_pos']=pos
            lblGroup['fc2_topn_vals']=vals

            for idx,ipos in enumerate(pos):
                idxLblGroup = lblGroup.create_group('pos_neuron_%d' % idx)
                fc1 = layer_output['fc1'][labels==lbl]
                fc1_act = np.mean(fc1 * fc2_W[:,ipos] + fc2_B[ipos], axis=0)
                pos_ipos, vals_ipos = topn(fc1_act, config.topn)
                idxLblGroup['fc1_act_hist'] = fc1_act
                idxLblGroup['fc1_topn_pos'] = pos_ipos
                idxLblGroup['fc1_topn_vals'] = vals_ipos

                for jdx, jpos in enumerate(pos_ipos):
                    jdxIdxLblGroup = idxLblGroup.create_group('pos_neuron_%d' % jdx)
                    pool5 = layer_output['pool5'][labels==lbl]
                    N = pool5.shape[0]
                    M = np.prod(pool5.shape[1:])
                    pool5 = np.resize(pool5, (N,M))
                    pool5_act = np.mean(pool5 * fc1_W[:,jpos] + fc1_B[jpos], axis=0)
                    pos_jpos, vals_jpos = topn(pool5_act, config.topn)
                    jdxIdxLblGroup['pool5_act_hist'] = pool5_act
                    jdxIdxLblGroup['pool5_topn_pos'] = pos_jpos
                    jdxIdxLblGroup['pool5_topn_vals'] = vals_jpos
                
        
    def gbprop(self, config, pipeline, step2h5list, output_files):
        assert config.layer_from == 'pool5', 'only implementing pool5 for now'
        num_outputs = self.get_datastat('num_outputs',step2h5list)
        h5in=h5py.File(step2h5list['neurons'][0],'r')
        h5vgg=h5py.File(step2h5list['vgg16_output'][0],'r')
        h5vgg_labels = h5vgg['label'][:]
        h5vgg_meta = h5vgg['meta'][:]

        gbprop_op, pl_pool5_grad = self.vgg16().gbprop_op_pool5()
        saliency_op, pl_pool5_grad = self.vgg16().saliency_op_pool5()
        pl_imgs = self.vgg16().imgs
        
        h5out=h5py.File(output_files[0],'w')

        for lbl in range(num_outputs):
            groupName = 'label_%d' % lbl
            if groupName not in h5in.keys():
                sys.stderr.write("WARNING: gbprop: %s not present, h5keys are: %s\n" % (groupName, h5in.keys()))
                continue
            
            groupLbl = h5in[groupName]
            meta = h5vgg_meta[h5vgg_labels==lbl][0:config.images_per_label]
            images = self.get_images(meta, step2h5list, do_vgg_prep=True)
            grad_ys=np.zeros((len(images),7,7,512), dtype=np.float32)
            top_pool5_neuron = groupLbl['pos_neuron_0/pos_neuron_0/pool5_topn_pos'][0]
            pool5_neuron_coords = np.unravel_index(top_pool5_neuron, (7,7,512))
            ii,jj,kk = pool5_neuron_coords
            grad_ys[:,ii,jj,kk] = 1.0
            gb_images = pipeline.session.run(gbprop_op, feed_dict={pl_imgs:images, pl_pool5_grad:grad_ys})
            assert gb_images.shape == images.shape
            saliency_images = pipeline.session.run(saliency_op, feed_dict={pl_imgs:images, pl_pool5_grad:grad_ys})
            assert saliency_images.shape == images.shape
            gr=h5out.create_group('label_%d' % lbl)
            gr['gbprop_images'] = gb_images
            gr['saliency_images'] = saliency_images

    def plot_gbprop(self, plot, pipeline, plotFigH, config, step2h5list):
        h5 = h5py.File(step2h5list['gbprop'][0],'r')
        gbprop_imgs = h5['label_3/gbprop_images'][:]
        saliency_imgs = h5['label_3/saliency_images'][:]
        plt = pipeline.plt
        plt.figure(plotFigH)
        plt.clf()
        for idx in range(len(gbprop_imgs)):
            imgA=gbprop_imgs[idx,:]
            imgB=saliency_imgs[idx,:]
            psplot.compareImages(plt, plotFigH, ("gbprop", imgA), ("saliency",imgB))
            raw_input('hit_enter')
        

### pipeline ###########
if __name__ == '__main__':
    stepImpl = XtcavVgg16Full()
    outputdir = psmlearn.dataloc.getDefalutOutputDir(project='xtcav')
    pipeline = Pipeline(stepImpl=stepImpl, outputdir=outputdir)
    stepImpl.add_arguments(pipeline.parser)
    pipeline.add_step_method(name='crop')
    if False:
        pipeline.add_step_method(name='compute_channel_mean')
        pipeline.add_step_method_plot(name='plot_vgg16_img_prep')
        pipeline.add_step_method(name='tsne_on_img_prep')
        pipeline.add_step_method(name='compute_vgg16_codewords',
                                 output_files=['_train','_validation','_test'])
        pipeline.add_step_method(name='tsne_on_vgg16_codewords')
        pipeline.add_step_method(name='train_on_codewords')
        pipeline.add_step_method(name='vgg16_output')
        pipeline.add_step_method(name='neurons')
        pipeline.add_step_method(name='gbprop')
        pipeline.add_step_method_plot(name='plot_gbprop')
    pipeline.init()
    pipeline.run()
