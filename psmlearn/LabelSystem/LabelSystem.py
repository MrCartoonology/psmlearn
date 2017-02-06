from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import numpy as np
import h5py
import matplotlib.cm as cm
import PIL.Image as Image
import getpass

try:
    import psana
except ImportError:
    psana = None

class LabelSystem(object):
    def __init__(self, expname):
        assert psana, "unable to import psana"
        self.expname = expname
        self.labelme_host = 'psexport'
        self.images_folder = '/reg/neh/home/davidsch/public_html/LabelMe/Images'
        self.offsite=False
        if not os.path.exists(self.images_folder):
            self.images_folder = '/home/davidsch/public_html/LabelMe/Images'
            self.offsite=True
        assert os.path.exists(self.images_folder), "path doesn't exist: %s" % self.images_folder
        
    @staticmethod
    def prepare_for_jpeg(img):
        jet = cm.get_cmap('jet')
        img = img.astype(np.float32)/np.float32(np.max(img))
        img = np.maximum(0.0, np.minimum(1.0, img))
        img = jet(img)[:,:,0:3]
        img *= 255.0
        img = np.uint8(img)
        img = Image.fromarray(img)
        return img
    
    @staticmethod
    def view_random(data_iter, figh=1, num=100, plt=None):
        if None is plt:
            import matplotlib.pyplot as plt
            plt.ion()
        plt.figure(figh)
        plt.clf()
        for ii, data in enumerate(data_iter):
            if ii > num:
                break
            img = data['img']
            plt.imshow(img)
            ii += 1
            plt.pause(.2)

    def get_jpeg_filename(self, detname, evt):
        assert detname.isalnum(), "datename must be alphanumeric, but it is %s" % detname
        evt_id = evt.get(psana.EventId)
        fname = '-'.join([self.expname,
                          'r%4.4d' % evt.run(),
                          detname,
                          's%9.9d' % evt_id.time()[0],
                          'n%9.9d' % evt_id.time()[1],
                          'f%6.6d' % evt_id.fiducials()])
        return fname + '.jpg'
    
    def prepare_jpegs_to_label(self, data_iter, jpeg_dir, detname, h5_output, num_to_label=200):
        next_label = 0
        jpeg_files = []
        for data in data_iter:
            if next_label == num_to_label:
                break
            arr = data['img']
            evt = data['evt']
            img = LabelSystem.prepare_for_jpeg(arr)
            filename = self.get_jpeg_filename(detname, evt)
            path = os.path.join(jpeg_dir, filename)
            img.save(path)
            jpeg_files.append(path)
            print(path)
            next_label += 1
        h5=h5py.File(h5_output,'w')
        h5['jpeg_files'] = jpeg_files
        h5.close()
    
    def move_jpegs_to_labelme(self, h5list, labels, h5out, labelme_folder=None):
        h5 = h5py.File(h5list,'r')
        jpeg_files = h5['jpeg_files'][:]
        if labelme_folder is None:
            labelme_folder = set([os.path.basename(path).split('-s')[0] for path in jpeg_files])
        assert len(labelme_folder)==1, "Expected collection from one expermiment/run/detector, but it is from: %s" % labelme_folder
        labelme_folder = labelme_folder.pop()
        print("labelme_folder: %s" % labelme_folder)
        labelme_path = os.path.join(self.images_folder, labelme_folder)
        if not os.path.exists(labelme_path): os.mkdir(labelme_path)
        copied = []
        for jpg in jpeg_files:
            dest = os.path.join(labelme_path, os.path.basename(jpg))
            cmd = 'cp %s %s' % (jpg, dest)
            print(cmd)
            assert 0 == os.system(cmd)
            copied.append(jpg)
        h5 = h5py.File(h5out,'w')
        h5['jpeg_files']=copied
        username=getpass.getuser()
        labelme_url = "https://pswww-dev.slac.stanford.edu/LabelMe/tool.html?collection=LabelMe&mode=f"
        labelme_url += "&folder=%s" % labelme_folder
        labelme_url += "&username=%s" % username
        labelme_url += "&objects=%s" % ','.join(labels)
        labelme_url += "&image=%s" % os.path.basename(jpeg_files[0])
        h5['labelme_url'] = labelme_url
        
    def get_labels(self, h5input):
        h5 = h5py.File(h5input, 'r')
        labelme_url = h5['labelme_url'].value
        print("=================")
        print(labelme_url)
        print("=================")
