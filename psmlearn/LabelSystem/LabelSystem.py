from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import random
import numpy as np


class LabelSystem(object):
    def __init__(self):
        pass

    @staticmethod
    def view_random(datasource, detector, img_prep_fn, figh=1, num=100, plt=None):
        run = datasource.runs().next()
        tms = list(run.times())
        random.shuffle(tms)
        if None is plt:
            import matplotlib.pyplot as plt
            plt.ion()
        plt.figure(figh)
        plt.clf()
        num_viewed = 0
        events_with_no_img = 0
        for tm in tms:
            if num_viewed > num:
                break
            if events_with_no_img > 300:
                sys.stderr.write("WARNING: went through 300 events where det.image is None, exiting early\n")
                break
            evt = run.event(tm)
            img = detector.image(evt)
            if None is img:
                events_with_no_img += 1
                continue
            img = img_prep_fn(img, evt)
            if None is img:
                continue
            plt.imshow(img)
            num_viewed += 1
            plt.pause(.2)

    def label(self, data_iter, num_to_label=200):
        next_label = 0
        for data in data_iter:
            if next_label == num_to_label:
                break
            img = data['img']
            img_id = data['id']
            assert isinstance(img_id, str)
            assert isinstance(img, np.ndarray)
            print(next_label)
            next_label += 1
