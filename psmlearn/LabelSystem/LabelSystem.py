from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import random


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
