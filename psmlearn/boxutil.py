from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def valid_box(bx):
    ymin,ymax,xmin,xmax=bx[:]
    if ymin>ymax: return False
    if xmin>xmax: return False
    return True
   
def plot_box(bx, plt, colorstr='w'):
    if not valid_box(bx):
        print("WARNING: plot_box: not valid: %s" % box_to_str(bx))
        return
    ymin,ymax,xmin,xmax=bx[:]
    plt.plot([xmin,xmin,xmax,xmax,xmin],
             [ymin,ymax,ymax,ymin,ymin], colorstr)

def plot_box_clipped_to(bx, plt, clipbx, colorstr='w'):
    def inside(x,ab):
        a,b=ab
        return x >= a and x <= b
    def overlap(ab,cd):
        a,b=ab
        c,d=cd
        if a>=d: return False
        if b<=c: return False
        return True
    def clip(ab,cd):
        a,b=ab
        c,d=cd
        return max(a,c),min(b,d)
    
    if not valid_box(bx):
        print("WARNING: plot_box_clipped_to: not valid: %s" % box_to_str(bx))
        return

    if not valid_box(clipbx):
        print("WARNING: plot_box_clipped_to: not valid: clipbx %s" % box_to_str(clipbx))
        return

    ymin,ymax,xmin,xmax=bx[:]
    ymin_clip,ymax_clip,xmin_clip,xmax_clip=clipbx[:]

    for x in [xmin,xmax]:
        if inside(x,[xmin_clip, xmax_clip]):
            y0,y1=clip([ymin,ymax],[ymin_clip,ymax_clip])
            plt.plot([x,x],[y0,y1],colorstr)
    for y in [ymin,ymax]:
        if inside(y,[ymin_clip, ymax_clip]):
            x0,x1=clip([xmin,xmax],[xmin_clip,xmax_clip])
            plt.plot([x0,x1],[y,y],colorstr)

def translate(bx, y, x):
    ymin,ymax,xmin,xmax = bx[:]
    return np.array([ymin+y, ymax+y, xmin+x,xmax+x])

def box_to_str(bx):
    ymin,ymax,xmin,xmax = bx[:]
    return "y=%.1f,%.1f x=%.1f,%.1f" % (ymin,ymax,xmin,xmax)

def containing_box(bx_for_center, bx_to_try_to_include, mx_expand=3.0):
    mx_expand=max(mx_expand, 1.1)
    ymin,ymax,xmin,xmax=bx_for_center[:]
    
    ylen=ymax-ymin
    xlen=xmax-xmin
    ylen_max = ylen * mx_expand
    xlen_max = xlen * mx_expand
    yhalf_outer=ylen_max/2.0
    xhalf_outer=xlen_max/2.0
    yhalf_inner=ylen*0.6
    xhalf_inner=xlen*0.6

    ymin_outer_lim = ymin-yhalf_outer
    ymax_outer_lim = ymax+yhalf_outer
    xmin_outer_lim = xmin-xhalf_outer
    xmax_outer_lim = xmax+xhalf_outer

    ymin_inner_lim = ymin-yhalf_inner
    ymax_inner_lim = ymax+yhalf_inner
    xmin_inner_lim = xmin-xhalf_inner
    xmax_inner_lim = xmax+xhalf_inner

    ymin_inc,ymax_inc,xmin_inc,xmax_inc=bx_to_try_to_include[:]

    ymin = max(ymin_outer_lim, min(ymin_inner_lim, ymin_inc-0.1*ylen))
    ymax = min(ymax_outer_lim, max(ymax_inner_lim, ymax_inc+0.1*ylen))

    xmin = min(xmin_outer_lim, min(xmin_inner_lim, xmin_inc-0.1*xlen))
    xmax = max(xmax_outer_lim, max(xmax_inner_lim, xmax_inc+0.1*xlen))

    bx = np.array([np.floor(ymin), np.ceil(ymax), np.floor(xmin), np.ceil(xmax)], dtype=np.int)
    return bx

def in_box(bx, y, x):
    ymin,ymax,xmin,xmax = bx[:]
    return y >= ymin and y <= ymax and x >= xmin and x <= xmax

def area_box(bx):
    ymin,ymax,xmin,xmax = bx[:]
    if ymin>=ymax: return 0
    if xmin>=xmax: return 0
    return (ymax-ymin)*(xmax-xmin)

def intersection_boxes(bxA, bxB):
    ymin1,ymax1,xmin1,xmax1 = bxA[:]
    ymin2,ymax2,xmin2,xmax2 = bxB[:]

    xmin_inter = max(xmin1,xmin2)
    xmax_inter = min(xmax1,xmax2)
    ymin_inter = max(ymin1,ymin2)
    ymax_inter = min(ymax1,ymax2)
    
    return np.array([ymin_inter, ymax_inter, xmin_inter, xmax_inter])

def intersection_over_union(bxA, bxB):
    inter = intersection_boxes(bxA, bxB)
    inter_area = area_box(inter)
    A_area = area_box(bxA)
    B_area = area_box(bxB)
    union_area = A_area + B_area - inter_area
    if union_area < 1e-12: return 0.0
    return inter_area/float(union_area)

def extract_box(img,bx):
    if not valid_box(bx):
        print("WARNING: extract_box: not valid: bx %s" % box_to_str(bx))
        return None

    ymin,ymax,xmin,xmax=bx[:]
    if ymin==ymax or xmin==xmax:
        return None
    return  img[ymin:ymax,xmin:xmax].copy()

def max_in_box(img, box):
    if not valid_box(box): return None
    ymin,ymax,xmin,xmax=box[:]
    roi = img[ymin:ymax,xmin:xmax]
    return np.max(roi)


