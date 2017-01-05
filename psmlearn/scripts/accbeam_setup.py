import os
import sys
os.environ['MOCK_TENSORFLOW']='1'
import psmlearn

def loadMat(fname_in):
    mat = sio.loadmat(fname_in)
    matdata = psmlearn.datasets.accbeam.parseBeamMat(mat)
    data = psmlearn.datasets.accbeam.mat2data(matdata)
    
    print("file %s has %d entries" % (fname_in, len(matdata['yagImg'])))
    yagImgs, vccImgs, yagBoxs, vccBoxs = \
        matdata['yagImg'], matdata['vccImg'], matdata['yagbox'], matdata['vccbox']
    for entry, yag, vcc, yagbx, vccbx in zip(range(len(yagImgs)), yagImgs, vccImgs, yagBoxs, vccBoxs):
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
        data['yag']['label'].append(yagLabel)
        data['vcc']['label'].append(vccLabel)
    for ky in ['yagbkg', 'vccbkg', 'yagbeam', 'vccbeam']:
        if ky in matdata:
            data[ky][filenum] = matdata[ky]
    return data

    
def translate_labelimg(fname_in, fname_out):
    mat = loadMat(fname_in)
    
def accbeam_setup(args):
    

if __name__ == '__main__':
    accbeam_setup(sys.argv)
