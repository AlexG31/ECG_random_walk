import pywt, pdb, json
import numpy as np
import matplotlib.pyplot as plt
from QTdata.loadQTdata import QTloader

from swt_plot import removeQRS

def loadChanggeng(recID = '42736'):
    signal_file = './changgeng/%s.json' % recID
    annot_file = './annots/%s.json' % recID

    QRS_ranges = list()
    with open(annot_file, 'r') as fin:
        annots = json.load(fin)
        annots.sort(key = lambda x:x[0], reverse = False)
        annots = filter(lambda x:('R' == x[1][0] and len(x[1]) > 1), annots)

        for ind in xrange(0, len(annots) - 1):
            pos, label = annots[ind]
            if label == 'Ronset':
                if annots[ind + 1][1] == 'Roffset':
                    QRS_ranges.append((pos, annots[ind + 1][0]))

    with open(signal_file, 'r') as fin:
        data = json.load(fin)
    return data

def loadChanggengAnnots(recID = '42736'):
    signal_file = './changgeng/%s.json' % recID
    annot_file = './annots/%s.json' % recID

    QRS_ranges = list()
    with open(annot_file, 'r') as fin:
        annots = json.load(fin)

    return annots

def amplify(coef, y, ax, level = 20, color = 'r'):
    coef_shape = coef.shape
    P_point_list = list()
    thres = 0.2
    for ind in xrange(0, coef_shape[1]):
        if coef[level, ind] > thres:
            P_point_list.append(ind)
    amplist = [y[x] for x in P_point_list]
    ax[0].plot(y)
    ax[0].plot(P_point_list, amplist, 'o', markerfacecolor = color, alpha = 0.8, label = str(level))


def getCwtRange(coef, y, level = 20):
    coef_shape = coef.shape
    P_point_list = list()
    thres = 0.2
    for ind in xrange(0, coef_shape[1]):
        if coef[level, ind] > thres:
            P_point_list.append(ind)
    return P_point_list


def swt_show():
    record_ID = '1269'
    y = loadChanggeng(record_ID)
    raw_sig = y[:]
    original_ecg = raw_sig[:]
    annots = loadChanggengAnnots(record_ID)
    y = removeQRS(y, annots)
    coef, freqs=pywt.cwt(y,np.arange(1,32),'mexh')

    coef_shape = coef.shape
    # Get P magnify ranges
    P_point_list = list()
    thres = 0.2
    for ind in xrange(0, len(y)):
        if coef[-1, ind] > thres:
            P_point_list.append(ind)



    x_range = (1000, 1640)
    # y = y[x_range[0]:x_range[1]]
    # coef = coef[x_range[0]:x_range[1]]

    fig, ax = plt.subplots(2,1)

    amplist = [y[x] for x in P_point_list]
    ax[0].plot(y)
    ax[0].plot(P_point_list, amplist, 'ro')

    amplify(coef, y, ax, level = 20, color = (0.1, 0.2,0.3))
    amplify(coef, y, ax, level = 10, color = (0.9, 0.3,0.8))
    # amplify
    for pos in P_point_list:
        y[pos] *= 10.0
        raw_sig[pos] *= 5.0
    ax[0].plot(y, 'y', lw = 4, alpha = 0.3)



    ax[0].set_xlim(x_range)
    ax[1].matshow(coef, cmap = plt.gray()) 
    # ax[1].set_clip_box(((0,0),(9,19)))
    ax[1].set_xlim(x_range)
    plt.legend(numpoints = 1)


    plt.figure(2)
    # plt.plot(raw_sig, 'k', lw = 2, alpha = 1)
    poslist = getCwtRange(coef, y, level = 30)
    amplist = [original_ecg[x] for x in poslist]
    plt.plot(original_ecg, 'b', lw = 2, alpha = 1)
    plt.plot(poslist, amplist, 'ro', markersize = 12, alpha = 0.5)
    

    plt.show() 


if __name__ == '__main__':
    swt_show()
