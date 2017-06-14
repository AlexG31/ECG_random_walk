#encoding:utf8
import os, sys, json

import matplotlib.pyplot as plt
from QTdata.loadQTdata import QTloader
import numpy as np

def plotExpertLabels(ax, raw_sig, annots, is_expert = False):

    #get label Dict
    labelSet = set()
    labelDict = dict()
    for pos,label in annots:
        if label in labelSet:
            labelDict[label].append(pos)
        else:
            labelSet.add(label)
            labelDict[label] = [pos,]

    # plot to axes
    for label,posList in labelDict.iteritems():
        # plot marker for current label
        if label[0]=='T':
            color = 'm'
        elif label[0]=='P':
            color  = 'b'
        elif label[0]=='R':
            color  = 'r'
        elif label[0]=='Q':
            color = 'y'
        elif label[0]=='S':
            color = 'c'
        # marker
        if 'onset' in label:
            marker = '<'
        elif 'offset' in label:
            marker = '>'
        else:
            marker = 'o'
        markersize = 8
        if is_expert:
            markersize = 14
            label = 'expert ' + label
        ax.plot(posList,map(lambda x:raw_sig[int(x)],posList),marker = marker,color = color,linestyle = 'none',markersize = markersize,label = label)
    ax.legend(numpoints = 1)


def view_result():
    file_path = './fixed_results/round1/sel37.json'
    record_name = 'sel37'
    qt = QTloader()
    sig = qt.load(record_name)

    
    # Testing results
    with open(file_path, 'r') as fin:
        data = json.load(fin)
        annots = data['result']

    # Expert annots
    expert_annots = qt.getExpert(record_name)
    pos_list = [x[0] for x in expert_annots]
    print expert_annots

    raw_sig = sig['sig']

    fig, ax = plt.subplots(1,1)
    plotExpertLabels(ax, raw_sig, annots)
    plotExpertLabels(ax, raw_sig, expert_annots, is_expert = True)
    plt.plot(raw_sig)
    plt.xlim((np.min(pos_list), np.max(pos_list)))
    plt.show()



if __name__ == '__main__':
    view_result()
    

