#encoding:utf-8
"""
Plot testing results.
"""
import os
import sys
import json
import math
import pickle
import random
import pywt
import time
import glob
import pdb
from multiprocessing import Pool

import numpy as np
## machine learning methods
from sklearn.ensemble import RandomForestClassifier
import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import pi, r_
from scipy import optimize

# project homepath
# 
curfilepath =  os.path.realpath(__file__)
curfolderpath = os.path.dirname(curfilepath)
projhomepath = os.path.dirname(curfolderpath)
from QTdata.loadQTdata import QTloader

def plotExpertLabels(ax, raw_sig, annots):

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
        # marker
        if 'onset' in label:
            marker = '<'
        elif 'offset' in label:
            marker = '>'
        else:
            marker = 'o'
        ax.plot(posList,map(lambda x:raw_sig[int(x)],posList),marker = marker,color = color,linestyle = 'none',markersize = 14,label = label)
    ax.legend(numpoints = 1)

def plotResult():
    '''Plot ECG delineation result.'''
    record_name = 'sel808'
    result_file_path = './round1/round2/sel808.json'
    qt = QTloader()
    sig = qt.load(record_name)
    raw_sig = sig['sig']

    # Load expert labels
    with open(result_file_path, 'r') as fin:
        data = json.load(fin)
        annots = data
    fig, ax = plt.subplots(1,1)
    plt.plot(raw_sig)
    plt.grid(true)
    plotExpertLabels(ax, raw_sig, annots)
    plt.show()




if __name__ == '__main__':
    plotResult()
