# -*- coding: utf-8 -*-  
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

from randomwalk.changgengLoader import ECGLoader as cLoader

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
        ax.plot(posList,map(lambda x:raw_sig[int(x)],posList),marker = marker,color = color,linestyle = 'none',markersize = 8,label = label)
    ax.legend(numpoints = 1)

def saveResult2Image(cID, annot_path, output_path):
    '''Save ECG delineation result as figure.'''
    cloader = cLoader(1, 1)
    raw_sig = cloader.loadID(cID)

    # Load expert labels
    with open(annot_path, 'r') as fin:
        data = json.load(fin)
        annots = data
    fig, ax = plt.subplots(1,1)
    fig.set_figheight(10, forward = True)
    fig.set_figwidth(30, forward = True)
    plt.plot(raw_sig, 'k')
    plt.title(cID)
    plt.grid(True)
    plotExpertLabels(ax, raw_sig, annots)
    plotECGGrid(ax)

    plt.xlim((-40, len(raw_sig) + 40))
    # plt.ylim((-1,3))
    plt.savefig(output_path)

def plotResult(record_name = '51087', annot_file_path = None):
    '''Plot ECG delineation result.'''
    cloader = cLoader(1, 1)
    raw_sig = cloader.loadID(record_name)

    # Load expert labels
    with open(annot_file_path, 'r') as fin:
        data = json.load(fin)
        annots = data


        from post_p import post_p
        annots = post_p(raw_sig, annots, 500)
    fig, ax = plt.subplots(1,1)
    fig.set_figheight(10, forward = True)
    fig.set_figwidth(30, forward = True)
    plt.plot(raw_sig, 'k')
    plt.grid(True)
    plotExpertLabels(ax, raw_sig, annots)
    plotECGGrid(ax)


    plt.title(record_name + ' post p')
    plt.xlim((-40, len(raw_sig) + 40))
    # plt.ylim((-1,3))
    # plt.savefig('./results/P-images/%s.png' % record_name)
    plt.show(block = True)
    # pdb.set_trace()


def plotECGGrid(ax):
    '''40ms, 0.1mV'''
    # -1 -> 1.5 mV
    x_range = (-10, 5000)
    y_range = (-1, 3)

    cur_y = y_range[0]
    while True:
        lw = 1
        alpha = 0.2
        if (abs(int(round(cur_y * 10))) % 5 == 0):
            lw = 2
            alpha = 0.4
        ax.plot(x_range,(cur_y, cur_y), color = 'r', lw = lw, alpha = alpha)
        cur_y += 0.1

        if cur_y >= y_range[1]:
            break
    
        
    cur_x = x_range[0]
    while True:
        lw = 1
        alpha = 0.2
        if (abs(int(round(cur_x - x_range[0]))) % 100 == 0):
            lw = 2
            alpha = 0.4
        ax.plot((cur_x, cur_x),y_range, color = 'r', lw = lw, alpha = alpha)
        cur_x += 20

        if cur_x >= x_range[1]:
            break

    # Remove xtick

    ax.set_xticks([])
    ax.set_yticks([])
    
        
    




if __name__ == '__main__':
    import glob
    json_files = glob.glob(curfolderpath + '/shortQT/test_results/*.json')
    for jsonfile in json_files:
       record_name = os.path.split(jsonfile)[-1]
       record_name = record_name.split('.')[0]
       plotResult(record_name, jsonfile)
    # plotResult('53789')
