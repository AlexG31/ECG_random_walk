#encoding:utf8
import os
import sys
import time
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib
import scipy.signal 
import pdb

from dpi.DPI_QRS_Detector import DPI_QRS_Detector as DPI
from QTdata.loadQTdata import QTloader
from random_walker import RandomWalker
from test_api import Testing
from test_api import GetModels

import scipy.io as sio



def Test1():
    '''Test case1.'''
    data = sio.loadmat('./data/ft.mat')
    v2 = np.squeeze(data['II'])
    raw_sig = v2
    fs = 500

    model_folder = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/m3_full_models'
    pattern_file_name = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/m3_full_models/random_pattern.json'
    model_list = GetModels(model_folder, pattern_file_name)
    start_time = time.time()
    # results = Testing_random_walk(raw_sig, 250.0, r_list, model_list)
    results = Testing(raw_sig, fs, model_list)
    print 'Testing time cost %f secs.' % (time.time() - start_time)

    samples_count = len(raw_sig)
    time_span = samples_count / fs
    print 'Span of testing range: %f samples(%f seconds).' % (samples_count, time_span)

    # Display results
    plt.figure(1)
    plt.plot(raw_sig, label = 'ECG')
    pos_list, label_list = zip(*results)
    labels = set(label_list)
    for label in labels:
        pos_list = [int(x[0]) for x in results if x[1] == label and x[0] < len(raw_sig)]
        amp_list = [raw_sig[x] for x in pos_list]
        plt.plot(pos_list, amp_list, 'o',
                markersize = 15,
                label = label)
    plt.title('ECG')
    plt.grid(True)
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    Test1()

