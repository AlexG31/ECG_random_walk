#encoding:utf8
"""
Author : Gaopengfei
Date: 2017.4
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
from sklearn.ensemble import RandomForestClassifier
import matplotlib
import matplotlib.pyplot as plt
from numpy import pi, r_
from scipy import optimize
import Tkinter
import tkMessageBox

curfilepath =  os.path.realpath(__file__)
current_folderpath = os.path.dirname(curfilepath)

# ECG data from changgeng
class ECGLoader(object):
    def __init__(self, fs, current_folder_path):
        '''Loader for changgeng data.'''
        self.fs = fs
        self.P_faillist = [8999,8374,6659, 6655,6059,5395,1401,1269,737,75,9524,9476]

    def load(self, record_name):
        '''Return loaded signal info.'''
        return self.getSignal('%s.json' % str(record_name), 'II')

    def loadAnnot(self, record_name, target_label = 'P'):
        '''Return loaded signal info.'''
        return self.getAnnot('%s.json' % str(record_name), target_label)

    def getSize(self, jsonfilename):
        '''Get fangchan record count.'''
        import json
        import codecs

        return len(self.P_faillist)

    def getAnnot(self, jsonfilename, target_label):
        '''Get annotations.'''
        import json
        import codecs
        import subprocess
        import scipy.io as sio

        matinfojson_filename = os.path.join(current_folderpath, 'path_info', 'P', '%s' % jsonfilename)
        with codecs.open(matinfojson_filename, 'r', 'utf8') as fin:
            data = json.load(fin)
            annot_list = data['II_label_pos'][target_label]
            return zip(annot_list, [target_label,] * len(annot_list))

    def getSignal(self, jsonfilename, leadname):
        '''Get fangchan signal.'''
        import json
        import codecs
        import subprocess
        import scipy.io as sio

        matinfojson_filename = os.path.join(current_folderpath, 'path_info', 'P', '%s' % jsonfilename)
        with codecs.open(matinfojson_filename, 'r', 'utf8') as fin:
            data = json.load(fin)

        # mat_rhythm is the data
        dlist = data
        matpath = dlist['mat_rhythm']
        diagnosis_text = dlist['diagnose']

        mat_file_name = os.path.split(matpath)[-1]
        save_mat_filepath = os.path.join(current_folderpath, 'data', mat_file_name)
        if (os.path.exists(save_mat_filepath) == False):
            subprocess.call(['scp', 'xinhe:%s' % matpath, save_mat_filepath])
        matdata = sio.loadmat(save_mat_filepath)
        raw_sig = np.squeeze(matdata[leadname])

        return [raw_sig, diagnosis_text, mat_file_name, dlist['II_label_pos']['P']]

if __name__ == '__main__':
    ecg = ECGLoader(1,1)
    for record_ID in ecg.P_faillist:
        raw_sig = ecg.load(record_ID)
        raw_sig = list(raw_sig[0])
        with open('./%d.json' % record_ID, 'w') as fout:
            plt.plot(raw_sig)
            plt.title('%d' % record_ID)
            plt.show()
            json.dump(raw_sig, fout, indent = 4)
