import os
import sys
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import joblib
import scipy.signal
import pdb
import json
import logging
import pywt
from contextlib import closing
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import scipy.io as sio
from multiprocessing import Pool
from dpi.DPI_QRS_Detector import DPI_QRS_Detector as DPI
from dpi.QrsTypeDetector import QrsTypeDetector
from QTdata.loadQTdata import QTloader
from random_walker import RandomWalker
from test_api import Testing
from test_api import GetModels
import time
import codecs

def changgeng():
    with codecs.open(
        '/home/lab/Programm/dataset/total_19308.json',mode='r', encoding='utf8') as fin:
        dinfo=json.load(fin)
    #pool=Pool()
    rec_list=dinfo['data']

    try:
        with closing(Pool(11)) as p:
            p.map(keypoint_detect, rec_list, len(rec_list)/11)
        #pool.map(keypoint_detect, rec_list)
    except Exception as e:
        logging.error("error_detail: {}".format(str(e)), exc_info=True)

def ConvertLabel(label):
    '''Convert random forest label to figure label.'''
    if label == 'T':
        mker = 'ro'
    elif label == 'R':
        mker = 'go'
    elif label == 'P':
        mker = 'bo'
    elif label == 'Tonset':
        mker = 'r<'
    elif label == 'Toffset':
        mker = 'r>'
    elif label == 'Ronset':
        mker = 'g<'
    elif label == 'Roffset':
        mker = 'g>'
    elif label == 'Ponset':
        mker = 'b<'
    elif label == 'Poffset':
        mker = 'b>'
    else:# white
        mker = 'w.'
    return mker

def read_hardware_data(filepath):
    file_object = open(filepath)
    sig = []
    try:
      list_lines = file_object.readlines()
      for line in list_lines:
        sig.append(int(line))
    finally:
      file_object.close()
    return sig

def keypoint_detect_hardware_data(filepath):
    sig = read_hardware_data(filepath)

def crop_data_for_swt(rawsig):
    '''Padding zeros to make the length of the signal to 2^N.'''
    # crop rawsig
    base2 = 1
    N_data = len(rawsig)
    if len(rawsig)<=1:
        raise Exception('len(rawsig)={}, not enough for swt!', len(rawsig))
    crop_len = base2
    while base2<N_data:
        if base2*2>=N_data:
            crop_len = base2*2
            break
        base2*=2
    # Extending this signal input with its tail element.
    if N_data< crop_len:
        rawsig += [rawsig[-1],]*(crop_len-N_data)
    return rawsig

def Denoise(raw_sig, level = 7, low = 1, high = 1):
    '''Denoise with wavelet.'''
    # sig = raw_sig[:]
    len_sig = len(raw_sig)
    raw_sig = crop_data_for_swt(raw_sig)
    coeflist = pywt.swt(raw_sig, 'db2', level)
    cAlist, cDlist = zip(*coeflist)

    cAlist = list(cAlist)
    cDlist = list(cDlist)

    # denoise
    for ind in xrange(0, low):
        cAlist[ind] = np.zeros(len(cAlist[ind]))
    for ind in xrange(len(cDlist) - high, len(cDlist)):
        cDlist[ind] = np.zeros(len(cDlist[ind]))
    
    coeflist = zip(cAlist, cDlist)
    denoised_sig = pywt.iswt(coeflist, 'db2')
    
    # plt.figure(1)
    # plt.plot(denoised_sig)
    # plt.plot(sig, 'r')
    # plt.show()

    raw_sig = denoised_sig[:len_sig]
    return raw_sig
    
def Test1():
    '''Test case1.'''
    record_name = 'sel30'
    fs = 250.0
    # sig = read_hardware_data('./hardware_data/04-39-18.dat')
    sig = read_hardware_data('./hardware_data/04-14-47.dat')
    raw_sig = sig
    # raw_sig = raw_sig[0:250 * 60 * 1]
    # raw_sig = raw_sig[39200:39200 + 250 * 60 * 1]

    # Denoise
    raw_sig = Denoise(raw_sig)

    model_folder = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/Lw3Np4000'
    pattern_file_name = os.path.join(model_folder, 'random_pattern.json')

    model_list = GetModels(model_folder, pattern_file_name)
    start_time = time.time()
    # results = Testing_random_walk(raw_sig, 250.0, r_list, model_list)
    results = Testing(raw_sig, 250.0, model_list, walker_stepsize = 10)
    print 'Testing time cost %f secs. Data length: %d sec' % (time.time() - start_time, len(raw_sig) / fs)

    samples_count = len(raw_sig)
    time_span = samples_count / fs
    #print 'Span of testing range: %f samples(%f seconds).' % (samples_count, time_span)

    # Display results
    plt.figure(1)
    plt.plot(raw_sig, label = 'ECG')
    pos_list, label_list = zip(*results)
    labels = set(label_list)
    for label in labels:
        pos_list = [int(x[0]) for x in results if x[1] == label]
        amp_list = [raw_sig[x] for x in pos_list]
        marker = ConvertLabel(label)
        plt.plot(pos_list, amp_list, marker,
                markersize = 16,
                label = label,
                alpha = 0.8)
    plt.title(record_name)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    Test1()
