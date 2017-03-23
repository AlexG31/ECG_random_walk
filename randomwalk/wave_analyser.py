#encoding:utf8
import os
import sys
import time
import scipy.io as sio
import glob
import numpy as np
import matplotlib.pyplot as plt
import json
import scipy.signal
import joblib
import scipy.signal
import pdb

from dpi.DPI_QRS_Detector import DPI_QRS_Detector as DPI
from dpi.QrsTypeDetector import QrsTypeDetector
from QTdata.loadQTdata import QTloader
from random_walker import RandomWalker
from mcmc.wave_delineator import WaveDelineator

def CutWave(raw_sig, left_bound, right_bound):
    '''Cut signal segment.'''
    left_bound = int(left_bound)
    right_bound = int(right_bound)
    sig = raw_sig
    fixed_window_length = right_bound - left_bound
    segment = []
    if right_bound <= 0:
        segment = [sig[0], ] * fixed_window_length
    elif left_bound >= len(sig):
        segment = [sig[-1], ] * fixed_window_length
    else:
        L = max(0, left_bound)
        R = min(len(sig), right_bound)
        if left_bound < 0:
            segment = [sig[0],] * (-left_bound)
        else:
            segment = []
        segment.extend(sig[L:R])
        if right_bound > len(sig):
            segment.extend([sig[-1],] * (right_bound - len(sig)))
    return segment

def Pshape(raw_sig, P_positions):
    '''Analyse P wave shape.'''
    print 'P positions:', P_positions
    fs = 250.0
    expand_length_left = 6
    expand_length_right = 10

    pd = WaveDelineator(raw_sig, fs = fs)

    left_bound = int(P_positions[0] - expand_length_left)
    right_bound = int(P_positions[2] + expand_length_right)
    p_wave_segment = CutWave(raw_sig,
            left_bound,
            right_bound)
    P_positions = np.array(P_positions) - left_bound


    # plt.figure(1)
    # plt.plot(p_wave_segment)
    # amp_list = [p_wave_segment[int(x)] for x in P_positions]
    # plt.plot(P_positions, amp_list, 'ro')
    # plt.show()
    pd.detect(p_wave_segment, fs)
    


def Test1():
    '''Test case'''
    P_width = 50
    data = sio.loadmat('./data/ft.mat')
    v2 = np.squeeze(data['II'])
    raw_sig = v2
    fs = 500
    with open('./data/new_result.json', 'r') as fin:
        results = json.load(fin)
        results = [[x[0] * 250 / fs, x[1]] for x in results]
    if abs(fs - 250.0) > 1e-6:
        raw_sig = scipy.signal.resample(raw_sig, int(len(raw_sig) / float(fs) * 250.0))
        fs_recover = fs
        fs = 250.0

    P_positions = filter(lambda x: x[1][0] == 'P', results)
    P_positions.sort(key = lambda x:x[0])

    for ind in xrange(0, len(P_positions)):
        pos, label = P_positions[ind]
        if label == 'P':
            Pon = pos - P_width / 2.0
            Poff = pos + P_width / 2.0
            for pi in xrange(ind - 1, -1, -1):
                if P_positions[pi][1] == 'Ponset':
                    Pon = P_positions[pi][0]
                    break
            for pi in xrange(ind + 1, len(P_positions)):
                if P_positions[pi][1] == 'Poffset':
                    Poff = P_positions[pi][0]
                    break
            Pshape(raw_sig, (Pon, pos, Poff))

def TestQT(record_index):
    '''Test case'''
    result_folder = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/test_results/r2'
    qt = QTloader()
    P_width = 50
    result_files = glob.glob(os.path.join(result_folder, 'sel*.json'))
    with open(result_files[record_index], 'r') as fin:
        data = json.load(fin)
        record_name = data[0][0]
        print 'Record name:', record_name

    sig = qt.load(record_name)
    raw_sig = np.array(sig['sig']) / 40.0 + 1.0
    results = data[0][1]
    fs = 250
    if abs(fs - 250.0) > 1e-6:
        raw_sig = scipy.signal.resample(raw_sig, int(len(raw_sig) / float(fs) * 250.0))
        fs_recover = fs
        fs = 250.0

    P_positions = results
    P_positions.sort(key = lambda x:x[0])

    show_count = 1
    segment_list = list()
    wholewave_list = list()
    for ind in xrange(0, len(P_positions)):
        if show_count > 2:
            break
        pos, label = P_positions[ind]
        if label == 'P':
            Pon = pos - P_width / 2.0
            Poff = pos + P_width / 2.0
            for pi in xrange(ind - 1, -1, -1):
                if P_positions[pi][1] == 'Ponset':
                    Pon = P_positions[pi][0]
                    break
                elif (P_positions[pi][1] == 'Roffset' or
                        P_positions[pi][1] == 'Toffset' or
                        P_positions[pi][1] == 'T'):
                    Pon = P_positions[pi][0]
                    break
            for pi in xrange(ind + 1, len(P_positions)):
                if P_positions[pi][1] == 'Poffset':
                    Poff = P_positions[pi][0]
                    break
                elif P_positions[pi][1] == 'Ronset':
                    Poff = P_positions[pi][0]
                    break
            Pshape(raw_sig, (Pon, pos, Poff))

def Glimpse(record_index):
    '''Test case'''
    result_folder = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/test_results/r2'
    qt = QTloader()
    P_width = 50
    result_files = glob.glob(os.path.join(result_folder, 'sel*.json'))
    with open(result_files[record_index], 'r') as fin:
        data = json.load(fin)
        record_name = data[0][0]
        print 'Record name:', record_name

    sig = qt.load(record_name)
    raw_sig = sig['sig']
    results = data[0][1]
    fs = 250
    if abs(fs - 250.0) > 1e-6:
        raw_sig = scipy.signal.resample(raw_sig, int(len(raw_sig) / float(fs) * 250.0))
        fs_recover = fs
        fs = 250.0

    P_positions = results
    P_positions.sort(key = lambda x:x[0])

    show_count = 1
    segment_list = list()
    wholewave_list = list()
    for ind in xrange(0, len(P_positions)):
        if show_count > 2:
            break
        pos, label = P_positions[ind]
        if label == 'P':
            Pon = pos - P_width / 2.0
            Poff = pos + P_width / 2.0
            for pi in xrange(ind - 1, -1, -1):
                if P_positions[pi][1] == 'Ponset':
                    Pon = P_positions[pi][0]
                    break
                elif (P_positions[pi][1] == 'Roffset' or
                        P_positions[pi][1] == 'Toffset' or
                        P_positions[pi][1] == 'T'):
                    Pon = P_positions[pi][0]
                    break
            for pi in xrange(ind + 1, len(P_positions)):
                if P_positions[pi][1] == 'Poffset':
                    Poff = P_positions[pi][0]
                    break
                elif P_positions[pi][1] == 'Ronset':
                    Poff = P_positions[pi][0]
                    break
            expand_length_left = 3
            expand_length_right = 10
            left_bound = int(Pon - expand_length_left)
            right_bound = int(Poff + expand_length_right)
            p_wave_segment = CutWave(
                    raw_sig,
                    left_bound,
                    right_bound)

            segment_list.append(p_wave_segment)
            wholewave_list.append((Pon, pos, Poff))
            if len(segment_list) == 6:
                plt.figure(1)
                plt.clf()
                for ind in xrange(0, len(segment_list)):
                    plt.subplot(3, 2, ind + 1)
                    plt.plot(segment_list[ind])
                    plt.grid(True)
                plt.suptitle('%s[%d]' % (record_name, record_index))

                # plt.figure(2)
                # plt.clf()
                # for ind in xrange(0, len(segment_list)):
                    # pos_list = wholewave_list[ind]
                    # amp_list = [raw_sig[x] for x in pos_list]
                    # plt.subplot(3, 2, ind + 1)
                    # plt.plot(raw_sig)
                    # plt.plot(pos_list, amp_list, 'o', markersize = 12)
                    # plt.grid(True)
                # plt.suptitle(record_name)
                plt.show(block = False)

                segment_list = list()
                wholewave_list = list()
                show_count += 1
                pdb.set_trace()
            
    

if __name__ == '__main__':
    # Test1()
    TestQT(5)
    # for ind in xrange(5, 30):
        # Glimpse(ind)

