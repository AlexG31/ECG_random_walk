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
from test_api import Testing_QS
from test_api import GetModels

import scipy.io as sio


def Testing_random_walk(raw_sig_in, fs, qrs_locations, model_list):
    '''
    Testing with random walk based on QRS locations.
    Inputs:
        raw_sig: ECG input.
        qrs_locations: indexes of QRS locations.
        model_list: random walker models and their biases relative to QRS.
    Output:
        List of (pos, label) pairs.
    '''
    if fs <= 1e-6:
        raise Exception('Unexpected sampling frequency of %f.' % fs)
    testing_results = list()
    raw_sig = raw_sig_in[:]
    fs_recover = fs
    if abs(fs - 250.0) > 1e-6:
        raw_sig = scipy.signal.resample(raw_sig, int(len(raw_sig) / float(fs) * 250.0))
        fs_recover = fs
        fs = 250.0
    

    # For benchmarking
    Tnew_list = list()
    walker_time_cost = 0
    walker_count = 0
    # Maybe batch walk?
    feature_extractor = None
    for R_pos in qrs_locations:
        R_pos = R_pos * 250.0 / fs_recover
        for walker, bias in model_list:
            bias = int(float(fs) * bias)

            if feature_extractor is None:
                feature_extractor = walker.GetFeatureExtractor(raw_sig)
            start_time = time.time()
            results = walker.testing_walk_extractor(feature_extractor, R_pos + bias,
                    iterations = 100,
                    stepsize = 10)
            # results = walker.testing_walk(raw_sig, R_pos + bias,
                    # iterations = 100,
                    # stepsize = 10)
            # print 'testing time cost: %f s.' % (time.time() - start_time)
            walker_time_cost += time.time() - start_time
            walker_count += 1

            path, probability = zip(*results)
            Tnew_list.append(len(set(path)))
            predict_position = int(np.mean(path[len(path) / 2:]) / 250.0 * fs_recover)
            testing_results.append((predict_position,
                    walker.target_label))

            # Debug plot path
            continue
            if walker.target_label != 'Poffset':
                continue
            plt.figure(1)
            plt.plot(raw_sig, linewidth = 2,
                    label = 'ECG')
            plt.plot(path, np.linspace(0, -1.5, len(path)),
                    'm',
                    alpha = 0.4,
                    linewidth = 3,
                    label = '%s Path' % walker.target_label)
            pos = predict_position * 250.0 / fs_recover
            plt.plot(pos, raw_sig[pos], 'ro', label = walker.target_label)
            plt.legend()
            plt.xlim(pos - 100, pos + 100)
            plt.show(block = False)
            pdb.set_trace()
            
    print 'Walker time cost %f seconds.' % walker_time_cost
    print 'Walker average time cost %f seconds.' % (walker_time_cost / walker_count)
    print 'Average number of new samples to test: %f.' % np.mean(Tnew_list)
    return testing_results


def Test1():
    '''Test case1.'''
    # data = sio.loadmat('./data/ft.mat')
    # v2 = np.squeeze(data['II'])
    # raw_sig = v2
    # fs = 500
    qt = QTloader()
    sig = qt.load('sel32')
    raw_sig = sig['sig'][1000:3000]
    fs = 250
    # raw_sig = scipy.signal.resample(raw_sig, len(raw_sig) / 2)
    # fs = 250

    model_folder = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/m3_full_models'
    pattern_file_name = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/m3_full_models/random_pattern.json'
    model_list = GetModels(model_folder, pattern_file_name)
    start_time = time.time()

    # First: QRS detection
    dpi = DPI(debug_info = dict())
    r_list = dpi.QRS_Detection(raw_sig, fs = fs)
    results = zip(r_list, len(r_list) * ['R', ])
    results.extend(Testing_QS(raw_sig, fs, r_list))
    walk_results = Testing_random_walk(raw_sig, fs, r_list, model_list)
    results.extend(walk_results)

    # results = Testing(raw_sig, fs, model_list)
    print 'Testing time cost %f secs.' % (time.time() - start_time)

    samples_count = len(raw_sig)
    time_span = samples_count / fs
    print 'Span of testing range: %f samples(%f seconds).' % (samples_count, time_span)

    with open('./data/new_result.json', 'w') as fout:
        json.dump(results, fout, indent = 4)

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

