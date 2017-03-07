#encoding:utf8
import os
import sys
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import joblib
import scipy.signal
import pdb

from dpi.DPI_QRS_Detector import DPI_QRS_Detector as DPI
from dpi.QrsTypeDetector import QrsTypeDetector
from QTdata.loadQTdata import QTloader
from random_walker import RandomWalker



def Testing(raw_sig_in, fs, model_list):
    '''Testing API.
    Input:
        raw_sig: ECG signal.
        model_list: random walker models and their biases relative to QRS.
    Output:
        List of (pos, label) pairs.
    '''
    if fs <= 1e-6:
        raise Exception('Unexpected sampling frequency of %f.' % fs)
    raw_sig = raw_sig_in[:]
    if abs(fs - 250.0) > 1e-6:
        raw_sig = scipy.signal.resample(raw_sig, int(len(raw_sig) / float(fs) * 250.0))
    fs_inner = 250

    dpi = DPI(debug_info = dict())
    r_list = dpi.QRS_Detection(raw_sig, fs = fs_inner)
    walk_results = Testing_random_walk(raw_sig, fs_inner, r_list, model_list)

    walk_results.extend(zip(r_list, ['R',] * len(r_list)))
    walk_results.extend(Testing_QS(raw_sig, fs_inner, r_list))
    # Change result indexes according to sampling frequency
    walk_results = [[x[0] / 250.0 * fs, x[1]] for x in walk_results]
    return walk_results

def Testing_QS(raw_sig, fs, r_list):
    '''Detect positions of Q and S based on given R position.'''
    if fs <= 1e-6:
        raise Exception('Unexpected sampling frequency of %f.' % fs)
    qrstype = QrsTypeDetector(fs)
    results = list()
    for r_pos in r_list:
        r_pos = int(r_pos)
        qrs_pos, qrs_text = qrstype.GetQrsType(
                raw_sig,
                r_pos - 10 / 250.0 * fs, r_pos, r_pos + 10 / 250.0 * fs,
                debug_plot = False)
        results.append((qrs_pos[0], 'Ronset'))
        results.append((qrs_pos[2], 'Roffset'))
    return results

def Testing_random_walk(raw_sig, fs, qrs_locations, model_list):
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
    

    # For benchmarking
    Tnew_list = list()
    walker_time_cost = 0
    walker_count = 0
    # Maybe batch walk?
    feature_extractor = None
    for R_pos in qrs_locations:
        R_pos = R_pos * 250.0 / fs
        for walker, bias in model_list:
            if abs(fs - 250.0) > 1e-6:
                raise Exception('Bias has default fs = 250.0Hz!')
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
            predict_position = int(np.mean(path[len(path) / 2:]) / 250.0 * fs)
            testing_results.append((predict_position,
                    walker.target_label))
            
    print 'Walker time cost %f seconds.' % walker_time_cost
    print 'Walker average time cost %f seconds.' % (walker_time_cost / walker_count)
    print 'Average number of new samples to test: %f.' % np.mean(Tnew_list)
    return testing_results



def GetModels(model_folder, pattern_file_name):
    '''Returns model dict.'''
    label_list = ['P', 'Ponset', 'Poffset',
            'T', 'Toffset', 'Tonset']
    # label_list = ['P', 'Ponset', 'Poffset',
            # 'T', 'Toffset',
            # 'Ronset', 'Roffset']
    bias_list = [
                -0.19, -0.195, -0.185,
                0.26, 0.27, 0.1,
                -0.02, 0.02,
            ]
    # Get model dict
    models = list()
    for target_label, bias in zip(label_list, bias_list):
        model_file_name = os.path.join(model_folder, target_label + '.mdl')
        walker = RandomWalker(target_label = target_label,
                random_pattern_file_name = pattern_file_name)
        walker.load_model(model_file_name)
        models.append((walker, bias))
    return models



def Test1():
    '''Test case1.'''
    record_name = 'sel30'
    fs = 250.0
    qt = QTloader()
    sig = qt.load(record_name)
    raw_sig = sig['sig']
    raw_sig = raw_sig[0:250 * 20]

    model_folder = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/m3_full_models'
    pattern_file_name = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/m3_full_models/random_pattern.json'
    model_list = GetModels(model_folder, pattern_file_name)
    start_time = time.time()
    # results = Testing_random_walk(raw_sig, 250.0, r_list, model_list)
    results = Testing(raw_sig, 250.0, model_list)
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
        pos_list = [int(x[0]) for x in results if x[1] == label]
        amp_list = [raw_sig[x] for x in pos_list]
        plt.plot(pos_list, amp_list, 'o',
                markersize = 15,
                label = label)
    plt.title(record_name)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    Test1()

