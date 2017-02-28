#encoding:utf8
import os
import sys
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pdb

from dpi.DPI_QRS_Detector import DPI_QRS_Detector as DPI
from QTdata.loadQTdata import QTloader
from random_walker import RandomWalker


def Testing(raw_sig, fs, model_list):
    '''Testing API.
    Input:
        raw_sig: ECG signal.
        model_list: random walker models and their biases relative to QRS.
    Output:
        List of (pos, label) pairs.
    '''
    dpi = DPI(debug_info = dict())
    r_list = dpi.QRS_Detection(raw_sig, fs = fs)
    walk_results = Testing_random_walk(raw_sig, fs, r_list, model_list)
    walk_results.extend(zip(r_list, ['R',] * len(r_list)))
    return walk_results

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
    testing_results = list()

    # For benchmarking
    Tnew_list = list()
    walker_time_cost = 0
    walker_count = 0
    # Maybe batch walk?
    feature_extractor = None
    for R_pos in qrs_locations:
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
            predict_position = int(np.mean(path[len(path) / 2:]))
            testing_results.append((predict_position,
                    walker.target_label))
            
    print 'Walker time cost %f seconds.' % walker_time_cost
    print 'Walker average time cost %f seconds.' % (walker_time_cost / walker_count)
    print 'Average number of new samples to test: %f.' % np.mean(Tnew_list)
    return testing_results



def GetModels():
    '''Returns model dict.'''
    label_list = ['P', 'Ponset', 'Poffset',
            'T', 'Toffset',
            'Ronset', 'Roffset']
    bias_list = [
                -0.19, -0.195, -0.185,
                0.26, 0.27,
                -0.02, 0.02,
            ]
    # Get model dict
    model_folder = '/home/alex/LabGit/ECG_random_walk/training/data/m4_models'
    models = list()
    for target_label, bias in zip(label_list, bias_list):
        model_file_name = os.path.join(model_folder, target_label + '.mdl')
        walker = RandomWalker(target_label = target_label)
        walker.load_model(model_file_name)
        models.append((walker, bias))
    return models



def Test1():
    '''Test case1.'''
    fs = 250.0
    qt = QTloader()
    sig = qt.load('sel103')
    raw_sig = sig['sig']
    raw_sig = raw_sig[0:250 * 20]

    model_list = GetModels()
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
    plt.title('sel32')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    Test1()

