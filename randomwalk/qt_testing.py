#encoding:utf8
import os
import sys
import time
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pdb

from dpi.DPI_QRS_Detector import DPI_QRS_Detector as DPI
from QTdata.loadQTdata import QTloader
from random_walker import RandomWalker
from test_api import Testing




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



def TestSignal(raw_sig, fs, test_range):
    '''Wrapper for testing.'''
    raw_sig = raw_sig[test_range[0]:test_range[1]]

    model_list = GetModels()
    start_time = time.time()
    # results = Testing_random_walk(raw_sig, 250.0, r_list, model_list)
    results = Testing(raw_sig, fs, model_list)
    print 'Testing time cost %f secs.' % (time.time() - start_time)

    samples_count = len(raw_sig)
    time_span = samples_count / fs
    print 'Span of testing range: %f samples(%f seconds).' % (samples_count, time_span)

    return results
    
def TestQT(save_result_folder):
    '''Test case1.'''
    fs = 250.0
    qt = QTloader()
    record_list = qt.getreclist()
    with open(os.path.join(save_result_folder, 'training_list.json'), 'r') as fin:
        training_list = json.load(fin)
        testing_list = list(set(record_list) - set(training_list))
    for record_name in testing_list:
        print 'Testing %s' % record_name
        sig = qt.load(record_name)
        expert_annotations = qt.getExpert(record_name)
        pos_list, label_list = zip(*expert_annotations)
        test_range = [np.min(pos_list) - 100, np.max(pos_list) + 100]
        
        result_mat = list()
        print 'Lead1'
        raw_sig = sig['sig']
        results = TestSignal(raw_sig, fs, test_range)
        for ind in xrange(0, len(results)):
            results[ind] = [results[ind][0] + test_range[0], results[ind][1]]
        result_mat.append((record_name, results))
        print 'Lead2'
        raw_sig = sig['sig2']
        results = TestSignal(raw_sig, fs, test_range)
        for ind in xrange(0, len(results)):
            results[ind] = [results[ind][0] + test_range[0], results[ind][1]]
        result_mat.append((record_name + '_sig2', results))
        
        result_file_name = os.path.join(save_result_folder, '%s.json' % record_name)
        with open(result_file_name, 'w') as fout:
            json.dump(result_mat, fout)
            print 'Results saved as %s.' % result_file_name

        # Display results
        # plt.figure(1)
        # plt.plot(raw_sig, label = 'ECG')
        # pos_list, label_list = zip(*results)
        # labels = set(label_list)
        # for label in labels:
            # pos_list = [int(x[0]) for x in results if x[1] == label]
            # amp_list = [raw_sig[x] for x in pos_list]
            # plt.plot(pos_list, amp_list, 'o',
                    # markersize = 15,
                    # label = label)
        # plt.title(record_name)
        # plt.grid(True)
        # plt.legend()
        # plt.show()

if __name__ == '__main__':
    result_folder = '/home/alex/LabGit/ECG_random_walk/testing/data/test_results/r2'
    TestQT(result_folder)

