#encoding:utf8
import os
import sys
import time
import glob
import multiprocessing
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import pdb

from dpi.DPI_QRS_Detector import DPI_QRS_Detector as DPI
from QTdata.loadQTdata import QTloader
from random_walker import RandomWalker
from test_api import Testing
from test_api import GetModels
import feature_extractor.random_patterns as random_patterns




def TestSignal(raw_sig, fs, test_range, model_folder, random_pattern_file_name):
    '''Wrapper for testing.'''
    raw_sig = raw_sig[test_range[0]:test_range[1]]

    model_list = GetModels(model_folder, random_pattern_file_name)
    start_time = time.time()
    # results = Testing_random_walk(raw_sig, 250.0, r_list, model_list)
    results = Testing(raw_sig, fs, model_list)
    print 'Testing time cost %f secs.' % (time.time() - start_time)

    samples_count = len(raw_sig)
    time_span = samples_count / fs
    print 'Span of testing range: %f samples(%f seconds).' % (samples_count, time_span)

    return results
    
def TestQT(record_name, save_result_folder, model_folder, random_pattern_file_name):
    '''Test case1.'''
    fs = 250.0
    qt = QTloader()

    sig = qt.load(record_name)
    expert_annotations = qt.getExpert(record_name)
    pos_list, label_list = zip(*expert_annotations)
    test_range = [np.min(pos_list) - 100, np.max(pos_list) + 100]
    
    result_mat = list()

    print 'Lead1'
    raw_sig = sig['sig']
    results = TestSignal(raw_sig, fs, test_range, model_folder, random_pattern_file_name)
    for ind in xrange(0, len(results)):
        results[ind] = [results[ind][0] + test_range[0], results[ind][1]]
    result_mat.append((record_name, results))

    print 'Lead2'
    raw_sig = sig['sig2']
    results = TestSignal(raw_sig, fs, test_range, model_folder, random_pattern_file_name)
    for ind in xrange(0, len(results)):
        results[ind] = [results[ind][0] + test_range[0], results[ind][1]]
    result_mat.append((record_name + '_sig2', results))
    
    result_file_name = os.path.join(save_result_folder, '%s.json' % record_name)
    with open(result_file_name, 'w') as fout:
        json.dump(result_mat, fout, indent = 4)
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

def SplitQTrecords(num_training = 75):
    '''Split records for testing & training.'''
    trianing_list = list()
    qt = QTloader()
    record_list = qt.getreclist()
    must_train_list = [
        "sel35", 
        "sel36", 
        "sel31", 
        "sel38", 
        "sel39", 
        "sel820", 
        "sel51", 
        "sele0104", 
        "sele0107", 
        "sel223", 
        "sele0607", 
        "sel102", 
        "sele0409", 
        "sel41", 
        "sel40", 
        "sel43", 
        "sel42", 
        "sel45", 
        "sel48", 
        "sele0133", 
        "sele0116", 
        "sel14172", 
        "sele0111", 
        "sel213", 
        "sel14157", 
        "sel301"
            ]
    num_training -= len(must_train_list)
    record_list = list(set(record_list) - set(must_train_list))
    training_list = must_train_list
    # Refresh training list
    if num_training > 0:
        training_list.extend(random.sample(record_list, num_training))
    testing_list = list(set(record_list) - set(training_list))

    return (training_list, testing_list)

def TrainingModels(target_label, model_file_name, training_list, random_pattern_file_name):
    '''Randomly select num_training records to train, and test others.'''
    qt = QTloader()

    random_forest_config = dict(
            max_depth = 10,
            n_jobs = 1)
    walker = RandomWalker(target_label = target_label,
            random_forest_config = random_forest_config,
            random_pattern_file_name = random_pattern_file_name)

    start_time = time.time()
    for record_name in training_list:
        print 'Collecting features from record %s.' % record_name
        sig = qt.load(record_name)
        walker.collect_training_data(sig['sig'], qt.getExpert(record_name))
    print 'random forest start training(%s)...' % target_label
    walker.training()
    print 'trianing used %.3f seconds' % (time.time() - start_time)

    start_time = time.time()
    walker.save_model(model_file_name)
    print 'Serializing model time cost %f' % (time.time() - start_time)

    
def pool_test(args):
    '''Function for pool testing.'''
    record_name, save_result_folder, model_folder, random_pattern_file_name = args
    print 'Testing %s' % record_name
    TestQT(record_name, save_result_folder, model_folder, random_pattern_file_name)
    
def RoundExperiment(save_result_folder):
    '''Training & Testing 30 records on QT'''
    qt = QTloader()
    # Training
    if os.path.exists(save_result_folder):
        cmd = raw_input('Folder: \n %s\nAlready exists, remove it?(Y/N)' % save_result_folder)
        if cmd in {'Y', 'y'}:
            import shutil
            shutil.rmtree(save_result_folder)
        else:
            return None

    os.makedirs(save_result_folder)
    model_folder = os.path.join(save_result_folder, 'models')
    os.makedirs(model_folder)
    label_list = ['P', 'Ponset', 'Poffset',
            'T', 'Toffset',]
    training_list, testing_list = SplitQTrecords()

    # Save training list
    with open(os.path.join(save_result_folder, 'training_list.json'), 'w') as fout:
        json.dump(training_list, fout, indent = 4)
    # Save random patterns
    random_pattern_file_name = os.path.join(os.path.join(model_folder, 'random_pattern.json'))
    # Refresh random patterns
    configure = dict(
            WT_LEVEL = 6,
            fs = 250.0,
            winlen_ratio_to_fs = 3,
            WTrandselfeaturenumber_apprx = 10000,
            totally_random_pair_number = 1000,
            )
    random_patterns.RefreshRswtPairs(configure, random_pattern_file_name)
    

    for target_label in label_list:
        model_file_name = os.path.join(model_folder, '%s.mdl' % target_label)
        TrainingModels(target_label, model_file_name, training_list, random_pattern_file_name)

    # Testing
    lt = len(testing_list)
    arg_list = zip(testing_list,
            [save_result_folder,] * lt,
            [model_folder,] * lt,
            [random_pattern_file_name,] * lt,
            )
    pool = multiprocessing.Pool(processes = 3)
    pool.map(pool_test, arg_list)
    pool.close()
    pool.join()
    
    

if __name__ == '__main__':
    # result_folder = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/test_results/r3/round1'
    # TestQT('sel40', result_folder, os.path.join(result_folder, 'models'), os.path.join(result_folder, 'models', 'random_pattern.json'))

    result_folder = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/test_results/r3'
    for ind in xrange(15, 30 + 1):
        RoundExperiment(os.path.join(result_folder, 'round%d' % ind))

