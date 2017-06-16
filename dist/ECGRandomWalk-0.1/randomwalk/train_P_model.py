#encoding:utf8
import os
import sys
import pickle
import matplotlib.pyplot as plt
import pdb
import time
import json
import random
import numpy as np

from random_walker import RandomWalker
from QTdata.loadQTdata import QTloader


curfilepath =  os.path.realpath(__file__)
current_folderpath = os.path.dirname(curfilepath)

def Test1(target_label = 'P', num_training = 25):
    '''Test case 1: random walk.'''
    qt = QTloader()
    record_list = qt.getreclist()
    training_list = random.sample(record_list, num_training)
    testing_list = list(set(record_list) - set(training_list))

    random_forest_config = dict(
            max_depth = 10)
    walker = RandomWalker(target_label = target_label,
            random_forest_config = random_forest_config)

    start_time = time.time()
    for record_name in training_list:
        print 'Collecting features from record %s.' % record_name
        sig = qt.load(record_name)
        walker.collect_training_data(sig['sig'], qt.getExpert(record_name))
    print 'random forest start training...'
    walker.training()
    print 'trianing used %.3f seconds' % (time.time() - start_time)

    for record_name in testing_list:
        sig = qt.load(record_name)
        raw_sig = sig['sig']

        seed_position = random.randint(100, len(raw_sig) - 100)
        plt.figure(1)
        plt.clf()
        plt.plot(sig['sig'], label = record_name)
        plt.title(target_label)
        for ti in xrange(0, 20):
            seed_position += random.randint(1,200)
            print 'testing...(position: %d)' % seed_position
            start_time = time.time()
            results = walker.testing_walk(sig['sig'], seed_position, iterations = 100,
                    stepsize = 10)
            print 'testing finished in %.3f seconds.' % (time.time() - start_time)

            pos_list, values = zip(*results)
            predict_pos = np.mean(pos_list[len(pos_list) / 2:])
            
            # amp_list = [raw_sig[int(x)] for x in pos_list]
            amp_list = []
            bias = raw_sig[pos_list[0]]
            for pos in pos_list:
                amp_list.append(bias)
                bias -= 0.01

            plt.plot(predict_pos,
                    raw_sig[int(predict_pos)],
                    'ro',
                    markersize = 14,
                    label = 'predict position')
            plt.plot(pos_list, amp_list, 'r',
                    label = 'walk path',
                    markersize = 3,
                    linewidth = 8,
                    alpha = 0.3)
            plt.xlim(min(pos_list) - 100, max(pos_list) + 100)
            plt.grid(True)
            plt.legend()
            plt.show(block = False)
            pdb.set_trace()

def testing(random_walker, raw_sig, seed_step_size = 200):
    result_list = list()
    start_time = time.time()
    for seed_position in xrange(0, len(raw_sig), seed_step_size):
        sys.stdout.write('\rTesting: %06d samples left.' % (len(raw_sig) - 1 - seed_position))
        sys.stdout.flush()

        results = random_walker.testing_walk(raw_sig,
                seed_position,
                iterations = 100,
                stepsize = 10)
        pos_list, values = zip(*results)
        predict_pos = np.mean(pos_list[len(pos_list) / 2:])
        confidence = 1.0
        result_list.append((predict_pos, random_walker.target_label, confidence, pos_list))
    print 'testing finished in %.3f seconds.' % (time.time() - start_time)
    return result_list

def TrainingModels(target_label, model_file_name, training_list):
    '''Randomly select num_training records to train, and test others.
    CP: Characteristic points
    '''
    qt = QTloader()
    record_list = qt.getreclist()
    testing_list = list(set(record_list) - set(training_list))

    random_forest_config = dict(
            max_depth = 10)
    walker = RandomWalker(target_label = target_label,
            random_forest_config = random_forest_config,
            random_pattern_file_name = os.path.join(os.path.dirname(model_file_name), 'random_pattern.json'))

    start_time = time.time()
    for record_name in training_list:
        CP_file_name = os.path.join('/home/alex/code/Python/EcgCharacterPointMarks', target_label, '%s_poslist.json' % record_name)

        # Add expert marks
        expert_marks = qt.getExpert(record_name)
        CP_marks = [x for x in expert_marks if x[1] == target_label]
        if len(CP_marks) == 0:
            continue

        # Add manual labels if possible
        if os.path.exists(CP_file_name) == True:
            with open(CP_file_name, 'r') as fin:
                CP_info = json.load(fin)
                poslist = CP_info['poslist']
                if len(poslist) == 0:
                    continue
                CP_marks.extend(zip(poslist, [target_label,] * len(poslist)))
            
        print 'Collecting features from record %s.' % record_name
        sig = qt.load(record_name)
        walker.collect_training_data(sig['sig'], CP_marks)
    print 'random forest start training(%s)...' % target_label
    walker.training()
    print 'trianing used %.3f seconds' % (time.time() - start_time)

    import joblib
    start_time = time.time()
    walker.save_model(model_file_name)
    print 'Serializing model time cost %f' % (time.time() - start_time)


def ContinueAddQtTrainingSamples(walker, target_label):
    '''Add QT training samples.'''
    qt = QTloader()
    record_list = qt.getreclist()

    start_time = time.time()
    for record_name in record_list:

        # Add expert marks
        expert_marks = qt.getExpert(record_name)
        CP_marks = [x for x in expert_marks if x[1] == target_label]
        if len(CP_marks) == 0:
            continue

        print 'Collecting features from QT record %s.' % record_name
        sig = qt.load(record_name)
        walker.collect_training_data(sig['sig'], CP_marks)

def TrainingModels_Changgeng(target_label, model_file_name):
    '''Randomly select num_training records to train, and test others.
    CP: Characteristic points
    '''
    
    import glob
    annot_jsonIDs = glob.glob(os.path.join(current_folderpath, 'data', 'labels', target_label, '*.json'))
    annot_jsonIDs = [os.path.split(x)[-1] for x in annot_jsonIDs]
    annot_jsonIDs = [x.split('.')[0] for x in annot_jsonIDs]
    # skip failed records        
    faillist = [8999,8374,6659, 6655,6059,5395,1401,1269,737,75,9524,9476]
    faillist = [str(x) for x in faillist]
    annot_jsonIDs = list(set(annot_jsonIDs) - set(faillist))

    

    from changgengLoader import ECGLoader
    ecg = ECGLoader(500, current_folderpath)

    random_forest_config = dict(
            max_depth = 10)
    walker = RandomWalker(target_label = target_label,
            random_forest_config = random_forest_config,
            random_pattern_file_name = os.path.join(os.path.dirname(model_file_name), 'random_pattern.json'))

    start_time = time.time()

    for record_ind in xrange(0, len(annot_jsonIDs)):
        record_name = annot_jsonIDs[record_ind]
        CP_file_name = os.path.join(current_folderpath, 'data', 'labels', target_label, '%s.json' % record_name)

        CP_marks = []

        # Add manual labels if possible
        if os.path.exists(CP_file_name) == True:
            with open(CP_file_name, 'r') as fin:
                CP_info = json.load(fin)
                poslist = CP_info['poslist']
                poslist = [int(x / 2) for x in poslist]
                mat_file_name = CP_info['mat_file_name']
                if len(poslist) == 0:
                    continue
                CP_marks.extend(zip(poslist, [target_label,] * len(poslist)))
            
        print 'Collecting features from record %s.' % record_name
        sig = ecg.load(record_name)
        raw_sig = sig[0]

        import scipy.signal
        resampled_sig = scipy.signal.resample_poly(raw_sig, 1, 2)
        raw_sig = resampled_sig
        # debug
        # plt.figure(1)
        # plt.plot(raw_sig, label = 'signal')
        # plt.plot(xrange(0, len(raw_sig), 2), resampled_sig, label = 'resmaple')
        # plt.legend()
        # plt.grid(True)
        # plt.title(record_name)
        # plt.show()

        walker.collect_training_data(raw_sig, CP_marks)

    # Add QT training samples
    # ContinueAddQtTrainingSamples(walker, target_label)

    print 'random forest start training(%s)...' % target_label
    walker.training()
    print 'trianing used %.3f seconds' % (time.time() - start_time)

    import joblib
    start_time = time.time()
    walker.save_model(model_file_name)
    print 'Serializing model time cost %f' % (time.time() - start_time)


if __name__ == '__main__':
    root_folder = 'data/Lw3Np4000/improved'
    # Refresh training list
    num_training = 105
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
    if num_training > 0:
        training_list.extend(random.sample(record_list, num_training))
    # Save training list
    with open(os.path.join(root_folder, 'training_list.json'), 'w') as fout:
        json.dump(training_list, fout, indent = 4)
    target_label = 'P'
    model_file_name = os.path.join(root_folder, '%s.mdl' % target_label)
    TrainingModels_Changgeng(target_label, model_file_name)
