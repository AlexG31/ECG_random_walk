#encoding:utf8
import os
import sys
import time
import glob
import numpy as np
import joblib
import scipy.signal
import pdb

from dpi.DPI_QRS_Detector import DPI_QRS_Detector as DPI
from randomwalk.random_walker import RandomWalker



def Testing(raw_sig_in, fs, model_list, walker_iterations = 100, walker_stepsize = 10):
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
    walk_results = Testing_random_walk_RR(raw_sig, fs_inner, r_list, model_list, iterations = walker_iterations, stepsize = walker_stepsize)

    walk_results.extend(zip(r_list, ['R',] * len(r_list)))
    # walk_results.extend(Testing_QS(raw_sig, fs_inner, r_list))
    # Change result indexes according to sampling frequency
    walk_results = [[x[0] / 250.0 * fs, x[1]] for x in walk_results]
    return walk_results

def Testing_QS(raw_sig, fs, r_list):
    '''Detect positions of Q and S based on given R position.'''
    if fs <= 1e-6:
        raise Exception('Unexpected sampling frequency of %f.' % fs)
    from dpi.QrsTypeDetector import QrsTypeDetector
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

def Testing_random_walk_RR(raw_sig, fs, qrs_locations, model_list, iterations = 100, stepsize = 10):
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
    if model_list is None or len(model_list) == 0:
        return []
    testing_results = list()

    # Maybe batch walk?
    # feature_extractor = None
    feature_extractor = model_list[0][0].GetFeatureExtractor(raw_sig)

    # For benchmarking
    longest_path_len = 0
    longest_path_Rpos = -1
    walker_time_cost = 0
    walker_count = 0
    Tnew_list = list()

    def RunWalkerModel(walker_model, seed_position, confined_range):
        '''Run random walk detection model.'''
        if abs(fs - 250.0) > 1e-6:
            raise Exception('Bias has default fs = 250.0Hz!')

        start_time = time.time()

        results = walker_model.testing_walk_extractor(feature_extractor,
                seed_position,
                iterations = 100,
                stepsize = 10,
                confined_range = confined_range)
        # print 'testing time cost: %f s.' % (time.time() - start_time)
        # walker_time_cost += time.time() - start_time
        # walker_count += 1

        path, probability = zip(*results)
        # path_len = len(set(path))
        # if path_len > longest_path_len:
            # longest_path_len = path_len
            # longest_path_Rpos = R_pos
        
        # Tnew_list.append(len(set(path)))
        predict_position = int(np.mean(path[len(path) / 2:]) / 250.0 * fs)
        testing_results.append((predict_position,
                walker_model.target_label))
        return predict_position

    
    # For QRS boundaries
    prev_Roffset = None
    back_Ronset = None

    # Get model dict
    model_dict = dict()
    for walker_model, bias, model_label in model_list:
        model_dict[model_label] = [walker_model, bias]

    # for R_pos in qrs_locations:
    for qrs_index in xrange(0, len(qrs_locations)):
        R_pos = qrs_locations[qrs_index]
        R_pos = R_pos * 250.0 / fs

        # Boundaries for Ponset and Toffset
        left_QRS_bound = 0
        right_QRS_bound = len(raw_sig)
        if qrs_index > 0:
            left_QRS_bound = qrs_locations[qrs_index - 1]
        if qrs_index + 1 < len(qrs_locations):
            right_QRS_bound = qrs_locations[qrs_index + 1]

        # Detect Ronset and Roffset first
        if qrs_index == 0:
            model_label = 'Ronset'
            walker_model, bias = model_dict[model_label]
            bias = int(float(fs) * bias)
            confined_range = [left_QRS_bound, R_pos]
            current_Ronset = RunWalkerModel(walker_model, R_pos + bias, confined_range)
        else:
            current_Ronset = back_Ronset

        model_label = 'Roffset'
        walker_model, bias = model_dict[model_label]
        bias = int(float(fs) * bias)
        confined_range = [R_pos, right_QRS_bound]
        current_Roffset = RunWalkerModel(walker_model, R_pos + bias, confined_range)

        # Get back_Ronset
        if qrs_index + 1 < len(qrs_locations):
            model_label = 'Ronset'
            walker_model, bias = model_dict[model_label]
            bias = int(float(fs) * bias)
            confined_range = [R_pos, qrs_locations[qrs_index + 1]]
            back_Ronset = RunWalkerModel(walker_model, qrs_locations[qrs_index + 1] + bias, confined_range)
            right_QRS_bound = back_Ronset
            # right_QRS_bound = qrs_locations[qrs_index + 1]
        else:
            back_Ronset = None

        # P wave
        model_label = 'P'
        walker_model, bias = model_dict[model_label]
        bias = int(float(fs) * bias)
        confined_range = [left_QRS_bound, R_pos]
        current_P = RunWalkerModel(walker_model, R_pos + bias, confined_range)


        # print 'Rpos:', R_pos
        # print 'left QRS bound:', left_QRS_bound
        # print 'right QRS bound:', right_QRS_bound
        # print 'current_Ronset:', current_Ronset
        # print 'current_Roffset:', current_Roffset
        # print 'Confined Range:', confined_range
        # plt.figure(1)
        # plt.clf()
        # plt.plot(raw_sig)
        # plt.plot(current_P, raw_sig[current_P], 'go', markersize = 12)
        # plt.grid(True)
        # plt.xlim((current_P - 50, current_P + 50))
        # plt.show(block = False)
        # pdb.set_trace()

        model_label = 'Ponset'
        walker_model, bias = model_dict[model_label]
        bias = int(float(fs) * bias)
        confined_range = [left_QRS_bound, current_P]
        current_Ponset = RunWalkerModel(walker_model, R_pos + bias, confined_range)

        model_label = 'Poffset'
        walker_model, bias = model_dict[model_label]
        bias = int(float(fs) * bias)
        confined_range = [current_P, R_pos]
        current_Ponset = RunWalkerModel(walker_model, R_pos + bias, confined_range)

        # T wave
        model_label = 'T'
        walker_model, bias = model_dict[model_label]
        bias = int(float(fs) * bias)
        confined_range = [R_pos, right_QRS_bound]
        current_T = RunWalkerModel(walker_model, R_pos + bias, confined_range)

        model_label = 'Tonset'
        walker_model, bias = model_dict[model_label]
        bias = int(float(fs) * bias)
        confined_range = [R_pos, current_T]
        current_Tonset = RunWalkerModel(walker_model, R_pos + bias, confined_range)

        model_label = 'Toffset'
        walker_model, bias = model_dict[model_label]
        bias = int(float(fs) * bias)
        confined_range = [current_T, right_QRS_bound]
        current_Tonset = RunWalkerModel(walker_model, R_pos + bias, confined_range)
        
        # Update prev QRS
        prev_Roffset = current_Roffset
         
            
    # print 'Walker time cost %f seconds.' % walker_time_cost
    # print 'Walker average time cost %f seconds.' % (walker_time_cost / walker_count)
    # print 'Average number of new samples to test: %f.' % np.mean(Tnew_list)
    # print 'Longest path:%d, with Rpos:%d' % (longest_path_len, longest_path_Rpos)
    # print 'Len of Tnewlist:', len(Tnew_list)

    return testing_results



def GetModels(model_folder, pattern_file_name):
    '''Returns model dict.'''
    # label_list = ['P', 'Ponset', 'Poffset',
            # 'T', 'Toffset', 'Tonset']
    label_list = ['P', 'Ponset', 'Poffset',
            'T', 'Toffset', 'Tonset',
            'Ronset', 'Roffset']
    bias_list = [
                -0.19, -0.195, -0.185,
                0.26, 0.27, 0.1,
                -0.02, 0.02,
            ]
    # Get model dict
    models = list()
    for target_label, bias in zip(label_list, bias_list):
        model_file_name = os.path.join(model_folder, target_label + '.mdl')
        walker_model = RandomWalker(target_label = target_label,
                random_pattern_file_name = pattern_file_name)
        walker_model.load_model(model_file_name)
        models.append((walker_model, bias, target_label))
    return models



def Test1():
    '''Test case1.'''
    import matplotlib.pyplot as plt
    record_name = 'sel30'
    fs = 250.0
    #from QTdata.loadQTdata import QTloader
    #qt = QTloader()
    #sig = qt.load(record_name)
    #raw_sig = sig['sig']
    raw_sig = [1.3, ] * 10000
    raw_sig = raw_sig[0:250 * 20]

    model_folder = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/m3_full_models'
    pattern_file_name = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/m3_full_models/random_pattern.json'
    model_list = GetModels(model_folder, pattern_file_name)
    start_time = time.time()
    # results = Testing_random_walk(raw_sig, 250.0, r_list, model_list)
    results = Testing(raw_sig, 250.0, model_list)
    #print 'Testing time cost %f secs.' % (time.time() - start_time)

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
        plt.plot(pos_list, amp_list, 'o',
                markersize = 15,
                label = label)
    plt.title(record_name)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    Test1()

