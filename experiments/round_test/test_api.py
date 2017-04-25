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


curfilepath =  os.path.realpath(__file__)
current_folderpath = os.path.dirname(curfilepath)

def Denoise(raw_sig, level = 7, low = 1, high = 1):
    '''Denoise with wavelet.'''
    def crop_data_for_swt(rawsig):
        '''Padding zeros to make the length of the signal to 2^N.'''
        import numpy as np
        if isinstance(rawsig, np.ndarray):
            rawsig = rawsig.tolist()
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

    import pywt
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
    print len_sig
    return raw_sig

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

    walk_results = Testing_random_walk_RR_batch(raw_sig, fs_inner, r_list, model_list, iterations = walker_iterations, stepsize = walker_stepsize)

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

def Testing_random_walk_RR_batch(raw_sig, fs, qrs_locations, model_list, iterations = 100, stepsize = 10, batch_size = 100):
    '''
    Batch testing with random walk based on QRS locations.
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

    def RunWalkerModel(walker_model, seed_positions, confined_ranges):
        '''Run random walk detection model.
        Input:
            walker_model: random walk regressor for a certain label.
            seed_positions: list of seed position
            confined_ranges: list of confined_range
        '''
        if abs(fs - 250.0) > 1e-6:
            raise Exception('Bias has default fs = 250.0Hz!')

        # First add to prepare testing list
        for seed_position, confined_range in zip(seed_positions, confined_ranges):
            walker_model.prepareTestSample(seed_position, confined_range)

        start_time = time.time()

        # Second, Testing all prepared positions
        path_list, scores_list = walker_model.runPreparedTesting(feature_extractor)

        predict_position_list = list()
        for path in path_list:
            # Tnew_list.append(len(set(path)))
            predict_position = int(np.mean(path[len(path) / 2:]) / 250.0 * fs)

            # For return value of this function
            predict_position_list.append(predict_position)
            # For return value of super function
            testing_results.append((predict_position,
                    walker_model.target_label))
        return predict_position_list

    
    # For QRS boundaries
    back_Ronset = None

    # Get model dict
    model_dict = dict()
    for walker_model, bias, model_label in model_list:
        model_dict[model_label] = [walker_model, bias]

    # for R_pos in qrs_locations:
    for batch_qrs_index in xrange(0, len(qrs_locations), batch_size):
        # Get R position batch
        # R_position_batch = list()
        # for R_pos in qrs_locations[batch_qrs_index:batch_qrs_index + batch_size]:
            # R_pos = qrs_locations[batch_qrs_index]
            # R_pos = R_pos * 250.0 / fs
            # R_position_batch.append(R_pos)

        seed_positions_dict = dict()
        confined_ranges_dict = dict()

        # Detect Ronset and Roffset first
        model_label = 'Ronset'
        seed_positions_dict[model_label] = list()
        confined_ranges_dict[model_label] = list()

        for qrs_index in xrange(batch_qrs_index, min(len(qrs_locations), batch_qrs_index + batch_size)):
            seed_position = None
            confined_range = None

            R_pos = qrs_locations[qrs_index]
            R_pos = R_pos * 250.0 / fs

            # Boundaries 
            left_QRS_bound = 0
            right_QRS_bound = len(raw_sig)
            if qrs_index > 0:
                left_QRS_bound = qrs_locations[qrs_index - 1]
            if qrs_index + 1 < len(qrs_locations):
                right_QRS_bound = qrs_locations[qrs_index + 1]

            if qrs_index == 0:
                walker_model, bias = model_dict[model_label]
                bias = int(float(fs) * bias)

                confined_range = [left_QRS_bound, R_pos]
                confined_ranges_dict[model_label].append(confined_range)
                seed_position = bias + R_pos
                seed_positions_dict[model_label].append(seed_position)
                # current_Ronset = RunWalkerModel(walker_model, R_pos + bias, confined_range)
            # Get back_Ronset
            if qrs_index + 1 < len(qrs_locations):
                walker_model, bias = model_dict[model_label]
                bias = int(float(fs) * bias)

                confined_range = [R_pos, qrs_locations[qrs_index + 1]]
                confined_ranges_dict[model_label].append(confined_range)
                seed_position = qrs_locations[qrs_index + 1] + bias
                seed_positions_dict[model_label].append(seed_position)
                # back_Ronset = RunWalkerModel(walker_model, qrs_locations[qrs_index + 1] + bias, confined_range)
                # right_QRS_bound = back_Ronset
                # right_QRS_bound = qrs_locations[qrs_index + 1]
            # else:
                # back_Ronset = None
        # Start testing
        walker_model, bias = model_dict[model_label]
        batch_Ronset_list = RunWalkerModel(walker_model, seed_positions_dict[model_label], confined_ranges_dict[model_label])


        # Testing Roffset
        model_label = 'Roffset'
        seed_positions_dict[model_label] = list()
        confined_ranges_dict[model_label] = list()

        for qrs_index in xrange(batch_qrs_index, min(len(qrs_locations), batch_qrs_index + batch_size)):
            seed_position = None
            confined_range = None

            R_pos = qrs_locations[qrs_index]
            R_pos = R_pos * 250.0 / fs
            # Boundaries 
            left_QRS_bound = 0
            right_QRS_bound = len(raw_sig)
            if qrs_index > 0:
                left_QRS_bound = qrs_locations[qrs_index - 1]
            if qrs_index + 1 < len(qrs_locations):
                right_QRS_bound = qrs_locations[qrs_index + 1]

            walker_model, bias = model_dict[model_label]
            bias = int(float(fs) * bias)

            confined_range = [R_pos, right_QRS_bound]
            confined_ranges_dict[model_label].append(confined_range)
            seed_position = bias + R_pos
            seed_positions_dict[model_label].append(seed_position)
            # current_Roffset = RunWalkerModel(walker_model, R_pos + bias, confined_range)
        # Start testing
        walker_model, bias = model_dict[model_label]
        batch_Roffset_list = RunWalkerModel(walker_model, seed_positions_dict[model_label], confined_ranges_dict[model_label])


        # P wave
        model_label = 'P'
        seed_positions_dict[model_label] = list()
        confined_ranges_dict[model_label] = list()

        for qrs_index in xrange(batch_qrs_index, min(len(qrs_locations), batch_qrs_index + batch_size)):
            seed_position = None
            confined_range = None

            R_pos = qrs_locations[qrs_index]
            R_pos = R_pos * 250.0 / fs
            # Boundaries 
            left_QRS_bound = 0
            right_QRS_bound = len(raw_sig)
            if qrs_index > 0:
                left_QRS_bound = qrs_locations[qrs_index - 1]
            if qrs_index + 1 < len(qrs_locations):
                right_QRS_bound = qrs_locations[qrs_index + 1]

            walker_model, bias = model_dict[model_label]
            bias = int(float(fs) * bias)

            confined_range = [left_QRS_bound, R_pos]
            confined_ranges_dict[model_label].append(confined_range)
            seed_position = bias + R_pos
            seed_positions_dict[model_label].append(seed_position)
            # current_P = RunWalkerModel(walker_model, R_pos + bias, confined_range)
        # Start testing
        walker_model, bias = model_dict[model_label]
        batch_P_list = RunWalkerModel(walker_model, seed_positions_dict[model_label], confined_ranges_dict[model_label])


        # Testing Ponset
        model_label = 'Ponset'
        seed_positions_dict[model_label] = list()
        confined_ranges_dict[model_label] = list()

        for qrs_index, current_P in zip(xrange(batch_qrs_index, min(len(qrs_locations), batch_qrs_index + batch_size)), batch_P_list):
            seed_position = None
            confined_range = None

            R_pos = qrs_locations[qrs_index]
            R_pos = R_pos * 250.0 / fs
            # Boundaries 
            left_QRS_bound = 0
            right_QRS_bound = len(raw_sig)
            if qrs_index > 0:
                left_QRS_bound = qrs_locations[qrs_index - 1]
            if qrs_index + 1 < len(qrs_locations):
                right_QRS_bound = qrs_locations[qrs_index + 1]

            walker_model, bias = model_dict[model_label]
            bias = int(float(fs) * bias)

            confined_range = [left_QRS_bound, current_P]
            confined_ranges_dict[model_label].append(confined_range)
            seed_position = bias + R_pos
            seed_positions_dict[model_label].append(seed_position)
            # current_Ponset = RunWalkerModel(walker_model, R_pos + bias, confined_range)
        # Start testing
        walker_model, bias = model_dict[model_label]
        batch_Ponset_list = RunWalkerModel(walker_model, seed_positions_dict[model_label], confined_ranges_dict[model_label])



        # Testing Poffset
        model_label = 'Poffset'
        seed_positions_dict[model_label] = list()
        confined_ranges_dict[model_label] = list()

        for qrs_index, current_P in zip(xrange(batch_qrs_index, min(len(qrs_locations), batch_qrs_index + batch_size)), batch_P_list):
            seed_position = None
            confined_range = None

            R_pos = qrs_locations[qrs_index]
            R_pos = R_pos * 250.0 / fs
            # Boundaries 
            left_QRS_bound = 0
            right_QRS_bound = len(raw_sig)
            if qrs_index > 0:
                left_QRS_bound = qrs_locations[qrs_index - 1]
            if qrs_index + 1 < len(qrs_locations):
                right_QRS_bound = qrs_locations[qrs_index + 1]

            walker_model, bias = model_dict[model_label]
            bias = int(float(fs) * bias)

            confined_range = [current_P, R_pos]
            confined_ranges_dict[model_label].append(confined_range)
            seed_position = bias + R_pos
            seed_positions_dict[model_label].append(seed_position)
            # current_Ponset = RunWalkerModel(walker_model, R_pos + bias, confined_range)
        # Start testing
        walker_model, bias = model_dict[model_label]
        batch_Poffset_list = RunWalkerModel(walker_model, seed_positions_dict[model_label], confined_ranges_dict[model_label])

        # # T wave
        model_label = 'T'
        seed_positions_dict[model_label] = list()
        confined_ranges_dict[model_label] = list()

        for qrs_index in xrange(batch_qrs_index, min(len(qrs_locations), batch_qrs_index + batch_size)):
            seed_position = None
            confined_range = None

            R_pos = qrs_locations[qrs_index]
            R_pos = R_pos * 250.0 / fs
            # Boundaries 
            left_QRS_bound = 0
            right_QRS_bound = len(raw_sig)
            if qrs_index > 0:
                left_QRS_bound = qrs_locations[qrs_index - 1]
            if qrs_index + 1 < len(qrs_locations):
                right_QRS_bound = qrs_locations[qrs_index + 1]

            walker_model, bias = model_dict[model_label]
            bias = int(float(fs) * bias)
            confined_range = [R_pos, right_QRS_bound]
            confined_ranges_dict[model_label].append(confined_range)
            seed_position = bias + R_pos
            seed_positions_dict[model_label].append(seed_position)
        # Start testing
        walker_model, bias = model_dict[model_label]
        batch_T_list = RunWalkerModel(walker_model, seed_positions_dict[model_label], confined_ranges_dict[model_label])


        # Testing Toffset
        model_label = 'Toffset'
        seed_positions_dict[model_label] = list()
        confined_ranges_dict[model_label] = list()

        for qrs_index, current_T in zip(xrange(batch_qrs_index, min(len(qrs_locations), batch_qrs_index + batch_size)), batch_T_list):
            seed_position = None
            confined_range = None

            R_pos = qrs_locations[qrs_index]
            R_pos = R_pos * 250.0 / fs
            # Boundaries 
            left_QRS_bound = 0
            right_QRS_bound = len(raw_sig)
            if qrs_index > 0:
                left_QRS_bound = qrs_locations[qrs_index - 1]
            if qrs_index + 1 < len(qrs_locations):
                right_QRS_bound = qrs_locations[qrs_index + 1]
            walker_model, bias = model_dict[model_label]
            bias = int(float(fs) * bias)

            confined_range = [current_T, right_QRS_bound]
            confined_ranges_dict[model_label].append(confined_range)
            seed_position = bias + R_pos
            seed_positions_dict[model_label].append(seed_position)
        # Start testing
        walker_model, bias = model_dict[model_label]
        batch_Toffset_list = RunWalkerModel(walker_model, seed_positions_dict[model_label], confined_ranges_dict[model_label])
         
    return testing_results



def GetModels(model_folder, pattern_file_name):
    '''Returns model dict.'''
    # label_list = ['P', 'Ponset', 'Poffset',
            # 'T', 'Toffset', 'Tonset']
    label_list = ['P', 'Ponset', 'Poffset',
            'T', 'Toffset', 
            'Ronset', 'Roffset']
    # label_list = ['P', 'Ponset', 'Poffset']
    # label_list = ['P', ]
    bias_list = [
                -0.19, -0.195, -0.185,
                0.26, 0.27,
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
    from QTdata.loadQTdata import QTloader
    qt = QTloader()
    sig = qt.load(record_name)
    raw_sig = sig['sig']
    raw_sig = raw_sig[0:250 * 120]

    model_folder = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/Lw3Np4000'
    pattern_file_name = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/Lw3Np4000/random_pattern.json'
    model_list = GetModels(model_folder, pattern_file_name)
    start_time = time.time()
    # results = Testing_random_walk(raw_sig, 250.0, r_list, model_list)
    results = Testing(raw_sig, 250.0, model_list, walker_iterations = 100)
    print 'Testing time cost %f secs.' % (time.time() - start_time)

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

def TestChanggeng(record_ind):
    '''Test case1.'''
    def RunWalkerModel(walker_model, seed_positions, confined_ranges, feature_extractor):
        '''Run random walk detection model.
        Input:
            walker_model: random walk regressor for a certain label.
            seed_positions: list of seed position
            confined_ranges: list of confined_range
        '''
        if abs(fs - 250.0) > 1e-6:
            raise Exception('Bias has default fs = 250.0Hz!')

        print 'fs = ', fs

        # First add to prepare testing list
        for seed_position, confined_range in zip(seed_positions, confined_ranges):
            walker_model.prepareTestSample(seed_position, confined_range)

        start_time = time.time()

        # Second, Testing all prepared positions
        path_list, scores_list = walker_model.runPreparedTesting(feature_extractor, iterations = 200, stepsize = 4)

        results = list()
        for path in path_list:
            # Tnew_list.append(len(set(path)))
            predict_position = int(np.mean(path[len(path) / 2:]) / 250.0 * fs)

            # For return value of super function
            results.append((predict_position,
                    walker_model.target_label))
        return (results, path_list)

    import matplotlib.pyplot as plt
    import random
    fs = 250.0
    from QTdata.loadQTdata import QTloader
    from changgengLoader import ECGLoader

    ecg = ECGLoader(500, current_folderpath)
    record_name = ecg.P_faillist[record_ind]
    sig = ecg.load(record_name)
    raw_sig = sig[0]
    import scipy.signal
    # raw_sig = Denoise(raw_sig)
    resampled_sig = scipy.signal.resample_poly(raw_sig, 1, 2)
    # plt.figure(1)
    # plt.plot(raw_sig, label = 'signal')
    # plt.plot(xrange(0, len(raw_sig), 2), resampled_sig, label = 'resmaple')
    # plt.legend()
    # plt.grid(True)
    # plt.title(record_name)
    # plt.show()

    raw_sig = resampled_sig

    model_folder = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/Lw3Np4000/improved'
    pattern_file_name = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/Lw3Np4000/random_pattern.json'
    model_list = GetModels(model_folder, pattern_file_name)
    start_time = time.time()

    # Start Testing
    results = list()
    # results = Testing_random_walk(raw_sig, 250.0, r_list, model_list)
    # results = Testing(raw_sig, 250.0, model_list, walker_iterations = 200)
    feature_extractor = model_list[0][0].GetFeatureExtractor(raw_sig)
    for walker_model, bias, model_label in model_list:
        if model_label != 'P':
            continue
        print 'Testing model label:', model_label
        seeds = list()
        confined_ranges = list()
        for pos in xrange(1, len(raw_sig), 200):
            seeds.append(pos)
            confined_ranges.append([0, len(raw_sig) - 1])
        seed_results, path_list = RunWalkerModel(walker_model, seeds, confined_ranges, feature_extractor)
        results.extend(seed_results)

    print 'Testing time cost %f secs.' % (time.time() - start_time)

    samples_count = len(raw_sig)
    time_span = samples_count / fs
    #print 'Span of testing range: %f samples(%f seconds).' % (samples_count, time_span)

    # Display results
    plt.figure(1)
    plt.clf()
    plt.plot(raw_sig, label = 'ECG')
    pos_list, label_list = zip(*results)
    labels = set(label_list)
    for label in labels:
        if label != 'P':
            continue
        pos_list = [int(x[0]) for x in results if x[1] == label]
        amp_list = [raw_sig[x] for x in pos_list]
        marker = 'o' if len(label) == 1 else '|'
        plt.plot(pos_list, amp_list,
                marker = marker,
                linestyle = 'none',
                markeredgewidth = 5,
                markersize = 15,
                alpha = 0.85,
                markerfacecolor = 'none',
                markeredgecolor = (random.random(),
                                    random.random(),
                                    random.random(),
                    ),
                label = label)

        # Plot path
        for path, up_amplitude in zip(path_list, amp_list):
            plt.plot(path, xrange(up_amplitude, up_amplitude - int(len(path) * 0.01) + 1, 0.01),'r', alpha = 0.43)

    # Plot failed test
    fail_results = ecg.loadAnnot(record_name, target_label = 'P')
    pos_list = [int(x[0] / 2) for x in fail_results if x[1] == 'P']
    amp_list = [raw_sig[x] for x in pos_list]
    plt.plot(pos_list, amp_list, 'x',
            markersize = 15,
            markeredgewidth = 5,
            alpha = 0.5,
            label = 'failed')
    

    plt.title(record_name)
    plt.grid(True)
    plt.legend()
    plt.show(block = False)
    pdb.set_trace()

if __name__ == '__main__':
    # Test1()
    for record_ind in xrange(0, 12):
        TestChanggeng(record_ind)

