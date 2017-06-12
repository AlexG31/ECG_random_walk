#encoding:utf8
import os
import sys
import time
import pdb
import bisect
import joblib
import scipy, scipy.signal
import numpy as np
import numpy.random as random
import random as pyrandom
import matplotlib.pyplot as plt
# from feature_extractor.feature_extractor import ECGfeatures
from sklearn.ensemble import RandomForestRegressor
from randomwalk.changgengLoader import ECGLoader


from randomwalk.random_walker import RandomWalker


def testing(changgengID = '8790', 
        leadname = 'II',
        model_folder = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/annots0605/db2/'):

    
    cloader = ECGLoader(1, 1)
    sig = cloader.loadID(changgengID, leadname = leadname)

    fs = 500.0
    # sig = scipy.signal.resample(sig, int(len(sig) / float(fs) * 250.0))

    walker = RandomWalker(target_label = 'T', random_pattern_file_name = model_folder + 'random_pattern.json')
    walker.load_model(model_folder + 'T.mdl')
    feature_extractor = walker.GetFeatureExtractor(sig)
    
    def test_seed(seed_position, label = 'T', plot_path = True):
        path = walker.testing_walk_extractor(feature_extractor, seed_position,
                iterations = 1000)
        path_len = len(path)
        pos_list = [x[0] for x in path]
        output_position = np.mean(pos_list[path_len / 2:])

        if plot_path:
            plt.figure(1)
            plt.clf()
            plt.subplot(2,1,1)
            plt.plot(sig)
            amp_list = np.linspace(0, -2, len(pos_list))
            plt.plot(pos_list, amp_list, 'r', alpha = 0.5, lw = 2)
            plt.xlim((min(pos_list) - 200, max(pos_list) + 200))
            plt.plot(output_position, sig[output_position], 'ro', markersize = 12)
            plt.plot(hiking(path), sig[output_position], 'sm', markersize = 13, alpha = 0.5)


            # Path and confidence
            # plt.figure(2)
            # plt.clf()
            plt.subplot(2,1,2)
            plt.plot(pos_list, label = 'Path')
            m1 = min(pos_list)
            m2 = max(pos_list)
            confidence_list = [x[1] * (m2 - m1) + m1 for x in path]
            plt.plot(confidence_list, label = 'Confidence', alpha = 0.9, )


            plt.figure(2)
            plt.clf()
            plt.hist(pos_list, bins = 30)

            plt.show(block = False)


            pdb.set_trace()

        return (output_position, label)
        
    annots = list()
    for seed in xrange(100, len(sig), 250):
        print 'testing seed:', seed
        annots.append(test_seed(seed))
        

    import testing_new_annots as test_tools
    test_tools.plot_result(sig, annots)

def hiking(walk_result):
    '''Hiking algorithm.
    Input:
        [(pos, confidence), ...]
    '''
    # print 'Walk result:', walk_result
    # pdb.set_trace()
    pos_list, confidence_list = zip(*walk_result)

    # 1. Top confidence
    Nc = 10
    sorted_result = sorted(walk_result, key = lambda x:x[1])
    if Nc > len(walk_result):
        raise Exception('Voter number Nc larger than total number of results!')
    voters = sorted_result[0:Nc]

    # 2. Histogram algorithm
    # number of bins must > 1
    bins = 30
    m0 = min(pos_list)
    m1 = max(pos_list)
    bin_width = float(m1 - m0) / bins
    

    if abs(bin_width) < 1e-6:
        return m1

    histogram = [0, ] * bins
    for val in pos_list:
        bin_index = int((val - m0) / bin_width)
        if bin_index >= bins:
            bin_index = bins - 1
        histogram[bin_index] += 1

    # Get peaks
    peak_positions = list()
    for ind in xrange(0, len(histogram)):
        if ind == 0:
            if histogram[ind] > histogram[ind + 1]:
                # peak_positions.append(m0 + (ind + 0.5) * bin_width)
                peak_positions.append(m0 + (ind) * bin_width)
        elif ind == len(histogram) - 1:
            if histogram[ind] > histogram[ind - 1]:
                # peak_positions.append(m0 + (ind + 0.5) * bin_width)
                peak_positions.append(m0 + (ind) * bin_width)
        else:
            if (histogram[ind] >= histogram[ind - 1] and
                histogram[ind] >= histogram[ind + 1]):
                # peak_positions.append(m0 + (ind + 0.5) * bin_width)
                peak_positions.append(m0 + (ind) * bin_width)
            
    # 3. Voting
    # If there are multiple winners, the average is used
    candidate_list = list()
    peak_positions.sort()
    voting_scores = [0, ] * len(peak_positions)
    max_score = 0

    import bisect
    for ind in xrange(0, len(voters)):
        pos, confidence = voters[ind]
        # Nearest peak position
        insert_index = bisect.bisect_left(peak_positions, pos)
        distance = None
        candidate_index = None
        if insert_index < len(peak_positions):
            distance = abs(peak_positions[insert_index] - pos)
            candidate_index = insert_index
        if insert_index - 1 > 0:
            if distance is None:
                distance = abs(peak_positions[insert_index - 1] - pos)
                candidate_index = insert_index - 1
            else:
                if distance > abs(peak_positions[insert_index - 1] - pos):
                    distance = abs(peak_positions[insert_index - 1] - pos)
                    candidate_index = insert_index - 1
        voting_scores[candidate_index] += 1
        max_score = max(max_score, voting_scores[candidate_index])
    for peak_pos, score in zip(peak_positions, voting_scores):
        if score == max_score:
            candidate_list.append(peak_pos)

    if len(candidate_list) > 1:
        print 'Warning: Multiple candidate for hiking peaks.'

    output = float(sum(candidate_list)) / len(candidate_list)
    return output

def debug_plot_result():
    import glob,json
    files = glob.glob('/home/alex/LabGit/ECG_random_walk/experiments/record_test/hiking/*.json')
    avg_result_folder = '/home/alex/LabGit/ECG_random_walk/experiments/record_test/hiking/avg/'
    cloader = ECGLoader(2, 1)
    for result_file_name in files:
        cID = os.path.split(result_file_name)[-1]
        cID = cID.split('.')[0]
        print 'Ploting cID:', cID
        with open(result_file_name, 'r') as fin:
            annots = json.load(fin)
        with open(avg_result_folder + '%s.json' % cID, 'r') as fin:
            avg_annots = json.load(fin)

        
        raw_sig = cloader.loadID(cID)
        import testing_new_annots as test_tools
        test_tools.plot_result(raw_sig, annots)
        # pdb.set_trace()


if __name__ == '__main__':
    debug_plot_result()


        


    



if __name__ == '__main__':
    testing()
