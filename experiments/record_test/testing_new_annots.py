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


            # Path and confidence
            # plt.figure(2)
            # plt.clf()
            plt.subplot(2,1,2)
            plt.plot(pos_list, label = 'Path')
            m1 = min(pos_list)
            m2 = max(pos_list)
            confidence_list = [x[1] * (m2 - m1) + m1 for x in path]
            plt.plot(confidence_list, label = 'Confidence', alpha = 0.9, )
            plt.show(block = False)
            pdb.set_trace()
            plt.savefig('/home/alex/图片/work_pngs/RandomWalkPath/confidence-%s.png' % seed_position)

        return (output_position, label)
        
    annots = list()
    for seed in xrange(100, len(sig), 250):
        print 'testing seed:', seed
        annots.append(test_seed(seed))
        

    plot_result(sig, annots)

def plotExpertLabels(ax, raw_sig, annots):

    #get label Dict
    labelSet = set()
    labelDict = dict()
    for pos,label in annots:
        if label in labelSet:
            labelDict[label].append(pos)
        else:
            labelSet.add(label)
            labelDict[label] = [pos,]

    # plot to axes
    for label,posList in labelDict.iteritems():
        # plot marker for current label
        if label[0]=='T':
            color = 'm'
        elif label[0]=='P':
            color  = 'b'
        elif label[0]=='R':
            color  = 'r'
        # marker
        if 'onset' in label:
            marker = '<'
        elif 'offset' in label:
            marker = '>'
        else:
            marker = 'o'
        ax.plot(posList,map(lambda x:raw_sig[int(x)],posList),marker = marker,color = color,linestyle = 'none',markersize = 14,label = label)
    ax.legend(numpoints = 1)


def plot_result(raw_sig, annots):
    fig, ax = plt.subplots(1,1)
    plt.plot(raw_sig)
    plt.grid(True)
    annots = filter(lambda x:x[0] >= 0 and x[0] < len(raw_sig), annots)
    plotExpertLabels(ax, raw_sig, annots)
    plt.show()
    
def show_result_ECG(cID = '8790'):
    loader = ECGLoader(1, 1)
    sig = loader.loadID(cID)

    plot_result(sig, annots)
    


if __name__ == '__main__':
    testing()
    # testing(changgengID = '56332', leadname = 'III')
