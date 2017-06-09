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


from dpi.DPI_QRS_Detector import DPI_QRS_Detector as DPI
from randomwalk.random_walker import RandomWalker


def testing(changgengID = '8790', 
        leadname = 'II',
        model_folder = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/annots0605/db2/', 
        test_method = 'normal'):

    
    cloader = ECGLoader(1, 1)
    sig = cloader.loadID(changgengID, leadname = leadname)

    fs = 500.0
    sig = scipy.signal.resample(sig, int(len(sig) / float(fs) * 250.0))
    fs_inner = 250.0

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
            # plt.savefig('/home/alex/图片/work_pngs/RandomWalkPath/confidence-%s.png' % seed_position)

        return (output_position, label)

    from hiking_test import hiking
    def hiking_test(seed_position, label = 'T'):
        path = walker.testing_walk_extractor(feature_extractor, seed_position,
                iterations = 1000)
        pos = hiking(path)
        return (pos, label)

    def avg_test(seed_position, label = 'T'):
        path = walker.testing_walk_extractor(feature_extractor, seed_position,
                iterations = 1000)
        pos_list = [x[0] for x in path]
        pos = np.mean(pos_list[len(pos_list) / 2:])
        return (pos, label)
        

    dpi = DPI(debug_info = dict())
    r_list = dpi.QRS_Detection(sig, fs = fs_inner)
    # annots = list()
    annots = zip(r_list, ['R', ] * len(r_list))

    # for seed in xrange(100, len(sig), 200):
    for rpos in r_list:
        seed = rpos + fs_inner * 0.26
        if seed >= len(sig):
            continue
        # print 'testing seed:', seed
        if test_method == 'hiking':
            annots.append(hiking_test(seed))
        elif test_method == 'normal':
            annots.append(avg_test(seed))
        elif test_method == 'test_seed':
            annots.append(test_seed(seed))
        

    # plot_result(sig, annots)
    return annots

def plotExpertLabels(ax, raw_sig, annots, marker_in = None, color_in = None, label_prefix = None):

    # Get label Dict
    labelSet = set()
    labelDict = dict()
    for pos,label in annots:
        if label in labelSet:
            labelDict[label].append(pos)
        else:
            labelSet.add(label)
            labelDict[label] = [pos,]

    # Plot to axes
    for label, posList in labelDict.iteritems():
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

        if marker is not None:
            marker = marker

        if marker_in is not None:
            marker = marker_in
        if color_in is not None:
            color = color_in
        if label_prefix is not None:
            label = label_prefix + label
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
    
def debug_plot_result():
    import glob,json
    files = glob.glob('/home/alex/LabGit/ECG_random_walk/experiments/record_test/hiking/avg/*.json')
    avg_result_folder = '/home/alex/LabGit/ECG_random_walk/experiments/record_test/hiking/'
    cloader = ECGLoader(2, 1)
    for result_file_name in files:
        cID = os.path.split(result_file_name)[-1]
        cID = cID.split('.')[0]
        print 'Ploting cID:', cID
        with open(result_file_name, 'r') as fin:
            hiking_annots = json.load(fin)
        with open(avg_result_folder + '%s.json' % cID, 'r') as fin:
            avg_annots = json.load(fin)

        
        raw_sig = cloader.loadID(cID)
        # plot_result(raw_sig, annots)
        fig, ax = plt.subplots(1,1, figsize=(25, 6))
        plt.plot(raw_sig)
        plt.grid(True)
        hiking_annots = filter(lambda x:x[0] >= 0 and x[0] < len(raw_sig), hiking_annots)
        plotExpertLabels(ax, raw_sig, hiking_annots, color_in = 'r', label_prefix = 'Normal ')
        plotExpertLabels(ax, raw_sig, avg_annots, label_prefix = 'Hiking ')
        plt.title(u'长庚 ' + cID)
        plt.xlim((750, len(raw_sig) - 750))
        
        # plt.savefig('/home/alex/图片/work_pngs/RandomWalkPath/hiking/hiking_%s.png' % cID)
        plt.show(block=False)
        pdb.set_trace()

def debug_plot_result_normal():
    import glob,json
    files = glob.glob('/home/alex/LabGit/ECG_random_walk/experiments/record_test/hiking/QRS_bias/hiking/*.json')
    avg_result_folder = '/home/alex/LabGit/ECG_random_walk/experiments/record_test/hiking/QRS_bias/avg/'
    cloader = ECGLoader(2, 1)
    # fig, ax = plt.subplots(1,1)
    fig, ax = plt.subplots(1,1, figsize=(25, 6))
    for result_file_name in files:
        cID = os.path.split(result_file_name)[-1]
        cID = cID.split('.')[0]
        print 'Ploting cID:', cID
        with open(result_file_name, 'r') as fin:
            hiking_annots = json.load(fin)
        with open(avg_result_folder + '%s.json' % cID, 'r') as fin:
            avg_annots = json.load(fin)

        
        raw_sig = cloader.loadID(cID)
        fs = 500.0
        raw_sig = scipy.signal.resample(raw_sig, int(len(raw_sig) / float(fs) * 250.0))
        # plot_result(raw_sig, annots)
        # fig, ax = plt.subplots(1,1, figsize=(25, 6))
        ax.cla()
        plt.plot(raw_sig)
        plt.grid(True)
        hiking_annots = filter(lambda x:x[0] >= 0 and x[0] < len(raw_sig), hiking_annots)
        avg_annots = filter(lambda x:x[0] >= 0 and x[0] < len(raw_sig), avg_annots)
        plotExpertLabels(ax, raw_sig, hiking_annots, color_in = 'm', label_prefix = 'Hiking')
        plotExpertLabels(ax, raw_sig, avg_annots, color_in = 'r', label_prefix = 'Normal ')
        plt.title(u'长庚 ' + cID)
        plt.xlim((750, len(raw_sig) - 750))
        
        # plt.savefig('/home/alex/图片/work_pngs/RandomWalkPath/hiking/hiking_%s.png' % cID)
        plt.show(block=False)
        pdb.set_trace()


def TestChanggeng(debug = False):
    import json
    with open('/home/alex/LabGit/ECG_random_walk/tools/annotations/inputs/IDlist.json', 'r') as fin:
    # with open('/home/alex/LabGit/ECG_random_walk/experiments/record_test/hiking/normal/IDlist.json', 'r') as fin:
        IDlist = json.load(fin)
    avg_save_result_folder = '/home/alex/LabGit/ECG_random_walk/experiments/record_test/hiking/QRS_bias/avg/'
    hiking_save_result_folder = '/home/alex/LabGit/ECG_random_walk/experiments/record_test/hiking/QRS_bias/hiking/'
    for cID in IDlist:
        print 'Testing cID:', cID

        if debug:
            annots = testing(changgengID = cID, test_method = 'test_seed')
            continue
            
        annots = testing(changgengID = cID, test_method = 'normal')
        with open(avg_save_result_folder + '%s.json' % cID, 'w') as fout:
            json.dump(annots, fout, indent = 4)

        annots = testing(changgengID = cID, test_method = 'hiking')
        with open(hiking_save_result_folder + '%s.json' % cID, 'w') as fout:
            json.dump(annots, fout, indent = 4)

def debug_plot_result_normal():
    import glob,json
    from mcmc.post_p import post_p_mcmc
    def fix_P_annots(raw_sig, annots):
        P_annots = filter(lambda x: x[1][0] =='P', annots)
        annots = filter(lambda x: x[1][0] !='P', annots)
        P_annots = post_p_mcmc(raw_sig, P_annots, 500.0)
        annots.extend(P_annots)
        return annots

    files = glob.glob('/home/alex/LabGit/ECG_random_walk/experiments/record_test/result0607/*.json')
    avg_result_folder = '/home/alex/LabGit/ECG_random_walk/experiments/record_test/hiking/avg/'
    cloader = ECGLoader(2, 1)
    for result_file_name in files:
        cID = os.path.split(result_file_name)[-1]
        cID = cID.split('.')[0]
        print 'Ploting cID:', cID
        with open(result_file_name, 'r') as fin:
            annots = json.load(fin)
        # with open(avg_result_folder + '%s.json' % cID, 'r') as fin:
            # avg_annots = json.load(fin)

        
        raw_sig = cloader.loadID(cID)

        annots = fix_P_annots(raw_sig, annots)
        plot_result(raw_sig, annots)
        # pdb.set_trace()
def run_post_p(output_folder = '/home/alex/LabGit/ECG_random_walk/experiments/record_test/result0607/post_p/'):
    import glob,json
    from mcmc.post_p import post_p_mcmc
    def fix_P_annots(raw_sig, annots):
        P_annots = filter(lambda x: x[1][0] =='P', annots)
        annots = filter(lambda x: x[1][0] !='P', annots)
        P_annots = post_p_mcmc(raw_sig, P_annots, 500.0)
        annots.extend(P_annots)
        return annots

    files = glob.glob('/home/alex/LabGit/ECG_random_walk/experiments/record_test/result0607/*.json')
    avg_result_folder = '/home/alex/LabGit/ECG_random_walk/experiments/record_test/hiking/avg/'
    cloader = ECGLoader(2, 1)
    for result_file_name in files:
        cID = os.path.split(result_file_name)[-1]
        cID = cID.split('.')[0]
        print 'Ploting cID:', cID
        with open(result_file_name, 'r') as fin:
            annots = json.load(fin)
        # with open(avg_result_folder + '%s.json' % cID, 'r') as fin:
            # avg_annots = json.load(fin)

        
        raw_sig = cloader.loadID(cID)

        annots = fix_P_annots(raw_sig, annots)
        with open(output_folder + '%s.json' % cID, 'w') as fout:
            json.dump(annots, fout)
        # plot_result(raw_sig, annots)
        # pdb.set_trace()

def debug_plot_result_post_p():
    import glob,json
    from mcmc.post_p import post_p_mcmc
    def fix_P_annots(raw_sig, annots):
        P_annots = filter(lambda x: x[1][0] =='P', annots)
        annots = filter(lambda x: x[1][0] !='P', annots)
        P_annots = post_p_mcmc(raw_sig, P_annots, 500.0)
        annots.extend(P_annots)
        return annots

    files = glob.glob('/home/alex/LabGit/ECG_random_walk/experiments/record_test/result0609/*.json')
    postp_result_folder = '/home/alex/LabGit/ECG_random_walk/experiments/record_test/result0609/post_p/'
    cloader = ECGLoader(2, 1)
    fig, ax = plt.subplots(1,1, figsize=(25, 6))
    for result_file_name in files:
        cID = os.path.split(result_file_name)[-1]
        cID = cID.split('.')[0]
        print 'Ploting cID:', cID
        with open(result_file_name, 'r') as fin:
            annots = json.load(fin)

        # if os.path.exists(postp_result_folder + '%s.json' % cID) == False:
            # continue
        # with open(postp_result_folder + '%s.json' % cID, 'r') as fin:
            # postp_annots = json.load(fin)

        
        raw_sig = cloader.loadID(cID)

        ax.cla()
        plt.plot(raw_sig)
        plotExpertLabels(ax, raw_sig, annots, label_prefix = 'Normal')

        # postp_annots = filter(lambda x: x[1][0] == 'P', postp_annots)
        # plotExpertLabels(ax, raw_sig, postp_annots, color_in = 'y', label_prefix = 'post_p ')
        plt.title(u'长庚 ' + cID)
        plt.xlim((750, len(raw_sig) - 750))
        
        plt.grid(True)
        # plt.savefig('/home/alex/图片/work_pngs/RandomWalkPath/hiking/hiking_%s.png' % cID)
        plt.show(block=False)
        pdb.set_trace()

def run_post_p_wt(output_folder = '/home/alex/LabGit/ECG_random_walk/experiments/record_test/result0607/post_p/'):
    import glob,json
    from mcmc.post_p import post_p_wt
    def fix_P_annots(raw_sig, annots):
        P_annots = filter(lambda x: x[1][0] =='P', annots)
        annots = filter(lambda x: x[1][0] !='P', annots)
        P_annots = post_p_wt(raw_sig, P_annots, 500.0)
        annots.extend(P_annots)
        return annots

    files = glob.glob('/home/alex/LabGit/ECG_random_walk/experiments/record_test/result0607/*.json')
    # files = glob.glob('/home/alex/LabGit/ECG_random_walk/experiments/record_test/hiking/normal/avg/*.json')
    avg_result_folder = '/home/alex/LabGit/ECG_random_walk/experiments/record_test/hiking/avg/'
    cloader = ECGLoader(2, 1)
    for result_file_name in files:
        cID = os.path.split(result_file_name)[-1]
        cID = cID.split('.')[0]
        print 'Ploting cID:', cID
        with open(result_file_name, 'r') as fin:
            annots = json.load(fin)
            # print 'Read annots:', annots
        # with open(avg_result_folder + '%s.json' % cID, 'r') as fin:
            # avg_annots = json.load(fin)

        
        raw_sig = cloader.loadID(cID)

        annots = fix_P_annots(raw_sig, annots)
        # with open(output_folder + '%s.json' % cID, 'w') as fout:
            # json.dump(annots, fout)
        # plot_result(raw_sig, annots)
        # pdb.set_trace()
if __name__ == '__main__':
    # testing(test_method = 'test_seed')
    # testing(changgengID = '56332', leadname = 'III')
    # TestChanggeng(debug = True)
    # debug_plot_result()
    # debug_plot_result_normal()
    # debug_plot_result_normal()
    # run_post_p()
    run_post_p_wt()
    # debug_plot_result_post_p()
