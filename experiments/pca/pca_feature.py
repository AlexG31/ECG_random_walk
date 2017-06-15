#encoding:utf8
import os, sys, json
import pdb
from QTdata.loadQTdata import QTloader
import matplotlib.pyplot as plt
from numpy.random import normal
import numpy as np
import math
from sklearn.decomposition import PCA
from randomwalk.feature_extractor.random_patterns import RefreshRswtPairs
from randomwalk.random_walker import RandomWalker
from randomwalk.feature_extractor.full_wt_feature_extractor import ECGfeatures


def gen_pairs():
    '''Generate Random Pattern.'''
    config = dict(
                WT_LEVEL = 9,
                fs = 250.0,
                winlen_ratio_to_fs = 3,
                WTrandselfeaturenumber_apprx = 3000,
                totally_random_pair_number = 6000,
            )

    output_pattern_path = './output.json'
    RefreshRswtPairs(config, output_pattern_path)


def pca_training(target_label):
    '''Do PCA feature re-projection on target label in QTdb.'''
    qt = QTloader()
    record_list = qt.getreclist()

    random_pattern_path = '/home/alex/LabGit/ECG_random_walk/experiments/pca/pattern/random_pattern.json'
    with open(random_pattern_path, 'r') as fin:
        random_pattern = json.load(fin)

    random_walker_fake = RandomWalker(random_pattern_file_name = random_pattern_path)

    matX = list()
    for record_name in record_list:
        print 'PCA training:', record_name

        sig = qt.load(record_name)
        expert_annots = qt.getExpert(record_name)
        pos_list = [x[0] for x in filter(lambda x:x[1] == target_label, expert_annots)]

        # Extract feature by gaussian distribution
        training_indexes = random_walker_fake.gaussian_training_sampling(pos_list)
        raw_sig = sig['sig']
        configuration_info = random_walker_fake.get_configuration()
        feature_extractor = ECGfeatures(raw_sig, configuration_info)

        for pos in training_indexes:
            feature_vector = feature_extractor.frompos(pos)
            matX.append(feature_vector)

    # Fit PCA
    pca = PCA(n_components = 3000)
    pca.fit(matX)

    return pca
        
        
    
if __name__ == '__main__':
    # gen_pairs()
    label_list = ['P', 'Ponset', 'Poffset', 'Ronset', 'Roffset', 'T', 'Toffset']
    for target_label in label_list:
        pca = pca_training(target_label)
        import joblib
        with open('./pca_models/%s.mdl' % target_label, 'wb') as fout:
            joblib.dump(pca, fout)
