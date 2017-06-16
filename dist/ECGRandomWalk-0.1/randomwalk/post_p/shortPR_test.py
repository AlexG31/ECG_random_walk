#encoding:utf8
import os, sys, pdb, json
import matplotlib.pyplot as plt

import gaussian_model as P_model_Gaussian
import numpy as np
import math
from pymc import MCMC
import scipy.signal as signal
from randomwalk.changgengLoader import ECGLoader as cLoader
from post_p import post_p, post_p_mcmc


def shortPR_test():
    '''Find more accurate characteristic point position.'''
    shortPR_folder = '/home/alex/LabGit/ECG_random_walk/experiments/record_test/shortQT/results0516/'
    shortPR_outputfolder = '/home/alex/LabGit/ECG_random_walk/experiments/record_test/shortQT/mcmc_enhanced157_Tonset/'
    import glob
    files = glob.glob(shortPR_folder + '*.json')

    
    loader = cLoader(2,1)
    
    file_ind = 0
    for result_file in files:
        recordID = os.path.split(result_file)[-1]
        recordID = recordID.split('.')[0]
        print 'Processing record: %s' % recordID
        file_ind += 1
        print '%d records left.' % (len(files) - file_ind)

        if os.path.exists(shortPR_outputfolder + '%s.json' % recordID) == True:
            continue

        sig = loader.loadID(recordID)

        with open(result_file, 'r') as fin:
            annots = json.load(fin)
            annots = post_p(sig, annots, 500)
            annots.sort(key = lambda x:x[0])
            annots = post_p_mcmc(sig, annots, 500)

        with open(shortPR_outputfolder + '%s.json' % recordID, 'w') as fout:
            json.dump(annots, fout, indent = 4)
        
if __name__ == '__main__':
    # enhance_test()
    shortPR_test()
