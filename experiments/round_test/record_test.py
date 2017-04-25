#encoding:utf-8
"""
ECG classification Module
Author : Gaopengfei
"""
import os
import sys
import json
import glob
import datetime
import math
import pickle
import logging
import random
import time
import shutil
import numpy as np
import pdb
## machine learning methods
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from QTdata.loadQTdata import QTloader 

# Get current file path & project homepath.
curfilepath =  os.path.realpath(__file__)
curfolderpath = os.path.dirname(curfilepath)
projhomepath = os.path.dirname(curfolderpath)



def backup_configure_file(saveresultpath):
    shutil.copy(os.path.join(projhomepath,'ECGconf.json'),saveresultpath)
def backupobj(obj,savefilename):
    with open(savefilename,'wb') as fout:
        pickle.dump(obj,fout)

def Round_Test(
        saveresultpath,RoundNumber = 1,
        number_of_test_record_per_round = 30,
        round_start_index = 1):
    '''Randomly select records from QTdb to test.
        Args:
            RoundNumber: Rounds to repeatedly select records form QTdb & test.
            number_of_test_record_per_round: Number of test records to randomly
            select per round.
    '''
    
    qt_loader = QTloader()
    QTreclist = qt_loader.getQTrecnamelist()

    # To randomly select 30 records from may_testlist
    may_testlist = QTreclist
    # Remove records that must be in the training set
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
    may_testlist = list(set(may_testlist) - set(must_train_list))
    N_may_test = len(may_testlist)
    
    # Start testing.
    for round_ind in xrange(round_start_index, RoundNumber+1):
        # Generate round folder.
        round_folder = os.path.join(saveresultpath, 'round{}'.format(round_ind))
        os.mkdir(round_folder)
        # Randomly select test records.
        test_ind_list = random.sample(xrange(0,N_may_test),number_of_test_record_per_round)
        testlist = map(lambda x:may_testlist[x], test_ind_list)
        # Run the test warpper.
        TestAllQTdata(round_folder, testlist)

def TestAllQTdata(saveresultpath, testinglist):
    '''Test all records in testinglist, training on remaining records in QTdb.'''
    qt_loader = QTloader()
    QTreclist = qt_loader.getQTrecnamelist()
    # Get training record list
    traininglist = list(set(QTreclist) - set(testinglist))

    
    # Testing
    from randomwalk.test_api import GetModels
    from randomwalk.test_api import Testing
    # pattern_filename = os.path.join(os.path.dirname(saveresultpath), 'randrel.json')
    pattern_filename = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/Lw3Np4000/random_pattern.json'
    model_folder = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/Lw3Np4000'
    model_list = GetModels(model_folder, pattern_filename)

    for record_name in testinglist:
        sig = qt_loader.load(record_name)
        raw_sig = sig['sig']
        
        start_time = time.time()
        results = Testing(raw_sig, 250.0, model_list, walker_iterations = 100)
        time_cost = time.time() - start_time

        with open(os.path.join(saveresultpath, '%s.json' % record_name), 'w') as fout:
            json.dump(results, fout)
            print 'Testing time %f s, data time %f s.' % (time_cost, len(raw_sig) / 250.0)


def RoundTesting(saveresultpath, testinglist, model_folder = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/Lw3Np4000', pattern_filename = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/Lw3Np4000/random_pattern.json'):
    '''Test all records in testinglist, training on remaining records in QTdb.'''
    qt_loader = QTloader()
    QTreclist = qt_loader.getQTrecnamelist()
    # Get training record list
    traininglist = list(set(QTreclist) - set(testinglist))

    
    # Testing
    from test_api import GetModels
    from test_api import Testing
    # pattern_filename = os.path.join(os.path.dirname(saveresultpath), 'randrel.json')
    
    
    model_list = GetModels(model_folder, pattern_filename)

    for record_name in testinglist:
        print 'testing %s' % record_name
        sig = qt_loader.load(record_name)
        raw_sig = sig['sig']
        
        start_time = time.time()
        results = Testing(raw_sig, 250.0, model_list, walker_iterations = 100)
        time_cost = time.time() - start_time

        with open(os.path.join(saveresultpath, '%s.json' % record_name), 'w') as fout:
            json.dump(results, fout, indent = 4)
            print 'Testing time %f s, data time %f s.' % (time_cost, len(raw_sig) / 250.0)


if __name__ == '__main__':

    # Debug
    number_of_test_record_per_round = 30

    saveresultpath = os.path.join(curfolderpath, 'round1')
    random_relation_file_path = os.path.join(curfolderpath, 'round1')

    # create result folder if not exist
    if os.path.exists(saveresultpath) == True:
        option = raw_input(
                'Result path "{}" already exists, remove it?(y/n)'.format(saveresultpath))
        if option in ['y', 'Y']:
            shutil.rmtree(saveresultpath)
        os.mkdir(saveresultpath)
    else:
        os.mkdir(saveresultpath)
    # Refresh randomly selected features json file and backup it.
    # RandomRelation.RefreshRswtPairs(random_relation_file_path)

    Round_Test(saveresultpath,
            number_of_test_record_per_round = number_of_test_record_per_round,
            RoundNumber = 100,
            round_start_index = 1)
