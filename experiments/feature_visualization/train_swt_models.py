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

# Get current file path & project homepath.
curfilepath =  os.path.realpath(__file__)
curfolderpath = os.path.dirname(curfilepath)
projhomepath = os.path.dirname(curfolderpath)
projhomepath = os.path.dirname(projhomepath)
# configure file
# conf is a dict containing keys
with open(os.path.join(projhomepath,'ECGconf.json'),'r') as fin:
    conf = json.load(fin)
sys.path.append(projhomepath)
# Logging config.
logging.basicConfig(
        filename = os.path.join(
            projhomepath,
            'logs',
            '%s.log'%datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')),
        format = ('%(asctime)-15s[%(levelname)s]%(filename)s:%(lineno)d,'
            ' in function:%(funcName)s\n    %(message)s'),
        level = 10
        )
# Logging
log = logging.getLogger()

# my project components
import RFclassifier.extractfeature.extractfeature as extfeature
import RFclassifier.extractfeature.randomrelations as RandomRelation
import QTdata.loadQTdata as QTdb
import RFclassifier.evaluation as ecgEval
from RFclassifier.ClassificationLearner import ECGrf
from RFclassifier.ClassificationLearner import timing_for
from QTdata.loadQTdata import QTloader 


def backup_configure_file(saveresultpath):
    shutil.copy(os.path.join(projhomepath,'ECGconf.json'),saveresultpath)
def backupobj(obj,savefilename):
    with open(savefilename,'wb') as fout:
        pickle.dump(obj,fout)


def TrainSwtModel(save_model_folder,
        save_sample_folder,
        target_label,
        random_relation_file_path):
    '''Test all records in testinglist, training on remaining records in QTdb.'''
    qt_loader = QTloader()
    QTreclist = qt_loader.getQTrecnamelist()

    # Get training record list
    random.shuffle(QTreclist)
    traininglist = QTreclist[0:75]

    random_relation_file_path = os.path.dirname(random_relation_file_path)
    rf_classifier = ECGrf(SaveTrainingSampleFolder = save_sample_folder,
            allowed_label_list = [target_label,],
            random_relation_path = random_relation_file_path)
    # Multi Process
    rf_classifier.TestRange = 'All'

    # Training
    time_cost_output = []
    timing_for(rf_classifier.TrainQtRecords,
            [traininglist,],
            prompt = 'Total Training time:',
            time_cost_output = time_cost_output)
    log.info('Total training time cost: %.2f seconds', time_cost_output[-1])
    # save trained mdl
    backupobj(rf_classifier.mdl, os.path.join(save_model_folder, 'trained_model.mdl'))

def ResetFolder(folder_path):
    # Create result folder if not exist.
    if os.path.exists(folder_path) == True:
        option = raw_input('Result path "{}" already exists, remove it?(y/n)'.format(
            folder_path))
        if option in ['y', 'Y']:
            shutil.rmtree(folder_path)
    else:
        os.mkdir(folder_path)
    
def TrainAndSaveModel(save_folder, random_relation_file_path, target_label = 'T'):
    '''
    Read random relation file from parent folder,
    training a model base on the target label,
    save the trained model to save_folder.
    '''
    saveresultpath = save_folder

    # Backup configuration file
    backup_configure_file(saveresultpath)

    TrainSwtModel(saveresultpath,
            os.path.join(
                os.path.dirname(save_folder),
        'training_samples'),
        target_label,
        random_relation_file_path)

if __name__ == '__main__':
    random_relation_file_path = os.path.join(curfolderpath,
            'trained_model',
            'rand_relations.json')

    RandomRelation.RefreshRswtPairs(random_relation_file_path)

    save_folder = os.path.join(curfolderpath,
            'trained_model',
            'T')
    TrainAndSaveModel(save_folder, random_relation_file_path, target_label = 'T')

    save_folder = os.path.join(curfolderpath,
            'trained_model',
            'P')
    TrainAndSaveModel(save_folder, random_relation_file_path, target_label = 'P')

    save_folder = os.path.join(curfolderpath,
            'trained_model',
            'R')
    TrainAndSaveModel(save_folder, random_relation_file_path, target_label = 'R')


