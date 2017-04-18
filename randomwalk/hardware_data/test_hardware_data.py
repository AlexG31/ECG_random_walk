import os
import sys
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import joblib
import scipy.signal
import pdb
import json
import logging
from contextlib import closing
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import scipy.io as sio
from multiprocessing import Pool
from dpi.DPI_QRS_Detector import DPI_QRS_Detector as DPI
from dpi.QrsTypeDetector import QrsTypeDetector
from QTdata.loadQTdata import QTloader
from random_walker import RandomWalker
from test_api import Testing
from test_api import GetModels
import time
import codecs

def changgeng():
    with codecs.open(
        '/home/lab/Programm/dataset/total_19308.json',mode='r', encoding='utf8') as fin:
        dinfo=json.load(fin)
    #pool=Pool()
    rec_list=dinfo['data']

    try:
        with closing(Pool(11)) as p:
            p.map(keypoint_detect, rec_list, len(rec_list)/11)
        #pool.map(keypoint_detect, rec_list)
    except Exception as e:
        logging.error("error_detail: {}".format(str(e)), exc_info=True)

def ConvertLabel(label):
    '''Convert random forest label to figure label.'''
    if label == 'T':
        mker = 'ro'
    elif label == 'R':
        mker = 'go'
    elif label == 'P':
        mker = 'bo'
    elif label == 'Tonset':
        mker = 'r<'
    elif label == 'Toffset':
        mker = 'r>'
    elif label == 'Ronset':
        mker = 'g<'
    elif label == 'Roffset':
        mker = 'g>'
    elif label == 'Ponset':
        mker = 'b<'
    elif label == 'Poffset':
        mker = 'b>'
    else:# white
        mker = 'w.'
    return mker

def read_hardware_data(filepath):
    file_object = open(filepath)
    sig = []
    try:
      list_lines = file_object.readlines()
      for line in list_lines:
        sig.append(int(line))
    finally:
      file_object.close()
    return sig

def keypoint_detect_hardware_data(filepath):
    sig = read_hardware_data(filepath)
    #sig = sig[0:1000]
    print len(sig)
    plt.figure(1, figsize=(20, 10))
    plt.clf()
    plt.plot(sig, label='signal')
    #plt.show()    
    sampling_frequency = 250
    adapt_frequency = 250
    model_folder = '/home/lab/Programm/pengfei/ECG_random_walk/randomwalk/data/m3_full_models'
    pattern_file_name = '/home/lab/Programm/pengfei/ECG_random_walk/randomwalk/data/m3_full_models/random_pattern.json'
    model_list = GetModels(model_folder, pattern_file_name)
    starttime = time.clock()
    keypoints = Testing(sig,sampling_frequency,model_list)
    endtime = time.clock()
    print 'excute time:%f seconds'%(endtime - starttime)
    target_label_set = set()
    II_label_pos = dict()
    map(lambda x:target_label_set.add(x[1]),keypoints)
    for target_label in target_label_set:
        II_label_pos[target_label]=[x[0] for x in keypoints if x[1]==target_label]
        II_pos_list = II_label_pos[target_label]
        II_amp_list = [sig[int(x)] for x in II_pos_list if x < len(sig)]
        II_pos_list = [x for x in II_pos_list if x < len(sig)]
        plt.plot(II_pos_list, II_amp_list, ConvertLabel(target_label), label=target_label)
        plt.title('hardware')
        plt.legend(loc='best')
    plt.show()

    detect_result_path = filepath+'.json'
    #filename_save_result=os.path.join(detect_result_path,str(ID)+'.json')
    result=dict()
    #result['id']=ID
    #result['diagnose']=diagnose
    result['II_label_pos']=II_label_pos
    #result['V1_label_pos']=V1_label_pos
    #result['V5_label_pos']=V5_label_pos
    with codecs.open(filename_save_result,'w',encoding='utf-8') as fout:
         json.dump(result,fout,ensure_ascii=False,encoding='utf-8')
    #print ID

def keypoint_detect(args):
    raw_data_path, ID, diagnose = args['mat_rhythm'],args['id'], args['diagnose']
    raw_data=sio.loadmat(raw_data_path)
    for key in raw_data.keys():
        raw_data[key] = np.squeeze(raw_data[key])
    sampling_frequency = 500
    adapt_frequency = 250
    model_folder = '/home/lab/Programm/pengfei/ECG_random_walk/randomwalk/data/m3_full_models'
    pattern_file_name = '/home/lab/Programm/pengfei/ECG_random_walk/randomwalk/data/m3_full_models/random_pattern.json'
    model_list = GetModels(model_folder, pattern_file_name)
    raw_sig_II = raw_data['II']

    II_label_pos=dict()
    if(len(raw_sig_II)>0):
      keypoints_II = Testing(raw_sig_II,sampling_frequency,model_list)
      target_label_set = set()
      map(lambda x:target_label_set.add(x[1]),keypoints_II)
      for target_label in target_label_set:
        II_label_pos[target_label]=[x[0] for x in keypoints_II if x[1]==target_label]
        #II_pos_list = II_label_pos[target_label]
        #II_amp_list = [raw_sig_II[int(x)] for x in II_pos_list if x < len(raw_sig_II)]
        #II_pos_list = [x for x in II_pos_list if x < len(raw_sig_II)]
        #plt.plot(II_pos_list, II_amp_list, ConvertLabel(target_label), label=target_label)
        #plt.title('II')
        #plt.legend(loc='best')
        #plt.show()
     
    raw_sig_V1 = raw_data['V1']
    V1_label_pos=dict()
    if(len(raw_sig_V1)>0):    
      keypoints_V1 = Testing(raw_sig_V1,sampling_frequency,model_list)
      target_label_set=set()
      map(lambda x:target_label_set.add(x[1]),keypoints_V1)
      for target_label in target_label_set:
        V1_label_pos[target_label]=[x[0] for x in keypoints_V1 if x[1]==target_label]
        #V1_pos_list = V1_label_pos[target_label]
        #V1_amp_list = [raw_sig_V1[int(x)] for x in V1_pos_list if x < len(raw_sig_V1)]
        #V1_pos_list = [x for x in V1_pos_list if x < len(raw_sig_V1)]
        #plt.plot(V1_pos_list, V1_amp_list, ConvertLabel(target_label), label=target_label)
        #plt.title('V1')
        #plt.legend(loc='best')

    #target_label_set=set()
    raw_sig_V5 = raw_data['V5']
    V5_label_pos=dict()
    if(len(raw_sig_V5)>0):
      target_label_set=set()
      keypoints_V5 = Testing(raw_sig_V5,sampling_frequency,model_list)
      map(lambda x:target_label_set.add(x[1]),keypoints_V5)
      for target_label in target_label_set:
        V5_label_pos[target_label]=[x[0] for x in keypoints_V5 if x[1]==target_label]
        #V5_pos_list = V5_label_pos[target_label]
        #V5_amp_list = [raw_sig_V5[int(x)] for x in V5_pos_list if x < len(raw_sig_V5)]
        #V5_pos_list = [x for x in V5_pos_list if x < len(raw_sig_V5)]
        #plt.plot(V5_pos_list, V5_amp_list, ConvertLabel(target_label), label=target_label)
        #plt.title('V5')
        #plt.legend(loc='best')
    #plt.show()


    #plt.figure(1, figsize=(20, 10))
    #plt.clf()
    #plt.subplot(3, 1, 1)
    #plt.plot(raw_sig_II, label='signal')
    #II_amp_list = list()

    #plt.subplot(3, 1, 2)
    #plt.plot(raw_sig_V1, label='signal')
    #V1_amp_list = list()

    #target_label_dict_V5=dict()
    #
    #plt.subplot(3, 1, 3)
    #plt.plot(raw_sig_V5, label='signal')

    #V5_amp_list = list()


    detect_result_path = '/home/share/generated_data/changgeng_detect/keypoints/'
    filename_save_result=os.path.join(detect_result_path,str(ID)+'.json')
    result=dict()
    result['id']=ID
    result['diagnose']=diagnose
    result['II_label_pos']=II_label_pos
    result['V1_label_pos']=V1_label_pos
    result['V5_label_pos']=V5_label_pos
    with codecs.open(filename_save_result,'w',encoding='utf-8') as fout:
         json.dump(result,fout,ensure_ascii=False,encoding='utf-8')
    print ID

filepath = '/home/lab/Programm/pengfei/ECG_random_walk/randomwalk/hardware_data/04-14-47.dat'
keypoint_detect_hardware_data(filepath)
