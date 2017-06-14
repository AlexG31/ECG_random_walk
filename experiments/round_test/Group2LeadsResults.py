#encoding:utf-8
"""
ECG Grouping Evaluation module
Author : Gaopengfei
"""
import os
import sys
import json
import glob
import bisect
import math
import pickle
import random
import time
import pdb
import pywt

import numpy as np
## machine learning methods
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# 
# project homepath
# 
curfilepath =  os.path.realpath(__file__)
curfolderpath = os.path.dirname(curfilepath)
projhomepath = os.path.dirname(curfolderpath)

# configure file
# conf is a dict containing keys
#
# my project components

from QTdata.loadQTdata import QTloader 

class GroupResult2Leads:
    ''' Simply Group nearby same labels and mean group position.
    '''
    def __init__(self,recname,reslist,reslist2,MaxSWTLevel = 9):
        self.recres = reslist
        self.LeadRes = (reslist,reslist2)

        self.recname = recname
        self.QTdb = QTloader()
        self.sig_struct = self.QTdb.load(self.recname)
        self.rawsig = self.sig_struct['sig']
        self.res_groups = None
        self.peak_dict = dict(T=[],P=[],Ponset = [],Poffset = [],Tonset = [],Toffset = [])
        #self.getSWTcoeflist(MaxLevel = MaxSWTLevel)

    def group_result(self,LeadNum,recres = None,white_del_thres = 20,cp_del_thres = 0):
        #
        # 参数说明：1.white_del_thres是删除较小白色组的阈值
        #           2.cp_del_thres是删除较小其他关键点组的阈值
        # Multiple prediction point -> single point output
        ## filter output for evaluation results
        #
        # parameters
        #
        #
        # the number of the group must be greater than:
        #
        # default parameter

        recres = self.LeadRes[LeadNum-1]

        # filtered test result
        frecres = []
        # in var
        prev_label = None
        posGroup = []
        #----------------------
        # [pos,label] in recres
        #----------------------
        for point_result in recres:
            pos, label = point_result[0:2]
            if len(point_result) >= 3:
                confidence = point_result[2]

            if prev_label is not None:
                if label != prev_label:
                    frecres.append((prev_label,posGroup))
                    posGroup = []
                
            prev_label = label
            posGroup.append(pos)
        # add last label group
        if len(posGroup)>0:
            frecres.append((prev_label,posGroup))
        #======================
        # 1.删除比较小的白色组和其他组(different threshold)
        # 2.合并删除后的相邻同色组
        #======================
        filtered_local_res = []
        for label,posGroup in frecres:
            if label == 'white' and len(posGroup) <= white_del_thres:
                continue
            if label != 'white' and len(posGroup) <= cp_del_thres:
                continue
            # can merge backward?
            if len(filtered_local_res)>0 and filtered_local_res[-1][0] == label:
                filtered_local_res[-1][1].extend(posGroup)
            else:
                filtered_local_res.append((label,posGroup))

        frecres = filtered_local_res
        # [(label,[poslist])]
        self.res_groups = frecres
                
        return frecres
    def filter_smaller_nearby_groups(self,res_groups,group_near_dist_thres = 100):
        # [(label,[poslist])]
        # filter close groups:
        # delete groups with smaller number
        frecres = res_groups
        N_groups = len(frecres)
        if N_groups == 0:
            return res_groups
        deleted_reslist = []
        
        deleted_reslist.append(frecres[0])
        for group_ind in xrange(1,N_groups):
            max_before = np.max(deleted_reslist[-1][1])
            min_after = np.min(frecres[group_ind][1])
            if min_after-max_before <=group_near_dist_thres:
                # keep the larger group
                if len(frecres[group_ind][1]) > len(deleted_reslist[-1][1]):
                    # del delete
                    del deleted_reslist[-1]
                    deleted_reslist.append(frecres[group_ind])
            else:
                deleted_reslist.append(frecres[group_ind])
        return deleted_reslist


    def getResultDict(self,debug = False):
        self.resDict = []
        for leadnum in xrange(1,3):
            self.resDict.append(dict())
            for label in ['P','R','T','Ponset','Poffset','Toffset','Ronset','Roffset']:
                # This function will set self.res_groups.
                self.group_result(leadnum)

                res_groups = filter(lambda x:x[0]==label,self.res_groups)
                res_groups = self.filter_smaller_nearby_groups(res_groups)
                if len(res_groups) == 0:
                    self.resDict[-1][label] = []
                    continue

                if debug == True:
                    print 'length of [{}] groups:'.format(label),len(res_groups)
                    pdb.set_trace()

                reslist = map(lambda x: np.mean(x[1]),res_groups)
                self.resDict[-1][label] = reslist

        return self.resDict


if __name__ == '__main__':
    
    # load the results
    RoundFolder = r'F:\LabGit\ECG_RSWT\TestResult\paper\MultiRound2'
    RoundInd = 1
    ResultFolder = os.path.join(RoundFolder,'Round{}'.format(RoundInd))

    # each result file
    resfiles = glob.glob(os.path.join(ResultFolder,'result_*'))
    for resfilepath in resfiles:
        with open(resfilepath,'r') as fin:
            prdRes = json.load(fin)
        recname = prdRes[0][0]
        reslist1= prdRes[0][1]
        reslist2= prdRes[1][1]

        # ------------------------------------------------------
        # Group Results
        eva = GroupResult2Leads(recname,reslist1,reslist2)
        resDict = eva.getResultDict(debug = False)
        
        # ------------------------------------------------------
        # save to Group Result
        GroupDict = dict(recname = recname,LeadResult=resDict)
        with open(os.path.join(curfolderpath,'MultiLead2','GroupResult',recname+'.json'),'w') as fout:
            json.dump(GroupDict,fout,indent = 4,sort_keys = True)

        # debug
        print 'record name:',recname
