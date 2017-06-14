#encoding:utf-8
"""
ECG Evaluation Module
Author : Gaopengfei
"""
import os
import sys
import json
import glob
import math
import pickle
import random
import bisect
import time
import importlib
import pdb

import numpy as np
## machine learning methods
# from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# project homepath
# 
curfilepath =  os.path.realpath(__file__)
curfolderpath = os.path.dirname(curfilepath)
projhomepath = os.path.dirname(curfolderpath)
# configure file
# conf is a dict containing keys
# with open(os.path.join(projhomepath,'ECGconf.json'),'r') as fin:
    # conf = json.load(fin)
# sys.path.append(projhomepath)
#
# my project components
from QTdata.loadQTdata import QTloader

#from RFclassifier.evaluation import ECGstatistics
#from ECGPostProcessing.PeakDetection import ECG_PeakDetection as PeakDetectionFilter


possibleLabels = [
            'P',
            'R',
            'T',
            'Ponset',
            'Poffset',
            'Ronset',
            'Roffset',
            'Toffset',
            ]

class EvaluationMultiLeads:
    '''Evaluation of raw detection result by random forest.'''
    def __init__(self, result_converter = None):
        self.QTdb = QTloader()
        self.labellist = []
        self.expertlist = []
        self.recname = None
        self.prdMatchList = []
        self.expMatchList = []
        # Converter that formatting given result format.
        self.result_converter_ = result_converter
        # color schemes
        tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
        self.colors = []
        for color_tri in tableau20:
            self.colors.append((color_tri[0]/255.0,color_tri[1]/255.0,color_tri[2]/255.0))

        # error List
        # [expPos - prdPos,...]
        self.errList = None
    def clear(self):
        # error List
        # [expPos - prdPos,...]
        self.errList = None
        self.labellist = []
        self.expertlist = []
        self.recname = None
        self.prdMatchList = []
        self.expMatchList = []

    def loadlabellist(self,filename, TargetLabel, supress_warning = False):
        '''Load label list with Target Label from json file.
        
            Result should have format:
                {
                    'recname': [],
                    'LeadResult': [
                        {
                            'P':[],
                            'T':[],
                        },
                        {
                            'P':[],
                            'T':[],
                        },
                    ],
                }
        '''
        
        with open(filename,'r') as fin:
            Res = json.load(fin)
            # Convert result format
            Res = self.result_converter_(Res)
        #reclist = Res.keys()

        #if len(reclist) ==0:
            #raise Exception('No result data in file:'+filename+'!')
        #elif len(reclist)>1:
            #print 'Rec List:',reclist
            #print '[Warning] Multiple result data, only the 1st one is used!'

        self.recname = Res['recname']
        if supress_warning == False:
            print '>>loading recname:',self.recname

        self.leadResults= Res['LeadResult']

        # Special case for empty dict
        if TargetLabel not in self.leadResults[0]:
            self.leadPosList = ([], [])
        else:
            self.leadPosList = (self.leadResults[0][TargetLabel],
                    self.leadResults[1][TargetLabel])

        # convert values to int
        #self.labellist = map(lambda x:int(x),self.labellist)

        expertlist = self.QTdb.getexpertlabeltuple(self.recname)
        self.expertlist = map(lambda x:x[0],filter(lambda x:x[1]==TargetLabel,expertlist))

    def getMatchList(self,prdposlist,MaxMatchDist = 50):
        '''Return the match result between prdposlist and expertlist.'''
        # Max Match Dist

        if len(prdposlist) == 0:
            return ([],[])
        prdposlist.sort()
        self.expertlist.sort()

        prdMatchList = [-1,]*len(prdposlist)
        expMatchList = [-1,]*len(self.expertlist)

        N_prdList = len(prdposlist)
        for expInd,exppos in enumerate(self.expertlist):
            insertPos = bisect.bisect_left(prdposlist,exppos)
            left_ind = insertPos-1
            right_ind = insertPos
            matchInd = -1
            matchDist = -1
            if left_ind<0:
                matchInd = right_ind
                matchDist = abs(exppos-prdposlist[right_ind])
            elif right_ind>=N_prdList:
                matchInd = left_ind
                matchDist = abs(exppos-prdposlist[left_ind])
            else:
                leftDist = abs(exppos-prdposlist[left_ind])
                rightDist = abs(exppos-prdposlist[right_ind])
                if leftDist>rightDist:
                    matchInd = right_ind
                    matchDist = rightDist
                else:
                    matchInd = left_ind
                    matchDist = leftDist
            if matchInd == -1 or matchDist>=MaxMatchDist:
                expMatchList[expInd] = -1
            else:
                expMatchList[expInd] = matchInd
                prdMatchList[matchInd] = expInd
        return (expMatchList,prdMatchList)

    def evaluate(self,TargetLabel):
        self.prdMatchList = []
        self.expMatchList = []

        # get 2 Leads match lists
        expMatchList,prdMatchList = self.getMatchList(self.leadPosList[0])
        self.prdMatchList.append(prdMatchList)
        self.expMatchList.append(expMatchList)
        expMatchList,prdMatchList = self.getMatchList(self.leadPosList[1])
        self.prdMatchList.append(prdMatchList)
        self.expMatchList.append(expMatchList)

        # get error statistics
        self.get_errList()

    def getFNlist(self):
        '''Return total number of False Negtives.'''
        return self.FNcnt

    def getFPlist(self):
        return min(self.FPcnt1, self.FPcnt2)

    def plot_evaluation_result(self):
        sigStruct = self.QTdb.load(self.recname)
        rawSig = sigStruct['sig']

        plt.figure(1)
        plt.subplot(211)
        plt.plot(rawSig)

        # plot expert labels
        expPosList = self.expertlist
        plt.plot(expPosList,map(lambda x:rawSig[x],expPosList),'d',color = self.colors[2],markersize = 12,label = 'Expert Label')
        # unmatched expert labels
        #FNPosList = map(lambda x:x[0],filter(lambda x:x[1]!=-1 or x[2]!=-1,zip(expPosList,self.expMatchList[0],self.expMatchList[1])))
        #plt.plot(FNPosList,map(lambda x:rawSig[x],FNPosList),'kd',markersize = 16,label = 'False Negtive')
        
        # T predict list
        prdPosList = self.leadPosList[0]
        plt.plot(prdPosList,map(lambda x:rawSig[int(x)],prdPosList),'*',color = self.colors[3],label = 'prd1',markersize = 12)

        prdPosList = self.leadPosList[1]
        plt.plot(prdPosList,map(lambda x:rawSig[int(x)],prdPosList),'*',color = self.colors[4],label = 'prd2',markersize = 12)

        # not matched list
        #prdPosList = map(lambda x:x[0],filter(lambda x:x[1]==-1,zip(self.labellist,self.prdMatchList)))
        #plt.plot(prdPosList,map(lambda x:rawSig[x],prdPosList),'k*',markersize = 14,label = 'False Positive')

        # plot match line
        #for expPos,matchInd in zip(self.expertlist,self.expMatchList):
            #if matchInd == -1:
                #continue
            #prdPos = self.labellist[matchInd]
            #plt.plot([expPos,prdPos],[rawSig[expPos],rawSig[prdPos]],lw = 14,color = self.colors[4],alpha = 0.3)

        plt.grid(True)
        plt.legend()
        plt.title(self.recname)
        plt.show()

    def getContinousRangeList(self,recname):
        FileFolder = os.path.join('/home/alex/LabGit/ProjectSwiper','QTdata','ContinousExpertMarkRangeList','{}_continousRange.json'.format(recname))
        with open(FileFolder,'r') as fin:
            range_list = json.load(fin)
        return range_list

    def get_errList(self):
        '''Choose the minimum error between the two leads, output is in ms.'''
        self.errList = []
        FNcnt = 0
        # Plot match line
        for expPos,matchInd1,matchInd2 in zip(
                self.expertlist,self.expMatchList[0],self.expMatchList[1]):
            if matchInd1 == -1 and matchInd2 == -1:
                FNcnt += 1
                continue
            elif matchInd1 == -1:
                prdpos2 = self.leadPosList[1][matchInd2]
                err2 = expPos - prdpos2
                self.errList.append(4.0*err2)
            elif matchInd2 == -1:
                prdpos1 = self.leadPosList[0][matchInd1]
                err1 = expPos - prdpos1
                self.errList.append(4.0*err1)
            else:
                prdpos1 = self.leadPosList[0][matchInd1]
                prdpos2 = self.leadPosList[1][matchInd2]
                err1 = expPos - prdpos1
                err2 = expPos - prdpos2

                # chooose the one with smaller error 
                if abs(err1)<abs(err2):
                    self.errList.append(4.0*err1)
                else:
                    self.errList.append(4.0*err2)

        # Total number of False Negtives
        self.FNcnt = FNcnt
        # Exclude FP that not in the Continous Range
        range_list = self.getContinousRangeList(self.recname)
        range_set = set()
        for current_range in range_list:
            range_set |= set(range(current_range[0],current_range[1]))

        self.FPcnt1 = 0
        for prdpos,match_index in zip(sorted(self.leadPosList[0]),self.prdMatchList[0]):
            if match_index == -1 and prdpos in range_set:
                self.FPcnt1 += 1
                print 'FP position index:', prdpos
        self.FPcnt2 = 0
        for prdpos, match_index in zip(sorted(self.leadPosList[1]), self.prdMatchList[1]):
            if match_index == -1 and prdpos in range_set:
                self.FPcnt2 += 1

        return self.errList

    def get_total_mean(self):
        meanVal = np.mean(self.errList)
        return meanVal
    def get_total_stdvar(self):
        stdvar = np.std(self.errList)
        return stdvar


if __name__ == '__main__':
    # Result converter
    ListResultConverterModule = importlib.import_module(
            'EvaluationSchemes.result-converters.list-result-converter')
    ListResultConverter = ListResultConverterModule.ListResultConverter

    resultfilelist = glob.glob(os.path.join(projhomepath, 'result',
        'run-6', 'round1', 'result_*'))
    result_converter = ListResultConverter.convert
    evalinfopath = os.path.join(projhomepath, 'EvaluationSchemes','results','run-6')
    

    # debug:print result files
    for ind, fp in enumerate(resultfilelist):
        print '[{}]'.format(ind),'fp:',fp

    ErrDict = dict()
    ErrData = dict()

    for label in ['P','R','T','Ponset','Poffset','Toffset','Ronset','Roffset']:
        ErrData[label] = dict()
        ErrDict[label] = dict()
        errList = []
        FNcnt = 0

        for file_ind in xrange(0,len(resultfilelist)):
            # progress info
            print 'label:',label
            print 'file_ind',file_ind

            eva= EvaluationMultiLeads(result_converter = result_converter)
            eva.loadlabellist(resultfilelist[file_ind],label)
            eva.evaluate(label)

            # total error
            errList.extend(eva.errList)
            FN = eva.getFNlist()
            FNcnt += FN

            # -----------------
            # error statistics

            #ErrDict[label]['mean'] = np.mean(eva.errList)
            #ErrDict[label]['std'] = np.std(eva.errList)
            #ErrDict[label]['FN'] = eva.getFNlist()

            # debug
            #print '--'
            #print 'record: {}'.format(os.path.split(resultfilelist[file_ind])[-1])
            #print 'Error Dict:','label:',label
            #print ErrDict[label]

            # ======
            #eva.plot_evaluation_result()
            #pdb.set_trace()
        ErrData[label]['errList'] = errList
        ErrData[label]['FN'] = FNcnt

        ErrDict[label]['mean'] = np.mean(errList)
        ErrDict[label]['std'] = np.std(errList)
        ErrDict[label]['FN'] = FNcnt

    print '-'*10
    print 'ErrDict'
    print ErrDict

    # write to json
    with open(os.path.join(evalinfopath,'ErrData.json'), 'w') as fout:
        json.dump(ErrData,fout,indent = 4,sort_keys = True)
        print '>>Dumped to json file: ''ErrData.json''.'
    # error statistics
    with open(os.path.join(evalinfopath,'ErrStat.json'), 'w') as fout:
        json.dump(ErrDict,fout,indent = 4,sort_keys = True)
        print '>>Dumped to json file: ''ErrStat.json''.'

