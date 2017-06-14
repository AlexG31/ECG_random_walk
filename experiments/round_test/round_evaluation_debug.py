#encoding:utf-8
"""
Date: 2016.10.10
ECG Grouping Evaluation module
Author : Gaopengfei

This file find the bad performance record, print its round number and record name.

1. This file tries to group the results as single outputs around the ground truth position.
2. This file computes the intra-record standard deviation with police object.
3. This file output stats in ms.
4. This file keep the smallest error between two leads.

"""
import os
import sys
import json
import glob
import string
import bisect
import math
import pickle
import shutil
import random
import time
import importlib
import pdb
import pywt

import numpy as np
## machine learning methods
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# project homepath
# 
curfilepath =  os.path.realpath(__file__)
curfolderpath = os.path.dirname(curfilepath)
projhomepath = os.path.dirname(curfolderpath)
sys.path.append(projhomepath)
#
# my project components

from Group2LeadsResults import GroupResult2Leads
from EvaluationMultiLeads import EvaluationMultiLeads
from round_result_converter import RoundResultConverter
# from TextLogger import TextLogger


def GroupingRawResults(RoundInd, round_folder, output_round_folder, file_name_pattern = 'result_*'):
    '''Grouping the raw detection result of random forest.'''
    # load the results
    RoundFolder = round_folder
    raw_result_folder = os.path.join(RoundFolder,'round{}'.format(RoundInd))
    output_round_folder = os.path.join(output_round_folder,'round{}'.format(RoundInd))

    # Remove existing folder
    if os.path.exists(output_round_folder) == True:
        shutil.rmtree(output_round_folder)
    os.mkdir(output_round_folder)

    # each result file
    resfiles = glob.glob(os.path.join(raw_result_folder, file_name_pattern))
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

        # save to Round folder 
        jsonfilepath = os.path.join(output_round_folder,recname+'.json')
        with open(jsonfilepath,'w') as fout:
            json.dump(GroupDict,fout,indent = 4,sort_keys = True)
    print 'Finished grouping for round %d' % RoundInd

class FindBadRecord:
    def __init__(
            self,
            mean_threshold,
            std_threshold,
            FN_threshold,
            FP_threshold,
            possible_label_list = None,
            result_converter = None):
        if possible_label_list is None:
            self.possible_label_list = ['P','T','Ponset','Poffset','Toffset']
        else:
            self.possible_label_list = possible_label_list

        self.round_record_statistics = list()
        self.total_error_diction =  dict()
        self.error_list_for_label = dict()
        self.false_negtive_list = dict()
        self.false_positive_list = dict()
        self.sensitivity = dict()
        self.Pplus = dict()
        # thresholds
        self.mean_threshold = mean_threshold
        self.std_threshold = std_threshold
        self.FN_threshold = FN_threshold
        self.FP_threshold = FP_threshold

        # result converter
        self.result_converter_ = result_converter

        # statistics for each record.
        self.statistics_list_for_label = dict()
        for label in self.possible_label_list:
            self.error_list_for_label[label] = []
            self.false_negtive_list[label] = 0
            self.false_positive_list[label] = 0
            self.sensitivity[label] = 0
            self.Pplus[label] = 0
            self.statistics_list_for_label[label] = dict(mean = [], std = [])

    def RunEval(self, RoundInd, GroupResultFolder, result_file_list):
        '''
        Evaluate results.

        RoundInd: only used by police_check function.
        '''
        # result_file_list = glob.glob(os.path.join(GroupResultFolder,'*.json'))

        ErrDict = dict()
        ErrData = dict()
        record_statistics = dict()

        for label in self.possible_label_list:
            # Target label to calculate stats from.
            ErrData[label] = dict()
            ErrDict[label] = dict()
            errList = []
            FNcnt = 0
            FPcnt = 0

            for file_ind in xrange(0,len(result_file_list)):
                # Process all the *grouped* results.
                result_file_name = result_file_list[file_ind]
                # Save record result
                record_name = os.path.split(result_file_name)[-1]
                if record_name not in record_statistics:
                    record_statistics[record_name] = dict()
                eva= EvaluationMultiLeads(self.result_converter_)
                eva.loadlabellist(result_file_name,label, supress_warning = True)
                eva.evaluate(label)

                # total error
                errList.extend(eva.errList)
                FN = eva.getFNlist()
                FP = eva.getFPlist()

                if FP > 0:
                    print FP
                    print result_file_name
                    pdb.set_trace()
                # intrarecord mean & std
                self.statistics_list_for_label[label]['mean'].append(np.mean(eva.errList))
                self.statistics_list_for_label[label]['std'].append(np.std(eva.errList))

                FNcnt += FN
                FPcnt += FP
                # Save record result
                record_statistics[record_name][label] = dict(
                        mean = np.mean(eva.errList),
                        std = np.std(eva.errList),
                        FN = FN,
                        FP = FP,
                        TP = len(eva.errList))
                
                # police check:
                # find error that exceeded thresholds
                self.police_check(eva,RoundInd,result_file_list[file_ind],label)

            self.error_list_for_label[label].extend(errList)
            self.false_negtive_list[label] += FNcnt
            self.false_positive_list[label] += FPcnt
        # Save record statistics of each round
        self.round_record_statistics.append(record_statistics)

    def police_check(self,eva,RoundInd,resultfilepath,label):
        '''Whether result statistics is worse than expected.'''
        if len(eva.errList) == 0:
            mean = -1
            std = -1
        else:
            mean = np.mean(eva.errList)
            std = np.std(eva.errList)
        FN = eva.getFNlist()
        FP = eva.getFPlist()

        # txt_logger = TextLogger(
                # os.path.join(curfolderpath,'Log_which_record_is_bad.txt'))
        # threshold
        good_record_mark = True
        if mean > self.mean_threshold:
            print 'mean = ',mean
            # txt_logger.dump('mean = {}'.format(mean))
            good_record_mark = False
        if std > self.std_threshold:
            print 'std = ', std
            good_record_mark = False
            # txt_logger.dump('std = {}'.format(std))
        if FN > self.FN_threshold:
            print 'FN = ', FN
            good_record_mark = False
            # txt_logger.dump('FN = {}'.format(FN))
        if FP > self.FP_threshold:
            print 'FP = ', FP
            good_record_mark = False
            # txt_logger.dump('FP = {}'.format(FP))
        if good_record_mark == False:
            print 'Round {}, record name {}'.format(RoundInd,os.path.split(resultfilepath)[-1])
            print '-'*20
            print '\n'
            # txt_logger.dump('Round {}, record name {} label [{}]'.format(RoundInd,os.path.split(resultfilepath)[-1],label))
            # txt_logger.dump('-'*20)

        

    def ReadEvaluationFolder(self,RoundInd):
        evalinfopath = os.path.join(curfolderpath,'MultiLead4','EvalInfoRound{}'.format(RoundInd))
        # write to json
        with open(os.path.join(evalinfopath,'ErrData.json'),'r') as fin:
            ErrData = json.load(fin)
        # add to total error list & False Negtive List
        for label in self.possible_label_list:
            self.error_list_for_label[label].extend(ErrData[label]['errList'])
            self.false_negtive_list[label] += ErrData[label]['FN']
            self.false_positive_list[label] += ErrData[label]['FP']

            #print 'label:',label,self.false_positive_list[label]
            #pdb.set_trace()

    def get_mean_and_std(self):
        '''
        The error value for all the records for given label is collected into a list.
        mean: mean(list)
        std: std(list)
        Results are saved in total_error_diction.
        '''
        for label in self.possible_label_list:
            self.total_error_diction[label] = dict()
            if len(self.error_list_for_label[label]) == 0:
                self.total_error_diction[label]['mean'] = -1
                self.total_error_diction[label]['std'] = -1
            else:
                self.total_error_diction[label]['mean'] = np.mean(
                        self.error_list_for_label[label])
                self.total_error_diction[label]['std'] = np.std(
                        self.error_list_for_label[label])

    def get_Sensitivity_and_Pplus(self):
        # Sensitivity & P plus
        for label in self.possible_label_list:
            FN = self.false_negtive_list[label]
            FP = self.false_positive_list[label]
            TP = len(self.error_list_for_label[label])
            if FN + TP == 0:
                self.sensitivity[label] = -1
                self.Pplus[label] = -1
                continue
                
            if abs(FN + TP) < 1e-6:
                self.sensitivity[label] = 1.0
            else:
                self.sensitivity[label] = 1.0*(1.0-float(FN)/(FN+TP))
            if abs(FP + TP) < 1e-6:
                self.Pplus[label] = 1.0
            else:
                self.Pplus[label] = 1.0*(1.0-float(FP)/(FP+TP))
        
    def ComputeStatistics(self):
        self.get_mean_and_std()
        self.get_Sensitivity_and_Pplus()
    def output_to_json(self,jsonfilename):
        with open(jsonfilename,'w') as fout:
            json.dump(self.total_error_diction,fout,indent = 4,sort_keys = True)
            print 'json file save to {}.'.format(jsonfilename)
    def display_error_statistics(self):
        '''Display error stats to stdout.'''
        print '\n'
        print '-'*30
        print '[label]  [mean]   [std]   [False Negtive]'
        for label in self.possible_label_list:
            print label, self.total_error_diction[label]['mean'],
            self.total_error_diction[label]['std'], self.false_negtive_list[label]
            print 'Sensitivity :{:.3f},\t Positive Predictivity: {:.3f}'.format(
                    self.sensitivity[label],self.Pplus[label])
            print 'Mean across records: ',
            print 'mean = {}, std = {}'.format(
                    np.nanmean(self.statistics_list_for_label[label]['mean']),
                    np.nanmean(self.statistics_list_for_label[label]['std']))

    def dump_statistics_to_HTML(self, html_file_name):
        '''Like display_error_statistics(), dump statistics data to txt file.'''
        def dump_round_statistics_to_Json():
            '''Dump record statistics of every round to json file.'''
            json_file_path = os.path.join(
                    os.path.dirname(html_file_name),
                    'round_statistics.json')
            with open(json_file_path, 'w') as fout:
                json.dump(self.round_record_statistics, fout)
        def dumpTotalResultToJson():
            '''Dump total result to json.'''
            json_data = dict()
            for label in self.possible_label_list:
                json_data[label] = dict()
                # Get mean & std
                label_mean = np.nanmean(self.statistics_list_for_label[label]['mean'])
                label_std = np.nanmean(self.statistics_list_for_label[label]['std'])
                json_data[label]['mean'] = label_mean
                json_data[label]['std'] = label_std
                # Get Se
                label_Se = self.sensitivity[label]
                json_data[label]['Se'] = label_Se
                # Get Pp
                label_Pp = self.Pplus[label]
                json_data[label]['Pp'] = label_Pp
            json_file_path = os.path.join(
                    os.path.dirname(html_file_name),
                    'total_statistics.json')
            with open(json_file_path, 'w') as fout:
                json.dump(json_data, fout, indent = 4, sort_keys = True)


        def format_record_result():
            '''Result of each round'''
            html_template_path = os.path.join(
                    curfolderpath,
                    'templates',
                    'record-template.html')
            html_template_str_list = []
            with open(html_template_path, 'r') as fin:
                for line in fin:
                    html_template_str_list.append(line)
                html_template_str = ''.join(html_template_str_list)
            html_template = string.Template(html_template_str)

            # Format output
            html_data = ''
            for round_ind in xrange(0, len(self.round_record_statistics)):

                html_data += '<br>' * 3 + '\n'
                html_data += '<table>\n'
                html_data += '<tr><td>Round %d</td></tr>\n' % (round_ind + 1)
                for record_name, result_stat in self.round_record_statistics[round_ind].iteritems():
                    
                    data_dict = dict(record_name = record_name)
                    for label in self.possible_label_list:
                        if label in ['P', 'R', 'T']:
                            html_label = label + 'peak'
                        else:
                            html_label = label
                        # Get mean & std
                        label_mean = result_stat[label]['mean']
                        label_std = result_stat[label]['std']
                        data_dict.update(
                                {html_label+'_error': '%.1f & %.1f' % (label_mean, label_std)})
                        # Get Se
                        label_Se = result_stat[label]['FN']
                        data_dict.update({html_label+'_se': '%.3f' % label_Se})

                        # Get Pp
                        label_Pp = result_stat[label]['FP']
                        data_dict.update({html_label+'_pp': '%.3f' % label_Pp})
                    round_data = html_template.substitute(data_dict)
                    html_data += round_data
                html_data += '</table>\n\n'
                
            return html_data

        # 1. Read html template to string
        html_template_path = os.path.join(
                curfolderpath,
                'templates',
                'result-table.html')
        html_template_str_list = []
        with open(html_template_path, 'r') as fin:
            for line in fin:
                html_template_str_list.append(line)
            html_template_str = ''.join(html_template_str_list)

        html_template = string.Template(html_template_str)
        
        css_file_path = os.path.join(
                projhomepath,
                'EvaluationSchemes',
                'templates',
                'result-table.css')
        data_dict = dict(css_file_path=css_file_path)
        # Fill in data, save in dict
        for label in self.possible_label_list:
            if label in ['P', 'R', 'T']:
                html_label = label + 'peak'
            else:
                html_label = label
            # Get mean & std
            label_mean = np.nanmean(self.statistics_list_for_label[label]['mean'])
            label_std = np.nanmean(self.statistics_list_for_label[label]['std'])
            data_dict.update({html_label+'_error': '%.1f & %.1f' % (label_mean, label_std)})
            # Get Se
            label_Se = self.sensitivity[label]
            data_dict.update({html_label+'_se': '%.3f' % label_Se})

            # Get Pp
            label_Pp = self.Pplus[label]
            data_dict.update({html_label+'_pp': '%.3f' % label_Pp})

        # Fill in null labels
        for label in ['Ponset', 'P', 'Poffset', 'Ronset', 'Roffset', 'T', 'Toffset', 'R']:
            if label in self.possible_label_list:
                continue
            if label in ['P', 'R', 'T']:
                html_label = label + 'peak'
            else:
                html_label = label
            # Get mean & std
            # print 'label:', label
            # print 'html:', html_label
            data_dict.update({html_label+'_error': 'NA & NA'})

            # Get Se
            data_dict.update({html_label+'_se': 'NA' })

            # Get Pp
            data_dict.update({html_label+'_pp': 'NA' })

        # Fill in the data
        # data_dict['append_data'] = format_record_result()
        data_dict['append_data'] = ''
        html_data = html_template.substitute(data_dict)
            
        with open(html_file_name, 'w') as fout:
            fout.write(html_data)

        # Dump to json
        dump_round_statistics_to_Json()
        # Dump total results to json
        dumpTotalResultToJson()

    def dump_statistics_to_file(self, log_file_name):
        '''Like display_error_statistics(), dump statistics data to txt file.'''
        def BreakLine(fout):
            fout.write('\n')
        with open(log_file_name, 'w') as fout:
            fout.write('\n')
            fout.write('-'*30 + '\n')
            fout.write('[label]  [mean]   [std]   [False Negtive]');
            BreakLine(fout)
            for label in self.possible_label_list:
                fout.write('label, self.total_error_diction[label][''mean'']')
                fout.write('{} {}\n'.format(self.total_error_diction[label]['std'],
                    self.false_negtive_list[label]))
                BreakLine(fout)
                fout.write('Sensitivity :{:.3f},\t Positive Predictivity: {:.3f}'.format(
                        self.sensitivity[label],self.Pplus[label]))
                BreakLine(fout)
                fout.write('Mean across records: ')
                fout.write('mean = {}, std = {}'.format(
                        np.nanmean(self.statistics_list_for_label[label]['mean']),
                        np.nanmean(self.statistics_list_for_label[label]['std'])))
                BreakLine(fout)

def ConverterFactory(converter_name):
    '''Return converter according to converter name.'''
    # Import Converters
    ListResultConverterModule = importlib.import_module(
            'list-result-converter')
    ListResultConverter = ListResultConverterModule.ListResultConverter
    # SimpleConverterModule = importlib.import_module(
            # 'EvaluationSchemes.result-converters.original-converter')
    # SimpleConverter = SimpleConverterModule.OriginalConverter

    # Converter function handle
    if converter_name == 'simple-converter':
        result_converter = SimpleConverter.convert
    elif converter_name == 'list-result-converter':
        result_converter = ListResultConverter
    return result_converter

def ResetFolder(folder_name):
    # Remove existing folder
    if os.path.exists(folder_name) == True:
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)

if __name__ == '__main__':

    # Name of the experiment
    experiment_name = 'regression-test'
    # Set folder path.
    prediction_result_folder = os.path.join(
            curfolderpath,
            'fixed_results')
    evaluation_result_path = os.path.join(curfolderpath,
            'eval_output')
    total_round_number = 30
    result_filename_pattern = '*.json'

    # Labels to extract statistics from.
    # possible_label_list = ['P',]
    # possible_label_list = None
    possible_label_list = ['Ponset', 'P', 'Poffset', 'Ronset', 'Roffset', 'T', 'Toffset']

    # Get converter
    result_converter = RoundResultConverter.convert

    ResetFolder(evaluation_result_path)

    # Construct Police object
    police_obj = FindBadRecord(10, 13, 20, 20,
            possible_label_list = possible_label_list,
            result_converter = result_converter)

    for RoundInd in xrange(1, total_round_number + 1):
        print 'processing Round :',RoundInd
        GroupResultFolder = os.path.join(prediction_result_folder,
                'round{}'.format(RoundInd))
        result_file_list = glob.glob(os.path.join(GroupResultFolder,
            result_filename_pattern))
        police_obj.RunEval(RoundInd,
                GroupResultFolder,
                result_file_list)

    # Compute error statistics
    police_obj.ComputeStatistics()
    # Display error statistics
    police_obj.display_error_statistics()
    police_obj.dump_statistics_to_HTML(os.path.join(
        evaluation_result_path, 
        'evaluation-statistics.html'))
