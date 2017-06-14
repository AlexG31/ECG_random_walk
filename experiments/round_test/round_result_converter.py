#encoding:utf-8
"""
Result Converter
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
import pdb

class Round2LeadResultConverter(object):
    def __init__(self):
        pass
    @staticmethod
    def convert(result_in):
        '''Convert list result format.
            Inputformat:
                {'record_name': sel1xx,
                 'result':[
                                [pos, label],
                                [...],
                            ]
                }
            Outputformat:
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
        result_out = dict()
        result_out['LeadResult'] = list()

        result_out['recname'] = result_in['record_name']

        label_dict = dict()
        for pos, label in result_in['result'][0]:
            if label not in label_dict:
                label_dict[label] = [pos,]
            else:
                label_dict[label].append(pos)
        result_out['LeadResult'] = [label_dict, ]

        label_dict = dict()
        for pos, label in result_in['result'][1]:
            if label not in label_dict:
                label_dict[label] = [pos,]
            else:
                label_dict[label].append(pos)
        result_out['LeadResult'].append(label_dict)

        return result_out

class RoundResultConverter(object):
    def __init__(self):
        pass
    @staticmethod
    def convert(result_in):
        '''Convert list result format.
            Inputformat:
                {'record_name': sel1xx,
                 'result':[
                                [pos, label],
                                [...],
                            ]
                }
            Outputformat:
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
        result_out = dict()
        result_out['LeadResult'] = list()

        result_out['recname'] = result_in['record_name']

        label_dict = dict()
        for pos, label in result_in['result']:
            if label not in label_dict:
                label_dict[label] = [pos,]
            else:
                label_dict[label].append(pos)
        import copy
        result_out['LeadResult'] = [label_dict, copy.deepcopy(label_dict)]
        return result_out
            

## TEST
if __name__ == '__main__':
    result_file_path = '/home/alex/LabGit/ProjectSwiper/result/run-6/round1/result_sel230'
    with open(result_file_path, 'r') as fin:
        result = json.load(fin)
        result = RoundResultConverter.convert(result)
        print result.keys()
        print result['LeadResult'][0].keys()
        print result['LeadResult'][0]['P'][0:10]
        pdb.set_trace()
