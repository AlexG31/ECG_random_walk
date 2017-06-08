#encoding:utf8
import os, sys, json


def parse():
    IDlist = list()
    with open('./positive_sample_48122.json', 'r') as fin:
        data = json.load(fin)
        
        for val in data:
            IDlist.append(val['id'])

    with open('./IDlist.json', 'w') as fout:
        json.dump(IDlist, fout)


parse()
