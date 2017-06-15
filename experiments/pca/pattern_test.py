#encoding:utf8
import os, sys, json
import pdb


current_folder = os.path.realpath(__file__)
current_folder = os.path.dirname(current_folder)

def test():
    '''Generated random pattern json analysis.'''

    random_pattern_path = os.path.join(current_folder, 'output.json')
    print '========%s======='%random_pattern_path
    with open(random_pattern_path, 'r') as fin:
        data = json.load(fin)
        print 'Total number of wavelet levels :', len(data)
        for ind in xrange(0, len(data)):
            print 'Number of pairs in level %d: %d' % (ind, len(data[ind]))


test()
