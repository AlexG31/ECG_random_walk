#encoding:utf8
import os
import sys
import matplotlib.pyplot as plt
import bisect
import numpy as np
import numpy.random as random
from QTdata.loadQTdata import QTloader
from feature_extractor.feature_extractor import ECGfeatures
import sklearn.ensemble.RandomForestRegressor as RandomForestRegressor

class RandomWalker(object):
    '''Random Tree Walk Model for ECG.'''
    def __init__(self, target_label = 'P'):
        '''This class only focus on a single label.'''
        self.target_label = target_label


    def gaussian_training_sampling(self,
            pos_list,
            draw_per_annotation = 3,
            max_distance = 100):
        '''Draw training samples according to gaussian distribution.
        Input:
            Position list of expert annotations.
        '''
        result_set = set()
        for pos in pos_list:
            for draw_ind in xrange(0, draw_per_annotation):
                max_loop_count = 10000
                while max_loop_count > 0:
                    max_loop_count -= 1
                    bias = random.normal(0, max_distance / 6.0, 1)
                    new_pos = pos + bias[0]
                    if new_pos not in result_set:
                        result_set.add(new_pos)
                        break
                if max_loop_count < 0:
                    raise Exception(
                            'Sampling failed(maximum loop count reached)!')
        return [int(x) for x in result_set]
            
            

    def get_configuration(self):
        '''Get configuration diction.'''
        current_file_path = os.path.realpath(__file__)
        current_folder = os.path.split(current_file_path)
        conf = dict(
                fs = 250,
                winlen_ratio_to_fs = 4,
                WT_LEVEL = 6,
                random_pattern_path = os.path.join(current_folder,
                    'data',
                    'random_pattern.json'),
                )
        return conf

    def do_training_on_qt(self, record_name = 'sel103'):
        '''Training test.'''
        
        qt = QTloader()
        sig = qt.load(record_name)
        raw_sig = sig['sig']

        expert_annotations = qt.getExpert(record_name)
        annot_pos_list = [x[0] for x in expert_annotations if x[1] == self.target_label]
        training_indexes = self.gaussian_training_sampling(annot_pos_list)

    
        configuration_info = self.get_configuration()
        feature_extractor = ECGfeatures(raw_sig, configuration_info)

        training_data = list()
        len_annotations = len(annot_pos_list)
        for pos in training_indexes:
            # Find closest target annotation
            lb = bisect.bisect_left(annot_pos_list, pos)
            near_pos = -1
            for si in xrange(lb - 1, lb + 1):
                if si >= 0 && si < len_annotations:
                    dist = abs(pos - annot_pos_list[si])
                    if (near_pos == -1 or
                            dist < abs(pos - near_pos)):
                        near_pos = annot_pos_list[si]
            if near_pos == -1:
                raise Exception(
                        'No %s annotations in training sample!' % (
                            self.target_label,))
            value = 1 if near_pos < pos else -1
            feature_vector = feature_extractor.frompos(pos)
            training_data.append(feature_vector, value)

                    
        self.regressor = RandomForestRegressor(30, max_depth = 5)
        self.regressor.fit(*training_data)
                
        # plt.figure(1)
        # plt.plot(sig['sig'])
        # amp_list = [raw_sig[x] for x in annot_pos_list]
        # plt.plot(annot_pos_list, amp_list, 'ro', markersize = 15)

        # amp_list = [raw_sig[x] for x in training_indexes]
        # plt.plot(training_indexes, amp_list, 'mx', markersize = 15)
        # plt.show()
            

    def testing_n(self, raw_signal, num2test = 100):
        '''
        Input:
            ECG signal.
        '''
        len_sig = len(raw_signal)
        pos_list = random.sample(num2test, xrange(0, len_sig))
        configuration_info = self.get_configuration()
        feature_extractor = ECGfeatures(raw_sig, configuration_info)

        results = list()
        for pos in pos_list:
            feature_vector = feature_extractor.frompos(pos)
            value = self.regressor.predict(feature_vector)
            results.append(value)
        return results
            



if __name__ == '__main__':
    walker = RandomWalker()
    walker.do_training_on_qt()
        
