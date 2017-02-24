#encoding:utf8
import os
import sys
import time
import matplotlib.pyplot as plt
import pdb
import bisect
import numpy as np
import numpy.random as random
import random as pyrandom
from QTdata.loadQTdata import QTloader
from feature_extractor.feature_extractor import ECGfeatures
from sklearn.ensemble import RandomForestRegressor

class RandomWalker(object):
    '''Random Tree Walk Model for ECG.'''
    def __init__(self, target_label = 'P',
            random_forest_config = dict()):
        '''This class only focus on a single label.'''
        self.target_label = target_label
        self.random_forest_config = random_forest_config
        if 'max_depth' not in self.random_forest_config:
            self.random_forest_config['max_depth'] = 15


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
        current_folder = os.path.dirname(current_file_path)
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
                if si >= 0 and si < len_annotations:
                    dist = abs(pos - annot_pos_list[si])
                    if (near_pos == -1 or
                            dist < abs(pos - near_pos)):
                        near_pos = annot_pos_list[si]
            if near_pos == -1:
                raise Exception(
                        'No %s annotations in training sample!' % (
                            self.target_label,))
            value = -1 if near_pos < pos else 1
            feature_vector = feature_extractor.frompos(pos)
            training_data.append((feature_vector, value))

                    
        self.regressor = RandomForestRegressor(30,
                **self.random_forest_config
                )
        X, y = zip(*training_data)
        self.regressor.fit(X, y)
                
        # plt.figure(1)
        # plt.plot(sig['sig'])
        # amp_list = [raw_sig[x] for x in annot_pos_list]
        # plt.plot(annot_pos_list, amp_list, 'ro', markersize = 15)

        # amp_list = [raw_sig[x] for x in training_indexes]
        # plt.plot(training_indexes, amp_list, 'mx', markersize = 15)
        # plt.show()
            

    def testing_n(self, raw_signal, num2test = 100):
        '''
        Selecting num2test samples to test.
        Input:
            ECG signal.
        '''
        len_sig = len(raw_signal)
        pos_list = pyrandom.sample(xrange(100000, 101000), num2test)
        configuration_info = self.get_configuration()
        feature_extractor = ECGfeatures(raw_signal, configuration_info)

        results = list()
        testing_features = list()
        for pos in pos_list:
            feature_vector = feature_extractor.frompos(pos)
            testing_features.append(feature_vector)

        results = self.regressor.predict(testing_features)
        return zip(pos_list, results)
            
    def sigmod_function(self, x):
        '''Increase the probability of left and right directions.'''
        return 1.0 / (1.0 + np.exp(-x)) - 0.5
        
    def testing_walk(self, raw_signal, seed_position,
            iterations =  100,
            stepsize = 4):
        '''
        Start random walk with seed position.
        Input:
            ECG signal.
        '''
        configuration_info = self.get_configuration()
        feature_extractor = ECGfeatures(raw_signal, configuration_info)

        results = list()
        pos_list = list()
        pos_dict = dict()

        pos = seed_position
        for pi in xrange(0, iterations):
            pos_list.append(pos)

            if pos not in pos_dict:
                feature_vector = np.array(feature_extractor.frompos(pos))
                feature_vector = feature_vector.reshape(1, -1)
                value = self.regressor.predict(feature_vector)
                pos_dict[pos] = value
            else:
                value = pos_dict[pos]
            results.append(value)

            # Update next position
            threshold = (value + 1.0) / 2.0
            # threshold = self.sigmod_function(threshold)
            direction = -1.0 if random.ranf() >= threshold else 1.0
            pos += int(direction * stepsize)

        return zip(pos_list, results)


def Test1():
    '''Test case 1'''
    qt = QTloader()
    record_name = 'sel103'
    sig = qt.load(record_name)
    raw_sig = sig['sig']
    walker = RandomWalker()
    print 'training...'
    walker.do_training_on_qt(record_name = record_name)

    print 'testing...'
    results = walker.testing_n(sig['sig'], 100)

    left_pos_list = [x[0] for x in results if x[1] <= 0]
    left_value_list = [x[1] for x in results if x[1] <= 0]

    right_pos_list = [x[0] for x in results if x[1] > 0]
    right_value_list = [x[1] for x in results if x[1] > 0]
    
    plt.figure(1)
    plt.plot(sig['sig'], label = record_name)
    amp_list = [raw_sig[x] for x in left_pos_list]
    plt.plot(left_pos_list, amp_list, 'r<', label = 'left',
            markersize = 15)
    # Annotate
    for x,y, score in zip(left_pos_list, amp_list, left_value_list):
        plt.annotate('%.3f' % score,
                xy = (x, y),
                xytext = (x + random.ranf() * 2.0 - 1, y + random.ranf()),
                arrowprops = dict(
                    arrowstyle = '->',
                    )
                )
    amp_list = [raw_sig[x] for x in right_pos_list]
    plt.plot(right_pos_list, amp_list, 'b>', label = 'right',
            markersize = 15)
    # Annotate
    for x,y, score in zip(right_pos_list, amp_list, right_value_list):
        plt.annotate('%.3f' % score,
                xy = (x, y),
                xytext = (x + random.ranf() * 2.0 - 1, y - random.ranf()),
                arrowprops = dict(
                    arrowstyle = '->',
                    )
                )
    plt.grid(True)
    plt.xlim(100000, 110000)
    plt.show()
    
def Test2():
    '''Test case 2: random walk.'''
    qt = QTloader()
    record_name = 'sel103'
    sig = qt.load(record_name)
    raw_sig = sig['sig']
    walker = RandomWalker()
    print 'training...'
    start_time = time.time()
    walker.do_training_on_qt(record_name = record_name)
    print 'trianing used %.3f seconds' % (time.time() - start_time)

    print 'testing...'
    start_time = time.time()
    results = walker.testing_walk(sig['sig'], 100000, iterations = 100,
            stepsize = 10)
    print 'testing used %.3f seconds' % (time.time() - start_time)

    pos_list, values = zip(*results)
    
    plt.figure(1)
    plt.plot(sig['sig'], label = record_name)
    # amp_list = [raw_sig[int(x)] for x in pos_list]
    amp_list = []
    bias = raw_sig[pos_list[0]]
    for pos in pos_list:
        amp_list.append(bias)
        bias -= 0.01

    plt.plot(pos_list, amp_list, 'r',
            label = 'walk path',
            markersize = 3,
            linewidth = 8,
            alpha = 0.3)
    plt.grid(True)
    plt.xlim(min(pos_list) - 100, max(pos_list) + 100)
    plt.show()
    
def Test3():
    '''Test case 3: random walk.'''
    qt = QTloader()
    record_name = 'sel100'
    sig = qt.load(record_name)
    raw_sig = sig['sig']

    random_forest_config = dict(
            max_depth = 10)
    walker = RandomWalker(random_forest_config = random_forest_config)

    print 'training...'
    start_time = time.time()
    walker.do_training_on_qt(record_name = 'sel103')
    print 'trianing used %.3f seconds' % (time.time() - start_time)

    seed_position = 100000
    plt.figure(1)
    plt.plot(sig['sig'], label = record_name)
    for ti in xrange(0, 20):
        seed_position += int(200.0 * random.ranf())
        print 'testing...(%d)' % seed_position
        start_time = time.time()
        results = walker.testing_walk(sig['sig'], seed_position, iterations = 100,
                stepsize = 10)
        print 'testing used %.3f seconds' % (time.time() - start_time)

        pos_list, values = zip(*results)
        
        # amp_list = [raw_sig[int(x)] for x in pos_list]
        amp_list = []
        bias = raw_sig[pos_list[0]]
        for pos in pos_list:
            amp_list.append(bias)
            bias -= 0.01

        plt.plot(pos_list, amp_list, 'r',
                label = 'walk path',
                markersize = 3,
                linewidth = 8,
                alpha = 0.3)
        plt.grid(True)
        plt.xlim(min(pos_list) - 100, max(pos_list) + 100)
        plt.legend()
        plt.show(block = False)
        pdb.set_trace()

if __name__ == '__main__':
    Test3()
        
