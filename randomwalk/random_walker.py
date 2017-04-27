#encoding:utf8
import os
import sys
import time
import pdb
import bisect
import joblib
import numpy as np
import numpy.random as random
import random as pyrandom
from feature_extractor.feature_extractor import ECGfeatures
from sklearn.ensemble import RandomForestRegressor

class RandomWalker(object):
    '''Random Tree Walk Model for ECG.
    [Important]:
        All input signals must resample to fs = 250Hz.
    '''
    def __init__(self, target_label = 'P',
            random_forest_config = dict(),
            random_pattern_file_name = None):
        '''This class only focus on a single label.'''
        self.target_label = target_label
        self.regressor = None
        self.training_data = list()
        self.random_forest_config = random_forest_config
        if 'max_depth' not in self.random_forest_config:
            self.random_forest_config['max_depth'] = 15
        self.random_pattern_file_name = random_pattern_file_name

        # For batch testing.
        self.position_bucket = list()
        self.confined_range_bucket = list()

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
                winlen_ratio_to_fs = 3,
                WT_LEVEL = 9
                )

        if self.random_pattern_file_name is not None:
            conf['random_pattern_path'] = self.random_pattern_file_name
        return conf

    def collect_training_data(self, raw_sig, expert_annotations):
        '''Incrementally collect training samples X and training values y.'''
        annot_pos_list = [x[0] for x in expert_annotations if x[1] == self.target_label]
        training_indexes = self.gaussian_training_sampling(annot_pos_list)

        configuration_info = self.get_configuration()
        feature_extractor = ECGfeatures(raw_sig, configuration_info)

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
            self.training_data.append((feature_vector, value))
        
    def save_model(self, model_file_name):
        with open(model_file_name, 'wb') as fout:
            joblib.dump(self.regressor, fout, compress = True)
            print 'Model save as %s.' % model_file_name

    def load_model(self, model_file_name):
        with open(model_file_name, 'rb') as fout:
            self.regressor = joblib.load(fout)

    def training(self):
        '''Trianing for samples X and their output values y.'''
        self.regressor = RandomForestRegressor(30,
                **self.random_forest_config
                )
        X, y = zip(*self.training_data)
        self.regressor.fit(X, y)

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
        
    def GetFeatureExtractor(self, raw_signal):
        '''Get feature extractor for a given signal.'''
        if len(raw_signal) == 0:
            raise Exception('Empty testing signal!')
        configuration_info = self.get_configuration()
        feature_extractor = ECGfeatures(raw_signal, configuration_info)
        return feature_extractor

    def testing_walk_extractor(self, feature_extractor,
            seed_position,
            iterations =  100,
            stepsize = 4,
            confined_range = None):
        '''
        Start random walk with seed position.
        Input:
            ECG signal feature extractor.
        Output:
            zip(pos_list, results):
                walk path & predict probability at each position.
        '''
        results = list()
        pos_list = list()
        pos_dict = dict()
        # Benchmarking
        feature_collecting_time_cost = 0
        predict_time_cost = 0

        pos = seed_position
        for pi in xrange(0, iterations):
            # Boundary
            if pos < 0:
                pos = 0
            elif pos >= len(feature_extractor.signal_in):
                pos = len(feature_extractor.signal_in) - 1

            pos_list.append(pos)

            if pos not in pos_dict:
                start_time = time.time()
                feature_vector = np.array(feature_extractor.frompos(pos))
                feature_vector = feature_vector.reshape(1, -1)
                feature_collecting_time_cost += time.time() - start_time
                start_time = time.time()
                value = self.regressor.predict(feature_vector)
                predict_time_cost += time.time() - start_time
                pos_dict[pos] = value
            else:
                value = pos_dict[pos]
            results.append(value)

            # Update next position
            threshold = (value + 1.0) / 2.0
            # threshold = self.sigmod_function(threshold)
            direction = -1.0 if random.ranf() >= threshold else 1.0
            pos += int(direction * stepsize)


            # Next position confined range
            if confined_range is not None:
                if pos > confined_range[1]:
                    pos = confined_range[1]
                elif pos < confined_range[0]:
                    pos = confined_range[0]

        # print 'Feature collecting time cost: %f secs.' % feature_collecting_time_cost
        # print 'Prediction time cost %f seconds.' % predict_time_cost
        return zip(pos_list, results)


    # Fast testing
    def prepareTestSample(self, seed_position, confined_range = None):
        '''Prepare sample for batch testing.'''
        self.position_bucket.append(seed_position)
        self.confined_range_bucket.append(confined_range)

    def runPreparedTesting(self,
            feature_extractor,
            iterations =  100,
            stepsize = 4,
            ):
        '''Start prepared batch testing. Clear prepare bucket afterwards.
        Returns:
            path_list
            scores_list
        '''
        if len(self.position_bucket) == 0:
            return list()

        
        # For scores of bucket elements
        scores_list = [list(), ] * len(self.position_bucket)

        # For path of bucket elements
        path = list()
        path_list = list()
        for ind in xrange(0, len(self.position_bucket)):
            path_list.append(list())

        # For tested position hash
        pos_dict = dict()

        # Benchmarking
        feature_collecting_time_cost = 0
        predict_time_cost = 0

        for pi in xrange(0, iterations):
            # List of pair (pos_to_test, bucket_index)
            prepare_test_info = list()
            scores = list()

            # print 'self.position_bucket:', self.position_bucket
            for bucket_index in xrange(0, len(self.position_bucket)):
                pos = self.position_bucket[bucket_index]
                confined_range = self.confined_range_bucket[bucket_index]
                # print '[%d] pos %d, range(%d, %d)' % (bucket_index, pos, confined_range[0], confined_range[1])

                # Boundary
                if pos < 0:
                    pos = 0
                elif pos >= len(feature_extractor.signal_in):
                    pos = len(feature_extractor.signal_in) - 1
                # confined range
                if confined_range is not None:
                    if pos > confined_range[1]:
                        pos = confined_range[1]
                    elif pos < confined_range[0]:
                        pos = confined_range[0]

                path_list[bucket_index].append(pos)

                # Prepare for testing
                if pos not in pos_dict:
                    prepare_test_info.append((pos, bucket_index))
                    scores.append(None)
                    # feature_vector = np.array(feature_extractor.frompos(pos))
                    # feature_vector = feature_vector.reshape(1, -1)
                    # feature_collecting_time_cost += time.time() - start_time
                    # start_time = time.time()
                    # value = self.regressor.predict(feature_vector)
                    # predict_time_cost += time.time() - start_time
                    # pos_dict[pos] = value
                else:
                    value = pos_dict[pos]
                    scores.append(value)

            # print 'prepare_test_info:', prepare_test_info
            # print 'scores:', scores 
            # print 'path:', zip(*path_list)[-1]
            # pdb.set_trace()
            # Collecting features
            feature_mat = list()
            for pos, bucket_index in prepare_test_info:
                feature_vector = feature_extractor.frompos(pos)
                # feature_vector = np.array(feature_extractor.frompos(pos))
                # feature_vector = feature_vector.reshape(1, -1)
                feature_mat.append(feature_vector)
            if len(feature_mat) > 0:
                # Start random forest testing
                tested_scores = self.regressor.predict(feature_mat)
                test_result_info = zip(prepare_test_info, tested_scores)
                for test_info_element, tested_score in test_result_info:
                    pos, bucket_index = test_info_element
                    scores[bucket_index] = tested_score
                    pos_dict[pos] = tested_score
            
            # Add to scores_list
            scores_list = zip(scores_list, scores)

            # Update seed_positions for next round
            for bucket_index in xrange(0, len(self.position_bucket)):
                score = scores[bucket_index]
                pos = self.position_bucket[bucket_index]
                confined_range = self.confined_range_bucket[bucket_index]

                # Update next position
                threshold = (score + 1.0) / 2.0
                # threshold = self.sigmod_function(threshold)
                direction = -1.0 if random.ranf() >= threshold else 1.0
                pos += int(direction * stepsize)


                # Next position confined range
                if confined_range is not None:
                    if pos > confined_range[1]:
                        pos = confined_range[1]
                    elif pos < confined_range[0]:
                        pos = confined_range[0]

                self.position_bucket[bucket_index] = pos

        # Clear data
        self.position_bucket = list()
        self.confined_range_bucket = list()

        return (path_list, scores_list)


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
    
    import matplotlib.pyplot as plt
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
    
    import matplotlib.pyplot as plt
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
    record_name = 'sel46'
    sig = qt.load(record_name)
    raw_sig = sig['sig']

    random_forest_config = dict(
            max_depth = 10)
    walker = RandomWalker(target_label = 'P',
            random_forest_config = random_forest_config)

    print 'training...'
    start_time = time.time()
    walker.do_training_on_qt(record_name = 'sel103')
    print 'trianing used %.3f seconds' % (time.time() - start_time)

    seed_position = 100000
    import matplotlib.pyplot as plt
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
        predict_pos = np.mean(pos_list[len(pos_list) / 2:])
        
        # amp_list = [raw_sig[int(x)] for x in pos_list]
        amp_list = []
        bias = raw_sig[pos_list[0]]
        for pos in pos_list:
            amp_list.append(bias)
            bias -= 0.01

        plt.plot(predict_pos,
                raw_sig[int(predict_pos)],
                'ro',
                markersize = 14,
                label = 'predict position')
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
        
