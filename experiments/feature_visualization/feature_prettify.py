#encoding:utf8
import os
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt

def prettify(ax, raw_sig, pairs, importance_list, center_index):
    '''Discretize RSWT pair importance.'''
    ax.plot(raw_sig, 'black')
    ax.set_xlim(0, len(raw_sig))
    # plt.plot(raw_sig)
    # plt.plot(center_index, raw_sig[center_index], 'ro',
            # markersize = 12,
            # label = 'Center Index')

    # print 'Lenght of pairs:', len(pairs)
    # for ind in xrange(0, 10):
        # print pairs[ind]
    # print 'length of signal:', len(raw_sig)
    # print '-' * 21

    # Fill triangles
    max_height = np.max(raw_sig)
    region_width = len(raw_sig) / 8
    max_alpha = 0
    for region_left in xrange(0, len(raw_sig), region_width):
        region_right = region_left + region_width
        region_right = min(len(raw_sig), region_right)
        # Number of pairs in this region
        weight = sum(map(lambda x:x[1],
                filter(lambda x: (x[0][0][0] >= region_left and
                    x[0][0][0] < region_right),
                zip(pairs, importance_list))))
        # alpha = 0.0 + float(weight) * 0.2
        alpha = 0.05 + float(weight) * 0.2
        max_alpha = max(max_alpha, alpha)
        alpha = min(1.0, alpha)
        ax.fill([region_left, center_index, region_right],
                [raw_sig[0], max_height, raw_sig[0]],
                color = (0.3, 0.3, 0.3),
                alpha = alpha)
    # plt.title('ECG')
    ax.grid(True)
    print 'Maximum alpha value: %f' % max_alpha
    
def prettify_stripe(ax, raw_sig, pairs, importance_list, center_index):
    '''Discretize RSWT pair importance into stripe.'''
    ax.plot(raw_sig, 'black')
    ax.set_xlim(0, len(raw_sig))
    # plt.plot(raw_sig)
    # plt.plot(center_index, raw_sig[center_index], 'ro',
            # markersize = 12,
            # label = 'Center Index')

    # print 'Lenght of pairs:', len(pairs)
    # for ind in xrange(0, 10):
        # print pairs[ind]
    # print 'length of signal:', len(raw_sig)
    # print '-' * 21

    # Fill triangles
    max_height = np.max(raw_sig)
    region_width = len(raw_sig) / 8
    max_alpha = 0
    for region_left in xrange(0, len(raw_sig), region_width):
        region_right = region_left + region_width
        region_right = min(len(raw_sig), region_right)

        # for p1, p2 in pairs:
            # print 'point1:', p1, 'point2:', p2
        # pdb.set_trace()
        # Number of pairs with first point's x in this region
        weight = sum(map(lambda x:x[1],
                filter(lambda x: (x[0][0][0] >= region_left and
                    x[0][0][0] < region_right and
                    x[0][0][0] != center_index),
                zip(pairs, importance_list))))
        # Number of pairs with second point's x in this region
        weight += sum(map(lambda x:x[1],
                filter(lambda x: (x[0][1][0] >= region_left and
                    x[0][1][0] < region_right and
                    x[0][1][0] != center_index),
                zip(pairs, importance_list))))
        # alpha = 0.0 + float(weight) * 0.2
        alpha = 0.05 + float(weight) * 0.2
        max_alpha = max(max_alpha, alpha)
        alpha = min(1.0, alpha)
        ax.fill([region_left, region_right, region_right, region_left],
                [-max_height, -max_height, max_height, max_height],
                color = (0.3, 0.3, 0.3),
                alpha = alpha)
    # plt.title('ECG')
    ax.grid(True)
    print 'Maximum alpha value: %f' % max_alpha

if __name__ == '__main__':
    prettify()
