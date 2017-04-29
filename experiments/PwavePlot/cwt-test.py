import pywt, pdb, json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from QTdata.loadQTdata import QTloader

from swt_plot import removeQRS, getQRSRanges

def loadChanggeng(recID = '42736'):
    signal_file = './changgeng/%s.json' % recID
    annot_file = './annots/%s.json' % recID

    QRS_ranges = list()
    with open(annot_file, 'r') as fin:
        annots = json.load(fin)
        annots.sort(key = lambda x:x[0], reverse = False)
        annots = filter(lambda x:('R' == x[1][0] and len(x[1]) > 1), annots)

        for ind in xrange(0, len(annots) - 1):
            pos, label = annots[ind]
            if label == 'Ronset':
                if annots[ind + 1][1] == 'Roffset':
                    QRS_ranges.append((pos, annots[ind + 1][0]))

    with open(signal_file, 'r') as fin:
        data = json.load(fin)
    return data

def loadChanggengAnnots(recID = '42736'):
    signal_file = './changgeng/%s.json' % recID
    annot_file = './annots/%s.json' % recID

    QRS_ranges = list()
    with open(annot_file, 'r') as fin:
        annots = json.load(fin)

    return annots

def amplify(coef, y, ax, level = 20, color = 'r'):
    coef_shape = coef.shape
    P_point_list = list()
    thres = 0.2
    for ind in xrange(0, coef_shape[1]):
        if coef[level, ind] > thres:
            P_point_list.append(ind)
    amplist = [y[x] for x in P_point_list]
    ax[0].plot(y)
    ax[0].plot(P_point_list, amplist, 'o', markerfacecolor = color, alpha = 0.8, label = str(level))


def getCwtRange(coef, y, level = 20, thres = 0.1):
    coef_shape = coef.shape
    P_point_list = list()
    for ind in xrange(0, coef_shape[1]):
        if coef[level, ind] > thres:
            P_point_list.append(ind)
    return P_point_list


def removeRangesInQRS(wave_ranges, qrs_ranges):
    p1 = 0
    p2 = 0
    
    remain_ranges = list()
    while p1 < len(wave_ranges) and p2 < len(qrs_ranges):
        # No common region
        if (wave_ranges[p1][1] <= qrs_ranges[p2][0] or
                wave_ranges[p1][0] >= qrs_ranges[p2][1]):
            remain_ranges.append(wave_ranges[p1])
        else:
            left = max(wave_ranges[p1][0], qrs_ranges[p2][0])
            right = min(wave_ranges[p1][1], qrs_ranges[p2][1])

            common_width = max(0, right - left)
            if (wave_ranges[p1][1] - wave_ranges[p1][0] <= 1 or
                    common_width / float(wave_ranges[p1][1] - wave_ranges[p1][0]) >= 0.5):
                pass
            else:
                remain_ranges.append(wave_ranges[p1])
        if wave_ranges[p1][0] < qrs_ranges[p2][0]:
            p1 += 1
        else:
            p2 += 1
    return remain_ranges
            
            
            

def getWaveRange(coef, y, qrs_ranges = list(), fs = 250, cwt_levels = [15, 10, 8], width_wiggle_threshold = 7.0 / 250):
    '''Get possible P wave ranges.'''
    
    wave_ranges = getCwtRange(coef, y, level = cwt_levels[0])
    # wave_ranges = filter(lambda x:x > 1214 and x < 1500, wave_ranges)
    for level in cwt_levels[1:]:

        current_ranges = getCwtRange(coef, y, level = level)

        # Merge
        p1 = 0
        p2 = 0
        remain_list = list()
        p2_tail = p2
        while p2_tail < len(current_ranges):
            p2_tail += 1
            if (p2_tail >= len(current_ranges) or
                    current_ranges[p2_tail] > current_ranges[p2_tail - 1] + 1):
                # End of group
                break
        while p1 < len(wave_ranges) and p2 < len(current_ranges):
            print 'Run:'
            print wave_ranges[p1], current_ranges[p2]
            
            if wave_ranges[p1] <= current_ranges[p2]:
                p1_start = p1
                while p1 < len(wave_ranges):
                    p1 += 1
                    if p1 >= len(wave_ranges) or wave_ranges[p1] > wave_ranges[p1 - 1] + 1:
                        # End of group
                        break

                # Merging ranges in different level
                if wave_ranges[p1 - 1] >= current_ranges[p2_tail - 1] - fs * width_wiggle_threshold:
                    remain_list.append((p1_start, p1))
                    p2 = p2_tail
                    p2_tail = p2
                    while p2_tail < len(current_ranges):
                        p2_tail += 1
                        if (p2_tail >= len(current_ranges) or
                                current_ranges[p2_tail] > current_ranges[p2_tail - 1] + 1):
                            # End of group
                            break
                # Range1 cannot cover range2, skipping range1
                else:
                    continue
            else:
                p2 = p2_tail
                p2_tail = p2
                while p2_tail < len(current_ranges):
                    p2_tail += 1
                    if (p2_tail >= len(current_ranges) or
                            current_ranges[p2_tail] > current_ranges[p2_tail - 1] + 1):
                        # End of group
                        break

        # Merge with qrs ranges, remove ranges in qrs
        pos_ranges = list()
        for start,end in remain_list:
            pos_ranges.append((wave_ranges[start], wave_ranges[end - 1]))
        wave_ranges = pos_ranges
        wave_ranges = removeRangesInQRS(wave_ranges, qrs_ranges)
        
        merge_ranges = list()
        for start,end in wave_ranges:
            merge_ranges.extend(xrange(start,end))
        wave_ranges = merge_ranges
    return wave_ranges
                


                    
            
            
        
        
    
def swt_show(record_ID = '1269'):
    
    y = loadChanggeng(record_ID)
    raw_sig = y[:]
    original_ecg = raw_sig[:]
    annots = loadChanggengAnnots(record_ID)
    y = removeQRS(y, annots)

    coef, freqs=pywt.cwt(y,np.arange(1,32),'mexh')

    coef_shape = coef.shape
    # Get P magnify ranges
    P_point_list = list()
    thres = 0.2
    for ind in xrange(0, len(y)):
        if coef[-1, ind] > thres:
            P_point_list.append(ind)



    x_range = (1000, 1640)
    # y = y[x_range[0]:x_range[1]]
    # coef = coef[x_range[0]:x_range[1]]

    # fig, ax = plt.subplots(2,1)

    # amplist = [y[x] for x in P_point_list]
    # ax[0].plot(y)
    # ax[0].plot(P_point_list, amplist, 'ro')

    # amplify(coef, y, ax, level = 20, color = (0.1, 0.2,0.3))
    # amplify(coef, y, ax, level = 10, color = (0.9, 0.3,0.8))
    # # amplify
    # for pos in P_point_list:
        # y[pos] *= 10.0
        # raw_sig[pos] *= 5.0
    # ax[0].plot(y, 'y', lw = 4, alpha = 0.3)



    # ax[0].set_xlim(x_range)
    # ax[1].matshow(coef, cmap = plt.gray()) 
    # # ax[1].set_clip_box(((0,0),(9,19)))
    # ax[1].set_xlim(x_range)
    # plt.legend(numpoints = 1)


    plt.figure(2)
    # plt.plot(raw_sig, 'k', lw = 2, alpha = 1)
    poslist = getWaveRange(coef, y)
    amplist = [original_ecg[x] for x in poslist]
    plt.plot(original_ecg, 'b', lw = 2, alpha = 1)
    plt.plot(poslist, amplist, 'ro', markersize = 12, alpha = 0.5)
    

    plt.title(record_ID)
    plt.show() 


def doCMT(raw_sig, annots, figure_title = 'ecg'):
    '''Processing ecg with CWT Multiscale Thresholding method.'''
    y = raw_sig[:]
    original_ecg = raw_sig[:]

    y = removeQRS(y, annots)
    qrs_ranges = getQRSRanges(annots)
    coef, freqs=pywt.cwt(y,np.arange(1, 102),'mexh')

    coef_shape = coef.shape
    # Get P magnify ranges
    P_point_list = list()
    thres = 0.2
    for ind in xrange(0, len(y)):
        if coef[-1, ind] > thres:
            P_point_list.append(ind)



    # x_range = (1000, 1640)

    # y = y[x_range[0]:x_range[1]]

    # coef = coef[:, x_range[0]:x_range[1]]

    fig, ax = plt.subplots(3,1)

    # amplist = [y[x] for x in P_point_list]
    # ax[0].plot(y)
    # ax[0].plot(P_point_list, amplist, 'ro')

    # amplify(coef, y, ax, level = 20, color = (0.1, 0.2,0.3))
    # amplify(coef, y, ax, level = 10, color = (0.9, 0.3,0.8))
    bar_height = 10
    for cwt_level in [15, 10, 8]:
        poslist = getCwtRange(coef, y, cwt_level)
        p1 = 0
        while p1 < len(poslist):
            p1_start = p1
            while p1 < len(poslist):
                p1 += 1
                if p1 >= len(poslist) or poslist[p1] > poslist[p1 - 1] + 1:
                    # End of group
                    break
            width = 1 + poslist[p1 - 1] - poslist[p1_start]
            left = poslist[p1_start]
            ax[0].bar(left, bar_height, width = width, alpha = 0.4)
        bar_height += 10
    # amplify
    # for pos in P_point_list:
        # y[pos] *= 10.0
        # raw_sig[pos] *= 5.0
    ax[0].plot(y, 'k', lw = 4, alpha = 0.3)


    # ax[0].set_xlim(x_range)
    ax[1].matshow(coef, cmap = plt.gray()) 
    bw_image = coef[:,:]
    bw_image[bw_image > 0.2] = 1.0
    bw_image[bw_image <= 0.2] = 0.0
    ax[2].matshow(bw_image, cmap = plt.gray()) 

    # ax[1].set_clip_box(((0,0),(9,19)))
    # ax[1].set_xlim(x_range)
    # plt.legend(numpoints = 1)


    plt.figure(2)
    # plt.plot(raw_sig, 'k', lw = 2, alpha = 1)
    poslist = getWaveRange(coef, y, qrs_ranges = qrs_ranges)
    amplist = [original_ecg[x] for x in poslist]
    plt.plot(original_ecg, 'b', lw = 2, alpha = 1)
    plt.plot(poslist, amplist, 'ro', markersize = 12, alpha = 0.5)
    

    # plt.bar(823, 50, width = 40, color = 'y', alpha = 0.3)
    plt.title(figure_title)
    plt.show() 


def viewCWTsignal(raw_sig, fs, figure_title = 'ecg'):
    '''Processing ecg raw_sig.'''

    # Testing
    from randomwalk.test_api import GetModels
    from randomwalk.test_api import Testing

    pattern_filename = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/db2WT/random_pattern.json'
    model_folder = '/home/alex/LabGit/ECG_random_walk/randomwalk/data/db2WT'
    models = GetModels(model_folder, pattern_filename)
    annots = Testing(raw_sig, fs, models)

    doCMT(raw_sig, annots, figure_title = figure_title)



def viewQT():
    qt = QTloader()
    record_list = qt.getreclist()
    index = 0
    for record_name in record_list[:]:
        print 'record index:', index
        sig = qt.load(record_name)
        raw_sig = sig['sig'][2000:7000]
        viewCWTsignal(raw_sig, 250, figure_title = record_name)

        index += 1

def viewChanggeng():
    import glob
    
    record_files = glob.glob('/home/alex/LabGit/ECG_random_walk/experiments/PwavePlot/changgeng/*.json')
    for record_file_path in record_files:
        record_file_name = os.path.split(record_file_path)[-1]
        record_ID = record_file_name.split('.')[0]
        print 'Record ID:', record_ID
        swt_show(record_ID = record_ID)
    
if __name__ == '__main__':
    viewQT()

