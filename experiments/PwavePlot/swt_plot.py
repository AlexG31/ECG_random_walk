#encoding:utf8
import os, sys, pywt, pdb
import matplotlib.pyplot as plt
import numpy as np


def crop_data_for_swt(rawsig):
    '''Padding zeros to make the length of the signal to 2^N.'''
    # crop rawsig
    base2 = 1
    N_data = len(rawsig)
    if len(rawsig)<=1:
        raise Exception('len(rawsig)={}, not enough for swt!', len(rawsig))
    crop_len = base2
    while base2<N_data:
        if base2*2>=N_data:
            crop_len = base2*2
            break
        base2 *=2
    # Extending this signal input with its tail element.
    if N_data < crop_len:
        rawsig += [rawsig[-1],]*(crop_len-N_data)
    return rawsig

def Pwave(record_ID = '42736'):
    '''Plot P wave swt coefs.'''
    import json
    import scipy.signal
    signal_file = './changgeng/%s.json' % str(record_ID)
    with open(signal_file, 'r') as fin:
        data = json.load(fin)
        data = crop_data_for_swt(data)
        data = scipy.signal.resample(data, len(data) / 2)

    
    wavelet = 'db2'
    coefs = pywt.swt(data, wavelet, level = 8)
    coefs = coefs[::-1]

    # plt.figure(1)
    # plt.plot(data)
    # plt.title('ECG')

    fig, ax = plt.subplots(4,2)
    ind = 0
    for ax_i in xrange(0, 4):
        for ax_j in xrange(0, 2):
            ax[ax_i, ax_j].plot(coefs[ind][1], lw = 2)
            ax[ax_i, ax_j].plot(data, 'r', alpha = 0.4)
            ax[ax_i,ax_j].set_title('swt level %d' % (ind + 1))
            ind += 1
    plt.suptitle(wavelet, fontsize = 22)
    plt.savefig('%s.png' % wavelet)
    plt.show()

def removeQRS(raw_sig, annots):
    '''Remove QRS regions in signal.'''
    QRS_ranges = getQRSRanges(annots)

    data = raw_sig[:]
    # Remove QRS
    for qrs_on, qrs_off in QRS_ranges:
        qrs_on = int(qrs_on)
        qrs_off = int(qrs_off)
        for ind in xrange(qrs_on, qrs_off):
            data[ind] = raw_sig[qrs_on] + (raw_sig[qrs_off] - raw_sig[qrs_on]) * (ind - qrs_on) / (qrs_off - qrs_on)

    return data

def removeRanges(raw_sig, QRS_ranges):
    '''Remove QRS(or T) regions in signal.'''

    data = raw_sig[:]
    # Remove QRS
    for qrs_on, qrs_off in QRS_ranges:
        qrs_on = int(qrs_on)
        qrs_off = int(qrs_off)
        for ind in xrange(qrs_on, qrs_off):
            data[ind] = raw_sig[qrs_on] + (raw_sig[qrs_off] - raw_sig[qrs_on]) * (ind - qrs_on) / (qrs_off - qrs_on)

    return data

def getQRSRanges(annots):
    '''Remove QRS regions in signal.'''
    QRS_ranges = list()

    annots.sort(key = lambda x:x[0], reverse = False)

    # 'Ronset' and 'Roffset' only
    annots = filter(lambda x:('R' == x[1][0] and len(x[1]) > 1), annots)

    for ind in xrange(0, len(annots) - 1):
        pos, label = annots[ind]
        if label == 'Ronset':
            if annots[ind + 1][1] == 'Roffset':
                QRS_ranges.append((pos, annots[ind + 1][0]))

    return QRS_ranges
    
def Pwave_noQRS(recID = 6655):
    '''Plot P wave swt coefs.'''
    recID = str(recID)
    import json
    import scipy.signal
    
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
        raw_sig = data[:]

        # Remove QRS
        for qrs_on, qrs_off in QRS_ranges:
            print '(%f, %f)' % (qrs_on, qrs_off)
            qrs_on = int(qrs_on)
            qrs_off = int(qrs_off)
            for ind in xrange(qrs_on, qrs_off):
                data[ind] = raw_sig[qrs_on] + (raw_sig[qrs_off] - raw_sig[qrs_on]) * (ind - qrs_on) / (qrs_off - qrs_on)

        data = crop_data_for_swt(data)
        # data = scipy.signal.resample(data, len(data) / 2)


    
    wavelet = 'db2'
    coefs = pywt.swt(data, wavelet, level = 8)
    coefs = coefs[::-1]

    # plt.figure(1)
    # plt.plot(data)
    # plt.title('ECG')

    fig, ax = plt.subplots(4,2)
    ind = 0
    for ax_i in xrange(0, 4):
        for ax_j in xrange(0, 2):
            ax[ax_i, ax_j].plot(coefs[ind][1], lw = 2)
            ax[ax_i, ax_j].plot(data, 'r', alpha = 0.4)
            ax[ax_i,ax_j].set_title('swt level %d' % (ind + 1))
            ind += 1
    plt.suptitle(wavelet, fontsize = 22)
    plt.savefig('%s.png' % wavelet)
    plt.show()

if __name__ == '__main__':
    Pwave_noQRS('42736')
    # Pwave()
