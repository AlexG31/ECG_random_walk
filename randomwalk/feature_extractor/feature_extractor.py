#encoding:utf-8
"""
This is ECG feature extractor class
Aiming to extract feature for training and classification

Author: Phil
"""

import os
import sys
import json
import math
import pdb
import numpy as np
import pywt
import array



class ECGfeatures:
    '''Collect ECG feature with SWT.

    1. Must initialise with raw ecg signal.
    2. then the frompos() function can be used to extract feature
        for position provided.
    '''
    def __init__(
                self,
                rawsig,
                configuration_info,
                wavelet = 'db2',
            ):
        '''
        Inputs:
            configuration_info:
                dictionary of feature extraction configurations.
        '''
        # Validation Check
        if (not isinstance(rawsig, list) and
                not isinstance(rawsig, array.array) and
                not isinstance(rawsig, np.ndarray)
        ):
            raise StandardError('Input rawsig is not a list type![WTdenoise]')
        if isinstance(rawsig, np.ndarray):
            rawsig = rawsig.tolist()

        # Default value of random relation files.
        # May denoise rawsig to get sig
        self.signal_in = rawsig
        self.rawsig = rawsig[:]
        self.config = configuration_info

        self.random_relation_path_ = self.config['random_pattern_path']
        self.fixed_window_length = self.config['fs'] * self.config['winlen_ratio_to_fs']

        # Do SWT once for all.
        wt_level = self.config['WT_LEVEL']
        self.rawsig = self.crop_data_for_swt(self.rawsig)
        coeflist = pywt.swt(self.rawsig, wavelet, wt_level)
        cAlist, cDlist = zip(*coeflist)
        self.cAlist = cAlist[::-1]
        self.cDlist = cDlist[::-1]
        
        
    def crop_data_for_swt(self, rawsig):
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
            base2*=2
        # Extending this signal input with its tail element.
        if N_data< crop_len:
            rawsig += [rawsig[-1],]*(crop_len-N_data)
        return rawsig

    def frompos(self,pos):
        '''WT feature Warpper.'''
        return self.getWTfeatureforpos(pos)

    
    def getWindowedSignal(self, x, sig, fixed_window_length):
        '''Get windowed signal segment centered in x.'''
        # Padding with head or tail element when out of bound.

        # padding zeros if near boundaries
        FixedWindowLen = fixed_window_length

        # Return windowed sig 
        segment = []
        left_bound = x - int(fixed_window_length / 2)
        right_bound = left_bound + fixed_window_length
        if right_bound <= 0:
            segment = [sig[0], ] * fixed_window_length
        elif left_bound >= len(sig):
            segment = [sig[-1], ] * fixed_window_length
        else:
            L = max(0, left_bound)
            R = min(len(sig), right_bound)
            if left_bound < 0:
                segment = [sig[0],] * (-left_bound)
            else:
                segment = []
            segment.extend(sig[L:R])
            if right_bound > len(sig):
                segment.extend([sig[-1],] * (right_bound - len(sig)))
            
                
        if len(segment) != FixedWindowLen:
            print 'Error: the windowed signal must be of equal length!'
            pdb.set_trace()

        return segment

    def GetWindowedMatrix(self, position):
        '''Windowing the rawsignal and SWT coefficients.'''
        fixed_window_length = self.fixed_window_length

        windowed_matrix = []
        
        # Adding time-domain windowed signal.
        # windowed_matrix.append(self.getWindowedSignal(position,
            # self.rawsig,
            # fixed_window_length))

        # Apply the window in each level of swt coefficients.
        for detail_coefficients in self.cDlist:
            windowed_matrix.append(self.getWindowedSignal(position,
                detail_coefficients,
                fixed_window_length))
        # Adding approximation level.
        # windowed_matrix.append(self.getWindowedSignal(position,
            # self.cAlist[0],
            # fixed_window_length))

        return windowed_matrix

    def debug_PlotWindowedMatrix(self,matrix,max_level = 4):
        '''Plot matrix in figure.'''
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.grid(True)
        axes_index = 1
        for signal in matrix:
            if axes_index > max_level:
                break
            plt.subplot(max_level,1, axes_index)
            axes_index += 1
            plt.plot(signal)
        plt.show()

        
    def getWTfeatureforpos(self, pos):
        '''Get WT feature from position in ECG time-domain waveform.'''
        pos = int(pos)
        if pos<0 or pos >= len(self.signal_in):
            raise StandardError('Input position posx must in range of sig indexs!')
        
        # Stateful... Apply window in each level of swt coefficients.
        windowed_matrix = self.GetWindowedMatrix(pos)

        # normalization
        windowed_ecg = windowed_matrix[0]
        Ampmax = max(windowed_ecg)
        Ampmin = min(windowed_ecg)
        sig_height = float(Ampmax-Ampmin)
        if sig_height <= 1e-6:
            sig_height = 1
        windowed_matrix = [[(val-Ampmin)/sig_height for val in signal]
                for signal in windowed_matrix]

        features = []

        with open(self.random_relation_path_, 'r') as fin:
            wt_pair_list = json.load(fin)

        for signal, pair_list in zip(windowed_matrix, wt_pair_list):
            fv = [signal[x[0]] - signal[x[1]] for x in pair_list]
            features.extend(fv)
            fv = [abs(signal[x[0]] - signal[x[1]]) for x in pair_list]
            features.extend(fv)
        
        return features


## for multiProcess
def frompos_with_denoisedsig(Params):
    # map function should have only 1 Param
    # ,so I combined them.
    #
    denoisedsig,pos = Params
    fvExtractor = ECGfeatures(denoisedsig,isdenoised = True)
    return fvExtractor.frompos(pos)


# debug
if __name__ == '__main__':
    from QTdata.loadQTdata import QTloader
    qt = QTloader()

    sigStruct = qt.load('sel100')
    # What's cAlist[-1]?
    feature_extractor = ECGfeatures(sigStruct['sig'])
    windowed_matrix = feature_extractor.GetWindowedMatrix(9000)
    # plot swt coefficients
    import matplotlib.pyplot as plt
    plt.figure(1)
    for ind in xrange(1,len(feature_extractor.cAlist)):
        plt.subplot(7,1,ind)
        plt.plot(feature_extractor.cAlist[ind])
        plt.title('App level %d' % ind)

    plt.show()



