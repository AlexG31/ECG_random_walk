#encoding:utf8
import os, sys, pdb, pywt
import feature_extractor


class NQRSfeatures(feature_extractor.ECGfeatures):
    '''RSWT feature extractor without QRS.
    NOTE: According to experiment, P wave will be spreaded in
            SWT levels (2,3,4,5,6).
    '''
    def __init__(self,
                rawsig,
                annots,
                configuration_info,
                wavelet='db2',
                ):
        super(NQRSfeatures, self).__init__(rawsig,
                configuration_info,
                wavelet = wavelet)

        # Get ecg signal without QRS: self.nQRS_ecg
        self.annots = annots
        self.QRS_ranges = self.getQRSRanges(annots)
        self.nQRS_ecg = self.removeRanges(rawsig, self.QRS_ranges)

        # Stationary wavelet transform
        WT_level = configuration_info['WT_LEVEL']
        self.nQRS_ecg = self.crop_data_for_swt(self.nQRS_ecg)
        coeflist = pywt.swt(self.nQRS_ecg, wavelet, WT_level)
        (cAlist, cDlist) = zip(*coeflist)
        self.nQRS_cAlist = cAlist[::-1]
        self.nQRS_cDlist = cDlist[::-1]

    def frompos(self, pos):
        '''Main function to extract feature from position.'''
        # Paramter Validation
        pos = int(pos)
        if pos < 0 or pos >= len(self.signal_in):
            raise StandardError(
                    'Input position posx must in range of sig indexs!'
                    )
        # RSWT feature
        feature_vector = super(NQRSfeatures, self).frompos(pos)
        # No-QRS feature
        windowed_matrix = self.GetWindowedMatrix(pos,
                cDlist = self.nQRS_cDlist,
                cAlist = self.nQRS_cAlist,
                )

        # Normalization
        windowed_ecg = windowed_matrix[0]
        Ampmax = max(windowed_ecg)
        Ampmin = min(windowed_ecg)
        sig_height = float(Ampmax - Ampmin)
        if sig_height <= 1e-6:
            sig_height = 1
        windowed_matrix = [[(val - Ampmin) / sig_height for val in
                           signal] for signal in windowed_matrix]


        wt_pair_list = self.getRandomPatterns()

        for (signal, pair_list) in zip(windowed_matrix, wt_pair_list):
            fv = [signal[x[0]] - signal[x[1]] for x in pair_list]
            feature_vector.extend(fv)
            fv = [abs(signal[x[0]] - signal[x[1]]) for x in pair_list]
            feature_vector.extend(fv)

        return feature_vector
        
    
    def removeRanges(self, raw_sig, QRS_ranges):
        '''Remove QRS(or T) regions in signal.'''

        data = raw_sig[:]
        # Remove QRS
        for qrs_on, qrs_off in QRS_ranges:
            qrs_on = int(qrs_on)
            qrs_off = int(qrs_off)
            for ind in xrange(qrs_on, qrs_off):
                data[ind] = raw_sig[qrs_on] + (raw_sig[qrs_off] - raw_sig[qrs_on]) * (ind - qrs_on) / (qrs_off - qrs_on)

        return data

    def getQRSRanges(self, annots):
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
