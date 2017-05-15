#encoding:utf8
import os, sys, pdb, json
import matplotlib.pyplot as plt

import gaussian_model as P_model_Gaussian
import numpy as np
import math
from pymc import MCMC
import scipy.signal as signal
from randomwalk.changgengLoader import ECGLoader as cLoader
from post_p import post_p

class PWave(object):
    def __init__(self, raw_sig, fs = 250.0, p_wave_lengthMs = 38 * 4):
        '''Delineator of P wave.'''
        self.raw_sig = raw_sig
        self.fs = fs

        # index of R wave
        result_dict = dict()
        # init
        result_dict['gamp'] = list()
        result_dict['gsigma'] = list()
        result_dict['gpos'] = list()
        result_dict['hermit_coefs'] = list()
        result_dict['segment_range'] = list()
        result_dict['peak_global_pos'] = list()

        self.result_dict = result_dict

    def detectGaussian(self,
            filtered_sig,
            annots,
            fs,
            step = 2,
            iter_count = 4000,
            burn_count = 2000,
            max_hermit_level = 4,
            ):
        '''Detect and returns P wave detection results.
            Note:
                This function returns None when detection fails.
        '''
        # r_list = self.r_list
        raw_sig = np.array(filtered_sig) * 10.0
        sig_seg = raw_sig
        

        p_model = P_model_Gaussian.MakeModel(sig_seg,
                annots,
                max_hermit_level = max_hermit_level)
        M = MCMC(p_model)

        M.sample(iter = iter_count, burn = burn_count, thin = 10)

        # retrieve parameters
        hermit_coefs = list()
        for h_ind in xrange(0, P_model_Gaussian.HermitFunction_max_level):
            hermit_value = np.mean(M.trace('hc%d' % h_ind)[:])
            hermit_coefs.append(hermit_value)

        fitting_curve = np.zeros(len(sig_seg),)
        for level, coef in zip(xrange(0, max_hermit_level), hermit_coefs):
            fitting_curve += P_model_Gaussian.HermitFunction(level, len(sig_seg)) * coef



        # Gaussian

        pos_ponset = None
        pos_p = None
        pos_poffset = None
        for pos, label in annots:
            if label == 'Ponset':
                pos_ponset = pos
            elif label == 'P':
                pos_p = pos
            elif label == 'Poffset':
                pos_poffset = pos

        gaussian_curve = np.zeros(len(sig_seg),)
        common_length = len(sig_seg)
        g_amp = np.mean(M.trace('g_amp')[:])
        g_sigma = np.mean(M.trace('g_sigma')[:])
        g_dc = np.mean(M.trace('dc')[:])
        gaussian_curve = P_model_Gaussian.GetGaussianPwave(len(sig_seg) * 2, g_amp, g_sigma / 3, g_dc)
        
        gaussian_segment = gaussian_curve[common_length - pos_p:2 * common_length - pos_p]
        baseline_curve = fitting_curve + g_dc
        baseline_curve = baseline_curve.tolist()
        fitting_curve += gaussian_segment
        

        plt.figure(1)
        plt.clf()
        plt.plot(sig_seg, label = 'ECG')
        plt.plot(fitting_curve, label = 'fitting curve')
        plt.plot(baseline_curve, 'r', alpha = 0.5, lw=5, label = 'Baseline curve')

        # Hermit coef vis
        plt.bar(xrange(0, len(hermit_coefs)),
                [0.12,] * len(hermit_coefs),
                width = 0.5,
                alpha = 0.3,
                color = 'grey')
        plt.bar(xrange(0, len(hermit_coefs)),
                [-0.12,] * len(hermit_coefs),
                width = 0.5,
                alpha = 0.3,
                color = 'grey')
        plt.bar(xrange(0, len(hermit_coefs)),
                np.array(hermit_coefs) * 0.2,
                width = 0.5,
                color = 'r')
        plt.legend()
        plt.grid(True)
        plt.show(block = False)
        # plt.savefig('./results/tmp/%d.png' % int(time.time()))

        results = dict()

        return results

    def detect(self,
            filtered_sig,
            fs,
            step = 2,
            iter_count = 4000,
            burn_count = 2000,
            ):
        '''Detect and returns P wave detection results.
            Note:
                This function returns None when detection fails.
        '''
        # r_list = self.r_list
        raw_sig = filtered_sig
        sig_seg = raw_sig
        max_hermit_level = 8

        p_model = P_model_Gaussian.MakeModel(sig_seg,
                max_hermit_level = max_hermit_level)
        M = MCMC(p_model)

        M.sample(iter = iter_count, burn = burn_count, thin = 10)

        # retrieve parameters
        hermit_coefs = list()
        for h_ind in xrange(0, P_model_Gaussian.HermitFunction_max_level):
            hermit_value = np.mean(M.trace('hc%d' % h_ind)[:])
            hermit_coefs.append(hermit_value)

        fitting_curve = np.zeros(len(sig_seg),)
        for level, coef in zip(xrange(0, max_hermit_level), hermit_coefs):
            fitting_curve += P_model_Gaussian.HermitFunction(level, len(sig_seg)) * coef


        plt.figure(1)
        plt.clf()
        plt.plot(sig_seg, label = 'ECG')
        plt.plot(fitting_curve, label = 'fitting curve')

        # Hermit coef vis
        plt.bar(xrange(0, len(hermit_coefs)),
                [0.12,] * len(hermit_coefs),
                width = 0.5,
                alpha = 0.3,
                color = 'grey')
        plt.bar(xrange(0, len(hermit_coefs)),
                [-0.12,] * len(hermit_coefs),
                width = 0.5,
                alpha = 0.3,
                color = 'grey')
        plt.bar(xrange(0, len(hermit_coefs)),
                np.array(hermit_coefs) * 0.2,
                width = 0.5,
                color = 'r')
        plt.legend()
        plt.grid(True)
        plt.show(block = False)
        # plt.savefig('./results/tmp/%d.png' % int(time.time()))

        results = dict()

        return results

    def run(self, debug_info = dict()):
        '''Run delineation process for each R-R interval.'''

        r_list = self.r_list
        result_dict = self.result_dict
        fs = self.fs
        raw_sig = self.raw_sig
        p_wave_length = self.p_wave_length
        r_detector = DPI_QRS_Detector()

        for ind in xrange(0, len(r_list) - 1):
            print 'Progress: %d R-R intervals left.' % (len(r_list) - 1 - ind)
            if ind > 1:
                print 'Debug break.'
                break
            region_left = r_list[ind]
            region_right = r_list[ind + 1]
            QR_length = fs / 46.0
            PR_length = 0.5 * (region_right - region_left)

            cut_x_list = [int(region_right - PR_length), int(region_right - QR_length)]

            sig_seg = raw_sig[cut_x_list[0]:cut_x_list[1]]
            sig_seg = r_detector.HPF(sig_seg, fs = fs, fc = 3.0)
            sig_seg = sig_seg[:cut_x_list[1] - cut_x_list[0]]
            if len(sig_seg) <= 75:
                print 'R-R interval %d is too short!' % len(sig_seg)
                continue


            p_model = P_model_Gaussian.MakeModel(sig_seg, p_wave_length, fs = fs)
            M = MCMC(p_model)

            M.sample(iter = 2000, burn = 1000, thin = 10)

            # retrieve parameters
            hermit_coefs = list()
            for h_ind in xrange(0, P_model_Gaussian.HermitFunction_max_level):
                hermit_value = np.mean(M.trace('hc%d' % h_ind)[:])
                hermit_coefs.append(hermit_value)
            # P wave shape parameters
            gpos = np.mean(M.trace('gaussian_start_position')[:])
            gsigma = np.mean(M.trace('gaussian_sigma')[:])
            gamp = np.mean(M.trace('gaussian_amplitude')[:])

            print 'Results:'
            print 'gpos = ', gpos
            print 'gsigma = ', gsigma
            print 'gamp = ', gamp
            
            len_sig = cut_x_list[1] - cut_x_list[0]
            fitting_curve = P_model_Gaussian.GetFittingCurve(len_sig,
                    gpos, gsigma,
                    gamp,
                    hermit_coefs)
            baseline_curve = P_model_Gaussian.GetFittingCurve(len_sig,
                    gpos, gsigma,
                    0,
                    hermit_coefs)
            plt.figure(1)
            plt.clf()
            plt.plot(sig_seg, label = 'ECG')
            plt.plot(fitting_curve, label = 'fitting curve')
            plt.plot(baseline_curve,
                    linestyle = '--',
                    alpha = 0.3, lw = 2,
                    label = 'baseline curve')
            plt.plot(gpos, fitting_curve[gpos], 'r^', markersize = 12)
            plt.legend()
            plt.grid(True)
            if 'plot_title' in debug_info:
                plt.title(debug_info['plot_title'], fontproperties = zn_font)
            if 'plot_xlabel' in debug_info:
                plt.xlabel(debug_info['plot_xlabel'])
            plt.show(block = False)
            plt.savefig('./results/tmp/%d.png' % int(time.time()))


            peak_pos = int(gpos + gsigma / 2.0)
            peak_global_pos = peak_pos + cut_x_list[0]


            # Save to result dict
            result_dict['gamp'].append(gamp)
            result_dict['gsigma'].append(gsigma)
            result_dict['gpos'].append(gpos)
            result_dict['hermit_coefs'].append(hermit_coefs)
            result_dict['segment_range'].append(cut_x_list)
            result_dict['peak_global_pos'].append(peak_global_pos)

            continue
        return result_dict

    def plot_results(self, raw_sig, result_dict, window_expand_size = 40):
        '''Visualize result diction.'''
        p_wave_length = self.p_wave_length
        fs = self.fs
        raw_sig = self.raw_sig
        r_detector = DPI_QRS_Detector()

        for ind in xrange(0, len(result_dict['gpos'])):
            cut_x_list = result_dict['segment_range'][ind]
            gamp = result_dict['gamp'][ind]
            gsigma = result_dict['gsigma'][ind]
            gpos = result_dict['gpos'][ind]
            hermit_coefs = result_dict['hermit_coefs'][ind]
            peak_global_pos = result_dict['peak_global_pos'][ind]

            len_sig = cut_x_list[1] - cut_x_list[0]
            sig_seg = raw_sig[cut_x_list[0]:cut_x_list[1]]
            sig_seg = r_detector.HPF(sig_seg, fs = fs, fc = 3.0)

            fitting_curve = P_model_Gaussian.GetFittingCurve(len_sig,
                    gpos, gsigma,
                    gamp,
                    hermit_coefs)
            baseline_curve = P_model_Gaussian.GetFittingCurve(len_sig,
                    gpos, gsigma,
                    0,
                    hermit_coefs)
            plt.figure(1)
            plt.clf()
            plt.plot(sig_seg, label = 'ECG')
            plt.plot(fitting_curve, label = 'fitting curve')
            plt.plot(baseline_curve,
                    linestyle = '--',
                    alpha = 0.3, lw = 2,
                    label = 'baseline curve')
            plt.plot(gpos, fitting_curve[gpos], 'r^', markersize = 12)
            # plt.plot(gpos + gsigma, fitting_curve[gpos + gsigma],
                    # 'r^', markersize = 12)
            plt.legend()
            plt.grid(True)
            plt.show(block = False)

            # Plot fitting curve
            plt.figure(1)
            plt.plot(raw_sig, label = 'ECG')
            plt.plot(peak_global_pos, raw_sig[peak_global_pos],
                    'ro', markersize = 12,label = 'P peak')
            plt.plot(xrange(cut_x_list[0], cut_x_list[1]),
                    fitting_curve,
                    linewidth = 2, color = 'orange', alpha = 1,
                    label = 'Fitting curve')
            plt.plot(xrange(cut_x_list[0], cut_x_list[1]),
                    baseline_curve,
                    linewidth = 3, color = 'black', alpha = 0.3,
                    label = 'Baseline')
            plt.xlim((cut_x_list[0] - window_expand_size,
                cut_x_list[1] + window_expand_size))
            plt.title('ECG %s (Peak %d)' % ('signal', ind))
            plt.legend()
            plt.grid(True)
            plt.show()

def plotExpertLabels(ax, raw_sig, annots):

    #get label Dict
    labelSet = set()
    labelDict = dict()
    for pos,label in annots:
        if label in labelSet:
            labelDict[label].append(pos)
        else:
            labelSet.add(label)
            labelDict[label] = [pos,]

    # plot to axes
    for label,posList in labelDict.iteritems():
        # plot marker for current label
        if label[0]=='T':
            color = 'm'
        elif label[0]=='P':
            color  = 'b'
        elif label[0]=='R':
            color  = 'r'
        elif label[0]=='Q':
            color = 'y'
        elif label[0]=='S':
            color = 'c'
        # marker
        if 'onset' in label:
            marker = '<'
        elif 'offset' in label:
            marker = '>'
        else:
            marker = 'o'
        ax.plot(posList,map(lambda x:raw_sig[int(x)],posList),marker = marker,color = color,linestyle = 'none',markersize = 8,label = label)
    ax.legend(numpoints = 1)

def enhance_demo():
    '''Find more accurate characteristic point position.'''
    loader = cLoader(2,1)
    recordID = '54722'
    sig = loader.loadID(recordID)
    with open('./data/tmpWT/%s.json' % recordID, 'r') as fin:
        annots = json.load(fin)
        annots = post_p(sig, annots, 500)
    

    x_range = (920, 1010)
    Pannots = map(lambda x: [x[0] - x_range[0], x[1]], filter(lambda x: x[0] >= x_range[0] and x[0] <= x_range[1], annots))
    pwave = PWave(sig, fs = 500.0)
    pwave.detectGaussian(sig[x_range[0]:x_range[1]], Pannots, 500.0)


    print 'Annotations:'
    print Pannots

    fig, ax = plt.subplots(1,1)
    plt.plot(sig)
    plotExpertLabels(ax, sig, annots)
    plt.show()

def enhance_test():
    '''Find more accurate characteristic point position.'''
    loader = cLoader(2,1)
    recordID = '54722'
    sig = loader.loadID(recordID)
    with open('./data/tmpWT/%s.json' % recordID, 'r') as fin:
        annots = json.load(fin)
        annots = post_p(sig, annots, 500)
        annots.sort(key = lambda x:x[0])
    
    x_range_list = list()
    x_range_start = None
    for ind in xrange(0, len(annots)):
        pos, label = annots[ind]
        if label == 'Ponset':
            x_range_start = pos - 5
        elif label == 'Poffset':
            if x_range_start is not None:
                x_range_list.append((x_range_start, pos + 5))
            x_range_start = None
            

    
    x_range = (920, 1010)
    for x_range in x_range_list:
        Pannots = map(lambda x: [x[0] - x_range[0], x[1]], filter(lambda x: x[0] >= x_range[0] and x[0] <= x_range[1], annots))
        pwave = PWave(sig, fs = 500.0)
        pwave.detectGaussian(sig[x_range[0]:x_range[1]], Pannots, 500.0)

        pdb.set_trace()


    print 'Annotations:'
    print Pannots

    fig, ax = plt.subplots(1,1)
    plt.plot(sig)
    plotExpertLabels(ax, sig, annots)
    plt.show()

if __name__ == '__main__':
    enhance_test()
