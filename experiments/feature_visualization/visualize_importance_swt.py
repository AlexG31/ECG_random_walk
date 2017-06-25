#encoding:utf-8
"""
Feature Importance Visualization
Author : Gaopengfei
"""
import os
import sys
import json
import math
import pickle
import random
import time
import pywt
import pdb
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool

import numpy as np
## machine learning methods
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# project homepath
# 
curfilepath =  os.path.realpath(__file__)
curfolderpath = os.path.dirname(curfilepath)
projhomepath = curfolderpath
projhomepath = os.path.dirname(projhomepath)
projhomepath = os.path.dirname(projhomepath)

# configure file
# conf is a dict containing keys
with open(os.path.join(curfolderpath, 'ECGconf.json'), 'r') as fin:
    conf = json.load(fin)
sys.path.append(projhomepath)
#
# my project components
from QTdata.loadQTdata import QTloader 
import joblib

## Main Scripts
# ==========
EPS = 1e-6



class FeatureVisualizationSwt:
    def __init__(self, rfmdl, RandomPairList_Path):
        # get random relations
        randrel_path = RandomPairList_Path
        with open(randrel_path, 'r') as fin:
            self.randrels = json.load(fin)

        self.rfmdl = rfmdl
        self.trees = rfmdl.estimators_
        self.qt = QTloader()
        self.qt_reclist = self.qt.getQTrecnamelist()
    def info(self):
        t0 = self.trees[0]
        # list dir
        for attr in dir(t0):
            print attr
        print 'tree-------------------------'
        for attr in dir(t0.tree_):
            print attr

        pdb.set_trace()
    def GetImportancePairs(self):
        '''
        Zipping pairs from random pairs file & importance values from classification model.
        Get a struct of importance:
            [#layer0:[((23, 2), 0.2# importance),
                ((pair), #importance), ()...],
             #layer1:[], ..]
        
        return:
             pairs & their importance
        '''
        # random relations 
        randrels = self.randrels
        importance_list = self.rfmdl.feature_importances_
        sum_rr = 0
        for rr in randrels:
            sum_rr += len(rr)
        print 'len(feature importance) = {}, len(random relations) = {}'.format(
                len(importance_list), sum_rr)

        print 'len(importance_list) = ', len(importance_list)
        print 'sum_rr * 2 = ', sum_rr * 2
        if len(importance_list) != 2*sum_rr:
            raise Exception('Length of important features is not equal '
                    'to the total number of pairs!')

        layer_index_start, layer_index_end = 0, 0
        relation_importance = []
        for rel_layer in randrels:
            layer_size = len(rel_layer)
            layer_index_end = layer_index_start + 2 * layer_size

            pair_diff_importance_list = importance_list[layer_index_start:layer_index_end:2]
            abs_diff_importance_list = importance_list[layer_index_start+1:layer_index_end:2]
            # Choose the maximum value from the importance of pair difference
            # and absolute pair difference.
            imp_arr = map(lambda x:max(x[0], x[1]),
                    zip(pair_diff_importance_list,
                abs_diff_importance_list))
            relation_importance.append(zip(rel_layer, imp_arr))
            layer_index_start += 2 * layer_size

        return relation_importance

    def get_min_Importance_threshold(self,relation_importance,top_importance_ratio = 9.5/10):
        '''Select top p% importance value.'''
        if relation_importance == None or len(relation_importance) == 0:
            raise Exception('relation_importance is empty!')
        imp_list = []
        for layer in relation_importance:
            # unzip the lists
            pairs,imps = zip(*layer)
            imp_list.extend(imps)
        # get the min_importance threshold
        N = len(imp_list)
        midpos = int(top_importance_ratio*float(N))
        imp_list.sort()
        return imp_list[midpos]

    def prettify_plot(self,
            rawsig,
            relation_importance,
            Window_Left = 1200,
            savefigname = None,
            figureID = 1,
            figsize = (8,6),
            figtitle = 'ECG Sample',
            showFigure = True,
            wavelet = 'db6',
            swt_level = 6,
            window_length_limit = 350):
        '''
        Prettify plot of RSWT pairs.
        '''
        import feature_prettify
        ## =========================================
        # 展示RSWT示意图
        # Plot Arrow
        #   with alpha value: png, pdf file
        ## =========================================
        N_subplot = 7
        # Importance pairs lower than this threshold is not shown in the figure.
        Thres_min_importance = self.get_min_Importance_threshold(
                relation_importance,
                top_importance_ratio = 0.97)
        fs = conf['fs']
        FixedWindowLen = conf['winlen_ratio_to_fs'] * fs

        xL = Window_Left
        xR = xL+FixedWindowLen
        tarpos  = (xL + xR) / 2
        # -----------------
        # Get median
        # -----------------
        importance_arr = []
        for rel_layer in relation_importance:
            importance_arr.extend([x[1] for x in rel_layer])
        N_imp = len(importance_arr)
        # Ascending order
        importance_arr.sort()
        IMP_MAX = importance_arr[-1]


        # Get SWT detail coefficient lists
        rawsig = self.crop_data_for_swt(rawsig)
        coeflist = pywt.swt(rawsig, wavelet, swt_level)
        cAlist, cDlist = zip(*coeflist)
        self.cAlist = cAlist[::-1]
        self.cDlist = cDlist[::-1]

        pltxLim = range(xL,xR)
        sigAmp = [rawsig[x] for x in pltxLim]

        detail_index = 1
        # ====================
        # Plot raw signal input
        # ====================
        # plot raw ECG
        fig_handle, ax_list = plt.subplots(N_subplot / 2,2)
        fig_handle.set_figwidth(figsize[0])
        fig_handle.set_figheight(figsize[1])
        # Get handle for annote arrow
        ax_current = ax_list[0,0]
        
        ax_current.plot(sigAmp, 'black')
        # plot reference point
        ax_current.plot(tarpos,rawsig[tarpos],'ro')
        ax_current.set_title(figtitle)

        window_center = len(sigAmp) / 2
        window_limit_left = int(window_center - window_length_limit / 2)
        window_limit_right = int(window_center + window_length_limit / 2)
        ax_current.set_xlim(0, len(sigAmp))

        ax_current.grid(True)
        # plt.figure(1)

        for i in range(2, N_subplot):
            # Relation&importance
            ax_current = ax_list[(i - 1) / 2, (i - 1) % 2]
            rel_layer = relation_importance[detail_index - 1]
            cD = self.cDlist[detail_index]
            detail_index += 1

            # Crop x range out
            xi = range(xL,xR)
            cDamp = [cD[x] for x in xi]

            # get relation points
            rel_x = []
            rel_y = []
            pairs = list()
            pairs_importance = list()
            cur_N = len(cDamp)

            window_center = len(cDamp) / 2
            window_limit_left = int(window_center - window_length_limit / 2)
            window_limit_right = int(window_center + window_length_limit / 2)
            # ax_current.set_xlim(window_limit_left, window_limit_right)

            # ------------
            # sub plot 
            # ------------
            ax_current.set_title('SWT Detail Coefficient {}'.format(detail_index))

            # Sort rel_layer by imp
            rel_layer.sort(key = lambda x:x[1])
            for rel_pair, imp in rel_layer:
                # Do not show imp lower than threshold
                if imp < Thres_min_importance:
                    continue 
                # Importance thres
                alpha = (imp-Thres_min_importance)/(IMP_MAX-Thres_min_importance)
                # Increase alpha for better visual effect.
                alpha_increase_ratio = 0.8
                alpha = alpha * alpha_increase_ratio + 1.0 - alpha_increase_ratio
                # alpha = 1.0

                pairs_importance.append(alpha)
                arrow_color = (1,0,0)
                arrowprops = dict(alpha = alpha,
                        width = 0.15,
                        linewidth = 0.15,
                        headwidth = 1.5,
                        headlength = 1.5,
                        facecolor=arrow_color,
                        edgecolor = arrow_color,
                        shrink=0)

                rel_x.append(rel_pair[0])
                if rel_x[-1] >= cur_N:
                    rel_y.append(0)
                else:
                    rel_y.append(cDamp[rel_pair[0]])
                rel_x.append(rel_pair[1])
                if rel_x[-1] >= cur_N:
                    rel_y.append(0)
                else:
                    rel_y.append(cDamp[rel_x[-1]])
                pairs.append(
                        (
                            (rel_x[-2] - window_limit_left, rel_y[-2]),
                            (rel_x[-1] - window_limit_left, rel_y[-1])
                            )
                        )

            
            # print 'Window_limit_left:', window_limit_left
            # for x_list_ind in xrange(0, len(rel_x), 2):
                # print 'point1:', rel_x[x_list_ind], 'point2:', rel_x[x_list_ind + 1]
            # for p1, p2 in pairs:
                # print 'point1:', p1, 'point2:', p2
            # pdb.set_trace()

            window_limit_right = min(len(cDamp), window_limit_right)
            feature_prettify.prettify_stripe(
                    ax_list[(i - 1) / 2, (i - 1) % 2],
                    cDamp,
                    pairs,
                    pairs_importance,
                    tarpos - xL,
                    )

        # plot result
        if showFigure == True:
            plt.show()
        # save fig
        if savefigname is not None:
            fig_handle.savefig(savefigname, dpi = fig_handle.dpi)
            fig_handle.clf()


    def PlotApproximationLevel(self, importance_list):
        '''Plot Approximation level importance.'''
        # plot Approximation Level
        rel_x = []
        rel_y = []

        rel_layer = importance_list
        fig = plt.subplot(4,2,8)
        plt.title('Approximation Coefficient')
        cAamp = [self.cAlist[-1][x] for x in xi]

        # sort rel_layer with imp
        rel_layer.sort(key = lambda x:x[1])
        for rel_pair, imp in rel_layer:
            # Hide importance lower than threshold
            if imp<Thres_min_importance:
                continue
            # importance thres
            alpha = (imp-Thres_min_importance) / (IMP_MAX-Thres_min_importance)
            #arrow_color = self.get_RGB_from_Alpha((1,0,0),alpha,(1,1,1))
            arrow_color = (1,0,0)
            #arrowprops = dict(width = 1, headwidth = 4, facecolor=arrow_color, edgecolor = arrow_color, shrink=0)
            arrowprops = dict(alpha = alpha,
                    width = 0.15,
                    linewidth = 0.15,
                    headwidth = 1.5,
                    headlength = 1.5,
                    facecolor=arrow_color,
                    edgecolor = arrow_color,
                    shrink=0)

            rel_x.append(rel_pair[0])
            if rel_x[-1] >= cur_N:
                rel_y.append(0)
            else:
                rel_y.append(cAamp[rel_pair[0]])
            rel_x.append(rel_pair[1])
            if rel_x[-1] >= cur_N:
                rel_y.append(0)
            else:
                rel_y.append(cAamp[rel_x[-1]])
            fig.annotate('', xy=(rel_x[-2],rel_y[-2]),
                    xytext=(rel_x[-1],rel_y[-1]),arrowprops=arrowprops)

        # reference point
        plt.plot(rel_x,rel_y,'.b')
        plt.plot(cAamp)
        plt.xlim(0,len(cAamp)-1)

    def crop_data_for_swt(self,rawsig):
        '''Padding zeros to make the length of the signal to 2^k.'''
        # crop rawsig
        base2 = 1
        N_data = len(rawsig)
        if len(rawsig)<=1:
            raise Exception('len(rawsig)={},not enough for swt!',len(rawsig))
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

    def get_RGB_from_Alpha(self,color,alpha,bgcolor):
        new_color = []
        for color_elem,bg_color_elem in zip(color,bgcolor):
            color_elem = float(color_elem)
            bg_color_elem = float(bg_color_elem)
            ncolor = (1.0-alpha)*bg_color_elem+alpha*color_elem
            new_color.append(ncolor)
        return new_color

    def plot_fv_importance(self):
        # cycling plot importance of list of positions
        rID = 2
        sig = self.qt.load(self.qt_reclist[rID])
        rel_imp = self.GetImportancePairs()
        # save current fig
        for i in xrange(0,130,10):
            savefigname = os.path.join(curfolderpath,'range_{}.png'.format(i))
            self.plot_dwt_pairs_arrow(sig['sig'],rel_imp,Window_Left = 1180+i,savefigname = savefigname,figsize = (16,12),figtitle = 'Window Start[{}]'.format(i))
    def plot_fv_importance_gswt(self, savefigname, showFigure, WindowLeftBias = 10):
        # QT record ID
        fs = conf['fs']
        rID = 2
        # sig = self.qt.load(self.qt_reclist[rID])
        sig = self.qt.load('sel103')

        rel_imp = self.GetImportancePairs()
        # Save current fig
        figtitle = 'ECG from QTdb Record {}'.format(self.qt_reclist[rID])
        window_length_limit = int(conf['fs'] * conf['winlen_ratio_to_fs'])
        self.prettify_plot(
                sig['sig'],
                rel_imp,
                Window_Left = 54100+WindowLeftBias,
                savefigname = savefigname,
                figsize = (14,14),
                figtitle = figtitle,
                showFigure = showFigure,
                window_length_limit = window_length_limit)


def VisualizeModel(target_label = 'T', WindowLeftBias = 140, show_figure = False):
    '''Visualize model.'''
    # target_label = 'P'
    mdlfilename = os.path.join(curfolderpath,
            'trained_model',
            target_label,
            'trained_model.mdl')
    rand_relation_path = os.path.join(curfolderpath,
            'trained_model',
            'random_pattern.json')

    with open(mdlfilename, 'rb') as fin:
        rfmdl = joblib.load(fin)
    fvis = FeatureVisualizationSwt(rfmdl, rand_relation_path)

    savefigname = os.path.join(curfolderpath, 'visualize_{}.png'.format(target_label))
    fvis.plot_fv_importance_gswt(
            savefigname, showFigure = show_figure, WindowLeftBias = WindowLeftBias)



if __name__ == '__main__':
    show_figure = False
    VisualizeModel('T', 50, show_figure = show_figure)
    VisualizeModel('P', 50, show_figure = show_figure)
    VisualizeModel('R', 50, show_figure = show_figure)
