#encoding:utf8
"""
Author : Gaopengfei
Date: 2017.4
OutputFormat:
    dict(
        key = (ID, database)
        value = list(label positions)
    )
"""
import os
import sys
import json
import math
import pickle
import random
import pywt
import time
import glob
import pdb
from multiprocessing import Pool

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib
import matplotlib.pyplot as plt
from numpy import pi, r_
from scipy import optimize
import Tkinter
import tkMessageBox

curfilepath =  os.path.realpath(__file__)
current_folderpath = os.path.dirname(curfilepath)
# my project components



debugmod = True

def getRecordList(target_label = 'P'):
    '''Return list of jsonIDs from local dir.'''
    jsonIDs = glob.glob(os.path.join(current_folderpath, 'path_info', target_label, '*.json'))
    jsonIDs = [os.path.split(x)[-1] for x in jsonIDs]


    # Skip records with annotations
    annot_jsonIDs = glob.glob(os.path.join(current_folderpath, 'labels', target_label, '*.json'))
    annot_jsonIDs = [os.path.split(x)[-1] for x in annot_jsonIDs]
    jsonIDs = list(set(jsonIDs) - set(annot_jsonIDs))

    return [x.split('.')[0] for x in jsonIDs]

# ECG data from changgeng
class ECGLoader(object):
    def __init__(self, fs, current_folderpath):
        '''Loader for changgeng data.'''
        self.fs = fs
        # self.P_faillist = [8999,8374,6659, 6655,6059,5395,1401,1269,737,75,9524,9476]
        self.P_faillist = getRecordList()

    def load(self, record_index):
        '''Return loaded signal info.'''
        return self.getSignal('%s.json' % self.P_faillist[record_index], 'II')

    def load_annotations(self, record_index, target_label = 'P'):
        '''Return auto-computed annotations.'''
        return self.getAnnot('%s.json' % self.P_faillist[record_index], target_label)

    def getSize(self, jsonfilename):
        '''Get fangchan record count.'''
        import json
        import codecs

        return len(self.P_faillist)

    def getAnnot(self, jsonfilename, target_label):
        '''Get annotations.'''
        import json
        import codecs
        import subprocess
        import scipy.io as sio

        matinfojson_filename = os.path.join(current_folderpath, 'path_info', 'P', '%s' % jsonfilename)
        with codecs.open(matinfojson_filename, 'r', 'utf8') as fin:
            data = json.load(fin)
            annot_list = data['II_label_pos'][target_label]
            return zip(annot_list, [target_label,] * len(annot_list))
        

    def getSignal(self, jsonfilename, leadname):
        '''Get fangchan signal.'''
        import json
        import codecs
        import subprocess
        import scipy.io as sio

        matinfojson_filename = os.path.join(current_folderpath, 'path_info', 'P', '%s' % jsonfilename)
        with codecs.open(matinfojson_filename, 'r', 'utf8') as fin:
            data = json.load(fin)

        # mat_rhythm is the data
        dlist = data
        matpath = dlist['mat_rhythm']
        diagnosis_text = dlist['diagnose']

        mat_file_name = os.path.split(matpath)[-1]
        save_mat_filepath = os.path.join(current_folderpath, 'data', mat_file_name)
        if (os.path.exists(save_mat_filepath) == False):
            subprocess.call(['scp', 'xinhe:%s' % matpath, save_mat_filepath])
        matdata = sio.loadmat(save_mat_filepath)
        raw_sig = np.squeeze(matdata[leadname])

        return [raw_sig, diagnosis_text, mat_file_name, dlist['II_label_pos']['P']]

class whiteSamplePicker:
    def __init__(self, target_label = 'P'):
        self.target_label = target_label
        tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
        self.colors = []
        for color_tri in tableau20:
            self.colors.append((color_tri[0]/255.0,color_tri[1]/255.0,color_tri[2]/255.0))



    def show(self, start_index = 0):
        fig = plt.figure(1)

        rect = 0.1,0.1,0.8,0.8
        ax = fig.add_axes(rect)
        ax.grid(color=(0.8,0.8,0.8), linestyle='--', linewidth=2)
        ax.set_title('Please Press space to refresh')
        browser = PointBrowser(fig, ax, start_index, self.target_label)

        fig.canvas.mpl_connect('pick_event', browser.onpick)
        fig.canvas.mpl_connect('key_press_event', browser.onpress)

        plt.show()

class PointBrowser(object):
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """


    def __init__(self, fig, ax, start_index, target_label):
        self.fig = fig
        self.ax = ax
        self.SaveFolder = os.path.join(current_folderpath, 'results')


        self.text = self.ax.text(0.05, 0.95, 'selected: none',
                            transform=self.ax.transAxes, va='top')
        # ============================
        self.ecgloader = ECGLoader(500, current_folderpath)
        self.recInd = start_index
        self.reloadData()
        self.target_label = target_label
        # self.expLabels = self.QTdb.getexpertlabeltuple(self.diag_text)

        tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
        self.colors = []
        for color_tri in tableau20:
            self.colors.append((color_tri[0]/255.0,color_tri[1]/255.0,color_tri[2]/255.0))
        # ===========================
        # Mark list
        self.poslist = []
        self.totalWhiteCount = 0

    def reloadData(self):
        '''Refresh data according to self.recInd'''
        raw_sig, diag_text, mat_file_name, target_poslist = self.ecgloader.load(self.recInd)
        self.diag_text = diag_text
        self.mat_file_name = mat_file_name
        self.rawSig = raw_sig

        # Get QRS locations
        from dpi.DPI_QRS_Detector import DPI_QRS_Detector as DPI
        dpi = DPI()
        print 'Testing QRS locations ...'
        results = dpi.QRS_Detection(raw_sig, fs = self.ecgloader.fs)
        self.expLabels = zip(results, ['R',] * len(results))
        self.expLabels.extend(self.ecgloader.load_annotations(self.recInd))

    def onpress(self, event):
        if event.key not in ('n', 'p',' ','x','a','d'):
            return

        if event.key == 'n':
            self.saveWhiteMarkList2Json()
            self.next_record()
            self.clearWhiteMarkList()

            # clear Marker List
            self.reDraw()
            return None
        elif event.key == ' ':
            self.reDraw()
            
            return None
        elif event.key == 'x':
            # Delete markers in stack
            if len(self.poslist) > 0:
                del self.poslist[-1]

        elif event.key == 'a':
            step = -200
            xlims = self.ax.get_xlim()
            new_xlims = [xlims[0]+step,xlims[1]+step]
            self.ax.set_xlim(new_xlims)
        elif event.key == 'd':
            step = 200
            xlims = self.ax.get_xlim()
            new_xlims = [xlims[0]+step,xlims[1]+step]
            self.ax.set_xlim(new_xlims)
        else:
            pass

        self.update()

    def saveWhiteMarkList2Json(self):
        import codecs
        changgengID = self.ecgloader.P_faillist[self.recInd]
        with codecs.open(os.path.join(current_folderpath, 'labels', self.target_label, '%s.json' % (changgengID)), 'w', 'utf8') as fout:
            result_info = dict(
                    mat_file_name = self.mat_file_name,
                    diag_text = self.diag_text,
                    database = 'changgeng',
                    poslist = self.poslist,
                    expertLabels = self.expLabels,
                    type = self.target_label)
            json.dump(result_info, fout, indent = 4, sort_keys = True, ensure_ascii = False)
            print 'Json file %s for record %s saved.' % (str(changgengID), self.mat_file_name)

    def clearWhiteMarkList(self):
        self.poslist = []
        self.totalWhiteCount = 0
        

    def addMarkx(self,x):
        # mark data
        pos = int(x)
        pos = min(pos, len(self.rawSig) - 1)
        self.poslist.append(pos)

        self.ax.plot(pos, self.rawSig[pos],
                marker = 'x',
                color = self.colors[7],
                markersize = 22,
                markeredgewidth = 4,
                alpha = 0.9,
                label = 'Tonset')
        self.ax.set_xlim(pos - 1000, pos + 1000)

        
    def onpick(self, event):
        '''Mouse click to mark target points.'''
        # The click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        # add white Mark
        self.addMarkx(x)
        self.text.set_text('Marking Tonset: ({}) [whiteCnt {}]'.format(self.poslist[-1],len(self.poslist)))

        # update canvas
        self.fig.canvas.draw()

    def RepeatCheck(self):
        '''Check repeat results.'''
        result_file_name = os.path.join(self.SaveFolder,'{}_poslist.json'.format(self.diag_text))
        if os.path.exists(result_file_name):
            window = Tkinter.Tk()
            window.wm_withdraw()
            tkMessageBox.showinfo(title = 'Repeat', message = 'The record %s is already marked!' % self.diag_text)
            window.destroy()

            # Go to next record
            self.next_record()
            self.clearWhiteMarkList()
            self.reDraw()
        
        
    def reDraw(self):

        # self.RepeatCheck()

        ax = self.ax
        ax.cla()

        self.text = self.ax.text(0.05, 0.95, 'selected: none',
                            transform=self.ax.transAxes, va='top')
        ax.grid(color=(0.8,0.8,0.8), linestyle='--', linewidth=2)

        # ====================================
        # load ECG signal

        ax.set_title(u'changgeng{} (Index = {}) target [{}]'.format(self.diag_text,self.recInd, self.target_label))
        ax.plot(self.rawSig, picker=5)  # 5 points tolerance
        # plot Expert Labels
        self.plotExpertLabels(ax)

        # draw Markers
        amplist = [self.rawSig[pos] for pos in self.poslist]
        # draw markers
        self.ax.plot(self.poslist, amplist,
                marker = 'x',
                color = self.colors[7],
                markersize = 22,
                markeredgewidth = 4,
                alpha = 0.9,
                label = self.target_label)

        self.ax.set_xlim(0, len(self.rawSig))
        # update draw
        self.fig.canvas.draw()

    def update(self):
        #self.ax2.text(0.05, 0.9, 'mu=%1.3f\nsigma=%1.3f' % (xs[dataind], ys[dataind]),
                 #transform=self.ax2.transAxes, va='top')
        #self.ax2.set_ylim(-0.5, 1.5)

        self.fig.canvas.draw()

    def next_record(self):
        self.recInd += 1
        if self.recInd >= self.ecgloader.getSize('fwave.json'):
            return False
        self.reloadData()
        return True

    def plotExpertLabels(self,ax):

        #get label Dict
        labelSet = set()
        labelDict = dict()
        for pos, label in self.expLabels:
            if label in labelSet:
                labelDict[label].append(pos)
            else:
                labelSet.add(label)
                labelDict[label] = [pos,]

        # plot to axes
        for label,posList in labelDict.iteritems():
            # plot marker for current label
            if label[0]=='T':
                color = self.colors[4]
            elif label[0]=='P':
                color  = self.colors[5]
            elif label[0]=='R':
                color  = self.colors[6]
            # marker
            if 'onset' in label:
                marker = '<'
            elif 'offset' in label:
                marker = '>'
            else:
                marker = 'o'
            ax.plot(posList,map(lambda x:self.rawSig[x],posList),marker = marker,color = color,linestyle = 'none',markersize = 14,label = label)
        ax.legend(numpoints = 1)




def get_QTdb_recordname(index = 1):
    QTdb = QTloader()
    reclist = QTdb.getQTrecnamelist()
    return reclist[index]


def TEST_loader():
    ecg = ECGLoader(500, current_folderpath)
    sig, diag_text, mat_filename, poslist = ecg.load(3)
    print diag_text, mat_filename 

    plt.plot(sig)
    amplist = [sig[int(x)] for x in poslist]
    plt.plot(poslist, amplist, 'go', label = 'P', markersize = 12)
    plt.grid(True)
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    tool = whiteSamplePicker(target_label = 'P')
    tool.show(start_index = 0)
    # TEST_loader()

