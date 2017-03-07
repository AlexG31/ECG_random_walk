#encoding:utf-8
"""
Author : Gaopengfei
Date: 2017.3
OutputFormat:
    dict(
        key = (ID, database)
        value = list(Tonset positions)
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
curfolderpath = os.path.dirname(curfilepath)
# my project components
from QTdata.loadQTdata import QTloader



debugmod = True

class whiteSamplePicker:
    def __init__(self):

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
        browser = PointBrowser(fig, ax, start_index)

        fig.canvas.mpl_connect('pick_event', browser.onpick)
        fig.canvas.mpl_connect('key_press_event', browser.onpress)

        plt.show()

class PointBrowser(object):
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """


    def __init__(self, fig, ax, start_index):
        self.fig = fig
        self.ax = ax
        self.SaveFolder = os.path.join(curfolderpath, 'results')


        self.text = self.ax.text(0.05, 0.95, 'selected: none',
                            transform=self.ax.transAxes, va='top')
        # ============================
        # QTdb
        self.QTdb = QTloader()
        self.reclist = self.QTdb.reclist
        self.recInd = start_index
        self.recname = self.reclist[self.recInd]
        self.sigStruct = self.QTdb.load(self.recname)
        self.rawSig = self.sigStruct['sig']
        self.expLabels = self.QTdb.getexpertlabeltuple(self.recname)

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
        with open(os.path.join(self.SaveFolder,'{}_poslist.json'.format(self.recname)),'w') as fout:
            result_info = dict(
                    ID = self.recname,
                    database = 'QTdb',
                    poslist = self.poslist,
                    type = 'Tonset')
            json.dump(result_info, fout, indent = 4, sort_keys = True)
            print 'Json file for record {} saved.'.format(self.recname)

    def clearWhiteMarkList(self):
        self.poslist = []
        self.totalWhiteCount = 0
        

    def addMarkx(self,x):
        # mark data
        pos = int(x)
        self.poslist.append(pos)

        self.ax.plot(pos, self.rawSig[pos],
                marker = 'x',
                color = self.colors[7],
                markersize = 22,
                markeredgewidth = 4,
                alpha = 0.9,
                label = 'Tonset')
        self.ax.set_xlim(pos - 500, pos + 500)

        
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
        result_file_name = os.path.join(self.SaveFolder,'{}_poslist.json'.format(self.recname))
        if os.path.exists(result_file_name):
            window = Tkinter.Tk()
            window.wm_withdraw()
            tkMessageBox.showinfo(title = 'Repeat', message = 'The record %s is already marked!' % self.recname)
            window.destroy()

            # Go to next record
            self.next_record()
            self.clearWhiteMarkList()
            self.reDraw()
        
        
    def reDraw(self):

        self.RepeatCheck()

        ax = self.ax
        ax.cla()

        self.text = self.ax.text(0.05, 0.95, 'selected: none',
                            transform=self.ax.transAxes, va='top')
        ax.grid(color=(0.8,0.8,0.8), linestyle='--', linewidth=2)

        # ====================================
        # load ECG signal

        ax.set_title('QT {} (Index = {})'.format(self.recname,self.recInd))
        ax.plot(self.rawSig, picker=5)  # 5 points tolerance
        # plot Expert Labels
        self.plotExpertLabels(ax)

        # draw Markers
        for pos in self.poslist:
            # draw markers
            self.ax.plot(pos, self.rawSig[pos],
                    marker = 'x',
                    color = self.colors[7],
                    markersize = 22,
                    markeredgewidth = 4,
                    alpha = 0.9,
                    label = 'Tonset')

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
        if self.recInd >= len(self.reclist):
            return False
        self.recname = self.reclist[self.recInd]
        self.sigStruct = self.QTdb.load(self.recname)
        self.rawSig = self.sigStruct['sig']
        self.expLabels = self.QTdb.getexpertlabeltuple(self.recname)
        return True

    def plotExpertLabels(self,ax):

        #get label Dict
        labelSet = set()
        labelDict = dict()
        for pos,label in self.expLabels:
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

if __name__ == '__main__':
    tool = whiteSamplePicker()
    tool.show(start_index = 0)
    pass
