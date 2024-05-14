# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:14:14 2019

@author: USUARIO, Jonas Zaehringer

@Modified NBS Group MINFLUX: Jonas Zaehringer

status: translated; 

"""

import numpy as np
import bisect
import time
from datetime import date, datetime
import os
import matplotlib.pyplot as plt
import tools.tools as tools
from tkinter import Tk, filedialog
import tifffile as tiff
import scipy.optimize as opt

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from pyqtgraph.dockarea import Dock, DockArea

import ctypes as ct
from ctypes import byref, POINTER

import tools.viewbox_tools as viewbox_tools
import drivers.hydraharp as hydraharp
import HydraHarp.Read_PTU as Read_PTU
#import drivers.ADwin as ADwin
import scan

import queue

import tools.pyqtsubclass as pyqtsc
import tools.colormaps as cmaps

import qdarkstyle
import ctypes
import threading
import fast_histogram


import multiprocessing as mp
from multiprocessing import Pool

def convert_ht3(data_array):
    relTimelive,absTimelive, channel_array = Read_PTU.convertHT3(data_array)
    
    return relTimelive,absTimelive, channel_array
    

class Frontend(QtGui.QFrame):
    
    paramSignal = pyqtSignal(list)
    measureSignal = pyqtSignal()
        
    def __init__(self,  *args, **kwargs):

        super().__init__(*args, **kwargs)
                
        # initial directory

        self.initialDir = r'C:\Data'
        self.setup_gui()
        self.colorcounter = 0
        
    def start_measurement(self):
        
        self.measureSignal.emit()
#        self.measureButton.setChecked(True) TO DO:  from backend that toggles button
    
    def load_folder(self):

        try:
            root = Tk()
            root.withdraw()
            folder = filedialog.askdirectory(parent=root,
                                             initialdir=self.initialDir)
            root.destroy()
            if folder != '':
                self.folderEdit.setText(folder)
        except OSError:
            pass
        
    def emit_param(self):
        
        # TO DO: change for dictionary
        
        filename = os.path.join(self.folderEdit.text(),
                                self.filenameEdit.text())
        
        name = filename
        res = int(self.resolutionEdit.text())
        tacq = int(self.acqtimeEdit.text())
        folder = self.folderEdit.text()
        liveTCSPC = self.liveTCSPCCheckbox.isChecked()
        liveCTS = self.liveCTSCheckbox.isChecked()
        
        paramlist = [name, res, tacq, folder, liveTCSPC, liveCTS]
        
        
        
        
        self.paramSignal.emit(paramlist)
        
    @pyqtSlot(float, float, float)
    def get_backend_parameters(self, cts0, cts1, cts2):
        
        # conversion to kHz
        
        cts0_khz = cts0/1000 
        cts1_khz = cts1/1000
        cts2_khz = cts2/1000
        
        if cts0 != 0:
            
            self.maxReltime = 1/ cts0 * 1e9
        else:
            self.maxReltime = 52
        self.reltimearray = np.arange(0, self.maxReltime, self.maxReltime/100)
        
        
        self.channel0Value.setText(('{}'.format(cts0_khz)))
        self.channel1Value.setText(('{}'.format(cts1_khz)))
        self.channel2Value.setText(('{}'.format(cts2_khz)))
    

    
    
    
    @pyqtSlot(np.ndarray, np.ndarray, bool, bool)    
    def plot_data_new(self, relTime, absTime, scanflag, liveFLUX = False):
        
#        print("got data", absTime)
        
        if scanflag or liveFLUX:
#            pass
            self.clear_data()
        
        
        
        if relTime != []: #and liveFLUX == False:
            
            counts= fast_histogram.histogram1d(relTime, range = [0, self.maxReltime], bins= 100) # TO DO: choose proper binning
            self.histPlot.plot(self.reltimearray[0:len(counts)], counts,  pen=pg.mkPen(self.colorcounter))

#        plt.hist(relTime, bins=300)
#        plt.xlabel('time (ns)')
#        plt.ylabel('ocurrences')
        
        binsize = 30 #ms
        
        if absTime.shape[0] == 0:
            #emtpy
            #TODO: Bad fix!
            numberbins = 2
            absTime = [0,2,3]
            maxabs = 1
            minabs = 0
            absbins = np.arange(0,1,1/numberbins)
        else:
            maxabs = max(absTime)
            minabs = min(absTime)#min(absTime)
#            print(absTime.shape)
            if (maxabs -minabs)/binsize <= 4000:
                numberbins = int((maxabs- minabs)/binsize)
            else:
                numberbins = 4000
                
            if numberbins < 3:
                numberbins = 3
                maxabs = 10
                
#            print("test")
#            print(minabs,maxabs, numberbins)            
            absbins =  np.arange(minabs, maxabs, (maxabs- minabs)/(numberbins))

#        print(absTime)
        
        timetrace = fast_histogram.histogram1d(absTime, range = [minabs, maxabs], bins = numberbins )
#        timetrace, time = np.histogram(absTime, bins= numberbins) # timetrace with 50 bins
#        print(len(timetrace))
#        print(numberbins)



        self.tracePlot.disableAutoRange()
        self.tracePlot.plot(absbins[0:len(timetrace)], timetrace / binsize * 1000, pen=pg.mkPen(self.colorcounter))# #TODO: make adaptable with bins
        self.tracePlot.autoRange()
#        self.tracePlot.addItem(lines)

#        plt.plot(time[0:-1], timetrace)
#        plt.xlabel('time (ms)')
#        plt.ylabel('counts')
        if liveFLUX == False:
            
            self.colorcounter = self.colorcounter + 1
            
            
    

    
    
    @pyqtSlot(np.ndarray, np.ndarray, bool, bool)    
    def plot_data(self, relTime, absTime, scanflag, liveFLUX = False):
        
#        print("got data", absTime)
        
        if scanflag:
            self.clear_data()
        
        
        
        if relTime != [] and liveFLUX == False:
            
            counts= fast_histogram.histogram1d(relTime, range = [0, self.maxReltime], bins= 100) # TO DO: choose proper binning
            self.histPlot.plot(self.reltimearray[0:len(counts)], counts,  pen=pg.mkPen(self.colorcounter))

#        plt.hist(relTime, bins=300)
#        plt.xlabel('time (ns)')
#        plt.ylabel('ocurrences')
        
        binsize = 30 #ms
        
        if absTime.shape[0] == 0:
            #emtpy
            #TODO: Bad fix!
            numberbins = 2
            absTime = [0,2,3]
            maxabs = 1
            minabs = 0
            absbins = np.arange(0,1,1/numberbins)
        else:
            maxabs = max(absTime)
            minabs = min(absTime)#min(absTime)
#            print(absTime.shape)
            if (maxabs -minabs)/binsize <= 4000:
                numberbins = int((maxabs- minabs)/binsize)
            else:
                numberbins = 4000
                maxabs = 10
                
                
                
#            print("test")
#            print(minabs,maxabs, numberbins)  
            if numberbins < 3:
                numberbins = 3
                maxabs = 60                

            absbins =  np.arange(minabs, maxabs, (maxabs- minabs)/(numberbins))
#        print(minabs,maxabs, numberbins)
#        print(absTime)
        if numberbins < 1:
            numberbins = 1
            
# TO DO: uncomment lines 248-260        
        
        timetrace = fast_histogram.histogram1d(absTime, range = [minabs, maxabs], bins = numberbins )
#        timetrace, time = np.histogram(absTime, bins= numberbins) # timetrace with 50 bins
#        print(len(timetrace))
#        print(numberbins)
#        print(absTime)
        self.tracePlot.disableAutoRange()
        if len(timetrace) <= len(absbins):
        
            self.tracePlot.plot(absbins[0:len(timetrace)], timetrace / binsize * 1000, pen=pg.mkPen(self.colorcounter))# #TODO: make adaptable with bins
            
        else:
            self.tracePlot.plot(absbins, timetrace[0:len(absbins)] / binsize * 1000, pen=pg.mkPen(self.colorcounter))# #TODO: make adaptable with bins    
        
        
        
        
        
        self.tracePlot.autoRange()

#        plt.plot(time[0:-1], timetrace)
#        plt.xlabel('time (ms)')
#        plt.ylabel('counts')
        if liveFLUX == False:
            
            self.colorcounter = self.colorcounter + 1
            
            

    def clear_data(self):
        
        self.histPlot.clear()
        self.tracePlot.clear()
        self.colorcounter = 0
    
    def make_connection(self, backend):
        
        backend.ctRatesSignal.connect(self.get_backend_parameters)
        backend.plotDataSignal.connect(self.plot_data)
        backend.clearSignal.connect(self.clear_data)
        
    def  setup_gui(self):
        
        # widget with tcspc parameters

        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)
        
        self.paramWidget.setFixedHeight(250)
        self.paramWidget.setFixedWidth(230)
        
#        phParamTitle = QtGui.QLabel('<h2><strong>TCSPC settings</strong></h2>')
        hhParamTitle = QtGui.QLabel('<h2>TCSPC settings</h2>')
        hhParamTitle.setTextFormat(QtCore.Qt.RichText)
        
        # widget to display data
        
        self.dataWidget = pg.GraphicsLayoutWidget()
        
        # file/folder widget
        
        self.fileWidget = QtGui.QFrame()
        self.fileWidget.setFrameStyle(QtGui.QFrame.Panel |
                                      QtGui.QFrame.Raised)
        
        self.fileWidget.setFixedHeight(120)
        self.fileWidget.setFixedWidth(230)
        
        # Shutter button
        
        self.shutterButton = QtGui.QPushButton('Shutter open/close')
        self.shutterButton.setCheckable(True)
        
        # Prepare button
        self.liveTCSPCCheckbox = QtGui.QCheckBox('TCSPC live')
        self.liveTCSPCCheckbox.setChecked(True)

        # show button
        self.liveCTSCheckbox = QtGui.QCheckBox('CTS live')
        self.liveCTSCheckbox.setChecked(True)
        
        
        
        self.prepareButton = QtGui.QPushButton('Prepare TTTR')
        
        # Measure button

        self.measureButton = QtGui.QPushButton('Measure TTTR')
        self.measureButton.setCheckable(True)
        
        # forced stop measurement
        
        self.stopButton = QtGui.QPushButton('Stop')
        
        # exportData button
        
        self.exportDataButton = QtGui.QPushButton('Export data')
#        self.exportDataButton.setCheckable(True)
        
        # Clear data
        
        self.clearButton = QtGui.QPushButton('Clear data')
#        self.clearButton.setCheckable(True)
        
        # TCSPC parameters

        self.acqtimeLabel = QtGui.QLabel('Acquisition time [s]')
        self.acqtimeEdit = QtGui.QLineEdit('1')
        self.resolutionLabel = QtGui.QLabel('Resolution [ps]')
        self.resolutionEdit = QtGui.QLineEdit('8')
        self.offsetLabel = QtGui.QLabel('Offset [ns]')
        self.offsetEdit = QtGui.QLineEdit('0')
        
        self.channel0Label = QtGui.QLabel('Input0 (sync) [kHz]')
        self.channel0Value = QtGui.QLineEdit('')
        self.channel0Value.setReadOnly(True)
        
        self.channel1Label = QtGui.QLabel('Input1 (APD) [kHz]')
        self.channel1Value = QtGui.QLineEdit('')
        self.channel1Value.setReadOnly(True)
        
        self.channel2Label = QtGui.QLabel('Input2 (APD) [kHz]')
        self.channel2Value = QtGui.QLineEdit('')
        self.channel2Value.setReadOnly(True)
        
        self.filenameLabel = QtGui.QLabel('File name')
        self.filenameEdit = QtGui.QLineEdit('filename')
        
        # microTime histogram and timetrace
        
        self.histPlot = self.dataWidget.addPlot(row=1, col=0, title="microTime histogram")
        self.histPlot.setLabels(bottom=('ns'),
                                left=('counts'))
        self.histPlot.setLogMode(False, True) 
        self.tracePlot = self.dataWidget.addPlot(row=2, col=0, title="Time trace")
        self.tracePlot.setLabels(bottom=('ms'),
                                left=('cps')) 
        
        # folder
        
        # TO DO: move this to backend
        
        today = str(date.today()).replace('-', '')
        root = r'C:\\Data\\'
        folder = root + today
        
        try:  
            os.mkdir(folder)
        except OSError:  
            print(datetime.now(), '[tcspc] Directory {} already exists'.format(folder))
        else:  
            print(datetime.now(), '[tcspc] Successfully created the directory {}'.format(folder))

        self.folderLabel = QtGui.QLabel('Folder')
        self.folderEdit = QtGui.QLineEdit(folder)
        self.browseFolderButton = QtGui.QPushButton('Browse')
        self.browseFolderButton.setCheckable(True)
        
        # GUI connections
        
        self.measureButton.clicked.connect(self.start_measurement)
        self.browseFolderButton.clicked.connect(self.load_folder)
        self.clearButton.clicked.connect(self.clear_data)    
        self.liveTCSPCCheckbox.stateChanged.connect(self.emit_param)
        self.acqtimeEdit.textChanged.connect(self.emit_param)
        self.resolutionEdit.textChanged.connect(self.emit_param)
        
        self.liveCTSCheckbox.stateChanged.connect(self.emit_param)
        

        # general GUI layout

        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.paramWidget, 0, 0)
        grid.addWidget(self.fileWidget, 1, 0)
        grid.addWidget(self.dataWidget, 0, 1, 2, 2)
        
        # param Widget layout
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)
        subgrid.addWidget(hhParamTitle, 0, 0, 2, 3)

        subgrid.addWidget(self.acqtimeLabel, 2, 0)
        subgrid.addWidget(self.acqtimeEdit, 2, 1)
        subgrid.addWidget(self.resolutionLabel, 4, 0)
        subgrid.addWidget(self.resolutionEdit, 4, 1)
        subgrid.addWidget(self.offsetLabel, 6, 0)
        subgrid.addWidget(self.offsetEdit, 6, 1)
        subgrid.addWidget(self.channel0Label, 8, 0)
        subgrid.addWidget(self.channel0Value, 8, 1)
        subgrid.addWidget(self.channel1Label, 9, 0)
        subgrid.addWidget(self.channel1Value, 9, 1)
        subgrid.addWidget(self.channel2Label, 10, 0)
        subgrid.addWidget(self.channel2Value, 10, 1)
        subgrid.addWidget(self.measureButton, 17, 0)
        subgrid.addWidget(self.prepareButton, 18, 0)
        subgrid.addWidget(self.shutterButton, 19, 0)
        subgrid.addWidget(self.stopButton, 17, 1)
        subgrid.addWidget(self.clearButton, 18, 1)
        subgrid.addWidget(self.liveTCSPCCheckbox, 19, 1)
        subgrid.addWidget(self.liveCTSCheckbox, 20, 1)
        
        
        file_subgrid = QtGui.QGridLayout()
        self.fileWidget.setLayout(file_subgrid)
        
        file_subgrid.addWidget(self.filenameLabel, 0, 0, 1, 2)
        file_subgrid.addWidget(self.filenameEdit, 1, 0, 1, 2)
        file_subgrid.addWidget(self.folderLabel, 2, 0, 1, 2)
        file_subgrid.addWidget(self.folderEdit, 3, 0, 1, 2)
        file_subgrid.addWidget(self.browseFolderButton, 4, 0)
        
class Backend(QtCore.QObject):

    ctRatesSignal = pyqtSignal(float, float, float)
    plotDataSignal = pyqtSignal(np.ndarray, np.ndarray, bool, bool)
    
    measureTCSPCSignal = pyqtSignal()
    HHStatusSignal = pyqtSignal(bool)
    
    xyzStartSignal = pyqtSignal(str)
    xyzEndSignal = pyqtSignal(str)
    
    tcspcDoneSignal = pyqtSignal()
    toggleShutterSignal = pyqtSignal(bool)
    toggleShutterSignalMINFLUX = pyqtSignal(bool)
    setODSignal = pyqtSignal()
    startAOTFsequenceSignal = pyqtSignal()
    MINFLUXdataSignal = pyqtSignal(np.ndarray)
    
    clearSignal = pyqtSignal()
    
    tcspcPickAndDestroySignal = pyqtSignal()
    
    def __init__(self, hh_device, scanTCSPCdata, MINFLUXdata,  *args, **kwargs): 
        
        super().__init__(*args, **kwargs)
        
        self.hh = hh_device 
        
        
        
        
        self.measureTCSPCSignal.connect(self.hh.startTTTR)
#        
#        hhWorkerThread = QtCore.QThread()
#        self.hh.moveToThread(hhWorkerThread)
#        
#        hhWorkerThread.start()
                
                
        
        self.scanTCSPCdata = scanTCSPCdata
        self.MINFLUXdata = MINFLUXdata
        self.syncTimer = QtCore.QTimer()     
        
        self.liveMINFLUXflag = False
        self.shutter_state = False
        self.syncTime = 250#300 # ms
        self.hhinit = False
        self.max_time_display = 10000 # ms
        print("initing")
        self.ptr = 0
        
        self.PickAndDestroyflag = False
        self.pool = Pool(processes=5)
        
    def init_view(self):
        self.last_plot_index = 0
        
        self.absdata_temp = np.zeros(0)
        self.reldata_temp = np.zeros(0)
        self.abstimemax = 0
        self.relTime = []
        self.absTime = []
        self.binary = True
        
        
    def update_view(self):
#        self.measure_count_rate()
        

        
        if self.hh.ctcdone == False:
#            print("updating")
            
            if self.liveTCSPC ==  True:
                
                self.measure_count_rate()
                if self.liveMINFLUXflag:
                    self.liveplot_MINFLUX()
                else:
                    self.liveplot()
            else:
                self.measure_count_rate()
            

        else:
            self.syncTimer.stop()
            print("measured minflux")
            self.tcspcDoneSignal.emit()
            #xyzEndSignal
            self.toggleShutterSignalMINFLUX.emit(False)
            #done xyzsignal
            self.xyzEndSignal.emit(self.currentfname)
            self.export_data()
        
        

        
    def measure_count_rate(self):
        self.cts0 = self.hh.getSyncRate()
        self.cts1 = self.hh.countrate(0)
        self.cts2 = self.hh.countrate(1)
        #print(self.ctssync)
        #print(type(self.cts1))
        self.ctRatesSignal.emit(self.cts0, self.cts1, self.cts2)
        # TO DO: fix this method to update cts0 and cts1 automatically
#        self.syncTime = 0 
        
          
    def prepare_hh(self):
        self.PickAndDestroyflag = False
        


        
        if self.hhinit == False:
                
            self.hh.ctcdone = True 
            
            self.hh.open()
            self.hh.initialize()
            self.hh.setup()
            self.hh.syncDivider = 4 # this parameter must be set such that the count rate at channel 0 (sync) is equal or lower than 10MHz
            print(self.resolution)
            self.hh.resolution = self.resolution # desired resolution in ps
            
            self.hh.lib.HH_SetBinning(ctypes.c_int(0), 
                                      ctypes.c_int(1)) # TO DO: fix this in a clean way (1 = 8 ps resolution)
    
            self.hh.offset = 0
            self.hh.tacq = self.tacq *1000 # time in ms
            
            #self.cts0 = self.hh.countrate(0)
            self.cts0 = self.hh.getSyncRate()
            self.cts1 = self.hh.countrate(0)
            self.cts2 = self.hh.countrate(1)
            print("\nSyncrate=%1d/s" % self.cts0)
            #print(self.ctssync)
            #print(type(self.cts1))
            self.ctRatesSignal.emit(self.cts0, self.cts1, self.cts2)
      
            print(datetime.now(), '[tcspc] Resolution = {} ps'.format(self.hh.resolution))
            print(datetime.now(), '[tcspc] Acquisition time = {} s'.format(self.hh.tacq))
        
            print(datetime.now(), '[tcspc] HydraHarp 400 prepared for TTTR measurement')
#            self.init_view() # 
            self.hhinit = True
        
        self.cts0 = self.hh.getSyncRate()
        if self.cts0 > 0:
            self.globRes = 1 / self.cts0  * 1e3 # ìn ms
        else: 
            self.globRes = 0
        self.timeRes = self.hh.resolution * 1e-12 * 1e9 # time resolution in ns
        
    
#    def prepare_hh_histo(self):
#        #TODO: delete histo stuff:
#        self.hh.open()
#        self.hh.initHistoScan()
#        
#        self.hh.setup()
#        self.hh.histosetup()
#        self.hh.syncDivider = 4 # this parameter must be set such that the count rate at channel 0 (sync) is equal or lower than 10MHz
#        print(self.resolution)
#        
#        self.hh.resolution = self.resolution # desired resolution in ps
#        
#        self.hh.lib.HH_SetBinning(ctypes.c_int(0), 
#                                  ctypes.c_int(1)) # TO DO: fix this in a clean way (1 = 8 ps resolution)
#
#        self.hh.offset = 0
#        #self.hh.tacq = self.tacq #* 1000 # time in ms
#        
#        #self.cts0 = self.hh.countrate(0)
#        self.cts0 = self.hh.getSyncRate()
#        self.cts1 = self.hh.countrate(0)
#        #print(self.ctssync)
#        #print(type(self.cts1))
#        self.ctRatesSignal.emit(self.cts0, self.cts1)
#  
#        print(datetime.now(), '[tcspc] Resolution = {} ps'.format(self.hh.resolution))
#        print(datetime.now(), '[tcspc] Acquisition time = {} s'.format(self.hh.tacq))
#    
#        print(datetime.now(), '[tcspc] HydraHarp 400 prepared for TTTR measurement')
#    
#    

    
    


    @pyqtSlot()           
    def measure_live(self):
        self.toggleShutterSignal.emit(True)
        t0 = time.time()

        self.currentfname = tools.getUniqueName(self.fname)

        t1 = time.time()
        
        print(datetime.now(), '[tcspc] starting the HH measurement took {} s'.format(t1-t0))
        self.init_view() 
#        self.hh.startTTTR_live()
        self.measureTCSPCSignal.emit()
        
        
        print(self.currentfname)
        #np.savetxt(self.currentfname + '.txt', [])
        
        while self.hh.ctcdone == False:
            
            
            self.liveplot()

        self.toggleShutterSignalMINFLUX.emit(False)
        self.export_data()  


    
    @pyqtSlot()           
    def measure(self):
        self.abstimemax = 0
        self.relTime = []
        self.absTime = []
        self.last_plot_index = 0
        
        self.toggleShutterSignal.emit(True)
        t0 = time.time()

        self.currentfname = tools.getUniqueName(self.fname)

        t1 = time.time()
        
        print(datetime.now(), '[tcspc] starting the HH measurement took {} s'.format(t1-t0))
        self.init_view() 
        self.hh.startTTTR()
        print(self.currentfname)
        #np.savetxt(self.currentfname + '.txt', [])
        
        while self.hh.ctcdone == False:
            pass

        self.toggleShutterSignal.emit(False)
        self.export_data()  
        
    @pyqtSlot(str, float, int, bool, bool,  np.ndarray)
    def prepare_minflux(self, fname, acqtime, n, PickAndDestroyFlag, liveMINFLUXflag, taus):
        
        self.abstimemax = 0
        self.relTime = []
        self.absTime = []
        self.last_plot_index = 0
        self.liveMINFLUXflag = liveMINFLUXflag
        self.PickAndDestroyflag = PickAndDestroyFlag
        
        if PickAndDestroyFlag:
            self.measType = "PickAndDestroy"
        else:
            self.measType = "MINFLUX"
            
        
        print(taus)
        self.taus = taus
        

        
        print(datetime.now(), ' [tcspc] preparing minflux measurement')
    
        t0 = time.time()
        
        self.setODSignal.emit()
        
        self.currentfname = fname #tools.getUniqueNameptu(fname)
        print(self.hhinit)
        self.prepare_hh()
        self.hh.tacq = int(acqtime * n * 1000)
        print(self.hhinit)
        print("hhtacq = ",self.hh.tacq)

        
        print(' [tcspc] self.hh.tacq', self.hh.tacq)
        
        #self.hh.lib.HH_SetBinning(ctypes.c_int(0), 
        #                       ctypes.c_int(1)) # TO DO: fix this in a clean way (1 = 8 ps resolution)
   
        t1 = time.time()
        
        print(datetime.now(), '[tcspc] preparing the HH measurement took {} s'.format(t1-t0))
        
        # if PickAndDestroyFlag == True:
        #     self.PickAndDestroyflag = True
        # else:
        #     self.PickAndDestroyflag = False
            
        if liveMINFLUXflag:
            self.clearSignal.emit()
            
            
        #Changed
        self.init_view()
#        self.syncTimer.start(self.syncTime)        
        
        
    @pyqtSlot()
    def measure_minflux(self):
        self.abstimemax = 0
        self.relTime = []
        self.absTime = []
        
        
        self.last_plot_index = 0
        self.hh.counts = []
        self.startAOTFsequenceSignal.emit()
        self.xyzStartSignal.emit(self.measType)
        self.toggleShutterSignalMINFLUX.emit(True)
        self.hh.ctcdone = False #HACK: !!!
#        self.hh.startTTTR()
        self.init_view()
        self.measureTCSPCSignal.emit()
        
        #np.savetxt(self.currentfname + '.txt', [])
#        print("livestuff")
        self.syncTimer.start(self.syncTime)
        
        
        
#        while 
        
        
    
    
    @pyqtSlot(int) 
    def prepare_scan(self, acqtime):
        
        print(datetime.now(), ' [tcspc] preparing scan measurement')
        
        t0 = time.time()

        #self.currentfname = tools.getUniqueName(fname)
                
        #self.prepare_hh_histo()
        self.prepare_hh() #TODO maybe not timer? close timer?
#        print("[tcspc]: acqtime", acqtime)
        self.hh.tacq = int(acqtime / 1000) #Unit? # TO DO: correspond to GUI !!!
        print(' [tcspc] self.hh.tacq', self.hh.tacq)
        
        
        t1 = time.time()
        
        print(datetime.now(), '[tcspc] preparing the HH measurement took {} s'.format(t1-t0))
        
        
#        fname = self.fname + "scan\\" 
#    
#        self.scanfolder = tools.getUniqueFolder(fname) 
#        try:  
#            os.mkdir(self.scanfolder)
#        except OSError:  
#            print(datetime.now(), '[tcspc] Directory {} already exists'.format(self.scanfolder))
#        else:  
#            print(datetime.now(), '[tcspc] Successfully created the directory {}'.format(self.scanfolder))
        self.scannumber = 0
        
        #Changed!
#        self.syncTimer.start(self.syncTime)
        self.init_view()
        
#    @pyqtSlot()
#    def measure_scan(self):
#
#        self.hh.ctcdone = False #HACK: !!!
#        
#        self.hh.startHistoScan()
#        
#
#        while self.hh.ctcdone == False:
#            pass
#
#        self.scanTCSPCdata.put([True, self.hh.integralCount]) 
    @pyqtSlot()
    def HH_status(self):
        
        
        self.HHStatusSignal.emit(self.hh.ctcdone)

        
    @pyqtSlot()
    def measure_fastscan(self):
        
        
        self.hh.startTTTR()
        
        
        while self.hh.ctcdone == False:
            pass
        #print(self.hh.integralCount)
        self.export_data(noscanFlag = False)
        #print(self.absTime)
        time.sleep(0.005) 
        self.scanTCSPCdata.put([True, self.absTime, self.relTime, self.channel_array]) 
        
    @pyqtSlot()
    def measure_fastscan_Point(self):
        
        
        self.hh.startTTTR()
        
        
        while self.hh.ctcdone == False:
            pass
        #print(self.hh.integralCount)
#        self.export_data(noscanFlag = False)
        #print(self.absTime)
        time.sleep(0.005) 
        self.scanTCSPCdata.put([True, self.hh.numRecords]) 
        
        
    
    @pyqtSlot()
    def closeconnection(self):
        self.hhinit = False
        self.hh.ctcdone = True 
        self.hh.stopTTTR()
        self.hh.closehh()

#        self.hh.closehh()
        print("[tcspc] HH closed")
        self.syncTimer.stop()

    
    
    
    def stop_measure(self):
        self.hhinit = False
#        self.toggleShutterSignal.emit(False)
        self.toggleShutterSignalMINFLUX.emit(False)
        self.hh.ctcdone = True 
        self.hh.stopTTTR()
        self.hh.closehh()

        
        print(datetime.now(), '[tcspc] stop measure function')
    
    def liveplot(self):
        
        
#        print("test")
#        self.countsflat = [val for sublist in self.hh.counts for val in sublist]
        countsflat = sum(self.hh.counts[self.last_plot_index::],[]) #make sublist?
        self.last_plot_index = len(self.hh.counts)
#        print(self.last_plot_index)

#        relTimelive, absTimelive, channel_array = Read_PTU.convertHT3(countsflat)
        self.pool.apply_async(convert_ht3,  args = [countsflat], callback = self.liveplot_callback)
    
    def liveplot_callback(self, callback_result):
        
        relTimelive,absTimelive, channel_array = callback_result
        # relTimelive = relTimelive * self.timeRes #* 1e9# in real time units (ns)
#        self.relTime = relTime   # in (ns)

        absTimelive = absTimelive * self.globRes #* 1e3 # in ms # true time in (ns), 4 comes from syncDivider, 10 Mhz = 40 MHz / syncDivider 
#        self.absTime = absTime
        
#        datasize = np.size(self.absTime[self.absTime != 0])
#        data = np.zeros((3, datasize))
##        print(self.absTime.shape)
##        print(channel_array.shape)
#        
#        data[0, :] = self.relTime[self.absTime != 0]
#        data[1, :] = self.absTime[self.absTime != 0]
#        print(type(absTime))
#        print(absTime)
        
        ##TODO if commented out helps????
#        if self.absdata_temp != []:
#            
#            
#            #todo in lists? Faster!
#            
#            #question is: where is the problem append or hist?
#            self.absdata_temp =  np.append(self.absdata_temp, max(self.absdata_temp)+ np.array(absTime[absTime != 0] ))
#            
#        else:
#            self.absdata_temp =  np.append(self.absdata_temp, np.array(absTime[absTime != 0] ))
#            
#        self.reldata_temp =  np.append(self.reldata_temp, np.array(relTime[absTime != 0] ))            
#        print(data[0, :])
        if absTimelive != []:
#            print(max(absTimelive))
#            self.plotDataSignal.emit(np.array(relTimelive[absTimelive != 0]), self.abstimemax + np.array(absTimelive[absTimelive != 0]), False, True)
            self.plotDataSignal.emit(np.array([]), self.abstimemax + np.array(absTimelive[absTimelive != 0]), False, True)
#            self.last_plot_index = len(self.hh.counts)
            self.abstimemax += max(absTimelive)
        
    
    
    
    def liveplot_MINFLUX(self):
        
        
#        print("test")
#        self.countsflat = [val for sublist in self.hh.counts for val in sublist]
        countsflat = sum(self.hh.counts[self.last_plot_index::],[]) #make sublist?
        self.last_plot_index = len(self.hh.counts)
#        print(self.last_plot_index)
        
        
#        print(len(self.hh.counts))
        
#        if self.cts0 > 0:
#            globRes = 1 / self.cts0  # Syncrate in kHz
#        else: 
#            globRes = 0
#        timeRes = self.hh.resolution * 1e-12 # time resolution in s

#        relTimelive, absTimelive, channel_array = Read_PTU.convertHT3(countsflat)
        self.pool.apply_async(convert_ht3,  args = [countsflat], callback = self.liveplot_MINFLUX_callback)
    
    
    def liveplot_MINFLUX_callback_new(self,callback_result):
        
        relTimelive,absTimelive, channel_array = callback_result
        
        if absTimelive != [] and relTimelive != []:
            self.relTime.extend(relTimelive * self.timeRes) # in real time units (ns)
            self.absTime.extend(self.abstimemax + absTimelive * self.globRes)
    #          + np.array(absTimelive[absTimelive != 0])
            self.abstimemax = max(self.absTime)
            
            #delete old events
            if self.abstimemax - min(self.absTime) > self.max_time_display:
                
                #find index where <  abstimemax - maxtimedisplay
                ind = bisect.bisect_left(self.absTime, self.abstimemax - self.max_time_display)
                self.absTime = self.absTime[ind:] 
                self.relTime = self.relTime[ind:] 
                
    
                
                
    
            
    #        fractions = fast_histogram.histogram1d(relTimelive, bins= self.taus)
            fractions, bins = np.histogram(relTimelive* self.timeRes, bins= self.taus)
#            print("TCSPC fractions", fractions)
            
            
            if (self.ptr % 8):
    #            print(max(absTimelive))
    #            self.plotDataSignal.emit(np.array(relTimelive[absTimelive != 0]), self.abstimemax + np.array(absTimelive[absTimelive != 0]), False, True)
    
                self.plotDataSignal.emit(np.array(self.relTime), np.array(self.absTime) , False, True)
    #            self.last_plot_index = len(self.hh.counts)
                
            self.MINFLUXdataSignal.emit(np.array([fractions[1],fractions[3],fractions[5],fractions[7]]))
            self.ptr += 1
        
    
    def liveplot_MINFLUX_callback(self,callback_result):
        
        relTimelive,absTimelive, channel_array = callback_result
        relTimelive = relTimelive * self.timeRes # in real time units (ns)
        absTimelive = absTimelive * self.globRes
#        fractions = fast_histogram.histogram1d(relTimelive, bins= self.taus)
        fractions, bins = np.histogram(relTimelive, bins= self.taus)
#        print("TCSPC fractions", fractions)
        
        
        if absTimelive != []:
#            print(max(absTimelive))
#            self.plotDataSignal.emit(np.array(relTimelive[absTimelive != 0]), self.abstimemax + np.array(absTimelive[absTimelive != 0]), False, True)
            self.plotDataSignal.emit(np.array([]),  self.abstimemax + np.array(absTimelive[absTimelive != 0]), False, True)
#            self.last_plot_index = len(self.hh.counts)
            self.abstimemax += max(absTimelive)
        self.MINFLUXdataSignal.emit(np.array([fractions[1],fractions[3],fractions[5],fractions[7]]))

        
        
        
    def liveplot_MINFLUX_callback_last(self,callback_result):
        #plots the TCSPC hist of the first 10.000 photons
        
        relTimelive,absTimelive, channel_array = callback_result
        relTimelive = relTimelive * self.timeRes # in real time units (ns)

        absTimelive = absTimelive * self.globRes
       
        
        if absTimelive != []:
#            print(max(absTimelive))
#            self.plotDataSignal.emit(np.array(relTimelive[absTimelive != 0]), self.abstimemax + np.array(absTimelive[absTimelive != 0]), False, True)
            self.plotDataSignal.emit(np.array(relTimelive), np.array(absTimelive[absTimelive != 0]), False, False)
#            self.last_plot_index = len(self.hh.counts)
            self.abstimemax += max(absTimelive)
        
        
        
    def export_data(self, noscanFlag = True):
        #self.currentfname = tools.getUniqueName(self.fname)
#        self.reset_data()
        

        
        countsflat = [val for sublist in self.hh.counts for val in sublist]
        if (self.binary == True) and noscanFlag and (self.PickAndDestroyflag == False):
            
            #plot TCSPC data of first 10000 photons:
            if len(self.hh.counts) > 10000:
                pass
#                countsflatlast = sum(self.hh.counts[:10000],[])                 
#                self.pool.apply_async(convert_ht3,  args = [countsflatlast], callback = self.liveplot_MINFLUX_callback_last)
            else:
#                self.pool.apply_async(convert_ht3,  args = [countsflat], callback = self.liveplot_MINFLUX_callback_last)
                pass
            filename = self.currentfname + '_arrays.ptu'
            np.savetxt(filename, np.array(countsflat).astype(np.uint32), fmt='%i') 
        
        else:
            
#            self.countsflat = [val for sublist in self.hh.counts for val in sublist]
            
            #print(datetime.now(), '[tcspc] opened {} file'.format(self.currentfname))
            
    
            #globRes = 2.5e-8  # in ns, corresponds to sync @40 MHz
            if self.cts0 > 0:
                globRes = 1 / self.cts0  # Syncrate in kHz
            else: 
                globRes = 0
            timeRes = self.hh.resolution * 1e-12 # time resolution in s
            print(self.hh.resolution)
            relTime, absTime, self.channel_array = Read_PTU.convertHT3(countsflat)
            if noscanFlag:
                print('{} events recorded'.format(len(relTime)))
                filename = self.currentfname + '_arrays.txt'
    #        print(channel_array)
            relTime = relTime * timeRes # in real time units (s)
            self.relTime = relTime * 1e9  # in (ns)
    
            absTime = absTime * globRes * 1e9  # true time in (ns), 4 comes from syncDivider, 10 Mhz = 40 MHz / syncDivider 
            self.absTime = absTime / 1e6 # in ms
    
    
            
            datasize = np.size(self.absTime[self.absTime != 0])
            data = np.zeros((3, datasize))
    #        print(self.absTime.shape)
    #        print(channel_array.shape)
            
            data[0, :] = self.relTime[self.absTime != 0]
            data[1, :] = self.absTime[self.absTime != 0]
            data[2,:] = self.channel_array[self.absTime != 0]
        
        
       
            
            
            
            
            
            if noscanFlag:
                
                
                
#                else:
                filename = self.currentfname + '_arrays.txt'
                np.savetxt(filename, data.T) # transpose for easier loading
                self.plotDataSignal.emit(data[0, :], data[1, :], False, False) #TODO: changed        
                print(datetime.now(), '[tcspc] tcspc data exported')
            else:
                if data[1,:] == np.array([]):
                    self.plotDataSignal.emit(np.array[0],np.array[1], np.array[2], True, False)
                else:
                    self.plotDataSignal.emit(data[0, :], data[1, :], True, False)
                #np.savetxt(self.scanfolder + "\\"+ "line_"+ str(self.scannumber) + ".txt", data.T) # transpose for easier loading
                self.scannumber =  self.scannumber + 1
                
            
            if self.PickAndDestroyflag == True:
                self.tcspcPickAndDestroySignal.emit()
                
                
                
                    
        if self.liveTCSPC == False:
        
        
            if self.cts0 > 0:
                globRes = 1 / self.cts0  # Syncrate in kHz
            else: 
                globRes = 0
            timeRes = self.hh.resolution * 1e-12 # time resolution in s
            print(self.hh.resolution)
            relTime, absTime, self.channel_array = Read_PTU.convertHT3(countsflat)
            
#        print(channel_array)
            relTime = relTime * timeRes # in real time units (s)
            self.relTime = relTime * 1e9  # in (ns)
    
            absTime = absTime * globRes * 1e9  # true time in (ns), 4 comes from syncDivider, 10 Mhz = 40 MHz / syncDivider 
            self.absTime = absTime / 1e6 # in ms
    
    
            
            datasize = np.size(self.absTime[self.absTime != 0])
            data = np.zeros((3, datasize))
    #        print(self.absTime.shape)
    #        print(channel_array.shape)
            
            data[0, :] = self.relTime[self.absTime != 0]
            data[1, :] = self.absTime[self.absTime != 0]
            data[2,:] = self.channel_array[self.absTime != 0]
            self.plotDataSignal.emit(data[0, :], data[1, :], False, False) #TODO: changed       
        
            
            
    @pyqtSlot(list)
    def get_frontend_parameters(self, paramlist):
        
        print(datetime.now(), '[tcspc] got frontend parameters')

        self.fname = paramlist[0]
        self.resolution = paramlist[1]
        self.tacq = paramlist[2]
        self.folder = paramlist[3]      
        self.liveTCSPC = paramlist[4]    
        self.liveCTS = paramlist[5]    
        
    @pyqtSlot(bool)
    def toggle_shutter(self, val):
        
        if val is True:
            
            self.shutter_state = True
            """
            self.adw.Set_Par(55, 0)
            self.adw.Set_Par(50, 1)
            self.adw.Start_Process(5)"""
            
            print(datetime.now(), '[tcspc] Shutter opened...')
            
        if val is False:
            
            self.shutter_state = False
            """
            self.adw.Set_Par(55, 0)
            self.adw.Set_Par(50, 0)
            self.adw.Start_Process(5)"""

            print(datetime.now(), '[tcspc] Shutter closed')

    def make_connection(self, frontend):

        frontend.paramSignal.connect(self.get_frontend_parameters)
        frontend.measureSignal.connect(self.measure)
        frontend.prepareButton.clicked.connect(self.prepare_hh)
        frontend.stopButton.clicked.connect(self.stop_measure)
        frontend.shutterButton.clicked.connect(lambda: self.toggle_shutter(frontend.shutterButton.isChecked()))

        frontend.emit_param() # TO DO: change such that backend has parameters defined from the start

    def stop(self):
        self.hh.ctcdone = True 
        self.hh.stopTTTR()
        self.hh.closehh()
#        self.toggleShutterSignal.emit(False)
        self.toggleShutterSignalMINFLUX.emit(False)
        self.hhinit = False
        print("[tcspc] HH closed")
        self.syncTimer.stop()


if __name__ == '__main__':

    app = QtGui.QApplication([])
    app.setStyle(QtGui.QStyleFactory.create('fusion'))
#    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    hh = hydraharp.HydraHarp400()
    scanTCSPCdata = queue.Queue()

    
    hhWorkerThread = QtCore.QThread()
    hh.moveToThread(hhWorkerThread)
    
    hhWorkerThread.start()
    
    
    
    gui = Frontend()
    worker = Backend(hh, scanTCSPCdata)
    workerThread = QtCore.QThread()
    workerThread.start()
    worker.moveToThread(workerThread)
    worker.syncTimer.moveToThread(workerThread)
    worker.syncTimer.timeout.connect(worker.update_view) # TODO connect

    
    
    worker.make_connection(gui)
    gui.make_connection(worker)

    gui.setWindowTitle('Time-correlated single-photon counting')
    gui.show()

    app.exec_()
    