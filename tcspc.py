# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:14:14 2019



PyFlux Masterfile

@author:  Jonas Zaehringer
original template: USUARIO


"""

import numpy as np
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
        paramlist = [name, res, tacq, folder]
        
        self.paramSignal.emit(paramlist)
        
    @pyqtSlot(float, float)
    def get_backend_parameters(self, cts0, cts1):
        
        # conversion to kHz
        
        cts0_khz = cts0/1000 
        cts1_khz = cts1/1000
        
        self.channel0Value.setText(('{}'.format(cts0_khz)))
        self.channel1Value.setText(('{}'.format(cts1_khz)))
    
    @pyqtSlot(np.ndarray, np.ndarray, bool)    
    def plot_data(self, relTime, absTime, scanflag):
        
        
        if scanflag:
            self.clear_data()
        
        
        
        if relTime != []:
            counts, bins = np.histogram(relTime, bins= 100) # TO DO: choose proper binning
            self.histPlot.plot(bins[0:-1], counts,  pen=pg.mkPen(self.colorcounter))

#        plt.hist(relTime, bins=300)
#        plt.xlabel('time (ns)')
#        plt.ylabel('ocurrences')
        
        binsize = 10 #ms
        
        if absTime.shape[0] == 0:
            #emtpy
            #TODO: Bad fix!
            numberbins = 2
            absTime = [0]
        else:
            print(absTime.shape)
            if max(absTime)/binsize <= 5000:
                numberbins = np.arange(0, max(absTime), binsize)
            else:
                numberbins = 10000
        
        timetrace, time = np.histogram(absTime, bins= numberbins) # timetrace with 50 bins

        self.tracePlot.plot(time[0:-1], timetrace * 50, pen=pg.mkPen(self.colorcounter)) #TODO: make adaptable with bins

#        plt.plot(time[0:-1], timetrace)
#        plt.xlabel('time (ms)')
#        plt.ylabel('counts')
        self.colorcounter = self.colorcounter + 1
    def clear_data(self):
        
        self.histPlot.clear()
        self.tracePlot.clear()
        self.colorcounter = 0
    
    def make_connection(self, backend):
        
        backend.ctRatesSignal.connect(self.get_backend_parameters)
        backend.plotDataSignal.connect(self.plot_data)
        
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
        
        self.filenameLabel = QtGui.QLabel('File name')
        self.filenameEdit = QtGui.QLineEdit('filename')
        
        # microTime histogram and timetrace
        
        self.histPlot = self.dataWidget.addPlot(row=1, col=0, title="microTime histogram")
        self.histPlot.setLabels(bottom=('ns'),
                                left=('counts'))
        self.histPlot.setLogMode(False, True) 
        self.tracePlot = self.dataWidget.addPlot(row=2, col=0, title="Time trace")
        self.tracePlot.setLabels(bottom=('ms'),
                                left=('cps')) #TODO: change to cps
        
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
        
        self.acqtimeEdit.textChanged.connect(self.emit_param)
        self.resolutionEdit.textChanged.connect(self.emit_param)

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
        
        subgrid.addWidget(self.measureButton, 17, 0)
        subgrid.addWidget(self.prepareButton, 18, 0)
        subgrid.addWidget(self.shutterButton, 19, 0)
        subgrid.addWidget(self.stopButton, 17, 1)
        subgrid.addWidget(self.clearButton, 18, 1)
        
        file_subgrid = QtGui.QGridLayout()
        self.fileWidget.setLayout(file_subgrid)
        
        file_subgrid.addWidget(self.filenameLabel, 0, 0, 1, 2)
        file_subgrid.addWidget(self.filenameEdit, 1, 0, 1, 2)
        file_subgrid.addWidget(self.folderLabel, 2, 0, 1, 2)
        file_subgrid.addWidget(self.folderEdit, 3, 0, 1, 2)
        file_subgrid.addWidget(self.browseFolderButton, 4, 0)
        
class Backend(QtCore.QObject):

    ctRatesSignal = pyqtSignal(float, float)
    plotDataSignal = pyqtSignal(np.ndarray, np.ndarray, bool)
    
    tcspcDoneSignal = pyqtSignal()
    toggleShutterSignal = pyqtSignal(bool)
    toggleShutterSignalMINFLUX = pyqtSignal(bool)
    setODSignal = pyqtSignal()
    
    tcspcPickAndDestroySignal = pyqtSignal()
    
    def __init__(self, hh_device, scanTCSPCdata,  *args, **kwargs): 
        
        super().__init__(*args, **kwargs)
        
        self.hh = hh_device 
        self.scanTCSPCdata = scanTCSPCdata
        self.syncTimer = QtCore.QTimer()     
        
        self.shutter_state = False
        self.syncTime = 500
        self.hhinit = False
        
        
        self.PickAndDestroyflag = False
        
    def init_view(self):
        self.last_plot_index = 0
        self.syncTimer.start(self.syncTime)
        
        
        
    def update_view(self):
        self.measure_count_rate()
#        self.liveplot()

        
    def measure_count_rate(self):
        self.cts0 = self.hh.getSyncRate()
        self.cts1 = self.hh.countrate(0)
        #print(self.ctssync)
        #print(type(self.cts1))
        self.ctRatesSignal.emit(self.cts0, self.cts1)
        # TO DO: fix this method to update cts0 and cts1 automatically
        self.syncTime = 0 
        
          
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
            print("\nSyncrate=%1d/s" % self.cts0)
            #print(self.ctssync)
            #print(type(self.cts1))
            self.ctRatesSignal.emit(self.cts0, self.cts1)
      
            print(datetime.now(), '[tcspc] Resolution = {} ps'.format(self.hh.resolution))
            print(datetime.now(), '[tcspc] Acquisition time = {} s'.format(self.hh.tacq))
        
            print(datetime.now(), '[tcspc] HydraHarp 400 prepared for TTTR measurement')
#            self.init_view() # 
            self.hhinit = True
        
    
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
    def measure(self):
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
        
    @pyqtSlot(str, int, int, bool)
    def prepare_minflux(self, fname, acqtime, n, PickAndDestroyFlag):
        
        
        
        print(datetime.now(), ' [tcspc] preparing minflux measurement')
    
        t0 = time.time()
        
        self.setODSignal.emit()
        
        self.currentfname = tools.getUniqueName(fname)

        self.prepare_hh()
        self.hh.tacq = acqtime * n * 1000 # TO DO: correspond to GUI !!!
        print("hhtacq = ",self.hh.tacq)

        
        print(' [tcspc] self.hh.tacq', self.hh.tacq)
        
        #self.hh.lib.HH_SetBinning(ctypes.c_int(0), 
        #                       ctypes.c_int(1)) # TO DO: fix this in a clean way (1 = 8 ps resolution)
   
        t1 = time.time()
        
        print(datetime.now(), '[tcspc] preparing the HH measurement took {} s'.format(t1-t0))
        
        if PickAndDestroyFlag == True:
            self.PickAndDestroyflag = True
        else:
            self.PickAndDestroyflag = False
            
            
            
            
        #Changed
        self.init_view()
#        self.syncTimer.start(self.syncTime)        
        
        
    @pyqtSlot()
    def measure_minflux(self):
        
        self.toggleShutterSignalMINFLUX.emit(True)
        self.hh.ctcdone = False #HACK: !!!
        self.hh.startTTTR()
        
        #np.savetxt(self.currentfname + '.txt', [])
        
        while self.hh.ctcdone == False:
            pass

        print("measured minflux")
        self.tcspcDoneSignal.emit()
        
        self.toggleShutterSignalMINFLUX.emit(False)
        #done xyzsignal
        self.export_data()
        
        
    
    
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
    def measure_fastscan(self):
        
        
        self.hh.startTTTR()
        
        
        while self.hh.ctcdone == False:
            pass
        #print(self.hh.integralCount)
        self.export_data(noscanFlag = False)
        #print(self.absTime)
        time.sleep(0.005) 
        self.scanTCSPCdata.put([True, self.absTime, self.relTime]) 
        
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

        self.hh.closehh()
        print("[tcspc] HH closed")
        self.syncTimer.stop()

    
    
    
    def stop_measure(self):
        self.hhinit = False
        self.hh.ctcdone = True 
        self.hh.stopTTTR()
        self.hh.closehh()
        self.toggleShutterSignal.emit(False)
        self.toggleShutterSignalMINFLUX.emit(False)
        print(datetime.now(), '[tcspc] stop measure function')
    
    def liveplot(self):
        
        
        
#        self.countsflat = [val for sublist in self.hh.counts for val in sublist]
        self.countsflat = sum(self.hh.counts[self.last_plot_index::],[]) #make sublist?
        self.last_plot_index = len(self.hh.counts)
        if self.cts0 > 0:
            globRes = 1 / self.cts0  # Syncrate in kHz
        else: 
            globRes = 0
        timeRes = self.hh.resolution * 1e-12 # time resolution in s

        relTime, absTime, channel_array = Read_PTU.convertHT3(self.countsflat)
        
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
        data[2,:] = channel_array[self.absTime != 0]
        print(data[0, :])
        if data[0, :] != []:
            self.plotDataSignal.emit(data[0, :], data[1, :], True)
    
    def export_data(self, noscanFlag = True):
        #self.currentfname = tools.getUniqueName(self.fname)
#        self.reset_data()
        self.countsflat = [val for sublist in self.hh.counts for val in sublist]
        
        #print(datetime.now(), '[tcspc] opened {} file'.format(self.currentfname))
        

        #globRes = 2.5e-8  # in ns, corresponds to sync @40 MHz
        if self.cts0 > 0:
            globRes = 1 / self.cts0  # Syncrate in kHz
        else: 
            globRes = 0
        timeRes = self.hh.resolution * 1e-12 # time resolution in s

        relTime, absTime, channel_array = Read_PTU.convertHT3(self.countsflat)
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
        data[2,:] = channel_array[self.absTime != 0]
        
        if noscanFlag:
            
            self.plotDataSignal.emit(data[0, :], data[1, :], False) #TODO: changed        
            np.savetxt(filename, data.T) # transpose for easier loading
        
            print(datetime.now(), '[tcspc] tcspc data exported')
        else:
            if data[1,:] == np.array([]):
                self.plotDataSignal.emit(np.array[0],np.array[1], np.array[2], True)
            else:
                self.plotDataSignal.emit(data[0, :], data[1, :], True)
            #np.savetxt(self.scanfolder + "\\"+ "line_"+ str(self.scannumber) + ".txt", data.T) # transpose for easier loading
            self.scannumber =  self.scannumber + 1
            
        
        if self.PickAndDestroyflag == True:
            self.tcspcPickAndDestroySignal.emit()
            
            
            
    @pyqtSlot(list)
    def get_frontend_parameters(self, paramlist):
        
        print(datetime.now(), '[tcspc] got frontend parameters')

        self.fname = paramlist[0]
        self.resolution = paramlist[1]
        self.tacq = paramlist[2]
        self.folder = paramlist[3]      
        
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
        self.hhinit = False
        print("[tcspc] HH closed")
        self.syncTimer.stop()


if __name__ == '__main__':

    app = QtGui.QApplication([])
    app.setStyle(QtGui.QStyleFactory.create('fusion'))
#    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    hh = hydraharp.HydraHarp400()
    scanTCSPCdata = queue.Queue()
    
    worker = Backend(hh, scanTCSPCdata)
    gui = Frontend()
    
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
    