# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:35:15 2019

Measurement Widget for MINFLUX

@author: Jonas Zaehringer

"""

import numpy as np
import time
import os
from datetime import date, datetime

#from threading import Thread

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from tkinter import Tk, filedialog
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QMessageBox

#import tools as tools
#
#import tools_MINFLUX as tools_MINFLUX



import tools.tools as tools
import tools.tools_MINFLUX as tools_MINFLUX
π = np.pi

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

    

class Frontend(QtGui.QFrame):
    
    filenameSignal = pyqtSignal(str)
    paramSignal = pyqtSignal(dict)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.setup_gui()
        
        self.emit_param()
        
        self.x0_final = [0, 0, 0, 0]
        self.y0_final = [0, 0, 0, 0]
        
        
    def emit_filename(self):  
        
        filename = os.path.join(self.folderEdit.text(),
                                self.filenameEdit.text())
        
        today = str(date.today()).replace('-', '')
        filename = tools.getUniqueName(filename + '_' + today)
        
        self.filenameSignal.emit(filename)
             
        
        
        
        
    def emit_param(self):
        
        params = dict()
        
        filename = os.path.join(self.folderEdit.text(),
                                self.filenameEdit.text())
        
        params['measType'] = self.measType.currentText()
        
        params['filename'] = filename
        params['patternType'] = self.patternType.currentText()
        params['patternLength'] = float(self.lengthEdit.text())
        params['preBleach'] = self.PreBleachBox.isChecked()
        params['LiveMINFLUX'] = self.LiveMINFLUXBox.isChecked()
        params['LiveMINFLUXmove'] = self.LiveMINFLUXmoveBox.isChecked()
        params['pulse_bin'] = np.array(self.pulseBinEdit.text().split(' '),
                                                  dtype=np.float32)
        params['startingColor'] = self.startingColor.currentText()
        params['aotfType'] = self.aotfType.currentText()
        
        try:
            params['acqtime'] = float(self.acqtimeEdit.text())
            params['alexTime'] = float(self.alexEdit.text())
            params['sequentialTime'] = float(self.sequentialt1Edit.text())
            
            
            self.paramSignal.emit(params)
        except(RuntimeError, ValueError):
            QMessageBox.about(self, "[MINFLUX] Warning", "Input error. Please check scan inputs.")
        
        
        
        

        
        
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
        
    def toggle_parameters(self):
        
        if self.measType.currentText() == 'Predefined positions':
            
            self.patternType.show()
            self.lengthLabel.show()
            self.lengthEdit.show()
      
        else:
            
            self.patternType.hide()
            self.lengthLabel.hide()
            self.lengthEdit.hide()
    
    
    def toggle_aotf_parameters(self):
        
        
        if self.aotfType.currentText() == 'alex':
            self.alexLabel.show()
            self.alexEdit.show()
            self.startingColorLabel.show()
            self.startingColor.show()
            self.sequentialt1Label.hide()
            self.sequentialt1Edit.hide()
            
        elif self.aotfType.currentText() == 'sequential':
            self.alexLabel.hide()
            self.alexEdit.hide()
            self.startingColorLabel.show()
            self.startingColor.show()
            self.sequentialt1Label.show()
            self.sequentialt1Edit.show()
        else:
            
            self.alexLabel.hide()
            self.alexEdit.hide()
            self.startingColorLabel.hide()
            self.startingColor.hide()
            self.sequentialt1Label.hide()
            self.sequentialt1Edit.hide()
        
           
    
    
    @pyqtSlot(np.ndarray, np.ndarray)
    def prepare_MINFLUXplot(self, x0_final , y0_final):
        
        self.x0_final = x0_final
        self.y0_final = y0_final
        self.canvas.axes.set_aspect('equal', 'box')
        marker = ['yo', 'ro', 'ko', 'bo'] 
        print("prepare MINFLUX plot")
        label_list = ["200", "455", "710", "965"]
        for k in np.arange(4):
            self.canvas.axes.plot(x0_final[k], y0_final[k], marker[k], markersize = 10, label = label_list[k])
    #        ax_EBP.set_aspect('equal', 'box')
        self.canvas.axes.set_xlabel('x [nm]')
        self.canvas.axes.set_ylabel('y [nm]')
        self.canvas.axes.set_title('green MINFLUX localization')
        self.canvas.axes.set_xlim(40, 170)
        self.canvas.axes.set_ylim(170,25)
        self.canvas.draw()

    @pyqtSlot(np.ndarray, np.ndarray)
    def prepare_MINFLUXplot2(self, x0_final , y0_final):
        
        self.x0_final2 = x0_final
        self.y0_final2 = y0_final
        self.canvas2.axes.set_aspect('equal', 'box')
        marker = ['yo', 'ro', 'ko', 'bo'] 
        print("prepare MINFLUX plot")
        label_list = ["200", "455", "710", "965"]
        for k in np.arange(4):
            self.canvas2.axes.plot(x0_final[k], y0_final[k], marker[k], markersize = 10, label = label_list[k])
    #        ax_EBP.set_aspect('equal', 'box')
        self.canvas2.axes.set_xlabel('x [nm]')
        self.canvas2.axes.set_ylabel('y [nm]')
        self.canvas2.axes.set_title('red MINFLUX localization')

        self.canvas2.axes.set_xlim(40, 170)
        self.canvas2.axes.set_ylim(170,25)
        self.canvas2.draw()
                
        
        
    def clearMINFLUXplot(self):
        
        
        print("clearing MINFLUX plot")
        self.canvas.axes.cla()
        self.canvas.draw()
        
        self.prepare_MINFLUXplot(self.x0_final, self.y0_final)
        
        
        
        
    def clearMINFLUXplot2(self):
        
        
        print("clearing MINFLUX plot2")
        self.canvas2.axes.cla()
        self.canvas2.draw()
        
        self.prepare_MINFLUXplot2(self.x0_final2, self.y0_final2)
        
    @pyqtSlot(np.ndarray)        
    def plotMINFLUX(self, indrec):    
#        markercolor = ['k']
        
#        print("plot")
        self.canvas.axes.plot(indrec[0], indrec[1],'o', color='k',marker='*', markersize = 5)
#        ax_EBP.plot(indrec[k,0], indrec[k,1],'o', color=markercolor[k],marker='*', markersize = 5)
        
        self.canvas.draw()
    

    @pyqtSlot(np.ndarray)        
    def plotMINFLUX2(self, indrec):    
#        markercolor = ['k']
        
#        print("plot")
        self.canvas2.axes.plot(indrec[0], indrec[1],'o', color='k',marker='*', markersize = 5)
#        ax_EBP.plot(indrec[k,0], indrec[k,1],'o', color=markercolor[k],marker='*', markersize = 5)
        
        self.canvas2.draw()


    @pyqtSlot(np.ndarray)        
    def untickLiveMINFLUX(self):    
        self.LiveMINFLUXBox.setChecked(False)
        

    
    
    def setup_gui(self):
        
        self.setWindowTitle('MINFLUX measurement')

        grid = QtGui.QGridLayout()

        self.setLayout(grid)
        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)

        grid.addWidget(self.paramWidget, 0, 0)
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)
        
        
        
        
        self.measLabel = QtGui.QLabel('Measurement type')
        
        self.measType = QtGui.QComboBox()
        self.measTypes = ['Standard', 'Predefined positions']
        self.measType.addItems(self.measTypes)
        
        self.patternType = QtGui.QComboBox()
        self.patternTypes = ['Grid','zRow', 'Row', 'Square', 'Triangle', 'Grid_6x6']
        self.patternType.addItems(self.patternTypes)
    

        
        
        self.lengthLabel = QtGui.QLabel('L [nm]')
        self.lengthEdit = QtGui.QLineEdit('30')
        
        self.patternType.hide()
        self.lengthLabel.hide()
        self.lengthEdit.hide()
        
        self.acqtimeLabel = QtGui.QLabel('Acq time [s]')
        self.acqtimeEdit = QtGui.QLineEdit('5')
        
        self.startButton = QtGui.QPushButton('Start')
        self.stopButton = QtGui.QPushButton('Stop')
        
        self.PreBleachBox = QtGui.QCheckBox('Prebleach')
        self.LiveMINFLUXBox = QtGui.QCheckBox('Live MINFLUX')
        self.LiveMINFLUXmoveBox = QtGui.QCheckBox('MINFLUX recenter')

        self.pulseBinLabel = QtGui.QLabel('Pulse Bin start')
        self.pulseBinEdit = QtGui.QLineEdit('0.1 12.6 25.2 37.3')
        
        
        self.aotfType = QtGui.QComboBox()
        self.aotfTypes = ['green', 'red', 'alex', 'sequential']
        self.aotfType.addItems(self.aotfTypes)
        
        self.alexLabel = QtGui.QLabel('color time / s')
        self.alexEdit = QtGui.QLineEdit('1')
        
        
        self.alexLabel.hide()
        self.alexEdit.hide()
        
        self.startingColorLabel = QtGui.QLabel('starting color')
        self.startingColor = QtGui.QComboBox()
        self.startingColors = ['green','red']
        self.startingColor.addItems(self.startingColors)
        
        
        self.sequentialt1Label = QtGui.QLabel('time first color / s')
        self.sequentialt1Edit = QtGui.QLineEdit('5')
        
        
        self.startingColorLabel.hide()
        self.startingColor.hide()
        self.sequentialt1Label.hide()
        self.sequentialt1Edit.hide()
        

        
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas2 = MplCanvas(self, width=5, height=4, dpi=100)

        

        

        subgrid.addWidget(self.measLabel, 0, 0, 1, 2)
        subgrid.addWidget(self.measType, 1, 0, 1, 2)
        
        subgrid.addWidget(self.patternType, 2, 0, 1, 2)
#        subgrid.addWidget(self.patternTypes, 3, 0)
        
        subgrid.addWidget(self.lengthLabel, 4, 0, 1, 1)
        subgrid.addWidget(self.lengthEdit, 4, 1, 1, 1)
        
        subgrid.addWidget(self.acqtimeLabel, 6, 0, 1, 1)
        subgrid.addWidget(self.acqtimeEdit, 6, 1, 1, 1)
        subgrid.addWidget(self.startButton, 7, 0, 1, 2)
        subgrid.addWidget(self.stopButton, 8, 0, 1, 2)
        subgrid.addWidget(self.PreBleachBox, 9, 0, 1, 2)
        subgrid.addWidget(self.LiveMINFLUXBox, 10, 0, 1, 2)
        subgrid.addWidget(self.LiveMINFLUXmoveBox, 11, 0, 1, 2)
        
        subgrid.addWidget(self.pulseBinLabel, 12, 0, 1, 1)
        subgrid.addWidget(self.pulseBinEdit, 13, 0, 1, 1)
        
        subgrid.addWidget(self.aotfType, 14, 0, 1, 1)
        
        subgrid.addWidget(self.alexLabel, 15, 0, 1, 1)
        subgrid.addWidget(self.alexEdit, 15, 1, 1, 1)
        
        subgrid.addWidget(self.startingColorLabel, 16, 0, 1, 1)
        subgrid.addWidget(self.startingColor, 16, 1, 1, 1)
        subgrid.addWidget(self.sequentialt1Label, 15, 0, 1, 1)
        subgrid.addWidget(self.sequentialt1Edit, 15, 1, 1, 1)
        
        subgrid.addWidget(self.canvas, 17, 0, 1, 1)
        subgrid.addWidget(self.canvas2, 18, 0, 1, 1)
        
        
        # file/folder widget
        
        self.fileWidget = QtGui.QFrame()
        self.fileWidget.setFrameStyle(QtGui.QFrame.Panel |
                                      QtGui.QFrame.Raised)
        
        self.fileWidget.setFixedHeight(400)
        self.fileWidget.setFixedWidth(400)
        
        # folder
        
        # TO DO: move this to backend
        
        today = str(date.today()).replace('-', '')
        root = r'C:\\Data\\'
        folder = root + today
        
        try:  
            os.mkdir(folder)
        except OSError:  
            print(datetime.now(), '[minflux] Directory {} already exists'.format(folder))
        else:  
            print(datetime.now(), '[minflux] Successfully created the directory {}'.format(folder))

        self.folderLabel = QtGui.QLabel('Folder')
        self.folderEdit = QtGui.QLineEdit(folder)
        self.browseFolderButton = QtGui.QPushButton('Browse')
        self.browseFolderButton.setCheckable(True)
        self.filenameLabel = QtGui.QLabel('File name')
        self.filenameEdit = QtGui.QLineEdit('minflux')
        
        
        self.clearButton = QtGui.QPushButton('Clear green plot')
        self.clearButton2 = QtGui.QPushButton('Clear red plot')
#        self.clearbutton.setCheckable(True)
        
        grid.addWidget(self.fileWidget, 1, 0)
        
        file_subgrid = QtGui.QGridLayout()
        self.fileWidget.setLayout(file_subgrid)
        
        file_subgrid.addWidget(self.filenameLabel, 0, 0, 1, 2)
        file_subgrid.addWidget(self.filenameEdit, 1, 0, 1, 2)
        file_subgrid.addWidget(self.folderLabel, 2, 0, 1, 2)
        file_subgrid.addWidget(self.folderEdit, 3, 0, 1, 2)
        file_subgrid.addWidget(self.browseFolderButton, 4, 0)
        file_subgrid.addWidget(self.clearButton, 6, 0)
        file_subgrid.addWidget(self.clearButton2, 7, 0)
        
        self.measType.currentIndexChanged.connect(self.toggle_parameters)
        self.aotfType.currentIndexChanged.connect(self.toggle_aotf_parameters)
        
        self.folderEdit.textChanged.connect(self.emit_param)
        
        
        
        
        self.filenameEdit.textChanged.connect(self.emit_param)
        self.acqtimeEdit.textChanged.connect(self.emit_param)
        self.lengthEdit.textChanged.connect(self.emit_param)
        self.patternType.activated.connect(self.emit_param)
        
        
        
        
        
        self.PreBleachBox.stateChanged.connect(self.emit_param)
        self.LiveMINFLUXBox.stateChanged.connect(self.emit_param)
        self.LiveMINFLUXmoveBox.stateChanged.connect(self.emit_param)
        self.pulseBinEdit.textChanged.connect(self.emit_param)
        
        
        self.startingColor.currentIndexChanged.connect(self.emit_param)
        self.aotfType.currentIndexChanged.connect(self.emit_param)
        
        self.alexEdit.textChanged.connect(self.emit_param)
        self.sequentialt1Edit.textChanged.connect(self.emit_param)
        
        
        self.clearButton.clicked.connect(self.clearMINFLUXplot)
        self.clearButton2.clicked.connect(self.clearMINFLUXplot2)

        self.emit_param()
        
    def make_connection(self, backend):
        
        backend.prepareMINFLUXplotSignal.connect(self.prepare_MINFLUXplot)
        backend.plotMINFLUXSignal.connect(self.plotMINFLUX)
        backend.prepareMINFLUXplotSignal2.connect(self.prepare_MINFLUXplot2)
        backend.plotMINFLUXSignal2.connect(self.plotMINFLUX2)
        backend.clearSignal.connect(self.clearMINFLUXplot)
        backend.clearSignal2.connect(self.clearMINFLUXplot2)
        
#        pass

class Backend(QtCore.QObject):
    
    
    untickLiveMINFLUXSignal = pyqtSignal()
    
    tcspcPrepareSignal = pyqtSignal(str, float, int, bool, bool,  np.ndarray)
    tcspcStartSignal = pyqtSignal()
    setODSignal = pyqtSignal()
    # xyzStartSignal = pyqtSignal()
    xyzEndSignal = pyqtSignal(str)
    
    preBleachSignal = pyqtSignal()
    
    TCSPCstopSignal = pyqtSignal()
    plotMINFLUXSignal = pyqtSignal(np.ndarray)
    prepareMINFLUXplotSignal = pyqtSignal(np.ndarray, np.ndarray)
    
    plotMINFLUXSignal2 = pyqtSignal(np.ndarray)
    prepareMINFLUXplotSignal2 = pyqtSignal(np.ndarray, np.ndarray)
    
    moveToSignal = pyqtSignal(np.ndarray, np.ndarray)
    
    paramSignal = pyqtSignal(np.ndarray, np.ndarray, float)
    
    initAOTFSignal = pyqtSignal(str, float, str)
    stopALEXSignal = pyqtSignal()
    
    clearSignal = pyqtSignal()
    clearSignal2 = pyqtSignal()
    
    def __init__(self, pidevice, MINFLUXdata, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.i = 0 # counter
        self.n = 1
        self.pidevice = pidevice
        self.pattern = np.array([0, 0])
        self.MINFLUXdata = MINFLUXdata
        self.measTimer = QtCore.QTimer()
        self.measTimer.timeout.connect(self.loop)
        
        
        
        
    @pyqtSlot(np.ndarray)    
    def get_ROI_center(self, center):
        #TODO:get working
#        ''' 
#        Connection: [scan] ROIcenterSignal
#        Description: gets the position selected by the user in [scan]
#        '''
        
#        self.r0 = center[0:2]
#        self.update_param()
#        
        time.sleep(0.4)
        
        print(datetime.now(), '[minflux] got ROI center')
        
        #self.xyzStartSignal.emit() #commented out
        
    @pyqtSlot(dict)
    def get_frontend_param(self, params):
        
        """
        Connection: [frontend] paramSignal
        """
        #TODO give params at init
        try:
            print("MINFLUX: getting frontend parameters")
            self.acqtime = params['acqtime']
            self.measType = params['measType']
            today = str(date.today()).replace('-', '')
            self.filename = params['filename'] #+ '_' + today
            
            self.patternType = params['patternType']
            self.patternLength = float(params['patternLength'])/1000 # in micrometer
            self.preBleach = params['preBleach']
            
            self.LiveMINFLUX = params['LiveMINFLUX']
            self.LiveMINFLUXmove = params['LiveMINFLUXmove']
            
            self.pulse_bin = np.array(params['pulse_bin'])
            
            self.aotfType = params['aotfType']
            self.startingColor = params['startingColor']
            self.alexTime = params['alexTime']
            self.sequentialTime = params['sequentialTime']
            
            
            
            
            self.update_param()
        except(RuntimeError, ValueError):
            QMessageBox.about(self, "[LiveMINFLUX] Warning", "input error. Please check scan inputs.")
        
        
    def update_param(self):
        self.taus = np.array([0])
        
        for i in self.pulse_bin:
            print(i)
            self.taus = np.append(self.taus, i)
            self.taus = np.append(self.taus, i+11)
            
        #LOAD EBP
        if self.LiveMINFLUX:
            middle_beam = 0
            middle_beam2 = 3
            
            
            
            try:
        
                if self.aotfType == "green":
                    self.PSF, self.x0_final, self.y0_final = tools_MINFLUX.initEBP(middle_beam, 'g')
                    self.prepareMINFLUXplotSignal.emit(np.array(self.x0_final), np.array(self.y0_final))
                if self.aotfType == "red":
                    self.PSF2, self.x0_final2, self.y0_final2 = tools_MINFLUX.initEBP(middle_beam2, 'r')
                    self.prepareMINFLUXplotSignal2.emit(np.array(self.x0_final2), np.array(self.y0_final2))
                if self.aotfType == "alex" or self.aotfType == "sequential":
                    self.PSF, self.x0_final, self.y0_final = tools_MINFLUX.initEBP(middle_beam, 'g')
                    self.PSF2, self.x0_final2, self.y0_final2 = tools_MINFLUX.initEBP(middle_beam2, 'r')
                    
                    self.prepareMINFLUXplotSignal.emit(np.array(self.x0_final), np.array(self.y0_final))
                    self.prepareMINFLUXplotSignal2.emit(np.array(self.x0_final2), np.array(self.y0_final2))
                    
                
            except(RuntimeError, ValueError, IOError, FileNotFoundError):
#                QMessageBox.about(self, "[LiveMINFLUX] Warning", "input error. Please check scan inputs.")
                print("no data found")
                
                
                
                self.LiveMINFLUX = False 
                self.untickLiveMINFLUXSignal.emit()
        
            
            
        l = self.patternLength
        h = np.sqrt(3/2)*l
        
        currentXposition = self.pidevice.qPOS(1)[1]
        currentYposition = self.pidevice.qPOS(2)[2]
                
        self.r0 = np.array([currentXposition, currentYposition])
        
        if self.aotfType == "green":
            self.initAOTFSignal.emit("green", 1000, "green")
        if self.aotfType == "red":
            self.initAOTFSignal.emit("red", 1000, "red")
        if self.aotfType == "alex":
            self.initAOTFSignal.emit("alex", self.alexTime, self.startingColor)
        if self.aotfType == "sequential":
            self.initAOTFSignal.emit("sequential", self.sequentialTime, self.startingColor)
            

        
        
        if self.measType == 'Predefined positions':
        
            if self.patternType == 'Row':
                
                self.pattern = np.array([[0, -2*l], [0, -l], [0, 0], [0, l], [0, 2*l]])
                
                print('ROW')
                
                
                
                

                
            if self.patternType == 'Square':
                
                self.pattern = np.array([[0, 0], [l/2, l/2], [l/2, -l/2],
                                        [-l/2, -l/2], [-l/2, l/2]])
            
    
                print('SQUARE') 
    
            if self.patternType == 'Grid_6x6':
            
                self.pattern = np.array([[0, 0],  [-3, 3], [-3, 2], [-3, 1],[-3, 0],[-3, -1],
                                         [-3, -2],[-3, -3], [-2, -3], [-2, -2], [-2, -1],[-2, 0],
                                         [-2, 1],[-2, 2], [-2, 3], [-1, 3], [-1, 2], [-1, 1],
                                         [-1, 0], [-1, -1],  [-1, -2],  [-1, -3],[0, -3],[0, -2],
                                         [0, -1], [0, 0], [0, 1], [0, 2], [0, 3], [1, 3],[1, 2],
                                         [1, 1], [1, 0], [1, -1], [1, -2], [1, -3], [2, -3],[2, -2],
                                         [2, -1], [2, 0], [2, 1], [2, 2], [2, 3] ,
                                          [3, +3], [3, +2], [3, +1],[3, 0],[3, -1],
                                         [3, -2],[3, -3],[0, 0]])*l
    
    
    
            if self.patternType == 'Grid':
                
#                self.pattern = np.array([[0, 0], [-2, -2], [-2, -1],[-2, 0],
#                                         [-2, 1],[-2, 2], [-1, 2], [-1, 1],
#                                         [-1, 0], [-1, -1],  [-1, -2],[0, -2],
#                                         [0, -1], [0, 0], [0, 1], [0, 2], [1, 2],
#                                         [1, 1], [1, 0], [1, -1], [1, -2], [2, -2],
#                                         [2, -1], [2, 0], [2, 1], [2, 2]  ])*l
##                #funzt
#                self.pattern = np.array([[0, 0], [-2, -2], [-1, -2], [-0, -2], [1, -2],
#                                         [2, -2], [2, -1], [1, -1], [0, -1], [-1, -1],
#                                         [-2, -1], [-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0],
#                                         [2, 1], [1, 1], [0, 1], [-1, 1], [-2, 1],
#                                         [-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2], [0,0] ])*l
                        
    
                self.pattern = np.array([[1, -1], [0, -1], [-1, -1],
                                          [-1, 0], [0, 0], [1, 0], [1, 1], [0, 1],
                                          [-1, 1],  [0,0] ])*l    
                print('GRID')
        
            if self.patternType == 'Triangle':
                
                self.pattern = np.array([[0, 0], [0, (2/3)*h], [l/2, -(1/3)*h],
                                        [-l/2, -(1/3)*h]])
        
                print('TRIANGLE')
                
            
            
            
            if self.patternType == 'zRow':
                
                self.pattern = np.array([-2*l, -l, 0, l , 2*l, 0])
                self.r = self.pidevice.qPOS(3)[3] + self.pattern
                print('zROW')
                
            else:
                self.r = self.r0 + self.pattern
            
            self.n = np.shape(self.r)[0]
                
        else:
            
            self.r = self.r0
            self.n = 1
    
    

        
        
#        print('[minflux] self.pattern', self.pattern)
#        print(datetime.now(), '[minflux] self.r', self.r)
#        print(datetime.now(), '[minflux] self.r.shape', self.r.shape)
                
    def start(self):
        today = str(date.today()).replace('-', '')
        
        if self.measType == 'Standard':
            self.curname = tools.getUniqueNameptu(self.filename +  '_' + self.aotfType , today)
        elif self.measType == 'Predefined positions':
            self.curname = tools.getUniqueNameptu(self.filename + '_' + self.aotfType +  '_' + self.patternType , today)
        else:
            self.curname = tools.getUniqueNameptu(self.filename , today)
        print(self.curname)
        
        if self.LiveMINFLUX:
            if self.aotfType == "green":
                self.clearSignal.emit()
            if self.aotfType == "red":
                self.clearSignal2.emit()
            if self.aotfType == "alex" or self.aotfType == "sequential":
                self.clearSignal2.emit()
                self.clearSignal.emit()
            
        

        
        self.pidevice.VEL("1", 10000)
        self.pidevice.VEL("2", 10000)
        self.pidevice.VEL("3", 5000)
        
        self.i = 0
        self.movecounter = 0
        
        
        currentXposition = self.pidevice.qPOS(1)[1]
        currentYposition = self.pidevice.qPOS(2)[2]
        currentZposition = self.pidevice.qPOS(3)[3]
        
        if self.preBleach:
            self.preBleachSignal.emit()
        self.initialPos = np.array([currentXposition, currentYposition, currentZposition])
        # self.curname = self.filename + '_' +today
        
        
        now = time.strftime("%c")
        tools.saveMINFLUXconfig(self, now, self.curname)
        

        
        
        
        if self.measType == 'Standard':
            
            print('[minflux] self.n, self.acqtime', self.n, self.acqtime)
            self.tcspcPrepareSignal.emit(self.curname, self.acqtime, self.n, False, self.LiveMINFLUX, self.taus) # signal emitted to tcspc module to start the measurement #TODO do once
            phtime = 4.0  # in s, it takes 4 s for the PH to start the measurement, TO DO: check if this can be reduced (send email to Picoquant, etc)
            self.setODSignal.emit()
            time.sleep(phtime)
            # self.xyzStartSignal.emit()
            self.tcspcStartSignal.emit()
            self.t0 = time.time()
            
        if self.measType == 'Predefined positions':
            
            print(datetime.now(), '[minflux] Predefined positions')
            print('[minflux] self.n, self.acqtime', self.n, self.acqtime)
            
            self.update_param()
            time.sleep(0.2)
            self.tcspcPrepareSignal.emit(self.curname, self.acqtime, self.n, False, self.LiveMINFLUX, self.taus) # signal emitted to tcspc module to start the measurement
            phtime = 8.0  # in s, it takes 4 s for the PH to start the measurement, TO DO: check if this can be reduced (send email to Picoquant, etc)
            self.setODSignal.emit()
            time.sleep(phtime)
            # self.xyzStartSignal.emit()
            self.tcspcStartSignal.emit()
            self.t0 = time.time()
            self.measTimer.start(0)
            
            
        if self.measType == 'PickAndDestroy':
            
            
            print('[minflux] PickAndDestroy self.n, self.acqtime', self.n, self.acqtime)
            self.tcspcPrepareSignal.emit(self.curname, self.acqtime, self.n, True, self.LiveMINFLUX, self.taus) # signal emitted to tcspc module to start the measurement #TODO do once
            phtime = 1.0  # in s, it takes 4 s for the PH to start the measurement, TO DO: check if this can be reduced (send email to Picoquant, etc)
            time.sleep(phtime)
            # self.xyzStartSignal.emit()
            self.tcspcStartSignal.emit()
            self.t0 = time.time()
            
            
            
            
            
    @pyqtSlot(dict) 
    def Destroy_PickAndDestroy(self,params):
        print("starting PnD")
        self.measType = 'Standard'
        
        self.acqtime = params['acqtime']
        self.measType = params['measType'] #PickAndDestroy
        
        self.filename = params['filename'] 
        
               
        
        
        self.start()
        
        
        
            
    
    def loop(self):
        
        now = time.time()
        
        if (now - (self.t0 + self.i * self.acqtime) + self.acqtime) > self.acqtime:
            
            print(datetime.now(), '[minflux] loop', self.i)
            
            
            if self.patternType == "zRow":
                self.moveToSignal.emit(np.array([self.r[self.i]]), np.array([self.pattern[self.i]]))    
            else:
                self.moveToSignal.emit(np.array([self.r[self.i]]), self.pattern[self.i])    

        
            self.i += 1
            
            if self.i == self.n :
                                
                self.stoploop()
                
                print(datetime.now(), '[minflux] measurement ended')
    
    def stoploop(self):
        self.stopALEXSignal.emit()
        self.measTimer.stop()
#        self.TCSPCstopSignal.emit()

            
    def stop(self):
        self.stopALEXSignal.emit()
        self.measTimer.stop()
        self.TCSPCstopSignal.emit()
        
    @pyqtSlot(np.ndarray) 
    def LiveMINFLUXanalysis(self, fractions):
#        print("MINFLUX", fractions)
#        PSF, x0_final, y0_final = tools_MINFLUX.initEBP()
        
#        fractions[0] = 1
        
        if self.aotfType == "green":
            [indrec, pos] = tools_MINFLUX.pos_MINFLUX(fractions, self.PSF, 4)
        if self.aotfType == "red":
            [indrec, pos] = tools_MINFLUX.pos_MINFLUX(fractions, self.PSF2, 4)
        if self.aotfType == "alex" or self.aotfType == "sequential":
            [indrec, pos] = tools_MINFLUX.pos_MINFLUX(fractions, self.PSF, 4)
            [indrec2, pos2] = tools_MINFLUX.pos_MINFLUX(fractions, self.PSF2, 4)
                
                
        
            
        
        #do analysis
#        print(indrec, pos)
        
        maxdistance = 50 #nm
        movedistance = 1/2**0.5 *((indrec[0]-100)**2 + (indrec[1]-100)**2)**0.5
        
        
        if (self.LiveMINFLUXmove)  and (self.movecounter % 2 == 0) :
            print(movedistance)
            if  movedistance < maxdistance:
                
                
                currentXposition = self.pidevice.qPOS(1)[1] #um
                currentYposition = self.pidevice.qPOS(2)[2] #um
                
                
                
                newx = currentXposition - (indrec[0] - 100) /1000 #um
                newy = currentYposition + (indrec[1] - 100) /1000 #um
                
                self.moveToSignal.emit(np.array([[newx,newy]]), np.array([]))
            else:
                print("[MINFLUX]: Movement is ", str(movedistance) , "nm and out of savedistance of ", str(maxdistance))
                safetyfactor = 0.5
                
                currentXposition = self.pidevice.qPOS(1)[1] #um
                currentYposition = self.pidevice.qPOS(2)[2] #um
                
                
                
                newx = currentXposition - safetyfactor * (indrec[0] - 100) /1000 #um
                newy = currentYposition + safetyfactor * (indrec[1] - 100) /1000 #um
                
                self.moveToSignal.emit(np.array([[newx,newy]]), np.array([]))
            
            self.movecounter += 1
            
        if self.aotfType == "green":
            self.plotMINFLUXSignal.emit(np.array(indrec))
        if self.aotfType == "red":
            self.plotMINFLUXSignal2.emit(np.array(indrec))
        if self.aotfType == "alex" or self.aotfType == "sequential":
            self.plotMINFLUXSignal.emit(np.array(indrec))
            self.plotMINFLUXSignal2.emit(np.array(indrec2))
        
        
    @pyqtSlot()  
    def get_tcspc_done_signal(self):
        
        """
        Connection: [tcspc] tcspcDoneSignal
        """
        
#        self.xyzEndSignal.emit(self.curname) #TODO:check if off bad
        self.stopALEXSignal.emit()
    def make_connection(self, frontend):
        
        frontend.paramSignal.connect(self.get_frontend_param)
#        frontend.filenameSignal.connect(self.get_filename)
        frontend.startButton.clicked.connect(self.start)
        frontend.stopButton.clicked.connect(self.stop)


if __name__ == '__main__':

    app = QtGui.QApplication([])
    app.setStyle(QtGui.QStyleFactory.create('fusion'))
#    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    hh = 0
    scanTCSPCdata = 0
    gui = Frontend()
    worker = Backend(hh, scanTCSPCdata)
    workerThread = QtCore.QThread()
    workerThread.start()
    worker.moveToThread(workerThread)
#    worker.syncTimer.moveToThread(workerThread)
#    worker.syncTimer.timeout.connect(worker.update_view) # TODO connect

    
    
    worker.make_connection(gui)
    gui.make_connection(worker)

    gui.setWindowTitle('Time-correlated single-photon counting')
    gui.show()

    app.exec_()
    