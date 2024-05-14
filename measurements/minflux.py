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

import tools.tools as tools


π = np.pi

class Frontend(QtGui.QFrame):
    
    filenameSignal = pyqtSignal(str)
    paramSignal = pyqtSignal(dict)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.setup_gui()
        
        self.emit_param()
        
        
        
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
        params['acqtime'] = int(self.acqtimeEdit.text())
        params['filename'] = filename
        params['patternType'] = self.patternType.currentText()
        params['patternLength'] = float(self.lengthEdit.text())
        params['preBleach'] = self.PreBleachBox.isChecked()
        
        self.paramSignal.emit(params)
        
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
        self.patternTypes = ['Row', 'Square', 'Triangle', 'Grid']
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
        
        
        # file/folder widget
        
        self.fileWidget = QtGui.QFrame()
        self.fileWidget.setFrameStyle(QtGui.QFrame.Panel |
                                      QtGui.QFrame.Raised)
        
        self.fileWidget.setFixedHeight(120)
        self.fileWidget.setFixedWidth(150)
        
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
        
        grid.addWidget(self.fileWidget, 0, 1)
        
        file_subgrid = QtGui.QGridLayout()
        self.fileWidget.setLayout(file_subgrid)
        
        file_subgrid.addWidget(self.filenameLabel, 0, 0, 1, 2)
        file_subgrid.addWidget(self.filenameEdit, 1, 0, 1, 2)
        file_subgrid.addWidget(self.folderLabel, 2, 0, 1, 2)
        file_subgrid.addWidget(self.folderEdit, 3, 0, 1, 2)
        file_subgrid.addWidget(self.browseFolderButton, 4, 0)
        
        self.measType.currentIndexChanged.connect(self.toggle_parameters)
        
        self.folderEdit.textChanged.connect(self.emit_param)
        self.filenameEdit.textChanged.connect(self.emit_param)
        self.acqtimeEdit.textChanged.connect(self.emit_param)
        self.lengthEdit.textChanged.connect(self.emit_param)
        self.patternType.activated.connect(self.emit_param)
        
        self.emit_param()
        
    def make_connection(self, backend):
        
#        backend.paramSignal.connect(self.get_backend_param)
        
        pass

class Backend(QtCore.QObject):
    
    tcspcPrepareSignal = pyqtSignal(str, int, int, bool)
    tcspcStartSignal = pyqtSignal()
    setODSignal = pyqtSignal()
    xyzStartSignal = pyqtSignal()
    xyzEndSignal = pyqtSignal(str)
    
    preBleachSignal = pyqtSignal()
    
    TCSPCstopSignal = pyqtSignal()
    
    moveToSignal = pyqtSignal(np.ndarray, np.ndarray)
    
    paramSignal = pyqtSignal(np.ndarray, np.ndarray, int)
    
    def __init__(self, pidevice, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.i = 0 # counter
        self.n = 1
        self.pidevice = pidevice
        self.pattern = np.array([0, 0])
        
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
            self.update_param()
        except(RuntimeError, ValueError):
            print("error")
        
        
    def update_param(self):
        
        l = self.patternLength
        h = np.sqrt(3/2)*l
        
        currentXposition = self.pidevice.qPOS(1)[1]
        currentYposition = self.pidevice.qPOS(2)[2]
                
        self.r0 = np.array([currentXposition, currentYposition])
        
        if self.measType == 'Predefined positions':
        
            if self.patternType == 'Row':
                
                self.pattern = np.array([[0, -2*l], [0, -l], [0, 0], [0, l], [0, 2*l]])
                
                print('ROW')
                
            if self.patternType == 'Square':
                
                self.pattern = np.array([[0, 0], [l/2, l/2], [l/2, -l/2],
                                        [-l/2, -l/2], [-l/2, l/2]])
            
    
                print('SQUARE') 
    
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
                
            self.r = self.r0 + self.pattern
            self.n = np.shape(self.r)[0]
                
        else:
            
            self.r = self.r0
            self.n = 1
    
#        print('[minflux] self.pattern', self.pattern)
#        print(datetime.now(), '[minflux] self.r', self.r)
#        print(datetime.now(), '[minflux] self.r.shape', self.r.shape)
                
    def start(self):
        
        self.pidevice.VEL("1", 10000)
        self.pidevice.VEL("2", 10000)
        self.pidevice.VEL("3", 5000)
        
        self.i = 0
        currentXposition = self.pidevice.qPOS(1)[1]
        currentYposition = self.pidevice.qPOS(2)[2]
        currentZposition = self.pidevice.qPOS(3)[3]
        
        
        self.preBleachSignal.emit()
        self.initialPos = np.array([currentXposition, currentYposition, currentZposition])
        today = str(date.today()).replace('-', '')
        name = tools.getUniqueName(self.filename)+ '_' +today
        
        
        print(name)
        now = time.strftime("%c")
        tools.saveMINFLUXconfig(self, now, name)
        
        
        
        if self.measType == 'Standard':
            
            print('[minflux] self.n, self.acqtime', self.n, self.acqtime)
            self.tcspcPrepareSignal.emit(name, self.acqtime, self.n, False) # signal emitted to tcspc module to start the measurement #TODO do once
            phtime = 4.0  # in s, it takes 4 s for the PH to start the measurement, TO DO: check if this can be reduced (send email to Picoquant, etc)
            self.setODSignal.emit()
            time.sleep(phtime)
            self.xyzStartSignal.emit()
            self.tcspcStartSignal.emit()
            self.t0 = time.time()
            
        if self.measType == 'Predefined positions':
            
            print(datetime.now(), '[minflux] Predefined positions')
            
            self.update_param()
            time.sleep(0.2)
            self.tcspcPrepareSignal.emit(self.filename, self.acqtime, self.n, False) # signal emitted to tcspc module to start the measurement
            phtime = 4.0  # in s, it takes 4 s for the PH to start the measurement, TO DO: check if this can be reduced (send email to Picoquant, etc)
            self.setODSignal.emit()
            time.sleep(phtime)
            self.xyzStartSignal.emit()
            self.tcspcStartSignal.emit()
            self.t0 = time.time()
            self.measTimer.start(0)
            
            
        if self.measType == 'PickAndDestroy':
            
            
            print('[minflux] PickAndDestroy self.n, self.acqtime', self.n, self.acqtime)
            self.tcspcPrepareSignal.emit(name, self.acqtime, self.n, True) # signal emitted to tcspc module to start the measurement #TODO do once
            phtime = 4.0  # in s, it takes 4 s for the PH to start the measurement, TO DO: check if this can be reduced (send email to Picoquant, etc)
            time.sleep(phtime)
            self.xyzStartSignal.emit()
            self.tcspcStartSignal.emit()
            self.t0 = time.time()
            
            
            
            
            
    @pyqtSlot(dict) 
    def Destroy_PickAndDestroy(self,params):
        
        self.measType = 'Standard'
        
        self.acqtime = params['acqtime']
        self.measType = params['measType']
        
        self.filename = params['filename'] 
        
               
        
        
        self.start()
        
        
        
            
    
    def loop(self):
        
        now = time.time()
        
        if (now - (self.t0 + self.i * self.acqtime) + self.acqtime) > self.acqtime:
            
            print(datetime.now(), '[minflux] loop', self.i)
                        
            self.moveToSignal.emit(self.r[self.i], self.pattern[self.i])
        
            self.i += 1
            
            if self.i == self.n:
                                
                self.stop()
                
                print(datetime.now(), '[minflux] measurement ended')
                
    def stop(self):
        
        self.measTimer.stop()
        self.TCSPCstopSignal.emit()
        
        
        
    @pyqtSlot()  
    def get_tcspc_done_signal(self):
        
        """
        Connection: [tcspc] tcspcDoneSignal
        """
        
        self.xyzEndSignal.emit(self.filename)
        
    def make_connection(self, frontend):
        
        frontend.paramSignal.connect(self.get_frontend_param)
#        frontend.filenameSignal.connect(self.get_filename)
        frontend.startButton.clicked.connect(self.start)
        frontend.stopButton.clicked.connect(self.stop)

