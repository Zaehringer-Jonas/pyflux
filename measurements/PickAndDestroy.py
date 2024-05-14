# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:12:09 2020

@author: Jonas Zaehringer

Not fully functioning - Fit <-> position not accurate
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
import tools.viewbox_tools as viewbox_tools
import tools.colormaps as cmaps
from tools.lineprofile import linePlotWidget
from tools.PSF import poly_func





π = np.pi

class Frontend(QtGui.QFrame):
    
    filenameSignal = pyqtSignal(str)
    paramSignal = pyqtSignal(dict)
    PickandDestroySignal = pyqtSignal(np.ndarray)
    Backend_request_ScanImageSignal = pyqtSignal()
    
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        
        
        self.clicks = []
        
        self.setup_gui()
        
        
        
        self.emit_param()
        self.PickedPoints = []
        
        
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
        
#        params['measType'] = self.measType.currentText()
        params['filename'] = filename
        params['positions'] = self.clicks # in px
        
        params['acqtime'] = int(self.acqtimeEdit.text())
#        params['patternType'] = self.patternType.currentText()
#        params['patternLength'] = float(self.lengthEdit.text())
        
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
        

    def requestScanImage(self):
        
        self.Backend_request_ScanImageSignal.emit()
        
        
    @pyqtSlot(np.ndarray)
    def getscanimage(self, image):
        
        self.img.setImage(image, autoLevels=True)
        
        
        

    def toggle_PickandDestroy(self,event):
        
        if self.startPickingButton.isChecked():

            self.clicks = []
            
            for picked in self.PickedPoints:
                self.vb.removeItem(picked)
            self.PickedPoints = []
            self.PickedPointsNumbers = []
            
            
        
        else:
        
            self.emit_param()
            print("PD: done with picking", self.clicks  )
        
#        self.PickandDestroySignal.emit(clicks)

    def mouse_release(self, event):
        print('click')   # never execute
#        print(event.scenePos())
#        print(event.pos())
        
        point = self.img.mapFromScene(event.scenePos())
#        print( self.img.mapFromScene(event.scenePos()))
        print([point.x(),point.y()])
        if self.startPickingButton.isChecked():
#            for event in pg.event.get():
#                if event.type == pg.MOUSEBUTTONDOWN:
            self.clicks.append([point.x(),point.y()]) # in px
            #put number on frontendposition
            mybrushorange = pg.mkBrush(255, 127, 80, 255) #orange
            mybrushblack = pg.mkBrush(0, 0, 0, 255) # black
            
            
            self.PickedPoints.append(pg.ScatterPlotItem([point.x()], 
                                                    [point.y()], 
                                                    size=10,
                                                    pen=pg.mkPen(None), 
                                                    brush=mybrushorange))
            #TODO. make numbered!!
#            self.PickedPointsNumbers.append(pg.ScatterPlotItem([point.x()], 
#                                                    [point.y()], 
#                                                    size=5,
#                                                    pen=pg.mkPen(width=5, color='b'), 
#                                                    brush=mybrushblack, symbol=str(len(self.PickedPoints))))

            self.vb.addItem(self.PickedPoints[-1])
#            self.vb.addItem(self.PickedPointsNumbers[-1])
            self.emit_param()
            
            
            
            
#            print(self.clicks)
                    
    def setup_gui(self):
        
        self.setWindowTitle('Pick And Destroy measurement')



        imageWidget = pg.GraphicsLayoutWidget()
        self.vb = imageWidget.addViewBox(row=0, col=0)
        self.lplotWidget = linePlotWidget()
        
        imageWidget.setFixedHeight(500)
        imageWidget.setFixedWidth(500)
        
        # Viewbox and image item where the liveview will be displayed

        self.vb.setMouseMode(pg.ViewBox.RectMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.vb.addItem(self.img)
        self.vb.setAspectLocked(True)
        imageWidget.setAspectLocked(True)
        
        
        self.img.scene().sigMouseClicked.connect(self.mouse_release)
        
        
        #create crosshair
        #double .vb in order to get actual viewbox and not just the thing
        #we called viewbox 
        vbox = self.vb
        self.ch = viewbox_tools.Crosshair(vbox)
        
        
        # set up histogram for the liveview image

        self.hist = pg.HistogramLUTItem(image=self.img)
        lut = viewbox_tools.generatePgColormap(cmaps.parula)
        self.hist.gradient.setColorMap(lut)
        self.hist.vb.setLimits(yMin=0, yMax=10000)

        for tick in self.hist.gradient.ticks:
            tick.hide()
        imageWidget.addItem(self.hist, row=0, col=1)
        


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







        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)

        
        
        grid = QtGui.QGridLayout()

        self.setLayout(grid)
                
        grid.addWidget(self.paramWidget, 0, 0)
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)
        

        

        
        self.acqtimeLabel = QtGui.QLabel('Acq time [s]')
        self.acqtimeEdit = QtGui.QLineEdit('5')
        
        
        self.getImage = QtGui.QPushButton('Get Scan Image')
        self.getImage.clicked.connect(self.requestScanImage)
        
        self.startPickingButton = QtGui.QPushButton('Toggle Picking')
        self.startPickingButton.setCheckable(True)
        
        
        self.startPickingButton.clicked.connect(self.toggle_PickandDestroy)
        
        
        
        self.startButton = QtGui.QPushButton('Start')
        self.stopButton = QtGui.QPushButton('Stop')

#        subgrid.addWidget(self.measLabel, 0, 0, 1, 2)

        subgrid.addWidget(self.acqtimeLabel, 6, 0, 1, 1)
        subgrid.addWidget(self.acqtimeEdit, 6, 1, 1, 1)
        subgrid.addWidget(self.getImage, 4, 0, 1, 2)
        subgrid.addWidget(self.startPickingButton, 5, 0, 1, 2)
        subgrid.addWidget(self.startButton, 7, 0, 1, 2)
        subgrid.addWidget(self.stopButton, 8, 0, 1, 2)
        
        
        
        
        
        
        
        subgrid.addWidget(self.fileWidget, 0, 1)
        
        file_subgrid = QtGui.QGridLayout()
        self.fileWidget.setLayout(file_subgrid)
        
        file_subgrid.addWidget(self.filenameLabel, 0, 0, 1, 2)
        file_subgrid.addWidget(self.filenameEdit, 1, 0, 1, 2)
        file_subgrid.addWidget(self.folderLabel, 2, 0, 1, 2)
        file_subgrid.addWidget(self.folderEdit, 3, 0, 1, 2)
        file_subgrid.addWidget(self.browseFolderButton, 4, 0)
        
#        self.measType.currentIndexChanged.connect(self.toggle_parameters)
        
        self.folderEdit.textChanged.connect(self.emit_param)
        self.filenameEdit.textChanged.connect(self.emit_param)
        self.acqtimeEdit.textChanged.connect(self.emit_param)


        
             
        
        
        dockArea = DockArea() 
        grid.addWidget(dockArea, 0, 0)
        
        
        paramDock = Dock('Pick and Destroy parameters')
        paramDock.setOrientation(o="vertical", force=True)
        paramDock.updateStyle()
        paramDock.addWidget(self.paramWidget)
        paramDock.addWidget(self.fileWidget)
        dockArea.addDock(paramDock)
        
        imageDock = Dock('Last Scan')
        imageDock.addWidget(imageWidget)
        dockArea.addDock(imageDock, 'right', paramDock)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        self.emit_param()
        
    def make_connection(self, backend):
        backend.sendImageFrontendSignal.connect(self.getscanimage)
#        backend.paramSignal.connect(self.get_backend_param)
        
        pass

class Backend(QtCore.QObject):
    
#    tcspcPrepareSignal = pyqtSignal(str, int, int)
#    tcspcStartSignal = pyqtSignal()
#    
#    xyzStartSignal = pyqtSignal()
#    xyzEndSignal = pyqtSignal(str)
#    
#    moveToSignal = pyqtSignal(np.ndarray, np.ndarray)
    
    paramSignal = pyqtSignal(np.ndarray, np.ndarray, int)
    requestScanImagetoScanSignal = pyqtSignal()
    scan_PickAndDestroySignal = pyqtSignal(np.ndarray)
    MINFLUXDestroy_PickAndDestroySignal = pyqtSignal(dict) # To MINFLUX
    sendImageFrontendSignal = pyqtSignal(np.ndarray)
    
    
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

#        self.i = 0 # counter
#        self.n = 1
        
  
#        self.pattern = np.array([0, 0])
        
#        self.measTimer = QtCore.QTimer()
#        self.measTimer.timeout.connect(self.loop)
            

    @pyqtSlot(dict)
    def get_frontend_param(self, params):
        
        """
        Connection: [frontend] paramSignal
        """
        #TODO give params at init
        
        print("MINFLUX: getting frontend parameters")
        self.acqtime = params['acqtime']
        self.clicks = params['positions']
#        self.measType = params['measType']
        today = str(date.today()).replace('-', '')
        self.filename = params['filename'] + '_' + today
        
#        self.patternType = params['patternType']
#        self.patternLength = float(params['patternLength'])/1000 # in micrometer
        
        self.update_param()
        
    def update_param(self):
        
            pass
    
#        print('[minflux] self.pattern', self.pattern)
#        print(datetime.now(), '[minflux] self.r', self.r)
#        print(datetime.now(), '[minflux] self.r.shape', self.r.shape)


    def init_PickandDestroy(self):
        
# =============================================================================
#       NEW GUI TAB????!!!!
#        open image new -> picks it 
#        
#        
#        1. calculate positions in xyz
#         2. loop over positions with pickanddestroy function
#        
#        
# =============================================================================
#        print(positions)
        
        
        #Turn ZFeedback on ?! 
        # What if NP is interrupting Z Feedback?
        print("[Pick and Destroy] starting routine")
        self.particle_nr = 0
        
        
        
        #TODO: calculate positions out of clicks
        self.positions = self.clicks
        
        
        
        
        self.loop_PickandDestroy()
        
#        for pos in positions:
##            self.pick_PickandDestroy(pos)
#            self.pick_scan_PickandDestroySignal.emit(pos, self.particle_nr)
##            self.Destroy_PickandDestroy(particle_nr)
#            self.particle_nr = self.particle_nr + 1
            
    @pyqtSlot()
    def loop_PickandDestroy(self):
        
        
        #should locate it automatically
        if self.particle_nr < len(self.positions):
            print("PD: at Particle Nr: ", str(self.particle_nr))
#            print(self.positions[self.particle_nr])
#            print(self.positions)
            self.scan_PickAndDestroySignal.emit(np.array(self.positions[self.particle_nr]))
        else:
            print("Finished Pick and Destroy Routine")
        
        
        





    @pyqtSlot(bool)
    def Destroy_PickandDestroy(self, foundFitFlag):
        #todo change color in frotnend of points if sucessfull green, there at the moment or did not find -> red        
        
        params = dict()

        params['acqtime'] = self.acqtime
        params['measType'] = 'PickAndDestroy'
        params['filename'] = self.filename + '_'+ str(self.particle_nr)
        
        if foundFitFlag:
            print("PD: Found Particle, doing MINFLUX", self.particle_nr)
            time.sleep(0.5)
            self.MINFLUXDestroy_PickAndDestroySignal.emit(params)
            self.particle_nr = self.particle_nr + 1
        else:
            print("PD: Did not find particle ", self.particle_nr)
            self.particle_nr = self.particle_nr + 1
            self.loop_PickandDestroy()
        
        
        
        #TODO: Pass Signal to MINFLUX part! Parameters, Time <-> MINFLUX?, filename <-> MINFLUXfilename + number 
        #TODO other side of the signal
    
    
    @pyqtSlot(np.ndarray)  
    def getScanImage(self, image):
        
        
        self.sendImageFrontendSignal.emit(image)
        


    def requestScanImage(self):
        
        self.requestScanImagetoScanSignal.emit()

    def stop(self):
        #TODO!!!
        pass
                
        
    @pyqtSlot()  
    def get_tcspc_done_signal(self):
        
        """
        Connection: [tcspc] tcspcDoneSignal
        """
        
        self.xyzEndSignal.emit(self.filename)
        
    def make_connection(self, frontend):
        
        
        frontend.Backend_request_ScanImageSignal.connect(self.requestScanImage)
        frontend.paramSignal.connect(self.get_frontend_param)
#        frontend.filenameSignal.connect(self.get_filename)
        frontend.startButton.clicked.connect(self.init_PickandDestroy)
        
        
        frontend.stopButton.clicked.connect(self.stop)

if __name__ == '__main__':

    app = QtGui.QApplication([])
    app.setStyle(QtGui.QStyleFactory.create('fusion'))
    worker = Backend()    
    gui = Frontend()

    worker.make_connection(gui)
    gui.make_connection(worker)
    gui.setWindowTitle('pick')
    gui.show()
    
    app.exec_()
