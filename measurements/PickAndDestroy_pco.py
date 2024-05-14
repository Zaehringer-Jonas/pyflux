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
from PIL import Image
from PIL.TiffTags import TAGS

#from threading import Thread

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from tkinter import Tk, filedialog

import tools.tools as tools
import tools.viewbox_tools as viewbox_tools
import tools.colormaps as cmaps
from tools.lineprofile import linePlotWidget
from tools.PSF import poly_func





import sys 

#
#sys.path.append('..\\tools')
#import tools as tools
#import viewbox_tools as viewbox_tools
#import colormaps as cmaps
#from lineprofile import linePlotWidget
#from PSF import poly_func


from skimage.feature import peak_local_max
#from skimage import data, img_as_float
#from scipy import ndimage as ndi
from skimage import morphology
import pco
#Kinesisstage
#import thorlabs_apt #this is the problem!
#import /../.drivers.xilab.crossplatform.wrappers.python.pyximc
from ctypes import *
import sys
import platform
import tempfile
import re


π = np.pi
#



class Frontend(QtGui.QFrame):
    
    filenameSignal = pyqtSignal(str)
    paramSignal = pyqtSignal(dict)
    PickandDestroySignal = pyqtSignal(np.ndarray)
    Backend_request_ImageSignal = pyqtSignal()
    Backend_request_liveImageSignal = pyqtSignal(bool)
    Backend_analyse_ImageSignal = pyqtSignal()
    Backend_setStageSignal = pyqtSignal(bool)
    Backend_calculate_SingleCoordSignal = pyqtSignal(np.ndarray)
    importDataSignal = pyqtSignal()
    
    
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        
        
        self.clicks = np.array([])
        
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
        try:
#        params['measType'] = self.measType.currentText()
            params['filename'] = filename
            params['positions'] = self.clicks # in px
            
            params['acqtime'] = float(self.acqtimeEdit.text())
            params['exptime'] = int(self.exptimeEdit.text())
            params['treshold'] = float(self.tresholdEdit.text())
            params['filter_radius'] = float(self.filterEdit.text())
            params['PDNr'] = float(self.PDNrEdit.text())
            params['dxWF'] = float(self.dxWFEdit.text())
            params['dyWF'] = float(self.dyWFEdit.text())
            params['StepSize'] = float(self.StepSizeEdit.text())
            
            
            
            
            
        
        except(RuntimeError, ValueError):
            QMessageBox.about(self, "[Pick and Destroy] Warning", "Input error. Please check inputs.")
        
        
        self.paramSignal.emit(params)
    
    
    
    
    
    def load_folder(self):

        try:
            root = Tk()
            root.withdraw()
            try:
                folder = filedialog.askdirectory(parent=root,
                                                 initialdir=self.folderEdit.text())
            except(AttributeError):
                folder = filedialog.askdirectory(parent=root)
            root.destroy()
            if folder != '':
                self.folderEdit.setText(folder)
        except OSError:
            pass
        

    def load_file(self):

        try:
            root = Tk()
            root.withdraw()
            try:
                fullfile = filedialog.askopenfilename(parent=root,
                                             initialdir=self.folderEdit.text(), filetypes=[ ("image",".tiff")])
            except(AttributeError):
                fullfile = filedialog.askopenfilename(parent=root, filetypes=[("image", ".tiff")])
            root.destroy()
            
            #
            
            
            if fullfile != '':
                folder, file = fullfile.rsplit('/', 1)
                self.folderEdit.setText(folder)
                self.filenameEdit.setText(file[:-5])
                
                self.emit_param()
                
                
                self.importDataSignal.emit()
                #TODO import
        except OSError:
            pass
        



    def requestImage(self):
        
        self.Backend_request_ImageSignal.emit()
        
        
        
    def liveImage(self): #lambda function?
        
        if self.getliveImage.isChecked():
            
        
            self.Backend_request_liveImageSignal.emit(True)
        else:
            self.reset_image()
            self.Backend_request_liveImageSignal.emit(False)
    
    
    
    
    def setStage(self):
        
        #TODO: reduce via lambda function
        
        
        # stage1pos_widefield = 90. # kinesis
        # stage2pos_widefield = 0. #xilinx
        # stage1pos_confocal = 0.# kinesis
        # stage2pos_confocal = 3000 #xilinx
        
        
        
        if self.setStageButton.isChecked():
            self.Backend_setStageSignal.emit(True )
            
        else:
            
            self.Backend_setStageSignal.emit(False)
    
    def reanalyse(self):
        self.reset_image()
        self.Backend_analyse_ImageSignal.emit()
    
    
    def reset_image(self):
        self.clicks = np.array([])
            
        for picked in self.PickedPoints:
            self.vb.removeItem(picked)
        self.PickedPoints = []
        # self.PickedPointsNumbers = []
    
    
    
    @pyqtSlot(np.ndarray)
    def getscanimage(self, image):
#        pass
        self.img.setImage(image)#, autoLevels=True)
        
    @pyqtSlot(np.ndarray)
    def getcoordinates(self, coordinates):
        mybrushorange = pg.mkBrush(255, 127, 80, 255) #orange
        mybrushblack = pg.mkBrush(0, 0, 0, 255) # black
        print("getting all coords")
        
        self.clicks = coordinates
#        print("Frontend")
#        print(coordinates)
#        print(coordinates[0])
        if coordinates.any():
            for i in np.arange(len(coordinates[0])):
#                print(coord)
#                print(coord[0])
                self.PickedPoints.append(pg.ScatterPlotItem(x=[int(coordinates[0][i])], 
                                                    y = [int(coordinates[1][i])], 
                                                    size=5,
                                                    pen=pg.mkPen(None), 
                                                    brush=mybrushblack))
            #TODO. make numbered!!
#            self.PickedPointsNumbers.append(pg.ScatterPlotItem([point.x()], 
#                                                    [point.y()], 
#                                                    size=5,
#                                                    pen=pg.mkPen(width=5, color='b'), 
#                                                    brush=mybrushblack, symbol=str(len(self.PickedPoints))))
#                print("here")
                self.vb.addItem(self.PickedPoints[-1])
#        self.img.setImage(image, autoLevels=True)
            


        
        

    def toggle_PickandDestroy(self,event):
        pass
    
    
#        if self.startPickingButton.isChecked():
#
#            self.clicks = np.array([])
#            
#            for picked in self.PickedPoints:
#                self.vb.removeItem(picked)
#            self.PickedPoints = []
#            # self.PickedPointsNumbers = []
#            
#            
#        
#        else:
#        
#            self.emit_param()
#            print("PD: done with picking", self.clicks  )
#        

        
    def mouse_release(self, event):
        print('click')   # never execute
#        print(event.scenePos())
#        print(event.pos())
        
        point = self.img.mapFromScene(event.scenePos())
#        print( self.img.mapFromScene(event.scenePos()))
        print([point.x(),point.y()])
        print("deleting", self.deleteButton.isChecked())
        if self.startPickingButton.isChecked():
#            for event in pg.event.get():
#                if event.type == pg.MOUSEBUTTONDOWN:
    
            if self.GaussFitCheckbox.isChecked():#new
                self.Backend_calculate_SingleCoordSignal.emit(np.array([point.x(),point.y()]))
            else:
                self.get_single_cooordinate(np.array([point.x(),point.y()]))#does it work?
            # self.clicks.append([point.x(),point.y()]) # in px
            # #put number on frontendposition
            # mybrushorange = pg.mkBrush(255, 127, 80, 255) #orange
            # mybrushblack = pg.mkBrush(0, 0, 0, 255) # black
            
            
            # self.PickedPoints.append(pg.ScatterPlotItem([point.x()], 
            #                                         [point.y()], 
            #                                         size=10,
            #                                         pen=pg.mkPen(None), 
            #                                         brush=mybrushorange))
            
            
            #todo gaussian fit there?!
            
            
            #TODO. make numbered!!
#            self.PickedPointsNumbers.append(pg.ScatterPlotItem([point.x()], 
#                                                    [point.y()], 
#                                                    size=5,
#                                                    pen=pg.mkPen(width=5, color='b'), 
#                                                    brush=mybrushblack, symbol=str(len(self.PickedPoints))))
    #         self.vb.addItem(self.PickedPoints[-1])
    # #            self.vb.addItem(self.PickedPointsNumbers[-1])
    #         # self.emit_param()

        elif self.deleteButton.isChecked():
#            print(self.clicks)
            
            
            # self.clicks.append([point.x(),point.y()])
            
            print("deleting")
            
            
            
            try:
            
                
                combined_x_y_arrays = np.dstack([np.array(self.clicks[0]).ravel(),np.array(self.clicks[1]).ravel()])[0]
                dist = lambda x, y: (x[0]-y[0])**2 + (x[1]-y[1])**2
                closest = min(combined_x_y_arrays, key=lambda co: dist(co, [point.x(),point.y()]))
                print(closest)
                
                
                if ((point.x() - closest[0])** 2 + (point.y() - closest[1])**2)**0.5 < 4:#smaller 4 px delete point!
                    
                    #find index
    #                ind = combined_x_y_arrays.index(closest)
                    print(np.where(combined_x_y_arrays == closest))
                    ind = np.where(combined_x_y_arrays == closest)[0][0]
                    print("delete at ", ind, " of ", closest)
                    #delete index
                    
                    x = np.delete(self.clicks[0], ind)
                    y = np.delete(self.clicks[1], ind)
                    self.vb.removeItem(self.PickedPoints[ind])
                    del self.PickedPoints[ind]

                    print(len(self.PickedPoints))
                    
                    self.clicks = np.array([x,y]) #to test
                    print(self.clicks)
                    
            except(ValueError):
                print("PD Error at deleting")
                
                
    @pyqtSlot(np.ndarray)
    def get_single_cooordinate(self, coord):
        
        print(coord)
#        self.clicks = np.append(self.clicks, [coord[0],coord[1]]) # in px

        if self.clicks.any():

            x = np.append(self.clicks[0], coord[0])
            y = np.append(self.clicks[1], coord[1])
        
            self.clicks = np.array([x,y])
            
        else:
            """Init"""

            x = coord[0]
            y = coord[1]
        
            self.clicks = np.array([x,y])
            
        print(self.clicks)
        
        
        
        #put number on frontendposition
        # mybrushorange = pg.mkBrush(255, 127, 80, 255) #orange
        # mybrushblack = pg.mkBrush(0, 0, 0, 255) # black
        mybrushred = pg.mkBrush(255, 0, 90, 255) #orange
        
        self.PickedPoints.append(pg.ScatterPlotItem(x=[int(coord[0])],
                                                y=[int(coord[1])], 
                                                size=7,
                                                pen=pg.mkPen(None), 
                                                brush=mybrushred))

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
        self.ImportButton = QtGui.QPushButton('Import Image')






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
        
        self.exptimeLabel = QtGui.QLabel('Exp time [ms]')
        self.exptimeEdit = QtGui.QLineEdit('800')
        
        self.tresholdLabel = QtGui.QLabel('treshold')
        self.tresholdEdit = QtGui.QLineEdit('0.3')        
        
        
        self.filterLabel = QtGui.QLabel('rolling ball')
        self.filterEdit = QtGui.QLineEdit('25')     
        
        self.setStageButton = QtGui.QPushButton('Set Stage')
        self.setStageButton.setCheckable(True)
        self.setStageButton.clicked.connect(self.setStage)
        
        self.getImage = QtGui.QPushButton('Get Scan Image')
        self.getImage.clicked.connect(self.requestImage)
        
        self.getliveImage = QtGui.QPushButton('Get live Image')
        self.getliveImage.setCheckable(True)
        self.getliveImage.clicked.connect(self.liveImage)
        
        
        self.startPickingButton = QtGui.QPushButton('Toggle Picking')
        self.startPickingButton.setCheckable(True)
        
        
        self.startPickingButton.clicked.connect(self.toggle_PickandDestroy)
        
        
        
        self.deleteButton = QtGui.QPushButton('Delete picks')
        self.deleteButton.setCheckable(True)
        
        self.deleteAllButton = QtGui.QPushButton('Reset all picks')
        self.deleteAllButton.clicked.connect(self.reset_image)
        # self.deleteAllButton.setCheckable(True)

        
        self.filterButton = QtGui.QPushButton('Reanalyse')
        
        self.filterButton.clicked.connect(self.reanalyse)
        
        self.startButton = QtGui.QPushButton('Init PD')
        self.PDButton = QtGui.QPushButton('Next PD')
        self.stopButton = QtGui.QPushButton('Stop')
        
        self.xUpButton = QtGui.QPushButton("(+x) ►")  # →
        self.xDownButton = QtGui.QPushButton("◄ (-x)")  # ←
        
        
        
        self.StepSizeLabel = QtGui.QLabel('Step Size Stage')
        self.StepSizeEdit = QtGui.QLineEdit('10')     
        
        self.PDNrLabel = QtGui.QLabel('Particle Nr.')
        self.PDNrEdit = QtGui.QLineEdit('0')    
        
        
        self.dxWFLabel = QtGui.QLabel('dx - [nm]')
        self.dxWFEdit = QtGui.QLineEdit('0')
        
        self.dyWFLabel = QtGui.QLabel('dy - [nm]')
        self.dyWFEdit = QtGui.QLineEdit('0')
        
        
        self.stageLabel = QtGui.QLabel('Move Stage')
        
        
        
        # toggle crosshair
        
        self.GaussFitCheckbox = QtGui.QCheckBox('GaussFit at Click')
#        self.GaussFitCheckbox.stateChanged.connect(self.ch.toggle)
        

#        subgrid.addWidget(self.measLabel, 0, 0, 1, 2)

        subgrid.addWidget(self.setStageButton, 4, 1, 1, 1)
        subgrid.addWidget(self.getliveImage, 4, 0, 1, 1)
        #
        subgrid.addWidget(self.getImage, 5, 0, 1, 1)
        subgrid.addWidget(self.filterButton, 5, 1, 1, 1)
        subgrid.addWidget(self.startPickingButton, 6, 0, 1, 1)
        subgrid.addWidget(self.deleteButton, 6, 1, 1, 1)
        subgrid.addWidget(self.GaussFitCheckbox, 7, 0, 1, 1)
        subgrid.addWidget(self.deleteAllButton, 7, 1, 1, 1)
        
        subgrid.addWidget(self.acqtimeLabel, 8, 0, 1, 1)
        subgrid.addWidget(self.acqtimeEdit, 8, 1, 1, 1)
        subgrid.addWidget(self.exptimeLabel, 9, 0, 1, 1)
        subgrid.addWidget(self.exptimeEdit, 9, 1, 1, 1)
        subgrid.addWidget(self.tresholdLabel, 10, 0, 1, 1)
        subgrid.addWidget(self.tresholdEdit, 10, 1, 1, 1)
        subgrid.addWidget(self.filterLabel, 11, 0, 1, 1)
        subgrid.addWidget(self.filterEdit, 11, 1, 1, 1)
        
        
        
        
        subgrid.addWidget(self.startButton, 12, 0, 1, 2)
        subgrid.addWidget(self.PDButton, 13, 0, 1, 1)
        subgrid.addWidget(self.stopButton, 13, 1, 1, 1)
        
        
        subgrid.addWidget(self.stageLabel, 14, 0, 1, 2)
        subgrid.addWidget(self.xUpButton, 15, 1, 1, 1)
        subgrid.addWidget(self.xDownButton, 15, 0, 1, 1)

        
         
        subgrid.addWidget(self.StepSizeLabel, 16, 0, 1, 1)
        subgrid.addWidget(self.StepSizeEdit, 16, 1, 1, 1)
               
        
        subgrid.addWidget(self.PDNrLabel, 16, 0, 1, 1)
        subgrid.addWidget(self.PDNrEdit, 16, 1, 1, 1)
        
        
        subgrid.addWidget(self.dxWFLabel, 17, 0, 1, 1)
        subgrid.addWidget(self.dxWFEdit, 17, 1, 1, 1)        
        
        subgrid.addWidget(self.dyWFLabel, 18, 0, 1, 1)
        subgrid.addWidget(self.dyWFEdit, 18, 1, 1, 1)       
        
        
        subgrid.addWidget(self.fileWidget, 0, 1)
        
        file_subgrid = QtGui.QGridLayout()
        self.fileWidget.setLayout(file_subgrid)
        
        file_subgrid.addWidget(self.filenameLabel, 0, 0, 1, 2)
        file_subgrid.addWidget(self.filenameEdit, 1, 0, 1, 2)
        file_subgrid.addWidget(self.folderLabel, 2, 0, 1, 2)
        file_subgrid.addWidget(self.folderEdit, 3, 0, 1, 2)
        file_subgrid.addWidget(self.browseFolderButton, 4, 0)
        file_subgrid.addWidget(self.ImportButton, 5, 0)
        
#        self.measType.currentIndexChanged.connect(self.toggle_parameters)
        
        self.folderEdit.textChanged.connect(self.emit_param)
        self.filenameEdit.textChanged.connect(self.emit_param)
        self.acqtimeEdit.textChanged.connect(self.emit_param)
        self.exptimeEdit.textChanged.connect(self.emit_param)
        self.tresholdEdit.textChanged.connect(self.emit_param)
        self.filterEdit.textChanged.connect(self.emit_param)
        self.browseFolderButton.clicked.connect(self.load_folder)
        self.ImportButton.clicked.connect(self.load_file)
        self.PDNrEdit.textChanged.connect(self.emit_param)
        self.dxWFEdit.textChanged.connect(self.emit_param)
        self.dyWFEdit.textChanged.connect(self.emit_param)

        
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
        
        
    @pyqtSlot(int)
    def get_backend_particle_nr(self, particle_nr):
        
        self.PDNrEdit.setText(str(particle_nr))
        
        
    def make_connection(self, backend):
        backend.sendImageFrontendSignal.connect(self.getscanimage)
        backend.sendCoordinatesFrontendSignal.connect(self.getcoordinates)
        backend.particleNumberSignal.connect(self.get_backend_particle_nr)
        
        backend.sendSingleCoordinateFrontendSignal.connect(self.get_single_cooordinate)



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
    WideFieldShutterSignal = pyqtSignal(bool)
    sendSingleCoordinateFrontendSignal = pyqtSignal(np.ndarray)
    
    
    sendCoordinatesFrontendSignal = pyqtSignal(np.ndarray)
    particleNumberSignal = pyqtSignal(int)
    detectAuNRSignal = pyqtSignal()
    
    
    def __init__(self, pidevice, lib, device_id, *args, **kwargs):
        
        
        
        super().__init__(*args, **kwargs)
        
        
        
        self.lib = lib
        self.device_id = device_id
        self.coordinates = np.array([])
        
        self.pco_pxsize = 74.06 #nm
        
        self.roi = [350, 800, 1249, 1699] # 1250, 1700]
        
        self.roi = [1100, 500, 2000, 1400] # 1250, 1700]
#        self.viewtimer.timeout.connect(self.getLiveImage)
        self.positions = 0
        self.particle_nr = 0
        self.pidevice = pidevice
        self.stageinit = False
        
        self.init_pco()
        self.livetimer = QtCore.QTimer()
        
    
        self.stage2pos_widefield = 200. #xilinx #700 EPI #225 TIRF
        self.stage2pos_confocal = 10000 #xilinx
        
#        self.initXilabStage()
        print("StageKinesis initing")
#        self.initKinesisStage() #works... VODOO
        print("StageKinesis inited") 
        
        
        self.importflag = False
        
#        self.i = 0 # counter
#        self.n = 1
        
  
#        self.pattern = np.array([0, 0])
        
#        self.measTimer = QtCore.QTimer()
#        self.measTimer.timeout.connect(self.loop)
        
    def piezo_xy(self, x_f, y_f):
        
#        print(datetime.now(), '[xyz_tracking] piezo x, y =', x_f, y_f)
        self.pidevice.MOV({1: x_f,2: y_f})
        time.sleep(0.05)
        for axis in self.pidevice.axes:
            position = self.pidevice.qPOS(axis)[axis]  # query single axis
            # position = pidevice.qPOS()[str(axis)] # query all axes
            print('current position of axis {} is {:.2f}'.format(axis, position))
        
    def initKinesisStage(self):
        
        

        self.kinesisMotor = thorlabs_apt.Motor(28250562)
        self.kinesisMotor.move_home(True)
        
        time.sleep(1)

    
    
    def init_pco(self):
        self.pcocam = pco.Camera()
        
        self.pcocam.record()
    
    


    
    
    
    def initXilabStage(self):
        
        print("[PD]: init xilab stage")
#        global lib, sbuf
        lib = self.lib
        sbuf = self.sbuf
        
    
        cur_dir = os.path.abspath(os.path.dirname(__file__)) # Specifies the current directory.

        print(cur_dir)
        
        print("init0")
        ximc_dir = os.path.join(cur_dir, "..","Xilab","ximc") # Formation of the directory name with all dependencies. The dependencies for the examples are located in the ximc directory.
        ximc_package_dir = os.path.join(ximc_dir, "crossplatform", "wrappers", "python") # Formation of the directory name with python dependencies.
           
#        print(lib)
#
#        sbuf = create_string_buffer(64)
#        lib.ximc_version(sbuf)
        
        # Set bindy (network) keyfile. Must be called before any call to "enumerate_devices" or "open_device" if you
        # wish to use network-attached controllers. Accepts both absolute and relative paths, relative paths are resolved
        # relative to the process working directory. If you do not need network devices then "set_bindy_key" is optional.
        # In Python make sure to pass byte-array object to this function (b"string literal").
        result = lib.set_bindy_key(os.path.join(ximc_dir, "win32", "keyfile.sqlite").encode("utf-8"))
        if result != Result.Ok:
            lib.set_bindy_key("keyfile.sqlite".encode("utf-8")) # Search for the key file in the current directory.
        
        # This is device search and enumeration with probing. It gives more information about devices.
        probe_flags = EnumerateFlags.ENUMERATE_PROBE + EnumerateFlags.ENUMERATE_NETWORK
        enum_hints = b"addr="
        # enum_hints = b"addr=" # Use this hint string for broadcast enumerate
        devenum = lib.enumerate_devices(probe_flags, enum_hints)
        print("Device enum handle: " + repr(devenum))
        print("Device enum handle type: " + repr(type(devenum)))
        dev_count = lib.get_device_count(devenum)
        controller_name = controller_name_t()
        for dev_ind in range(0, dev_count):
            enum_name = lib.get_device_name(devenum, dev_ind)
            result = lib.get_enumerate_device_controller_name(devenum, dev_ind, byref(controller_name))
            if result == Result.Ok:
                print("Enumerated device #{} name (port name): ".format(dev_ind) + repr(enum_name) + ". Friendly name: " + repr(controller_name.ControllerName) + ".")

        
        open_name = None
        if len(sys.argv) > 1:
            open_name = sys.argv[1]
        elif dev_count > 0:
            open_name = lib.get_device_name(devenum, 0)
        elif sys.version_info >= (3,0):
            # use URI for virtual device when there is new urllib python3 API
            tempdir = tempfile.gettempdir() + "/testdevice.bin"
            if os.altsep:
                tempdir = tempdir.replace(os.sep, os.altsep)
            # urlparse build wrong path if scheme is not file
#            uri = urllib.parse.urlunparse(urllib.parse.ParseResult(scheme="file", \
#                netloc=None, path=tempdir, params=None, query=None, fragment=None))
#            open_name = re.sub(r'^file', 'xi-emu', uri).encode()
#            flag_virtual = 1
            print("The real controller is not found or busy with another app.")
        if type(open_name) is str:
            open_name = open_name.encode()
    
        print("\nOpen device " + repr(open_name))
        self.device_id = lib.open_device(open_name)
        print("Device id: " + repr(self.device_id))

        #do something

    def xilinx_right(self):
        print("\nMoving right")
        result = self.lib.right(self.device_id)
        print("Result: " + repr(result))
    
    def xilinx_left(self):
        print("\nMoving left")
        result = self.lib.left(self.device_id)
        print("Result: " + repr(result))
    
    def xilinx_moveto(self, distance, udistance):
        print("\nGoing to {0} steps, {1} microsteps".format(distance, udistance))
        result = self.lib.command_move(self.device_id, distance, udistance)
        print("Result: " + repr(result))



    @pyqtSlot(dict)
    def get_frontend_param(self, params):
        
        """
        Connection: [frontend] paramSignal
        """
        #TODO give params at init
        
        print("MINFLUX: getting frontend parameters")
        self.acqtime = params['acqtime']
        self.exptime = params['exptime']
        self.treshold = params['treshold']
        self.clicks = params['positions']
        
#        self.positions = []

        
        self.filter_radius = params['filter_radius']
        
        
        #TODO test if init flag needed when running!
        self.particle_nr = int(params['PDNr'])
        
        
        
#        self.measType = params['measType']
#        today = str(date.today()).replace('-', '')
        self.filename = params['filename'] #+ '_' + today
        
        self.dxWF = int(params['dxWF'])
        self.dyWF = int(params['dyWF'])
        
        
        self.StepSizeXilinx = float(params['StepSize'])
        
        
#        self.patternType = params['patternType']
#        self.patternLength = float(params['patternLength'])/1000 # in micrometer
        self.calculate_position()
        self.update_param()
        
        
        
    def relative_move(self, axis, direction):
        
        
        if axis == 'x' and direction == 'up':
            self.stage2pos_widefield += self.StepSizeXilinx
            self.xilinx_moveto(int(self.stage2pos_widefield), 256 )
        

            
        if axis == 'x' and direction == 'down':
            
            self.stage2pos_widefield -= self.StepSizeXilinx
            self.xilinx_moveto(int(self.stage2pos_widefield), 256 )
        
        
        
    def calculate_position(self):
        self.positions = []
        
        if np.array(self.clicks).any():
            #affine transformation calibrated before
#            A =  [[ 0.07403606,  0.00023699], [-0.00016932,  0.07406459]]
##            t = [-1.73615122, -20.79150288] 
#            t = [-11.73615122, -30.79150288] 
            
            
            
    
    
#            def funcsingle(data, pxsizex,  pxsizey, offsetx, offsety):
#                """
#                poptx pxsize,offset
#                Out[27]: array([6.67704953e-02, 7.27692440e+01])
#                
#                popty
#                Out[28]: array([ 0.06793414, 29.37827808])
#                """
#            
#                data[:,0] = data[:,0] * pxsizex - offsetx
#                data[:,1] = data[:,1] * pxsizey - offsety
#              
#                return data
            
#            A =  [[ 7.39652844e-02,  3.26428392e-05], [-9.85528341e-05,  7.42431499e-02]]
#            t = [-11.79119116 ,-31.18748218]
#            print("calc positions")
            poptx = [6.67704953e-02 , 33.22101227338967  ]
            popty = [0.06793414 , 29.15441355451215 ] # 10 wegen symphotime 
            
            try:
                for p in np.arange(len(self.clicks[0])):
                    print(self.clicks[0][p], self.clicks[1][p])
                    
                    #pco to imagej
                    #x = yprog+350; y = xprog+800    
                    
                    #pos = [(self.clicks[1][p]+self.roi[0]) * 6.67704953e-02 + 7.27692440e+01   +35.58936243  , (self.clicks[0][p]+self.roi[1]) *0.06793414 + 29.37827808  + 1.1514979 ] # 10 wegen symphotime 
#                    pos = [(self.clicks[1][p]+self.roi[0]) * 6.67704953e-02 - 7.27692440e+01    , (self.clicks[0][p]+self.roi[1]) *0.06793414 - 29.37827808  ] # 10 wegen symphotime 
                    pos = [(self.clicks[1][p]+self.roi[0]) * poptx[0] - poptx[1]    , (self.clicks[0][p]+self.roi[1]) * popty[0] - popty[1] ] # 10 wegen symphotime 

                    print(pos) 
                    #pq 10 10 -> 10 90 in pminflux
                    
#                    pos = np.dot(A, [self.clicks[1][p] + self.roi[0] , (self.clicks[0][p]) + self.roi[1]]) + t # x y px vertauscht zu Imagej/PCO stuff...
                    pos[0] = pos[0] + 10  #-7.2 #Sympotime calibration... + XFactor?
                    pos[1] = 100 - (pos[1]  + 10)# -0.9#Sympotime calibration...
                    self.positions.append(pos)
            except(TypeError): #wow bad style
                try:
                    print(self.clicks[0], self.clicks[1])
                
                    #pco to imagej
                    #x = yprog+350; y = xprog+800    
                    
#                    pos = [(self.clicks[1]+self.roi[0]) * 6.67704953e-02 - 7.27692440e+01    , (self.clicks[0]+self.roi[1]) *0.06793414 - 29.37827808  ] # 10 wegen symphotime 
                    pos = [(self.clicks[1]+self.roi[0]) * poptx[0] - poptx[1]  , (self.clicks[0]+self.roi[1]) * popty[0] - popty[1] ] # 10 wegen symphotime 
                    
                    #pos = [(self.clicks[1][p]+self.roi[0]) * 6.67704953e-02 - 7.27692440e+01   +35.58936243  , (self.clicks[0][p]+self.roi[1]) *0.06793414 - 29.37827808  + 1.1514979 ] # 10 wegen symphotime 
##                    pos = [(self.clicks[1]+self.roi[0]) * 6.67704953e-02 - 73.92074188338967  ,  (self.clicks[0]+self.roi[1]) *0.06793414 -64.96764051451215 ] # 10 wegen symphotime 
    
                    print(pos) 
                    #pq 10 10 -> 10 90 in pminflux
                    
    #                    pos = np.dot(A, [self.clicks[1][p] + self.roi[0] , (self.clicks[0][p]) + self.roi[1]]) + t # x y px vertauscht zu Imagej/PCO stuff...
                    pos[0] = pos[0] + 10  #-7.2 #Sympotime calibration... + XFactor?
                    pos[1] = 100 - (pos[1]  + 10)# -0.9#Sympotime calibration...
                    self.positions.append(pos)
                    
                    #todo single [503.5670781776028, 796.2900890586694] instead of
                    #[[448.74378871 503.56707818]
                    #[791.59094996 796.29008906]]
                except(TypeError):                
                
                    print("[PD] error no positions no convert")
        print(self.positions)

            
        
        
        
        
    def update_param(self):
#        print(set)
        self.pcocam.set_exposure_time(self.exptime / 1000 )
        
        print(self.pcocam.get_exposure_time())
        
#            self.pcocam.configuration = {'exposure time': }
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
        
        
        
        #save image
        #save spotmap
        
        
#        self.setstages(False)
        
        
        
        
        
        
        print("[Pick and Destroy] starting routine")
        

        
            
        #TODO: calculate positions out of clicks
#        self.positions = self.clicks
        #ABfrage < 100 um
        
        self.save_current_data()
        
        
        
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
            print("PD: at Particle Nr: ", str(self.particle_nr), " of ", str(len(self.positions)))
#            print(self.positions[self.particle_nr])
#            print(self.positions)
            # self.scan_PickAndDestroySignal.emit(np.array(self.positions[self.particle_nr]))
            
            
            """#move to pos"""
#            print(self.positions)
            x = self.positions[self.particle_nr][0]+ self.dxWF
            y = self.positions[self.particle_nr][1]+ self.dyWF
#            print(x)
#            print(y)
#            print(self.positions)
            if (x > 0) and (y > 0) and (x < 100) and (y < 100):
                self.piezo_xy(x,y)
                
                self.detectAuNRSignal.emit()
            #
            #When fully automated: Uncomment
            # self.Destroy_PickandDestroy(True)
            
            #copy self.moveToSignal from Liveminflx
            
            
        else:
            print("Finished Pick and Destroy Routine")
        
        
        





    @pyqtSlot(bool)
    def Destroy_PickandDestroy(self, foundFitFlag = True):
        #todo change color in frotnend of points if sucessfull green, there at the moment or did not find -> red        
        
        params = dict()
        foundFitFlag = True
        params['acqtime'] = self.acqtime
        params['exptime'] = self.exptime
        params['measType'] = 'PickAndDestroy'
        params['filename'] = self.filename + '_'+ str(self.particle_nr)
        
        if foundFitFlag:
            print("PD: Found Particle, performing MINFLUX", self.particle_nr)
#            time.sleep(0.1) # TODO: why?
            
            
            
            self.MINFLUXDestroy_PickAndDestroySignal.emit(params)

            
        else:
            #TODO delte soon
            print("PD: Did not find particle ", self.particle_nr)
            self.particle_nr = self.particle_nr + 1
            self.loop_PickandDestroy()
        
        
        
        #TODO: Pass Signal to MINFLUX part! Parameters, Time <-> MINFLUX?, filename <-> MINFLUXfilename + number 
        #TODO other side of the signal
    @pyqtSlot(np.ndarray)  
    #end of measurement of xyz
    
    def CorrectedMovement(self, Movement):
        self.positions = self.positions + Movement #TODO: put in first
        
        
        self.particle_nr = self.particle_nr + 1
        self.particleNumberSignal.emit(self.particle_nr)
        
        self.loop_PickandDestroy()
        
        
        
    @pyqtSlot(np.ndarray)  
    def getScanImage(self, image):
        
        
        self.sendImageFrontendSignal.emit(image)
    
    
    def getLiveImage(self):
#        print("getImage")
        self.pcocam.record()

        image, meta = self.pcocam.image(-1, roi = self.roi)
        self.sendImageFrontendSignal.emit(image)    
        
        

    
    def toggle_liveImage(self, toggle_flag):
        print("[Pick and Destroy]: Toggling liveImage to", toggle_flag)
        if toggle_flag:
            
            
            self.WideFieldShutterSignal.emit(True)
#            self.pcocam.record()
            self.livetimer.start(self.exptime)
            print("started Timer to ", int(self.exptime ))
            
#            self.getLiveImage()
#        self.image, self.meta = self.pcocam.image(-1)
            
        else:
            self.WideFieldShutterSignal.emit(False)
            self.livetimer.stop()
            

    
    @pyqtSlot(bool)
    def setstages(self, stageFlag):
        """


        Parameters
        ----------
        stageFlag : bool
            True - Widefield
            False - confocal
        
        Returns
        -------
        None.
        
        Sets Stages according to StageFlag
        """
        #posFlag True        

        stage1pos_widefield = 89. # kinesis

        stage1pos_confocal = 0.001# kinesis

        
        
        self.stageFlag = stageFlag
        
        

        if self.stageinit == False:
            
            
#            
#            self.initXilabStage()
#            self.initKinesisStage()
            
            self.stageinit = True
            
        
        if self.stageFlag:
            self.piezo_xy(10,90)
            print("[Pick and Destroy]: Setting Stages to Widefield")
#            self.kinesisMotor.move_to(stage1pos_widefield)
            self.xilinx_moveto(int(self.stage2pos_widefield), 256 )
            
            position = self.pidevice.qPOS(3)[3]
            self.pidevice.MOV('3', position - 2.4)
        
        else:
            print("[Pick and Destroy]: Setting Stages to Confocal")
#            self.kinesisMotor.move_to(stage1pos_confocal)
            self.xilinx_moveto(int(self.stage2pos_confocal), 256 )
            position = self.pidevice.qPOS(3)[3]
            self.pidevice.MOV('3', position + 2.4)
            
        # self.posStage1 = posStage1
        # self.posStage2 = posStage2
        # print(self.posStage2)
        
        

        
        



    def getImage(self):
        
#        self.requestScanImagetoScanSignal.emit()
        
        self.importflag = False

        self.livetimer.stop()
#        self.piezo_xy(40,40)
        #IncreasePower - manual?
        #Shutter
        #shutter 4 open
        self.WideFieldShutterSignal.emit(True)
#        time.sleep(self.exptime / 1000)
#        self.WideFieldShutterSignal.emit(False)
        #sleep
        #shutter close
#        wait_for_first_image
        
        self.pcocam.record(4)
#        self.image, self.meta = self.pcocam.image(-1)
        self.image = self.pcocam.image_average(roi = self.roi)
        
        
        
        
        
#        print(self.pcocam.get_exposure_time())

#        im = self.subtract_background(self.image, radius = self.filter_radius)
        
#        image_max = ndi.maximum_filter(im, size=20, mode='constant')
#        self.coordinates = peak_local_max(im, threshold_rel = self.treshold)
#
#        print(len(self.coordinates ))
#        
##        print(self.meta)
#        self.sendImageFrontendSignal.emit(im)
#        self.sendCoordinatesFrontendSignal.emit(np.array(self.coordinates))
        
        self.WideFieldShutterSignal.emit(False)
        self.sendImageFrontendSignal.emit(self.image)
        self.analyse()



    @pyqtSlot(np.ndarray)  
    def find_single_gauss(self, coords):
        """TODO: Attention if self.coordinates no inited"""
#        print(coords)
        print(coords)
        print("trying to fit")
        newcoord = np.array(tools.gaussianfit(self.wavelets, [coords], 3, 1 ))
#        self.coordinates[0].append(self.coordinates[0],newcoord[0])
#        self.coordinates[1].append(self.coordinates[1],newcoord[1])
        #self coordinates needed?
        
        #if found
#        if newcoord != :
        self.sendSingleCoordinateFrontendSignal.emit(newcoord)#[self.coordinates[0][-1],self.coordinates[1][-1]]))


    def analyse(self):
        #todo get min size cleaned
        
    

        
        
        
        self.im = tools.subtract_background(self.image, radius = self.filter_radius)
        self.sendImageFrontendSignal.emit(self.im)
        try:
            wavelets = tools.calculate_wavelet(self.im,max_level=3,k=2.2)  #k level = 2 max level 3 seem to be good
            wavelets[wavelets == 0 ] += wavelets[wavelets != 0].min() 
        
            wavelets -= wavelets.min()
            self.wavelets  = morphology.remove_small_objects(wavelets>0, min_size=4) * wavelets
            
            
            peaks = peak_local_max(wavelets, threshold_abs = 0.05, min_distance=6 )
            
            self.coordinates = tools.gaussianfit(self.wavelets, peaks, 3, 1 )
            
            
    #        a = self.gaussianfit(wavelets, peaks, 5, 100 )
    
    #        image_max = ndi.maximum_filter(im, size=20, mode='constant')
    #        self.coordinates = peaks #peak_local_max(im, threshold_rel = self.treshold)
    
            print(len(self.coordinates))
            
            #filter some out:
            x = []
            y = []
            for i in range(len(self.coordinates[0])):
                if (self.coordinates[0][i] != 0) and (self.coordinates[1][i] != 0):
                    
                    x.append(self.coordinates[0][i])
                    y.append(self.coordinates[1][i])
                    
                    
            self.coordinates = [x,y]
                    
                    
                
            
            
    #        print(self.meta)
            print("[PD]: Data analyzed. Found ", str(len(self.coordinates[0])), " spots.")
            self.sendCoordinatesFrontendSignal.emit(np.array(self.coordinates))
        except(RuntimeError, ValueError):
            print("[PD] Warning", "Analysis Gone Wrong")
            self.coordinates = []
            self.positions = []
#            QMessageBox.about(self, "[PD] Warning", "Analysis Gone Wrong")
        
    def save_current_data(self):
        
        
        name = tools.getUniqueName(self.filename)
        print(name)
        
        
        
        # save image
        if self.importflag == False:
            data = self.image
#        print(data)
            result = Image.fromarray(data.astype('uint16')) #TODO: check if astype bad
            #TODO: check rotation
            result.save(r'{}.tiff'.format(name))#,  resolution_unit =  "cm", resolution = 1/ self.pxSize *10**4) .rotate(270)
    
    
            data = self.wavelets
    #        print(data)
            result = Image.fromarray(data.astype('uint16')) #TODO: check if astype bad
            #TODO: check rotation
            result.save(r'{}_wavelets.tif'.format(name))#,  resolution_unit =  "cm", resolution = 1/ self.pxSize *10**4) .rotate(270)



        else:
            
            pass
            
            #TODO: only if not exisits
#            data = self.image
#            data.save(r'{}.tiff'.format(name))#,  resolution_unit =  "cm", resolution = 1/ self.pxSize *10**4) .rotate(270)
#    
#    
#            data = self.wavelets
#    #        print(data)
#            
#            #TODO: check rotation
#            data.save(r'{}_wavelets.tif'.format(name))#,  resolution_unit =  "cm", resolution = 1/ self.pxSize *10**4) .rotate(270)

        
        #save Spotmap in px
        
        np.savetxt(name+ '_spots_px.txt', self.clicks)
#        print(self.coordinates)

        #save Spotmap in um:
        np.savetxt(name+ '_spots.txt', self.positions)

        
        print('[scan] Saved current frame', name)

    
    @pyqtSlot()  
    def importData(self):
        
        self.importflag = True

        self.coordinates = np.loadtxt(self.filename + '_spots_px.txt')
#        

        self.positions = np.loadtxt(self.filename + '_spots.txt')
        #TODO make functional
        
        self.image = np.asarray(Image.open(self.filename + ".tiff"))
        self.wavelets = np.asarray(Image.open(self.filename + "_wavelets.tif"))
        

        self.sendImageFrontendSignal.emit(np.array(self.image))
        self.sendCoordinatesFrontendSignal.emit(np.array(self.coordinates))        
        

    def stop(self):
        #TODO!!!
        self.pcocam.close()
#        del self.kinesisMotor
        self.lib.close_device(byref(cast(self.device_id, POINTER(c_int))))
#        del die #errors kills easier the program
        
        print("shutting down xilinxstage")        
        
    @pyqtSlot()  
    def get_tcspc_done_signal(self):
        
        """
        Connection: [tcspc] tcspcDoneSignal
        """
        
        self.xyzEndSignal.emit(self.filename)
        
    def make_connection(self, frontend):
        
        frontend.Backend_calculate_SingleCoordSignal.connect(self.find_single_gauss)
        frontend.Backend_analyse_ImageSignal.connect(self.analyse)
        frontend.Backend_request_ImageSignal.connect(self.getImage)
        frontend.Backend_request_liveImageSignal.connect(self.toggle_liveImage)
        frontend.Backend_setStageSignal.connect(self.setstages)
        
        frontend.paramSignal.connect(self.get_frontend_param)
#        frontend.filenameSignal.connect(self.get_filename)
        frontend.startButton.clicked.connect(self.init_PickandDestroy)
        frontend.PDButton.clicked.connect(self.Destroy_PickandDestroy)
        
        frontend.importDataSignal.connect(self.importData)
        
        
        frontend.xUpButton.pressed.connect(lambda: self.relative_move('x', 'up'))
        frontend.xDownButton.pressed.connect(lambda: self.relative_move('x', 'down')) 
        
        frontend.stopButton.clicked.connect(self.stop)

if __name__ == '__main__':
    sys.path.append('..\\tools')
    import tools as tools
    import viewbox_tools as viewbox_tools
    import colormaps as cmaps
    from lineprofile import linePlotWidget
    from PSF import poly_func

    app = QtGui.QApplication([])
    app.setStyle(QtGui.QStyleFactory.create('fusion'))
    worker = Backend(0)    

    gui = Frontend()


    
    pickThread = QtCore.QThread()
    worker.moveToThread(pickThread)
    worker.livetimer.moveToThread(pickThread)
    worker.livetimer.timeout.connect(worker.getLiveImage)
    pickThread.start()
    
    worker.make_connection(gui)

    gui.make_connection(worker)
    gui.setWindowTitle('pick')
    gui.show()
    

    
    
    app.exec_()
